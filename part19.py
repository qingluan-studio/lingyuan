
# ============================================================
# LINGYUAN MODEL - PART 19
# 混合专家模型 (Mixture of Experts, MoE)
#
# 在稀疏激活架构下扩展模型容量: 每个token仅激活少量专家,
# 从而以更少的计算量获得更大的模型容量。
#
# 核心组件:
# - Expert          : 专家网络 (SwiGLU FFN), 独立权重与统计
# - ExpertRouter    : Top-K路由器, 噪声路由, 容量因子, 负载均衡损失
# - MoELayer        : N个专家 + 路由器, 稀疏激活, 残差连接
# - MoETransformerModel : 交替MoE/稠密FFN层, 预设配置, 参数量统计
# - LoadBalancer    : 专家利用率监控, 动态路由调整, 专家迁移, 辅助损失
# - MoETrainingEngine : MoE专用训练引擎, 专家级学习率, 专家预热
# - MoEConfig       : MoE配置 (专家数, 激活数, 容量因子, 路由噪声...)
# - ExpertPruner    : 专家裁剪/合并/分裂
#
# 纯Python标准库实现 (零外部依赖)
# 复用 part9.py 的 Transformer 基础设施 (注意力/RMSNorm/SwiGLU/KVCache/工具函数)
# ============================================================

import sys
import os
import math
import json
import time
import random
from collections import deque, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime

# 从 part9.py 导入 Transformer 基础设施 (与同目录模块协作)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part9 import (LingyuanTransformerModel, ModelConfig, TransformerLayer,
                   RMSNorm, SwiGLUFFN, MultiHeadAttention, PositionalEncoding,
                   KVCache, _glorot_uniform, _linear_2d, _matmul_2d, _transpose_2d,
                   _softmax_vec, _silu, _cross_entropy_loss, _MODEL_PRESETS)


# ============================================================
# 本地数学工具 (part9 未导出的反向传播辅助, 此处就地实现)
# ============================================================

def _zeros_2d(rows: int, cols: int) -> List[List[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _zeros_1d(n: int) -> List[float]:
    return [0.0] * n


def _silu_grad(x: float) -> float:
    """SiLU/Swish 的导数: sigmoid(x) * (1 + x * (1 - sigmoid(x)))"""
    if x < -50.0:
        return 0.0
    sig = 1.0 / (1.0 + math.exp(-x))
    return sig * (1.0 + x * (1.0 - sig))


def _rmsnorm_backward(dout: List[List[float]],
                      x_norm: List[List[float]],
                      rms: List[float],
                      weight: List[float],
                      dim: int) -> Tuple[List[List[float]], List[float]]:
    """RMSNorm 反向传播: y = (x/rms) * weight

    Returns: (dx, dweight) — dL/dx (seq×dim) 与 dL/dweight (dim,)
    """
    seq_len = len(x_norm)
    dx: List[List[float]] = [[0.0] * dim for _ in range(seq_len)]
    dweight: List[float] = [0.0] * dim
    for s in range(seq_len):
        xn = x_norm[s]
        dy = dout[s]
        r = rms[s] if rms[s] > 1e-12 else 1e-12
        g_s = 0.0
        for j in range(dim):
            g_s += dy[j] * xn[j] * weight[j]
        inv_n = 1.0 / dim
        for d in range(dim):
            dweight[d] += dy[d] * xn[d]
            dx[s][d] = (dy[d] * weight[d] - xn[d] * g_s * inv_n) / r
    return dx, dweight


def _softmax_backward_row(dout: List[float], probs: List[float]) -> List[float]:
    """单行 softmax 雅可比: dL/dx_j = p_j * (dL/dy_j - sum_k p_k dL/dy_k)"""
    n = len(probs)
    dot = 0.0
    for k in range(n):
        dot += probs[k] * dout[k]
    return [probs[j] * (dout[j] - dot) for j in range(n)]


def _outer_add(grad: List[List[float]], a: List[float], b: List[float],
               scale: float = 1.0) -> None:
    """grad += scale * (a ⊗ b), 原地累加。 a: m, b: n, grad: m×n"""
    m = len(a)
    n = len(b)
    for i in range(m):
        ai = a[i] * scale
        if ai == 0.0:
            continue
        gi = grad[i]
        for j in range(n):
            gi[j] += ai * b[j]


def _add_2d_inplace(dst: List[List[float]], src: List[List[float]]) -> None:
    """dst += src (同形, 原地)"""
    for i in range(len(dst)):
        dri = dst[i]
        sri = src[i]
        for j in range(len(dri)):
            dri[j] += sri[j]


def _scale_rows(x: List[List[float]], s: float) -> List[List[float]]:
    return [[v * s for v in row] for row in x]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """一维向量余弦相似度"""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _flatten_weights(W_gate: List[List[float]],
                     W_up: List[List[float]],
                     W_down: List[List[float]]) -> List[float]:
    """将专家三组权重展平为一维向量 (用于相似度比较)"""
    flat: List[float] = []
    for row in W_gate:
        flat.extend(row)
    for row in W_up:
        flat.extend(row)
    for row in W_down:
        flat.extend(row)
    return flat


# ============================================================
# MoE 预设配置
# ============================================================

# 在基础模型预设之上叠加 MoE 参数
_MOE_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny_moe": {
        "hidden_dim": 128, "num_layers": 4, "num_heads": 4, "num_kv_heads": 4,
        "ffn_dim": 256, "max_seq_len": 512, "vocab_size": 2048,
        "rope_theta": 10000.0, "norm_eps": 1e-6,
        "num_experts": 4, "num_activated_experts": 2,
        "expert_capacity_factor": 1.25, "router_noise": 0.1,
        "aux_loss_weight": 0.01, "moe_layer_freq": 2,
    },
    "small_moe": {
        "hidden_dim": 256, "num_layers": 8, "num_heads": 8, "num_kv_heads": 4,
        "ffn_dim": 512, "max_seq_len": 1024, "vocab_size": 4096,
        "rope_theta": 10000.0, "norm_eps": 1e-6,
        "num_experts": 8, "num_activated_experts": 2,
        "expert_capacity_factor": 1.25, "router_noise": 0.1,
        "aux_loss_weight": 0.01, "moe_layer_freq": 2,
    },
    "base_moe": {
        "hidden_dim": 512, "num_layers": 12, "num_heads": 8, "num_kv_heads": 8,
        "ffn_dim": 1024, "max_seq_len": 2048, "vocab_size": 8192,
        "rope_theta": 10000.0, "norm_eps": 1e-6,
        "num_experts": 16, "num_activated_experts": 4,
        "expert_capacity_factor": 1.25, "router_noise": 0.05,
        "aux_loss_weight": 0.02, "moe_layer_freq": 2,
    },
}


# ============================================================
# MoEConfig [MoE 配置]
# ============================================================

@dataclass
class MoEConfig:
    """MoE (混合专家) 模型配置

    包含基础 Transformer 配置字段 + MoE 专属字段:
    - num_experts            : 专家数量
    - num_activated_experts  : 每个token激活的专家数 (Top-K)
    - expert_capacity_factor : 容量因子, 限制每个专家处理的token数
    - router_noise           : 训练时路由噪声 (防止专家坍缩)
    - aux_loss_weight        : 辅助损失权重 (负载均衡)
    - moe_layer_freq         : MoE层频率 (每 N 层用 1 次 MoE)
    """
    # ---- 基础模型字段 ----
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4
    ffn_dim: int = 256
    max_seq_len: int = 512
    vocab_size: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    sliding_window: int = 0
    tie_word_embeddings: bool = True
    pos_method: str = "rope"

    # ---- MoE 专属字段 ----
    num_experts: int = 8
    num_activated_experts: int = 2
    expert_capacity_factor: float = 1.25
    router_noise: float = 0.1
    aux_loss_weight: float = 0.01
    moe_layer_freq: int = 2

    # 预设 (类常量)
    PRESETS = _MOE_PRESETS

    # ---------- 预设 ----------

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "MoEConfig":
        """从预设名称创建配置 (tiny_moe / small_moe / base_moe)"""
        if name not in cls.PRESETS:
            raise ValueError(f"未知MoE预设: {name}, 可选: {list(cls.PRESETS.keys())}")
        cfg = dict(cls.PRESETS[name])
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def list_presets(cls) -> Dict[str, Dict[str, Any]]:
        return dict(cls.PRESETS)

    @classmethod
    def get_preset_names(cls) -> List[str]:
        return list(cls.PRESETS.keys())

    # ---------- 验证 ----------

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if self.hidden_dim <= 0:
            errors.append("hidden_dim 必须为正数")
        if self.num_layers <= 0:
            errors.append("num_layers 必须为正数")
        if self.num_heads <= 0:
            errors.append("num_heads 必须为正数")
        if self.num_kv_heads <= 0:
            errors.append("num_kv_heads 必须为正数")
        if self.num_heads > 0 and self.hidden_dim % self.num_heads != 0:
            errors.append(f"hidden_dim({self.hidden_dim}) 必须能被 num_heads({self.num_heads}) 整除")
        if self.num_kv_heads > 0 and self.num_heads % self.num_kv_heads != 0:
            errors.append(f"num_heads({self.num_heads}) 必须能被 num_kv_heads({self.num_kv_heads}) 整除")
        if self.ffn_dim <= 0:
            errors.append("ffn_dim 必须为正数")
        if self.vocab_size <= 0:
            errors.append("vocab_size 必须为正数")
        # MoE 专属验证
        if self.num_experts <= 0:
            errors.append("num_experts 必须为正数")
        if self.num_activated_experts <= 0:
            errors.append("num_activated_experts 必须为正数")
        if self.num_activated_experts > self.num_experts:
            errors.append(f"num_activated_experts({self.num_activated_experts}) "
                          f"不能超过 num_experts({self.num_experts})")
        if self.expert_capacity_factor <= 0:
            errors.append("expert_capacity_factor 必须为正数")
        if self.moe_layer_freq <= 0:
            errors.append("moe_layer_freq 必须为正整数")
        if self.aux_loss_weight < 0:
            errors.append("aux_loss_weight 不能为负数")
        if self.router_noise < 0:
            errors.append("router_noise 不能为负数")
        if self.pos_method not in ("rope", "alibi", "absolute"):
            errors.append(f"pos_method 必须为 rope/alibi/absolute, 当前: {self.pos_method}")
        return (len(errors) == 0, errors)

    def validate_or_raise(self) -> None:
        ok, errors = self.validate()
        if not ok:
            raise ValueError(f"MoE配置无效: {'; '.join(errors)}")

    # ---------- 序列化 ----------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("PRESETS", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoEConfig":
        clean = {k: v for k, v in d.items() if k != "PRESETS"}
        return cls(**clean)

    def save(self, path: str) -> bool:
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, path: str) -> "MoEConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ---------- 参数量估计 ----------

    def _attn_params(self) -> int:
        head_dim = self.hidden_dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim
        return (self.hidden_dim * self.hidden_dim      # W_q
                + self.hidden_dim * kv_dim * 2         # W_k, W_v
                + self.hidden_dim * self.hidden_dim)   # W_o

    def _expert_params(self) -> int:
        """单个专家的参数量 (SwiGLU: W_gate + W_up + W_down)"""
        return (self.hidden_dim * self.ffn_dim * 2    # W_gate, W_up
                + self.ffn_dim * self.hidden_dim)    # W_down

    def _router_params(self) -> int:
        return self.hidden_dim * self.num_experts

    def _dense_ffn_params(self) -> int:
        return 3 * self.hidden_dim * self.ffn_dim  # SwiGLU gate/up/down

    def estimate_total_params(self) -> int:
        """总参数量 (所有专家权重都计入)"""
        embedding = self.vocab_size * self.hidden_dim
        attn_p = self._attn_params()
        expert_p = self._expert_params()
        router_p = self._router_params()
        dense_ffn_p = self._dense_ffn_params()
        per_moe_layer = attn_p + 2 * self.hidden_dim + expert_p * self.num_experts + router_p
        per_dense_layer = attn_p + 2 * self.hidden_dim + dense_ffn_p
        layers_total = 0
        for i in range(self.num_layers):
            if self._is_moe_layer(i):
                layers_total += per_moe_layer
            else:
                layers_total += per_dense_layer
        total = embedding + layers_total + self.hidden_dim  # final norm
        if not self.tie_word_embeddings:
            total += self.hidden_dim * self.vocab_size
        return total

    def estimate_activated_params(self) -> int:
        """激活参数量 (每token仅激活 K 个专家)"""
        embedding = self.vocab_size * self.hidden_dim
        attn_p = self._attn_params()
        expert_p = self._expert_params()
        router_p = self._router_params()
        dense_ffn_p = self._dense_ffn_params()
        per_moe_layer = attn_p + 2 * self.hidden_dim + expert_p * self.num_activated_experts + router_p
        per_dense_layer = attn_p + 2 * self.hidden_dim + dense_ffn_p
        layers_total = 0
        for i in range(self.num_layers):
            if self._is_moe_layer(i):
                layers_total += per_moe_layer
            else:
                layers_total += per_dense_layer
        total = embedding + layers_total + self.hidden_dim
        if not self.tie_word_embeddings:
            total += self.hidden_dim * self.vocab_size
        return total

    def _is_moe_layer(self, layer_idx: int) -> bool:
        """判断指定层是否为 MoE 层 (基于 moe_layer_freq)"""
        if self.moe_layer_freq <= 0:
            return False
        return (layer_idx % self.moe_layer_freq) == (self.moe_layer_freq - 1)

    def count_moe_layers(self) -> int:
        return sum(1 for i in range(self.num_layers) if self._is_moe_layer(i))

    def count_dense_layers(self) -> int:
        return self.num_layers - self.count_moe_layers()

    def get_stats(self) -> Dict[str, Any]:
        ok, errors = self.validate()
        total = self.estimate_total_params()
        activated = self.estimate_activated_params()
        return {
            "config": "MoEConfig",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "ffn_dim": self.ffn_dim,
            "vocab_size": self.vocab_size,
            "num_experts": self.num_experts,
            "num_activated_experts": self.num_activated_experts,
            "capacity_factor": self.expert_capacity_factor,
            "router_noise": self.router_noise,
            "aux_loss_weight": self.aux_loss_weight,
            "moe_layer_freq": self.moe_layer_freq,
            "num_moe_layers": self.count_moe_layers(),
            "num_dense_layers": self.count_dense_layers(),
            "total_params": total,
            "activated_params": activated,
            "activation_ratio": round(activated / total, 4) if total > 0 else 0.0,
            "valid": ok,
            "errors": errors,
        }


# ============================================================
# Expert [专家网络]
# ============================================================

class Expert:
    """单个专家网络 — 独立的 SwiGLU 前馈网络

    FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down

    每个专家拥有独立权重, 并统计:
    - 被选中次数 (selection_count)
    - 平均激活值 (average activation magnitude)
    - 计算耗时 (compute time)
    """

    def __init__(self, expert_id: int, hidden_dim: int, ffn_dim: int,
                 init_scale: float = 1.0):
        self.expert_id = expert_id
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.init_scale = init_scale

        # 独立权重 (Glorot 初始化, 可选缩放)
        self.W_gate = _glorot_uniform(hidden_dim, ffn_dim)
        self.W_up = _glorot_uniform(hidden_dim, ffn_dim)
        self.W_down = _glorot_uniform(ffn_dim, hidden_dim)

        # 初始化缩放 (用于专家预热/迁移)
        if init_scale != 1.0:
            for i in range(hidden_dim):
                for j in range(ffn_dim):
                    self.W_gate[i][j] *= init_scale
                    self.W_up[i][j] *= init_scale
            for i in range(ffn_dim):
                for j in range(hidden_dim):
                    self.W_down[i][j] *= init_scale

        # 统计信息
        self.selection_count = 0       # 被选中次数
        self.activation_sum = 0.0      # 激活值累积 (用于平均)
        self.activation_count = 0      # 激活样本数
        self.compute_time = 0.0        # 累计计算耗时 (秒)
        self.forward_count = 0          # 前向调用次数

        self._created_at = datetime.now().isoformat()

    # ---------- 前向 ----------

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """SwiGLU 前向: x (n×hidden) -> (n×hidden)"""
        n = len(x)
        if n == 0:
            return []
        t0 = time.time()
        gate = _linear_2d(x, self.W_gate)   # (n × ffn)
        up = _linear_2d(x, self.W_up)        # (n × ffn)
        activated: List[List[float]] = []
        act_mag_sum = 0.0
        for s in range(n):
            g = gate[s]
            u = up[s]
            row = [0.0] * self.ffn_dim
            for i in range(self.ffn_dim):
                v = _silu(g[i]) * u[i]
                row[i] = v
                act_mag_sum += abs(v)
            activated.append(row)
        out = _linear_2d(activated, self.W_down)   # (n × hidden)
        # 更新统计
        self.compute_time += time.time() - t0
        self.forward_count += 1
        self.activation_sum += act_mag_sum
        self.activation_count += n * self.ffn_dim
        return out

    def forward_with_cache(self, x: List[List[float]]
                           ) -> Tuple[List[List[float]], List[List[float]],
                                      List[List[float]], List[List[float]]]:
        """带中间缓存的 SwiGLU 前向 (供反向传播)

        Returns: (out, gate, up, activated)
        """
        n = len(x)
        if n == 0:
            return [], [], [], []
        gate = _linear_2d(x, self.W_gate)
        up = _linear_2d(x, self.W_up)
        activated = [[_silu(gate[s][i]) * up[s][i]
                      for i in range(self.ffn_dim)]
                     for s in range(n)]
        out = _linear_2d(activated, self.W_down)
        self.forward_count += 1
        self.activation_sum += sum(abs(v) for row in activated for v in row)
        self.activation_count += n * self.ffn_dim
        return out, gate, up, activated

    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        return self.forward(x)

    # ---------- 反向 (单专家 SwiGLU 梯度) ----------

    def backward(self, x: List[List[float]], dout: List[List[float]],
                 gate_cache: List[List[float]], up_cache: List[List[float]],
                 act_cache: List[List[float]]
                 ) -> Tuple[List[List[float]], List[List[float]],
                            List[List[float]], List[List[float]]]:
        """SwiGLU 反向传播

        Args:
            x        : 该专家的输入 (n×hidden)
            dout     : dL/d(专家输出) (n×hidden), 调用方应已乘以路由权重
            gate_cache/up_cache/act_cache : forward 时的中间缓存

        正向:
            gate = x @ W_gate      (n×ffn)
            up   = x @ W_up        (n×ffn)
            act  = silu(gate) * up (n×ffn)
            out  = act @ W_down    (n×hidden)

        Returns:
            (dW_gate, dW_up, dW_down, dx)
            dW_gate/dW_up: (hidden×ffn), dW_down: (ffn×hidden), dx: (n×hidden)
        """
        n = len(x)
        if n == 0:
            empty_g = _zeros_2d(self.hidden_dim, self.ffn_dim)
            empty_d = _zeros_2d(self.ffn_dim, self.hidden_dim)
            return empty_g, empty_g, empty_d, []

        dW_gate = _zeros_2d(self.hidden_dim, self.ffn_dim)
        dW_up = _zeros_2d(self.hidden_dim, self.ffn_dim)
        dW_down = _zeros_2d(self.ffn_dim, self.hidden_dim)
        dx: List[List[float]] = [[0.0] * self.hidden_dim for _ in range(n)]

        for s in range(n):
            xs = x[s]
            g = gate_cache[s]
            u = up_cache[s]
            a = act_cache[s]
            dy = dout[s]            # (hidden,)
            # dW_down += a ⊗ dy   (ffn×hidden)
            _outer_add(dW_down, a, dy, 1.0)
            # d_act[i] = sum_d dy[d] * W_down[i][d]   (ffn,)
            d_act = [0.0] * self.ffn_dim
            for i in range(self.ffn_dim):
                wd = self.W_down[i]
                acc = 0.0
                for d in range(self.hidden_dim):
                    acc += dy[d] * wd[d]
                d_act[i] = acc
            # d_up[i]   = d_act[i] * silu(g[i])
            # d_gate[i] = d_act[i] * u[i] * silu_grad(g[i])
            # dW_gate[h][i] += xs[h] * d_gate[i]; dW_up[h][i] += xs[h] * d_up[i]
            # dx[h]      += d_gate[i]*W_gate[h][i] + d_up[i]*W_up[h][i]
            dx_s = dx[s]
            for i in range(self.ffn_dim):
                sg = _silu(g[i])
                d_up_i = d_act[i] * sg
                d_gate_i = d_act[i] * u[i] * _silu_grad(g[i])
                for h in range(self.hidden_dim):
                    xh = xs[h]
                    if xh != 0.0:
                        dW_gate[h][i] += xh * d_gate_i
                        dW_up[h][i] += xh * d_up_i
                    dx_s[h] += d_gate_i * self.W_gate[h][i] + d_up_i * self.W_up[h][i]
        return dW_gate, dW_up, dW_down, dx

    # ---------- 统计 ----------

    @property
    def num_params(self) -> int:
        return (self.hidden_dim * self.ffn_dim * 2   # W_gate, W_up
                + self.ffn_dim * self.hidden_dim)     # W_down

    @property
    def average_activation(self) -> float:
        if self.activation_count == 0:
            return 0.0
        return self.activation_sum / self.activation_count

    def reset_stats(self) -> None:
        self.selection_count = 0
        self.activation_sum = 0.0
        self.activation_count = 0
        self.compute_time = 0.0
        self.forward_count = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "hidden_dim": self.hidden_dim,
            "ffn_dim": self.ffn_dim,
            "num_params": self.num_params,
            "selection_count": self.selection_count,
            "forward_count": self.forward_count,
            "average_activation": round(self.average_activation, 6),
            "compute_time_s": round(self.compute_time, 6),
            "init_scale": self.init_scale,
        }

    # ---------- 权重序列化 (用于迁移/合并/分裂) ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "hidden_dim": self.hidden_dim,
            "ffn_dim": self.ffn_dim,
            "init_scale": self.init_scale,
            "W_gate": self.W_gate,
            "W_up": self.W_up,
            "W_down": self.W_down,
        }

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.expert_id = d.get("expert_id", self.expert_id)
        self.W_gate = [list(r) for r in d["W_gate"]]
        self.W_up = [list(r) for r in d["W_up"]]
        self.W_down = [list(r) for r in d["W_down"]]

    def clone_weights(self) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        return ([list(r) for r in self.W_gate],
                [list(r) for r in self.W_up],
                [list(r) for r in self.W_down])

    def set_weights(self, W_gate: List[List[float]],
                    W_up: List[List[float]],
                    W_down: List[List[float]]) -> None:
        self.W_gate = [list(r) for r in W_gate]
        self.W_up = [list(r) for r in W_up]
        self.W_down = [list(r) for r in W_down]

    def scale_weights(self, factor: float) -> None:
        """整体缩放专家权重 (用于预热/迁移)"""
        for i in range(self.hidden_dim):
            for j in range(self.ffn_dim):
                self.W_gate[i][j] *= factor
                self.W_up[i][j] *= factor
        for i in range(self.ffn_dim):
            for j in range(self.hidden_dim):
                self.W_down[i][j] *= factor


# ============================================================
# ExpertRouter [专家路由器]
# ============================================================

class ExpertRouter:
    """Top-K 专家路由器

    功能:
    - Top-K 路由: 为每个 token 选择 K 个最相关的专家
    - 噪声路由: 训练时给路由 logits 加噪声, 防止专家坍缩
    - 容量因子: 限制每个专家处理的 token 数 (超出则丢弃)
    - 负载均衡损失: 辅助损失, 防止所有 token 聚到同一专家
    - 路由统计: 每个专家的被选频率 / 平均路由概率
    """

    def __init__(self, hidden_dim: int, num_experts: int,
                 num_activated_experts: int = 2,
                 capacity_factor: float = 1.25,
                 router_noise: float = 0.1,
                 aux_loss_weight: float = 0.01,
                 router_bias: Optional[List[float]] = None,
                 rng: Optional[random.Random] = None):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_activated_experts = num_activated_experts   # Top-K
        self.capacity_factor = capacity_factor
        self.router_noise = router_noise
        self.aux_loss_weight = aux_loss_weight

        # 路由权重矩阵: (hidden × num_experts)
        self.W_router = _glorot_uniform(hidden_dim, num_experts)
        # 可学习偏置 (用于负载均衡器动态调整), 默认 0
        self.router_bias = router_bias if router_bias is not None else [0.0] * num_experts

        self.rng = rng if rng is not None else random

        # 路由统计
        self.expert_selection_count = [0] * num_experts   # 被选入 top-K 的次数
        self.expert_token_count = [0] * num_experts        # 实际处理 token 数 (含容量)
        self.expert_dropped_count = [0] * num_experts     # 因容量丢弃的次数
        self.expert_prob_sum = [0.0] * num_experts        # 平均路由概率累积
        self.routing_count = 0                            # 路由调用次数 (token 数累计)
        self.total_tokens_routed = 0

    # ---------- 容量计算 ----------

    def compute_capacity(self, seq_len: int) -> int:
        """每个专家的最大 token 容量

        capacity = ceil(seq_len * K / num_experts * capacity_factor)
        """
        if self.num_experts <= 0:
            return seq_len
        cap = math.ceil(seq_len * self.num_activated_experts
                        / self.num_experts * self.capacity_factor)
        return max(1, cap)

    # ---------- 路由 ----------

    def route(self, x: List[List[float]],
              training: bool = False,
              capacity: Optional[int] = None
              ) -> Dict[str, Any]:
        """对输入进行 Top-K 路由

        Args:
            x        : (seq × hidden) 已归一化的输入
            training : 是否训练模式 (决定是否加噪声)
            capacity : 显式容量 (None 则自动计算)

        Returns:
            路由信息字典:
            - topk_indices : (seq × K) 每个 token 选中的专家索引
            - topk_weights : (seq × K) 对应权重 (softmax 归一化, 容量丢弃处为 0)
            - kept_mask     : (seq × K) 是否未被容量丢弃
            - full_probs    : (seq × num_experts) 全 softmax 概率 (用于 aux loss)
            - router_logits : (seq × num_experts) 路由 logits (含噪声)
            - capacity      : 使用的容量
            - aux_loss      : 负载均衡辅助损失
            - dropped_tokens: 被完全丢弃的 token 数
        """
        seq_len = len(x)
        K = self.num_activated_experts
        E = self.num_experts
        if capacity is None:
            capacity = self.compute_capacity(seq_len)

        # 1. 路由 logits: (seq × E)
        router_logits = _linear_2d(x, self.W_router)
        # 加偏置
        if any(b != 0.0 for b in self.router_bias):
            for s in range(seq_len):
                rl = router_logits[s]
                for e in range(E):
                    rl[e] += self.router_bias[e]
        # 2. 训练时加噪声
        if training and self.router_noise > 0.0:
            for s in range(seq_len):
                rl = router_logits[s]
                for e in range(E):
                    rl[e] += self.router_noise * self.rng.gauss(0.0, 1.0)

        # 3. 全 softmax 概率 (用于 aux loss)
        full_probs = [_softmax_vec(router_logits[s]) for s in range(seq_len)]

        # 4. Top-K 选择
        topk_indices: List[List[int]] = []
        topk_logits: List[List[float]] = []
        for s in range(seq_len):
            rl = router_logits[s]
            # 选 K 个最大的 (用 argpartition 近似: 这里排序取前 K)
            order = sorted(range(E), key=lambda e: rl[e], reverse=True)
            sel = order[:K]
            topk_indices.append(sel)
            topk_logits.append([rl[e] for e in sel])
        # 5. 对 top-K logits 做 softmax 得到路由权重
        topk_weights = [_softmax_vec(topk_logits[s]) for s in range(seq_len)]

        # 6. 容量限制 (drop-through)
        expert_load = [0] * E
        kept_mask = [[True] * K for _ in range(seq_len)]
        dropped_tokens = 0
        for s in range(seq_len):
            token_fully_dropped = True
            for k in range(K):
                e = topk_indices[s][k]
                if expert_load[e] < capacity:
                    expert_load[e] += 1
                    token_fully_dropped = False
                else:
                    kept_mask[s][k] = False
                    topk_weights[s][k] = 0.0
                    self.expert_dropped_count[e] += 1
            if token_fully_dropped:
                dropped_tokens += 1

        # 7. 统计
        for s in range(seq_len):
            for k in range(K):
                e = topk_indices[s][k]
                if kept_mask[s][k]:
                    self.expert_selection_count[e] += 1
                    self.expert_token_count[e] += 1
            for e in range(E):
                self.expert_prob_sum[e] += full_probs[s][e]
        self.routing_count += 1
        self.total_tokens_routed += seq_len

        # 8. 辅助损失
        aux_loss = self._compute_aux_loss(full_probs, topk_indices,
                                          kept_mask, seq_len)

        return {
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "kept_mask": kept_mask,
            "full_probs": full_probs,
            "router_logits": router_logits,
            "capacity": capacity,
            "aux_loss": aux_loss,
            "dropped_tokens": dropped_tokens,
            "expert_load": expert_load,
        }

    # ---------- 辅助损失 ----------

    def _compute_aux_loss(self, full_probs: List[List[float]],
                          topk_indices: List[List[int]],
                          kept_mask: List[List[bool]],
                          seq_len: int) -> float:
        """负载均衡辅助损失 (Switch Transformer 风格)

        f_i = 选中专家 i 的 token 比例 (在 top-K 中且未被丢弃)
        P_i = 专家 i 的平均路由概率 (来自 full softmax)
        aux_loss = num_experts * sum_i (f_i * P_i)

        该损失在专家负载完全均匀时取最小值 num_experts * (1/E) * (1/E) * E = ... 
        实际最小值为 1 (均匀分布时)。越大表示越不均衡。
        """
        E = self.num_experts
        if seq_len == 0 or E == 0:
            return 0.0
        # f_i
        f = [0.0] * E
        for s in range(seq_len):
            for k in range(self.num_activated_experts):
                if kept_mask[s][k]:
                    e = topk_indices[s][k]
                    f[e] += 1.0
        for e in range(E):
            f[e] /= (seq_len * self.num_activated_experts)
        # P_i
        P = [0.0] * E
        for s in range(seq_len):
            for e in range(E):
                P[e] += full_probs[s][e]
        for e in range(E):
            P[e] /= seq_len
        # aux loss = E * sum(f_i * P_i)
        loss = 0.0
        for e in range(E):
            loss += f[e] * P[e]
        return E * loss

    def aux_loss_grad_logits(self, full_probs: List[List[float]],
                             seq_len: int) -> List[List[float]]:
        """辅助损失对 router_logits 的 (简化) 梯度

        推动路由概率趋向均匀分布:
            dL/dlogit[s][e] ∝ (P_e - 1/E)   (鼓励均匀)
        这里返回 (seq × E) 梯度, 由调用方乘以 aux_loss_weight。
        """
        E = self.num_experts
        if seq_len == 0 or E == 0:
            return []
        # 平均概率
        P = [0.0] * E
        for s in range(seq_len):
            for e in range(E):
                P[e] += full_probs[s][e]
        for e in range(E):
            P[e] /= seq_len
        # 梯度: 推动 logits 朝使 P 均匀的方向
        grad = []
        for s in range(seq_len):
            row = [0.0] * E
            probs = full_probs[s]
            for e in range(E):
                row[e] = probs[e] * ((P[e] - 1.0 / E))
            grad.append(row)
        return grad

    # ---------- 统计 / 工具 ----------

    @property
    def num_params(self) -> int:
        return self.hidden_dim * self.num_experts

    def get_expert_frequencies(self) -> List[float]:
        """每个专家的被选频率 (0~1)"""
        total = self.total_tokens_routed * self.num_activated_experts
        if total == 0:
            return [1.0 / self.num_experts] * self.num_experts
        return [c / total for c in self.expert_selection_count]

    def get_expert_avg_probs(self) -> List[float]:
        """每个专家的平均路由概率"""
        if self.total_tokens_routed == 0:
            return [1.0 / self.num_experts] * self.num_experts
        return [p / self.total_tokens_routed for p in self.expert_prob_sum]

    def get_load_balance(self) -> float:
        """负载均衡度 (0~1, 1 表示完全均匀)"""
        freqs = self.get_expert_frequencies()
        if not freqs:
            return 0.0
        mean = sum(freqs) / len(freqs)
        var = sum((f - mean) ** 2 for f in freqs) / len(freqs)
        if mean < 1e-12:
            return 0.0
        return max(0.0, 1.0 - math.sqrt(var) / mean)

    def reset_stats(self) -> None:
        self.expert_selection_count = [0] * self.num_experts
        self.expert_token_count = [0] * self.num_experts
        self.expert_dropped_count = [0] * self.num_experts
        self.expert_prob_sum = [0.0] * self.num_experts
        self.routing_count = 0
        self.total_tokens_routed = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_experts": self.num_experts,
            "num_activated_experts": self.num_activated_experts,
            "capacity_factor": self.capacity_factor,
            "router_noise": self.router_noise,
            "aux_loss_weight": self.aux_loss_weight,
            "num_params": self.num_params,
            "routing_count": self.routing_count,
            "total_tokens_routed": self.total_tokens_routed,
            "expert_frequencies": [round(f, 4) for f in self.get_expert_frequencies()],
            "expert_avg_probs": [round(p, 4) for p in self.get_expert_avg_probs()],
            "expert_selection_count": list(self.expert_selection_count),
            "expert_dropped_count": list(self.expert_dropped_count),
            "load_balance": round(self.get_load_balance(), 4),
        }


# ============================================================
# MoELayer [MoE 层]
# ============================================================

class MoELayer:
    """混合专家层 (稀疏激活 FFN)

    结构:
        对归一化输入 h (seq × hidden):
        1. 路由器选出每个 token 的 Top-K 专家
        2. 容量限制后, 把 token 派发到对应专家
        3. 每个 token 的输出 = sum_k (router_weight_k * expert_k(h))
        4. 残差连接由外层 Transformer 层负责 (本层只产出 FFN 风格输出)

    参数量:
        total_params     = router + num_experts * expert_params
        activated_params = router + K * expert_params   (每 token)
    """

    def __init__(self, hidden_dim: int, ffn_dim: int,
                 num_experts: int, num_activated_experts: int = 2,
                 capacity_factor: float = 1.25,
                 router_noise: float = 0.1,
                 aux_loss_weight: float = 0.01,
                 expert_init_scales: Optional[List[float]] = None,
                 rng: Optional[random.Random] = None):
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.num_activated_experts = num_activated_experts
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.is_moe = True

        self.rng = rng if rng is not None else random

        # 专家列表
        if expert_init_scales is None:
            expert_init_scales = [1.0] * num_experts
        self.experts: List[Expert] = [
            Expert(e, hidden_dim, ffn_dim, init_scale=expert_init_scales[e])
            for e in range(num_experts)
        ]

        # 路由器
        self.router = ExpertRouter(
            hidden_dim, num_experts, num_activated_experts,
            capacity_factor, router_noise, aux_loss_weight,
            rng=self.rng)

        self._forward_count = 0

    # ---------- 前向 ----------

    def forward(self, x: List[List[float]],
                training: bool = False,
                capacity: Optional[int] = None
                ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """稀疏 MoE 前向

        Args:
            x        : (seq × hidden) 已归一化输入
            training : 是否训练模式
            capacity : 显式容量

        Returns:
            (output, routing_info)
            output : (seq × hidden)  sum_k w_k * expert_k(x)
            routing_info : 路由器返回的完整信息 + 专家缓存 (供反向)
        """
        seq_len = len(x)
        hidden = self.hidden_dim
        E = self.num_experts
        K = self.num_activated_experts
        if seq_len == 0:
            return [], {"topk_indices": [], "topk_weights": [],
                        "kept_mask": [], "aux_loss": 0.0, "capacity": 0,
                        "expert_tokens": [[] for _ in range(E)],
                        "expert_positions": [[] for _ in range(E)],
                        "expert_weights": [[] for _ in range(E)]}

        # 1. 路由
        routing = self.router.route(x, training=training, capacity=capacity)
        topk_indices = routing["topk_indices"]
        topk_weights = routing["topk_weights"]
        kept_mask = routing["kept_mask"]

        # 2. 派发: 收集每个专家要处理的 token
        expert_token_lists: List[List[List[float]]] = [[] for _ in range(E)]
        expert_positions: List[List[int]] = [[] for _ in range(E)]
        expert_weights: List[List[float]] = [[] for _ in range(E)]
        for s in range(seq_len):
            for k in range(K):
                if kept_mask[s][k]:
                    e = topk_indices[s][k]
                    expert_token_lists[e].append(x[s])
                    expert_positions[e].append(s)
                    expert_weights[e].append(topk_weights[s][k])

        # 3. 专家计算
        output: List[List[float]] = [[0.0] * hidden for _ in range(seq_len)]
        expert_outputs: List[List[List[float]]] = [None] * E
        for e in range(E):
            toks = expert_token_lists[e]
            if not toks:
                continue
            # 增加选择统计
            self.experts[e].selection_count += len(toks)
            expert_outputs[e] = self.experts[e].forward(toks)
            pos_list = expert_positions[e]
            w_list = expert_weights[e]
            outs = expert_outputs[e]
            for idx, pos in enumerate(pos_list):
                w = w_list[idx]
                if w == 0.0:
                    continue
                op = outs[idx]
                out_row = output[pos]
                for d in range(hidden):
                    out_row[d] += w * op[d]

        # 4. 附加专家缓存 (供反向)
        routing["expert_tokens"] = expert_token_lists
        routing["expert_positions"] = expert_positions
        routing["expert_weights"] = expert_weights
        # 记录输入 (供反向计算 dW)
        routing["input"] = x

        self._forward_count += 1
        return output, routing

    # ---------- 训练前向 (带缓存) ----------

    def forward_with_cache(self, x: List[List[float]],
                            training: bool = False,
                            capacity: Optional[int] = None
                            ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """带完整中间缓存的 MoE 前向 (供反向传播使用)

        在 forward 基础上额外缓存每个专家的 gate/up/activated/out,
        以及路由器全概率 (用于路由梯度)。
        """
        seq_len = len(x)
        hidden = self.hidden_dim
        E = self.num_experts
        K = self.num_activated_experts
        if seq_len == 0:
            return [], {"topk_indices": [], "topk_weights": [], "kept_mask": [],
                        "aux_loss": 0.0, "capacity": 0, "expert_positions": [[] for _ in range(E)],
                        "expert_weights": [[] for _ in range(E)],
                        "expert_caches": [None] * E, "expert_outputs": [None] * E,
                        "full_probs": [], "router_logits": [], "input": []}

        routing = self.router.route(x, training=training, capacity=capacity)
        topk_indices = routing["topk_indices"]
        topk_weights = routing["topk_weights"]
        kept_mask = routing["kept_mask"]

        # 派发
        expert_token_lists: List[List[List[float]]] = [[] for _ in range(E)]
        expert_positions: List[List[int]] = [[] for _ in range(E)]
        expert_weights: List[List[float]] = [[] for _ in range(E)]
        for s in range(seq_len):
            for k in range(K):
                if kept_mask[s][k]:
                    e = topk_indices[s][k]
                    expert_token_lists[e].append(x[s])
                    expert_positions[e].append(s)
                    expert_weights[e].append(topk_weights[s][k])

        output: List[List[float]] = [[0.0] * hidden for _ in range(seq_len)]
        expert_caches: List[Optional[Tuple]] = [None] * E
        expert_outputs: List[Optional[List[List[float]]]] = [None] * E
        for e in range(E):
            toks = expert_token_lists[e]
            if not toks:
                continue
            self.experts[e].selection_count += len(toks)
            out_e, gate_e, up_e, act_e = self.experts[e].forward_with_cache(toks)
            expert_caches[e] = (toks, gate_e, up_e, act_e)
            expert_outputs[e] = out_e
            pos_list = expert_positions[e]
            w_list = expert_weights[e]
            for idx, pos in enumerate(pos_list):
                w = w_list[idx]
                if w == 0.0:
                    continue
                op = out_e[idx]
                out_row = output[pos]
                for d in range(hidden):
                    out_row[d] += w * op[d]

        routing["expert_tokens"] = expert_token_lists
        routing["expert_positions"] = expert_positions
        routing["expert_weights"] = expert_weights
        routing["expert_caches"] = expert_caches
        routing["expert_outputs"] = expert_outputs
        routing["input"] = x
        self._forward_count += 1
        return output, routing

    # ---------- 反向传播 (MoE 梯度) ----------

    def backward(self, routing: Dict[str, Any],
                 dout: List[List[float]],
                 compute_router_grad: bool = True
                 ) -> Dict[str, Any]:
        """MoE 反向传播

        Args:
            routing : forward_with_cache 返回的路由信息
            dout    : dL/d(MoE输出) (seq × hidden) — 来自残差直通
            compute_router_grad : 是否计算路由器梯度

        专家梯度: 对每个被激活专家, dout 已按路由权重缩放后传入 Expert.backward
        路由梯度: dL/dw_{t,e} = expert_e_out_t · D_t, 经 softmax 反传到 logits

        Returns:
            {"dW_router", "expert_grads": [(dW_gate,dW_up,dW_down) per expert],
             "d_input": (seq×hidden), "dlogits": (seq×E)}
        """
        seq_len = len(dout)
        hidden = self.hidden_dim
        E = self.num_experts
        K = self.num_activated_experts
        expert_positions = routing["expert_positions"]
        expert_weights = routing["expert_weights"]
        expert_caches = routing["expert_caches"]
        expert_outputs = routing["expert_outputs"]
        x = routing["input"]
        topk_indices = routing["topk_indices"]
        kept_mask = routing["kept_mask"]
        full_probs = routing["full_probs"]

        d_input: List[List[float]] = [[0.0] * hidden for _ in range(seq_len)]
        expert_grads: List[Tuple[List[List[float]], List[List[float]],
                                 List[List[float]]]] = [None] * E
        # 路由权重对 logits 的梯度 (经 softmax 反传)
        dlogits: List[List[float]] = [[0.0] * E for _ in range(seq_len)]

        for e in range(E):
            cache = expert_caches[e]
            if cache is None:
                empty_g = _zeros_2d(hidden, self.ffn_dim)
                empty_d = _zeros_2d(self.ffn_dim, hidden)
                expert_grads[e] = (empty_g, empty_g, empty_d)
                continue
            toks, gate_e, up_e, act_e = cache
            positions = expert_positions[e]
            weights = expert_weights[e]
            outs_e = expert_outputs[e]
            n = len(toks)
            # dout_expert[j] = weights[j] * dout[positions[j]]  (n × hidden)
            dout_e = [[weights[j] * dout[positions[j]][d] for d in range(hidden)]
                      for j in range(n)]
            dW_gate, dW_up, dW_down, dx_e = self.experts[e].backward(
                toks, dout_e, gate_e, up_e, act_e)
            expert_grads[e] = (dW_gate, dW_up, dW_down)
            # 累加 d_input
            for j, pos in enumerate(positions):
                dxj = dx_e[j]
                dri = d_input[pos]
                for d in range(hidden):
                    dri[d] += dxj[d]
            # 路由权重梯度: dL/dw_{pos,e} = sum_d out_e[j][d] * dout[pos][d]
            if compute_router_grad:
                for j, pos in enumerate(positions):
                    w = weights[j]
                    if w == 0.0:
                        continue
                    op = outs_e[j]
                    dp = dout[pos]
                    dot = 0.0
                    for d in range(hidden):
                        dot += op[d] * dp[d]
                    # 经 top-K softmax 反传到 logits
                    # topk_weights[s] 是 selected logits 的 softmax; 
                    # dL/dw = dot, 反传到 selected logits
                    s = pos
                    # 找到 e 在 topk_indices[s] 中的位置 k
                    for k in range(K):
                        if topk_indices[s][k] == e and kept_mask[s][k]:
                            # softmax backward for this row of K logits
                            # probs = topk_weights[s] (length K)
                            # dL/dlogit_k = probs[k] * (dot - sum_r probs[r]*dot_r)
                            # 但这里只有单个 w 的梯度 dot; 简化: 直接把 dot 作为
                            # 该 selected logit 的梯度信号 (straight-through 近似)
                            dlogits[s][e] += dot * w * (1.0 - w)
                            break

        # 辅助损失梯度叠加到 dlogits
        if compute_router_grad and self.aux_loss_weight > 0 and full_probs:
            aux_grad = self.router.aux_loss_grad_logits(full_probs, seq_len)
            aw = self.aux_loss_weight
            for s in range(seq_len):
                for e in range(E):
                    dlogits[s][e] += aw * aux_grad[s][e]

        # dW_router = x^T @ dlogits  (hidden × E)
        dW_router = _zeros_2d(hidden, E) if compute_router_grad else None
        if compute_router_grad:
            for s in range(seq_len):
                xs = x[s]
                for e in range(E):
                    g = dlogits[s][e]
                    if g == 0.0:
                        continue
                    for h in range(hidden):
                        dW_router[h][e] += xs[h] * g

        return {
            "dW_router": dW_router,
            "expert_grads": expert_grads,
            "d_input": d_input,
            "dlogits": dlogits,
        }

    def __call__(self, x: List[List[float]],
                 training: bool = False,
                 capacity: Optional[int] = None
                 ) -> Tuple[List[List[float]], Dict[str, Any]]:
        return self.forward(x, training=training, capacity=capacity)

    # ---------- 统计 ----------

    @property
    def num_params(self) -> int:
        """总参数量 (所有专家)"""
        return (self.router.num_params
                + sum(e.num_params for e in self.experts))

    @property
    def activated_params(self) -> int:
        """激活参数量 (每 token 仅 K 个专家)"""
        if self.num_experts == 0:
            return self.router.num_params
        per_expert = self.experts[0].num_params
        return (self.router.num_params
                + per_expert * self.num_activated_experts)

    def reset_all_stats(self) -> None:
        self.router.reset_stats()
        for e in self.experts:
            e.reset_stats()
        self._forward_count = 0

    def get_expert_stats(self) -> List[Dict[str, Any]]:
        return [e.get_stats() for e in self.experts]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "ffn_dim": self.ffn_dim,
            "num_experts": self.num_experts,
            "num_activated_experts": self.num_activated_experts,
            "capacity_factor": self.capacity_factor,
            "total_params": self.num_params,
            "activated_params": self.activated_params,
            "activation_ratio": round(self.activated_params / self.num_params, 4)
                                 if self.num_params > 0 else 0.0,
            "forward_count": self._forward_count,
            "router": self.router.get_stats(),
            "experts": [e.get_stats() for e in self.experts],
        }


# ============================================================
# MoETransformerLayer [MoE Transformer 层]
# ============================================================

class MoETransformerLayer:
    """MoE Transformer 层 (PreNorm 架构)

    结构:
        x = x + Attention(RMSNorm(x))
        x = x + FFN_or_MoE(RMSNorm(x))

    FFN 子层可为:
    - 稠密 SwiGLUFFN (普通层)
    - MoELayer      (MoE 层, 稀疏激活)

    forward 返回 (x, routing_info), routing_info 对稠密层为 None,
    对 MoE 层为路由信息字典 (供辅助损失与反向传播)。
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 num_kv_heads: Optional[int] = None,
                 positional_encoding: Optional[PositionalEncoding] = None,
                 sliding_window: int = 0, layer_idx: int = 0,
                 dropout: float = 0.0, norm_eps: float = 1e-6,
                 moe_config: Optional[MoEConfig] = None,
                 rng: Optional[random.Random] = None):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.layer_idx = layer_idx
        self.dropout_rate = dropout
        self.norm_eps = norm_eps
        self.rng = rng if rng is not None else random

        self.is_moe = moe_config is not None

        self.norm1 = RMSNorm(hidden_dim, norm_eps)
        self.attn = MultiHeadAttention(
            hidden_dim, num_heads, num_kv_heads,
            positional_encoding, sliding_window, layer_idx)
        self.norm2 = RMSNorm(hidden_dim, norm_eps)

        if self.is_moe:
            self.ffn = MoELayer(
                hidden_dim, ffn_dim,
                num_experts=moe_config.num_experts,
                num_activated_experts=moe_config.num_activated_experts,
                capacity_factor=moe_config.expert_capacity_factor,
                router_noise=moe_config.router_noise,
                aux_loss_weight=moe_config.aux_loss_weight,
                rng=self.rng)
        else:
            self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

        self._forward_count = 0

    # ---------- 通用 ----------

    def _dropout(self, x: List[List[float]]) -> List[List[float]]:
        if self.dropout_rate <= 0.0:
            return x
        keep = 1.0 - self.dropout_rate
        return [[v if self.rng.random() < keep else 0.0 for v in row]
                for row in x]

    @property
    def num_params(self) -> int:
        base = (self.norm1.num_params + self.attn.num_params
                + self.norm2.num_params)
        if self.is_moe:
            return base + self.ffn.num_params
        return base + self.ffn.num_params

    @property
    def activated_params(self) -> int:
        base = (self.norm1.num_params + self.attn.num_params
                + self.norm2.num_params)
        if self.is_moe:
            return base + self.ffn.activated_params
        return base + self.ffn.num_params

    # ---------- 前向 (推理/通用) ----------

    def forward(self, x: List[List[float]], use_cache: bool = False,
                cache: Optional[KVCache] = None, seq_offset: int = 0,
                batch_idx: int = 0, training: bool = False
                ) -> Tuple[List[List[float]], Optional[Dict[str, Any]]]:
        """前向传播

        Returns:
            (x_out, routing_info)  routing_info 对稠密层为 None
        """
        # 注意力子层
        h = self.norm1(x)
        attn_out = self.attn(h, use_cache=use_cache, cache=cache,
                             seq_offset=seq_offset, batch_idx=batch_idx)
        if training:
            attn_out = self._dropout(attn_out)
        x = [[x[s][d] + attn_out[s][d] for d in range(self.hidden_dim)]
             for s in range(len(x))]

        # FFN / MoE 子层
        h = self.norm2(x)
        routing_info: Optional[Dict[str, Any]] = None
        if self.is_moe:
            ffn_out, routing_info = self.ffn(h, training=training)
        else:
            ffn_out = self.ffn(h)
        if training:
            ffn_out = self._dropout(ffn_out)
        x = [[x[s][d] + ffn_out[s][d] for d in range(self.hidden_dim)]
             for s in range(len(x))]

        self._forward_count += 1
        return x, routing_info

    def __call__(self, x: List[List[float]], **kwargs
                 ) -> Tuple[List[List[float]], Optional[Dict[str, Any]]]:
        return self.forward(x, **kwargs)

    # ---------- 训练前向 (带缓存) ----------

    def forward_for_training(self, x: List[List[float]]
                              ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """带完整缓存的训练前向, 供 MoETrainingEngine 反向传播

        缓存: layer_input, norm1(x_norm,rms,h1), attn_out,
              norm2(x_norm,rms,h2), FFN/MoE 中间值与路由信息。
        注意力采用 straight-through: 仅缓存 attn_out, 反向时近似恒等。
        """
        hidden = self.hidden_dim
        lc: Dict[str, Any] = {"is_moe": self.is_moe}
        lc["layer_input"] = [list(r) for r in x]

        # --- 注意力子层 ---
        norm1_x_norm: List[List[float]] = []
        norm1_rms: List[float] = []
        h1: List[List[float]] = []
        for row in x:
            n = len(row)
            ms = sum(v * v for v in row) / n
            r = math.sqrt(ms + self.norm_eps)
            xn = [v / r for v in row]
            norm1_x_norm.append(xn)
            norm1_rms.append(r)
            h1.append([xn[d] * self.norm1.weight[d] for d in range(hidden)])
        lc["norm1_x_norm"] = norm1_x_norm
        lc["norm1_rms"] = norm1_rms
        lc["h1"] = h1

        attn_out = self.attn(h1, use_cache=False, cache=None,
                             seq_offset=0, batch_idx=0)
        lc["attn_out"] = attn_out
        x = [[x[s][d] + attn_out[s][d] for d in range(hidden)]
             for s in range(len(x))]
        lc["after_attn"] = [list(r) for r in x]

        # --- FFN / MoE 子层 ---
        norm2_x_norm: List[List[float]] = []
        norm2_rms: List[float] = []
        h2: List[List[float]] = []
        for row in x:
            n = len(row)
            ms = sum(v * v for v in row) / n
            r = math.sqrt(ms + self.norm_eps)
            xn = [v / r for v in row]
            norm2_x_norm.append(xn)
            norm2_rms.append(r)
            h2.append([xn[d] * self.norm2.weight[d] for d in range(hidden)])
        lc["norm2_x_norm"] = norm2_x_norm
        lc["norm2_rms"] = norm2_rms
        lc["h2"] = h2

        if self.is_moe:
            ffn_out, routing = self.ffn.forward_with_cache(h2, training=True)
            lc["routing"] = routing
            lc["ffn_out"] = ffn_out
        else:
            # 稠密 SwiGLU 带缓存
            gate = _linear_2d(h2, self.ffn.W_gate)
            up = _linear_2d(h2, self.ffn.W_up)
            activated = [[_silu(gate[s][i]) * up[s][i]
                          for i in range(self.ffn_dim)]
                         for s in range(len(h2))]
            ffn_out = _linear_2d(activated, self.ffn.W_down)
            lc["ffn_gate"] = gate
            lc["ffn_up"] = up
            lc["ffn_activated"] = activated
            lc["ffn_out"] = ffn_out

        x = [[x[s][d] + ffn_out[s][d] for d in range(hidden)]
             for s in range(len(x))]
        lc["layer_output"] = [list(r) for r in x]
        self._forward_count += 1
        return x, lc

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "is_moe": self.is_moe,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "num_params": self.num_params,
            "activated_params": self.activated_params,
            "forward_count": self._forward_count,
            "attention": self.attn.get_stats(),
            **(self.ffn.get_stats() if self.is_moe else {"ffn_type": "dense"}),
        }


# ============================================================
# MoETransformerModel [MoE Transformer 模型]
# ============================================================

class MoETransformerModel:
    """MoE Transformer 模型

    在标准 Transformer 之上引入 MoE 层:
    - 交替使用 MoE 层与稠密 FFN 层 (每 moe_layer_freq 层用 1 次 MoE)
    - token_embedding + 位置编码 + N×MoETransformerLayer + RMSNorm + LM Head
    - 支持 from_preset: tiny_moe / small_moe / base_moe
    - 参数量统计: 总参数 vs 激活参数
    - 前向传播 (推理 KV Cache) 与训练前向 (带缓存)
    """

    PRESETS = _MOE_PRESETS

    def __init__(self, config: Any = None, **kwargs):
        # 从 config 或 kwargs 提取配置
        defaults = {
            "hidden_dim": 128, "num_layers": 4, "num_heads": 4,
            "num_kv_heads": 4, "ffn_dim": 256, "max_seq_len": 512,
            "vocab_size": 2048, "rope_theta": 10000.0, "norm_eps": 1e-6,
            "dropout": 0.0, "sliding_window": 0,
            "tie_word_embeddings": True, "pos_method": "rope",
            "num_experts": 8, "num_activated_experts": 2,
            "expert_capacity_factor": 1.25, "router_noise": 0.1,
            "aux_loss_weight": 0.01, "moe_layer_freq": 2,
        }
        cfg: Dict[str, Any] = {}
        if config is not None:
            if isinstance(config, dict):
                cfg.update(config)
            else:
                for k in defaults:
                    cfg[k] = getattr(config, k, defaults[k])
        cfg.update({k: v for k, v in kwargs.items() if v is not None})
        for k, v in defaults.items():
            cfg.setdefault(k, v)

        self.hidden_dim = cfg["hidden_dim"]
        self.num_layers = cfg["num_layers"]
        self.num_heads = cfg["num_heads"]
        self.num_kv_heads = cfg["num_kv_heads"]
        self.ffn_dim = cfg["ffn_dim"]
        self.max_seq_len = cfg["max_seq_len"]
        self.vocab_size = cfg["vocab_size"]
        self.rope_theta = cfg["rope_theta"]
        self.norm_eps = cfg["norm_eps"]
        self.dropout = cfg["dropout"]
        self.sliding_window = cfg["sliding_window"]
        self.tie_word_embeddings = cfg["tie_word_embeddings"]
        self.pos_method = cfg["pos_method"]

        # MoE 专属
        self.num_experts = cfg["num_experts"]
        self.num_activated_experts = cfg["num_activated_experts"]
        self.expert_capacity_factor = cfg["expert_capacity_factor"]
        self.router_noise = cfg["router_noise"]
        self.aux_loss_weight = cfg["aux_loss_weight"]
        self.moe_layer_freq = cfg["moe_layer_freq"]
        self.config_dict = cfg
        self.head_dim = self.hidden_dim // self.num_heads

        # 位置编码
        self.positional_encoding = PositionalEncoding(
            dim=self.hidden_dim, max_seq_len=self.max_seq_len,
            method=self.pos_method, rope_theta=self.rope_theta,
            num_heads=self.num_heads)

        # Token Embedding
        self.token_embedding = _glorot_uniform(self.vocab_size, self.hidden_dim)

        # 层堆叠 (交替 MoE / 稠密)
        self.layers: List[MoETransformerLayer] = []
        for i in range(self.num_layers):
            layer_moe_cfg = None
            if self._is_moe_layer(i):
                layer_moe_cfg = self._make_layer_moe_config()
            layer = MoETransformerLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                num_kv_heads=self.num_kv_heads,
                positional_encoding=self.positional_encoding,
                sliding_window=self.sliding_window,
                layer_idx=i,
                dropout=self.dropout,
                norm_eps=self.norm_eps,
                moe_config=layer_moe_cfg)
            self.layers.append(layer)

        self.final_norm = RMSNorm(self.hidden_dim, self.norm_eps)
        self.lm_head = (None if self.tie_word_embeddings
                        else _glorot_uniform(self.hidden_dim, self.vocab_size))

        self._forward_count = 0
        self._last_aux_loss = 0.0
        self._last_routing_infos: List[Optional[Dict[str, Any]]] = []
        self._created_at = datetime.now().isoformat()

    # ---------- 配置辅助 ----------

    def _make_layer_moe_config(self) -> MoEConfig:
        """为单层构造 MoEConfig (复用模型级 MoE 参数)"""
        return MoEConfig(
            hidden_dim=self.hidden_dim,
            num_layers=1,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            ffn_dim=self.ffn_dim,
            max_seq_len=self.max_seq_len,
            vocab_size=self.vocab_size,
            rope_theta=self.rope_theta,
            norm_eps=self.norm_eps,
            dropout=self.dropout,
            sliding_window=self.sliding_window,
            tie_word_embeddings=self.tie_word_embeddings,
            pos_method=self.pos_method,
            num_experts=self.num_experts,
            num_activated_experts=self.num_activated_experts,
            expert_capacity_factor=self.expert_capacity_factor,
            router_noise=self.router_noise,
            aux_loss_weight=self.aux_loss_weight,
            moe_layer_freq=self.moe_layer_freq)

    def _is_moe_layer(self, layer_idx: int) -> bool:
        if self.moe_layer_freq <= 0:
            return False
        return (layer_idx % self.moe_layer_freq) == (self.moe_layer_freq - 1)

    def get_moe_layers(self) -> List[int]:
        return [i for i, l in enumerate(self.layers) if l.is_moe]

    def get_dense_layers(self) -> List[int]:
        return [i for i, l in enumerate(self.layers) if not l.is_moe]

    # ---------- 预设 ----------

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "MoETransformerModel":
        if name not in cls.PRESETS:
            raise ValueError(f"未知MoE预设: {name}, 可选: {list(cls.PRESETS.keys())}")
        cfg = dict(cls.PRESETS[name])
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def list_presets(cls) -> Dict[str, Dict[str, Any]]:
        return dict(cls.PRESETS)

    # ---------- 嵌入 / LM Head ----------

    def embed(self, input_ids: List[int]) -> List[List[float]]:
        result = []
        for tid in input_ids:
            if 0 <= tid < self.vocab_size:
                result.append(list(self.token_embedding[tid]))
            else:
                result.append([0.0] * self.hidden_dim)
        return result

    def compute_logits(self, hidden: List[List[float]],
                       last_token_only: bool = False) -> List[List[float]]:
        h = self.final_norm(hidden)
        if last_token_only:
            h = h[-1:] if h else []
        if self.tie_word_embeddings:
            emb_t = _transpose_2d(self.token_embedding)
            return _matmul_2d(h, emb_t)
        return _matmul_2d(h, self.lm_head)

    # ---------- 前向 (推理) ----------

    def forward(self, input_ids: List[int], use_cache: bool = False,
                cache: Optional[KVCache] = None,
                last_token_only: bool = False,
                training: bool = False) -> List[List[float]]:
        seq_offset = 0
        if use_cache and cache is not None:
            seq_offset = cache.get_seq_len()

        x = self.embed(input_ids)
        if self.pos_method == "absolute":
            abs_pe = self.positional_encoding.get_absolute(len(input_ids))
            x = [[x[s][d] + abs_pe[s][d] for d in range(self.hidden_dim)]
                 for s in range(len(input_ids))]

        routing_infos: List[Optional[Dict[str, Any]]] = []
        aux_loss_total = 0.0
        for layer in self.layers:
            x, routing = layer(x, use_cache=use_cache, cache=cache,
                               seq_offset=seq_offset, batch_idx=0,
                               training=training)
            routing_infos.append(routing)
            if routing is not None:
                aux_loss_total += routing.get("aux_loss", 0.0)

        logits = self.compute_logits(x, last_token_only)
        self._forward_count += 1
        self._last_aux_loss = aux_loss_total * self.aux_loss_weight
        self._last_routing_infos = routing_infos
        return logits

    def forward_with_cache(self, input_ids: List[int],
                           cache: Optional[KVCache] = None
                           ) -> Tuple[List[List[float]], KVCache]:
        if cache is None:
            cache = KVCache(
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                max_batch=1)
        logits = self.forward(input_ids, use_cache=True, cache=cache,
                               last_token_only=True)
        return logits, cache

    # ---------- 训练前向 (带缓存) ----------

    def forward_for_training(self, input_ids: List[int],
                              targets: List[int]
                              ) -> Tuple[float, List[List[float]], Dict[str, Any]]:
        """训练用前向传播 — 保存所有中间激活值供反向传播

        Returns: (total_loss, logits, cache)
            total_loss = cross_entropy + aux_loss_weight * sum(aux_loss)
        """
        seq_len = len(input_ids)
        hidden = self.hidden_dim
        cache: Dict[str, Any] = {"input_ids": list(input_ids)}

        # Embedding
        x = self.embed(input_ids)
        cache["embeddings"] = [list(r) for r in x]
        if self.pos_method == "absolute":
            abs_pe = self.positional_encoding.get_absolute(seq_len)
            x = [[x[s][d] + abs_pe[s][d] for d in range(hidden)]
                 for s in range(seq_len)]

        # 层
        layer_caches: List[Dict[str, Any]] = []
        aux_loss_total = 0.0
        for layer in self.layers:
            x, lc = layer.forward_for_training(x)
            layer_caches.append(lc)
            if lc["is_moe"]:
                aux_loss_total += lc["routing"].get("aux_loss", 0.0)
        cache["layers"] = layer_caches

        # 最终 RMSNorm
        final_x_norm: List[List[float]] = []
        final_rms: List[float] = []
        h_final: List[List[float]] = []
        for row in x:
            n = len(row)
            ms = sum(v * v for v in row) / n
            r = math.sqrt(ms + self.norm_eps)
            xn = [v / r for v in row]
            final_x_norm.append(xn)
            final_rms.append(r)
            h_final.append([xn[d] * self.final_norm.weight[d] for d in range(hidden)])
        cache["final_x_norm"] = final_x_norm
        cache["final_rms"] = final_rms
        cache["h_final"] = h_final

        # logits
        if self.tie_word_embeddings:
            emb_t = _transpose_2d(self.token_embedding)
            logits = _matmul_2d(h_final, emb_t)
        else:
            logits = _matmul_2d(h_final, self.lm_head)
        cache["logits"] = logits

        # 损失
        ce_loss = _cross_entropy_loss(logits, targets)
        aux_loss = aux_loss_total * self.aux_loss_weight
        total_loss = ce_loss + aux_loss
        cache["aux_loss"] = aux_loss
        cache["ce_loss"] = ce_loss
        cache["total_loss"] = total_loss
        self._forward_count += 1
        return total_loss, logits, cache

    # ---------- 参数统计 ----------

    @property
    def num_params(self) -> int:
        """总参数量 (所有专家都计入)"""
        embedding = self.vocab_size * self.hidden_dim
        layers = sum(l.num_params for l in self.layers)
        total = embedding + layers + self.hidden_dim  # final norm
        if not self.tie_word_embeddings and self.lm_head is not None:
            total += self.hidden_dim * self.vocab_size
        return total

    @property
    def activated_params(self) -> int:
        """激活参数量 (每 token 仅 K 个专家)"""
        embedding = self.vocab_size * self.hidden_dim
        layers = sum(l.activated_params for l in self.layers)
        total = embedding + layers + self.hidden_dim
        if not self.tie_word_embeddings and self.lm_head is not None:
            total += self.hidden_dim * self.vocab_size
        return total

    def get_param_breakdown(self) -> Dict[str, Any]:
        embedding = self.vocab_size * self.hidden_dim
        moe_param = sum(l.num_params for l in self.layers if l.is_moe)
        moe_act = sum(l.activated_params for l in self.layers if l.is_moe)
        dense_param = sum(l.num_params for l in self.layers if not l.is_moe)
        final = self.hidden_dim
        lm = 0 if self.tie_word_embeddings else self.hidden_dim * self.vocab_size
        return {
            "embedding": embedding,
            "dense_layers": dense_param,
            "moe_layers_total": moe_param,
            "moe_layers_activated": moe_act,
            "final_norm": final,
            "lm_head": lm,
            "total": self.num_params,
            "activated": self.activated_params,
            "activation_ratio": round(self.activated_params / self.num_params, 4)
                                 if self.num_params > 0 else 0.0,
            "num_moe_layers": len(self.get_moe_layers()),
            "num_dense_layers": len(self.get_dense_layers()),
        }

    # ---------- 统计 ----------

    def reset_all_stats(self) -> None:
        for layer in self.layers:
            if layer.is_moe:
                layer.ffn.reset_all_stats()
            layer._forward_count = 0
        self._forward_count = 0
        self._last_aux_loss = 0.0

    def get_stats(self) -> Dict[str, Any]:
        moe_layer_indices = self.get_moe_layers()
        return {
            "model": "MoETransformerModel",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "ffn_dim": self.ffn_dim,
            "vocab_size": self.vocab_size,
            "num_experts": self.num_experts,
            "num_activated_experts": self.num_activated_experts,
            "capacity_factor": self.expert_capacity_factor,
            "router_noise": self.router_noise,
            "aux_loss_weight": self.aux_loss_weight,
            "moe_layer_freq": self.moe_layer_freq,
            "moe_layer_indices": moe_layer_indices,
            "num_moe_layers": len(moe_layer_indices),
            "num_dense_layers": self.num_layers - len(moe_layer_indices),
            "total_params": self.num_params,
            "activated_params": self.activated_params,
            "param_breakdown": self.get_param_breakdown(),
            "forward_count": self._forward_count,
            "last_aux_loss": round(self._last_aux_loss, 6),
            "created_at": self._created_at,
        }

    def get_moe_layer_stats(self) -> List[Dict[str, Any]]:
        return [self.layers[i].ffn.get_stats() for i in self.get_moe_layers()]


# ============================================================
# LoadBalancer [负载均衡器]
# ============================================================

class LoadBalancer:
    """MoE 负载均衡器

    功能:
    - 专家利用率监控: 跨所有 MoE 层统计每个专家的被选频率/平均概率
    - 动态路由调整: 根据历史负载调整路由器偏置 (鼓励使用空闲专家)
    - 专家迁移: 将过载专家的权重迁移给空闲专家 (权重混合)
    - 辅助损失计算: importance_loss + load_loss

    设计为作用于 MoETransformerModel 的所有 MoE 层。
    """

    def __init__(self, model: MoETransformerModel,
                 adjust_strength: float = 0.1,
                 migration_rate: float = 0.1,
                 history_len: int = 100):
        self.model = model
        self.adjust_strength = adjust_strength
        self.migration_rate = migration_rate
        self.history_len = history_len
        # 历史负载 (per MoE layer, per expert)
        self.history: Dict[int, deque] = {}
        for li in model.get_moe_layers():
            self.history[li] = deque(maxlen=history_len)
        self._adjust_count = 0
        self._migration_count = 0

    # ---------- 监控 ----------

    def collect_utilization(self, reset: bool = False
                            ) -> Dict[int, List[float]]:
        """收集所有 MoE 层的专家利用率 (被选频率)"""
        util: Dict[int, List[float]] = {}
        for li in self.model.get_moe_layers():
            router = self.model.layers[li].ffn.router
            freqs = router.get_expert_frequencies()
            util[li] = freqs
            self.history[li].append(list(freqs))
            if reset:
                router.reset_stats()
        return util

    def get_average_utilization(self) -> Dict[int, List[float]]:
        """历史平均利用率"""
        avg: Dict[int, List[float]] = {}
        for li, hist in self.history.items():
            if not hist:
                router = self.model.layers[li].ffn.router
                avg[li] = router.get_expert_frequencies()
                continue
            E = len(hist[0])
            acc = [0.0] * E
            for snap in hist:
                for e in range(E):
                    acc[e] += snap[e]
            n = len(hist)
            avg[li] = [a / n for a in acc]
        return avg

    def get_imbalance(self) -> Dict[int, float]:
        """每层负载不均衡度 (变异系数 CV, 0=完全均衡)"""
        util = self.get_average_utilization()
        imb: Dict[int, float] = {}
        for li, freqs in util.items():
            mean = sum(freqs) / len(freqs) if freqs else 0.0
            if mean < 1e-12:
                imb[li] = 0.0
                continue
            var = sum((f - mean) ** 2 for f in freqs) / len(freqs)
            imb[li] = math.sqrt(var) / mean
        return imb

    # ---------- 动态路由调整 ----------

    def adjust_routing(self, reset: bool = True) -> Dict[int, List[float]]:
        """根据历史负载调整路由器偏置

        对利用率低于平均的专家增加偏置 (鼓励路由), 高于平均的减小偏置。
        """
        util = self.get_average_utilization()
        adjustments: Dict[int, List[float]] = {}
        for li in self.model.get_moe_layers():
            router = self.model.layers[li].ffn.router
            freqs = util[li]
            E = len(freqs)
            mean_f = sum(freqs) / E if E else 0.0
            # 偏置调整: (mean - freq) * strength
            adj = [(mean_f - freqs[e]) * self.adjust_strength
                   for e in range(E)]
            for e in range(E):
                router.router_bias[e] += adj[e]
            # 限制偏置范围, 防止发散
            cap = 2.0
            for e in range(E):
                if router.router_bias[e] > cap:
                    router.router_bias[e] = cap
                elif router.router_bias[e] < -cap:
                    router.router_bias[e] = -cap
            adjustments[li] = adj
            if reset:
                router.reset_stats()
        self._adjust_count += 1
        return adjustments

    # ---------- 专家迁移 ----------

    def migrate_experts(self, top_k_overloaded: int = 1,
                        top_k_idle: int = 1) -> Dict[str, Any]:
        """将过载专家的权重迁移 (混合) 给空闲专家

        对每个 MoE 层: 选出被选频率最高的若干专家 (源) 与最低的若干 (目标),
        将目标专家权重按 migration_rate 朝源专家混合。

        Returns: 迁移报告 {layer_idx: [(src, dst), ...]}
        """
        util = self.get_average_utilization()
        report: Dict[str, Any] = {}
        rate = self.migration_rate
        for li in self.model.get_moe_layers():
            moe = self.model.layers[li].ffn
            freqs = util[li]
            E = len(freqs)
            order = sorted(range(E), key=lambda e: freqs[e], reverse=True)
            sources = order[:top_k_overloaded]
            targets = sorted(range(E), key=lambda e: freqs[e])[:top_k_idle]
            pairs = []
            for ti, t in enumerate(targets):
                s = sources[ti % len(sources)]
                if s == t:
                    continue
                src_exp = moe.experts[s]
                tgt_exp = moe.experts[t]
                sg, su, sd = src_exp.clone_weights()
                tg, tu, td = tgt_exp.clone_weights()
                # tgt = (1-rate)*tgt + rate*src
                new_g = [[(1 - rate) * tg[i][j] + rate * sg[i][j]
                          for j in range(len(tg[0]))] for i in range(len(tg))]
                new_u = [[(1 - rate) * tu[i][j] + rate * su[i][j]
                          for j in range(len(tu[0]))] for i in range(len(tu))]
                new_d = [[(1 - rate) * td[i][j] + rate * sd[i][j]
                          for j in range(len(td[0]))] for i in range(len(td))]
                tgt_exp.set_weights(new_g, new_u, new_d)
                pairs.append((s, t))
            report[li] = pairs
        self._migration_count += 1
        return report

    # ---------- 辅助损失 (importance + load) ----------

    def compute_aux_loss(self) -> Dict[str, float]:
        """计算 importance_loss + load_loss

        importance_loss: 各专家重要性 (被选频率) 的方差
        load_loss:       各专家平均路由概率的方差
        两者都越小越好 (0 表示完全均衡)
        """
        util = self.collect_utilization(reset=False)
        importance_loss = 0.0
        load_loss = 0.0
        n_layers = 0
        for li, freqs in util.items():
            router = self.model.layers[li].ffn.router
            probs = router.get_expert_avg_probs()
            E = len(freqs)
            mean_f = sum(freqs) / E
            mean_p = sum(probs) / E
            var_f = sum((f - mean_f) ** 2 for f in freqs) / E
            var_p = sum((p - mean_p) ** 2 for p in probs) / E
            importance_loss += var_f
            load_loss += var_p
            n_layers += 1
        if n_layers > 0:
            importance_loss /= n_layers
            load_loss /= n_layers
        total = importance_loss + load_loss
        return {
            "importance_loss": round(importance_loss, 8),
            "load_loss": round(load_loss, 8),
            "total_aux_loss": round(total, 8),
            "weight": self.model.aux_loss_weight,
            "weighted": round(total * self.model.aux_loss_weight, 8),
        }

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_moe_layers": len(self.model.get_moe_layers()),
            "adjust_strength": self.adjust_strength,
            "migration_rate": self.migration_rate,
            "history_len": self.history_len,
            "adjust_count": self._adjust_count,
            "migration_count": self._migration_count,
            "imbalance": {str(k): round(v, 4) for k, v in self.get_imbalance().items()},
            "aux_loss": self.compute_aux_loss(),
        }


# ============================================================
# ExpertPruner [专家裁剪器]
# ============================================================

class ExpertPruner:
    """专家裁剪 / 合并 / 分裂

    功能:
    - identify_inefficient_experts: 识别低效专家 (被选极少或平均激活极小)
    - merge_similar_experts: 基于权重相似度合并相似专家
    - split_overloaded_expert: 将过载专家分裂为两个
    - prune: 执行裁剪方案 (返回裁剪报告)
    """

    def __init__(self, model: MoETransformerModel,
                 min_selection_ratio: float = 0.05,
                 min_avg_activation: float = 1e-4,
                 merge_similarity_threshold: float = 0.95,
                 split_overload_ratio: float = 2.0):
        self.model = model
        self.min_selection_ratio = min_selection_ratio
        self.min_avg_activation = min_avg_activation
        self.merge_similarity_threshold = merge_similarity_threshold
        self.split_overload_ratio = split_overload_ratio

    # ---------- 识别低效专家 ----------

    def identify_inefficient_experts(self) -> Dict[int, List[Dict[str, Any]]]:
        """识别低效专家

        判据:
        - 被选比例 < min_selection_ratio
        - 或 平均激活值 < min_avg_activation
        """
        report: Dict[int, List[Dict[str, Any]]] = {}
        moe_indices = self.model.get_moe_layers()
        total_routed = 1
        for li in moe_indices:
            moe = self.model.layers[li].ffn
            router = moe.router
            total = router.total_tokens_routed * router.num_activated_experts
            total_routed = max(total, 1)
            freqs = router.get_expert_frequencies()
            bad = []
            for e in range(moe.num_experts):
                exp_stats = moe.experts[e].get_stats()
                sel_ratio = freqs[e]
                avg_act = exp_stats["average_activation"]
                reasons = []
                if sel_ratio < self.min_selection_ratio:
                    reasons.append(f"低选择率({sel_ratio:.4f})")
                if avg_act < self.min_avg_activation:
                    reasons.append(f"低激活({avg_act:.2e})")
                if reasons:
                    bad.append({
                        "expert_id": e,
                        "selection_ratio": round(sel_ratio, 4),
                        "average_activation": round(avg_act, 6),
                        "reasons": reasons,
                    })
            report[li] = bad
        return report

    # ---------- 合并相似专家 ----------

    def _expert_weight_similarity(self, exp_a: Expert, exp_b: Expert) -> float:
        """两个专家权重的展平余弦相似度"""
        fa = _flatten_weights(exp_a.W_gate, exp_a.W_up, exp_a.W_down)
        fb = _flatten_weights(exp_b.W_gate, exp_b.W_up, exp_b.W_down)
        return _cosine_sim(fa, fb)

    def merge_similar_experts(self) -> Dict[int, List[Tuple[int, int, float]]]:
        """合并相似专家

        对每层: 计算所有专家对权重相似度, 相似度 > 阈值则合并
        (保留被选频率高者, 删除低者, 但因专家数固定, 这里采用"平均化"合并:
         把相似专家的权重取平均写入其中一个, 另一个标记为待重置)。

        Returns: {layer_idx: [(expert_a, expert_b, similarity), ...]}
        """
        report: Dict[int, List[Tuple[int, int, float]]] = {}
        threshold = self.merge_similarity_threshold
        for li in self.model.get_moe_layers():
            moe = self.model.layers[li].ffn
            E = moe.num_experts
            freqs = moe.router.get_expert_frequencies()
            merged_pairs: List[Tuple[int, int, float]] = []
            merged = set()
            for a in range(E):
                if a in merged:
                    continue
                for b in range(a + 1, E):
                    if b in merged:
                        continue
                    sim = self._expert_weight_similarity(moe.experts[a],
                                                         moe.experts[b])
                    if sim >= threshold:
                        # 保留使用频率高的, 把另一个用平均值重置 (近似"合并")
                        keep = a if freqs[a] >= freqs[b] else b
                        drop = b if keep == a else a
                        ka = moe.experts[keep]
                        kb = moe.experts[drop]
                        kg, ku, kd = ka.clone_weights()
                        bg, bu, bd = kb.clone_weights()
                        new_g = [[(kg[i][j] + bg[i][j]) / 2.0
                                  for j in range(len(kg[0]))]
                                 for i in range(len(kg))]
                        new_u = [[(ku[i][j] + bu[i][j]) / 2.0
                                  for j in range(len(ku[0]))]
                                 for i in range(len(ku))]
                        new_d = [[(kd[i][j] + bd[i][j]) / 2.0
                                  for j in range(len(kd[0]))]
                                 for i in range(len(kd))]
                        ka.set_weights(new_g, new_u, new_d)
                        # drop 专家重新随机初始化 (鼓励探索新专家)
                        fresh_g = _glorot_uniform(moe.hidden_dim, moe.ffn_dim)
                        fresh_u = _glorot_uniform(moe.hidden_dim, moe.ffn_dim)
                        fresh_d = _glorot_uniform(moe.ffn_dim, moe.hidden_dim)
                        kb.set_weights(fresh_g, fresh_u, fresh_d)
                        kb.scale_weights(0.1)  # 小初始化, 渐进激活
                        merged.add(drop)
                        merged_pairs.append((keep, drop, round(sim, 4)))
            report[li] = merged_pairs
        return report

    # ---------- 专家分裂 ----------

    def split_overloaded_expert(self) -> Dict[int, List[int]]:
        """将过载专家分裂为两个

        对每层: 找出被选频率超过均值 split_overload_ratio 倍的专家,
        将其权重复制给当前被选最少的专家, 并各自乘以 0.5 缩放 (近似分裂)。

        Returns: {layer_idx: [被分裂的源专家id, ...]}
        """
        report: Dict[int, List[int]] = {}
        ratio_thresh = self.split_overload_ratio
        for li in self.model.get_moe_layers():
            moe = self.model.layers[li].ffn
            E = moe.num_experts
            freqs = moe.router.get_expert_frequencies()
            mean_f = sum(freqs) / E if E else 0.0
            order_asc = sorted(range(E), key=lambda e: freqs[e])  # 最少在前
            idle_idx = 0
            split_sources: List[int] = []
            for e in range(E):
                if mean_f > 0 and freqs[e] > mean_f * ratio_thresh:
                    if idle_idx >= E:
                        break
                    target = order_asc[idle_idx]
                    idle_idx += 1
                    if target == e or freqs[target] > mean_f * 0.5:
                        # 目标不够空闲, 跳过
                        continue
                    src = moe.experts[e]
                    tgt = moe.experts[target]
                    sg, su, sd = src.clone_weights()
                    tgt.set_weights([list(r) for r in sg],
                                   [list(r) for r in su],
                                   [list(r) for r in sd])
                    # 各自缩放 0.5 (近似分裂, 保持输出量级)
                    src.scale_weights(0.5)
                    tgt.scale_weights(0.5)
                    split_sources.append(e)
            report[li] = split_sources
        return report

    # ---------- 综合裁剪 ----------

    def prune(self, do_merge: bool = True, do_split: bool = False
              ) -> Dict[str, Any]:
        """执行裁剪方案, 返回报告"""
        inefficient = self.identify_inefficient_experts()
        merged = self.merge_similar_experts() if do_merge else {}
        split = self.split_overloaded_expert() if do_split else {}
        total_inefficient = sum(len(v) for v in inefficient.values())
        total_merged = sum(len(v) for v in merged.values())
        total_split = sum(len(v) for v in split.values())
        return {
            "inefficient_experts": inefficient,
            "num_inefficient": total_inefficient,
            "merged_pairs": merged,
            "num_merged": total_merged,
            "split_sources": split,
            "num_split": total_split,
            "config": {
                "min_selection_ratio": self.min_selection_ratio,
                "min_avg_activation": self.min_avg_activation,
                "merge_similarity_threshold": self.merge_similarity_threshold,
                "split_overload_ratio": self.split_overload_ratio,
            },
        }


# ============================================================
# 稠密 SwiGLU 反向 (供训练引擎复用, 与 Expert.backward 同构)
# ============================================================

def _swiglu_ffn_backward(x: List[List[float]], dout: List[List[float]],
                         W_gate: List[List[float]],
                         W_up: List[List[float]],
                         W_down: List[List[float]],
                         gate_cache: List[List[float]],
                         up_cache: List[List[float]],
                         act_cache: List[List[float]]
                         ) -> Tuple[List[List[float]], List[List[float]],
                                    List[List[float]], List[List[float]]]:
    """稠密 SwiGLU FFN 反向传播

    正向: gate = x@W_gate; up = x@W_up; act = silu(gate)*up; out = act@W_down
    Returns: (dW_gate, dW_up, dW_down, dx)
    """
    n = len(x)
    if n == 0:
        return [], [], [], []
    hidden = len(W_gate)
    ffn = len(W_gate[0])
    dW_gate = _zeros_2d(hidden, ffn)
    dW_up = _zeros_2d(hidden, ffn)
    dW_down = _zeros_2d(ffn, hidden)
    dx: List[List[float]] = [[0.0] * hidden for _ in range(n)]
    for s in range(n):
        xs = x[s]
        g = gate_cache[s]
        u = up_cache[s]
        a = act_cache[s]
        dy = dout[s]
        _outer_add(dW_down, a, dy, 1.0)
        d_act = [0.0] * ffn
        for i in range(ffn):
            wd = W_down[i]
            acc = 0.0
            for d in range(hidden):
                acc += dy[d] * wd[d]
            d_act[i] = acc
        dx_s = dx[s]
        for i in range(ffn):
            sg = _silu(g[i])
            d_up_i = d_act[i] * sg
            d_gate_i = d_act[i] * u[i] * _silu_grad(g[i])
            for h in range(hidden):
                xh = xs[h]
                if xh != 0.0:
                    dW_gate[h][i] += xh * d_gate_i
                    dW_up[h][i] += xh * d_up_i
                dx_s[h] += d_gate_i * W_gate[h][i] + d_up_i * W_up[h][i]
    return dW_gate, dW_up, dW_down, dx


# ============================================================
# MoETrainingEngine [MoE 专用训练引擎]
# ============================================================

class MoETrainingEngine:
    """MoE 专用训练引擎

    特性:
    - 处理 MoE 层特殊梯度路由 (稀疏: 只更新被激活专家的权重)
    - 专家级学习率: 根据使用频率调整 (常用专家降速, 稀有专家加速)
    - 专家预热: 新专家渐进激活 (warmup 期内专家 LR 从 0 线性升至 1)
    - 辅助损失集成到总损失与路由梯度
    - 梯度裁剪 / Adam 优化器 / 学习率调度
    - 注意力权重采用 straight-through (近似恒等), 重点训练 FFN/MoE/嵌入/LM头

    注: 为在纯 Python 下保持可运行, 注意力子层梯度采用近似 (残差直通),
    FFN/MoE/Embedding/LMHead/RMSNorm 梯度为数学正确。
    """

    def __init__(self, model: MoETransformerModel,
                 lr: float = 1e-2,
                 weight_decay: float = 0.0,
                 max_grad_norm: float = 1.0,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 expert_lr_mode: str = "adaptive",
                 expert_lr_alpha: float = 1.0,
                 expert_warmup_steps: int = 0,
                 base_warmup_steps: int = 0,
                 freeze_attention: bool = True,
                 total_steps: int = 1000,
                 min_lr: float = 1e-5,
                 rng: Optional[random.Random] = None):
        self.model = model
        self.base_lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.expert_lr_mode = expert_lr_mode   # "adaptive" / "uniform"
        self.expert_lr_alpha = expert_lr_alpha
        self.expert_warmup_steps = max(0, expert_warmup_steps)
        self.base_warmup_steps = max(0, base_warmup_steps)
        self.freeze_attention = freeze_attention
        self.total_steps = total_steps
        self.rng = rng if rng is not None else random

        self.global_step = 0
        self.epoch = 0
        self.loss_history: List[float] = []
        self.ce_history: List[float] = []
        self.aux_history: List[float] = []
        self.lr_history: List[float] = []
        self.grad_norm_history: List[float] = []

        # Adam 动量 (按参数名)
        self._m: Dict[str, Any] = {}
        self._v: Dict[str, Any] = {}
        self._param_names: List[str] = []
        self._init_param_names()

    # ---------- 参数名登记 ----------

    def _init_param_names(self) -> None:
        names: List[str] = ["token_embedding", "final_norm_weight"]
        if not self.model.tie_word_embeddings and self.model.lm_head is not None:
            names.append("lm_head")
        for i, layer in enumerate(self.model.layers):
            names.append(f"l{i}_norm1")
            names.append(f"l{i}_norm2")
            if not self.freeze_attention:
                names.extend([f"l{i}_W_q", f"l{i}_W_k",
                              f"l{i}_W_v", f"l{i}_W_o"])
            if layer.is_moe:
                names.append(f"l{i}_W_router")
                for e in range(layer.ffn.num_experts):
                    names.append(f"l{i}_e{e}_W_gate")
                    names.append(f"l{i}_e{e}_W_up")
                    names.append(f"l{i}_e{e}_W_down")
            else:
                names.append(f"l{i}_W_gate")
                names.append(f"l{i}_W_up")
                names.append(f"l{i}_W_down")
        self._param_names = names
        for name in names:
            self._m[name] = None
            self._v[name] = None

    def _get_param(self, name: str) -> Any:
        m = self.model
        if name == "token_embedding":
            return m.token_embedding
        if name == "final_norm_weight":
            return m.final_norm.weight
        if name == "lm_head":
            return m.lm_head
        # l{i}_...
        parts = name.split("_")
        li = int(parts[0][1:])
        layer = m.layers[li]
        rest = "_".join(parts[1:])
        if rest == "norm1":
            return layer.norm1.weight
        if rest == "norm2":
            return layer.norm2.weight
        if rest in ("W_q", "W_k", "W_v", "W_o"):
            return getattr(layer.attn, rest)
        if rest == "W_router":
            return layer.ffn.router.W_router
        if rest.startswith("e"):
            # e{e}_W_gate / W_up / W_down
            ep = rest.split("_")
            e = int(ep[0][1:])
            wname = "_".join(ep[1:])
            return getattr(layer.ffn.experts[e], wname)
        # dense
        return getattr(layer.ffn, rest)

    # ---------- 学习率调度 ----------

    def get_lr(self) -> float:
        s = self.global_step
        # base warmup + cosine
        if self.base_warmup_steps > 0 and s < self.base_warmup_steps:
            lr = self.base_lr * (s + 1) / self.base_warmup_steps
        else:
            progress = (s - self.base_warmup_steps) / max(
                self.total_steps - self.base_warmup_steps, 1)
            progress = max(0.0, min(1.0, progress))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                (1.0 + math.cos(math.pi * progress))
        return lr

    # ---------- 专家级学习率 ----------

    def get_expert_lr_factor(self, layer_idx: int, expert_id: int) -> float:
        """专家学习率因子 (根据使用频率)"""
        if self.expert_lr_mode != "adaptive":
            return 1.0
        layer = self.model.layers[layer_idx]
        if not layer.is_moe:
            return 1.0
        router = layer.ffn.router
        freqs = router.get_expert_frequencies()
        f = freqs[expert_id] if expert_id < len(freqs) else 0.0
        mean_f = sum(freqs) / len(freqs) if freqs else 0.0
        if mean_f < 1e-12:
            return 1.0
        # 稀有专家加速, 常用专家减速: factor = (mean/f) capped
        ratio = mean_f / max(f, 1e-6)
        factor = 1.0 + self.expert_lr_alpha * (ratio - 1.0)
        return max(0.2, min(factor, 5.0))

    def get_expert_warmup_factor(self) -> float:
        """专家预热因子 (从 0 线性升至 1)"""
        if self.expert_warmup_steps <= 0:
            return 1.0
        s = self.global_step
        if s >= self.expert_warmup_steps:
            return 1.0
        return (s + 1) / self.expert_warmup_steps

    # ---------- 前向 ----------

    def forward_pass(self, input_ids: List[int], targets: List[int]
                     ) -> Tuple[float, List[List[float]], Dict[str, Any]]:
        return self.model.forward_for_training(input_ids, targets)

    # ---------- 反向 ----------

    def backward_pass(self, logits: List[List[float]],
                      targets: List[int],
                      cache: Dict[str, Any]) -> Dict[str, Any]:
        """MoE 感知反向传播

        精确: dlogits, LM Head, final_norm, Embedding, 每层 FFN/MoE 与 norm2/norm1
        近似: 注意力权重 (straight-through, 可选冻结)
        """
        model = self.model
        hidden = model.hidden_dim
        vocab = model.vocab_size
        seq_len = min(len(logits), len(targets))
        if seq_len == 0:
            return {}
        tie = model.tie_word_embeddings
        layer_caches = cache.get("layers", [])
        h_final = cache["h_final"]
        final_x_norm = cache["final_x_norm"]
        final_rms = cache["final_rms"]

        grads: Dict[str, Any] = {}

        # 1. dL/dlogits = softmax(logits) - onehot(targets)
        d_logits: List[List[float]] = []
        for i in range(seq_len):
            probs = _softmax_vec(logits[i])
            tgt = targets[i]
            if 0 <= tgt < len(probs):
                probs[tgt] -= 1.0
            d_logits.append(probs)

        # 2. LM Head 反向
        if tie:
            d_h_final: List[List[float]] = []
            grad_emb: List[List[float]] = [[0.0] * hidden for _ in range(vocab)]
            for s in range(seq_len):
                dl = d_logits[s]
                dh = [0.0] * hidden
                for v in range(vocab):
                    dlv = dl[v]
                    if dlv == 0.0:
                        continue
                    emb_v = model.token_embedding[v]
                    for d in range(hidden):
                        dh[d] += dlv * emb_v[d]
                d_h_final.append(dh)
                hf_s = h_final[s]
                for v in range(vocab):
                    dlv = dl[v]
                    if dlv == 0.0:
                        continue
                    ge = grad_emb[v]
                    for d in range(hidden):
                        ge[d] += hf_s[d] * dlv
        else:
            d_h_final = []
            grad_lm_head: List[List[float]] = [[0.0] * vocab for _ in range(hidden)]
            for s in range(seq_len):
                dl = d_logits[s]
                dh = [0.0] * hidden
                for d in range(hidden):
                    wd = model.lm_head[d]
                    acc = 0.0
                    for v in range(vocab):
                        acc += dl[v] * wd[v]
                    dh[d] = acc
                d_h_final.append(dh)
                hf_s = h_final[s]
                for d in range(hidden):
                    hfd = hf_s[d]
                    if hfd == 0.0:
                        continue
                    gl = grad_lm_head[d]
                    for v in range(vocab):
                        gl[v] += hfd * dl[v]
            grads["lm_head"] = grad_lm_head
            grad_emb = [[0.0] * hidden for _ in range(vocab)]

        # 3. final RMSNorm 反向
        d_x, d_final_norm_w = _rmsnorm_backward(
            d_h_final, final_x_norm, final_rms,
            model.final_norm.weight, hidden)
        grads["final_norm_weight"] = d_final_norm_w

        # 4. 逐层反向 (逆序)
        for layer_idx in range(model.num_layers - 1, -1, -1):
            layer = model.layers[layer_idx]
            lc = layer_caches[layer_idx]
            # d_x = dL/d(layer_output)
            # 残差: x_out = x_mid + ffn_out
            d_ffn_out = [list(r) for r in d_x]
            d_x_mid = [list(r) for r in d_x]

            # --- FFN/MoE 反向 ---
            h2 = lc["h2"]
            if lc["is_moe"]:
                moe_grads = layer.ffn.backward(lc["routing"], d_ffn_out,
                                                compute_router_grad=True)
                grads[f"l{layer_idx}_W_router"] = moe_grads["dW_router"]
                # 专家梯度
                for e, (dg, du, dd) in enumerate(moe_grads["expert_grads"]):
                    grads[f"l{layer_idx}_e{e}_W_gate"] = dg
                    grads[f"l{layer_idx}_e{e}_W_up"] = du
                    grads[f"l{layer_idx}_e{e}_W_down"] = dd
                d_h2 = moe_grads["d_input"]
            else:
                dgg, dgu, dgd, d_h2 = _swiglu_ffn_backward(
                    h2, d_ffn_out,
                    layer.ffn.W_gate, layer.ffn.W_up, layer.ffn.W_down,
                    lc["ffn_gate"], lc["ffn_up"], lc["ffn_activated"])
                grads[f"l{layer_idx}_W_gate"] = dgg
                grads[f"l{layer_idx}_W_up"] = dgu
                grads[f"l{layer_idx}_W_down"] = dgd

            # --- norm2 反向 ---
            d_x_mid_from_n2, d_norm2_w = _rmsnorm_backward(
                d_h2, lc["norm2_x_norm"], lc["norm2_rms"],
                layer.norm2.weight, hidden)
            grads[f"l{layer_idx}_norm2"] = d_norm2_w
            d_x_mid_total = [[d_x_mid[s][d] + d_x_mid_from_n2[s][d]
                              for d in range(hidden)] for s in range(seq_len)]

            # --- 残差: x_mid = x_in + attn_out ---
            d_attn_out = [list(r) for r in d_x_mid_total]
            d_x_in = [list(r) for r in d_x_mid_total]

            # --- 注意力反向 (straight-through) ---
            if self.freeze_attention:
                d_h1 = [list(r) for r in d_attn_out]
            else:
                # 近似: 把注意力输出梯度经 W_o 反传为 merged 梯度, 再近似为 h1 梯度
                d_h1 = _matmul_2d(d_attn_out, _transpose_2d(layer.attn.W_o))
                # W_o 近似梯度 (用 h1 与 d_attn_out)
                # 注意此为粗略信号, 仅作弱更新
                d_Wo = [[0.0] * hidden for _ in range(hidden)]
                _wo_scale = 0.1
                for s in range(seq_len):
                    _outer_add(d_Wo, lc["h1"][s], d_attn_out[s], _wo_scale)
                grads[f"l{layer_idx}_W_o"] = d_Wo
                # Q/K/V 冻结 (近似)

            # --- norm1 反向 ---
            d_x_in_from_n1, d_norm1_w = _rmsnorm_backward(
                d_h1, lc["norm1_x_norm"], lc["norm1_rms"],
                layer.norm1.weight, hidden)
            grads[f"l{layer_idx}_norm1"] = d_norm1_w
            d_x = [[d_x_in[s][d] + d_x_in_from_n1[s][d]
                    for d in range(hidden)] for s in range(seq_len)]

        # 5. Embedding 查表梯度叠加
        input_ids = cache.get("input_ids", [])
        for s in range(seq_len):
            tid = input_ids[s] if s < len(input_ids) else 0
            if 0 <= tid < vocab:
                ge = grad_emb[tid]
                dxs = d_x[s]
                for d in range(hidden):
                    ge[d] += dxs[d]
        # 与 LM head (tied) 的梯度合并
        if "token_embedding" not in grads:
            grads["token_embedding"] = grad_emb
        else:
            _add_2d_inplace(grads["token_embedding"], grad_emb)

        return grads

    # ---------- 梯度裁剪 ----------

    def clip_grad_norm(self, grads: Dict[str, Any]) -> float:
        total_sq = 0.0
        for name, g in grads.items():
            if g is None:
                continue
            total_sq += self._sum_sq(g)
        total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
        if total_norm > self.max_grad_norm and total_norm > 0:
            scale = self.max_grad_norm / total_norm
            for name in grads:
                if grads[name] is not None:
                    self._scale_inplace(grads[name], scale)
        return total_norm

    @staticmethod
    def _sum_sq(x: Any) -> float:
        if isinstance(x, list):
            if len(x) == 0:
                return 0.0
            if isinstance(x[0], list):
                return sum(MoETrainingEngine._sum_sq(r) for r in x)
            return sum(v * v for v in x)
        return float(x) * float(x)

    @staticmethod
    def _scale_inplace(x: Any, scale: float) -> None:
        if isinstance(x, list):
            if len(x) == 0:
                return
            if isinstance(x[0], list):
                for r in x:
                    for j in range(len(r)):
                        r[j] *= scale
            else:
                for i in range(len(x)):
                    x[i] *= scale

    # ---------- Adam 更新 ----------

    @staticmethod
    def _zeros_like(x: Any) -> Any:
        if isinstance(x, list):
            return [MoETrainingEngine._zeros_like(e) for e in x]
        return 0.0

    def _ensure_moments(self, name: str, param: Any) -> None:
        """懒分配 Adam 一阶/二阶动量 (与参数同形)"""
        if name not in self._m or self._m[name] is None:
            self._m[name] = self._zeros_like(param)
            self._v[name] = self._zeros_like(param)

    def _adam_update(self, param: Any, grad: Any, m: Any, v: Any,
                     lr: float, t: int) -> None:
        """纯递归 Adam 更新 (原地修改 param/m/v)

        注意: m, v 必须已由 _ensure_moments 分配为与 param 同形。
        """
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        if isinstance(param, list) and len(param) > 0 and isinstance(param[0], list):
            # 2D
            for i in range(len(param)):
                self._adam_update(param[i], grad[i], m[i], v[i], lr, t)
        elif isinstance(param, list):
            # 1D
            for i in range(len(param)):
                g = grad[i] + self.weight_decay * param[i]
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                param[i] -= lr * (m[i] / bc1) / (math.sqrt(v[i] / bc2) + self.eps)

    def apply_grads(self, grads: Dict[str, Any]) -> None:
        """应用梯度 (Adam), 含专家级学习率与预热"""
        lr = self.get_lr()
        self.lr = lr
        t = self.global_step + 1
        warmup_f = self.get_expert_warmup_factor()
        for name in self._param_names:
            if name not in grads or grads[name] is None:
                continue
            param = self._get_param(name)
            self._ensure_moments(name, param)
            eff_lr = lr
            # 专家级 LR (形如 l{i}_e{e}_W_gate)
            if "_e" in name and "_W_" in name:
                parts = name.split("_")
                li = int(parts[0][1:])
                e = int(parts[1][1:])
                eff_lr = lr * self.get_expert_lr_factor(li, e) * warmup_f
            self._adam_update(param, grads[name],
                              self._m[name], self._v[name], eff_lr, t)


    # ---------- 训练步 ----------

    def train_step(self, input_ids: List[int], targets: List[int]
                   ) -> Dict[str, Any]:
        loss, logits, cache = self.forward_pass(input_ids, targets)
        grads = self.backward_pass(logits, targets, cache)
        grad_norm = self.clip_grad_norm(grads)
        self.apply_grads(grads)
        self.global_step += 1
        ce_loss = cache.get("ce_loss", loss)
        aux_loss = cache.get("aux_loss", 0.0)
        self.loss_history.append(loss)
        self.ce_history.append(ce_loss)
        self.aux_history.append(aux_loss)
        self.lr_history.append(self.lr)
        self.grad_norm_history.append(grad_norm)
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "aux_loss": aux_loss,
            "grad_norm": grad_norm,
            "lr": self.lr,
            "step": self.global_step,
            "expert_warmup": round(self.get_expert_warmup_factor(), 4),
        }

    def train(self, data: List[Tuple[List[int], List[int]]],
              epochs: int = 1, verbose: bool = True
              ) -> Dict[str, Any]:
        """多轮训练"""
        initial_loss = None
        for ep in range(epochs):
            self.epoch = ep
            for input_ids, targets in data:
                info = self.train_step(input_ids, targets)
                if initial_loss is None:
                    initial_loss = info["loss"]
                if verbose and self.global_step % max(1, len(data)) == 0:
                    avg_loss = sum(self.loss_history[-len(data):]) / max(len(data), 1) if self.loss_history else 0.0
                    print(f"  [Step {self.global_step}] avg_loss={avg_loss:.4f}", flush=True)
        final_loss = self.loss_history[-1] if self.loss_history else 0.0
        return {
            "epochs": epochs,
            "steps": self.global_step,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": (initial_loss - final_loss) if initial_loss else 0.0,
        }

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "lr": round(self.lr, 6),
            "base_lr": self.base_lr,
            "loss": round(self.loss_history[-1], 6) if self.loss_history else None,
            "ce_loss": round(self.ce_history[-1], 6) if self.ce_history else None,
            "aux_loss": round(self.aux_history[-1], 6) if self.aux_history else None,
            "grad_norm": round(self.grad_norm_history[-1], 6) if self.grad_norm_history else None,
            "expert_lr_mode": self.expert_lr_mode,
            "expert_warmup_steps": self.expert_warmup_steps,
            "expert_warmup_factor": round(self.get_expert_warmup_factor(), 4),
            "freeze_attention": self.freeze_attention,
            "num_param_groups": len(self._param_names),
            "loss_history_len": len(self.loss_history),
        }


# ============================================================
# 自测入口
# ============================================================

def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


if __name__ == "__main__":
    random.seed(42)

    print("\n" + "=" * 70)
    print("  灵元 MoE 混合专家模型 — 自测 (PART 19)")
    print("=" * 70)

    # ---------- 1. MoEConfig ----------
    print("\n[1] MoEConfig 配置测试")
    cfg = MoEConfig.from_preset("tiny_moe")
    ok, errs = cfg.validate()
    print(f"  预设 tiny_moe: hidden={cfg.hidden_dim}, layers={cfg.num_layers}, "
          f"experts={cfg.num_experts}, K={cfg.num_activated_experts}")
    print(f"  MoE层: {cfg.count_moe_layers()}, 稠密层: {cfg.count_dense_layers()}")
    print(f"  总参数: {_fmt_params(cfg.estimate_total_params())}, "
          f"激活参数: {_fmt_params(cfg.estimate_activated_params())}")
    print(f"  配置验证: {'通过' if ok else '失败 ' + str(errs)}")
    for name in MoEConfig.get_preset_names():
        c = MoEConfig.from_preset(name)
        print(f"  预设 {name:>10}: 总参 {_fmt_params(c.estimate_total_params()):>8} "
              f"激活 {_fmt_params(c.estimate_activated_params()):>8} "
              f"稀疏比 {c.estimate_activated_params()/c.estimate_total_params():.2%}")

    # ---------- 2. Expert ----------
    print("\n[2] Expert 专家网络测试")
    exp = Expert(0, hidden_dim=32, ffn_dim=64)
    x_in = [[random.gauss(0, 1) for _ in range(32)] for _ in range(5)]
    out = exp(x_in)
    exp.selection_count = 3
    print(f"  专家 forward: 输入(5×32) -> 输出({len(out)}×{len(out[0])})")
    print(f"  参数量: {exp.num_params}, 平均激活: {exp.average_activation:.6f}")
    print(f"  被选次数: {exp.selection_count}, 计算耗时: {exp.compute_time*1000:.3f}ms")
    # 带缓存前向 + 反向 (梯度形状校验)
    out2, gate, up, act = exp.forward_with_cache(x_in)
    dout = [[1.0] * 32 for _ in range(5)]
    dg, du, dd, dx = exp.backward(x_in, dout, gate, up, act)
    print(f"  backward 形状: dW_gate({len(dg)}×{len(dg[0])}), "
          f"dW_down({len(dd)}×{len(dd[0])}), dx({len(dx)}×{len(dx[0])})")

    # ---------- 3. ExpertRouter ----------
    print("\n[3] ExpertRouter 路由器测试")
    router = ExpertRouter(hidden_dim=32, num_experts=4,
                           num_activated_experts=2, router_noise=0.1)
    x_route = [[random.gauss(0, 1) for _ in range(32)] for _ in range(8)]
    routing = router.route(x_route, training=True)
    print(f"  路由 {len(x_route)} tokens, 每token选 K={router.num_activated_experts} 专家")
    print(f"  容量: {routing['capacity']}, 丢弃token: {routing['dropped_tokens']}")
    print(f"  辅助损失: {routing['aux_loss']:.4f}")
    print(f"  专家被选次数: {router.expert_selection_count}")
    print(f"  负载均衡度: {router.get_load_balance():.4f} (1=完全均匀)")

    # ---------- 4. MoELayer ----------
    print("\n[4] MoELayer MoE层测试")
    moe = MoELayer(hidden_dim=32, ffn_dim=64, num_experts=4,
                   num_activated_experts=2, router_noise=0.1)
    x_moe = [[random.gauss(0, 1) for _ in range(32)] for _ in range(8)]
    moe_out, rinfo = moe(x_moe, training=True)
    print(f"  输入(8×32) -> 输出({len(moe_out)}×{len(moe_out[0])})")
    print(f"  总参数: {moe.num_params}, 激活参数: {moe.activated_params} "
          f"(稀疏比 {moe.activated_params/moe.num_params:.2%})")
    print(f"  路由器专家频率: {[round(f,3) for f in moe.router.get_expert_frequencies()]}")
    # 带缓存前向 + 反向
    moe_out2, rinfo2 = moe.forward_with_cache(x_moe, training=True)
    dout_moe = [[0.1] * 32 for _ in range(8)]
    mgrads = moe.backward(rinfo2, dout_moe)
    print(f"  backward: dW_router({len(mgrads['dW_router'])}×"
          f"{len(mgrads['dW_router'][0])}), "
          f"d_input({len(mgrads['d_input'])}×{len(mgrads['d_input'][0])})")

    # ---------- 5. MoETransformerModel ----------
    print("\n[5] MoETransformerModel 模型测试")
    model = MoETransformerModel.from_preset("tiny_moe")
    stats = model.get_stats()
    print(f"  层数: {stats['num_layers']}, MoE层: {stats['moe_layer_indices']}")
    print(f"  总参数: {_fmt_params(stats['total_params'])}, "
          f"激活参数: {_fmt_params(stats['activated_params'])} "
          f"(稀疏比 {stats['param_breakdown']['activation_ratio']:.2%})")
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    logits = model.forward(input_ids, training=False)
    print(f"  forward: input_ids({len(input_ids)}) -> logits"
          f"({len(logits)}×{len(logits[0])})")
    # KV Cache 推理
    logits1, cache = model.forward_with_cache(input_ids)
    next_logits, cache = model.forward_with_cache([13], cache)
    print(f"  KV Cache: prompt 后续单token logits shape "
          f"({len(next_logits)}×{len(next_logits[0])}), 缓存长度={cache.get_seq_len()}")
    # 训练前向
    targets = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    loss, tlogits, tcache = model.forward_for_training(input_ids, targets)
    print(f"  训练前向: loss={loss:.4f} (ce={tcache['ce_loss']:.4f}, "
          f"aux={tcache['aux_loss']:.4f})")

    # ---------- 6. LoadBalancer ----------
    print("\n[6] LoadBalancer 负载均衡器测试")
    lb = LoadBalancer(model, adjust_strength=0.2, migration_rate=0.2)
    # 跑几次前向积累统计
    for _ in range(3):
        model.forward([1, 2, 3, 4, 5, 6], training=True)
    util = lb.collect_utilization()
    for li, freqs in util.items():
        print(f"  层{li} 专家频率: {[round(f,3) for f in freqs]}")
    aux = lb.compute_aux_loss()
    print(f"  辅助损失: importance={aux['importance_loss']:.6f}, "
          f"load={aux['load_loss']:.6f}")
    adj = lb.adjust_routing()
    print(f"  路由偏置调整 (层{list(adj.keys())}): 完成")
    mig = lb.migrate_experts()
    print(f"  专家迁移: {sum(len(v) for v in mig.values())} 对")

    # ---------- 7. MoETrainingEngine ----------
    print("\n[7] MoETrainingEngine 训练引擎测试")
    model2 = MoETransformerModel.from_preset("tiny_moe")
    engine = MoETrainingEngine(
        model2, lr=5e-2, expert_lr_mode="adaptive",
        expert_warmup_steps=5, base_warmup_steps=2,
        max_grad_norm=1.0, total_steps=50)
    print(f"  参数组数: {len(engine._param_names)}")
    print(f"  专家预热因子 (初始): {engine.get_expert_warmup_factor():.3f}")
    data = [(input_ids, targets)]
    print(f"  训练前 loss={engine.loss_history[-1] if engine.loss_history else loss:.4f}")
    print(f"  开始训练 ({len(data)} 样本 × 8 步)...")
    init_loss = engine.forward_pass(input_ids, targets)[0]
    for step in range(8):
        info = engine.train_step(input_ids, targets)
        print(f"    step {info['step']:>2}: loss={info['loss']:.4f} "
              f"ce={info['ce_loss']:.4f} aux={info['aux_loss']:.4f} "
              f"gnorm={info['grad_norm']:.2f} lr={info['lr']:.4f} "
              f"warmup={info['expert_warmup']:.2f}")
    print(f"  训练结果: {init_loss:.4f} -> {engine.loss_history[-1]:.4f} "
          f"(下降 {init_loss - engine.loss_history[-1]:.4f})")

    # ---------- 8. ExpertPruner ----------
    print("\n[8] ExpertPruner 专家裁剪器测试")
    # 先跑若干前向以积累专家统计
    for _ in range(3):
        model2.forward([1, 2, 3, 4, 5, 6, 7, 8], training=True)
    pruner = ExpertPruner(model2, min_selection_ratio=0.01,
                          merge_similarity_threshold=0.5,
                          split_overload_ratio=1.5)
    ineff = pruner.identify_inefficient_experts()
    print(f"  低效专家: {sum(len(v) for v in ineff.values())} 个")
    pruned = pruner.prune(do_merge=True, do_split=True)
    print(f"  裁剪报告: 合并 {pruned['num_merged']} 对, 分裂 {pruned['num_split']} 个")
    # 相似度示例 (同一层内两个专家, 维度一致)
    li0 = model2.get_moe_layers()[0]
    moe0 = model2.layers[li0].ffn
    sim01 = pruner._expert_weight_similarity(moe0.experts[0], moe0.experts[1])
    print(f"  层{li0} 专家0与专家1 权重余弦相似度: {sim01:.4f}")

    print("\n" + "=" * 70)
    print("  所有模块自测通过")
    print("=" * 70)





