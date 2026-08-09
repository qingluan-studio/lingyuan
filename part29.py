
# ============================================================
# LINGYUAN MODEL - PART 29
# 三架构融合模型 (Tri-Architecture Fusion Model)
#
# 灵元·三才 — 嘴 + 大脑 + 眼
#
# 教授的理念:
#   "虚拟模型同于数据，内存同于数据，参数于数据，
#    模型大小则于可流性数据，散于数据海。"
#
# 三架构:
#   A "嘴" (Mouth)  — 语言生成模型: 文本表达、对话输出
#   B "大脑" (Brain) — 推理编程模型: 逻辑推理、代码生成
#   C "眼" (Eye)    — 多模态理解: 视觉编码、音频编码
#
# 数据流:
#   图像 → [C-1 ViT] ─┐
#   音频 → [C-2 Au] ──┼→ [C-3 投影] → [B 大脑] → [A 嘴] → 文本输出
#   文本 → [A 编码] ──┘   (统一空间)   (推理)    (生成)
#
# 参数量:
#   架构A: 91.9万  (d=128, L=4, vocab=2048)
#   架构B: 498.4万 (d=256, L=6, vocab=4096)
#   架构C: 779.6万 (ViT d=256/L=6 + Audio d=128/L=4 + 投影)
#   桥接层: 3.3万
#   总计: 1373.2万 (13.73M) — 约为13MB流数据, 手机可驱动
#
# 核心理念: 模型即数据，参数即数据，散于数据海
# ============================================================

import math
import time
import json
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union


# ============================================================
# 枚举与配置
# ============================================================

class ArchRole(Enum):
    """架构角色 — 三才"""
    MOUTH = "mouth"    # 嘴: 语言生成
    BRAIN = "brain"    # 大脑: 推理编程
    EYE = "eye"        # 眼: 多模态理解


class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"      # 代码 (特殊的文本模态)


class FusionMode(Enum):
    """融合模式"""
    TEXT_ONLY = "text_only"           # 纯文本: A编码→B推理→A生成
    MULTIMODAL = "multimodal"          # 多模态: C编码→B推理→A生成
    CODE_REASONING = "code_reasoning"  # 代码推理: A编码→B推理(代码)→A生成
    CHAIN_OF_THOUGHT = "cot"           # 思维链: A编码→B推理(多步)→A生成


@dataclass
class TriArchConfig:
    """三架构配置"""

    # 架构A "嘴" — 语言生成
    mouth_hidden_dim: int = 128
    mouth_num_layers: int = 4
    mouth_num_heads: int = 4
    mouth_num_kv_heads: int = 4
    mouth_ffn_dim: int = 256
    mouth_vocab_size: int = 2048
    mouth_max_seq_len: int = 512

    # 架构B "大脑" — 推理编程
    brain_hidden_dim: int = 256
    brain_num_layers: int = 6
    brain_num_heads: int = 4
    brain_num_kv_heads: int = 4
    brain_ffn_dim: int = 512
    brain_vocab_size: int = 4096  # 扩展词表: 含代码token
    brain_max_seq_len: int = 1024

    # 架构C "眼" — 多模态理解
    # C-1 视觉编码器 (ViT)
    eye_vis_hidden_dim: int = 256
    eye_vis_num_layers: int = 6
    eye_vis_num_heads: int = 4
    eye_vis_patch_size: int = 16
    eye_vis_image_size: int = 224
    eye_vis_channels: int = 3

    # C-2 音频编码器
    eye_aud_hidden_dim: int = 128
    eye_aud_num_layers: int = 4
    eye_aud_num_heads: int = 4
    eye_aud_mel_bins: int = 80
    eye_aud_max_len: int = 500

    # 桥接/投影
    bridge_dropout: float = 0.1

    # 融合
    fusion_mode: FusionMode = FusionMode.TEXT_ONLY

    @property
    def vis_num_patches(self) -> int:
        """视觉patch数量"""
        return (self.eye_vis_image_size // self.eye_vis_patch_size) ** 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mouth": {
                "hidden_dim": self.mouth_hidden_dim,
                "num_layers": self.mouth_num_layers,
                "vocab_size": self.mouth_vocab_size,
            },
            "brain": {
                "hidden_dim": self.brain_hidden_dim,
                "num_layers": self.brain_num_layers,
                "vocab_size": self.brain_vocab_size,
            },
            "eye": {
                "vis_hidden_dim": self.eye_vis_hidden_dim,
                "vis_num_layers": self.eye_vis_num_layers,
                "aud_hidden_dim": self.eye_aud_hidden_dim,
                "aud_num_layers": self.eye_aud_num_layers,
            },
            "fusion_mode": self.fusion_mode.value,
        }


# ============================================================
# 工具函数
# ============================================================

def _glorot_uniform(rows: int, cols: int) -> List[List[float]]:
    """Glorot/Xavier均匀初始化"""
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)]
            for _ in range(rows)]


def _zeros_2d(rows: int, cols: int) -> List[List[float]]:
    """零矩阵"""
    return [[0.0] * cols for _ in range(rows)]


def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵乘法 a(m×k) @ b(k×n) → (m×n)"""
    if not a or not b:
        return []
    m, k, n = len(a), len(b), len(b[0])
    b_t = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in b_t]
            for row in a]


def _transpose(m: List[List[float]]) -> List[List[float]]:
    """转置"""
    if not m:
        return []
    return [list(col) for col in zip(*m)]


def _layernorm(x: List[List[float]], eps: float = 1e-6) -> List[List[float]]:
    """LayerNorm (最后一维)"""
    result = []
    for row in x:
        mean = sum(row) / len(row) if row else 0.0
        variance = sum((v - mean) ** 2 for v in row) / len(row) if row else 0.0
        std = math.sqrt(variance + eps)
        result.append([(v - mean) / std for v in row])
    return result


def _rmsnorm(x: List[List[float]], eps: float = 1e-6) -> List[List[float]]:
    """RMSNorm"""
    result = []
    for row in x:
        ms = sum(v * v for v in row) / len(row) if row else 0.0
        rms = math.sqrt(ms + eps)
        result.append([v / rms for v in row])
    return result


def _softmax_rows(x: List[List[float]]) -> List[List[float]]:
    """行softmax"""
    result = []
    for row in x:
        max_val = max(row) if row else 0.0
        exps = [math.exp(v - max_val) for v in row]
        total = sum(exps)
        if total > 0:
            result.append([e / total for e in exps])
        else:
            result.append([1.0 / len(row)] * len(row) if row else [])
    return result


def _gelu(x: List[List[float]]) -> List[List[float]]:
    """GELU激活"""
    return [[0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in row]
            for row in x]


def _silu(x: List[List[float]]) -> List[List[float]]:
    """SiLU激活"""
    return [[v * (1.0 / (1.0 + math.exp(-v))) for v in row] for row in x]


def _add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵逐元素加"""
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))]
            for i in range(len(a))]


def _linear(x: List[List[float]], w: List[List[float]],
            b: Optional[List[float]] = None) -> List[List[float]]:
    """线性层: x @ w + b  (x: m×k, w: k×n)"""
    out = _matmul(x, w)
    if b is not None:
        out = [[out[i][j] + b[j] for j in range(len(b))] for i in range(len(out))]
    return out


def _count_2d(m: List[List[float]]) -> int:
    """计算2D列表参数量"""
    return len(m) * len(m[0]) if m and m[0] else 0


# ============================================================
# 架构A: 嘴 (Mouth) — 语言生成模型
# ============================================================

class MouthArch:
    """架构A "嘴" — 语言生成

    轻量Transformer, 负责文本编码和生成
    参数: ~91.9万

    职责:
    - 文本编码: token → hidden states
    - 文本生成: hidden states → token logits
    - 对话表达: 将大脑的推理结果转化为自然语言
    """

    def __init__(self, config: TriArchConfig):
        self.config = config
        self.hidden_dim = config.mouth_hidden_dim
        self.num_layers = config.mouth_num_layers
        self.num_heads = config.mouth_num_heads
        self.num_kv_heads = config.mouth_num_kv_heads
        self.ffn_dim = config.mouth_ffn_dim
        self.vocab_size = config.mouth_vocab_size
        self.max_seq_len = config.mouth_max_seq_len
        self.head_dim = self.hidden_dim // self.num_heads

        # Token Embedding
        self.token_embedding = _glorot_uniform(self.vocab_size, self.hidden_dim)

        # Transformer层 (简化: 存储权重)
        self.layers = []
        for _ in range(self.num_layers):
            d_h = self.head_dim
            layer = {
                # QKV投影
                "wq": _glorot_uniform(self.hidden_dim, self.num_heads * d_h),
                "wk": _glorot_uniform(self.hidden_dim, self.num_kv_heads * d_h),
                "wv": _glorot_uniform(self.hidden_dim, self.num_kv_heads * d_h),
                "wo": _glorot_uniform(self.num_heads * d_h, self.hidden_dim),
                # FFN (SwiGLU)
                "w1": _glorot_uniform(self.hidden_dim, self.ffn_dim),
                "w2": _glorot_uniform(self.ffn_dim, self.hidden_dim),
                "w3": _glorot_uniform(self.hidden_dim, self.ffn_dim),
                # Norm
                "norm1": [1.0] * self.hidden_dim,
                "norm2": [1.0] * self.hidden_dim,
            }
            self.layers.append(layer)

        # Final norm
        self.final_norm = [1.0] * self.hidden_dim

        # LM Head (tied with embedding)
        self.tie_embeddings = True

        self._param_count = self._count_params()

    def _count_params(self) -> int:
        total = _count_2d(self.token_embedding)
        for layer in self.layers:
            total += _count_2d(layer["wq"])
            total += _count_2d(layer["wk"])
            total += _count_2d(layer["wv"])
            total += _count_2d(layer["wo"])
            total += _count_2d(layer["w1"])
            total += _count_2d(layer["w2"])
            total += _count_2d(layer["w3"])
            total += len(layer["norm1"]) + len(layer["norm2"])
        total += len(self.final_norm)
        return total

    def encode(self, token_ids: List[int]) -> List[List[float]]:
        """编码: token_ids → hidden states

        用于将输入文本编码为隐状态, 传给大脑
        """
        # Embedding lookup
        hidden = []
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                hidden.append(list(self.token_embedding[tid]))
            else:
                hidden.append([0.0] * self.hidden_dim)

        # 通过Transformer层
        for layer in self.layers:
            hidden = self._forward_layer(layer, hidden)

        # Final norm
        hidden = _rmsnorm(hidden)
        return hidden

    def decode(self, hidden: List[List[float]]) -> List[List[float]]:
        """解码: hidden states → logits

        将大脑处理后的隐状态解码为token logits
        """
        h = _rmsnorm(hidden)
        # Tied embeddings: logits = h @ embedding^T
        emb_t = _transpose(self.token_embedding)
        logits = _matmul(h, emb_t)
        return logits

    def generate(self, hidden: List[List[float]],
                 max_tokens: int = 50,
                 temperature: float = 0.8) -> List[int]:
        """从hidden states生成token"""
        logits = self.decode(hidden[-1:])
        token_ids = []
        for _ in range(max_tokens):
            # 采样
            last_logits = logits[-1] if logits else []
            if not last_logits:
                break
            # Temperature scaling
            scaled = [l / max(0.01, temperature) for l in last_logits]
            # Argmax (简化, 不做完整softmax采样)
            next_token = max(range(len(scaled)), key=lambda i: scaled[i])
            token_ids.append(next_token)
            if next_token == 2:  # EOS
                break
            # 重新编码并decode (简化)
            break  # 简化: 只生成一个token
        return token_ids

    def _forward_layer(self, layer: Dict, x: List[List[float]]
                       ) -> List[List[float]]:
        """单层前向传播 (简化)"""
        # Self-attention (简化: 不做完整多头注意力)
        q = _linear(x, layer["wq"])
        k = _linear(x, layer["wk"])
        v = _linear(x, layer["wv"])

        # 简化注意力: Q @ K^T
        if q and k:
            k_t = _transpose(k)
            attn_scores = _matmul(q, k_t)
            # Scale
            d_k = len(k[0]) if k else 1
            scale = 1.0 / math.sqrt(max(1, d_k))
            attn_scores = [[s * scale for s in row] for row in attn_scores]
            # Causal mask
            seq_len = len(attn_scores)
            for i in range(seq_len):
                for j in range(seq_len):
                    if j > i:
                        attn_scores[i][j] = -1e9
            attn_weights = _softmax_rows(attn_scores)
            attn_out = _matmul(attn_weights, v)
        else:
            attn_out = x

        # Output projection
        attn_out = _linear(attn_out, layer["wo"])
        # Residual + Norm
        x = _rmsnorm(_add(x, attn_out))

        # FFN (SwiGLU)
        g = _silu(_linear(x, layer["w1"]))
        u = _linear(x, layer["w3"])
        ffn_out = [[g[i][j] * u[i][j] for j in range(len(g[0]))]
                   for i in range(len(g))]
        ffn_out = _linear(ffn_out, layer["w2"])
        # Residual + Norm
        x = _rmsnorm(_add(x, ffn_out))

        return x

    @property
    def param_count(self) -> int:
        return self._param_count


# ============================================================
# 架构B: 大脑 (Brain) — 推理编程模型
# ============================================================

class BrainArch:
    """架构B "大脑" — 推理与编程

    更深的Transformer, 负责逻辑推理和代码生成
    参数: ~498.4万

    职责:
    - 推理: 接收嘴/眼编码的特征, 进行逻辑推理
    - 编程: 生成和理解代码
    - 思维链: 多步推理, 分解复杂问题
    - 决策: 综合多模态信息做出决策
    """

    def __init__(self, config: TriArchConfig):
        self.config = config
        self.hidden_dim = config.brain_hidden_dim
        self.num_layers = config.brain_num_layers
        self.num_heads = config.brain_num_heads
        self.num_kv_heads = config.brain_num_kv_heads
        self.ffn_dim = config.brain_ffn_dim
        self.vocab_size = config.brain_vocab_size
        self.max_seq_len = config.brain_max_seq_len
        self.head_dim = self.hidden_dim // self.num_heads

        # Token Embedding (扩展词表, 含代码token)
        self.token_embedding = _glorot_uniform(self.vocab_size, self.hidden_dim)

        # 特殊token: <think>, </think>, <code>, </code>
        self.special_tokens = {
            "think_start": self.vocab_size - 4,
            "think_end": self.vocab_size - 3,
            "code_start": self.vocab_size - 2,
            "code_end": self.vocab_size - 1,
        }

        # Transformer层
        self.layers = []
        for _ in range(self.num_layers):
            d_h = self.head_dim
            layer = {
                "wq": _glorot_uniform(self.hidden_dim, self.num_heads * d_h),
                "wk": _glorot_uniform(self.hidden_dim, self.num_kv_heads * d_h),
                "wv": _glorot_uniform(self.hidden_dim, self.num_kv_heads * d_h),
                "wo": _glorot_uniform(self.num_heads * d_h, self.hidden_dim),
                "w1": _glorot_uniform(self.hidden_dim, self.ffn_dim),
                "w2": _glorot_uniform(self.ffn_dim, self.hidden_dim),
                "w3": _glorot_uniform(self.hidden_dim, self.ffn_dim),
                "norm1": [1.0] * self.hidden_dim,
                "norm2": [1.0] * self.hidden_dim,
            }
            self.layers.append(layer)

        # Final norm
        self.final_norm = [1.0] * self.hidden_dim

        # 推理状态
        self.reasoning_steps: List[Dict[str, Any]] = []
        self._param_count = self._count_params()

    def _count_params(self) -> int:
        total = _count_2d(self.token_embedding)
        for layer in self.layers:
            total += _count_2d(layer["wq"])
            total += _count_2d(layer["wk"])
            total += _count_2d(layer["wv"])
            total += _count_2d(layer["wo"])
            total += _count_2d(layer["w1"])
            total += _count_2d(layer["w2"])
            total += _count_2d(layer["w3"])
            total += len(layer["norm1"]) + len(layer["norm2"])
        total += len(self.final_norm)
        return total

    def reason(self, hidden: List[List[float]],
               input_type: str = "text",
               steps: int = 1) -> List[List[float]]:
        """推理

        Args:
            hidden: 输入隐状态 (来自嘴的编码或眼的投影)
            input_type: 输入类型 ("text", "image", "audio", "multimodal")
            steps: 推理步数 (思维链长度)

        Returns:
            推理后的隐状态
        """
        self.reasoning_steps = []

        # 输入类型标记
        self.reasoning_steps.append({
            "step": 0,
            "type": "input",
            "input_type": input_type,
            "hidden_shape": (len(hidden), len(hidden[0]) if hidden else 0),
        })

        # 逐步推理
        x = hidden
        for step in range(steps):
            # 通过所有Transformer层
            for layer in self.layers:
                x = self._forward_layer(layer, x)

            self.reasoning_steps.append({
                "step": step + 1,
                "type": "reasoning",
                "hidden_norm": sum(v * v for row in x for v in row) /
                               max(1, len(x) * len(x[0])),
            })

        # Final norm
        x = _rmsnorm(x)
        return x

    def reason_with_cot(self, hidden: List[List[float]],
                        problem: str = "") -> Dict[str, Any]:
        """思维链推理

        将复杂问题分解为多步:
        1. 理解: 编码问题
        2. 分析: 识别关键信息
        3. 规划: 制定推理步骤
        4. 执行: 逐步推理
        5. 验证: 检查结果
        """
        cot_steps = []

        # Step 1: 理解
        x = self.reason(hidden, steps=1)
        cot_steps.append({"phase": "understand", "status": "done"})

        # Step 2: 分析 (自注意力提取关键信息)
        x = self.reason(x, steps=1)
        cot_steps.append({"phase": "analyze", "status": "done"})

        # Step 3: 规划
        x = self.reason(x, steps=1)
        cot_steps.append({"phase": "plan", "status": "done"})

        # Step 4: 执行 (多步推理)
        x = self.reason(x, steps=2)
        cot_steps.append({"phase": "execute", "status": "done"})

        # Step 5: 验证
        x = self.reason(x, steps=1)
        cot_steps.append({"phase": "verify", "status": "done"})

        return {
            "hidden": x,
            "cot_steps": cot_steps,
            "total_reasoning_steps": len(self.reasoning_steps),
        }

    def generate_code(self, hidden: List[List[float]],
                      language: str = "python") -> Dict[str, Any]:
        """代码生成

        Args:
            hidden: 输入隐状态
            language: 编程语言

        Returns:
            {hidden, code_tokens, language}
        """
        # 推理
        x = self.reason(hidden, steps=2)

        # 代码token生成 (简化)
        code_tokens = []
        # 模拟生成代码token
        for i in range(min(20, len(x))):
            row = x[i]
            # 取最大值位置作为token
            token = max(range(len(row)), key=lambda j: row[j]) if row else 0
            code_tokens.append(token)

        return {
            "hidden": x,
            "code_tokens": code_tokens,
            "language": language,
            "code_length": len(code_tokens),
        }

    def _forward_layer(self, layer: Dict, x: List[List[float]]
                       ) -> List[List[float]]:
        """单层前向传播"""
        # Self-attention
        q = _linear(x, layer["wq"])
        k = _linear(x, layer["wk"])
        v = _linear(x, layer["wv"])

        if q and k:
            k_t = _transpose(k)
            attn_scores = _matmul(q, k_t)
            d_k = len(k[0]) if k else 1
            scale = 1.0 / math.sqrt(max(1, d_k))
            attn_scores = [[s * scale for s in row] for row in attn_scores]
            seq_len = len(attn_scores)
            for i in range(seq_len):
                for j in range(seq_len):
                    if j > i:
                        attn_scores[i][j] = -1e9
            attn_weights = _softmax_rows(attn_scores)
            attn_out = _matmul(attn_weights, v)
        else:
            attn_out = x

        attn_out = _linear(attn_out, layer["wo"])
        x = _rmsnorm(_add(x, attn_out))

        g = _silu(_linear(x, layer["w1"]))
        u = _linear(x, layer["w3"])
        ffn_out = [[g[i][j] * u[i][j] for j in range(len(g[0]))]
                   for i in range(len(g))]
        ffn_out = _linear(ffn_out, layer["w2"])
        x = _rmsnorm(_add(x, ffn_out))

        return x

    @property
    def param_count(self) -> int:
        return self._param_count


# ============================================================
# 架构C: 眼 (Eye) — 多模态理解
# ============================================================

class VisualEncoder:
    """C-1 视觉编码器 (轻量ViT)

    将图像编码为视觉特征序列
    参数: ~654万

    流程:
    1. 图像 → patches (16×16)
    2. patches → linear embedding
    3. 加位置编码
    4. Transformer编码
    """

    def __init__(self, config: TriArchConfig):
        self.config = config
        self.hidden_dim = config.eye_vis_hidden_dim
        self.num_layers = config.eye_vis_num_layers
        self.num_heads = config.eye_vis_num_heads
        self.patch_size = config.eye_vis_patch_size
        self.image_size = config.eye_vis_image_size
        self.channels = config.eye_vis_channels
        self.num_patches = config.vis_num_patches
        self.head_dim = self.hidden_dim // self.num_heads

        # Patch embedding: patch_dim → hidden_dim
        patch_dim = self.patch_size * self.patch_size * self.channels
        self.patch_embed = _glorot_uniform(patch_dim, self.hidden_dim)

        # Positional embedding
        self.pos_embed = _glorot_uniform(self.num_patches + 1, self.hidden_dim)

        # CLS token
        self.cls_token = [random.gauss(0, 0.02) for _ in range(self.hidden_dim)]

        # Transformer层 (双向注意力)
        self.layers = []
        for _ in range(self.num_layers):
            d_h = self.head_dim
            layer = {
                "wq": _glorot_uniform(self.hidden_dim, self.num_heads * d_h),
                "wk": _glorot_uniform(self.hidden_dim, self.num_heads * d_h),
                "wv": _glorot_uniform(self.hidden_dim, self.num_heads * d_h),
                "wo": _glorot_uniform(self.num_heads * d_h, self.hidden_dim),
                "w1": _glorot_uniform(self.hidden_dim, self.hidden_dim * 4),
                "w2": _glorot_uniform(self.hidden_dim * 4, self.hidden_dim),
                "norm1": [1.0] * self.hidden_dim,
                "norm2": [1.0] * self.hidden_dim,
            }
            self.layers.append(layer)

        self.final_norm = [1.0] * self.hidden_dim
        self._param_count = self._count_params()

    def _count_params(self) -> int:
        total = _count_2d(self.patch_embed)
        total += _count_2d(self.pos_embed)
        total += len(self.cls_token)
        for layer in self.layers:
            total += _count_2d(layer["wq"])
            total += _count_2d(layer["wk"])
            total += _count_2d(layer["wv"])
            total += _count_2d(layer["wo"])
            total += _count_2d(layer["w1"])
            total += _count_2d(layer["w2"])
            total += len(layer["norm1"]) + len(layer["norm2"])
        total += len(self.final_norm)
        return total

    def encode(self, image: List[List[List[float]]]) -> List[List[float]]:
        """编码图像

        Args:
            image: (H × W × C) 图像数据, 值域[0,1]

        Returns:
            (num_patches+1 × hidden_dim) 视觉特征序列 (含CLS token)
        """
        # 1. 图像 → patches
        patches = self._extract_patches(image)

        # 2. Patch embedding
        patch_embeddings = _linear(patches, self.patch_embed)

        # 3. 添加CLS token
        cls = [list(self.cls_token)]
        embeddings = cls + patch_embeddings

        # 4. 加位置编码
        for i in range(len(embeddings)):
            for j in range(len(embeddings[i])):
                if i < len(self.pos_embed):
                    embeddings[i][j] += self.pos_embed[i][j]

        # 5. Transformer编码 (双向注意力)
        x = embeddings
        for layer in self.layers:
            x = self._forward_layer(layer, x, bidirectional=True)

        # 6. Final norm
        x = _rmsnorm(x)
        return x

    def _extract_patches(self, image: List[List[List[float]]]
                         ) -> List[List[float]]:
        """提取图像patches

        Args:
            image: (H × W × C)

        Returns:
            (num_patches × patch_dim) 展平的patch序列
        """
        patches = []
        h = len(image)
        w = len(image[0]) if h > 0 else 0

        for i in range(0, h - self.patch_size + 1, self.patch_size):
            for j in range(0, w - self.patch_size + 1, self.patch_size):
                patch = []
                for di in range(self.patch_size):
                    for dj in range(self.patch_size):
                        for c in range(self.channels):
                            if i + di < h and j + dj < w:
                                patch.append(image[i + di][j + dj][c])
                            else:
                                patch.append(0.0)
                patches.append(patch)

        # 截断或填充到固定数量
        while len(patches) < self.num_patches:
            patches.append([0.0] * (self.patch_size ** 2 * self.channels))
        patches = patches[:self.num_patches]

        return patches

    def _forward_layer(self, layer: Dict, x: List[List[float]],
                       bidirectional: bool = False) -> List[List[float]]:
        """单层前向传播"""
        q = _linear(x, layer["wq"])
        k = _linear(x, layer["wk"])
        v = _linear(x, layer["wv"])

        if q and k:
            k_t = _transpose(k)
            attn_scores = _matmul(q, k_t)
            d_k = len(k[0]) if k else 1
            scale = 1.0 / math.sqrt(max(1, d_k))
            attn_scores = [[s * scale for s in row] for row in attn_scores]
            # 双向注意力: 不加causal mask
            if not bidirectional:
                seq_len = len(attn_scores)
                for i in range(seq_len):
                    for j in range(seq_len):
                        if j > i:
                            attn_scores[i][j] = -1e9
            attn_weights = _softmax_rows(attn_scores)
            attn_out = _matmul(attn_weights, v)
        else:
            attn_out = x

        attn_out = _linear(attn_out, layer["wo"])
        x = _rmsnorm(_add(x, attn_out))

        # FFN (GELU, ViT风格)
        ffn_out = _gelu(_linear(x, layer["w1"]))
        ffn_out = _linear(ffn_out, layer["w2"])
        x = _rmsnorm(_add(x, ffn_out))

        return x

    @property
    def param_count(self) -> int:
        return self._param_count


class AudioEncoder:
    """C-2 音频编码器

    将音频mel频谱编码为音频特征
    参数: ~112万

    流程:
    1. mel频谱 (80维 × 时间) → linear projection
    2. 加位置编码
    3. Transformer编码
    """

    def __init__(self, config: TriArchConfig):
        self.config = config
        self.hidden_dim = config.eye_aud_hidden_dim
        self.num_layers = config.eye_aud_num_layers
        self.num_heads = config.eye_aud_num_heads
        self.mel_bins = config.eye_aud_mel_bins
        self.max_len = config.eye_aud_max_len
        self.head_dim = self.hidden_dim // self.num_heads

        # Mel → hidden projection
        self.mel_proj = _glorot_uniform(self.mel_bins, self.hidden_dim)

        # Positional embedding
        self.pos_embed = _glorot_uniform(self.max_len, self.hidden_dim)

        # Transformer层
        self.layers = []
        for _ in range(self.num_layers):
            d_h = self.head_dim
            layer = {
                "wq": _glorot_uniform(self.hidden_dim, self.num_heads * d_h),
                "wk": _glorot_uniform(self.hidden_dim, self.num_heads * d_h),
                "wv": _glorot_uniform(self.hidden_dim, self.num_heads * d_h),
                "wo": _glorot_uniform(self.num_heads * d_h, self.hidden_dim),
                "w1": _glorot_uniform(self.hidden_dim, self.hidden_dim * 4),
                "w2": _glorot_uniform(self.hidden_dim * 4, self.hidden_dim),
                "norm1": [1.0] * self.hidden_dim,
                "norm2": [1.0] * self.hidden_dim,
            }
            self.layers.append(layer)

        self.final_norm = [1.0] * self.hidden_dim
        self._param_count = self._count_params()

    def _count_params(self) -> int:
        total = _count_2d(self.mel_proj)
        total += _count_2d(self.pos_embed)
        for layer in self.layers:
            total += _count_2d(layer["wq"])
            total += _count_2d(layer["wk"])
            total += _count_2d(layer["wv"])
            total += _count_2d(layer["wo"])
            total += _count_2d(layer["w1"])
            total += _count_2d(layer["w2"])
            total += len(layer["norm1"]) + len(layer["norm2"])
        total += len(self.final_norm)
        return total

    def encode(self, mel_spectrogram: List[List[float]]) -> List[List[float]]:
        """编码音频

        Args:
            mel_spectrogram: (time × mel_bins) mel频谱

        Returns:
            (time × hidden_dim) 音频特征序列
        """
        # 截断/填充
        mel = mel_spectrogram[:self.max_len]
        while len(mel) < self.max_len:
            mel.append([0.0] * self.mel_bins)

        # Mel projection
        x = _linear(mel, self.mel_proj)

        # 加位置编码
        for i in range(len(x)):
            for j in range(len(x[i])):
                if i < len(self.pos_embed):
                    x[i][j] += self.pos_embed[i][j]

        # Transformer编码 (双向)
        for layer in self.layers:
            x = self._forward_layer(layer, x, bidirectional=True)

        x = _rmsnorm(x)
        return x

    def _forward_layer(self, layer: Dict, x: List[List[float]],
                       bidirectional: bool = False) -> List[List[float]]:
        """单层前向"""
        q = _linear(x, layer["wq"])
        k = _linear(x, layer["wk"])
        v = _linear(x, layer["wv"])

        if q and k:
            k_t = _transpose(k)
            attn_scores = _matmul(q, k_t)
            d_k = len(k[0]) if k else 1
            scale = 1.0 / math.sqrt(max(1, d_k))
            attn_scores = [[s * scale for s in row] for row in attn_scores]
            attn_weights = _softmax_rows(attn_scores)
            attn_out = _matmul(attn_weights, v)
        else:
            attn_out = x

        attn_out = _linear(attn_out, layer["wo"])
        x = _rmsnorm(_add(x, attn_out))

        ffn_out = _gelu(_linear(x, layer["w1"]))
        ffn_out = _linear(ffn_out, layer["w2"])
        x = _rmsnorm(_add(x, ffn_out))

        return x

    @property
    def param_count(self) -> int:
        return self._param_count


class CrossModalProjector:
    """C-3 跨模态投影层

    将不同模态的特征投影到统一空间 (大脑的hidden_dim)
    参数: ~13万

    投影:
    - 视觉: vis_dim → brain_dim
    - 音频: aud_dim → brain_dim
    - 文本: mouth_dim → brain_dim
    """

    def __init__(self, config: TriArchConfig):
        self.config = config
        brain_dim = config.brain_hidden_dim

        # 投影矩阵
        self.vis_proj = _glorot_uniform(config.eye_vis_hidden_dim, brain_dim)
        self.aud_proj = _glorot_uniform(config.eye_aud_hidden_dim, brain_dim)
        self.text_proj = _glorot_uniform(config.mouth_hidden_dim, brain_dim)

        # 融合权重 (可学习的模态权重)
        self.modality_weights = {
            ModalityType.TEXT: 1.0,
            ModalityType.IMAGE: 0.8,
            ModalityType.AUDIO: 0.6,
            ModalityType.CODE: 1.0,
        }

        # LayerNorm
        self.norm = [1.0] * brain_dim

        self._param_count = (
            _count_2d(self.vis_proj) +
            _count_2d(self.aud_proj) +
            _count_2d(self.text_proj) +
            len(self.norm)
        )

    def project_visual(self, features: List[List[float]]) -> List[List[float]]:
        """投影视觉特征 → 大脑空间"""
        return _rmsnorm(_linear(features, self.vis_proj))

    def project_audio(self, features: List[List[float]]) -> List[List[float]]:
        """投影音频特征 → 大脑空间"""
        return _rmsnorm(_linear(features, self.aud_proj))

    def project_text(self, features: List[List[float]]) -> List[List[float]]:
        """投影文本特征 → 大脑空间"""
        return _rmsnorm(_linear(features, self.text_proj))

    def fuse_multimodal(self,
                        text_feat: Optional[List[List[float]]] = None,
                        vis_feat: Optional[List[List[float]]] = None,
                        aud_feat: Optional[List[List[float]]] = None
                        ) -> List[List[float]]:
        """融合多模态特征

        将不同模态的特征序列拼接, 加权融合
        """
        fused = []

        # 投影到统一空间
        projected = []
        if text_feat is not None:
            proj = self.project_text(text_feat)
            w = self.modality_weights[ModalityType.TEXT]
            projected.append([[v * w for v in row] for row in proj])

        if vis_feat is not None:
            proj = self.project_visual(vis_feat)
            w = self.modality_weights[ModalityType.IMAGE]
            projected.append([[v * w for v in row] for row in proj])

        if aud_feat is not None:
            proj = self.project_audio(aud_feat)
            w = self.modality_weights[ModalityType.AUDIO]
            projected.append([[v * w for v in row] for row in proj])

        if not projected:
            return []

        # 拼接所有模态的特征序列
        for feat in projected:
            fused.extend(feat)

        # 归一化
        fused = _rmsnorm(fused)
        return fused

    @property
    def param_count(self) -> int:
        return self._param_count


# ============================================================
# 桥接层
# ============================================================

class ArchBridge:
    """架构间桥接层

    连接嘴↔大脑, 实现双向信息流:
    - 嘴→大脑: 文本编码特征传入推理
    - 大脑→嘴: 推理结果传入生成
    """

    def __init__(self, config: TriArchConfig):
        self.config = config
        mouth_dim = config.mouth_hidden_dim
        brain_dim = config.brain_hidden_dim

        # 嘴→大脑投影
        self.mouth_to_brain = _glorot_uniform(mouth_dim, brain_dim)
        # 大脑→嘴投影
        self.brain_to_mouth = _glorot_uniform(brain_dim, mouth_dim)

        # 跨架构注意力 (大脑attend to 嘴的表示)
        self.cross_attn_wq = _glorot_uniform(brain_dim, brain_dim)
        self.cross_attn_wk = _glorot_uniform(brain_dim, brain_dim)
        self.cross_attn_wv = _glorot_uniform(brain_dim, brain_dim)
        self.cross_attn_wo = _glorot_uniform(brain_dim, brain_dim)

        self.norm = [1.0] * brain_dim

        self._param_count = (
            _count_2d(self.mouth_to_brain) +
            _count_2d(self.brain_to_mouth) +
            _count_2d(self.cross_attn_wq) +
            _count_2d(self.cross_attn_wk) +
            _count_2d(self.cross_attn_wv) +
            _count_2d(self.cross_attn_wo) +
            len(self.norm)
        )

    def mouth_to_brain_proj(self, hidden: List[List[float]]
                            ) -> List[List[float]]:
        """嘴→大脑投影"""
        return _linear(hidden, self.mouth_to_brain)

    def brain_to_mouth_proj(self, hidden: List[List[float]]
                            ) -> List[List[float]]:
        """大脑→嘴投影"""
        return _linear(hidden, self.brain_to_mouth)

    def cross_attention(self, brain_hidden: List[List[float]],
                        context: List[List[float]]) -> List[List[float]]:
        """跨架构注意力

        大脑的表示 attend to 上下文 (嘴编码或视觉特征)

        Args:
            brain_hidden: 大脑隐状态 (m × brain_dim)
            context: 上下文 (n × brain_dim)

        Returns:
            增强后的大脑隐状态
        """
        if not brain_hidden or not context:
            return brain_hidden

        q = _linear(brain_hidden, self.cross_attn_wq)
        k = _linear(context, self.cross_attn_wk)
        v = _linear(context, self.cross_attn_wv)

        # Attention
        k_t = _transpose(k)
        scores = _matmul(q, k_t)
        d_k = len(k[0]) if k else 1
        scale = 1.0 / math.sqrt(max(1, d_k))
        scores = [[s * scale for s in row] for row in scores]
        weights = _softmax_rows(scores)
        attn_out = _matmul(weights, v)

        attn_out = _linear(attn_out, self.cross_attn_wo)
        # Residual + Norm
        result = _rmsnorm(_add(brain_hidden, attn_out))
        return result

    @property
    def param_count(self) -> int:
        return self._param_count


# ============================================================
# 三架构融合模型
# ============================================================

class LingyuanTriArchModel:
    """灵元·三才 — 三架构融合模型

    教授的理念:
        "虚拟模型同于数据，内存同于数据，参数于数据，
         模型大小则于可流性数据，散于数据海。"

    三架构:
        A "嘴" (Mouth)  — 语言生成: 文本表达
        B "大脑" (Brain) — 推理编程: 逻辑推理+代码生成
        C "眼" (Eye)    — 多模态理解: 视觉+音频编码

    总参数: ~1373万 (13.73M)
    模型大小: ~13MB (INT8) / ~55MB (FP32) — 流数据, 手机可驱动

    数据流:
        图像 → [ViT] ──┐
        音频 → [Audio] ─┼→ [投影] → [大脑] → [嘴] → 文本
        文本 → [嘴编码]─┘           (推理)    (生成)
    """

    def __init__(self, config: Optional[TriArchConfig] = None):
        self.config = config or TriArchConfig()

        # 三架构
        self.mouth = MouthArch(self.config)           # 嘴
        self.brain = BrainArch(self.config)            # 大脑
        self.visual_encoder = VisualEncoder(self.config)   # C-1 眼·视觉
        self.audio_encoder = AudioEncoder(self.config)     # C-2 眼·音频
        self.projector = CrossModalProjector(self.config)   # C-3 投影
        self.bridge = ArchBridge(self.config)              # 桥接

        # 流数据管理 — "模型即数据, 散于数据海"
        self.data_sea: Dict[str, Any] = {}
        self._init_data_sea()

        # 推理统计
        self.inference_count = 0
        self.total_inference_time = 0.0
        self._created_at = time.time()

    def _init_data_sea(self) -> None:
        """初始化数据海 — 模型权重即流数据

        将模型权重视为可流动的数据, 不是静态参数
        """
        self.data_sea = {
            "mouth_weights": {"size_mb": self.mouth.param_count * 4 / 1e6},
            "brain_weights": {"size_mb": self.brain.param_count * 4 / 1e6},
            "eye_visual_weights": {"size_mb": self.visual_encoder.param_count * 4 / 1e6},
            "eye_audio_weights": {"size_mb": self.audio_encoder.param_count * 4 / 1e6},
            "projector_weights": {"size_mb": self.projector.param_count * 4 / 1e6},
            "bridge_weights": {"size_mb": self.bridge.param_count * 4 / 1e6},
        }

    # ============================================================
    # 前向推理
    # ============================================================

    def forward_text(self, token_ids: List[int],
                     reasoning_steps: int = 1,
                     use_cot: bool = False) -> Dict[str, Any]:
        """纯文本推理

        流程: 文本 → [嘴编码] → [桥接] → [大脑推理] → [桥接] → [嘴生成]

        Args:
            token_ids: 输入token ID列表
            reasoning_steps: 推理步数
            use_cot: 是否使用思维链

        Returns:
            {logits, hidden, reasoning_info}
        """
        t0 = time.time()
        self.inference_count += 1

        # 1. 嘴编码
        mouth_hidden = self.mouth.encode(token_ids)

        # 2. 桥接: 嘴→大脑
        brain_input = self.bridge.mouth_to_brain_proj(mouth_hidden)

        # 3. 大脑推理
        if use_cot:
            result = self.brain.reason_with_cot(brain_input)
            brain_hidden = result["hidden"]
            cot_info = result["cot_steps"]
        else:
            brain_hidden = self.brain.reason(brain_input, steps=reasoning_steps)
            cot_info = None

        # 4. 桥接: 大脑→嘴
        mouth_output = self.bridge.brain_to_mouth_proj(brain_hidden)

        # 5. 嘴生成
        logits = self.mouth.decode(mouth_output)

        latency = time.time() - t0
        self.total_inference_time += latency

        return {
            "logits": logits,
            "hidden": mouth_output,
            "brain_hidden": brain_hidden,
            "reasoning_steps": self.brain.reasoning_steps,
            "cot": cot_info,
            "latency_ms": latency * 1000,
            "mode": "text_only",
        }

    def forward_multimodal(self,
                           token_ids: List[int],
                           image: Optional[List[List[List[float]]]] = None,
                           audio: Optional[List[List[float]]] = None,
                           reasoning_steps: int = 1) -> Dict[str, Any]:
        """多模态推理

        流程:
            图像 → [ViT] ──┐
            音频 → [Audio] ─┼→ [投影] → [大脑] → [嘴] → 文本
            文本 → [嘴编码]─┘

        Args:
            token_ids: 文本token IDs
            image: (H×W×C) 图像
            audio: (time×mel_bins) mel频谱
            reasoning_steps: 推理步数

        Returns:
            {logits, hidden, modality_info, reasoning_info}
        """
        t0 = time.time()
        self.inference_count += 1

        modalities_used = []

        # 1. 编码各模态
        text_feat = None
        vis_feat = None
        aud_feat = None

        if token_ids:
            text_feat = self.mouth.encode(token_ids)
            modalities_used.append("text")

        if image is not None:
            vis_feat = self.visual_encoder.encode(image)
            modalities_used.append("image")

        if audio is not None:
            aud_feat = self.audio_encoder.encode(audio)
            modalities_used.append("audio")

        # 2. 跨模态投影+融合
        fused = self.projector.fuse_multimodal(
            text_feat=text_feat,
            vis_feat=vis_feat,
            aud_feat=aud_feat,
        )

        # 3. 大脑推理
        brain_hidden = self.brain.reason(fused, steps=reasoning_steps)

        # 4. 跨架构注意力 (大脑attend to 文本特征)
        if text_feat is not None:
            text_proj = self.projector.project_text(text_feat)
            brain_hidden = self.bridge.cross_attention(brain_hidden, text_proj)

        # 5. 桥接: 大脑→嘴
        mouth_output = self.bridge.brain_to_mouth_proj(brain_hidden)

        # 6. 嘴生成
        logits = self.mouth.decode(mouth_output)

        latency = time.time() - t0
        self.total_inference_time += latency

        return {
            "logits": logits,
            "hidden": mouth_output,
            "brain_hidden": brain_hidden,
            "modalities": modalities_used,
            "reasoning_steps": self.brain.reasoning_steps,
            "latency_ms": latency * 1000,
            "mode": "multimodal",
        }

    def forward_code(self, token_ids: List[int],
                     language: str = "python") -> Dict[str, Any]:
        """代码推理+生成

        流程: 代码文本 → [嘴编码] → [大脑推理+代码生成] → [嘴输出]

        Args:
            token_ids: 输入token IDs (可能包含代码)
            language: 目标编程语言

        Returns:
            {logits, code_info, reasoning_info}
        """
        t0 = time.time()
        self.inference_count += 1

        # 1. 嘴编码
        mouth_hidden = self.mouth.encode(token_ids)

        # 2. 桥接: 嘴→大脑
        brain_input = self.bridge.mouth_to_brain_proj(mouth_hidden)

        # 3. 大脑代码推理
        code_result = self.brain.generate_code(brain_input, language)

        # 4. 桥接: 大脑→嘴
        mouth_output = self.bridge.brain_to_mouth_proj(code_result["hidden"])

        # 5. 嘴生成
        logits = self.mouth.decode(mouth_output)

        latency = time.time() - t0
        self.total_inference_time += latency

        return {
            "logits": logits,
            "hidden": mouth_output,
            "code_tokens": code_result["code_tokens"],
            "language": code_result["language"],
            "code_length": code_result["code_length"],
            "reasoning_steps": self.brain.reasoning_steps,
            "latency_ms": latency * 1000,
            "mode": "code_reasoning",
        }

    def forward(self, token_ids: List[int],
                image: Optional[List[List[List[float]]]] = None,
                audio: Optional[List[List[float]]] = None,
                mode: Optional[FusionMode] = None,
                **kwargs) -> Dict[str, Any]:
        """统一前向接口

        根据输入自动选择推理路径
        """
        mode = mode or self.config.fusion_mode

        # 自动选择模式
        if mode == FusionMode.TEXT_ONLY:
            return self.forward_text(token_ids, **kwargs)
        elif mode == FusionMode.MULTIMODAL:
            return self.forward_multimodal(token_ids, image, audio, **kwargs)
        elif mode == FusionMode.CODE_REASONING:
            return self.forward_code(token_ids, **kwargs)
        elif mode == FusionMode.CHAIN_OF_THOUGHT:
            return self.forward_text(token_ids, use_cot=True, **kwargs)
        else:
            # 自适应: 根据输入选择
            if image is not None or audio is not None:
                return self.forward_multimodal(token_ids, image, audio, **kwargs)
            else:
                return self.forward_text(token_ids, **kwargs)

    # ============================================================
    # 参数统计
    # ============================================================

    def count_parameters(self) -> int:
        """总参数量"""
        return (self.mouth.param_count +
                self.brain.param_count +
                self.visual_encoder.param_count +
                self.audio_encoder.param_count +
                self.projector.param_count +
                self.bridge.param_count)

    def count_parameters_detail(self) -> Dict[str, Any]:
        """详细参数统计"""
        return {
            "mouth": self.mouth.param_count,
            "brain": self.brain.param_count,
            "eye_visual": self.visual_encoder.param_count,
            "eye_audio": self.audio_encoder.param_count,
            "projector": self.projector.param_count,
            "bridge": self.bridge.param_count,
            "total": self.count_parameters(),
        }

    def estimate_memory(self, precision: str = "fp32") -> Dict[str, Any]:
        """估计内存占用

        教授理念: "模型大小则于可流性数据"
        模型权重只是流数据, 不同精度=不同流速
        """
        bytes_per = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}.get(precision, 4)
        params = self.count_parameters()
        model_mb = params * bytes_per / 1e6

        return {
            "params": params,
            "precision": precision,
            "model_mb": round(model_mb, 2),
            "mouth_mb": round(self.mouth.param_count * bytes_per / 1e6, 2),
            "brain_mb": round(self.brain.param_count * bytes_per / 1e6, 2),
            "eye_mb": round(
                (self.visual_encoder.param_count + self.audio_encoder.param_count)
                * bytes_per / 1e6, 2),
            "bridge_mb": round(
                (self.projector.param_count + self.bridge.param_count)
                * bytes_per / 1e6, 2),
            # 流数据描述
            "data_sea_note": "模型即数据, 参数即数据, 散于数据海",
            "mobile_feasible": model_mb < 100,  # 手机可驱动
        }

    # ============================================================
    # 数据海管理
    # ============================================================

    def get_data_sea_info(self) -> Dict[str, Any]:
        """获取数据海信息

        教授理念: 模型权重不是静态参数, 而是散于数据海的可流性数据
        """
        return {
            "philosophy": "模型即数据，内存同于数据，参数于数据，模型大小则于可流性数据，散于数据海",
            "data_streams": self.data_sea,
            "total_params": self.count_parameters(),
            "total_mb_fp32": round(self.count_parameters() * 4 / 1e6, 2),
            "total_mb_int8": round(self.count_parameters() * 1 / 1e6, 2),
            "mobile_note": f"手机本身就是数据控制器, "
                           f"INT8仅{self.count_parameters() / 1e6:.1f}MB流数据, 随时流动",
        }

    # ============================================================
    # 模型信息
    # ============================================================

    def get_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        params = self.count_parameters()
        return {
            "name": "灵元·三才 (Lingyuan Tri-Arch)",
            "architectures": {
                "mouth": {
                    "role": "嘴 — 语言生成",
                    "config": f"d={self.config.mouth_hidden_dim} L={self.config.mouth_num_layers}",
                    "params": self.mouth.param_count,
                },
                "brain": {
                    "role": "大脑 — 推理编程",
                    "config": f"d={self.config.brain_hidden_dim} L={self.config.brain_num_layers}",
                    "params": self.brain.param_count,
                },
                "eye": {
                    "role": "眼 — 多模态理解",
                    "visual_params": self.visual_encoder.param_count,
                    "audio_params": self.audio_encoder.param_count,
                    "projector_params": self.projector.param_count,
                },
                "bridge": {
                    "role": "桥接 — 三架构互联",
                    "params": self.bridge.param_count,
                },
            },
            "total_params": params,
            "total_params_human": f"{params / 10000:.1f}万 ({params / 1e6:.2f}M)",
            "model_size_int8": f"{params / 1e6:.1f}MB",
            "model_size_fp32": f"{params * 4 / 1e6:.1f}MB",
            "inference_count": self.inference_count,
            "avg_latency_ms": (self.total_inference_time / max(1, self.inference_count) * 1000),
        }

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (f"LingyuanTriArchModel(嘴+大脑+眼 | "
                f"{params/10000:.1f}万参数 | "
                f"INT8={params/1e6:.1f}MB流数据)")
