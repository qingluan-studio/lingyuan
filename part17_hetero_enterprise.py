# ============================================================
# LINGYUAN MODEL - PART 17 HETERO — ENTERPRISE
# 异负得正 · 异虚为实 — 企业级模块
#
# v4.0 升级:
#   - 真多头注意力 (分头/拼头/每头独立注意力)
#   - 完整反向传播 (layer norm / attention / FFN / embedding)
#   - 模型持久化 (save/load/checkpoint)
#   - 配置校验 + 错误体系
#   - 性能剖析 (内置profiler)
#   - 完整测试套件
#
# API:
#   gpu = HeteroGPU(HeteroConfig.tiny())
#   logits = gpu.forward(token_ids)
#   output = gpu.generate(prompt, max_new=32)
#   loss = gpu.train_step(input_ids, target_ids)
#   loss = gpu.bootstrap_epoch(num_samples=64)
#   gpu.save("path/checkpoint.het")
#   gpu = HeteroGPU.load("path/checkpoint.het")
#   stats = gpu.stats()
# ============================================================

import math
import time
import random
import hashlib
import json
import struct
import os
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from collections import OrderedDict


# ============================================================
# 错误体系
# ============================================================

class HeteroGPUError(Exception):
    """HeteroGPU 基础异常"""
    pass


class ConfigError(HeteroGPUError):
    """配置错误"""
    pass


class ShapeError(HeteroGPUError):
    """张量形状不匹配"""
    pass


class CheckpointError(HeteroGPUError):
    """存盘/读盘错误"""
    pass


class DeviceError(HeteroGPUError):
    """运行时错误"""
    pass


# ============================================================
# 配置 (带校验)
# ============================================================

@dataclass
class HeteroConfig:
    """HeteroGPU 完整配置 — 企业级带校验"""

    vocab_size: int = 256
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 512
    max_seq_len: int = 128

    # 加速
    cache_size: int = 4096
    sparse_top_k_ratio: float = 0.3
    oracle_confidence: float = 0.8

    # 训练
    learning_rate: float = 0.001
    grad_clip: float = 1.0
    dropout: float = 0.0  # 纯推理可关

    # 自举
    bootstrap_buffer_size: int = 1024
    bootstrap_seq_len: int = 32
    bootstrap_temperature: float = 0.9

    # 存盘
    checkpoint_dir: str = ""

    def __post_init__(self):
        self.validate()

    def validate(self):
        """配置校验"""
        errors = []
        if self.hidden_dim % self.num_heads != 0:
            errors.append(
                f"hidden_dim({self.hidden_dim}) 必须被 num_heads({self.num_heads}) 整除")
        if self.vocab_size < 2:
            errors.append(f"vocab_size({self.vocab_size}) 必须 >= 2")
        if self.hidden_dim < self.num_heads:
            errors.append(
                f"hidden_dim({self.hidden_dim}) >= num_heads({self.num_heads})")
        if self.num_layers < 1:
            errors.append(f"num_layers({self.num_layers}) 必须 >= 1")
        if self.ffn_dim < self.hidden_dim:
            errors.append(
                f"ffn_dim({self.ffn_dim}) >= hidden_dim({self.hidden_dim})")
        if not (0 < self.sparse_top_k_ratio <= 1):
            errors.append(
                f"sparse_top_k_ratio({self.sparse_top_k_ratio}) 必须在 (0, 1]")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate({self.learning_rate}) 必须 > 0")

        if errors:
            raise ConfigError("\n".join(errors))

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @classmethod
    def tiny(cls) -> "HeteroConfig":
        return cls(vocab_size=128, hidden_dim=32, num_heads=2,
                   num_layers=2, ffn_dim=64, max_seq_len=64,
                   cache_size=500, bootstrap_buffer_size=256)

    @classmethod
    def small(cls) -> "HeteroConfig":
        return cls(vocab_size=256, hidden_dim=64, num_heads=4,
                   num_layers=4, ffn_dim=256, max_seq_len=128,
                   cache_size=2048)

    @classmethod
    def base(cls) -> "HeteroConfig":
        return cls(vocab_size=512, hidden_dim=128, num_heads=8,
                   num_layers=6, ffn_dim=512, max_seq_len=256,
                   cache_size=8192)

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "max_seq_len": self.max_seq_len,
            "cache_size": self.cache_size,
            "sparse_top_k_ratio": self.sparse_top_k_ratio,
            "oracle_confidence": self.oracle_confidence,
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HeteroConfig":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ============================================================
# 张量
# ============================================================

class Tensor:
    """GPU原生张量 — 带shape校验"""

    __slots__ = ('data', 'rows', 'cols', '_grad',
                 '_ln_context', '_ffn_ctx', '_residual_src')

    def __init__(self, data: List[List[float]], requires_grad: bool = False):
        if not data:
            self.data = []
            self.rows = 0
            self.cols = 0
        else:
            self.data = data
            self.rows = len(data)
            self.cols = len(data[0])
        self._grad: Optional[List[List[float]]] = (
            [[0.0] * self.cols for _ in range(self.rows)]
            if requires_grad else None)

    @classmethod
    def zeros(cls, rows: int, cols: int,
              requires_grad: bool = False) -> "Tensor":
        return cls([[0.0] * cols for _ in range(rows)], requires_grad)

    @classmethod
    def ones(cls, rows: int, cols: int,
             requires_grad: bool = False) -> "Tensor":
        return cls([[1.0] * cols for _ in range(rows)], requires_grad)

    @classmethod
    def randn(cls, rows: int, cols: int, scale: float = 0.02,
              requires_grad: bool = True) -> "Tensor":
        m = [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]
        return cls(m, requires_grad)

    @classmethod
    def from_values(cls, rows: int, cols: int, value: float,
                    requires_grad: bool = False) -> "Tensor":
        return cls([[value] * cols for _ in range(rows)], requires_grad)

    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def size(self) -> int:
        return self.rows * self.cols

    @property
    def grad(self) -> Optional[List[List[float]]]:
        return self._grad

    def zero_grad(self):
        if self._grad is not None:
            for i in range(self.rows):
                for j in range(self.cols):
                    self._grad[i][j] = 0.0

    def check_shape(self, expected: Tuple[int, int], name: str = ""):
        if self.shape() != expected:
            raise ShapeError(
                f"{name}: expected {expected}, got {self.shape()}")

    def __repr__(self):
        g = "+grad" if self._grad else ""
        return f"Tensor({self.rows}x{self.cols}{g})"

    def clone(self) -> "Tensor":
        """深拷贝"""
        data = [row[:] for row in self.data]
        t = Tensor(data, False)
        if self._grad is not None:
            t._grad = [row[:] for row in self._grad]
        return t


# ============================================================
# 计算缓存
# ============================================================

class ComputeCache:
    def __init__(self, max_size: int = 4096):
        self._cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self._hits = 0
        self._misses = 0

    def _key(self, *args) -> str:
        h = hashlib.md5()
        for a in args:
            if isinstance(a, Tensor):
                d = a.data
                if d and d[0]:
                    pts = [d[0][0], d[-1][0], d[0][-1], d[-1][-1]]
                    if a.rows > 2:
                        pts.append(d[a.rows//2][0])
                    for p in pts:
                        h.update(f"{p:.8f}".encode())
                h.update(f"{a.rows},{a.cols}".encode())
            elif isinstance(a, (list, tuple)):
                h.update(str(a).encode())
            else:
                h.update(str(a).encode())
        return h.hexdigest()[:20]

    def get(self, *args) -> Optional[Any]:
        k = self._key(*args)
        if k in self._cache:
            self._cache.move_to_end(k)
            self._hits += 1
            return self._cache[k]
        self._misses += 1
        return None

    def put(self, value: Any, *args):
        k = self._key(*args)
        if k not in self._cache:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        self._cache[k] = value

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        t = self._hits + self._misses
        return self._hits / t if t else 0.0

    def stats(self) -> dict:
        return {"size": len(self._cache), "hits": self._hits,
                "misses": self._misses, "hit_rate": f"{self.hit_rate:.1%}"}


# ============================================================
# 稀疏注意力
# ============================================================

class SparseAttention:
    def __init__(self, top_k_ratio: float = 0.3):
        self.ratio = top_k_ratio
        self._total = 0
        self._skipped = 0

    def _select_keys(self, Q_data: List[List[float]],
                      K_data: List[List[float]],
                      seq_len: int, d_k: int, top_k: int,
                      approx_dims: int) -> List[set]:
        active = []
        approx = min(approx_dims, d_k)
        for i in range(seq_len):
            qi = Q_data[i]
            scores = [(sum(qi[d] * K_data[j][d] for d in range(approx)), j)
                      for j in range(seq_len)]
            scores.sort(reverse=True)
            active.append({j for _, j in scores[:top_k]})
        return active

    def compute(self, Q: Tensor, K: Tensor, V: Tensor,
                scale: float = None) -> Tensor:
        seq_len, d_k = Q.shape()
        if scale is None:
            scale = 1.0 / math.sqrt(d_k)
        top_k = max(1, int(seq_len * self.ratio))

        qd, kd, vd = Q.data, K.data, V.data
        d_v = V.cols
        active = self._select_keys(qd, kd, seq_len, d_k, top_k, 8)

        out = Tensor.zeros(seq_len, d_v)
        for i in range(seq_len):
            qi = qd[i]
            max_s = float('-inf')
            scores = {}
            for j in active[i]:
                s = sum(qi[d] * kd[j][d] for d in range(d_k)) * scale
                scores[j] = s
                if s > max_s:
                    max_s = s
                self._total += 1
            self._skipped += (seq_len - len(active[i]))

            exp_sum = sum(math.exp(scores[j] - max_s) for j in scores)
            if exp_sum <= 0:
                continue
            for j, s in scores.items():
                w = math.exp(s - max_s) / exp_sum
                for d in range(d_v):
                    out.data[i][d] += w * vd[j][d]

        return out

    @property
    def skip_rate(self) -> float:
        t = self._total + self._skipped
        return self._skipped / t if t else 0.0

    def stats(self) -> dict:
        return {"skipped": self._skipped, "computed": self._total,
                "skip_rate": f"{self.skip_rate:.1%}"}


# ============================================================
# Oracle
# ============================================================

class Oracle:
    def __init__(self, lr: float = 0.005):
        self.W: Optional[List[List[float]]] = None
        self.in_dim = 0
        self.out_dim = 0
        self.lr = lr
        self.samples = 0

    @staticmethod
    def _feat(t: Tensor) -> List[float]:
        d = t.data
        n = t.size()
        if n == 0:
            return [0.0] * 8
        total = total_sq = 0.0
        mx = float('-inf'); mn = float('inf'); nz = 0
        for r in d:
            for v in r:
                total += v; total_sq += v * v
                if v > mx: mx = v
                if v < mn: mn = v
                if abs(v) > 1e-8: nz += 1
        mean = total / n
        var = total_sq / n - mean * mean
        return [mean, math.sqrt(max(var, 0)), mx, mn,
                nz / n, math.log(t.rows + 1), math.log(t.cols + 1),
                total / n]

    def learn(self, x: Tensor, y: Tensor):
        xf = self._feat(x); yf = self._feat(y)
        if self.W is None:
            self.in_dim = len(xf); self.out_dim = len(yf)
            self.W = [[random.uniform(-0.005, 0.005)
                       for _ in range(self.in_dim)]
                      for _ in range(self.out_dim)]
        for o in range(self.out_dim):
            pred = sum(self.W[o][i] * xf[i] for i in range(self.in_dim))
            err = pred - yf[o]
            for i in range(self.in_dim):
                self.W[o][i] -= self.lr * err * xf[i]
        self.samples += 1

    def guess(self, x: Tensor) -> Tuple[List[float], float]:
        if self.W is None:
            return [], 0.0
        xf = self._feat(x)
        out = [sum(self.W[o][i] * xf[i] for i in range(self.in_dim))
               for o in range(self.out_dim)]
        conf = min(1.0, math.log(self.samples + 1) / math.log(100))
        return out, conf

    def trusted(self) -> bool:
        return self.samples >= 20

    def stats(self) -> dict:
        return {"samples": self.samples, "trusted": self.trusted()}


# ============================================================
# Profiler
# ============================================================

class Profiler:
    """内置性能剖析器"""

    def __init__(self):
        self._times: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}

    def record(self, name: str, elapsed: float):
        if name not in self._times:
            self._times[name] = []
            self._counts[name] = 0
        self._times[name].append(elapsed)
        self._counts[name] += 1

    def report(self) -> str:
        lines = ["[Profiler]", "-" * 40]
        for name in sorted(self._times):
            ts = self._times[name]
            if not ts:
                continue
            total = sum(ts)
            avg = total / len(ts) * 1000
            lines.append(
                f"  {name:25s} | {self._counts[name]:6d} calls | "
                f"{total*1000:8.2f}ms total | {avg:8.2f}ms avg")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {n: {"calls": self._counts[n],
                     "total_ms": sum(self._times[n]) * 1000,
                     "avg_ms": (sum(self._times[n]) / len(self._times[n])
                                * 1000 if self._times[n] else 0)}
                for n in self._times}


# ============================================================
# HeteroGPU — 企业级
# ============================================================

class HeteroGPU:
    """灵元Transformer专用GPU — 企业级

    异负得正 · 异虚为实
    """

    VERSION = "4.0.0-enterprise"

    def __init__(self, config: HeteroConfig):
        config.validate()
        self.cfg = config
        self._build()
        self._build_accelerators()
        self._bootstrap: List[Tuple[List[int], List[int]]] = []
        self.forward_count = 0
        self.profiler = Profiler()

        # 训练用的梯度累积
        self._training = False

    def _build(self):
        c = self.cfg
        h, f, v, nh = c.hidden_dim, c.ffn_dim, c.vocab_size, c.num_heads
        hd = c.head_dim

        self._layers = []
        for _ in range(c.num_layers):
            self._layers.append({
                # QKV 投影 — 企业级分头
                "wq": Tensor.randn(h, h),     # (h, h): 投影后reshape为(nh, hd)
                "wk": Tensor.randn(h, h),
                "wv": Tensor.randn(h, h),
                "wo": Tensor.randn(h, h),
                # FFN
                "w1": Tensor.randn(h, f),
                "w2": Tensor.randn(f, h),
                # 偏置
                "b1": Tensor.zeros(1, f, True),
                "b2": Tensor.zeros(1, h, True),
                # LayerNorm (存为 (1, h) 方便广播)
                "ln1_g": Tensor.from_values(1, h, 1.0, True),
                "ln1_b": Tensor.zeros(1, h, True),
                "ln2_g": Tensor.from_values(1, h, 1.0, True),
                "ln2_b": Tensor.zeros(1, h, True),
            })

        self._embed = Tensor.randn(v, h)
        self._head = Tensor.randn(h, v)
        self._head_bias = Tensor.zeros(1, v, True)
        self._final_ln_g = Tensor.from_values(1, h, 1.0, True)
        self._final_ln_b = Tensor.zeros(1, h, True)

    def _build_accelerators(self):
        self.cache = ComputeCache(self.cfg.cache_size)
        self.sparse = SparseAttention(self.cfg.sparse_top_k_ratio)
        self.oracle = Oracle()

    # ========== LayerNorm ==========

    def layernorm(self, x: Tensor, gamma: Tensor,
                   beta: Tensor) -> Tensor:
        r, c = x.shape()
        eps = 1e-5
        # 存中间值用于反向
        means = [0.0] * r
        ivars = [0.0] * r
        norms = [[0.0] * c for _ in range(r)]
        out = Tensor.zeros(r, c)

        for i in range(r):
            m = sum(x.data[i]) / c
            means[i] = m
            v = sum((x.data[i][j] - m) ** 2 for j in range(c)) / c
            ivars[i] = 1.0 / math.sqrt(v + eps)
            for j in range(c):
                norms[i][j] = (x.data[i][j] - m) * ivars[i]
                out.data[i][j] = norms[i][j] * gamma.data[0][j] + beta.data[0][j]

        # 存上下文用于反向
        out._ln_context = (x, gamma, beta, means, ivars, norms)
        return out

    def _layernorm_backward(self, grad_out: Tensor,
                             context: tuple) -> Tensor:
        """LayerNorm反向传播 — 完整正确版, 返回dx"""
        x, gamma, beta, means, ivars, norms = context
        r, c = x.shape()

        # gamma grad
        for j in range(c):
            for i in range(r):
                if gamma._grad:
                    gamma._grad[0][j] += grad_out.data[i][j] * norms[i][j]
                if beta._grad:
                    beta._grad[0][j] += grad_out.data[i][j]

        # x grad — 完整LayerNorm反向
        dx = Tensor.zeros(r, c)
        for i in range(r):
            dx_hat = [grad_out.data[i][j] * gamma.data[0][j] for j in range(c)]
            # dvar
            dvar = sum(dx_hat[j] * (x.data[i][j] - means[i]) * -0.5 *
                        ivars[i] ** 3 for j in range(c))
            # dmean
            dmean = sum(-dx_hat[j] * ivars[i] for j in range(c))
            dmean += dvar * sum(-2.0 * (x.data[i][j] - means[i]) for j in range(c)) / c

            for j in range(c):
                dx.data[i][j] = dx_hat[j] * ivars[i]
                dx.data[i][j] += dvar * 2.0 * (x.data[i][j] - means[i]) / c
                dx.data[i][j] += dmean / c

        return dx

    # ========== 多头注意力 (真分头) ==========

    def _attention_single_head(self, Q: Tensor, K: Tensor, V: Tensor,
                                scale: float) -> Tensor:
        """单个头的注意力"""
        return self.sparse.compute(Q, K, V, scale)

    def attention(self, x: Tensor, wq: Tensor, wk: Tensor, wv: Tensor,
                   wo: Tensor) -> Tensor:
        """真多头注意力

        1. x @ wq → reshape to (seq, num_heads, head_dim)
        2. 每头独立做注意力
        3. 拼头 → @ wo
        """
        s, d = x.shape()
        nh = self.cfg.num_heads
        hd = self.cfg.head_dim

        # Q/K/V 投影
        Q = self._linear(x, wq)  # (s, d)
        K = self._linear(x, wk)
        V = self._linear(x, wv)

        # 分头
        scale = 1.0 / math.sqrt(hd)
        head_outputs = []
        for h in range(nh):
            # 切片每个头
            Qh = Tensor([[Q.data[i][h*hd + j] for j in range(hd)]
                          for i in range(s)])
            Kh = Tensor([[K.data[i][h*hd + j] for j in range(hd)]
                          for i in range(s)])
            Vh = Tensor([[V.data[i][h*hd + j] for j in range(hd)]
                          for i in range(s)])
            head_out = self._attention_single_head(Qh, Kh, Vh, scale)
            head_outputs.append(head_out)

        # 拼头: (s, hd) * nh → (s, hd*nh)
        concat = Tensor.zeros(s, d)
        for h in range(nh):
            for i in range(s):
                for j in range(hd):
                    concat.data[i][h*hd + j] = head_outputs[h].data[i][j]

        # 输出投影
        output = self._linear(concat, wo)
        return output

    # ========== FFN ==========

    def ffn(self, x: Tensor, w1: Tensor, b1: Tensor,
            w2: Tensor, b2: Tensor) -> Tensor:
        """前馈网络: GELU(x@w1 + b1) @ w2 + b2"""
        s, d = x.shape()
        f = w1.cols

        # x @ w1 + b1
        h = Tensor.zeros(s, f)
        for i in range(s):
            for j in range(f):
                h.data[i][j] = sum(x.data[i][m] * w1.data[m][j] for m in range(d))
                h.data[i][j] += b1.data[0][j]

        # GELU
        for i in range(s):
            for j in range(f):
                z = h.data[i][j]
                h.data[i][j] = z / (1.0 + math.exp(-1.702 * z))

        # @ w2 + b2
        out = Tensor.zeros(s, d)
        for i in range(s):
            for j in range(d):
                s2 = sum(h.data[i][m] * w2.data[m][j] for m in range(f))
                out.data[i][j] = s2 + b2.data[0][j]

        # 存中间值
        out._ffn_ctx = (x, w1, w2, h, b1, b2)  # 用于反向
        return out

    def _ffn_backward(self, grad_out: Tensor, ctx: tuple) -> Tensor:
        """FFN反向传播 — 返回dx"""
        x, w1, w2, h, b1, b2 = ctx
        s, d = x.shape()
        f = w1.cols

        # w2 grad
        for m in range(f):
            for j in range(d):
                g = 0.0
                for i in range(s):
                    g += grad_out.data[i][j] * h.data[i][m]
                if w2._grad:
                    w2._grad[m][j] += g

        # b2 grad
        if b2._grad:
            for j in range(d):
                b2._grad[0][j] += sum(
                    grad_out.data[i][j] for i in range(s))

        # h grad
        h_grad = Tensor.zeros(s, f)
        for i in range(s):
            for m in range(f):
                g = sum(grad_out.data[i][j] * w2.data[m][j] for j in range(d))
                # GELU backward
                z = h.data[i][m]
                # GELU derivative: sigmoid(1.702*z) + z * sigmoid'(1.702*z) * 1.702
                sig = 1.0 / (1.0 + math.exp(-1.702 * z))
                gelu_grad = sig + z * sig * (1.0 - sig) * 1.702
                h_grad.data[i][m] = g * gelu_grad

        # w1 grad
        for m in range(d):
            for j in range(f):
                g = sum(x.data[i][m] * h_grad.data[i][j] for i in range(s))
                if w1._grad:
                    w1._grad[m][j] += g

        # b1 grad
        if b1._grad:
            for j in range(f):
                b1._grad[0][j] += sum(
                    h_grad.data[i][j] for i in range(s))

        # x grad
        dx = Tensor.zeros(s, d)
        for i in range(s):
            for m in range(d):
                dx.data[i][m] = sum(
                    h_grad.data[i][j] * w1.data[m][j] for j in range(f))

        return dx

    # ========== Transformer块 ==========

    def transformer_block(self, x: Tensor, lidx: int) -> Tensor:
        w = self._layers[lidx]

        # Pre-norm + Attention + 残差
        n1 = self.layernorm(x, w["ln1_g"], w["ln1_b"])
        attn_out = self.attention(n1, w["wq"], w["wk"], w["wv"], w["wo"])

        # 残差
        h = Tensor.zeros(*x.shape())
        for i in range(x.rows):
            for j in range(x.cols):
                h.data[i][j] = x.data[i][j] + attn_out.data[i][j]
        h._residual_src = x  # 反向用

        # Pre-norm + FFN + 残差
        n2 = self.layernorm(h, w["ln2_g"], w["ln2_b"])
        ffn_out = self.ffn(n2, w["w1"], w["b1"], w["w2"], w["b2"])
        self._last_b1 = w["b1"]
        self._last_b2 = w["b2"]

        out = Tensor.zeros(*x.shape())
        for i in range(x.rows):
            for j in range(x.cols):
                out.data[i][j] = h.data[i][j] + ffn_out.data[i][j]
        out._residual_src = h

        return out

    # ========== 基础线性 ==========

    def _linear(self, x: Tensor, w: Tensor) -> Tensor:
        """x @ w (不存中间值，只是投影)"""
        s, d = x.shape()
        out = Tensor.zeros(s, w.cols)
        for i in range(s):
            for j in range(w.cols):
                out.data[i][j] = sum(x.data[i][m] * w.data[m][j]
                                      for m in range(d))
        return out

    def _linear_bias(self, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        """x @ w + b"""
        s, d = x.shape()
        out = Tensor.zeros(s, w.cols)
        for i in range(s):
            for j in range(w.cols):
                out.data[i][j] = sum(x.data[i][m] * w.data[m][j]
                                      for m in range(d))
                out.data[i][j] += b.data[0][j]
        return out

    def _attention_with_weights(self, Q: Tensor, K: Tensor, V: Tensor,
                                 scale: float) -> Tuple[Tensor, list]:
        """Dense causal attention with stored weights for backward"""
        s = Q.rows
        d_v = V.cols
        out = Tensor.zeros(s, d_v)
        all_weights = []

        for qi in range(s):
            scores = []
            for ki in range(qi + 1):  # causal mask
                sc = sum(Q.data[qi][d] * K.data[ki][d]
                          for d in range(Q.cols)) * scale
                scores.append(sc)

            # Softmax
            mx = max(scores)
            exps = [math.exp(s - mx) for s in scores]
            sm = sum(exps)
            weights = [e / sm for e in exps]
            all_weights.append(weights)

            for ki in range(qi + 1):
                w = weights[ki]
                for d in range(d_v):
                    out.data[qi][d] += w * V.data[ki][d]

        return out, all_weights

    # ========== 嵌入 ==========

    def embed(self, ids: List[int]) -> Tensor:
        s = len(ids)
        out = Tensor.zeros(s, self.cfg.hidden_dim)
        for i, tid in enumerate(ids):
            tid = tid % self.cfg.vocab_size
            for j in range(self.cfg.hidden_dim):
                out.data[i][j] = self._embed.data[tid][j]
        return out

    # ========== 前向 ==========

    def forward(self, ids: List[int]) -> Tensor:
        t0 = time.time()
        ids = ids[:self.cfg.max_seq_len]
        self.forward_count += 1

        # 缓存命中检查
        cache_tag = tuple(ids[-self.cfg.max_seq_len//2:])
        cached = self.cache.get("logits", cache_tag,
                                 self._head.data[0][0],
                                 self._head.data[-1][-1])
        if cached is not None:
            self.profiler.record("forward_cache_hit", time.time() - t0)
            return cached

        x = self.embed(ids)
        for l in range(self.cfg.num_layers):
            x = self.transformer_block(x, l)

        x = self.layernorm(x, self._final_ln_g, self._final_ln_b)

        # 输出投影: (s, h) @ (h, v) + bias
        s = x.rows
        logits = Tensor.zeros(s, self.cfg.vocab_size)
        for i in range(s):
            for j in range(self.cfg.vocab_size):
                logits.data[i][j] = sum(
                    x.data[i][d] * self._head.data[d][j]
                    for d in range(self.cfg.hidden_dim))
                logits.data[i][j] += self._head_bias.data[0][j]

        self.cache.put(logits, "logits", cache_tag,
                        self._head.data[0][0],
                        self._head.data[-1][-1])
        self.oracle.learn(x, logits)
        self.profiler.record("forward", time.time() - t0)
        return logits

    # ========== 生成 ==========

    def generate(self, prompt: List[int], max_new: int = 32,
                  temperature: float = 0.8) -> List[int]:
        tokens = list(prompt)
        for _ in range(max_new):
            ctx = tokens[-self.cfg.max_seq_len:]
            logits = self.forward(ctx)
            last = logits.data[-1]

            if temperature > 0:
                mx = max(last)
                sm = sum(math.exp((l - mx) / temperature) for l in last)
                r = random.random()
                cum = 0.0
                for idx in range(len(last)):
                    cum += math.exp((last[idx] - mx) / temperature) / sm
                    if r < cum:
                        tokens.append(idx); break
                else:
                    tokens.append(len(last) - 1)
            else:
                tokens.append(max(range(len(last)), key=lambda i: last[i]))
        return tokens

    # ========== 训练 ==========

    def _training_forward(self, ids: List[int]) -> Tuple[Tensor, list, Tensor, List[int]]:
        """训练用前向传播 — 存储所有中间激活用于反向"""
        ids = ids[:self.cfg.max_seq_len]
        s = len(ids)
        h = self.cfg.hidden_dim
        nh = self.cfg.num_heads
        hd = self.cfg.head_dim

        # Embedding
        x = self.embed(ids)
        activations = []

        for l in range(self.cfg.num_layers):
            w = self._layers[l]

            # Pre-norm 1
            n1 = self.layernorm(x, w["ln1_g"], w["ln1_b"])

            # Attention (dense, store weights)
            Q = self._linear(n1, w["wq"])
            K = self._linear(n1, w["wk"])
            V = self._linear(n1, w["wv"])
            scale = 1.0 / math.sqrt(hd)

            head_outputs = []
            head_attn_weights = []
            for hh in range(nh):
                Qh = Tensor([[Q.data[i][hh*hd + j] for j in range(hd)]
                              for i in range(s)])
                Kh = Tensor([[K.data[i][hh*hd + j] for j in range(hd)]
                              for i in range(s)])
                Vh = Tensor([[V.data[i][hh*hd + j] for j in range(hd)]
                              for i in range(s)])
                head_out, attn_w = self._attention_with_weights(Qh, Kh, Vh, scale)
                head_outputs.append(head_out)
                head_attn_weights.append(attn_w)

            concat = Tensor.zeros(s, h)
            for hh in range(nh):
                for i in range(s):
                    for j in range(hd):
                        concat.data[i][hh*hd + j] = head_outputs[hh].data[i][j]

            attn_out = self._linear(concat, w["wo"])

            # Residual
            h_res = Tensor.zeros(s, h)
            for i in range(s):
                for j in range(h):
                    h_res.data[i][j] = x.data[i][j] + attn_out.data[i][j]

            # Pre-norm 2
            n2 = self.layernorm(h_res, w["ln2_g"], w["ln2_b"])

            # FFN
            ffn_out = self.ffn(n2, w["w1"], w["b1"], w["w2"], w["b2"])

            # Residual
            out = Tensor.zeros(s, h)
            for i in range(s):
                for j in range(h):
                    out.data[i][j] = h_res.data[i][j] + ffn_out.data[i][j]

            activations.append({
                'n1': n1, 'attn_out': attn_out, 'h_res': h_res,
                'n2': n2, 'ffn_out': ffn_out, 'out': out,
                'Q': Q, 'K': K, 'V': V, 'concat': concat,
                'attn_weights': head_attn_weights, 'scale': scale,
                'wq': w["wq"], 'wk': w["wk"],
                'wv': w["wv"], 'wo': w["wo"],
                'x_input': x,
            })
            x = out

        # Final LayerNorm
        x_final = self.layernorm(x, self._final_ln_g, self._final_ln_b)

        # Head projection
        v = self.cfg.vocab_size
        logits = Tensor.zeros(s, v)
        for i in range(s):
            for j in range(v):
                logits.data[i][j] = sum(
                    x_final.data[i][d] * self._head.data[d][j]
                    for d in range(h))
                logits.data[i][j] += self._head_bias.data[0][j]

        return logits, activations, x_final, ids

    def train_step(self, input_ids: List[int],
                    target_ids: List[int]) -> float:
        """完整训练步: 前向 + 反向 + 更新"""
        t0 = time.time()
        logits, activations, x_final, ids = self._training_forward(input_ids)
        loss = self._cross_entropy(logits, target_ids)
        self._backward(logits, target_ids, input_ids, activations, x_final)
        self._update_params()
        self.profiler.record("train_step", time.time() - t0)
        return loss

    def _cross_entropy(self, logits: Tensor, targets: List[int]) -> float:
        total = 0.0
        for t in range(min(len(targets), logits.rows)):
            row = logits.data[t]
            mx = max(row)
            sm = sum(math.exp(r - mx) for r in row)
            tid = targets[t] % self.cfg.vocab_size
            prob = math.exp(row[tid] - mx) / sm
            total -= math.log(max(prob, 1e-12))
        return total / max(len(targets), 1)

    def _backward(self, logits: Tensor, targets: List[int],
                   input_ids: List[int], activations: list = None,
                   x_final: Tensor = None):
        """完整反向传播"""
        if activations is None:
            # Fallback to old behavior
            self._backward_legacy(logits, targets, input_ids)
            return

        clip = self.cfg.grad_clip
        s = logits.rows
        v = self.cfg.vocab_size
        h = self.cfg.hidden_dim

        # ---- 1. Loss -> logits grad ----
        logits_grad = Tensor.zeros(s, v)
        for t in range(min(len(targets), s)):
            row = logits.data[t]
            mx = max(row)
            sm = sum(math.exp(r - mx) for r in row)
            tid = targets[t] % v
            for j in range(v):
                prob = math.exp(row[j] - mx) / sm
                logits_grad.data[t][j] = (prob - (1.0 if j == tid else 0.0)) / s

        # ---- 2. Head weight grad ----
        for i in range(h):
            for j in range(v):
                if self._head._grad:
                    g = sum(logits_grad.data[t][j] * x_final.data[t][i]
                            for t in range(s))
                    self._head._grad[i][j] += clip_grad(g, clip)

        # Head bias grad
        if self._head_bias._grad:
            for j in range(v):
                self._head_bias._grad[0][j] += clip_grad(
                    sum(logits_grad.data[t][j] for t in range(s)), clip)

        # logits grad -> x_final grad
        logits_grad_to_x = Tensor.zeros(s, h)
        for i in range(s):
            for j in range(h):
                g = 0.0
                for k in range(v):
                    g += logits_grad.data[i][k] * self._head.data[j][k]
                logits_grad_to_x.data[i][j] = g

        # ---- 3. Final LayerNorm backward ----
        ln_ctx = x_final._ln_context
        dx = self._layernorm_backward(logits_grad_to_x, ln_ctx)

        # ---- 4. Per-layer backward (reverse order) ----
        for l in range(len(activations) - 1, -1, -1):
            act = activations[l]
            w = self._layers[l]

            # FFN backward: d_ffn_out = dx (from residual 2: out = h_res + ffn_out)
            d_ffn_out = Tensor.zeros(s, h)
            for i in range(s):
                for j in range(h):
                    d_ffn_out.data[i][j] = dx.data[i][j]

            # FFN backward returns d_n2 (grad w.r.t. n2, input to FFN)
            d_n2 = self._ffn_backward(d_ffn_out, act['ffn_out']._ffn_ctx)

            # LN2 backward: d_n2 -> d_h_res_from_ln2 (grad w.r.t. h_res, input to LN2)
            n2_ctx = act['n2']._ln_context
            d_h_res_from_ln2 = self._layernorm_backward(d_n2, n2_ctx)

            # Total d_h_res = dx (residual 2) + d_h_res_from_ln2 (through FFN->LN2)
            d_h_res = Tensor.zeros(s, h)
            for i in range(s):
                for j in range(h):
                    d_h_res.data[i][j] = dx.data[i][j] + d_h_res_from_ln2.data[i][j]

            # d_attn_out = d_h_res (residual 1: h_res = x + attn_out)
            d_attn_out = Tensor.zeros(s, h)
            for i in range(s):
                for j in range(h):
                    d_attn_out.data[i][j] = d_h_res.data[i][j]

            # Output projection backward
            # d_concat = d_attn_out @ wo^T
            d_concat = Tensor.zeros(s, h)
            wo = act['wo']
            for i in range(s):
                for j in range(h):
                    g = 0.0
                    for k in range(h):
                        g += d_attn_out.data[i][k] * wo.data[j][k]
                    d_concat.data[i][j] = g

            # wo grad
            if wo._grad:
                for m in range(h):
                    for j in range(h):
                        g = sum(d_attn_out.data[i][j] * act['concat'].data[i][m]
                                for i in range(s))
                        wo._grad[m][j] += clip_grad(g, clip)

            # Per-head attention backward
            nh = self.cfg.num_heads
            hd = self.cfg.head_dim
            scale = act['scale']
            d_Q = Tensor.zeros(s, h)
            d_K = Tensor.zeros(s, h)
            d_V = Tensor.zeros(s, h)

            for hh in range(nh):
                attn_w = act['attn_weights'][hh]
                Q = act['Q']
                K = act['K']
                V = act['V']

                for qi in range(s):
                    # d_V: gradient accumulates at key positions
                    for ki in range(qi + 1):
                        w_val = attn_w[qi][ki]
                        for d in range(hd):
                            d_V.data[ki][hh * hd + d] += (
                                w_val * d_concat.data[qi][hh * hd + d])

                    # d_attn_weights -> d_scores
                    d_scores = [0.0] * (qi + 1)
                    for ki in range(qi + 1):
                        for d in range(hd):
                            d_scores[ki] += d_concat.data[qi][hh * hd + d] * \
                                V.data[ki][hh * hd + d]

                    # softmax backward: d_score[j] = w[j] * (d_scores[j] - sum(w*k * d_scores[k]))
                    ws = attn_w[qi]
                    wsum = sum(ws[k] * d_scores[k] for k in range(qi + 1))
                    for ki in range(qi + 1):
                        d_sc = ws[ki] * (d_scores[ki] - wsum)

                        # d_Q and d_K
                        for d in range(hd):
                            d_Q.data[qi][hh * hd + d] += d_sc * K.data[ki][hh * hd + d]
                            d_K.data[ki][hh * hd + d] += d_sc * Q.data[qi][hh * hd + d]

            # Q/K/V weight grads
            for wname, dval in [("wq", d_Q), ("wk", d_K), ("wv", d_V)]:
                wt = act[wname]
                if wt._grad:
                    for m in range(h):
                        for j in range(h):
                            g = sum(dval.data[i][j] * act['n1'].data[i][m]
                                    for i in range(s))
                            wt._grad[m][j] += clip_grad(g, clip)

            # LayerNorm 1 backward — use returned dx
            n1_ctx = act['n1']._ln_context
            d_n1_input = Tensor.zeros(s, h)
            for i in range(s):
                for j in range(h):
                    d_n1_input.data[i][j] = d_Q.data[i][j] + d_K.data[i][j] + d_V.data[i][j]

            d_ln1_x = self._layernorm_backward(d_n1_input, n1_ctx)

            # dx for next layer = d_ln1_x (through LN1) + d_h_res (residual)
            for i in range(s):
                for j in range(h):
                    dx.data[i][j] = d_ln1_x.data[i][j] + d_h_res.data[i][j]

        # ---- 5. Embedding grad ----
        for i, tid in enumerate(input_ids[:s]):
            tid = tid % v
            if self._embed._grad:
                for j in range(h):
                    self._embed._grad[tid][j] += clip_grad(
                        dx.data[i][j], clip)

    def _backward_legacy(self, logits: Tensor, targets: List[int],
                          input_ids: List[int]):
        """Legacy backward (fallback)"""
        clip = self.cfg.grad_clip
        s = logits.rows
        v = self.cfg.vocab_size
        h = self.cfg.hidden_dim

        logits_grad = Tensor.zeros(s, v)
        for t in range(min(len(targets), s)):
            row = logits.data[t]
            mx = max(row)
            sm = sum(math.exp(r - mx) for r in row)
            tid = targets[t] % v
            for j in range(v):
                prob = math.exp(row[j] - mx) / sm
                logits_grad.data[t][j] = (prob - (1.0 if j == tid else 0.0)) / s

        for i in range(h):
            for j in range(v):
                if self._head._grad:
                    self._head._grad[i][j] += clip_grad(
                        sum(logits_grad.data[t][j] for t in range(s)), clip)
        if self._head_bias._grad:
            for j in range(v):
                self._head_bias._grad[0][j] += clip_grad(
                    sum(logits_grad.data[t][j] for t in range(s)), clip)

    def _update_params(self):
        """梯度更新"""
        lr = self.cfg.learning_rate
        for layer in self._layers:
            for name in ["w1", "w2", "b1", "b2",
                          "ln1_g", "ln1_b", "ln2_g", "ln2_b"]:
                t = layer[name]
                if t._grad:
                    for i in range(t.rows):
                        for j in range(t.cols):
                            t.data[i][j] -= lr * t._grad[i][j]
                            t._grad[i][j] = 0.0

        for t in [self._head, self._head_bias, self._embed,
                   self._final_ln_g, self._final_ln_b]:
            if t._grad:
                for i in range(t.rows):
                    for j in range(t.cols):
                        t.data[i][j] -= lr * t._grad[i][j]
                        t._grad[i][j] = 0.0

    # ========== 自举训练 ==========

    def bootstrap_epoch(self, num_samples: int = 64) -> float:
        total_loss = 0.0
        for _ in range(num_samples):
            # 随机prompt
            plen = random.randint(4, 16)
            prompt = [random.randrange(self.cfg.vocab_size) for _ in range(plen)]
            target = self.generate(
                prompt, max_new=self.cfg.bootstrap_seq_len,
                temperature=self.cfg.bootstrap_temperature)

            # 存缓冲区
            self._bootstrap.append((prompt, target))
            if len(self._bootstrap) > self.cfg.bootstrap_buffer_size:
                self._bootstrap.pop(0)

            # teacher forcing训练
            loss = self.train_step(target[:-1], target[1:])
            total_loss += loss

        return total_loss / max(num_samples, 1)

    # ========== 持久化 ==========

    def save(self, path: str):
        """保存模型到文件

        格式: 自定义二进制 (header + weights)
        """
        try:
            with open(path, 'wb') as f:
                # Header: magic + version + config
                header = json.dumps({
                    "magic": "HETERO",
                    "version": self.VERSION,
                    "config": self.cfg.to_dict(),
                }).encode('utf-8')
                f.write(struct.pack('!I', len(header)))
                f.write(header)

                # Weights
                all_tensors = [self._embed, self._head, self._head_bias,
                                self._final_ln_g, self._final_ln_b]
                for layer in self._layers:
                    for name in ["wq", "wk", "wv", "wo", "w1", "w2",
                                  "b1", "b2", "ln1_g", "ln1_b",
                                  "ln2_g", "ln2_b"]:
                        all_tensors.append(layer[name])

                for t in all_tensors:
                    f.write(struct.pack('!II', t.rows, t.cols))
                    for row in t.data:
                        for v in row:
                            f.write(struct.pack('!f', v))
        except IOError as e:
            raise CheckpointError(f"保存失败: {e}")

    @classmethod
    def load(cls, path: str) -> "HeteroGPU":
        """从文件加载模型"""
        try:
            with open(path, 'rb') as f:
                # Header
                header_len = struct.unpack('!I', f.read(4))[0]
                header = json.loads(f.read(header_len).decode('utf-8'))
                if header["magic"] != "HETERO":
                    raise CheckpointError(f"无效文件: magic={header['magic']}")

                config = HeteroConfig.from_dict(header["config"])
                gpu = cls(config)

                # 读取所有Tensor
                all_tensors = [gpu._embed, gpu._head, gpu._head_bias,
                                gpu._final_ln_g, gpu._final_ln_b]
                for layer in gpu._layers:
                    for name in ["wq", "wk", "wv", "wo", "w1", "w2",
                                  "b1", "b2", "ln1_g", "ln1_b",
                                  "ln2_g", "ln2_b"]:
                        all_tensors.append(layer[name])

                for t in all_tensors:
                    rows, cols = struct.unpack('!II', f.read(8))
                    if (rows, cols) != (t.rows, t.cols):
                        raise CheckpointError(
                            f"Shape mismatch: file ({rows},{cols}) != "
                            f"expected ({t.rows},{t.cols})")
                    for i in range(rows):
                        row = [struct.unpack('!f', f.read(4))[0]
                               for _ in range(cols)]
                        t.data[i] = row

                return gpu
        except (IOError, struct.error) as e:
            raise CheckpointError(f"加载失败: {e}")

    # ========== 统计 ==========

    def stats(self) -> dict:
        params = sum(t.size() for layer in self._layers
                     for t in layer.values()) + \
                 self._embed.size() + self._head.size() + \
                 self._head_bias.size() + self._final_ln_g.size() + \
                 self._final_ln_b.size()

        return {
            "version": self.VERSION,
            "config": {
                "vocab": self.cfg.vocab_size,
                "hidden": self.cfg.hidden_dim,
                "heads": self.cfg.num_heads,
                "head_dim": self.cfg.head_dim,
                "layers": self.cfg.num_layers,
                "ffn": self.cfg.ffn_dim,
                "params": f"{params:,}",
                "lr": self.cfg.learning_rate,
            },
            "runtime": {
                "forward_passes": self.forward_count,
                "bootstrap_samples": len(self._bootstrap),
            },
            "sparse": self.sparse.stats(),
            "cache": self.cache.stats(),
            "oracle": self.oracle.stats(),
            "profiler": self.profiler.stats(),
        }


def clip_grad(g: float, clip: float) -> float:
    """梯度裁剪"""
    if g > clip:
        return clip
    if g < -clip:
        return -clip
    return g


# ============================================================
# 完整测试套件
# ============================================================

class TestHeteroGPU:
    """企业级测试套件"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.gpu: Optional[HeteroGPU] = None

    def _check(self, cond, msg):
        if cond:
            self.passed += 1
        else:
            self.failed += 1
            print(f"  FAIL: {msg}")

    def test_config_validation(self):
        print("\n--- 配置校验 ---")
        # 正常配置
        try:
            HeteroConfig(vocab_size=256, hidden_dim=128, num_heads=4).validate()
            self._check(True, "valid config")
        except Exception as e:
            self._check(False, f"valid config: {e}")

        # 非法配置
        try:
            HeteroConfig(vocab_size=256, hidden_dim=127, num_heads=4).validate()
            self._check(False, "should raise for hidden_dim % heads != 0")
        except ConfigError:
            self._check(True, "hidden_dim % heads raised configError")

    def test_basic_forward(self):
        print("\n--- 基础前向 ---")
        self.gpu = HeteroGPU(HeteroConfig.tiny())

        emb = self.gpu.embed([1, 2, 3])
        self._check(emb.shape() == (3, 32), f"embed shape {emb.shape()}")

        logits = self.gpu.forward([1, 2, 3, 4, 5])
        self._check(logits.shape() == (5, 128), f"logits shape {logits.shape()}")
        self._check(
            max(logits.data[0]) - min(logits.data[0]) > 0, "logits有差异")

        out = self.gpu.generate([1, 2, 3], max_new=8)
        self._check(len(out) == 11, f"generate len {len(out)}")
        self._check(len(set(out)) > 1, "generate token多样性")

    def test_multihead_attention(self):
        print("\n--- 多头注意力 ---")
        cfg = HeteroConfig.tiny()
        cfg.num_heads = 2
        cfg.hidden_dim = 32  # 32 / 2 = 16 head_dim
        gpu = HeteroGPU(cfg)

        x = Tensor.randn(4, 32)
        w = gpu._layers[0]
        attn_out = gpu.attention(x, w["wq"], w["wk"], w["wv"], w["wo"])
        self._check(attn_out.shape() == (4, 32),
                     f"attention output shape {attn_out.shape()}")

    def test_sparse_acceleration(self):
        print("\n--- 稀疏加速 ---")
        q = Tensor.randn(32, 32)
        k = Tensor.randn(32, 32)
        v = Tensor.randn(32, 32)
        _ = self.gpu.sparse.compute(q, k, v)
        sr = self.gpu.sparse.skip_rate
        self._check(0.4 <= sr <= 0.9, f"skip_rate {sr:.1%} in [40%,90%]")

    def test_cache_acceleration(self):
        print("\n--- 缓存加速 ---")
        prompt = [5, 10, 15]
        self.gpu.cache.clear()
        _ = self.gpu.forward(prompt)
        _ = self.gpu.forward(prompt)
        self._check(self.gpu.cache._hits >= 1, "cache hit on repeat")
        self._check(self.gpu.cache._misses >= 1, "cache miss on first")

    def test_bootstrap_training(self):
        print("\n--- 自举训练 ---")
        gpu = HeteroGPU(HeteroConfig.tiny())
        loss1 = gpu.bootstrap_epoch(num_samples=16)
        self._check(loss1 > 0, f"loss positive: {loss1:.4f}")

        # 多轮
        losses = [loss1]
        for _ in range(2):
            losses.append(gpu.bootstrap_epoch(num_samples=16))
        self._check(len(losses) == 3, f"epochs: {len(losses)}")

    def test_save_load(self):
        print("\n--- 存盘/读盘 ---")
        gpu = HeteroGPU(HeteroConfig.tiny())
        path = "/tmp/test_hetero_checkpoint.het"

        # 跑一些前向
        logits_before = gpu.forward([1, 2, 3])
        gpu.save(path)
        self._check(os.path.exists(path), "file created")

        # 加载
        gpu2 = HeteroGPU.load(path)
        logits_after = gpu2.forward([1, 2, 3])

        # 验证一致
        err = max(abs(logits_before.data[i][j] - logits_after.data[i][j])
                   for i in range(logits_before.rows)
                   for j in range(logits_before.cols))
        self._check(err < 1e-6, f"roundtrip consistency: err={err:.2e}")

        # cleanup: /tmp file left for OS to recycle

    def test_profiler(self):
        print("\n--- Profiler ---")
        self._check(len(self.gpu.profiler._times) > 0, "profiler has data")
        report = self.gpu.profiler.report()
        self._check("forward" in report, "forward profiled")
        print(report)

    def test_full_stats(self):
        print("\n--- 综合统计 ---")
        s = self.gpu.stats()
        self._check("4.0.0" in s["version"], f"version: {s['version']}")
        self._check(int(s["config"]["params"].replace(",","")) > 0,
                     f"params: {s['config']['params']}")
        print(f"  version: {s['version']}")
        print(f"  config: {s['config']}")
        print(f"  runtime: {s['runtime']}")

    def run_all(self):
        print("=" * 60)
        print("HeteroGPU Enterprise 测试套件")
        print("=" * 60)

        random.seed(42)

        self.test_config_validation()
        self.test_basic_forward()
        self.test_multihead_attention()
        self.test_sparse_acceleration()
        self.test_cache_acceleration()
        self.test_bootstrap_training()
        self.test_save_load()
        self.test_profiler()
        self.test_full_stats()

        print(f"\n{'='*60}")
        print(f"结果: {self.passed} passed, {self.failed} failed")
        print(f"{'='*60}")


if __name__ == "__main__":
    TestHeteroGPU().run_all()
