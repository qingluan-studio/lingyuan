#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灵元 V7.0 ULTRA — 自包含训练系统
DeepNorm + GQA + MoE + RoPE/ALiBi 混合位置编码
纯Python标准库 · 零外部依赖
"""

import os, sys, json, time, math, random, struct, argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# ============================================================
# Tensor — 二维矩阵
# ============================================================

class Tensor:
    __slots__ = ['data', 'rows', 'cols', 'grad', '_residual_src']

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.data = [[0.0]*cols for _ in range(rows)]
        self.grad = [[0.0]*cols for _ in range(rows)]
        self._residual_src = None

    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Tensor":
        return cls(rows, cols)

    @classmethod
    def ones(cls, rows: int, cols: int) -> "Tensor":
        t = cls(rows, cols)
        for i in range(rows):
            for j in range(cols):
                t.data[i][j] = 1.0
        return t

    @classmethod
    def randn(cls, rows: int, cols: int, scale: float = 0.02) -> "Tensor":
        t = cls(rows, cols)
        for i in range(rows):
            for j in range(cols):
                t.data[i][j] = random.gauss(0, 1) * scale
        return t

    def size(self) -> int:
        return self.rows * self.cols

    def zero_grad(self):
        self.grad = [[0.0]*self.cols for _ in range(self.rows)]

    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)


def glorot(fan_in: int, fan_out: int) -> Tensor:
    scale = math.sqrt(2.0 / (fan_in + fan_out))
    return Tensor.randn(fan_in, fan_out, scale)


# ============================================================
# Tokenizer — 字符级频率自适应
# ============================================================

class CharTokenizer:
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        # 特殊token
        for i, ch in enumerate(['<PAD>', '<UNK>', '<BOS>', '<EOS>']):
            self.char2id[ch] = i
            self.id2char[i] = ch

    def fit_on_text(self, text: str):
        from collections import Counter
        freq = Counter(text)
        sorted_chars = sorted(freq.keys(), key=lambda c: -freq[c])
        idx = len(self.char2id)
        for ch in sorted_chars:
            if ch not in self.char2id and idx < self.vocab_size:
                self.char2id[ch] = idx
                self.id2char[idx] = ch
                idx += 1

    def encode(self, text: str) -> List[int]:
        return [self.char2id.get(ch, 1) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.id2char.get(i, '<UNK>') for i in ids)

    @property
    def actual_size(self) -> int:
        return len(self.char2id)


# ============================================================
# Data Loader
# ============================================================

class TextDataLoader:
    def __init__(self, tokenizer: CharTokenizer, seq_len: int = 64, batch_size: int = 4):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self._flat_ids: List[int] = []
        self._data: List[List[int]] = []

    def load_text(self, text: str):
        self.tokenizer.fit_on_text(text)
        ids = self.tokenizer.encode(text)
        self._flat_ids = ids
        step = max(1, self.seq_len // 2)
        for i in range(0, len(ids) - self.seq_len - 1, step):
            self._data.append(ids[i:i+self.seq_len+1])

    def load_file(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.load_text(f.read())

    def sample_batch(self) -> Tuple[List[List[int]], List[List[int]]]:
        if not self._data:
            return self._synthetic()
        inputs, targets = [], []
        for _ in range(self.batch_size):
            seq = random.choice(self._data)
            inputs.append(seq[:self.seq_len])
            targets.append(seq[1:self.seq_len+1])
        return inputs, targets

    def _synthetic(self) -> Tuple[List[List[int]], List[List[int]]]:
        inputs, targets = [], []
        for _ in range(self.batch_size):
            seq = [random.randint(0, self.tokenizer.vocab_size-1) for _ in range(self.seq_len)]
            inputs.append(seq[:-1])
            targets.append(seq[1:])
        return inputs, targets


# ============================================================
# Config
# ============================================================

@dataclass
class ModelConfig:
    vocab_size: int = 512
    hidden_dim: int = 64
    num_heads: int = 4
    num_kv_heads: int = 1      # GQA
    num_layers: int = 4
    ffn_dim: int = 256
    max_seq_len: int = 64
    learning_rate: float = 0.001
    # V7.0 ULTRA
    pos_encoding: str = "hybrid"  # rope+alibi
    norm_type: str = "deepnorm"
    ffn_type: str = "moe"
    num_experts: int = 4
    num_activated_experts: int = 2
    sliding_window: int = 32
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def kv_group_size(self) -> int:
        return self.num_heads // max(1, self.num_kv_heads)

    @property
    def deepnorm_alpha(self) -> float:
        return (2 * self.num_layers) ** 0.25

    @classmethod
    def ultra_lite(cls) -> "ModelConfig":
        return cls(
            vocab_size=512, hidden_dim=64, num_heads=4,
            num_kv_heads=1, num_layers=4, ffn_dim=256,
            max_seq_len=64,
        )

    @classmethod
    def tiny(cls) -> "ModelConfig":
        return cls(
            vocab_size=256, hidden_dim=32, num_heads=2,
            num_kv_heads=1, num_layers=2, ffn_dim=64,
            max_seq_len=32,
        )


# ============================================================
# Model — V7.0 ULTRA Transformer
# ============================================================

class LingyuanModel:
    """V7.0 ULTRA — DeepNorm + GQA + MoE + RoPE/ALiBi"""

    VERSION = "7.0.0-ultra"

    def __init__(self, config: ModelConfig):
        self.cfg = config
        c = config
        H, V, L = c.hidden_dim, c.vocab_size, c.num_layers
        F = c.ffn_dim
        E = c.num_experts

        # Embedding
        self.embed = Tensor.randn(V, H, 0.02)

        # Layers
        self.layers = []
        alpha = c.deepnorm_alpha
        for _ in range(L):
            layer = {
                # Norm 1 (DeepNorm)
                "ln1_g": Tensor.ones(1, H),
                "ln1_b": Tensor.zeros(1, H),
                # GQA: Q heads full, KV heads reduced
                "wq": glorot(H, H),
                "wk": glorot(H, c.head_dim * c.num_kv_heads),
                "wv": glorot(H, c.head_dim * c.num_kv_heads),
                "wo": glorot(H, H),
                # Norm 2
                "ln2_g": Tensor.ones(1, H),
                "ln2_b": Tensor.zeros(1, H),
                # MoE
                "moe_router": glorot(H, E),
                "moe_experts": [
                    {
                        "w_gate": glorot(H, F),
                        "w_up": glorot(F, H),
                        "w_down": glorot(H, F),  # for SwiGLU
                    } for _ in range(E)
                ],
                # 保留标准FFN权重作为fallback
                "w1": glorot(H, F),
                "b1": Tensor.zeros(1, F),
                "w2": glorot(F, H),
                "b2": Tensor.zeros(1, H),
            }
            self.layers.append(layer)

        # Final norm
        self.final_ln_g = Tensor.ones(1, H)
        self.final_ln_b = Tensor.zeros(1, H)

        # Output head
        self.head = glorot(H, V)
        self.head_bias = Tensor.zeros(1, V)

        # RoPE 旋转矩阵预计算
        self._rope_cos = {}
        self._rope_sin = {}
        self._alibi = self._compute_alibi()

        # 训练状态
        self.forward_count = 0
        self._params_list = None
        self._cache: Optional[dict] = None

    # --------------------------------------------------------
    # 参数收集 (用于梯度清零 / 裁剪 / 更新)
    # --------------------------------------------------------
    def _all_params(self) -> List[Tensor]:
        params = [self.embed, self.head, self.head_bias,
                  self.final_ln_g, self.final_ln_b]
        for layer in self.layers:
            for k in ["ln1_g", "ln1_b", "wq", "wk", "wv", "wo",
                      "ln2_g", "ln2_b", "moe_router"]:
                params.append(layer[k])
            for exp in layer["moe_experts"]:
                for k in ["w_gate", "w_up", "w_down"]:
                    params.append(exp[k])
            for k in ["w1", "b1", "w2", "b2"]:
                params.append(layer[k])
        return params

    # --------------------------------------------------------
    # 位置编码
    # --------------------------------------------------------
    def _compute_rope(self, seq_len: int):
        """预计算RoPE旋转矩阵"""
        d = self.cfg.head_dim
        theta = self.cfg.rope_theta
        cos = [[0.0]*d for _ in range(seq_len)]
        sin = [[0.0]*d for _ in range(seq_len)]
        for pos in range(seq_len):
            for i in range(d // 2):
                angle = pos / (theta ** (2*i / d))
                cos[pos][2*i] = math.cos(angle)
                cos[pos][2*i+1] = math.cos(angle)
                sin[pos][2*i] = math.sin(angle)
                sin[pos][2*i+1] = math.sin(angle)
        return cos, sin

    def _compute_alibi(self) -> List[List[List[float]]]:
        """ALiBi 偏置矩阵 — 每个头独立斜率 (per-head slopes)"""
        n = self.cfg.max_seq_len
        heads = self.cfg.num_heads
        # 每个头一个几何递减斜率
        slopes = [1.0 / (2.0 ** (h + 1)) for h in range(heads)]
        bias = [[[0.0]*n for _ in range(n)] for _ in range(heads)]
        for h in range(heads):
            s = slopes[h]
            for i in range(n):
                for j in range(n):
                    bias[h][i][j] = -s * abs(i - j)
        return bias

    # --------------------------------------------------------
    # LayerNorm (带缓存)
    # --------------------------------------------------------
    def _layernorm(self, x: Tensor, g: Tensor, b: Tensor,
                   cache_key=None) -> Tensor:
        """LayerNorm / DeepNorm"""
        rows, cols = x.rows, x.cols
        out = Tensor.zeros(rows, cols)
        x_hat = [[0.0]*cols for _ in range(rows)]
        means = [0.0]*rows
        stds = [0.0]*rows
        for i in range(rows):
            mean = sum(x.data[i]) / cols
            var = sum((v - mean)**2 for v in x.data[i]) / cols
            std = math.sqrt(var + 1e-6)
            means[i] = mean
            stds[i] = std
            for j in range(cols):
                xh = (x.data[i][j] - mean) / std
                x_hat[i][j] = xh
                out.data[i][j] = xh * g.data[0][j] + b.data[0][j]
        if self._cache is not None and cache_key is not None:
            self._cache[cache_key] = {
                'x_hat': x_hat, 'mean': means, 'std': stds,
                'g': g, 'b': b, 'input': x,
            }
        return out

    def _layernorm_backward(self, dy, lncache, g: Tensor, b: Tensor):
        """LayerNorm 反向: dy -> dx, 累积 dgamma/dbeta 到 g.grad/b.grad"""
        x_hat = lncache['x_hat']
        stds = lncache['std']
        rows = len(x_hat)
        cols = len(x_hat[0]) if rows > 0 else 0
        n = cols
        # dgamma / dbeta
        for j in range(cols):
            dg = 0.0
            db = 0.0
            for i in range(rows):
                dg += dy[i][j] * x_hat[i][j]
                db += dy[i][j]
            g.grad[0][j] += dg
            b.grad[0][j] += db
        # dx
        dx = [[0.0]*cols for _ in range(rows)]
        for i in range(rows):
            sigma = stds[i]
            inv = 1.0 / (sigma * n)
            dx_hat = [dy[i][j] * g.data[0][j] for j in range(cols)]
            sum_dx_hat = sum(dx_hat[j] for j in range(cols))
            sum_dx_hat_xhat = sum(dx_hat[j] * x_hat[i][j] for j in range(cols))
            for j in range(cols):
                dx[i][j] = inv * (n * dx_hat[j] - sum_dx_hat
                                  - x_hat[i][j] * sum_dx_hat_xhat)
        return dx

    # --------------------------------------------------------
    # RoPE
    # --------------------------------------------------------
    def _apply_rope(self, x: List[List[float]], cos, sin) -> List[List[float]]:
        """应用RoPE旋转位置编码"""
        seq = len(x)
        dim = len(x[0]) if seq > 0 else 0
        out = [[0.0]*dim for _ in range(seq)]
        for pos in range(seq):
            for j in range(0, dim - 1, 2):
                c_val = cos[pos][j] if pos < len(cos) else 1.0
                s_val = sin[pos][j] if pos < len(sin) else 0.0
                out[pos][j] = x[pos][j] * c_val - x[pos][j+1] * s_val
                out[pos][j+1] = x[pos][j] * s_val + x[pos][j+1] * c_val
        return out

    def _apply_rope_backward(self, dout, cos, sin):
        """RoPE 反向: 逆旋转"""
        seq = len(dout)
        dim = len(dout[0]) if seq > 0 else 0
        dx = [[0.0]*dim for _ in range(seq)]
        for pos in range(seq):
            for j in range(0, dim - 1, 2):
                c_val = cos[pos][j] if pos < len(cos) else 1.0
                s_val = sin[pos][j] if pos < len(sin) else 0.0
                dx[pos][j] = dout[pos][j] * c_val + dout[pos][j+1] * s_val
                dx[pos][j+1] = -dout[pos][j] * s_val + dout[pos][j+1] * c_val
        return dx

    # --------------------------------------------------------
    # Attention (GQA + RoPE + ALiBi + sliding window) — 带缓存
    # --------------------------------------------------------
    def _attention_gqa(self, x: Tensor, wq, wk, wv, wo, cache_key=None) -> Tensor:
        """GQA + RoPE/ALiBi 注意力"""
        c = self.cfg
        H, S = c.hidden_dim, x.rows
        hd = c.head_dim
        nq = c.num_heads
        nkv = c.num_kv_heads
        group = c.kv_group_size
        caching = self._cache is not None

        # Q/KV 投影
        q = self._matmul(x, wq)   # (S, H)
        k = self._matmul(x, wk)   # (S, hd*nkv)
        v = self._matmul(x, wv)   # (S, hd*nkv)

        # 分头
        q_heads = [[[q.data[s][h*hd + j] for j in range(hd)] for s in range(S)]
                   for h in range(nq)]
        kv_heads_k = [[[k.data[s][h*hd + j] for j in range(hd)] for s in range(S)]
                      for h in range(nkv)]
        kv_heads_v = [[[v.data[s][h*hd + j] for j in range(hd)] for s in range(S)]
                      for h in range(nkv)]

        # RoPE
        if S not in self._rope_cos:
            cos, sin = self._compute_rope(S)
            self._rope_cos[S] = cos
            self._rope_sin[S] = sin
        cos, sin = self._rope_cos[S], self._rope_sin[S]

        q_heads_rot = [self._apply_rope(q_heads[h], cos, sin) for h in range(nq)]
        kv_heads_k_rot = [self._apply_rope(kv_heads_k[h], cos, sin) for h in range(nkv)]

        scale = 1.0 / math.sqrt(hd)
        attn_out = Tensor.zeros(S, H)
        attn_weights_all = []

        for h in range(nq):
            kv_idx = h // group
            qh = q_heads_rot[h]
            kh = kv_heads_k_rot[kv_idx]
            vh = kv_heads_v[kv_idx]

            scores = [[0.0]*S for _ in range(S)]
            for i in range(S):
                for j in range(S):
                    if c.sliding_window > 0 and abs(i - j) > c.sliding_window:
                        scores[i][j] = -1e9
                        continue
                    dot = sum(qh[i][d] * kh[j][d] for d in range(hd)) * scale
                    dot += self._alibi[h][i][j]
                    scores[i][j] = dot

                # Softmax
                mx = max(scores[i])
                if mx < -1e8:
                    scores[i] = [0.0]*S
                else:
                    exp_sum = sum(math.exp(s - mx) for s in scores[i])
                    for j in range(S):
                        scores[i][j] = math.exp(scores[i][j] - mx) / max(exp_sum, 1e-8)
            attn_weights_all.append(scores)

            # Output = attn @ V
            for i in range(S):
                for d in range(hd):
                    val = sum(scores[i][j] * vh[j][d] for j in range(S))
                    attn_out.data[i][h*hd + d] = val

        # Output projection
        output = self._matmul(attn_out, wo)

        if caching and cache_key is not None:
            self._cache[cache_key] = {
                'input': x,
                'wq': wq, 'wk': wk, 'wv': wv, 'wo': wo,
                'q_heads_rot': q_heads_rot,
                'kv_heads_k_rot': kv_heads_k_rot,
                'kv_heads_v': kv_heads_v,
                'attn_weights': attn_weights_all,
                'attn_out': attn_out,
                'cos': cos, 'sin': sin,
                'scale': scale,
            }
        return output

    def _attention_backward(self, doutput, layer, li):
        """注意力反向传播: doutput -> dx(=dn1), 累积 wq/wk/wv/wo 梯度"""
        c = self.cfg
        H = c.hidden_dim
        hd = c.head_dim
        nq = c.num_heads
        nkv = c.num_kv_heads
        group = c.kv_group_size
        S = len(doutput)

        ac = self._cache[('attn', li)]
        wq = ac['wq']; wk = ac['wk']; wv = ac['wv']; wo = ac['wo']
        q_heads_rot = ac['q_heads_rot']
        kv_heads_k_rot = ac['kv_heads_k_rot']
        kv_heads_v = ac['kv_heads_v']
        attn_weights = ac['attn_weights']   # nq x S x S
        attn_out = ac['attn_out']           # (S, H)
        cos = ac['cos']; sin = ac['sin']
        scale = ac['scale']
        x = ac['input']                      # n1 (S, H)

        # output = attn_out @ wo  ->  dwo, dattn_out
        for a in range(H):
            for b in range(H):
                g = 0.0
                for i in range(S):
                    g += attn_out.data[i][a] * doutput[i][b]
                wo.grad[a][b] += g
        dattn_out = [[0.0]*H for _ in range(S)]
        for i in range(S):
            for a in range(H):
                s = 0.0
                for b in range(H):
                    s += doutput[i][b] * wo.data[a][b]
                dattn_out[i][a] = s

        # 拆分到每个Q头; dK/dV 在共享的kv头上累积
        dQ_rot = [[[0.0]*hd for _ in range(S)] for _ in range(nq)]
        dK_rot = [[[0.0]*hd for _ in range(S)] for _ in range(nkv)]
        dV = [[[0.0]*hd for _ in range(S)] for _ in range(nkv)]

        for h in range(nq):
            kv_idx = h // group
            aw = attn_weights[h]          # (S, S)
            V = kv_heads_v[kv_idx]        # (S, hd)
            K = kv_heads_k_rot[kv_idx]    # (S, hd)
            Q = q_heads_rot[h]            # (S, hd)

            # attn_out_h[i][d] = sum_j aw[i][j] * V[j][d]
            # dattn_weights[i][j] = sum_d dattn_head[i][d] * V[j][d]
            dattn_weights = [[0.0]*S for _ in range(S)]
            for i in range(S):
                for j in range(S):
                    aw_ij = aw[i][j]
                    if aw_ij == 0.0:
                        continue
                    dw = 0.0
                    base = h * hd
                    for d in range(hd):
                        dah = dattn_out[i][base + d]
                        dV[kv_idx][j][d] += aw_ij * dah
                        dw += dah * V[j][d]
                    dattn_weights[i][j] = dw

            # Softmax 反向 + QK 反向
            for i in range(S):
                row_dot = 0.0
                for j in range(S):
                    row_dot += dattn_weights[i][j] * aw[i][j]
                for j in range(S):
                    aw_ij = aw[i][j]
                    if aw_ij == 0.0:
                        continue
                    dscores = aw_ij * (dattn_weights[i][j] - row_dot)
                    if c.sliding_window > 0 and abs(i - j) > c.sliding_window:
                        dscores = 0.0
                    if dscores == 0.0:
                        continue
                    sc = dscores * scale
                    for d in range(hd):
                        dQ_rot[h][i][d] += sc * K[j][d]
                        dK_rot[kv_idx][j][d] += sc * Q[i][d]

        # RoPE 反向 (V 不需要)
        dQ_pre = [self._apply_rope_backward(dQ_rot[h], cos, sin) for h in range(nq)]
        dK_pre = [self._apply_rope_backward(dK_rot[kv], cos, sin) for kv in range(nkv)]

        # 拼接回 (S, H) / (S, hd*nkv)
        dQ_concat = [[0.0]*H for _ in range(S)]
        for i in range(S):
            for h in range(nq):
                base = h * hd
                for d in range(hd):
                    dQ_concat[i][base + d] = dQ_pre[h][i][d]
        kcols = hd * nkv
        dK_concat = [[0.0]*kcols for _ in range(S)]
        dV_concat = [[0.0]*kcols for _ in range(S)]
        for i in range(S):
            for kv in range(nkv):
                base = kv * hd
                for d in range(hd):
                    dK_concat[i][base + d] = dK_pre[kv][i][d]
                    dV_concat[i][base + d] = dV[kv][i][d]

        # q = x @ wq ; k = x @ wk ; v = x @ wv
        dx = [[0.0]*H for _ in range(S)]

        # wq (H, H)
        for h in range(H):
            for f in range(H):
                g = 0.0
                for i in range(S):
                    g += x.data[i][h] * dQ_concat[i][f]
                wq.grad[h][f] += g
        for i in range(S):
            for h in range(H):
                s = 0.0
                for f in range(H):
                    s += dQ_concat[i][f] * wq.data[h][f]
                dx[i][h] += s

        # wk (H, kcols)
        for h in range(H):
            for f in range(kcols):
                g = 0.0
                for i in range(S):
                    g += x.data[i][h] * dK_concat[i][f]
                wk.grad[h][f] += g
        for i in range(S):
            for h in range(H):
                s = 0.0
                for f in range(kcols):
                    s += dK_concat[i][f] * wk.data[h][f]
                dx[i][h] += s

        # wv (H, kcols)
        for h in range(H):
            for f in range(kcols):
                g = 0.0
                for i in range(S):
                    g += x.data[i][h] * dV_concat[i][f]
                wv.grad[h][f] += g
        for i in range(S):
            for h in range(H):
                s = 0.0
                for f in range(kcols):
                    s += dV_concat[i][f] * wv.data[h][f]
                dx[i][h] += s

        return dx  # = dn1

    # --------------------------------------------------------
    # MoE FFN (SwiGLU) — 带缓存
    # --------------------------------------------------------
    def _moe_ffn(self, x: Tensor, router, experts, cache_key=None) -> Tuple[Tensor, float]:
        """MoE FFN with Top-K routing + SwiGLU"""
        c = self.cfg
        S, H = x.rows, x.cols
        E = c.num_experts
        K = c.num_activated_experts
        F = c.ffn_dim
        caching = self._cache is not None

        # 路由: (S, H) @ (H, E) = (S, E)
        logits = self._matmul(x, router)

        # Top-K 选择
        routes = []
        aux_loss = 0.0
        for i in range(S):
            scores = [(logits.data[i][e], e) for e in range(E)]
            scores.sort(reverse=True)
            topk = scores[:K]
            mx = topk[0][0]
            exp_sum = sum(math.exp(s - mx) for s, _ in topk)
            weights = [(math.exp(s - mx) / max(exp_sum, 1e-8), e) for s, e in topk]
            routes.append(weights)

        # 辅助损失 (负载均衡)
        load = [0]*E
        for i in range(S):
            for w, e in routes[i]:
                load[e] += 1
        for e in range(E):
            aux_loss += (load[e] / max(S, 1)) ** 2

        # 加权专家输出
        out = Tensor.zeros(S, H)
        per_token_cache = []
        for i in range(S):
            token_cache = []
            for weight, expert_idx in routes[i]:
                exp = experts[expert_idx]
                xv = x.data[i]
                w_gate = exp["w_gate"].data   # (H, F)
                w_up = exp["w_up"].data       # (F, H)
                w_down = exp["w_down"].data   # (H, F)

                # SwiGLU: gate_pre 只计算一次
                gate_pre = [sum(xv[h] * w_gate[h][f] for h in range(H))
                            for f in range(F)]
                gate = [0.0]*F
                for f in range(F):
                    gp = gate_pre[f]
                    sig = 1.0 / (1.0 + math.exp(-gp))
                    gate[f] = gp * sig          # SiLU(gate_pre)
                up = [sum(xv[h] * w_down[h][f] for h in range(H))
                      for f in range(F)]
                combined = [gate[f] * up[f] for f in range(F)]
                expert_out = [sum(combined[f] * w_up[f][j] for f in range(F))
                              for j in range(H)]

                for j in range(H):
                    out.data[i][j] += weight * expert_out[j]

                if caching:
                    token_cache.append({
                        'expert_idx': expert_idx,
                        'weight': weight,
                        'gate_pre': gate_pre,
                        'gate': gate,
                        'up': up,
                        'combined': combined,
                        'expert_out': expert_out,
                    })
            per_token_cache.append(token_cache)

        if caching and cache_key is not None:
            self._cache[cache_key] = {
                'input': x,
                'router': router,
                'experts': experts,
                'routes': routes,
                'per_token': per_token_cache,
                'logits': logits,
                'aux_loss': aux_loss,
            }
        return out, aux_loss

    def _moe_backward(self, dout, layer, li):
        """MoE 反向: dout -> dx(=dn2), 累积 router/专家梯度"""
        c = self.cfg
        S = len(dout)
        H = c.hidden_dim
        F = c.ffn_dim
        E = c.num_experts

        fc = self._cache[('ffn', li)]
        x = fc['input']              # n2 (S, H)
        router = fc['router']
        experts = fc['experts']
        routes = fc['routes']
        per_token = fc['per_token']

        dx = [[0.0]*H for _ in range(S)]
        drouter_logits = [[0.0]*E for _ in range(S)]

        for i in range(S):
            xv = x.data[i]
            token_routes = routes[i]
            tc_list = per_token[i]
            n_act = len(tc_list)
            dweights = [0.0]*n_act

            # 每个激活专家的反向
            for idx in range(n_act):
                tc = tc_list[idx]
                e = tc['expert_idx']
                weight = tc['weight']
                exp = experts[e]
                w_gate = exp["w_gate"].data   # (H, F)
                w_up = exp["w_up"].data       # (F, H)
                w_down = exp["w_down"].data   # (H, F)
                gate_pre = tc['gate_pre']
                gate = tc['gate']
                up = tc['up']
                combined = tc['combined']
                expert_out = tc['expert_out']

                # out[i][j] += weight * expert_out[j]
                # dweight = sum_j dout[i][j] * expert_out[j]
                dweight = 0.0
                for j in range(H):
                    dweight += dout[i][j] * expert_out[j]
                dweights[idx] = dweight

                # expert_out[j] = sum_f combined[f] * w_up[f][j]
                # dw_up[f][j] += weight * combined[f] * dout[i][j]
                # dcombined[f] = weight * sum_j dout[i][j] * w_up[f][j]
                dcombined = [0.0]*F
                for f in range(F):
                    cf = combined[f]
                    s = 0.0
                    for j in range(H):
                        s += dout[i][j] * w_up[f][j]
                    dcombined[f] = weight * s
                    if cf != 0.0:
                        wup_f = w_up[f]
                        wup_grad_f = exp["w_up"].grad[f]
                        for j in range(H):
                            wup_grad_f[j] += weight * cf * dout[i][j]

                # combined[f] = gate[f] * up[f]
                dgate = [dcombined[f] * up[f] for f in range(F)]
                dup = [dcombined[f] * gate[f] for f in range(F)]

                # gate[f] = gate_pre[f] * sigmoid(gate_pre[f])
                # dgate_pre[f] = dgate[f] * sig * (1 + gate_pre[f]*(1-sig))
                dgate_pre = [0.0]*F
                for f in range(F):
                    gp = gate_pre[f]
                    sig = 1.0 / (1.0 + math.exp(-gp))
                    dgate_pre[f] = dgate[f] * sig * (1.0 + gp * (1.0 - sig))

                # gate_pre[f] = sum_h xv[h]*w_gate[h][f] ; up[f] = sum_h xv[h]*w_down[h][f]
                # dw_gate[h][f] += xv[h]*dgate_pre[f] ; dw_down[h][f] += xv[h]*dup[f]
                for h in range(H):
                    xh = xv[h]
                    if xh == 0.0:
                        continue
                    wg_grad_h = exp["w_gate"].grad[h]
                    wd_grad_h = exp["w_down"].grad[h]
                    for f in range(F):
                        wg_grad_h[f] += xh * dgate_pre[f]
                        wd_grad_h[f] += xh * dup[f]

                # dx[i][h] += sum_f dgate_pre[f]*w_gate[h][f] + sum_f dup[f]*w_down[h][f]
                for h in range(H):
                    sg = 0.0
                    sd = 0.0
                    wg_h = w_gate[h]
                    wd_h = w_down[h]
                    for f in range(F):
                        sg += dgate_pre[f] * wg_h[f]
                        sd += dup[f] * wd_h[f]
                    dx[i][h] += sg + sd

            # 路由 softmax 反向 (top-k)
            # weights = softmax(topk_logits)
            # drouter_logits[i][e] = w * (dweight - sum(dweight*w))
            wsum = 0.0
            for idx in range(n_act):
                wsum += dweights[idx] * token_routes[idx][0]
            for idx in range(n_act):
                e = tc_list[idx]['expert_idx']
                w = token_routes[idx][0]
                drouter_logits[i][e] += w * (dweights[idx] - wsum)

        # router = x @ router (H, E)
        # drouter[h][e] = sum_i x[i][h]*drouter_logits[i][e]
        # dx[i][h] += sum_e drouter_logits[i][e]*router[h][e]
        for h in range(H):
            for e in range(E):
                g = 0.0
                for i in range(S):
                    g += x.data[i][h] * drouter_logits[i][e]
                router.grad[h][e] += g
        for i in range(S):
            for h in range(H):
                s = 0.0
                for e in range(E):
                    s += drouter_logits[i][e] * router.data[h][e]
                dx[i][h] += s

        return dx  # = dn2

    # --------------------------------------------------------
    # 标准 GELU FFN (fallback, 带缓存)
    # --------------------------------------------------------
    def _ffn(self, x: Tensor, w1, b1, w2, b2, cache_key=None) -> Tensor:
        """标准 GELU FFN (fallback)"""
        c = self.cfg
        S, H = x.rows, x.cols
        F = c.ffn_dim
        caching = self._cache is not None
        pre = [[0.0]*F for _ in range(S)]
        gelu = [[0.0]*F for _ in range(S)]
        for i in range(S):
            for f in range(F):
                val = sum(x.data[i][h] * w1.data[h][f] for h in range(H)) + b1.data[0][f]
                pre[i][f] = val
                gelu[i][f] = 0.5 * val * (1 + math.tanh(0.7978 * (val + 0.0447 * val**3)))
        out = Tensor.zeros(S, H)
        for i in range(S):
            for j in range(H):
                out.data[i][j] = sum(gelu[i][f] * w2.data[f][j] for f in range(F)) + b2.data[0][j]
        if caching and cache_key is not None:
            self._cache[cache_key] = {
                'input': x, 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,
                'pre': pre, 'gelu': gelu,
            }
        return out

    def _ffn_backward(self, dout, layer, li):
        """标准 GELU FFN 反向"""
        c = self.cfg
        S = len(dout); H = c.hidden_dim; F = c.ffn_dim
        fc = self._cache[('ffn', li)]
        x = fc['input']; w1 = fc['w1']; b1 = fc['b1']; w2 = fc['w2']; b2 = fc['b2']
        pre = fc['pre']; gelu = fc['gelu']

        dgelu = [[0.0]*F for _ in range(S)]
        for i in range(S):
            for f in range(F):
                s = 0.0
                for j in range(H):
                    s += dout[i][j] * w2.data[f][j]
                dgelu[i][f] = s
                gf = gelu[i][f]
                if gf != 0.0:
                    w2g_f = w2.grad[f]
                    for j in range(H):
                        w2g_f[j] += gf * dout[i][j]
            for j in range(H):
                b2.grad[0][j] += dout[i][j]

        a = 0.7978
        bcoef = 0.7978 * 0.0447
        dpre = [[0.0]*F for _ in range(S)]
        for i in range(S):
            for f in range(F):
                v = pre[i][f]
                u = a * (v + 0.0447 * v**3)
                t = math.tanh(u)
                du = a * (1.0 + 3.0 * 0.0447 * v * v)
                dpre[i][f] = dgelu[i][f] * (0.5*(1.0+t) + 0.5*v*(1.0-t*t)*du)

        dx = [[0.0]*H for _ in range(S)]
        for i in range(S):
            for h in range(H):
                s = 0.0
                for f in range(F):
                    s += dpre[i][f] * w1.data[h][f]
                dx[i][h] = s
            for f in range(F):
                b1.grad[0][f] += dpre[i][f]
        for h in range(H):
            for f in range(F):
                g = 0.0
                for i in range(S):
                    g += x.data[i][h] * dpre[i][f]
                w1.grad[h][f] += g
        return dx

    # --------------------------------------------------------
    # 工具
    # --------------------------------------------------------
    def _matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """矩阵乘法: (S, in) @ (in, out) -> (S, out)"""
        out = Tensor.zeros(a.rows, b.cols)
        for i in range(a.rows):
            for j in range(b.cols):
                val = 0.0
                for k in range(a.cols):
                    val += a.data[i][k] * b.data[k][j]
                out.data[i][j] = val
        return out

    # --------------------------------------------------------
    # Transformer Block (带缓存)
    # --------------------------------------------------------
    def _transformer_block(self, x: Tensor, layer: dict, layer_idx: int) -> Tensor:
        c = self.cfg
        alpha = c.deepnorm_alpha if c.norm_type == "deepnorm" else 1.0
        caching = self._cache is not None
        lcache = self._cache['layers'][layer_idx] if caching else None

        # Pre-norm + Attention + Residual (DeepNorm: x*alpha + sublayer)
        n1 = self._layernorm(x, layer["ln1_g"], layer["ln1_b"],
                             cache_key=('ln1', layer_idx))
        attn = self._attention_gqa(n1, layer["wq"], layer["wk"],
                                   layer["wv"], layer["wo"],
                                   cache_key=('attn', layer_idx))
        h = Tensor.zeros(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                h.data[i][j] = x.data[i][j] * alpha + attn.data[i][j]

        # Pre-norm + FFN + Residual
        n2 = self._layernorm(h, layer["ln2_g"], layer["ln2_b"],
                             cache_key=('ln2', layer_idx))
        if c.ffn_type == "moe":
            ffn_out, aux = self._moe_ffn(n2, layer["moe_router"],
                                         layer["moe_experts"],
                                         cache_key=('ffn', layer_idx))
        else:
            ffn_out = self._ffn(n2, layer["w1"], layer["b1"],
                                layer["w2"], layer["b2"],
                                cache_key=('ffn', layer_idx))
            aux = 0.0

        out = Tensor.zeros(h.rows, h.cols)
        for i in range(h.rows):
            for j in range(h.cols):
                out.data[i][j] = h.data[i][j] * alpha + ffn_out.data[i][j]

        if caching:
            lcache['x_input'] = x
            lcache['n1'] = n1
            lcache['h'] = h
            lcache['n2'] = n2
            lcache['alpha'] = alpha
            lcache['aux_loss'] = aux
        return out

    def _block_backward(self, dx, layer, li):
        """单个 Transformer block 反向: dx(w.r.t. out) -> dx(w.r.t. block 输入)"""
        c = self.cfg
        H = c.hidden_dim
        S = len(dx)
        lcache = self._cache['layers'][li]
        alpha = lcache['alpha']

        # out = h * alpha + ffn_out
        dh = [[dx[i][j] * alpha for j in range(H)] for i in range(S)]
        dffn_out = [row[:] for row in dx]

        # FFN 反向 -> dn2
        if c.ffn_type == "moe":
            dn2 = self._moe_backward(dffn_out, layer, li)
        else:
            dn2 = self._ffn_backward(dffn_out, layer, li)

        # n2 = LN2(h) -> dh += LN2_backward(dn2)
        dh_ln2 = self._layernorm_backward(dn2, self._cache[('ln2', li)],
                                          layer["ln2_g"], layer["ln2_b"])
        for i in range(S):
            for j in range(H):
                dh[i][j] += dh_ln2[i][j]

        # h = x * alpha + attn
        dx_in = [[dh[i][j] * alpha for j in range(H)] for i in range(S)]
        dattn = [row[:] for row in dh]

        # Attention 反向 -> dn1
        dn1 = self._attention_backward(dattn, layer, li)

        # n1 = LN1(x) -> dx_in += LN1_backward(dn1)
        dx_ln1 = self._layernorm_backward(dn1, self._cache[('ln1', li)],
                                          layer["ln1_g"], layer["ln1_b"])
        for i in range(S):
            for j in range(H):
                dx_in[i][j] += dx_ln1[i][j]

        return dx_in

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, ids: List[int], training: bool = False) -> Tensor:
        """前向传播: ids -> logits"""
        c = self.cfg
        ids = ids[:c.max_seq_len]
        S = len(ids)

        # 设置缓存
        self._cache = {} if training else None
        if training:
            self._cache['ids'] = list(ids)
            self._cache['layers'] = [{} for _ in range(c.num_layers)]

        # Embedding
        x = Tensor.zeros(S, c.hidden_dim)
        for i, tid in enumerate(ids):
            if 0 <= tid < c.vocab_size:
                for j in range(c.hidden_dim):
                    x.data[i][j] = self.embed.data[tid][j]
        if training:
            self._cache['x_embed'] = x

        # Transformer blocks
        for li, layer in enumerate(self.layers):
            x = self._transformer_block(x, layer, li)

        # Final norm
        x = self._layernorm(x, self.final_ln_g, self.final_ln_b,
                            cache_key='final_ln')
        if training:
            self._cache['x_final'] = x

        # Output projection
        logits = Tensor.zeros(S, c.vocab_size)
        for i in range(S):
            for j in range(c.vocab_size):
                val = sum(x.data[i][d] * self.head.data[d][j]
                          for d in range(c.hidden_dim))
                logits.data[i][j] = val + self.head_bias.data[0][j]

        self.forward_count += 1
        if not training:
            self._cache = None
        return logits

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    def _cross_entropy(self, logits: Tensor, targets: List[int]) -> float:
        """交叉熵损失"""
        S = logits.rows
        V = logits.cols
        loss = 0.0
        for i in range(min(S, len(targets))):
            tgt = targets[i]
            if tgt < 0 or tgt >= V:
                continue
            mx = max(logits.data[i])
            exp_sum = sum(math.exp(logits.data[i][j] - mx) for j in range(V))
            prob = math.exp(logits.data[i][tgt] - mx) / max(exp_sum, 1e-8)
            loss += -math.log(max(prob, 1e-8))
        return loss / max(S, 1)

    # --------------------------------------------------------
    # Backward (真正的链式法则反向传播)
    # --------------------------------------------------------
    def _backward(self, logits: Tensor, targets: List[int], lr: float):
        """真正的链式法则反向传播 + 梯度裁剪 + SGD 更新"""
        c = self.cfg
        S = logits.rows
        V = c.vocab_size
        H = c.hidden_dim
        cache = self._cache

        # 0. 清零所有梯度
        for p in self._all_params():
            p.zero_grad()

        # 1. Softmax + Cross-Entropy 反向: dlogits
        dlogits = [[0.0]*V for _ in range(S)]
        for i in range(min(S, len(targets))):
            tgt = targets[i]
            if tgt < 0 or tgt >= V:
                continue
            mx = max(logits.data[i])
            exp_sum = sum(math.exp(logits.data[i][j] - mx) for j in range(V))
            for j in range(V):
                prob = math.exp(logits.data[i][j] - mx) / max(exp_sum, 1e-8)
                dlogits[i][j] = prob
            dlogits[i][tgt] -= 1.0
            for j in range(V):
                dlogits[i][j] /= S   # 损失对 S 取平均

        # 2. Head 反向: logits = x_final @ head + head_bias
        x_final = cache['x_final']   # (S, H)
        for j in range(V):
            gb = 0.0
            for i in range(S):
                gb += dlogits[i][j]
            self.head_bias.grad[0][j] += gb
        for d in range(H):
            for j in range(V):
                g = 0.0
                for i in range(S):
                    g += x_final.data[i][d] * dlogits[i][j]
                self.head.grad[d][j] += g
        # dx_final
        dx = [[0.0]*H for _ in range(S)]
        for i in range(S):
            for d in range(H):
                s = 0.0
                for j in range(V):
                    s += dlogits[i][j] * self.head.data[d][j]
                dx[i][d] = s

        # 3. Final LayerNorm 反向
        dx = self._layernorm_backward(dx, cache['final_ln'],
                                      self.final_ln_g, self.final_ln_b)

        # 4. Transformer blocks 反向 (逆序)
        for li in range(c.num_layers - 1, -1, -1):
            dx = self._block_backward(dx, self.layers[li], li)

        # 5. Embedding 反向 (scatter-add)
        ids = cache['ids']
        for i in range(S):
            tid = ids[i]
            if 0 <= tid < V:
                for j in range(H):
                    self.embed.grad[tid][j] += dx[i][j]

        # 6. 梯度裁剪 (global norm, max_norm=1.0)
        self._clip_grads(1.0)

        # 7. SGD 更新
        for p in self._all_params():
            for i in range(p.rows):
                for j in range(p.cols):
                    p.data[i][j] -= lr * p.grad[i][j]

        # 8. 清空缓存
        self._cache = None

    def _clip_grads(self, max_norm: float = 1.0):
        """全局梯度范数裁剪"""
        total_sq = 0.0
        for p in self._all_params():
            for i in range(p.rows):
                row = p.grad[i]
                for j in range(p.cols):
                    g = row[j]
                    total_sq += g * g
        total_norm = math.sqrt(total_sq)
        if total_norm > max_norm and total_norm > 0.0:
            scale = max_norm / total_norm
            for p in self._all_params():
                for i in range(p.rows):
                    row = p.grad[i]
                    for j in range(p.cols):
                        row[j] *= scale

    # --------------------------------------------------------
    # Train step
    # --------------------------------------------------------
    def train_step(self, input_ids: List[int], target_ids: List[int],
                   lr: float = None) -> float:
        """训练步: forward + backward + update"""
        c = self.cfg
        lr = lr or c.learning_rate
        logits = self.forward(input_ids, training=True)
        loss = self._cross_entropy(logits, target_ids)
        self._backward(logits, target_ids, lr)
        return loss

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------
    def generate(self, prompt: List[int], max_new: int = 32,
                 temperature: float = 0.8, top_k: int = 0) -> List[int]:
        """自回归生成"""
        c = self.cfg
        ids = list(prompt[:c.max_seq_len])
        for _ in range(max_new):
            ctx = ids[-c.max_seq_len:]
            logits = self.forward(ctx)   # 不缓存
            last = logits.data[-1]
            # Temperature
            scaled = [l / max(temperature, 1e-8) for l in last]
            mx = max(scaled)
            exp_vals = [math.exp(s - mx) for s in scaled]
            exp_sum = sum(exp_vals)
            probs = [e / max(exp_sum, 1e-8) for e in exp_vals]

            # Top-k
            if top_k > 0 and top_k < len(probs):
                indexed = [(p, i) for i, p in enumerate(probs)]
                indexed.sort(reverse=True)
                topk_sum = sum(p for p, _ in indexed[:top_k])
                probs = [0.0]*len(probs)
                for p, i in indexed[:top_k]:
                    probs[i] = p / max(topk_sum, 1e-8)

            # Sample
            r = random.random()
            cum = 0.0
            for i, p in enumerate(probs):
                cum += p
                if r < cum:
                    ids.append(i)
                    break
            else:
                ids.append(len(probs) - 1)
        return ids

    # --------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------
    def save(self, path: str):
        """保存模型"""
        c = self.cfg
        data = {
            "version": self.VERSION,
            "config": {
                "vocab_size": c.vocab_size, "hidden_dim": c.hidden_dim,
                "num_heads": c.num_heads, "num_kv_heads": c.num_kv_heads,
                "num_layers": c.num_layers, "ffn_dim": c.ffn_dim,
                "max_seq_len": c.max_seq_len,
                "pos_encoding": c.pos_encoding, "norm_type": c.norm_type,
                "ffn_type": c.ffn_type, "num_experts": c.num_experts,
                "num_activated_experts": c.num_activated_experts,
                "sliding_window": c.sliding_window,
            },
            "embed": self.embed.data,
            "head": self.head.data,
            "head_bias": self.head_bias.data,
            "final_ln_g": self.final_ln_g.data,
            "final_ln_b": self.final_ln_b.data,
            "layers": [],
        }
        for layer in self.layers:
            l = {}
            for k in ["ln1_g", "ln1_b", "ln2_g", "ln2_b",
                       "wq", "wk", "wv", "wo", "moe_router",
                       "w1", "b1", "w2", "b2"]:
                if k in layer:
                    l[k] = layer[k].data
            l["moe_experts"] = []
            for exp in layer["moe_experts"]:
                l["moe_experts"].append({
                    "w_gate": exp["w_gate"].data,
                    "w_up": exp["w_up"].data,
                    "w_down": exp["w_down"].data,
                })
            data["layers"].append(l)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "LingyuanModel":
        """加载模型"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = ModelConfig(**data["config"])
        model = cls(cfg)
        model.embed.data = data["embed"]
        model.head.data = data["head"]
        model.head_bias.data = data["head_bias"]
        model.final_ln_g.data = data["final_ln_g"]
        model.final_ln_b.data = data["final_ln_b"]
        for li, ldata in enumerate(data["layers"]):
            layer = model.layers[li]
            for k in ["ln1_g", "ln1_b", "ln2_g", "ln2_b",
                       "wq", "wk", "wv", "wo", "moe_router",
                       "w1", "b1", "w2", "b2"]:
                if k in ldata and k in layer:
                    layer[k].data = ldata[k]
            for ei, edata in enumerate(ldata.get("moe_experts", [])):
                exp = layer["moe_experts"][ei]
                exp["w_gate"].data = edata["w_gate"]
                exp["w_up"].data = edata["w_up"]
                exp["w_down"].data = edata["w_down"]
        return model

    def stats(self) -> dict:
        c = self.cfg
        params = (self.embed.size() + self.head.size() +
                  self.head_bias.size() + self.final_ln_g.size() +
                  self.final_ln_b.size())
        moe_params = 0
        for layer in self.layers:
            for k in ["wq", "wk", "wv", "wo", "ln1_g", "ln1_b",
                       "ln2_g", "ln2_b", "moe_router", "w1", "b1", "w2", "b2"]:
                if k in layer:
                    params += layer[k].size()
            for exp in layer["moe_experts"]:
                for k in ["w_gate", "w_up", "w_down"]:
                    moe_params += exp[k].size()
        return {
            "version": self.VERSION,
            "params": f"{params:,}",
            "moe_params": f"{moe_params:,}",
            "total_params": f"{params + moe_params:,}",
            "config": {
                "layers": c.num_layers, "hidden": c.hidden_dim,
                "heads": c.num_heads, "kv_heads": c.num_kv_heads,
                "gqa_ratio": c.kv_group_size,
                "experts": c.num_experts,
                "activated": c.num_activated_experts,
                "norm": c.norm_type, "ffn": c.ffn_type,
                "pos": c.pos_encoding,
                "deepnorm_alpha": round(c.deepnorm_alpha, 4),
            },
            "forward_count": self.forward_count,
        }


# ============================================================
# Training Engine
# ============================================================

class Trainer:
    """训练引擎 — warmup+cosine LR, 早停, 断点"""

    def __init__(self, model: LingyuanModel, tokenizer: CharTokenizer,
                 loader: TextDataLoader):
        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.best_loss = float('inf')
        self.no_improve = 0
        self.history = []

    def _compute_lr(self, step, total, base_lr, warmup=0.1):
        warmup_steps = max(1, int(total * warmup))
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def train(self, epochs=10, steps_per_epoch=50, base_lr=0.001,
              patience=5, log_interval=10, resume=False) -> dict:
        total_steps = epochs * steps_per_epoch
        current_step = 0
        t0 = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start = time.time()

            for step in range(steps_per_epoch):
                current_step += 1
                lr = self._compute_lr(current_step - 1, total_steps, base_lr)

                inputs, targets = self.loader.sample_batch()
                batch_loss = 0.0
                for inp, tgt in zip(inputs, targets):
                    loss = self.model.train_step(inp, tgt, lr)
                    batch_loss += loss

                avg_loss = batch_loss / max(len(inputs), 1)
                epoch_loss += avg_loss

                if step % log_interval == 0 or step == steps_per_epoch - 1:
                    elapsed = time.time() - epoch_start
                    print(f"  [E{epoch+1} S{step+1}] loss={avg_loss:.4f} "
                          f"lr={lr:.6f} t={elapsed:.0f}s", flush=True)

            avg_epoch = epoch_loss / steps_per_epoch
            epoch_time = time.time() - epoch_start
            self.history.append({
                "epoch": epoch + 1, "loss": avg_epoch, "time": epoch_time
            })

            print(f"  >> Epoch {epoch+1}: loss={avg_epoch:.4f} "
                  f"time={epoch_time:.0f}s best={self.best_loss:.4f}", flush=True)

            if avg_epoch < self.best_loss - 1e-6:
                self.best_loss = avg_epoch
                self.no_improve = 0
            else:
                self.no_improve += 1
                if self.no_improve >= patience:
                    print(f"  !! 早停: {patience}轮无改善", flush=True)
                    break

        total_time = time.time() - t0
        return {
            "epochs": len(self.history),
            "steps": current_step,
            "time": f"{total_time:.0f}s",
            "best_loss": round(self.best_loss, 4),
            "history": self.history,
        }


# ============================================================
# 内置语料
# ============================================================

BUILTIN_CORPUS = """
春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
床前明月光，疑是地上霜。举头望明月，低头思故乡。
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。
离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。
两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。
独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。
空山新雨后，天气晚来秋。明月松间照，清泉石上流。
千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。
月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。
远上寒山石径斜，白云生处有人家。停车坐爱枫林晚，霜叶红于二月花。
葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。
秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。
寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。
渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。
故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。
朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。
日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。
天门中断楚江开，碧水东流至此回。两岸青山相对出，孤帆一片日边来。
莫愁前路无知己，天下谁人不识君。六翮飘飖私自怜，一离京洛十余年。
"""


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="灵元 V7.0 ULTRA 训练")
    parser.add_argument("--data", type=str, default="", help="训练数据文件")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--config", type=str, default="ultra_lite",
                        help="ultra_lite / tiny")
    parser.add_argument("--model", type=str, default="", help="加载已有模型续训")
    parser.add_argument("--output", type=str, default="lingyuan_v7.het")
    parser.add_argument("--generate", action="store_true", help="仅生成")
    parser.add_argument("--prompt", type=str, default="春眠不觉晓")
    args = parser.parse_args()

    output_dir = "/workspace/lingyuan_train"
    os.makedirs(output_dir, exist_ok=True)

    if args.generate:
        # 生成模式
        model_path = args.model or os.path.join(output_dir, args.output)
        if not os.path.exists(model_path):
            print(f"模型不存在: {model_path}")
            return
        model = LingyuanModel.load(model_path)
        tokenizer = CharTokenizer(vocab_size=model.cfg.vocab_size)
        tokenizer.fit_on_text(BUILTIN_CORPUS)
        prompt_ids = tokenizer.encode(args.prompt)
        if prompt_ids:
            output = model.generate(prompt_ids, max_new=64, temperature=0.8, top_k=20)
            text = tokenizer.decode(output)
            print(f"Prompt: {args.prompt}")
            print(f"生成: {text}")
        return

    # 配置
    if args.config == "tiny":
        cfg = ModelConfig.tiny()
    else:
        cfg = ModelConfig.ultra_lite()
    cfg.learning_rate = args.lr

    # 分词器 + 数据
    tokenizer = CharTokenizer(vocab_size=cfg.vocab_size)
    loader = TextDataLoader(tokenizer, seq_len=cfg.max_seq_len,
                            batch_size=args.batch_size)

    if args.data and os.path.exists(args.data):
        loader.load_file(args.data)
        print(f"数据加载: {args.data}")
    else:
        loader.load_text(BUILTIN_CORPUS)
        print("使用内置古诗词语料")

    print(f"词表: {tokenizer.actual_size} | 序列数: {len(loader._data)}")

    # 模型
    if args.model and os.path.exists(args.model):
        model = LingyuanModel.load(args.model)
        print(f"模型加载: {args.model} (续训)")
    else:
        model = LingyuanModel(cfg)
        print("新建模型")

    stats = model.stats()
    print(f"版本: {stats['version']}")
    print(f"参数: {stats['total_params']} (基础: {stats['params']}, MoE: {stats['moe_params']})")
    print(f"架构: {stats['config']}")

    # 训练
    trainer = Trainer(model, tokenizer, loader)
    result = trainer.train(
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        base_lr=args.lr,
        patience=args.patience,
        log_interval=args.log_interval,
    )

    # 保存模型
    model_path = os.path.join(output_dir, args.output)
    model.save(model_path)
    print(f"\n模型保存: {model_path}")

    # 保存训练报告
    report_path = os.path.join(output_dir, "training_report.json")
    report = {
        "version": stats["version"],
        "config": stats["config"],
        "params": stats["total_params"],
        "training": result,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告保存: {report_path}")

    # 生成测试
    prompt = "春眠不觉晓"
    prompt_ids = tokenizer.encode(prompt)
    if prompt_ids:
        gen = model.generate(prompt_ids, max_new=32, temperature=0.8, top_k=20)
        text = tokenizer.decode(gen)
        print(f"\n生成测试 (prompt='{prompt}'):")
        print(f"  {text}")

    print(f"\n{'='*50}")
    print(f"训练完成: {result['epochs']}轮 {result['steps']}步")
    print(f"用时: {result['time']} | 最优loss: {result['best_loss']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
