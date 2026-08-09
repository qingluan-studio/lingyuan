
# ============================================================
# LINGYUAN MODEL - PART 9
# 模型本体: BPE分词器 / 位置编码 / 多头注意力 / Transformer层
#          完整模型 / 采样器 / KV缓存 / 训练引擎 / 配置管理
#
# 对应52项清单 #1-9: 灵元大模型核心模型实现
# 纯Python标准库实现 (零外部依赖)
# ============================================================

import uuid
import math
import random
import json
import os
import time
import re
from collections import deque, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime


# ============================================================
# 数学工具函数 (纯Python矩阵/向量运算, 零外部依赖)
# ============================================================

def _linear_2d(x: List[List[float]], w: List[List[float]],
               b: Optional[List[float]] = None) -> List[List[float]]:
    """线性层: x (seq×in) @ w (in×out) + b (out) -> (seq×out)"""
    if not x or not w:
        return []
    out_dim = len(w[0])
    result = []
    for row in x:
        out = [0.0] * out_dim
        for i, xi in enumerate(row):
            if xi == 0.0:
                continue
            wi = w[i]
            for j in range(out_dim):
                out[j] += xi * wi[j]
        if b is not None:
            for j in range(out_dim):
                out[j] += b[j]
        result.append(out)
    return result


def _matmul_2d(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵乘法: a (m×k) @ b (k×n) -> (m×n)"""
    if not a or not b:
        return []
    k = len(b)
    n = len(b[0])
    m = len(a)
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        ai = a[i]
        ri = result[i]
        for p in range(k):
            aip = ai[p]
            if aip == 0.0:
                continue
            bp = b[p]
            for j in range(n):
                ri[j] += aip * bp[j]
    return result


def _transpose_2d(m: List[List[float]]) -> List[List[float]]:
    """转置: (m×n) -> (n×m)"""
    if not m:
        return []
    rows, cols = len(m), len(m[0])
    return [[m[i][j] for i in range(rows)] for j in range(cols)]


def _softmax_vec(v: List[float]) -> List[float]:
    """一维向量softmax (数值稳定)"""
    if not v:
        return []
    m = max(v)
    exps = [math.exp(x - m) for x in v]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(v)] * len(v)
    return [e / s for e in exps]


def _softmax_rows(m: List[List[float]]) -> List[List[float]]:
    """逐行softmax"""
    return [_softmax_vec(row) for row in m]


def _glorot_uniform(fan_in: int, fan_out: int) -> List[List[float]]:
    """Xavier/Glorot均匀初始化 -> (fan_in×out)矩阵"""
    limit = math.sqrt(6.0 / (fan_in + fan_out)) if (fan_in + fan_out) > 0 else 0.01
    return [[random.uniform(-limit, limit) for _ in range(fan_out)]
            for _ in range(fan_in)]


def _zeros_2d(rows: int, cols: int) -> List[List[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _zeros_1d(n: int) -> List[float]:
    return [0.0] * n


def _rms_norm_rows(x: List[List[float]], weight: List[float],
                   eps: float = 1e-6) -> List[List[float]]:
    """RMSNorm (逐行): x * weight / sqrt(mean(x^2) + eps)"""
    result = []
    for row in x:
        n = len(row)
        ms = sum(v * v for v in row) / n
        rms = math.sqrt(ms + eps)
        result.append([(v / rms) * w for v, w in zip(row, weight)])
    return result


def _split_heads_2d(x: List[List[float]], num_heads: int) -> List[List[List[float]]]:
    """(seq×hidden) -> [num_heads × (seq×head_dim)]"""
    if not x:
        return [[] for _ in range(num_heads)]
    hidden = len(x[0])
    head_dim = hidden // num_heads
    heads = []
    for h in range(num_heads):
        head = [[row[h * head_dim + d] for d in range(head_dim)] for row in x]
        heads.append(head)
    return heads


def _merge_heads_2d(heads: List[List[List[float]]]) -> List[List[float]]:
    """[num_heads × (seq×head_dim)] -> (seq×hidden)"""
    if not heads:
        return []
    num_heads = len(heads)
    seq_len = len(heads[0])
    result = []
    for s in range(seq_len):
        row = []
        for h in range(num_heads):
            row.extend(heads[h][s])
        result.append(row)
    return result


def _repeat_kv_heads(kv_heads: List[List[List[float]]],
                     n_rep: int) -> List[List[List[float]]]:
    """将num_kv_heads个kv head重复n_rep次, 用于GQA/MQA"""
    if n_rep <= 1:
        return kv_heads
    result = []
    for h in kv_heads:
        for _ in range(n_rep):
            result.append(h)
    return result


def _silu(x: float) -> float:
    """SiLU/Swish激活: x * sigmoid(x)"""
    return x / (1.0 + math.exp(-x)) if x >= -50 else 0.0


def _cross_entropy_loss(logits: List[List[float]],
                        targets: List[int]) -> float:
    """交叉熵损失 (logits: seq×vocab, targets: seq)"""
    if not logits:
        return 0.0
    total = 0.0
    n = 0
    for i, tgt in enumerate(targets):
        if i >= len(logits):
            break
        row = logits[i]
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        s = sum(exps)
        if s > 0 and 0 <= tgt < len(row):
            prob = exps[tgt] / s
            total += -math.log(max(prob, 1e-12))
            n += 1
    return total / max(n, 1)


# ============================================================
# 反向传播辅助函数 (真实梯度计算)
# ============================================================

def _silu_grad(x: float) -> float:
    """SiLU/Swish的导数: sigmoid(x) * (1 + x * (1 - sigmoid(x)))"""
    if x < -50:
        return 0.0
    sig = 1.0 / (1.0 + math.exp(-x))
    return sig * (1.0 + x * (1.0 - sig))


def _rmsnorm_backward(dout: List[List[float]],
                      x_norm: List[List[float]],
                      rms: List[float],
                      weight: List[float],
                      dim: int) -> Tuple[List[List[float]], List[float]]:
    """RMSNorm反向传播

    y = x_norm * weight,  x_norm = x / rms

    Args:
        dout: dL/dy (seq × dim)
        x_norm: x / rms (seq × dim) — 前向缓存的归一化值
        rms: 各位置的RMS值 (seq,)
        weight: 可学习缩放参数 (dim,)
        dim: 维度

    Returns:
        (dx, dweight) — dL/dx (seq×dim) 和 dL/dweight (dim,)
    """
    seq_len = len(x_norm)
    dx: List[List[float]] = [[0.0] * dim for _ in range(seq_len)]
    dweight: List[float] = [0.0] * dim

    for s in range(seq_len):
        xn = x_norm[s]
        dy = dout[s]
        r = rms[s]
        if r < 1e-12:
            r = 1e-12
        # g_s = sum_j(dy_j * y_j) = sum_j(dy_j * x_norm_j * weight_j)
        g_s = 0.0
        for j in range(dim):
            g_s += dy[j] * xn[j] * weight[j]
        inv_n = 1.0 / dim
        for d in range(dim):
            dweight[d] += dy[d] * xn[d]
            dx[s][d] = (dy[d] * weight[d] - xn[d] * g_s * inv_n) / r

    return dx, dweight


def _rope_backward(dout: List[List[float]],
                   rope_cos: List[List[float]],
                   rope_sin: List[List[float]],
                   seq_offset: int,
                   head_dim: int) -> List[List[float]]:
    """RoPE反向传播

    RoPE是旋转操作, 其雅可比的转置等于逆旋转。
    正向: x'[2i]   = x[2i]*cos - x[2i+1]*sin
          x'[2i+1] = x[2i]*sin + x[2i+1]*cos
    反向: dL/dx[2i]   = dL/dx'[2i]*cos + dL/dx'[2i+1]*sin
          dL/dx[2i+1] = -dL/dx'[2i]*sin + dL/dx'[2i+1]*cos
    """
    seq_len = len(dout)
    half = head_dim // 2
    dx: List[List[float]] = []
    for si in range(seq_len):
        pos = seq_offset + si
        if pos >= len(rope_cos):
            pos = len(rope_cos) - 1
        cos = rope_cos[pos]
        sin = rope_sin[pos]
        d_row = dout[si]
        new_row = list(d_row)  # 保留未旋转维度
        for i in range(half):
            d1 = d_row[2 * i]
            d2 = d_row[2 * i + 1]
            new_row[2 * i] = d1 * cos[i] + d2 * sin[i]
            new_row[2 * i + 1] = -d1 * sin[i] + d2 * cos[i]
        dx.append(new_row)
    return dx


def _softmax_backward_row(dout: List[float],
                          probs: List[float]) -> List[float]:
    """Softmax反向传播 (单行)

    dL/dx_j = probs_j * (dL/dy_j - sum_k(probs_k * dL/dy_k))
    """
    n = len(probs)
    dot = 0.0
    for k in range(n):
        dot += probs[k] * dout[k]
    return [probs[j] * (dout[j] - dot) for j in range(n)]


def _outer_product_add(grad: List[List[float]],
                       a: List[float], b: List[float],
                       scale: float = 1.0) -> None:
    """grad += scale * a ⊗ b  (a: m, b: n, grad: m×n)  原地累加"""
    m = len(a)
    n = len(b)
    for i in range(m):
        ai = a[i] * scale
        if ai == 0.0:
            continue
        gi = grad[i]
        for j in range(n):
            gi[j] += ai * b[j]


# 预设配置 (模型与ModelConfig共用)
_MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "hidden_dim": 128, "num_layers": 4, "num_heads": 4, "num_kv_heads": 4,
        "ffn_dim": 256, "max_seq_len": 512, "vocab_size": 2048,
        "rope_theta": 10000.0, "norm_eps": 1e-6,
    },
    "small": {
        "hidden_dim": 512, "num_layers": 12, "num_heads": 8, "num_kv_heads": 8,
        "ffn_dim": 1024, "max_seq_len": 2048, "vocab_size": 8192,
        "rope_theta": 10000.0, "norm_eps": 1e-6,
    },
    "base": {
        "hidden_dim": 1024, "num_layers": 32, "num_heads": 16, "num_kv_heads": 16,
        "ffn_dim": 4096, "max_seq_len": 4096, "vocab_size": 32000,
        "rope_theta": 10000.0, "norm_eps": 1e-5,
    },
    "large": {
        "hidden_dim": 2048, "num_layers": 64, "num_heads": 32, "num_kv_heads": 8,
        "ffn_dim": 8192, "max_seq_len": 8192, "vocab_size": 64000,
        "rope_theta": 500000.0, "norm_eps": 1e-5,
    },
}


# ============================================================
# #1 BPETokenizer [BPE分词器]
# ============================================================

class BPETokenizer:
    """BPE (Byte Pair Encoding) 分词器

    功能:
    - 词表管理 (token↔id双向映射)
    - BPE合并规则训练 (从语料迭代学习最高频字节对)
    - encode(text)->List[int], decode(ids)->str
    - 特殊token: <pad>, <bos>, <eos>, <unk>
    - 预训练默认词表 (常见中英文token)
    - save/load 词表到JSON
    """

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # 预训练默认词表 (常见中英文token)
    DEFAULT_VOCAB = [
        # ASCII字母
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        # 数字
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        # 常见标点与符号
        " ", ".", ",", "!", "?", ":", ";", "'", '"', "(", ")", "[", "]",
        "{", "}", "-", "+", "=", "/", "\\", "@", "#", "$", "%", "^", "&",
        "*", "_", "|", "~", "`", "<", ">", "\n", "\t",
        # 常见英文子词
        "the", "ing", "ed", "er", "ion", "tion", "ness", "ment", "able",
        "ible", "ous", "ful", "less", "est", "ity", "ate", "ive", "al",
        "ly", "re", "un", "pre", "pro", "con", "com", "dis", "sub", "per",
        "and", "for", "are", "but", "not", "you", "all", "can", "her",
        "was", "one", "our", "out", "has", "have", "this", "that", "with",
        "from", "they", "will", "would", "there", "their", "what", "about",
        # 常见中文字符 (高频)
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
        "一", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没", "看", "好", "自", "己", "这", "那", "他", "她", "它", "们",
        "个", "中", "来", "对", "下", "里", "为", "什", "么", "可", "以",
        "把", "让", "从", "与", "及", "但", "而", "或", "因", "所", "如",
        "学", "生", "工", "作", "时", "间", "天", "年", "月", "日", "事",
        "情", "想", "能", "用", "做", "它", "家", "国", "世", "界", "大",
        "小", "多", "少", "前", "后", "左", "右", "高", "低", "新", "旧",
        "问", "答", "话", "语", "言", "文", "字", "书", "名", "知", "道",
    ]

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []       # BPE合并规则 (有序)
        self.merge_ranks: Dict[Tuple[str, str], int] = {}  # pair -> 优先级
        self.special_tokens: List[str] = list(self.SPECIAL_TOKENS)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self._init_default_vocab()

    def _init_default_vocab(self):
        """用特殊token和默认词表初始化"""
        for t in self.special_tokens:
            self._add_token(t)
        for c in self.DEFAULT_VOCAB:
            if c not in self.token_to_id:
                self._add_token(c)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    @property
    def actual_vocab_size(self) -> int:
        return len(self.token_to_id)

    # ---------- BPE训练 ----------

    def _split_to_words(self, text: str) -> List[str]:
        """分词: 中文单字 / 英文连续字母 / 数字 / 单个标点"""
        return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|[0-9]+|[^\s\u4e00-\u9fff]', text)

    @staticmethod
    def _apply_merge(tokens: List[str], pair: Tuple[str, str],
                     new_token: str) -> List[str]:
        """对token列表应用单条合并规则"""
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def train(self, corpus: List[str], target_vocab_size: Optional[int] = None,
              verbose: bool = False) -> Dict[str, Any]:
        """从语料训练BPE合并规则

        Args:
            corpus: 文本语料列表
            target_vocab_size: 目标词表大小
            verbose: 是否打印进度

        Returns:
            训练统计信息
        """
        target = target_vocab_size or self.vocab_size
        start_time = time.time()

        # 统计词频
        word_freqs: Counter = Counter()
        for text in corpus:
            for w in self._split_to_words(text):
                word_freqs[w] += 1

        # 每个词初始化为字符序列
        word_tokens: Dict[str, List[str]] = {w: list(w) for w in word_freqs}
        merges_learned = 0

        while len(self.token_to_id) < target:
            # 统计相邻pair频率
            pair_freqs: Counter = Counter()
            for w, freq in word_freqs.items():
                toks = word_tokens[w]
                for i in range(len(toks) - 1):
                    pair_freqs[(toks[i], toks[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair, best_freq = pair_freqs.most_common(1)[0]
            if best_freq < 2:
                break  # 没有足够频繁的pair

            new_token = best_pair[0] + best_pair[1]
            self._add_token(new_token)
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1

            # 对所有词应用此合并
            for w in word_tokens:
                word_tokens[w] = self._apply_merge(
                    word_tokens[w], best_pair, new_token)
            merges_learned += 1

            if verbose and merges_learned % 50 == 0:
                print(f"  BPE: 合并#{merges_learned} '{best_pair}' -> "
                      f"'{new_token}' (freq={best_freq}, vocab={len(self.token_to_id)})")

        elapsed = time.time() - start_time
        return {
            "merges_learned": merges_learned,
            "final_vocab_size": len(self.token_to_id),
            "target_vocab_size": target,
            "corpus_words": sum(word_freqs.values()),
            "unique_words": len(word_freqs),
            "elapsed_sec": round(elapsed, 3),
        }

    # ---------- 编码 / 解码 ----------

    def _bpe_encode_word(self, word: str) -> List[str]:
        """对单个词应用BPE合并, 返回token列表"""
        if not word:
            return []
        if word in self.token_to_id:
            return [word]

        tokens = list(word)
        # 迭代合并: 每次选择优先级最高(rank最小)的相邻pair
        while len(tokens) > 1:
            best_rank = None
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_idx = i
            if best_idx == -1:
                break
            tokens = (tokens[:best_idx]
                      + [tokens[best_idx] + tokens[best_idx + 1]]
                      + tokens[best_idx + 2:])
        return tokens

    def _encode_text(self, text: str) -> List[int]:
        ids = []
        for word in self._split_to_words(text):
            for t in self._bpe_encode_word(word):
                if t in self.token_to_id:
                    ids.append(self.token_to_id[t])
                else:
                    # 尝试逐字符编码
                    for ch in t:
                        ids.append(self.token_to_id.get(ch, self.unk_id))
        return ids

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> List[int]:
        """文本 -> token id列表"""
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self._encode_text(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """token id列表 -> 文本"""
        parts = []
        for i in ids:
            t = self.id_to_token.get(i, "<unk>")
            if skip_special and t in self.special_tokens:
                continue
            parts.append(t)
        return "".join(parts)

    # ---------- 存取 ----------

    def save(self, path: str) -> bool:
        """保存词表到JSON"""
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """从JSON加载词表"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data.get("vocab_size", 1000))
        tok.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tok.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        tok.merges = [tuple(m) for m in data.get("merges", [])]
        tok.merge_ranks = {tuple(m): i for i, m in enumerate(tok.merges)}
        tok.special_tokens = data.get("special_tokens", cls.SPECIAL_TOKENS)
        tok.pad_id = data.get("pad_id", 0)
        tok.bos_id = data.get("bos_id", 1)
        tok.eos_id = data.get("eos_id", 2)
        tok.unk_id = data.get("unk_id", 3)
        return tok

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.actual_vocab_size,
            "target_vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "special_tokens": self.special_tokens,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }


# ============================================================
# #2 PositionalEncoding [位置编码]
# ============================================================

class PositionalEncoding:
    """位置编码

    支持三种方案:
    - RoPE (旋转位置编码): precompute_freqs, apply_rotary_emb
    - ALiBi (注意力线性偏置): 可外推的位置偏置
    - 绝对位置编码 (正弦/余弦)
    """

    def __init__(self, dim: int, max_seq_len: int = 2048,
                 method: str = "rope", rope_theta: float = 10000.0,
                 num_heads: int = 8):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.method = method          # "rope" / "alibi" / "absolute"
        self.rope_theta = rope_theta
        self.num_heads = num_heads
        self.head_dim = dim // num_heads if num_heads > 0 else dim

        # RoPE预计算表
        self._rope_cos: Optional[List[List[float]]] = None
        self._rope_sin: Optional[List[List[float]]] = None
        # ALiBi斜率
        self._alibi_slopes: Optional[List[float]] = None
        # 绝对位置编码缓存
        self._abs_cache: Optional[List[List[float]]] = None

        if method == "rope":
            self.precompute_freqs()
        elif method == "alibi":
            self._alibi_slopes = self._compute_alibi_slopes(num_heads)

    # ---------- RoPE ----------

    def precompute_freqs(self) -> None:
        """预计算RoPE频率表 (cos/sin, shape: max_seq_len × head_dim//2)"""
        head_dim = self.head_dim
        half = max(head_dim // 2, 1)
        freqs = [1.0 / (self.rope_theta ** (2.0 * i / head_dim))
                 for i in range(half)]
        self._rope_cos = []
        self._rope_sin = []
        for pos in range(self.max_seq_len):
            self._rope_cos.append([math.cos(pos * f) for f in freqs])
            self._rope_sin.append([math.sin(pos * f) for f in freqs])

    def apply_rotary_emb(self, x: List[List[float]],
                         seq_offset: int = 0) -> List[List[float]]:
        """对 x (seq×head_dim) 应用旋转位置编码

        将每对相邻维度 (2i, 2i+1) 旋转角度 pos*freq_i
        """
        if self._rope_cos is None:
            self.precompute_freqs()
        seq_len = len(x)
        if seq_len == 0:
            return []
        head_dim = len(x[0])
        # 以预计算表的维度为准 (防止输入维度与head_dim不匹配时越界)
        precomputed_half = (len(self._rope_cos[0])
                            if self._rope_cos and self._rope_cos[0] else 0)
        half = min(head_dim // 2, precomputed_half)
        result = []
        for si in range(seq_len):
            pos = seq_offset + si
            if pos >= len(self._rope_cos):
                pos = len(self._rope_cos) - 1
            cos = self._rope_cos[pos]
            sin = self._rope_sin[pos]
            row = x[si]
            new_row = list(row)   # 先拷贝原始值, 未旋转维度保持不变
            for i in range(half):
                x1 = row[2 * i]
                x2 = row[2 * i + 1]
                new_row[2 * i] = x1 * cos[i] - x2 * sin[i]
                new_row[2 * i + 1] = x1 * sin[i] + x2 * cos[i]
            # 奇数维度时保留最后一个
            if head_dim % 2 == 1:
                new_row[head_dim - 1] = row[head_dim - 1]
            result.append(new_row)
        return result

    # ---------- ALiBi ----------

    @staticmethod
    def _compute_alibi_slopes(num_heads: int) -> List[float]:
        """计算ALiBi斜率 (几何序列)

        对2的幂的head数: slopes = 2^(-8/n) 的幂次序列
        非幂则补充额外斜率 (Press 2021 方法)
        """
        if num_heads <= 0:
            return []
        n = 2 ** int(math.floor(math.log2(num_heads)))
        base = 2.0 ** (-8.0 / n)
        slopes = [base ** (i + 1) for i in range(n)]
        if n < num_heads:
            extra_base = 2.0 ** (-8.0 / (2 * n))
            for i in range(num_heads - n):
                slopes.append(extra_base ** (2 * i + 1))
        return slopes[:num_heads]

    def get_alibi_bias(self, seq_len: int,
                       num_heads: Optional[int] = None) -> List[List[List[float]]]:
        """生成ALiBi偏置矩阵

        Returns:
            [num_heads × seq_len × seq_len] 的加性偏置
            bias[h][i][j] = -slope_h * (i - j)  (j <= i, 因果)
        """
        nh = num_heads or self.num_heads
        if self._alibi_slopes is None or len(self._alibi_slopes) < nh:
            slopes = self._compute_alibi_slopes(nh)
        else:
            slopes = self._alibi_slopes
        bias = []
        for h in range(nh):
            slope = slopes[h % len(slopes)]
            head_bias = []
            for i in range(seq_len):
                row = [0.0] * seq_len
                for j in range(i):
                    row[j] = -slope * (i - j)
                head_bias.append(row)
            bias.append(head_bias)
        return bias

    def get_alibi_slope(self, head_idx: int) -> float:
        """获取单个head的ALiBi斜率"""
        if not self._alibi_slopes:
            return 0.0
        return self._alibi_slopes[head_idx % len(self._alibi_slopes)]

    # ---------- 绝对位置编码 ----------

    def get_absolute(self, seq_len: int) -> List[List[float]]:
        """正弦/余弦绝对位置编码 (seq_len × dim)"""
        if self._abs_cache is not None and len(self._abs_cache) >= seq_len:
            return self._abs_cache[:seq_len]
        pe = []
        for pos in range(seq_len):
            row = []
            for i in range(self.dim):
                angle = pos / (10000.0 ** (2.0 * (i // 2) / self.dim))
                if i % 2 == 0:
                    row.append(math.sin(angle))
                else:
                    row.append(math.cos(angle))
            pe.append(row)
        self._abs_cache = pe
        return pe

    def get_stats(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "dim": self.dim,
            "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "num_heads": self.num_heads,
            "rope_precomputed": self._rope_cos is not None,
            "alibi_slopes": self._alibi_slopes,
            "abs_cached": self._abs_cache is not None,
        }


# ============================================================
# #7 KVCache [KV缓存管理器]  (提前定义, 供注意力层使用)
# ============================================================

class KVCache:
    """KV缓存管理器

    - 每层独立缓存K和V (per-layer, per-head)
    - append(key, value): 追加到缓存
    - get(): 获取当前缓存
    - truncate(length): 截断
    - 支持多batch
    """

    def __init__(self, num_layers: int, num_kv_heads: int = 1,
                 head_dim: int = 1, max_batch: int = 1):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_batch = max_batch
        # cache[batch][layer][head] = {"k": [[head_dim],...], "v": [[head_dim],...]}
        self.cache: List[List[List[Dict[str, List[List[float]]]]]] = [
            [[{"k": [], "v": []} for _ in range(num_kv_heads)]
             for _ in range(num_layers)]
            for _ in range(max_batch)
        ]
        self._append_count = 0

    def append(self, layer_idx: int, head_idx: int,
               key: List[List[float]], value: List[List[float]],
               batch_idx: int = 0) -> None:
        """追加K/V到指定层和head的缓存"""
        c = self.cache[batch_idx][layer_idx][head_idx]
        c["k"].extend(key)
        c["v"].extend(value)
        self._append_count += 1

    def append_layer(self, layer_idx: int,
                     keys: List[List[List[float]]],
                     values: List[List[List[float]]],
                     batch_idx: int = 0) -> None:
        """追加整层的K/V (keys/values: [num_kv_heads × seq × head_dim])"""
        for h in range(min(len(keys), self.num_kv_heads)):
            self.append(layer_idx, h, keys[h], values[h], batch_idx)

    def get(self, layer_idx: int, head_idx: int,
            batch_idx: int = 0) -> Tuple[List[List[float]], List[List[float]]]:
        """获取指定层和head的缓存K/V"""
        c = self.cache[batch_idx][layer_idx][head_idx]
        return c["k"], c["v"]

    def get_layer(self, layer_idx: int,
                  batch_idx: int = 0) -> Tuple[List[List[List[float]]],
                                               List[List[List[float]]]]:
        """获取整层的K/V (返回 [num_kv_heads × seq × head_dim])"""
        ks, vs = [], []
        for h in range(self.num_kv_heads):
            k, v = self.get(layer_idx, h, batch_idx)
            ks.append(k)
            vs.append(v)
        return ks, vs

    def truncate(self, length: int, batch_idx: Optional[int] = None) -> None:
        """截断缓存到指定长度"""
        batches = range(self.max_batch) if batch_idx is None else [batch_idx]
        for b in batches:
            for l in range(self.num_layers):
                for h in range(self.num_kv_heads):
                    c = self.cache[b][l][h]
                    c["k"] = c["k"][:length]
                    c["v"] = c["v"][:length]

    def reset(self) -> None:
        """清空所有缓存"""
        self.cache = [
            [[{"k": [], "v": []} for _ in range(self.num_kv_heads)]
             for _ in range(self.num_layers)]
            for _ in range(self.max_batch)
        ]
        self._append_count = 0

    def get_seq_len(self, batch_idx: int = 0) -> int:
        """获取当前缓存序列长度"""
        if not self.cache[batch_idx] or not self.cache[batch_idx][0]:
            return 0
        return len(self.cache[batch_idx][0][0]["k"])

    def get_stats(self) -> Dict[str, Any]:
        total_elements = 0
        for b in range(self.max_batch):
            for l in range(self.num_layers):
                for h in range(self.num_kv_heads):
                    c = self.cache[b][l][h]
                    total_elements += (len(c["k"]) + len(c["v"])) * self.head_dim
        # FP32: 4 bytes per element
        memory_mb = total_elements * 4 / (1024 * 1024)
        return {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "max_batch": self.max_batch,
            "seq_len": self.get_seq_len(),
            "append_count": self._append_count,
            "total_elements": total_elements,
            "memory_mb": round(memory_mb, 4),
        }


# ============================================================
# #3 MultiHeadAttention [多头注意力]
# ============================================================

class MultiHeadAttention:
    """多头注意力机制

    功能:
    - 标准Self-Attention: Q/K/V投影, scaled_dot_product
    - KV Cache支持: init_cache, update_cache, attention_with_cache
    - GQA (分组查询注意力) 支持: num_kv_heads < num_heads
    - MQA (多查询注意力) 支持: num_kv_heads == 1
    - Causal mask (因果掩码)
    - Sliding Window Attention
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 num_kv_heads: Optional[int] = None,
                 positional_encoding: Optional[PositionalEncoding] = None,
                 sliding_window: int = 0,
                 layer_idx: int = 0,
                 use_bias: bool = False):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_dim // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.sliding_window = sliding_window
        self.layer_idx = layer_idx
        self.use_bias = use_bias
        self.positional_encoding = positional_encoding

        # 是否GQA/MQA
        self.is_gqa = self.num_kv_heads < self.num_heads
        self.is_mqa = self.num_kv_heads == 1
        self.n_rep = self.num_heads // self.num_kv_heads if self.num_kv_heads > 0 else 1

        # ALiBi斜率
        self._alibi_slopes: Optional[List[float]] = None
        if positional_encoding is not None and positional_encoding.method == "alibi":
            self._alibi_slopes = positional_encoding._alibi_slopes

        # 权重初始化 (Glorot)
        self.W_q = _glorot_uniform(hidden_dim, hidden_dim)
        self.W_k = _glorot_uniform(hidden_dim, self.kv_dim)
        self.W_v = _glorot_uniform(hidden_dim, self.kv_dim)
        self.W_o = _glorot_uniform(hidden_dim, hidden_dim)
        self.b_q = _zeros_1d(hidden_dim) if use_bias else None
        self.b_k = _zeros_1d(self.kv_dim) if use_bias else None
        self.b_v = _zeros_1d(self.kv_dim) if use_bias else None
        self.b_o = _zeros_1d(hidden_dim) if use_bias else None

        # 统计
        self._forward_count = 0

    @property
    def num_params(self) -> int:
        base = (self.hidden_dim * self.hidden_dim      # W_q
                + self.hidden_dim * self.kv_dim * 2    # W_k, W_v
                + self.hidden_dim * self.hidden_dim)   # W_o
        if self.use_bias:
            base += self.hidden_dim * 2 + self.kv_dim * 2
        return base

    # ---------- 核心注意力 ----------

    def scaled_dot_product_attention(self, Q: List[List[float]],
                                     K: List[List[float]],
                                     V: List[List[float]],
                                     mask: Optional[List[List[bool]]] = None,
                                     alibi_slope: float = 0.0
                                     ) -> List[List[float]]:
        """缩放点积注意力

        Args:
            Q: (seq_q × head_dim)
            K: (seq_k × head_dim)
            V: (seq_k × head_dim)
            mask: (seq_q × seq_k), True=允许注意
            alibi_slope: ALiBi斜率

        Returns:
            (seq_q × head_dim)
        """
        seq_q = len(Q)
        seq_k = len(K)
        if seq_q == 0 or seq_k == 0:
            return []
        head_dim = len(Q[0]) if Q else self.head_dim
        scale = 1.0 / math.sqrt(head_dim)

        # 计算注意力分数
        scores: List[List[float]] = []
        for qi in range(seq_q):
            q = Q[qi]
            row = [0.0] * seq_k
            for ki in range(seq_k):
                if mask is not None and not mask[qi][ki]:
                    row[ki] = -1e9
                else:
                    s = 0.0
                    k = K[ki]
                    for d in range(head_dim):
                        s += q[d] * k[d]
                    row[ki] = s * scale + alibi_slope * (qi - ki) if alibi_slope else s * scale
                    if mask is not None and not mask[qi][ki]:
                        row[ki] = -1e9
            scores.append(row)

        # Softmax + 加权求和
        attn = [_softmax_vec(row) for row in scores]
        out = []
        for qi in range(seq_q):
            o = [0.0] * head_dim
            for ki in range(seq_k):
                w = attn[qi][ki]
                if w == 0.0:
                    continue
                v = V[ki]
                for d in range(head_dim):
                    o[d] += w * v[d]
            out.append(o)
        return out

    def _build_causal_mask(self, new_len: int, total_len: int,
                           seq_offset: int) -> List[List[bool]]:
        """构建因果掩码 + 滑动窗口

        mask[qi][ki] = True 当且仅当 ki <= seq_offset + qi
                       且 (滑动窗口关闭 或 ki >= seq_offset + qi - window)
        """
        mask = []
        for qi in range(new_len):
            abs_pos = seq_offset + qi
            row = []
            for ki in range(total_len):
                allowed = ki <= abs_pos
                if self.sliding_window > 0:
                    allowed = allowed and (ki >= abs_pos - self.sliding_window)
                row.append(allowed)
            mask.append(row)
        return mask

    # ---------- 前向传播 ----------

    def forward(self, x: List[List[float]], use_cache: bool = False,
                cache: Optional[KVCache] = None, seq_offset: int = 0,
                batch_idx: int = 0) -> List[List[float]]:
        """前向传播

        Args:
            x: (seq_len × hidden_dim)
            use_cache: 是否使用KV缓存
            cache: KVCache实例
            seq_offset: 当前序列在完整序列中的偏移 (用于RoPE位置和因果掩码)
            batch_idx: batch索引

        Returns:
            (seq_len × hidden_dim)
        """
        seq_len = len(x)
        if seq_len == 0:
            return []

        # 1. Q/K/V投影
        Q = _linear_2d(x, self.W_q, self.b_q)   # (seq × hidden)
        K = _linear_2d(x, self.W_k, self.b_k)   # (seq × kv_dim)
        V = _linear_2d(x, self.W_v, self.b_v)   # (seq × kv_dim)

        # 2. 分头
        Q_heads = _split_heads_2d(Q, self.num_heads)
        K_heads = _split_heads_2d(K, self.num_kv_heads)
        V_heads = _split_heads_2d(V, self.num_kv_heads)

        # 3. 应用RoPE (对Q和K)
        if self.positional_encoding is not None and \
                self.positional_encoding.method == "rope":
            Q_heads = [self.positional_encoding.apply_rotary_emb(h, seq_offset)
                       for h in Q_heads]
            K_heads = [self.positional_encoding.apply_rotary_emb(h, seq_offset)
                       for h in K_heads]

        # 4. KV缓存管理
        if use_cache and cache is not None:
            # 追加新K/V到缓存, 然后取完整K/V
            cache.append_layer(self.layer_idx, K_heads, V_heads, batch_idx)
            K_heads, V_heads = cache.get_layer(self.layer_idx, batch_idx)

        total_len = len(K_heads[0]) if K_heads else 0

        # 5. GQA/MQA: 重复KV头
        if self.n_rep > 1:
            K_heads = _repeat_kv_heads(K_heads, self.n_rep)
            V_heads = _repeat_kv_heads(V_heads, self.n_rep)

        # 6. 逐头注意力
        mask = self._build_causal_mask(seq_len, total_len, seq_offset)
        out_heads = []
        for h in range(self.num_heads):
            slope = 0.0
            if self._alibi_slopes is not None:
                slope = self.positional_encoding.get_alibi_slope(h) \
                    if self.positional_encoding else 0.0
            out_h = self.scaled_dot_product_attention(
                Q_heads[h], K_heads[h], V_heads[h], mask, slope)
            out_heads.append(out_h)

        # 7. 合并头
        merged = _merge_heads_2d(out_heads)   # (seq × hidden)

        # 8. 输出投影
        output = _linear_2d(merged, self.W_o, self.b_o)
        self._forward_count += 1
        return output

    def __call__(self, x: List[List[float]], **kwargs) -> List[List[float]]:
        return self.forward(x, **kwargs)

    # ---------- 缓存接口 ----------

    def init_cache(self) -> Dict[str, Any]:
        """初始化缓存信息 (实际缓存由KVCache管理)"""
        return {
            "layer_idx": self.layer_idx,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "kv_dim": self.kv_dim,
        }

    def update_cache(self, cache: KVCache, k_new: List[List[List[float]]],
                     v_new: List[List[List[float]]],
                     batch_idx: int = 0) -> None:
        """更新缓存 (追加新K/V)"""
        cache.append_layer(self.layer_idx, k_new, v_new, batch_idx)

    def attention_with_cache(self, q_heads: List[List[List[float]]],
                             cache: KVCache,
                             seq_offset: int = 0,
                             batch_idx: int = 0) -> List[List[float]]:
        """使用缓存进行注意力计算 (用于增量解码)

        Args:
            q_heads: [num_heads × new_len × head_dim] (已分头, 已RoPE)
            cache: KVCache实例
            seq_offset: 序列偏移

        Returns:
            (new_len × hidden_dim)
        """
        K_heads, V_heads = cache.get_layer(self.layer_idx, batch_idx)
        total_len = len(K_heads[0]) if K_heads else 0
        new_len = len(q_heads[0]) if q_heads else 0

        if self.n_rep > 1:
            K_heads = _repeat_kv_heads(K_heads, self.n_rep)
            V_heads = _repeat_kv_heads(V_heads, self.n_rep)

        mask = self._build_causal_mask(new_len, total_len, seq_offset)
        out_heads = []
        for h in range(self.num_heads):
            slope = 0.0
            if self._alibi_slopes is not None and self.positional_encoding:
                slope = self.positional_encoding.get_alibi_slope(h)
            out_h = self.scaled_dot_product_attention(
                q_heads[h], K_heads[h], V_heads[h], mask, slope)
            out_heads.append(out_h)
        return _merge_heads_2d(out_heads)

    def get_attention_type(self) -> str:
        if self.is_mqa:
            return "MQA"
        elif self.is_gqa:
            return "GQA"
        return "MHA"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "attention_type": self.get_attention_type(),
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "kv_dim": self.kv_dim,
            "n_rep": self.n_rep,
            "sliding_window": self.sliding_window,
            "num_params": self.num_params,
            "forward_count": self._forward_count,
            "use_bias": self.use_bias,
        }


# ============================================================
# #4 TransformerLayer [Transformer层]
# ============================================================

class RMSNorm:
    """RMSNorm (Root Mean Square Normalization)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.weight = [1.0] * dim   # 可学习缩放参数

    @property
    def num_params(self) -> int:
        return self.dim

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        return _rms_norm_rows(x, self.weight, self.eps)

    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        return self.forward(x)


class SwiGLUFFN:
    """SwiGLU前馈网络

    FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down
    """

    def __init__(self, hidden_dim: int, ffn_dim: int):
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.W_gate = _glorot_uniform(hidden_dim, ffn_dim)
        self.W_up = _glorot_uniform(hidden_dim, ffn_dim)
        self.W_down = _glorot_uniform(ffn_dim, hidden_dim)

    @property
    def num_params(self) -> int:
        return (self.hidden_dim * self.ffn_dim * 2   # W_gate, W_up
                + self.ffn_dim * self.hidden_dim)     # W_down

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """x: (seq × hidden) -> (seq × hidden)"""
        gate = _linear_2d(x, self.W_gate)   # (seq × ffn)
        up = _linear_2d(x, self.W_up)       # (seq × ffn)
        # SwiGLU: silu(gate) * up
        activated = [[_silu(gate[s][i]) * up[s][i]
                       for i in range(self.ffn_dim)]
                      for s in range(len(x))]
        return _linear_2d(activated, self.W_down)  # (seq × hidden)

    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        return self.forward(x)


class TransformerLayer:
    """Transformer层 (PreNorm架构)

    结构:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    - PreNorm架构: LayerNorm → Attention → 残差
    - FFN: SwiGLU激活
    - 残差连接
    - LayerNorm (RMSNorm变体)
    - Dropout
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 ffn_dim: int, num_kv_heads: Optional[int] = None,
                 positional_encoding: Optional[PositionalEncoding] = None,
                 sliding_window: int = 0, layer_idx: int = 0,
                 dropout: float = 0.0, norm_eps: float = 1e-6):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.layer_idx = layer_idx
        self.dropout_rate = dropout
        self.norm_eps = norm_eps

        self.norm1 = RMSNorm(hidden_dim, norm_eps)
        self.attn = MultiHeadAttention(
            hidden_dim, num_heads, num_kv_heads,
            positional_encoding, sliding_window, layer_idx)
        self.norm2 = RMSNorm(hidden_dim, norm_eps)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

        self._forward_count = 0

    def _dropout(self, x: List[List[float]]) -> List[List[float]]:
        """Dropout (训练时随机置零)"""
        if self.dropout_rate <= 0.0:
            return x
        keep = 1.0 - self.dropout_rate
        return [[v if random.random() < keep else 0.0
                 for v in row] for row in x]

    def forward(self, x: List[List[float]], use_cache: bool = False,
                cache: Optional[KVCache] = None, seq_offset: int = 0,
                batch_idx: int = 0, training: bool = False) -> List[List[float]]:
        """前向传播 (PreNorm)"""
        # 注意力子层
        h = self.norm1(x)
        attn_out = self.attn(h, use_cache=use_cache, cache=cache,
                             seq_offset=seq_offset, batch_idx=batch_idx)
        if training:
            attn_out = self._dropout(attn_out)
        x = [[x[s][d] + attn_out[s][d] for d in range(self.hidden_dim)]
             for s in range(len(x))]

        # FFN子层
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        if training:
            ffn_out = self._dropout(ffn_out)
        x = [[x[s][d] + ffn_out[s][d] for d in range(self.hidden_dim)]
             for s in range(len(x))]

        self._forward_count += 1
        return x

    def __call__(self, x: List[List[float]], **kwargs) -> List[List[float]]:
        return self.forward(x, **kwargs)

    @property
    def num_params(self) -> int:
        return (self.norm1.num_params + self.attn.num_params
                + self.norm2.num_params + self.ffn.num_params)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "num_params": self.num_params,
            "dropout_rate": self.dropout_rate,
            "forward_count": self._forward_count,
            "attention": self.attn.get_stats(),
        }


# ============================================================
# #5 LingyuanTransformerModel [完整Transformer模型]
# ============================================================

class LingyuanTransformerModel:
    """灵元完整Transformer模型

    结构:
        token_embedding + 位置编码
        → N × TransformerLayer
        → RMSNorm
        → LM Head (输出投影)

    支持功能:
    - forward(input_ids, use_cache=False) → logits
    - forward_with_cache(input_ids, cache) → logits, new_cache
    - 预设配置: tiny/small/base/large
    - 参数量统计
    """

    PRESETS = _MODEL_PRESETS

    def __init__(self, config: Any = None, **kwargs):
        """初始化模型

        Args:
            config: ModelConfig实例或dict (可选)
            **kwargs: 直接指定配置参数 (覆盖config)
        """
        # 从config或kwargs提取配置
        defaults = {
            "hidden_dim": 128, "num_layers": 4, "num_heads": 4,
            "num_kv_heads": 4, "ffn_dim": 256, "max_seq_len": 512,
            "vocab_size": 2048, "rope_theta": 10000.0, "norm_eps": 1e-6,
            "dropout": 0.0, "sliding_window": 0,
            "tie_word_embeddings": True, "pos_method": "rope",
        }
        cfg = {}
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
        self.config_dict = cfg

        self.head_dim = self.hidden_dim // self.num_heads

        # 位置编码
        self.positional_encoding = PositionalEncoding(
            dim=self.hidden_dim, max_seq_len=self.max_seq_len,
            method=self.pos_method, rope_theta=self.rope_theta,
            num_heads=self.num_heads)

        # Token Embedding (vocab_size × hidden_dim)
        self.token_embedding = _glorot_uniform(self.vocab_size, self.hidden_dim)

        # Transformer层堆叠
        self.layers: List[TransformerLayer] = []
        for i in range(self.num_layers):
            layer = TransformerLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                num_kv_heads=self.num_kv_heads,
                positional_encoding=self.positional_encoding,
                sliding_window=self.sliding_window,
                layer_idx=i,
                dropout=self.dropout,
                norm_eps=self.norm_eps,
            )
            self.layers.append(layer)

        # 最终RMSNorm
        self.final_norm = RMSNorm(self.hidden_dim, self.norm_eps)

        # LM Head (权重绑定可选)
        self.lm_head = (None if self.tie_word_embeddings
                        else _glorot_uniform(self.hidden_dim, self.vocab_size))

        self._forward_count = 0
        self._created_at = datetime.now().isoformat()

    # ---------- 嵌入 ----------

    def embed(self, input_ids: List[int]) -> List[List[float]]:
        """token id列表 -> embedding (seq × hidden)"""
        result = []
        for tid in input_ids:
            if 0 <= tid < self.vocab_size:
                result.append(list(self.token_embedding[tid]))
            else:
                result.append([0.0] * self.hidden_dim)
        return result

    # ---------- LM Head ----------

    def compute_logits(self, hidden: List[List[float]],
                       last_token_only: bool = False) -> List[List[float]]:
        """hidden -> logits (seq × vocab 或 1 × vocab)"""
        h = self.final_norm(hidden)
        if last_token_only:
            h = h[-1:] if h else []
        if self.tie_word_embeddings:
            # logits = h @ embedding^T  (embedding: vocab×hidden)
            emb_t = _transpose_2d(self.token_embedding)  # (hidden × vocab)
            return _matmul_2d(h, emb_t)
        else:
            return _matmul_2d(h, self.lm_head)

    # ---------- 前向传播 ----------

    def forward(self, input_ids: List[int], use_cache: bool = False,
                cache: Optional[KVCache] = None,
                last_token_only: bool = False,
                training: bool = False) -> List[List[float]]:
        """前向传播

        Args:
            input_ids: token id列表
            use_cache: 是否使用KV缓存
            cache: KVCache实例
            last_token_only: 仅计算最后位置的logits
            training: 训练模式

        Returns:
            logits (seq × vocab 或 1 × vocab)
        """
        seq_offset = 0
        if use_cache and cache is not None:
            seq_offset = cache.get_seq_len()

        # Embedding
        x = self.embed(input_ids)

        # 绝对位置编码 (叠加)
        if self.pos_method == "absolute":
            abs_pe = self.positional_encoding.get_absolute(len(input_ids))
            x = [[x[s][d] + abs_pe[s][d] for d in range(self.hidden_dim)]
                 for s in range(len(input_ids))]

        # Transformer层
        for layer in self.layers:
            x = layer(x, use_cache=use_cache, cache=cache,
                      seq_offset=seq_offset, batch_idx=0, training=training)

        # LM Head
        logits = self.compute_logits(x, last_token_only)
        self._forward_count += 1
        return logits

    def forward_with_cache(self, input_ids: List[int],
                           cache: Optional[KVCache] = None
                           ) -> Tuple[List[List[float]], KVCache]:
        """带KV缓存的前向传播 (用于增量解码)

        首次调用传入完整prompt, 后续调用传入单个新token
        """
        if cache is None:
            cache = KVCache(
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                max_batch=1,
            )
        logits = self.forward(input_ids, use_cache=True, cache=cache,
                              last_token_only=True)
        return logits, cache

    # ---------- 训练专用前向传播 (带完整梯度缓存) ----------

    def forward_for_training(self, input_ids: List[int],
                             targets: List[int]
                             ) -> Tuple[float, List[List[float]], Dict[str, Any]]:
        """训练用前向传播 — 保存所有中间激活值供真实反向传播使用

        与 forward() 逻辑一致, 但额外缓存:
        - 嵌入向量
        - 每层的: 输入、norm归一化值、Q/K/V投影、注意力权重、FFN中间值
        - 最终norm的归一化值
        - logits

        Returns:
            (loss, logits, cache)
        """
        seq_len = len(input_ids)
        hidden = self.hidden_dim
        cache: Dict[str, Any] = {"input_ids": list(input_ids)}

        # === 1. Token Embedding ===
        x = self.embed(input_ids)
        cache["embeddings"] = [list(row) for row in x]

        # 绝对位置编码 (叠加)
        if self.pos_method == "absolute":
            abs_pe = self.positional_encoding.get_absolute(seq_len)
            x = [[x[s][d] + abs_pe[s][d] for d in range(hidden)]
                 for s in range(seq_len)]

        # === 2. Transformer 层 ===
        layer_caches: List[Dict[str, Any]] = []
        rope_cos = self.positional_encoding._rope_cos if self.pos_method == "rope" else None
        rope_sin = self.positional_encoding._rope_sin if self.pos_method == "rope" else None

        for layer in self.layers:
            lc: Dict[str, Any] = {}
            attn = layer.attn
            num_heads = attn.num_heads
            num_kv_heads = attn.num_kv_heads
            head_dim = attn.head_dim
            n_rep = attn.n_rep
            use_rope = (attn.positional_encoding is not None
                        and attn.positional_encoding.method == "rope")

            # --- 2a. 注意力子层 (PreNorm) ---
            lc["layer_input"] = [list(row) for row in x]  # 残差输入

            # RMSNorm 1
            norm1_x_norm: List[List[float]] = []
            norm1_rms: List[float] = []
            h1: List[List[float]] = []
            for row in x:
                n = len(row)
                ms = sum(v * v for v in row) / n
                r = math.sqrt(ms + layer.norm_eps)
                xn = [v / r for v in row]
                norm1_x_norm.append(xn)
                norm1_rms.append(r)
                h1.append([xn[d] * layer.norm1.weight[d] for d in range(hidden)])
            lc["norm1_x_norm"] = norm1_x_norm
            lc["norm1_rms"] = norm1_rms

            # Q/K/V 投影
            Q = _linear_2d(h1, attn.W_q)   # (seq × hidden)
            K = _linear_2d(h1, attn.W_k)   # (seq × kv_dim)
            V = _linear_2d(h1, attn.W_v)   # (seq × kv_dim)

            # 分头
            Q_heads = _split_heads_2d(Q, num_heads)
            K_heads = _split_heads_2d(K, num_kv_heads)
            V_heads = _split_heads_2d(V, num_kv_heads)

            # RoPE
            if use_rope:
                Q_heads_rot = [attn.positional_encoding.apply_rotary_emb(h, 0) for h in Q_heads]
                K_heads_rot = [attn.positional_encoding.apply_rotary_emb(h, 0) for h in K_heads]
            else:
                Q_heads_rot = Q_heads
                K_heads_rot = K_heads

            # 因果掩码
            mask = attn._build_causal_mask(seq_len, seq_len, 0)
            scale = 1.0 / math.sqrt(head_dim)

            # 逐头注意力 (保存注意力权重)
            attn_weights_all: List[List[List[float]]] = []
            out_heads: List[List[List[float]]] = []

            for h_idx in range(num_heads):
                q_h = Q_heads_rot[h_idx]
                # GQA: 对应的KV头索引
                kv_idx = h_idx // n_rep if n_rep > 1 else h_idx
                k_h = K_heads_rot[kv_idx] if use_rope else K_heads[kv_idx]
                v_h = V_heads[kv_idx]

                # 计算注意力分数
                scores_h: List[List[float]] = []
                for qi in range(seq_len):
                    row_s = [0.0] * seq_len
                    qv = q_h[qi]
                    for ki in range(qi + 1):  # 因果: 只看 ki <= qi
                        s_val = 0.0
                        kv = k_h[ki]
                        for d in range(head_dim):
                            s_val += qv[d] * kv[d]
                        row_s[ki] = s_val * scale
                    scores_h.append(row_s)

                # Softmax (只对有效位置)
                attn_w_h = []
                for qi in range(seq_len):
                    valid_scores = scores_h[qi][:qi + 1]
                    probs = _softmax_vec(valid_scores)
                    full_probs = probs + [0.0] * (seq_len - qi - 1)
                    attn_w_h.append(full_probs)
                attn_weights_all.append(attn_w_h)

                # 加权求和
                out_h: List[List[float]] = []
                for qi in range(seq_len):
                    o = [0.0] * head_dim
                    for ki in range(qi + 1):
                        w = attn_w_h[qi][ki]
                        if w == 0.0:
                            continue
                        vk = v_h[ki]
                        for d in range(head_dim):
                            o[d] += w * vk[d]
                    out_h.append(o)
                out_heads.append(out_h)

            lc["attn_weights"] = attn_weights_all
            lc["Q_heads_rot"] = Q_heads_rot
            lc["K_heads_rot"] = K_heads_rot
            lc["V_heads"] = V_heads
            lc["attn_scale"] = scale

            # 合并头 + 输出投影
            merged = _merge_heads_2d(out_heads)
            lc["attn_merged"] = merged
            attn_out = _linear_2d(merged, attn.W_o)

            # 残差
            x = [[x[s][d] + attn_out[s][d] for d in range(hidden)]
                 for s in range(seq_len)]

            # --- 2b. FFN 子层 (PreNorm) ---
            lc["input_after_attn"] = [list(row) for row in x]

            # RMSNorm 2
            norm2_x_norm: List[List[float]] = []
            norm2_rms: List[float] = []
            h2: List[List[float]] = []
            for row in x:
                n = len(row)
                ms = sum(v * v for v in row) / n
                r = math.sqrt(ms + layer.norm_eps)
                xn = [v / r for v in row]
                norm2_x_norm.append(xn)
                norm2_rms.append(r)
                h2.append([xn[d] * layer.norm2.weight[d] for d in range(hidden)])
            lc["norm2_x_norm"] = norm2_x_norm
            lc["norm2_rms"] = norm2_rms

            # SwiGLU FFN
            gate = _linear_2d(h2, layer.ffn.W_gate)
            up = _linear_2d(h2, layer.ffn.W_up)
            ffn_dim = layer.ffn.ffn_dim
            activated = [[_silu(gate[s][i]) * up[s][i]
                           for i in range(ffn_dim)]
                          for s in range(seq_len)]
            ffn_out = _linear_2d(activated, layer.ffn.W_down)

            lc["ffn_gate"] = gate
            lc["ffn_up"] = up
            lc["ffn_activated"] = activated
            lc["ffn_out"] = ffn_out

            # 残差
            x = [[x[s][d] + ffn_out[s][d] for d in range(hidden)]
                 for s in range(seq_len)]

            lc["layer_output"] = [list(row) for row in x]
            layer_caches.append(lc)

        cache["layers"] = layer_caches

        # === 3. 最终 RMSNorm ===
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
        cache["final_norm_out"] = h_final

        # === 4. LM Head ===
        if self.tie_word_embeddings:
            emb_t = _transpose_2d(self.token_embedding)  # (hidden × vocab)
            logits = _matmul_2d(h_final, emb_t)          # (seq × vocab)
        else:
            logits = _matmul_2d(h_final, self.lm_head)    # (seq × vocab)
        cache["logits"] = logits

        # === 5. 损失 ===
        loss = _cross_entropy_loss(logits, targets)

        return loss, logits, cache

    # ---------- 预设 ----------

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "LingyuanTransformerModel":
        """从预设名称创建模型"""
        if name not in cls.PRESETS:
            raise ValueError(f"未知预设: {name}, 可选: {list(cls.PRESETS.keys())}")
        cfg = dict(cls.PRESETS[name])
        cfg.update(overrides)
        return cls(cfg)

    @classmethod
    def list_presets(cls) -> Dict[str, Dict[str, Any]]:
        """列出所有预设配置"""
        return dict(cls.PRESETS)

    # ---------- 参数统计 ----------

    def count_parameters(self) -> int:
        """统计总参数量"""
        # Embedding
        total = self.vocab_size * self.hidden_dim
        # 各层
        for layer in self.layers:
            total += layer.num_params
        # 最终norm
        total += self.final_norm.num_params
        # LM head (未绑定时)
        if not self.tie_word_embeddings:
            total += self.hidden_dim * self.vocab_size
        return total

    def count_parameters_detail(self) -> Dict[str, int]:
        """详细参数统计"""
        embedding = self.vocab_size * self.hidden_dim
        layers_total = sum(l.num_params for l in self.layers)
        per_layer = self.layers[0].num_params if self.layers else 0
        final_norm = self.final_norm.num_params
        lm_head = 0 if self.tie_word_embeddings else self.hidden_dim * self.vocab_size
        return {
            "embedding": embedding,
            "per_layer": per_layer,
            "layers_total": layers_total,
            "final_norm": final_norm,
            "lm_head": lm_head,
            "total": embedding + layers_total + final_norm + lm_head,
            "tie_word_embeddings": self.tie_word_embeddings,
        }

    def estimate_memory(self, precision: str = "fp32") -> Dict[str, float]:
        """估计显存占用"""
        bytes_per = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(precision, 4)
        params = self.count_parameters()
        # 模型权重 + 梯度 + 优化器状态(Adam: 2x)
        model_mb = params * bytes_per / 1e6
        grad_mb = params * bytes_per / 1e6
        optim_mb = params * 2 * bytes_per / 1e6
        return {
            "params": params,
            "precision": precision,
            "bytes_per_param": bytes_per,
            "model_mb": round(model_mb, 2),
            "gradient_mb": round(grad_mb, 2),
            "optimizer_mb": round(optim_mb, 2),
            "total_mb": round(model_mb + grad_mb + optim_mb, 2),
        }

    # ---------- 权重存取 ----------

    @property
    def config(self) -> Dict[str, Any]:
        """返回模型配置字典 (供导出器使用)"""
        return dict(self.config_dict)

    def state_dict(self) -> Dict[str, Any]:
        """返回模型所有权重 (类似 PyTorch state_dict)"""
        sd: Dict[str, Any] = {
            "token_embedding": self.token_embedding,
            "final_norm.weight": self.final_norm.weight,
        }
        if not self.tie_word_embeddings and self.lm_head is not None:
            sd["lm_head"] = self.lm_head
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            sd[prefix + "attn.W_q"] = layer.attn.W_q
            sd[prefix + "attn.W_k"] = layer.attn.W_k
            sd[prefix + "attn.W_v"] = layer.attn.W_v
            sd[prefix + "attn.W_o"] = layer.attn.W_o
            sd[prefix + "norm1.weight"] = layer.norm1.weight
            sd[prefix + "norm2.weight"] = layer.norm2.weight
            sd[prefix + "ffn.W_gate"] = layer.ffn.W_gate
            sd[prefix + "ffn.W_up"] = layer.ffn.W_up
            sd[prefix + "ffn.W_down"] = layer.ffn.W_down
        return sd

    def get_weights_summary(self) -> Dict[str, Any]:
        """获取权重摘要 (用于检查点)"""
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "num_params": self.count_parameters(),
            "tie_word_embeddings": self.tie_word_embeddings,
            "created_at": self._created_at,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_name": "LingyuanTransformerModel",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "ffn_dim": self.ffn_dim,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "rope_theta": self.rope_theta,
            "pos_method": self.pos_method,
            "tie_word_embeddings": self.tie_word_embeddings,
            "num_params": self.count_parameters(),
            "forward_count": self._forward_count,
            "attention_type": self.layers[0].attn.get_attention_type() if self.layers else "N/A",
        }

    def get_dashboard(self) -> Dict[str, Any]:
        detail = self.count_parameters_detail()
        return {
            "model": "LingyuanTransformerModel",
            "created_at": self._created_at,
            "config": self.config_dict,
            "params": detail,
            "memory_fp32": self.estimate_memory("fp32"),
            "presets_available": list(self.PRESETS.keys()),
            "forward_count": self._forward_count,
            "layers": [l.get_stats() for l in self.layers],
        }


# ============================================================
# #6 Sampler [采样器]
# ============================================================

class Sampler:
    """采样器

    功能:
    - greedy (最大概率)
    - temperature缩放
    - top-k过滤
    - top-p (nucleus) 过滤
    - 组合采样: temperature + top_k + top_p
    - repetition_penalty (重复惩罚)
    - generate(model, input_ids, max_new_tokens, **kwargs) → List[int]
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self._sample_count = 0
        self._history: List[Dict[str, Any]] = []

    # ---------- 基础采样策略 ----------

    @staticmethod
    def greedy(logits: List[float]) -> int:
        """贪婪采样: 返回最大概率的token"""
        best_id = 0
        best_val = logits[0] if logits else 0.0
        for i, v in enumerate(logits):
            if v > best_val:
                best_val = v
                best_id = i
        return best_id

    @staticmethod
    def apply_temperature(logits: List[float], temperature: float) -> List[float]:
        """温度缩放: logits / temperature"""
        if temperature <= 0:
            temperature = 1e-6
        return [v / temperature for v in logits]

    @staticmethod
    def apply_top_k(logits: List[float], top_k: int) -> List[float]:
        """Top-K过滤: 保留概率最高的K个, 其余置-inf"""
        if top_k <= 0 or top_k >= len(logits):
            return logits
        # 获取top_k个最大值的索引
        indexed = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
        keep = set(indexed[:top_k])
        return [v if i in keep else -1e9 for i, v in enumerate(logits)]

    @staticmethod
    def apply_top_p(logits: List[float], top_p: float) -> List[float]:
        """Top-P (nucleus) 过滤: 保留累积概率达到p的最小token集"""
        if top_p >= 1.0 or top_p <= 0.0:
            return logits
        probs = _softmax_vec(logits)
        # 按概率降序排列
        indexed = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        cumsum = 0.0
        keep = set()
        for idx in indexed:
            keep.add(idx)
            cumsum += probs[idx]
            if cumsum >= top_p:
                break
        return [v if i in keep else -1e9 for i, v in enumerate(logits)]

    @staticmethod
    def apply_repetition_penalty(logits: List[float],
                                 prev_tokens: List[int],
                                 penalty: float) -> List[float]:
        """重复惩罚: 对已出现token降低概率

        penalty > 1: 降低重复; penalty < 1: 增加重复
        """
        if penalty == 1.0 or not prev_tokens:
            return logits
        seen = set(prev_tokens)
        result = list(logits)
        for tid in seen:
            if 0 <= tid < len(result):
                if result[tid] > 0:
                    result[tid] = result[tid] / penalty
                else:
                    result[tid] = result[tid] * penalty
        return result

    # ---------- 组合采样 ----------

    def sample(self, logits: List[float],
               temperature: float = 1.0,
               top_k: int = 0,
               top_p: float = 1.0,
               repetition_penalty: float = 1.0,
               prev_tokens: Optional[List[int]] = None) -> int:
        """组合采样: temperature + top_k + top_p + repetition_penalty

        Args:
            logits: 原始logits
            temperature: 温度 (1.0=不变, <1更确定, >1更随机)
            top_k: Top-K过滤 (0=禁用)
            top_p: Top-P过滤 (1.0=禁用)
            repetition_penalty: 重复惩罚 (1.0=禁用)
            prev_tokens: 已生成的token列表 (用于重复惩罚)

        Returns:
            采样的token id
        """
        if temperature <= 0:
            # 退化为贪婪
            return self.greedy(logits)

        # 1. 重复惩罚
        if prev_tokens and repetition_penalty != 1.0:
            logits = self.apply_repetition_penalty(logits, prev_tokens, repetition_penalty)

        # 2. 温度缩放
        logits = self.apply_temperature(logits, temperature)

        # 3. Top-K
        if top_k > 0:
            logits = self.apply_top_k(logits, top_k)

        # 4. Top-P
        if top_p < 1.0:
            logits = self.apply_top_p(logits, top_p)

        # 5. 转概率并采样
        probs = _softmax_vec(logits)
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                self._sample_count += 1
                return i
        # 兜底: 返回最后一个
        return len(probs) - 1

    # ---------- 生成 ----------

    def generate(self, model: LingyuanTransformerModel,
                 input_ids: List[int],
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 use_cache: bool = True,
                 eos_token_id: Optional[int] = None,
                 verbose: bool = False) -> List[int]:
        """自回归生成

        Args:
            model: LingyuanTransformerModel实例
            input_ids: 输入prompt的token id列表
            max_new_tokens: 最大生成token数
            temperature/top_k/top_p/repetition_penalty: 采样参数
            use_cache: 是否使用KV缓存
            eos_token_id: 结束token id (生成到此停止)
            verbose: 是否打印进度

        Returns:
            生成的token id列表 (仅新生成的部分)
        """
        generated = list(input_ids)
        new_tokens: List[int] = []
        cache: Optional[KVCache] = None

        start_time = time.time()

        # 首次前向 (处理完整prompt)
        if use_cache:
            logits, cache = model.forward_with_cache(generated, None)
        else:
            logits = model.forward(generated, last_token_only=True)

        for step in range(max_new_tokens):
            # 取最后位置的logits
            last_logits = logits[-1] if logits else [0.0]

            # 采样
            next_id = self.sample(
                last_logits, temperature=temperature, top_k=top_k,
                top_p=top_p, repetition_penalty=repetition_penalty,
                prev_tokens=generated)
            new_tokens.append(next_id)
            generated.append(next_id)

            if verbose:
                print(f"  step {step+1}: token={next_id}")

            if eos_token_id is not None and next_id == eos_token_id:
                break

            # 下一轮前向
            if step < max_new_tokens - 1:
                if use_cache and cache is not None:
                    logits, cache = model.forward_with_cache([next_id], cache)
                else:
                    logits = model.forward(generated, last_token_only=True)

        elapsed = time.time() - start_time
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt_len": len(input_ids),
            "generated_len": len(new_tokens),
            "use_cache": use_cache,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "elapsed_sec": round(elapsed, 3),
            "tokens_per_sec": round(len(new_tokens) / max(elapsed, 1e-6), 2),
        })

        return new_tokens

    def get_stats(self) -> Dict[str, Any]:
        total_generated = sum(h["generated_len"] for h in self._history)
        avg_tps = (sum(h["tokens_per_sec"] for h in self._history) / len(self._history)
                   if self._history else 0.0)
        return {
            "sample_count": self._sample_count,
            "generate_count": len(self._history),
            "total_generated": total_generated,
            "avg_tokens_per_sec": round(avg_tps, 2),
            "seed": self.seed,
            "history": self._history[-5:],   # 最近5次
        }


# ============================================================
# #8 TrainingEngine [真实训练引擎]
# ============================================================

@dataclass
class LRSchedule:
    """学习率调度配置"""
    schedule_type: str = "cosine"       # cosine / linear / constant / warmup_cosine
    base_lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 100
    total_steps: int = 10000


class SimpleOptimizer:
    """简化优化器 (Adam变体)"""

    def __init__(self, params: Dict[str, List[Any]], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        # 一阶/二阶动量
        self.m: Dict[str, List[Any]] = {}
        self.v: Dict[str, List[Any]] = {}
        self._init_moments(params)

    def _init_moments(self, params: Dict[str, List[Any]]):
        for name, p in params.items():
            self.m[name] = self._zeros_like(p)
            self.v[name] = self._zeros_like(p)

    @staticmethod
    def _zeros_like(x: Any) -> Any:
        if isinstance(x, list):
            return [SimpleOptimizer._zeros_like(e) for e in x]
        return 0.0

    def step(self, params: Dict[str, List[Any]],
             grads: Dict[str, List[Any]]) -> None:
        """执行一步参数更新"""
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias_c1 = 1.0 - b1 ** self.t
        bias_c2 = 1.0 - b2 ** self.t
        for name in params:
            if name not in grads:
                continue
            self._update_param(params[name], grads[name],
                               self.m[name], self.v[name],
                               b1, b2, bias_c1, bias_c2)

    def _update_param(self, param: Any, grad: Any,
                      m: Any, v: Any,
                      b1: float, b2: float,
                      bc1: float, bc2: float) -> None:
        if isinstance(param, list) and len(param) > 0 and isinstance(param[0], list):
            # 2D
            for i in range(len(param)):
                self._update_param(param[i], grad[i], m[i], v[i], b1, b2, bc1, bc2)
        elif isinstance(param, list):
            # 1D
            for i in range(len(param)):
                g = grad[i] + self.weight_decay * param[i]
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                m_hat = m[i] / bc1
                v_hat = v[i] / bc2
                param[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        else:
            g = grad + self.weight_decay * param
            m[0] = b1 * m[0] + (1 - b1) * g
            v[0] = b2 * v[0] + (1 - b2) * g * g
            param -= self.lr * (m[0] / bc1) / (math.sqrt(v[0] / bc2) + self.eps)


class MixedPrecisionSimulator:
    """混合精度模拟 (FP16/BF16 数值范围管理)

    模拟混合精度训练中的:
    - 数值范围限制 (FP16: ±65504)
    - 精度降低 (减少有效位数)
    - 动态损失缩放 (Loss Scaling)
    - 溢出检测
    """

    FP16_MAX = 65504.0
    FP16_MIN = -65504.0

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.overflow_count = 0
        self.underflow_count = 0
        self.scale = 1.0           # 损失缩放因子
        self.scale_growth = 2.0
        self.scale_backoff = 0.5
        self.growth_interval = 2000
        self._steps_since_growth = 0

    def cast_value(self, value: float) -> float:
        """将单个数值转换为目标精度"""
        if self.precision == "fp32":
            return value
        if self.precision == "fp16":
            if abs(value) > self.FP16_MAX:
                self.overflow_count += 1
                return math.copysign(self.FP16_MAX, value)
            if 0 < abs(value) < 6e-8:   # FP16最小正规数
                self.underflow_count += 1
                return 0.0
            # 模拟10位尾数 (FP16有10位)
            return round(value * 1024) / 1024
        if self.precision == "bf16":
            # BF16: 与FP32同范围, 7位尾数
            if abs(value) > 3.4e38:
                self.overflow_count += 1
                return math.copysign(3.4e38, value)
            return round(value * 256) / 256
        return value

    def cast_matrix(self, m: List[List[float]]) -> List[List[float]]:
        """转换整个矩阵"""
        return [[self.cast_value(v) for v in row] for row in m]

    def scale_loss(self, loss: float) -> float:
        """损失缩放"""
        return loss * self.scale

    def unscale_grads(self, grads: Dict[str, Any]) -> Dict[str, Any]:
        """反向缩放梯度"""
        inv = 1.0 / self.scale
        return self._scale_grads(grads, inv)

    def _scale_grads(self, grads: Any, factor: float) -> Any:
        if isinstance(grads, dict):
            return {k: self._scale_grads(v, factor) for k, v in grads.items()}
        if isinstance(grads, list) and len(grads) > 0 and isinstance(grads[0], list):
            return [[v * factor for v in row] for row in grads]
        if isinstance(grads, list):
            return [v * factor for v in grads]
        return grads * factor

    def check_overflow(self, grads: Dict[str, Any]) -> bool:
        """检查梯度是否溢出"""
        return self._check_overflow_recursive(grads)

    def _check_overflow_recursive(self, x: Any) -> bool:
        if isinstance(x, dict):
            return any(self._check_overflow_recursive(v) for v in x.values())
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
            return any(self._check_overflow_recursive(row) for row in x)
        if isinstance(x, list):
            return any(math.isinf(v) or math.isnan(v) or abs(v) > self.FP16_MAX
                       for v in x)
        return math.isinf(x) or math.isnan(x)

    def update_scale(self, overflow: bool) -> None:
        """更新损失缩放因子"""
        if overflow:
            self.scale *= self.scale_backoff
            self._steps_since_growth = 0
        else:
            self._steps_since_growth += 1
            if self._steps_since_growth >= self.growth_interval:
                self.scale *= self.scale_growth
                self._steps_since_growth = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "loss_scale": self.scale,
            "overflow_count": self.overflow_count,
            "underflow_count": self.underflow_count,
            "steps_since_growth": self._steps_since_growth,
        }


class DataParallelSimulator:
    """数据并行模拟器

    模拟多GPU数据并行:
    - 批次切分到多个设备
    - 各设备独立前向/反向
    - All-Reduce梯度聚合
    """

    def __init__(self, num_devices: int = 1, device_speed: float = 1.0):
        self.num_devices = num_devices
        self.device_speed = device_speed
        self.throughput_history: List[float] = []
        self.comm_overhead: float = 0.0

    def split_batch(self, batch: List[Any]) -> List[List[Any]]:
        """将batch切分到各设备"""
        if self.num_devices <= 1:
            return [batch]
        per = max(1, len(batch) // self.num_devices)
        shards = []
        for i in range(self.num_devices):
            start = i * per
            end = start + per if i < self.num_devices - 1 else len(batch)
            shards.append(batch[start:end])
        return shards

    def all_reduce(self, grads: List[Any]) -> List[Any]:
        """模拟All-Reduce: 平均各设备梯度"""
        if not grads:
            return []
        n = len(grads)
        dim = len(grads[0]) if grads[0] else 0
        result = [0.0] * dim
        for g in grads:
            for i in range(dim):
                result[i] += g[i]
        result = [v / n for v in result]
        # 通信开销 (与设备数和梯度大小成正比)
        self.comm_overhead += 0.001 * n * dim
        return result

    def estimate_speedup(self) -> float:
        """估计加速比"""
        if self.num_devices <= 1:
            return 1.0
        # Amdahl定律简化 (通信开销)
        ideal = self.num_devices * self.device_speed
        comm_ratio = min(0.3, 0.05 * self.num_devices)
        return ideal / (1.0 + comm_ratio * (ideal - 1))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_devices": self.num_devices,
            "device_speed": self.device_speed,
            "estimated_speedup": round(self.estimate_speedup(), 3),
            "comm_overhead": round(self.comm_overhead, 4),
            "throughput_samples": len(self.throughput_history),
        }


class PipelineParallelSimulator:
    """流水线并行模拟器

    模拟将模型层切分到多个阶段(stage):
    - 各stage负责连续的若干层
    - 微批次流水线执行
    - 气泡开销 (pipeline bubble)
    """

    def __init__(self, num_stages: int = 1, num_layers: int = 0):
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.stage_splits = self._split_layers()
        self.bubble_overhead: float = 0.0

    def _split_layers(self) -> List[Tuple[int, int]]:
        """将层均匀切分到各stage"""
        if self.num_stages <= 0 or self.num_layers <= 0:
            return [(0, self.num_layers)]
        per = self.num_layers // self.num_stages
        splits = []
        for i in range(self.num_stages):
            start = i * per
            end = start + per if i < self.num_stages - 1 else self.num_layers
            splits.append((start, end))
        return splits

    def forward_stage(self, stage_idx: int, x: List[List[float]],
                      layers: List[TransformerLayer]) -> List[List[float]]:
        """执行单个stage的前向"""
        start, end = self.stage_splits[stage_idx]
        for i in range(start, end):
            x = layers[i].forward(x, training=True)
        return x

    def estimate_bubble(self, num_microbatches: int) -> float:
        """估计流水线气泡比例"""
        if self.num_stages <= 1 or num_microbatches <= 0:
            return 0.0
        # 气泡 = (stages - 1) / (microbatches + stages - 1)
        bubble = (self.num_stages - 1) / (num_microbatches + self.num_stages - 1)
        self.bubble_overhead = bubble
        return bubble

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_stages": self.num_stages,
            "num_layers": self.num_layers,
            "stage_splits": self.stage_splits,
            "layers_per_stage": [e - s for s, e in self.stage_splits],
            "bubble_overhead": round(self.bubble_overhead, 4),
        }


class TrainingEngine:
    """真实训练引擎

    功能:
    - forward_pass(model, input_ids, targets) → loss, logits
    - backward_pass (简化梯度计算)
    - 混合精度: FP16/BF16 模拟
    - 梯度累积: accumulate_gradients(steps)
    - 梯度裁剪: clip_grad_norm
    - 学习率调度: cosine, linear, warmup
    - 训练循环: train_step, train_epoch
    - 检查点保存/加载
    - 分布式训练协调器: DataParallel/PipelineParallel模拟
    - 断点续训: resume_from_checkpoint
    """

    def __init__(self, model: LingyuanTransformerModel,
                 config: Any = None,
                 lr: float = 1e-3,
                 schedule: Optional[LRSchedule] = None,
                 weight_decay: float = 0.0,
                 max_grad_norm: float = 1.0,
                 grad_accumulation_steps: int = 1,
                 precision: str = "fp32",
                 num_dp_devices: int = 1,
                 num_pp_stages: int = 1):
        self.model = model
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.grad_accumulation_steps = max(1, grad_accumulation_steps)
        self.weight_decay = weight_decay

        # 学习率调度
        self.schedule = schedule or LRSchedule(
            base_lr=lr, total_steps=10000)

        # 优化器
        self.optimizer = SimpleOptimizer(
            self._get_params(), lr=lr, weight_decay=weight_decay)

        # 混合精度
        self.mixed_precision = MixedPrecisionSimulator(precision)

        # 分布式
        self.data_parallel = DataParallelSimulator(num_dp_devices)
        self.pipeline_parallel = PipelineParallelSimulator(
            num_pp_stages, model.num_layers)

        # 状态
        self.step = 0
        self.epoch = 0
        self.global_step = 0
        self.loss_history: List[float] = []
        self.lr_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self._accumulated_grads: Dict[str, Any] = {}
        self._accumulation_count = 0
        self._start_time = time.time()

    def _get_params(self) -> Dict[str, List[Any]]:
        """获取模型所有可训练参数的引用"""
        params: Dict[str, List[Any]] = {
            "token_embedding": self.model.token_embedding,
            "final_norm_weight": self.model.final_norm.weight,
        }
        if not self.model.tie_word_embeddings and self.model.lm_head is not None:
            params["lm_head"] = self.model.lm_head
        for i, layer in enumerate(self.model.layers):
            params[f"layer_{i}_W_q"] = layer.attn.W_q
            params[f"layer_{i}_W_k"] = layer.attn.W_k
            params[f"layer_{i}_W_v"] = layer.attn.W_v
            params[f"layer_{i}_W_o"] = layer.attn.W_o
            params[f"layer_{i}_norm1"] = layer.norm1.weight
            params[f"layer_{i}_norm2"] = layer.norm2.weight
            params[f"layer_{i}_W_gate"] = layer.ffn.W_gate
            params[f"layer_{i}_W_up"] = layer.ffn.W_up
            params[f"layer_{i}_W_down"] = layer.ffn.W_down
        return params

    # ---------- 学习率调度 ----------

    def get_lr(self, step: Optional[int] = None) -> float:
        """计算当前学习率"""
        s = step if step is not None else self.global_step
        sched = self.schedule
        base = sched.base_lr

        # Warmup
        if sched.warmup_steps > 0 and s < sched.warmup_steps:
            return base * (s + 1) / sched.warmup_steps

        progress = (s - sched.warmup_steps) / max(
            sched.total_steps - sched.warmup_steps, 1)
        progress = max(0.0, min(1.0, progress))

        if sched.schedule_type == "cosine":
            return sched.min_lr + 0.5 * (base - sched.min_lr) * \
                (1.0 + math.cos(math.pi * progress))
        elif sched.schedule_type == "linear":
            return base - (base - sched.min_lr) * progress
        elif sched.schedule_type == "constant":
            return base
        elif sched.schedule_type == "warmup_cosine":
            return sched.min_lr + 0.5 * (base - sched.min_lr) * \
                (1.0 + math.cos(math.pi * progress))
        return base

    # ---------- 前向 / 反向 (真实梯度) ----------

    def forward_pass(self, input_ids: List[int],
                     targets: List[int]
                     ) -> Tuple[float, List[List[float]], Dict[str, Any]]:
        """前向传播 + 损失计算 + 梯度缓存

        使用 model.forward_for_training() 保存所有中间激活值,
        供 backward_pass 计算真实梯度。

        Returns:
            (loss, logits, cache)
        """
        loss, logits, cache = self.model.forward_for_training(input_ids, targets)
        # 混合精度: 转换logits
        if self.mixed_precision.precision != "fp32":
            logits = self.mixed_precision.cast_matrix(logits)
            loss = self.mixed_precision.scale_loss(loss)
        return loss, logits, cache

    def backward_pass(self, logits: List[List[float]],
                      targets: List[int],
                      cache: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, Any]:
        """真实反向传播 — 基于前向缓存计算数学正确的梯度

        逐层使用链式法则:
        1. dL/dlogits = softmax(logits) - onehot(targets)
        2. LM Head: dL/dh_final = d_logits @ W_lmhead^T; dL/dW = h^T @ d_logits
        3. 最终RMSNorm: 真实梯度
        4. 每层 (逆序):
           - FFN (SwiGLU): 真实梯度 through gate/up/down
           - RMSNorm2: 真实梯度
           - Attention: 真实梯度 through Q/K/V/O + softmax Jacobian
           - RMSNorm1: 真实梯度
        5. Embedding: 真实梯度

        Returns:
            梯度字典 {param_name: grad}
        """
        if cache is None:
            # 退化: 无法计算真实梯度
            return self._backward_pass_legacy(logits, targets)

        seq_len = min(len(logits), len(targets))
        if seq_len == 0:
            return {}

        model = self.model
        hidden = model.hidden_dim
        vocab = model.vocab_size
        layer_caches = cache.get("layers", [])
        input_ids = cache.get("input_ids", [])
        h_final = cache["final_norm_out"]
        final_x_norm = cache["final_x_norm"]
        final_rms = cache["final_rms"]
        tie = model.tie_word_embeddings

        grads: Dict[str, Any] = {}

        # === 1. dL/dlogits = softmax(logits) - onehot(targets) ===
        d_logits: List[List[float]] = []
        for i in range(seq_len):
            probs = _softmax_vec(logits[i])
            tgt = targets[i]
            if 0 <= tgt < len(probs):
                probs[tgt] -= 1.0
            d_logits.append(probs)

        # === 2. LM Head 反向 ===
        # logits = h_final @ W_lmhead (或 token_embedding^T)
        # dL/dh_final = d_logits @ W_lmhead^T  (seq × hidden)
        # dL/dW_lmhead = h_final^T @ d_logits   (hidden × vocab)
        if tie:
            # W_lmhead = token_embedding^T (hidden × vocab)
            # dL/dh_final[s] = sum_v d_logits[s][v] * token_embedding[v]  (对v求和)
            d_h_final: List[List[float]] = []
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
            # dL/dtoken_embedding (from LM head) = h_final^T @ d_logits
            # 只累加, 后面 embedding 查表部分再叠加
            grad_emb: List[List[float]] = [[0.0] * hidden for _ in range(vocab)]
            for s in range(seq_len):
                hf_s = h_final[s]
                dl = d_logits[s]
                for v in range(vocab):
                    dlv = dl[v]
                    if dlv == 0.0:
                        continue
                    ge = grad_emb[v]
                    for d in range(hidden):
                        ge[d] += hf_s[d] * dlv
        else:
            # W_lmhead: (hidden × vocab)
            # dL/dh_final = d_logits @ W_lmhead^T
            d_h_final = []
            for s in range(seq_len):
                dl = d_logits[s]
                dh = [0.0] * hidden
                for d in range(hidden):
                    s_val = 0.0
                    wd = model.lm_head[d]
                    for v in range(vocab):
                        s_val += dl[v] * wd[v]
                    dh[d] = s_val
                d_h_final.append(dh)
            # dL/dlm_head = h_final^T @ d_logits  (hidden × vocab)
            grad_lm_head: List[List[float]] = [[0.0] * vocab for _ in range(hidden)]
            for s in range(seq_len):
                hf_s = h_final[s]
                dl = d_logits[s]
                for d in range(hidden):
                    hfd = hf_s[d]
                    if hfd == 0.0:
                        continue
                    gl = grad_lm_head[d]
                    for v in range(vocab):
                        gl[v] += hfd * dl[v]
            grads["lm_head"] = grad_lm_head
            grad_emb = [[0.0] * hidden for _ in range(vocab)]

        # === 3. 最终 RMSNorm 反向 ===
        d_x, d_final_norm_w = _rmsnorm_backward(
            d_h_final, final_x_norm, final_rms,
            model.final_norm.weight, hidden)
        grads["final_norm_weight"] = d_final_norm_w

        # === 4. 逐层反向 (逆序) ===
        for layer_idx in range(model.num_layers - 1, -1, -1):
            layer = model.layers[layer_idx]
            lc = layer_caches[layer_idx]
            attn = layer.attn
            ffn = layer.ffn
            num_heads = attn.num_heads
            num_kv_heads = attn.num_kv_heads
            head_dim = attn.head_dim
            n_rep = attn.n_rep
            kv_dim = attn.kv_dim
            ffn_dim = ffn.ffn_dim
            norm_eps = layer.norm_eps

            # d_x 当前是 dL/d(layer_output)
            # 残差: x_out = x_mid + ffn_out
            # dL/dx_mid = dL/dx_out (残差直通)
            # dL/dffn_out = dL/dx_out (残差直通)
            d_x_mid = [list(row) for row in d_x]
            d_ffn_out = [list(row) for row in d_x]

            # --- 4a. FFN (SwiGLU) 反向 ---
            # ffn_out = activated @ W_down
            # dL/dW_down = activated^T @ d_ffn_out  (ffn_dim × hidden)
            # dL/dactivated = d_ffn_out @ W_down^T  (seq × ffn_dim)
            activated = lc["ffn_activated"]
            h2 = lc["norm2_x_norm"]  # norm2 output (x_norm * weight) used as FFN input
            # Actually h2 stored is x_norm, need to multiply by weight for norm2 output
            # Wait - in forward_for_training, h2 = [xn[d] * weight[d] for d] which is the norm2 output
            # But we stored norm2_x_norm (the x/rms values, not multiplied by weight)
            # We need the actual h2 values (norm2 output) for FFN backward
            # h2 = norm2_x_norm * norm2.weight
            norm2_weight = layer.norm2.weight
            h2_actual = [[lc["norm2_x_norm"][s][d] * norm2_weight[d]
                          for d in range(hidden)]
                         for s in range(seq_len)]

            W_down_t = _transpose_2d(ffn.W_down)  # (hidden × ffn_dim)
            d_activated = _matmul_2d(d_ffn_out, W_down_t)  # (seq × ffn_dim)

            grad_W_down: List[List[float]] = [[0.0] * hidden for _ in range(ffn_dim)]
            for s in range(seq_len):
                _outer_product_add(grad_W_down, activated[s], d_ffn_out[s])

            # activated = silu(gate) * up
            # dL/dgate = dL/dactivated * up * silu'(gate)
            # dL/dup = dL/dactivated * silu(gate)
            gate = lc["ffn_gate"]
            up = lc["ffn_up"]
            d_gate: List[List[float]] = [[0.0] * ffn_dim for _ in range(seq_len)]
            d_up: List[List[float]] = [[0.0] * ffn_dim for _ in range(seq_len)]
            for s in range(seq_len):
                da = d_activated[s]
                gs = gate[s]
                us = up[s]
                dg = d_gate[s]
                du = d_up[s]
                for i in range(ffn_dim):
                    sg = _silu(gs[i])
                    dg[i] = da[i] * us[i] * _silu_grad(gs[i])
                    du[i] = da[i] * sg

            # gate = h2 @ W_gate, up = h2 @ W_up
            # dL/dW_gate = h2^T @ d_gate  (hidden × ffn_dim)
            # dL/dW_up = h2^T @ d_up      (hidden × ffn_dim)
            # dL/dh2 = d_gate @ W_gate^T + d_up @ W_up^T  (seq × hidden)
            grad_W_gate: List[List[float]] = [[0.0] * ffn_dim for _ in range(hidden)]
            grad_W_up: List[List[float]] = [[0.0] * ffn_dim for _ in range(hidden)]
            for s in range(seq_len):
                h2s = h2_actual[s]
                _outer_product_add(grad_W_gate, h2s, d_gate[s])
                _outer_product_add(grad_W_up, h2s, d_up[s])

            W_gate_t = _transpose_2d(ffn.W_gate)  # (ffn_dim × hidden)
            W_up_t = _transpose_2d(ffn.W_up)
            d_h2_from_gate = _matmul_2d(d_gate, W_gate_t)  # (seq × hidden)
            d_h2_from_up = _matmul_2d(d_up, W_up_t)
            d_h2 = [[d_h2_from_gate[s][d] + d_h2_from_up[s][d]
                      for d in range(hidden)]
                     for s in range(seq_len)]

            # --- 4b. RMSNorm 2 反向 ---
            d_x_from_norm2, d_norm2_w = _rmsnorm_backward(
                d_h2, lc["norm2_x_norm"], lc["norm2_rms"],
                norm2_weight, hidden)
            grads[f"layer_{layer_idx}_norm2"] = d_norm2_w

            # 累加到 d_x_mid (残差 + norm2反向)
            for s in range(seq_len):
                for d in range(hidden):
                    d_x_mid[s][d] += d_x_from_norm2[s][d]

            # --- 4c. Attention 反向 ---
            # 残差: x_mid = x_in + attn_out
            # dL/dx_in = dL/dx_mid (残差直通)
            # dL/d_attn_out = dL/dx_mid (残差直通)
            d_x_in = [list(row) for row in d_x_mid]
            d_attn_out = [list(row) for row in d_x_mid]

            # attn_out = merged @ W_o
            # dL/dW_o = merged^T @ d_attn_out  (hidden × hidden)
            # dL/dmerged = d_attn_out @ W_o^T  (seq × hidden)
            merged = lc["attn_merged"]
            grad_W_o: List[List[float]] = [[0.0] * hidden for _ in range(hidden)]
            for s in range(seq_len):
                _outer_product_add(grad_W_o, merged[s], d_attn_out[s])

            W_o_t = _transpose_2d(attn.W_o)  # (hidden × hidden)
            d_merged = _matmul_2d(d_attn_out, W_o_t)  # (seq × hidden)

            # 拆分回各头: d_merged (seq × hidden) -> [num_heads × (seq × head_dim)]
            d_out_heads = _split_heads_2d(d_merged, num_heads)

            # 逐头注意力反向
            Q_heads_rot = lc["Q_heads_rot"]
            K_heads_rot = lc["K_heads_rot"]
            V_heads = lc["V_heads"]
            attn_weights = lc["attn_weights"]
            scale = lc["attn_scale"]

            # 累积各头的 Q/K/V 梯度 (post-RoPE)
            d_Q_heads_rot: List[List[List[float]]] = [
                [[0.0] * head_dim for _ in range(seq_len)]
                for _ in range(num_heads)]
            d_K_heads_rot: List[List[List[float]]] = [
                [[0.0] * head_dim for _ in range(seq_len)]
                for _ in range(num_kv_heads)]
            d_V_heads: List[List[List[float]]] = [
                [[0.0] * head_dim for _ in range(seq_len)]
                for _ in range(num_kv_heads)]

            for h_idx in range(num_heads):
                q_h = Q_heads_rot[h_idx]
                kv_idx = h_idx // n_rep if n_rep > 1 else h_idx
                k_h = K_heads_rot[kv_idx] if attn.positional_encoding and \
                    attn.positional_encoding.method == "rope" else \
                    lc.get("K_heads_rot", lc.get("K_heads", [[]]))[kv_idx]
                v_h = V_heads[kv_idx]
                aw = attn_weights[h_idx]
                dout_h = d_out_heads[h_idx]

                # 1. dL/dV[j] = sum_i aw[i][j] * dout[i]
                for j in range(seq_len):
                    dv = d_V_heads[kv_idx][j]
                    for i in range(j, seq_len):  # 因果: i >= j
                        w = aw[i][j]
                        if w == 0.0:
                            continue
                        di = dout_h[i]
                        for d in range(head_dim):
                            dv[d] += w * di[d]

                # 2. dL/daw[i][j] = sum_d dout[i][d] * V[j][d]
                d_aw: List[List[float]] = [[0.0] * seq_len for _ in range(seq_len)]
                for i in range(seq_len):
                    di = dout_h[i]
                    for j in range(i + 1):  # 因果
                        vj = v_h[j]
                        s_val = 0.0
                        for d in range(head_dim):
                            s_val += di[d] * vj[d]
                        d_aw[i][j] = s_val

                # 3. Softmax 反向: dL/dscores[i][j] = aw[i][j] * (d_aw[i][j] - sum_k aw[i][k]*d_aw[i][k])
                d_scores: List[List[float]] = [[0.0] * seq_len for _ in range(seq_len)]
                for i in range(seq_len):
                    probs = aw[i]
                    dout_row = d_aw[i]
                    d_scores[i] = _softmax_backward_row(dout_row, probs)

                # 4. dL/dQ[i] = sum_j d_scores[i][j] * scale * K[j]
                for i in range(seq_len):
                    dq = d_Q_heads_rot[h_idx][i]
                    for j in range(i + 1):
                        ds = d_scores[i][j] * scale
                        if ds == 0.0:
                            continue
                        kj = k_h[j]
                        for d in range(head_dim):
                            dq[d] += ds * kj[d]

                # 5. dL/dK[j] = sum_i d_scores[i][j] * scale * Q[i]
                for j in range(seq_len):
                    dk = d_K_heads_rot[kv_idx][j]
                    for i in range(j, seq_len):  # 因果: i >= j
                        ds = d_scores[i][j] * scale
                        if ds == 0.0:
                            continue
                        qi = q_h[i]
                        for d in range(head_dim):
                            dk[d] += ds * qi[d]

            # 合并 Q 梯度 (post-RoPE) -> pre-RoPE
            d_Q_merged = _merge_heads_2d(d_Q_heads_rot)  # (seq × hidden)
            d_K_merged = _merge_heads_2d(d_K_heads_rot)  # (seq × kv_dim)
            d_V_merged = _merge_heads_2d(d_V_heads)      # (seq × kv_dim)

            # RoPE 反向 (逆旋转)
            use_rope = (attn.positional_encoding is not None
                        and attn.positional_encoding.method == "rope")
            if use_rope:
                rope_cos = attn.positional_encoding._rope_cos
                rope_sin = attn.positional_encoding._rope_sin
                d_Q_pre = _rope_backward(d_Q_merged, rope_cos, rope_sin, 0, head_dim)
                # 注意: Q_merged 是 hidden 维, 但 RoPE 只作用于 head_dim 维
                # 分头 -> 逆旋转 -> 合并
                d_Q_heads_pre = [_rope_backward(d_Q_heads_rot[h], rope_cos, rope_sin, 0, head_dim)
                                  for h in range(num_heads)]
                d_Q_pre = _merge_heads_2d(d_Q_heads_pre)
                d_K_heads_pre = [_rope_backward(d_K_heads_rot[k], rope_cos, rope_sin, 0, head_dim)
                                  for k in range(num_kv_heads)]
                d_K_pre = _merge_heads_2d(d_K_heads_pre)
            else:
                d_Q_pre = d_Q_merged
                d_K_pre = d_K_merged
            d_V_pre = d_V_merged

            # Q/K/V 投影反向: Q = h1 @ W_q
            # dL/dW_q = h1^T @ d_Q_pre  (hidden × hidden)
            # dL/dW_k = h1^T @ d_K_pre  (hidden × kv_dim)
            # dL/dW_v = h1^T @ d_V_pre  (hidden × kv_dim)
            # dL/dh1 = d_Q_pre @ W_q^T + d_K_pre @ W_k^T + d_V_pre @ W_v^T
            h1_actual = [[lc["norm1_x_norm"][s][d] * layer.norm1.weight[d]
                          for d in range(hidden)]
                         for s in range(seq_len)]

            grad_W_q: List[List[float]] = [[0.0] * hidden for _ in range(hidden)]
            grad_W_k: List[List[float]] = [[0.0] * kv_dim for _ in range(hidden)]
            grad_W_v: List[List[float]] = [[0.0] * kv_dim for _ in range(hidden)]
            for s in range(seq_len):
                h1s = h1_actual[s]
                _outer_product_add(grad_W_q, h1s, d_Q_pre[s])
                _outer_product_add(grad_W_k, h1s, d_K_pre[s])
                _outer_product_add(grad_W_v, h1s, d_V_pre[s])

            W_q_t = _transpose_2d(attn.W_q)
            W_k_t = _transpose_2d(attn.W_k)
            W_v_t = _transpose_2d(attn.W_v)
            d_h1 = _matmul_2d(d_Q_pre, W_q_t)
            d_h1_k = _matmul_2d(d_K_pre, W_k_t)
            d_h1_v = _matmul_2d(d_V_pre, W_v_t)
            for s in range(seq_len):
                for d in range(hidden):
                    d_h1[s][d] += d_h1_k[s][d] + d_h1_v[s][d]

            # --- 4d. RMSNorm 1 反向 ---
            d_x_from_norm1, d_norm1_w = _rmsnorm_backward(
                d_h1, lc["norm1_x_norm"], lc["norm1_rms"],
                layer.norm1.weight, hidden)
            grads[f"layer_{layer_idx}_norm1"] = d_norm1_w

            # 累加到 d_x_in (残差 + norm1反向)
            for s in range(seq_len):
                for d in range(hidden):
                    d_x_in[s][d] += d_x_from_norm1[s][d]

            # 存储层梯度
            grads[f"layer_{layer_idx}_W_q"] = grad_W_q
            grads[f"layer_{layer_idx}_W_k"] = grad_W_k
            grads[f"layer_{layer_idx}_W_v"] = grad_W_v
            grads[f"layer_{layer_idx}_W_o"] = grad_W_o
            grads[f"layer_{layer_idx}_W_gate"] = grad_W_gate
            grads[f"layer_{layer_idx}_W_up"] = grad_W_up
            grads[f"layer_{layer_idx}_W_down"] = grad_W_down

            # d_x_in 传递给上一层
            d_x = d_x_in

        # === 5. Embedding 反向 ===
        # d_x 是 dL/d(embeddings)
        # 对于每个 input_ids[s], 累加梯度到 token_embedding[input_ids[s]]
        for s in range(seq_len):
            tid = input_ids[s] if s < len(input_ids) else -1
            if 0 <= tid < vocab:
                ge = grad_emb[tid]
                dxs = d_x[s]
                for d in range(hidden):
                    ge[d] += dxs[d]

        grads["token_embedding"] = grad_emb

        return grads

    def _backward_pass_legacy(self, logits: List[List[float]],
                              targets: List[int]) -> Dict[str, Any]:
        """退化反向传播 (无缓存时使用近似梯度)

        当 forward_pass 未提供 cache 时回退到此方法。
        """
        seq_len = min(len(logits), len(targets))
        if seq_len == 0:
            return {}

        grads: Dict[str, Any] = {}
        vocab = self.model.vocab_size

        d_logits = []
        for i in range(seq_len):
            probs = _softmax_vec(logits[i])
            tgt = targets[i]
            if 0 <= tgt < len(probs):
                probs[tgt] -= 1.0
            d_logits.append(probs)

        loss_signal = sum(sum(abs(v) for v in row) for row in d_logits) / max(seq_len, 1)

        if not self.model.tie_word_embeddings and self.model.lm_head is not None:
            grads["lm_head"] = [[loss_signal * 0.01 * random.gauss(0, 1)
                                  for _ in range(vocab)]
                                 for _ in range(self.model.hidden_dim)]

        grads["token_embedding"] = [[loss_signal * 0.01 * random.gauss(0, 1)
                                      for _ in range(self.model.hidden_dim)]
                                     for _ in range(vocab)]

        for i, layer in enumerate(self.model.layers):
            attn_dim = layer.attn.hidden_dim
            kv_dim = layer.attn.kv_dim
            ffn_dim = layer.ffn.ffn_dim
            scale = loss_signal * 0.005
            grads[f"layer_{i}_W_q"] = [[scale * random.gauss(0, 1) for _ in range(attn_dim)] for _ in range(attn_dim)]
            grads[f"layer_{i}_W_k"] = [[scale * random.gauss(0, 1) for _ in range(kv_dim)] for _ in range(attn_dim)]
            grads[f"layer_{i}_W_v"] = [[scale * random.gauss(0, 1) for _ in range(kv_dim)] for _ in range(attn_dim)]
            grads[f"layer_{i}_W_o"] = [[scale * random.gauss(0, 1) for _ in range(attn_dim)] for _ in range(attn_dim)]
            grads[f"layer_{i}_norm1"] = [scale * random.gauss(0, 1) for _ in range(attn_dim)]
            grads[f"layer_{i}_norm2"] = [scale * random.gauss(0, 1) for _ in range(attn_dim)]
            grads[f"layer_{i}_W_gate"] = [[scale * random.gauss(0, 1) for _ in range(ffn_dim)] for _ in range(attn_dim)]
            grads[f"layer_{i}_W_up"] = [[scale * random.gauss(0, 1) for _ in range(ffn_dim)] for _ in range(attn_dim)]
            grads[f"layer_{i}_W_down"] = [[scale * random.gauss(0, 1) for _ in range(attn_dim)] for _ in range(ffn_dim)]

        grads["final_norm_weight"] = [loss_signal * 0.01 * random.gauss(0, 1)
                                       for _ in range(self.model.hidden_dim)]
        return grads

    # ---------- 梯度管理 ----------

    def clip_grad_norm(self, grads: Dict[str, Any],
                       max_norm: Optional[float] = None) -> float:
        """梯度裁剪 (全局L2范数)"""
        max_norm = max_norm or self.max_grad_norm
        total_norm_sq = 0.0

        def _accumulate(x: Any) -> None:
            nonlocal total_norm_sq
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                for row in x:
                    _accumulate(row)
            elif isinstance(x, list):
                for v in x:
                    total_norm_sq += v * v
            else:
                total_norm_sq += x * x

        for g in grads.values():
            _accumulate(g)

        total_norm = math.sqrt(total_norm_sq)
        self.grad_norm_history.append(total_norm)

        if total_norm > max_norm and total_norm > 0:
            scale = max_norm / total_norm

            def _scale(x: Any) -> Any:
                if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                    return [_scale(row) for row in x]
                elif isinstance(x, list):
                    return [v * scale for v in x]
                return x * scale

            grads = {k: _scale(v) for k, v in grads.items()}

        return total_norm

    def accumulate_gradients(self, grads: Dict[str, Any]) -> None:
        """梯度累积"""
        if not self._accumulated_grads:
            self._accumulated_grads = {k: self._deep_copy_grad(v)
                                       for k, v in grads.items()}
        else:
            for k in grads:
                if k in self._accumulated_grads:
                    self._accumulated_grads[k] = self._add_grads(
                        self._accumulated_grads[k], grads[k])
        self._accumulation_count += 1

    @staticmethod
    def _deep_copy_grad(x: Any) -> Any:
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
            return [list(row) for row in x]
        elif isinstance(x, list):
            return list(x)
        return x

    @staticmethod
    def _add_grads(a: Any, b: Any) -> Any:
        if isinstance(a, list) and len(a) > 0 and isinstance(a[0], list):
            return [[a[i][j] + b[i][j] for j in range(len(a[i]))]
                    for i in range(len(a))]
        elif isinstance(a, list):
            return [a[i] + b[i] for i in range(len(a))]
        return a + b

    def _reset_accumulation(self) -> None:
        self._accumulated_grads = {}
        self._accumulation_count = 0

    # ---------- 训练循环 ----------

    def train_step(self, batch: List[Tuple[List[int], List[int]]]) -> Dict[str, Any]:
        """单个训练步

        Args:
            batch: [(input_ids, targets), ...]

        Returns:
            步骤统计
        """
        step_loss = 0.0
        step_grad_norm = 0.0
        n_samples = len(batch)

        # 数据并行: 切分batch
        shards = self.data_parallel.split_batch(batch)

        all_grads: List[Dict[str, Any]] = []
        for shard in shards:
            for input_ids, targets in shard:
                # 前向 (带梯度缓存)
                loss, logits, fwd_cache = self.forward_pass(input_ids, targets)
                step_loss += loss

                # 反向 (真实梯度, 使用前向缓存)
                grads = self.backward_pass(logits, targets, cache=fwd_cache)

                # 混合精度: 反向缩放
                if self.mixed_precision.precision != "fp32":
                    grads = self.mixed_precision.unscale_grads(grads)

                # 梯度累积
                self.accumulate_gradients(grads)

        # 是否达到累积步数
        if self._accumulation_count >= self.grad_accumulation_steps:
            # 平均累积梯度
            avg_grads = {k: self._scale_grad(v, 1.0 / self._accumulation_count)
                         for k, v in self._accumulated_grads.items()}

            # 梯度裁剪
            step_grad_norm = self.clip_grad_norm(avg_grads)

            # 更新学习率
            current_lr = self.get_lr()
            self.optimizer.lr = current_lr

            # 优化器步进
            self.optimizer.step(self._get_params(), avg_grads)

            # 重置累积
            self._reset_accumulation()
            self.global_step += 1
            self.lr_history.append(current_lr)

        avg_loss = step_loss / max(n_samples, 1)
        self.loss_history.append(avg_loss)
        self.step += 1

        return {
            "step": self.step,
            "global_step": self.global_step,
            "loss": round(avg_loss, 6),
            "grad_norm": round(step_grad_norm, 4),
            "lr": round(self.get_lr(), 8),
            "samples": n_samples,
            "accumulation_count": self._accumulation_count,
        }

    @staticmethod
    def _scale_grad(x: Any, factor: float) -> Any:
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
            return [[v * factor for v in row] for row in x]
        elif isinstance(x, list):
            return [v * factor for v in x]
        return x * factor

    def train_epoch(self, dataset: List[Tuple[List[int], List[int]]],
                    batch_size: int = 4, verbose: bool = False) -> Dict[str, Any]:
        """训练一个epoch

        Args:
            dataset: [(input_ids, targets), ...]
            batch_size: 批次大小
            verbose: 打印进度

        Returns:
            epoch统计
        """
        self.epoch += 1
        epoch_losses: List[float] = []
        epoch_start = time.time()

        # 打乱数据
        shuffled = list(dataset)
        random.shuffle(shuffled)

        n_batches = 0
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            if not batch:
                continue
            result = self.train_step(batch)
            epoch_losses.append(result["loss"])
            n_batches += 1
            if verbose and n_batches % 10 == 0:
                print(f"  Epoch {self.epoch} | Batch {n_batches} | "
                      f"Loss {result['loss']:.4f} | LR {result['lr']:.6f}")

        elapsed = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        return {
            "epoch": self.epoch,
            "batches": n_batches,
            "avg_loss": round(avg_loss, 6),
            "min_loss": round(min(epoch_losses), 6) if epoch_losses else 0.0,
            "max_loss": round(max(epoch_losses), 6) if epoch_losses else 0.0,
            "elapsed_sec": round(elapsed, 3),
            "samples_per_sec": round(len(shuffled) / max(elapsed, 1e-6), 2),
        }

    # ---------- 检查点 ----------

    def save_checkpoint(self, path: str) -> bool:
        """保存检查点"""
        checkpoint = {
            "step": self.step,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "lr": self.get_lr(),
            "loss_history": self.loss_history[-100:],
            "lr_history": self.lr_history[-100:],
            "grad_norm_history": self.grad_norm_history[-100:],
            "optimizer_t": self.optimizer.t,
            "mixed_precision": self.mixed_precision.get_stats(),
            "model_summary": self.model.get_weights_summary(),
            "saved_at": datetime.now().isoformat(),
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"检查点保存失败: {e}")
            return False

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载检查点 (仅恢复训练状态, 不恢复权重)"""
        with open(path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        self.step = checkpoint.get("step", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.loss_history = checkpoint.get("loss_history", [])
        self.lr_history = checkpoint.get("lr_history", [])
        self.grad_norm_history = checkpoint.get("grad_norm_history", [])
        self.optimizer.t = checkpoint.get("optimizer_t", 0)
        return checkpoint

    def resume_from_checkpoint(self, path: str) -> bool:
        """断点续训"""
        try:
            info = self.load_checkpoint(path)
            print(f"从检查点恢复: step={info['step']}, "
                  f"epoch={info['epoch']}, "
                  f"last_loss={info['loss_history'][-1] if info['loss_history'] else 'N/A'}")
            return True
        except FileNotFoundError:
            print(f"检查点不存在: {path}")
            return False
        except Exception as e:
            print(f"恢复失败: {e}")
            return False

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "current_lr": round(self.get_lr(), 8),
            "avg_loss": round(sum(self.loss_history) / max(len(self.loss_history), 1), 6),
            "last_loss": self.loss_history[-1] if self.loss_history else None,
            "last_grad_norm": self.grad_norm_history[-1] if self.grad_norm_history else None,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "precision": self.mixed_precision.precision,
            "uptime_sec": round(time.time() - self._start_time, 1),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        return {
            "training": self.get_stats(),
            "model": self.model.get_stats(),
            "optimizer": {
                "lr": self.optimizer.lr,
                "betas": (self.optimizer.beta1, self.optimizer.beta2),
                "weight_decay": self.optimizer.weight_decay,
                "step": self.optimizer.t,
            },
            "schedule": asdict(self.schedule),
            "mixed_precision": self.mixed_precision.get_stats(),
            "data_parallel": self.data_parallel.get_stats(),
            "pipeline_parallel": self.pipeline_parallel.get_stats(),
            "loss_trend": self.loss_history[-20:],
            "lr_trend": self.lr_history[-20:],
        }


# ============================================================
# #9 ModelConfig [模型配置管理]
# ============================================================

@dataclass
class ModelConfig:
    """模型配置管理

    功能:
    - 预设配置管理 (tiny/small/base/large)
    - 配置验证
    - 配置序列化 (save/load JSON)
    - 配置自动推导 (根据参数量/显存需求推荐配置)
    """

    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: int = 8
    ffn_dim: int = 1024
    max_seq_len: int = 2048
    vocab_size: int = 32000
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    sliding_window: int = 0
    tie_word_embeddings: bool = True
    pos_method: str = "rope"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # 预设配置 (无类型注解 → 非dataclass字段, 为类常量)
    PRESETS = _MODEL_PRESETS

    # ---------- 预设 ----------

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ModelConfig":
        """从预设名称创建配置"""
        if name not in cls.PRESETS:
            raise ValueError(f"未知预设: {name}, 可选: {list(cls.PRESETS.keys())}")
        cfg = dict(cls.PRESETS[name])
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def list_presets(cls) -> Dict[str, Dict[str, Any]]:
        """列出所有预设"""
        return dict(cls.PRESETS)

    @classmethod
    def get_preset_names(cls) -> List[str]:
        return list(cls.PRESETS.keys())

    # ---------- 验证 ----------

    def validate(self) -> Tuple[bool, List[str]]:
        """配置验证

        Returns:
            (是否有效, 错误消息列表)
        """
        errors: List[str] = []

        if self.hidden_dim <= 0:
            errors.append("hidden_dim必须为正数")
        if self.num_layers <= 0:
            errors.append("num_layers必须为正数")
        if self.num_heads <= 0:
            errors.append("num_heads必须为正数")
        if self.num_kv_heads <= 0:
            errors.append("num_kv_heads必须为正数")
        if self.ffn_dim <= 0:
            errors.append("ffn_dim必须为正数")
        if self.max_seq_len <= 0:
            errors.append("max_seq_len必须为正数")
        if self.vocab_size <= 0:
            errors.append("vocab_size必须为正数")

        # hidden_dim必须能被num_heads整除
        if self.num_heads > 0 and self.hidden_dim % self.num_heads != 0:
            errors.append(
                f"hidden_dim({self.hidden_dim})必须能被num_heads({self.num_heads})整除")

        # num_heads必须能被num_kv_heads整除 (GQA要求)
        if self.num_kv_heads > 0 and self.num_heads % self.num_kv_heads != 0:
            errors.append(
                f"num_heads({self.num_heads})必须能被num_kv_heads({self.num_kv_heads})整除")

        # num_kv_heads不能超过num_heads
        if self.num_kv_heads > self.num_heads:
            errors.append(f"num_kv_heads({self.num_kv_heads})不能超过num_heads({self.num_heads})")

        # ffn_dim通常为hidden_dim的倍数
        if self.ffn_dim < self.hidden_dim:
            errors.append(f"ffn_dim({self.ffn_dim})建议不小于hidden_dim({self.hidden_dim})")

        # rope_theta
        if self.rope_theta <= 0:
            errors.append("rope_theta必须为正数")

        # pos_method
        if self.pos_method not in ("rope", "alibi", "absolute"):
            errors.append(f"pos_method必须为rope/alibi/absolute, 当前: {self.pos_method}")

        return (len(errors) == 0, errors)

    def validate_or_raise(self) -> None:
        """验证配置, 失败则抛出异常"""
        ok, errors = self.validate()
        if not ok:
            raise ValueError(f"配置无效: {'; '.join(errors)}")

    # ---------- 序列化 ----------

    def save(self, path: str) -> bool:
        """保存配置到JSON"""
        data = asdict(self)
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                        exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """从JSON加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        """从字典创建"""
        return cls(**d)

    # ---------- 参数/显存估计 ----------

    def estimate_params(self) -> int:
        """估计参数量"""
        head_dim = self.hidden_dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim
        embedding = self.vocab_size * self.hidden_dim
        per_layer = (
            self.hidden_dim * self.hidden_dim    # W_q
            + self.hidden_dim * kv_dim           # W_k
            + self.hidden_dim * kv_dim           # W_v
            + self.hidden_dim * self.hidden_dim  # W_o
            + 3 * self.hidden_dim * self.ffn_dim  # SwiGLU (gate, up, down)
            + 2 * self.hidden_dim                # 2×RMSNorm
        )
        total = embedding + per_layer * self.num_layers + self.hidden_dim  # final norm
        if not self.tie_word_embeddings:
            total += self.hidden_dim * self.vocab_size
        return total

    def estimate_memory(self, precision: str = "fp32") -> Dict[str, float]:
        """估计显存占用

        包含: 模型权重 + 梯度 + 优化器状态(Adam) + 激活值
        """
        bytes_per = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(precision, 4)
        params = self.estimate_params()

        model_mb = params * bytes_per / 1e6
        grad_mb = params * bytes_per / 1e6
        optim_mb = params * 2 * bytes_per / 1e6   # Adam: momentum + variance

        # 激活值估计 (近似: 与seq_len, hidden_dim, layers成正比)
        activation_mb = (self.max_seq_len * self.hidden_dim * self.num_layers
                         * bytes_per * 4) / 1e6   # 4: 近似激活系数

        total_mb = model_mb + grad_mb + optim_mb + activation_mb
        return {
            "params": params,
            "precision": precision,
            "bytes_per_param": bytes_per,
            "model_mb": round(model_mb, 2),
            "gradient_mb": round(grad_mb, 2),
            "optimizer_mb": round(optim_mb, 2),
            "activation_mb": round(activation_mb, 2),
            "total_mb": round(total_mb, 2),
            "total_gb": round(total_mb / 1024, 3),
        }

    # ---------- 自动推导 ----------

    @classmethod
    def auto_derive(cls, target_params: Optional[int] = None,
                    memory_gb: Optional[float] = None,
                    max_layers: Optional[int] = None,
                    preferred_preset: Optional[str] = None) -> "ModelConfig":
        """根据约束自动推导推荐配置

        Args:
            target_params: 目标参数量 (如 1.4e9 for 1.4B)
            memory_gb: 可用显存(GB)
            max_layers: 最大层数限制
            preferred_preset: 偏好的预设起点

        Returns:
            推荐的ModelConfig
        """
        # 1. 选择起点预设
        if preferred_preset and preferred_preset in cls.PRESETS:
            cfg = cls.from_preset(preferred_preset)
        elif target_params is not None:
            # 根据目标参数量选最近预设
            best_name = "small"
            best_diff = float("inf")
            for name in cls.PRESETS:
                preset_cfg = cls.from_preset(name)
                diff = abs(preset_cfg.estimate_params() - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_name = name
            cfg = cls.from_preset(best_name)
        else:
            cfg = cls.from_preset("small")

        # 2. 根据目标参数量调整
        if target_params is not None:
            current = cfg.estimate_params()
            if current > 0:
                ratio = (target_params / current) ** (1.0 / 3.0)  # 立方根缩放
                cfg.hidden_dim = max(64, int(cfg.hidden_dim * ratio))
                cfg.hidden_dim = (cfg.hidden_dim // cfg.num_heads) * cfg.num_heads
                cfg.ffn_dim = cfg.hidden_dim * 2
                # 调整层数
                if max_layers:
                    cfg.num_layers = min(max_layers, int(cfg.num_layers * ratio))
                else:
                    cfg.num_layers = max(1, int(cfg.num_layers * ratio))

        # 3. 根据显存约束调整
        if memory_gb is not None:
            for prec in ["fp32", "fp16", "bf16"]:
                mem = cfg.estimate_memory(prec)
                if mem["total_gb"] <= memory_gb:
                    break
            else:
                # 都超了, 缩小配置
                while cfg.estimate_memory("fp16")["total_gb"] > memory_gb:
                    cfg.hidden_dim = max(64, cfg.hidden_dim - 64)
                    cfg.hidden_dim = (cfg.hidden_dim // cfg.num_heads) * cfg.num_heads
                    cfg.ffn_dim = cfg.hidden_dim * 2
                    cfg.num_layers = max(1, cfg.num_layers - 1)
                    if cfg.hidden_dim <= 64 and cfg.num_layers <= 1:
                        break

        # 4. 确保num_kv_heads兼容
        if cfg.num_heads % cfg.num_kv_heads != 0:
            cfg.num_kv_heads = cfg.num_heads

        # 5. 验证
        ok, errors = cfg.validate()
        if not ok:
            # 回退到small预设
            cfg = cls.from_preset("small")

        return cfg

    @classmethod
    def recommend_for_gpu(cls, gpu_memory_gb: float,
                          gpu_name: str = "") -> "ModelConfig":
        """根据GPU显存推荐配置"""
        # 粗略映射: 每GB显存约支持 ~0.5B参数 (FP16训练)
        approx_params = int(gpu_memory_gb * 0.4e9)
        return cls.auto_derive(target_params=approx_params, memory_gb=gpu_memory_gb)

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, Any]:
        ok, errors = self.validate()
        params = self.estimate_params()
        return {
            "config_name": "ModelConfig",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.hidden_dim // self.num_heads if self.num_heads > 0 else 0,
            "ffn_dim": self.ffn_dim,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "rope_theta": self.rope_theta,
            "pos_method": self.pos_method,
            "tie_word_embeddings": self.tie_word_embeddings,
            "num_params": params,
            "num_params_human": cls_human_count(params),
            "is_valid": ok,
            "validation_errors": errors,
            "memory_fp32": self.estimate_memory("fp32"),
            "memory_fp16": self.estimate_memory("fp16"),
            "presets_available": list(self.PRESETS.keys()),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        stats = self.get_stats()
        stats["config_dict"] = self.to_dict()
        # 各预设参数量对比
        preset_comparison = {}
        for name in self.PRESETS:
            pc = self.from_preset(name)
            preset_comparison[name] = {
                "params": pc.estimate_params(),
                "params_human": cls_human_count(pc.estimate_params()),
                "memory_gb": pc.estimate_memory("fp16")["total_gb"],
            }
        stats["preset_comparison"] = preset_comparison
        return stats


def cls_human_count(n: int) -> str:
    """将参数量转为人类可读格式 (如 1.4B)"""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
