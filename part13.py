"""
灵元大模型 - 微调与持续学习模块 (part13.py)
对应52项清单 #30-35

子系统:
  #30 LoRATuner        - LoRA/QLoRA 低秩微调
  #31 FullFineTuner    - 全参数微调
  #32 SFTTrainer       - 指令微调 (Supervised Fine-Tuning)
  #33 DPOTrainer       - 直接偏好优化 (Direct Preference Optimization)
  #34 ContinualLearner - 持续学习
  #35 DomainAdapter    - 领域适配

纯 Python 标准库实现, 零外部依赖.
此文件在 lingyuan_full.py 之后加载, 可使用全局变量: DATA_DIR, LOG_DIR, CONFIG_DIR
"""

import uuid
import math
import random
import json
import os
import time
from collections import deque, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime

# ============================================================
# 全局路径 (优先使用 lingyuan_full.py 中定义的全局变量)
# ============================================================
_DATA_DIR = globals().get('DATA_DIR', os.path.join(os.path.expanduser('~'), '.lingyuan', 'data'))
_LOG_DIR = globals().get('LOG_DIR', os.path.join(os.path.expanduser('~'), '.lingyuan', 'logs'))
_CONFIG_DIR = globals().get('CONFIG_DIR', os.path.join(os.path.expanduser('~'), '.lingyuan', 'config'))

for _d in [_DATA_DIR, _LOG_DIR, _CONFIG_DIR]:
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception:
        pass


# ============================================================
# 第一部分: 矩阵运算工具 (纯 Python 实现)
# ============================================================

def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩阵乘法 A[m×n] @ B[n×p] → C[m×p]"""
    if not A or not B or not A[0] or not B[0]:
        return []
    m, n, p = len(A), len(A[0]), len(B[0])
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            for j in range(p):
                Ci[j] += a * Bk[j]
    return C


def matadd(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩阵逐元素加法 A + B"""
    if not A:
        return []
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def matsub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩阵逐元素减法 A - B"""
    if not A:
        return []
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def transpose(A: List[List[float]]) -> List[List[float]]:
    """矩阵转置 A[m×n] → A^T[n×m]"""
    if not A or not A[0]:
        return []
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def zeros(rows: int, cols: int) -> List[List[float]]:
    """创建 rows×cols 零矩阵"""
    return [[0.0] * cols for _ in range(rows)]


def ones(rows: int, cols: int) -> List[List[float]]:
    """创建 rows×cols 全 1 矩阵"""
    return [[1.0] * cols for _ in range(rows)]


def identity(n: int) -> List[List[float]]:
    """创建 n×n 单位矩阵"""
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def random_gauss(rows: int, cols: int, mean: float = 0.0, std: float = 0.02) -> List[List[float]]:
    """创建高斯随机矩阵"""
    return [[random.gauss(mean, std) for _ in range(cols)] for _ in range(rows)]


def random_uniform(rows: int, cols: int, low: float = -0.02, high: float = 0.02) -> List[List[float]]:
    """创建均匀随机矩阵"""
    return [[random.uniform(low, high) for _ in range(cols)] for _ in range(rows)]


def scalar_mul(A: List[List[float]], s: float) -> List[List[float]]:
    """矩阵标量乘法 A * s"""
    if not A:
        return []
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]


def mat_add_inplace(A: List[List[float]], B: List[List[float]]) -> None:
    """矩阵原地加法 A += B"""
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] += B[i][j]


def mat_sub_inplace(A: List[List[float]], B: List[List[float]]) -> None:
    """矩阵原地减法 A -= B"""
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] -= B[i][j]


def mat_mul_scalar_inplace(A: List[List[float]], s: float) -> None:
    """矩阵原地标量乘法 A *= s"""
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] *= s


def outer_product(a: List[float], b: List[float]) -> List[List[float]]:
    """外积 a[m] × b[n] → [m×n]"""
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


def frobenius_norm(A: List[List[float]]) -> float:
    """计算矩阵的 Frobenius 范数"""
    return math.sqrt(sum(v ** 2 for row in A for v in row))


def clip_matrix_norm(grad: List[List[float]], max_norm: float) -> List[List[float]]:
    """按 Frobenius 范数裁剪矩阵"""
    norm = frobenius_norm(grad)
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        return scalar_mul(grad, scale)
    return grad


def mat_sum(A: List[List[float]]) -> float:
    """矩阵所有元素之和"""
    return sum(sum(row) for row in A)


def mat_mean(A: List[List[float]]) -> float:
    """矩阵所有元素均值"""
    if not A or not A[0]:
        return 0.0
    return mat_sum(A) / (len(A) * len(A[0]))


def deep_copy_matrix(A: List[List[float]]) -> List[List[float]]:
    """深拷贝矩阵"""
    return [row[:] for row in A]


def mat_to_list(A: List[List[float]]) -> List[List[float]]:
    """深拷贝矩阵 (别名)"""
    return [row[:] for row in A]


# ============================================================
# 第二部分: 神经网络工具函数
# ============================================================

def softmax(logits: List[float]) -> List[float]:
    """Softmax 函数, 数值稳定版"""
    if not logits:
        return []
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


def log_softmax(logits: List[float]) -> List[float]:
    """Log-Softmax 函数, 数值稳定版"""
    if not logits:
        return []
    max_val = max(logits)
    shifted = [v - max_val for v in logits]
    total = sum(math.exp(v) for v in shifted)
    log_total = math.log(total) if total > 0 else 0.0
    return [v - log_total for v in shifted]


def gelu(x: float) -> float:
    """GELU 激活函数: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_grad(x: float) -> float:
    """GELU 激活函数的导数"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) + \
           x * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def relu(x: float) -> float:
    """ReLU 激活函数"""
    return max(0.0, x)


def relu_grad(x: float) -> float:
    """ReLU 的导数"""
    return 1.0 if x > 0 else 0.0


def sigmoid(x: float) -> float:
    """Sigmoid 函数, 数值稳定版"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def cross_entropy_loss(logits: List[float], target: int) -> float:
    """单个位置的交叉熵损失: -log(softmax(logits)[target])"""
    ls = log_softmax(logits)
    return -ls[target] if 0 <= target < len(ls) else 0.0


def cross_entropy_grad(logits: List[float], target: int) -> List[float]:
    """交叉熵对 logits 的梯度: softmax(logits) - one_hot(target)"""
    sm = softmax(logits)
    grad = sm[:]
    if 0 <= target < len(grad):
        grad[target] -= 1.0
    return grad


def sequence_cross_entropy(logits: List[List[float]], targets: List[int],
                           mask: Optional[List[int]] = None) -> float:
    """序列交叉熵损失, 可选 mask (只对 mask=1 的位置计算)"""
    seq = len(logits)
    if seq == 0:
        return 0.0
    if mask is None:
        mask = [1] * seq
    total_loss = 0.0
    total_mask = 0
    for t in range(seq):
        if mask[t] == 1 and t < len(targets):
            total_loss += cross_entropy_loss(logits[t], targets[t])
            total_mask += 1
    return total_loss / max(total_mask, 1)


def sequence_ce_grad(logits: List[List[float]], targets: List[int],
                     mask: Optional[List[int]] = None) -> List[List[float]]:
    """序列交叉熵对 logits 的梯度"""
    seq = len(logits)
    if seq == 0:
        return []
    if mask is None:
        mask = [1] * seq
    total_mask = max(sum(mask), 1)
    grad = []
    for t in range(seq):
        if mask[t] == 1 and t < len(targets):
            g = cross_entropy_grad(logits[t], targets[t])
            grad.append([v / total_mask for v in g])
        else:
            grad.append([0.0] * len(logits[t]))
    return grad


def sequence_log_probs(logits: List[List[float]], targets: List[int],
                       mask: Optional[List[int]] = None) -> float:
    """计算序列的对数概率 (用于 DPO)"""
    seq = len(logits)
    if seq == 0:
        return 0.0
    if mask is None:
        mask = [1] * seq
    total_lp = 0.0
    for t in range(seq):
        if mask[t] == 1 and t < len(targets):
            ls = log_softmax(logits[t])
            if 0 <= targets[t] < len(ls):
                total_lp += ls[targets[t]]
    return total_lp


def sequence_log_probs_grad(logits: List[List[float]], targets: List[int],
                            weight: float,
                            mask: Optional[List[int]] = None) -> List[List[float]]:
    """序列对数概率对 logits 的梯度, 乘以 weight (用于 DPO 反向传播)"""
    seq = len(logits)
    if seq == 0:
        return []
    if mask is None:
        mask = [1] * seq
    grad = []
    for t in range(seq):
        if mask[t] == 1 and t < len(targets):
            sm = softmax(logits[t])
            g = [(-sm[j]) * weight for j in range(len(sm))]
            if 0 <= targets[t] < len(g):
                g[targets[t]] += weight
            grad.append(g)
        else:
            grad.append([0.0] * len(logits[t]))
    return grad


# ============================================================
# 第三部分: 通用辅助函数
# ============================================================

def argmax(lst: List[float]) -> int:
    """返回最大值的索引"""
    if not lst:
        return 0
    best_idx, best_val = 0, lst[0]
    for i in range(1, len(lst)):
        if lst[i] > best_val:
            best_val = lst[i]
            best_idx = i
    return best_idx


def flatten(nested: List[List[Any]]) -> List[Any]:
    """展平二维列表"""
    return [item for sublist in nested for item in sublist]


def mean(lst: List[float]) -> float:
    """均值"""
    return sum(lst) / len(lst) if lst else 0.0


def std(lst: List[float]) -> float:
    """标准差"""
    if len(lst) < 2:
        return 0.0
    m = mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))


def safe_exp(x: float, max_val: float = 50.0) -> float:
    """安全的 exp, 防止溢出"""
    return math.exp(max(-max_val, min(max_val, x)))


def tokenize_simple(text: str, vocab: Optional[Dict[str, int]] = None,
                    max_vocab: int = 256) -> Tuple[List[int], Dict[str, int]]:
    """简单的字符级分词器 (纯 Python)"""
    if vocab is None:
        vocab = {}
    ids = []
    for ch in text:
        if ch not in vocab:
            if len(vocab) < max_vocab:
                vocab[ch] = len(vocab)
            else:
                vocab[ch] = 0
        ids.append(vocab[ch])
    return ids, vocab


def detokenize_simple(ids: List[int], id_to_token: Dict[int, str]) -> str:
    """简单字符级解码"""
    return ''.join(id_to_token.get(i, '') for i in ids)


def greedy_generate(model, input_ids: List[int], max_new_tokens: int = 20,
                    eos_token: int = -1) -> List[int]:
    """贪心解码生成"""
    ids = input_ids[:]
    for _ in range(max_new_tokens):
        logits = model.forward(ids)
        if not logits:
            break
        next_token = argmax(logits[-1])
        ids.append(next_token)
        if next_token == eos_token:
            break
    return ids


def count_parameters(params: Dict[str, List[List[float]]]) -> int:
    """统计参数量"""
    total = 0
    for v in params.values():
        if isinstance(v, list) and v and isinstance(v[0], list):
            total += len(v) * len(v[0])
        elif isinstance(v, list):
            total += len(v)
    return total


def save_json(obj: Any, path: str) -> None:
    """保存 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def now_str() -> str:
    """当前时间字符串"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ============================================================
# 第四部分: 简化语言模型 (作为灵元模型的接口/占位)
# ============================================================

@dataclass
class SimpleModelConfig:
    """简化语言模型配置"""
    vocab_size: int = 256
    dim: int = 64
    num_layers: int = 1
    init_std: float = 0.02


class SimpleLanguageModel:
    """
    简化语言模型 - 纯 Python 实现
    架构: Embedding → q_proj(GELU) → k_proj(GELU) → v_proj(GELU) → o_proj → lm_head
    支持: 前向传播, 反向传播, 参数管理, 保存/加载, 拷贝, 冻结
    """

    def __init__(self, config: Optional[SimpleModelConfig] = None):
        self.config = config or SimpleModelConfig()
        self.vocab_size = self.config.vocab_size
        self.dim = self.config.dim
        self.params: Dict[str, List[List[float]]] = {}
        self.grads: Dict[str, List[List[float]]] = {}
        self.cache: Dict[str, Any] = {}
        self._frozen: bool = False
        self._init_params()

    def _init_params(self):
        """初始化模型参数"""
        v, d = self.vocab_size, self.dim
        std = self.config.init_std
        self.params['embedding'] = random_gauss(v, d, std=std)
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            self.params[name] = random_gauss(d, d, std=std)
        self.params['lm_head'] = random_gauss(d, v, std=std)

    def forward(self, input_ids: List[int]) -> List[List[float]]:
        """
        前向传播
        input_ids: [seq_len] token ID 列表
        返回: logits [seq_len, vocab_size]
        """
        seq = len(input_ids)
        if seq == 0:
            return []
        d = self.dim

        # Embedding 查表
        emb = [list(self.params['embedding'][tid]) for tid in input_ids]
        self.cache['emb'] = emb
        self.cache['input_ids'] = input_ids

        # q_proj: z1 = emb @ W_q, h1 = GELU(z1)
        z1 = matmul(emb, self.params['q_proj'])
        h1 = [[gelu(v) for v in row] for row in z1]
        self.cache['z1'] = z1
        self.cache['h1'] = h1

        # k_proj: z2 = h1 @ W_k, h2 = GELU(z2)
        z2 = matmul(h1, self.params['k_proj'])
        h2 = [[gelu(v) for v in row] for row in z2]
        self.cache['z2'] = z2
        self.cache['h2'] = h2

        # v_proj: z3 = h2 @ W_v, h3 = GELU(z3)
        z3 = matmul(h2, self.params['v_proj'])
        h3 = [[gelu(v) for v in row] for row in z3]
        self.cache['z3'] = z3
        self.cache['h3'] = h3

        # o_proj: h4 = h3 @ W_o (无激活)
        h4 = matmul(h3, self.params['o_proj'])
        self.cache['h4'] = h4

        # lm_head: logits = h4 @ W_lm
        logits = matmul(h4, self.params['lm_head'])
        return logits

    def backward(self, grad_logits: List[List[float]]) -> None:
        """
        反向传播, 填充 self.grads
        grad_logits: [seq, vocab] 损失对 logits 的梯度
        """
        seq = len(grad_logits)
        if seq == 0:
            return
        d = self.dim

        # lm_head: logits = h4 @ W_lm
        h4 = self.cache['h4']
        dW_lm = matmul(transpose(h4), grad_logits)
        dh4 = matmul(grad_logits, transpose(self.params['lm_head']))
        self.grads['lm_head'] = dW_lm

        # o_proj: h4 = h3 @ W_o
        h3 = self.cache['h3']
        dW_o = matmul(transpose(h3), dh4)
        dh3 = matmul(dh4, transpose(self.params['o_proj']))
        self.grads['o_proj'] = dW_o

        # v_proj: h3 = GELU(z3), z3 = h2 @ W_v
        z3 = self.cache['z3']
        dz3 = [[dh3[i][j] * gelu_grad(z3[i][j]) for j in range(d)] for i in range(seq)]
        h2 = self.cache['h2']
        dW_v = matmul(transpose(h2), dz3)
        dh2 = matmul(dz3, transpose(self.params['v_proj']))
        self.grads['v_proj'] = dW_v

        # k_proj: h2 = GELU(z2), z2 = h1 @ W_k
        z2 = self.cache['z2']
        dz2 = [[dh2[i][j] * gelu_grad(z2[i][j]) for j in range(d)] for i in range(seq)]
        h1 = self.cache['h1']
        dW_k = matmul(transpose(h1), dz2)
        dh1 = matmul(dz2, transpose(self.params['k_proj']))
        self.grads['k_proj'] = dW_k

        # q_proj: h1 = GELU(z1), z1 = emb @ W_q
        z1 = self.cache['z1']
        dz1 = [[dh1[i][j] * gelu_grad(z1[i][j]) for j in range(d)] for i in range(seq)]
        emb = self.cache['emb']
        dW_q = matmul(transpose(emb), dz1)
        d_emb = matmul(dz1, transpose(self.params['q_proj']))
        self.grads['q_proj'] = dW_q

        # embedding: 散射加梯度
        input_ids = self.cache['input_ids']
        dE = zeros(self.vocab_size, self.dim)
        for i in range(seq):
            tid = input_ids[i]
            for j in range(d):
                dE[tid][j] += d_emb[i][j]
        self.grads['embedding'] = dE

    def zero_grad(self) -> None:
        """清零梯度"""
        self.grads = {}

    def apply_gradients(self, lr: float, weight_decay: float = 0.0) -> None:
        """应用梯度更新参数"""
        if self._frozen:
            return
        for name in self.params:
            if name not in self.grads:
                continue
            grad = self.grads[name]
            if weight_decay > 0:
                grad = matadd(grad, scalar_mul(self.params[name], weight_decay))
            mat_sub_inplace(self.params[name], scalar_mul(grad, lr))

    def parameters(self) -> Dict[str, List[List[float]]]:
        """返回参数字典"""
        return self.params

    def get_param(self, name: str) -> Optional[List[List[float]]]:
        """获取指定参数"""
        return self.params.get(name)

    def set_param(self, name: str, value: List[List[float]]) -> None:
        """设置指定参数"""
        self.params[name] = value

    def num_parameters(self) -> int:
        """总参数量"""
        return count_parameters(self.params)

    def num_trainable_parameters(self) -> int:
        """可训练参数量"""
        return 0 if self._frozen else self.num_parameters()

    def freeze(self) -> None:
        """冻结模型"""
        self._frozen = True

    def unfreeze(self) -> None:
        """解冻模型"""
        self._frozen = False

    def copy(self) -> 'SimpleLanguageModel':
        """深拷贝模型"""
        new_model = SimpleLanguageModel(self.config)
        for name, param in self.params.items():
            new_model.params[name] = deep_copy_matrix(param)
        new_model._frozen = self._frozen
        return new_model

    def state_dict(self) -> Dict[str, Any]:
        """获取模型状态字典 (可序列化)"""
        return {
            'config': asdict(self.config),
            'params': self.params,
            'frozen': self._frozen,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从状态字典加载模型"""
        cfg = state.get('config', {})
        self.config = SimpleModelConfig(**cfg)
        self.vocab_size = self.config.vocab_size
        self.dim = self.config.dim
        self.params = state.get('params', {})
        self._frozen = state.get('frozen', False)

    def save(self, path: str) -> None:
        """保存模型到文件"""
        save_json(self.state_dict(), path)

    def load(self, path: str) -> None:
        """从文件加载模型"""
        self.load_state_dict(load_json(path))


# ============================================================
# 第五部分: LoRATuner — LoRA/QLoRA 低秩微调 (#30)
# ============================================================

@dataclass
class LoRAConfig:
    """LoRA 微调配置"""
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    use_qlora: bool = False
    qlora_bits: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    init_std: float = 0.02

    @property
    def scaling(self) -> float:
        """LoRA 缩放因子: alpha / rank"""
        return self.alpha / self.rank if self.rank > 0 else 0.0


@dataclass
class LoRAAdapter:
    """单个 LoRA 适配器"""
    name: str
    rank: int
    alpha: int
    target_modules: List[str]
    A: Dict[str, List[List[float]]] = field(default_factory=dict)  # [dim, rank]
    B: Dict[str, List[List[float]]] = field(default_factory=dict)  # [rank, dim]
    grads_A: Dict[str, List[List[float]]] = field(default_factory=dict)
    grads_B: Dict[str, List[List[float]]] = field(default_factory=dict)
    enabled: bool = True
    merged: bool = False
    num_steps: int = 0
    total_loss: float = 0.0

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank if self.rank > 0 else 0.0

    def num_parameters(self) -> int:
        """可训练参数量"""
        total = 0
        for name in self.target_modules:
            if name in self.A:
                total += len(self.A[name]) * len(self.A[name][0])
            if name in self.B:
                total += len(self.B[name]) * len(self.B[name][0])
        return total


class LoRATuner:
    """
    LoRA/QLoRA 微调器
    - 低秩矩阵 A(rank×dim) 和 B(dim×rank), B=0, A=高斯
    - 前向传播: h = Wx + (alpha/rank) * B(Ax)
    - QLoRA: 先 4bit 量化基座, 再加 LoRA 适配器
    - 多适配器: 支持加载多个 LoRA 适配器, 可切换/融合
    """

    def __init__(self, config: Optional[LoRAConfig] = None, model: Optional[Any] = None):
        self.config = config or LoRAConfig()
        self.model = model
        self.adapters: Dict[str, LoRAAdapter] = {}
        self.active_adapter_name: Optional[str] = None
        self._quantized: bool = False
        self._quant_params: Dict[str, Any] = {}  # 量化参数存储
        self._original_params: Dict[str, List[List[float]]] = {}  # 量化前的原始权重

        # 如果提供了模型, 初始化默认适配器
        if self.model is not None:
            self._init_adapter('default')
            self.active_adapter_name = 'default'
            if self.config.use_qlora:
                self.apply_qlora()

    def _get_model_dim(self) -> int:
        """获取模型维度"""
        if self.model is not None and hasattr(self.model, 'dim'):
            return self.model.dim
        return 64

    def _init_adapter(self, name: str) -> LoRAAdapter:
        """初始化一个 LoRA 适配器"""
        d = self._get_model_dim()
        r = self.config.rank
        adapter = LoRAAdapter(
            name=name,
            rank=r,
            alpha=self.config.alpha,
            target_modules=list(self.config.target_modules),
        )
        for module_name in self.config.target_modules:
            # A 矩阵: [dim, rank], 高斯初始化
            adapter.A[module_name] = random_gauss(d, r, std=self.config.init_std)
            # B 矩阵: [rank, dim], 零初始化
            adapter.B[module_name] = zeros(r, d)
        self.adapters[name] = adapter
        return adapter

    def add_adapter(self, name: str, config: Optional[LoRAConfig] = None) -> LoRAAdapter:
        """添加一个新的 LoRA 适配器"""
        if name in self.adapters:
            raise ValueError(f"适配器 '{name}' 已存在")
        old_config = self.config
        if config is not None:
            self.config = config
        adapter = self._init_adapter(name)
        self.config = old_config
        return adapter

    def switch_adapter(self, name: str) -> None:
        """切换当前激活的适配器"""
        if name not in self.adapters:
            raise ValueError(f"适配器 '{name}' 不存在")
        self.active_adapter_name = name

    def get_active_adapter(self) -> Optional[LoRAAdapter]:
        """获取当前激活的适配器"""
        if self.active_adapter_name is None:
            return None
        return self.adapters.get(self.active_adapter_name)

    def _merge_lora_into_weights(self, adapter: LoRAAdapter) -> None:
        """将 LoRA 权重合并到模型基座权重中 (临时)"""
        if self.model is None:
            return
        scaling = adapter.scaling
        for module_name in adapter.target_modules:
            W = self.model.params.get(module_name)
            if W is None:
                continue
            A = adapter.A[module_name]
            B = adapter.B[module_name]
            # delta = scaling * A @ B, 其中 A[dim,rank], B[rank,dim] → delta[dim,dim]
            delta = scalar_mul(matmul(A, B), scaling)
            mat_add_inplace(W, delta)

    def _unmerge_lora_from_weights(self, adapter: LoRAAdapter) -> None:
        """从模型基座权重中移除 LoRA 权重 (恢复)"""
        if self.model is None:
            return
        scaling = adapter.scaling
        for module_name in adapter.target_modules:
            W = self.model.params.get(module_name)
            if W is None:
                continue
            A = adapter.A[module_name]
            B = adapter.B[module_name]
            delta = scalar_mul(matmul(A, B), scaling)
            mat_sub_inplace(W, delta)

    def forward_with_lora(self, input_ids: List[int]) -> List[List[float]]:
        """带 LoRA 的前向传播"""
        adapter = self.get_active_adapter()
        if adapter is None or not adapter.enabled or self.model is None:
            return self.model.forward(input_ids) if self.model else []
        # 临时合并 LoRA, 前向传播, 再恢复
        self._merge_lora_into_weights(adapter)
        logits = self.model.forward(input_ids)
        self._unmerge_lora_from_weights(adapter)
        return logits

    def train_step(self, input_ids: List[int], targets: List[int]) -> float:
        """
        LoRA 训练步
        只更新 LoRA 参数 (A 和 B), 基座权重冻结
        返回: loss
        """
        if self.model is None:
            return 0.0
        adapter = self.get_active_adapter()
        if adapter is None:
            return 0.0

        # 1. 合并 LoRA 到权重
        self._merge_lora_into_weights(adapter)

        # 2. 前向传播
        logits = self.model.forward(input_ids)

        # 3. 计算损失
        loss = sequence_cross_entropy(logits, targets)

        # 4. 计算梯度
        grad_logits = sequence_ce_grad(logits, targets)

        # 5. 反向传播 (计算模型梯度, 即 dL/dW_eff)
        self.model.backward(grad_logits)

        # 6. 提取 LoRA 梯度
        # 由于 W_eff = W + scaling * A @ B, 且 dL/dW_eff = dL/d(delta) = x^T @ g
        # dL/dA = scaling * (dL/dW_eff) @ B^T
        # dL/dB = scaling * A^T @ (dL/dW_eff)
        scaling = adapter.scaling
        for module_name in adapter.target_modules:
            grad_W = self.model.grads.get(module_name)
            if grad_W is None:
                continue
            A = adapter.A[module_name]
            B = adapter.B[module_name]
            # grad_A = scaling * grad_W @ B^T  → [dim,dim] @ [dim,rank] = [dim,rank]
            adapter.grads_A[module_name] = scalar_mul(matmul(grad_W, transpose(B)), scaling)
            # grad_B = scaling * A^T @ grad_W  → [rank,dim] @ [dim,dim] = [rank,dim]
            adapter.grads_B[module_name] = scalar_mul(matmul(transpose(A), grad_W), scaling)

        # 7. 恢复原始权重
        self._unmerge_lora_from_weights(adapter)

        # 8. 更新 LoRA 参数
        lr = self.config.learning_rate
        wd = self.config.weight_decay
        for module_name in adapter.target_modules:
            if module_name in adapter.grads_A:
                grad_a = adapter.grads_A[module_name]
                if wd > 0:
                    grad_a = matadd(grad_a, scalar_mul(adapter.A[module_name], wd))
                mat_sub_inplace(adapter.A[module_name], scalar_mul(grad_a, lr))
            if module_name in adapter.grads_B:
                grad_b = adapter.grads_B[module_name]
                if wd > 0:
                    grad_b = matadd(grad_b, scalar_mul(adapter.B[module_name], wd))
                mat_sub_inplace(adapter.B[module_name], scalar_mul(grad_b, lr))

        # 9. 记录统计
        adapter.num_steps += 1
        adapter.total_loss += loss

        return loss

    # ---------- QLoRA 4-bit 量化 ----------

    def quantize_4bit(self, W: List[List[float]]) -> Tuple[List[List[float]], float]:
        """
        4-bit 量化
        返回: (量化值矩阵 Q, 缩放因子 scale)
        """
        if not W or not W[0]:
            return W, 1.0
        max_abs = max(abs(v) for row in W for v in row)
        scale = max_abs / 7.0 if max_abs > 0 else 1.0
        Q = [[round(W[i][j] / scale) for j in range(len(W[0]))] for i in range(len(W))]
        # 裁剪到 4-bit 范围 [-8, 7]
        Q = [[max(-8.0, min(7.0, q)) for q in row] for row in Q]
        return Q, scale

    def dequantize_4bit(self, Q: List[List[float]], scale: float) -> List[List[float]]:
        """4-bit 反量化"""
        return [[q * scale for q in row] for row in Q]

    def apply_qlora(self) -> None:
        """对基座模型进行 4-bit 量化 (QLoRA)"""
        if self.model is None or self._quantized:
            return
        self._original_params = {}
        for name, W in self.model.params.items():
            self._original_params[name] = deep_copy_matrix(W)
            Q, scale = self.quantize_4bit(W)
            self._quant_params[name] = {'Q': Q, 'scale': scale}
            # 用反量化后的权重替换 (模拟量化误差)
            self.model.params[name] = self.dequantize_4bit(Q, scale)
        self._quantized = True

    def restore_quantized(self) -> None:
        """恢复量化前的原始权重"""
        if not self._quantized:
            return
        for name, W in self._original_params.items():
            self.model.params[name] = deep_copy_matrix(W)
        self._quantized = False
        self._quant_params = {}

    # ---------- 适配器管理 ----------

    def save_adapter(self, path: str, name: Optional[str] = None) -> None:
        """保存 LoRA 适配器到文件"""
        adapter_name = name or self.active_adapter_name
        if adapter_name is None or adapter_name not in self.adapters:
            raise ValueError("未指定有效的适配器")
        adapter = self.adapters[adapter_name]
        data = {
            'name': adapter.name,
            'rank': adapter.rank,
            'alpha': adapter.alpha,
            'target_modules': adapter.target_modules,
            'A': adapter.A,
            'B': adapter.B,
            'num_steps': adapter.num_steps,
            'total_loss': adapter.total_loss,
        }
        save_json(data, path)

    def load_adapter(self, path: str, name: Optional[str] = None) -> LoRAAdapter:
        """从文件加载 LoRA 适配器"""
        data = load_json(path)
        adapter_name = name or data.get('name', f'adapter_{len(self.adapters)}')
        adapter = LoRAAdapter(
            name=adapter_name,
            rank=data['rank'],
            alpha=data['alpha'],
            target_modules=data['target_modules'],
            A=data['A'],
            B=data['B'],
            num_steps=data.get('num_steps', 0),
            total_loss=data.get('total_loss', 0.0),
        )
        self.adapters[adapter_name] = adapter
        return adapter

    def merge_adapter(self, name: Optional[str] = None) -> None:
        """将 LoRA 适配器永久合并到基座权重"""
        adapter_name = name or self.active_adapter_name
        if adapter_name is None or adapter_name not in self.adapters:
            raise ValueError("未指定有效的适配器")
        adapter = self.adapters[adapter_name]
        if self.model is None:
            return
        scaling = adapter.scaling
        for module_name in adapter.target_modules:
            W = self.model.params.get(module_name)
            if W is None:
                continue
            A = adapter.A[module_name]
            B = adapter.B[module_name]
            delta = scalar_mul(matmul(A, B), scaling)
            mat_add_inplace(W, delta)
        adapter.merged = True

    def merge_adapters(self, names: List[str], weights: Optional[List[float]] = None) -> None:
        """融合多个 LoRA 适配器 (加权平均后合并)"""
        if weights is None:
            weights = [1.0 / len(names)] * len(names)
        if self.model is None:
            return
        for module_name in self.config.target_modules:
            W = self.model.params.get(module_name)
            if W is None:
                continue
            for adapter_name, w in zip(names, weights):
                adapter = self.adapters.get(adapter_name)
                if adapter is None:
                    continue
                A = adapter.A.get(module_name)
                B = adapter.B.get(module_name)
                if A is None or B is None:
                    continue
                scaling = adapter.scaling
                delta = scalar_mul(matmul(A, B), scaling * w)
                mat_add_inplace(W, delta)

    def list_adapters(self) -> List[str]:
        """列出所有适配器名称"""
        return list(self.adapters.keys())

    def remove_adapter(self, name: str) -> None:
        """移除适配器"""
        if name in self.adapters:
            del self.adapters[name]
        if self.active_adapter_name == name:
            self.active_adapter_name = None

    # ---------- 参数统计 ----------

    def get_trainable_param_count(self) -> int:
        """可训练参数量 (仅 LoRA 参数)"""
        total = 0
        for adapter in self.adapters.values():
            if adapter.enabled:
                total += adapter.num_parameters()
        return total

    def get_total_param_count(self) -> int:
        """总参数量 (基座 + LoRA)"""
        base = self.model.num_parameters() if self.model is not None else 0
        return base + self.get_trainable_param_count()

    def get_trainable_ratio(self) -> float:
        """可训练参数比例"""
        total = self.get_total_param_count()
        return self.get_trainable_param_count() / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        adapter = self.get_active_adapter()
        return {
            'num_adapters': len(self.adapters),
            'active_adapter': self.active_adapter_name,
            'adapter_names': list(self.adapters.keys()),
            'rank': self.config.rank,
            'alpha': self.config.alpha,
            'scaling': self.config.scaling,
            'target_modules': self.config.target_modules,
            'use_qlora': self.config.use_qlora,
            'is_quantized': self._quantized,
            'trainable_params': self.get_trainable_param_count(),
            'total_params': self.get_total_param_count(),
            'trainable_ratio': round(self.get_trainable_ratio(), 6),
            'active_steps': adapter.num_steps if adapter else 0,
            'active_avg_loss': (adapter.total_loss / adapter.num_steps) if adapter and adapter.num_steps > 0 else 0.0,
        }

    def get_dashboard(self) -> str:
        """获取仪表盘字符串"""
        s = self.get_stats()
        lines = [
            "========== LoRATuner 仪表盘 ==========",
            f"  适配器数量:       {s['num_adapters']}",
            f"  当前适配器:       {s['active_adapter']}",
            f"  适配器列表:       {s['adapter_names']}",
            f"  LoRA rank:        {s['rank']}",
            f"  LoRA alpha:       {s['alpha']}",
            f"  缩放因子:         {s['scaling']:.4f}",
            f"  目标模块:         {s['target_modules']}",
            f"  QLoRA 启用:       {s['use_qlora']}",
            f"  基座已量化:       {s['is_quantized']}",
            f"  可训练参数:       {s['trainable_params']:,}",
            f"  总参数:           {s['total_params']:,}",
            f"  可训练比例:       {s['trainable_ratio']:.4%}",
            f"  训练步数:         {s['active_steps']}",
            f"  平均损失:         {s['active_avg_loss']:.6f}",
            "======================================",
        ]
        return '\n'.join(lines)


# ============================================================
# 第六部分: FullFineTuner — 全参数微调 (#31)
# ============================================================

@dataclass
class FullFTConfig:
    """全参数微调配置"""
    learning_rate: float = 1e-3
    epochs: int = 3
    batch_size: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    patience: int = 3
    min_delta: float = 1e-4
    scheduler: str = 'warmup_cosine'  # cosine, linear, warmup_cosine
    lr_min: float = 1e-6
    save_dir: str = field(default_factory=lambda: os.path.join(_DATA_DIR, 'full_ft'))
    log_interval: int = 10
    seed: int = 42


@dataclass
class TrainingLog:
    """训练日志"""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    epochs: List[Dict[str, Any]] = field(default_factory=list)
    best_eval_loss: float = float('inf')
    best_step: int = 0
    total_steps: int = 0
    start_time: str = ''
    end_time: str = ''
    early_stopped: bool = False

    def add_step(self, step: int, loss: float, lr: float, grad_norm: float):
        self.steps.append({
            'step': step, 'loss': loss, 'lr': lr, 'grad_norm': grad_norm
        })
        self.total_steps = step

    def add_epoch(self, epoch: int, train_loss: float, eval_loss: float, eval_acc: float):
        self.epochs.append({
            'epoch': epoch, 'train_loss': train_loss,
            'eval_loss': eval_loss, 'eval_acc': eval_acc
        })


class FullFineTuner:
    """
    全参数微调器
    - 支持: 学习率调度 (cosine/linear/warmup_cosine), 梯度累积, 梯度裁剪
    - 早停, 最佳模型保存, 从 checkpoint 恢复
    - 训练日志: 每步 loss/lr/grad_norm
    """

    def __init__(self, config: Optional[FullFTConfig] = None):
        self.config = config or FullFTConfig()
        self.log = TrainingLog()
        self._best_model_state: Optional[Dict[str, Any]] = None
        self._no_improve_count: int = 0
        random.seed(self.config.seed)

    def get_lr(self, step: int, total_steps: int) -> float:
        """根据调度器计算当前学习率"""
        lr_max = self.config.learning_rate
        lr_min = self.config.lr_min
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        if self.config.scheduler == 'linear':
            if step < warmup_steps:
                return lr_max * step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return lr_max - (lr_max - lr_min) * progress

        elif self.config.scheduler == 'cosine':
            if step < warmup_steps:
                return lr_max * step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

        else:  # warmup_cosine (默认)
            if step < warmup_steps:
                return lr_max * step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

    def clip_gradients(self, model) -> float:
        """梯度裁剪, 返回裁剪前的全局梯度范数"""
        total_norm = 0.0
        for name, grad in model.grads.items():
            total_norm += frobenius_norm(grad) ** 2
        total_norm = math.sqrt(total_norm)
        if total_norm > self.config.max_grad_norm and total_norm > 0:
            scale = self.config.max_grad_norm / total_norm
            for name in model.grads:
                mat_mul_scalar_inplace(model.grads[name], scale)
        return total_norm

    def train_step(self, model, input_ids: List[int], targets: List[int]) -> Tuple[float, float]:
        """
        单步训练
        返回: (loss, grad_norm)
        """
        # 前向传播
        logits = model.forward(input_ids)
        # 计算损失
        loss = sequence_cross_entropy(logits, targets)
        # 计算梯度
        grad_logits = sequence_ce_grad(logits, targets)
        model.zero_grad()
        model.backward(grad_logits)
        # 梯度裁剪
        grad_norm = self.clip_gradients(model)
        return loss, grad_norm

    def train(self, model, dataset: List[Dict[str, Any]],
              eval_dataset: Optional[List[Dict[str, Any]]] = None) -> TrainingLog:
        """
        完整训练循环
        dataset: [{'input_ids': [...], 'targets': [...]}, ...]
        返回: TrainingLog
        """
        self.log = TrainingLog()
        self.log.start_time = now_str()
        self._no_improve_count = 0
        self._best_model_state = None

        total_steps = len(dataset) * self.config.epochs // max(self.config.batch_size, 1)
        total_steps = max(total_steps, 1)
        global_step = 0

        for epoch in range(self.config.epochs):
            # 打乱数据
            shuffled = dataset[:]
            random.shuffle(shuffled)

            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            accum_count = 0

            for i, example in enumerate(shuffled):
                input_ids = example['input_ids']
                targets = example.get('targets', input_ids[1:] + [0])
                # 对齐长度
                min_len = min(len(input_ids), len(targets))
                if min_len == 0:
                    continue
                input_ids = input_ids[:min_len]
                targets = targets[:min_len]

                loss, grad_norm = self.train_step(model, input_ids, targets)

                # 梯度累积
                accum_count += 1
                epoch_loss_sum += loss
                epoch_loss_count += 1

                if accum_count >= self.config.gradient_accumulation:
                    lr = self.get_lr(global_step, total_steps)
                    model.apply_gradients(lr, self.config.weight_decay)
                    model.zero_grad()
                    global_step += 1
                    accum_count = 0

                    # 记录日志
                    if global_step % self.config.log_interval == 0:
                        self.log.add_step(global_step, loss, lr, grad_norm)

            # 每个 epoch 结束后评估
            train_avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
            eval_loss, eval_acc = 0.0, 0.0
            if eval_dataset:
                metrics = self.evaluate(model, eval_dataset)
                eval_loss = metrics['loss']
                eval_acc = metrics['accuracy']

            self.log.add_epoch(epoch + 1, train_avg_loss, eval_loss, eval_acc)

            # 早停检查
            if eval_dataset and eval_loss < self.log.best_eval_loss - self.config.min_delta:
                self.log.best_eval_loss = eval_loss
                self.log.best_step = global_step
                self._best_model_state = model.state_dict()
                self._no_improve_count = 0
            elif eval_dataset:
                self._no_improve_count += 1
                if self._no_improve_count >= self.config.patience:
                    self.log.early_stopped = True
                    break

        # 恢复最佳模型
        if self._best_model_state is not None:
            model.load_state_dict(self._best_model_state)

        self.log.end_time = now_str()
        return self.log

    def evaluate(self, model, eval_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估循环
        返回: {'loss': float, 'accuracy': float, 'perplexity': float}
        """
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        count = 0

        for example in eval_dataset:
            input_ids = example['input_ids']
            targets = example.get('targets', input_ids[1:] + [0])
            min_len = min(len(input_ids), len(targets))
            if min_len == 0:
                continue
            input_ids = input_ids[:min_len]
            targets = targets[:min_len]

            logits = model.forward(input_ids)
            loss = sequence_cross_entropy(logits, targets)
            total_loss += loss
            count += 1

            # 准确率
            for t in range(min_len):
                pred = argmax(logits[t])
                if pred == targets[t]:
                    total_correct += 1
                total_tokens += 1

        avg_loss = total_loss / max(count, 1)
        accuracy = total_correct / max(total_tokens, 1)
        perplexity = safe_exp(avg_loss)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
        }

    def save_checkpoint(self, model, path: str) -> None:
        """保存 checkpoint"""
        checkpoint = {
            'model_state': model.state_dict(),
            'config': asdict(self.config),
            'log': {
                'total_steps': self.log.total_steps,
                'best_eval_loss': self.log.best_eval_loss,
                'best_step': self.log.best_step,
                'epochs': self.log.epochs,
            },
            'timestamp': now_str(),
        }
        save_json(checkpoint, path)

    def load_checkpoint(self, model, path: str) -> None:
        """从 checkpoint 恢复"""
        checkpoint = load_json(path)
        model.load_state_dict(checkpoint['model_state'])
        log_data = checkpoint.get('log', {})
        self.log.best_eval_loss = log_data.get('best_eval_loss', float('inf'))
        self.log.best_step = log_data.get('best_step', 0)
        self.log.total_steps = log_data.get('total_steps', 0)
        self.log.epochs = log_data.get('epochs', [])

    def save_best_model(self, model, path: str) -> None:
        """保存最佳模型"""
        model.save(path)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'config': asdict(self.config),
            'total_steps': self.log.total_steps,
            'best_eval_loss': self.log.best_eval_loss if self.log.best_eval_loss != float('inf') else None,
            'best_step': self.log.best_step,
            'early_stopped': self.log.early_stopped,
            'num_epochs': len(self.log.epochs),
            'start_time': self.log.start_time,
            'end_time': self.log.end_time,
            'epoch_summaries': self.log.epochs,
            'recent_steps': self.log.steps[-10:] if self.log.steps else [],
        }

    def get_dashboard(self) -> str:
        """获取仪表盘字符串"""
        s = self.get_stats()
        lines = [
            "========== FullFineTuner 仪表盘 ==========",
            f"  学习率:           {s['config']['learning_rate']}",
            f"  调度器:           {s['config']['scheduler']}",
            f"  训练轮数:         {s['config']['epochs']}",
            f"  批大小:           {s['config']['batch_size']}",
            f"  梯度累积:         {s['config']['gradient_accumulation']}",
            f"  梯度裁剪:         {s['config']['max_grad_norm']}",
            f"  早停耐心:         {s['config']['patience']}",
            f"  总步数:           {s['total_steps']}",
            f"  最佳验证损失:     {s['best_eval_loss']}",
            f"  最佳步数:         {s['best_step']}",
            f"  是否早停:         {s['early_stopped']}",
            f"  实际轮数:         {s['num_epochs']}",
            f"  开始时间:         {s['start_time']}",
            f"  结束时间:         {s['end_time']}",
        ]
        for ep in s['epoch_summaries']:
            lines.append(f"  Epoch {ep['epoch']}: train_loss={ep['train_loss']:.6f}, "
                         f"eval_loss={ep['eval_loss']:.6f}, eval_acc={ep['eval_acc']:.4f}")
        lines.append("==========================================")
        return '\n'.join(lines)


# ============================================================
# 第七部分: SFTTrainer — 指令微调 (#32)
# ============================================================

@dataclass
class SFTConfig:
    """SFT 指令微调配置"""
    learning_rate: float = 5e-4
    epochs: int = 3
    batch_size: int = 1
    max_length: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation: int = 1
    # 数据增强
    shuffle: bool = True
    repeat: int = 1
    truncate: bool = True
    # 打包
    packing: bool = False
    # 模板
    user_tag: str = '<|user|>'
    assistant_tag: str = '<|assistant|>'
    end_tag: str = '<|end|>'
    # 评估
    max_gen_tokens: int = 32
    save_dir: str = field(default_factory=lambda: os.path.join(_DATA_DIR, 'sft'))
    seed: int = 42


class SFTTrainer:
    """
    指令微调 (Supervised Fine-Tuning) 训练器
    - 数据格式: {instruction, input, output} 或 {messages: [{role, content}]}
    - 使用 ChatTemplate 格式化训练数据
    - 只对 output 部分计算 loss (instruction 部分 mask 掉)
    - 支持多轮对话, 数据增强, 序列打包
    - 评估: 简化版 ROUGE / BLEU
    """

    def __init__(self, config: Optional[SFTConfig] = None,
                 tokenizer: Optional[Any] = None):
        self.config = config or SFTConfig()
        self.tokenizer = tokenizer
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._step_count: int = 0
        self._total_loss: float = 0.0
        self._training_log: List[Dict[str, Any]] = []
        random.seed(self.config.seed)

    def _tokenize(self, text: str) -> List[int]:
        """分词 (字符级)"""
        ids = []
        for ch in text:
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.id_to_token[idx] = ch
            ids.append(self.vocab[ch])
        return ids

    def _detokenize(self, ids: List[int]) -> str:
        """解码"""
        return ''.join(self.id_to_token.get(i, '') for i in ids)

    def format_instruction(self, example: Dict[str, Any]) -> str:
        """
        格式化指令数据
        支持格式: {instruction, input, output} 或 {messages: [...]}
        """
        if 'messages' in example:
            return self.apply_chat_template(example['messages'])
        instruction = example.get('instruction', '')
        inp = example.get('input', '')
        output = example.get('output', '')
        text = f"{self.config.user_tag}{instruction}"
        if inp:
            text += f"\n{inp}"
        text += f"{self.config.assistant_tag}{output}{self.config.end_tag}"
        return text

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """应用聊天模板格式化多轮对话"""
        text = ''
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                text += f"{self.config.user_tag}{content}"
            elif role == 'assistant':
                text += f"{self.config.assistant_tag}{content}{self.config.end_tag}"
            else:
                text += f"<|{role}|>{content}"
        return text

    def tokenize_with_mask(self, example: Dict[str, Any]) -> Tuple[List[int], List[int], List[int]]:
        """
        分词并生成 loss mask
        返回: (input_ids, targets, mask)
        mask[t]=1 表示该位置计算 loss (output 部分)
        """
        if 'messages' in example:
            return self._tokenize_multi_turn(example['messages'])

        instruction = example.get('instruction', '')
        inp = example.get('input', '')
        output = example.get('output', '')

        # 构建指令部分
        inst_text = f"{self.config.user_tag}{instruction}"
        if inp:
            inst_text += f"\n{inp}"
        inst_text += f"{self.config.assistant_tag}"
        # 输出部分
        out_text = f"{output}{self.config.end_tag}"

        inst_ids = self._tokenize(inst_text)
        out_ids = self._tokenize(out_text)

        input_ids = inst_ids + out_ids
        # targets: 右移一位
        targets = input_ids[1:] + [0]
        # mask: 只对 output 部分为 1
        mask = [0] * len(inst_ids) + [1] * len(out_ids)
        # 对齐 targets
        mask = mask[1:] + [0]

        # 截断
        if self.config.truncate and len(input_ids) > self.config.max_length:
            input_ids = input_ids[:self.config.max_length]
            targets = targets[:self.config.max_length]
            mask = mask[:self.config.max_length]

        return input_ids, targets, mask

    def _tokenize_multi_turn(self, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int], List[int]]:
        """处理多轮对话的 SFT 分词"""
        input_ids = []
        mask = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                prefix = f"{self.config.user_tag}{content}{self.config.assistant_tag}"
                prefix_ids = self._tokenize(prefix)
                input_ids.extend(prefix_ids)
                mask.extend([0] * len(prefix_ids))
            elif role == 'assistant':
                content_ids = self._tokenize(content)
                end_ids = self._tokenize(self.config.end_tag)
                full_ids = content_ids + end_ids
                input_ids.extend(full_ids)
                mask.extend([1] * len(full_ids))

        targets = input_ids[1:] + [0]
        mask = mask[1:] + [0]

        if self.config.truncate and len(input_ids) > self.config.max_length:
            input_ids = input_ids[:self.config.max_length]
            targets = targets[:self.config.max_length]
            mask = mask[:self.config.max_length]

        return input_ids, targets, mask

    def pack_sequences(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将多条短指令打包到一个序列, 提高利用率
        """
        if not self.config.packing:
            return [{'input_ids': e[0], 'targets': e[1], 'mask': e[2]}
                    for e in [self.tokenize_with_mask(ex) for ex in examples]]

        packed = []
        current_ids = []
        current_targets = []
        current_mask = []

        for ex in examples:
            ids, targets, mask = self.tokenize_with_mask(ex)
            if len(current_ids) + len(ids) > self.config.max_length:
                if current_ids:
                    packed.append({
                        'input_ids': current_ids,
                        'targets': current_targets,
                        'mask': current_mask,
                    })
                current_ids = ids[:self.config.max_length]
                current_targets = targets[:self.config.max_length]
                current_mask = mask[:self.config.max_length]
            else:
                current_ids.extend(ids)
                current_targets.extend(targets)
                current_mask.extend(mask)

        if current_ids:
            packed.append({
                'input_ids': current_ids,
                'targets': current_targets,
                'mask': current_mask,
            })

        return packed

    def augment_data(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """数据增强: 随机打乱/重复/截断"""
        augmented = []
        for _ in range(self.config.repeat):
            batch = examples[:]
            if self.config.shuffle:
                random.shuffle(batch)
            augmented.extend(batch)
        return augmented

    def compute_loss(self, model, input_ids: List[int], targets: List[int],
                     mask: List[int]) -> Tuple[float, List[List[float]]]:
        """
        计算 SFT 损失 (只对 mask=1 的位置计算)
        返回: (loss, grad_logits)
        """
        logits = model.forward(input_ids)
        loss = sequence_cross_entropy(logits, targets, mask)
        grad_logits = sequence_ce_grad(logits, targets, mask)
        return loss, grad_logits

    def train_sft(self, model, sft_dataset: List[Dict[str, Any]],
                  eval_dataset: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        SFT 训练
        返回: {'steps': int, 'avg_loss': float, 'eval_metrics': dict, 'log': list}
        """
        self._step_count = 0
        self._total_loss = 0.0
        self._training_log = []

        # 数据增强
        augmented = self.augment_data(sft_dataset)

        # 分词 + 打包
        packed = self.pack_sequences(augmented)

        total_steps = len(packed) * self.config.epochs
        global_step = 0

        for epoch in range(self.config.epochs):
            epoch_data = packed[:]
            if self.config.shuffle:
                random.shuffle(epoch_data)

            epoch_loss = 0.0
            epoch_count = 0

            for example in epoch_data:
                input_ids = example['input_ids']
                targets = example['targets']
                mask = example['mask']
                if len(input_ids) < 2:
                    continue

                loss, grad_logits = self.compute_loss(model, input_ids, targets, mask)
                model.zero_grad()
                model.backward(grad_logits)

                # 梯度裁剪
                total_norm = 0.0
                for g in model.grads.values():
                    total_norm += frobenius_norm(g) ** 2
                total_norm = math.sqrt(total_norm)
                if total_norm > self.config.max_grad_norm and total_norm > 0:
                    scale = self.config.max_grad_norm / total_norm
                    for g in model.grads.values():
                        mat_mul_scalar_inplace(g, scale)

                # 学习率
                lr = self._get_lr_sft(global_step, total_steps)
                model.apply_gradients(lr, self.config.weight_decay)
                model.zero_grad()

                global_step += 1
                epoch_loss += loss
                epoch_count += 1
                self._step_count += 1
                self._total_loss += loss

                if global_step % 10 == 0:
                    self._training_log.append({
                        'step': global_step,
                        'epoch': epoch + 1,
                        'loss': loss,
                        'lr': lr,
                        'grad_norm': total_norm,
                    })

        avg_loss = self._total_loss / max(self._step_count, 1)
        eval_metrics = {}
        if eval_dataset:
            eval_metrics = self.evaluate_sft(model, eval_dataset)

        return {
            'steps': self._step_count,
            'avg_loss': avg_loss,
            'eval_metrics': eval_metrics,
            'log': self._training_log,
        }

    def _get_lr_sft(self, step: int, total_steps: int) -> float:
        """SFT 学习率调度"""
        lr_max = self.config.learning_rate
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        if step < warmup_steps:
            return lr_max * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr_max * 0.5 * (1 + math.cos(math.pi * progress))

    def evaluate_sft(self, model, eval_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        SFT 评估: 计算损失 + 生成质量评估 (ROUGE/BLEU 简化版)
        """
        total_loss = 0.0
        count = 0
        rouge_scores = []
        bleu_scores = []

        for example in eval_dataset:
            ids, targets, mask = self.tokenize_with_mask(example)
            if len(ids) < 2:
                continue

            logits = model.forward(ids)
            loss = sequence_cross_entropy(logits, targets, mask)
            total_loss += loss
            count += 1

            # 生成评估
            instruction = example.get('instruction', '')
            reference = example.get('output', '')
            prompt_ids = self._tokenize(
                f"{self.config.user_tag}{instruction}{self.config.assistant_tag}")
            generated_ids = greedy_generate(model, prompt_ids, self.config.max_gen_tokens)
            generated_text = self._detokenize(generated_ids[len(prompt_ids):])

            rouge_scores.append(self.rouge_score(generated_text, reference))
            bleu_scores.append(self.bleu_score(generated_text, reference))

        return {
            'loss': total_loss / max(count, 1),
            'rouge_1': mean([r['f1'] for r in rouge_scores]) if rouge_scores else 0.0,
            'bleu_1': mean(bleu_scores) if bleu_scores else 0.0,
            'num_examples': count,
        }

    def rouge_score(self, generated: str, reference: str) -> Dict[str, float]:
        """简化版 ROUGE-1 评分"""
        gen_words = list(generated)
        ref_words = list(reference)
        if not gen_words or not ref_words:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        ref_counter = Counter(ref_words)
        gen_counter = Counter(gen_words)
        overlap = sum((ref_counter & gen_counter).values())

        precision = overlap / len(gen_words) if gen_words else 0.0
        recall = overlap / len(ref_words) if ref_words else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    def bleu_score(self, generated: str, reference: str) -> float:
        """简化版 BLEU-1 评分"""
        gen_words = list(generated)
        ref_words = list(reference)
        if not gen_words:
            return 0.0

        ref_counter = Counter(ref_words)
        gen_counter = Counter(gen_words)
        overlap = sum((ref_counter & gen_counter).values())

        return overlap / len(gen_words)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'config': asdict(self.config),
            'vocab_size': len(self.vocab),
            'step_count': self._step_count,
            'avg_loss': self._total_loss / max(self._step_count, 1),
            'recent_log': self._training_log[-10:],
        }

    def get_dashboard(self) -> str:
        """获取仪表盘字符串"""
        s = self.get_stats()
        lines = [
            "========== SFTTrainer 仪表盘 ==========",
            f"  学习率:           {s['config']['learning_rate']}",
            f"  训练轮数:         {s['config']['epochs']}",
            f"  最大长度:         {s['config']['max_length']}",
            f"  序列打包:         {s['config']['packing']}",
            f"  数据增强重复:     {s['config']['repeat']}",
            f"  词表大小:         {s['vocab_size']}",
            f"  训练步数:         {s['step_count']}",
            f"  平均损失:         {s['avg_loss']:.6f}",
            "=======================================",
        ]
        for entry in s['recent_log'][-5:]:
            lines.append(f"  Step {entry['step']} (Epoch {entry['epoch']}): "
                         f"loss={entry['loss']:.6f}, lr={entry['lr']:.6f}")
        return '\n'.join(lines)


# ============================================================
# 第八部分: DPOTrainer — 直接偏好优化 (#33)
# ============================================================

@dataclass
class DPOConfig:
    """DPO 配置"""
    beta: float = 0.1
    learning_rate: float = 5e-4
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    use_ipo: bool = False  # IPO 变体
    ipo_tau: float = 0.1   # IPO 参数
    max_length: int = 128
    log_interval: int = 10
    seed: int = 42


class DPOTrainer:
    """
    直接偏好优化 (Direct Preference Optimization) 训练器
    - 数据格式: {prompt, chosen, rejected}
    - DPO 损失: L = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    - 参考模型: 冻结的基座模型
    - 支持 IPO 变体
    - 评估: chosen_rate
    """

    def __init__(self, config: Optional[DPOConfig] = None):
        self.config = config or DPOConfig()
        self._step_count: int = 0
        self._total_loss: float = 0.0
        self._total_chosen_reward: float = 0.0
        self._total_rejected_reward: float = 0.0
        self._training_log: List[Dict[str, Any]] = []
        random.seed(self.config.seed)

    def _tokenize_text(self, text: str) -> List[int]:
        """简单字符级分词"""
        return [ord(ch) % 256 for ch in text]

    def compute_log_probs(self, model, prompt_ids: List[int],
                          response_ids: List[int]) -> Tuple[float, List[List[float]], List[int], List[int]]:
        """
        计算模型对 response 的对数概率
        返回: (log_prob, logits, full_ids, response_mask)
        """
        full_ids = prompt_ids + response_ids
        response_start = len(prompt_ids)
        # mask: 只对 response 部分为 1
        mask = [0] * response_start + [1] * len(response_ids)
        # targets: 右移一位
        targets = full_ids[1:] + [0]
        mask = mask[1:] + [0]

        logits = model.forward(full_ids)
        # 截断
        min_len = min(len(logits), len(targets), len(mask))
        logits = logits[:min_len]
        targets = targets[:min_len]
        mask = mask[:min_len]

        log_prob = sequence_log_probs(logits, targets, mask)
        return log_prob, logits, full_ids, mask

    def dpo_loss(self, log_ratio_chosen: float, log_ratio_rejected: float) -> float:
        """
        DPO 损失: L = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
        """
        logits = self.config.beta * (log_ratio_chosen - log_ratio_rejected)
        # -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
        return math.log(1.0 + safe_exp(-logits))

    def ipo_loss(self, log_ratio_chosen: float, log_ratio_rejected: float) -> float:
        """
        IPO 损失 (Identity Preference Optimization 简化版):
        L = (log_ratio_chosen - log_ratio_rejected - 1/(2*beta))^2
        """
        target = 1.0 / (2.0 * self.config.beta) if self.config.beta > 0 else 0.0
        diff = (log_ratio_chosen - log_ratio_rejected) - target
        return diff * diff

    def dpo_loss_grad(self, log_ratio_chosen: float, log_ratio_rejected: float) -> Tuple[float, float]:
        """
        DPO 损失对 log_ratio_chosen 和 log_ratio_rejected 的梯度
        返回: (dL/dlog_ratio_chosen, dL/dlog_ratio_rejected)
        """
        if self.config.use_ipo:
            target = 1.0 / (2.0 * self.config.beta) if self.config.beta > 0 else 0.0
            diff = (log_ratio_chosen - log_ratio_rejected) - target
            return 2.0 * diff, -2.0 * diff
        else:
            logits = self.config.beta * (log_ratio_chosen - log_ratio_rejected)
            # dL/dlogits = -sigmoid(-logits) = -(1 - sigmoid(logits))
            sig_neg = sigmoid(-logits)
            d_logits = -sig_neg
            d_chosen = self.config.beta * d_logits
            d_rejected = -self.config.beta * d_logits
            return d_chosen, d_rejected

    def train_dpo(self, model, ref_model, preference_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        DPO 训练
        preference_data: [{'prompt': str, 'chosen': str, 'rejected': str}, ...]
        返回: {'steps': int, 'avg_loss': float, 'chosen_rate': float, 'log': list}
        """
        self._step_count = 0
        self._total_loss = 0.0
        self._total_chosen_reward = 0.0
        self._total_rejected_reward = 0.0
        self._training_log = []

        # 确保参考模型冻结
        ref_model.freeze()

        total_steps = len(preference_data) * self.config.epochs
        global_step = 0

        for epoch in range(self.config.epochs):
            data = preference_data[:]
            random.shuffle(data)

            for example in data:
                prompt_ids = self._tokenize_text(example['prompt'])
                chosen_ids = self._tokenize_text(example['chosen'])
                rejected_ids = self._tokenize_text(example['rejected'])

                # 截断
                max_resp = max(1, self.config.max_length - len(prompt_ids))
                chosen_ids = chosen_ids[:max_resp]
                rejected_ids = rejected_ids[:max_resp]

                if not chosen_ids or not rejected_ids:
                    continue

                # 参考模型的 log probs (不需要梯度)
                ref_lp_chosen, _, _, _ = self.compute_log_probs(ref_model, prompt_ids, chosen_ids)
                ref_lp_rejected, _, _, _ = self.compute_log_probs(ref_model, prompt_ids, rejected_ids)

                # 策略模型的 log probs (需要梯度)
                policy_lp_chosen, logits_chosen, full_ids_chosen, mask_chosen = \
                    self.compute_log_probs(model, prompt_ids, chosen_ids)
                policy_lp_rejected, logits_rejected, full_ids_rejected, mask_rejected = \
                    self.compute_log_probs(model, prompt_ids, rejected_ids)

                # log ratios
                log_ratio_chosen = policy_lp_chosen - ref_lp_chosen
                log_ratio_rejected = policy_lp_rejected - ref_lp_rejected

                # 记录 reward (用于统计)
                self._total_chosen_reward += self.config.beta * log_ratio_chosen
                self._total_rejected_reward += self.config.beta * log_ratio_rejected

                # 计算损失
                if self.config.use_ipo:
                    loss = self.ipo_loss(log_ratio_chosen, log_ratio_rejected)
                else:
                    loss = self.dpo_loss(log_ratio_chosen, log_ratio_rejected)

                # 计算梯度
                d_chosen, d_rejected = self.dpo_loss_grad(log_ratio_chosen, log_ratio_rejected)

                # 反向传播: chosen
                targets_chosen = full_ids_chosen[1:] + [0]
                min_len = min(len(logits_chosen), len(targets_chosen), len(mask_chosen))
                grad_logits_chosen = sequence_log_probs_grad(
                    logits_chosen[:min_len], targets_chosen[:min_len],
                    d_chosen, mask_chosen[:min_len])

                # 反向传播: rejected
                targets_rejected = full_ids_rejected[1:] + [0]
                min_len = min(len(logits_rejected), len(targets_rejected), len(mask_rejected))
                grad_logits_rejected = sequence_log_probs_grad(
                    logits_rejected[:min_len], targets_rejected[:min_len],
                    d_rejected, mask_rejected[:min_len])

                # 累积梯度
                model.zero_grad()
                model.forward(full_ids_chosen)
                model.backward(grad_logits_chosen)
                # 保存梯度
                grads_copy = {k: deep_copy_matrix(v) for k, v in model.grads.items()}

                model.zero_grad()
                model.forward(full_ids_rejected)
                model.backward(grad_logits_rejected)
                # 累加梯度
                for k in model.grads:
                    if k in grads_copy:
                        mat_add_inplace(model.grads[k], grads_copy[k])
                    else:
                        model.grads[k] = grads_copy[k]

                # 梯度裁剪
                total_norm = 0.0
                for g in model.grads.values():
                    total_norm += frobenius_norm(g) ** 2
                total_norm = math.sqrt(total_norm)
                if total_norm > self.config.max_grad_norm and total_norm > 0:
                    scale = self.config.max_grad_norm / total_norm
                    for g in model.grads.values():
                        mat_mul_scalar_inplace(g, scale)

                # 学习率
                lr = self._get_lr_dpo(global_step, total_steps)
                model.apply_gradients(lr, self.config.weight_decay)
                model.zero_grad()

                global_step += 1
                self._step_count += 1
                self._total_loss += loss

                if global_step % self.config.log_interval == 0:
                    self._training_log.append({
                        'step': global_step,
                        'epoch': epoch + 1,
                        'loss': loss,
                        'lr': lr,
                        'log_ratio_chosen': log_ratio_chosen,
                        'log_ratio_rejected': log_ratio_rejected,
                        'grad_norm': total_norm,
                    })

        avg_loss = self._total_loss / max(self._step_count, 1)
        chosen_rate = self._compute_chosen_rate(model, ref_model, preference_data)

        return {
            'steps': self._step_count,
            'avg_loss': avg_loss,
            'chosen_rate': chosen_rate,
            'avg_chosen_reward': self._total_chosen_reward / max(self._step_count, 1),
            'avg_rejected_reward': self._total_rejected_reward / max(self._step_count, 1),
            'log': self._training_log,
        }

    def _get_lr_dpo(self, step: int, total_steps: int) -> float:
        """DPO 学习率调度"""
        lr_max = self.config.learning_rate
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        if step < warmup_steps:
            return lr_max * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr_max * 0.5 * (1 + math.cos(math.pi * progress))

    def _compute_chosen_rate(self, model, ref_model, data: List[Dict[str, Any]]) -> float:
        """计算 chosen_rate: 模型给 chosen 更高分数的比例"""
        if not data:
            return 0.0
        chosen_count = 0
        for example in data[:50]:  # 限制评估数量
            prompt_ids = self._tokenize_text(example['prompt'])
            chosen_ids = self._tokenize_text(example['chosen'])[:max(1, self.config.max_length - len(prompt_ids))]
            rejected_ids = self._tokenize_text(example['rejected'])[:max(1, self.config.max_length - len(prompt_ids))]

            if not chosen_ids or not rejected_ids:
                continue

            policy_lp_c, _, _, _ = self.compute_log_probs(model, prompt_ids, chosen_ids)
            policy_lp_r, _, _, _ = self.compute_log_probs(model, prompt_ids, rejected_ids)
            ref_lp_c, _, _, _ = self.compute_log_probs(ref_model, prompt_ids, chosen_ids)
            ref_lp_r, _, _, _ = self.compute_log_probs(ref_model, prompt_ids, rejected_ids)

            reward_c = self.config.beta * (policy_lp_c - ref_lp_c)
            reward_r = self.config.beta * (policy_lp_r - ref_lp_r)

            if reward_c > reward_r:
                chosen_count += 1

        return chosen_count / min(len(data), 50)

    def evaluate_dpo(self, model, ref_model, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """DPO 评估"""
        chosen_rate = self._compute_chosen_rate(model, ref_model, eval_data)
        return {
            'chosen_rate': chosen_rate,
            'num_examples': len(eval_data),
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'config': asdict(self.config),
            'step_count': self._step_count,
            'avg_loss': self._total_loss / max(self._step_count, 1),
            'avg_chosen_reward': self._total_chosen_reward / max(self._step_count, 1),
            'avg_rejected_reward': self._total_rejected_reward / max(self._step_count, 1),
            'use_ipo': self.config.use_ipo,
            'beta': self.config.beta,
            'recent_log': self._training_log[-10:],
        }

    def get_dashboard(self) -> str:
        """获取仪表盘字符串"""
        s = self.get_stats()
        lines = [
            "========== DPOTrainer 仪表盘 ==========",
            f"  Beta:             {s['beta']}",
            f"  使用 IPO 变体:    {s['use_ipo']}",
            f"  学习率:           {s['config']['learning_rate']}",
            f"  训练步数:         {s['step_count']}",
            f"  平均损失:         {s['avg_loss']:.6f}",
            f"  平均 chosen 奖励: {s['avg_chosen_reward']:.6f}",
            f"  平均 rejected 奖励:{s['avg_rejected_reward']:.6f}",
            "=======================================",
        ]
        for entry in s['recent_log'][-5:]:
            lines.append(f"  Step {entry['step']}: loss={entry['loss']:.6f}, "
                         f"lr_c={entry['log_ratio_chosen']:.4f}, lr_r={entry['log_ratio_rejected']:.4f}")
        return '\n'.join(lines)


# ============================================================
# 第九部分: ContinualLearner — 持续学习 (#34)
# ============================================================

@dataclass
class ContinualConfig:
    """持续学习配置"""
    buffer_size: int = 200          # 经验回放缓冲区大小
    ewc_lambda: float = 100.0       # EWC 约束强度
    kd_lambda: float = 1.0          # 知识蒸馏强度
    kd_temperature: float = 2.0     # 蒸馏温度
    learning_rate: float = 1e-3
    epochs_per_task: int = 2
    batch_size: int = 1
    max_grad_norm: float = 1.0
    replay_ratio: float = 0.3       # 回放数据比例
    forgetting_threshold: float = 0.1  # 遗忘检测阈值
    boundary_threshold: float = 0.15  # 任务边界检测阈值
    use_ewc: bool = True
    use_kd: bool = True
    use_replay: bool = True
    seed: int = 42


@dataclass
class TaskRecord:
    """任务记录"""
    task_id: str
    data_stats: Dict[str, Any] = field(default_factory=dict)
    performance: float = 0.0
    fisher_info: Dict[str, Any] = field(default_factory=dict)
    param_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ''
    num_samples: int = 0


class ContinualLearner:
    """
    持续学习器
    - 经验回放 (Experience Replay): 从旧数据 buffer 采样混合新数据
    - 弹性权重巩固 (EWC) 简化版: 约束重要参数不变
    - 知识蒸馏约束: 新模型输出对齐旧模型
    - 遗忘检测 / 任务边界检测
    - Reservoir sampling 维护缓冲区
    - 多任务记忆
    """

    def __init__(self, config: Optional[ContinualConfig] = None):
        self.config = config or ContinualConfig()
        self.buffer: deque = deque(maxlen=self.config.buffer_size)  # 经验回放缓冲区
        self._total_seen: int = 0  # 总样本计数 (用于 reservoir sampling)
        self.tasks: Dict[str, TaskRecord] = {}  # 任务记录
        self._current_task_id: Optional[str] = None
        self._old_model: Optional[Any] = None  # 旧模型 (用于蒸馏)
        self._ewc_fisher: Dict[str, List[List[float]]] = {}  # Fisher 信息矩阵
        self._ewc_old_params: Dict[str, List[List[float]]] = {}  # EWC 参考参数
        self._training_log: List[Dict[str, Any]] = []
        self._forgetting_log: List[Dict[str, Any]] = []
        random.seed(self.config.seed)

    def reservoir_sample(self, item: Any) -> None:
        """
        Reservoir sampling 维护缓冲区
        当缓冲区满时, 以 buffer_size/total_seen 的概率替换随机位置
        """
        self._total_seen += 1
        if len(self.buffer) < self.config.buffer_size:
            self.buffer.append(item)
        else:
            j = random.randint(0, self._total_seen - 1)
            if j < self.config.buffer_size:
                self.buffer[j] = item

    def manage_buffer(self, new_data: List[Any]) -> None:
        """批量管理缓冲区: 将新数据加入缓冲"""
        for item in new_data:
            self.reservoir_sample(item)

    def get_replay_batch(self, batch_size: int) -> List[Any]:
        """从缓冲区采样回放数据"""
        if not self.buffer:
            return []
        n = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def compute_fisher(self, model, data: List[Dict[str, Any]]) -> Dict[str, List[List[float]]]:
        """
        计算 Fisher 信息矩阵 (简化版)
        F_i ≈ (gradient_i)^2, 在数据上平均
        """
        fisher = {}
        count = 0
        for example in data[:50]:  # 限制数据量
            input_ids = example.get('input_ids', [])
            targets = example.get('targets', input_ids[1:] + [0])
            min_len = min(len(input_ids), len(targets))
            if min_len < 2:
                continue
            input_ids = input_ids[:min_len]
            targets = targets[:min_len]

            logits = model.forward(input_ids)
            grad_logits = sequence_ce_grad(logits, targets)
            model.zero_grad()
            model.backward(grad_logits)

            for name, grad in model.grads.items():
                if name not in fisher:
                    fisher[name] = zeros(len(grad), len(grad[0]))
                for i in range(len(grad)):
                    for j in range(len(grad[0])):
                        fisher[name][i][j] += grad[i][j] ** 2
            count += 1

        # 平均
        if count > 0:
            for name in fisher:
                mat_mul_scalar_inplace(fisher[name], 1.0 / count)

        return fisher

    def ewc_penalty(self, model) -> float:
        """
        计算 EWC 惩罚项
        L_ewc = (lambda/2) * sum_i F_i * (theta_i - theta_old_i)^2
        """
        if not self._ewc_fisher or not self._ewc_old_params:
            return 0.0
        penalty = 0.0
        for name, fisher in self._ewc_fisher.items():
            old_param = self._ewc_old_params.get(name)
            cur_param = model.params.get(name)
            if old_param is None or cur_param is None:
                continue
            for i in range(len(fisher)):
                for j in range(len(fisher[0])):
                    diff = cur_param[i][j] - old_param[i][j]
                    penalty += fisher[i][j] * diff * diff
        return 0.5 * self.config.ewc_lambda * penalty

    def ewc_grad(self, model) -> Dict[str, List[List[float]]]:
        """EWC 惩罚对参数的梯度: lambda * F * (theta - theta_old)"""
        grads = {}
        if not self._ewc_fisher or not self._ewc_old_params:
            return grads
        for name, fisher in self._ewc_fisher.items():
            old_param = self._ewc_old_params.get(name)
            cur_param = model.params.get(name)
            if old_param is None or cur_param is None:
                continue
            g = zeros(len(fisher), len(fisher[0]))
            for i in range(len(fisher)):
                for j in range(len(fisher[0])):
                    g[i][j] = self.config.ewc_lambda * fisher[i][j] * (cur_param[i][j] - old_param[i][j])
            grads[name] = g
        return grads

    def distillation_loss(self, student_logits: List[List[float]],
                          teacher_logits: List[List[float]]) -> float:
        """
        知识蒸馏损失 (MSE 简化版)
        L_kd = mean((student - teacher)^2)
        """
        if not student_logits or not teacher_logits:
            return 0.0
        total = 0.0
        count = 0
        for i in range(min(len(student_logits), len(teacher_logits))):
            s = student_logits[i]
            t = teacher_logits[i]
            for j in range(min(len(s), len(t))):
                total += (s[j] - t[j]) ** 2
                count += 1
        return total / max(count, 1)

    def distillation_grad(self, student_logits: List[List[float]],
                          teacher_logits: List[List[float]]) -> List[List[float]]:
        """蒸馏损失对 student logits 的梯度: 2*(student - teacher)/N"""
        if not student_logits or not teacher_logits:
            return []
        n = 0
        for i in range(min(len(student_logits), len(teacher_logits))):
            n += min(len(student_logits[i]), len(teacher_logits[i]))
        n = max(n, 1)
        grad = []
        for i in range(len(student_logits)):
            row_grad = [0.0] * len(student_logits[i])
            if i < len(teacher_logits):
                for j in range(min(len(student_logits[i]), len(teacher_logits[i]))):
                    row_grad[j] = 2.0 * (student_logits[i][j] - teacher_logits[i][j]) / n
            grad.append(row_grad)
        return grad

    def detect_forgetting(self, model, old_task_data: List[Dict[str, Any]],
                          old_performance: float) -> Dict[str, Any]:
        """
        遗忘检测: 评估旧任务性能下降程度
        """
        # 计算当前模型在旧任务数据上的性能
        total_loss = 0.0
        count = 0
        for example in old_task_data[:50]:
            input_ids = example.get('input_ids', [])
            targets = example.get('targets', input_ids[1:] + [0])
            min_len = min(len(input_ids), len(targets))
            if min_len < 2:
                continue
            logits = model.forward(input_ids[:min_len])
            loss = sequence_cross_entropy(logits, targets[:min_len])
            total_loss += loss
            count += 1

        current_perf = total_loss / max(count, 1) if count > 0 else float('inf')
        # 性能下降 = 当前损失 - 旧损失 (越大说明遗忘越严重)
        forgetting = current_perf - old_performance if old_performance > 0 else 0.0
        is_forgetting = forgetting > self.config.forgetting_threshold

        result = {
            'old_performance': old_performance,
            'current_performance': current_perf,
            'forgetting_amount': forgetting,
            'is_forgetting': is_forgetting,
            'timestamp': now_str(),
        }
        self._forgetting_log.append(result)
        return result

    def detect_task_boundary(self, new_data: List[Dict[str, Any]],
                             old_data_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        任务边界检测: 检测数据分布变化
        通过比较 token 频率分布和平均序列长度
        """
        # 计算新数据的统计特征
        new_stats = self._compute_data_stats(new_data)

        if old_data_stats is None:
            return {
                'is_boundary': False,
                'shift_score': 0.0,
                'new_stats': new_stats,
                'old_stats': None,
            }

        # 计算分布差异 (简化: 使用平均长度差异和 token 分布 KL 散度)
        length_diff = abs(new_stats['avg_length'] - old_data_stats.get('avg_length', 0))
        old_avg = old_data_stats.get('avg_length', 1) or 1
        length_shift = length_diff / old_avg

        # Token 频率分布差异 (简化: 使用 Jaccard 相似度)
        new_tokens = set(new_stats.get('token_freq', {}).keys())
        old_tokens = set(old_data_stats.get('token_freq', {}).keys())
        if new_tokens or old_tokens:
            jaccard = len(new_tokens & old_tokens) / len(new_tokens | old_tokens)
        else:
            jaccard = 1.0
        token_shift = 1.0 - jaccard

        shift_score = 0.5 * length_shift + 0.5 * token_shift
        is_boundary = shift_score > self.config.boundary_threshold

        return {
            'is_boundary': is_boundary,
            'shift_score': shift_score,
            'length_shift': length_shift,
            'token_shift': token_shift,
            'new_stats': new_stats,
            'old_stats': old_data_stats,
        }

    def _compute_data_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算数据统计特征"""
        lengths = []
        token_freq = Counter()
        for example in data:
            ids = example.get('input_ids', [])
            lengths.append(len(ids))
            token_freq.update(ids)
        return {
            'num_samples': len(data),
            'avg_length': mean(lengths) if lengths else 0.0,
            'std_length': std(lengths) if lengths else 0.0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'token_freq': dict(token_freq),
            'unique_tokens': len(token_freq),
        }

    def record_task(self, task_id: str, model, data: List[Dict[str, Any]],
                    performance: float) -> TaskRecord:
        """记录任务信息"""
        # 计算数据统计
        data_stats = self._compute_data_stats(data)
        # 计算 Fisher 信息
        fisher = self.compute_fisher(model, data) if self.config.use_ewc else {}
        # 保存参数快照
        param_snapshot = {k: deep_copy_matrix(v) for k, v in model.params.items()}

        record = TaskRecord(
            task_id=task_id,
            data_stats=data_stats,
            performance=performance,
            fisher_info=fisher,
            param_snapshot=param_snapshot,
            timestamp=now_str(),
            num_samples=len(data),
        )
        self.tasks[task_id] = record

        # 更新 EWC 信息
        if self.config.use_ewc:
            self._ewc_fisher = fisher
            self._ewc_old_params = param_snapshot

        # 保存旧模型用于蒸馏
        if self.config.use_kd:
            self._old_model = model.copy()
            self._old_model.freeze()

        return record

    def learn_new_data(self, model, new_data: List[Dict[str, Any]],
                       task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        在线学习新数据
        1. 检测任务边界
        2. 经验回放: 混合新数据和旧数据
        3. EWC 约束 + 知识蒸馏
        4. 记录任务
        返回: 训练结果
        """
        # 生成任务 ID
        if task_id is None:
            task_id = f'task_{len(self.tasks)}'

        # 任务边界检测
        old_stats = self.tasks[self._current_task_id].data_stats if self._current_task_id else None
        boundary_result = self.detect_task_boundary(new_data, old_stats)

        # 准备训练数据: 混合新数据和回放数据
        train_data = new_data[:]
        if self.config.use_replay and self.buffer:
            replay_size = int(len(new_data) * self.config.replay_ratio)
            replay_batch = self.get_replay_batch(replay_size)
            train_data.extend(replay_batch)
        random.shuffle(train_data)

        # 训练
        total_steps = 0
        total_loss = 0.0
        total_ewc_loss = 0.0
        total_kd_loss = 0.0

        for epoch in range(self.config.epochs_per_task):
            for example in train_data:
                input_ids = example.get('input_ids', [])
                targets = example.get('targets', input_ids[1:] + [0])
                min_len = min(len(input_ids), len(targets))
                if min_len < 2:
                    continue
                input_ids = input_ids[:min_len]
                targets = targets[:min_len]

                # 前向传播
                logits = model.forward(input_ids)
                # 任务损失
                task_loss = sequence_cross_entropy(logits, targets)
                grad_logits = sequence_ce_grad(logits, targets)

                # EWC 惩罚梯度
                ewc_loss = 0.0
                if self.config.use_ewc and self._ewc_fisher:
                    ewc_loss = self.ewc_penalty(model)
                    ewc_grads = self.ewc_grad(model)
                    # 将 EWC 梯度加到模型梯度
                    # 先反向传播任务梯度
                    model.zero_grad()
                    model.backward(grad_logits)
                    for name, eg in ewc_grads.items():
                        if name in model.grads:
                            mat_add_inplace(model.grads[name], eg)
                        else:
                            model.grads[name] = eg
                else:
                    model.zero_grad()
                    model.backward(grad_logits)

                # 知识蒸馏
                kd_loss = 0.0
                if self.config.use_kd and self._old_model is not None:
                    teacher_logits = self._old_model.forward(input_ids)
                    kd_loss = self.distillation_loss(logits, teacher_logits)
                    kd_grad = self.distillation_grad(logits, teacher_logits)
                    # 将蒸馏梯度加到 grad_logits
                    for i in range(min(len(grad_logits), len(kd_grad))):
                        for j in range(min(len(grad_logits[i]), len(kd_grad[i]))):
                            model.grads_deps = None  # placeholder
                    # 重新计算: 蒸馏梯度需要加到参数梯度上
                    # 简化: 直接将蒸馏梯度作为额外的 grad_logits 反向传播
                    # (累积到已有梯度上)
                    # 这里用简化方式: 将 kd_loss * kd_lambda 加到总损失, 梯度近似处理
                    kd_weighted_grad = scalar_mul(kd_grad, self.config.kd_lambda)
                    model.backward(kd_weighted_grad)  # 累积梯度

                # 梯度裁剪
                total_norm = 0.0
                for g in model.grads.values():
                    total_norm += frobenius_norm(g) ** 2
                total_norm = math.sqrt(total_norm)
                if total_norm > self.config.max_grad_norm and total_norm > 0:
                    scale = self.config.max_grad_norm / total_norm
                    for g in model.grads.values():
                        mat_mul_scalar_inplace(g, scale)

                # 更新参数
                model.apply_gradients(self.config.learning_rate)
                model.zero_grad()

                total_steps += 1
                total_loss += task_loss
                total_ewc_loss += ewc_loss
                total_kd_loss += kd_loss

        # 更新缓冲区
        if self.config.use_replay:
            self.manage_buffer(new_data)

        # 评估性能
        perf = total_loss / max(total_steps, 1)

        # 记录任务
        self.record_task(task_id, model, new_data, perf)
        self._current_task_id = task_id

        result = {
            'task_id': task_id,
            'steps': total_steps,
            'avg_task_loss': total_loss / max(total_steps, 1),
            'avg_ewc_loss': total_ewc_loss / max(total_steps, 1),
            'avg_kd_loss': total_kd_loss / max(total_steps, 1),
            'boundary_detected': boundary_result['is_boundary'],
            'shift_score': boundary_result['shift_score'],
            'buffer_size': len(self.buffer),
            'num_tasks': len(self.tasks),
        }
        self._training_log.append(result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'config': asdict(self.config),
            'buffer_size': len(self.buffer),
            'buffer_capacity': self.config.buffer_size,
            'total_seen': self._total_seen,
            'num_tasks': len(self.tasks),
            'current_task': self._current_task_id,
            'task_ids': list(self.tasks.keys()),
            'ewc_active': bool(self._ewc_fisher),
            'kd_active': self._old_model is not None,
            'training_log': self._training_log[-10:],
            'forgetting_log': self._forgetting_log[-5:],
        }

    def get_dashboard(self) -> str:
        """获取仪表盘字符串"""
        s = self.get_stats()
        lines = [
            "========== ContinualLearner 仪表盘 ==========",
            f"  缓冲区大小:       {s['buffer_size']}/{s['buffer_capacity']}",
            f"  总样本数:         {s['total_seen']}",
            f"  任务数量:         {s['num_tasks']}",
            f"  当前任务:         {s['current_task']}",
            f"  任务列表:         {s['task_ids']}",
            f"  EWC 激活:         {s['ewc_active']}",
            f"  知识蒸馏激活:     {s['kd_active']}",
            f"  EWC lambda:       {s['config']['ewc_lambda']}",
            f"  KD lambda:        {s['config']['kd_lambda']}",
            f"  回放比例:         {s['config']['replay_ratio']}",
        ]
        for entry in s['training_log'][-3:]:
            lines.append(f"  Task {entry['task_id']}: steps={entry['steps']}, "
                         f"loss={entry['avg_task_loss']:.6f}, boundary={entry['boundary_detected']}")
        if s['forgetting_log']:
            for entry in s['forgetting_log'][-2:]:
                lines.append(f"  遗忘检测: amount={entry['forgetting_amount']:.6f}, "
                             f"is_forgetting={entry['is_forgetting']}")
        lines.append("=============================================")
        return '\n'.join(lines)


# ============================================================
# 第十部分: DomainAdapter — 领域适配 (#35)
# ============================================================

@dataclass
class DomainConfig:
    """领域适配配置"""
    max_vocab_extension: int = 100      # 最大扩展词表数
    domain_pretrain_epochs: int = 2     # 领域预训练轮数
    sft_epochs: int = 2                 # SFT 轮数
    mixed_ratio: float = 0.5            # 领域数据比例 (混合训练)
    learning_rate: float = 1e-3
    max_length: int = 128
    max_grad_norm: float = 1.0
    analysis_sample_size: int = 200     # 分析采样数
    save_dir: str = field(default_factory=lambda: os.path.join(_DATA_DIR, 'domain'))
    seed: int = 42


@dataclass
class DomainInfo:
    """领域信息"""
    name: str
    data_path: str = ''
    data: List[str] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    vocab: Dict[str, int] = field(default_factory=dict)
    new_tokens: List[str] = field(default_factory=list)
    num_samples: int = 0
    registered_at: str = ''


class DomainRouter:
    """
    领域路由器 (简化版朴素贝叶斯分类器)
    根据输入文本判断属于哪个领域
    """

    def __init__(self):
        self.domain_word_counts: Dict[str, Counter] = {}
        self.domain_totals: Dict[str, int] = {}
        self.domain_vocab_sizes: Dict[str, int] = {}
        self.domains: List[str] = []

    def train(self, domain: str, text: str) -> None:
        """训练路由器: 记录领域词频"""
        if domain not in self.domain_word_counts:
            self.domain_word_counts[domain] = Counter()
            self.domain_totals[domain] = 0
            self.domain_vocab_sizes[domain] = 0
            self.domains.append(domain)
        words = text.split()
        self.domain_word_counts[domain].update(words)
        self.domain_totals[domain] += len(words)
        self.domain_vocab_sizes[domain] = len(self.domain_word_counts[domain])

    def predict(self, text: str) -> Optional[str]:
        """预测输入文本属于哪个领域"""
        if not self.domains:
            return None
        words = text.split()
        best_domain = None
        best_score = float('-inf')

        total_vocab = sum(self.domain_vocab_sizes.values())
        for domain in self.domains:
            score = 0.0
            word_counts = self.domain_word_counts[domain]
            total = self.domain_totals[domain]
            vocab_size = self.domain_vocab_sizes[domain]
            # 朴素贝叶斯 (对数概率, 拉普拉斯平滑)
            for w in words:
                count = word_counts.get(w, 0)
                prob = (count + 1) / (total + total_vocab)
                score += math.log(max(prob, 1e-10))
            if score > best_score:
                best_score = score
                best_domain = domain

        return best_domain

    def get_confidence(self, text: str) -> Dict[str, float]:
        """获取各领域的置信度"""
        if not self.domains:
            return {}
        words = text.split()
        scores = {}
        total_vocab = sum(self.domain_vocab_sizes.values())
        for domain in self.domains:
            score = 0.0
            word_counts = self.domain_word_counts[domain]
            total = self.domain_totals[domain]
            for w in words:
                count = word_counts.get(w, 0)
                prob = (count + 1) / (total + total_vocab)
                score += math.log(max(prob, 1e-10))
            scores[domain] = score

        # 归一化为概率
        max_score = max(scores.values()) if scores else 0
        exp_scores = {d: safe_exp(s - max_score) for d, s in scores.items()}
        total_exp = sum(exp_scores.values())
        if total_exp > 0:
            return {d: s / total_exp for d, s in exp_scores.items()}
        return {d: 1.0 / len(scores) for d in scores}


class DomainAdapter:
    """
    领域适配器
    - 领域数据加载 / 领域分析
    - 领域词汇扩展 (提取领域特有 token, 扩展词表)
    - 领域适配微调: 先继续预训练, 再 SFT
    - 混合训练: 领域数据 + 通用数据按比例混合
    - 领域评估: 领域内/外性能对比
    - 多领域管理 / 领域路由
    """

    def __init__(self, config: Optional[DomainConfig] = None):
        self.config = config or DomainConfig()
        self.domains: Dict[str, DomainInfo] = {}
        self.router = DomainRouter()
        self._training_log: List[Dict[str, Any]] = []
        random.seed(self.config.seed)

    def register_domain(self, domain_name: str, data_path: str = '',
                        data: Optional[List[str]] = None) -> DomainInfo:
        """注册一个领域"""
        if data is None:
            data = []
        info = DomainInfo(
            name=domain_name,
            data_path=data_path,
            data=data,
            num_samples=len(data),
            registered_at=now_str(),
        )
        self.domains[domain_name] = info
        # 如果提供了数据, 训练领域路由器
        for text in data:
            self.router.train(domain_name, text)
        return info

    def load_domain_data(self, domain_name: str, data_path: str) -> List[str]:
        """从文件加载领域数据"""
        data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(line)
        except Exception:
            pass

        if domain_name in self.domains:
            self.domains[domain_name].data = data
            self.domains[domain_name].data_path = data_path
            self.domains[domain_name].num_samples = len(data)
        else:
            self.register_domain(domain_name, data_path, data)

        # 训练路由器
        for text in data:
            self.router.train(domain_name, text)

        return data

    def analyze_domain(self, domain_name: str,
                       data: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析领域数据特征
        - 词汇分析: 词频, 独特词数
        - 长度分析: 平均/标准差/最小/最大长度
        - 主题分布: 简化关键词提取
        """
        if data is None:
            info = self.domains.get(domain_name)
            data = info.data if info else []

        sample = data[:self.config.analysis_sample_size]

        # 长度分析
        lengths = [len(text) for text in sample]
        word_lengths = [len(text.split()) for text in sample]

        # 词汇分析
        word_freq = Counter()
        char_freq = Counter()
        for text in sample:
            word_freq.update(text.split())
            char_freq.update(text)

        # 主题关键词 (高频词, 排除单字符)
        keywords = [(w, c) for w, c in word_freq.most_common(20) if len(w) > 1]

        analysis = {
            'domain_name': domain_name,
            'num_samples': len(data),
            'avg_char_length': mean(lengths) if lengths else 0.0,
            'std_char_length': std(lengths) if lengths else 0.0,
            'min_char_length': min(lengths) if lengths else 0,
            'max_char_length': max(lengths) if lengths else 0,
            'avg_word_length': mean(word_lengths) if word_lengths else 0.0,
            'unique_words': len(word_freq),
            'unique_chars': len(char_freq),
            'top_keywords': keywords[:10],
            'top_chars': char_freq.most_common(10),
            'vocab_richness': len(word_freq) / max(sum(word_freq.values()), 1),
        }

        if domain_name in self.domains:
            self.domains[domain_name].analysis = analysis

        return analysis

    def extract_domain_vocab(self, domain_name: str,
                             general_data: Optional[List[str]] = None,
                             threshold: float = 2.0) -> List[str]:
        """
        提取领域特有 token
        通过比较领域数据和通用数据的词频, 找出在领域中显著高频的词
        threshold: 领域频率/通用频率 的阈值
        """
        info = self.domains.get(domain_name)
        if info is None or not info.data:
            return []

        # 领域词频
        domain_freq = Counter()
        for text in info.data[:self.config.analysis_sample_size]:
            domain_freq.update(text.split())
        domain_total = sum(domain_freq.values()) or 1

        # 通用词频
        general_freq = Counter()
        if general_data:
            for text in general_data[:self.config.analysis_sample_size]:
                general_freq.update(text.split())
        general_total = sum(general_freq.values()) or 1

        # 找出领域特有词
        new_tokens = []
        for word, count in domain_freq.most_common(self.config.max_vocab_extension * 2):
            if len(word) < 2:
                continue
            domain_ratio = count / domain_total
            general_ratio = general_freq.get(word, 0) / general_total
            if general_ratio == 0:
                ratio = float('inf')
            else:
                ratio = domain_ratio / general_ratio
            if ratio >= threshold:
                new_tokens.append(word)
            if len(new_tokens) >= self.config.max_vocab_extension:
                break

        if domain_name in self.domains:
            self.domains[domain_name].new_tokens = new_tokens
            self.domains[domain_name].vocab = {tok: idx for idx, tok in enumerate(new_tokens)}

        return new_tokens

    def extend_vocab(self, model, new_tokens: List[str]) -> int:
        """
        扩展模型词表
        为新 token 添加 embedding 行
        返回: 新增的 token 数量
        """
        if not new_tokens or model is None:
            return 0

        old_vocab = model.vocab_size
        old_embedding = model.params.get('embedding', [])
        old_lm_head = model.params.get('lm_head', [])
        d = model.dim

        # 扩展 embedding
        new_rows = random_gauss(len(new_tokens), d, std=model.config.init_std)
        model.params['embedding'] = old_embedding + new_rows

        # 扩展 lm_head
        new_lm_rows = random_gauss(d, len(new_tokens), std=model.config.init_std)
        # lm_head 是 [dim, vocab], 需要在列方向扩展
        for i in range(len(old_lm_head)):
            model.params['lm_head'][i] = old_lm_head[i] + new_lm_rows[i]

        model.vocab_size = old_vocab + len(new_tokens)
        return len(new_tokens)

    def domain_pretrain(self, model, domain_name: str,
                        epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        领域继续预训练
        在领域数据上进行语言建模训练
        """
        info = self.domains.get(domain_name)
        if info is None or not info.data:
            return {'steps': 0, 'avg_loss': 0.0}

        epochs = epochs or self.config.domain_pretrain_epochs
        total_steps = 0
        total_loss = 0.0

        for epoch in range(epochs):
            data = info.data[:]
            random.shuffle(data)

            for text in data:
                # 简单字符级编码
                input_ids = [ord(ch) % model.vocab_size for ch in text[:self.config.max_length]]
                if len(input_ids) < 2:
                    continue
                targets = input_ids[1:] + [0]
                input_ids = input_ids[:len(targets)]

                logits = model.forward(input_ids)
                loss = sequence_cross_entropy(logits, targets)
                grad_logits = sequence_ce_grad(logits, targets)

                model.zero_grad()
                model.backward(grad_logits)

                # 梯度裁剪
                total_norm = 0.0
                for g in model.grads.values():
                    total_norm += frobenius_norm(g) ** 2
                total_norm = math.sqrt(total_norm)
                if total_norm > self.config.max_grad_norm and total_norm > 0:
                    scale = self.config.max_grad_norm / total_norm
                    for g in model.grads.values():
                        mat_mul_scalar_inplace(g, scale)

                model.apply_gradients(self.config.learning_rate)
                model.zero_grad()

                total_steps += 1
                total_loss += loss

        result = {
            'domain': domain_name,
            'phase': 'pretrain',
            'steps': total_steps,
            'avg_loss': total_loss / max(total_steps, 1),
        }
        self._training_log.append(result)
        return result

    def mixed_training(self, model, domain_data: List[str],
                       general_data: List[str],
                       ratio: Optional[float] = None,
                       epochs: int = 1) -> Dict[str, Any]:
        """
        混合训练: 领域数据 + 通用数据按比例混合
        """
        ratio = ratio if ratio is not None else self.config.mixed_ratio
        # 按比例采样
        domain_size = int(len(domain_data) * ratio)
        general_size = int(len(general_data) * (1 - ratio))

        mixed_data = random.sample(domain_data, min(domain_size, len(domain_data))) + \
                     random.sample(general_data, min(general_size, len(general_data)))
        random.shuffle(mixed_data)

        total_steps = 0
        total_loss = 0.0

        for epoch in range(epochs):
            for text in mixed_data:
                input_ids = [ord(ch) % model.vocab_size for ch in text[:self.config.max_length]]
                if len(input_ids) < 2:
                    continue
                targets = input_ids[1:] + [0]
                input_ids = input_ids[:len(targets)]

                logits = model.forward(input_ids)
                loss = sequence_cross_entropy(logits, targets)
                grad_logits = sequence_ce_grad(logits, targets)

                model.zero_grad()
                model.backward(grad_logits)

                # 梯度裁剪
                total_norm = 0.0
                for g in model.grads.values():
                    total_norm += frobenius_norm(g) ** 2
                total_norm = math.sqrt(total_norm)
                if total_norm > self.config.max_grad_norm and total_norm > 0:
                    scale = self.config.max_grad_norm / total_norm
                    for g in model.grads.values():
                        mat_mul_scalar_inplace(g, scale)

                model.apply_gradients(self.config.learning_rate)
                model.zero_grad()

                total_steps += 1
                total_loss += loss

        result = {
            'phase': 'mixed_training',
            'domain_ratio': ratio,
            'domain_samples': min(domain_size, len(domain_data)),
            'general_samples': min(general_size, len(general_data)),
            'steps': total_steps,
            'avg_loss': total_loss / max(total_steps, 1),
        }
        self._training_log.append(result)
        return result

    def domain_adaptive_finetune(self, model, domain_name: str,
                                 sft_data: Optional[List[Dict[str, Any]]] = None,
                                 general_data: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        领域适配微调: 先继续预训练, 再 SFT
        """
        results = {}

        # 阶段 1: 领域继续预训练
        pretrain_result = self.domain_pretrain(model, domain_name)
        results['pretrain'] = pretrain_result

        # 阶段 2: 混合训练 (如果有通用数据)
        info = self.domains.get(domain_name)
        if general_data and info and info.data:
            mixed_result = self.mixed_training(model, info.data, general_data)
            results['mixed'] = mixed_result

        # 阶段 3: SFT (如果有 SFT 数据)
        if sft_data:
            sft_config = SFTConfig(epochs=self.config.sft_epochs,
                                   learning_rate=self.config.learning_rate,
                                   max_length=self.config.max_length)
            sft_trainer = SFTTrainer(sft_config)
            sft_result = sft_trainer.train_sft(model, sft_data)
            results['sft'] = {
                'steps': sft_result['steps'],
                'avg_loss': sft_result['avg_loss'],
            }

        results['domain'] = domain_name
        return results

    def evaluate_domain(self, model, in_domain_data: List[str],
                        out_domain_data: List[str]) -> Dict[str, Any]:
        """
        领域评估: 领域内/外性能对比
        """
        def _eval(data):
            total_loss = 0.0
            count = 0
            for text in data[:50]:
                input_ids = [ord(ch) % model.vocab_size for ch in text[:self.config.max_length]]
                if len(input_ids) < 2:
                    continue
                targets = input_ids[1:] + [0]
                input_ids = input_ids[:len(targets)]
                logits = model.forward(input_ids)
                loss = sequence_cross_entropy(logits, targets)
                total_loss += loss
                count += 1
            return total_loss / max(count, 1)

        in_loss = _eval(in_domain_data)
        out_loss = _eval(out_domain_data)

        return {
            'in_domain_loss': in_loss,
            'out_domain_loss': out_loss,
            'loss_gap': out_loss - in_loss,
            'in_domain_perplexity': safe_exp(in_loss),
            'out_domain_perplexity': safe_exp(out_loss),
            'num_in_domain': len(in_domain_data),
            'num_out_domain': len(out_domain_data),
        }

    def route_domain(self, input_text: str) -> Tuple[Optional[str], Dict[str, float]]:
        """
        领域路由: 根据输入判断属于哪个领域
        返回: (预测领域, 各领域置信度)
        """
        confidence = self.router.get_confidence(input_text)
        predicted = self.router.predict(input_text)
        return predicted, confidence

    def list_domains(self) -> List[str]:
        """列出所有注册的领域"""
        return list(self.domains.keys())

    def get_domain_info(self, domain_name: str) -> Optional[DomainInfo]:
        """获取领域信息"""
        return self.domains.get(domain_name)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        domain_stats = {}
        for name, info in self.domains.items():
            domain_stats[name] = {
                'num_samples': info.num_samples,
                'num_new_tokens': len(info.new_tokens),
                'registered_at': info.registered_at,
                'has_analysis': bool(info.analysis),
            }

        return {
            'config': asdict(self.config),
            'num_domains': len(self.domains),
            'domain_names': list(self.domains.keys()),
            'domain_stats': domain_stats,
            'router_domains': self.router.domains,
            'training_log': self._training_log[-10:],
        }

    def get_dashboard(self) -> str:
        """获取仪表盘字符串"""
        s = self.get_stats()
        lines = [
            "========== DomainAdapter 仪表盘 ==========",
            f"  领域数量:         {s['num_domains']}",
            f"  领域列表:         {s['domain_names']}",
            f"  最大词表扩展:     {s['config']['max_vocab_extension']}",
            f"  预训练轮数:       {s['config']['domain_pretrain_epochs']}",
            f"  SFT 轮数:         {s['config']['sft_epochs']}",
            f"  混合比例:         {s['config']['mixed_ratio']}",
        ]
        for name, ds in s['domain_stats'].items():
            lines.append(f"  --- 领域: {name} ---")
            lines.append(f"    样本数:          {ds['num_samples']}")
            lines.append(f"    新增 token 数:   {ds['num_new_tokens']}")
            lines.append(f"    注册时间:        {ds['registered_at']}")
        for entry in s['training_log'][-5:]:
            phase = entry.get('phase', 'unknown')
            domain = entry.get('domain', '')
            steps = entry.get('steps', 0)
            avg_loss = entry.get('avg_loss', 0.0)
            lines.append(f"  训练记录: phase={phase}, domain={domain}, steps={steps}, loss={avg_loss:.6f}")
        lines.append("==========================================")
        return '\n'.join(lines)
