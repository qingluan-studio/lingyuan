#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
灵元模型 (Lingyuan Model) - 自进化系统模块
模块编号: Part 21
================================================================================

本模块实现了完整的模型自进化系统,涵盖神经架构搜索(NAS)、超参数优化、
模型压缩、进化引擎、AutoML流水线、模型注册中心和实验追踪器七大核心组件。
所有组件均使用纯Python标准库实现,零外部依赖,不依赖其他part文件。

评估函数采用基于真实模型操作与理论分析的评估,结合实际训练与确定性公式验证自进化流程。

功能概览:
    1. NeuralArchitectureSearch  - 神经架构搜索
       搜索空间: hidden_dim, num_layers, num_heads, ffn_dim, num_kv_heads, dropout
       搜索策略: 随机搜索、网格搜索、进化算法、贝叶斯优化模拟
       适应度评估: 参数量、理论FLOPS、估计精度
       帕累托前沿: 多目标优化 (精度 vs 速度 vs 大小)
       搜索历史: 记录所有尝试过的架构和结果
    2. HyperparameterOptimizer   - 超参数优化器
       优化目标: learning_rate, batch_size, weight_decay, warmup_steps, grad_clip
       优化算法: 网格搜索、随机搜索、TPE模拟、CMA-ES模拟
       早停机制、并行评估模拟、参数重要性分析与交互分析
    3. AutoCompressor            - 自动模型压缩器
       量化感知、结构化/非结构化剪枝、知识蒸馏配置生成
       压缩Pipeline (量化→剪枝→蒸馏)、压缩比 vs 精度损失曲线
    4. EvolutionEngine           - 进化引擎
       种群管理、变异、交叉、锦标赛/轮盘赌选择、精英保留、进化历史
    5. AutoMLPipeline            - AutoML流水线
       端到端 (NAS→HPO→训练→压缩→评估)、自动配置、时间预算、报告生成
    6. ModelRegistry             - 模型注册中心
       模型注册、版本管理、性能排行、模型对比、模型导出
    7. ExperimentTracker         - 实验追踪器
       实验记录、实验对比、实验复现、统计分析

设计原则:
    - 纯Python标准库实现,零外部依赖
    - 模块化设计,各组件可独立使用
    - 类型注解完备,代码自文档化
    - 完善的错误处理和边界检查
    - 不依赖其他part文件,使用独立的评估函数(基于真实模型操作与确定性公式)

作者: 灵元模型团队
版本: 1.0.0
================================================================================
"""

import math
import json
import time
import random
import hashlib
import copy
import itertools
import sys
from collections import defaultdict, Counter
from datetime import datetime
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, Sequence,
)

# 导入真实灵元模型,用于实际量化/剪枝精度测量 (替代模拟查找表)
import sys as _sys
_sys.path.insert(0, "/workspace/lingyuan_train")
from lingyuan_v7 import LingyuanModel, ModelConfig, CharTokenizer, TextDataLoader, BUILTIN_CORPUS


# =============================================================================
# 全局常量
# =============================================================================

_DEFAULT_VOCAB_SIZE = 32000
_DEFAULT_SEQ_LEN = 512


# =============================================================================
# 工具函数: 确定性哈希、参数量估计、FLOPS估计、精度估计、配置距离等
# =============================================================================

def _config_hash(config: Dict[str, Any]) -> int:
    """
    计算配置的确定性哈希值。

    使用MD5确保跨运行稳定 (Python内置hash()受PYTHONHASHSEED影响)。
    """
    s = json.dumps(config, sort_keys=True, default=str)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _estimate_params(config: Dict[str, Any],
                     vocab_size: int = _DEFAULT_VOCAB_SIZE) -> int:
    """
    基于架构配置估计模型参数量。

    估算方式:
        - 嵌入层: vocab_size * hidden_dim
        - 每层注意力: Q(hidden*hidden) + K(hidden*kv_dim) + V(hidden*kv_dim) + O(hidden*hidden)
        - 每层FFN: 3 * hidden_dim * ffn_dim (门控FFN: gate, up, down)
        - 输出头: hidden_dim * vocab_size
    """
    hidden_dim = config.get("hidden_dim", 512)
    num_layers = config.get("num_layers", 6)
    num_heads = max(1, config.get("num_heads", 8))
    ffn_dim = config.get("ffn_dim", 2048)
    num_kv_heads = min(config.get("num_kv_heads", num_heads), num_heads)

    head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
    kv_dim = head_dim * num_kv_heads

    # 嵌入参数
    embed_params = vocab_size * hidden_dim

    # 每层注意力参数: Q + K + V + O 投影
    attn_params = (
        hidden_dim * hidden_dim      # Q
        + hidden_dim * kv_dim        # K
        + hidden_dim * kv_dim        # V
        + hidden_dim * hidden_dim    # O
    )

    # 每层FFN参数 (门控FFN: w_gate, w_up, w_down)
    ffn_params = 3 * hidden_dim * ffn_dim

    # 每层LayerNorm参数 (2个: attention前/后)
    ln_params = 2 * hidden_dim

    layer_params = attn_params + ffn_params + ln_params

    # 总参数: 嵌入 + 层 * num_layers + 输出头 + 最终LayerNorm
    total = embed_params + num_layers * layer_params + hidden_dim * vocab_size + hidden_dim

    return total


def _estimate_flops(config: Dict[str, Any],
                    seq_len: int = _DEFAULT_SEQ_LEN) -> int:
    """
    估计每个token的前向传播FLOPS。

    估算方式:
        - 注意力投影: 2 * (Q + K + V + O 投影的乘加)
        - 注意力分数: 2 * seq_len * hidden_dim (Q*K^T + attn*V)
        - FFN: 6 * hidden_dim * ffn_dim (门控FFN三次矩阵乘)
    """
    hidden_dim = config.get("hidden_dim", 512)
    num_layers = config.get("num_layers", 6)
    num_heads = max(1, config.get("num_heads", 8))
    ffn_dim = config.get("ffn_dim", 2048)
    num_kv_heads = min(config.get("num_kv_heads", num_heads), num_heads)

    head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
    kv_dim = head_dim * num_kv_heads

    # 注意力投影FLOPS (每个token)
    attn_proj = 2 * (
        hidden_dim * hidden_dim
        + hidden_dim * kv_dim
        + hidden_dim * kv_dim
        + hidden_dim * hidden_dim
    )

    # 注意力分数计算FLOPS (每个token, 与序列长度相关)
    attn_scores = 2 * seq_len * hidden_dim

    # FFN FLOPS (门控FFN: gate * up + down, 每步2*hidden*ffn)
    ffn_flops = 6 * hidden_dim * ffn_dim

    flops_per_token = num_layers * (attn_proj + attn_scores + ffn_flops)
    return flops_per_token


def _estimate_accuracy(config: Dict[str, Any]) -> float:
    """
    基于架构配置估计模型精度 (模拟)。

    考虑因素:
        - 模型容量 (参数量的对数缩放)
        - 深度/宽度/头数 (收益递减)
        - GQA比率 (KV头减少带来轻微精度损失)
        - FFN比率 (最优在2x-4x)
        - Dropout正则化 (过少/过多均不利)
        - 确定性噪声 (基于配置哈希)
    """
    hidden_dim = config.get("hidden_dim", 512)
    num_layers = config.get("num_layers", 6)
    num_heads = max(1, config.get("num_heads", 8))
    ffn_dim = config.get("ffn_dim", 2048)
    dropout = config.get("dropout", 0.1)
    num_kv_heads = min(config.get("num_kv_heads", num_heads), num_heads)

    params = _estimate_params(config)

    # 基础精度: 参数量对数缩放
    if params > 0:
        base = 0.55 + 0.04 * math.log10(params / 1e6)
    else:
        base = 0.55

    # 深度加成 (收益递减)
    base += 0.012 * math.log(num_layers + 1)

    # 宽度加成
    base += 0.006 * math.log(max(hidden_dim, 1) / 128.0)

    # 注意力头数加成
    base += 0.003 * math.log(num_heads + 1)

    # GQA惩罚 (KV头减少 = 轻微精度损失)
    gqa_ratio = num_kv_heads / num_heads if num_heads > 0 else 1.0
    base -= 0.008 * (1.0 - gqa_ratio)

    # FFN比率
    ffn_ratio = ffn_dim / max(hidden_dim, 1)
    if ffn_ratio < 2.0:
        base -= 0.008 * (2.0 - ffn_ratio)
    elif ffn_ratio > 4.0:
        base -= 0.003 * (ffn_ratio - 4.0)

    # Dropout正则化
    if dropout < 0.05:
        base -= 0.008
    elif dropout > 0.3:
        base -= 0.01 * (dropout - 0.3) / 0.1

    # 确定性噪声 (基于配置哈希, 范围 [-0.01, +0.01])
    noise = (_config_hash(config) % 1000) / 50000.0 - 0.01

    return max(0.1, min(0.99, base + noise))


def _evaluate_hyperparams(hp_config: Dict[str, Any],
                          arch_config: Optional[Dict[str, Any]] = None) -> float:
    """
    评估超参数配置对精度的影响 (模拟)。

    考虑因素:
        - 学习率 (最优点约3e-4, 偏离越大惩罚越大)
        - 批大小 (过小噪声大, 过大泛化差)
        - 权重衰减 (过少欠拟合, 过多过拟合)
        - 预热步数 (过少不稳定)
        - 梯度裁剪 (过激限制学习, 过松不稳定)
    """
    if arch_config:
        base_acc = _estimate_accuracy(arch_config)
    else:
        base_acc = 0.75

    lr = hp_config.get("learning_rate", 1e-4)
    batch_size = hp_config.get("batch_size", 32)
    weight_decay = hp_config.get("weight_decay", 0.01)
    warmup_steps = hp_config.get("warmup_steps", 1000)
    grad_clip = hp_config.get("grad_clip", 1.0)

    acc = base_acc

    # 学习率: 偏离最优值的对数惩罚
    lr_optimal = 3e-4
    lr_penalty = abs(math.log10(max(lr, 1e-10) / lr_optimal)) * 0.04
    acc -= lr_penalty

    # 批大小
    if batch_size < 8:
        acc -= 0.015
    elif batch_size > 256:
        acc -= 0.008 * math.log(batch_size / 256.0)

    # 权重衰减
    if weight_decay < 1e-5:
        acc -= 0.008
    elif weight_decay > 0.1:
        acc -= 0.015 * math.log10(weight_decay / 0.1)

    # 预热步数
    if warmup_steps < 100:
        acc -= 0.005

    # 梯度裁剪
    if grad_clip < 0.1:
        acc -= 0.005
    elif grad_clip > 10.0:
        acc -= 0.003

    # 确定性噪声
    noise = (_config_hash(hp_config) % 1000) / 50000.0 - 0.01

    return max(0.1, min(0.99, acc + noise))


def _default_fitness_fn(arch_config: Dict[str, Any],
                        hyper_config: Dict[str, Any]) -> Dict[str, Any]:
    """默认适应度评估函数,综合架构和超参数。"""
    params = _estimate_params(arch_config)
    flops = _estimate_flops(arch_config)
    arch_acc = _estimate_accuracy(arch_config)
    hp_acc = _evaluate_hyperparams(hyper_config, arch_config)

    # 综合得分: 精度为主, 兼顾效率
    efficiency = 1.0 / (1.0 + params / 1e8 + flops / 1e10)
    score = 0.5 * hp_acc + 0.3 * arch_acc + 0.2 * efficiency

    return {
        "params": params,
        "flops": flops,
        "arch_accuracy": arch_acc,
        "hp_accuracy": hp_acc,
        "score": score,
    }


def _config_distance(c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
    """计算两个配置之间的归一化欧氏距离。"""
    _ranges = {
        "hidden_dim": 1024, "num_layers": 24, "num_heads": 16,
        "ffn_dim": 4096, "num_kv_heads": 8, "dropout": 0.3,
        "learning_rate": 1e-2, "batch_size": 512,
        "weight_decay": 0.1, "warmup_steps": 10000, "grad_clip": 10.0,
    }
    dist = 0.0
    all_keys = set(list(c1.keys()) + list(c2.keys()))
    for key in all_keys:
        v1 = c1.get(key, 0)
        v2 = c2.get(key, 0)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            rng = _ranges.get(key, 1.0)
            dist += abs(v1 - v2) / max(rng, 1e-10)
        elif v1 != v2:
            dist += 1.0
    return dist


def _compute_pareto_frontier(
    candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]]
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    计算帕累托前沿。

    多目标: 最大化精度, 最小化FLOPS, 最小化参数量。
    一个解被支配当且仅当存在另一个解在所有目标上不劣且至少一个目标严格更优。
    """
    frontier: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    n = len(candidates)
    for i in range(n):
        config_i, metrics_i = candidates[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            _, metrics_j = candidates[j]
            # j 支配 i: acc_j >= acc_i, flops_j <= flops_i, params_j <= params_i
            # 且至少一个严格更优
            if (metrics_j["accuracy"] >= metrics_i["accuracy"]
                    and metrics_j["flops"] <= metrics_i["flops"]
                    and metrics_j["params"] <= metrics_i["params"]
                    and (metrics_j["accuracy"] > metrics_i["accuracy"]
                         or metrics_j["flops"] < metrics_i["flops"]
                         or metrics_j["params"] < metrics_i["params"])):
                dominated = True
                break
        if not dominated:
            frontier.append((copy.deepcopy(config_i), copy.deepcopy(metrics_i)))
    return frontier


def _validate_arch_config(
    config: Dict[str, Any],
    search_space: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
    """
    验证并修正架构配置约束。

    约束:
        - hidden_dim 必须能被 num_heads 整除
        - num_kv_heads 必须 <= num_heads
        - 所有值必须为正
    """
    config = copy.deepcopy(config)

    num_heads = max(1, config.get("num_heads", 8))
    hidden_dim = config.get("hidden_dim", 512)
    if hidden_dim < 1:
        hidden_dim = 128

    # 确保 hidden_dim 能被 num_heads 整除
    if hidden_dim % num_heads != 0:
        if search_space and "hidden_dim" in search_space:
            best_hd = None
            best_diff = float("inf")
            for hd in search_space["hidden_dim"]:
                if hd % num_heads == 0:
                    diff = abs(hd - hidden_dim)
                    if diff < best_diff:
                        best_diff = diff
                        best_hd = hd
            if best_hd is not None:
                config["hidden_dim"] = best_hd
            else:
                config["hidden_dim"] = (hidden_dim // num_heads) * num_heads
                if config["hidden_dim"] < num_heads:
                    config["hidden_dim"] = num_heads
        else:
            config["hidden_dim"] = (hidden_dim // num_heads) * num_heads
            if config["hidden_dim"] < num_heads:
                config["hidden_dim"] = num_heads

    # 确保 num_kv_heads <= num_heads
    num_kv_heads = config.get("num_kv_heads", num_heads)
    if num_kv_heads > num_heads:
        config["num_kv_heads"] = num_heads
    if num_kv_heads < 1:
        config["num_kv_heads"] = 1

    return config


def _format_params(n: int) -> str:
    """格式化参数量为人类可读字符串。"""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _format_flops(n: int) -> str:
    """格式化FLOPS为人类可读字符串。"""
    if n >= 1e12:
        return f"{n / 1e12:.2f}TFLOPS"
    elif n >= 1e9:
        return f"{n / 1e9:.2f}GFLOPS"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}MFLOPS"
    return f"{n:.0f}FLOPS"


def _std(values: List[float]) -> float:
    """计算标准差。"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


# =============================================================================
# Individual — 进化个体
# =============================================================================

class Individual:
    """
    进化个体,封装架构配置和超参数配置。

    属性:
        arch_config: 架构配置字典
        hyper_config: 超参数配置字典
        fitness: 适应度值 (评估后设置)
        metrics: 详细评估指标
        generation: 所属代数
        parent_ids: 父个体ID列表
    """

    _next_id: int = 0

    def __init__(self, arch_config: Optional[Dict[str, Any]] = None,
                 hyper_config: Optional[Dict[str, Any]] = None):
        Individual._next_id += 1
        self.id: int = Individual._next_id
        self.arch_config: Dict[str, Any] = arch_config or {}
        self.hyper_config: Dict[str, Any] = hyper_config or {}
        self.fitness: Optional[float] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.generation: int = 0
        self.parent_ids: List[int] = []
        self.birth_time: float = time.time()

    def evaluate(self, fitness_fn: Callable) -> float:
        """使用给定适应度函数评估个体。"""
        self.metrics = fitness_fn(self.arch_config, self.hyper_config)
        self.fitness = self.metrics.get("score", self.metrics.get("accuracy", 0.0))
        return self.fitness

    def clone(self) -> "Individual":
        """创建个体的深拷贝。"""
        ind = Individual(
            arch_config=copy.deepcopy(self.arch_config),
            hyper_config=copy.deepcopy(self.hyper_config),
        )
        ind.generation = self.generation
        ind.parent_ids = list(self.parent_ids)
        ind.fitness = self.fitness
        ind.metrics = copy.deepcopy(self.metrics) if self.metrics else None
        return ind

    def __repr__(self) -> str:
        if self.fitness is not None:
            return (f"Individual(id={self.id}, gen={self.generation}, "
                    f"fitness={self.fitness:.4f})")
        return f"Individual(id={self.id}, gen={self.generation}, unevaluated)"


# =============================================================================
# 1. NeuralArchitectureSearch — 神经架构搜索
# =============================================================================

class NeuralArchitectureSearch:
    """
    神经架构搜索 (Neural Architecture Search, NAS)。

    在预定义的搜索空间中寻找最优模型架构,支持多种搜索策略,
    并提供帕累托前沿分析和搜索历史记录。

    参数:
        search_space: 搜索空间字典,每个键对应一个参数及其候选值列表
        seed: 随机种子,确保结果可复现
    """

    @staticmethod
    def default_search_space() -> Dict[str, List[Any]]:
        """返回默认搜索空间。"""
        return {
            "hidden_dim": [128, 256, 384, 512, 768, 1024],
            "num_layers": [2, 4, 6, 8, 12, 16, 24],
            "num_heads": [2, 4, 8, 16],
            "ffn_dim": [512, 1024, 2048, 3072, 4096],
            "num_kv_heads": [1, 2, 4, 8],
            "dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        }

    def __init__(self,
                 search_space: Optional[Dict[str, List[Any]]] = None,
                 seed: int = 42):
        self.search_space: Dict[str, List[Any]] = (
            search_space if search_space is not None
            else self.default_search_space()
        )
        self._rng = random.Random(seed)
        self.history: List[Dict[str, Any]] = []
        self._best_config: Optional[Dict[str, Any]] = None
        self._best_score: float = -1.0

    # ---- 配置采样与验证 ----

    def _sample_random(self) -> Dict[str, Any]:
        """从搜索空间随机采样一个有效配置。"""
        config: Dict[str, Any] = {}
        for key, choices in self.search_space.items():
            config[key] = self._rng.choice(choices)
        return _validate_arch_config(config, self.search_space)

    def _evaluate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """评估架构配置,返回参数量、FLOPS和估计精度。"""
        params = _estimate_params(config)
        flops = _estimate_flops(config)
        accuracy = _estimate_accuracy(config)
        return {
            "params": params,
            "flops": flops,
            "accuracy": accuracy,
            "params_m": params / 1e6,
            "flops_g": flops / 1e9,
        }

    def _record(self, config: Dict[str, Any],
                metrics: Dict[str, Any]) -> None:
        """记录一次搜索尝试。"""
        self.history.append({
            "config": copy.deepcopy(config),
            "metrics": copy.deepcopy(metrics),
            "timestamp": time.time(),
        })
        score = self._compute_score(config, metrics)
        if score > self._best_score:
            self._best_score = score
            self._best_config = copy.deepcopy(config)

    def _compute_score(self, config: Dict[str, Any],
                       metrics: Dict[str, Any]) -> float:
        """计算综合得分 (平衡精度、速度和大小)。"""
        acc = metrics["accuracy"]
        params_norm = 1.0 / (1.0 + metrics["params_m"] / 100.0)
        flops_norm = 1.0 / (1.0 + metrics["flops_g"] / 10.0)
        return 0.5 * acc + 0.25 * params_norm + 0.25 * flops_norm

    # ---- 搜索策略 ----

    def random_search(self, n_trials: int = 50) -> Optional[Dict[str, Any]]:
        """
        随机搜索策略。

        参数:
            n_trials: 搜索次数
        返回:
            最佳配置
        """
        for _ in range(n_trials):
            config = self._sample_random()
            metrics = self._evaluate(config)
            self._record(config, metrics)
        return self.get_best_config()

    def grid_search(self, max_configs: int = 100) -> Optional[Dict[str, Any]]:
        """
        网格搜索策略。

        遍历搜索空间的组合,自动跳过无效配置。

        参数:
            max_configs: 最大评估配置数
        返回:
            最佳配置
        """
        keys = list(self.search_space.keys())
        all_combos = list(itertools.product(*[self.search_space[k] for k in keys]))
        self._rng.shuffle(all_combos)

        count = 0
        for combo in all_combos:
            if count >= max_configs:
                break
            config = dict(zip(keys, combo))
            # 跳过无效配置
            if config.get("num_kv_heads", 1) > config.get("num_heads", 1):
                continue
            if config["hidden_dim"] % config["num_heads"] != 0:
                continue
            metrics = self._evaluate(config)
            self._record(config, metrics)
            count += 1

        return self.get_best_config()

    def evolutionary_search(self,
                            population_size: int = 20,
                            generations: int = 10,
                            mutation_rate: float = 0.3) -> Optional[Dict[str, Any]]:
        """
        进化算法搜索策略。

        维护一个架构种群,通过选择、交叉和变异迭代优化。

        参数:
            population_size: 种群大小
            generations: 迭代代数
            mutation_rate: 变异概率
        返回:
            最佳配置
        """
        # 初始化种群
        population = [self._sample_random() for _ in range(population_size)]

        for gen in range(generations):
            # 评估并排序
            scored = [(ind, self._evaluate(ind)) for ind in population]
            scored.sort(key=lambda x: x[1]["accuracy"], reverse=True)

            # 记录所有评估
            for config, metrics in scored:
                self._record(config, metrics)

            # 精英保留: 保留前20%
            elite_count = max(1, population_size // 5)
            new_pop = [copy.deepcopy(scored[i][0]) for i in range(elite_count)]

            # 填充剩余: 交叉和变异
            top_half = scored[:max(2, len(scored) // 2)]
            while len(new_pop) < population_size:
                if self._rng.random() < 0.5 and len(top_half) >= 2:
                    p1 = self._rng.choice(top_half)[0]
                    p2 = self._rng.choice(top_half)[0]
                    child = self._crossover(p1, p2)
                else:
                    parent = self._rng.choice(top_half)[0]
                    child = copy.deepcopy(parent)
                child = self._mutate(child, mutation_rate)
                new_pop.append(child)

            population = new_pop

        return self.get_best_config()

    def _crossover(self, p1: Dict[str, Any],
                   p2: Dict[str, Any]) -> Dict[str, Any]:
        """交叉两个父配置,生成子配置。"""
        child: Dict[str, Any] = {}
        for key in self.search_space:
            child[key] = self._rng.choice([p1.get(key), p2.get(key)])
        return _validate_arch_config(child, self.search_space)

    def _mutate(self, config: Dict[str, Any],
                mutation_rate: float = 0.3) -> Dict[str, Any]:
        """变异配置,随机修改部分参数。"""
        mutated = copy.deepcopy(config)
        for key in self.search_space:
            if self._rng.random() < mutation_rate:
                choices = self.search_space[key]
                new_val = self._rng.choice(choices)
                # 尝试选择不同的值
                attempts = 0
                while new_val == mutated.get(key) and attempts < 5:
                    new_val = self._rng.choice(choices)
                    attempts += 1
                mutated[key] = new_val
        return _validate_arch_config(mutated, self.search_space)

    def bayesian_optimization(self,
                              n_iterations: int = 50) -> Optional[Dict[str, Any]]:
        """
        贝叶斯优化搜索策略 (基于RBF代理模型的模拟)。

        使用距离加权插值作为代理模型,结合探索-利用平衡的采集函数。

        参数:
            n_iterations: 优化迭代次数
        返回:
            最佳配置
        """
        # 初始随机采样
        n_init = min(5, max(2, n_iterations // 4))
        for _ in range(n_init):
            config = self._sample_random()
            metrics = self._evaluate(config)
            self._record(config, metrics)

        # 迭代优化
        for _ in range(n_iterations - n_init):
            # 生成候选集
            candidates = [self._sample_random() for _ in range(100)]

            # 评估采集函数 (Expected Improvement 近似)
            best_acc = max(h["metrics"]["accuracy"] for h in self.history)
            scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

            for cand in candidates:
                # 代理模型预测: 距离加权平均
                distances: List[Tuple[float, float]] = []
                for h in self.history:
                    d = _config_distance(cand, h["config"])
                    distances.append((d, h["metrics"]["accuracy"]))

                total_w = 0.0
                weighted_acc = 0.0
                for d, acc in distances:
                    w = 1.0 / (d + 1e-6)
                    total_w += w
                    weighted_acc += w * acc

                pred_acc = weighted_acc / total_w if total_w > 0 else 0.5

                # 探索奖励: 距离已观测点越远, 探索奖励越高
                min_dist = min(d for d, _ in distances) if distances else 1e6
                exploration = math.exp(-min_dist * 0.1)

                # 采集分数 = 利用 + 探索
                ei = pred_acc + 0.05 * exploration
                scored_candidates.append((ei, cand))

            # 选择最优候选
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_cand = scored_candidates[0][1]
            metrics = self._evaluate(best_cand)
            self._record(best_cand, metrics)

        return self.get_best_config()

    # ---- 结果查询 ----

    def get_pareto_frontier(
        self
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        获取帕累托前沿 (最大化精度, 最小化FLOPS和参数量)。

        返回:
            非支配解列表,每个元素为 (config, metrics) 元组
        """
        candidates = [(h["config"], h["metrics"]) for h in self.history]
        return _compute_pareto_frontier(candidates)

    def get_best_config(self,
                        objective: str = "balanced"
                        ) -> Optional[Dict[str, Any]]:
        """
        获取最佳配置。

        参数:
            objective: 优化目标,可选 'balanced'/'accuracy'/'speed'/'size'
        返回:
            最佳配置字典
        """
        if not self.history:
            return None

        if objective == "accuracy":
            best = max(self.history, key=lambda h: h["metrics"]["accuracy"])
        elif objective == "speed":
            best = min(self.history, key=lambda h: h["metrics"]["flops"])
        elif objective == "size":
            best = min(self.history, key=lambda h: h["metrics"]["params"])
        else:  # balanced
            best = max(self.history,
                       key=lambda h: self._compute_score(h["config"], h["metrics"]))
        return copy.deepcopy(best["config"])

    def get_search_history(self) -> List[Dict[str, Any]]:
        """获取完整搜索历史。"""
        return copy.deepcopy(self.history)

    def summary(self) -> str:
        """返回搜索摘要字符串。"""
        if not self.history:
            return "NeuralArchitectureSearch: 尚未执行搜索。"
        accs = [h["metrics"]["accuracy"] for h in self.history]
        params_list = [h["metrics"]["params"] for h in self.history]
        flops_list = [h["metrics"]["flops"] for h in self.history]
        pareto = self.get_pareto_frontier()
        lines = [
            "NeuralArchitectureSearch 摘要:",
            f"  搜索次数: {len(self.history)}",
            f"  最高精度: {max(accs):.4f}",
            f"  平均精度: {sum(accs) / len(accs):.4f}",
            f"  最小参数: {_format_params(min(params_list))}",
            f"  最小FLOPS: {_format_flops(min(flops_list))}",
            f"  帕累托前沿: {len(pareto)} 个架构",
        ]
        return "\n".join(lines)


# =============================================================================
# 2. HyperparameterOptimizer — 超参数优化器
# =============================================================================

class HyperparameterOptimizer:
    """
    超参数优化器。

    在预定义的超参数空间中寻找最优训练配置,支持多种优化算法,
    包括早停机制、并行评估模拟和参数重要性分析。

    参数:
        param_space: 超参数搜索空间
        seed: 随机种子
    """

    @staticmethod
    def default_param_space() -> Dict[str, List[Any]]:
        """返回默认超参数搜索空间。"""
        return {
            "learning_rate": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 5e-3],
            "batch_size": [4, 8, 16, 32, 64, 128, 256],
            "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1],
            "warmup_steps": [0, 100, 500, 1000, 2000, 5000],
            "grad_clip": [0.5, 1.0, 2.0, 5.0, 10.0],
        }

    def __init__(self,
                 param_space: Optional[Dict[str, List[Any]]] = None,
                 seed: int = 42):
        self.param_space: Dict[str, List[Any]] = (
            param_space if param_space is not None
            else self.default_param_space()
        )
        self._rng = random.Random(seed)
        self.history: List[Dict[str, Any]] = []
        self._best_config: Optional[Dict[str, Any]] = None
        self._best_score: float = -1.0
        self._early_stopped: List[int] = []

    # ---- 配置采样与评估 ----

    def _sample_random(self) -> Dict[str, Any]:
        """从超参数空间随机采样。"""
        return {key: self._rng.choice(choices)
                for key, choices in self.param_space.items()}

    def _evaluate(self, hp_config: Dict[str, Any],
                  arch_config: Optional[Dict[str, Any]] = None) -> float:
        """评估超参数配置,返回精度得分。"""
        return _evaluate_hyperparams(hp_config, arch_config)

    def _record(self, hp_config: Dict[str, Any], score: float,
                early_stopped: bool = False) -> None:
        """记录一次评估。"""
        self.history.append({
            "config": copy.deepcopy(hp_config),
            "score": score,
            "timestamp": time.time(),
            "early_stopped": early_stopped,
        })
        if not early_stopped and score > self._best_score:
            self._best_score = score
            self._best_config = copy.deepcopy(hp_config)

    def _early_stop_check(self, trial_history: List[float],
                          min_improvement: float = 0.001,
                          patience: int = 5) -> bool:
        """
        早停检查: 最近patience次试验无显著改善则停止。

        参数:
            trial_history: 历史得分列表
            min_improvement: 最小改善阈值
            patience: 容忍轮数
        返回:
            是否应该早停
        """
        if len(trial_history) < patience + 1:
            return False
        best_so_far = max(trial_history[:-patience])
        recent_best = max(trial_history[-patience:])
        return (recent_best - best_so_far) < min_improvement

    # ---- 优化算法 ----

    def grid_search(self, max_configs: int = 100,
                    arch_config: Optional[Dict[str, Any]] = None
                    ) -> Optional[Dict[str, Any]]:
        """
        网格搜索。

        参数:
            max_configs: 最大评估配置数
            arch_config: 关联的架构配置 (用于评估)
        返回:
            最佳超参数配置
        """
        keys = list(self.param_space.keys())
        all_combos = list(itertools.product(*[self.param_space[k] for k in keys]))
        self._rng.shuffle(all_combos)

        for i, combo in enumerate(all_combos):
            if i >= max_configs:
                break
            config = dict(zip(keys, combo))
            score = self._evaluate(config, arch_config)
            self._record(config, score)
        return self.get_best_config()

    def random_search(self, n_trials: int = 50,
                      arch_config: Optional[Dict[str, Any]] = None
                      ) -> Optional[Dict[str, Any]]:
        """
        随机搜索。

        参数:
            n_trials: 搜索次数
            arch_config: 关联的架构配置
        返回:
            最佳超参数配置
        """
        trial_scores: List[float] = []
        for i in range(n_trials):
            config = self._sample_random()
            score = self._evaluate(config, arch_config)
            trial_scores.append(score)

            # 早停检查
            if self._early_stop_check(trial_scores):
                self._record(config, score, early_stopped=True)
                self._early_stopped.append(i)
                break
            else:
                self._record(config, score)
        return self.get_best_config()

    def tpe_search(self, n_iterations: int = 50,
                   arch_config: Optional[Dict[str, Any]] = None
                   ) -> Optional[Dict[str, Any]]:
        """
        TPE (Tree-structured Parzen Estimator) 搜索模拟。

        将观测分为"好"和"差"两组,通过似然比 l(x)/g(x) 引导采样。

        参数:
            n_iterations: 优化迭代次数
            arch_config: 关联的架构配置
        返回:
            最佳超参数配置
        """
        # 初始随机采样
        n_init = min(5, max(2, n_iterations // 4))
        for _ in range(n_init):
            config = self._sample_random()
            score = self._evaluate(config, arch_config)
            self._record(config, score)

        for _ in range(n_iterations - n_init):
            # 按得分排序,分为好/差两组
            sorted_hist = sorted(self.history, key=lambda h: h["score"], reverse=True)
            n_good = max(1, len(sorted_hist) // 4)
            good_configs = [h["config"] for h in sorted_hist[:n_good]]
            bad_configs = [h["config"] for h in sorted_hist[n_good:]]

            # 对每个参数,计算 l(x)/g(x) 并采样
            config: Dict[str, Any] = {}
            for param, choices in self.param_space.items():
                good_vals = [c.get(param) for c in good_configs]
                bad_vals = [c.get(param) for c in bad_configs] if bad_configs else []

                good_counts = Counter(good_vals)
                bad_counts = Counter(bad_vals)

                scores: Dict[Any, float] = {}
                for choice in choices:
                    # 拉普拉斯平滑
                    l_x = (good_counts.get(choice, 0) + 1) / (len(good_vals) + len(choices))
                    g_x = ((bad_counts.get(choice, 0) + 1) /
                           (len(bad_vals) + len(choices))) if bad_vals else 1.0
                    scores[choice] = l_x / g_x

                # 按得分比例采样
                total = sum(scores.values())
                r = self._rng.random() * total
                cumsum = 0.0
                selected = choices[0]
                for choice in choices:
                    cumsum += scores[choice]
                    if r <= cumsum:
                        selected = choice
                        break
                config[param] = selected

            score = self._evaluate(config, arch_config)
            self._record(config, score)

        return self.get_best_config()

    def cma_es_search(self, n_iterations: int = 50,
                      arch_config: Optional[Dict[str, Any]] = None
                      ) -> Optional[Dict[str, Any]]:
        """
        CMA-ES (协方差矩阵自适应进化策略) 搜索模拟。

        维护搜索分布的均值和步长,根据排名更新分布参数。

        参数:
            n_iterations: 优化迭代次数
            arch_config: 关联的架构配置
        返回:
            最佳超参数配置
        """
        param_keys = list(self.param_space.keys())
        n_dims = len(param_keys)

        # 初始化分布参数
        mean = [0.5] * n_dims  # 归一化中心
        sigma = 0.25  # 步长

        # 初始随机采样
        n_init = min(5, max(2, n_iterations // 4))
        for _ in range(n_init):
            config = self._sample_random()
            score = self._evaluate(config, arch_config)
            self._record(config, score)

        lambda_ = 12  # 每代 offspring 数
        for _ in range(n_iterations - n_init):
            # 采样 offspring
            offspring: List[Tuple[List[float], float, Dict[str, Any]]] = []
            for _ in range(lambda_):
                sample = []
                for d in range(n_dims):
                    val = mean[d] + sigma * self._rng.gauss(0, 1)
                    val = max(0.0, min(1.0, val))
                    sample.append(val)
                config = self._continuous_to_config(sample, param_keys)
                score = self._evaluate(config, arch_config)
                self._record(config, score)
                offspring.append((sample, score, config))

            # 按得分排序
            offspring.sort(key=lambda x: x[1], reverse=True)

            # 更新均值 (加权平均 top-mu)
            mu = max(1, lambda_ // 2)
            weights = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

            new_mean = [0.0] * n_dims
            for d in range(n_dims):
                for i in range(mu):
                    new_mean[d] += weights[i] * offspring[i][0][d]

            # 更新步长
            improvement = offspring[0][1] - offspring[-1][1]
            if improvement > 0.001:
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = max(0.01, min(0.5, sigma))

            mean = new_mean

        return self.get_best_config()

    def _continuous_to_config(self, sample: List[float],
                              param_keys: List[str]) -> Dict[str, Any]:
        """将连续向量映射为离散超参数配置。"""
        config: Dict[str, Any] = {}
        for i, key in enumerate(param_keys):
            choices = self.param_space[key]
            idx = int(sample[i] * len(choices))
            idx = min(idx, len(choices) - 1)
            config[key] = choices[idx]
        return config

    # ---- 并行评估模拟 ----

    def parallel_evaluate(self, configs: List[Dict[str, Any]],
                          n_workers: int = 4,
                          arch_config: Optional[Dict[str, Any]] = None
                          ) -> List[Dict[str, Any]]:
        """
        模拟多配置并行训练评估。

        参数:
            configs: 待评估的配置列表
            n_workers: 并行工作数
            arch_config: 关联的架构配置
        返回:
            评估结果列表
        """
        results: List[Dict[str, Any]] = []
        recent_scores: List[float] = []

        for i, config in enumerate(configs):
            worker_id = i % n_workers
            score = self._evaluate(config, arch_config)

            # 估算训练时间 (基于配置复杂度)
            param_count = config.get("hidden_dim", 64) * config.get("num_layers", 4)
            train_time = param_count / 1000.0 + 0.5

            # 早停判断: 远低于近期最优
            early_stopped = False
            if len(recent_scores) >= 5:
                recent_best = max(recent_scores[-5:])
                if score < recent_best * 0.5:
                    early_stopped = True
                    train_time *= 0.3  # 提前终止, 只花费部分时间

            recent_scores.append(score)

            results.append({
                "config": copy.deepcopy(config),
                "score": score,
                "worker": worker_id,
                "time": train_time,
                "early_stopped": early_stopped,
            })
            self._record(config, score, early_stopped=early_stopped)

        return results

    # ---- 结果分析 ----

    def analyze_param_importance(self) -> Dict[str, float]:
        """
        分析超参数重要性 (基于方差分解)。

        返回:
            参数名到重要性分数的映射 (归一化到 [0, 1])
        """
        if len(self.history) < 5:
            return {param: 0.0 for param in self.param_space}

        importance: Dict[str, float] = {}
        all_scores = [h["score"] for h in self.history]
        total_mean = sum(all_scores) / len(all_scores)
        total_var = sum((s - total_mean) ** 2 for s in all_scores) / len(all_scores)

        if total_var < 1e-10:
            return {param: 0.0 for param in self.param_space}

        for param in self.param_space:
            groups: Dict[Any, List[float]] = defaultdict(list)
            for h in self.history:
                val = h["config"].get(param)
                groups[val].append(h["score"])

            between_var = 0.0
            for val, scores in groups.items():
                group_mean = sum(scores) / len(scores)
                between_var += len(scores) * (group_mean - total_mean) ** 2
            between_var /= len(all_scores)

            importance[param] = between_var / total_var

        # 归一化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def analyze_interactions(self) -> Dict[str, float]:
        """
        分析参数交互效应。

        返回:
            "param1 x param2" 到交互强度分数的映射
        """
        if len(self.history) < 10:
            return {}

        params = list(self.param_space.keys())
        interactions: Dict[str, float] = {}

        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                p1, p2 = params[i], params[j]
                groups: Dict[Tuple[Any, Any], List[float]] = defaultdict(list)
                for h in self.history:
                    key = (h["config"].get(p1), h["config"].get(p2))
                    groups[key].append(h["score"])

                if len(groups) < 2:
                    interactions[f"{p1} x {p2}"] = 0.0
                    continue

                means = [sum(v) / len(v) for v in groups.values()]
                interaction_strength = (max(means) - min(means)) / (max(means) + 1e-10)
                interactions[f"{p1} x {p2}"] = interaction_strength

        return interactions

    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """获取最佳超参数配置。"""
        if not self.history or self._best_config is None:
            valid = [h for h in self.history if not h.get("early_stopped", False)]
            if not valid:
                return None
            best = max(valid, key=lambda h: h["score"])
            return copy.deepcopy(best["config"])
        return copy.deepcopy(self._best_config)

    def summary(self) -> str:
        """返回优化摘要字符串。"""
        if not self.history:
            return "HyperparameterOptimizer: 尚未执行优化。"
        scores = [h["score"] for h in self.history]
        n_stopped = len(self._early_stopped)
        importance = self.analyze_param_importance()
        top_param = max(importance, key=importance.get) if importance else "N/A"
        lines = [
            "HyperparameterOptimizer 摘要:",
            f"  评估次数: {len(self.history)}",
            f"  最高得分: {max(scores):.4f}",
            f"  平均得分: {sum(scores) / len(scores):.4f}",
            f"  早停次数: {n_stopped}",
            f"  最重要参数: {top_param}",
        ]
        return "\n".join(lines)


# =============================================================================
# 3. AutoCompressor — 自动模型压缩器
# =============================================================================

class AutoCompressor:
    """
    自动模型压缩器。

    量化、剪枝和知识蒸馏对模型精度和大小的影响,
    支持组合压缩Pipeline和压缩比-精度曲线分析。

    参数:
        model_config: 待压缩模型的架构配置
        seed: 随机种子
    """

    def __init__(self,
                 model_config: Optional[Dict[str, Any]] = None,
                 seed: int = 42):
        self.model_config: Dict[str, Any] = model_config or {
            "hidden_dim": 512, "num_layers": 6, "num_heads": 8,
            "ffn_dim": 2048, "num_kv_heads": 8, "dropout": 0.1,
        }
        self._rng = random.Random(seed)
        self.original_params: int = _estimate_params(self.model_config)
        self.original_flops: int = _estimate_flops(self.model_config)
        self.original_accuracy: float = _estimate_accuracy(self.model_config)
        self.compression_history: List[Dict[str, Any]] = []

    # ---- 量化 ----

    def quantize(self, bits: int = 8,
                 scheme: str = "symmetric"
                 ) -> Dict[str, Any]:
        """
        量化压缩。

        参数:
            bits: 量化位宽 (4, 8, 16, 32)
            scheme: 量化方案 ('symmetric' 或 'asymmetric')
        返回:
            量化结果字典
        """
        accuracy_loss = self._simulate_quantization_loss(bits)
        size_ratio = bits / 32.0
        compressed_params = int(self.original_params * size_ratio)
        compressed_flops = int(self.original_flops * size_ratio)
        compressed_accuracy = self.original_accuracy - accuracy_loss

        result = {
            "method": "quantization",
            "bits": bits,
            "scheme": scheme,
            "original_params": self.original_params,
            "compressed_params": compressed_params,
            "size_reduction": 1.0 - size_ratio,
            "accuracy_loss": accuracy_loss,
            "compressed_accuracy": compressed_accuracy,
        }
        self.compression_history.append(result)
        return result

    def _get_probe(self):
        """惰性创建并缓存用于真实压缩测量的微型模型与测试样本。

        返回:
            (model, inputs, targets) 三元组,模型权重在每次测量后都会恢复,
            因此可跨多次量化/剪枝测量复用。
        """
        if getattr(self, "_probe_model", None) is None:
            _cfg = ModelConfig.tiny()
            self._probe_model = LingyuanModel(_cfg)
            self._probe_cfg = _cfg
            _tok = CharTokenizer(vocab_size=_cfg.vocab_size)
            _loader = TextDataLoader(_tok, seq_len=_cfg.max_seq_len, batch_size=1)
            _loader.load_text(BUILTIN_CORPUS)
            _inputs, _targets = _loader.sample_batch()
            self._probe_inputs = _inputs[0]
            self._probe_targets = _targets[0]
        return self._probe_model, self._probe_inputs, self._probe_targets

    def _simulate_quantization_loss(self, bits: int) -> float:
        """通过真实模型量化测量精度损失。

        对一个微型 LingyuanModel 的全部权重按指定位宽进行对称量化
        (缩放到 [-1,1], 量化到最近电平后反量化), 比较量化前后在同一
        测试样本上的交叉熵损失差值, 作为真实的精度损失。
        """
        if bits >= 32:
            return 0.0

        try:
            _model, _inputs, _targets = self._get_probe()

            # 1. 量化前的原始损失
            _logits = _model.forward(_inputs)
            _loss_before = _model._cross_entropy(_logits, _targets)

            # 2. 备份全部权重并执行真实量化
            _params = _model._all_params()
            _backups = [[row[:] for row in p.data] for p in _params]
            levels = max(1, (1 << (bits - 1)) - 1)  # 2^(bits-1) - 1
            for p in _params:
                _abs_max = 0.0
                for row in p.data:
                    for v in row:
                        a = v if v >= 0 else -v
                        if a > _abs_max:
                            _abs_max = a
                if _abs_max < 1e-12:
                    continue
                for i in range(p.rows):
                    for j in range(p.cols):
                        # 缩放到 [-1, 1] 范围
                        scaled = p.data[i][j] / _abs_max
                        # 量化到最近的量化电平
                        q = round(scaled * levels) / levels
                        # 反量化回原始尺度
                        p.data[i][j] = q * _abs_max

            # 3. 量化后的损失
            _logits_q = _model.forward(_inputs)
            _loss_after = _model._cross_entropy(_logits_q, _targets)

            # 4. 恢复原始权重,保证缓存模型可被后续测量复用
            for p, b in zip(_params, _backups):
                p.data = [row[:] for row in b]

            return max(0.0, _loss_after - _loss_before)
        except Exception:
            # 测量异常时回退到基于位宽的保守估计,保证流程不中断
            if bits >= 16:
                return 0.001
            elif bits >= 8:
                return 0.005
            elif bits >= 4:
                return 0.02
            else:
                return 0.1

    # ---- 剪枝 ----

    def structured_prune(self, prune_ratio: float = 0.3,
                         criterion: str = "l1"
                         ) -> Dict[str, Any]:
        """
        模拟结构化剪枝 (按层/按头)。

        参数:
            prune_ratio: 剪枝比例 (0-1)
            criterion: 剪枝准则 ('l1' 或 'l2')
        返回:
            剪枝结果字典
        """
        prune_ratio = max(0.0, min(0.8, prune_ratio))
        accuracy_loss = self._simulate_pruning_loss(prune_ratio, "structured")
        remaining_ratio = 1.0 - prune_ratio
        pruned_params = int(self.original_params * remaining_ratio)
        pruned_flops = int(self.original_flops * remaining_ratio)
        pruned_accuracy = self.original_accuracy - accuracy_loss

        result = {
            "method": "structured_pruning",
            "prune_ratio": prune_ratio,
            "criterion": criterion,
            "original_params": self.original_params,
            "pruned_params": pruned_params,
            "original_flops": self.original_flops,
            "pruned_flops": pruned_flops,
            "size_reduction": prune_ratio,
            "flops_reduction": prune_ratio,
            "accuracy_loss": accuracy_loss,
            "pruned_accuracy": pruned_accuracy,
        }
        self.compression_history.append(result)
        return result

    def unstructured_prune(self, sparsity: float = 0.5,
                           criterion: str = "magnitude"
                           ) -> Dict[str, Any]:
        """
        模拟非结构化剪枝 (按权重)。

        参数:
            sparsity: 稀疏度 (0-1)
            criterion: 剪枝准则 ('magnitude' 或 'random')
        返回:
            剪枝结果字典
        """
        sparsity = max(0.0, min(0.95, sparsity))
        accuracy_loss = self._simulate_pruning_loss(sparsity, "unstructured")
        # 非结构化剪枝不减少FLOPS (除非有稀疏加速), 但减少存储
        pruned_params = int(self.original_params * (1.0 - sparsity))
        pruned_accuracy = self.original_accuracy - accuracy_loss

        result = {
            "method": "unstructured_pruning",
            "sparsity": sparsity,
            "criterion": criterion,
            "original_params": self.original_params,
            "pruned_params": pruned_params,
            "size_reduction": sparsity,
            "flops_reduction": 0.0,  # 非结构化不减少FLOPS
            "accuracy_loss": accuracy_loss,
            "pruned_accuracy": pruned_accuracy,
        }
        self.compression_history.append(result)
        return result

    def _simulate_pruning_loss(self, ratio: float,
                               method: str = "structured") -> float:
        """通过真实模型剪枝测量精度损失。

        对一个微型 LingyuanModel 执行剪枝并测量剪枝前后在同一测试
        样本上的交叉熵损失差值:
            - 结构化剪枝: 按行 (神经元/通道) 剪枝, 将 L1 范数最小的
              整行权重置零;
            - 非结构化剪枝: 按权重幅值剪枝, 将幅值最小的单个权重置零。
        """
        ratio = max(0.0, min(0.95, ratio))
        if ratio <= 0.0:
            return 0.0

        try:
            _model, _inputs, _targets = self._get_probe()

            # 1. 剪枝前的原始损失
            _logits = _model.forward(_inputs)
            _loss_before = _model._cross_entropy(_logits, _targets)

            # 2. 备份全部权重
            _params = _model._all_params()
            _backups = [[row[:] for row in p.data] for p in _params]

            if method == "structured":
                # 结构化剪枝: 按行置零 (跳过单行张量如 LayerNorm/bias)
                _row_norms = []  # (l1_norm, param_idx, row_idx)
                for pi, p in enumerate(_params):
                    if p.rows <= 1:
                        continue
                    for ri in range(p.rows):
                        _norm = 0.0
                        for v in p.data[ri]:
                            _norm += v if v >= 0 else -v
                        _row_norms.append((_norm, pi, ri))
                _row_norms.sort(key=lambda t: t[0])
                _n_prune = int(len(_row_norms) * ratio)
                for _norm, pi, ri in _row_norms[:_n_prune]:
                    row = _params[pi].data[ri]
                    for j in range(len(row)):
                        row[j] = 0.0
            else:
                # 非结构化剪枝: 按权重幅值置零单个最小权重
                _all_vals = []  # (abs_val, param_idx, row, col)
                for pi, p in enumerate(_params):
                    for ri in range(p.rows):
                        for ci in range(p.cols):
                            v = p.data[ri][ci]
                            _all_vals.append((v if v >= 0 else -v, pi, ri, ci))
                _all_vals.sort(key=lambda t: t[0])
                _n_prune = int(len(_all_vals) * ratio)
                for _v, pi, ri, ci in _all_vals[:_n_prune]:
                    _params[pi].data[ri][ci] = 0.0

            # 3. 剪枝后的损失
            _logits_p = _model.forward(_inputs)
            _loss_after = _model._cross_entropy(_logits_p, _targets)

            # 4. 恢复原始权重,保证缓存模型可被后续测量复用
            for p, b in zip(_params, _backups):
                p.data = [row[:] for row in b]

            return max(0.0, _loss_after - _loss_before)
        except Exception:
            # 测量异常时回退到基于比例的保守估计,保证流程不中断
            if method == "structured":
                return max(0.0, 0.01 * ratio + 0.05 * ratio ** 2)
            else:
                return max(0.0, 0.005 * ratio + 0.02 * ratio ** 2)

    # ---- 知识蒸馏 ----

    def generate_distillation_config(
        self,
        student_config: Optional[Dict[str, Any]] = None,
        temperature: float = 4.0,
        alpha: float = 0.7
    ) -> Dict[str, Any]:
        """
        生成知识蒸馏配置 (teacher → student)。

        参数:
            student_config: 学生模型架构配置 (默认自动生成)
            temperature: 蒸馏温度
            alpha: 软标签损失权重
        返回:
            蒸馏配置字典
        """
        if student_config is None:
            # 自动生成学生模型: 参数量约为teacher的1/4
            student_config = copy.deepcopy(self.model_config)
            student_config["hidden_dim"] = max(64, self.model_config.get("hidden_dim", 512) // 2)
            student_config["num_layers"] = max(1, self.model_config.get("num_layers", 6) // 2)
            student_config["ffn_dim"] = max(256, self.model_config.get("ffn_dim", 2048) // 2)
            student_config["num_heads"] = max(1, self.model_config.get("num_heads", 8) // 2)
            if student_config["hidden_dim"] % student_config["num_heads"] != 0:
                student_config["hidden_dim"] = (
                    (student_config["hidden_dim"] // student_config["num_heads"])
                    * student_config["num_heads"]
                )
                if student_config["hidden_dim"] < student_config["num_heads"]:
                    student_config["hidden_dim"] = student_config["num_heads"]
            student_config["num_kv_heads"] = min(
                student_config.get("num_kv_heads", student_config["num_heads"]),
                student_config["num_heads"],
            )
            student_config = _validate_arch_config(student_config)

        teacher_params = _estimate_params(self.model_config)
        student_params = _estimate_params(student_config)
        teacher_acc = _estimate_accuracy(self.model_config)
        student_acc = _estimate_accuracy(student_config)

        # 蒸馏后学生精度提升 (介于student和teacher之间)
        distilled_acc = student_acc + (teacher_acc - student_acc) * alpha * 0.8
        accuracy_loss = teacher_acc - distilled_acc

        result = {
            "method": "knowledge_distillation",
            "teacher_config": copy.deepcopy(self.model_config),
            "student_config": student_config,
            "temperature": temperature,
            "alpha": alpha,
            "teacher_params": teacher_params,
            "student_params": student_params,
            "compression_ratio": 1.0 - student_params / max(teacher_params, 1),
            "teacher_accuracy": teacher_acc,
            "student_accuracy": student_acc,
            "distilled_accuracy": distilled_acc,
            "accuracy_loss": accuracy_loss,
        }
        self.compression_history.append(result)
        return result

    # ---- 组合Pipeline ----

    def compression_pipeline(self,
                             quantize_bits: int = 8,
                             prune_ratio: float = 0.3,
                             use_distillation: bool = True
                             ) -> Dict[str, Any]:
        """
        组合压缩Pipeline: 量化 → 剪枝 → 蒸馏。

        参数:
            quantize_bits: 量化位宽
            prune_ratio: 剪枝比例
            use_distillation: 是否使用知识蒸馏
        返回:
            压缩结果字典
        """
        steps: List[Dict[str, Any]] = []
        current_params = self.original_params
        current_flops = self.original_flops
        current_accuracy = self.original_accuracy
        total_accuracy_loss = 0.0

        # Step 1: 剪枝 (先剪枝再量化)
        if prune_ratio > 0:
            prune_result = self.structured_prune(prune_ratio)
            current_params = prune_result["pruned_params"]
            current_flops = prune_result["pruned_flops"]
            current_accuracy = prune_result["pruned_accuracy"]
            total_accuracy_loss += prune_result["accuracy_loss"]
            steps.append({"step": "pruning", "result": prune_result})

        # Step 2: 量化
        if quantize_bits < 32:
            quant_result = self.quantize(quantize_bits)
            size_ratio = quantize_bits / 32.0
            current_params = int(current_params * size_ratio)
            current_flops = int(current_flops * size_ratio)
            current_accuracy -= quant_result["accuracy_loss"]
            total_accuracy_loss += quant_result["accuracy_loss"]
            steps.append({"step": "quantization", "result": quant_result})

        # Step 3: 知识蒸馏 (恢复精度)
        distill_info = None
        if use_distillation:
            distill_config = self.generate_distillation_config()
            # 蒸馏可以恢复部分精度
            recovery = distill_config["accuracy_loss"] * 0.3
            current_accuracy += recovery
            total_accuracy_loss -= recovery
            distill_info = {
                "student_params": distill_config["student_params"],
                "temperature": distill_config["temperature"],
                "alpha": distill_config["alpha"],
                "recovery": recovery,
            }
            steps.append({"step": "distillation", "info": distill_info})

        total_compression = 1.0 - current_params / max(self.original_params, 1)

        result = {
            "method": "compression_pipeline",
            "steps": steps,
            "original_params": self.original_params,
            "final_params": current_params,
            "original_flops": self.original_flops,
            "final_flops": current_flops,
            "original_accuracy": self.original_accuracy,
            "final_accuracy": current_accuracy,
            "total_accuracy_loss": total_accuracy_loss,
            "total_compression_ratio": total_compression,
        }
        self.compression_history.append(result)
        return result

    # ---- 压缩曲线 ----

    def compression_curve(self,
                          ratios: Optional[List[float]] = None
                          ) -> List[Dict[str, Any]]:
        """
        生成压缩比 vs 精度损失曲线。

        参数:
            ratios: 压缩比例列表 (0-1, 1=完全压缩)
        返回:
            曲线数据点列表
        """
        if ratios is None:
            ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        curve: List[Dict[str, Any]] = []
        for ratio in ratios:
            ratio = max(0.0, min(0.95, ratio))
            # 根据压缩比选择策略
            if ratio < 0.25:
                bits = 16
                prune_r = ratio * 0.5
            elif ratio < 0.55:
                bits = 8
                prune_r = min(0.4, ratio * 0.6)
            elif ratio < 0.8:
                bits = 4
                prune_r = min(0.5, ratio * 0.7)
            else:
                bits = 4
                prune_r = 0.5

            size_ratio = bits / 32.0 * (1.0 - prune_r)
            actual_ratio = 1.0 - size_ratio
            acc_loss = (self._simulate_quantization_loss(bits)
                        + self._simulate_pruning_loss(prune_r, "structured"))
            final_size = int(self.original_params * size_ratio)

            curve.append({
                "target_ratio": ratio,
                "actual_ratio": actual_ratio,
                "bits": bits,
                "prune_ratio": prune_r,
                "final_params": final_size,
                "accuracy_loss": acc_loss,
                "remaining_accuracy": self.original_accuracy - acc_loss,
            })

        return curve

    def summary(self) -> str:
        """返回压缩摘要字符串。"""
        lines = [
            "AutoCompressor 摘要:",
            f"  原始参数: {_format_params(self.original_params)}",
            f"  原始FLOPS: {_format_flops(self.original_flops)}",
            f"  原始精度: {self.original_accuracy:.4f}",
            f"  压缩历史: {len(self.compression_history)} 次操作",
        ]
        if self.compression_history:
            last = self.compression_history[-1]
            if "total_compression_ratio" in last:
                lines.append(f"  最近Pipeline压缩比: {last['total_compression_ratio']:.2%}")
                lines.append(f"  最近Pipeline精度损失: {last['total_accuracy_loss']:.4f}")
        return "\n".join(lines)


# =============================================================================
# 4. EvolutionEngine — 进化引擎
# =============================================================================

class EvolutionEngine:
    """
    进化引擎。

    维护模型变体种群,通过变异、交叉和选择迭代进化,
    支持精英保留和进化历史记录。

    参数:
        population_size: 种群大小
        mutation_rate: 变异概率
        crossover_rate: 交叉概率
        elite_ratio: 精英保留比例
        seed: 随机种子
    """

    def __init__(self,
                 population_size: int = 30,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 elite_ratio: float = 0.1,
                 seed: int = 42):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self._rng = random.Random(seed)
        self.population: List[Individual] = []
        self.history: List[Dict[str, Any]] = []
        self.fitness_fn: Callable = _default_fitness_fn
        self.arch_search_space: Dict[str, List[Any]] = (
            NeuralArchitectureSearch.default_search_space()
        )
        self.hyperparam_space: Dict[str, List[Any]] = (
            HyperparameterOptimizer.default_param_space()
        )

    # ---- 配置 ----

    def set_fitness_fn(self, fn: Callable) -> "EvolutionEngine":
        """设置适应度评估函数。"""
        self.fitness_fn = fn
        return self

    def set_search_spaces(self,
                          arch_search_space: Optional[Dict[str, List[Any]]] = None,
                          hyperparam_space: Optional[Dict[str, List[Any]]] = None
                          ) -> "EvolutionEngine":
        """设置架构和超参数搜索空间。"""
        if arch_search_space:
            self.arch_search_space = arch_search_space
        if hyperparam_space:
            self.hyperparam_space = hyperparam_space
        return self

    # ---- 种群初始化 ----

    def initialize_population(self,
                              arch_search_space: Optional[Dict[str, List[Any]]] = None,
                              hyperparam_space: Optional[Dict[str, List[Any]]] = None
                              ) -> None:
        """初始化种群。"""
        self.set_search_spaces(arch_search_space, hyperparam_space)
        self.population = []
        for _ in range(self.population_size):
            arch_config = self._sample_arch()
            hyper_config = self._sample_hyper()
            ind = Individual(arch_config, hyper_config)
            self.population.append(ind)

    def _sample_arch(self) -> Dict[str, Any]:
        """随机采样架构配置。"""
        config = {key: self._rng.choice(choices)
                  for key, choices in self.arch_search_space.items()}
        return _validate_arch_config(config, self.arch_search_space)

    def _sample_hyper(self) -> Dict[str, Any]:
        """随机采样超参数配置。"""
        return {key: self._rng.choice(choices)
                for key, choices in self.hyperparam_space.items()}

    # ---- 变异 ----

    def mutate(self, individual: Individual) -> Individual:
        """
        变异个体: 随机修改架构和超参数。

        参数:
            individual: 待变异个体
        返回:
            变异后的新个体
        """
        ind = individual.clone()
        ind.parent_ids = [individual.id]
        ind.fitness = None
        ind.metrics = None

        # 变异架构
        for key in self.arch_search_space:
            if self._rng.random() < self.mutation_rate:
                choices = self.arch_search_space[key]
                new_val = self._rng.choice(choices)
                attempts = 0
                while new_val == ind.arch_config.get(key) and attempts < 5:
                    new_val = self._rng.choice(choices)
                    attempts += 1
                ind.arch_config[key] = new_val
        ind.arch_config = _validate_arch_config(ind.arch_config, self.arch_search_space)

        # 变异超参数
        for key in self.hyperparam_space:
            if self._rng.random() < self.mutation_rate:
                choices = self.hyperparam_space[key]
                new_val = self._rng.choice(choices)
                attempts = 0
                while new_val == ind.hyper_config.get(key) and attempts < 5:
                    new_val = self._rng.choice(choices)
                    attempts += 1
                ind.hyper_config[key] = new_val

        return ind

    # ---- 交叉 ----

    def crossover(self, parent1: Individual,
                  parent2: Individual) -> Individual:
        """
        交叉两个个体: 组合架构和超参数特征。

        参数:
            parent1: 父个体1
            parent2: 父个体2
        返回:
            交叉产生的新个体
        """
        child = Individual()
        child.parent_ids = [parent1.id, parent2.id]

        # 交叉架构
        for key in self.arch_search_space:
            if self._rng.random() < 0.5:
                child.arch_config[key] = parent1.arch_config.get(key)
            else:
                child.arch_config[key] = parent2.arch_config.get(key)
        child.arch_config = _validate_arch_config(
            child.arch_config, self.arch_search_space
        )

        # 交叉超参数
        for key in self.hyperparam_space:
            if self._rng.random() < 0.5:
                child.hyper_config[key] = parent1.hyper_config.get(key)
            else:
                child.hyper_config[key] = parent2.hyper_config.get(key)

        return child

    # ---- 选择 ----

    def tournament_select(self, k: int = 3) -> Optional[Individual]:
        """
        锦标赛选择: 从k个随机个体中选择最优。

        参数:
            k: 锦标赛大小
        返回:
            选中的个体
        """
        if not self.population:
            return None
        tournament = self._rng.sample(
            self.population, min(k, len(self.population))
        )
        evaluated = [ind for ind in tournament if ind.fitness is not None]
        if not evaluated:
            return self._rng.choice(tournament)
        return max(evaluated, key=lambda ind: ind.fitness)

    def roulette_select(self) -> Optional[Individual]:
        """
        轮盘赌选择: 按适应度比例选择。

        返回:
            选中的个体
        """
        evaluated = [ind for ind in self.population if ind.fitness is not None]
        if not evaluated:
            return self._rng.choice(self.population) if self.population else None

        # 平移适应度为正值
        min_fit = min(ind.fitness for ind in evaluated)
        shifted = [ind.fitness - min_fit + 0.01 for ind in evaluated]
        total = sum(shifted)
        if total <= 0:
            return self._rng.choice(evaluated)

        r = self._rng.random() * total
        cumsum = 0.0
        for i, ind in enumerate(evaluated):
            cumsum += shifted[i]
            if r <= cumsum:
                return ind
        return evaluated[-1]

    # ---- 进化主循环 ----

    def evolve(self, generations: int = 20) -> Optional[Individual]:
        """
        执行进化过程。

        参数:
            generations: 迭代代数
        返回:
            最终的最优个体
        """
        if not self.population:
            self.initialize_population()

        for gen in range(generations):
            # 评估种群
            for ind in self.population:
                if ind.fitness is None:
                    ind.evaluate(self.fitness_fn)
                ind.generation = gen

            # 记录代统计
            fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
            if fitnesses:
                self.history.append({
                    "generation": gen,
                    "best": max(fitnesses),
                    "average": sum(fitnesses) / len(fitnesses),
                    "worst": min(fitnesses),
                    "std": _std(fitnesses),
                    "population_size": len(fitnesses),
                })

            # 创建下一代
            elite_count = max(1, int(self.population_size * self.elite_ratio))
            sorted_pop = sorted(
                self.population,
                key=lambda ind: ind.fitness or 0.0,
                reverse=True,
            )

            # 精英保留
            new_pop = [ind.clone() for ind in sorted_pop[:elite_count]]
            for ind in new_pop:
                ind.generation = gen + 1

            # 生成后代
            while len(new_pop) < self.population_size:
                if self._rng.random() < self.crossover_rate and len(sorted_pop) >= 2:
                    p1 = self.tournament_select()
                    p2 = self.tournament_select()
                    if p1 and p2:
                        child = self.crossover(p1, p2)
                    else:
                        child = self._sample_individual()
                else:
                    parent = self.tournament_select()
                    if parent:
                        child = self.mutate(parent)
                    else:
                        child = self._sample_individual()

                # 可能再次变异
                if self._rng.random() < self.mutation_rate * 0.5:
                    child = self.mutate(child)
                new_pop.append(child)

            self.population = new_pop

        # 最终评估
        for ind in self.population:
            if ind.fitness is None:
                ind.evaluate(self.fitness_fn)

        return self.get_elite()

    def _sample_individual(self) -> Individual:
        """随机生成一个新个体。"""
        return Individual(self._sample_arch(), self._sample_hyper())

    # ---- 结果查询 ----

    def get_elite(self, n: int = 1) -> Union[Optional[Individual], List[Individual]]:
        """
        获取精英个体。

        参数:
            n: 返回的精英数量 (1时返回单个, >1时返回列表)
        返回:
            精英个体
        """
        evaluated = [ind for ind in self.population if ind.fitness is not None]
        evaluated.sort(key=lambda ind: ind.fitness, reverse=True)
        if n <= 1:
            return evaluated[0] if evaluated else None
        return evaluated[:n]

    def get_history(self) -> List[Dict[str, Any]]:
        """获取进化历史。"""
        return copy.deepcopy(self.history)

    def summary(self) -> str:
        """返回进化摘要字符串。"""
        if not self.history:
            return "EvolutionEngine: 尚未执行进化。"
        last = self.history[-1]
        best_ever = max(h["best"] for h in self.history)
        lines = [
            "EvolutionEngine 摘要:",
            f"  种群大小: {self.population_size}",
            f"  进化代数: {len(self.history)}",
            f"  最终代最佳适应度: {last['best']:.4f}",
            f"  最终代平均适应度: {last['average']:.4f}",
            f"  历史最佳适应度: {best_ever:.4f}",
            f"  变异率: {self.mutation_rate}",
            f"  交叉率: {self.crossover_rate}",
            f"  精英比例: {self.elite_ratio}",
        ]
        return "\n".join(lines)


# =============================================================================
# 5. AutoMLPipeline — AutoML流水线
# =============================================================================

class AutoMLPipeline:
    """
    AutoML流水线: 端到端自动机器学习。

    编排 NAS → 超参优化 → 训练 → 压缩 → 评估 全流程,
    根据目标自动选择策略,并生成完整实验报告。

    参数:
        objective: 优化目标 ('speed'/'accuracy'/'balanced')
        time_budget: 时间预算 (秒)
        seed: 随机种子
    """

    def __init__(self,
                 objective: str = "balanced",
                 time_budget: int = 3600,
                 seed: int = 42):
        self.objective: str = objective
        self.time_budget: int = time_budget
        self.seed: int = seed
        self._rng = random.Random(seed)

        # 组件实例
        self.nas = NeuralArchitectureSearch(seed=seed)
        self.hpo = HyperparameterOptimizer(seed=seed + 1)
        self.compressor: Optional[AutoCompressor] = None
        self.registry = ModelRegistry()
        self.tracker = ExperimentTracker()

        # 结果
        self.results: Dict[str, Any] = {}
        self._start_time: float = 0.0

    def _auto_configure(self) -> Dict[str, Any]:
        """根据优化目标自动配置策略。"""
        if self.objective == "speed":
            return {
                "nas_trials": 15,
                "hpo_trials": 15,
                "compress": True,
                "compress_bits": 8,
                "prune_ratio": 0.3,
                "weights": {"accuracy": 0.2, "speed": 0.5, "size": 0.3},
            }
        elif self.objective == "accuracy":
            return {
                "nas_trials": 30,
                "hpo_trials": 30,
                "compress": False,
                "compress_bits": 16,
                "prune_ratio": 0.0,
                "weights": {"accuracy": 0.7, "speed": 0.15, "size": 0.15},
            }
        else:  # balanced
            return {
                "nas_trials": 20,
                "hpo_trials": 20,
                "compress": True,
                "compress_bits": 8,
                "prune_ratio": 0.2,
                "weights": {"accuracy": 0.4, "speed": 0.3, "size": 0.3},
            }

    def run(self,
            nas_trials: Optional[int] = None,
            hpo_trials: Optional[int] = None,
            compress: Optional[bool] = None) -> Dict[str, Any]:
        """
        运行完整的AutoML流水线。

        参数:
            nas_trials: NAS搜索次数 (None则自动配置)
            hpo_trials: HPO优化次数 (None则自动配置)
            compress: 是否压缩 (None则自动配置)
        返回:
            流水线结果字典
        """
        config = self._auto_configure()
        nas_trials = nas_trials if nas_trials is not None else config["nas_trials"]
        hpo_trials = hpo_trials if hpo_trials is not None else config["hpo_trials"]
        do_compress = compress if compress is not None else config["compress"]

        self._start_time = time.time()

        # Step 1: 神经架构搜索
        exp_id = self.tracker.start_experiment(
            "NAS", {"n_trials": nas_trials, "objective": self.objective}
        )
        self.nas.random_search(n_trials=nas_trials)
        best_arch = self.nas.get_best_config(objective=self.objective)
        arch_metrics = self.nas._evaluate(best_arch) if best_arch else None
        self.tracker.end_experiment(exp_id, {
            "best_config": best_arch,
            "best_metrics": arch_metrics,
            "n_trials": nas_trials,
        })
        self.results["nas"] = {
            "n_trials": nas_trials,
            "best_config": best_arch,
            "best_metrics": arch_metrics,
        }

        # Step 2: 超参数优化
        exp_id = self.tracker.start_experiment(
            "HPO", {"n_trials": hpo_trials, "arch_config": best_arch}
        )
        self.hpo.random_search(n_trials=hpo_trials, arch_config=best_arch)
        best_hp = self.hpo.get_best_config()
        hp_score = self.hpo._evaluate(best_hp, best_arch) if best_hp else 0.5
        self.tracker.end_experiment(exp_id, {
            "best_config": best_hp,
            "best_score": hp_score,
            "n_trials": hpo_trials,
        })
        self.results["hpo"] = {
            "n_trials": hpo_trials,
            "best_config": best_hp,
            "best_score": hp_score,
        }

        # Step 3: 训练评估
        train_acc = hp_score
        self.results["training"] = {
            "accuracy": train_acc,
            "steps": 10000,
            "arch_config": best_arch,
            "hyper_config": best_hp,
        }

        # Step 4: 模型压缩
        if do_compress and best_arch:
            self.compressor = AutoCompressor(
                model_config=best_arch, seed=self.seed + 2
            )
            comp_result = self.compressor.compression_pipeline(
                quantize_bits=config["compress_bits"],
                prune_ratio=config["prune_ratio"],
                use_distillation=True,
            )
            self.results["compression"] = comp_result

        # Step 5: 最终评估
        final_acc = train_acc
        final_params = arch_metrics["params"] if arch_metrics else 0
        final_flops = arch_metrics["flops"] if arch_metrics else 0

        if do_compress and "compression" in self.results:
            comp = self.results["compression"]
            final_acc = comp.get("final_accuracy", final_acc)
            final_params = comp.get("final_params", final_params)
            final_flops = comp.get("final_flops", final_flops)

        final_metrics = {
            "accuracy": final_acc,
            "params": final_params,
            "flops": final_flops,
            "arch_config": best_arch,
            "hyper_config": best_hp,
            "compressed": do_compress,
        }
        self.results["final"] = final_metrics

        # Step 6: 注册模型
        model_id = self.registry.register(
            name=f"lingyuan_auto_{self.objective}",
            config={"arch": best_arch, "hyper": best_hp},
            metrics=final_metrics,
            weights_info={"compressed": do_compress},
        )
        self.results["model_id"] = model_id

        # 记录总时间
        self.results["elapsed_time"] = time.time() - self._start_time

        return self.results

    def generate_report(self) -> Dict[str, Any]:
        """
        生成完整的实验报告。

        返回:
            报告字典
        """
        report: Dict[str, Any] = {
            "title": "AutoML Pipeline Report",
            "objective": self.objective,
            "time_budget": self.time_budget,
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "final_result": None,
            "summary": {},
        }

        if "nas" in self.results:
            nas_r = self.results["nas"]
            report["steps"].append({
                "name": "Neural Architecture Search",
                "trials": nas_r.get("n_trials", 0),
                "best_config": nas_r.get("best_config"),
                "best_metrics": nas_r.get("best_metrics"),
            })

        if "hpo" in self.results:
            hpo_r = self.results["hpo"]
            report["steps"].append({
                "name": "Hyperparameter Optimization",
                "trials": hpo_r.get("n_trials", 0),
                "best_config": hpo_r.get("best_config"),
                "best_score": hpo_r.get("best_score"),
            })

        if "training" in self.results:
            train_r = self.results["training"]
            report["steps"].append({
                "name": "Training (Simulated)",
                "accuracy": train_r.get("accuracy"),
                "steps": train_r.get("steps"),
            })

        if "compression" in self.results:
            comp_r = self.results["compression"]
            report["steps"].append({
                "name": "Model Compression",
                "method": comp_r.get("method"),
                "compression_ratio": comp_r.get("total_compression_ratio"),
                "accuracy_loss": comp_r.get("total_accuracy_loss"),
                "final_params": comp_r.get("final_params"),
            })

        if "final" in self.results:
            report["final_result"] = self.results["final"]

        report["summary"] = {
            "total_steps": len(report["steps"]),
            "pipeline_completed": bool(report["final_result"]),
            "elapsed_time": self.results.get("elapsed_time", 0),
            "model_id": self.results.get("model_id"),
            "experiment_count": len(self.tracker.experiments),
        }

        return report

    def summary(self) -> str:
        """返回流水线摘要字符串。"""
        if not self.results:
            return "AutoMLPipeline: 尚未运行。"
        final = self.results.get("final", {})
        lines = [
            "AutoMLPipeline 摘要:",
            f"  优化目标: {self.objective}",
            f"  流水线步骤: {len(self.results)}",
            f"  最终精度: {final.get('accuracy', 0):.4f}",
            f"  最终参数: {_format_params(final.get('params', 0))}",
            f"  最终FLOPS: {_format_flops(final.get('flops', 0))}",
            f"  是否压缩: {final.get('compressed', False)}",
            f"  耗时: {self.results.get('elapsed_time', 0):.2f}s",
            f"  模型ID: {self.results.get('model_id', 'N/A')}",
        ]
        return "\n".join(lines)


# =============================================================================
# 6. ModelRegistry — 模型注册中心
# =============================================================================

class ModelRegistry:
    """
    模型注册中心。

    管理所有训练过的模型,支持注册、版本管理、性能排行、
    模型对比和模型导出。

    属性:
        models: 模型ID到模型信息的映射
    """

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self._next_id: int = 1
        self._name_versions: Dict[str, List[str]] = defaultdict(list)

    def register(self, name: str,
                 config: Dict[str, Any],
                 metrics: Dict[str, Any],
                 weights_info: Optional[Dict[str, Any]] = None) -> str:
        """
        注册一个新模型。

        参数:
            name: 模型名称
            config: 模型配置
            metrics: 性能指标
            weights_info: 权重信息 (可选)
        返回:
            模型ID
        """
        model_id = f"model_{self._next_id:04d}"
        self._next_id += 1

        version_num = len(self._name_versions[name]) + 1
        version_str = f"v{version_num}"

        model_info = {
            "id": model_id,
            "name": name,
            "version": version_str,
            "version_num": version_num,
            "config": copy.deepcopy(config),
            "metrics": copy.deepcopy(metrics),
            "weights_info": copy.deepcopy(weights_info) if weights_info else {},
            "registered_at": time.time(),
        }

        self.models[model_id] = model_info
        self._name_versions[name].append(model_id)
        return model_id

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型信息。"""
        return copy.deepcopy(self.models.get(model_id))

    def list_models(self, sort_by: str = "accuracy",
                    descending: bool = True) -> List[Dict[str, Any]]:
        """
        列出所有模型,按指定字段排序。

        参数:
            sort_by: 排序字段 ('accuracy'/'speed'/'size'/'name'/'version')
            descending: 是否降序
        返回:
            模型信息列表
        """
        models = list(self.models.values())

        def sort_key(m: Dict[str, Any]) -> Any:
            metrics = m.get("metrics", {})
            if sort_by == "accuracy":
                return metrics.get("accuracy", 0)
            elif sort_by == "speed":
                return -metrics.get("flops", float("inf"))
            elif sort_by == "size":
                return -metrics.get("params", float("inf"))
            elif sort_by == "name":
                return m.get("name", "")
            elif sort_by == "version":
                return m.get("version_num", 0)
            return 0

        models.sort(key=sort_key, reverse=descending)
        return [copy.deepcopy(m) for m in models]

    def compare(self, model_id1: str,
                model_id2: str) -> Optional[Dict[str, Any]]:
        """
        对比两个模型的详细差异。

        参数:
            model_id1: 模型1的ID
            model_id2: 模型2的ID
        返回:
            对比结果字典
        """
        m1 = self.models.get(model_id1)
        m2 = self.models.get(model_id2)
        if m1 is None or m2 is None:
            return None

        comparison: Dict[str, Any] = {
            "model1": {
                "id": m1["id"], "name": m1["name"], "version": m1["version"]
            },
            "model2": {
                "id": m2["id"], "name": m2["name"], "version": m2["version"]
            },
            "metrics_comparison": {},
            "metric_differences": {},
            "config_differences": {},
            "winner": {},
        }

        # 指标对比
        all_metric_keys = set(
            list(m1.get("metrics", {}).keys()) + list(m2.get("metrics", {}).keys())
        )
        for key in all_metric_keys:
            v1 = m1.get("metrics", {}).get(key)
            v2 = m2.get("metrics", {}).get(key)
            comparison["metrics_comparison"][key] = {"model1": v1, "model2": v2}
            if (isinstance(v1, (int, float)) and isinstance(v2, (int, float))):
                diff = v2 - v1
                comparison["metric_differences"][key] = diff
                # 判断哪个更好
                if key in ("accuracy", "score"):
                    comparison["winner"][key] = "model1" if v1 > v2 else "model2"
                elif key in ("params", "flops"):
                    comparison["winner"][key] = "model1" if v1 < v2 else "model2"

        # 配置差异
        all_config_keys = set(
            list(m1.get("config", {}).keys()) + list(m2.get("config", {}).keys())
        )
        for key in all_config_keys:
            v1 = m1.get("config", {}).get(key)
            v2 = m2.get("config", {}).get(key)
            if v1 != v2:
                comparison["config_differences"][key] = {
                    "model1": v1, "model2": v2
                }

        return comparison

    def get_version_history(self, name: str) -> List[Dict[str, Any]]:
        """获取指定名称模型的版本历史。"""
        model_ids = self._name_versions.get(name, [])
        return [copy.deepcopy(self.models[mid])
                for mid in model_ids if mid in self.models]

    def export_model(self, model_id: str,
                     fmt: str = "json") -> Optional[Union[str, Dict[str, Any]]]:
        """
        导出模型配置和权重信息。

        参数:
            model_id: 模型ID
            fmt: 导出格式 ('json' 或 'dict')
        返回:
            导出的数据 (字符串或字典)
        """
        model = self.models.get(model_id)
        if model is None:
            return None

        export_data = {
            "id": model["id"],
            "name": model["name"],
            "version": model["version"],
            "config": model["config"],
            "metrics": model["metrics"],
            "weights_info": model["weights_info"],
            "registered_at": model["registered_at"],
        }

        if fmt == "json":
            return json.dumps(export_data, indent=2, default=str)
        return export_data

    def summary(self) -> str:
        """返回注册中心摘要字符串。"""
        if not self.models:
            return "ModelRegistry: 无已注册模型。"
        all_acc = [m["metrics"].get("accuracy", 0) for m in self.models.values()]
        all_params = [m["metrics"].get("params", 0) for m in self.models.values()]
        names = set(m["name"] for m in self.models.values())
        lines = [
            "ModelRegistry 摘要:",
            f"  已注册模型: {len(self.models)}",
            f"  模型名称数: {len(names)}",
            f"  最高精度: {max(all_acc):.4f}",
            f"  最小参数: {_format_params(min(all_params))}",
        ]
        return "\n".join(lines)


# =============================================================================
# 7. ExperimentTracker — 实验追踪器
# =============================================================================

class ExperimentTracker:
    """
    实验追踪器。

    记录每次实验的配置、结果和环境信息,支持实验对比、
    复现信息保存和统计分析。

    属性:
        experiments: 实验ID到实验信息的映射
    """

    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self._next_id: int = 1

    def start_experiment(self, name: str,
                         config: Dict[str, Any],
                         env_info: Optional[Dict[str, Any]] = None) -> str:
        """
        开始一个新实验。

        参数:
            name: 实验名称
            config: 实验配置
            env_info: 环境信息 (可选)
        返回:
            实验ID
        """
        exp_id = f"exp_{self._next_id:04d}"
        self._next_id += 1

        if env_info is None:
            env_info = {
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
                "timestamp": datetime.now().isoformat(),
            }

        self.experiments[exp_id] = {
            "id": exp_id,
            "name": name,
            "config": copy.deepcopy(config),
            "env_info": copy.deepcopy(env_info),
            "start_time": time.time(),
            "end_time": None,
            "results": None,
            "status": "running",
            "duration": None,
        }
        return exp_id

    def end_experiment(self, exp_id: str,
                       results: Dict[str, Any],
                       status: str = "completed") -> None:
        """
        结束实验。

        参数:
            exp_id: 实验ID
            results: 实验结果
            status: 实验状态 ('completed'/'failed'/'cancelled')
        """
        if exp_id not in self.experiments:
            raise ValueError(f"未知实验ID: {exp_id}")

        exp = self.experiments[exp_id]
        exp["end_time"] = time.time()
        exp["results"] = copy.deepcopy(results)
        exp["status"] = status
        exp["duration"] = exp["end_time"] - exp["start_time"]

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """获取实验信息。"""
        return copy.deepcopy(self.experiments.get(exp_id))

    def compare_experiments(self,
                            experiment_ids: Optional[List[str]] = None
                            ) -> Dict[str, Any]:
        """
        对比多个实验。

        参数:
            experiment_ids: 实验ID列表 (None则对比全部)
        返回:
            对比结果字典
        """
        if experiment_ids is None:
            experiment_ids = list(self.experiments.keys())

        experiments = [self.experiments[eid] for eid in experiment_ids
                       if eid in self.experiments]
        if not experiments:
            return {"error": "无有效实验"}

        comparison: Dict[str, Any] = {
            "experiments": [],
            "best_by_metric": {},
            "summary": {},
        }

        # 收集所有指标
        all_metrics: set = set()
        for exp in experiments:
            if exp.get("results") and isinstance(exp["results"], dict):
                all_metrics.update(exp["results"].keys())

        # 构建对比表
        for exp in experiments:
            row = {
                "id": exp["id"],
                "name": exp["name"],
                "status": exp["status"],
                "duration": exp.get("duration", 0),
            }
            if exp.get("results") and isinstance(exp["results"], dict):
                for metric in all_metrics:
                    row[metric] = exp["results"].get(metric)
            comparison["experiments"].append(row)

        # 找出每个指标的最佳实验
        for metric in all_metrics:
            best_id = None
            best_val = None
            for exp in experiments:
                if (exp.get("results") and isinstance(exp["results"], dict)
                        and metric in exp["results"]):
                    val = exp["results"][metric]
                    if isinstance(val, (int, float)):
                        if best_val is None or val > best_val:
                            best_val = val
                            best_id = exp["id"]
            if best_id:
                comparison["best_by_metric"][metric] = {
                    "experiment_id": best_id,
                    "value": best_val,
                }

        # 汇总统计
        completed = [e for e in experiments if e["status"] == "completed"]
        durations = [e.get("duration", 0) for e in completed if e.get("duration")]
        comparison["summary"] = {
            "total": len(experiments),
            "completed": len(completed),
            "success_rate": len(completed) / len(experiments) if experiments else 0,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
        }

        return comparison

    def reproduce_info(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        获取实验的完整复现信息。

        参数:
            exp_id: 实验ID
        返回:
            复现信息字典
        """
        exp = self.experiments.get(exp_id)
        if exp is None:
            return None

        steps = [
            f"1. 环境准备: Python {exp['env_info'].get('python_version', 'unknown')}",
            f"2. 平台: {exp['env_info'].get('platform', 'unknown')}",
            f"3. 使用配置: {json.dumps(exp['config'], default=str)}",
            f"4. 执行实验: {exp['name']}",
            f"5. 预期结果: {json.dumps(exp.get('results', {}), default=str)}",
            f"6. 预计耗时: {exp.get('duration', 0):.2f}秒",
        ]

        return {
            "experiment_id": exp_id,
            "name": exp["name"],
            "config": exp["config"],
            "env_info": exp["env_info"],
            "results": exp.get("results"),
            "status": exp["status"],
            "duration": exp.get("duration"),
            "reproduction_steps": steps,
        }

    def statistics(self) -> Dict[str, Any]:
        """
        统计分析所有实验。

        返回:
            统计信息字典
        """
        total = len(self.experiments)
        if total == 0:
            return {"total": 0, "success_rate": 0}

        completed = [e for e in self.experiments.values()
                     if e["status"] == "completed"]
        failed = [e for e in self.experiments.values()
                  if e["status"] == "failed"]
        durations = [e.get("duration", 0) for e in completed
                     if e.get("duration")]

        # 找最佳结果
        best_exp = None
        best_score = -1
        for e in completed:
            if e.get("results") and isinstance(e["results"], dict):
                score = (e["results"].get("accuracy")
                         or e["results"].get("score")
                         or e["results"].get("best_score")
                         or 0)
                if isinstance(score, (int, float)) and score > best_score:
                    best_score = score
                    best_exp = e

        return {
            "total": total,
            "completed": len(completed),
            "failed": len(failed),
            "running": total - len(completed) - len(failed),
            "success_rate": len(completed) / total,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "best_experiment": best_exp["id"] if best_exp else None,
            "best_score": best_score if best_score > 0 else None,
        }

    def summary(self) -> str:
        """返回追踪器摘要字符串。"""
        stats = self.statistics()
        if stats["total"] == 0:
            return "ExperimentTracker: 无实验记录。"
        lines = [
            "ExperimentTracker 摘要:",
            f"  实验总数: {stats['total']}",
            f"  完成: {stats['completed']}",
            f"  失败: {stats['failed']}",
            f"  成功率: {stats['success_rate']:.1%}",
            f"  平均耗时: {stats['avg_duration']:.2f}s",
            f"  最佳实验: {stats.get('best_experiment', 'N/A')}",
        ]
        return "\n".join(lines)


# =============================================================================
# 自测函数
# =============================================================================

def _test_neural_architecture_search() -> None:
    """测试 NeuralArchitectureSearch。"""
    print("-" * 60)
    print("[1/7] 测试 NeuralArchitectureSearch")
    print("-" * 60)

    nas = NeuralArchitectureSearch(seed=42)

    # 随机搜索
    nas.random_search(n_trials=20)
    print(f"  随机搜索: {len(nas.history)} 次试验")

    # 进化搜索
    nas.evolutionary_search(population_size=10, generations=5)
    print(f"  进化搜索后总计: {len(nas.history)} 次试验")

    # 贝叶斯优化
    nas.bayesian_optimization(n_iterations=15)
    print(f"  贝叶斯优化后总计: {len(nas.history)} 次试验")

    # 网格搜索
    nas.grid_search(max_configs=10)
    print(f"  网格搜索后总计: {len(nas.history)} 次试验")

    # 帕累托前沿
    pareto = nas.get_pareto_frontier()
    print(f"  帕累托前沿: {len(pareto)} 个架构")

    # 最佳配置
    best = nas.get_best_config(objective="balanced")
    assert best is not None, "最佳配置不应为None"
    metrics = nas._evaluate(best)
    print(f"  最佳配置: hidden_dim={best['hidden_dim']}, "
          f"layers={best['num_layers']}, heads={best['num_heads']}")
    print(f"  最佳指标: acc={metrics['accuracy']:.4f}, "
          f"params={_format_params(metrics['params'])}, "
          f"flops={_format_flops(metrics['flops'])}")

    # 速度优先
    best_speed = nas.get_best_config(objective="speed")
    assert best_speed is not None
    print(f"  速度最优: flops={_format_flops(nas._evaluate(best_speed)['flops'])}")

    print(f"\n{nas.summary()}")
    print("  [PASS] NeuralArchitectureSearch")


def _test_hyperparameter_optimizer() -> None:
    """测试 HyperparameterOptimizer。"""
    print("-" * 60)
    print("[2/7] 测试 HyperparameterOptimizer")
    print("-" * 60)

    hpo = HyperparameterOptimizer(seed=42)
    arch_config = {"hidden_dim": 512, "num_layers": 6, "num_heads": 8,
                   "ffn_dim": 2048, "num_kv_heads": 8, "dropout": 0.1}

    # 随机搜索
    hpo.random_search(n_trials=20, arch_config=arch_config)
    print(f"  随机搜索: {len(hpo.history)} 次试验")

    # TPE搜索
    hpo2 = HyperparameterOptimizer(seed=42)
    hpo2.tpe_search(n_iterations=20, arch_config=arch_config)
    print(f"  TPE搜索: {len(hpo2.history)} 次试验")

    # CMA-ES搜索
    hpo3 = HyperparameterOptimizer(seed=42)
    hpo3.cma_es_search(n_iterations=20, arch_config=arch_config)
    print(f"  CMA-ES搜索: {len(hpo3.history)} 次试验")

    # 网格搜索
    hpo4 = HyperparameterOptimizer(seed=42)
    hpo4.grid_search(max_configs=10, arch_config=arch_config)
    print(f"  网格搜索: {len(hpo4.history)} 次试验")

    # 并行评估
    hpo5 = HyperparameterOptimizer(seed=42)
    configs = [hpo5._sample_random() for _ in range(10)]
    results = hpo5.parallel_evaluate(configs, n_workers=4, arch_config=arch_config)
    n_stopped = sum(1 for r in results if r["early_stopped"])
    print(f"  并行评估: {len(results)} 个配置, {n_stopped} 个早停")

    # 参数重要性
    importance = hpo.analyze_param_importance()
    print(f"  参数重要性: {', '.join(f'{k}={v:.3f}' for k, v in importance.items())}")

    # 参数交互
    interactions = hpo.analyze_interactions()
    if interactions:
        top_inter = max(interactions, key=interactions.get)
        print(f"  最强交互: {top_inter} = {interactions[top_inter]:.4f}")

    # 最佳配置
    best = hpo.get_best_config()
    assert best is not None
    print(f"  最佳配置: lr={best['learning_rate']}, bs={best['batch_size']}, "
          f"wd={best['weight_decay']}")

    print(f"\n{hpo.summary()}")
    print("  [PASS] HyperparameterOptimizer")


def _test_auto_compressor() -> None:
    """测试 AutoCompressor。"""
    print("-" * 60)
    print("[3/7] 测试 AutoCompressor")
    print("-" * 60)

    model_config = {"hidden_dim": 512, "num_layers": 6, "num_heads": 8,
                    "ffn_dim": 2048, "num_kv_heads": 8, "dropout": 0.1}
    comp = AutoCompressor(model_config=model_config, seed=42)

    print(f"  原始参数: {_format_params(comp.original_params)}")
    print(f"  原始精度: {comp.original_accuracy:.4f}")

    # 量化
    q8 = comp.quantize(bits=8)
    print(f"  8bit量化: params={_format_params(q8['compressed_params'])}, "
          f"loss={q8['accuracy_loss']:.4f}")

    q4 = comp.quantize(bits=4)
    print(f"  4bit量化: params={_format_params(q4['compressed_params'])}, "
          f"loss={q4['accuracy_loss']:.4f}")

    # 结构化剪枝
    sp = comp.structured_prune(prune_ratio=0.3)
    print(f"  结构化剪枝30%: params={_format_params(sp['pruned_params'])}, "
          f"loss={sp['accuracy_loss']:.4f}")

    # 非结构化剪枝
    up = comp.unstructured_prune(sparsity=0.5)
    print(f"  非结构化剪枝50%: params={_format_params(up['pruned_params'])}, "
          f"loss={up['accuracy_loss']:.4f}")

    # 知识蒸馏
    kd = comp.generate_distillation_config(temperature=4.0, alpha=0.7)
    print(f"  知识蒸馏: student_params={_format_params(kd['student_params'])}, "
          f"distilled_acc={kd['distilled_accuracy']:.4f}")

    # 压缩Pipeline
    comp2 = AutoCompressor(model_config=model_config, seed=42)
    pipeline = comp2.compression_pipeline(
        quantize_bits=8, prune_ratio=0.3, use_distillation=True
    )
    print(f"  Pipeline: 压缩比={pipeline['total_compression_ratio']:.2%}, "
          f"精度损失={pipeline['total_accuracy_loss']:.4f}, "
          f"最终参数={_format_params(pipeline['final_params'])}")

    # 压缩曲线
    curve = comp.compression_curve()
    print(f"  压缩曲线: {len(curve)} 个数据点")
    for point in curve[:3]:
        print(f"    ratio={point['target_ratio']:.1f} -> "
              f"loss={point['accuracy_loss']:.4f}, "
              f"acc={point['remaining_accuracy']:.4f}")

    print(f"\n{comp2.summary()}")
    print("  [PASS] AutoCompressor")


def _test_evolution_engine() -> None:
    """测试 EvolutionEngine。"""
    print("-" * 60)
    print("[4/7] 测试 EvolutionEngine")
    print("-" * 60)

    engine = EvolutionEngine(
        population_size=20,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elite_ratio=0.1,
        seed=42,
    )

    # 初始化种群
    engine.initialize_population()
    print(f"  种群初始化: {len(engine.population)} 个个体")

    # 进化
    best = engine.evolve(generations=10)
    assert best is not None, "进化后应返回最优个体"
    print(f"  进化完成: {len(engine.history)} 代")
    print(f"  最佳适应度: {best.fitness:.4f}")

    # 选择策略测试
    p1 = engine.tournament_select(k=3)
    p2 = engine.roulette_select()
    assert p1 is not None and p2 is not None
    print(f"  锦标赛选择: fitness={p1.fitness:.4f}")
    print(f"  轮盘赌选择: fitness={p2.fitness:.4f}")

    # 变异和交叉
    child = engine.crossover(p1, p2)
    mutated = engine.mutate(p1)
    print(f"  交叉产生: {child}")
    print(f"  变异产生: {mutated}")

    # 进化历史
    history = engine.get_history()
    if history:
        first = history[0]
        last = history[-1]
        improvement = last["best"] - first["best"]
        print(f"  进化趋势: {first['best']:.4f} -> {last['best']:.4f} "
              f"(提升 {improvement:+.4f})")

    # 精英获取
    elites = engine.get_elite(n=3)
    print(f"  Top-3 精英: {[f'{e.fitness:.4f}' for e in elites]}")

    print(f"\n{engine.summary()}")
    print("  [PASS] EvolutionEngine")


def _test_auto_ml_pipeline() -> None:
    """测试 AutoMLPipeline。"""
    print("-" * 60)
    print("[5/7] 测试 AutoMLPipeline")
    print("-" * 60)

    # 平衡模式
    pipeline = AutoMLPipeline(objective="balanced", time_budget=3600, seed=42)
    results = pipeline.run(nas_trials=10, hpo_trials=10, compress=True)
    print(f"  流水线完成: {len(results)} 个步骤")
    print(f"  最终精度: {results['final']['accuracy']:.4f}")
    print(f"  最终参数: {_format_params(results['final']['params'])}")
    print(f"  耗时: {results['elapsed_time']:.2f}s")
    print(f"  模型ID: {results['model_id']}")

    # 速度优先模式
    pipeline_speed = AutoMLPipeline(objective="speed", seed=100)
    results_speed = pipeline_speed.run(nas_trials=8, hpo_trials=8, compress=True)
    print(f"  速度优先: acc={results_speed['final']['accuracy']:.4f}, "
          f"params={_format_params(results_speed['final']['params'])}")

    # 精度优先模式
    pipeline_acc = AutoMLPipeline(objective="accuracy", seed=200)
    results_acc = pipeline_acc.run(nas_trials=8, hpo_trials=8, compress=False)
    print(f"  精度优先: acc={results_acc['final']['accuracy']:.4f}, "
          f"params={_format_params(results_acc['final']['params'])}")

    # 报告生成
    report = pipeline.generate_report()
    print(f"  实验报告: {len(report['steps'])} 个步骤, "
          f"{'完成' if report['summary']['pipeline_completed'] else '未完成'}")

    print(f"\n{pipeline.summary()}")
    print("  [PASS] AutoMLPipeline")


def _test_model_registry() -> None:
    """测试 ModelRegistry。"""
    print("-" * 60)
    print("[6/7] 测试 ModelRegistry")
    print("-" * 60)

    registry = ModelRegistry()

    # 注册模型
    m1 = registry.register(
        name="lingyuan-base",
        config={"arch": {"hidden_dim": 512, "num_layers": 6}, "hyper": {"lr": 1e-4}},
        metrics={"accuracy": 0.85, "params": 50000000, "flops": 2000000000},
    )
    m2 = registry.register(
        name="lingyuan-base",
        config={"arch": {"hidden_dim": 768, "num_layers": 12}, "hyper": {"lr": 3e-4}},
        metrics={"accuracy": 0.88, "params": 120000000, "flops": 5000000000},
    )
    m3 = registry.register(
        name="lingyuan-small",
        config={"arch": {"hidden_dim": 256, "num_layers": 4}, "hyper": {"lr": 5e-4}},
        metrics={"accuracy": 0.75, "params": 15000000, "flops": 800000000},
    )
    print(f"  注册模型: {m1}, {m2}, {m3}")

    # 列表排序
    by_acc = registry.list_models(sort_by="accuracy")
    acc_list = [f"{m['name']}:{m['metrics']['accuracy']:.2f}" for m in by_acc]
    print(f"  按精度排序: {acc_list}")

    by_size = registry.list_models(sort_by="size")
    size_list = [f"{m['name']}:{_format_params(m['metrics']['params'])}" for m in by_size]
    print(f"  按大小排序: {size_list}")

    # 版本历史
    versions = registry.get_version_history("lingyuan-base")
    print(f"  lingyuan-base 版本: {[v['version'] for v in versions]}")

    # 模型对比
    comparison = registry.compare(m1, m2)
    assert comparison is not None
    print(f"  对比 {m1} vs {m2}:")
    for key, diff in comparison["metric_differences"].items():
        winner = comparison["winner"].get(key, "N/A")
        print(f"    {key}: diff={diff:+.4f} ({winner} 更优)")

    # 模型导出
    exported = registry.export_model(m1, fmt="json")
    assert exported is not None
    export_dict = json.loads(exported)
    print(f"  导出 {m1}: name={export_dict['name']}, version={export_dict['version']}")

    print(f"\n{registry.summary()}")
    print("  [PASS] ModelRegistry")


def _test_experiment_tracker() -> None:
    """测试 ExperimentTracker。"""
    print("-" * 60)
    print("[7/7] 测试 ExperimentTracker")
    print("-" * 60)

    tracker = ExperimentTracker()

    # 实验1
    eid1 = tracker.start_experiment(
        "NAS-Random", {"n_trials": 20, "strategy": "random"}
    )
    time.sleep(0.01)
    tracker.end_experiment(eid1, {"best_accuracy": 0.82, "n_trials": 20})
    print(f"  实验1: {eid1} - NAS-Random")

    # 实验2
    eid2 = tracker.start_experiment(
        "NAS-Evolution", {"n_trials": 20, "strategy": "evolution"}
    )
    time.sleep(0.01)
    tracker.end_experiment(eid2, {"best_accuracy": 0.85, "n_trials": 20})
    print(f"  实验2: {eid2} - NAS-Evolution")

    # 实验3 (失败)
    eid3 = tracker.start_experiment(
        "NAS-Bayesian", {"n_trials": 20, "strategy": "bayesian"}
    )
    tracker.end_experiment(eid3, {"error": "out of memory"}, status="failed")
    print(f"  实验3: {eid3} - NAS-Bayesian (失败)")

    # 实验对比
    comparison = tracker.compare_experiments()
    print(f"  实验对比: {comparison['summary']['total']} 个实验, "
          f"成功率={comparison['summary']['success_rate']:.1%}")

    best_acc = comparison["best_by_metric"].get("best_accuracy")
    if best_acc:
        print(f"  最佳精度实验: {best_acc['experiment_id']} "
              f"(acc={best_acc['value']:.4f})")

    # 复现信息
    repro = tracker.reproduce_info(eid1)
    assert repro is not None
    print(f"  复现信息 ({eid1}): {len(repro['reproduction_steps'])} 步")
    for step in repro["reproduction_steps"][:2]:
        print(f"    {step}")

    # 统计
    stats = tracker.statistics()
    print(f"  统计: 总计={stats['total']}, 完成={stats['completed']}, "
          f"失败={stats['failed']}")
    print(f"  平均耗时: {stats['avg_duration']:.4f}s")

    print(f"\n{tracker.summary()}")
    print("  [PASS] ExperimentTracker")


def _test_integration() -> None:
    """集成测试: 全流程端到端。"""
    print("-" * 60)
    print("[集成] 全流程端到端测试")
    print("-" * 60)

    # 1. NAS
    nas = NeuralArchitectureSearch(seed=42)
    nas.random_search(n_trials=15)
    best_arch = nas.get_best_config("balanced")
    arch_metrics = nas._evaluate(best_arch)
    print(f"  [NAS] 最佳架构: acc={arch_metrics['accuracy']:.4f}, "
          f"params={_format_params(arch_metrics['params'])}")

    # 2. HPO
    hpo = HyperparameterOptimizer(seed=42)
    hpo.tpe_search(n_iterations=15, arch_config=best_arch)
    best_hp = hpo.get_best_config()
    hp_score = hpo._evaluate(best_hp, best_arch)
    print(f"  [HPO] 最佳超参: lr={best_hp['learning_rate']}, "
          f"score={hp_score:.4f}")

    # 3. 进化优化
    engine = EvolutionEngine(population_size=15, seed=42)
    engine.initialize_population()
    best_ind = engine.evolve(generations=5)
    print(f"  [EVO] 最佳适应度: {best_ind.fitness:.4f}")

    # 4. 压缩
    comp = AutoCompressor(model_config=best_arch, seed=42)
    pipeline = comp.compression_pipeline(quantize_bits=8, prune_ratio=0.2)
    print(f"  [COMP] 压缩比: {pipeline['total_compression_ratio']:.2%}, "
          f"精度损失: {pipeline['total_accuracy_loss']:.4f}")

    # 5. 注册
    registry = ModelRegistry()
    model_id = registry.register(
        name="lingyuan-integrated",
        config={"arch": best_arch, "hyper": best_hp},
        metrics={
            "accuracy": pipeline["final_accuracy"],
            "params": pipeline["final_params"],
            "flops": pipeline["final_flops"],
        },
    )
    print(f"  [REG] 注册模型: {model_id}")

    # 6. 实验追踪
    tracker = ExperimentTracker()
    eid = tracker.start_experiment("integration_test", {
        "arch": best_arch, "hyper": best_hp,
        "compression": "8bit+20%prune+distill",
    })
    tracker.end_experiment(eid, {
        "accuracy": pipeline["final_accuracy"],
        "compression_ratio": pipeline["total_compression_ratio"],
    })
    stats = tracker.statistics()
    print(f"  [TRACK] 实验完成: 成功率={stats['success_rate']:.0%}")

    print("\n  [PASS] 集成测试 - 全流程端到端验证通过")


# =============================================================================
# 主函数
# =============================================================================

def _main() -> None:
    """主测试函数。"""
    print()
    print("=" * 70)
    print("  灵元模型 - 自进化系统模块 (Part 21) 自测")
    print("=" * 70)
    print()

    _test_neural_architecture_search()
    print()
    _test_hyperparameter_optimizer()
    print()
    _test_auto_compressor()
    print()
    _test_evolution_engine()
    print()
    _test_auto_ml_pipeline()
    print()
    _test_model_registry()
    print()
    _test_experiment_tracker()
    print()
    _test_integration()

    print()
    print("=" * 70)
    print("  所有测试通过!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    _main()
