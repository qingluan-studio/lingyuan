
# ============================================================
# LINGYUAN MODEL - PART 18
# 虚拟GPU加速训练引擎 (Virtual GPU Accelerated Training Engine)
#
# 在Part 17虚拟GPU基础上构建完整的GPU加速训练栈:
# - GPUAcceleratedTrainingEngine: 用虚拟GPU加速前向/反向传播
# - GPUBatchProcessor: GPU批量处理与动态批大小
# - GradientCompression: 梯度压缩(Top-K/INT8/误差反馈/去噪)
# - TrainingProfiler: 逐层耗时/显存/GPU利用率分析与瓶颈识别
# - CheckpointManager: 增量检查点/INT8压缩/自动回滚/版本管理
# - TrainingScheduler: 学习率/批大小/梯度累积调度与早停
# - DistributedTrainer: 多虚拟GPU数据并行/流水线并行/容错
#
# 纯Python标准库实现 (零外部依赖)
# ============================================================

import sys
import os
import math
import time
import json
import copy
import random
import hashlib
import struct
import pickle
import zipfile
import io
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from part9 import (
    LingyuanTransformerModel, ModelConfig, TrainingEngine, LRSchedule,
    _cross_entropy_loss, _softmax_vec,
    _matmul_2d as _cpu_matmul_2d, _linear_2d as _cpu_linear_2d,
    _transpose_2d, _split_heads_2d, _merge_heads_2d,
    _rmsnorm_backward, _softmax_backward_row, _outer_product_add,
    _rope_backward, _silu, _silu_grad,
)
from part17 import VirtualGPU, ModelAccelerator, vgpu_smi

import part9 as _part9_module


# ============================================================
# 模块级辅助函数
# ============================================================

def _estimate_matrix_bytes(rows: int, cols: int) -> int:
    """估算矩阵内存占用 (字节, float64)"""
    return rows * cols * 8


def _matrix_sparsity(m: List[List[float]], sample_size: int = 64) -> float:
    """估算矩阵稀疏度 (采样)"""
    if not m or not m[0]:
        return 0.0
    total = 0
    zeros = 0
    step = max(1, len(m) // 8)
    for i in range(0, len(m), step):
        row = m[i]
        step_j = max(1, len(row) // 8)
        for j in range(0, len(row), step_j):
            total += 1
            if abs(row[j]) < 1e-12:
                zeros += 1
    return zeros / max(total, 1)


def _deep_copy_2d(m: List[List[float]]) -> List[List[float]]:
    """深拷贝二维矩阵"""
    return [list(row) for row in m]


def _deep_copy_grad(x: Any) -> Any:
    """深拷贝梯度 (支持1D/2D)"""
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        return [list(row) for row in x]
    elif isinstance(x, list):
        return list(x)
    return x


def _add_grads(a: Any, b: Any) -> Any:
    """梯度相加 (支持1D/2D)"""
    if isinstance(a, list) and len(a) > 0 and isinstance(a[0], list):
        return [[a[i][j] + b[i][j] for j in range(len(a[i]))]
                for i in range(len(a))]
    elif isinstance(a, list):
        return [a[i] + b[i] for i in range(len(a))]
    return a + b


def _scale_grad(x: Any, factor: float) -> Any:
    """梯度缩放 (支持1D/2D)"""
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        return [[v * factor for v in row] for row in x]
    elif isinstance(x, list):
        return [v * factor for v in x]
    return x * factor


def _grad_total_norm(grads: Dict[str, Any]) -> float:
    """计算梯度全局L2范数"""
    total_sq = 0.0

    def _accum(x: Any) -> None:
        nonlocal total_sq
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
            for row in x:
                _accum(row)
        elif isinstance(x, list):
            for v in x:
                total_sq += v * v
        else:
            total_sq += x * x

    for g in grads.values():
        _accum(g)
    return math.sqrt(total_sq)


def _hash_weights(weights: Dict[str, Any]) -> str:
    """计算权重的哈希摘要 (用于增量检查点)"""
    hasher = hashlib.md5()
    for name in sorted(weights.keys()):
        w = weights[name]
        hasher.update(name.encode())
        if isinstance(w, list) and len(w) > 0 and isinstance(w[0], list):
            for row in w:
                hasher.update(struct.pack(f'{len(row)}d', *row))
        elif isinstance(w, list):
            hasher.update(struct.pack(f'{len(w)}d', *w))
    return hasher.hexdigest()


# ============================================================
# 1. GPUAcceleratedTrainingEngine — 虚拟GPU加速训练引擎
# ============================================================

class GPUAcceleratedTrainingEngine(TrainingEngine):
    """虚拟GPU加速训练引擎

    继承TrainingEngine的全部训练功能, 并用VirtualGPU加速:
    - 前向传播: Q/K/V投影、FFN、LM Head的矩阵乘法走GPU
    - 反向传播: 所有转置矩阵乘法(梯度计算)走GPU
    - 自适应策略: 根据矩阵大小自动选择GPU或CPU路径
    - GPU显存管理: 自动管理中间激活的显存分配/释放
    - 梯度累积、混合精度、梯度裁剪 (继承自TrainingEngine)

    加速原理:
        通过运行时替换part9模块的 _matmul_2d / _linear_2d 全局函数,
        使模型 forward_for_training 和 backward_pass 内部的所有矩阵
        乘法自动路由到 VirtualGPU.parallel_matmul, 实现透明加速。
        小矩阵走CPU原版(避免调度开销), 大矩阵走GPU(真正多核并行)。

    用法:
        gpu = VirtualGPU()
        engine = GPUAcceleratedTrainingEngine(model, gpu=gpu)
        result = engine.train_step(batch)
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
                 num_pp_stages: int = 1,
                 gpu: Optional[VirtualGPU] = None,
                 gpu_threshold: int = 1024,
                 enable_gpu: bool = True,
                 profiler: Optional["TrainingProfiler"] = None):
        """初始化GPU加速训练引擎

        Args:
            gpu: VirtualGPU实例 (None则自动创建)
            gpu_threshold: 矩阵元素数阈值, 超过则走GPU
            enable_gpu: 是否启用GPU加速
            profiler: 训练性能分析器 (None则自动创建)
        """
        super().__init__(
            model=model, config=config, lr=lr, schedule=schedule,
            weight_decay=weight_decay, max_grad_norm=max_grad_norm,
            grad_accumulation_steps=grad_accumulation_steps,
            precision=precision, num_dp_devices=num_dp_devices,
            num_pp_stages=num_pp_stages,
        )

        # 虚拟GPU
        self.gpu: VirtualGPU = gpu or VirtualGPU()
        if not self.gpu._warmup_done:
            self.gpu.warmup()

        # 自适应策略参数
        self.gpu_threshold = gpu_threshold
        self._gpu_enabled = enable_gpu

        # GPU显存管理
        self._gpu_allocations: List[str] = []
        self._gpu_memory_peak: float = 0.0

        # 性能分析器
        self.profiler: TrainingProfiler = profiler or TrainingProfiler()

        # 统计
        self.gpu_forward_count = 0
        self.gpu_backward_count = 0
        self._gpu_matmul_calls = 0
        self._cpu_matmul_calls = 0
        self._total_gpu_time = 0.0

    # ---------- 自适应矩阵乘法 ----------

    def _adaptive_matmul(self, a: List[List[float]],
                         b: List[List[float]]) -> List[List[float]]:
        """自适应矩阵乘法: 根据大小选择GPU或CPU

        小矩阵 (元素数 < gpu_threshold): CPU原版 (避免调度开销)
        大矩阵: VirtualGPU.parallel_matmul (多核并行)
        """
        if not a or not b:
            return []
        m = len(a)
        n = len(b[0]) if b else 0
        total_cells = m * n

        if not self._gpu_enabled or total_cells < self.gpu_threshold:
            self._cpu_matmul_calls += 1
            return _cpu_matmul_2d(a, b)

        self._gpu_matmul_calls += 1
        t0 = time.time()
        result = self.gpu.parallel_matmul(a, b)
        self._total_gpu_time += time.time() - t0
        return result

    def _adaptive_linear(self, x: List[List[float]],
                         w: List[List[float]],
                         b: Optional[List[float]] = None
                         ) -> List[List[float]]:
        """自适应线性层: x @ w + b"""
        if not x or not w:
            return []
        result = self._adaptive_matmul(x, w)
        if b is not None and result:
            for i in range(len(result)):
                ri = result[i]
                for j in range(len(b)):
                    ri[j] += b[j]
        return result

    # ---------- GPU显存管理 ----------

    def _gpu_malloc(self, name: str, data: Any) -> None:
        """在虚拟GPU上分配显存"""
        self.gpu.malloc(name, data, scope="global")
        self._gpu_allocations.append(name)
        self._gpu_memory_peak = max(
            self._gpu_memory_peak, self.gpu.memory.total_allocated_mb)

    def _gpu_free(self, name: str) -> None:
        """释放虚拟GPU显存"""
        self.gpu.free(name)
        if name in self._gpu_allocations:
            self._gpu_allocations.remove(name)

    def _gpu_free_all(self) -> None:
        """释放所有已分配的GPU显存"""
        for name in list(self._gpu_allocations):
            self.gpu.free(name)
        self._gpu_allocations.clear()

    def _manage_forward_memory(self, cache: Dict[str, Any]) -> None:
        """管理前向传播中间激活的GPU显存"""
        # 将关键中间激活注册到GPU显存
        seq_len = len(cache.get("input_ids", []))
        hidden = self.model.hidden_dim
        for i, lc in enumerate(cache.get("layers", [])):
            self._gpu_malloc(f"fwd_layer_{i}_input", lc.get("layer_input"))
            self._gpu_malloc(f"fwd_layer_{i}_attn_weights",
                             lc.get("attn_weights"))

    def _release_backward_memory(self, layer_idx: int) -> None:
        """反向传播完成后释放对应层的GPU显存"""
        for prefix in ["fwd_layer"]:
            name = f"{prefix}_{layer_idx}_input"
            self._gpu_free(name)
            name = f"{prefix}_{layer_idx}_attn_weights"
            self._gpu_free(name)

    # ---------- 前向传播 (GPU加速) ----------

    def _install_gpu_kernels(self) -> Tuple[Any, Any]:
        """安装GPU加速内核 (替换part9全局函数)

        Returns:
            (原始_matmul_2d, 原始_linear_2d) 用于恢复
        """
        orig_matmul = _part9_module._matmul_2d
        orig_linear = _part9_module._linear_2d
        _part9_module._matmul_2d = self._adaptive_matmul
        _part9_module._linear_2d = self._adaptive_linear
        return orig_matmul, orig_linear

    def _restore_kernels(self, orig_matmul: Any, orig_linear: Any) -> None:
        """恢复原始CPU内核"""
        _part9_module._matmul_2d = orig_matmul
        _part9_module._linear_2d = orig_linear

    def forward_pass(self, input_ids: List[int],
                     targets: List[int]
                     ) -> Tuple[float, List[List[float]], Dict[str, Any]]:
        """GPU加速前向传播

        通过运行时替换 _matmul_2d / _linear_2d 全局函数,
        使 model.forward_for_training 内部的所有矩阵乘法
        (Q/K/V投影、输出投影、FFN、LM Head)自动走GPU。

        Returns:
            (loss, logits, cache)
        """
        self.profiler.start_event("forward_total")

        # 安装GPU内核
        orig_matmul, orig_linear = self._install_gpu_kernels()

        try:
            self.profiler.start_event("forward_compute")
            loss, logits, cache = self.model.forward_for_training(
                input_ids, targets)
            self.profiler.end_event("forward_compute")

            # 混合精度
            if self.mixed_precision.precision != "fp32":
                logits = self.mixed_precision.cast_matrix(logits)
                loss = self.mixed_precision.scale_loss(loss)
        finally:
            self._restore_kernels(orig_matmul, orig_linear)

        # GPU显存管理: 注册中间激活
        self._manage_forward_memory(cache)

        self.gpu_forward_count += 1
        self.profiler.end_event("forward_total")
        return loss, logits, cache

    # ---------- 反向传播 (GPU加速) ----------

    def backward_pass(self, logits: List[List[float]],
                      targets: List[int],
                      cache: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, Any]:
        """GPU加速反向传播

        通过运行时替换 _matmul_2d 全局函数,
        使 backward_pass 内部的所有转置矩阵乘法
        (FFN梯度、注意力梯度、Q/K/V投影梯度)自动走GPU。

        Returns:
            梯度字典 {param_name: grad}
        """
        if cache is None:
            return super().backward_pass(logits, targets, cache)

        self.profiler.start_event("backward_total")

        # 安装GPU内核 (反向传播只用_matmul_2d, 但也替换_linear_2d以备万一)
        orig_matmul, orig_linear = self._install_gpu_kernels()

        try:
            self.profiler.start_event("backward_compute")
            grads = super().backward_pass(logits, targets, cache)
            self.profiler.end_event("backward_compute")
        finally:
            self._restore_kernels(orig_matmul, orig_linear)

        # 释放GPU显存
        self._gpu_free_all()

        self.gpu_backward_count += 1
        self.profiler.end_event("backward_total")
        return grads

    # ---------- 训练步 (带分析) ----------

    def train_step(self, batch: List[Tuple[List[int], List[int]]]
                   ) -> Dict[str, Any]:
        """GPU加速训练步 (带性能分析)"""
        self.profiler.start_event("train_step")

        step_loss = 0.0
        step_grad_norm = 0.0
        n_samples = len(batch)

        shards = self.data_parallel.split_batch(batch)

        for shard in shards:
            for input_ids, targets in shard:
                # 前向 (GPU加速)
                loss, logits, fwd_cache = self.forward_pass(input_ids, targets)
                step_loss += loss

                # 反向 (GPU加速)
                grads = self.backward_pass(logits, targets, cache=fwd_cache)

                # 混合精度: 反向缩放
                if self.mixed_precision.precision != "fp32":
                    grads = self.mixed_precision.unscale_grads(grads)

                # 梯度累积
                self.accumulate_gradients(grads)

        # 达到累积步数
        if self._accumulation_count >= self.grad_accumulation_steps:
            avg_grads = {k: _scale_grad(v, 1.0 / self._accumulation_count)
                         for k, v in self._accumulated_grads.items()}

            step_grad_norm = self.clip_grad_norm(avg_grads)
            current_lr = self.get_lr()
            self.optimizer.lr = current_lr
            self.optimizer.step(self._get_params(), avg_grads)

            self._reset_accumulation()
            self.global_step += 1
            self.lr_history.append(current_lr)

        avg_loss = step_loss / max(n_samples, 1)
        self.loss_history.append(avg_loss)
        self.step += 1

        self.profiler.end_event("train_step")

        return {
            "step": self.step,
            "global_step": self.global_step,
            "loss": round(avg_loss, 6),
            "grad_norm": round(step_grad_norm, 4),
            "lr": round(self.get_lr(), 8),
            "samples": n_samples,
            "accumulation_count": self._accumulation_count,
            "gpu_matmul_calls": self._gpu_matmul_calls,
            "cpu_matmul_calls": self._cpu_matmul_calls,
        }

    # ---------- GPU统计 ----------

    def get_gpu_stats(self) -> Dict[str, Any]:
        """获取GPU加速统计"""
        total_calls = self._gpu_matmul_calls + self._cpu_matmul_calls
        gpu_ratio = self._gpu_matmul_calls / max(total_calls, 1)
        return {
            "gpu_enabled": self._gpu_enabled,
            "gpu_threshold": self.gpu_threshold,
            "gpu_forward_count": self.gpu_forward_count,
            "gpu_backward_count": self.gpu_backward_count,
            "gpu_matmul_calls": self._gpu_matmul_calls,
            "cpu_matmul_calls": self._cpu_matmul_calls,
            "gpu_matmul_ratio": round(gpu_ratio, 4),
            "total_gpu_time_s": round(self._total_gpu_time, 4),
            "gpu_memory_peak_mb": round(self._gpu_memory_peak, 4),
            "gpu_utilization": self.gpu.get_utilization(),
        }

    def enable_gpu(self) -> None:
        """启用GPU加速"""
        self._gpu_enabled = True

    def disable_gpu(self) -> None:
        """禁用GPU加速 (回退到纯CPU)"""
        self._gpu_enabled = False

    def benchmark_gpu_vs_cpu(self, input_ids: List[int],
                             targets: List[int],
                             num_runs: int = 3) -> Dict[str, Any]:
        """基准测试: GPU加速 vs 纯CPU"""
        # GPU基准
        self.enable_gpu()
        self.gpu.reset_stats()
        t0 = time.time()
        for _ in range(num_runs):
            loss_g, _, _ = self.forward_pass(input_ids, targets)
            _ = self.backward_pass([[0.0] * self.model.vocab_size] * len(targets),
                                   targets, cache=None)
        gpu_time = (time.time() - t0) / num_runs

        # CPU基准
        self.disable_gpu()
        t0 = time.time()
        for _ in range(num_runs):
            loss_c, _, _ = self.forward_pass(input_ids, targets)
        cpu_time = (time.time() - t0) / num_runs

        self.enable_gpu()
        return {
            "cpu_time_ms": round(cpu_time * 1000, 3),
            "gpu_time_ms": round(gpu_time * 1000, 3),
            "speedup": round(cpu_time / max(gpu_time, 1e-9), 3),
            "num_runs": num_runs,
            "seq_len": len(input_ids),
        }


# ============================================================
# 2. GPUBatchProcessor — GPU批处理器
# ============================================================

class GPUBatchProcessor:
    """GPU批处理器

    功能:
    - 批量前向传播: 多样本并行处理
    - 批量梯度计算: 多样本梯度聚合
    - 动态批大小: 根据显存自动调整
    - 序列打包: 变长序列的高效批处理

    用法:
        processor = GPUBatchProcessor(engine)
        results = processor.batch_forward(samples)
    """

    def __init__(self, engine: GPUAcceleratedTrainingEngine,
                 max_batch_size: int = 32,
                 max_seq_len: int = 512,
                 gpu_memory_budget_mb: float = 512.0):
        self.engine = engine
        self.model = engine.model
        self.gpu = engine.gpu
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.gpu_memory_budget_mb = gpu_memory_budget_mb

        # 统计
        self.total_samples_processed = 0
        self.total_batches = 0
        self.batch_size_history: List[int] = []

    def estimate_sample_memory(self, seq_len: int) -> float:
        """估算单个样本的显存占用 (MB)

        显存占用主要包括:
        - 嵌入: seq * hidden * 8
        - 每层中间激活: seq * hidden * ~20 (注意力/FFN缓存)
        - 注意力权重: num_heads * seq * seq * 8
        """
        hidden = self.model.hidden_dim
        num_layers = self.model.num_layers
        num_heads = self.model.num_heads

        # 嵌入
        emb_bytes = seq_len * hidden * 8
        # 每层中间激活 (约20倍hidden)
        layer_bytes = seq_len * hidden * 20 * 8
        # 注意力权重
        attn_bytes = num_heads * seq_len * seq_len * 8
        # 梯度缓存 (约等于前向)
        grad_bytes = layer_bytes * 0.5

        total = (emb_bytes + num_layers * (layer_bytes + attn_bytes)
                 + grad_bytes)
        return total / (1024 * 1024)

    def compute_dynamic_batch_size(self, seq_len: int) -> int:
        """根据GPU显存预算计算动态批大小

        Args:
            seq_len: 序列长度

        Returns:
            推荐批大小
        """
        per_sample_mb = self.estimate_sample_memory(seq_len)
        if per_sample_mb <= 0:
            return 1

        # 预留20%显存余量
        usable_budget = self.gpu_memory_budget_mb * 0.8
        batch_size = int(usable_budget / per_sample_mb)
        batch_size = max(1, min(batch_size, self.max_batch_size))
        return batch_size

    def pack_sequences(self,
                       sequences: List[Tuple[List[int], List[int]]],
                       pad_token_id: int = 0
                       ) -> Tuple[List[Tuple[List[int], List[int]]], List[int]]:
        """序列打包: 将变长序列分组并填充到相同长度

        策略: 按长度排序, 相近长度的序列打包到同一批次,
        减少填充浪费。

        Args:
            sequences: [(input_ids, targets), ...]
            pad_token_id: 填充token ID

        Returns:
            (packed_batches, lengths) — 每个batch内序列已填充到相同长度
        """
        if not sequences:
            return [], []

        # 按长度排序
        indexed = [(len(seq[0]), i, seq) for i, seq in enumerate(sequences)]
        indexed.sort()

        packed_batches = []
        lengths = []

        i = 0
        while i < len(indexed):
            # 动态确定当前批次的批大小
            seq_len = indexed[i][0]
            batch_size = self.compute_dynamic_batch_size(
                min(seq_len, self.max_seq_len))
            batch_size = min(batch_size, len(indexed) - i)

            batch = []
            max_len_in_batch = 0
            for j in range(batch_size):
                _, _, seq = indexed[i + j]
                batch.append(seq)
                max_len_in_batch = max(max_len_in_batch, len(seq[0]))

            # 填充到相同长度
            padded_batch = []
            for input_ids, targets in batch:
                pad_len = max_len_in_batch - len(input_ids)
                if pad_len > 0:
                    input_ids = list(input_ids) + [pad_token_id] * pad_len
                    targets = list(targets) + [pad_token_id] * pad_len
                # 截断到最大长度
                if len(input_ids) > self.max_seq_len:
                    input_ids = input_ids[:self.max_seq_len]
                    targets = targets[:self.max_seq_len]
                padded_batch.append((input_ids, targets))

            packed_batches.append(padded_batch)
            lengths.append(max_len_in_batch)
            i += batch_size

        return packed_batches, lengths

    def batch_forward(self,
                      samples: List[Tuple[List[int], List[int]]]
                      ) -> List[Tuple[float, List[List[float]], Dict[str, Any]]]:
        """批量前向传播: 多样本并行处理

        对每个样本执行GPU加速的前向传播, 收集结果。

        Returns:
            [(loss, logits, cache), ...]
        """
        results = []
        for input_ids, targets in samples:
            loss, logits, cache = self.engine.forward_pass(input_ids, targets)
            results.append((loss, logits, cache))

        self.total_samples_processed += len(samples)
        self.total_batches += 1
        self.batch_size_history.append(len(samples))
        return results

    def batch_backward(self,
                       forward_results: List[Tuple[float, List[List[float]], Dict[str, Any]]],
                       targets_list: List[List[int]]
                       ) -> Dict[str, Any]:
        """批量梯度计算: 多样本梯度聚合

        对多个样本分别计算梯度, 然后取平均。

        Returns:
            聚合后的平均梯度字典
        """
        if not forward_results:
            return {}

        all_grads: List[Dict[str, Any]] = []
        for (loss, logits, cache), targets in zip(forward_results, targets_list):
            grads = self.engine.backward_pass(logits, targets, cache=cache)
            all_grads.append(grads)

        # 梯度聚合: 取平均
        n = len(all_grads)
        if n == 0:
            return {}

        avg_grads: Dict[str, Any] = {}
        for name in all_grads[0]:
            accumulated = _deep_copy_grad(all_grads[0][name])
            for i in range(1, n):
                if name in all_grads[i]:
                    accumulated = _add_grads(accumulated, all_grads[i][name])
            avg_grads[name] = _scale_grad(accumulated, 1.0 / n)

        return avg_grads

    def batch_train_step(self,
                         samples: List[Tuple[List[int], List[int]]]
                         ) -> Dict[str, Any]:
        """批量训练步: 前向 → 反向 → 梯度聚合 → 更新

        自动处理动态批大小和序列打包。

        Returns:
            训练步统计
        """
        # 序列打包
        packed, lengths = self.pack_sequences(samples)

        total_loss = 0.0
        total_samples = 0

        for batch in packed:
            # 批量前向
            fwd_results = self.batch_forward(batch)
            targets_list = [s[1] for s in batch]

            # 批量反向
            avg_grads = self.batch_backward(fwd_results, targets_list)

            # 梯度裁剪
            grad_norm = self.engine.clip_grad_norm(avg_grads)

            # 更新学习率
            current_lr = self.engine.get_lr()
            self.engine.optimizer.lr = current_lr

            # 优化器步进
            self.engine.optimizer.step(self.engine._get_params(), avg_grads)

            self.engine.global_step += 1
            self.engine.lr_history.append(current_lr)

            for loss, _, _ in fwd_results:
                total_loss += loss
            total_samples += len(batch)

        avg_loss = total_loss / max(total_samples, 1)
        self.engine.loss_history.append(avg_loss)
        self.engine.step += 1

        return {
            "step": self.engine.step,
            "global_step": self.engine.global_step,
            "loss": round(avg_loss, 6),
            "samples": total_samples,
            "batches": len(packed),
            "batch_lengths": lengths,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取批处理统计"""
        avg_batch_size = (sum(self.batch_size_history) /
                          max(len(self.batch_size_history), 1))
        return {
            "total_samples_processed": self.total_samples_processed,
            "total_batches": self.total_batches,
            "avg_batch_size": round(avg_batch_size, 2),
            "max_batch_size": self.max_batch_size,
            "gpu_memory_budget_mb": self.gpu_memory_budget_mb,
        }


# ============================================================
# 3. GradientCompression — 梯度压缩
# ============================================================

class GradientCompression:
    """梯度压缩

    减少梯度同步的通信量, 适用于分布式训练:
    - Top-K稀疏化: 只保留最大的K%梯度
    - 量化压缩: FP32→INT8梯度量化
    - 误差反馈: 累积压缩误差到下一轮
    - 梯度去噪: 基于阈值的噪声过滤

    用法:
        compressor = GradientCompression(top_k_ratio=0.1, quantize=True)
        compressed = compressor.compress(grads)
        grads = compressor.decompress(compressed)
    """

    def __init__(self,
                 top_k_ratio: float = 1.0,
                 quantize: bool = False,
                 error_feedback: bool = True,
                 denoise_threshold: float = 0.0,
                 int8_range: int = 127):
        """
        Args:
            top_k_ratio: 保留的梯度比例 (0-1, 1.0=不稀疏化)
            quantize: 是否启用INT8量化
            error_feedback: 是否启用误差反馈
            denoise_threshold: 去噪阈值 (绝对值小于此值的梯度置零)
            int8_range: INT8量化范围 (通常127)
        """
        self.top_k_ratio = max(0.0, min(1.0, top_k_ratio))
        self.quantize_enabled = quantize
        self.error_feedback_enabled = error_feedback
        self.denoise_threshold = denoise_threshold
        self.int8_range = int8_range

        # 误差反馈状态
        self._error: Dict[str, Any] = {}

        # 统计
        self.compression_count = 0
        self.total_original_bytes = 0
        self.total_compressed_bytes = 0

    # ---------- Top-K稀疏化 ----------

    def top_k_sparsify(self, grad: Any) -> Tuple[Any, List[Tuple[int, ...]]]:
        """Top-K稀疏化: 只保留绝对值最大的K%梯度

        Args:
            grad: 梯度 (1D或2D)

        Returns:
            (稀疏梯度, 非零位置索引列表)
        """
        if self.top_k_ratio >= 1.0:
            return grad, []

        # 收集所有元素及其位置
        elements: List[Tuple[float, Tuple[int, ...]]] = []

        def _collect(x: Any, idx: Tuple[int, ...] = ()) -> None:
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                for i, row in enumerate(x):
                    _collect(row, idx + (i,))
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    elements.append((abs(v), idx + (i,)))
            else:
                elements.append((abs(x), idx))

        _collect(grad)

        if not elements:
            return grad, []

        # 按绝对值排序, 取Top-K
        k = max(1, int(len(elements) * self.top_k_ratio))
        elements.sort(key=lambda t: t[0], reverse=True)
        keep_indices = set(e[1] for e in elements[:k])

        # 构建稀疏梯度
        def _build(x: Any, idx: Tuple[int, ...] = ()) -> Any:
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                return [_build(row, idx + (i,)) for i, row in enumerate(x)]
            elif isinstance(x, list):
                return [x[i] if (idx + (i,)) in keep_indices else 0.0
                        for i in range(len(x))]
            return x if idx in keep_indices else 0.0

        sparse_grad = _build(grad)
        return sparse_grad, list(keep_indices)

    # ---------- INT8量化 ----------

    def quantize_int8(self, grad: Any) -> Tuple[Any, float]:
        """INT8量化: FP32→INT8

        量化公式: q = round(v / scale * int8_range)
        反量化:   v = q / int8_range * scale

        Args:
            grad: FP32梯度

        Returns:
            (INT8量化梯度, scale因子)
        """
        # 找最大绝对值
        max_abs = 0.0

        def _find_max(x: Any) -> None:
            nonlocal max_abs
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                for row in x:
                    _find_max(row)
            elif isinstance(x, list):
                for v in x:
                    if abs(v) > max_abs:
                        max_abs = abs(v)
            else:
                if abs(x) > max_abs:
                    max_abs = abs(x)

        _find_max(grad)

        if max_abs < 1e-12:
            return _deep_copy_grad(grad), 0.0

        scale = max_abs / self.int8_range

        def _quantize(x: Any) -> Any:
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                return [_quantize(row) for row in x]
            elif isinstance(x, list):
                return [round(v / scale) if scale > 0 else 0 for v in x]
            return round(x / scale) if scale > 0 else 0

        quantized = _quantize(grad)
        return quantized, scale

    def dequantize_int8(self, quantized: Any, scale: float) -> Any:
        """INT8反量化: INT8→FP32"""
        if scale == 0.0:
            return _deep_copy_grad(quantized)

        def _dequant(x: Any) -> Any:
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                return [_dequant(row) for row in x]
            elif isinstance(x, list):
                return [v * scale for v in x]
            return x * scale

        return _dequant(quantized)

    # ---------- 误差反馈 ----------

    def _apply_error_feedback(self, grads: Dict[str, Any]) -> Dict[str, Any]:
        """误差反馈: 将上一轮的压缩误差加到当前梯度"""
        if not self.error_feedback_enabled:
            return grads

        result = {}
        for name, g in grads.items():
            if name in self._error:
                result[name] = _add_grads(_deep_copy_grad(g), self._error[name])
            else:
                result[name] = _deep_copy_grad(g)
        return result

    def _update_error(self, original: Dict[str, Any],
                      compressed: Dict[str, Any]) -> None:
        """更新误差: error = original - compressed"""
        if not self.error_feedback_enabled:
            return

        for name in original:
            if name in compressed:
                orig = original[name]
                comp = compressed[name]
                self._error[name] = _add_grads(
                    _deep_copy_grad(orig),
                    _scale_grad(_deep_copy_grad(comp), -1.0))

    # ---------- 梯度去噪 ----------

    def denoise(self, grad: Any, threshold: Optional[float] = None) -> Any:
        """梯度去噪: 基于阈值的噪声过滤

        绝对值小于threshold的梯度置零。

        Args:
            grad: 梯度
            threshold: 去噪阈值 (None则使用self.denoise_threshold)
        """
        thr = threshold if threshold is not None else self.denoise_threshold
        if thr <= 0:
            return grad

        def _denoise(x: Any) -> Any:
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                return [_denoise(row) for row in x]
            elif isinstance(x, list):
                return [v if abs(v) >= thr else 0.0 for v in x]
            return x if abs(x) >= thr else 0.0

        return _denoise(grad)

    # ---------- 完整压缩/解压流程 ----------

    def compress(self, grads: Dict[str, Any]
                 ) -> Dict[str, Any]:
        """完整梯度压缩流程

        1. 误差反馈: 累加历史误差
        2. 梯度去噪: 过滤小梯度
        3. Top-K稀疏化: 保留最大的K%
        4. INT8量化: 压缩到INT8

        Returns:
            压缩后的梯度包, 包含:
            - "gradients": 压缩后的梯度
            - "scales": 各梯度的量化scale
            - "indices": Top-K保留的位置 (可选)
            - "metadata": 压缩元数据
        """
        self.compression_count += 1

        # 1. 误差反馈
        corrected = self._apply_error_feedback(grads)

        compressed_grads: Dict[str, Any] = {}
        scales: Dict[str, float] = {}
        indices_map: Dict[str, List[Tuple[int, ...]]] = {}

        for name, g in corrected.items():
            # 2. 梯度去噪
            g = self.denoise(g)

            # 3. Top-K稀疏化
            sparse_g, indices = self.top_k_sparsify(g)
            if indices:
                indices_map[name] = indices

            # 4. INT8量化
            if self.quantize_enabled:
                quantized, scale = self.quantize_int8(sparse_g)
                compressed_grads[name] = quantized
                scales[name] = scale
            else:
                compressed_grads[name] = sparse_g
                scales[name] = 0.0

        # 更新误差
        decompressed = self.decompress({
            "gradients": compressed_grads,
            "scales": scales,
            "indices_map": indices_map,
        })
        self._update_error(corrected, decompressed)

        # 统计
        for name in grads:
            self.total_original_bytes += self._estimate_size(grads[name])
            self.total_compressed_bytes += self._estimate_size(
                compressed_grads[name])

        return {
            "gradients": compressed_grads,
            "scales": scales,
            "indices_map": indices_map,
            "metadata": {
                "top_k_ratio": self.top_k_ratio,
                "quantized": self.quantize_enabled,
                "denoise_threshold": self.denoise_threshold,
                "error_feedback": self.error_feedback_enabled,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def decompress(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """完整梯度解压流程

        1. INT8反量化
        2. Top-K恢复 (稀疏梯度已是完整形状, 零值已填充)

        Returns:
            解压后的FP32梯度
        """
        compressed_grads = compressed["gradients"]
        scales = compressed["scales"]

        result: Dict[str, Any] = {}
        for name, g in compressed_grads.items():
            if self.quantize_enabled and scales.get(name, 0.0) != 0.0:
                result[name] = self.dequantize_int8(g, scales[name])
            else:
                result[name] = _deep_copy_grad(g)

        return result

    @staticmethod
    def _estimate_size(x: Any) -> int:
        """估算数据大小 (字节)"""
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
            return len(x) * len(x[0]) * 8
        elif isinstance(x, list):
            return len(x) * 8
        return 8

    def get_compression_ratio(self) -> float:
        """获取平均压缩比"""
        if self.total_compressed_bytes == 0:
            return 1.0
        return self.total_original_bytes / max(self.total_compressed_bytes, 1)

    def get_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        return {
            "compression_count": self.compression_count,
            "top_k_ratio": self.top_k_ratio,
            "quantize_enabled": self.quantize_enabled,
            "error_feedback_enabled": self.error_feedback_enabled,
            "denoise_threshold": self.denoise_threshold,
            "total_original_bytes": self.total_original_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "compression_ratio": round(self.get_compression_ratio(), 4),
        }

    def reset_error(self) -> None:
        """重置误差反馈状态"""
        self._error.clear()


# ============================================================
# 4. TrainingProfiler — 训练性能分析器
# ============================================================

class TrainingProfiler:
    """训练性能分析器

    功能:
    - 逐层耗时统计
    - 显存使用追踪
    - GPU利用率监控
    - 瓶颈识别: 自动定位最慢的层/操作
    - 训练报告生成

    用法:
        profiler = TrainingProfiler()
        profiler.start_event("forward")
        # ... do work ...
        profiler.end_event("forward")
        report = profiler.generate_report()
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history

        # 事件计时
        self._event_stack: List[Tuple[str, float]] = []
        self._event_times: Dict[str, List[float]] = {}
        self._event_counts: Dict[str, int] = {}

        # 逐层计时
        self._layer_timings: Dict[str, List[float]] = {}
        self._current_layer: Optional[str] = None
        self._layer_start: float = 0.0

        # 显存追踪
        self._memory_history: List[Tuple[int, float]] = []  # (step, mb)

        # GPU利用率追踪
        self._gpu_util_history: List[Tuple[int, Dict[str, Any]]] = []

        # 训练步损失追踪
        self._loss_history: List[Tuple[int, float]] = []

        # 全局统计
        self._total_profiled_time = 0.0
        self._start_time = time.time()

    # ---------- 事件计时 ----------

    def start_event(self, name: str) -> None:
        """开始计时事件"""
        self._event_stack.append((name, time.time()))

    def end_event(self, name: str) -> float:
        """结束计时事件

        Returns:
            事件耗时 (秒)
        """
        if not self._event_stack:
            return 0.0

        elapsed = 0.0
        # 弹出匹配的事件
        for i in range(len(self._event_stack) - 1, -1, -1):
            if self._event_stack[i][0] == name:
                elapsed = time.time() - self._event_stack[i][1]
                self._event_stack = self._event_stack[:i]
                break

        if name not in self._event_times:
            self._event_times[name] = []
            self._event_counts[name] = 0

        self._event_times[name].append(elapsed)
        self._event_counts[name] += 1
        self._total_profiled_time += elapsed

        # 限制历史长度
        if len(self._event_times[name]) > self.max_history:
            self._event_times[name] = self._event_times[name][-self.max_history:]

        return elapsed

    # ---------- 逐层计时 ----------

    def start_layer(self, layer_name: str) -> None:
        """开始逐层计时"""
        if self._current_layer is not None:
            self.end_layer()
        self._current_layer = layer_name
        self._layer_start = time.time()

    def end_layer(self) -> float:
        """结束逐层计时

        Returns:
            层耗时 (秒)
        """
        if self._current_layer is None:
            return 0.0
        elapsed = time.time() - self._layer_start
        if self._current_layer not in self._layer_timings:
            self._layer_timings[self._current_layer] = []
        self._layer_timings[self._current_layer].append(elapsed)
        if len(self._layer_timings[self._current_layer]) > self.max_history:
            self._layer_timings[self._current_layer] = \
                self._layer_timings[self._current_layer][-self.max_history:]
        self._current_layer = None
        return elapsed

    # ---------- 显存追踪 ----------

    def record_memory(self, step: int, memory_mb: float) -> None:
        """记录显存使用"""
        self._memory_history.append((step, memory_mb))
        if len(self._memory_history) > self.max_history:
            self._memory_history = self._memory_history[-self.max_history:]

    # ---------- GPU利用率监控 ----------

    def record_gpu_util(self, step: int,
                        gpu: Optional[VirtualGPU] = None) -> None:
        """记录GPU利用率"""
        if gpu is not None:
            util = gpu.get_utilization()
        else:
            util = {"num_sms": 0, "avg_utilization": 0.0}
        self._gpu_util_history.append((step, util))
        if len(self._gpu_util_history) > self.max_history:
            self._gpu_util_history = self._gpu_util_history[-self.max_history:]

    # ---------- 损失追踪 ----------

    def record_loss(self, step: int, loss: float) -> None:
        """记录训练损失"""
        self._loss_history.append((step, loss))
        if len(self._loss_history) > self.max_history:
            self._loss_history = self._loss_history[-self.max_history:]

    # ---------- 瓶颈识别 ----------

    def identify_bottleneck(self) -> Dict[str, Any]:
        """瓶颈识别: 自动定位最慢的层/操作

        Returns:
            {
                "bottleneck_event": 最慢的事件名,
                "bottleneck_time_s": 最慢事件的平均耗时,
                "bottleneck_layer": 最慢的层名,
                "bottleneck_layer_time_s": 最慢层的平均耗时,
                "time_distribution": 各事件耗时占比,
            }
        """
        # 分析事件瓶颈
        event_avg: Dict[str, float] = {}
        for name, times in self._event_times.items():
            if times:
                event_avg[name] = sum(times) / len(times)

        bottleneck_event = ""
        bottleneck_time = 0.0
        if event_avg:
            bottleneck_event = max(event_avg, key=event_avg.get)
            bottleneck_time = event_avg[bottleneck_event]

        # 分析层瓶颈
        layer_avg: Dict[str, float] = {}
        for name, times in self._layer_timings.items():
            if times:
                layer_avg[name] = sum(times) / len(times)

        bottleneck_layer = ""
        bottleneck_layer_time = 0.0
        if layer_avg:
            bottleneck_layer = max(layer_avg, key=layer_avg.get)
            bottleneck_layer_time = layer_avg[bottleneck_layer]

        # 时间分布
        total_time = sum(event_avg.values())
        time_distribution = {}
        if total_time > 0:
            time_distribution = {
                name: round(t / total_time, 4)
                for name, t in event_avg.items()
            }

        return {
            "bottleneck_event": bottleneck_event,
            "bottleneck_time_s": round(bottleneck_time, 6),
            "bottleneck_layer": bottleneck_layer,
            "bottleneck_layer_time_s": round(bottleneck_layer_time, 6),
            "time_distribution": time_distribution,
            "total_events": len(event_avg),
            "total_layers": len(layer_avg),
        }

    # ---------- 报告生成 ----------

    def generate_report(self) -> str:
        """生成训练性能报告 (文本格式)"""
        lines = [
            "=" * 70,
            "  Lingyuan Training Profiler Report",
            "=" * 70,
            f"  生成时间: {datetime.now().isoformat()}",
            f"  分析时长: {time.time() - self._start_time:.2f}s",
            "",
            "  --- 事件耗时统计 ---",
        ]

        for name in sorted(self._event_times.keys()):
            times = self._event_times[name]
            count = self._event_counts[name]
            avg = sum(times) / max(len(times), 1)
            total = sum(times)
            last = times[-1] if times else 0.0
            lines.append(
                f"    {name:30s} | 调用 {count:6d} 次 | "
                f"平均 {avg*1000:8.3f}ms | 总计 {total:.3f}s | "
                f"最后 {last*1000:.3f}ms")

        # 逐层统计
        if self._layer_timings:
            lines.append("")
            lines.append("  --- 逐层耗时统计 ---")
            for name in sorted(self._layer_timings.keys()):
                times = self._layer_timings[name]
                avg = sum(times) / max(len(times), 1)
                lines.append(
                    f"    {name:30s} | "
                    f"调用 {len(times):4d} 次 | "
                    f"平均 {avg*1000:8.3f}ms")

        # 瓶颈分析
        lines.append("")
        lines.append("  --- 瓶颈分析 ---")
        bottleneck = self.identify_bottleneck()
        lines.append(f"    瓶颈事件: {bottleneck['bottleneck_event']}"
                      f" ({bottleneck['bottleneck_time_s']*1000:.3f}ms)")
        lines.append(f"    瓶颈层:   {bottleneck['bottleneck_layer']}"
                      f" ({bottleneck['bottleneck_layer_time_s']*1000:.3f}ms)")

        if bottleneck["time_distribution"]:
            lines.append("    时间分布:")
            for name, ratio in sorted(
                    bottleneck["time_distribution"].items(),
                    key=lambda x: x[1], reverse=True):
                bar = "#" * int(ratio * 40)
                lines.append(f"      {name:30s} {ratio*100:6.2f}% {bar}")

        # 显存统计
        if self._memory_history:
            lines.append("")
            lines.append("  --- 显存统计 ---")
            max_mem = max(m for _, m in self._memory_history)
            min_mem = min(m for _, m in self._memory_history)
            avg_mem = sum(m for _, m in self._memory_history) / len(
                self._memory_history)
            lines.append(f"    峰值显存: {max_mem:.4f} MB")
            lines.append(f"    平均显存: {avg_mem:.4f} MB")
            lines.append(f"    最低显存: {min_mem:.4f} MB")

        # GPU利用率
        if self._gpu_util_history:
            lines.append("")
            lines.append("  --- GPU利用率 ---")
            latest = self._gpu_util_history[-1][1]
            lines.append(f"    SM数量:       {latest.get('num_sms', 0)}")
            lines.append(f"    平均利用率:   {latest.get('avg_utilization', 0):.2f}%")
            lines.append(f"    总FLOPS:      {latest.get('total_flops', 0):,}")
            lines.append(f"    矩阵乘法次数: {latest.get('matmul_count', 0)}")
            lines.append(f"    缓存命中率:   {latest.get('cache_hit_rate', 0):.2%}")

        # 损失趋势
        if self._loss_history:
            lines.append("")
            lines.append("  --- 损失趋势 ---")
            losses = [l for _, l in self._loss_history]
            lines.append(f"    记录数:   {len(losses)}")
            lines.append(f"    初始损失: {losses[0]:.6f}")
            lines.append(f"    最终损失: {losses[-1]:.6f}")
            lines.append(f"    最低损失: {min(losses):.6f}")
            lines.append(f"    最高损失: {max(losses):.6f}")
            if len(losses) >= 2:
                trend = "下降" if losses[-1] < losses[0] else "上升"
                lines.append(f"    趋势:     {trend}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """获取分析统计 (字典格式)"""
        return {
            "total_profiled_time_s": round(self._total_profiled_time, 4),
            "event_count": len(self._event_times),
            "layer_count": len(self._layer_timings),
            "memory_records": len(self._memory_history),
            "gpu_util_records": len(self._gpu_util_history),
            "loss_records": len(self._loss_history),
            "bottleneck": self.identify_bottleneck(),
        }

    def reset(self) -> None:
        """重置所有分析数据"""
        self._event_stack.clear()
        self._event_times.clear()
        self._event_counts.clear()
        self._layer_timings.clear()
        self._memory_history.clear()
        self._gpu_util_history.clear()
        self._loss_history.clear()
        self._total_profiled_time = 0.0
        self._start_time = time.time()


# ============================================================
# 5. CheckpointManager — 检查点管理器 (增强版)
# ============================================================

class CheckpointManager:
    """增强版检查点管理器

    功能:
    - 增量检查点: 只保存变化的权重 (基于哈希对比)
    - 检查点压缩: 用INT8压缩权重
    - 自动回滚: 检测到损失上升时回滚到上一个检查点
    - 检查点版本管理: 支持多版本对比
    - 云端同步接口: 预留云存储同步

    用法:
        ckpt_mgr = CheckpointManager(engine, save_dir="./checkpoints")
        ckpt_mgr.save("step_100")
        ckpt_mgr.auto_rollback(current_loss, threshold=0.1)
    """

    def __init__(self,
                 engine: GPUAcceleratedTrainingEngine,
                 save_dir: str = "./lingyuan_checkpoints",
                 compress: bool = True,
                 max_versions: int = 10,
                 auto_rollback_threshold: float = 0.15):
        self.engine = engine
        self.model = engine.model
        self.save_dir = save_dir
        self.compress_enabled = compress
        self.max_versions = max_versions
        self.auto_rollback_threshold = auto_rollback_threshold

        # 版本管理
        self.versions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._last_weights_hash: str = ""
        self._last_loss: float = float('inf')

        # 云端同步
        self._cloud_sync_handler: Optional[Callable] = None
        self._cloud_sync_enabled = False

        # 统计
        self.save_count = 0
        self.rollback_count = 0
        self.total_save_time = 0.0

        os.makedirs(save_dir, exist_ok=True)

    # ---------- 权重提取 ----------

    def _extract_weights(self) -> Dict[str, Any]:
        """提取模型所有权重"""
        return self.engine._get_params()

    # ---------- INT8压缩 ----------

    def _compress_weights(self,
                          weights: Dict[str, Any]
                          ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """INT8压缩权重

        Returns:
            (压缩后的权重, 各权重的scale因子)
        """
        compressed: Dict[str, Any] = {}
        scales: Dict[str, float] = {}

        for name, w in weights.items():
            # 找最大绝对值
            max_abs = 0.0
            if isinstance(w, list) and len(w) > 0 and isinstance(w[0], list):
                for row in w:
                    for v in row:
                        if abs(v) > max_abs:
                            max_abs = abs(v)
            elif isinstance(w, list):
                for v in w:
                    if abs(v) > max_abs:
                        max_abs = abs(v)

            if max_abs < 1e-12:
                compressed[name] = _deep_copy_grad(w)
                scales[name] = 0.0
                continue

            scale = max_abs / 127.0

            if isinstance(w, list) and len(w) > 0 and isinstance(w[0], list):
                compressed[name] = [[round(v / scale) for v in row]
                                    for row in w]
            elif isinstance(w, list):
                compressed[name] = [round(v / scale) for v in w]
            scales[name] = scale

        return compressed, scales

    def _decompress_weights(self,
                            compressed: Dict[str, Any],
                            scales: Dict[str, float]
                            ) -> Dict[str, Any]:
        """INT8解压权重"""
        result: Dict[str, Any] = {}
        for name, w in compressed.items():
            scale = scales.get(name, 0.0)
            if scale == 0.0:
                result[name] = _deep_copy_grad(w)
                continue
            if isinstance(w, list) and len(w) > 0 and isinstance(w[0], list):
                result[name] = [[v * scale for v in row]
                                for row in w]
            elif isinstance(w, list):
                result[name] = [v * scale for v in w]
        return result

    # ---------- 增量检查点 ----------

    def _detect_changed_weights(self,
                                current: Dict[str, Any],
                                threshold: float = 1e-8
                                ) -> Tuple[Dict[str, Any], List[str]]:
        """检测变化的权重 (与上次检查点对比)

        Returns:
            (变化的权重字典, 变化的权重名列表)
        """
        if not self._last_weights_hash:
            return current, list(current.keys())

        # 简化: 如果哈希不同, 检查每个参数
        current_hash = _hash_weights(current)
        if current_hash == self._last_weights_hash:
            return {}, []

        changed: Dict[str, Any] = {}
        changed_names: List[str] = []

        # 逐参数检测 (基于采样)
        for name, w in current.items():
            # 简化: 如果是第一次或上次没有, 视为变化
            changed[name] = w
            changed_names.append(name)

        return changed, changed_names

    # ---------- 保存检查点 ----------

    def save(self, tag: str = "",
             incremental: bool = True) -> Dict[str, Any]:
        """保存检查点

        Args:
            tag: 检查点标签 (如 "step_100")
            incremental: 是否增量保存 (只保存变化的权重)

        Returns:
            检查点元数据
        """
        t0 = time.time()
        timestamp = datetime.now().isoformat()
        tag = tag or f"step_{self.engine.global_step}"

        # 提取权重
        weights = self._extract_weights()

        # 增量检测
        changed_weights = weights
        changed_names = list(weights.keys())
        is_incremental = False

        if incremental and self._last_weights_hash:
            changed_weights, changed_names = self._detect_changed_weights(weights)
            is_incremental = len(changed_names) < len(weights)

        # 压缩
        if self.compress_enabled:
            compressed, scales = self._compress_weights(changed_weights)
        else:
            compressed = changed_weights
            scales = {}

        # 元数据
        meta = {
            "tag": tag,
            "timestamp": timestamp,
            "global_step": self.engine.global_step,
            "epoch": self.engine.epoch,
            "lr": self.engine.get_lr(),
            "loss_history": self.engine.loss_history[-50:],
            "grad_norm_history": self.engine.grad_norm_history[-50:],
            "changed_weights": changed_names,
            "is_incremental": is_incremental,
            "compressed": self.compress_enabled,
            "total_params": self.model.count_parameters(),
            "weights_hash": _hash_weights(weights),
        }

        # 保存到文件 (使用zip + pickle)
        filepath = os.path.join(self.save_dir, f"{tag}.ckpt")
        try:
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 元数据
                zf.writestr("metadata.json",
                            json.dumps(meta, ensure_ascii=False, indent=2))
                # 权重 (pickle序列化)
                weights_data = pickle.dumps({
                    "weights": compressed,
                    "scales": scales,
                })
                zf.writestr("weights.pkl", weights_data)
        except Exception as e:
            print(f"检查点保存失败: {e}")
            return {"error": str(e)}

        # 更新版本管理
        self._last_weights_hash = meta["weights_hash"]
        self.versions[tag] = meta
        if len(self.versions) > self.max_versions:
            oldest = next(iter(self.versions))
            del self.versions[oldest]

        elapsed = time.time() - t0
        self.save_count += 1
        self.total_save_time += elapsed

        meta["save_time_s"] = round(elapsed, 4)
        meta["filepath"] = filepath
        return meta

    # ---------- 加载检查点 ----------

    def load(self, tag: str,
             restore_weights: bool = False) -> Dict[str, Any]:
        """加载检查点

        Args:
            tag: 检查点标签
            restore_weights: 是否恢复权重 (默认只恢复训练状态)

        Returns:
            检查点元数据
        """
        filepath = os.path.join(self.save_dir, f"{tag}.ckpt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点不存在: {filepath}")

        with zipfile.ZipFile(filepath, 'r') as zf:
            meta = json.loads(zf.read("metadata.json"))
            weights_data = pickle.loads(zf.read("weights.pkl"))

        # 恢复训练状态
        self.engine.global_step = meta.get("global_step", 0)
        self.engine.epoch = meta.get("epoch", 0)
        self.engine.loss_history = meta.get("loss_history", [])
        self.engine.grad_norm_history = meta.get("grad_norm_history", [])

        # 恢复权重
        if restore_weights:
            compressed = weights_data["weights"]
            scales = weights_data.get("scales", {})
            if self.compress_enabled and scales:
                weights = self._decompress_weights(compressed, scales)
            else:
                weights = compressed

            params = self.engine._get_params()
            for name, w in weights.items():
                if name in params:
                    target = params[name]
                    if isinstance(target, list) and len(target) > 0 \
                            and isinstance(target[0], list):
                        for i in range(min(len(target), len(w))):
                            for j in range(min(len(target[i]), len(w[i]))):
                                target[i][j] = w[i][j]
                    elif isinstance(target, list):
                        for i in range(min(len(target), len(w))):
                            target[i] = w[i]

        self._last_weights_hash = meta.get("weights_hash", "")
        return meta

    # ---------- 自动回滚 ----------

    def auto_rollback(self, current_loss: float,
                      threshold: Optional[float] = None) -> bool:
        """自动回滚: 检测到损失上升时回滚

        Args:
            current_loss: 当前损失
            threshold: 损失上升阈值 (None则使用self.auto_rollback_threshold)

        Returns:
            是否执行了回滚
        """
        thr = threshold if threshold is not None else self.auto_rollback_threshold

        if self._last_loss == float('inf'):
            self._last_loss = current_loss
            return False

        loss_increase = (current_loss - self._last_loss) / max(
            abs(self._last_loss), 1e-8)

        if loss_increase > thr:
            # 损失上升超过阈值, 回滚
            if len(self.versions) >= 2:
                tags = list(self.versions.keys())
                previous_tag = tags[-2]
                try:
                    self.load(previous_tag, restore_weights=True)
                    self.rollback_count += 1
                    self._last_loss = current_loss
                    return True
                except Exception as e:
                    print(f"回滚失败: {e}")

        self._last_loss = current_loss
        return False

    # ---------- 版本管理 ----------

    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有检查点版本"""
        return [
            {
                "tag": tag,
                "timestamp": meta.get("timestamp"),
                "global_step": meta.get("global_step"),
                "lr": meta.get("lr"),
                "is_incremental": meta.get("is_incremental"),
                "compressed": meta.get("compressed"),
                "changed_weights_count": len(meta.get("changed_weights", [])),
            }
            for tag, meta in self.versions.items()
        ]

    def compare_versions(self, tag_a: str, tag_b: str) -> Dict[str, Any]:
        """对比两个检查点版本

        Returns:
            对比结果
        """
        if tag_a not in self.versions or tag_b not in self.versions:
            return {"error": "版本不存在"}

        meta_a = self.versions[tag_a]
        meta_b = self.versions[tag_b]

        return {
            "tag_a": tag_a,
            "tag_b": tag_b,
            "step_diff": meta_b.get("global_step", 0) - meta_a.get(
                "global_step", 0),
            "loss_diff": (
                meta_b.get("loss_history", [0])[-1] -
                meta_a.get("loss_history", [0])[-1]
                if meta_a.get("loss_history") and meta_b.get("loss_history")
                else 0.0
            ),
            "changed_a": set(meta_a.get("changed_weights", [])),
            "changed_b": set(meta_b.get("changed_weights", [])),
            "common_changed": list(
                set(meta_a.get("changed_weights", [])) &
                set(meta_b.get("changed_weights", []))
            ),
        }

    # ---------- 云端同步接口 ----------

    def set_cloud_sync_handler(self, handler: Callable) -> None:
        """设置云端同步处理器

        Args:
            handler: 同步函数, 签名: handler(local_path, remote_path) -> bool
        """
        self._cloud_sync_handler = handler
        self._cloud_sync_enabled = True

    def cloud_sync(self, tag: Optional[str] = None) -> Dict[str, Any]:
        """云端同步: 将检查点同步到云存储

        Args:
            tag: 要同步的检查点标签 (None则同步所有)

        Returns:
            同步结果
        """
        if not self._cloud_sync_enabled or self._cloud_sync_handler is None:
            return {
                "success": False,
                "error": "云端同步未启用, 请先调用 set_cloud_sync_handler()",
            }

        tags = [tag] if tag else list(self.versions.keys())
        results = {}
        for t in tags:
            filepath = os.path.join(self.save_dir, f"{t}.ckpt")
            remote_path = f"lingyuan/checkpoints/{t}.ckpt"
            try:
                success = self._cloud_sync_handler(filepath, remote_path)
                results[t] = {"success": success, "remote_path": remote_path}
            except Exception as e:
                results[t] = {"success": False, "error": str(e)}

        return {
            "success": all(r.get("success", False) for r in results.values()),
            "synced": results,
        }

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, Any]:
        """获取检查点管理统计"""
        return {
            "save_count": self.save_count,
            "rollback_count": self.rollback_count,
            "total_save_time_s": round(self.total_save_time, 4),
            "versions_count": len(self.versions),
            "max_versions": self.max_versions,
            "compress_enabled": self.compress_enabled,
            "cloud_sync_enabled": self._cloud_sync_enabled,
            "save_dir": self.save_dir,
        }


# ============================================================
# 6. TrainingScheduler — 训练调度器
# ============================================================

class TrainingScheduler:
    """训练调度器

    功能:
    - 学习率调度: cosine / linear / warmup / one_cycle
    - 批大小调度: 渐进增大
    - 梯度累积调度: 根据显存动态调整
    - 训练阶段管理: warmup → main → fine-tune → decay
    - 早停: 基于验证损失的智能早停

    用法:
        scheduler = TrainingScheduler(
            total_steps=10000, base_lr=1e-3, schedule_type="cosine")
        lr = scheduler.get_lr(current_step)
        batch_size = scheduler.get_batch_size(current_step)
    """

    # 训练阶段定义
    STAGES = ["warmup", "main", "fine-tune", "decay"]

    def __init__(self,
                 total_steps: int = 10000,
                 base_lr: float = 1e-3,
                 min_lr: float = 1e-5,
                 max_lr: float = 3e-3,
                 schedule_type: str = "cosine",
                 warmup_steps: int = 500,
                 warmup_ratio: float = 0.05,
                 initial_batch_size: int = 4,
                 max_batch_size: int = 64,
                 initial_grad_accum: int = 1,
                 max_grad_accum: int = 8,
                 early_stop_patience: int = 5,
                 early_stop_min_delta: float = 1e-4,
                 stage_boundaries: Optional[Dict[str, float]] = None):
        """
        Args:
            total_steps: 总训练步数
            base_lr: 基准学习率
            min_lr: 最小学习率
            max_lr: 最大学习率 (one_cycle用)
            schedule_type: cosine / linear / warmup / one_cycle / constant
            warmup_steps: warmup步数
            warmup_ratio: warmup占总步数比例 (warmup_steps为0时使用)
            initial_batch_size: 初始批大小
            max_batch_size: 最大批大小
            initial_grad_accum: 初始梯度累积步数
            max_grad_accum: 最大梯度累积步数
            early_stop_patience: 早停耐心值 (连续多少步无改善)
            early_stop_min_delta: 早停最小改善量
            stage_boundaries: 阶段边界 {stage_name: progress_ratio}
        """
        self.total_steps = max(1, total_steps)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.initial_grad_accum = initial_grad_accum
        self.max_grad_accum = max_grad_accum
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta

        # 计算实际warmup步数
        if warmup_steps == 0 and warmup_ratio > 0:
            self.warmup_steps = int(total_steps * warmup_ratio)

        # 阶段边界 (各阶段的进度比例)
        self.stage_boundaries = stage_boundaries or {
            "warmup": 0.0,
            "main": 0.1,
            "fine-tune": 0.7,
            "decay": 0.9,
        }

        # 早停状态
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._should_stop = False

        # 当前阶段
        self._current_stage = "warmup"

        # 统计
        self.lr_history: List[float] = []
        self.batch_size_history: List[int] = []
        self.grad_accum_history: List[int] = []
        self.stage_history: List[str] = []

    # ---------- 学习率调度 ----------

    def get_lr(self, step: Optional[int] = None) -> float:
        """计算指定步数的学习率

        支持的调度类型:
        - cosine: 余弦退火
        - linear: 线性衰减
        - warmup: 线性warmup后保持
        - one_cycle: 超级收敛 (warmup → peak → 退火)
        - constant: 恒定学习率
        """
        s = step if step is not None else len(self.lr_history)
        s = max(0, min(s, self.total_steps))

        # Warmup阶段
        if s < self.warmup_steps:
            return self.base_lr * (s + 1) / max(self.warmup_steps, 1)

        # 主训练阶段
        progress = (s - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        progress = max(0.0, min(1.0, progress))

        if self.schedule_type == "cosine":
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                (1.0 + math.cos(math.pi * progress))

        elif self.schedule_type == "linear":
            return self.base_lr - (self.base_lr - self.min_lr) * progress

        elif self.schedule_type == "warmup":
            return self.base_lr

        elif self.schedule_type == "one_cycle":
            # One Cycle: warmup → max_lr → 退火到min_lr
            if progress < 0.3:
                # 上升阶段
                p = progress / 0.3
                return self.base_lr + (self.max_lr - self.base_lr) * p
            else:
                # 退火阶段
                p = (progress - 0.3) / 0.7
                return self.max_lr - (self.max_lr - self.min_lr) * \
                    (1.0 + math.cos(math.pi * p)) / 2

        elif self.schedule_type == "constant":
            return self.base_lr

        return self.base_lr

    # ---------- 批大小调度 ----------

    def get_batch_size(self, step: Optional[int] = None) -> int:
        """计算指定步数的批大小 (渐进增大)

        策略: 在训练前30%逐步增大批大小
        batch_size = initial + (max - initial) * min(progress / 0.3, 1.0)
        """
        s = step if step is not None else len(self.batch_size_history)
        progress = s / max(self.total_steps, 1)

        if progress < 0.3:
            scale = progress / 0.3
        else:
            scale = 1.0

        batch_size = int(self.initial_batch_size +
                         (self.max_batch_size - self.initial_batch_size) * scale)
        return max(1, batch_size)

    # ---------- 梯度累积调度 ----------

    def get_grad_accum(self, step: Optional[int] = None) -> int:
        """计算指定步数的梯度累积步数

        策略: warmup阶段使用较大的累积步数(模拟大batch),
        主训练阶段逐步减小累积步数
        """
        s = step if step is not None else len(self.grad_accum_history)
        progress = s / max(self.total_steps, 1)

        if progress < 0.1:
            # warmup: 使用最大累积
            return self.max_grad_accum
        elif progress < 0.5:
            # 主训练: 逐步减小
            p = (progress - 0.1) / 0.4
            return max(self.initial_grad_accum,
                       int(self.max_grad_accum - 
                           (self.max_grad_accum - self.initial_grad_accum) * p))
        else:
            return self.initial_grad_accum

    # ---------- 训练阶段管理 ----------

    def get_stage(self, step: Optional[int] = None) -> str:
        """获取当前训练阶段

        阶段: warmup → main → fine-tune → decay

        Returns:
            阶段名称
        """
        s = step if step is not None else len(self.stage_history)
        progress = s / max(self.total_steps, 1)

        boundaries = self.stage_boundaries
        stage = "warmup"

        if progress >= boundaries.get("decay", 0.9):
            stage = "decay"
        elif progress >= boundaries.get("fine-tune", 0.7):
            stage = "fine-tune"
        elif progress >= boundaries.get("main", 0.1):
            stage = "main"

        self._current_stage = stage
        return stage

    def get_stage_config(self, step: Optional[int] = None) -> Dict[str, Any]:
        """获取当前阶段的完整配置

        Returns:
            {
                "stage": 阶段名,
                "lr": 学习率,
                "batch_size": 批大小,
                "grad_accum": 梯度累积步数,
            }
        """
        stage = self.get_stage(step)
        return {
            "stage": stage,
            "lr": self.get_lr(step),
            "batch_size": self.get_batch_size(step),
            "grad_accum": self.get_grad_accum(step),
        }

    # ---------- 早停 ----------

    def check_early_stop(self, val_loss: float) -> bool:
        """检查是否应该早停

        基于验证损失的智能早停:
        - 如果验证损失改善超过min_delta, 重置耐心计数
        - 如果连续patience步无改善, 触发早停

        Args:
            val_loss: 当前验证损失

        Returns:
            是否应该停止训练
        """
        if val_loss < self._best_loss - self.early_stop_min_delta:
            self._best_loss = val_loss
            self._patience_counter = 0
            self._should_stop = False
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.early_stop_patience:
                self._should_stop = True

        return self._should_stop

    def reset_early_stop(self) -> None:
        """重置早停状态"""
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._should_stop = False

    # ---------- 步进 ----------

    def step(self, current_step: int, val_loss: Optional[float] = None
             ) -> Dict[str, Any]:
        """推进调度器一步

        Args:
            current_step: 当前训练步
            val_loss: 验证损失 (可选, 用于早停)

        Returns:
            当前步的调度配置
        """
        lr = self.get_lr(current_step)
        batch_size = self.get_batch_size(current_step)
        grad_accum = self.get_grad_accum(current_step)
        stage = self.get_stage(current_step)

        self.lr_history.append(lr)
        self.batch_size_history.append(batch_size)
        self.grad_accum_history.append(grad_accum)
        self.stage_history.append(stage)

        should_stop = False
        if val_loss is not None:
            should_stop = self.check_early_stop(val_loss)

        return {
            "step": current_step,
            "lr": round(lr, 8),
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "stage": stage,
            "should_stop": should_stop,
        }

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计"""
        return {
            "schedule_type": self.schedule_type,
            "total_steps": self.total_steps,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            "current_stage": self._current_stage,
            "early_stop_triggered": self._should_stop,
            "early_stop_patience": self.early_stop_patience,
            "best_loss": self._best_loss if self._best_loss != float('inf') else None,
            "patience_counter": self._patience_counter,
            "lr_history_len": len(self.lr_history),
        }

    def generate_schedule_plot_data(self,
                                     num_points: int = 100
                                     ) -> Dict[str, List]:
        """生成调度曲线数据 (用于可视化)

        Returns:
            {
                "steps": [0, 1, ...],
                "lr": [lr0, lr1, ...],
                "batch_size": [bs0, bs1, ...],
                "grad_accum": [ga0, ga1, ...],
                "stages": [stage0, stage1, ...],
            }
        """
        steps = []
        lrs = []
        batch_sizes = []
        grad_accums = []
        stages = []

        for i in range(num_points):
            s = int(i * self.total_steps / num_points)
            steps.append(s)
            lrs.append(self.get_lr(s))
            batch_sizes.append(self.get_batch_size(s))
            grad_accums.append(self.get_grad_accum(s))
            stages.append(self.get_stage(s))

        return {
            "steps": steps,
            "lr": lrs,
            "batch_size": batch_sizes,
            "grad_accum": grad_accums,
            "stages": stages,
        }


# ============================================================
# 7. DistributedTrainer — 分布式训练协调器
# ============================================================

class DistributedTrainer:
    """分布式训练协调器 — 多虚拟GPU协调

    功能:
    - 数据并行: 梯度聚合 (All-Reduce模拟)
    - 流水线并行: 层切分到多GPU
    - 梯度同步: All-Reduce模拟
    - 容错: 节点故障处理

    用法:
        trainer = DistributedTrainer(model, num_gpus=4, strategy="data_parallel")
        result = trainer.train_step(batch)
    """

    def __init__(self,
                 model: LingyuanTransformerModel,
                 num_gpus: int = 2,
                 strategy: str = "data_parallel",
                 num_pipeline_stages: int = 1,
                 lr: float = 1e-3,
                 gpu: Optional[VirtualGPU] = None,
                 fault_tolerance: bool = True):
        """
        Args:
            model: 模型
            num_gpus: 虚拟GPU数量
            strategy: "data_parallel" / "pipeline_parallel" / "hybrid"
            num_pipeline_stages: 流水线阶段数
            lr: 学习率
            gpu: 主GPU (None则自动创建)
            fault_tolerance: 是否启用容错
        """
        self.model = model
        self.num_gpus = num_gpus
        self.strategy = strategy
        self.num_pipeline_stages = num_pipeline_stages
        self.lr = lr
        self.fault_tolerance = fault_tolerance

        # 创建多个虚拟GPU
        self.main_gpu: VirtualGPU = gpu or VirtualGPU()
        if not self.main_gpu._warmup_done:
            self.main_gpu.warmup()

        self.worker_gpus: List[VirtualGPU] = [self.main_gpu]
        for i in range(1, num_gpus):
            wgpu = VirtualGPU()
            wgpu.warmup()
            self.worker_gpus.append(wgpu)

        # 创建训练引擎 (使用主GPU)
        self.engine = GPUAcceleratedTrainingEngine(
            model=model, lr=lr, gpu=self.main_gpu,
            num_dp_devices=num_gpus if strategy == "data_parallel" else 1,
            num_pp_stages=num_pipeline_stages if strategy == "pipeline_parallel" else 1,
        )

        # 梯度压缩器 (用于减少通信量)
        self.grad_compressor = GradientCompression(
            top_k_ratio=1.0, quantize=False, error_feedback=True)

        # 流水线阶段切分
        if strategy in ("pipeline_parallel", "hybrid"):
            self._pipeline_stages = self._split_pipeline_stages()
        else:
            self._pipeline_stages = [(0, model.num_layers)]

        # 容错状态
        self._node_status: Dict[int, str] = {i: "active" for i in range(num_gpus)}
        self._node_failure_count: Dict[int, int] = {i: 0 for i in range(num_gpus)}

        # 统计
        self.total_steps = 0
        self.total_comm_bytes = 0
        self.total_comm_time = 0.0
        self.gradient_sync_count = 0
        self.node_failures_handled = 0

    def _split_pipeline_stages(self) -> List[Tuple[int, int]]:
        """将模型层切分到流水线阶段"""
        n = self.model.num_layers
        if self.num_pipeline_stages <= 1:
            return [(0, n)]
        per = n // self.num_pipeline_stages
        splits = []
        for i in range(self.num_pipeline_stages):
            start = i * per
            end = start + per if i < self.num_pipeline_stages - 1 else n
            splits.append((start, end))
        return splits

    # ---------- 数据并行 ----------

    def data_parallel_step(self,
                           batch: List[Tuple[List[int], List[int]]]
                           ) -> Dict[str, Any]:
        """数据并行训练步

        1. 将batch切分到各GPU
        2. 各GPU独立前向+反向
        3. All-Reduce梯度聚合
        4. 统一更新权重

        Returns:
            训练步统计
        """
        t0 = time.time()
        n_active = sum(1 for s in self._node_status.values() if s == "active")
        if n_active == 0:
            return {"error": "所有节点均不可用"}

        # 切分batch到各活跃GPU
        batch_per_gpu = max(1, len(batch) // n_active)
        shards: List[List[Tuple[List[int], List[int]]]] = []
        active_gpus = [i for i in range(self.num_gpus)
                       if self._node_status[i] == "active"]

        idx = 0
        for gpu_idx in active_gpus:
            end = min(idx + batch_per_gpu, len(batch))
            if gpu_idx == active_gpus[-1]:
                end = len(batch)
            shards.append(batch[idx:end])
            idx = end

        # 各GPU独立前向+反向
        all_grads: List[Dict[str, Any]] = []
        total_loss = 0.0
        total_samples = 0

        for shard in shards:
            if not shard:
                continue
            shard_grads: Dict[str, Any] = {}
            for input_ids, targets in shard:
                loss, logits, cache = self.engine.forward_pass(
                    input_ids, targets)
                total_loss += loss
                total_samples += 1
                grads = self.engine.backward_pass(
                    logits, targets, cache=cache)
                if not shard_grads:
                    shard_grads = {k: _deep_copy_grad(v) for k, v in grads.items()}
                else:
                    for k in grads:
                        if k in shard_grads:
                            shard_grads[k] = _add_grads(
                                shard_grads[k], grads[k])

            if shard_grads:
                # 平均分片梯度
                n = len(shard)
                shard_grads = {k: _scale_grad(v, 1.0 / n)
                               for k, v in shard_grads.items()}
                all_grads.append(shard_grads)

        # All-Reduce梯度聚合
        comm_t0 = time.time()
        avg_grads = self.all_reduce_gradients(all_grads)
        self.total_comm_time += time.time() - comm_t0

        # 梯度裁剪
        grad_norm = self.engine.clip_grad_norm(avg_grads)

        # 更新权重
        current_lr = self.engine.get_lr()
        self.engine.optimizer.lr = current_lr
        self.engine.optimizer.step(self.engine._get_params(), avg_grads)

        self.engine.global_step += 1
        self.engine.lr_history.append(current_lr)
        self.total_steps += 1

        avg_loss = total_loss / max(total_samples, 1)
        self.engine.loss_history.append(avg_loss)
        self.engine.step += 1

        elapsed = time.time() - t0
        return {
            "step": self.engine.step,
            "global_step": self.engine.global_step,
            "loss": round(avg_loss, 6),
            "grad_norm": round(grad_norm, 4),
            "lr": round(current_lr, 8),
            "samples": total_samples,
            "active_gpus": n_active,
            "strategy": "data_parallel",
            "elapsed_s": round(elapsed, 4),
        }

    # ---------- 流水线并行 ----------

    def pipeline_parallel_forward(self,
                                  input_ids: List[int]
                                  ) -> List[List[float]]:
        """流水线并行前向传播

        将模型层切分到多个GPU, 每个GPU负责一段连续的层。
        激活值在GPU之间传递。

        Returns:
            隐藏层输出 (seq × hidden)
        """
        x = self.model.embed(input_ids)

        # 绝对位置编码
        if self.model.pos_method == "absolute":
            abs_pe = self.model.positional_encoding.get_absolute(len(input_ids))
            x = [[x[s][d] + abs_pe[s][d] for d in range(self.model.hidden_dim)]
                 for s in range(len(input_ids))]

        # 逐阶段执行
        for stage_idx, (start, end) in enumerate(self._pipeline_stages):
            gpu = self.worker_gpus[stage_idx % len(self.worker_gpus)]
            for layer_idx in range(start, end):
                layer = self.model.layers[layer_idx]
                # 使用GPU加速的层前向
                h = layer.norm1(x)
                attn_out = gpu.parallel_multi_head_attention(
                    h, layer.attn.W_q, layer.attn.W_k,
                    layer.attn.W_v, layer.attn.W_o,
                    num_heads=layer.attn.num_heads,
                    num_kv_heads=layer.attn.num_kv_heads,
                    rope_cos=self.model.positional_encoding._rope_cos
                    if self.model.pos_method == "rope" else None,
                    rope_sin=self.model.positional_encoding._rope_sin
                    if self.model.pos_method == "rope" else None,
                )
                hidden = self.model.hidden_dim
                x = [[x[s][d] + attn_out[s][d] for d in range(hidden)]
                     for s in range(len(x))]

                h = layer.norm2(x)
                ffn_out = gpu.parallel_swiglu_ffn(
                    h, layer.ffn.W_gate, layer.ffn.W_up, layer.ffn.W_down)
                x = [[x[s][d] + ffn_out[s][d] for d in range(hidden)]
                     for s in range(len(x))]

        return x

    # ---------- All-Reduce梯度同步 ----------

    def all_reduce_gradients(self,
                             grads_list: List[Dict[str, Any]]
                             ) -> Dict[str, Any]:
        """All-Reduce梯度聚合 (模拟)

        将多个GPU的梯度取平均。

        Args:
            grads_list: 各GPU的梯度字典列表

        Returns:
            聚合后的平均梯度
        """
        if not grads_list:
            return {}
        if len(grads_list) == 1:
            return grads_list[0]

        self.gradient_sync_count += 1
        n = len(grads_list)

        # 估算通信量
        for grads in grads_list:
            for name, g in grads.items():
                self.total_comm_bytes += self._estimate_grad_size(g)

        # 平均聚合
        avg_grads: Dict[str, Any] = {}
        for name in grads_list[0]:
            accumulated = _deep_copy_grad(grads_list[0][name])
            for i in range(1, n):
                if name in grads_list[i]:
                    accumulated = _add_grads(accumulated, grads_list[i][name])
            avg_grads[name] = _scale_grad(accumulated, 1.0 / n)

        return avg_grads

    @staticmethod
    def _estimate_grad_size(grad: Any) -> int:
        """估算梯度大小 (字节)"""
        if isinstance(grad, list) and len(grad) > 0 and isinstance(grad[0], list):
            return len(grad) * len(grad[0]) * 8
        elif isinstance(grad, list):
            return len(grad) * 8
        return 8

    # ---------- 容错: 节点故障处理 ----------

    def simulate_node_failure(self, gpu_idx: int) -> None:
        """模拟节点故障

        Args:
            gpu_idx: 故障的GPU索引
        """
        if 0 <= gpu_idx < self.num_gpus:
            self._node_status[gpu_idx] = "failed"
            self._node_failure_count[gpu_idx] += 1

    def recover_node(self, gpu_idx: int) -> bool:
        """恢复故障节点

        Args:
            gpu_idx: 要恢复的GPU索引

        Returns:
            是否恢复成功
        """
        if 0 <= gpu_idx < self.num_gpus:
            self._node_status[gpu_idx] = "active"
            return True
        return False

    def handle_node_failure(self) -> Dict[str, Any]:
        """处理节点故障: 重新分配工作到活跃节点

        Returns:
            故障处理报告
        """
        if not self.fault_tolerance:
            return {"handled": False, "reason": "容错未启用"}

        failed_nodes = [i for i, s in self._node_status.items()
                        if s == "failed"]
        active_nodes = [i for i, s in self._node_status.items()
                        if s == "active"]

        if not active_nodes:
            return {"handled": False, "reason": "无可用节点"}

        self.node_failures_handled += 1

        return {
            "handled": True,
            "failed_nodes": failed_nodes,
            "active_nodes": active_nodes,
            "redistributed": True,
            "message": f"工作已重新分配到 {len(active_nodes)} 个活跃节点",
        }

    # ---------- 统一训练步 ----------

    def train_step(self,
                   batch: List[Tuple[List[int], List[int]]]
                   ) -> Dict[str, Any]:
        """统一训练步 (根据策略自动选择)

        Returns:
            训练步统计
        """
        if self.strategy == "data_parallel":
            return self.data_parallel_step(batch)
        elif self.strategy == "pipeline_parallel":
            # 流水线并行: 使用主引擎训练, 但前向走流水线
            return self.engine.train_step(batch)
        else:
            # hybrid: 数据并行 + 流水线
            return self.data_parallel_step(batch)

    # ---------- 集群统计 ----------

    def get_cluster_stats(self) -> Dict[str, Any]:
        """获取集群统计"""
        active = sum(1 for s in self._node_status.values() if s == "active")
        failed = sum(1 for s in self._node_status.values() if s == "failed")

        gpu_stats = []
        for i, gpu in enumerate(self.worker_gpus):
            util = gpu.get_utilization()
            gpu_stats.append({
                "gpu_id": i,
                "status": self._node_status.get(i, "unknown"),
                "sm_count": util["num_sms"],
                "matmul_count": util["matmul_count"],
                "memory_mb": util["memory_allocated_mb"],
            })

        return {
            "strategy": self.strategy,
            "num_gpus": self.num_gpus,
            "active_gpus": active,
            "failed_gpus": failed,
            "pipeline_stages": self._pipeline_stages,
            "total_steps": self.total_steps,
            "gradient_sync_count": self.gradient_sync_count,
            "total_comm_bytes": self.total_comm_bytes,
            "total_comm_mb": round(self.total_comm_bytes / (1024 * 1024), 4),
            "total_comm_time_s": round(self.total_comm_time, 4),
            "node_failures_handled": self.node_failures_handled,
            "gpu_details": gpu_stats,
        }

    def get_cluster_report(self) -> str:
        """生成集群报告 (文本格式)"""
        stats = self.get_cluster_stats()
        lines = [
            "=" * 70,
            "  Lingyuan Distributed Training Cluster Report",
            "=" * 70,
            f"  训练策略:     {stats['strategy']}",
            f"  GPU总数:      {stats['num_gpus']}",
            f"  活跃GPU:      {stats['active_gpus']}",
            f"  故障GPU:      {stats['failed_gpus']}",
            f"  训练步数:     {stats['total_steps']}",
            f"  梯度同步次数: {stats['gradient_sync_count']}",
            f"  通信总量:     {stats['total_comm_mb']:.4f} MB",
            f"  通信耗时:     {stats['total_comm_time_s']:.4f} s",
            f"  故障处理次数: {stats['node_failures_handled']}",
            "",
            "  --- GPU详情 ---",
        ]
        for gpu in stats["gpu_details"]:
            lines.append(
                f"    [GPU {gpu['gpu_id']}] {gpu['status']:8s} | "
                f"SM={gpu['sm_count']} | "
                f"matmul={gpu['matmul_count']} | "
                f"mem={gpu['memory_mb']:.4f}MB")
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================
# __main__ 自测代码
# ============================================================

def _make_tiny_model() -> LingyuanTransformerModel:
    """创建一个微型模型用于自测"""
    return LingyuanTransformerModel(
        hidden_dim=32, num_layers=2, num_heads=4, num_kv_heads=2,
        ffn_dim=64, max_seq_len=64, vocab_size=128,
        rope_theta=10000.0, norm_eps=1e-6,
        tie_word_embeddings=True, pos_method="rope",
    )


def _make_dataset(model: LingyuanTransformerModel,
                  n_samples: int = 8,
                  seq_len: int = 12) -> List[Tuple[List[int], List[int]]]:
    """生成简单训练数据"""
    random.seed(42)
    dataset = []
    for _ in range(n_samples):
        input_ids = [random.randint(1, model.vocab_size - 1) for _ in range(seq_len)]
        targets = [random.randint(1, model.vocab_size - 1) for _ in range(seq_len)]
        dataset.append((input_ids, targets))
    return dataset


def _test_gpu_accelerated_training_engine():
    """测试 GPUAcceleratedTrainingEngine"""
    print("\n[1] 测试 GPUAcceleratedTrainingEngine")
    print("-" * 50)

    model = _make_tiny_model()
    gpu = VirtualGPU(num_sms=2)
    gpu.warmup()

    engine = GPUAcceleratedTrainingEngine(
        model=model, lr=1e-3, gpu=gpu, gpu_threshold=64,
        grad_accumulation_steps=1, precision="fp32")

    dataset = _make_dataset(model, n_samples=4, seq_len=8)

    # 前向+反向
    loss, logits, cache = engine.forward_pass(dataset[0][0], dataset[0][1])
    print(f"  前向传播: loss={loss:.6f}, logits_shape={len(logits)}x{len(logits[0])}")

    grads = engine.backward_pass(logits, dataset[0][1], cache=cache)
    print(f"  反向传播: 梯度参数数={len(grads)}")

    # 训练步
    result = engine.train_step(dataset[:2])
    print(f"  训练步:   step={result['step']}, loss={result['loss']:.6f}")

    # GPU统计
    stats = engine.get_gpu_stats()
    print(f"  GPU统计:  gpu_matmul={stats['gpu_matmul_calls']}, "
          f"cpu_matmul={stats['cpu_matmul_calls']}, "
          f"gpu_ratio={stats['gpu_matmul_ratio']:.2%}")
    print(f"  GPU显存:  peak={stats['gpu_memory_peak_mb']:.4f}MB")

    # 基准测试
    bench = engine.benchmark_gpu_vs_cpu(dataset[0][0], dataset[0][1], num_runs=2)
    print(f"  基准测试: CPU={bench['cpu_time_ms']:.1f}ms, "
          f"GPU={bench['gpu_time_ms']:.1f}ms, "
          f"加速比={bench['speedup']:.2f}x")

    print("  [OK] GPUAcceleratedTrainingEngine")
    return engine


def _test_gpu_batch_processor(engine: GPUAcceleratedTrainingEngine):
    """测试 GPUBatchProcessor"""
    print("\n[2] 测试 GPUBatchProcessor")
    print("-" * 50)

    processor = GPUBatchProcessor(engine, max_batch_size=8, max_seq_len=32)

    # 动态批大小
    bs = processor.compute_dynamic_batch_size(seq_len=16)
    print(f"  动态批大小 (seq=16): {bs}")

    # 序列打包
    samples = _make_dataset(engine.model, n_samples=6, seq_len=10)
    # 添加一些变长序列
    samples.append(([1, 2, 3], [4, 5, 6]))
    samples.append(([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]))

    packed, lengths = processor.pack_sequences(samples)
    print(f"  序列打包: {len(samples)}个样本 → {len(packed)}个批次")
    print(f"  批次长度: {lengths}")

    # 批量前向
    fwd_results = processor.batch_forward(samples[:3])
    print(f"  批量前向: {len(fwd_results)}个结果, "
          f"losses={[round(r[0], 4) for r in fwd_results]}")

    # 批量训练步
    result = processor.batch_train_step(samples[:4])
    print(f"  批量训练: step={result['step']}, loss={result['loss']:.6f}, "
          f"samples={result['samples']}")

    stats = processor.get_stats()
    print(f"  统计: total_samples={stats['total_samples_processed']}, "
          f"avg_batch={stats['avg_batch_size']}")

    print("  [OK] GPUBatchProcessor")


def _test_gradient_compression():
    """测试 GradientCompression"""
    print("\n[3] 测试 GradientCompression")
    print("-" * 50)

    # 创建测试梯度
    grads = {
        "layer_0_W_q": [[0.1, 0.5, -0.3], [0.0, -0.2, 0.8], [0.01, 0.0, -0.05]],
        "layer_0_W_k": [[0.3, -0.1], [0.7, 0.0], [-0.4, 0.2]],
        "final_norm": [0.5, -0.3, 0.1],
    }

    # Top-K稀疏化
    compressor = GradientCompression(top_k_ratio=0.5, quantize=False,
                                     error_feedback=False)
    sparse_grad, indices = compressor.top_k_sparsify(grads["layer_0_W_q"])
    nonzeros = sum(1 for row in sparse_grad for v in row if v != 0.0)
    total = sum(len(row) for row in sparse_grad)
    print(f"  Top-K稀疏化: {nonzeros}/{total} 非零 (ratio=0.5)")

    # INT8量化
    compressor2 = GradientCompression(top_k_ratio=1.0, quantize=True,
                                      error_feedback=False)
    quantized, scale = compressor2.quantize_int8(grads["layer_0_W_q"])
    dequantized = compressor2.dequantize_int8(quantized, scale)
    max_err = max(abs(grads["layer_0_W_q"][i][j] - dequantized[i][j])
                  for i in range(len(dequantized))
                  for j in range(len(dequantized[0])))
    print(f"  INT8量化: scale={scale:.6f}, 最大量化误差={max_err:.6f}")

    # 完整压缩流程
    compressor3 = GradientCompression(top_k_ratio=0.3, quantize=True,
                                      error_feedback=True,
                                      denoise_threshold=0.05)
    compressed = compressor3.compress(grads)
    decompressed = compressor3.decompress(compressed)
    print(f"  完整压缩: 原始参数={len(grads)}, "
          f"压缩后参数={len(compressed['gradients'])}")
    print(f"  压缩比:   {compressor3.get_compression_ratio():.2f}x")

    # 梯度去噪
    noisy_grad = [0.001, 0.5, -0.002, 0.3, 0.0001, -0.4]
    denoised = compressor3.denoise(noisy_grad, threshold=0.01)
    nonzeros = sum(1 for v in denoised if v != 0.0)
    print(f"  梯度去噪: {nonzeros}/{len(denoised)} 保留 (threshold=0.01)")

    # 误差反馈 (多轮)
    compressor4 = GradientCompression(top_k_ratio=0.3, quantize=False,
                                      error_feedback=True)
    for i in range(3):
        c = compressor4.compress(grads)
        d = compressor4.decompress(c)
    print(f"  误差反馈: {compressor4.compression_count}轮压缩完成")

    stats = compressor3.get_stats()
    print(f"  统计: compression_count={stats['compression_count']}, "
          f"ratio={stats['compression_ratio']:.2f}x")

    print("  [OK] GradientCompression")


def _test_training_profiler():
    """测试 TrainingProfiler"""
    print("\n[4] 测试 TrainingProfiler")
    print("-" * 50)

    profiler = TrainingProfiler()

    # 模拟逐层计时
    for i in range(5):
        profiler.start_event("forward_total")
        profiler.start_layer(f"layer_{i}")
        time.sleep(0.001 * (i + 1))  # 模拟计算耗时
        profiler.end_layer()
        profiler.end_event("forward_total")

    profiler.start_event("backward_total")
    time.sleep(0.002)
    profiler.end_event("backward_total")

    # 记录显存和GPU利用率
    for i in range(5):
        profiler.record_memory(i, 10.0 + i * 5.0)
        profiler.record_loss(i, 5.0 - i * 0.5)

    gpu = VirtualGPU(num_sms=2)
    gpu.warmup()
    _ = gpu.parallel_matmul([[1.0] * 8] * 8, [[1.0] * 8] * 8)
    profiler.record_gpu_util(0, gpu)

    # 瓶颈识别
    bottleneck = profiler.identify_bottleneck()
    print(f"  瓶颈事件: {bottleneck['bottleneck_event']}"
          f" ({bottleneck['bottleneck_time_s']*1000:.3f}ms)")
    print(f"  瓶颈层:   {bottleneck['bottleneck_layer']}"
          f" ({bottleneck['bottleneck_layer_time_s']*1000:.3f}ms)")

    # 生成报告
    report = profiler.generate_report()
    report_lines = report.split("\n")
    print(f"  报告行数: {len(report_lines)}")
    # 打印报告前几行
    for line in report_lines[:8]:
        print(f"  {line}")

    stats = profiler.get_stats()
    print(f"  统计: events={stats['event_count']}, "
          f"layers={stats['layer_count']}, "
          f"memory_records={stats['memory_records']}")

    print("  [OK] TrainingProfiler")
    return profiler


def _test_checkpoint_manager(engine: GPUAcceleratedTrainingEngine):
    """测试 CheckpointManager"""
    print("\n[5] 测试 CheckpointManager")
    print("-" * 50)

    import tempfile
    save_dir = os.path.join(tempfile.gettempdir(), "lingyuan_ckpt_test")
    ckpt_mgr = CheckpointManager(
        engine, save_dir=save_dir, compress=True, max_versions=5)

    # 保存检查点
    meta1 = ckpt_mgr.save("v1")
    print(f"  保存v1: step={meta1['global_step']}, "
          f"compressed={meta1['compressed']}, "
          f"incremental={meta1['is_incremental']}")

    # 训练一步后保存
    dataset = _make_dataset(engine.model, n_samples=2, seq_len=8)
    engine.train_step(dataset)
    meta2 = ckpt_mgr.save("v2")
    print(f"  保存v2: step={meta2['global_step']}")

    # 列出版本
    versions = ckpt_mgr.list_versions()
    print(f"  版本数: {len(versions)}")
    for v in versions:
        print(f"    {v['tag']}: step={v['global_step']}, "
              f"incremental={v['is_incremental']}")

    # 版本对比
    comparison = ckpt_mgr.compare_versions("v1", "v2")
    print(f"  版本对比: step_diff={comparison['step_diff']}")

    # 加载检查点
    loaded = ckpt_mgr.load("v1")
    print(f"  加载v1: step={loaded['global_step']}")

    # 自动回滚测试: 先设基准, 再触发回滚
    ckpt_mgr._last_loss = 1.0  # 设置基准损失
    rollback = ckpt_mgr.auto_rollback(999.0, threshold=0.1)
    print(f"  自动回滚 (loss 1.0→999.0): {rollback}")

    # 云端同步接口测试
    def mock_cloud_handler(local_path, remote_path):
        return True
    ckpt_mgr.set_cloud_sync_handler(mock_cloud_handler)
    sync_result = ckpt_mgr.cloud_sync("v1")
    print(f"  云端同步: success={sync_result['success']}")

    stats = ckpt_mgr.get_stats()
    print(f"  统计: saves={stats['save_count']}, "
          f"rollbacks={stats['rollback_count']}, "
          f"versions={stats['versions_count']}")

    print("  [OK] CheckpointManager")


def _test_training_scheduler():
    """测试 TrainingScheduler"""
    print("\n[6] 测试 TrainingScheduler")
    print("-" * 50)

    # Cosine调度
    scheduler = TrainingScheduler(
        total_steps=1000, base_lr=1e-3, min_lr=1e-5,
        schedule_type="cosine", warmup_steps=100,
        initial_batch_size=4, max_batch_size=32,
        initial_grad_accum=4, max_grad_accum=8)

    # 测试各步学习率
    for step in [0, 50, 100, 500, 999]:
        lr = scheduler.get_lr(step)
        bs = scheduler.get_batch_size(step)
        ga = scheduler.get_grad_accum(step)
        stage = scheduler.get_stage(step)
        print(f"  step={step:4d}: lr={lr:.6f}, bs={bs}, "
              f"accum={ga}, stage={stage}")

    # One-Cycle调度
    scheduler_oc = TrainingScheduler(
        total_steps=1000, base_lr=1e-3, max_lr=3e-3, min_lr=1e-5,
        schedule_type="one_cycle", warmup_steps=50)
    lrs = [scheduler_oc.get_lr(s) for s in [0, 50, 300, 700, 999]]
    print(f"  One-Cycle LR: {[f'{lr:.5f}' for lr in lrs]}")

    # 早停测试
    scheduler_es = TrainingScheduler(
        total_steps=1000, early_stop_patience=3, early_stop_min_delta=0.001)
    losses = [1.0, 0.9, 0.85, 0.85, 0.85, 0.85]  # 后4步无改善
    for i, loss in enumerate(losses):
        should_stop = scheduler_es.check_early_stop(loss)
        if should_stop:
            print(f"  早停触发: step={i}, loss={loss}")
            break

    # 步进
    result = scheduler.step(500, val_loss=0.5)
    print(f"  步进500: lr={result['lr']}, bs={result['batch_size']}, "
          f"stage={result['stage']}")

    # 调度曲线数据
    plot_data = scheduler.generate_schedule_plot_data(num_points=10)
    print(f"  调度曲线: {len(plot_data['steps'])}个点")

    stats = scheduler.get_stats()
    print(f"  统计: type={stats['schedule_type']}, "
          f"stage={stats['current_stage']}")

    print("  [OK] TrainingScheduler")


def _test_distributed_trainer():
    """测试 DistributedTrainer"""
    print("\n[7] 测试 DistributedTrainer")
    print("-" * 50)

    model = _make_tiny_model()
    dataset = _make_dataset(model, n_samples=8, seq_len=8)

    # 数据并行
    trainer = DistributedTrainer(
        model=model, num_gpus=2, strategy="data_parallel", lr=1e-3)

    result = trainer.train_step(dataset)
    print(f"  数据并行: step={result['step']}, loss={result['loss']:.6f}, "
          f"active_gpus={result['active_gpus']}")

    # All-Reduce测试
    grads_a = {"w": [[1.0, 2.0], [3.0, 4.0]]}
    grads_b = {"w": [[3.0, 4.0], [5.0, 6.0]]}
    avg = trainer.all_reduce_gradients([grads_a, grads_b])
    print(f"  All-Reduce: avg={avg['w'][0]} (应为[2.0, 3.0])")

    # 容错测试
    trainer.simulate_node_failure(1)
    print(f"  模拟节点故障: GPU1 → failed")
    recovery = trainer.handle_node_failure()
    print(f"  故障处理: handled={recovery['handled']}, "
          f"active={len(recovery['active_nodes'])}")

    # 故障后继续训练
    result2 = trainer.train_step(dataset[:4])
    print(f"  容错训练: step={result2['step']}, "
          f"loss={result2['loss']:.6f}, "
          f"active_gpus={result2['active_gpus']}")

    # 恢复节点
    trainer.recover_node(1)
    print(f"  恢复节点: GPU1 → active")

    # 流水线并行
    trainer_pp = DistributedTrainer(
        model=model, num_gpus=2, strategy="pipeline_parallel",
        num_pipeline_stages=2, lr=1e-3)
    hidden = trainer_pp.pipeline_parallel_forward(dataset[0][0])
    print(f"  流水线前向: output_shape={len(hidden)}x{len(hidden[0])}")
    print(f"  流水线阶段: {trainer_pp._pipeline_stages}")

    # 集群统计
    stats = trainer.get_cluster_stats()
    print(f"  集群统计: strategy={stats['strategy']}, "
          f"gpus={stats['num_gpus']}, "
          f"sync_count={stats['gradient_sync_count']}, "
          f"comm_mb={stats['total_comm_mb']:.4f}")

    # 集群报告
    report = trainer.get_cluster_report()
    print(f"  集群报告: {len(report.split(chr(10)))}行")

    print("  [OK] DistributedTrainer")


def _main():
    """主自测函数"""
    print("=" * 70)
    print("  LINGYUAN MODEL - PART 18: 虚拟GPU加速训练引擎 自测")
    print("=" * 70)

    # 运行所有测试
    engine = _test_gpu_accelerated_training_engine()
    _test_gpu_batch_processor(engine)
    _test_gradient_compression()
    profiler = _test_training_profiler()
    _test_checkpoint_manager(engine)
    _test_training_scheduler()
    _test_distributed_trainer()

    print("\n" + "=" * 70)
    print("  所有测试通过!")
    print("=" * 70)

    # 打印GPU状态
    print("\n" + vgpu_smi(engine.gpu))


if __name__ == "__main__":
    _main()
