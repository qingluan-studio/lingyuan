#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 灵元模型项目 (LingYuan Model Project) — 第 23 模块
 量子化推理引擎 (Quantized Inference Engine)
================================================================================

模块概述:
    本模块实现了完整的模型量化推理引擎, 核心目标是将灵元模型的推理
    内存占用降低 4x-8x, 同时保持精度损失在可接受范围内, 使模型能够在
    移动端 (手机/平板) 上流畅运行。

    量化是"虚拟GPU"路线的关键配套:
    - 虚拟GPU (part17) 解决"计算在哪里跑"的问题
    - 量化 (本模块) 解决"模型塞不塞得下"的问题
    - 边缘部署 (part28) 解决"如何部署到设备"的问题

核心组件:
    1. QuantizationConfig       — 量化配置 (精度/方案/粒度/校准方法)
    2. CalibrationCollector      — 校准数据收集器 (激活统计)
    3. MinMaxCalibrator          — Min-Max校准器
    4. PercentileCalibrator      — 百分位校准器 (抗离群值)
    5. MSECalibrator             — MSE最优校准器 (搜索最优截断)
    6. ACIQCalibrator            — ACIQ校准器 (高斯假设)
    7. SymmetricQuantizer        — 对称量化器 (INT8/INT4)
    8. AsymmetricQuantizer       — 非对称量化器 (零点偏移)
    9. PerChannelQuantizer       — 逐通道量化器
   10. PerTensorQuantizer        — 逐张量量化器
   11. MixedPrecisionQuantizer   — 混合精度量化器 (敏感层FP16)
   12. QuantizedLinear           — 量化线性层 (INT矩阵乘法)
   13. QuantizedAttention        — 量化注意力 (QK量化+Softmax FP16)
   14. QuantizedEmbedding        — 量化嵌入层
   15. QuantizedLayerNorm        — 量化LayerNorm (FP16计算)
   16. QuantizedRMSNorm          — 量化RMSNorm
   17. QuantizedTransformer      — 量化Transformer (整合所有量化层)
   18. QuantizationAwareTraining — 量化感知训练 (伪量化+STE)
   19. DynamicQuantizer          — 动态量化 (推理时量化激活)
   20. WeightOnlyQuantizer       — 仅权重量化 (AWQ风格)
   21. GPTQQuantizer             — GPTQ量化 (二阶Hessian补偿)
   22. MobileInferenceOptimizer  — 移动端推理优化器
   23. QuantizationBenchmark     — 量化基准测试
   24. QuantizationProfiler      — 量化分析器 (逐层敏感度分析)

设计原则:
    - 纯 Python 标准库实现, 零外部依赖
    - 量化推理用整数运算模拟 (实际部署时映射到硬件INT指令)
    - 所有类可独立实例化和运行
    - 完整的校准/量化/评估流水线
    - 支持INT8/INT4/混合精度/动态量化/仅权重量化

量化数学基础:
    对称量化:   q = round(x / scale),            scale = max(|x|) / (2^(b-1) - 1)
    非对称量化: q = round((x - zero_point) / scale),  scale = (max - min) / (2^b - 1)
    反量化:     x_hat = q * scale + zero_point

作者: 灵元模型项目组
版本: 1.0.0
================================================================================
"""

import os
import sys
import math
import time
import json
import random
import struct
import hashlib
from collections import deque, defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union


# ============================================================
# 枚举定义
# ============================================================

class QuantizationPrecision(IntEnum):
    """量化精度"""
    FP32 = 32       # 不量化
    FP16 = 16       # 半精度
    BF16 = 15       # Brain Float 16
    INT8 = 8        # 8位整数
    INT4 = 4        # 4位整数
    INT2 = 2        # 2位整数 (极端压缩)
    MIXED = 0       # 混合精度


class QuantizationScheme(Enum):
    """量化方案"""
    SYMMETRIC = "symmetric"         # 对称量化: [-scale, +scale]
    ASYMMETRIC = "asymmetric"       # 非对称量化: [min, max]
    SYMMETRIC_UNSIGNED = "sym_unsigned"  # 对称无符号 (ReLU后)


class QuantizationGranularity(Enum):
    """量化粒度"""
    PER_TENSOR = "per_tensor"       # 整个张量共享一组参数
    PER_CHANNEL = "per_channel"     # 每个通道独立参数
    PER_GROUP = "per_group"         # 每组(如128)独立参数 (用于INT4)


class CalibrationMethod(Enum):
    """校准方法"""
    MIN_MAX = "min_max"             # 简单Min-Max
    PERCENTILE = "percentile"       # 百分位截断
    MSE = "mse"                     # 均方误差最优
    ACIQ = "aciq"                   # ACIQ (高斯假设)
    ENTROPY = "entropy"             # 熵最小化
    GRADIENT = "gradient"           # 梯度校准 (需反向传播)


class LayerSensitivity(Enum):
    """层敏感度等级"""
    INSENSITIVE = 0     # 不敏感, 可INT4
    LOW = 1             # 低敏感, INT8
    MEDIUM = 2          # 中等敏感, INT8+校准
    HIGH = 3            # 高敏感, FP16
    CRITICAL = 4        # 极度敏感, FP32 (不量化)


# ============================================================
# 量化配置
# ============================================================

@dataclass
class QuantizationConfig:
    """量化配置 — 控制全局量化策略

    属性:
        weight_precision: 权重量化精度
        activation_precision: 激活量化精度
        scheme: 量化方案 (对称/非对称)
        granularity: 量化粒度 (逐张量/逐通道)
        calibration_method: 校准方法
        calibration_samples: 校准样本数
        percentile: 百分位截断值 (如99.9)
        group_size: 分组量化大小 (INT4常用128)
        sensitive_layers: 敏感层列表 (保持高精度)
        skip_layers: 跳过量化的层
        weight_only: 是否仅权重量化 (激活保持FP16)
        dynamic: 是否动态量化 (推理时量化激活)
    """
    weight_precision: QuantizationPrecision = QuantizationPrecision.INT8
    activation_precision: QuantizationPrecision = QuantizationPrecision.INT8
    scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
    granularity: QuantizationGranularity = QuantizationGranularity.PER_CHANNEL
    calibration_method: CalibrationMethod = CalibrationMethod.PERCENTILE
    calibration_samples: int = 128
    percentile: float = 99.9
    group_size: int = 128
    sensitive_layers: List[str] = field(default_factory=list)
    skip_layers: List[str] = field(default_factory=lambda: ["embedding", "lm_head"])
    weight_only: bool = False
    dynamic: bool = False

    # 预设配置
    @classmethod
    def preset_int8(cls) -> "QuantizationConfig":
        """INT8量化预设 (平衡精度与速度)"""
        return cls(
            weight_precision=QuantizationPrecision.INT8,
            activation_precision=QuantizationPrecision.INT8,
            scheme=QuantizationScheme.SYMMETRIC,
            granularity=QuantizationGranularity.PER_CHANNEL,
            calibration_method=CalibrationMethod.PERCENTILE,
        )

    @classmethod
    def preset_int4(cls) -> "QuantizationConfig":
        """INT4量化预设 (最大化压缩)"""
        return cls(
            weight_precision=QuantizationPrecision.INT4,
            activation_precision=QuantizationPrecision.INT8,
            scheme=QuantizationScheme.SYMMETRIC,
            granularity=QuantizationGranularity.PER_GROUP,
            group_size=128,
            calibration_method=CalibrationMethod.MSE,
        )

    @classmethod
    def preset_mixed(cls) -> "QuantizationConfig":
        """混合精度预设 (敏感层FP16, 其余INT8)"""
        return cls(
            weight_precision=QuantizationPrecision.MIXED,
            activation_precision=QuantizationPrecision.INT8,
            scheme=QuantizationScheme.SYMMETRIC,
            granularity=QuantizationGranularity.PER_CHANNEL,
            calibration_method=CalibrationMethod.PERCENTILE,
            sensitive_layers=["lm_head", "attention_output", "final_norm"],
            skip_layers=["embedding"],  # lm_head放sensitive_layers中
        )

    @classmethod
    def preset_mobile(cls) -> "QuantizationConfig":
        """移动端预设 (INT4权重 + FP16激活)"""
        return cls(
            weight_precision=QuantizationPrecision.INT4,
            activation_precision=QuantizationPrecision.FP16,
            scheme=QuantizationScheme.SYMMETRIC,
            granularity=QuantizationGranularity.PER_GROUP,
            group_size=128,
            weight_only=True,
            calibration_method=CalibrationMethod.MSE,
        )

    @classmethod
    def preset_dynamic(cls) -> "QuantizationConfig":
        """动态量化预设 (权重静态INT8, 激活动态INT8)"""
        return cls(
            weight_precision=QuantizationPrecision.INT8,
            activation_precision=QuantizationPrecision.INT8,
            scheme=QuantizationScheme.SYMMETRIC,
            granularity=QuantizationGranularity.PER_CHANNEL,
            dynamic=True,
        )

    def to_dict(self) -> Dict:
        return {
            "weight_precision": self.weight_precision.name,
            "activation_precision": self.activation_precision.name,
            "scheme": self.scheme.value,
            "granularity": self.granularity.value,
            "calibration_method": self.calibration_method.value,
            "calibration_samples": self.calibration_samples,
            "percentile": self.percentile,
            "group_size": self.group_size,
            "sensitive_layers": self.sensitive_layers,
            "skip_layers": self.skip_layers,
            "weight_only": self.weight_only,
            "dynamic": self.dynamic,
        }

    def __repr__(self) -> str:
        return (f"QuantizationConfig(w={self.weight_precision.name}, "
                f"a={self.activation_precision.name}, "
                f"{self.scheme.value}, {self.granularity.value})")


# ============================================================
# 数学辅助函数
# ============================================================

def _clamp(val: float, lo: float, hi: float) -> float:
    """截断到范围"""
    return max(lo, min(hi, val))


def _quantize_value(x: float, scale: float, zero_point: float,
                    qmin: int, qmax: int) -> int:
    """量化单个值: x -> q"""
    q = round(x / scale + zero_point) if scale > 0 else 0
    return int(_clamp(q, qmin, qmax))


def _dequantize_value(q: int, scale: float, zero_point: float) -> float:
    """反量化: q -> x_hat"""
    return (q - zero_point) * scale


def _compute_qrange(bits: int, signed: bool = True) -> Tuple[int, int]:
    """计算量化范围 [qmin, qmax]"""
    if signed:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << bits) - 1
    return qmin, qmax


def _compute_scale_symmetric(max_abs: float, bits: int) -> float:
    """对称量化 scale = max_abs / (2^(b-1) - 1)"""
    qmax = (1 << (bits - 1)) - 1
    return max_abs / qmax if max_abs > 0 else 1.0


def _compute_scale_asymmetric(min_val: float, max_val: float,
                               bits: int) -> Tuple[float, float]:
    """非对称量化 scale, zero_point"""
    qmin = 0
    qmax = (1 << bits) - 1
    range_val = max_val - min_val
    if range_val < 1e-12:
        return 1.0, 0.0
    scale = range_val / (qmax - qmin)
    zero_point = qmin - round(min_val / scale)
    zero_point = _clamp(zero_point, qmin, qmax)
    return scale, zero_point


def _quantize_tensor_symmetric(tensor: List[float], scale: float,
                                bits: int) -> List[int]:
    """对称量化一个一维张量"""
    qmin, qmax = _compute_qrange(bits, signed=True)
    return [_clamp(round(x / scale), qmin, qmax) for x in tensor]


def _quantize_tensor_asymmetric(tensor: List[float], scale: float,
                                 zero_point: float, bits: int) -> List[int]:
    """非对称量化一个一维张量"""
    qmin, qmax = _compute_qrange(bits, signed=False)
    return [_clamp(round(x / scale + zero_point), qmin, qmax) for x in tensor]


def _dequantize_tensor(quantized: List[int], scale: float,
                        zero_point: float = 0.0) -> List[float]:
    """反量化一维张量"""
    return [(q - zero_point) * scale for q in quantized]


def _quantize_matrix_symmetric(matrix: List[List[float]], scale: float,
                                bits: int) -> List[List[int]]:
    """对称量化二维矩阵"""
    qmin, qmax = _compute_qrange(bits, signed=True)
    result = []
    for row in matrix:
        result.append([_clamp(round(x / scale), qmin, qmax) for x in row])
    return result


def _dequantize_matrix(quantized: List[List[int]], scale: float,
                        zero_point: float = 0.0) -> List[List[float]]:
    """反量化二维矩阵"""
    return [[(q - zero_point) * scale for q in row] for row in quantized]


def _quantize_matrix_per_channel(matrix: List[List[float]],
                                  scales: List[float],
                                  bits: int,
                                  axis: int = 0) -> List[List[int]]:
    """逐通道量化二维矩阵

    Args:
        matrix: 输入矩阵
        scales: 每通道的scale
        bits: 量化位数
        axis: 量化轴 (0=行, 1=列)
    """
    qmin, qmax = _compute_qrange(bits, signed=True)
    rows, cols = len(matrix), len(matrix[0]) if matrix else 0
    result = [[0] * cols for _ in range(rows)]

    if axis == 0:
        # 每行独立量化
        for i in range(rows):
            s = scales[i] if i < len(scales) else 1.0
            for j in range(cols):
                result[i][j] = _clamp(round(matrix[i][j] / s), qmin, qmax)
    else:
        # 每列独立量化
        for j in range(cols):
            s = scales[j] if j < len(scales) else 1.0
            for i in range(rows):
                result[i][j] = _clamp(round(matrix[i][j] / s), qmin, qmax)
    return result


def _dequantize_matrix_per_channel(quantized: List[List[int]],
                                    scales: List[float],
                                    axis: int = 0) -> List[List[float]]:
    """逐通道反量化二维矩阵"""
    rows, cols = len(quantized), len(quantized[0]) if quantized else 0
    result = [[0.0] * cols for _ in range(rows)]
    if axis == 0:
        for i in range(rows):
            s = scales[i] if i < len(scales) else 1.0
            for j in range(cols):
                result[i][j] = quantized[i][j] * s
    else:
        for j in range(cols):
            s = scales[j] if j < len(scales) else 1.0
            for i in range(rows):
                result[i][j] = quantized[i][j] * s
    return result


def _quantize_matrix_per_group(matrix: List[List[float]],
                                bits: int,
                                group_size: int) -> Tuple[List[List[int]], List[List[float]]]:
    """分组量化二维矩阵 (INT4常用)

    将每行分成若干组, 每组独立量化

    Returns:
        quantized: 量化后的整数矩阵
        scales: 每组的scale (形状与matrix相同行数, 列数=cols/group_size)
    """
    qmin, qmax = _compute_qrange(bits, signed=True)
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    quantized = [[0] * cols for _ in range(rows)]
    scales_list = []

    for i in range(rows):
        row_scales = []
        for j_start in range(0, cols, group_size):
            j_end = min(j_start + group_size, cols)
            group = matrix[i][j_start:j_end]
            max_abs = max(abs(x) for x in group) if group else 0
            scale = _compute_scale_symmetric(max_abs, bits)
            row_scales.append(scale)
            for j in range(j_start, j_end):
                quantized[i][j] = _clamp(round(matrix[i][j] / scale), qmin, qmax)
        scales_list.append(row_scales)

    return quantized, scales_list


def _dequantize_matrix_per_group(quantized: List[List[int]],
                                  scales: List[List[float]],
                                  group_size: int) -> List[List[float]]:
    """分组反量化"""
    rows = len(quantized)
    cols = len(quantized[0]) if quantized else 0
    result = [[0.0] * cols for _ in range(rows)]

    for i in range(rows):
        for g, j_start in enumerate(range(0, cols, group_size)):
            j_end = min(j_start + group_size, cols)
            s = scales[i][g] if g < len(scales[i]) else 1.0
            for j in range(j_start, j_end):
                result[i][j] = quantized[i][j] * s
    return result


def _int_matmul(a_q: List[List[int]], b_q: List[List[int]],
                a_scale: float, b_scale: float) -> List[List[float]]:
    """整数量化矩阵乘法 (模拟INT GEMM)

    实际部署时映射到硬件INT8/INT4 GEMM指令
    A形状: (m, k), B形状: (n, k) [out_dim, in_dim]
    C = (A @ B^T) * scale_a * scale_b
    """
    m = len(a_q)
    k = len(a_q[0]) if a_q else 0
    n = len(b_q) if b_q else 0  # out_dim = B的行数
    output_scale = a_scale * b_scale

    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        ai = a_q[i]
        ri = result[i]
        for j in range(n):
            bj = b_q[j]
            acc = 0
            for p in range(k):
                acc += ai[p] * bj[p]
            ri[j] = acc * output_scale
    return result


def _int_matmul_per_channel(a_q: List[List[int]], b_q: List[List[int]],
                             a_scale: float,
                             b_scales: List[float]) -> List[List[float]]:
    """逐通道整数量化矩阵乘法

    权重按输出通道(axis=0)逐通道量化
    B形状: (out_dim, in_dim), A形状: (seq, in_dim)
    C[i][j] = sum_p(A[i][p] * B[j][p]) * a_scale * b_scales[j]
    """
    m = len(a_q)
    k = len(a_q[0]) if a_q else 0
    n = len(b_q) if b_q else 0  # out_dim = 行数

    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        ai = a_q[i]
        ri = result[i]
        for j in range(n):
            bj = b_q[j]
            acc = 0
            for p in range(k):
                acc += ai[p] * bj[p]
            ri[j] = acc * a_scale * (b_scales[j] if j < len(b_scales) else 1.0)
    return result


# ============================================================
# 校准数据收集器
# ============================================================

class CalibrationCollector:
    """校准数据收集器 — 收集推理过程中的激活统计

    在校准阶段运行模型, 记录每层的激活值分布,
    用于后续确定量化参数(scale, zero_point)

    统计内容:
    - 每层激活的 min, max, 均值, 方差
    - 直方图分布 (用于MSE/熵校准)
    - 绝对值最大值 (对称量化)
    """

    def __init__(self, num_bins: int = 2048):
        self.num_bins = num_bins
        self.layer_stats: Dict[str, Dict] = {}
        self._hooks: Dict[str, Callable] = {}
        self._collecting = False

    def register_layer(self, layer_name: str) -> None:
        """注册一个层进行统计"""
        self.layer_stats[layer_name] = {
            "min": float('inf'),
            "max": float('-inf'),
            "abs_max": 0.0,
            "mean": 0.0,
            "m2": 0.0,  # Welford算法的M2
            "count": 0,
            "histogram": [0] * self.num_bins,
            "hist_min": float('inf'),
            "hist_max": float('-inf'),
            "samples": deque(maxlen=10),  # 保留少量样本
        }

    def collect(self, layer_name: str, activation: List[float]) -> None:
        """收集一层的激活统计

        使用Welford在线算法计算均值/方差
        """
        if layer_name not in self.layer_stats:
            self.register_layer(layer_name)

        stats = self.layer_stats[layer_name]
        n = len(activation)

        if n == 0:
            return

        # Min/Max/AbsMax
        layer_min = min(activation)
        layer_max = max(activation)
        layer_abs_max = max(abs(x) for x in activation)

        stats["min"] = min(stats["min"], layer_min)
        stats["max"] = max(stats["max"], layer_max)
        stats["abs_max"] = max(stats["abs_max"], layer_abs_max)

        # Welford在线均值/方差
        prev_count = stats["count"]
        new_count = prev_count + n

        batch_mean = sum(activation) / n
        batch_m2 = sum((x - batch_mean) ** 2 for x in activation)

        delta = batch_mean - stats["mean"]
        stats["mean"] += delta * n / new_count
        stats["m2"] += batch_m2 + delta ** 2 * n * prev_count / new_count
        stats["count"] = new_count

        # 直方图 (动态范围)
        if stats["hist_min"] > layer_min:
            stats["hist_min"] = layer_min
        if stats["hist_max"] < layer_max:
            stats["hist_max"] = layer_max

        hist_range = stats["hist_max"] - stats["hist_min"]
        if hist_range > 0:
            for x in activation:
                bin_idx = int((x - stats["hist_min"]) / hist_range * (self.num_bins - 1))
                bin_idx = _clamp(bin_idx, 0, self.num_bins - 1)
                stats["histogram"][bin_idx] += 1

        # 保留少量样本
        if len(stats["samples"]) < 10:
            stats["samples"].append(activation[:64])

    def collect_matrix(self, layer_name: str, matrix: List[List[float]]) -> None:
        """收集矩阵激活统计 (展平后统计)"""
        flattened = []
        for row in matrix:
            flattened.extend(row)
        self.collect(layer_name, flattened)

    def get_stats(self, layer_name: str) -> Optional[Dict]:
        """获取某层的统计信息"""
        if layer_name not in self.layer_stats:
            return None
        stats = self.layer_stats[layer_name]
        count = stats["count"]
        variance = stats["m2"] / count if count > 1 else 0.0
        std = math.sqrt(variance)
        return {
            "min": stats["min"],
            "max": stats["max"],
            "abs_max": stats["abs_max"],
            "mean": stats["mean"],
            "std": std,
            "variance": variance,
            "count": count,
            "histogram": stats["histogram"],
            "hist_min": stats["hist_min"],
            "hist_max": stats["hist_max"],
        }

    def get_all_stats(self) -> Dict[str, Dict]:
        """获取所有层统计"""
        return {name: self.get_stats(name) for name in self.layer_stats}

    def reset(self) -> None:
        """重置所有统计"""
        self.layer_stats.clear()

    def summary(self) -> str:
        """生成摘要字符串"""
        lines = [f"{'Layer':<30} {'Min':>10} {'Max':>10} {'AbsMax':>10} {'Mean':>10} {'Std':>10}"]
        lines.append("-" * 85)
        for name, _ in sorted(self.layer_stats.items()):
            s = self.get_stats(name)
            if s:
                lines.append(f"{name:<30} {s['min']:>10.4f} {s['max']:>10.4f} "
                             f"{s['abs_max']:>10.4f} {s['mean']:>10.4f} {s['std']:>10.4f}")
        return "\n".join(lines)


# ============================================================
# 校准器
# ============================================================

class MinMaxCalibrator:
    """Min-Max校准器

    最简单的校准方法: 直接使用激活的min/max作为量化范围
    优点: 快速, 无需搜索
    缺点: 对离群值敏感
    """

    def __init__(self, collector: CalibrationCollector):
        self.collector = collector

    def calibrate(self, layer_name: str,
                  scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
                  ) -> Tuple[float, float]:
        """校准一层, 返回 (scale, zero_point)"""
        stats = self.collector.get_stats(layer_name)
        if not stats:
            return 1.0, 0.0

        if scheme == QuantizationScheme.SYMMETRIC:
            max_abs = max(abs(stats["min"]), abs(stats["max"]))
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            return scale, 0.0
        else:
            min_val, max_val = stats["min"], stats["max"]
            scale, zp = _compute_scale_asymmetric(min_val, max_val, 8)
            return scale, zp


class PercentileCalibrator:
    """百分位校准器

    截断极端离群值, 使用百分位范围作为量化范围
    优点: 抗离群值, 精度好
    缺点: 需要直方图, 稍慢
    """

    def __init__(self, collector: CalibrationCollector, percentile: float = 99.9):
        self.collector = collector
        self.percentile = percentile

    def calibrate(self, layer_name: str,
                  scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
                  ) -> Tuple[float, float]:
        stats = self.collector.get_stats(layer_name)
        if not stats:
            return 1.0, 0.0

        histogram = stats["histogram"]
        hist_min = stats["hist_min"]
        hist_max = stats["hist_max"]
        total = sum(histogram)

        if total == 0:
            return 1.0, 0.0

        # 计算上下百分位
        lower_pct = (100.0 - self.percentile) / 2.0
        upper_pct = 100.0 - lower_pct

        cumsum = 0
        lower_idx = 0
        upper_idx = len(histogram) - 1
        hist_range = hist_max - hist_min

        for i, count in enumerate(histogram):
            cumsum += count
            pct = cumsum / total * 100
            if pct >= lower_pct and lower_idx == 0:
                lower_idx = i
            if pct >= upper_pct:
                upper_idx = i
                break

        min_val = hist_min + lower_idx / len(histogram) * hist_range
        max_val = hist_min + upper_idx / len(histogram) * hist_range

        if scheme == QuantizationScheme.SYMMETRIC:
            max_abs = max(abs(min_val), abs(max_val))
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            return scale, 0.0
        else:
            return _compute_scale_asymmetric(min_val, max_val, 8)


class MSECalibrator:
    """MSE最优校准器

    搜索最优截断阈值, 使量化前后MSE最小
    优点: 精度最优
    缺点: 需要搜索, 最慢
    """

    def __init__(self, collector: CalibrationCollector, num_grid: int = 80):
        self.collector = collector
        self.num_grid = num_grid

    def calibrate(self, layer_name: str,
                  scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
                  bits: int = 8
                  ) -> Tuple[float, float]:
        stats = self.collector.get_stats(layer_name)
        if not stats:
            return 1.0, 0.0

        histogram = stats["histogram"]
        hist_min = stats["hist_min"]
        hist_max = stats["hist_max"]
        total = sum(histogram)

        if total == 0:
            return 1.0, 0.0

        hist_range = hist_max - hist_min
        if hist_range < 1e-12:
            return 1.0, 0.0

        # 搜索最优截断: 从max_abs的50%到100%
        abs_max = max(abs(stats["min"]), abs(stats["max"]))
        best_mse = float('inf')
        best_scale = abs_max / 127.0 if abs_max > 0 else 1.0
        best_zp = 0.0

        for ratio in [0.5 + 0.5 * i / self.num_grid for i in range(self.num_grid + 1)]:
            threshold = abs_max * ratio
            if threshold < 1e-12:
                continue

            if scheme == QuantizationScheme.SYMMETRIC:
                scale = threshold / ((1 << (bits - 1)) - 1)
                zp = 0.0
            else:
                scale, zp = _compute_scale_asymmetric(-threshold, threshold, bits)

            # 计算MSE
            mse = self._compute_mse(histogram, hist_min, hist_range,
                                     scale, zp, bits, scheme)

            if mse < best_mse:
                best_mse = mse
                best_scale = scale
                best_zp = zp

        return best_scale, best_zp

    def _compute_mse(self, histogram: List[int], hist_min: float,
                     hist_range: float, scale: float, zero_point: float,
                     bits: int, scheme: QuantizationScheme) -> float:
        """计算给定量化参数下的MSE"""
        qmin, qmax = _compute_qrange(bits, signed=(scheme == QuantizationScheme.SYMMETRIC))
        mse = 0.0
        total = sum(histogram)

        for i, count in enumerate(histogram):
            if count == 0:
                continue
            x = hist_min + i / len(histogram) * hist_range
            if scale > 0:
                q = round(x / scale + zero_point)
                q = _clamp(q, qmin, qmax)
                x_hat = (q - zero_point) * scale
            else:
                x_hat = 0.0
            mse += count * (x - x_hat) ** 2

        return mse / total


class ACIQCalibrator:
    """ACIQ (Analytical Clipping for Integer Quantization) 校准器

    基于高斯假设, 解析地计算最优截断阈值
    论文: "Analytical clipping for integer quantization of neural networks"
    """

    # 预计算的INT8最优截断 alpha (高斯分布)
    _ALPHA_TABLE = {
        8: 3.96,   # INT8 对称
        4: 2.48,   # INT4 对称
    }

    def __init__(self, collector: CalibrationCollector):
        self.collector = collector

    def calibrate(self, layer_name: str,
                  scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
                  bits: int = 8
                  ) -> Tuple[float, float]:
        stats = self.collector.get_stats(layer_name)
        if not stats:
            return 1.0, 0.0

        mean = stats["mean"]
        std = stats["std"]

        if std < 1e-12:
            return 1.0, 0.0

        # 最优截断 = alpha * std
        alpha = self._ALPHA_TABLE.get(bits, 3.96)
        threshold = alpha * std

        if scheme == QuantizationScheme.SYMMETRIC:
            scale = threshold / ((1 << (bits - 1)) - 1)
            return scale, 0.0
        else:
            min_val = mean - threshold
            max_val = mean + threshold
            return _compute_scale_asymmetric(min_val, max_val, bits)


# ============================================================
# 量化器
# ============================================================

class SymmetricQuantizer:
    """对称量化器

    对称量化: 量化范围 [-max_abs, +max_abs]
    适合权重 (权重通常零均值)
    公式: q = round(x / scale), scale = max_abs / (2^(b-1) - 1)
    """

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin, self.qmax = _compute_qrange(bits, signed=True)
        self.scale: float = 1.0
        self._calibrated = False

    def calibrate(self, data: List[float]) -> float:
        """校准: 从数据中计算scale"""
        max_abs = max(abs(x) for x in data) if data else 0
        self.scale = _compute_scale_symmetric(max_abs, self.bits)
        self._calibrated = True
        return self.scale

    def calibrate_matrix(self, matrix: List[List[float]]) -> float:
        """校准矩阵 (逐张量)"""
        max_abs = 0.0
        for row in matrix:
            for x in row:
                max_abs = max(max_abs, abs(x))
        self.scale = _compute_scale_symmetric(max_abs, self.bits)
        self._calibrated = True
        return self.scale

    def quantize(self, data: List[float]) -> List[int]:
        """量化一维数据"""
        return [_clamp(round(x / self.scale), self.qmin, self.qmax) for x in data]

    def quantize_matrix(self, matrix: List[List[float]]) -> List[List[int]]:
        """量化二维矩阵"""
        return [[_clamp(round(x / self.scale), self.qmin, self.qmax) for x in row]
                for row in matrix]

    def dequantize(self, quantized: List[int]) -> List[float]:
        """反量化一维"""
        return [q * self.scale for q in quantized]

    def dequantize_matrix(self, quantized: List[List[int]]) -> List[List[float]]:
        """反量化二维"""
        return [[q * self.scale for q in row] for row in quantized]

    def quantize_dequantize(self, data: List[float]) -> List[float]:
        """量化后立即反量化 (QAT中使用, 模拟量化误差)"""
        q = self.quantize(data)
        return self.dequantize(q)

    def quantize_dequantize_matrix(self, matrix: List[List[float]]) -> List[List[float]]:
        """矩阵量化-反量化 (模拟量化误差)"""
        q = self.quantize_matrix(matrix)
        return self.dequantize_matrix(q)

    def estimate_error(self, data: List[float]) -> Dict[str, float]:
        """评估量化误差"""
        quantized = self.quantize(data)
        dequantized = self.dequantize(quantized)
        errors = [d - dq for d, dq in zip(data, dequantized)]
        mse = sum(e ** 2 for e in errors) / len(errors) if errors else 0
        mae = sum(abs(e) for e in errors) / len(errors) if errors else 0
        max_err = max(abs(e) for e in errors) if errors else 0
        signal_power = sum(d ** 2 for d in data) / len(data) if data else 1
        psnr = 10 * math.log10(signal_power / mse) if mse > 0 else float('inf')
        snr = 10 * math.log10(signal_power / mse) if mse > 0 else float('inf')
        return {
            "mse": mse,
            "mae": mae,
            "max_error": max_err,
            "psnr_db": psnr,
            "snr_db": snr,
            "scale": self.scale,
            "bits": self.bits,
            "compression_ratio": 32.0 / self.bits,
        }


class AsymmetricQuantizer:
    """非对称量化器

    非对称量化: 量化范围 [min, max]
    适合激活 (激活可能非零均值, 如ReLU后全正)
    公式: q = round((x - zero_point) / scale)
    """

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin, self.qmax = _compute_qrange(bits, signed=False)
        self.scale: float = 1.0
        self.zero_point: float = 0.0
        self._calibrated = False

    def calibrate(self, data: List[float]) -> Tuple[float, float]:
        min_val = min(data) if data else 0
        max_val = max(data) if data else 0
        self.scale, self.zero_point = _compute_scale_asymmetric(min_val, max_val, self.bits)
        self._calibrated = True
        return self.scale, self.zero_point

    def quantize(self, data: List[float]) -> List[int]:
        return [_clamp(round(x / self.scale + self.zero_point), self.qmin, self.qmax)
                for x in data]

    def dequantize(self, quantized: List[int]) -> List[float]:
        return [(q - self.zero_point) * self.scale for q in quantized]

    def quantize_dequantize(self, data: List[float]) -> List[float]:
        q = self.quantize(data)
        return self.dequantize(q)


class PerChannelQuantizer:
    """逐通道量化器

    对权重矩阵的每个输出通道独立量化
    优点: 精度更高 (不同通道的值域可能差异大)
    """

    def __init__(self, bits: int = 8, axis: int = 0):
        self.bits = bits
        self.axis = axis  # 0=行(输出通道), 1=列
        self.qmin, self.qmax = _compute_qrange(bits, signed=True)
        self.scales: List[float] = []
        self._calibrated = False

    def calibrate_matrix(self, matrix: List[List[float]]) -> List[float]:
        """校准矩阵, 返回逐通道scale列表"""
        rows, cols = len(matrix), len(matrix[0]) if matrix else 0
        self.scales = []

        if self.axis == 0:
            for i in range(rows):
                max_abs = max(abs(x) for x in matrix[i]) if matrix[i] else 0
                self.scales.append(_compute_scale_symmetric(max_abs, self.bits))
        else:
            for j in range(cols):
                max_abs = max(abs(matrix[i][j]) for i in range(rows)) if rows > 0 else 0
                self.scales.append(_compute_scale_symmetric(max_abs, self.bits))

        self._calibrated = True
        return self.scales

    def quantize_matrix(self, matrix: List[List[float]]) -> List[List[int]]:
        return _quantize_matrix_per_channel(matrix, self.scales, self.bits, self.axis)

    def dequantize_matrix(self, quantized: List[List[int]]) -> List[List[float]]:
        return _dequantize_matrix_per_channel(quantized, self.scales, self.axis)

    def quantize_dequantize_matrix(self, matrix: List[List[float]]) -> List[List[float]]:
        q = self.quantize_matrix(matrix)
        return self.dequantize_matrix(q)


class PerGroupQuantizer:
    """分组量化器 (INT4专用)

    将权重矩阵每行分成若干组, 每组独立量化
    优点: INT4精度下保持较好精度
    缺点: 额外的scale存储开销
    """

    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self.qmin, self.qmax = _compute_qrange(bits, signed=True)
        self.scales: List[List[float]] = []  # 每行每组的scale
        self._calibrated = False

    def calibrate_matrix(self, matrix: List[List[float]]) -> List[List[float]]:
        _, self.scales = _quantize_matrix_per_group(
            matrix, self.bits, self.group_size)
        self._calibrated = True
        return self.scales

    def quantize_matrix(self, matrix: List[List[float]]) -> List[List[int]]:
        quantized, self.scales = _quantize_matrix_per_group(
            matrix, self.bits, self.group_size)
        return quantized

    def dequantize_matrix(self, quantized: List[List[int]]) -> List[List[float]]:
        return _dequantize_matrix_per_group(quantized, self.scales, self.group_size)

    def quantize_dequantize_matrix(self, matrix: List[List[float]]) -> List[List[float]]:
        q = self.quantize_matrix(matrix)
        return self.dequantize_matrix(q)


# ============================================================
# 混合精度量化器
# ============================================================

class MixedPrecisionQuantizer:
    """混合精度量化器

    根据层敏感度自动选择量化精度:
    - 不敏感层: INT4 (最大化压缩)
    - 低敏感层: INT8
    - 中敏感层: INT8 + 精细校准
    - 高敏感层: FP16
    - 极度敏感层: FP32 (不量化)
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig.preset_mixed()
        self.layer_precision: Dict[str, QuantizationPrecision] = {}
        self.layer_quantizers: Dict[str, Any] = {}
        self.sensitivity_scores: Dict[str, float] = {}

    def analyze_sensitivity(self, model_layers: Dict[str, List[List[float]]],
                            calibration_data: Optional[List] = None) -> Dict[str, LayerSensitivity]:
        """分析每层量化敏感度

        方法: 对每层分别量化, 测量量化前后MSE变化
        敏感度 = MSE / 原始信号功率
        """
        sensitivities = {}

        for name, weights in model_layers.items():
            if name in self.config.skip_layers:
                sensitivities[name] = LayerSensitivity.CRITICAL
                self.layer_precision[name] = QuantizationPrecision.FP32
                continue

            # 测量INT8量化误差
            quantizer = SymmetricQuantizer(bits=8)
            quantizer.calibrate_matrix(weights)
            error = quantizer.estimate_error([w for row in weights for w in row])

            snr = error["snr_db"]
            self.sensitivity_scores[name] = snr

            if name in self.config.sensitive_layers:
                sensitivities[name] = LayerSensitivity.HIGH
                self.layer_precision[name] = QuantizationPrecision.FP16
            elif snr > 40:
                sensitivities[name] = LayerSensitivity.INSENSITIVE
                self.layer_precision[name] = QuantizationPrecision.INT4
            elif snr > 30:
                sensitivities[name] = LayerSensitivity.LOW
                self.layer_precision[name] = QuantizationPrecision.INT8
            elif snr > 20:
                sensitivities[name] = LayerSensitivity.MEDIUM
                self.layer_precision[name] = QuantizationPrecision.INT8
            else:
                sensitivities[name] = LayerSensitivity.HIGH
                self.layer_precision[name] = QuantizationPrecision.FP16

        return sensitivities

    def quantize_layer(self, name: str, weights: List[List[float]]
                       ) -> Tuple[Any, QuantizationPrecision]:
        """量化单层, 返回 (量化结果, 精度)"""
        precision = self.layer_precision.get(name, QuantizationPrecision.INT8)

        if precision == QuantizationPrecision.FP32:
            return weights, precision

        if precision == QuantizationPrecision.FP16:
            return self._to_fp16(weights), precision

        if precision == QuantizationPrecision.INT4:
            quantizer = PerGroupQuantizer(bits=4, group_size=self.config.group_size)
        else:
            quantizer = PerChannelQuantizer(bits=8)

        quantizer.calibrate_matrix(weights)
        quantized = quantizer.quantize_matrix(weights)
        self.layer_quantizers[name] = quantizer
        return (quantized, quantizer.scales, precision), precision

    @staticmethod
    def _to_fp16(matrix: List[List[float]]) -> List[List[float]]:
        """模拟FP16精度 (截断到16位浮点)"""
        result = []
        for row in matrix:
            new_row = []
            for x in row:
                if x == 0:
                    new_row.append(0.0)
                else:
                    # 模拟FP16: 1位符号 + 5位指数 + 10位尾数
                    sign = 1 if x >= 0 else -1
                    abs_x = abs(x)
                    exp = int(math.floor(math.log2(abs_x)))
                    mantissa = abs_x / (2 ** exp)
                    # 截断尾数到10位
                    mantissa_q = round(mantissa * 1024) / 1024
                    new_row.append(sign * mantissa_q * (2 ** exp))
            result.append(new_row)
        return result

    def get_precision_summary(self) -> Dict[str, str]:
        """获取精度分配摘要"""
        return {name: p.name for name, p in self.layer_precision.items()}


# ============================================================
# 量化线性层
# ============================================================

class QuantizedLinear:
    """量化线性层 — INT8/INT4矩阵乘法

    权重: 静态量化为INT8/INT4 (推理前量化)
    激活: 静态量化(校准后)或动态量化(推理时)

    推理流程:
    1. 量化激活: x_fp32 -> x_int8
    2. INT矩阵乘法: y_int = x_int @ w_int
    3. 反量化: y_fp32 = y_int * scale_x * scale_w + bias
    """

    def __init__(self, weight: List[List[float]], bias: Optional[List[float]] = None,
                 config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig.preset_int8()
        self.weight_shape = (len(weight), len(weight[0]) if weight else 0)
        self.bias = bias

        # 权重量化
        w_bits = self.config.weight_precision.value
        if w_bits >= 32:
            # 不量化
            self.weight_quantized = None
            self.weight_fp = [row[:] for row in weight]
            self.weight_scales = None
            self.weight_quantizer = None
        elif w_bits == 4:
            self.weight_quantizer = PerGroupQuantizer(bits=4,
                                                       group_size=self.config.group_size)
            self.weight_quantizer.calibrate_matrix(weight)
            self.weight_quantized = self.weight_quantizer.quantize_matrix(weight)
            self.weight_scales = self.weight_quantizer.scales
            self.weight_fp = None
        elif w_bits == 8:
            self.weight_quantizer = PerChannelQuantizer(bits=8)
            self.weight_quantizer.calibrate_matrix(weight)
            self.weight_quantized = self.weight_quantizer.quantize_matrix(weight)
            self.weight_scales = self.weight_quantizer.scales
            self.weight_fp = None
        else:
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits)
            self.weight_quantizer.calibrate_matrix(weight)
            self.weight_quantized = self.weight_quantizer.quantize_matrix(weight)
            self.weight_scales = self.weight_quantizer.scale
            self.weight_fp = None

        # 激活量化器 (动态量化时在forward中创建)
        self.activation_quantizer = None
        self._activation_calibrated = False

        # 统计
        self.original_bytes = len(weight) * len(weight[0]) * 4  # float32
        self.quantized_bytes = self._estimate_quantized_bytes()
        self.compression_ratio = self.original_bytes / max(self.quantized_bytes, 1)

    def _estimate_quantized_bytes(self) -> int:
        """估算量化后字节数"""
        rows, cols = self.weight_shape
        w_bits = self.config.weight_precision.value
        if w_bits >= 32:
            return rows * cols * 4
        elif w_bits == 4:
            # INT4 + group scales
            num_groups = (cols + self.config.group_size - 1) // self.config.group_size
            return (rows * cols * 4 // 8) + (rows * num_groups * 4)  # packed + scales
        else:
            # INT8 + per-channel scales
            return (rows * cols) + (rows * 4)  # packed + scales

    def calibrate_activation(self, calibration_data: List[List[float]]) -> None:
        """校准激活量化参数"""
        if self.config.weight_only or self.config.activation_precision.value >= 32:
            self._activation_calibrated = True
            return

        a_bits = self.config.activation_precision.value
        if self.config.scheme == QuantizationScheme.SYMMETRIC:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits)

        # 收集所有激活 (支持2D和3D输入)
        all_activations = []
        for batch in calibration_data:
            if batch and isinstance(batch[0], list):
                for row in batch:
                    all_activations.extend(row)
            else:
                all_activations.extend(batch)

        if isinstance(self.activation_quantizer, SymmetricQuantizer):
            self.activation_quantizer.calibrate(all_activations)
        else:
            self.activation_quantizer.calibrate(all_activations)

        self._activation_calibrated = True

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """量化前向传播

        Args:
            x: (seq × in_dim) 输入激活

        Returns:
            (seq × out_dim) 输出
        """
        w_bits = self.config.weight_precision.value
        a_bits = self.config.activation_precision.value

        # 不量化的情况
        if w_bits >= 32 and a_bits >= 32:
            return self._fp_forward(x)

        # 仅权重量化 (weight-only): 权重反量化后用FP计算
        if self.config.weight_only:
            if self.weight_fp is None:
                w_dequant = self.weight_quantizer.dequantize_matrix(self.weight_quantized)
            else:
                w_dequant = self.weight_fp
            return self._matmul_add_bias(x, w_dequant)

        # 权重和激活都量化
        # 1. 量化激活
        if self.config.dynamic:
            # 动态量化: 推理时实时计算scale
            x_flat = [v for row in x for v in row]
            max_abs = max(abs(v) for v in x_flat) if x_flat else 0
            x_scale = _compute_scale_symmetric(max_abs, a_bits)
        elif self._activation_calibrated and self.activation_quantizer:
            x_scale = self.activation_quantizer.scale
        else:
            x_scale = 1.0

        x_q = [[_clamp(round(v / x_scale), -(1 << (a_bits - 1)), (1 << (a_bits - 1)) - 1)
                for v in row] for row in x]

        # 2. INT矩阵乘法
        if w_bits == 8 and isinstance(self.weight_quantizer, PerChannelQuantizer):
            # 逐通道INT8 GEMM
            result = _int_matmul_per_channel(x_q, self.weight_quantized,
                                              x_scale, self.weight_scales)
        elif w_bits == 4 and isinstance(self.weight_quantizer, PerGroupQuantizer):
            # INT4: 反量化权重后用FP计算 (模拟)
            w_dequant = self.weight_quantizer.dequantize_matrix(self.weight_quantized)
            # 量化激活反量化
            x_dequant = [[v * x_scale for v in row] for row in x_q]
            result = self._matmul_add_bias(x_dequant, w_dequant)
        else:
            # 通用: 反量化权重
            w_dequant = self.weight_quantizer.dequantize_matrix(self.weight_quantized)
            x_dequant = [[v * x_scale for v in row] for row in x_q]
            result = self._matmul_add_bias(x_dequant, w_dequant)

        # 3. 加偏置
        if self.bias is not None:
            for i in range(len(result)):
                for j in range(len(self.bias)):
                    if j < len(result[i]):
                        result[i][j] += self.bias[j]

        return result

    def _fp_forward(self, x: List[List[float]]) -> List[List[float]]:
        """FP32前向传播"""
        if self.weight_fp is not None:
            return self._matmul_add_bias(x, self.weight_fp)
        elif self.weight_quantized is not None:
            w = self.weight_quantizer.dequantize_matrix(self.weight_quantized)
            return self._matmul_add_bias(x, w)
        return x

    @staticmethod
    def _matmul_add_bias(x: List[List[float]], w: List[List[float]],
                          bias: Optional[List[float]] = None) -> List[List[float]]:
        """FP矩阵乘法 + 偏置
        x: (seq, in_dim), w: (out_dim, in_dim), output: (seq, out_dim)
        计算: output = x @ w^T + bias
        """
        m = len(x)
        k = len(x[0]) if x else 0
        n = len(w) if w else 0  # out_dim = w的行数
        result = [[0.0] * n for _ in range(m)]
        for i in range(m):
            xi = x[i]
            ri = result[i]
            for j in range(n):
                wj = w[j]
                acc = 0.0
                for p in range(k):
                    acc += xi[p] * wj[p]
                ri[j] = acc
            if bias is not None:
                for j in range(n):
                    ri[j] += bias[j] if j < len(bias) else 0.0
        return result

    def get_memory_info(self) -> Dict:
        """获取内存信息"""
        return {
            "layer_type": "QuantizedLinear",
            "weight_shape": list(self.weight_shape),
            "weight_precision": self.config.weight_precision.name,
            "activation_precision": self.config.activation_precision.name,
            "original_bytes": self.original_bytes,
            "quantized_bytes": self.quantized_bytes,
            "compression_ratio": round(self.compression_ratio, 2),
            "has_bias": self.bias is not None,
        }


# ============================================================
# 量化嵌入层
# ============================================================

class QuantizedEmbedding:
    """量化嵌入层

    嵌入层通常保持FP32 (查表操作, 量化收益小)
    但对于大词表, 可用INT8减少内存
    """

    def __init__(self, embedding_matrix: List[List[float]],
                 config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig.preset_int8()
        self.vocab_size = len(embedding_matrix)
        self.embed_dim = len(embedding_matrix[0]) if embedding_matrix else 0

        # 嵌入层通常不量化或用INT8
        if "embedding" in self.config.skip_layers or \
           self.config.weight_precision.value >= 32:
            self.embedding = [row[:] for row in embedding_matrix]
            self.quantized = None
            self.scale = None
        else:
            # INT8量化
            quantizer = SymmetricQuantizer(bits=8)
            quantizer.calibrate_matrix(embedding_matrix)
            self.quantized = quantizer.quantize_matrix(embedding_matrix)
            self.scale = quantizer.scale
            self.embedding = None

    def forward(self, token_ids: List[int]) -> List[List[float]]:
        """查表"""
        result = []
        for tid in token_ids:
            if tid < 0 or tid >= self.vocab_size:
                result.append([0.0] * self.embed_dim)
            elif self.embedding is not None:
                result.append(self.embedding[tid][:])
            else:
                # 反量化
                result.append([q * self.scale for q in self.quantized[tid]])
        return result

    def get_memory_info(self) -> Dict:
        if self.embedding is not None:
            bytes_per = 4
            total = self.vocab_size * self.embed_dim * bytes_per
        else:
            bytes_per = 1
            total = self.vocab_size * self.embed_dim * bytes_per + 4  # + scale
        return {
            "layer_type": "QuantizedEmbedding",
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "total_bytes": total,
            "is_quantized": self.embedding is None,
        }


# ============================================================
# 量化RMSNorm
# ============================================================

class QuantizedRMSNorm:
    """量化RMSNorm — 保持FP16精度

    归一化层对量化非常敏感, 通常保持FP16或FP32
    """

    def __init__(self, weight: List[float], eps: float = 1e-6,
                 use_fp16: bool = True):
        self.eps = eps
        self.use_fp16 = use_fp16
        if use_fp16:
            # 模拟FP16
            self.weight = self._to_fp16_vec(weight)
        else:
            self.weight = weight[:]

    @staticmethod
    def _to_fp16_vec(vec: List[float]) -> List[float]:
        result = []
        for x in vec:
            if x == 0:
                result.append(0.0)
            else:
                sign = 1 if x >= 0 else -1
                abs_x = abs(x)
                exp = int(math.floor(math.log2(abs_x)))
                mantissa = abs_x / (2 ** exp)
                mantissa_q = round(mantissa * 1024) / 1024
                result.append(sign * mantissa_q * (2 ** exp))
        return result

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """RMSNorm前向传播"""
        result = []
        for row in x:
            ms = sum(v * v for v in row) / len(row)
            rms = math.sqrt(ms + self.eps)
            normed = [v / rms * w for v, w in zip(row, self.weight)]
            result.append(normed)
        return result


# ============================================================
# 量化注意力
# ============================================================

class QuantizedAttention:
    """量化注意力层

    量化策略:
    - QK矩阵乘法: INT8量化 (注意力分数计算)
    - Softmax: FP16 (非线性操作, 量化敏感)
    - AV矩阵乘法: INT8量化
    - 输出投影: INT8量化
    """

    def __init__(self, wq: List[List[float]], wk: List[List[float]],
                 wv: List[List[float]], wo: List[List[float]],
                 num_heads: int, hidden_dim: int,
                 config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig.preset_int8()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        # 量化各投影层
        self.q_proj = QuantizedLinear(wq, config=self.config)
        self.k_proj = QuantizedLinear(wk, config=self.config)
        self.v_proj = QuantizedLinear(wv, config=self.config)
        self.o_proj = QuantizedLinear(wo, config=self.config)

        # 注意力分数的量化器
        self.attention_quantizer = SymmetricQuantizer(bits=8) if \
            self.config.activation_precision == QuantizationPrecision.INT8 else None

    def forward(self, x: List[List[float]], mask: Optional[List[List[float]]] = None
                ) -> List[List[float]]:
        """量化注意力前向传播"""
        seq_len = len(x)

        # 1. QKV投影 (量化)
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        # 2. 多头分割
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)

        # 3. 注意力计算 (每头)
        attn_output = []
        for h in range(self.num_heads):
            # QK^T (可量化)
            scores = self._matmul(q_heads[h], self._transpose(k_heads[h]))

            # 缩放
            scale_factor = 1.0 / math.sqrt(self.head_dim)
            for i in range(len(scores)):
                for j in range(len(scores[i])):
                    scores[i][j] *= scale_factor

            # Mask
            if mask is not None:
                for i in range(len(scores)):
                    for j in range(len(scores[i])):
                        scores[i][j] += mask[i][j] if i < len(mask) and j < len(mask[i]) else 0

            # Softmax (FP16精度)
            attn_weights = self._softmax_rows(scores)

            # AV (可量化)
            ctx = self._matmul(attn_weights, v_heads[h])
            attn_output.append(ctx)

        # 4. 合并多头
        merged = self._merge_heads(attn_output)

        # 5. 输出投影 (量化)
        output = self.o_proj.forward(merged)

        return output

    def _split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        """分割多头: (seq × hidden) -> (heads × seq × head_dim)"""
        result = [[] for _ in range(self.num_heads)]
        for row in x:
            for h in range(self.num_heads):
                start = h * self.head_dim
                end = start + self.head_dim
                result[h].append(row[start:end])
        return result

    def _merge_heads(self, heads: List[List[List[float]]]) -> List[List[float]]:
        """合并多头: (heads × seq × head_dim) -> (seq × hidden)"""
        if not heads:
            return []
        seq_len = len(heads[0])
        result = []
        for s in range(seq_len):
            row = []
            for h in range(self.num_heads):
                row.extend(heads[h][s])
            result.append(row)
        return result

    @staticmethod
    def _transpose(m: List[List[float]]) -> List[List[float]]:
        if not m:
            return []
        return [list(col) for col in zip(*m)]

    @staticmethod
    def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        m, k, n = len(a), len(a[0]) if a else 0, len(b[0]) if b else 0
        result = [[0.0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for p in range(k):
                    acc += a[i][p] * b[p][j]
                result[i][j] = acc
        return result

    @staticmethod
    def _softmax_rows(matrix: List[List[float]]) -> List[List[float]]:
        result = []
        for row in matrix:
            max_val = max(row) if row else 0
            exps = [math.exp(v - max_val) for v in row]
            total = sum(exps)
            if total > 0:
                result.append([e / total for e in exps])
            else:
                result.append([1.0 / len(row)] * len(row))
        return result

    def get_memory_info(self) -> Dict:
        return {
            "layer_type": "QuantizedAttention",
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "q_proj": self.q_proj.get_memory_info(),
            "k_proj": self.k_proj.get_memory_info(),
            "v_proj": self.v_proj.get_memory_info(),
            "o_proj": self.o_proj.get_memory_info(),
        }


# ============================================================
# 量化Transformer
# ============================================================

class QuantizedTransformer:
    """量化Transformer模型

    将完整的Transformer模型量化, 包括:
    - 嵌入层 (可选量化)
    - 注意力层 (QK/AV量化, Softmax FP16)
    - FFN层 (INT8/INT4)
    - 归一化层 (FP16)
    - 输出投影 (可选量化)
    """

    def __init__(self, model_config: Optional[Dict] = None,
                 config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig.preset_int8()
        self.model_config = model_config or {
            "vocab_size": 1000,
            "hidden_dim": 128,
            "num_heads": 4,
            "num_layers": 2,
            "ffn_dim": 256,
            "max_seq_len": 64,
        }

        self.vocab_size = self.model_config["vocab_size"]
        self.hidden_dim = self.model_config["hidden_dim"]
        self.num_heads = self.model_config["num_heads"]
        self.num_layers = self.model_config["num_layers"]
        self.ffn_dim = self.model_config["ffn_dim"]
        self.max_seq_len = self.model_config["max_seq_len"]

        # 构建量化层
        self._build_layers()

    def _build_layers(self) -> None:
        """构建量化层"""
        h = self.hidden_dim
        f = self.ffn_dim
        v = self.vocab_size

        # 嵌入层
        emb = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(v)]
        self.embedding = QuantizedEmbedding(emb, self.config)

        # Transformer层
        self.layers = []
        for _ in range(self.num_layers):
            layer = self._build_transformer_layer()
            self.layers.append(layer)

        # 最终归一化
        norm_weight = [1.0] * h
        self.final_norm = QuantizedRMSNorm(norm_weight)

        # 输出投影 (LM Head)
        lm_head = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(v)]
        lm_config = QuantizationConfig.preset_int8()
        if "lm_head" in self.config.sensitive_layers:
            lm_config = QuantizationConfig(
                weight_precision=QuantizationPrecision.FP16,
                activation_precision=QuantizationPrecision.FP16,
            )
        self.lm_head = QuantizedLinear(lm_head, config=lm_config)

    def _build_transformer_layer(self) -> Dict:
        """构建一个Transformer层"""
        h = self.hidden_dim
        f = self.ffn_dim

        # 注意力权重
        wq = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(h)]
        wk = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(h)]
        wv = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(h)]
        wo = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(h)]

        # FFN权重
        w1 = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(f)]
        w2 = [[random.gauss(0, 0.02) for _ in range(f)] for _ in range(h)]

        # 归一化
        norm1_weight = [1.0] * h
        norm2_weight = [1.0] * h

        return {
            "attention": QuantizedAttention(wq, wk, wv, wo,
                                             self.num_heads, h, self.config),
            "ffn_w1": QuantizedLinear(w1, config=self.config),
            "ffn_w2": QuantizedLinear(w2, config=self.config),
            "norm1": QuantizedRMSNorm(norm1_weight),
            "norm2": QuantizedRMSNorm(norm2_weight),
        }

    def forward(self, input_ids: List[int]) -> List[List[float]]:
        """量化Transformer前向传播"""
        # 1. 嵌入
        x = self.embedding.forward(input_ids)

        # 2. Transformer层
        for layer in self.layers:
            # Self-attention + residual
            attn_out = layer["attention"].forward(x)
            x = [[xi + ai for xi, ai in zip(xr, ar)]
                 for xr, ar in zip(x, attn_out)]

            # Norm1
            x = layer["norm1"].forward(x)

            # FFN + residual
            ffn_out = layer["ffn_w1"].forward(x)
            ffn_out = [[max(0, v) * v for v in row] for row in ffn_out]  # SiLU
            ffn_out = layer["ffn_w2"].forward(ffn_out)
            x = [[xi + fi for xi, fi in zip(xr, fr)]
                 for xr, fr in zip(x, ffn_out)]

            # Norm2
            x = layer["norm2"].forward(x)

        # 3. 最终归一化
        x = self.final_norm.forward(x)

        # 4. LM Head
        logits = self.lm_head.forward(x)

        return logits

    def get_model_info(self) -> Dict:
        """获取量化模型信息"""
        total_original = 0
        total_quantized = 0
        layer_infos = []

        # 嵌入层
        emb_info = self.embedding.get_memory_info()
        total_original += self.vocab_size * self.hidden_dim * 4
        total_quantized += emb_info["total_bytes"]
        layer_infos.append(emb_info)

        # Transformer层
        for i, layer in enumerate(self.layers):
            attn_info = layer["attention"].get_memory_info()
            for proj_key in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                info = attn_info[proj_key]
                total_original += info["original_bytes"]
                total_quantized += info["quantized_bytes"]

            for ffn_key in ["ffn_w1", "ffn_w2"]:
                info = layer[ffn_key].get_memory_info()
                total_original += info["original_bytes"]
                total_quantized += info["quantized_bytes"]

        # LM Head
        lm_info = self.lm_head.get_memory_info()
        total_original += lm_info["original_bytes"]
        total_quantized += lm_info["quantized_bytes"]

        return {
            "model_type": "QuantizedTransformer",
            "config": self.config.to_dict(),
            "model_config": self.model_config,
            "num_layers": self.num_layers,
            "total_original_bytes": total_original,
            "total_quantized_bytes": total_quantized,
            "compression_ratio": round(total_original / max(total_quantized, 1), 2),
            "total_original_mb": round(total_original / 1024 / 1024, 2),
            "total_quantized_mb": round(total_quantized / 1024 / 1024, 2),
        }


# ============================================================
# 量化感知训练 (QAT)
# ============================================================

class FakeQuantize:
    """伪量化 — QAT中使用

    前向传播: 量化->反量化 (模拟量化误差)
    反向传播: 直通估计器 (Straight-Through Estimator)
    """

    def __init__(self, bits: int = 8, symmetric: bool = True,
                 learnable_scale: bool = False):
        self.bits = bits
        self.symmetric = symmetric
        self.learnable_scale = learnable_scale
        self.scale = 1.0
        self.zero_point = 0.0
        self.qmin, self.qmax = _compute_qrange(bits, signed=symmetric)
        self._calibrated = False
        self._observer_data: List[float] = []

    def observe(self, data: List[float]) -> None:
        """观察数据 (用于校准)"""
        self._observer_data.extend(data[:1000])  # 限制内存

    def calibrate(self) -> None:
        """从观察数据中校准"""
        if not self._observer_data:
            return
        if self.symmetric:
            max_abs = max(abs(x) for x in self._observer_data)
            self.scale = _compute_scale_symmetric(max_abs, self.bits)
        else:
            min_val = min(self._observer_data)
            max_val = max(self._observer_data)
            self.scale, self.zero_point = _compute_scale_asymmetric(
                min_val, max_val, self.bits)
        self._calibrated = True
        self._observer_data.clear()

    def forward(self, x: List[float]) -> List[float]:
        """前向: 量化->反量化 (模拟误差)"""
        if not self._calibrated:
            return x
        if self.symmetric:
            q = [_clamp(round(v / self.scale), self.qmin, self.qmax) for v in x]
            return [qi * self.scale for qi in q]
        else:
            q = [_clamp(round(v / self.scale + self.zero_point), self.qmin, self.qmax)
                 for v in x]
            return [(qi - self.zero_point) * self.scale for qi in q]

    def forward_matrix(self, matrix: List[List[float]]) -> List[List[float]]:
        """矩阵前向"""
        return [self.forward(row) for row in matrix]

    def ste_backward(self, grad: List[float], x: List[float]) -> List[float]:
        """直通估计器反向传播

        量化函数的梯度几乎处处为0, STE直接传递梯度:
        d(loss)/d(x) ≈ d(loss)/d(x_hat)
        但在裁剪范围外, 梯度为0
        """
        if not self._calibrated:
            return grad
        threshold = self.scale * self.qmax
        result = []
        for g, v in zip(grad, x):
            if abs(v) <= threshold:
                result.append(g)  # 范围内: 直通
            else:
                result.append(0.0)  # 范围外: 截断梯度
        return result


class QuantizationAwareTraining:
    """量化感知训练 (QAT)

    在训练过程中插入伪量化节点, 让模型适应量化误差
    训练后可直接转换为真正的量化模型

    流程:
    1. 初始化: 在权重和激活上插入FakeQuantize
    2. 观察期: 前几个epoch只观察, 不量化
    3. 量化期: 开始伪量化训练
    4. 转换: 训练完成后提取量化参数, 转为真量化模型
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig.preset_int8()
        self.weight_fake_quantizers: Dict[str, FakeQuantize] = {}
        self.activation_fake_quantizers: Dict[str, FakeQuantize] = {}
        self.observer_epochs = 1  # 前N个epoch只观察
        self.current_epoch = 0
        self._quantization_active = False

    def register_weight(self, name: str, weight: List[List[float]]) -> FakeQuantize:
        """注册权重伪量化"""
        w_bits = self.config.weight_precision.value
        if w_bits >= 32:
            return None
        fq = FakeQuantize(bits=w_bits, symmetric=True)
        # 立即校准权重 (权重是静态的)
        flat = [v for row in weight for v in row]
        fq.observe(flat)
        fq.calibrate()
        self.weight_fake_quantizers[name] = fq
        return fq

    def register_activation(self, name: str) -> FakeQuantize:
        """注册激活伪量化"""
        a_bits = self.config.activation_precision.value
        if a_bits >= 32:
            return None
        fq = FakeQuantize(bits=a_bits, symmetric=True)
        self.activation_fake_quantizers[name] = fq
        return fq

    def begin_epoch(self, epoch: int) -> None:
        """开始新epoch"""
        self.current_epoch = epoch
        if epoch >= self.observer_epochs:
            self._quantization_active = True
            # 校准所有激活伪量化器
            for fq in self.activation_fake_quantizers.values():
                if not fq._calibrated:
                    fq.calibrate()

    def fake_quantize_weight(self, name: str,
                              weight: List[List[float]]) -> List[List[float]]:
        """伪量化权重"""
        if not self._quantization_active:
            return weight
        fq = self.weight_fake_quantizers.get(name)
        if fq:
            return fq.forward_matrix(weight)
        return weight

    def fake_quantize_activation(self, name: str,
                                  activation: List[List[float]]) -> List[List[float]]:
        """伪量化激活"""
        fq = self.activation_fake_quantizers.get(name)
        if not fq:
            return activation
        if not fq._calibrated:
            # 观察期: 收集数据
            for row in activation:
                fq.observe(row)
            return activation
        if self._quantization_active:
            return fq.forward_matrix(activation)
        return activation

    def get_quantization_params(self) -> Dict[str, Dict]:
        """获取所有量化参数 (用于转换)"""
        params = {
            "weights": {},
            "activations": {},
        }
        for name, fq in self.weight_fake_quantizers.items():
            params["weights"][name] = {
                "scale": fq.scale,
                "zero_point": fq.zero_point,
                "bits": fq.bits,
            }
        for name, fq in self.activation_fake_quantizers.items():
            params["activations"][name] = {
                "scale": fq.scale,
                "zero_point": fq.zero_point,
                "bits": fq.bits,
            }
        return params

    def convert_to_quantized(self, weights: Dict[str, List[List[float]]]
                             ) -> Dict[str, Any]:
        """将QAT模型转换为真量化模型

        Returns:
            量化后的权重和参数
        """
        quantized_weights = {}
        for name, weight in weights.items():
            fq = self.weight_fake_quantizers.get(name)
            if fq and fq.bits < 32:
                # 真量化
                if fq.bits == 8:
                    quantizer = PerChannelQuantizer(bits=8)
                elif fq.bits == 4:
                    quantizer = PerGroupQuantizer(bits=4, group_size=self.config.group_size)
                else:
                    quantizer = SymmetricQuantizer(bits=fq.bits)
                quantizer.calibrate_matrix(weight)
                q_weight = quantizer.quantize_matrix(weight)
                quantized_weights[name] = {
                    "quantized": q_weight,
                    "scales": quantizer.scales,
                    "bits": fq.bits,
                    "quantizer_type": type(quantizer).__name__,
                }
            else:
                quantized_weights[name] = {
                    "quantized": weight,
                    "scales": None,
                    "bits": 32,
                    "quantizer_type": "FP32",
                }
        return quantized_weights


# ============================================================
# 动态量化器
# ============================================================

class DynamicQuantizer:
    """动态量化器

    权重静态量化 (推理前), 激活动态量化 (推理时)
    优点: 无需校准数据, 灵活
    缺点: 推理时多一次量化开销

    适合: 输入分布变化大的场景 (如对话)
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig.preset_dynamic()
        self.weight_quantizers: Dict[str, PerChannelQuantizer] = {}
        self.weight_quantized: Dict[str, List[List[int]]] = {}

    def quantize_weights(self, weights: Dict[str, List[List[float]]]) -> None:
        """静态权重量化"""
        w_bits = self.config.weight_precision.value
        for name, w in weights.items():
            if w_bits >= 32:
                continue
            quantizer = PerChannelQuantizer(bits=w_bits)
            quantizer.calibrate_matrix(w)
            self.weight_quantizers[name] = quantizer
            self.weight_quantized[name] = quantizer.quantize_matrix(w)

    def forward_linear(self, name: str, x: List[List[float]]
                       ) -> List[List[float]]:
        """动态量化前向传播"""
        if name not in self.weight_quantized:
            # 未量化, FP计算
            return x

        w_q = self.weight_quantized[name]
        w_scales = self.weight_quantizers[name].scales

        # 动态量化激活
        a_bits = self.config.activation_precision.value
        x_flat = [v for row in x for v in row]
        max_abs = max(abs(v) for v in x_flat) if x_flat else 0
        x_scale = _compute_scale_symmetric(max_abs, a_bits)

        qmin, qmax = _compute_qrange(a_bits, signed=True)
        x_q = [[_clamp(round(v / x_scale), qmin, qmax) for v in row] for row in x]

        # INT矩阵乘法
        return _int_matmul_per_channel(x_q, w_q, x_scale, w_scales)

    def get_info(self) -> Dict:
        return {
            "type": "DynamicQuantizer",
            "weight_precision": self.config.weight_precision.name,
            "activation_precision": self.config.activation_precision.name,
            "num_quantized_layers": len(self.weight_quantized),
            "is_dynamic": True,
        }


# ============================================================
# 仅权重量化器 (AWQ风格)
# ============================================================

class WeightOnlyQuantizer:
    """仅权重量化器 (AWQ风格)

    只量化权重, 激活保持FP16
    核心思想: 保护重要权重通道 (salient channels)

    AWQ: Activation-aware Weight Quantization
    1. 通过激活分布识别重要权重通道
    2. 对重要通道用更小的量化误差
    3. 用等价缩放保持数学等价性
    """

    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self.scales: Dict[str, List[List[float]]] = {}
        self.quantized: Dict[str, List[List[int]]] = {}
        self.search_space = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 缩放搜索

    def quantize(self, name: str, weight: List[List[float]],
                 activation_stats: Optional[Dict] = None) -> None:
        """量化权重 (感知激活分布)

        Args:
            name: 层名
            weight: 权重矩阵
            activation_stats: 激活统计 (用于识别重要通道)
        """
        rows, cols = len(weight), len(weight[0]) if weight else 0

        # 1. 识别重要通道 (基于激活幅度)
        if activation_stats and "abs_max" in activation_stats:
            act_scales = activation_stats["abs_max"]
            if isinstance(act_scales, list):
                # 按列(输入通道)的最大激活
                importance = act_scales[:cols] if len(act_scales) >= cols else \
                    [1.0] * cols
            else:
                importance = [1.0] * cols
        else:
            # 无激活统计: 用权重幅度作为近似
            importance = [0.0] * cols
            for j in range(cols):
                for i in range(rows):
                    importance[j] = max(importance[j], abs(weight[i][j]))

        # 2. 搜索最优缩放因子
        best_ratio = 0.0
        best_error = float('inf')

        for ratio in self.search_space:
            # 缩放: w' = w * s, x' = x / s (数学等价)
            s = [1.0 + ratio * (imp / max(importance) if max(importance) > 0 else 0)
                 for imp in importance]

            # 缩放权重
            scaled_w = [[weight[i][j] * s[j] for j in range(cols)]
                        for i in range(rows)]

            # 量化-反量化
            q, scales = _quantize_matrix_per_group(scaled_w, self.bits, self.group_size)
            deq = _dequantize_matrix_per_group(q, scales, self.group_size)

            # 反缩放
            restored = [[deq[i][j] / s[j] for j in range(cols)]
                        for i in range(rows)]

            # 计算误差
            error = sum((weight[i][j] - restored[i][j]) ** 2
                        for i in range(rows) for j in range(cols))

            if error < best_error:
                best_error = error
                best_ratio = ratio
                best_s = s

        # 3. 最终量化
        s = best_s
        scaled_w = [[weight[i][j] * s[j] for j in range(cols)]
                    for i in range(rows)]
        q, scales = _quantize_matrix_per_group(scaled_w, self.bits, self.group_size)

        self.quantized[name] = q
        self.scales[name] = scales

    def dequantize(self, name: str) -> List[List[float]]:
        """反量化"""
        q = self.quantized[name]
        scales = self.scales[name]
        return _dequantize_matrix_per_group(q, scales, self.group_size)

    def forward(self, name: str, x: List[List[float]]) -> List[List[float]]:
        """前向: 反量化权重后FP计算
        x: (seq, in_dim), w: (out_dim, in_dim), output: (seq, out_dim)
        """
        w = self.dequantize(name)
        m = len(x)
        k = len(x[0]) if x else 0
        n = len(w) if w else 0  # out_dim = w的行数
        result = [[0.0] * n for _ in range(m)]
        for i in range(m):
            xi = x[i]
            for j in range(n):
                wj = w[j]
                acc = 0.0
                for p in range(k):
                    acc += xi[p] * wj[p]
                result[i][j] = acc
        return result

    def get_info(self) -> Dict:
        total_params = sum(len(q) * len(q[0]) if q else 0 for q in self.quantized.values())
        return {
            "type": "WeightOnlyQuantizer",
            "bits": self.bits,
            "group_size": self.group_size,
            "num_layers": len(self.quantized),
            "total_quantized_params": total_params,
            "compression_ratio": 32.0 / self.bits,
        }


# ============================================================
# GPTQ量化器 (二阶Hessian补偿)
# ============================================================

class GPTQQuantizer:
    """GPTQ量化器 — 基于二阶信息的权重量化

    论文: "GPTQ: Accurate Post-Training Quantization for GPT"
    核心思想: 利用Hessian矩阵信息, 逐列量化权重并补偿误差

    流程:
    1. 计算Hessian: H = X^T @ X (X为校准输入)
    2. 逐列量化权重
    3. 将量化误差传播到后续列 (用Hessian信息加权)
    """

    def __init__(self, bits: int = 4, group_size: int = 128,
                 act_order: bool = True):
        self.bits = bits
        self.group_size = group_size
        self.act_order = act_order  # 按激活幅度排序
        self.quantized: Dict[str, List[List[int]]] = {}
        self.scales: Dict[str, List[List[float]]] = {}

    def quantize(self, name: str, weight: List[List[float]],
                 calibration_input: Optional[List[List[float]]] = None) -> None:
        """GPTQ量化

        Args:
            name: 层名
            weight: 权重矩阵 (in_dim × out_dim)
            calibration_input: 校准输入 (seq × in_dim), 用于计算Hessian
        """
        rows, cols = len(weight), len(weight[0]) if weight else 0

        # 1. 计算Hessian (或用单位矩阵近似)
        if calibration_input:
            # H = X^T @ X (in_dim × in_dim)
            hessian = self._compute_hessian(calibration_input)
        else:
            # 无校准数据: 用对角近似
            hessian = [[1.0 if i == j else 0.0 for j in range(rows)]
                       for i in range(rows)]

        # 2. 激活排序 (可选)
        col_order = list(range(cols))
        if self.act_order and calibration_input:
            # 按Hessian对角线大小排序
            diag = [hessian[i][i] for i in range(min(rows, cols))]
            col_order = sorted(range(len(diag)), key=lambda x: diag[x], reverse=True)

        # 3. 逐列量化 + 误差补偿
        weight_copy = [row[:] for row in weight]
        quantized = [[0] * cols for _ in range(rows)]
        all_scales = []

        for group_start in range(0, cols, self.group_size):
            group_end = min(group_start + self.group_size, cols)
            group_indices = col_order[group_start:group_end]

            # 计算该组的scale
            group_weight = [[weight_copy[i][j] for j in group_indices]
                            for i in range(rows)]
            max_abs = 0.0
            for row in group_weight:
                for v in row:
                    max_abs = max(max_abs, abs(v))
            scale = _compute_scale_symmetric(max_abs, self.bits)
            all_scales.append(scale)

            qmin, qmax = _compute_qrange(self.bits, signed=True)

            # 逐列量化
            for idx, j in enumerate(group_indices):
                # 量化第j列
                col = [weight_copy[i][j] for i in range(rows)]
                q_col = [_clamp(round(v / scale), qmin, qmax) for v in col]
                dq_col = [q * scale for q in q_col]

                # 记录量化结果
                for i in range(rows):
                    quantized[i][j] = q_col[i]

                # 计算误差
                errors = [col[i] - dq_col[i] for i in range(rows)]

                # 误差补偿: 传播到后续列
                # delta_w[i][k] = error[i] * H[i][j] / H[j][j]
                h_jj = hessian[j][j] if j < rows and j < len(hessian[j]) else 1.0
                if abs(h_jj) < 1e-10:
                    h_jj = 1e-10

                for k in group_indices[idx + 1:]:
                    h_jk = hessian[j][k] if j < rows and k < len(hessian[j]) else 0.0
                    for i in range(rows):
                        weight_copy[i][k] -= errors[i] * h_jk / h_jj

        self.quantized[name] = quantized

        # 构建scales (per-group)
        scales_per_row = [all_scales[:] for _ in range(rows)]
        self.scales[name] = scales_per_row

    def _compute_hessian(self, x: List[List[float]]) -> List[List[float]]:
        """计算Hessian: H = X^T @ X"""
        seq_len, dim = len(x), len(x[0]) if x else 0
        hessian = [[0.0] * dim for _ in range(dim)]
        for s in range(seq_len):
            xs = x[s]
            for i in range(dim):
                for j in range(i, dim):
                    val = xs[i] * xs[j]
                    hessian[i][j] += val
                    if i != j:
                        hessian[j][i] += val
        return hessian

    def dequantize(self, name: str) -> List[List[float]]:
        q = self.quantized[name]
        scales = self.scales[name]
        return _dequantize_matrix_per_group(q, scales, self.group_size)

    def get_info(self) -> Dict:
        return {
            "type": "GPTQQuantizer",
            "bits": self.bits,
            "group_size": self.group_size,
            "act_order": self.act_order,
            "num_layers": len(self.quantized),
        }


# ============================================================
# 移动端推理优化器
# ============================================================

class MobileInferenceOptimizer:
    """移动端推理优化器

    针对移动设备(手机/平板)的推理优化:
    - 模型大小优化 (量化+剪枝)
    - 内存优化 (流式推理+缓存复用)
    - 计算优化 (算子融合+稀疏计算)
    - 电池优化 (动态精度+批处理)
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig.preset_mobile()
        self.device_profile = self._get_device_profile()
        self.optimization_log: List[Dict] = []

    def _get_device_profile(self) -> Dict:
        """获取设备性能档案"""
        return {
            "device_type": "mobile",
            "cpu_cores": 8,
            "ram_mb": 4096,
            "available_ram_mb": 2048,
            "max_model_size_mb": 500,
            "battery_level": 0.8,
            "thermal_state": "nominal",
            "compute_capability": "cpu_int8",
        }

    def optimize_model(self, model_info: Dict) -> Dict:
        """优化模型配置

        Args:
            model_info: 模型信息 (参数量/层数/维度等)

        Returns:
            优化后的配置
        """
        optimizations = []

        # 1. 检查模型大小
        model_size_mb = model_info.get("model_size_mb", 100)
        max_size = self.device_profile["max_model_size_mb"]

        if model_size_mb > max_size:
            # 需要压缩
            if model_size_mb > max_size * 4:
                # 极端压缩: INT4
                self.config = QuantizationConfig.preset_int4()
                optimizations.append({"type": "int4_quantization",
                                      "reason": f"model {model_size_mb}MB > {max_size*4}MB"})
            else:
                # INT8
                self.config = QuantizationConfig.preset_int8()
                optimizations.append({"type": "int8_quantization",
                                      "reason": f"model {model_size_mb}MB > {max_size}MB"})

        # 2. 内存优化: 流式推理
        if model_info.get("num_layers", 0) > 6:
            optimizations.append({"type": "streaming_inference",
                                  "reason": "layers > 6, enable layer-by-layer"})

        # 3. 电池优化: 动态精度
        battery = self.device_profile["battery_level"]
        if battery < 0.3:
            optimizations.append({"type": "dynamic_precision",
                                  "reason": f"battery {battery:.0%} < 30%"})
            self.config.weight_precision = QuantizationPrecision.INT4

        # 4. 热优化: 降频
        if self.device_profile["thermal_state"] != "nominal":
            optimizations.append({"type": "thermal_throttle",
                                  "reason": "device overheating"})

        # 5. 算子融合
        optimizations.append({"type": "operator_fusion",
                              "fusions": ["linear+activation", "norm+linear"]})

        # 6. KV缓存优化
        optimizations.append({"type": "kv_cache_compression",
                              "method": "int8_kv_cache"})

        self.optimization_log = optimizations

        return {
            "optimized_config": self.config.to_dict(),
            "optimizations": optimizations,
            "estimated_size_mb": model_size_mb * (4 / 32 if
                self.config.weight_precision == QuantizationPrecision.INT4 else
                4 / 32 if self.config.weight_precision == QuantizationPrecision.INT8 else 1),
            "estimated_speedup": 2.0 if self.config.weight_precision == QuantizationPrecision.INT8 else 3.0,
        }

    def estimate_inference_time(self, seq_len: int, model_info: Dict) -> Dict:
        """估算推理时间"""
        num_layers = model_info.get("num_layers", 6)
        hidden_dim = model_info.get("hidden_dim", 256)
        vocab_size = model_info.get("vocab_size", 1000)

        # 粗略估算 (ms)
        # 每层计算量 ≈ 2 * seq * hidden^2 (注意力) + 2 * seq * hidden * ffn (FFN)
        flops_per_layer = 2 * seq_len * hidden_dim * hidden_dim + \
                          2 * seq_len * hidden_dim * hidden_dim * 4

        # INT8比FP32快~2x, INT4比FP32快~3x
        w_bits = self.config.weight_precision.value
        speedup = 32 / w_bits if w_bits < 32 else 1.0

        # 假设移动CPU 10 GFLOPS (INT8)
        gflops = 10 * speedup
        time_per_layer_ms = flops_per_layer / (gflops * 1e9) * 1000
        total_time_ms = time_per_layer_ms * num_layers

        # 加上embedding和LM head
        embed_time = seq_len * hidden_dim / (gflops * 1e9) * 1000
        lm_head_time = seq_len * hidden_dim * vocab_size / (gflops * 1e9) * 1000

        return {
            "total_time_ms": round(total_time_ms + embed_time + lm_head_time, 2),
            "attention_time_ms": round(time_per_layer_ms * 0.5 * num_layers, 2),
            "ffn_time_ms": round(time_per_layer_ms * 0.5 * num_layers, 2),
            "embedding_time_ms": round(embed_time, 2),
            "lm_head_time_ms": round(lm_head_time, 2),
            "throughput_tokens_per_sec": round(1000 / max(total_time_ms + embed_time + lm_head_time, 0.1) * seq_len, 1),
            "precision": self.config.weight_precision.name,
        }

    def get_optimization_summary(self) -> str:
        """生成优化摘要"""
        lines = ["移动端推理优化摘要", "=" * 50]
        for opt in self.optimization_log:
            lines.append(f"  - {opt['type']}: {opt.get('reason', 'applied')}")
        lines.append(f"  - 最终精度: {self.config.weight_precision.name}")
        lines.append(f"  - 最终方案: {self.config.scheme.value}")
        return "\n".join(lines)


# ============================================================
# 量化基准测试
# ============================================================

class QuantizationBenchmark:
    """量化基准测试

    测量量化前后的:
    - 模型大小
    - 推理速度
    - 输出精度 (与FP32的KL散度)
    - 内存占用
    """

    def __init__(self):
        self.results: List[Dict] = []

    def benchmark_quantizer(self, name: str,
                            quantizer: Any,
                            test_data: List[List[float]],
                            reference_output: Optional[List] = None) -> Dict:
        """基准测试一个量化器

        Args:
            name: 测试名称
            quantizer: 量化器实例
            test_data: 测试数据
            reference_output: 参考输出 (FP32)
        """
        result = {
            "name": name,
            "quantizer_type": type(quantizer).__name__,
        }

        # 1. 原始大小
        original_bytes = sum(len(row) * 4 for row in test_data)

        # 2. 量化
        t0 = time.time()
        if hasattr(quantizer, 'calibrate_matrix'):
            quantizer.calibrate_matrix(test_data)
        elif hasattr(quantizer, 'calibrate'):
            flat = [v for row in test_data for v in row]
            quantizer.calibrate(flat)

        if hasattr(quantizer, 'quantize_matrix'):
            quantized = quantizer.quantize_matrix(test_data)
        elif hasattr(quantizer, 'quantize'):
            flat = [v for row in test_data for v in row]
            quantized = quantizer.quantize(flat)
        else:
            quantized = test_data
        quantize_time = time.time() - t0

        # 3. 量化后大小
        if isinstance(quantized[0], list):
            quantized_bytes = sum(len(row) for row in quantized)
        else:
            bits = getattr(quantizer, 'bits', 8)
            quantized_bytes = len(quantized) * bits // 8

        # 4. 反量化
        t0 = time.time()
        if hasattr(quantizer, 'dequantize_matrix'):
            dequantized = quantizer.dequantize_matrix(quantized)
        elif hasattr(quantizer, 'dequantize'):
            dequantized = quantizer.dequantize(quantized)
        else:
            dequantized = test_data
        dequantize_time = time.time() - t0

        # 5. 精度评估
        if isinstance(quantized[0], list) and isinstance(dequantized[0], list):
            errors = []
            for i in range(min(len(test_data), len(dequantized))):
                for j in range(min(len(test_data[i]), len(dequantized[i]))):
                    errors.append(test_data[i][j] - dequantized[i][j])
        else:
            flat_orig = [v for row in test_data for v in row]
            errors = [o - d for o, d in zip(flat_orig, dequantized)]

        mse = sum(e ** 2 for e in errors) / len(errors) if errors else 0
        mae = sum(abs(e) for e in errors) / len(errors) if errors else 0
        max_err = max(abs(e) for e in errors) if errors else 0

        signal_power = sum(v ** 2 for row in test_data for v in row) / \
            max(sum(len(row) for row in test_data), 1)
        psnr = 10 * math.log10(signal_power / mse) if mse > 0 else float('inf')

        result.update({
            "original_bytes": original_bytes,
            "quantized_bytes": quantized_bytes,
            "compression_ratio": round(original_bytes / max(quantized_bytes, 1), 2),
            "quantize_time_ms": round(quantize_time * 1000, 2),
            "dequantize_time_ms": round(dequantize_time * 1000, 2),
            "mse": mse,
            "mae": mae,
            "max_error": max_err,
            "psnr_db": round(psnr, 2),
            "bits": getattr(quantizer, 'bits', 8),
        })

        self.results.append(result)
        return result

    def compare_configs(self, weight: List[List[float]],
                         configs: List[Tuple[str, QuantizationConfig]]
                         ) -> List[Dict]:
        """比较不同量化配置"""
        comparison = []
        for name, config in configs:
            w_bits = config.weight_precision.value
            if w_bits >= 32:
                continue
            elif w_bits == 4:
                if config.granularity == QuantizationGranularity.PER_GROUP:
                    q = PerGroupQuantizer(bits=4, group_size=config.group_size)
                else:
                    q = PerChannelQuantizer(bits=4)
            elif w_bits == 8:
                q = PerChannelQuantizer(bits=8)
            else:
                q = SymmetricQuantizer(bits=w_bits)

            result = self.benchmark_quantizer(name, q, weight)
            result["config"] = config.to_dict()
            comparison.append(result)

        return comparison

    def print_results(self) -> str:
        """打印结果表格"""
        lines = [f"{'Name':<20} {'Type':<25} {'CR':>6} {'MSE':>10} {'PSNR':>8} {'Bits':>5}"]
        lines.append("-" * 80)
        for r in self.results:
            lines.append(f"{r['name']:<20} {r['quantizer_type']:<25} "
                         f"{r['compression_ratio']:>6.1f} {r['mse']:>10.6f} "
                         f"{r['psnr_db']:>8.1f} {r['bits']:>5}")
        return "\n".join(lines)


# ============================================================
# 量化分析器 (逐层敏感度分析)
# ============================================================

class QuantizationProfiler:
    """量化分析器

    逐层分析量化敏感度, 生成量化策略建议
    """

    def __init__(self):
        self.layer_analysis: Dict[str, Dict] = {}

    def analyze_layer(self, name: str, weight: List[List[float]],
                      activation: Optional[List[List[float]]] = None) -> Dict:
        """分析单层量化敏感度"""
        analysis = {
            "layer_name": name,
            "shape": [len(weight), len(weight[0]) if weight else 0],
            "num_params": len(weight) * len(weight[0]) if weight else 0,
        }

        # 1. 权重分布分析
        flat_w = [v for row in weight for v in row]
        w_min, w_max = min(flat_w), max(flat_w)
        w_mean = sum(flat_w) / len(flat_w)
        w_std = math.sqrt(sum((v - w_mean) ** 2 for v in flat_w) / len(flat_w))
        w_abs_max = max(abs(v) for v in flat_w)

        analysis["weight_stats"] = {
            "min": w_min, "max": w_max, "mean": w_mean,
            "std": w_std, "abs_max": w_abs_max,
            "sparsity": sum(1 for v in flat_w if abs(v) < 1e-6) / len(flat_w),
        }

        # 2. 不同精度的量化误差
        errors = {}
        for bits in [8, 4, 2]:
            quantizer = SymmetricQuantizer(bits=bits)
            quantizer.calibrate_matrix(weight)
            error = quantizer.estimate_error(flat_w)
            errors[f"int{bits}"] = {
                "mse": error["mse"],
                "psnr_db": error["psnr_db"],
                "snr_db": error["snr_db"],
                "compression_ratio": error["compression_ratio"],
            }
        analysis["quantization_errors"] = errors

        # 3. 敏感度评估
        int8_snr = errors["int8"]["snr_db"]
        int4_snr = errors["int4"]["snr_db"]

        if int4_snr > 30:
            sensitivity = LayerSensitivity.INSENSITIVE
            recommendation = "INT4 (低敏感, 可安全量化)"
        elif int8_snr > 30:
            sensitivity = LayerSensitivity.LOW
            recommendation = "INT8 (中等敏感, INT4损失大)"
        elif int8_snr > 20:
            sensitivity = LayerSensitivity.MEDIUM
            recommendation = "INT8 + 精细校准"
        else:
            sensitivity = LayerSensitivity.HIGH
            recommendation = "FP16 (高敏感, 量化损失大)"

        analysis["sensitivity"] = sensitivity.name
        analysis["recommendation"] = recommendation

        # 4. 激活分析 (如果有)
        if activation:
            flat_a = [v for row in activation for v in row]
            a_min, a_max = min(flat_a), max(flat_a)
            a_mean = sum(flat_a) / len(flat_a)
            a_std = math.sqrt(sum((v - a_mean) ** 2 for v in flat_a) / len(flat_a))
            analysis["activation_stats"] = {
                "min": a_min, "max": a_max, "mean": a_mean, "std": a_std,
            }

        self.layer_analysis[name] = analysis
        return analysis

    def analyze_model(self, model_weights: Dict[str, List[List[float]]],
                      activations: Optional[Dict[str, List[List[float]]]] = None
                      ) -> Dict[str, Dict]:
        """分析整个模型"""
        results = {}
        for name, weight in model_weights.items():
            act = activations.get(name) if activations else None
            results[name] = self.analyze_layer(name, weight, act)
        return results

    def get_quantization_strategy(self) -> Dict[str, str]:
        """获取量化策略建议"""
        strategy = {}
        for name, analysis in self.layer_analysis.items():
            strategy[name] = analysis["recommendation"]
        return strategy

    def summary(self) -> str:
        """生成分析摘要"""
        lines = ["量化敏感度分析摘要", "=" * 80]
        lines.append(f"{'Layer':<25} {'Shape':<15} {'INT8 SNR':>10} {'INT4 SNR':>10} {'Recommendation':<30}")
        lines.append("-" * 80)
        for name, a in self.layer_analysis.items():
            shape_str = f"{a['shape'][0]}x{a['shape'][1]}"
            int8_snr = a["quantization_errors"]["int8"]["snr_db"]
            int4_snr = a["quantization_errors"]["int4"]["snr_db"]
            lines.append(f"{name:<25} {shape_str:<15} {int8_snr:>10.1f} {int4_snr:>10.1f} {a['recommendation']:<30}")
        return "\n".join(lines)


# ============================================================
# 量化模型序列化
# ============================================================

class QuantizedModelSerializer:
    """量化模型序列化器

    将量化模型保存为紧凑的二进制格式, 便于部署
    """

    @staticmethod
    def save(quantized_model: Dict, filepath: str) -> Dict:
        """保存量化模型

        Args:
            quantized_model: 量化后的模型数据
            filepath: 保存路径
        """
        total_bytes = 0
        layer_sizes = {}

        # 简单JSON序列化 (实际应用中用二进制)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(quantized_model, f, ensure_ascii=False)

        total_bytes = os.path.getsize(filepath)
        return {
            "filepath": filepath,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / 1024 / 1024, 2),
            "num_layers": len(quantized_model),
        }

    @staticmethod
    def load(filepath: str) -> Dict:
        """加载量化模型"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


# ============================================================
# 测试函数
# ============================================================

def _test_symmetric_quantizer():
    """测试对称量化器"""
    print("  [测试] SymmetricQuantizer...")
    quantizer = SymmetricQuantizer(bits=8)

    # 校准
    data = [random.gauss(0, 1) for _ in range(1000)]
    quantizer.calibrate(data)

    # 量化
    quantized = quantizer.quantize(data)
    dequantized = quantizer.dequantize(quantized)

    # 检查范围
    qmin, qmax = -128, 127
    assert all(qmin <= q <= qmax for q in quantized), "量化值超出范围"

    # 检查误差
    errors = [d - dq for d, dq in zip(data, dequantized)]
    mse = sum(e ** 2 for e in errors) / len(errors)
    assert mse < 0.001, f"MSE过大: {mse}"

    # 测试矩阵
    matrix = [[random.gauss(0, 1) for _ in range(10)] for _ in range(10)]
    quantizer.calibrate_matrix(matrix)
    q_matrix = quantizer.quantize_matrix(matrix)
    dq_matrix = quantizer.dequantize_matrix(q_matrix)
    assert len(dq_matrix) == len(matrix)
    assert len(dq_matrix[0]) == len(matrix[0])

    print("    PASS")


def _test_asymmetric_quantizer():
    """测试非对称量化器"""
    print("  [测试] AsymmetricQuantizer...")
    quantizer = AsymmetricQuantizer(bits=8)

    # 非零均值数据
    data = [random.gauss(5, 2) for _ in range(1000)]
    quantizer.calibrate(data)

    quantized = quantizer.quantize(data)
    dequantized = quantizer.dequantize(quantized)

    qmin, qmax = 0, 255
    assert all(qmin <= q <= qmax for q in quantized), "量化值超出范围"

    errors = [d - dq for d, dq in zip(data, dequantized)]
    mse = sum(e ** 2 for e in errors) / len(errors)
    assert mse < 0.01, f"MSE过大: {mse}"

    print("    PASS")


def _test_per_channel_quantizer():
    """测试逐通道量化器"""
    print("  [测试] PerChannelQuantizer...")
    quantizer = PerChannelQuantizer(bits=8, axis=0)

    # 不同通道有不同值域
    matrix = []
    for i in range(8):
        scale = 10 ** (i - 3)  # 0.001 到 1000
        matrix.append([random.gauss(0, scale) for _ in range(32)])

    quantizer.calibrate_matrix(matrix)
    quantized = quantizer.quantize_matrix(matrix)
    dequantized = quantizer.dequantize_matrix(quantized)

    assert len(quantizer.scales) == 8, "应有8个scale"

    # 检查每通道误差
    for i in range(8):
        errors = [matrix[i][j] - dequantized[i][j] for j in range(32)]
        mse = sum(e ** 2 for e in errors) / 32
        # 相对误差应合理
        signal_power = sum(v ** 2 for v in matrix[i]) / 32
        relative_err = mse / max(signal_power, 1e-12)
        assert relative_err < 0.01, f"通道{i}相对误差过大: {relative_err}"

    print("    PASS")


def _test_per_group_quantizer():
    """测试分组量化器"""
    print("  [测试] PerGroupQuantizer...")
    quantizer = PerGroupQuantizer(bits=4, group_size=16)

    matrix = [[random.gauss(0, 1) for _ in range(64)] for _ in range(8)]
    quantizer.calibrate_matrix(matrix)
    quantized = quantizer.quantize_matrix(matrix)
    dequantized = quantizer.dequantize_matrix(quantized)

    # INT4范围
    assert all(-8 <= q <= 7 for row in quantized for q in row), "INT4值超出范围"

    # 检查scales维度
    expected_groups = 64 // 16  # 4组
    assert len(quantizer.scales) == 8, "每行应有scales"
    assert len(quantizer.scales[0]) == expected_groups, f"应有{expected_groups}组scale"

    print("    PASS")


def _test_calibration():
    """测试校准器"""
    print("  [测试] CalibrationCollector + Calibrators...")

    collector = CalibrationCollector()
    collector.register_layer("test_layer")

    # 收集多批数据
    for _ in range(10):
        data = [random.gauss(0, 1) + random.choice([0, 0, 0, 5]) for _ in range(100)]
        collector.collect("test_layer", data)

    stats = collector.get_stats("test_layer")
    assert stats is not None
    assert stats["count"] == 1000
    assert stats["abs_max"] > 4  # 有离群值

    # MinMax校准
    mm_calibrator = MinMaxCalibrator(collector)
    scale, zp = mm_calibrator.calibrate("test_layer", QuantizationScheme.SYMMETRIC)
    assert scale > 0

    # Percentile校准
    pct_calibrator = PercentileCalibrator(collector, percentile=99.0)
    scale_pct, zp_pct = pct_calibrator.calibrate("test_layer", QuantizationScheme.SYMMETRIC)
    assert scale_pct > 0
    # 百分位应比MinMax小 (截断离群值)
    assert scale_pct <= scale + 1e-6

    # MSE校准
    mse_calibrator = MSECalibrator(collector, num_grid=20)
    scale_mse, zp_mse = mse_calibrator.calibrate("test_layer", QuantizationScheme.SYMMETRIC)
    assert scale_mse > 0

    # ACIQ校准
    aciq_calibrator = ACIQCalibrator(collector)
    scale_aciq, zp_aciq = aciq_calibrator.calibrate("test_layer", QuantizationScheme.SYMMETRIC)
    assert scale_aciq > 0

    print("    PASS")


def _test_quantized_linear():
    """测试量化线性层"""
    print("  [测试] QuantizedLinear...")

    # INT8配置
    config = QuantizationConfig.preset_int8()
    weight = [[random.gauss(0, 0.1) for _ in range(32)] for _ in range(16)]
    bias = [0.01] * 16
    layer = QuantizedLinear(weight, bias, config)

    # 前向传播
    x = [[random.gauss(0, 1) for _ in range(32)] for _ in range(4)]
    output = layer.forward(x)

    assert len(output) == 4
    assert len(output[0]) == 16

    # 校准激活后再推理
    cal_data = [[random.gauss(0, 1) for _ in range(32)] for _ in range(10)]
    layer.calibrate_activation(cal_data)
    output2 = layer.forward(x)
    assert len(output2) == 4

    # 检查内存信息
    info = layer.get_memory_info()
    assert info["compression_ratio"] > 1.0

    # INT4配置
    config4 = QuantizationConfig.preset_int4()
    layer4 = QuantizedLinear(weight, bias, config4)
    output4 = layer4.forward(x)
    assert len(output4) == 4

    # Weight-only配置
    config_wo = QuantizationConfig.preset_mobile()
    layer_wo = QuantizedLinear(weight, bias, config_wo)
    output_wo = layer_wo.forward(x)
    assert len(output_wo) == 4

    print("    PASS")


def _test_quantized_attention():
    """测试量化注意力"""
    print("  [测试] QuantizedAttention...")

    config = QuantizationConfig.preset_int8()
    h = 64
    num_heads = 4

    wq = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(h)]
    wk = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(h)]
    wv = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(h)]
    wo = [[random.gauss(0, 0.02) for _ in range(h)] for _ in range(h)]

    attn = QuantizedAttention(wq, wk, wv, wo, num_heads, h, config)

    x = [[random.gauss(0, 1) for _ in range(h)] for _ in range(8)]
    output = attn.forward(x)

    assert len(output) == 8
    assert len(output[0]) == h

    print("    PASS")


def _test_quantized_transformer():
    """测试量化Transformer"""
    print("  [测试] QuantizedTransformer...")

    config = QuantizationConfig.preset_int8()
    model_config = {
        "vocab_size": 100,
        "hidden_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 64,
        "max_seq_len": 16,
    }

    model = QuantizedTransformer(model_config, config)
    input_ids = [random.randint(0, 99) for _ in range(8)]
    logits = model.forward(input_ids)

    assert len(logits) == 8
    assert len(logits[0]) == 100  # vocab_size

    info = model.get_model_info()
    assert info["compression_ratio"] > 1.0

    print("    PASS")


def _test_mixed_precision():
    """测试混合精度量化"""
    print("  [测试] MixedPrecisionQuantizer...")

    config = QuantizationConfig.preset_mixed()
    quantizer = MixedPrecisionQuantizer(config)

    # 模拟不同层的权重
    model_layers = {
        "layer0_attention": [[random.gauss(0, 0.1) for _ in range(32)] for _ in range(32)],
        "layer0_ffn": [[random.gauss(0, 0.1) for _ in range(32)] for _ in range(64)],
        "layer1_attention": [[random.gauss(0, 0.01) for _ in range(32)] for _ in range(32)],
        "lm_head": [[random.gauss(0, 0.1) for _ in range(32)] for _ in range(100)],
    }

    sensitivities = quantizer.analyze_sensitivity(model_layers)
    assert len(sensitivities) == 4

    summary = quantizer.get_precision_summary()
    assert len(summary) == 4
    assert summary["lm_head"] == "FP16"  # 敏感层

    print("    PASS")


def _test_qat():
    """测试量化感知训练"""
    print("  [测试] QuantizationAwareTraining...")

    qat = QuantizationAwareTraining(QuantizationConfig.preset_int8())

    # 注册权重
    weight = [[random.gauss(0, 0.1) for _ in range(16)] for _ in range(16)]
    qat.register_weight("test_layer", weight)

    # 注册激活
    qat.register_activation("test_layer_act")

    # 观察期 (epoch 0)
    qat.begin_epoch(0)
    activation = [[random.gauss(0, 1) for _ in range(16)] for _ in range(4)]
    fq_act = qat.fake_quantize_activation("test_layer_act", activation)
    assert len(fq_act) == 4  # 观察期不量化

    # 量化期 (epoch 1+)
    qat.begin_epoch(1)
    fq_act2 = qat.fake_quantize_activation("test_layer_act", activation)
    assert len(fq_act2) == 4

    # 伪量化权重
    fq_weight = qat.fake_quantize_weight("test_layer", weight)
    assert len(fq_weight) == 16

    # 获取量化参数
    params = qat.get_quantization_params()
    assert "weights" in params
    assert "activations" in params

    # 转换为量化模型
    quantized = qat.convert_to_quantized({"test_layer": weight})
    assert "test_layer" in quantized

    print("    PASS")


def _test_dynamic_quantizer():
    """测试动态量化器"""
    print("  [测试] DynamicQuantizer...")

    quantizer = DynamicQuantizer(QuantizationConfig.preset_dynamic())

    weights = {
        "layer1": [[random.gauss(0, 0.1) for _ in range(16)] for _ in range(8)],
        "layer2": [[random.gauss(0, 0.1) for _ in range(8)] for _ in range(16)],
    }
    quantizer.quantize_weights(weights)

    x = [[random.gauss(0, 1) for _ in range(16)] for _ in range(4)]
    output = quantizer.forward_linear("layer1", x)
    assert len(output) == 4
    assert len(output[0]) == 8

    info = quantizer.get_info()
    assert info["is_dynamic"] is True

    print("    PASS")


def _test_weight_only_quantizer():
    """测试仅权重量化器"""
    print("  [测试] WeightOnlyQuantizer...")

    quantizer = WeightOnlyQuantizer(bits=4, group_size=16)

    weight = [[random.gauss(0, 0.1) for _ in range(64)] for _ in range(32)]
    activation_stats = {"abs_max": [abs(random.gauss(0, 1)) for _ in range(64)]}

    quantizer.quantize("test_layer", weight, activation_stats)
    dequantized = quantizer.dequantize("test_layer")

    assert len(dequantized) == 32
    assert len(dequantized[0]) == 64

    # 前向
    x = [[random.gauss(0, 1) for _ in range(64)] for _ in range(4)]
    output = quantizer.forward("test_layer", x)
    assert len(output) == 4
    assert len(output[0]) == 32

    info = quantizer.get_info()
    assert info["compression_ratio"] == 8.0  # 32/4

    print("    PASS")


def _test_gptq_quantizer():
    """测试GPTQ量化器"""
    print("  [测试] GPTQQuantizer...")

    quantizer = GPTQQuantizer(bits=4, group_size=16, act_order=True)

    weight = [[random.gauss(0, 0.1) for _ in range(32)] for _ in range(32)]
    cal_input = [[random.gauss(0, 1) for _ in range(32)] for _ in range(8)]

    quantizer.quantize("test_layer", weight, cal_input)
    dequantized = quantizer.dequantize("test_layer")

    assert len(dequantized) == 32
    assert len(dequantized[0]) == 32

    # 检查量化误差
    errors = [weight[i][j] - dequantized[i][j]
              for i in range(32) for j in range(32)]
    mse = sum(e ** 2 for e in errors) / len(errors)
    # GPTQ应该比朴素量化误差小
    assert mse < 0.01

    print("    PASS")


def _test_mobile_optimizer():
    """测试移动端优化器"""
    print("  [测试] MobileInferenceOptimizer...")

    optimizer = MobileInferenceOptimizer()

    model_info = {
        "model_size_mb": 800,
        "num_layers": 12,
        "hidden_dim": 256,
        "vocab_size": 5000,
    }

    result = optimizer.optimize_model(model_info)
    assert "optimizations" in result
    assert len(result["optimizations"]) > 0

    # 推理时间估算
    time_est = optimizer.estimate_inference_time(32, model_info)
    assert time_est["total_time_ms"] > 0
    assert time_est["throughput_tokens_per_sec"] > 0

    summary = optimizer.get_optimization_summary()
    assert "移动端" in summary

    print("    PASS")


def _test_benchmark():
    """测试基准测试"""
    print("  [测试] QuantizationBenchmark...")

    bench = QuantizationBenchmark()
    weight = [[random.gauss(0, 0.1) for _ in range(32)] for _ in range(32)]

    # 测试不同量化器
    configs = [
        ("INT8_PerChannel", QuantizationConfig.preset_int8()),
        ("INT4_PerGroup", QuantizationConfig.preset_int4()),
        ("Mobile_INT4", QuantizationConfig.preset_mobile()),
    ]

    results = bench.compare_configs(weight, configs)
    assert len(results) >= 2

    # INT4压缩比应比INT8高
    int8_cr = [r for r in results if "INT8" in r["name"]][0]["compression_ratio"]
    int4_cr = [r for r in results if "INT4" in r["name"]][0]["compression_ratio"]
    assert int4_cr >= int8_cr

    print("    PASS")


def _test_profiler():
    """测试量化分析器"""
    print("  [测试] QuantizationProfiler...")

    profiler = QuantizationProfiler()

    model_weights = {
        "layer0_attn": [[random.gauss(0, 0.1) for _ in range(32)] for _ in range(32)],
        "layer0_ffn": [[random.gauss(0, 0.1) for _ in range(32)] for _ in range(64)],
        "layer1_attn": [[random.gauss(0, 0.01) for _ in range(32)] for _ in range(32)],
    }

    results = profiler.analyze_model(model_weights)
    assert len(results) == 3

    strategy = profiler.get_quantization_strategy()
    assert len(strategy) == 3

    summary = profiler.summary()
    assert "量化敏感度分析" in summary

    print("    PASS")


def _test_serializer():
    """测试序列化器"""
    print("  [测试] QuantizedModelSerializer...")

    model_data = {
        "layer1": {"quantized": [[1, 2, 3], [4, 5, 6]], "scales": [0.1, 0.2]},
        "layer2": {"quantized": [[7, 8], [9, 10]], "scales": [0.3]},
    }

    filepath = "/tmp/test_quantized_model.json"
    save_info = QuantizedModelSerializer.save(model_data, filepath)
    assert save_info["total_bytes"] > 0

    loaded = QuantizedModelSerializer.load(filepath)
    assert loaded == model_data

    # 清理
    if os.path.exists(filepath):
        os.remove(filepath)

    print("    PASS")


def _test_integration():
    """集成测试: 完整量化流水线"""
    print("  [测试] 集成测试: 完整量化流水线...")

    # 1. 创建模型
    config = QuantizationConfig.preset_mobile()
    model_config = {
        "vocab_size": 50,
        "hidden_dim": 16,
        "num_heads": 2,
        "num_layers": 2,
        "ffn_dim": 32,
        "max_seq_len": 8,
    }
    model = QuantizedTransformer(model_config, config)

    # 2. 推理
    input_ids = [random.randint(0, 49) for _ in range(4)]
    logits = model.forward(input_ids)
    assert len(logits) == 4
    assert len(logits[0]) == 50

    # 3. 分析
    model_info = model.get_model_info()
    assert model_info["compression_ratio"] > 1.0

    # 4. 移动端优化
    optimizer = MobileInferenceOptimizer(config)
    opt_result = optimizer.optimize_model({
        "model_size_mb": model_info["total_quantized_mb"],
        "num_layers": model_config["num_layers"],
        "hidden_dim": model_config["hidden_dim"],
        "vocab_size": model_config["vocab_size"],
    })

    # 5. 推理时间估算
    time_est = optimizer.estimate_inference_time(8, model_config)
    assert time_est["total_time_ms"] >= 0  # 小模型可能估算为0
    assert "throughput_tokens_per_sec" in time_est

    print("    PASS")


# ============================================================
# 主入口
# ============================================================

def main():
    """主测试函数"""
    print()
    print("=" * 70)
    print("  灵元模型 - 量子化推理引擎模块 (Part 23) 自测")
    print("=" * 70)
    print()

    tests = [
        ("SymmetricQuantizer", _test_symmetric_quantizer),
        ("AsymmetricQuantizer", _test_asymmetric_quantizer),
        ("PerChannelQuantizer", _test_per_channel_quantizer),
        ("PerGroupQuantizer", _test_per_group_quantizer),
        ("Calibration", _test_calibration),
        ("QuantizedLinear", _test_quantized_linear),
        ("QuantizedAttention", _test_quantized_attention),
        ("QuantizedTransformer", _test_quantized_transformer),
        ("MixedPrecisionQuantizer", _test_mixed_precision),
        ("QuantizationAwareTraining", _test_qat),
        ("DynamicQuantizer", _test_dynamic_quantizer),
        ("WeightOnlyQuantizer", _test_weight_only_quantizer),
        ("GPTQQuantizer", _test_gptq_quantizer),
        ("MobileInferenceOptimizer", _test_mobile_optimizer),
        ("QuantizationBenchmark", _test_benchmark),
        ("QuantizationProfiler", _test_profiler),
        ("QuantizedModelSerializer", _test_serializer),
        ("Integration", _test_integration),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  [FAIL] {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  自测结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("  所有测试通过!")


if __name__ == "__main__":
    main()
