#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 灵元模型项目 (LingYuan Model Project) — 第 24 模块
 多模态融合引擎 (Multimodal Fusion Engine)
================================================================================

模块概述:
    本模块实现了多模态融合引擎, 支持文本/图像/音频三种模态的编码、对齐与融合。
    核心目标: 让灵元模型不仅理解文本, 还能理解图像和音频。

核心组件:
    1. TextEncoder          — 文本编码器 (基于Transformer)
    2. ImageEncoder         — 图像编码器 (模拟ViT patch embedding)
    3. AudioEncoder         — 音频编码器 (模拟spectrogram embedding)
    4. ModalityProjector    — 模态投影器 (将不同模态映射到统一空间)
    5. CrossModalAttention  — 跨模态注意力 (图文/音文交互)
    6. FusionStrategist     — 融合策略选择器 (早融合/中融合/晚融合)
    7. MultimodalFusionModel — 多模态融合模型 (端到端)
    8. ContrastiveLearner   — 对比学习器 (CLIP风格对齐)
    9. VisualQuestionAnswering — 视觉问答 (VQA)
   10. ImageCaptioner       — 图像描述生成器
   11. SpeechRecognizer     — 语音识别 (模拟ASR)
   12. TextToSpeech         — 文本转语音 (模拟TTS)
   13. MultimodalEmbedder   — 多模态嵌入器 (统一嵌入空间)
   14. ModalityRouter       — 模态路由器 (动态选择激活模态)
   15. MultimodalPipeline   — 多模态流水线 (端到端)

设计原则:
    - 纯 Python 标准库实现, 零外部依赖
    - 图像/音频用模拟数据, 实际部署时替换为真实编码器
    - 所有类可独立实例化和运行
    - 统一的隐藏维度设计

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
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union


# ============================================================
# 枚举定义
# ============================================================

class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class FusionStrategy(Enum):
    """融合策略"""
    EARLY = "early"      # 早融合: 特征级合并
    LATE = "late"        # 晚融合: 决策级合并
    HYBRID = "hybrid"    # 混合融合: 中间层交叉注意力
    ADAPTIVE = "adaptive"  # 自适应: 动态选择


class AttentionType(Enum):
    """注意力类型"""
    SELF = "self"
    CROSS = "cross"
    BIDIRECTIONAL = "bidirectional"


# ============================================================
# 配置
# ============================================================

@dataclass
class MultimodalConfig:
    """多模态配置"""
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    vocab_size: int = 1000
    image_patch_size: int = 16
    image_size: int = 64  # 假设64x64图像
    audio_sample_rate: int = 16000
    audio_duration: float = 1.0  # 秒
    text_max_len: int = 64
    fusion_strategy: FusionStrategy = FusionStrategy.HYBRID
    temperature: float = 0.07  # 对比学习温度

    @property
    def num_image_patches(self) -> int:
        return (self.image_size // self.image_patch_size) ** 2

    @property
    def audio_num_frames(self) -> int:
        return int(self.audio_sample_rate * self.audio_duration / 160)  # 10ms帧


# ============================================================
# 数学辅助函数
# ============================================================

def _softmax(vec: List[float]) -> List[float]:
    if not vec:
        return []
    m = max(vec)
    exps = [math.exp(v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [1.0 / len(vec)] * len(vec)


def _softmax_rows(matrix: List[List[float]]) -> List[List[float]]:
    return [_softmax(row) for row in matrix]


def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    m = len(a)
    k = len(a[0]) if a else 0
    n = len(b[0]) if b else 0
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += a[i][p] * b[p][j]
            result[i][j] = acc
    return result


def _matmul_t(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """A @ B^T"""
    m = len(a)
    k = len(a[0]) if a else 0
    n = len(b) if b else 0
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += a[i][p] * b[j][p]
            result[i][j] = acc
    return result


def _transpose(m: List[List[float]]) -> List[List[float]]:
    if not m:
        return []
    return [list(col) for col in zip(*m)]


def _layer_norm(x: List[List[float]], eps: float = 1e-6) -> List[List[float]]:
    result = []
    for row in x:
        mean = sum(row) / len(row) if row else 0
        var = sum((v - mean) ** 2 for v in row) / len(row) if row else 0
        std = math.sqrt(var + eps)
        result.append([(v - mean) / std for v in row])
    return result


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _gelu_vec(vec: List[float]) -> List[float]:
    return [_gelu(v) for v in vec]


def _init_weight(rows: int, cols: int, scale: float = 0.02) -> List[List[float]]:
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def _init_bias(size: int) -> List[float]:
    return [0.0] * size


def _add_residual(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[i]))]
            for i in range(len(a))]


# ============================================================
# 文本编码器
# ============================================================

class TextEncoder:
    """文本编码器 — 基于简化Transformer

    输入: token_ids
    输出: (seq_len × hidden_dim) 文本特征
    """

    def __init__(self, config: MultimodalConfig):
        self.config = config
        h = config.hidden_dim
        v = config.vocab_size

        # 嵌入
        self.token_emb = _init_weight(v, h)
        self.pos_emb = _init_weight(config.text_max_len, h)

        # Transformer层
        self.layers = []
        for _ in range(config.num_layers):
            self.layers.append({
                "wq": _init_weight(h, h), "wk": _init_weight(h, h),
                "wv": _init_weight(h, h), "wo": _init_weight(h, h),
                "w1": _init_weight(h, h * 4), "w2": _init_weight(h * 4, h),
                "norm1_w": [1.0] * h, "norm2_w": [1.0] * h,
            })

    def forward(self, token_ids: List[int]) -> List[List[float]]:
        seq_len = len(token_ids)
        h = self.config.hidden_dim

        # 嵌入
        x = []
        for i, tid in enumerate(token_ids):
            tid = min(tid, self.config.vocab_size - 1)
            row = [self.token_emb[tid][j] + self.pos_emb[min(i, self.config.text_max_len - 1)][j]
                   for j in range(h)]
            x.append(row)

        # Transformer层
        for layer in self.layers:
            x = self._attention_block(x, layer)
            x = self._ffn_block(x, layer)

        return x

    def _attention_block(self, x: List[List[float]], layer: Dict) -> List[List[float]]:
        h = self.config.hidden_dim
        n = self.config.num_heads
        hd = h // n

        q = _matmul(x, layer["wq"])
        k = _matmul(x, layer["wk"])
        v = _matmul(x, layer["wv"])

        # 多头注意力 (简化: 直接用完整维度)
        scores = _matmul_t(q, k)
        scale = 1.0 / math.sqrt(h)
        scores = [[s * scale for s in row] for row in scores]
        attn = _softmax_rows(scores)
        ctx = _matmul(attn, v)
        out = _matmul(ctx, layer["wo"])

        x = _add_residual(x, out)
        x = _layer_norm(x)
        return x

    def _ffn_block(self, x: List[List[float]], layer: Dict) -> List[List[float]]:
        h1 = [_gelu_vec(row) for row in _matmul(x, layer["w1"])]
        h2 = _matmul(h1, layer["w2"])
        x = _add_residual(x, h2)
        x = _layer_norm(x)
        return x


# ============================================================
# 图像编码器
# ============================================================

class ImageEncoder:
    """图像编码器 — 模拟ViT (Vision Transformer)

    将图像分成patch, 每个patch线性投影到隐藏维度
    输入: image (模拟, List[List[float]] 像素值)
    输出: (num_patches × hidden_dim) 图像特征
    """

    def __init__(self, config: MultimodalConfig):
        self.config = config
        h = config.hidden_dim
        p = config.image_patch_size
        img = config.image_size

        # Patch投影: (p*p) -> hidden_dim
        self.patch_proj = _init_weight(p * p, h)
        self.pos_emb = _init_weight(config.num_image_patches + 1, h)  # +1 for CLS
        self.cls_token = [random.gauss(0, 0.02) for _ in range(h)]

        # Transformer层
        self.layers = []
        for _ in range(config.num_layers):
            self.layers.append({
                "wq": _init_weight(h, h), "wk": _init_weight(h, h),
                "wv": _init_weight(h, h), "wo": _init_weight(h, h),
                "w1": _init_weight(h, h * 4), "w2": _init_weight(h * 4, h),
            })

    def forward(self, image: List[List[float]]) -> List[List[float]]:
        p = self.config.image_patch_size
        img = self.config.image_size
        h = self.config.hidden_dim

        # 1. 提取patches
        patches = []
        for i in range(0, img, p):
            for j in range(0, img, p):
                patch = []
                for pi in range(i, min(i + p, img)):
                    for pj in range(j, min(j + p, img)):
                        if pi < len(image) and pj < len(image[pi]):
                            patch.append(image[pi][pj])
                        else:
                            patch.append(0.0)
                # 补齐到 p*p
                while len(patch) < p * p:
                    patch.append(0.0)
                patches.append(patch)

        # 2. 线性投影
        features = _matmul(patches, self.patch_proj)

        # 3. 添加CLS token
        features.insert(0, self.cls_token[:])

        # 4. 位置编码
        for i in range(len(features)):
            pos = self.pos_emb[min(i, len(self.pos_emb) - 1)]
            features[i] = [features[i][j] + pos[j] for j in range(h)]

        # 5. Transformer层
        for layer in self.layers:
            features = self._attention_block(features, layer)
            features = self._ffn_block(features, layer)

        return features

    def _attention_block(self, x, layer):
        h = self.config.hidden_dim
        q = _matmul(x, layer["wq"])
        k = _matmul(x, layer["wk"])
        v = _matmul(x, layer["wv"])
        scores = _matmul_t(q, k)
        scale = 1.0 / math.sqrt(h)
        scores = [[s * scale for s in row] for row in scores]
        attn = _softmax_rows(scores)
        ctx = _matmul(attn, v)
        out = _matmul(ctx, layer["wo"])
        x = _add_residual(x, out)
        x = _layer_norm(x)
        return x

    def _ffn_block(self, x, layer):
        h1 = [_gelu_vec(row) for row in _matmul(x, layer["w1"])]
        h2 = _matmul(h1, layer["w2"])
        x = _add_residual(x, h2)
        x = _layer_norm(x)
        return x


# ============================================================
# 音频编码器
# ============================================================

class AudioEncoder:
    """音频编码器 — 模拟spectrogram embedding

    将音频波形转为频谱图, 然后用卷积+Transformer编码
    输入: waveform (List[float])
    输出: (num_frames × hidden_dim) 音频特征
    """

    def __init__(self, config: MultimodalConfig):
        self.config = config
        h = config.hidden_dim

        # 频谱投影
        self.spec_proj = _init_weight(80, h)  # 80 mel bins
        self.pos_emb = _init_weight(config.audio_num_frames, h)

        # Transformer层
        self.layers = []
        for _ in range(config.num_layers):
            self.layers.append({
                "wq": _init_weight(h, h), "wk": _init_weight(h, h),
                "wv": _init_weight(h, h), "wo": _init_weight(h, h),
                "w1": _init_weight(h, h * 4), "w2": _init_weight(h * 4, h),
            })

    def forward(self, waveform: List[float]) -> List[List[float]]:
        h = self.config.hidden_dim
        n_frames = self.config.audio_num_frames

        # 1. 模拟STFT -> Mel spectrogram
        spec = []
        frame_size = max(1, len(waveform) // n_frames)
        for f in range(n_frames):
            start = f * frame_size
            end = min(start + frame_size, len(waveform))
            frame = waveform[start:end]
            # 简化: 用帧的能量分布模拟mel bins
            mel = []
            for b in range(80):
                if frame:
                    # 模拟mel filterbank
                    idx = int(b * len(frame) / 80)
                    val = abs(frame[min(idx, len(frame) - 1)]) if frame else 0
                    mel.append(val)
                else:
                    mel.append(0.0)
            spec.append(mel)

        # 2. 线性投影
        features = _matmul(spec, self.spec_proj)

        # 3. 位置编码
        for i in range(len(features)):
            pos = self.pos_emb[min(i, len(self.pos_emb) - 1)]
            features[i] = [features[i][j] + pos[j] for j in range(h)]

        # 4. Transformer层
        for layer in self.layers:
            features = self._attention_block(features, layer)
            features = self._ffn_block(features, layer)

        return features

    def _attention_block(self, x, layer):
        q = _matmul(x, layer["wq"])
        k = _matmul(x, layer["wk"])
        v = _matmul(x, layer["wv"])
        scores = _matmul_t(q, k)
        scale = 1.0 / math.sqrt(self.config.hidden_dim)
        scores = [[s * scale for s in row] for row in scores]
        attn = _softmax_rows(scores)
        ctx = _matmul(attn, v)
        out = _matmul(ctx, layer["wo"])
        x = _add_residual(x, out)
        x = _layer_norm(x)
        return x

    def _ffn_block(self, x, layer):
        h1 = [_gelu_vec(row) for row in _matmul(x, layer["w1"])]
        h2 = _matmul(h1, layer["w2"])
        x = _add_residual(x, h2)
        x = _layer_norm(x)
        return x


# ============================================================
# 模态投影器
# ============================================================

class ModalityProjector:
    """模态投影器 — 将不同模态的特征映射到统一空间"""

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.text_proj = _init_weight(hidden_dim, hidden_dim)
        self.image_proj = _init_weight(hidden_dim, hidden_dim)
        self.audio_proj = _init_weight(hidden_dim, hidden_dim)

    def project_text(self, features: List[List[float]]) -> List[List[float]]:
        return _matmul(features, self.text_proj)

    def project_image(self, features: List[List[float]]) -> List[List[float]]:
        return _matmul(features, self.image_proj)

    def project_audio(self, features: List[List[float]]) -> List[List[float]]:
        return _matmul(features, self.audio_proj)


# ============================================================
# 跨模态注意力
# ============================================================

class CrossModalAttention:
    """跨模态注意力 — 模态间的交互

    Cross-attention: Q来自模态A, K/V来自模态B
    Bidirectional: 双向cross-attention
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.wq = _init_weight(hidden_dim, hidden_dim)
        self.wk = _init_weight(hidden_dim, hidden_dim)
        self.wv = _init_weight(hidden_dim, hidden_dim)
        self.wo = _init_weight(hidden_dim, hidden_dim)

    def forward(self, query_feat: List[List[float]],
                kv_feat: List[List[float]]) -> List[List[float]]:
        """cross-attention: query attends to key-value

        Args:
            query_feat: (seq_q × hidden) 查询模态特征
            kv_feat: (seq_kv × hidden) 键值模态特征

        Returns:
            (seq_q × hidden) 融合后特征
        """
        q = _matmul(query_feat, self.wq)
        k = _matmul(kv_feat, self.wk)
        v = _matmul(kv_feat, self.wv)

        # Attention: Q @ K^T
        scores = _matmul_t(q, k)
        scale = 1.0 / math.sqrt(self.hidden_dim)
        scores = [[s * scale for s in row] for row in scores]
        attn = _softmax_rows(scores)
        ctx = _matmul(attn, v)
        out = _matmul(ctx, self.wo)

        # Residual + Norm
        out = _add_residual(query_feat, out)
        out = _layer_norm(out)
        return out

    def bidirectional(self, feat_a: List[List[float]],
                      feat_b: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """双向cross-attention"""
        a_to_b = self.forward(feat_a, feat_b)
        b_to_a = self.forward(feat_b, feat_a)
        return a_to_b, b_to_a


# ============================================================
# 融合策略选择器
# ============================================================

class FusionStrategist:
    """融合策略选择器

    根据输入模态和任务自动选择最优融合策略:
    - 早融合: 在特征级直接拼接 (适合模态高度相关)
    - 晚融合: 各模态独立处理后在决策级合并 (适合模态独立)
    - 混合融合: 用cross-attention交互 (最灵活)
    - 自适应: 根据模态置信度动态选择
    """

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.cross_attn = CrossModalAttention(hidden_dim)
        self.fusion_gate = _init_weight(hidden_dim * 2, hidden_dim)

    def early_fusion(self, feat_a: List[List[float]],
                     feat_b: List[List[float]]) -> List[List[float]]:
        """早融合: 拼接 + 线性投影"""
        # 对齐长度
        min_len = min(len(feat_a), len(feat_b))
        concat = [feat_a[i] + feat_b[i] for i in range(min_len)]
        return _matmul(concat, self.fusion_gate)

    def late_fusion(self, feat_a: List[List[float]],
                    feat_b: List[List[float]]) -> List[List[float]]:
        """晚融合: 各模态独立均值池化后加权合并"""
        # 均值池化
        pool_a = self._mean_pool(feat_a)
        pool_b = self._mean_pool(feat_b)

        # 加权合并
        weight = 0.5
        merged = [pool_a[i] * weight + pool_b[i] * (1 - weight)
                  for i in range(self.hidden_dim)]
        return [merged]

    def hybrid_fusion(self, feat_a: List[List[float]],
                      feat_b: List[List[float]]) -> List[List[float]]:
        """混合融合: cross-attention + 拼接"""
        # Cross-attention交互
        a_fused, b_fused = self.cross_attn.bidirectional(feat_a, feat_b)

        # 拼接融合结果
        min_len = min(len(a_fused), len(b_fused))
        concat = [a_fused[i] + b_fused[i] for i in range(min_len)]
        return _matmul(concat, self.fusion_gate)

    def adaptive_fusion(self, feat_a: List[List[float]],
                        feat_b: List[List[float]],
                        confidence_a: float = 0.5,
                        confidence_b: float = 0.5) -> List[List[float]]:
        """自适应融合: 根据置信度选择策略"""
        if abs(confidence_a - confidence_b) > 0.3:
            # 某模态明显更可信 -> 晚融合
            return self.late_fusion(feat_a, feat_b)
        elif confidence_a > 0.7 and confidence_b > 0.7:
            # 两模态都高可信 -> 早融合
            return self.early_fusion(feat_a, feat_b)
        else:
            # 默认 -> 混合融合
            return self.hybrid_fusion(feat_a, feat_b)

    @staticmethod
    def _mean_pool(features: List[List[float]]) -> List[float]:
        if not features:
            return [0.0]
        h = len(features[0])
        pooled = [0.0] * h
        for row in features:
            for j in range(h):
                pooled[j] += row[j]
        return [p / len(features) for p in pooled]


# ============================================================
# 对比学习器 (CLIP风格)
# ============================================================

class ContrastiveLearner:
    """对比学习器 — CLIP风格模态对齐

    通过对比学习将不同模态映射到同一嵌入空间
    目标: 匹配的图文对相似度最大化, 不匹配的最小化
    """

    def __init__(self, hidden_dim: int, temperature: float = 0.07):
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def compute_similarity(self, feat_a: List[List[float]],
                           feat_b: List[List[float]]) -> List[List[float]]:
        """计算模态间的相似度矩阵

        Returns:
            (batch_a × batch_b) 相似度矩阵
        """
        # 均值池化
        pool_a = [self._mean_pool(f) for f in feat_a] if feat_a and isinstance(feat_a[0], list) and isinstance(feat_a[0][0], list) else feat_a
        pool_b = [self._mean_pool(f) for f in feat_b] if feat_b and isinstance(feat_b[0], list) and isinstance(feat_b[0][0], list) else feat_b

        # L2归一化
        norm_a = [self._l2_normalize(v) for v in pool_a]
        norm_b = [self._l2_normalize(v) for v in pool_b]

        # 余弦相似度
        sim = _matmul_t(norm_a, norm_b)
        # 缩放
        sim = [[s / self.temperature for s in row] for row in sim]
        return sim

    def contrastive_loss(self, feat_a: List[List[float]],
                         feat_b: List[List[float]]) -> Dict[str, float]:
        """计算对比损失

        对于N个匹配对, 损失 = -log(exp(sim[i][i]/T) / sum_j exp(sim[i][j]/T))
        """
        sim = self.compute_similarity(feat_a, feat_b)
        n = len(sim)

        if n == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        # 行方向损失 (A->B)
        loss_a = 0.0
        correct_a = 0
        for i in range(n):
            probs = _softmax(sim[i])
            loss_a -= math.log(max(probs[i], 1e-10))
            if max(range(n), key=lambda j: sim[i][j]) == i:
                correct_a += 1

        # 列方向损失 (B->A)
        loss_b = 0.0
        correct_b = 0
        for j in range(n):
            col = [sim[i][j] for i in range(n)]
            probs = _softmax(col)
            loss_b -= math.log(max(probs[j], 1e-10))
            if max(range(n), key=lambda i: sim[i][j]) == j:
                correct_b += 1

        total_loss = (loss_a + loss_b) / (2 * n)
        accuracy = (correct_a + correct_b) / (2 * n)

        return {"loss": total_loss, "accuracy": accuracy}

    @staticmethod
    def _mean_pool(features: List[List[float]]) -> List[float]:
        if not features:
            return [0.0]
        h = len(features[0])
        pooled = [0.0] * h
        for row in features:
            for j in range(h):
                pooled[j] += row[j]
        return [p / len(features) for p in pooled]

    @staticmethod
    def _l2_normalize(vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm < 1e-12:
            return vec
        return [v / norm for v in vec]


# ============================================================
# 多模态融合模型
# ============================================================

class MultimodalFusionModel:
    """多模态融合模型 — 端到端

    整合文本/图像/音频编码器, 通过融合策略统一处理
    """

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()
        self.text_encoder = TextEncoder(self.config)
        self.image_encoder = ImageEncoder(self.config)
        self.audio_encoder = AudioEncoder(self.config)
        self.projector = ModalityProjector(self.config.hidden_dim)
        self.fusion = FusionStrategist(self.config.hidden_dim)
        self.contrastive = ContrastiveLearner(self.config.hidden_dim,
                                               self.config.temperature)

    def encode_text(self, token_ids: List[int]) -> List[List[float]]:
        features = self.text_encoder.forward(token_ids)
        return self.projector.project_text(features)

    def encode_image(self, image: List[List[float]]) -> List[List[float]]:
        features = self.image_encoder.forward(image)
        return self.projector.project_image(features)

    def encode_audio(self, waveform: List[float]) -> List[List[float]]:
        features = self.audio_encoder.forward(waveform)
        return self.projector.project_audio(features)

    def fuse_text_image(self, token_ids: List[int],
                        image: List[List[float]],
                        strategy: FusionStrategy = FusionStrategy.HYBRID
                        ) -> List[List[float]]:
        """融合文本和图像"""
        text_feat = self.encode_text(token_ids)
        image_feat = self.encode_image(image)

        if strategy == FusionStrategy.EARLY:
            return self.fusion.early_fusion(text_feat, image_feat)
        elif strategy == FusionStrategy.LATE:
            return self.fusion.late_fusion(text_feat, image_feat)
        elif strategy == FusionStrategy.HYBRID:
            return self.fusion.hybrid_fusion(text_feat, image_feat)
        else:
            return self.fusion.adaptive_fusion(text_feat, image_feat)

    def fuse_text_audio(self, token_ids: List[int],
                        waveform: List[float],
                        strategy: FusionStrategy = FusionStrategy.HYBRID
                        ) -> List[List[float]]:
        """融合文本和音频"""
        text_feat = self.encode_text(token_ids)
        audio_feat = self.encode_audio(waveform)

        if strategy == FusionStrategy.EARLY:
            return self.fusion.early_fusion(text_feat, audio_feat)
        elif strategy == FusionStrategy.LATE:
            return self.fusion.late_fusion(text_feat, audio_feat)
        elif strategy == FusionStrategy.HYBRID:
            return self.fusion.hybrid_fusion(text_feat, audio_feat)
        else:
            return self.fusion.adaptive_fusion(text_feat, audio_feat)

    def fuse_all(self, token_ids: List[int],
                 image: List[List[float]],
                 waveform: List[float]) -> List[List[float]]:
        """融合所有模态"""
        text_feat = self.encode_text(token_ids)
        image_feat = self.encode_image(image)
        audio_feat = self.encode_audio(waveform)

        # 先融合文本和图像
        text_image = self.fusion.hybrid_fusion(text_feat, image_feat)
        # 再融合结果和音频
        result = self.fusion.hybrid_fusion(text_image, audio_feat)
        return result

    def get_model_info(self) -> Dict:
        return {
            "model_type": "MultimodalFusionModel",
            "hidden_dim": self.config.hidden_dim,
            "num_heads": self.config.num_heads,
            "num_layers": self.config.num_layers,
            "modalities": ["text", "image", "audio"],
            "fusion_strategy": self.config.fusion_strategy.value,
            "num_image_patches": self.config.num_image_patches,
        }


# ============================================================
# 视觉问答 (VQA)
# ============================================================

class VisualQuestionAnswering:
    """视觉问答 — 给定图像和问题, 生成答案

    流程:
    1. 编码图像
    2. 编码问题文本
    3. 融合图文特征
    4. 生成答案
    """

    def __init__(self, model: Optional[MultimodalFusionModel] = None):
        self.model = model or MultimodalFusionModel()
        self.answer_vocab = ["是", "否", "红色", "蓝色", "绿色", "大", "小",
                             "人", "动物", "植物", "建筑", "天空", "地面",
                             "左边", "右边", "上方", "下方", "一个", "多个", "没有"]

    def answer(self, image: List[List[float]], question_ids: List[int]) -> Dict:
        """回答视觉问题

        Args:
            image: 图像像素矩阵
            question_ids: 问题的token ids

        Returns:
            答案信息
        """
        # 融合图文
        fused = self.model.fuse_text_image(question_ids, image,
                                            FusionStrategy.HYBRID)

        # 简化: 用融合特征的均值选择答案
        pooled = self._mean_pool(fused)

        # 用特征向量计算答案分数
        scores = []
        for i in range(len(self.answer_vocab)):
            score = sum(pooled[j] * (1 + i * 0.01) for j in range(min(len(pooled), 10))
                        if j < len(pooled))
            scores.append(score)

        # Softmax
        probs = _softmax(scores)
        best_idx = max(range(len(probs)), key=lambda i: probs[i])

        return {
            "answer": self.answer_vocab[best_idx],
            "confidence": probs[best_idx],
            "top_3": [(self.answer_vocab[i], round(probs[i], 4))
                      for i in sorted(range(len(probs)), key=lambda i: -probs[i])[:3]],
        }

    @staticmethod
    def _mean_pool(features: List[List[float]]) -> List[float]:
        if not features:
            return [0.0]
        h = len(features[0])
        pooled = [0.0] * h
        for row in features:
            for j in range(h):
                pooled[j] += row[j]
        return [p / len(features) for p in pooled]


# ============================================================
# 图像描述生成器
# ============================================================

class ImageCaptioner:
    """图像描述生成器 — 给定图像生成文字描述"""

    def __init__(self, model: Optional[MultimodalFusionModel] = None):
        self.model = model or MultimodalFusionModel()
        self.vocab = ["<sos>", "<eos>", "图", "片", "中", "有", "一", "个", "多",
                       "大", "小", "红", "蓝", "绿", "色", "的", "人", "物", "动",
                       "植", "建", "筑", "天", "空", "地", "面", "左", "右", "上",
                       "下", "是", "在", "和", "被", "这", "些"]

    def caption(self, image: List[List[float]], max_len: int = 12) -> Dict:
        """生成图像描述

        Args:
            image: 图像像素矩阵
            max_len: 最大生成长度

        Returns:
            描述信息
        """
        # 编码图像
        image_feat = self.model.encode_image(image)

        # 自回归生成 (简化: 用图像特征直接预测)
        tokens = [0]  # <sos>
        for _ in range(max_len - 1):
            # 用最后一个token + 图像特征预测下一个token
            text_feat = self.model.encode_text(tokens[-3:] if len(tokens) >= 3 else tokens)
            # 融合
            fused = self.model.fusion.late_fusion(text_feat, image_feat)
            pooled = self._mean_pool(fused)

            # 计算词表分数
            scores = []
            for i in range(len(self.vocab)):
                s = sum(pooled[j] * (1 + i * 0.001) for j in range(min(len(pooled), 20))
                        if j < len(pooled))
                scores.append(s)

            probs = _softmax(scores)
            next_token = max(range(len(probs)), key=lambda i: probs[i])

            if next_token == 1:  # <eos>
                break
            tokens.append(next_token)

        caption_text = "".join(self.vocab[t] for t in tokens if t > 1)
        return {
            "caption": caption_text,
            "token_ids": tokens,
            "length": len(tokens),
        }

    @staticmethod
    def _mean_pool(features: List[List[float]]) -> List[float]:
        if not features:
            return [0.0]
        h = len(features[0])
        pooled = [0.0] * h
        for row in features:
            for j in range(h):
                pooled[j] += row[j]
        return [p / len(features) for p in pooled]


# ============================================================
# 语音识别 (ASR)
# ============================================================

class SpeechRecognizer:
    """语音识别 — 将音频转为文本

    流程:
    1. 编码音频
    2. 解码为token序列
    """

    def __init__(self, model: Optional[MultimodalFusionModel] = None):
        self.model = model or MultimodalFusionModel()
        self.vocab = ["", "你", "好", "世", "界", "我", "是", "灵", "元",
                       "模", "型", "说", "听", "看", "想", "做", "的", "了", "在"]

    def recognize(self, waveform: List[float]) -> Dict:
        """语音识别

        Args:
            waveform: 音频波形

        Returns:
            识别结果
        """
        # 编码音频
        audio_feat = self.model.encode_audio(waveform)

        # 简化: 直接从特征生成token
        tokens = []
        for feat in audio_feat[:10]:  # 最多10个token
            scores = []
            for i in range(len(self.vocab)):
                s = sum(feat[j] * (1 + i * 0.01) for j in range(min(len(feat), 20))
                        if j < len(feat))
                scores.append(s)
            probs = _softmax(scores)
            token = max(range(len(probs)), key=lambda i: probs[i])
            if token > 0:
                tokens.append(token)

        text = "".join(self.vocab[t] for t in tokens)
        return {
            "text": text,
            "token_ids": tokens,
            "confidence": 0.75 + random.random() * 0.2,
        }


# ============================================================
# 文本转语音 (TTS)
# ============================================================

class TextToSpeech:
    """文本转语音 — 将文本转为音频波形"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.freq_map = {
            "你": 440, "好": 523, "世": 392, "界": 466,
            "我": 440, "是": 392, "灵": 523, "元": 466,
        }

    def synthesize(self, text: str) -> Dict:
        """文本转语音

        Args:
            text: 输入文本

        Returns:
            合成结果
        """
        waveform = []
        for char in text:
            freq = self.freq_map.get(char, 400 + ord(char) % 200)
            duration = int(self.sample_rate * 0.15)  # 150ms per char
            for t in range(duration):
                sample = math.sin(2 * math.pi * freq * t / self.sample_rate) * 0.5
                # 添加谐波
                sample += math.sin(4 * math.pi * freq * t / self.sample_rate) * 0.15
                # 包络
                envelope = min(1.0, t / 1000) * min(1.0, (duration - t) / 1000)
                waveform.append(sample * envelope)

        return {
            "waveform": waveform,
            "duration_s": len(waveform) / self.sample_rate,
            "sample_rate": self.sample_rate,
            "text": text,
        }


# ============================================================
# 多模态嵌入器
# ============================================================

class MultimodalEmbedder:
    """多模态嵌入器 — 将任意模态映射到统一嵌入空间

    用途: 多模态检索 (以文搜图, 以图搜文, 以音搜文等)
    """

    def __init__(self, model: Optional[MultimodalFusionModel] = None,
                 embed_dim: int = 256):
        self.model = model or MultimodalFusionModel()
        self.embed_dim = embed_dim
        self.embedding_db: Dict[str, Dict] = {}  # id -> {embedding, modality, content}

    def embed_text(self, token_ids: List[int]) -> List[float]:
        """嵌入文本"""
        features = self.model.encode_text(token_ids)
        return self._mean_pool(features)

    def embed_image(self, image: List[List[float]]) -> List[float]:
        """嵌入图像"""
        features = self.model.encode_image(image)
        return self._mean_pool(features)

    def embed_audio(self, waveform: List[float]) -> List[float]:
        """嵌入音频"""
        features = self.model.encode_audio(waveform)
        return self._mean_pool(features)

    def add_to_db(self, item_id: str, modality: str, content: Any,
                  embedding: List[float]) -> None:
        """添加到嵌入数据库"""
        self.embedding_db[item_id] = {
            "modality": modality,
            "content": content,
            "embedding": embedding,
        }

    def search(self, query_embedding: List[float], top_k: int = 5,
               modality_filter: Optional[str] = None) -> List[Dict]:
        """搜索最相似的项"""
        results = []
        for item_id, item in self.embedding_db.items():
            if modality_filter and item["modality"] != modality_filter:
                continue
            sim = self._cosine_sim(query_embedding, item["embedding"])
            results.append({
                "id": item_id,
                "modality": item["modality"],
                "similarity": round(sim, 4),
            })

        results.sort(key=lambda x: -x["similarity"])
        return results[:top_k]

    @staticmethod
    def _mean_pool(features: List[List[float]]) -> List[float]:
        if not features:
            return [0.0]
        h = len(features[0])
        pooled = [0.0] * h
        for row in features:
            for j in range(h):
                pooled[j] += row[j]
        return [p / len(features) for p in pooled]

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ============================================================
# 模态路由器
# ============================================================

class ModalityRouter:
    """模态路由器 — 动态选择激活哪些模态

    根据输入类型和任务需求, 选择最优的模态组合
    """

    def __init__(self):
        self.modality_weights: Dict[str, float] = {
            "text": 1.0,
            "image": 0.8,
            "audio": 0.6,
        }
        self.task_modality_map = {
            "text_generation": ["text"],
            "image_captioning": ["image", "text"],
            "vqa": ["image", "text"],
            "asr": ["audio", "text"],
            "tts": ["text", "audio"],
            "multimodal_search": ["text", "image", "audio"],
        }

    def route(self, task: str, available_modalities: List[str]) -> Dict:
        """路由到最优模态组合

        Args:
            task: 任务类型
            available_modalities: 可用模态列表

        Returns:
            路由结果
        """
        required = self.task_modality_map.get(task, ["text"])
        active = [m for m in required if m in available_modalities]

        if not active:
            active = available_modalities[:1] if available_modalities else ["text"]

        return {
            "task": task,
            "required_modalities": required,
            "available_modalities": available_modalities,
            "active_modalities": active,
            "modality_weights": {m: self.modality_weights.get(m, 0.5) for m in active},
        }

    def update_weight(self, modality: str, weight: float) -> None:
        """更新模态权重"""
        self.modality_weights[modality] = weight


# ============================================================
# 多模态流水线
# ============================================================

class MultimodalPipeline:
    """多模态流水线 — 端到端处理

    整合所有多模态组件, 提供统一接口
    """

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()
        self.model = MultimodalFusionModel(self.config)
        self.vqa = VisualQuestionAnswering(self.model)
        self.captioner = ImageCaptioner(self.model)
        self.asr = SpeechRecognizer(self.model)
        self.tts = TextToSpeech(self.config.audio_sample_rate)
        self.embedder = MultimodalEmbedder(self.model)
        self.router = ModalityRouter()

    def process(self, inputs: Dict[str, Any], task: str = "auto") -> Dict:
        """处理多模态输入

        Args:
            inputs: {"text": ..., "image": ..., "audio": ...}
            task: 任务类型 ("auto"自动推断)

        Returns:
            处理结果
        """
        # 确定可用模态
        available = []
        if "text" in inputs:
            available.append("text")
        if "image" in inputs:
            available.append("image")
        if "audio" in inputs:
            available.append("audio")

        # 自动推断任务
        if task == "auto":
            if "image" in inputs and "text" in inputs:
                if isinstance(inputs.get("text"), list):
                    task = "vqa"
                else:
                    task = "image_captioning"
            elif "audio" in inputs:
                task = "asr"
            elif "text" in inputs and "image" not in inputs:
                task = "text_generation"
            else:
                task = "multimodal_search"

        # 路由
        routing = self.router.route(task, available)

        # 执行任务
        result = {"task": task, "routing": routing}

        if task == "vqa" and "image" in inputs and "text" in inputs:
            question_ids = inputs["text"] if isinstance(inputs["text"], list) else [1, 2, 3]
            result["vqa_result"] = self.vqa.answer(inputs["image"], question_ids)

        elif task == "image_captioning" and "image" in inputs:
            result["caption_result"] = self.captioner.caption(inputs["image"])

        elif task == "asr" and "audio" in inputs:
            result["asr_result"] = self.asr.recognize(inputs["audio"])

        elif task == "tts" and "text" in inputs:
            text = inputs["text"] if isinstance(inputs["text"], str) else "你好"
            result["tts_result"] = self.tts.synthesize(text)

        elif task == "multimodal_search":
            # 嵌入所有模态
            embeddings = {}
            if "text" in inputs:
                text_ids = inputs["text"] if isinstance(inputs["text"], list) else [1, 2, 3]
                embeddings["text"] = self.embedder.embed_text(text_ids)
            if "image" in inputs:
                embeddings["image"] = self.embedder.embed_image(inputs["image"])
            if "audio" in inputs:
                embeddings["audio"] = self.embedder.embed_audio(inputs["audio"])
            result["embeddings"] = {
                k: v[:10] for k, v in embeddings.items()  # 截断显示
            }

        return result

    def get_pipeline_info(self) -> Dict:
        return {
            "pipeline_type": "MultimodalPipeline",
            "config": {
                "hidden_dim": self.config.hidden_dim,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
            },
            "components": [
                "TextEncoder", "ImageEncoder", "AudioEncoder",
                "ModalityProjector", "CrossModalAttention", "FusionStrategist",
                "ContrastiveLearner", "VQA", "ImageCaptioner",
                "ASR", "TTS", "MultimodalEmbedder", "ModalityRouter",
            ],
            "supported_tasks": list(self.router.task_modality_map.keys()),
        }


# ============================================================
# 测试函数
# ============================================================

def _test_text_encoder():
    print("  [测试] TextEncoder...")
    config = MultimodalConfig(hidden_dim=32, num_heads=2, num_layers=1, vocab_size=50, text_max_len=16)
    encoder = TextEncoder(config)
    token_ids = [random.randint(0, 49) for _ in range(8)]
    features = encoder.forward(token_ids)
    assert len(features) == 8
    assert len(features[0]) == 32
    print("    PASS")


def _test_image_encoder():
    print("  [测试] ImageEncoder...")
    config = MultimodalConfig(hidden_dim=32, num_heads=2, num_layers=1,
                              image_patch_size=8, image_size=32)
    encoder = ImageEncoder(config)
    image = [[random.random() for _ in range(32)] for _ in range(32)]
    features = encoder.forward(image)
    expected_patches = (32 // 8) ** 2 + 1  # +CLS
    assert len(features) == expected_patches
    assert len(features[0]) == 32
    print("    PASS")


def _test_audio_encoder():
    print("  [测试] AudioEncoder...")
    config = MultimodalConfig(hidden_dim=32, num_heads=2, num_layers=1)
    encoder = AudioEncoder(config)
    waveform = [random.gauss(0, 0.5) for _ in range(1600)]
    features = encoder.forward(waveform)
    assert len(features) > 0
    assert len(features[0]) == 32
    print("    PASS")


def _test_modality_projector():
    print("  [测试] ModalityProjector...")
    proj = ModalityProjector(32)
    feat = [[random.gauss(0, 1) for _ in range(32)] for _ in range(4)]
    out = proj.project_text(feat)
    assert len(out) == 4 and len(out[0]) == 32
    out = proj.project_image(feat)
    assert len(out) == 4
    out = proj.project_audio(feat)
    assert len(out) == 4
    print("    PASS")


def _test_cross_modal_attention():
    print("  [测试] CrossModalAttention...")
    attn = CrossModalAttention(32)
    feat_a = [[random.gauss(0, 1) for _ in range(32)] for _ in range(4)]
    feat_b = [[random.gauss(0, 1) for _ in range(32)] for _ in range(6)]
    out = attn.forward(feat_a, feat_b)
    assert len(out) == 4 and len(out[0]) == 32
    a_out, b_out = attn.bidirectional(feat_a, feat_b)
    assert len(a_out) == 4 and len(b_out) == 6
    print("    PASS")


def _test_fusion_strategist():
    print("  [测试] FusionStrategist...")
    fusion = FusionStrategist(32)
    feat_a = [[random.gauss(0, 1) for _ in range(32)] for _ in range(4)]
    feat_b = [[random.gauss(0, 1) for _ in range(32)] for _ in range(4)]

    early = fusion.early_fusion(feat_a, feat_b)
    assert len(early) == 4 and len(early[0]) == 32

    late = fusion.late_fusion(feat_a, feat_b)
    assert len(late) == 1 and len(late[0]) == 32

    hybrid = fusion.hybrid_fusion(feat_a, feat_b)
    assert len(hybrid) == 4

    adaptive = fusion.adaptive_fusion(feat_a, feat_b, 0.8, 0.8)
    assert len(adaptive) > 0
    print("    PASS")


def _test_contrastive_learner():
    print("  [测试] ContrastiveLearner...")
    learner = ContrastiveLearner(32, temperature=0.1)
    feat_a = [[random.gauss(0, 1) for _ in range(32)] for _ in range(4)]
    feat_b = [[random.gauss(0, 1) for _ in range(32)] for _ in range(4)]
    sim = learner.compute_similarity(feat_a, feat_b)
    assert len(sim) == 4 and len(sim[0]) == 4
    loss_info = learner.contrastive_loss(feat_a, feat_b)
    assert "loss" in loss_info and "accuracy" in loss_info
    assert loss_info["loss"] > 0
    print("    PASS")


def _test_multimodal_model():
    print("  [测试] MultimodalFusionModel...")
    config = MultimodalConfig(hidden_dim=32, num_heads=2, num_layers=1,
                              vocab_size=50, image_patch_size=8, image_size=32)
    model = MultimodalFusionModel(config)

    token_ids = [random.randint(0, 49) for _ in range(8)]
    image = [[random.random() for _ in range(32)] for _ in range(32)]
    waveform = [random.gauss(0, 0.5) for _ in range(800)]

    text_feat = model.encode_text(token_ids)
    assert len(text_feat) == 8

    image_feat = model.encode_image(image)
    assert len(image_feat) > 0

    audio_feat = model.encode_audio(waveform)
    assert len(audio_feat) > 0

    fused = model.fuse_text_image(token_ids, image)
    assert len(fused) > 0

    fused_all = model.fuse_all(token_ids, image, waveform)
    assert len(fused_all) > 0

    info = model.get_model_info()
    assert info["hidden_dim"] == 32
    print("    PASS")


def _test_vqa():
    print("  [测试] VisualQuestionAnswering...")
    config = MultimodalConfig(hidden_dim=32, num_heads=2, num_layers=1,
                              vocab_size=50, image_patch_size=8, image_size=32)
    model = MultimodalFusionModel(config)
    vqa = VisualQuestionAnswering(model)

    image = [[random.random() for _ in range(32)] for _ in range(32)]
    question = [1, 2, 3, 4, 5]
    result = vqa.answer(image, question)
    assert "answer" in result
    assert "confidence" in result
    assert "top_3" in result
    assert len(result["top_3"]) == 3
    print("    PASS")


def _test_captioner():
    print("  [测试] ImageCaptioner...")
    config = MultimodalConfig(hidden_dim=32, num_heads=2, num_layers=1,
                              vocab_size=50, image_patch_size=8, image_size=32)
    model = MultimodalFusionModel(config)
    captioner = ImageCaptioner(model)

    image = [[random.random() for _ in range(32)] for _ in range(32)]
    result = captioner.caption(image, max_len=6)
    assert "caption" in result
    assert "token_ids" in result
    assert len(result["token_ids"]) <= 6
    print("    PASS")


def _test_asr():
    print("  [测试] SpeechRecognizer...")
    config = MultimodalConfig(hidden_dim=32, num_heads=2, num_layers=1)
    model = MultimodalFusionModel(config)
    asr = SpeechRecognizer(model)

    waveform = [random.gauss(0, 0.5) for _ in range(1600)]
    result = asr.recognize(waveform)
    assert "text" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    print("    PASS")


def _test_tts():
    print("  [测试] TextToSpeech...")
    tts = TextToSpeech(sample_rate=8000)
    result = tts.synthesize("你好世界")
    assert "waveform" in result
    assert len(result["waveform"]) > 0
    assert result["duration_s"] > 0
    print("    PASS")


def _test_embedder():
    print("  [测试] MultimodalEmbedder...")
    config = MultimodalConfig(hidden_dim=32, num_heads=2, num_layers=1, vocab_size=50)
    model = MultimodalFusionModel(config)
    embedder = MultimodalEmbedder(model, embed_dim=32)

    # 添加项目
    embedder.add_to_db("text1", "text", "hello", embedder.embed_text([1, 2, 3]))
    embedder.add_to_db("image1", "image", "photo", embedder.embed_image([[0.5]*16]*16))

    # 搜索
    query = embedder.embed_text([1, 2, 3])
    results = embedder.search(query, top_k=2)
    assert len(results) <= 2
    assert results[0]["id"] == "text1"  # 完全匹配

    # 模态过滤
    image_results = embedder.search(query, top_k=5, modality_filter="image")
    assert all(r["modality"] == "image" for r in image_results)
    print("    PASS")


def _test_router():
    print("  [测试] ModalityRouter...")
    router = ModalityRouter()
    result = router.route("vqa", ["text", "image", "audio"])
    assert "image" in result["active_modalities"]
    assert "text" in result["active_modalities"]

    result = router.route("asr", ["text", "image"])
    assert "text" in result["active_modalities"]
    print("    PASS")


def _test_pipeline():
    print("  [测试] MultimodalPipeline...")
    config = MultimodalConfig(hidden_dim=32, num_heads=2, num_layers=1,
                              vocab_size=50, image_patch_size=8, image_size=32)
    pipeline = MultimodalPipeline(config)

    # VQA
    result = pipeline.process(
        {"image": [[random.random() for _ in range(32)] for _ in range(32)],
         "text": [1, 2, 3]},
        task="auto"
    )
    assert "task" in result

    # ASR
    result = pipeline.process(
        {"audio": [random.gauss(0, 0.5) for _ in range(800)]},
        task="auto"
    )
    assert "task" in result

    # Pipeline info
    info = pipeline.get_pipeline_info()
    assert "components" in info
    print("    PASS")


def _test_integration():
    print("  [测试] 集成测试: 完整多模态流水线...")
    config = MultimodalConfig(hidden_dim=64, num_heads=4, num_layers=2,
                              vocab_size=100, image_patch_size=16, image_size=64)
    pipeline = MultimodalPipeline(config)

    # 1. 编码各模态
    text_feat = pipeline.model.encode_text([1, 2, 3, 4, 5])
    image_feat = pipeline.model.encode_image([[random.random() for _ in range(64)] for _ in range(64)])
    audio_feat = pipeline.model.encode_audio([random.gauss(0, 0.5) for _ in range(1600)])

    assert len(text_feat[0]) == 64
    assert len(image_feat[0]) == 64
    assert len(audio_feat[0]) == 64

    # 2. 对比学习
    loss_info = pipeline.model.contrastive.contrastive_loss(text_feat, image_feat[:5])
    assert loss_info["loss"] > 0

    # 3. VQA
    vqa_result = pipeline.vqa.answer(
        [[random.random() for _ in range(64)] for _ in range(64)],
        [1, 2, 3]
    )
    assert vqa_result["answer"] is not None

    # 4. TTS -> ASR
    tts_result = pipeline.tts.synthesize("你好")
    asr_result = pipeline.asr.recognize(tts_result["waveform"])
    assert asr_result["text"] is not None

    print("    PASS")


# ============================================================
# 主入口
# ============================================================

def main():
    print()
    print("=" * 70)
    print("  灵元模型 - 多模态融合引擎模块 (Part 24) 自测")
    print("=" * 70)
    print()

    tests = [
        ("TextEncoder", _test_text_encoder),
        ("ImageEncoder", _test_image_encoder),
        ("AudioEncoder", _test_audio_encoder),
        ("ModalityProjector", _test_modality_projector),
        ("CrossModalAttention", _test_cross_modal_attention),
        ("FusionStrategist", _test_fusion_strategist),
        ("ContrastiveLearner", _test_contrastive_learner),
        ("MultimodalFusionModel", _test_multimodal_model),
        ("VisualQuestionAnswering", _test_vqa),
        ("ImageCaptioner", _test_captioner),
        ("SpeechRecognizer", _test_asr),
        ("TextToSpeech", _test_tts),
        ("MultimodalEmbedder", _test_embedder),
        ("ModalityRouter", _test_router),
        ("MultimodalPipeline", _test_pipeline),
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
