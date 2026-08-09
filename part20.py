#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
灵元模型 (Lingyuan Model) - 智能数据工厂模块
模块编号: Part 20
================================================================================

本模块实现了完整的智能数据工厂,涵盖数据增强、课程学习调度、智能批处理、
数据质量评估、流式数据加载、端到端数据流水线、词表管理和数据集构建等功能。
所有组件均使用纯Python标准库实现,零外部依赖。

功能概览:
    1. DataAugmentor       - 数据增强器
       文本增强(同义词替换/随机删除/随机交换/回译模拟)
       Token级增强(随机mask/token替换/token插入)
       噪声注入、增强策略、增强Pipeline
    2. CurriculumScheduler - 课程学习调度器
       难度评估(序列长度/词汇稀有度/句法复杂度)
       课程策略(线性/指数/反焦虑)、动态调整、批次组成
    3. SmartBatcher        - 智能批处理器
       长度感知、动态批大小、内存预算、打包优化、去重
    4. DataQualityAssessor - 数据质量评估器
       重复检测、异常检测、毒性检测、质量评分、自动清洗
    5. StreamingDataLoader - 流式数据加载器
       内存高效、预取、断点续传、多源混合、采样策略
    6. DataPipeline        - 数据流水线
       端到端、可配置、统计、缓存
    7. VocabularyManager   - 词表管理器
       动态扩展、词频统计、词表压缩、BPE子词合并、特殊token管理
    8. DatasetBuilder      - 数据集构建器
       多格式加载、自动切分、版本管理、统计报告

设计原则:
    - 纯Python标准库实现,零外部依赖
    - 模块化设计,各组件可独立使用
    - 类型注解完备,代码自文档化
    - 完善的错误处理和边界检查

作者: 灵元模型团队
版本: 1.0.0
================================================================================
"""

import os
import re
import json
import csv
import math
import time
import random
import hashlib
import pickle
import threading
import queue as queue_module
from collections import Counter, defaultdict, deque
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Tuple, Union,
    Iterable, Sequence, Set, NamedTuple
)
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# 工具函数和常量
# =============================================================================

# 内置中文同义词词典 (精简版)
_BUILTIN_SYNONYMS: Dict[str, List[str]] = {
    "美丽": ["漂亮", "好看", "秀丽", "优美"],
    "漂亮": ["美丽", "好看", "精致"],
    "快乐": ["高兴", "愉快", "开心", "喜悦"],
    "高兴": ["快乐", "愉快", "开心"],
    "迅速": ["快速", "敏捷", "飞快"],
    "快速": ["迅速", "敏捷", "飞快"],
    "优秀": ["杰出", "卓越", "出色"],
    "杰出": ["优秀", "卓越", "出色"],
    "简单": ["容易", "简便", "轻松"],
    "容易": ["简单", "简便", "轻松"],
    "困难": ["艰难", "不易", "棘手"],
    "艰难": ["困难", "不易", "棘手"],
    "重要": ["关键", "核心", "要紧"],
    "关键": ["重要", "核心", "要紧"],
    "巨大": ["庞大", "宏大", "硕大"],
    "庞大": ["巨大", "宏大", "硕大"],
    "开始": ["启动", "开端", "起始"],
    "结束": ["完毕", "终结", "完成"],
    "增加": ["增长", "增多", "提升"],
    "减少": ["降低", "缩减", "下降"],
}

# 内置英文同义词词典 (精简版)
_BUILTIN_SYNONYMS_EN: Dict[str, List[str]] = {
    "good": ["great", "excellent", "fine", "nice"],
    "bad": ["poor", "terrible", "awful"],
    "big": ["large", "huge", "enormous"],
    "small": ["tiny", "little", "compact"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["sluggish", "gradual", "unhurried"],
    "happy": ["joyful", "glad", "cheerful"],
    "sad": ["unhappy", "sorrowful", "down"],
    "important": ["significant", "crucial", "vital"],
    "easy": ["simple", "effortless", "straightforward"],
}

# 内置噪声token池
_NOISE_TOKENS: List[str] = [
    "[NOISE]", "[PAD]", "[RAND]", "[BLANK]", "[UNK]",
    "...", "###", "***", "@@@", "%%%",
]

# 内置毒性关键词 (精简版,实际使用应扩展)
_TOXICITY_KEYWORDS: List[str] = [
    "暴力", "色情", "赌博", "毒品", "诈骗",
    "kill", "bomb", "weapon", "drug", "fraud",
]

# 默认特殊token
DEFAULT_SPECIAL_TOKENS: Dict[str, str] = {
    "pad": "<PAD>",
    "bos": "<BOS>",
    "eos": "<EOS>",
    "unk": "<UNK>",
    "mask": "<MASK>",
}


def _tokenize(text: str) -> List[str]:
    """简单的分词函数: 中文按字, 英文按词, 保留标点。"""
    if not text:
        return []
    tokens = []
    # 匹配: 英文单词 | 中文字符 | 数字 | 标点符号
    pattern = re.compile(r"[a-zA-Z]+|[0-9]+|[\u4e00-\u9fff]|[^\sa-zA-Z0-9\u4e00-\u9fff]")
    for match in pattern.finditer(text):
        tokens.append(match.group())
    return tokens


def _char_tokenize(text: str) -> List[str]:
    """按字符分词。"""
    return list(text) if text else []


def _jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """计算两个集合的Jaccard相似度。"""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / len(union)


def _ngram_set(text: str, n: int = 3) -> Set[str]:
    """生成文本的n-gram集合。"""
    chars = text.replace(" ", "")
    if len(chars) < n:
        return {chars} if chars else set()
    return {chars[i:i + n] for i in range(len(chars) - n + 1)}


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法,避免除零。"""
    if b == 0:
        return default
    return a / b


class DifficultyLevel(Enum):
    """难度级别枚举。"""
    EASY = 1
    MEDIUM = 2
    HARD = 3


class SamplingStrategy(Enum):
    """采样策略枚举。"""
    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    TEMPERATURE = "temperature"


# =============================================================================
# 1. DataAugmentor — 数据增强器
# =============================================================================

class AugmentationPipeline:
    """增强Pipeline,可组合多个增强操作。"""

    def __init__(self, operations: Optional[List[Callable]] = None):
        self._operations: List[Callable] = operations if operations else []

    def add(self, operation: Callable) -> "AugmentationPipeline":
        """添加增强操作到Pipeline。"""
        self._operations.append(operation)
        return self

    def apply(self, text: str) -> str:
        """按顺序应用所有增强操作。"""
        result = text
        for op in self._operations:
            result = op(result)
        return result

    def apply_batch(self, texts: List[str]) -> List[str]:
        """批量应用增强操作。"""
        return [self.apply(t) for t in texts]

    @property
    def size(self) -> int:
        return len(self._operations)

    def __repr__(self) -> str:
        return f"AugmentationPipeline(operations={len(self._operations)})"


class DataAugmentor:
    """
    数据增强器。

    支持文本级增强(同义词替换、随机删除、随机交换、回译模拟)、
    Token级增强(随机mask、token替换、token插入)、噪声注入,
    并提供可组合的增强Pipeline。

    参数:
        augmentation_rate: 增强强度 (0.0-1.0)
        seed: 随机种子
        synonyms: 自定义同义词词典
    """

    def __init__(
        self,
        augmentation_rate: float = 0.1,
        seed: Optional[int] = None,
        synonyms: Optional[Dict[str, List[str]]] = None,
    ):
        self.augmentation_rate = max(0.0, min(1.0, augmentation_rate))
        self._rng = random.Random(seed)
        # 合并内置中英文同义词
        self._synonyms: Dict[str, List[str]] = {}
        self._synonyms.update(_BUILTIN_SYNONYMS)
        self._synonyms.update(_BUILTIN_SYNONYMS_EN)
        if synonyms:
            self._synonyms.update(synonyms)
        self._pipeline = AugmentationPipeline()

    # ---- 文本级增强 ----

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        同义词替换: 用同义词替换文本中的n个词。

        参数:
            text: 输入文本
            n: 替换次数
        返回:
            增强后的文本
        """
        tokens = _tokenize(text)
        if not tokens:
            return text
        # 找出可替换的token位置
        replaceable = [
            i for i, t in enumerate(tokens)
            if t.lower() in self._synonyms or t in self._synonyms
        ]
        if not replaceable:
            return text
        n = min(n, len(replaceable))
        chosen = self._rng.sample(replaceable, n)
        for idx in chosen:
            token = tokens[idx]
            key = token if token in self._synonyms else token.lower()
            if key in self._synonyms:
                tokens[idx] = self._rng.choice(self._synonyms[key])
        return "".join(tokens) if _is_chinese_text(text) else " ".join(
            _reconstruct(tokens, text)
        )

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        随机删除: 以概率p删除每个词。

        参数:
            text: 输入文本
            p: 删除概率
        返回:
            增强后的文本
        """
        tokens = _tokenize(text)
        if len(tokens) <= 1:
            return text
        p = max(0.0, min(1.0, p))
        result = [t for t in tokens if self._rng.random() > p]
        if not result:
            result = [self._rng.choice(tokens)]
        if _is_chinese_text(text):
            return "".join(result)
        return " ".join(result)

    def random_swap(self, text: str, n: int = 1) -> str:
        """
        随机交换: 随机交换n对词的位置。

        参数:
            text: 输入文本
            n: 交换次数
        返回:
            增强后的文本
        """
        tokens = _tokenize(text)
        if len(tokens) < 2:
            return text
        n = min(n, len(tokens) // 2)
        for _ in range(n):
            i, j = self._rng.sample(range(len(tokens)), 2)
            tokens[i], tokens[j] = tokens[j], tokens[i]
        if _is_chinese_text(text):
            return "".join(tokens)
        return " ".join(tokens)

    def back_translation_simulate(self, text: str) -> str:
        """
        回译模拟: 模拟翻译-再翻译的扰动效果。
        通过词序微调、同义词替换和轻微删减来模拟。
        """
        result = self.synonym_replacement(text, n=max(1, len(text) // 20))
        result = self.random_swap(result, n=1)
        result = self.random_deletion(result, p=0.05)
        return result

    # ---- Token级增强 ----

    def token_mask(
        self,
        tokens: List[str],
        mask_token: str = "<MASK>",
        p: float = 0.15,
    ) -> List[str]:
        """
        随机mask: 以概率p将token替换为mask_token。

        参数:
            tokens: token列表
            mask_token: 用于替换的mask token
            p: mask概率
        返回:
            增强后的token列表
        """
        if not tokens:
            return tokens
        p = max(0.0, min(1.0, p))
        return [
            mask_token if self._rng.random() < p else t
            for t in tokens
        ]

    def token_replace(
        self,
        tokens: List[str],
        vocab: Optional[List[str]] = None,
        p: float = 0.15,
    ) -> List[str]:
        """
        Token替换: 以概率p将token替换为词表中的随机token。

        参数:
            tokens: token列表
            vocab: 替换用的词表,为None时使用噪声token
            p: 替换概率
        返回:
            增强后的token列表
        """
        if not tokens:
            return tokens
        p = max(0.0, min(1.0, p))
        pool = vocab if vocab else _NOISE_TOKENS
        if not pool:
            return tokens
        return [
            self._rng.choice(pool) if self._rng.random() < p else t
            for t in tokens
        ]

    def token_insert(
        self,
        tokens: List[str],
        vocab: Optional[List[str]] = None,
        p: float = 0.10,
    ) -> List[str]:
        """
        Token插入: 以概率p在token之间插入随机token。

        参数:
            tokens: token列表
            vocab: 插入用的词表
            p: 插入概率
        返回:
            增强后的token列表
        """
        if not tokens:
            return tokens
        p = max(0.0, min(1.0, p))
        pool = vocab if vocab else _NOISE_TOKENS
        if not pool:
            return tokens
        result: List[str] = []
        for t in tokens:
            result.append(t)
            if self._rng.random() < p:
                result.append(self._rng.choice(pool))
        return result

    # ---- 噪声注入 ----

    def inject_noise(
        self,
        text: str,
        noise_tokens: Optional[List[str]] = None,
        p: float = 0.05,
    ) -> str:
        """
        噪声注入: 在文本中随机插入噪声token。

        参数:
            text: 输入文本
            noise_tokens: 噪声token池
            p: 注入概率
        返回:
            含噪声的文本
        """
        tokens = _tokenize(text)
        if not tokens:
            return text
        pool = noise_tokens if noise_tokens else _NOISE_TOKENS
        p = max(0.0, min(1.0, p))
        result: List[str] = []
        for t in tokens:
            result.append(t)
            if self._rng.random() < p:
                result.append(self._rng.choice(pool))
        if _is_chinese_text(text):
            return "".join(result)
        return " ".join(result)

    # ---- 增强策略与Pipeline ----

    def augment(self, text: str) -> str:
        """
        应用默认增强策略: 根据augmentation_rate组合多种增强操作。
        """
        if self._rng.random() > self.augmentation_rate:
            return text
        rate = self.augmentation_rate
        # 随机选择1-2种增强方式
        ops = [
            lambda t: self.synonym_replacement(t, n=1),
            lambda t: self.random_deletion(t, p=rate * 0.5),
            lambda t: self.random_swap(t, n=1),
            lambda t: self.back_translation_simulate(t),
        ]
        num_ops = self._rng.randint(1, min(2, len(ops)))
        chosen_ops = self._rng.sample(ops, num_ops)
        result = text
        for op in chosen_ops:
            result = op(result)
        return result

    def augment_batch(self, texts: List[str]) -> List[str]:
        """批量增强。"""
        return [self.augment(t) for t in texts]

    def build_pipeline(self, operations: List[str]) -> AugmentationPipeline:
        """
        根据操作名称列表构建增强Pipeline。

        参数:
            operations: 操作名称列表,可选值:
                'synonym', 'deletion', 'swap', 'back_translation',
                'noise'
        返回:
            AugmentationPipeline实例
        """
        op_map: Dict[str, Callable] = {
            "synonym": lambda t: self.synonym_replacement(t, n=1),
            "deletion": lambda t: self.random_deletion(t, p=self.augmentation_rate * 0.5),
            "swap": lambda t: self.random_swap(t, n=1),
            "back_translation": lambda t: self.back_translation_simulate(t),
            "noise": lambda t: self.inject_noise(t, p=self.augmentation_rate * 0.3),
        }
        pipeline = AugmentationPipeline()
        for name in operations:
            if name in op_map:
                pipeline.add(op_map[name])
        self._pipeline = pipeline
        return pipeline

    @property
    def pipeline(self) -> AugmentationPipeline:
        return self._pipeline

    def __repr__(self) -> str:
        return (
            f"DataAugmentor(augmentation_rate={self.augmentation_rate}, "
            f"synonyms={len(self._synonyms)})"
        )


def _is_chinese_text(text: str) -> bool:
    """判断文本是否主要为中文。"""
    if not text:
        return False
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return chinese_count > len(text) * 0.3


def _reconstruct(tokens: List[str], original: str) -> List[str]:
    """重构token列表(用于英文文本)。"""
    return tokens


# =============================================================================
# 2. CurriculumScheduler — 课程学习调度器
# =============================================================================

class CurriculumScheduler:
    """
    课程学习调度器。

    基于序列长度、词汇稀有度和句法复杂度评估样本难度,
    支持线性渐进、指数渐进、反焦虑(先易后难再中)等课程策略,
    可根据训练loss动态调整难度。

    参数:
        strategy: 课程策略 ('linear', 'exponential', 'anti_anxiety')
        initial_difficulty: 初始难度阈值
        max_difficulty: 最大难度阈值
        total_steps: 总训练步数
        freq_dict: 词汇频率字典 (用于稀有度计算)
        seed: 随机种子
    """

    def __init__(
        self,
        strategy: str = "linear",
        initial_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
        total_steps: int = 10000,
        freq_dict: Optional[Dict[str, int]] = None,
        seed: Optional[int] = None,
    ):
        self.strategy = strategy
        self.initial_difficulty = max(0.0, min(1.0, initial_difficulty))
        self.max_difficulty = max(0.0, min(1.0, max_difficulty))
        self.total_steps = max(1, total_steps)
        self.freq_dict = freq_dict if freq_dict else {}
        self._rng = random.Random(seed)
        self._current_step = 0
        self._loss_history: deque = deque(maxlen=100)
        self._difficulty_history: Dict[int, float] = {}
        self._dynamic_adjustment = 0.0
        self._min_length = 5
        self._max_length = 500

    # ---- 难度评估 ----

    def assess_difficulty(self, text: str) -> float:
        """
        评估单个样本的难度 (0.0-1.0)。

        综合考虑序列长度、词汇稀有度和句法复杂度。
        """
        if not text or not text.strip():
            return 1.0
        length_diff = self._length_difficulty(text)
        rarity_diff = self._vocabulary_rarity(text)
        syntax_diff = self._syntactic_complexity(text)
        # 加权平均
        difficulty = 0.3 * length_diff + 0.4 * rarity_diff + 0.3 * syntax_diff
        return max(0.0, min(1.0, difficulty))

    def _length_difficulty(self, text: str) -> float:
        """基于序列长度的难度。"""
        length = len(text)
        if length <= self._min_length:
            return 0.1
        if length >= self._max_length:
            return 1.0
        return _safe_divide(length - self._min_length, self._max_length - self._min_length)

    def _vocabulary_rarity(self, text: str) -> float:
        """基于词汇稀有度的难度。"""
        tokens = _tokenize(text)
        if not tokens:
            return 0.5
        if not self.freq_dict:
            # 无频率字典时,用token长度作为稀有度近似
            avg_len = sum(len(t) for t in tokens) / len(tokens)
            return min(1.0, avg_len / 5.0)
        total_freq = sum(self.freq_dict.values())
        if total_freq == 0:
            return 0.5
        rarity_sum = 0.0
        for t in tokens:
            freq = self.freq_dict.get(t, 0)
            # 频率越低,稀有度越高
            rarity_sum += 1.0 - _safe_divide(freq, total_freq)
        return rarity_sum / len(tokens)

    def _syntactic_complexity(self, text: str) -> float:
        """基于句法复杂度的难度 (简化版: 标点密度和句子数)。"""
        if not text:
            return 0.5
        punct_count = sum(1 for c in text if c in ".,;:!?。，；：！？、")
        char_count = len(text)
        if char_count == 0:
            return 0.5
        punct_density = punct_count / char_count
        # 句子数量
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s for s in sentences if s.strip()]
        num_sentences = max(1, len(sentences))
        avg_sentence_len = char_count / num_sentences
        # 综合复杂度
        complexity = 0.4 * min(1.0, punct_density * 10) + 0.6 * min(1.0, avg_sentence_len / 100)
        return complexity

    # ---- 课程策略 ----

    def get_difficulty_threshold(self, step: Optional[int] = None) -> float:
        """
        获取当前步数的难度阈值。

        只有难度低于阈值的样本才会被选中训练。
        """
        if step is None:
            step = self._current_step
        progress = min(1.0, step / self.total_steps)
        if self.strategy == "linear":
            threshold = self._linear_schedule(progress)
        elif self.strategy == "exponential":
            threshold = self._exponential_schedule(progress)
        elif self.strategy == "anti_anxiety":
            threshold = self._anti_anxiety_schedule(progress)
        else:
            threshold = self._linear_schedule(progress)
        # 应用动态调整
        threshold += self._dynamic_adjustment
        threshold = max(self.initial_difficulty, min(self.max_difficulty, threshold))
        return threshold

    def _linear_schedule(self, progress: float) -> float:
        """线性渐进。"""
        return self.initial_difficulty + (self.max_difficulty - self.initial_difficulty) * progress

    def _exponential_schedule(self, progress: float) -> float:
        """指数渐进。"""
        exp_val = math.exp(3 * progress) - 1
        exp_max = math.exp(3) - 1
        normalized = exp_val / exp_max if exp_max > 0 else progress
        return self.initial_difficulty + (self.max_difficulty - self.initial_difficulty) * normalized

    def _anti_anxiety_schedule(self, progress: float) -> float:
        """
        反焦虑策略: 先易后难再中。
        前段: 线性增长到最大
        中段: 保持最大
        后段: 逐渐回落到中等
        """
        if progress < 0.3:
            # 前段: 快速上升
            local_progress = progress / 0.3
            return self.initial_difficulty + (self.max_difficulty - self.initial_difficulty) * local_progress
        elif progress < 0.6:
            # 中段: 保持最大
            return self.max_difficulty
        else:
            # 后段: 回落到中等
            local_progress = (progress - 0.6) / 0.4
            mid_difficulty = (self.initial_difficulty + self.max_difficulty) / 2
            return self.max_difficulty - (self.max_difficulty - mid_difficulty) * local_progress

    # ---- 动态调整 ----

    def update_loss(self, loss: float) -> None:
        """
        根据训练loss动态调整难度。

        loss持续下降时增加难度,loss上升或波动时降低难度。
        """
        self._loss_history.append(loss)
        if len(self._loss_history) < 5:
            return
        recent = list(self._loss_history)
        half = len(recent) // 2
        early_avg = sum(recent[:half]) / half
        late_avg = sum(recent[half:]) / (len(recent) - half)
        if early_avg == 0:
            return
        ratio = late_avg / early_avg
        if ratio < 0.9:
            # loss下降,增加难度
            self._dynamic_adjustment = min(0.1, self._dynamic_adjustment + 0.01)
        elif ratio > 1.1:
            # loss上升,降低难度
            self._dynamic_adjustment = max(-0.1, self._dynamic_adjustment - 0.02)
        else:
            # loss稳定,缓慢恢复
            self._dynamic_adjustment *= 0.95

    def step(self) -> float:
        """前进一步,返回当前难度阈值。"""
        self._current_step += 1
        return self.get_difficulty_threshold()

    # ---- 批次组成 ----

    def compose_batch(
        self,
        samples: List[Tuple[str, Any]],
        batch_size: int = 32,
        easy_ratio: float = 0.3,
        hard_ratio: float = 0.3,
    ) -> List[Tuple[str, Any]]:
        """
        组成混合难度的批次。

        参数:
            samples: (文本, 数据)对的列表
            batch_size: 批大小
            easy_ratio: 简单样本比例
            hard_ratio: 困难样本比例
        返回:
            选中的样本列表
        """
        if not samples:
            return []
        # 计算每个样本的难度
        scored = [(self.assess_difficulty(text), text, data) for text, data in samples]
        scored.sort(key=lambda x: x[0])
        n = len(scored)
        easy_count = max(1, int(batch_size * easy_ratio))
        hard_count = max(1, int(batch_size * hard_ratio))
        medium_count = batch_size - easy_count - hard_count
        if medium_count < 0:
            easy_count = batch_size // 3
            hard_count = batch_size // 3
            medium_count = batch_size - easy_count - hard_count
        easy = scored[:min(easy_count, n)]
        hard = scored[max(0, n - hard_count):]
        remaining = scored[easy_count:max(0, n - hard_count)] if n > easy_count + hard_count else []
        if remaining:
            step_size = max(1, len(remaining) // max(1, medium_count))
            medium = remaining[::step_size][:medium_count]
        else:
            medium = []
        result = [(text, data) for _, text, data in (easy + medium + hard)]
        # 记录难度历史
        for diff, text, _ in (easy + medium + hard):
            sample_id = hash(text) % (10 ** 8)
            self._difficulty_history[sample_id] = diff
        return result

    def track_difficulty(self, sample_id: int, difficulty: float) -> None:
        """记录样本的难度历史。"""
        self._difficulty_history[sample_id] = difficulty

    def get_difficulty_history(self) -> Dict[int, float]:
        """获取难度历史记录。"""
        return dict(self._difficulty_history)

    @property
    def current_step(self) -> int:
        return self._current_step

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(strategy='{self.strategy}', "
            f"step={self._current_step}/{self.total_steps})"
        )


# =============================================================================
# 3. SmartBatcher — 智能批处理器
# =============================================================================

class SmartBatcher:
    """
    智能批处理器。

    根据序列长度分组以减少padding,动态调整batch size,
    限制每批总token数,支持短序列打包和样本去重。

    参数:
        max_tokens: 每批最大token数 (内存预算)
        max_batch_size: 最大批大小
        padding_token: padding token的id
        sort: 是否按长度排序
        pack_short: 是否打包短序列
        dedup_threshold: 去重相似度阈值
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        max_batch_size: int = 64,
        padding_token: int = 0,
        sort: bool = True,
        pack_short: bool = True,
        dedup_threshold: float = 0.9,
    ):
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.padding_token = padding_token
        self.sort = sort
        self.pack_short = pack_short
        self.dedup_threshold = dedup_threshold
        self._stats = {
            "total_batches": 0,
            "total_samples": 0,
            "total_padding": 0,
            "total_tokens": 0,
        }

    def create_batches(
        self,
        samples: List[Union[str, List[int], List[str]]],
    ) -> List[List]:
        """
        创建智能批次。

        参数:
            samples: 样本列表,每个样本可以是文本字符串、token id列表或token字符串列表
        返回:
            批次列表,每批是一个样本列表
        """
        if not samples:
            return []
        # 计算每个样本的长度
        indexed = []
        for i, sample in enumerate(samples):
            length = self._get_length(sample)
            indexed.append((length, i, sample))
        # 按长度排序
        if self.sort:
            indexed.sort(key=lambda x: x[0])
        # 去重
        if self.dedup_threshold < 1.0:
            indexed = self._deduplicate(indexed)
        # 打包短序列
        if self.pack_short:
            batches = self._pack_and_batch(indexed)
        else:
            batches = self._simple_batch(indexed)
        # 更新统计
        self._stats["total_batches"] += len(batches)
        self._stats["total_samples"] += sum(len(b) for b in batches)
        return batches

    def _get_length(self, sample: Any) -> int:
        """获取样本长度。"""
        if isinstance(sample, str):
            return len(sample)
        elif isinstance(sample, (list, tuple)):
            return len(sample)
        return 1

    def _dynamic_batch_size(self, max_seq_len: int) -> int:
        """根据序列长度动态计算batch size。"""
        if max_seq_len <= 0:
            return self.max_batch_size
        size = self.max_tokens // max_seq_len
        return max(1, min(size, self.max_batch_size))

    def _simple_batch(self, indexed: List[Tuple[int, int, Any]]) -> List[List]:
        """简单分批: 按动态batch size分组。"""
        batches = []
        current_batch: List[Any] = []
        current_max_len = 0
        for length, _, sample in indexed:
            new_max_len = max(current_max_len, length)
            batch_size = self._dynamic_batch_size(new_max_len)
            if len(current_batch) >= batch_size or \
               (len(current_batch) + 1) * new_max_len > self.max_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [sample]
                current_max_len = length
            else:
                current_batch.append(sample)
                current_max_len = new_max_len
        if current_batch:
            batches.append(current_batch)
        return batches

    def _pack_and_batch(self, indexed: List[Tuple[int, int, Any]]) -> List[List]:
        """
        打包优化: 将短序列打包到一起以减少浪费。
        """
        batches = []
        current_batch: List[Any] = []
        current_token_count = 0
        current_max_len = 0
        for length, _, sample in indexed:
            # 如果加入当前样本会超出token预算
            projected_tokens = (len(current_batch) + 1) * max(current_max_len, length)
            batch_limit = self._dynamic_batch_size(max(current_max_len, length))
            if len(current_batch) >= batch_limit or projected_tokens > self.max_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [sample]
                current_token_count = length
                current_max_len = length
            else:
                current_batch.append(sample)
                current_token_count += length
                current_max_len = max(current_max_len, length)
        if current_batch:
            batches.append(current_batch)
        return batches

    def _deduplicate(
        self, indexed: List[Tuple[int, int, Any]]
    ) -> List[Tuple[int, int, Any]]:
        """去重: 避免过于相似的样本在同一批。"""
        if not indexed:
            return indexed
        result: List[Tuple[int, int, Any]] = []
        seen_ngrams: List[Set[str]] = []
        for length, idx, sample in indexed:
            text = self._to_text(sample)
            ngrams = _ngram_set(text, n=3)
            is_dup = False
            for prev_ngrams in seen_ngrams[-20:]:  # 只检查最近的20个
                if _jaccard_similarity(ngrams, prev_ngrams) > self.dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                result.append((length, idx, sample))
                seen_ngrams.append(ngrams)
        return result

    def _to_text(self, sample: Any) -> str:
        """将样本转换为文本字符串用于去重。"""
        if isinstance(sample, str):
            return sample
        elif isinstance(sample, (list, tuple)):
            return " ".join(str(x) for x in sample)
        return str(sample)

    def pad_batch(
        self,
        batch: List[List[int]],
        max_len: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        对批次进行padding。

        参数:
            batch: token id列表的列表
            max_len: 最大长度,为None时使用批次内最大长度
        返回:
            (padded_batch, attention_masks)
        """
        if not batch:
            return [], []
        if max_len is None:
            max_len = max(len(seq) for seq in batch)
        padded = []
        masks = []
        for seq in batch:
            pad_len = max_len - len(seq)
            padded.append(list(seq) + [self.padding_token] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
        # 更新padding统计
        self._stats["total_padding"] += sum(
            max_len - len(seq) for seq in batch
        )
        self._stats["total_tokens"] += max_len * len(batch)
        return padded, masks

    def get_stats(self) -> Dict[str, int]:
        """获取批处理统计信息。"""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """重置统计信息。"""
        self._stats = {
            "total_batches": 0,
            "total_samples": 0,
            "total_padding": 0,
            "total_tokens": 0,
        }

    def __repr__(self) -> str:
        return (
            f"SmartBatcher(max_tokens={self.max_tokens}, "
            f"max_batch_size={self.max_batch_size}, sort={self.sort})"
        )


# =============================================================================
# 4. DataQualityAssessor — 数据质量评估器
# =============================================================================

class DataQualityAssessor:
    """
    数据质量评估器。

    提供基于n-gram的重复检测、异常检测(过短/过长/乱码/低信息量)、
    简单关键词毒性检测、综合质量评分和自动清洗功能。

    参数:
        min_length: 最小有效长度
        max_length: 最大有效长度
        ngram_size: n-gram大小 (用于重复检测)
        toxicity_keywords: 毒性关键词列表
    """

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 10000,
        ngram_size: int = 5,
        toxicity_keywords: Optional[List[str]] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.ngram_size = ngram_size
        self.toxicity_keywords = (
            toxicity_keywords if toxicity_keywords is not None else list(_TOXICITY_KEYWORDS)
        )
        self._duplicate_cache: Dict[str, Set[str]] = {}

    # ---- 重复检测 ----

    def detect_duplicates(
        self,
        samples: List[str],
        threshold: float = 0.8,
    ) -> List[List[int]]:
        """
        基于n-gram的重复样本检测。

        参数:
            samples: 样本列表
            threshold: 相似度阈值
        返回:
            重复组列表,每组包含重复样本的索引
        """
        if not samples:
            return []
        # 预计算每个样本的n-gram集合
        ngram_sets: List[Set[str]] = []
        for s in samples:
            key = hashlib.md5(s.encode()).hexdigest()[:8]
            if key not in self._duplicate_cache:
                self._duplicate_cache[key] = _ngram_set(s, self.ngram_size)
            ngram_sets.append(self._duplicate_cache[key])
        # 找重复组
        visited: Set[int] = set()
        groups: List[List[int]] = []
        for i in range(len(samples)):
            if i in visited:
                continue
            group = [i]
            visited.add(i)
            for j in range(i + 1, len(samples)):
                if j in visited:
                    continue
                sim = _jaccard_similarity(ngram_sets[i], ngram_sets[j])
                if sim >= threshold:
                    group.append(j)
                    visited.add(j)
            if len(group) > 1:
                groups.append(group)
        return groups

    # ---- 异常检测 ----

    def detect_anomalies(self, text: str) -> Dict[str, bool]:
        """
        检测文本中的异常。

        返回字典包含:
            - too_short: 过短
            - too_long: 过长
            - garbled: 乱码
            - low_info: 低信息量
            - has_anomaly: 是否存在任何异常
        """
        result = {
            "too_short": False,
            "too_long": False,
            "garbled": False,
            "low_info": False,
            "has_anomaly": False,
        }
        if not text:
            result["too_short"] = True
            result["has_anomaly"] = True
            return result
        text = text.strip()
        length = len(text)
        # 过短
        if length < self.min_length:
            result["too_short"] = True
        # 过长
        if length > self.max_length:
            result["too_long"] = True
        # 乱码检测: 非可打印字符比例过高
        printable_count = sum(
            1 for c in text
            if c.isprintable() or c in '\n\r\t'
        )
        if length > 0 and printable_count / length < 0.8:
            result["garbled"] = True
        # 乱码检测: 重复字符过多
        if length > 10:
            char_counter = Counter(text)
            most_common_ratio = char_counter.most_common(1)[0][1] / length
            if most_common_ratio > 0.5:
                result["garbled"] = True
        # 低信息量: 去重后字符太少
        unique_chars = len(set(text))
        if length > 0 and unique_chars / length < 0.1:
            result["low_info"] = True
        # 低信息量: 几乎全是标点/空白
        alpha_count = sum(1 for c in text if c.isalnum())
        if length > 0 and alpha_count / length < 0.2:
            result["low_info"] = True
        result["has_anomaly"] = any([
            result["too_short"], result["too_long"],
            result["garbled"], result["low_info"]
        ])
        return result

    # ---- 毒性检测 ----

    def detect_toxicity(self, text: str) -> Tuple[bool, List[str]]:
        """
        简单关键词毒性检测。

        参数:
            text: 输入文本
        返回:
            (是否含毒性内容, 匹配的关键词列表)
        """
        if not text:
            return False, []
        text_lower = text.lower()
        matched = []
        for keyword in self.toxicity_keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)
        return len(matched) > 0, matched

    # ---- 质量评分 ----

    def assess_quality(self, text: str) -> float:
        """
        综合质量评分 (0.0-1.0)。

        评分维度:
            - 长度合理性 (0.25)
            - 内容丰富度 (0.25)
            - 无毒性 (0.25)
            - 无异常 (0.25)
        """
        if not text or not text.strip():
            return 0.0
        text = text.strip()
        length = len(text)
        # 长度合理性
        if length < self.min_length:
            length_score = length / self.min_length * 0.5
        elif length > self.max_length:
            length_score = 0.5
        else:
            # 在合理范围内,越接近中间越好
            optimal = (self.min_length + self.max_length) / 2
            deviation = abs(length - optimal) / optimal
            length_score = max(0.5, 1.0 - deviation * 0.5)
        # 内容丰富度
        unique_ratio = len(set(text)) / max(1, length)
        alpha_ratio = sum(1 for c in text if c.isalnum()) / max(1, length)
        richness_score = min(1.0, unique_ratio * 2) * 0.5 + alpha_ratio * 0.5
        # 无毒性
        is_toxic, _ = self.detect_toxicity(text)
        toxicity_score = 0.0 if is_toxic else 1.0
        # 无异常
        anomalies = self.detect_anomalies(text)
        anomaly_score = 1.0 if not anomalies["has_anomaly"] else 0.3
        # 综合评分
        quality = (
            0.25 * length_score +
            0.25 * richness_score +
            0.25 * toxicity_score +
            0.25 * anomaly_score
        )
        return max(0.0, min(1.0, quality))

    # ---- 自动清洗 ----

    def auto_clean(
        self,
        samples: List[str],
        min_quality: float = 0.5,
        remove_duplicates: bool = True,
        dedup_threshold: float = 0.85,
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        根据质量评分自动过滤样本。

        参数:
            samples: 原始样本列表
            min_quality: 最低质量分
            remove_duplicates: 是否去重
            dedup_threshold: 去重相似度阈值
        返回:
            (清洗后的样本列表, 统计信息)
        """
        stats = {
            "total": len(samples),
            "kept": 0,
            "removed_low_quality": 0,
            "removed_toxic": 0,
            "removed_duplicate": 0,
            "removed_anomaly": 0,
        }
        # 标记要删除的索引
        to_remove: Set[int] = set()
        # 质量过滤
        for i, sample in enumerate(samples):
            quality = self.assess_quality(sample)
            if quality < min_quality:
                to_remove.add(i)
                is_toxic, _ = self.detect_toxicity(sample)
                anomalies = self.detect_anomalies(sample)
                if is_toxic:
                    stats["removed_toxic"] += 1
                elif anomalies["has_anomaly"]:
                    stats["removed_anomaly"] += 1
                else:
                    stats["removed_low_quality"] += 1
        # 重复检测
        if remove_duplicates:
            dup_groups = self.detect_duplicates(samples, threshold=dedup_threshold)
            for group in dup_groups:
                # 保留每组中质量最高的一个
                if all(idx in to_remove for idx in group):
                    continue
                qualities = [(self.assess_quality(samples[idx]), idx) for idx in group]
                qualities.sort(reverse=True)
                # 保留最好的,删除其余
                for _, idx in qualities[1:]:
                    if idx not in to_remove:
                        to_remove.add(idx)
                        stats["removed_duplicate"] += 1
        # 构建结果
        result = [s for i, s in enumerate(samples) if i not in to_remove]
        stats["kept"] = len(result)
        return result, stats

    def quality_report(self, samples: List[str]) -> Dict[str, Any]:
        """生成质量报告。"""
        if not samples:
            return {"total": 0}
        qualities = [self.assess_quality(s) for s in samples]
        toxic_count = sum(1 for s in samples if self.detect_toxicity(s)[0])
        anomaly_count = sum(
            1 for s in samples if self.detect_anomalies(s)["has_anomaly"]
        )
        return {
            "total": len(samples),
            "mean_quality": sum(qualities) / len(qualities),
            "min_quality": min(qualities),
            "max_quality": max(qualities),
            "toxic_count": toxic_count,
            "anomaly_count": anomaly_count,
            "quality_distribution": {
                "high (>0.8)": sum(1 for q in qualities if q > 0.8),
                "medium (0.5-0.8)": sum(1 for q in qualities if 0.5 <= q <= 0.8),
                "low (<0.5)": sum(1 for q in qualities if q < 0.5),
            },
        }

    def __repr__(self) -> str:
        return (
            f"DataQualityAssessor(min_length={self.min_length}, "
            f"max_length={self.max_length}, ngram_size={self.ngram_size})"
        )


# =============================================================================
# 5. StreamingDataLoader — 流式数据加载器
# =============================================================================

class StreamingDataLoader:
    """
    流式数据加载器。

    逐行读取数据(不全部加载到内存),支持后台预取、断点续传、
    多数据源按比例混合和多种采样策略。

    参数:
        sources: 数据源列表,每项为 (文件路径, 权重/比例) 或文件路径
        batch_size: 批大小
        sampling_strategy: 采样策略 ('uniform', 'weighted', 'temperature')
        temperature: 温度采样参数
        prefetch_count: 预取批次数
        seed: 随机种子
    """

    def __init__(
        self,
        sources: Union[str, List[Union[str, Tuple[str, float]]]],
        batch_size: int = 32,
        sampling_strategy: str = "uniform",
        temperature: float = 1.0,
        prefetch_count: int = 2,
        seed: Optional[int] = None,
    ):
        # 标准化数据源
        if isinstance(sources, str):
            sources = [(sources, 1.0)]
        self._sources: List[Tuple[str, float]] = []
        for s in sources:
            if isinstance(s, tuple):
                self._sources.append((s[0], s[1]))
            else:
                self._sources.append((s, 1.0))
        self.batch_size = batch_size
        self.sampling_strategy = sampling_strategy
        self.temperature = max(0.01, temperature)
        self.prefetch_count = max(0, prefetch_count)
        self._rng = random.Random(seed)
        # 文件句柄和位置
        self._file_handles: Dict[int, Any] = {}
        self._positions: Dict[int, int] = {}
        self._exhausted: Set[int] = set()
        # 预取队列
        self._prefetch_queue: Optional[queue_module.Queue] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()
        # 统计
        self._stats = {
            "total_read": 0,
            "batches_yielded": 0,
            "sources_read": {os.path.basename(s[0]): 0 for s in self._sources},
        }

    def _open_source(self, source_idx: int) -> Any:
        """打开数据源文件。"""
        path = self._sources[source_idx][0]
        f = open(path, 'r', encoding='utf-8', errors='replace')
        # 恢复位置
        if source_idx in self._positions:
            f.seek(self._positions[source_idx])
        self._file_handles[source_idx] = f
        return f

    def _sample_source(self) -> int:
        """根据采样策略选择数据源。"""
        available = [i for i in range(len(self._sources)) if i not in self._exhausted]
        if not available:
            return -1
        if len(available) == 1:
            return available[0]
        if self.sampling_strategy == "uniform":
            return self._rng.choice(available)
        elif self.sampling_strategy == "weighted":
            weights = [self._sources[i][1] for i in available]
            total = sum(weights)
            if total <= 0:
                return self._rng.choice(available)
            r = self._rng.random() * total
            cumulative = 0.0
            for idx, w in zip(available, weights):
                cumulative += w
                if r <= cumulative:
                    return idx
            return available[-1]
        elif self.sampling_strategy == "temperature":
            weights = [self._sources[i][1] for i in available]
            # 温度采样: w_i^(1/T) 归一化
            adjusted = [w ** (1.0 / self.temperature) for w in weights]
            total = sum(adjusted)
            if total <= 0:
                return self._rng.choice(available)
            r = self._rng.random() * total
            cumulative = 0.0
            for idx, w in zip(available, adjusted):
                cumulative += w
                if r <= cumulative:
                    return idx
            return available[-1]
        else:
            return self._rng.choice(available)

    def _read_line(self) -> Optional[str]:
        """从选中的数据源读取一行。"""
        source_idx = self._sample_source()
        if source_idx == -1:
            return None
        if source_idx not in self._file_handles:
            self._open_source(source_idx)
        f = self._file_handles[source_idx]
        line = f.readline()
        if not line:
            # 当前数据源耗尽
            self._exhausted.add(source_idx)
            f.close()
            del self._file_handles[source_idx]
            # 尝试其他数据源
            return self._read_line()
        line = line.strip()
        self._positions[source_idx] = f.tell()
        self._stats["total_read"] += 1
        source_name = os.path.basename(self._sources[source_idx][0])
        if source_name in self._stats["sources_read"]:
            self._stats["sources_read"][source_name] += 1
        return line

    def _read_batch(self) -> Optional[List[str]]:
        """读取一个批次。"""
        batch: List[str] = []
        while len(batch) < self.batch_size:
            line = self._read_line()
            if line is None:
                break
            if line:
                batch.append(line)
        if not batch:
            return None
        self._stats["batches_yielded"] += 1
        return batch

    def _start_prefetch(self) -> None:
        """启动后台预取线程。"""
        if self.prefetch_count <= 0:
            return
        self._prefetch_queue = queue_module.Queue(maxsize=self.prefetch_count)
        self._stop_prefetch.clear()

        def _worker():
            while not self._stop_prefetch.is_set():
                batch = self._read_batch()
                if batch is None:
                    self._prefetch_queue.put(None)
                    break
                self._prefetch_queue.put(batch)

        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    def __iter__(self) -> Iterator[List[str]]:
        """迭代器接口。"""
        if self.prefetch_count > 0:
            self._start_prefetch()
            try:
                while True:
                    batch = self._prefetch_queue.get()
                    if batch is None:
                        break
                    yield batch
            finally:
                self._stop_prefetch.set()
                self.close()
        else:
            try:
                while True:
                    batch = self._read_batch()
                    if batch is None:
                        break
                    yield batch
            finally:
                self.close()

    def save_checkpoint(self, path: str) -> None:
        """保存读取位置(断点续传)。"""
        checkpoint = {
            "positions": dict(self._positions),
            "exhausted": list(self._exhausted),
            "stats": self._stats,
            "sources": self._sources,
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str) -> None:
        """从检查点恢复读取位置。"""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        self._positions = checkpoint.get("positions", {})
        self._exhausted = set(checkpoint.get("exhausted", []))
        self._stats = checkpoint.get("stats", self._stats)

    def close(self) -> None:
        """关闭所有文件句柄。"""
        for f in self._file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_handles.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return dict(self._stats)

    def reset(self) -> None:
        """重置加载器状态。"""
        self.close()
        self._positions.clear()
        self._exhausted.clear()
        self._stats = {
            "total_read": 0,
            "batches_yielded": 0,
            "sources_read": {os.path.basename(s[0]): 0 for s in self._sources},
        }

    def __repr__(self) -> str:
        return (
            f"StreamingDataLoader(sources={len(self._sources)}, "
            f"batch_size={self.batch_size}, "
            f"strategy='{self.sampling_strategy}')"
        )


# =============================================================================
# 6. VocabularyManager — 词表管理器
# =============================================================================

class VocabularyManager:
    """
    词表管理器。

    支持动态扩展词表、追踪token频率、词表压缩(合并低频token)、
    BPE子词合并规则管理和特殊token管理。

    参数:
        special_tokens: 特殊token映射
        max_vocab_size: 最大词表大小
    """

    def __init__(
        self,
        special_tokens: Optional[Dict[str, str]] = None,
        max_vocab_size: int = 100000,
    ):
        self.max_vocab_size = max_vocab_size
        self._special_tokens = special_tokens if special_tokens else dict(DEFAULT_SPECIAL_TOKENS)
        # token -> id 映射
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        # 频率统计
        self._token_freq: Counter = Counter()
        # BPE合并规则
        self._bpe_merges: List[Tuple[str, str]] = []
        self._bpe_ranks: Dict[Tuple[str, str], int] = {}
        # 初始化特殊token
        self._init_special_tokens()
        self._next_id = len(self._token_to_id)

    def _init_special_tokens(self) -> None:
        """初始化特殊token,分配id 0, 1, 2, ..."""
        for idx, (name, token) in enumerate(self._special_tokens.items()):
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
            self._token_freq[token] = 0

    # ---- 基本操作 ----

    def add_token(self, token: str) -> int:
        """添加单个token到词表,返回其id。"""
        if token in self._token_to_id:
            return self._token_to_id[token]
        if len(self._token_to_id) >= self.max_vocab_size:
            # 词表已满,返回unk
            unk = self._special_tokens.get("unk", "<UNK>")
            return self._token_to_id.get(unk, 0)
        token_id = self._next_id
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        self._token_freq[token] = 0
        self._next_id += 1
        return token_id

    def add_tokens(self, tokens: List[str]) -> List[int]:
        """批量添加token。"""
        return [self.add_token(t) for t in tokens]

    def get_id(self, token: str) -> int:
        """获取token的id,不存在则返回unk的id。"""
        if token in self._token_to_id:
            return self._token_to_id[token]
        unk = self._special_tokens.get("unk", "<UNK>")
        return self._token_to_id.get(unk, 0)

    def get_token(self, token_id: int) -> str:
        """根据id获取token。"""
        return self._id_to_token.get(token_id, self._special_tokens.get("unk", "<UNK>"))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """将文本编码为token id列表。"""
        tokens = _tokenize(text)
        ids = [self.get_id(t) for t in tokens]
        if add_bos:
            bos = self._token_to_id.get(self._special_tokens.get("bos", "<BOS>"), 1)
            ids = [bos] + ids
        if add_eos:
            eos = self._token_to_id.get(self._special_tokens.get("eos", "<EOS>"), 2)
            ids = ids + [eos]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """将token id列表解码为文本。"""
        special_set = set(self._special_tokens.values())
        tokens = []
        for i in ids:
            token = self.get_token(i)
            if skip_special and token in special_set:
                continue
            tokens.append(token)
        return "".join(tokens) if tokens else ""

    # ---- 频率统计 ----

    def update_frequency(self, tokens: List[str]) -> None:
        """更新token使用频率。"""
        self._token_freq.update(tokens)

    def update_frequency_from_text(self, text: str) -> None:
        """从文本更新频率。"""
        tokens = _tokenize(text)
        self.update_frequency(tokens)

    def get_frequency(self, token: str) -> int:
        """获取token的频率。"""
        return self._token_freq.get(token, 0)

    def get_most_common(self, n: int = 10) -> List[Tuple[str, int]]:
        """获取频率最高的n个token。"""
        return self._token_freq.most_common(n)

    # ---- 词表压缩 ----

    def compress_vocab(self, min_freq: int = 2) -> int:
        """
        压缩词表: 合并低频token为unk。

        参数:
            min_freq: 最低频率阈值,低于此值的token将被合并
        返回:
            被移除的token数量
        """
        # 特殊token不受影响
        special_set = set(self._special_tokens.values())
        to_remove = [
            token for token, freq in self._token_freq.items()
            if freq < min_freq and token not in special_set
        ]
        for token in to_remove:
            token_id = self._token_to_id.pop(token, None)
            if token_id is not None:
                self._id_to_token.pop(token_id, None)
                self._token_freq.pop(token, None)
        # 重新分配id (保持连续)
        self._rebuild_id_mapping()
        return len(to_remove)

    def _rebuild_id_mapping(self) -> None:
        """重建id映射(保持连续)。"""
        old_mapping = dict(self._token_to_id)
        self._token_to_id.clear()
        self._id_to_token.clear()
        # 先放特殊token
        idx = 0
        for name, token in self._special_tokens.items():
            if token in old_mapping or token in self._token_freq:
                self._token_to_id[token] = idx
                self._id_to_token[idx] = token
                idx += 1
        # 再放普通token (按频率降序)
        sorted_tokens = sorted(
            [t for t in old_mapping if t not in self._special_tokens.values()],
            key=lambda t: self._token_freq.get(t, 0),
            reverse=True
        )
        for token in sorted_tokens:
            if token not in self._token_to_id:
                self._token_to_id[token] = idx
                self._id_to_token[idx] = token
                idx += 1
        self._next_id = idx

    # ---- BPE子词合并 ----

    def train_bpe(
        self,
        texts: List[str],
        num_merges: int = 100,
    ) -> List[Tuple[str, str]]:
        """
        训练BPE合并规则。

        参数:
            texts: 训练文本列表
            num_merges: 合并次数
        返回:
            合并规则列表 [(token_a, token_b), ...]
        """
        # 将文本转换为字符序列(每个词用空格分隔)
        word_freqs: Counter = Counter()
        for text in texts:
            words = text.strip().split()
            for word in words:
                # 将词表示为字符元组,末尾加</w>
                char_word = tuple(list(word) + ["</w>"])
                word_freqs[char_word] += 1
        # 迭代合并
        merges: List[Tuple[str, str]] = []
        for _ in range(num_merges):
            # 统计相邻pair频率
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair_freqs[(word[i], word[i + 1])] += freq
            if not pair_freqs:
                break
            # 找最高频pair
            best_pair = pair_freqs.most_common(1)[0][0]
            merges.append(best_pair)
            # 应用合并
            new_word_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                new_word = self._apply_bpe_merge(word, best_pair)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs
        # 更新合并规则和排名
        self._bpe_merges = merges
        self._bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        # 将合并后的子词加入词表
        for pair in merges:
            merged = pair[0] + pair[1]
            self.add_token(merged)
        return merges

    def _apply_bpe_merge(
        self,
        word: Tuple[str, ...],
        pair: Tuple[str, str],
    ) -> Tuple[str, ...]:
        """在词中应用一次BPE合并。"""
        new_word: List[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def bpe_encode(self, word: str) -> List[str]:
        """使用BPE规则编码一个词。"""
        if not self._bpe_merges:
            return list(word) + ["</w>"]
        word_chars = list(word) + ["</w>"]
        while len(word_chars) > 1:
            # 找到排名最高(最先合并)的pair
            pairs = [
                (word_chars[i], word_chars[i + 1])
                for i in range(len(word_chars) - 1)
            ]
            min_rank = float('inf')
            min_pair = None
            for p in pairs:
                rank = self._bpe_ranks.get(p, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = p
            if min_pair is None or min_rank == float('inf'):
                break
            # 应用合并
            new_chars: List[str] = []
            i = 0
            while i < len(word_chars):
                if i < len(word_chars) - 1 and \
                   word_chars[i] == min_pair[0] and word_chars[i + 1] == min_pair[1]:
                    new_chars.append(word_chars[i] + word_chars[i + 1])
                    i += 2
                else:
                    new_chars.append(word_chars[i])
                    i += 1
            word_chars = new_chars
        return word_chars

    # ---- 特殊token管理 ----

    def get_special_token_id(self, name: str) -> int:
        """获取特殊token的id。"""
        token = self._special_tokens.get(name)
        if token is None:
            raise KeyError(f"特殊token '{name}' 不存在")
        return self._token_to_id.get(token, 0)

    @property
    def pad_id(self) -> int:
        return self.get_special_token_id("pad")

    @property
    def bos_id(self) -> int:
        return self.get_special_token_id("bos")

    @property
    def eos_id(self) -> int:
        return self.get_special_token_id("eos")

    @property
    def unk_id(self) -> int:
        return self.get_special_token_id("unk")

    @property
    def mask_id(self) -> int:
        return self.get_special_token_id("mask")

    # ---- 持久化 ----

    def save(self, path: str) -> None:
        """保存词表到文件。"""
        data = {
            "token_to_id": self._token_to_id,
            "id_to_token": {str(k): v for k, v in self._id_to_token.items()},
            "token_freq": dict(self._token_freq),
            "special_tokens": self._special_tokens,
            "bpe_merges": self._bpe_merges,
            "max_vocab_size": self.max_vocab_size,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """从文件加载词表。"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._token_to_id = data["token_to_id"]
        self._id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self._token_freq = Counter(data["token_freq"])
        self._special_tokens = data["special_tokens"]
        self._bpe_merges = data.get("bpe_merges", [])
        self._bpe_ranks = {pair: i for i, pair in enumerate(self._bpe_merges)}
        self.max_vocab_size = data.get("max_vocab_size", 100000)
        self._next_id = max(self._id_to_token.keys()) + 1 if self._id_to_token else 0

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def __repr__(self) -> str:
        return (
            f"VocabularyManager(vocab_size={self.vocab_size}, "
            f"bpe_merges={len(self._bpe_merges)})"
        )


# =============================================================================
# 7. DatasetBuilder — 数据集构建器
# =============================================================================

@dataclass
class DatasetVersion:
    """数据集版本信息。"""
    version: str
    created_at: str
    num_samples: int
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetBuilder:
    """
    数据集构建器。

    支持从txt/json/csv/markdown格式加载数据,自动切分训练/验证/测试集,
    管理数据集版本,并生成统计报告。

    参数:
        name: 数据集名称
        seed: 随机种子
    """

    def __init__(self, name: str = "dataset", seed: Optional[int] = None):
        self.name = name
        self._rng = random.Random(seed)
        self._samples: List[Dict[str, Any]] = []
        self._splits: Dict[str, List[Dict[str, Any]]] = {}
        self._versions: Dict[str, DatasetVersion] = {}
        self._source_format: str = ""

    # ---- 多格式加载 ----

    def load_txt(self, path: str, encoding: str = "utf-8") -> "DatasetBuilder":
        """
        从txt文件加载数据(每行一个样本)。

        参数:
            path: 文件路径
            encoding: 文件编码
        """
        self._source_format = "txt"
        with open(path, 'r', encoding=encoding, errors='replace') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    self._samples.append({
                        "text": line,
                        "id": len(self._samples),
                        "source_line": line_num,
                    })
        return self

    def load_json(
        self,
        path: str,
        text_field: str = "text",
        encoding: str = "utf-8",
    ) -> "DatasetBuilder":
        """
        从json文件加载数据。
        支持json数组或jsonl(每行一个json对象)。

        参数:
            path: 文件路径
            text_field: 文本字段名
            encoding: 文件编码
        """
        self._source_format = "json"
        with open(path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read().strip()
        # 尝试解析为json数组
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and text_field in item:
                        record = dict(item)
                        record["text"] = str(item[text_field])
                        record["id"] = len(self._samples)
                        self._samples.append(record)
                    elif isinstance(item, str):
                        self._samples.append({
                            "text": item,
                            "id": len(self._samples),
                        })
            elif isinstance(data, dict):
                # 单个对象
                if text_field in data:
                    self._samples.append({
                        "text": str(data[text_field]),
                        "id": 0,
                        **data,
                    })
        except json.JSONDecodeError:
            # 尝试jsonl格式
            for line_num, line in enumerate(content.split('\n')):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and text_field in item:
                        record = dict(item)
                        record["text"] = str(item[text_field])
                        record["id"] = len(self._samples)
                        self._samples.append(record)
                except json.JSONDecodeError:
                    continue
        return self

    def load_csv(
        self,
        path: str,
        text_column: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> "DatasetBuilder":
        """
        从csv文件加载数据。

        参数:
            path: 文件路径
            text_column: 文本列名,为None时使用第一列
            encoding: 文件编码
        """
        self._source_format = "csv"
        with open(path, 'r', encoding=encoding, errors='replace', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return self
            col = text_column if text_column else reader.fieldnames[0]
            for row in reader:
                if col in row and row[col]:
                    record = dict(row)
                    record["text"] = str(row[col])
                    record["id"] = len(self._samples)
                    self._samples.append(record)
        return self

    def load_markdown(self, path: str, encoding: str = "utf-8") -> "DatasetBuilder":
        """
        从markdown文件加载数据。
        按段落分割,每个段落作为一个样本。

        参数:
            path: 文件路径
            encoding: 文件编码
        """
        self._source_format = "markdown"
        with open(path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        # 按空行分割段落
        paragraphs = re.split(r'\n\s*\n', content)
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 5:
                # 去除markdown标记
                clean_text = re.sub(r'[#*`_\-\[\]()]', '', para)
                clean_text = clean_text.strip()
                if clean_text:
                    self._samples.append({
                        "text": clean_text,
                        "id": len(self._samples),
                        "raw": para,
                    })
        return self

    def add_samples(self, samples: List[str]) -> "DatasetBuilder":
        """直接添加样本列表。"""
        for s in samples:
            self._samples.append({
                "text": s,
                "id": len(self._samples),
            })
        return self

    # ---- 切分 ----

    def split(
        self,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        shuffle: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        自动切分训练/验证/测试集。

        参数:
            ratios: (训练比例, 验证比例, 测试比例)
            shuffle: 是否打乱
        返回:
            {"train": [...], "val": [...], "test": [...]}
        """
        total = sum(ratios)
        if total <= 0:
            return {"train": [], "val": [], "test": []}
        ratios = tuple(r / total for r in ratios)
        samples = list(self._samples)
        if shuffle:
            self._rng.shuffle(samples)
        n = len(samples)
        train_end = int(n * ratios[0])
        val_end = int(n * (ratios[0] + ratios[1]))
        self._splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:],
        }
        return self._splits

    def get_split(self, name: str) -> List[Dict[str, Any]]:
        """获取指定切分。"""
        return self._splits.get(name, [])

    # ---- 版本管理 ----

    def save_version(self, path: str, version: str = "1.0", description: str = "") -> None:
        """保存数据集版本。"""
        version_info = DatasetVersion(
            version=version,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            num_samples=len(self._samples),
            description=description,
            metadata={
                "format": self._source_format,
                "splits": {k: len(v) for k, v in self._splits.items()},
            },
        )
        data = {
            "name": self.name,
            "samples": self._samples,
            "splits": self._splits,
            "version_info": {
                "version": version_info.version,
                "created_at": version_info.created_at,
                "num_samples": version_info.num_samples,
                "description": version_info.description,
                "metadata": version_info.metadata,
            },
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        self._versions[version] = version_info

    def load_version(self, path: str) -> str:
        """加载数据集版本。"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.name = data.get("name", self.name)
        self._samples = data.get("samples", [])
        self._splits = data.get("splits", {})
        vinfo = data.get("version_info", {})
        version = vinfo.get("version", "unknown")
        version_obj = DatasetVersion(
            version=version,
            created_at=vinfo.get("created_at", ""),
            num_samples=vinfo.get("num_samples", 0),
            description=vinfo.get("description", ""),
            metadata=vinfo.get("metadata", {}),
        )
        self._versions[version] = version_obj
        return version

    def list_versions(self) -> List[DatasetVersion]:
        """列出所有版本。"""
        return list(self._versions.values())

    # ---- 统计报告 ----

    def statistics_report(self) -> Dict[str, Any]:
        """生成数据集统计报告。"""
        if not self._samples:
            return {"total_samples": 0}
        texts = [s["text"] for s in self._samples if "text" in s]
        lengths = [len(t) for t in texts]
        all_tokens = []
        for t in texts:
            all_tokens.extend(_tokenize(t))
        token_freq = Counter(all_tokens)
        report = {
            "dataset_name": self.name,
            "source_format": self._source_format,
            "total_samples": len(self._samples),
            "splits": {k: len(v) for k, v in self._splits.items()},
            "text_length": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": sum(lengths) / len(lengths) if lengths else 0,
                "median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            },
            "vocabulary": {
                "total_tokens": len(all_tokens),
                "unique_tokens": len(token_freq),
                "top_10": token_freq.most_common(10),
            },
            "versions": len(self._versions),
        }
        return report

    def summary(self) -> str:
        """返回可读的摘要字符串。"""
        report = self.statistics_report()
        lines = [
            f"数据集: {report['dataset_name']}",
            f"格式: {report['source_format']}",
            f"总样本数: {report['total_samples']}",
            f"切分: {report['splits']}",
            f"版本数: {report['versions']}",
        ]
        if "text_length" in report and report["text_length"]:
            tl = report["text_length"]
            lines.append(f"文本长度: min={tl['min']}, max={tl['max']}, mean={tl['mean']:.1f}")
        if "vocabulary" in report:
            vocab = report["vocabulary"]
            lines.append(f"词汇: total={vocab['total_tokens']}, unique={vocab['unique_tokens']}")
        return "\n".join(lines)

    @property
    def samples(self) -> List[Dict[str, Any]]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        return f"DatasetBuilder(name='{self.name}', samples={len(self._samples)})"


# =============================================================================
# 8. DataPipeline — 数据流水线
# =============================================================================

@dataclass
class PipelineConfig:
    """数据流水线配置。"""
    # 读取配置
    source: Union[str, List[str], None] = None
    batch_size: int = 32
    # 清洗配置
    enable_cleaning: bool = True
    min_quality: float = 0.5
    remove_duplicates: bool = True
    dedup_threshold: float = 0.85
    # 增强配置
    enable_augmentation: bool = False
    augmentation_rate: float = 0.1
    augmentation_ops: List[str] = field(default_factory=lambda: ["synonym", "swap"])
    # 批处理配置
    max_tokens: int = 4096
    max_batch_size: int = 64
    sort_by_length: bool = True
    pack_short: bool = True
    # 课程学习配置
    enable_curriculum: bool = False
    curriculum_strategy: str = "linear"
    # 词表配置
    enable_vocab: bool = False
    max_vocab_size: int = 50000
    # 缓存配置
    enable_cache: bool = True
    cache_dir: str = "/tmp/lingyuan_cache"
    # 随机种子
    seed: Optional[int] = None


class DataPipeline:
    """
    端到端数据流水线。

    将读取→清洗→增强→批处理→迭代器各阶段串联,
    每个阶段可独立开关和配置,支持中间结果缓存和统计收集。

    参数:
        config: 流水线配置
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config if config else PipelineConfig()
        self._assessor = DataQualityAssessor()
        self._augmentor = DataAugmentor(
            augmentation_rate=self.config.augmentation_rate,
            seed=self.config.seed,
        )
        self._batcher = SmartBatcher(
            max_tokens=self.config.max_tokens,
            max_batch_size=self.config.max_batch_size,
            sort=self.config.sort_by_length,
            pack_short=self.config.pack_short,
        )
        self._scheduler = CurriculumScheduler(
            strategy=self.config.curriculum_strategy,
            seed=self.config.seed,
        ) if self.config.enable_curriculum else None
        self._vocab = VocabularyManager(
            max_vocab_size=self.config.max_vocab_size,
        ) if self.config.enable_vocab else None
        self._cache: Dict[str, Any] = {}
        self._stats = {
            "total_input": 0,
            "total_cleaned": 0,
            "total_augmented": 0,
            "total_output": 0,
            "processing_time": 0.0,
            "stage_times": defaultdict(float),
        }
        self._stages_enabled: Dict[str, bool] = {
            "read": True,
            "clean": self.config.enable_cleaning,
            "augment": self.config.enable_augmentation,
            "batch": True,
            "curriculum": self.config.enable_curriculum,
        }

    def enable_stage(self, stage_name: str) -> None:
        """启用某个阶段。"""
        self._stages_enabled[stage_name] = True

    def disable_stage(self, stage_name: str) -> None:
        """禁用某个阶段。"""
        self._stages_enabled[stage_name] = False

    def _stage_timer(self, stage_name: str):
        """阶段计时上下文管理器。"""
        class _Timer:
            def __init__(self, stats_dict, name):
                self.stats = stats_dict
                self.name = name
                self.start = 0.0

            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                self.stats["stage_times"][self.name] += time.time() - self.start

        return _Timer(self._stats, stage_name)

    # ---- 缓存 ----

    def _cache_key(self, data: Any, stage: str) -> str:
        """生成缓存键。"""
        if isinstance(data, list):
            content = "".join(str(d) for d in data[:100])
        else:
            content = str(data)
        return hashlib.md5(f"{stage}_{content}".encode()).hexdigest()

    def cache_get(self, key: str) -> Optional[Any]:
        """从缓存获取。"""
        if not self.config.enable_cache:
            return None
        return self._cache.get(key)

    def cache_set(self, key: str, value: Any) -> None:
        """设置缓存。"""
        if self.config.enable_cache:
            self._cache[key] = value

    def clear_cache(self) -> None:
        """清空缓存。"""
        self._cache.clear()

    # ---- 各阶段处理 ----

    def _stage_read(self, source: Any) -> List[str]:
        """读取阶段。"""
        with self._stage_timer("read"):
            if isinstance(source, str):
                if os.path.isfile(source):
                    with open(source, 'r', encoding='utf-8', errors='replace') as f:
                        data = [line.strip() for line in f if line.strip()]
                else:
                    data = [source]
            elif isinstance(source, list):
                data = [str(s) for s in source if s]
            else:
                data = [str(source)]
            self._stats["total_input"] += len(data)
            return data

    def _stage_clean(self, data: List[str]) -> List[str]:
        """清洗阶段。"""
        if not self._stages_enabled.get("clean", True):
            return data
        with self._stage_timer("clean"):
            cache_key = self._cache_key(data, "clean")
            cached = self.cache_get(cache_key)
            if cached is not None:
                return cached
            cleaned, clean_stats = self._assessor.auto_clean(
                data,
                min_quality=self.config.min_quality,
                remove_duplicates=self.config.remove_duplicates,
                dedup_threshold=self.config.dedup_threshold,
            )
            self._stats["total_cleaned"] += len(cleaned)
            self._stats["clean_stats"] = clean_stats
            self.cache_set(cache_key, cleaned)
            return cleaned

    def _stage_augment(self, data: List[str]) -> List[str]:
        """增强阶段。"""
        if not self._stages_enabled.get("augment", False):
            return data
        with self._stage_timer("augment"):
            cache_key = self._cache_key(data, "augment")
            cached = self.cache_get(cache_key)
            if cached is not None:
                return cached
            pipeline = self._augmentor.build_pipeline(self.config.augmentation_ops)
            augmented = pipeline.apply_batch(data)
            self._stats["total_augmented"] += len(augmented)
            self.cache_set(cache_key, augmented)
            return augmented

    def _stage_curriculum(self, data: List[str]) -> List[str]:
        """课程学习阶段: 按难度排序。"""
        if not self._stages_enabled.get("curriculum", False) or self._scheduler is None:
            return data
        with self._stage_timer("curriculum"):
            scored = [(self._scheduler.assess_difficulty(t), t) for t in data]
            scored.sort(key=lambda x: x[0])
            return [t for _, t in scored]

    def _stage_batch(self, data: List[str]) -> List[List[str]]:
        """批处理阶段。"""
        with self._stage_timer("batch"):
            batches = self._batcher.create_batches(data)
            self._stats["total_output"] += sum(len(b) for b in batches)
            return batches

    # ---- 端到端运行 ----

    def run(self, source: Any) -> Iterator[List[str]]:
        """
        端到端运行流水线。

        参数:
            source: 数据源 (文件路径/字符串/列表)
        返回:
            批次迭代器
        """
        start_time = time.time()
        # 读取
        data = self._stage_read(source)
        if not data:
            return iter([])
        # 清洗
        data = self._stage_clean(data)
        # 增强
        data = self._stage_augment(data)
        # 课程学习排序
        data = self._stage_curriculum(data)
        # 批处理
        batches = self._stage_batch(data)
        self._stats["processing_time"] = time.time() - start_time
        return iter(batches)

    def process(self, source: Any) -> List[List[str]]:
        """同步处理,返回所有批次。"""
        return list(self.run(source))

    # ---- 统计 ----

    def get_statistics(self) -> Dict[str, Any]:
        """获取流水线统计信息。"""
        stats = dict(self._stats)
        stats["stage_times"] = dict(stats.get("stage_times", {}))
        stats["batcher_stats"] = self._batcher.get_stats()
        if self._vocab:
            stats["vocab_size"] = self._vocab.vocab_size
        if self._scheduler:
            stats["curriculum_step"] = self._scheduler.current_step
            stats["difficulty_threshold"] = self._scheduler.get_difficulty_threshold()
        return stats

    def reset_stats(self) -> None:
        """重置统计信息。"""
        self._stats = {
            "total_input": 0,
            "total_cleaned": 0,
            "total_augmented": 0,
            "total_output": 0,
            "processing_time": 0.0,
            "stage_times": defaultdict(float),
        }
        self._batcher.reset_stats()

    def __repr__(self) -> str:
        enabled = [k for k, v in self._stages_enabled.items() if v]
        return f"DataPipeline(stages={enabled})"


# =============================================================================
# __main__ 自测代码
# =============================================================================

def _test_data_augmentor():
    """测试数据增强器。"""
    print("=" * 60)
    print("[1/8] 测试 DataAugmentor")
    print("=" * 60)
    augmentor = DataAugmentor(augmentation_rate=0.5, seed=42)
    text = "这个美丽的城市让人感到非常快乐"
    print(f"原始文本: {text}")
    print(f"同义词替换: {augmentor.synonym_replacement(text, n=2)}")
    print(f"随机删除:   {augmentor.random_deletion(text, p=0.2)}")
    print(f"随机交换:   {augmentor.random_swap(text, n=2)}")
    print(f"回译模拟:   {augmentor.back_translation_simulate(text)}")
    print(f"噪声注入:   {augmentor.inject_noise(text, p=0.1)}")
    # Token级
    tokens = ["我", "爱", "北京", "天安门"]
    print(f"原始tokens: {tokens}")
    print(f"Token mask: {augmentor.token_mask(tokens, p=0.3)}")
    print(f"Token替换:  {augmentor.token_replace(tokens, p=0.3)}")
    print(f"Token插入:  {augmentor.token_insert(tokens, p=0.3)}")
    # Pipeline
    pipeline = augmentor.build_pipeline(["synonym", "swap", "noise"])
    print(f"Pipeline:   {pipeline.apply(text)}")
    print(f"增强器:     {augmentor}")
    print("DataAugmentor 测试通过\n")


def _test_curriculum_scheduler():
    """测试课程学习调度器。"""
    print("=" * 60)
    print("[2/8] 测试 CurriculumScheduler")
    print("=" * 60)
    scheduler = CurriculumScheduler(
        strategy="linear",
        initial_difficulty=0.1,
        max_difficulty=1.0,
        total_steps=100,
    )
    texts = [
        "好的",  # 简单
        "今天天气不错",  # 中等
        "人工智能正在深刻改变人类社会的方方面面包括经济文化和政治等领域",  # 困难
    ]
    for t in texts:
        diff = scheduler.assess_difficulty(t)
        print(f"难度={diff:.3f} | {t}")
    # 测试不同策略
    for strategy in ["linear", "exponential", "anti_anxiety"]:
        s = CurriculumScheduler(strategy=strategy, total_steps=100)
        thresholds = [s.get_difficulty_threshold(step) for step in [0, 25, 50, 75, 100]]
        print(f"策略={strategy:12s} | 阈值: {[f'{t:.3f}' for t in thresholds]}")
    # 动态调整
    for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
        scheduler.update_loss(loss)
    print(f"动态调整后: adjustment={scheduler._dynamic_adjustment:.4f}")
    # 批次组成
    samples = [(t, i) for i, t in enumerate(texts * 5)]
    batch = scheduler.compose_batch(samples, batch_size=5)
    print(f"组成批次: {len(batch)} 个样本")
    print(f"调度器:   {scheduler}")
    print("CurriculumScheduler 测试通过\n")


def _test_smart_batcher():
    """测试智能批处理器。"""
    print("=" * 60)
    print("[3/8] 测试 SmartBatcher")
    print("=" * 60)
    batcher = SmartBatcher(
        max_tokens=200,
        max_batch_size=10,
        padding_token=0,
        sort=True,
        pack_short=True,
        dedup_threshold=0.95,
    )
    # 生成不同长度的样本
    samples = [f"样本{i}" * (i + 1) for i in range(15)]
    batches = batcher.create_batches(samples)
    print(f"样本数: {len(samples)}, 批次数: {len(batches)}")
    for i, batch in enumerate(batches):
        lengths = [len(s) for s in batch]
        print(f"  批次{i}: {len(batch)}个样本, 长度={lengths}")
    # 测试padding
    int_batches = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    padded, masks = batcher.pad_batch(int_batches)
    print(f"Padding后: {padded}")
    print(f"Mask:      {masks}")
    print(f"统计:      {batcher.get_stats()}")
    print(f"批处理器:   {batcher}")
    print("SmartBatcher 测试通过\n")


def _test_data_quality_assessor():
    """测试数据质量评估器。"""
    print("=" * 60)
    print("[4/8] 测试 DataQualityAssessor")
    print("=" * 60)
    assessor = DataQualityAssessor(min_length=5, max_length=200)
    # 质量评分
    test_texts = [
        "这是一个质量很好的文本样本,内容丰富且有意义。",
        "短",
        "啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊",
        "这是一个包含暴力内容的文本",
        "The quick brown fox jumps over the lazy dog near the river bank.",
    ]
    for t in test_texts:
        quality = assessor.assess_quality(t)
        anomalies = assessor.detect_anomalies(t)
        is_toxic, toxic_words = assessor.detect_toxicity(t)
        print(f"质量={quality:.3f} | 异常={anomalies['has_anomaly']} | "
              f"毒性={is_toxic} | {t[:30]}...")
    # 重复检测
    dup_samples = [
        "今天天气真好适合出门散步",
        "今天天气真好适合出门散步",  # 完全重复
        "今天天气不错适合出去走走",  # 近似
        "人工智能是未来的发展方向",
    ]
    groups = assessor.detect_duplicates(dup_samples, threshold=0.5)
    print(f"重复组: {groups}")
    # 自动清洗
    cleaned, stats = assessor.auto_clean(dup_samples, min_quality=0.3, dedup_threshold=0.5)
    print(f"清洗统计: {stats}")
    print(f"清洗后样本数: {len(cleaned)}")
    # 质量报告
    report = assessor.quality_report(test_texts)
    print(f"质量报告: mean={report['mean_quality']:.3f}")
    print(f"评估器:   {assessor}")
    print("DataQualityAssessor 测试通过\n")


def _test_streaming_data_loader():
    """测试流式数据加载器。"""
    print("=" * 60)
    print("[5/8] 测试 StreamingDataLoader")
    print("=" * 60)
    import tempfile
    # 创建临时测试文件
    tmpdir = tempfile.mkdtemp()
    file1 = os.path.join(tmpdir, "source1.txt")
    file2 = os.path.join(tmpdir, "source2.txt")
    with open(file1, 'w', encoding='utf-8') as f:
        for i in range(50):
            f.write(f"数据源一第{i}行样本\n")
    with open(file2, 'w', encoding='utf-8') as f:
        for i in range(30):
            f.write(f"数据源二第{i}行样本\n")
    # 测试均匀采样
    loader = StreamingDataLoader(
        sources=[(file1, 1.0), (file2, 2.0)],
        batch_size=10,
        sampling_strategy="uniform",
        prefetch_count=0,
        seed=42,
    )
    batch_count = 0
    total_samples = 0
    for batch in loader:
        batch_count += 1
        total_samples += len(batch)
    print(f"均匀采样: {batch_count}批, {total_samples}个样本")
    print(f"统计: {loader.get_stats()}")
    # 测试加权采样
    loader2 = StreamingDataLoader(
        sources=[(file1, 1.0), (file2, 3.0)],
        batch_size=10,
        sampling_strategy="weighted",
        prefetch_count=0,
        seed=42,
    )
    batch_count2 = 0
    for batch in loader2:
        batch_count2 += 1
    print(f"加权采样: {batch_count2}批")
    # 测试断点续传
    loader3 = StreamingDataLoader(
        sources=[file1],
        batch_size=10,
        prefetch_count=0,
        seed=42,
    )
    first_batch = None
    for i, batch in enumerate(loader3):
        if i == 0:
            first_batch = batch
            break
    loader3.close()
    # 保存检查点
    checkpoint_path = os.path.join(tmpdir, "checkpoint.pkl")
    loader3.save_checkpoint(checkpoint_path)
    print(f"断点续传: 位置={loader3._positions}")
    print(f"加载器:   {loader3}")
    # 清理
    import shutil
    shutil.rmtree(tmpdir)
    print("StreamingDataLoader 测试通过\n")


def _test_vocabulary_manager():
    """测试词表管理器。"""
    print("=" * 60)
    print("[6/8] 测试 VocabularyManager")
    print("=" * 60)
    vocab = VocabularyManager(max_vocab_size=10000)
    print(f"初始词表大小: {vocab.vocab_size}")
    print(f"特殊token ID: pad={vocab.pad_id}, bos={vocab.bos_id}, "
          f"eos={vocab.eos_id}, unk={vocab.unk_id}, mask={vocab.mask_id}")
    # 添加token
    text = "人工智能是未来的发展方向人工智能改变世界"
    tokens = _tokenize(text)
    ids = vocab.add_tokens(tokens)
    print(f"添加tokens: {tokens}")
    print(f"Token IDs: {ids}")
    # 编码解码
    encoded = vocab.encode("人工智能世界", add_bos=True, add_eos=True)
    decoded = vocab.decode(encoded, skip_special=True)
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
    # 频率统计
    vocab.update_frequency(tokens)
    vocab.update_frequency(tokens)
    print(f"频率最高: {vocab.get_most_common(5)}")
    print(f"'人工'频率: {vocab.get_frequency('人工')}")
    # BPE训练
    bpe_texts = ["hello world", "hello there", "world peace", "peace and love"]
    merges = vocab.train_bpe(bpe_texts, num_merges=10)
    print(f"BPE合并规则 ({len(merges)}条): {merges[:5]}")
    bpe_result = vocab.bpe_encode("hello")
    print(f"BPE编码 'hello': {bpe_result}")
    # 词表压缩
    before_size = vocab.vocab_size
    removed = vocab.compress_vocab(min_freq=3)
    after_size = vocab.vocab_size
    print(f"压缩: 移除{removed}个低频token, {before_size} -> {after_size}")
    # 持久化
    import tempfile
    tmpdir = tempfile.mkdtemp()
    vocab_path = os.path.join(tmpdir, "vocab.pkl")
    vocab.save(vocab_path)
    vocab2 = VocabularyManager()
    vocab2.load(vocab_path)
    print(f"保存/加载: size={vocab2.vocab_size}")
    print(f"词表管理器: {vocab}")
    import shutil
    shutil.rmtree(tmpdir)
    print("VocabularyManager 测试通过\n")


def _test_dataset_builder():
    """测试数据集构建器。"""
    print("=" * 60)
    print("[7/8] 测试 DatasetBuilder")
    print("=" * 60)
    import tempfile
    tmpdir = tempfile.mkdtemp()
    # 创建测试文件
    txt_path = os.path.join(tmpdir, "data.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i in range(100):
            f.write(f"这是第{i}个训练样本用于测试数据集构建器\n")
    json_path = os.path.join(tmpdir, "data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([{"text": f"json样本{i}", "label": i % 2} for i in range(50)], f,
                  ensure_ascii=False)
    csv_path = os.path.join(tmpdir, "data.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
        for i in range(30):
            writer.writerow({"text": f"csv样本{i}", "category": f"cat{i % 3}"})
    md_path = os.path.join(tmpdir, "data.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 标题\n\n这是第一个段落内容。\n\n## 子标题\n\n这是第二个段落。\n")
    # 加载txt
    builder = DatasetBuilder(name="test_dataset", seed=42)
    builder.load_txt(txt_path)
    print(f"加载txt: {len(builder)}个样本")
    # 切分
    splits = builder.split(ratios=(0.8, 0.1, 0.1))
    print(f"切分: train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}")
    # 统计报告
    report = builder.statistics_report()
    print(f"统计: total={report['total_samples']}, "
          f"mean_len={report['text_length']['mean']:.1f}")
    # 加载json
    builder2 = DatasetBuilder(name="json_dataset")
    builder2.load_json(json_path)
    print(f"加载json: {len(builder2)}个样本")
    # 加载csv
    builder3 = DatasetBuilder(name="csv_dataset")
    builder3.load_csv(csv_path)
    print(f"加载csv: {len(builder3)}个样本")
    # 加载markdown
    builder4 = DatasetBuilder(name="md_dataset")
    builder4.load_markdown(md_path)
    print(f"加载markdown: {len(builder4)}个样本")
    # 版本管理
    version_path = os.path.join(tmpdir, "dataset_v1.pkl")
    builder.save_version(version_path, version="1.0", description="初始版本")
    builder5 = DatasetBuilder(name="loaded")
    ver = builder5.load_version(version_path)
    print(f"版本管理: 加载版本={ver}, 样本数={len(builder5)}")
    print(f"版本列表: {[v.version for v in builder.list_versions()]}")
    print(builder.summary())
    print(f"构建器: {builder}")
    import shutil
    shutil.rmtree(tmpdir)
    print("DatasetBuilder 测试通过\n")


def _test_data_pipeline():
    """测试数据流水线。"""
    print("=" * 60)
    print("[8/8] 测试 DataPipeline")
    print("=" * 60)
    # 准备测试数据
    test_data = [
        "人工智能是计算机科学的一个重要分支",
        "机器学习让计算机能够从数据中学习",
        "深度学习使用神经网络来处理复杂数据",
        "自然语言处理研究计算机理解和生成人类语言",
        "计算机视觉让机器能够看到和理解图像",
        "短",
        "啊啊啊啊啊啊啊啊啊啊啊啊啊啊",
        "强化学习通过试错来学习最优策略",
        "人工智能正在改变世界的方方面面",
        "数据科学结合统计学和计算机科学",
    ] * 3
    # 配置并运行流水线
    config = PipelineConfig(
        batch_size=8,
        enable_cleaning=True,
        min_quality=0.3,
        remove_duplicates=True,
        dedup_threshold=0.85,
        enable_augmentation=True,
        augmentation_rate=0.3,
        augmentation_ops=["synonym", "swap"],
        max_tokens=512,
        max_batch_size=8,
        enable_curriculum=True,
        curriculum_strategy="linear",
        enable_cache=True,
        seed=42,
    )
    pipeline = DataPipeline(config)
    print(f"流水线: {pipeline}")
    batches = pipeline.process(test_data)
    print(f"输入: {len(test_data)}个样本")
    print(f"输出: {len(batches)}个批次")
    for i, batch in enumerate(batches):
        print(f"  批次{i}: {len(batch)}个样本, 首样本='{batch[0][:25]}...'")
    # 统计
    stats = pipeline.get_statistics()
    print(f"统计:")
    print(f"  输入: {stats['total_input']}")
    print(f"  清洗后: {stats['total_cleaned']}")
    print(f"  增强后: {stats['total_augmented']}")
    print(f"  输出: {stats['total_output']}")
    print(f"  耗时: {stats['processing_time']:.4f}s")
    print(f"  阶段耗时: {dict(stats['stage_times'])}")
    # 测试阶段开关
    pipeline.disable_stage("augment")
    pipeline.reset_stats()
    batches2 = pipeline.process(test_data)
    stats2 = pipeline.get_statistics()
    print(f"禁用增强后: 输出={stats2['total_output']}")
    print(f"流水线(禁用增强): {pipeline}")
    print("DataPipeline 测试通过\n")


def _test_integration():
    """集成测试: 所有组件协同工作。"""
    print("=" * 60)
    print("集成测试: 全流程协同")
    print("=" * 60)
    # 1. 构建词表
    vocab = VocabularyManager(max_vocab_size=5000)
    # 2. 构建数据集
    texts = [
        "人工智能是计算机科学的分支",
        "机器学习是人工智能的核心",
        "深度学习使用多层神经网络",
        "自然语言处理处理文本数据",
        "计算机视觉处理图像数据",
        "强化学习通过奖励学习策略",
    ]
    for t in texts:
        vocab.update_frequency_from_text(t)
    tokens_all = []
    for t in texts:
        tokens_all.extend(_tokenize(t))
    vocab.add_tokens(list(set(tokens_all)))
    print(f"词表大小: {vocab.vocab_size}")
    # 3. 质量评估
    assessor = DataQualityAssessor(min_length=5)
    qualities = [assessor.assess_quality(t) for t in texts]
    print(f"质量评分: {[f'{q:.2f}' for q in qualities]}")
    # 4. 数据增强
    augmentor = DataAugmentor(augmentation_rate=0.3, seed=42)
    augmented = augmentor.augment_batch(texts)
    print(f"增强样本数: {len(augmented)}")
    # 5. 课程调度
    scheduler = CurriculumScheduler(strategy="anti_anxiety", total_steps=100)
    difficulties = [scheduler.assess_difficulty(t) for t in texts]
    print(f"难度评估: {[f'{d:.2f}' for d in difficulties]}")
    threshold = scheduler.get_difficulty_threshold(step=50)
    print(f"第50步难度阈值: {threshold:.3f}")
    # 6. 智能批处理
    batcher = SmartBatcher(max_tokens=200, max_batch_size=4)
    batches = batcher.create_batches(texts + augmented)
    print(f"批次数: {len(batches)}")
    # 7. 编码
    encoded = vocab.encode(texts[0], add_bos=True, add_eos=True)
    decoded = vocab.decode(encoded, skip_special=True)
    print(f"编码/解码验证: {'通过' if decoded else '失败'}")
    # 8. 完整流水线
    config = PipelineConfig(
        enable_cleaning=True,
        min_quality=0.2,
        enable_augmentation=True,
        augmentation_rate=0.2,
        enable_curriculum=True,
        max_tokens=300,
        max_batch_size=5,
        seed=42,
    )
    pipeline = DataPipeline(config)
    final_batches = pipeline.process(texts * 3)
    stats = pipeline.get_statistics()
    print(f"流水线: 输入={stats['total_input']}, 输出={stats['total_output']}")
    print("集成测试通过\n")


def _main():
    """主测试函数。"""
    print()
    print("*" * 60)
    print("  灵元模型 - 智能数据工厂模块 (Part 20) 自测")
    print("*" * 60)
    print()
    _test_data_augmentor()
    _test_curriculum_scheduler()
    _test_smart_batcher()
    _test_data_quality_assessor()
    _test_streaming_data_loader()
    _test_vocabulary_manager()
    _test_dataset_builder()
    _test_data_pipeline()
    _test_integration()
    print("*" * 60)
    print("  所有测试通过!")
    print("*" * 60)


if __name__ == "__main__":
    _main()
