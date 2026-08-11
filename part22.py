#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 灵元模型项目 (LingYuan Model Project) — 第 22 模块
 分布式推理引擎 (Distributed Inference Engine)
================================================================================

项目说明:
    灵元模型项目是一个端到端的大语言模型训练与推理框架，涵盖从数据预处理、
    分词器、模型架构、分布式训练到推理部署的全栈实现。本文件为第 22 模块，
    聚焦于分布式推理引擎的完整实现。

模块概述:
    本模块实现了生产级分布式推理引擎的核心组件，包括:
      1. DistributedInferenceEngine    — 推理引擎核心 (单GPU/批量/流式/采样/KV缓存/统计)
      2. RequestScheduler   — 请求调度器 (优先级队列/连续批处理/超时/公平调度)
      3. ModelPartitioner   — 模型分区器 (层级分区/流水线/显存均衡/通信优化)
      4. KVCacheManager     — KV缓存管理器 (缓存池/LRU/压缩/共享/显存追踪)
      5. EdgeOptimizer      — 边缘部署优化器 (模型分析/设备匹配/量化/退化策略)
      6. LoadBalancer       — 负载均衡器 (健康检查/路由/自动扩缩容/故障转移)
      7. InferenceServer    — 推理服务器 (HTTP API模拟/SSE流式/并发控制/指标)
      8. PerformanceMonitor — 性能监控器 (QPS/延迟分位/资源监控/告警/仪表板)

设计原则:
    - 纯 Python 标准库实现，零外部依赖
    - 模拟推理通过内部函数实现，不依赖其他 part 文件
    - 所有类可独立实例化和运行
    - 线程安全设计 (使用 threading.Lock 保护共享状态)
    - 完整的统计与监控能力

作者: 灵元模型项目组
版本: 1.0.0
================================================================================
"""

import os
import sys
import time
import math
import json
import heapq
import random
import hashlib
import threading
import queue
import logging
from collections import deque, defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, Iterator,
    Generator, Set, NamedTuple
)

# ============================================================================
# 日志配置
# ============================================================================

logger = logging.getLogger("lingyuan.inference")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING)


# ============================================================================
# 枚举与常量
# ============================================================================

class Priority(IntEnum):
    """请求优先级枚举。数值越小优先级越高。"""
    HIGH = 0
    MEDIUM = 1
    LOW = 2


class SamplingStrategy(Enum):
    """采样策略枚举。"""
    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    TOP_P = "top_nucleus"


class RequestStatus(Enum):
    """请求状态枚举。"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class PartitionStrategy(Enum):
    """模型分区策略枚举。"""
    UNIFORM = "uniform"
    BY_LAYER_SIZE = "by_layer_size"
    BY_COMPUTE = "by_compute"


class QuantizationLevel(Enum):
    """量化级别枚举。"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class NodeStatus(Enum):
    """推理节点状态枚举。"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class DeviceType(Enum):
    """设备类型枚举。"""
    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"
    EDGE = "edge"


# 模拟词表大小
_MOCK_VOCAB_SIZE = 32000
# 模拟词表: token id -> 字符串
_MOCK_VOCAB = {i: f"<tok_{i}>" for i in range(_MOCK_VOCAB_SIZE)}
_MOCK_TOKEN_TEXT = [
    "the", "model", "is", "a", "powerful", "language", "system", "that",
    "can", "generate", "coherent", "text", "and", "answer", "questions",
    "with", "high", "quality", "responses", "for", "various", "tasks",
]


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class InferenceRequest:
    """推理请求数据结构。"""
    request_id: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    strategy: SamplingStrategy = SamplingStrategy.GREEDY
    priority: Priority = Priority.MEDIUM
    stream: bool = False
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None  # 超时时间戳
    status: RequestStatus = RequestStatus.PENDING
    prompt_token_ids: List[int] = field(default_factory=list)
    output_token_ids: List[int] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _arrival_order: int = 0  # 用于公平调度排序

    @property
    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


@dataclass
class InferenceResult:
    """推理结果数据结构。"""
    request_id: str
    text: str
    token_ids: List[int]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    finish_reason: str = "stop"
    error: Optional[str] = None


@dataclass
class InferenceStats:
    """推理统计信息。"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    total_latency_ms: float = 0.0
    batch_count: int = 0
    _latencies: List[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def throughput_tps(self) -> float:
        """每秒生成的 token 数。"""
        if self.total_latency_ms <= 0:
            return 0.0
        return self.total_tokens_generated / (self.total_latency_ms / 1000.0)

    @property
    def qps(self) -> float:
        if self.total_latency_ms <= 0:
            return 0.0
        return self.completed_requests / (self.total_latency_ms / 1000.0)

    def percentile(self, p: float) -> float:
        """计算延迟分位数。"""
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        idx = max(0, min(len(sorted_lat) - 1, int(math.ceil(p / 100 * len(sorted_lat))) - 1))
        return sorted_lat[idx]

    def record(self, latency_ms: float, tokens: int, prompt_tokens: int = 0):
        self._latencies.append(latency_ms)
        self.total_latency_ms += latency_ms
        self.total_tokens_generated += tokens
        self.total_prompt_tokens += prompt_tokens

    def summary(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "completed": self.completed_requests,
            "failed": self.failed_requests,
            "tokens_generated": self.total_tokens_generated,
            "prompt_tokens": self.total_prompt_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_ms": round(self.percentile(50), 2),
            "p95_ms": round(self.percentile(95), 2),
            "p99_ms": round(self.percentile(99), 2),
            "throughput_tps": round(self.throughput_tps, 2),
            "qps": round(self.qps, 2),
            "batch_count": self.batch_count,
        }


@dataclass
class CacheEntry:
    """KV缓存条目。"""
    key: str
    token_ids: List[int]
    kv_data: List[Any]  # 模拟 KV 张量数据
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    ref_count: int = 1  # 引用计数，用于缓存共享

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class LayerInfo:
    """Transformer 层信息。"""
    layer_id: int
    name: str
    param_count: int
    flops_per_token: float  # 每个 token 的计算量
    memory_bytes: int  # 显存占用
    kv_cache_bytes_per_token: int = 0  # 每 token 的 KV 缓存大小


@dataclass
class DeviceInfo:
    """设备信息。"""
    device_id: str
    device_type: DeviceType
    total_memory_bytes: int
    compute_flops: float  # 算力 (FLOPS)
    bandwidth_bytes_per_sec: float = 0
    quantization_support: List[QuantizationLevel] = field(default_factory=list)
    status: NodeStatus = NodeStatus.HEALTHY

    @property
    def available(self) -> bool:
        return self.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED)


@dataclass
class PartitionPlan:
    """分区方案。"""
    strategy: PartitionStrategy
    assignments: List[Tuple[int, str]]  # (layer_id, device_id)
    device_memory_usage: Dict[str, int] = field(default_factory=dict)
    estimated_comm_bytes: int = 0
    pipeline_stages: int = 1
    balance_score: float = 0.0


@dataclass
class InferenceNode:
    """推理节点。"""
    node_id: str
    endpoint: str
    device_info: DeviceInfo
    status: NodeStatus = NodeStatus.HEALTHY
    active_requests: int = 0
    queue_length: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    _latency_history: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_latency(self, latency_ms: float):
        self._latency_history.append(latency_ms)
        if self._latency_history:
            self.avg_latency_ms = sum(self._latency_history) / len(self._latency_history)

    @property
    def load_score(self) -> float:
        """负载分数，越高表示越忙。"""
        return self.active_requests * 0.5 + self.queue_length * 0.3 + self.avg_latency_ms * 0.002

    @property
    def is_available(self) -> bool:
        return self.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED)


# ============================================================================
# 模拟推理函数 (不依赖其他 part 文件)
# ============================================================================

def _mock_tokenize(text: str) -> List[int]:
    """模拟分词器: 将文本转为 token id 列表。"""
    if not text:
        return []
    # 基于 hash 的确定性分词
    tokens = []
    words = text.split()
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        tokens.append(h % _MOCK_VOCAB_SIZE)
    if not tokens:
        tokens = [hash(text) % _MOCK_VOCAB_SIZE]
    return tokens


def _mock_detokenize(token_ids: List[int]) -> str:
    """模拟反分词器: 将 token id 列表转为文本。"""
    parts = []
    for i, tid in enumerate(token_ids):
        if tid < len(_MOCK_TOKEN_TEXT):
            parts.append(_MOCK_TOKEN_TEXT[tid])
        else:
            parts.append(_MOCK_VOCAB.get(tid, f"<tok_{tid}>"))
    return " ".join(parts)


def _mock_forward_pass(
    input_ids: List[int],
    kv_cache: Optional[List[Any]] = None,
    num_layers: int = 12,
    hidden_size: int = 768
) -> Tuple[List[float], Optional[List[Any]]]:
    """
    模拟 Transformer 前向传播。
    返回 (logits, kv_cache)。
    logtis 为词表大小的概率分布 (未归一化)。
    """
    vocab = _MOCK_VOCAB_SIZE
    # 基于输入生成确定性的 logits
    seed = sum(input_ids) % vocab if input_ids else 0
    logits = [0.0] * vocab
    for i in range(vocab):
        # 确定性伪随机 logit
        val = math.sin(seed * 0.001 + i * 0.017) + math.cos((seed + i) * 0.003)
        logits[i] = val
    # 增强最近 token 的影响
    if input_ids:
        last_token = input_ids[-1] % vocab
        logits[last_token] += 2.0
        # 鼓励生成一些常见 token
        for idx in range(min(22, vocab)):
            logits[idx] += 1.0

    # 模拟 KV cache 更新
    if kv_cache is None:
        kv_cache = [[] for _ in range(num_layers)]
    for layer in range(num_layers):
        kv_cache[layer].append(input_ids[-1] if input_ids else 0)

    return logits, kv_cache


def _softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    """数值稳定的 softmax。"""
    if temperature <= 0:
        temperature = 1e-8
    scaled = [l / temperature for l in logits]
    max_val = max(scaled) if scaled else 0.0
    exps = [math.exp(l - max_val) for l in scaled]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits) if logits else []
    return [e / total for e in exps]


# ============================================================================
# KVCacheManager — KV缓存管理器
# ============================================================================

class KVCacheManager:
    """
    KV缓存管理器 (推理专用)。
    管理推理过程中的 Key-Value 缓存，支持缓存池预分配、LRU淘汰、
    缓存压缩、prefix共享和实时显存追踪。
    """

    def __init__(
        self,
        pool_size_bytes: int = 1024 * 1024 * 1024,  # 1GB
        max_entries: int = 256,
        num_layers: int = 12,
        hidden_size: int = 768,
        compression_threshold: int = 300,  # 秒，超过此时间不活跃则压缩
        eviction_threshold: float = 0.9,  # 显存使用率超过此值开始淘汰
    ):
        self._pool_size = pool_size_bytes
        self._max_entries = max_entries
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._compression_threshold = compression_threshold
        self._eviction_threshold = eviction_threshold
        self._lock = threading.RLock()

        # 缓存存储: key -> CacheEntry
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        # prefix 索引: prefix_hash -> set of cache keys (用于共享)
        self._prefix_index: Dict[str, Set[str]] = defaultdict(set)
        # 显存追踪
        self._used_bytes = 0
        self._compressed_bytes = 0

        # 统计
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "shared_count": 0,
            "allocations": 0,
        }

    # ---- 属性 ----

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    @property
    def free_bytes(self) -> int:
        with self._lock:
            return self._pool_size - self._used_bytes

    @property
    def usage_ratio(self) -> float:
        with self._lock:
            if self._pool_size == 0:
                return 0.0
            return self._used_bytes / self._pool_size

    @property
    def num_entries(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "used_bytes": self._used_bytes,
                "free_bytes": self.free_bytes,
                "usage_ratio": round(self.usage_ratio, 4),
                "num_entries": len(self._cache),
                "compressed_entries": sum(1 for e in self._cache.values() if e.compressed),
                "hit_rate": (
                    self._stats["hits"] /
                    max(1, self._stats["hits"] + self._stats["misses"])
                ),
            }

    # ---- 缓存大小估算 ----

    def _estimate_cache_size(self, num_tokens: int) -> int:
        """估算 KV 缓存大小 (字节)。"""
        # KV cache: 2 (K+V) * num_layers * num_tokens * hidden_size * 2 bytes (fp16)
        return 2 * self._num_layers * num_tokens * self._hidden_size * 2

    def _prefix_hash(self, token_ids: List[int], prefix_len: int) -> str:
        """计算 token prefix 的哈希。"""
        prefix = tuple(token_ids[:prefix_len])
        return hashlib.md5(str(prefix).encode()).hexdigest()

    # ---- 核心操作 ----

    def get(self, key: str) -> Optional[CacheEntry]:
        """获取缓存条目。"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            entry.touch()
            # 移动到末尾 (LRU)
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return entry

    def find_shared(self, token_ids: List[int]) -> Optional[CacheEntry]:
        """查找可共享的 prefix 缓存。"""
        with self._lock:
            if not token_ids:
                return None
            # 尝试不同长度的 prefix
            for prefix_len in range(len(token_ids), 0, -1):
                ph = self._prefix_hash(token_ids, prefix_len)
                keys = self._prefix_index.get(ph, set())
                for ck in keys:
                    entry = self._cache.get(ck)
                    if entry is not None:
                        # 验证 prefix 匹配
                        cached = entry.token_ids[:prefix_len]
                        if cached == token_ids[:prefix_len]:
                            self._stats["shared_count"] += 1
                            return entry
            return None

    def put(
        self,
        key: str,
        token_ids: List[int],
        kv_data: Optional[List[Any]] = None,
        shareable: bool = True,
    ) -> CacheEntry:
        """存入缓存条目。"""
        with self._lock:
            # 如果已存在，更新
            if key in self._cache:
                entry = self._cache[key]
                entry.touch()
                entry.ref_count += 1
                self._cache.move_to_end(key)
                return entry

            size = self._estimate_cache_size(len(token_ids))
            # 检查是否需要淘汰
            while (self._used_bytes + size > self._pool_size or
                   len(self._cache) >= self._max_entries):
                if not self._evict_one():
                    break

            if kv_data is None:
                kv_data = [list(token_ids) for _ in range(self._num_layers)]

            entry = CacheEntry(
                key=key,
                token_ids=list(token_ids),
                kv_data=kv_data,
                size_bytes=size,
            )
            self._cache[key] = entry
            self._used_bytes += size
            self._stats["allocations"] += 1

            # 建立 prefix 索引
            if shareable and token_ids:
                for prefix_len in [len(token_ids), max(1, len(token_ids) // 2)]:
                    ph = self._prefix_hash(token_ids, prefix_len)
                    self._prefix_index[ph].add(key)

            return entry

    def _evict_one(self) -> bool:
        """淘汰一个最久未使用的缓存条目 (LRU)。"""
        if not self._cache:
            return False
        # 优先淘汰压缩过的、引用计数低的
        key_to_evict = None
        for k, entry in self._cache.items():
            if entry.ref_count <= 1:
                key_to_evict = k
                break
        if key_to_evict is None:
            key_to_evict = next(iter(self._cache))

        entry = self._cache.pop(key_to_evict)
        self._used_bytes -= entry.size_bytes
        self._stats["evictions"] += 1
        # 清理 prefix 索引
        if entry.token_ids:
            for prefix_len in [len(entry.token_ids), max(1, len(entry.token_ids) // 2)]:
                ph = self._prefix_hash(entry.token_ids, prefix_len)
                if ph in self._prefix_index:
                    self._prefix_index[ph].discard(key_to_evict)
                    if not self._prefix_index[ph]:
                        del self._prefix_index[ph]
        return True

    def compress_inactive(self, current_time: Optional[float] = None) -> int:
        """压缩不活跃的缓存条目。返回压缩数量。"""
        with self._lock:
            if current_time is None:
                current_time = time.time()
            count = 0
            for entry in self._cache.values():
                if (not entry.compressed and
                    current_time - entry.last_accessed > self._compression_threshold):
                    # 压缩: 大小减半
                    old_size = entry.size_bytes
                    entry.size_bytes = old_size // 2
                    self._used_bytes -= (old_size - entry.size_bytes)
                    self._compressed_bytes += entry.size_bytes
                    entry.compressed = True
                    count += 1
                    self._stats["compressions"] += 1
            return count

    def release(self, key: str):
        """释放缓存引用。"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return
            entry.ref_count -= 1
            if entry.ref_count <= 0:
                self._cache.pop(key)
                self._used_bytes -= entry.size_bytes

    def invalidate(self, key: str):
        """使缓存条目失效。"""
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry:
                self._used_bytes -= entry.size_bytes

    def clear(self):
        """清空所有缓存。"""
        with self._lock:
            self._cache.clear()
            self._prefix_index.clear()
            self._used_bytes = 0
            self._compressed_bytes = 0

    def get_memory_report(self) -> Dict[str, Any]:
        """获取显存使用报告。"""
        with self._lock:
            return {
                "pool_size_bytes": self._pool_size,
                "used_bytes": self._used_bytes,
                "free_bytes": self.free_bytes,
                "usage_ratio": round(self.usage_ratio, 4),
                "num_entries": len(self._cache),
                "compressed_bytes": self._compressed_bytes,
                "stats": dict(self._stats),
            }


# ============================================================================
# DistributedInferenceEngine — 推理引擎核心
# ============================================================================

class DistributedInferenceEngine:
    """
    推理引擎核心。
    支持单GPU推理、批量推理、流式推理、多种采样策略、KV缓存管理和延迟/吞吐统计。
    """

    def __init__(
        self,
        model_name: str = "lingyuan-base",
        num_layers: int = 12,
        hidden_size: int = 768,
        vocab_size: int = _MOCK_VOCAB_SIZE,
        max_batch_size: int = 32,
        cache_manager: Optional[KVCacheManager] = None,
        device_id: str = "gpu-0",
        simulate_latency: bool = True,
        base_latency_ms: float = 1.0,
    ):
        self.model_name = model_name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size
        self.device_id = device_id
        self.simulate_latency = simulate_latency
        self.base_latency_ms = base_latency_ms

        self.cache_manager = cache_manager or KVCacheManager(
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
        self._lock = threading.RLock()
        self._stats = InferenceStats()
        self._running = True
        self._request_counter = 0

    # ---- 属性 ----

    @property
    def stats(self) -> InferenceStats:
        return self._stats

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False

    def _gen_request_id(self) -> str:
        with self._lock:
            self._request_counter += 1
            return f"req-{self._request_counter}"

    def _simulate_delay(self, num_tokens: int = 1):
        """模拟推理延迟。"""
        if self.simulate_latency:
            time.sleep(self.base_latency_ms * num_tokens / 1000.0)

    # ---- 采样策略 ----

    def _sample(
        self,
        logits: List[float],
        strategy: SamplingStrategy,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> int:
        """根据策略采样 token。"""
        if strategy == SamplingStrategy.GREEDY or temperature <= 0:
            return max(range(len(logits)), key=lambda i: logits[i])

        # 应用温度
        probs = _softmax(logits, temperature)

        if strategy == SamplingStrategy.TEMPERATURE:
            return self._sample_from_probs(probs)

        if strategy == SamplingStrategy.TOP_K:
            return self._sample_top_k(logits, probs, top_k)

        if strategy == SamplingStrategy.TOP_P:
            return self._sample_top_p(probs, top_p)

        # 默认 greedy
        return max(range(len(logits)), key=lambda i: logits[i])

    def _sample_from_probs(self, probs: List[float]) -> int:
        """从概率分布中采样。"""
        r = random.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if cum >= r:
                return i
        return len(probs) - 1

    def _sample_top_k(self, logits: List[float], probs: List[float], k: int) -> int:
        """Top-K 采样。"""
        k = min(k, len(probs))
        # 获取 top-k 的索引
        top_indices = sorted(range(len(probs)), key=lambda i: logits[i], reverse=True)[:k]
        top_probs = [probs[i] for i in top_indices]
        total = sum(top_probs)
        if total <= 0:
            return random.choice(top_indices)
        top_probs = [p / total for p in top_probs]
        r = random.random()
        cum = 0.0
        for idx, p in zip(top_indices, top_probs):
            cum += p
            if cum >= r:
                return idx
        return top_indices[-1]

    def _sample_top_p(self, probs: List[float], p: float) -> int:
        """Top-P (nucleus) 采样。"""
        # 按概率降序排列
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        cum = 0.0
        nucleus = []
        for idx in sorted_indices:
            nucleus.append(idx)
            cum += probs[idx]
            if cum >= p:
                break
        if not nucleus:
            return sorted_indices[0] if sorted_indices else 0
        nucleus_probs = [probs[i] for i in nucleus]
        total = sum(nucleus_probs)
        if total <= 0:
            return random.choice(nucleus)
        nucleus_probs = [pr / total for pr in nucleus_probs]
        r = random.random()
        cum = 0.0
        for idx, pr in zip(nucleus, nucleus_probs):
            cum += pr
            if cum >= r:
                return idx
        return nucleus[-1]

    # ---- 单序列推理 ----

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        strategy: SamplingStrategy = SamplingStrategy.GREEDY,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        request_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> InferenceResult:
        """单序列推理 (非流式)。"""
        if request_id is None:
            request_id = self._gen_request_id()

        start_time = time.time()
        prompt_ids = _mock_tokenize(prompt)

        with self._lock:
            self._stats.total_requests += 1

        # 检查/创建 KV 缓存
        cache_key = f"{request_id}_kv"
        kv_cache = None
        if use_cache:
            # 尝试共享 prefix
            shared = self.cache_manager.find_shared(prompt_ids)
            if shared is not None:
                kv_cache = list(shared.kv_data)
            entry = self.cache_manager.put(cache_key, prompt_ids, kv_data=kv_cache)
            kv_cache = entry.kv_data

        output_ids: List[int] = []
        current_ids = list(prompt_ids)

        for step in range(max_tokens):
            self._simulate_delay(1)
            logits, kv_cache = _mock_forward_pass(
                current_ids, kv_cache=kv_cache,
                num_layers=self.num_layers, hidden_size=self.hidden_size,
            )
            next_token = self._sample(
                logits, strategy, temperature, top_k, top_p
            )
            output_ids.append(next_token)
            current_ids.append(next_token)

            # 更新缓存
            if use_cache:
                self.cache_manager.put(cache_key, current_ids, kv_data=kv_cache)

            # 简单的停止条件
            if next_token == 0 or len(output_ids) >= max_tokens:
                break

        latency_ms = (time.time() - start_time) * 1000
        text = _mock_detokenize(output_ids)

        with self._lock:
            self._stats.completed_requests += 1
            self._stats.record(latency_ms, len(output_ids), len(prompt_ids))

        if use_cache:
            self.cache_manager.release(cache_key)

        return InferenceResult(
            request_id=request_id,
            text=text,
            token_ids=output_ids,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(output_ids),
            latency_ms=latency_ms,
        )

    # ---- 流式推理 ----

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        strategy: SamplingStrategy = SamplingStrategy.GREEDY,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        request_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式推理: token-by-token 生成 (生成器)。"""
        if request_id is None:
            request_id = self._gen_request_id()

        start_time = time.time()
        prompt_ids = _mock_tokenize(prompt)

        with self._lock:
            self._stats.total_requests += 1

        cache_key = f"{request_id}_kv_stream"
        kv_cache = None
        if use_cache:
            shared = self.cache_manager.find_shared(prompt_ids)
            if shared is not None:
                kv_cache = list(shared.kv_data)
            entry = self.cache_manager.put(cache_key, prompt_ids, kv_data=kv_cache)
            kv_cache = entry.kv_data

        output_ids: List[int] = []
        current_ids = list(prompt_ids)

        # 首先发送 prompt 处理事件
        yield {
            "request_id": request_id,
            "type": "prompt",
            "prompt_tokens": len(prompt_ids),
            "token_id": None,
            "text": "",
        }

        for step in range(max_tokens):
            self._simulate_delay(1)
            logits, kv_cache = _mock_forward_pass(
                current_ids, kv_cache=kv_cache,
                num_layers=self.num_layers, hidden_size=self.hidden_size,
            )
            next_token = self._sample(logits, strategy, temperature, top_k, top_p)
            output_ids.append(next_token)
            current_ids.append(next_token)

            if use_cache:
                self.cache_manager.put(cache_key, current_ids, kv_data=kv_cache)

            token_text = _mock_detokenize([next_token])
            yield {
                "request_id": request_id,
                "type": "token",
                "step": step,
                "token_id": next_token,
                "text": token_text,
            }

            if next_token == 0 or len(output_ids) >= max_tokens:
                break

        latency_ms = (time.time() - start_time) * 1000

        with self._lock:
            self._stats.completed_requests += 1
            self._stats.record(latency_ms, len(output_ids), len(prompt_ids))

        if use_cache:
            self.cache_manager.release(cache_key)

        yield {
            "request_id": request_id,
            "type": "done",
            "completion_tokens": len(output_ids),
            "latency_ms": latency_ms,
            "finish_reason": "stop" if len(output_ids) < max_tokens else "length",
        }

    # ---- 批量推理 ----

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        strategy: SamplingStrategy = SamplingStrategy.GREEDY,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_cache: bool = True,
    ) -> List[InferenceResult]:
        """批量推理: 多请求并行处理 (模拟)。"""
        start_time = time.time()
        results: List[InferenceResult] = []
        batch_size = len(prompts)

        if batch_size > self.max_batch_size:
            # 分批处理
            for i in range(0, batch_size, self.max_batch_size):
                batch = prompts[i:i + self.max_batch_size]
                results.extend(self.generate_batch(
                    batch, max_tokens, strategy, temperature, top_k, top_p, use_cache
                ))
            return results

        with self._lock:
            self._stats.total_requests += batch_size
            self._stats.batch_count += 1

        # 模拟批量前向传播
        all_prompt_ids = [_mock_tokenize(p) for p in prompts]
        all_output_ids: List[List[int]] = [[] for _ in range(batch_size)]
        all_current_ids = [list(pids) for pids in all_prompt_ids]

        # 批量生成
        for step in range(max_tokens):
            self._simulate_delay(1)
            all_done = True
            for i in range(batch_size):
                if len(all_output_ids[i]) >= max_tokens:
                    continue
                all_done = False
                logits, _ = _mock_forward_pass(
                    all_current_ids[i], kv_cache=None,
                    num_layers=self.num_layers, hidden_size=self.hidden_size,
                )
                next_token = self._sample(logits, strategy, temperature, top_k, top_p)
                all_output_ids[i].append(next_token)
                all_current_ids[i].append(next_token)
            if all_done:
                break

        latency_ms = (time.time() - start_time) * 1000

        for i in range(batch_size):
            text = _mock_detokenize(all_output_ids[i])
            results.append(InferenceResult(
                request_id=self._gen_request_id(),
                text=text,
                token_ids=all_output_ids[i],
                prompt_tokens=len(all_prompt_ids[i]),
                completion_tokens=len(all_output_ids[i]),
                latency_ms=latency_ms / batch_size,
            ))
            with self._lock:
                self._stats.completed_requests += 1
                self._stats.record(
                    latency_ms / batch_size,
                    len(all_output_ids[i]),
                    len(all_prompt_ids[i]),
                )

        return results

    # ---- 前向传播 (低级接口) ----

    def forward(
        self,
        input_ids: List[int],
        kv_cache: Optional[List[Any]] = None,
    ) -> Tuple[List[float], Optional[List[Any]]]:
        """单次前向传播 (低级接口)。"""
        self._simulate_delay(1)
        return _mock_forward_pass(
            input_ids, kv_cache=kv_cache,
            num_layers=self.num_layers, hidden_size=self.hidden_size,
        )

    # ---- 统计 ----

    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计信息。"""
        with self._lock:
            stats = self._stats.summary()
            stats["cache"] = self.cache_manager.stats
            stats["model_name"] = self.model_name
            stats["device_id"] = self.device_id
            stats["max_batch_size"] = self.max_batch_size
            return stats

    def reset_stats(self):
        """重置统计信息。"""
        with self._lock:
            self._stats = InferenceStats()


# ============================================================================
# RequestScheduler — 请求调度器
# ============================================================================

class RequestScheduler:
    """
    请求调度器。
    支持优先级队列管理、连续批处理 (continuous batching)、请求优先级、
    超时处理和公平调度 (防止大请求饿死小请求)。
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_queue_size: int = 1000,
        timeout_seconds: float = 30.0,
        fairness_window: int = 10,  # 公平调度窗口: 连续处理 N 个高优先级后强制处理低优先级
        schedule_interval_ms: float = 10.0,
    ):
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.timeout_seconds = timeout_seconds
        self.fairness_window = fairness_window
        self.schedule_interval_ms = schedule_interval_ms

        self._lock = threading.RLock()
        # 优先级队列: (priority, arrival_order, request)
        self._heap: List[Tuple[int, int, InferenceRequest]] = []
        self._arrival_counter = 0
        self._request_map: Dict[str, InferenceRequest] = {}

        # 公平调度追踪
        self._priority_counts: Dict[Priority, int] = {p: 0 for p in Priority}
        self._total_scheduled = 0

        # 统计
        self._stats = {
            "total_submitted": 0,
            "total_scheduled": 0,
            "total_expired": 0,
            "total_cancelled": 0,
            "batches_formed": 0,
            "avg_batch_size": 0.0,
            "max_queue_depth": 0,
        }
        self._batch_sizes: deque = deque(maxlen=100)

        self._running = True

    # ---- 属性 ----

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_bs = (sum(self._batch_sizes) / len(self._batch_sizes)) if self._batch_sizes else 0.0
            return {
                **self._stats,
                "avg_batch_size": round(avg_bs, 2),
                "current_queue_size": len(self._heap),
                "priority_counts": dict(self._priority_counts),
            }

    # ---- 请求管理 ----

    def submit(self, request: InferenceRequest) -> bool:
        """提交请求到调度器。返回是否成功入队。"""
        with self._lock:
            if len(self._heap) >= self.max_queue_size:
                logger.warning(f"Scheduler queue full, rejecting request {request.request_id}")
                return False

            if request.deadline is None:
                request.deadline = time.time() + self.timeout_seconds

            self._arrival_counter += 1
            request._arrival_order = self._arrival_counter
            request.status = RequestStatus.QUEUED

            heapq.heappush(self._heap, (request.priority, request._arrival_order, request))
            self._request_map[request.request_id] = request
            self._stats["total_submitted"] += 1
            self._stats["max_queue_depth"] = max(
                self._stats["max_queue_depth"], len(self._heap)
            )
            return True

    def submit_batch(self, requests: List[InferenceRequest]) -> int:
        """批量提交请求。返回成功入队数。"""
        count = 0
        for req in requests:
            if self.submit(req):
                count += 1
        return count

    def cancel(self, request_id: str) -> bool:
        """取消请求。"""
        with self._lock:
            req = self._request_map.get(request_id)
            if req is None:
                return False
            req.status = RequestStatus.CANCELLED
            self._stats["total_cancelled"] += 1
            # 从堆中移除 (标记为取消，下次调度时跳过)
            return True

    def _is_valid(self, request: InferenceRequest) -> bool:
        """检查请求是否有效 (未过期、未取消)。"""
        if request.status in (RequestStatus.CANCELLED, RequestStatus.COMPLETED,
                              RequestStatus.FAILED, RequestStatus.TIMEOUT):
            return False
        if request.is_expired:
            request.status = RequestStatus.TIMEOUT
            self._stats["total_expired"] += 1
            return False
        return True

    def _should_fair_schedule(self) -> Optional[Priority]:
        """
        公平调度: 检查是否需要强制调度低优先级请求。
        如果高优先级连续调度超过 fairness_window 次，则提升低优先级。
        """
        high_count = self._priority_counts[Priority.HIGH]
        med_count = self._priority_counts[Priority.MEDIUM]
        if high_count >= self.fairness_window and med_count < self.fairness_window:
            return Priority.MEDIUM
        if (high_count + med_count) >= self.fairness_window * 2:
            return Priority.LOW
        return None

    def next_batch(self) -> List[InferenceRequest]:
        """
        获取下一批请求 (连续批处理)。
        从队列中取出最多 max_batch_size 个有效请求。
        """
        with self._lock:
            batch: List[InferenceRequest] = []
            skipped: List[Tuple[int, int, InferenceRequest]] = []

            # 清理过期请求
            while self._heap:
                priority, order, request = heapq.heappop(self._heap)
                if not self._is_valid(request):
                    self._request_map.pop(request.request_id, None)
                    continue
                skipped.append((priority, order, request))

            # 放回
            for item in skipped:
                heapq.heappush(self._heap, item)

            # 公平调度检查
            fair_priority = self._should_fair_schedule()

            # 收集批次
            temp_skipped: List[Tuple[int, int, InferenceRequest]] = []
            while self._heap and len(batch) < self.max_batch_size:
                priority, order, request = heapq.heappop(self._heap)

                # 公平调度: 跳过高优先级，找低优先级
                if fair_priority is not None and priority < fair_priority:
                    temp_skipped.append((priority, order, request))
                    continue

                if not self._is_valid(request):
                    self._request_map.pop(request.request_id, None)
                    continue

                request.status = RequestStatus.RUNNING
                batch.append(request)
                self._priority_counts[request.priority] += 1
                self._total_scheduled += 1
                self._stats["total_scheduled"] += 1

            # 放回跳过的
            for item in temp_skipped:
                heapq.heappush(self._heap, item)

            # 重置公平调度计数器
            if fair_priority is not None:
                self._priority_counts = {p: 0 for p in Priority}

            if batch:
                self._stats["batches_formed"] += 1
                self._batch_sizes.append(len(batch))

            # 清理已完成请求的映射
            for req in batch:
                self._request_map.pop(req.request_id, None)

            return batch

    def get_request(self, request_id: str) -> Optional[InferenceRequest]:
        """获取请求。"""
        with self._lock:
            return self._request_map.get(request_id)

    def purge_expired(self) -> int:
        """清理所有过期请求。返回清理数量。"""
        with self._lock:
            count = 0
            remaining: List[Tuple[int, int, InferenceRequest]] = []
            while self._heap:
                priority, order, request = heapq.heappop(self._heap)
                if self._is_valid(request):
                    remaining.append((priority, order, request))
                else:
                    count += 1
                    self._request_map.pop(request.request_id, None)
            for item in remaining:
                heapq.heappush(self._heap, item)
            return count

    def stop(self):
        self._running = False


# ============================================================================
# ModelPartitioner — 模型分区器
# ============================================================================

class ModelPartitioner:
    """
    模型分区器。
    将 Transformer 层分配到多设备，支持层级分区、流水线推理、
    显存均衡、通信优化和多种分区策略。
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        devices: Optional[List[DeviceInfo]] = None,
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.devices = devices or self._default_devices()
        self._lock = threading.RLock()
        self._current_plan: Optional[PartitionPlan] = None

    def _default_devices(self) -> List[DeviceInfo]:
        """默认设备列表。"""
        return [
            DeviceInfo(
                device_id="gpu-0",
                device_type=DeviceType.GPU,
                total_memory_bytes=24 * 1024**3,
                compute_flops=300 * 1e12,
                bandwidth_bytes_per_sec=900 * 1e9,
                quantization_support=[QuantizationLevel.FP32, QuantizationLevel.FP16,
                                      QuantizationLevel.INT8, QuantizationLevel.INT4],
            ),
            DeviceInfo(
                device_id="gpu-1",
                device_type=DeviceType.GPU,
                total_memory_bytes=24 * 1024**3,
                compute_flops=300 * 1e12,
                bandwidth_bytes_per_sec=900 * 1e9,
                quantization_support=[QuantizationLevel.FP32, QuantizationLevel.FP16,
                                      QuantizationLevel.INT8],
            ),
        ]

    def _build_layer_info(self) -> List[LayerInfo]:
        """构建每一层的信息。"""
        layers = []
        base_params = 4 * self.hidden_size * self.hidden_size  # QKVO
        for i in range(self.num_layers):
            # 模拟不同层的参数差异 (某些层更大)
            mult = 1.0 + 0.1 * (i % 3)
            param_count = int(base_params * mult)
            flops = param_count * 2  # 每个参数约 2 FLOPs
            memory = param_count * 4  # fp32
            kv_cache = 2 * self.hidden_size * 2 * 2  # K+V, fp16, per token
            layers.append(LayerInfo(
                layer_id=i,
                name=f"transformer_layer_{i}",
                param_count=param_count,
                flops_per_token=flops,
                memory_bytes=memory,
                kv_cache_bytes_per_token=kv_cache,
            ))
        return layers

    # ---- 分区策略 ----

    def partition(
        self,
        strategy: PartitionStrategy = PartitionStrategy.UNIFORM,
    ) -> PartitionPlan:
        """执行模型分区。"""
        with self._lock:
            layers = self._build_layer_info()
            num_devices = len(self.devices)

            if strategy == PartitionStrategy.UNIFORM:
                plan = self._partition_uniform(layers, num_devices)
            elif strategy == PartitionStrategy.BY_LAYER_SIZE:
                plan = self._partition_by_size(layers, num_devices)
            elif strategy == PartitionStrategy.BY_COMPUTE:
                plan = self._partition_by_compute(layers, num_devices)
            else:
                plan = self._partition_uniform(layers, num_devices)

            plan.strategy = strategy
            plan.pipeline_stages = num_devices
            plan.balance_score = self._compute_balance_score(plan, layers)
            self._current_plan = plan
            return plan

    def _partition_uniform(
        self, layers: List[LayerInfo], num_devices: int
    ) -> PartitionPlan:
        """均匀分区: 每个设备分配相同数量的层。"""
        assignments: List[Tuple[int, str]] = []
        device_mem: Dict[str, int] = {d.device_id: 0 for d in self.devices}
        layers_per_device = math.ceil(len(layers) / num_devices)

        for i, layer in enumerate(layers):
            device_idx = min(i // layers_per_device, num_devices - 1)
            device = self.devices[device_idx]
            assignments.append((layer.layer_id, device.device_id))
            device_mem[device.device_id] += layer.memory_bytes

        comm_bytes = self._estimate_communication(assignments, layers)
        return PartitionPlan(
            strategy=PartitionStrategy.UNIFORM,
            assignments=assignments,
            device_memory_usage=device_mem,
            estimated_comm_bytes=comm_bytes,
        )

    def _partition_by_size(
        self, layers: List[LayerInfo], num_devices: int
    ) -> PartitionPlan:
        """按层大小分区: 贪心算法，将层分配到当前显存最少的设备。"""
        assignments: List[Tuple[int, str]] = []
        device_mem: Dict[str, int] = {d.device_id: 0 for d in self.devices}

        # 按层大小降序排列
        sorted_layers = sorted(layers, key=lambda l: l.memory_bytes, reverse=True)

        for layer in sorted_layers:
            # 找到当前显存最少的设备
            min_device = min(self.devices, key=lambda d: device_mem[d.device_id])
            assignments.append((layer.layer_id, min_device.device_id))
            device_mem[min_device.device_id] += layer.memory_bytes

        assignments.sort(key=lambda x: x[0])
        comm_bytes = self._estimate_communication(assignments, layers)
        return PartitionPlan(
            strategy=PartitionStrategy.BY_LAYER_SIZE,
            assignments=assignments,
            device_memory_usage=device_mem,
            estimated_comm_bytes=comm_bytes,
        )

    def _partition_by_compute(
        self, layers: List[LayerInfo], num_devices: int
    ) -> PartitionPlan:
        """按计算量分区: 平衡每个设备的计算负载。"""
        assignments: List[Tuple[int, str]] = []
        device_compute: Dict[str, float] = {d.device_id: 0.0 for d in self.devices}
        device_mem: Dict[str, int] = {d.device_id: 0 for d in self.devices}

        # 按计算量降序排列
        sorted_layers = sorted(layers, key=lambda l: l.flops_per_token, reverse=True)

        for layer in sorted_layers:
            # 考虑设备的算力比例
            def device_load(d: DeviceInfo) -> float:
                if d.compute_flops <= 0:
                    return float('inf')
                return device_compute[d.device_id] / d.compute_flops

            min_device = min(self.devices, key=device_load)
            assignments.append((layer.layer_id, min_device.device_id))
            device_compute[min_device.device_id] += layer.flops_per_token
            device_mem[min_device.device_id] += layer.memory_bytes

        assignments.sort(key=lambda x: x[0])
        comm_bytes = self._estimate_communication(assignments, layers)
        return PartitionPlan(
            strategy=PartitionStrategy.BY_COMPUTE,
            assignments=assignments,
            device_memory_usage=device_mem,
            estimated_comm_bytes=comm_bytes,
        )

    def _estimate_communication(
        self, assignments: List[Tuple[int, str]], layers: List[LayerInfo]
    ) -> int:
        """估算设备间通信量。"""
        comm = 0
        layer_map = {l.layer_id: l for l in layers}
        for i in range(len(assignments) - 1):
            curr_layer_id, curr_device = assignments[i]
            next_layer_id, next_device = assignments[i + 1]
            if curr_device != next_device:
                # 隐藏层大小的激活值传输
                comm += self.hidden_size * 4  # fp32
        return comm

    def _compute_balance_score(
        self, plan: PartitionPlan, layers: List[LayerInfo]
    ) -> float:
        """计算分区均衡分数 (0-1, 越高越均衡)。"""
        if not plan.device_memory_usage:
            return 0.0
        values = list(plan.device_memory_usage.values())
        if not values or max(values) == 0:
            return 0.0
        avg = sum(values) / len(values)
        if avg == 0:
            return 0.0
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        cv = std / avg  # 变异系数
        return max(0.0, 1.0 - cv)

    # ---- 流水线推理 ----

    def pipeline_inference(
        self,
        input_ids: List[int],
        plan: Optional[PartitionPlan] = None,
    ) -> List[float]:
        """流水线推理: 请求在设备间流水线传递。"""
        with self._lock:
            if plan is None:
                plan = self._current_plan or self.partition()

            # 按设备分组层
            device_layers: Dict[str, List[int]] = defaultdict(list)
            for layer_id, device_id in plan.assignments:
                device_layers[device_id].append(layer_id)

            # 模拟流水线: 激活值在设备间传递
            hidden = list(input_ids)  # 模拟激活值
            logits = [0.0] * _MOCK_VOCAB_SIZE

            for device in self.devices:
                layer_ids = sorted(device_layers.get(device.device_id, []))
                if not layer_ids:
                    continue
                # 模拟在该设备上执行层
                time.sleep(0.001)  # 模拟计算延迟
                for lid in layer_ids:
                    seed = sum(hidden) % _MOCK_VOCAB_SIZE if hidden else 0
                    for i in range(min(100, _MOCK_VOCAB_SIZE)):
                        logits[i] = math.sin(seed * 0.001 + lid * 0.01 + i * 0.017)
                # 传输到下一设备 (通信开销)
                if device.device_id != plan.assignments[-1][1]:
                    time.sleep(0.0005)

            return logits

    # ---- 通信优化 ----

    def optimize_communication(self, plan: PartitionPlan) -> PartitionPlan:
        """优化分区方案的通信开销。"""
        with self._lock:
            # 尝试将相邻层尽量放在同一设备
            assignments = plan.assignments
            optimized: List[Tuple[int, str]] = []
            current_device = assignments[0][1] if assignments else ""

            for layer_id, device_id in assignments:
                # 如果当前设备还能容纳，保持
                optimized.append((layer_id, current_device if device_id else device_id))

            plan.estimated_comm_bytes = self._estimate_communication(
                optimized, self._build_layer_info()
            )
            return plan

    def get_plan_info(self) -> Dict[str, Any]:
        """获取当前分区方案信息。"""
        with self._lock:
            if self._current_plan is None:
                return {"status": "no_plan"}
            plan = self._current_plan
            return {
                "strategy": plan.strategy.value,
                "pipeline_stages": plan.pipeline_stages,
                "balance_score": round(plan.balance_score, 4),
                "estimated_comm_bytes": plan.estimated_comm_bytes,
                "device_memory_usage": {
                    k: v // (1024**2) for k, v in plan.device_memory_usage.items()
                },  # MB
                "assignments": plan.assignments,
                "num_devices": len(self.devices),
            }


# ============================================================================
# EdgeOptimizer — 边缘部署优化器
# ============================================================================

class EdgeOptimizer:
    """
    边缘部署优化器。
    分析模型的计算/内存特征，根据设备能力选择最优配置，
    生成设备特定的推理计划，并支持资源不足时的降级方案。
    """

    def __init__(
        self,
        model_name: str = "lingyuan-base",
        num_layers: int = 12,
        hidden_size: int = 768,
        vocab_size: int = _MOCK_VOCAB_SIZE,
    ):
        self.model_name = model_name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._lock = threading.RLock()

    def _model_param_count(self) -> int:
        """估算模型参数量。"""
        # 嵌入层 + Transformer 层 + 输出层
        embed = self.vocab_size * self.hidden_size
        per_layer = 4 * self.hidden_size * self.hidden_size + 2 * self.hidden_size
        output = self.vocab_size * self.hidden_size
        return embed + self.num_layers * per_layer + output

    def analyze_model(self) -> Dict[str, Any]:
        """分析模型的计算/内存特征。"""
        with self._lock:
            params = self._model_param_count()
            # 不同精度的显存占用
            mem_fp32 = params * 4
            mem_fp16 = params * 2
            mem_int8 = params * 1
            mem_int4 = params // 2

            # 每个 token 的计算量 (FLOPs)
            flops_per_token = 2 * params  # 约等于 2 * 参数量
            # KV 缓存 (每 token)
            kv_per_token = 2 * self.num_layers * self.hidden_size * 2  # fp16

            return {
                "model_name": self.model_name,
                "num_layers": self.num_layers,
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size,
                "param_count": params,
                "param_count_B": round(params / 1e9, 3),
                "memory": {
                    "fp32_bytes": mem_fp32,
                    "fp32_MB": round(mem_fp32 / (1024**2), 2),
                    "fp16_MB": round(mem_fp16 / (1024**2), 2),
                    "int8_MB": round(mem_int8 / (1024**2), 2),
                    "int4_MB": round(mem_int4 / (1024**2), 2),
                },
                "flops_per_token": flops_per_token,
                "flops_per_token_G": round(flops_per_token / 1e9, 3),
                "kv_cache_per_token_bytes": kv_per_token,
                "kv_cache_per_token_KB": round(kv_per_token / 1024, 2),
            }

    def match_device(
        self, device: DeviceInfo, analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """根据设备能力选择最优配置。"""
        with self._lock:
            if analysis is None:
                analysis = self.analyze_model()

            device_mem_mb = device.total_memory_bytes / (1024**2)
            best_quant = QuantizationLevel.FP32
            best_mem = analysis["memory"]["fp32_MB"]

            # 选择能放下的最高精度 (最低量化)
            quant_priority = [
                (QuantizationLevel.FP32, analysis["memory"]["fp32_MB"]),
                (QuantizationLevel.FP16, analysis["memory"]["fp16_MB"]),
                (QuantizationLevel.INT8, analysis["memory"]["int8_MB"]),
                (QuantizationLevel.INT4, analysis["memory"]["int4_MB"]),
            ]

            for quant, mem_mb in quant_priority:
                if quant in device.quantization_support and mem_mb < device_mem_mb * 0.8:
                    best_quant = quant
                    best_mem = mem_mb
                    break

            # 估算最大 KV 缓存容量
            available_mem = device_mem_mb * 0.8 - best_mem
            kv_per_token_kb = analysis["kv_cache_per_token_KB"]
            max_kv_tokens = int(available_mem * 1024 / max(1, kv_per_token_kb))

            # 估算吞吐 (tokens/sec)
            flops_per_token = analysis["flops_per_token"]
            if device.compute_flops > 0:
                # 假设 30% 算力用于推理
                estimated_tps = (device.compute_flops * 0.3) / flops_per_token
            else:
                estimated_tps = 10.0

            return {
                "device_id": device.device_id,
                "device_type": device.device_type.value,
                "recommended_quantization": best_quant.value,
                "model_memory_MB": round(best_mem, 2),
                "device_memory_MB": round(device_mem_mb, 2),
                "available_for_kv_MB": round(available_mem, 2),
                "max_kv_cache_tokens": max_kv_tokens,
                "estimated_tps": round(estimated_tps, 2),
                "fits": best_mem < device_mem_mb * 0.8,
            }

    def generate_inference_plan(
        self,
        device: DeviceInfo,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
    ) -> Dict[str, Any]:
        """生成设备特定的推理计划。"""
        with self._lock:
            analysis = self.analyze_model()
            match = self.match_device(device, analysis)

            if not match["fits"]:
                # 需要降级
                degraded = self.degrade_strategy(device, max_batch_size, max_seq_len)
                return {
                    "device_id": device.device_id,
                    "feasible": False,
                    "degradation": degraded,
                    "analysis": analysis,
                    "match": match,
                }

            # 计算最优 batch size
            kv_per_token = analysis["kv_cache_per_token_bytes"]
            available_kv = match["available_for_kv_MB"] * (1024**2)
            kv_per_batch = kv_per_token * max_seq_len * max_batch_size
            optimal_batch = max(1, min(max_batch_size, int(available_kv / max(1, kv_per_token * max_seq_len))))

            quant = QuantizationLevel(match["recommended_quantization"])

            return {
                "device_id": device.device_id,
                "feasible": True,
                "quantization": quant.value,
                "optimal_batch_size": optimal_batch,
                "max_seq_len": max_seq_len,
                "max_kv_tokens": match["max_kv_cache_tokens"],
                "estimated_tps": match["estimated_tps"],
                "model_memory_MB": match["model_memory_MB"],
                "analysis": analysis,
                "match": match,
                "config": {
                    "num_layers": self.num_layers,
                    "hidden_size": self.hidden_size,
                    "vocab_size": self.vocab_size,
                    "use_kv_cache": True,
                    "pipeline_stages": 1 if device.device_type != DeviceType.CPU else 0,
                },
            }

    def degrade_strategy(
        self,
        device: DeviceInfo,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
    ) -> Dict[str, Any]:
        """资源不足时的降级方案。"""
        with self._lock:
            analysis = self.analyze_model()
            device_mem_mb = device.total_memory_bytes / (1024**2)

            # 逐步降级
            steps = []
            current_quant = QuantizationLevel.FP32
            current_mem = analysis["memory"]["fp32_MB"]

            for quant, mem_mb in [
                (QuantizationLevel.FP16, analysis["memory"]["fp16_MB"]),
                (QuantizationLevel.INT8, analysis["memory"]["int8_MB"]),
                (QuantizationLevel.INT4, analysis["memory"]["int4_MB"]),
            ]:
                if quant in device.quantization_support:
                    steps.append({
                        "action": f"quantize_to_{quant.value}",
                        "memory_MB": round(mem_mb, 2),
                        "fits": mem_mb < device_mem_mb * 0.8,
                    })
                    if mem_mb < device_mem_mb * 0.8:
                        current_quant = quant
                        current_mem = mem_mb
                        break

            # 如果仍然放不下，减少层数 (层丢弃)
            if current_mem >= device_mem_mb * 0.8:
                max_fit_layers = int(self.num_layers * (device_mem_mb * 0.7) / current_mem)
                max_fit_layers = max(1, min(max_fit_layers, self.num_layers))
                steps.append({
                    "action": f"layer_drop_to_{max_fit_layers}",
                    "remaining_layers": max_fit_layers,
                    "memory_MB": round(current_mem * max_fit_layers / self.num_layers, 2),
                    "fits": True,
                })

            # 减少 batch size 和序列长度
            steps.append({
                "action": "reduce_batch_size",
                "new_batch_size": 1,
            })
            steps.append({
                "action": "reduce_seq_len",
                "new_seq_len": min(512, max_seq_len),
            })

            return {
                "feasible_after": True,
                "original_memory_MB": round(analysis["memory"]["fp32_MB"], 2),
                "device_memory_MB": round(device_mem_mb, 2),
                "steps": steps,
                "final_quantization": current_quant.value,
            }

    def compare_devices(self, devices: List[DeviceInfo]) -> List[Dict[str, Any]]:
        """对比多个设备的部署能力。"""
        results = []
        for device in devices:
            plan = self.generate_inference_plan(device)
            results.append({
                "device_id": device.device_id,
                "device_type": device.device_type.value,
                "feasible": plan["feasible"],
                "quantization": plan.get("quantization", plan.get("degradation", {}).get("final_quantization", "N/A")),
                "estimated_tps": plan.get("estimated_tps", 0),
                "optimal_batch_size": plan.get("optimal_batch_size", 1),
            })
        # 按吞吐排序
        results.sort(key=lambda x: x["estimated_tps"], reverse=True)
        return results


# ============================================================================
# LoadBalancer — 负载均衡器
# ============================================================================

class LoadBalancer:
    """
    负载均衡器。
    支持健康检查、请求路由、自动扩缩容、故障转移和统计。
    """

    def __init__(
        self,
        health_check_interval: float = 5.0,
        unhealthy_threshold: int = 3,
        max_nodes: int = 10,
        min_nodes: int = 1,
        scale_up_threshold: int = 20,  # 队列长度超过此值扩容
        scale_down_threshold: int = 5,  # 队列长度低于此值缩容
        routing_strategy: str = "least_load",  # least_load / round_robin / least_latency
    ):
        self.health_check_interval = health_check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.routing_strategy = routing_strategy

        self._lock = threading.RLock()
        self._nodes: Dict[str, InferenceNode] = {}
        self._rr_index = 0  # round robin 索引
        self._node_counter = 0

        # 统计
        self._stats = {
            "total_routed": 0,
            "total_failed": 0,
            "total_failovers": 0,
            "total_scaled_up": 0,
            "total_scaled_down": 0,
            "routing_decisions": defaultdict(int),
        }
        self._qps_history: deque = deque(maxlen=60)
        self._latency_history: deque = deque(maxlen=1000)
        self._error_history: deque = deque(maxlen=100)

        self._health_thread: Optional[threading.Thread] = None
        self._running = False

    # ---- 节点管理 ----

    def add_node(self, node: InferenceNode) -> bool:
        """添加推理节点。"""
        with self._lock:
            if node.node_id in self._nodes:
                return False
            if len(self._nodes) >= self.max_nodes:
                return False
            self._nodes[node.node_id] = node
            logger.info(f"Added node {node.node_id}")
            return True

    def remove_node(self, node_id: str) -> bool:
        """移除推理节点。"""
        with self._lock:
            if node_id not in self._nodes:
                return False
            del self._nodes[node_id]
            logger.info(f"Removed node {node_id}")
            return True

    def register_node(
        self,
        endpoint: str,
        device_info: Optional[DeviceInfo] = None,
    ) -> InferenceNode:
        """注册新节点。"""
        with self._lock:
            self._node_counter += 1
            node_id = f"node-{self._node_counter}"
            if device_info is None:
                device_info = DeviceInfo(
                    device_id=node_id,
                    device_type=DeviceType.GPU,
                    total_memory_bytes=24 * 1024**3,
                    compute_flops=300 * 1e12,
                )
            node = InferenceNode(
                node_id=node_id,
                endpoint=endpoint,
                device_info=device_info,
            )
            self._nodes[node_id] = node
            return node

    def get_nodes(self) -> List[InferenceNode]:
        """获取所有节点。"""
        with self._lock:
            return list(self._nodes.values())

    def get_available_nodes(self) -> List[InferenceNode]:
        """获取可用节点。"""
        with self._lock:
            return [n for n in self._nodes.values() if n.is_available]

    # ---- 健康检查 ----

    def start_health_check(self):
        """启动健康检查线程。"""
        if self._running:
            return
        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self._health_thread.start()
        logger.info("Health check started")

    def stop_health_check(self):
        """停止健康检查。"""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=2.0)

    def _health_check_loop(self):
        """健康检查循环。"""
        while self._running:
            try:
                self.check_all_health()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            time.sleep(self.health_check_interval)

    def check_all_health(self):
        """检查所有节点健康状态。"""
        with self._lock:
            for node in self._nodes.values():
                self._check_node_health(node)

    def _check_node_health(self, node: InferenceNode):
        """检查单个节点健康状态 (模拟)。"""
        # 模拟: 随机心跳
        node.last_heartbeat = time.time()
        # 模拟偶尔的故障
        if random.random() < 0.001 and node.status == NodeStatus.HEALTHY:
            node.status = NodeStatus.DEGRADED
            node.error_count += 1
            logger.warning(f"Node {node.node_id} degraded")
        elif node.error_count > self.unhealthy_threshold:
            node.status = NodeStatus.UNHEALTHY
        elif node.status == NodeStatus.DEGRADED and random.random() < 0.5:
            node.status = NodeStatus.HEALTHY
            node.error_count = 0

    def heartbeat(self, node_id: str):
        """节点心跳。"""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.last_heartbeat = time.time()
                if node.status == NodeStatus.UNHEALTHY:
                    node.status = NodeStatus.HEALTHY
                    node.error_count = 0

    # ---- 请求路由 ----

    def route(self, request: Optional[InferenceRequest] = None) -> Optional[InferenceNode]:
        """路由请求到最优节点。"""
        with self._lock:
            available = [n for n in self._nodes.values() if n.is_available]
            if not available:
                logger.error("No available nodes for routing")
                self._stats["total_failed"] += 1
                return None

            if self.routing_strategy == "round_robin":
                node = self._route_round_robin(available)
            elif self.routing_strategy == "least_latency":
                node = self._route_least_latency(available)
            else:  # least_load
                node = self._route_least_load(available)

            if node:
                node.active_requests += 1
                self._stats["total_routed"] += 1
                self._stats["routing_decisions"][self.routing_strategy] += 1
            return node

    def _route_round_robin(self, nodes: List[InferenceNode]) -> InferenceNode:
        """轮询路由。"""
        node = nodes[self._rr_index % len(nodes)]
        self._rr_index += 1
        return node

    def _route_least_load(self, nodes: List[InferenceNode]) -> InferenceNode:
        """最小负载路由。"""
        return min(nodes, key=lambda n: n.load_score)

    def _route_least_latency(self, nodes: List[InferenceNode]) -> InferenceNode:
        """最低延迟路由。"""
        return min(nodes, key=lambda n: n.avg_latency_ms)

    def complete_request(self, node_id: str, latency_ms: float, success: bool = True):
        """请求完成回调。"""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.active_requests = max(0, node.active_requests - 1)
                node.record_latency(latency_ms)
                if not success:
                    node.error_count += 1
                    self._stats["total_failed"] += 1
                    self._error_history.append(time.time())
            self._latency_history.append(latency_ms)

    # ---- 自动扩缩容 ----

    def auto_scale(self, total_queue_length: int) -> Dict[str, Any]:
        """根据队列长度自动扩缩容。"""
        with self._lock:
            actions = []
            current_nodes = len(self._nodes)

            if total_queue_length > self.scale_up_threshold and current_nodes < self.max_nodes:
                # 扩容
                new_node = self.register_node(f"endpoint-{self._node_counter + 1}")
                self._stats["total_scaled_up"] += 1
                actions.append({
                    "action": "scale_up",
                    "new_node_id": new_node.node_id,
                    "reason": f"queue_length={total_queue_length} > {self.scale_up_threshold}",
                })
                logger.info(f"Scaling up: added {new_node.node_id}")

            elif (total_queue_length < self.scale_down_threshold and
                  current_nodes > self.min_nodes):
                # 缩容: 移除负载最低的节点
                nodes = list(self._nodes.values())
                if nodes:
                    node_to_remove = min(nodes, key=lambda n: n.load_score)
                    self.remove_node(node_to_remove.node_id)
                    self._stats["total_scaled_down"] += 1
                    actions.append({
                        "action": "scale_down",
                        "removed_node_id": node_to_remove.node_id,
                        "reason": f"queue_length={total_queue_length} < {self.scale_down_threshold}",
                    })
                    logger.info(f"Scaling down: removed {node_to_remove.node_id}")

            return {
                "actions": actions,
                "current_nodes": len(self._nodes),
                "queue_length": total_queue_length,
            }

    # ---- 故障转移 ----

    def failover(self, failed_node_id: str) -> Optional[InferenceNode]:
        """节点故障时自动切换到其他节点。"""
        with self._lock:
            node = self._nodes.get(failed_node_id)
            if node:
                node.status = NodeStatus.UNHEALTHY
                node.error_count += 1
            self._stats["total_failovers"] += 1

            available = [n for n in self._nodes.values()
                         if n.is_available and n.node_id != failed_node_id]
            if available:
                return min(available, key=lambda n: n.load_score)
            return None

    # ---- 统计 ----

    def record_request(self, latency_ms: float, success: bool = True):
        """记录请求统计。"""
        with self._lock:
            self._latency_history.append(latency_ms)
            if not success:
                self._error_history.append(time.time())

    def get_stats(self) -> Dict[str, Any]:
        """获取负载均衡器统计。"""
        with self._lock:
            latencies = list(self._latency_history)
            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
            sorted_lat = sorted(latencies)
            p50 = sorted_lat[int(len(sorted_lat) * 0.5)] if sorted_lat else 0.0
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0.0
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0.0

            # 错误率 (最近60秒)
            now = time.time()
            recent_errors = sum(1 for t in self._error_history if now - t < 60)
            error_rate = recent_errors / max(1, len(latencies)) if latencies else 0.0

            return {
                "total_routed": self._stats["total_routed"],
                "total_failed": self._stats["total_failed"],
                "total_failovers": self._stats["total_failovers"],
                "total_scaled_up": self._stats["total_scaled_up"],
                "total_scaled_down": self._stats["total_scaled_down"],
                "num_nodes": len(self._nodes),
                "num_available": len(self.get_available_nodes()),
                "avg_latency_ms": round(avg_lat, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "error_rate": round(error_rate, 4),
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "status": n.status.value,
                        "active_requests": n.active_requests,
                        "queue_length": n.queue_length,
                        "avg_latency_ms": round(n.avg_latency_ms, 2),
                        "load_score": round(n.load_score, 2),
                    }
                    for n in self._nodes.values()
                ],
            }


# ============================================================================
# PerformanceMonitor — 性能监控器
# ============================================================================

class PerformanceMonitor:
    """
    性能监控器。
    支持实时指标 (QPS/延迟/P50/P95/P99/吞吐)、资源监控、
    延迟告警、时间窗口聚合和仪表板数据生成。
    """

    def __init__(
        self,
        window_seconds: int = 60,
        alert_latency_threshold_ms: float = 5000.0,
        alert_error_rate_threshold: float = 0.05,
        alert_queue_threshold: int = 100,
    ):
        self.window_seconds = window_seconds
        self.alert_latency_threshold_ms = alert_latency_threshold_ms
        self.alert_error_rate_threshold = alert_error_rate_threshold
        self.alert_queue_threshold = alert_queue_threshold

        self._lock = threading.RLock()
        # 指标存储: timestamp -> metrics
        self._metrics: deque = deque(maxlen=10000)
        # 请求记录: (timestamp, latency_ms, success)
        self._requests: deque = deque(maxlen=10000)
        # 资源使用记录: timestamp -> {cpu, memory, gpu}
        self._resources: deque = deque(maxlen=10000)
        # 告警
        self._alerts: deque = deque(maxlen=100)
        # 聚合窗口
        self._aggregations: Dict[str, deque] = {
            "1s": deque(maxlen=3600),
            "1m": deque(maxlen=1440),
            "5m": deque(maxlen=288),
        }
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

    # ---- 指标记录 ----

    def record_request(self, latency_ms: float, success: bool = True, tokens: int = 0):
        """记录请求指标。"""
        with self._lock:
            now = time.time()
            self._requests.append((now, latency_ms, success, tokens))

    def record_resource(
        self,
        cpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        gpu_percent: float = 0.0,
        gpu_memory_percent: float = 0.0,
    ):
        """记录资源使用。"""
        with self._lock:
            now = time.time()
            self._resources.append({
                "timestamp": now,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "gpu_percent": gpu_percent,
                "gpu_memory_percent": gpu_memory_percent,
            })

    def record_metric(self, name: str, value: float):
        """记录自定义指标。"""
        with self._lock:
            now = time.time()
            self._metrics.append({"timestamp": now, "name": name, "value": value})

    # ---- 实时指标计算 ----

    def get_realtime_metrics(self) -> Dict[str, Any]:
        """获取实时指标。"""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            # 过滤窗口内请求
            window_requests = [
                (ts, lat, succ, tok) for ts, lat, succ, tok in self._requests
                if ts >= window_start
            ]

            if not window_requests:
                return {
                    "qps": 0.0,
                    "avg_latency_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "throughput_tps": 0.0,
                    "error_rate": 0.0,
                    "total_requests": 0,
                    "window_seconds": self.window_seconds,
                }

            latencies = sorted([r[1] for r in window_requests])
            successes = sum(1 for r in window_requests if r[2])
            total_tokens = sum(r[3] for r in window_requests)
            n = len(latencies)

            qps = n / self.window_seconds
            avg_lat = sum(latencies) / n
            p50 = latencies[int(n * 0.5)]
            p95 = latencies[min(n - 1, int(n * 0.95))]
            p99 = latencies[min(n - 1, int(n * 0.99))]
            error_rate = 1.0 - (successes / n)
            throughput = total_tokens / self.window_seconds

            return {
                "qps": round(qps, 2),
                "avg_latency_ms": round(avg_lat, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "throughput_tps": round(throughput, 2),
                "error_rate": round(error_rate, 4),
                "total_requests": n,
                "window_seconds": self.window_seconds,
            }

    def get_resource_usage(self) -> Dict[str, Any]:
        """获取最新资源使用。"""
        with self._lock:
            if not self._resources:
                return {
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "gpu_percent": 0.0,
                    "gpu_memory_percent": 0.0,
                }
            latest = self._resources[-1]
            return {
                "cpu_percent": round(latest["cpu_percent"], 2),
                "memory_percent": round(latest["memory_percent"], 2),
                "gpu_percent": round(latest["gpu_percent"], 2),
                "gpu_memory_percent": round(latest["gpu_memory_percent"], 2),
            }

    # ---- 告警 ----

    def check_alerts(self) -> List[Dict[str, Any]]:
        """检查告警条件。"""
        with self._lock:
            metrics = self.get_realtime_metrics()
            new_alerts = []

            if metrics["p99_ms"] > self.alert_latency_threshold_ms:
                alert = {
                    "type": "high_latency",
                    "severity": "warning",
                    "message": f"P99 latency {metrics['p99_ms']}ms exceeds threshold {self.alert_latency_threshold_ms}ms",
                    "timestamp": time.time(),
                    "value": metrics["p99_ms"],
                    "threshold": self.alert_latency_threshold_ms,
                }
                new_alerts.append(alert)
                self._alerts.append(alert)

            if metrics["error_rate"] > self.alert_error_rate_threshold:
                alert = {
                    "type": "high_error_rate",
                    "severity": "critical",
                    "message": f"Error rate {metrics['error_rate']:.2%} exceeds threshold {self.alert_error_rate_threshold:.2%}",
                    "timestamp": time.time(),
                    "value": metrics["error_rate"],
                    "threshold": self.alert_error_rate_threshold,
                }
                new_alerts.append(alert)
                self._alerts.append(alert)

            return new_alerts

    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取告警历史。"""
        with self._lock:
            return list(self._alerts)[-limit:]

    # ---- 指标聚合 ----

    def aggregate(self, window: str = "1m") -> Dict[str, Any]:
        """按时间窗口聚合指标。"""
        with self._lock:
            now = time.time()
            if window == "1s":
                w = 1
            elif window == "1m":
                w = 60
            elif window == "5m":
                w = 300
            else:
                w = 60

            window_start = now - w
            window_requests = [
                (ts, lat, succ, tok) for ts, lat, succ, tok in self._requests
                if ts >= window_start
            ]

            if not window_requests:
                return {"window": window, "count": 0}

            latencies = [r[1] for r in window_requests]
            successes = sum(1 for r in window_requests if r[2])

            agg = {
                "window": window,
                "window_seconds": w,
                "count": len(window_requests),
                "qps": round(len(window_requests) / w, 2),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
                "min_latency_ms": round(min(latencies), 2),
                "max_latency_ms": round(max(latencies), 2),
                "success_rate": round(successes / len(window_requests), 4),
                "total_tokens": sum(r[3] for r in window_requests),
            }

            self._aggregations[window].append({"timestamp": now, "data": agg})
            return agg

    # ---- 仪表板数据 ----

    def get_dashboard_data(self) -> Dict[str, Any]:
        """生成监控仪表板数据。"""
        with self._lock:
            realtime = self.get_realtime_metrics()
            resources = self.get_resource_usage()
            alerts = self.get_alerts(10)

            # 资源使用趋势 (最近10个数据点)
            resource_trend = list(self._resources)[-10:]

            # 延迟趋势
            recent_requests = list(self._requests)[-50:]
            latency_trend = [
                {"timestamp": ts, "latency_ms": lat}
                for ts, lat, _, _ in recent_requests
            ]

            return {
                "realtime": realtime,
                "resources": resources,
                "resource_trend": resource_trend,
                "latency_trend": latency_trend,
                "alerts": alerts,
                "alert_count": len(self._alerts),
                "aggregations": {
                    "1m": self.aggregate("1m"),
                    "5m": self.aggregate("5m"),
                },
                "timestamp": time.time(),
            }

    # ---- 监控线程 ----

    def start_monitoring(self, interval: float = 1.0):
        """启动后台监控。"""
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        """停止监控。"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _monitor_loop(self, interval: float):
        """监控循环。"""
        while self._running:
            try:
                # 模拟资源采集
                self.record_resource(
                    cpu_percent=random.uniform(10, 80),
                    memory_percent=random.uniform(30, 70),
                    gpu_percent=random.uniform(20, 90),
                    gpu_memory_percent=random.uniform(40, 80),
                )
                self.check_alerts()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            time.sleep(interval)


# ============================================================================
# InferenceServer — 推理服务器
# ============================================================================

class InferenceServer:
    """
    推理服务器。
    模拟 HTTP API (/v1/completions, /v1/chat/completions)、SSE 流式响应、
    并发控制、健康检查端点和指标端点。
    """

    def __init__(
        self,
        engine: Optional[DistributedInferenceEngine] = None,
        scheduler: Optional[RequestScheduler] = None,
        monitor: Optional[PerformanceMonitor] = None,
        max_concurrent: int = 16,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.engine = engine or DistributedInferenceEngine(simulate_latency=False)
        self.scheduler = scheduler or RequestScheduler()
        self.monitor = monitor or PerformanceMonitor()
        self.max_concurrent = max_concurrent
        self.host = host
        self.port = port

        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.RLock()
        self._running = False
        self._request_counter = 0

        # 模拟路由表
        self._routes: Dict[str, Callable] = {
            "/health": self._handle_health,
            "/metrics": self._handle_metrics,
            "/v1/completions": self._handle_completions,
            "/v1/chat/completions": self._handle_chat_completions,
            "/v1/models": self._handle_models,
        }

        # 服务器统计
        self._server_stats = {
            "total_requests": 0,
            "total_errors": 0,
            "start_time": time.time(),
        }

    # ---- 属性 ----

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def uptime(self) -> float:
        return time.time() - self._server_stats["start_time"]

    # ---- 请求分发 ----

    def handle_request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """处理模拟 HTTP 请求。"""
        with self._lock:
            self._server_stats["total_requests"] += 1
            self._request_counter += 1
            request_id = f"http-{self._request_counter}"

        start_time = time.time()

        handler = self._routes.get(path)
        if handler is None:
            return self._error_response(404, f"Path not found: {path}")

        if method not in ("POST", "GET"):
            return self._error_response(405, f"Method not allowed: {method}")

        try:
            with self._semaphore:
                response = handler(body or {}, headers or {})
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_request(latency_ms, success=True)
            response["request_id"] = request_id
            response["latency_ms"] = round(latency_ms, 2)
            return response
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_request(latency_ms, success=False)
            with self._lock:
                self._server_stats["total_errors"] += 1
            return self._error_response(500, str(e))

    # ---- 端点处理器 ----

    def _handle_health(self, body: Dict, headers: Dict) -> Dict[str, Any]:
        """健康检查端点 /health。"""
        return {
            "status": "healthy",
            "engine_running": self.engine.is_running,
            "uptime_seconds": round(self.uptime, 2),
            "queue_size": self.scheduler.queue_size,
        }

    def _handle_metrics(self, body: Dict, headers: Dict) -> Dict[str, Any]:
        """指标端点 /metrics。"""
        return {
            "server": {
                "total_requests": self._server_stats["total_requests"],
                "total_errors": self._server_stats["total_errors"],
                "uptime_seconds": round(self.uptime, 2),
                "max_concurrent": self.max_concurrent,
            },
            "engine": self.engine.get_stats(),
            "scheduler": self.scheduler.stats,
            "monitor": self.monitor.get_realtime_metrics(),
            "resources": self.monitor.get_resource_usage(),
        }

    def _handle_models(self, body: Dict, headers: Dict) -> Dict[str, Any]:
        """模型列表端点 /v1/models。"""
        return {
            "object": "list",
            "data": [
                {
                    "id": self.engine.model_name,
                    "object": "model",
                    "created": int(self._server_stats["start_time"]),
                    "owned_by": "lingyuan",
                }
            ],
        }

    def _handle_completions(self, body: Dict, headers: Dict) -> Dict[str, Any]:
        """/v1/completions 端点。"""
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 128)
        temperature = body.get("temperature", 1.0)
        top_k = body.get("top_k", 50)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)
        strategy_str = body.get("strategy", "greedy")

        strategy = SamplingStrategy(strategy_str) if strategy_str in [
            s.value for s in SamplingStrategy
        ] else SamplingStrategy.GREEDY

        if stream:
            # 返回 SSE 流式响应模拟
            chunks = []
            for chunk in self.engine.generate_stream(
                prompt, max_tokens=max_tokens, strategy=strategy,
                temperature=temperature, top_k=top_k, top_p=top_p,
            ):
                if chunk["type"] == "token":
                    chunks.append({
                        "object": "text_completion_chunk",
                        "choices": [{"text": chunk["text"], "finish_reason": None}],
                    })
                elif chunk["type"] == "done":
                    chunks.append({
                        "object": "text_completion_chunk",
                        "choices": [{"text": "", "finish_reason": chunk["finish_reason"]}],
                    })
            return {
                "status": "streaming",
                "object": "text_completion",
                "chunks": chunks,
                "stream": True,
            }
        else:
            result = self.engine.generate(
                prompt, max_tokens=max_tokens, strategy=strategy,
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            return {
                "status": "success",
                "object": "text_completion",
                "id": result.request_id,
                "model": self.engine.model_name,
                "choices": [{
                    "text": result.text,
                    "finish_reason": result.finish_reason,
                    "token_ids": result.token_ids,
                }],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            }

    def _handle_chat_completions(self, body: Dict, headers: Dict) -> Dict[str, Any]:
        """/v1/chat/completions 端点。"""
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 128)
        temperature = body.get("temperature", 1.0)
        stream = body.get("stream", False)

        # 将消息合并为 prompt
        prompt = " ".join(m.get("content", "") for m in messages)

        if stream:
            chunks = []
            for chunk in self.engine.generate_stream(
                prompt, max_tokens=max_tokens,
                strategy=SamplingStrategy.TEMPERATURE,
                temperature=temperature,
            ):
                if chunk["type"] == "token":
                    chunks.append({
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "delta": {"content": chunk["text"]},
                            "finish_reason": None,
                        }],
                    })
                elif chunk["type"] == "done":
                    chunks.append({
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "delta": {},
                            "finish_reason": chunk["finish_reason"],
                        }],
                    })
            return {
                "status": "streaming",
                "object": "chat.completion",
                "chunks": chunks,
                "stream": True,
            }
        else:
            result = self.engine.generate(
                prompt, max_tokens=max_tokens,
                strategy=SamplingStrategy.TEMPERATURE,
                temperature=temperature,
            )
            return {
                "status": "success",
                "object": "chat.completion",
                "id": result.request_id,
                "model": self.engine.model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text,
                    },
                    "finish_reason": result.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            }

    # ---- 辅助方法 ----

    def _error_response(self, status: int, message: str) -> Dict[str, Any]:
        """生成错误响应。"""
        return {
            "status": "error",
            "error": {
                "code": status,
                "message": message,
            },
        }

    def simulate_sse_stream(self, prompt: str, max_tokens: int = 32) -> List[str]:
        """模拟 SSE (Server-Sent Events) 流式响应。"""
        sse_events = []
        for chunk in self.engine.generate_stream(prompt, max_tokens=max_tokens):
            if chunk["type"] == "token":
                data = json.dumps({
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": chunk["text"]}}],
                })
                sse_events.append(f"data: {data}\n\n")
            elif chunk["type"] == "done":
                sse_events.append("data: [DONE]\n\n")
        return sse_events

    # ---- 服务器生命周期 ----

    def start(self):
        """启动服务器 (模拟)。"""
        self._running = True
        self._server_stats["start_time"] = time.time()
        self.monitor.start_monitoring()
        logger.info(f"InferenceServer started on {self.host}:{self.port}")

    def stop(self):
        """停止服务器。"""
        self._running = False
        self.monitor.stop_monitoring()
        self.engine.stop()
        logger.info("InferenceServer stopped")

    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息。"""
        return {
            "host": self.host,
            "port": self.port,
            "running": self._running,
            "uptime_seconds": round(self.uptime, 2),
            "max_concurrent": self.max_concurrent,
            "model": self.engine.model_name,
            "routes": list(self._routes.keys()),
            "stats": dict(self._server_stats),
        }


# ============================================================================
# __main__ 自测代码
# ============================================================================

def _print_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _test_kv_cache_manager():
    """测试 KVCacheManager。"""
    _print_separator("测试 KVCacheManager")
    mgr = KVCacheManager(pool_size_bytes=10 * 1024 * 1024, max_entries=10, num_layers=6, hidden_size=512)

    # 存入缓存
    for i in range(5):
        token_ids = list(range(i * 10, (i + 1) * 10))
        mgr.put(f"cache-{i}", token_ids)

    print(f"  缓存条目数: {mgr.num_entries}")
    print(f"  已用字节: {mgr.used_bytes}")
    print(f"  使用率: {mgr.usage_ratio:.4f}")

    # 缓存命中
    entry = mgr.get("cache-0")
    assert entry is not None, "缓存命中失败"
    print(f"  缓存命中: cache-0, access_count={entry.access_count}")

    # 缓存未命中
    entry = mgr.get("nonexistent")
    assert entry is None, "缓存未命中检测失败"
    print(f"  缓存未命中: nonexistent (正确)")

    # LRU 淘汰
    for i in range(5, 15):
        mgr.put(f"cache-{i}", list(range(i * 10, (i + 1) * 10)))
    print(f"  添加更多缓存后条目数: {mgr.num_entries} (应 <= 10)")

    # 缓存压缩
    time.sleep(0.1)
    mgr._compression_threshold = 0  # 立即压缩
    compressed = mgr.compress_inactive()
    print(f"  压缩不活跃缓存: {compressed} 条")

    # prefix 共享
    shared = mgr.find_shared([0, 1, 2, 3, 4])
    print(f"  Prefix 共享查找: {'找到' if shared else '未找到'}")

    report = mgr.get_memory_report()
    print(f"  内存报告: entries={report['num_entries']}, usage={report['usage_ratio']:.4f}")
    print(f"  统计: hits={report['stats']['hits']}, misses={report['stats']['misses']}, evictions={report['stats']['evictions']}")
    print("  [PASS] KVCacheManager 测试通过")


def _test_inference_engine():
    """测试 DistributedInferenceEngine。"""
    _print_separator("测试 DistributedInferenceEngine")
    engine = DistributedInferenceEngine(simulate_latency=False, max_batch_size=4)

    # 单序列推理 - greedy
    result = engine.generate("hello world", max_tokens=10, strategy=SamplingStrategy.GREEDY)
    print(f"  Greedy 推理: prompt_tokens={result.prompt_tokens}, output_tokens={result.completion_tokens}")
    print(f"    输出: {result.text[:50]}...")
    assert result.completion_tokens > 0, "推理未生成 token"

    # 温度采样
    result = engine.generate("test prompt", max_tokens=10, strategy=SamplingStrategy.TEMPERATURE, temperature=0.8)
    print(f"  Temperature 推理: {result.completion_tokens} tokens, 延迟={result.latency_ms:.2f}ms")

    # Top-K 采样
    result = engine.generate("top k test", max_tokens=10, strategy=SamplingStrategy.TOP_K, top_k=20)
    print(f"  Top-K 推理: {result.completion_tokens} tokens")

    # Top-P 采样
    result = engine.generate("nucleus test", max_tokens=10, strategy=SamplingStrategy.TOP_P, top_p=0.8)
    print(f"  Top-P 推理: {result.completion_tokens} tokens")

    # 流式推理
    tokens_received = 0
    for chunk in engine.generate_stream("stream test", max_tokens=5):
        if chunk["type"] == "token":
            tokens_received += 1
        elif chunk["type"] == "done":
            print(f"  流式推理: 收到 {tokens_received} tokens, 延迟={chunk['latency_ms']:.2f}ms")
    assert tokens_received > 0, "流式推理未收到 token"

    # 批量推理
    prompts = ["batch 1", "batch 2", "batch 3"]
    results = engine.generate_batch(prompts, max_tokens=5)
    print(f"  批量推理: {len(results)} 个结果")
    assert len(results) == 3, "批量推理结果数不匹配"

    # 统计
    stats = engine.get_stats()
    print(f"  统计: completed={stats['completed']}, avg_latency={stats['avg_latency_ms']:.2f}ms")
    print(f"    p50={stats['p50_ms']:.2f}ms, p95={stats['p95_ms']:.2f}ms, p99={stats['p99_ms']:.2f}ms")
    print(f"    throughput={stats['throughput_tps']:.2f} tps, qps={stats['qps']:.2f}")
    print("  [PASS] DistributedInferenceEngine 测试通过")


def _test_request_scheduler():
    """测试 RequestScheduler。"""
    _print_separator("测试 RequestScheduler")
    scheduler = RequestScheduler(max_batch_size=4, timeout_seconds=10, fairness_window=3)

    # 提交不同优先级的请求
    for i in range(3):
        req = InferenceRequest(
            request_id=f"high-{i}", prompt=f"high priority {i}",
            priority=Priority.HIGH, max_tokens=5,
        )
        scheduler.submit(req)

    for i in range(3):
        req = InferenceRequest(
            request_id=f"med-{i}", prompt=f"medium priority {i}",
            priority=Priority.MEDIUM, max_tokens=5,
        )
        scheduler.submit(req)

    for i in range(3):
        req = InferenceRequest(
            request_id=f"low-{i}", prompt=f"low priority {i}",
            priority=Priority.LOW, max_tokens=5,
        )
        scheduler.submit(req)

    print(f"  队列大小: {scheduler.queue_size}")

    # 获取批次
    batch = scheduler.next_batch()
    print(f"  第一批: {len(batch)} 个请求")
    for req in batch:
        print(f"    {req.request_id} (priority={req.priority.name})")

    # 继续获取
    batch2 = scheduler.next_batch()
    print(f"  第二批: {len(batch2)} 个请求")

    batch3 = scheduler.next_batch()
    print(f"  第三批: {len(batch3)} 个请求")

    # 超时测试
    expired_req = InferenceRequest(
        request_id="expired-0", prompt="will expire",
        priority=Priority.HIGH, deadline=time.time() - 1,  # 已过期
    )
    scheduler.submit(expired_req)
    batch = scheduler.next_batch()
    print(f"  过期请求批次: {len(batch)} (应为0)")
    stats = scheduler.stats
    print(f"  过期请求数: {stats['total_expired']}")

    # 取消测试
    cancel_req = InferenceRequest(
        request_id="cancel-0", prompt="will cancel", priority=Priority.MEDIUM,
    )
    scheduler.submit(cancel_req)
    scheduler.cancel("cancel-0")
    print(f"  取消请求: {scheduler.stats['total_cancelled']}")

    print(f"  统计: submitted={stats['total_submitted']}, scheduled={stats['total_scheduled']}")
    print(f"    batches={stats['batches_formed']}, avg_batch_size={stats['avg_batch_size']}")
    print("  [PASS] RequestScheduler 测试通过")


def _test_model_partitioner():
    """测试 ModelPartitioner。"""
    _print_separator("测试 ModelPartitioner")
    partitioner = ModelPartitioner(num_layers=12, hidden_size=768)

    # 均匀分区
    plan = partitioner.partition(PartitionStrategy.UNIFORM)
    print(f"  均匀分区: {len(plan.assignments)} 层分配到 {len(partitioner.devices)} 设备")
    print(f"    均衡分数: {plan.balance_score:.4f}")
    print(f"    通信量: {plan.estimated_comm_bytes} bytes")
    for dev, mem in plan.device_memory_usage.items():
        print(f"    {dev}: {mem // (1024**2)} MB")

    # 按层大小分区
    plan2 = partitioner.partition(PartitionStrategy.BY_LAYER_SIZE)
    print(f"  按大小分区: 均衡分数={plan2.balance_score:.4f}")

    # 按计算量分区
    plan3 = partitioner.partition(PartitionStrategy.BY_COMPUTE)
    print(f"  按计算分区: 均衡分数={plan3.balance_score:.4f}")

    # 流水线推理
    logits = partitioner.pipeline_inference([1, 2, 3, 4, 5])
    print(f"  流水线推理: 生成 {len(logits)} 个 logits")

    # 分区信息
    info = partitioner.get_plan_info()
    print(f"  当前方案: strategy={info['strategy']}, stages={info['pipeline_stages']}")
    print("  [PASS] ModelPartitioner 测试通过")


def _test_edge_optimizer():
    """测试 EdgeOptimizer。"""
    _print_separator("测试 EdgeOptimizer")
    optimizer = EdgeOptimizer(num_layers=12, hidden_size=768)

    # 模型分析
    analysis = optimizer.analyze_model()
    print(f"  模型分析: params={analysis['param_count_B']}B, flops/token={analysis['flops_per_token_G']}G")
    print(f"    显存: fp32={analysis['memory']['fp32_MB']}MB, fp16={analysis['memory']['fp16_MB']}MB")
    print(f"          int8={analysis['memory']['int8_MB']}MB, int4={analysis['memory']['int4_MB']}MB")

    # GPU 设备匹配
    gpu_device = DeviceInfo(
        device_id="gpu-0", device_type=DeviceType.GPU,
        total_memory_bytes=24 * 1024**3, compute_flops=300 * 1e12,
        quantization_support=[QuantizationLevel.FP32, QuantizationLevel.FP16, QuantizationLevel.INT8],
    )
    match = optimizer.match_device(gpu_device)
    print(f"  GPU 匹配: quant={match['recommended_quantization']}, tps={match['estimated_tps']}")
    print(f"    fits={match['fits']}, max_kv_tokens={match['max_kv_cache_tokens']}")

    # 边缘设备
    edge_device = DeviceInfo(
        device_id="edge-0", device_type=DeviceType.EDGE,
        total_memory_bytes=4 * 1024**3, compute_flops=10 * 1e12,
        quantization_support=[QuantizationLevel.FP16, QuantizationLevel.INT8, QuantizationLevel.INT4],
    )
    plan = optimizer.generate_inference_plan(edge_device, max_batch_size=4, max_seq_len=512)
    print(f"  边缘设备计划: feasible={plan['feasible']}")
    if plan["feasible"]:
        print(f"    quant={plan['quantization']}, batch={plan['optimal_batch_size']}, tps={plan['estimated_tps']}")
    else:
        print(f"    降级: {plan['degradation']['steps']}")

    # 小设备降级
    tiny_device = DeviceInfo(
        device_id="tiny-0", device_type=DeviceType.CPU,
        total_memory_bytes=512 * 1024**2, compute_flops=1 * 1e12,
        quantization_support=[QuantizationLevel.INT8, QuantizationLevel.INT4],
    )
    degraded = optimizer.degrade_strategy(tiny_device)
    print(f"  小设备降级: {len(degraded['steps'])} 步, final_quant={degraded['final_quantization']}")

    # 设备对比
    comparison = optimizer.compare_devices([gpu_device, edge_device, tiny_device])
    print(f"  设备对比: {len(comparison)} 个设备")
    for c in comparison:
        print(f"    {c['device_id']}: feasible={c['feasible']}, quant={c['quantization']}, tps={c['estimated_tps']}")
    print("  [PASS] EdgeOptimizer 测试通过")


def _test_load_balancer():
    """测试 LoadBalancer。"""
    _print_separator("测试 LoadBalancer")
    lb = LoadBalancer(routing_strategy="least_load", max_nodes=5, min_nodes=1)

    # 注册节点
    for i in range(3):
        lb.register_node(f"endpoint-{i}")

    print(f"  注册节点数: {len(lb.get_nodes())}")

    # 请求路由
    for i in range(6):
        node = lb.route()
        if node:
            lb.complete_request(node.node_id, latency_ms=random.uniform(10, 100))
    print(f"  路由 {lb.get_stats()['total_routed']} 个请求")

    # 健康检查
    lb.check_all_health()
    nodes = lb.get_nodes()
    print(f"  健康检查完成, 可用节点: {len(lb.get_available_nodes())}")

    # 自动扩缩容
    result = lb.auto_scale(total_queue_length=25)
    print(f"  扩容 (queue=25): actions={len(result['actions'])}, nodes={result['current_nodes']}")

    result = lb.auto_scale(total_queue_length=2)
    print(f"  缩容 (queue=2): actions={len(result['actions'])}, nodes={result['current_nodes']}")

    # 故障转移
    nodes = lb.get_nodes()
    if nodes:
        failed = nodes[0].node_id
        backup = lb.failover(failed)
        print(f"  故障转移: {failed} -> {backup.node_id if backup else 'None'}")

    stats = lb.get_stats()
    print(f"  统计: routed={stats['total_routed']}, failovers={stats['total_failovers']}")
    print(f"    scaled_up={stats['total_scaled_up']}, scaled_down={stats['total_scaled_down']}")
    print(f"    avg_latency={stats['avg_latency_ms']}ms, error_rate={stats['error_rate']}")
    print("  [PASS] LoadBalancer 测试通过")


def _test_performance_monitor():
    """测试 PerformanceMonitor。"""
    _print_separator("测试 PerformanceMonitor")
    monitor = PerformanceMonitor(window_seconds=60, alert_latency_threshold_ms=100)

    # 记录请求
    for i in range(20):
        latency = random.uniform(5, 50)
        monitor.record_request(latency, success=True, tokens=random.randint(10, 50))
    # 记录一些高延迟请求触发告警
    for i in range(3):
        monitor.record_request(150.0, success=False, tokens=5)

    # 记录资源
    for i in range(5):
        monitor.record_resource(
            cpu_percent=random.uniform(20, 70),
            memory_percent=random.uniform(30, 60),
            gpu_percent=random.uniform(40, 80),
            gpu_memory_percent=random.uniform(50, 70),
        )

    # 实时指标
    metrics = monitor.get_realtime_metrics()
    print(f"  实时指标: qps={metrics['qps']}, avg_lat={metrics['avg_latency_ms']}ms")
    print(f"    p50={metrics['p50_ms']}ms, p95={metrics['p95_ms']}ms, p99={metrics['p99_ms']}ms")
    print(f"    throughput={metrics['throughput_tps']} tps, error_rate={metrics['error_rate']}")

    # 资源使用
    resources = monitor.get_resource_usage()
    print(f"  资源: cpu={resources['cpu_percent']}%, mem={resources['memory_percent']}%")
    print(f"    gpu={resources['gpu_percent']}%, gpu_mem={resources['gpu_memory_percent']}%")

    # 告警检查
    alerts = monitor.check_alerts()
    print(f"  告警: {len(alerts)} 条")
    for a in alerts:
        print(f"    [{a['severity']}] {a['type']}: {a['message']}")

    # 指标聚合
    agg_1m = monitor.aggregate("1m")
    print(f"  1分钟聚合: count={agg_1m['count']}, qps={agg_1m['qps']}, avg_lat={agg_1m['avg_latency_ms']}ms")

    agg_5m = monitor.aggregate("5m")
    print(f"  5分钟聚合: count={agg_5m['count']}, qps={agg_5m['qps']}")

    # 仪表板数据
    dashboard = monitor.get_dashboard_data()
    print(f"  仪表板: alerts={dashboard['alert_count']}, latency_trend={len(dashboard['latency_trend'])} points")
    print("  [PASS] PerformanceMonitor 测试通过")


def _test_inference_server():
    """测试 InferenceServer。"""
    _print_separator("测试 InferenceServer")
    engine = DistributedInferenceEngine(simulate_latency=False)
    monitor = PerformanceMonitor()
    server = InferenceServer(engine=engine, monitor=monitor, max_concurrent=4)
    server.start()

    # 健康检查
    resp = server.handle_request("GET", "/health")
    print(f"  /health: status={resp['status']}, engine_running={resp['engine_running']}")

    # 指标
    resp = server.handle_request("GET", "/metrics")
    print(f"  /metrics: server_requests={resp['server']['total_requests']}")

    # 模型列表
    resp = server.handle_request("GET", "/v1/models")
    print(f"  /v1/models: {resp['data'][0]['id']}")

    # Completions
    resp = server.handle_request("POST", "/v1/completions", {
        "prompt": "hello world test",
        "max_tokens": 8,
        "strategy": "greedy",
    })
    print(f"  /v1/completions: status={resp['status']}, tokens={resp['usage']['completion_tokens']}")
    print(f"    输出: {resp['choices'][0]['text'][:40]}...")

    # Chat Completions
    resp = server.handle_request("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hello chat"}],
        "max_tokens": 8,
        "temperature": 0.8,
    })
    print(f"  /v1/chat/completions: status={resp['status']}, tokens={resp['usage']['completion_tokens']}")

    # 流式 Completions
    resp = server.handle_request("POST", "/v1/completions", {
        "prompt": "stream test",
        "max_tokens": 5,
        "stream": True,
        "strategy": "temperature",
        "temperature": 0.7,
    })
    print(f"  /v1/completions (stream): status={resp['status']}, chunks={len(resp['chunks'])}")

    # SSE 模拟
    sse_events = server.simulate_sse_stream("sse test", max_tokens=5)
    print(f"  SSE 流: {len(sse_events)} 个事件")
    if sse_events:
        print(f"    首事件: {sse_events[0].strip()[:60]}...")

    # 404 测试
    resp = server.handle_request("GET", "/unknown")
    print(f"  /unknown: error_code={resp['error']['code']}")

    # 服务器信息
    info = server.get_server_info()
    print(f"  服务器信息: routes={len(info['routes'])}, uptime={info['uptime_seconds']}s")

    server.stop()
    print("  [PASS] InferenceServer 测试通过")


def _test_integration():
    """集成测试: 所有组件协同工作。"""
    _print_separator("集成测试: 全链路推理")
    print("  场景: 模拟多请求通过调度器 -> 引擎推理 -> 监控统计的完整流程")

    # 初始化所有组件
    engine = DistributedInferenceEngine(simulate_latency=False, max_batch_size=8)
    scheduler = RequestScheduler(max_batch_size=4)
    monitor = PerformanceMonitor()
    lb = LoadBalancer()

    # 注册推理节点
    node = lb.register_node("integration-endpoint")

    # 提交多个请求
    prompts = ["integration test one", "integration test two", "integration test three"]
    requests = []
    for i, prompt in enumerate(prompts):
        req = InferenceRequest(
            request_id=f"integ-{i}",
            prompt=prompt,
            max_tokens=8,
            priority=Priority.HIGH if i == 0 else Priority.MEDIUM,
            strategy=SamplingStrategy.TEMPERATURE,
            temperature=0.7,
        )
        scheduler.submit(req)
        requests.append(req)

    # 调度并推理
    batch = scheduler.next_batch()
    print(f"  调度批次: {len(batch)} 个请求")

    for req in batch:
        # 负载均衡路由
        target_node = lb.route(req)
        # 引擎推理
        result = engine.generate(
            req.prompt, max_tokens=req.max_tokens,
            strategy=req.strategy, temperature=req.temperature,
            request_id=req.request_id,
        )
        # 监控记录
        monitor.record_request(result.latency_ms, success=True, tokens=result.completion_tokens)
        lb.complete_request(target_node.node_id, result.latency_ms, success=True)
        req.status = RequestStatus.COMPLETED
        print(f"    {req.request_id}: {result.completion_tokens} tokens, {result.latency_ms:.2f}ms")

    # 输出综合统计
    print(f"\n  引擎统计: {engine.get_stats()['completed']} completed, avg={engine.get_stats()['avg_latency_ms']:.2f}ms")
    print(f"  调度器统计: scheduled={scheduler.stats['total_scheduled']}, batches={scheduler.stats['batches_formed']}")
    lb_stats = lb.get_stats()
    print(f"  负载均衡: routed={lb_stats['total_routed']}, nodes={lb_stats['num_nodes']}")
    rt = monitor.get_realtime_metrics()
    print(f"  监控: qps={rt['qps']}, p50={rt['p50_ms']}ms, p95={rt['p95_ms']}ms, tps={rt['throughput_tps']}")
    print("  [PASS] 集成测试通过")


def main():
    """主测试函数。"""
    print("=" * 60)
    print("  灵元模型项目 - 第22模块: 分布式推理引擎")
    print("  自测程序开始")
    print("=" * 60)

    random.seed(42)  # 固定随机种子保证可复现

    tests = [
        ("KVCacheManager", _test_kv_cache_manager),
        ("DistributedInferenceEngine", _test_inference_engine),
        ("RequestScheduler", _test_request_scheduler),
        ("ModelPartitioner", _test_model_partitioner),
        ("EdgeOptimizer", _test_edge_optimizer),
        ("LoadBalancer", _test_load_balancer),
        ("PerformanceMonitor", _test_performance_monitor),
        ("InferenceServer", _test_inference_server),
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
