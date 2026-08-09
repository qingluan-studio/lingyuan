#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# LINGYUAN MODEL - PART 11
# 推理服务模块 (Inference Service)
#
# 对应52项清单 #19-24, 共6个子系统:
#   1. InferenceEngine             真实推理引擎
#   2. ContinuousBatcher           连续批处理
#   3. StreamingOutput             流式输出
#   4. InferenceCache              推理结果缓存
#   5. FunctionCaller              Function Calling
#   6. ChatTemplateManager         对话模板系统
#
# 纯Python标准库实现(零外部依赖)。
# 本文件在 lingyuan_full.py 之后加载, 可使用全局变量: DATA_DIR / LOG_DIR / CONFIG_DIR。
# ============================================================

import uuid
import math
import random
import json
import os
import time
import re
import hashlib
import threading
from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple, Generator, Union
from datetime import datetime


# ============================================================
# 全局目录容错: 若 lingyuan_full.py 未提供则自行创建默认目录
# ============================================================
try:
    _ = DATA_DIR    # noqa: F821
    _ = LOG_DIR     # noqa: F821
    _ = CONFIG_DIR  # noqa: F821
except NameError:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    DATA_DIR = os.path.join(_BASE_DIR, 'lingyuan_data')
    LOG_DIR = os.path.join(_BASE_DIR, 'lingyuan_logs')
    CONFIG_DIR = os.path.join(_BASE_DIR, 'lingyuan_config')
    for _d in (DATA_DIR, LOG_DIR, CONFIG_DIR):
        os.makedirs(_d, exist_ok=True)

# 推理服务子目录
INFERENCE_LOG_DIR = os.path.join(LOG_DIR, "inference")
INFERENCE_CACHE_DIR = os.path.join(DATA_DIR, "inference_cache")
for _d in (INFERENCE_LOG_DIR, INFERENCE_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)


# ============================================================
# 辅助函数 (纯Python, 零依赖)
# ============================================================

def _softmax(logits: List[float]) -> List[float]:
    """一维向量 softmax (数值稳定)"""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / s for e in exps]


def _stable_hash(text: str) -> str:
    """稳定的字符串哈希 (基于 md5, 跨进程一致)"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    """当前时间 ISO 字符串"""
    return datetime.now().isoformat()


def _safe_call(fn: Callable, *args, **kwargs) -> Tuple[bool, Any]:
    """安全调用函数, 返回 (成功, 结果或异常信息)"""
    try:
        return True, fn(*args, **kwargs)
    except Exception as e:
        return False, str(e)


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数 (CJK字符≈1 token, 4个ASCII≈1 token)"""
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            cjk += 1
        elif 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF:
            cjk += 1
        else:
            other += 1
    return cjk + max(1, other // 4)


# ============================================================
# 1. InferenceEngine [真实推理引擎]
# ============================================================

@dataclass
class InferenceConfig:
    """推理配置参数"""
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    stop_words: List[Any] = field(default_factory=list)   # str 或 List[int]
    use_cache: bool = True
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceRecord:
    """单次推理记录"""
    record_id: str
    input_ids: List[int]
    output_ids: List[int]
    config: Dict[str, Any]
    latency: float
    tokens_per_sec: float
    cache_hit_rate: float
    timestamp: str
    prompt_len: int = 0
    output_len: int = 0
    forward_calls: int = 0
    cache_hits: int = 0
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DefaultMockModel:
    """默认模拟模型 (零依赖, 用于无外部模型时的推理演示)

    采用简化的大词表 + 状态转移表, 使输出具有一定连贯性。
    接口与 LingyuanTransformerModel 兼容:
      - forward(input_ids, last_token_only=False, **kwargs) -> List[List[float]]
      - forward_with_cache(input_ids, cache) -> (logits, cache)
      - vocab_size / num_layers / num_kv_heads / head_dim 属性
    """

    def __init__(self, vocab_size: int = 1024, seed: int = 42):
        self.vocab_size = vocab_size
        self._rng = random.Random(seed)
        # 构建简单转移表: 每个 token 偏好若干后续 token
        self._transitions: Dict[int, Tuple[List[int], List[float]]] = {}
        for t in range(vocab_size):
            n = self._rng.randint(3, 6)
            preferred = self._rng.sample(range(vocab_size), n)
            weights = sorted([self._rng.random() for _ in range(n)], reverse=True)
            self._transitions[t] = (preferred, weights)
        # 兼容属性
        self.num_layers = 4
        self.num_kv_heads = 4
        self.head_dim = 32
        self.hidden_dim = 128
        self._forward_count = 0

    def _compute_logits(self, input_ids: List[int]) -> List[float]:
        """计算最后位置的 logits"""
        if not input_ids:
            return [0.0] * self.vocab_size
        last_id = input_ids[-1] % self.vocab_size
        preferred, weights = self._transitions[last_id]
        logits = [-5.0] * self.vocab_size
        for tok, w in zip(preferred, weights):
            logits[tok] = w * 10.0
        # 加入位置相关的微扰, 使输出更丰富
        pos = len(input_ids)
        for i in range(self.vocab_size):
            logits[i] += math.sin(pos * 0.13 + i * 0.007) * 0.3
        return logits

    def forward(self, input_ids: List[int], last_token_only: bool = False,
                **kwargs) -> List[List[float]]:
        """前向传播 -> logits (seq × vocab 或 1 × vocab)"""
        if not input_ids:
            return [[0.0] * self.vocab_size]
        self._forward_count += 1
        if last_token_only:
            return [self._compute_logits(input_ids)]
        return [self._compute_logits(input_ids[:i + 1]) for i in range(len(input_ids))]

    def forward_with_cache(self, input_ids: List[int],
                           cache: Any = None) -> Tuple[List[List[float]], Any]:
        """带缓存的前向传播 (模拟: 缓存仅记录已处理长度)"""
        logits = self.forward(input_ids, last_token_only=True)
        # 模拟缓存对象
        if cache is None:
            cache = {"processed_len": 0}
        cache = {"processed_len": cache.get("processed_len", 0) + len(input_ids)}
        return logits, cache


class InferenceEngine:
    """真实推理引擎

    功能:
    - 前向推理: infer(input_ids, ...) -> output_ids
    - KV Cache 管理: 每步增量计算, 复用已计算的 K/V
    - 采样配置: temperature / top_k / top_p / repetition_penalty
    - 批量推理: batch_infer(inputs) -> outputs
    - 推理统计: latency / token_per_sec / cache_hit_rate
    - 支持外部模型 (有 forward 方法的任意对象) 或默认模拟模型
    - 停止条件: max_tokens / EOS token / stop_words
    - 日志记录: 每次推理的输入/输出/参数/耗时
    """

    def __init__(self,
                 model: Any = None,
                 config: Optional[InferenceConfig] = None,
                 log_dir: Optional[str] = None,
                 decode_fn: Optional[Callable[[List[int]], str]] = None):
        self.model = model if model is not None else DefaultMockModel()
        self.config = config or InferenceConfig()
        self.log_dir = log_dir or INFERENCE_LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        # decode 函数: token ids -> 文本 (用于 stop_words 字符串匹配)
        self.decode_fn = decode_fn
        # 统计
        self._records: deque = deque(maxlen=500)
        self._total_forward_calls = 0
        self._total_cache_hits = 0
        self._total_tokens_generated = 0
        self._total_latency = 0.0
        self._infer_count = 0
        self._lock = threading.Lock()
        # 日志文件
        self._log_file = os.path.join(self.log_dir, "inference_log.jsonl")
        # 探测模型能力
        self._supports_cache = hasattr(self.model, "forward_with_cache") and callable(
            getattr(self.model, "forward_with_cache", None))
        # 采样随机源
        self._rng = random.Random(self.config.seed)

    # ---------- 模型前向 ----------

    def _get_logits(self, input_ids: List[int], cache: Any,
                    use_cache: bool) -> Tuple[List[List[float]], Any, bool]:
        """获取 logits, 返回 (logits, cache, cache_hit)

        cache_hit=True 表示本次前向复用了已有缓存 (增量解码)。
        """
        cache_hit = False
        if use_cache and self._supports_cache:
            logits, cache = self.model.forward_with_cache(input_ids, cache)
            cache_hit = cache is not None
        else:
            # 不支持缓存: 每次全量前向
            try:
                logits = self.model.forward(input_ids, last_token_only=True)
            except TypeError:
                logits = self.model.forward(input_ids)
            cache_hit = False
        return logits, cache, cache_hit

    # ---------- 采样 ----------

    def _sample(self, logits: List[float],
                temperature: float, top_k: int, top_p: float,
                repetition_penalty: float,
                prev_tokens: List[int]) -> int:
        """组合采样: temperature + top_k + top_p + repetition_penalty"""
        if not logits:
            return 0
        # 贪婪模式
        if temperature <= 0:
            best_id, best_val = 0, logits[0]
            for i, v in enumerate(logits):
                if v > best_val:
                    best_val, best_id = v, i
            return best_id

        vec = list(logits)

        # 1. 重复惩罚
        if prev_tokens and repetition_penalty != 1.0:
            seen = set(prev_tokens)
            for tid in seen:
                if 0 <= tid < len(vec):
                    if vec[tid] > 0:
                        vec[tid] = vec[tid] / repetition_penalty
                    else:
                        vec[tid] = vec[tid] * repetition_penalty

        # 2. 温度缩放
        vec = [v / temperature for v in vec]

        # 3. Top-K 过滤
        if 0 < top_k < len(vec):
            indexed = sorted(range(len(vec)), key=lambda i: vec[i], reverse=True)
            keep = set(indexed[:top_k])
            vec = [v if i in keep else -1e9 for i, v in enumerate(vec)]

        # 4. Top-P (nucleus) 过滤
        if 0.0 < top_p < 1.0:
            probs = _softmax(vec)
            indexed = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            cumsum, keep = 0.0, set()
            for idx in indexed:
                keep.add(idx)
                cumsum += probs[idx]
                if cumsum >= top_p:
                    break
            vec = [v if i in keep else -1e9 for i, v in enumerate(vec)]

        # 5. 转概率并采样
        probs = _softmax(vec)
        r = self._rng.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i
        return len(probs) - 1

    # ---------- 停止条件检测 ----------

    def _check_stop(self, generated: List[int],
                    eos_token_id: Optional[int],
                    stop_words: List[Any],
                    decode_fn: Optional[Callable]) -> Tuple[bool, int]:
        """检测是否满足停止条件, 返回 (是否停止, 截断长度)

        截断长度 = 保留的 token 数 (排除停止序列)。
        """
        # EOS token
        if eos_token_id is not None and generated and generated[-1] == eos_token_id:
            return True, len(generated) - 1
        # stop_words
        for sw in stop_words:
            if isinstance(sw, str):
                if decode_fn is not None:
                    text = decode_fn(generated)
                    if text.endswith(sw):
                        return True, len(generated)  # 字符串停止词不截断 token
            elif isinstance(sw, (list, tuple)):
                seq = list(sw)
                n = len(seq)
                if n > 0 and len(generated) >= n and generated[-n:] == seq:
                    return True, len(generated) - n
        return False, len(generated)

    # ---------- 核心推理 ----------

    def infer(self,
              input_ids: List[int],
              max_new_tokens: Optional[int] = None,
              temperature: Optional[float] = None,
              top_k: Optional[int] = None,
              top_p: Optional[float] = None,
              repetition_penalty: Optional[float] = None,
              eos_token_id: Optional[int] = None,
              stop_words: Optional[List[Any]] = None,
              use_cache: Optional[bool] = None,
              decode_fn: Optional[Callable] = None) -> List[int]:
        """前向推理 (自回归生成)

        Args:
            input_ids: 输入 prompt 的 token id 列表
            max_new_tokens: 最大生成 token 数 (None 取默认配置)
            temperature / top_k / top_p / repetition_penalty: 采样参数
            eos_token_id: 结束 token id
            stop_words: 停止词列表 (str 需配合 decode_fn, 或 List[int] token 序列)
            use_cache: 是否使用 KV 缓存
            decode_fn: token ids -> 文本 (用于 stop_words 字符串匹配)

        Returns:
            生成的 token id 列表 (仅新生成部分, 不含停止序列)
        """
        # 参数合并 (显式参数优先于配置)
        _max = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        _temp = temperature if temperature is not None else self.config.temperature
        _topk = top_k if top_k is not None else self.config.top_k
        _topp = top_p if top_p is not None else self.config.top_p
        _rep = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        _eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        _stops = stop_words if stop_words is not None else self.config.stop_words
        _cache = use_cache if use_cache is not None else self.config.use_cache
        _decode = decode_fn if decode_fn is not None else self.decode_fn

        if not input_ids:
            return []

        start_time = time.time()
        generated: List[int] = list(input_ids)
        new_tokens: List[int] = []
        cache: Any = None
        forward_calls = 0
        cache_hits = 0
        stop_reason = "max_tokens"

        # 首次前向 (处理完整 prompt)
        logits, cache, hit = self._get_logits(generated, cache, _cache)
        forward_calls += 1
        if hit:
            cache_hits += 1

        for step in range(_max):
            last_logits = logits[-1] if logits else [0.0]

            # 采样下一个 token
            next_id = self._sample(
                last_logits, _temp, _topk, _topp, _rep, generated)
            new_tokens.append(next_id)
            generated.append(next_id)

            # 检查停止条件
            stopped, trim_len = self._check_stop(generated, _eos, _stops, _decode)
            if stopped:
                if _eos is not None and generated[-1] == _eos:
                    stop_reason = "eos"
                else:
                    stop_reason = "stop_words"
                # 截断停止序列
                if trim_len < len(generated):
                    removed = len(generated) - trim_len
                    new_tokens = new_tokens[:len(new_tokens) - removed]
                break

            # 下一轮前向 (增量)
            if step < _max - 1:
                logits, cache, hit = self._get_logits([next_id], cache, _cache)
                forward_calls += 1
                if hit:
                    cache_hits += 1

        elapsed = time.time() - start_time
        tps = len(new_tokens) / max(elapsed, 1e-6)
        hit_rate = cache_hits / max(forward_calls, 1)

        # 记录
        record = InferenceRecord(
            record_id=str(uuid.uuid4())[:12],
            input_ids=list(input_ids),
            output_ids=list(new_tokens),
            config={
                "max_new_tokens": _max, "temperature": _temp,
                "top_k": _topk, "top_p": _topp,
                "repetition_penalty": _rep, "use_cache": _cache,
                "eos_token_id": _eos, "stop_words": [str(s) for s in _stops],
            },
            latency=round(elapsed, 4),
            tokens_per_sec=round(tps, 2),
            cache_hit_rate=round(hit_rate, 4),
            timestamp=_now_iso(),
            prompt_len=len(input_ids),
            output_len=len(new_tokens),
            forward_calls=forward_calls,
            cache_hits=cache_hits,
            stop_reason=stop_reason,
        )
        with self._lock:
            self._records.append(record)
            self._total_forward_calls += forward_calls
            self._total_cache_hits += cache_hits
            self._total_tokens_generated += len(new_tokens)
            self._total_latency += elapsed
            self._infer_count += 1
        self._log_record(record)

        return new_tokens

    # ---------- 批量推理 ----------

    def batch_infer(self, inputs: List[List[int]], **kwargs) -> List[List[int]]:
        """批量推理: 对多个输入依次推理

        Args:
            inputs: 多个 prompt 的 token id 列表
            **kwargs: 传递给 infer 的采样参数

        Returns:
            每个输入对应的输出 token id 列表
        """
        results: List[List[int]] = []
        for input_ids in inputs:
            out = self.infer(input_ids, **kwargs)
            results.append(out)
        return results

    # ---------- 日志 ----------

    def _log_record(self, record: InferenceRecord):
        """将推理记录写入日志文件 (JSONL)"""
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---------- 统计与仪表盘 ----------

    def get_stats(self) -> Dict[str, Any]:
        """推理统计"""
        avg_latency = self._total_latency / max(self._infer_count, 1)
        avg_tps = (self._total_tokens_generated / max(self._total_latency, 1e-6))
        overall_hit_rate = self._total_cache_hits / max(self._total_forward_calls, 1)
        return {
            "infer_count": self._infer_count,
            "total_tokens_generated": self._total_tokens_generated,
            "total_forward_calls": self._total_forward_calls,
            "total_cache_hits": self._total_cache_hits,
            "cache_hit_rate": round(overall_hit_rate, 4),
            "avg_latency_sec": round(avg_latency, 4),
            "avg_tokens_per_sec": round(avg_tps, 2),
            "model_type": type(self.model).__name__,
            "supports_cache": self._supports_cache,
            "recent_records": len(self._records),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """推理仪表盘 (含详细分解)"""
        records = list(self._records)
        recent = [r.to_dict() for r in records[-10:]]
        # 按停止原因统计
        stop_reasons: Dict[str, int] = {}
        for r in records:
            stop_reasons[r.stop_reason] = stop_reasons.get(r.stop_reason, 0) + 1
        # 平均输出长度
        avg_out_len = (sum(r.output_len for r in records) / len(records)
                       if records else 0.0)
        return {
            "engine": "InferenceEngine",
            "model": type(self.model).__name__,
            "supports_cache": self._supports_cache,
            "config": self.config.to_dict(),
            "stats": self.get_stats(),
            "stop_reason_distribution": stop_reasons,
            "avg_output_length": round(avg_out_len, 1),
            "log_file": self._log_file,
            "recent_records": recent,
        }


# ============================================================
# 2. ContinuousBatcher [连续批处理]
# ============================================================

@dataclass
class BatchRequest:
    """批处理请求"""
    request_id: str
    input_ids: List[int]
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    stop_words: List[Any] = field(default_factory=list)
    priority: int = 0                          # 数值越大优先级越高
    status: str = "waiting"                    # waiting / running / completed / failed
    output_ids: List[int] = field(default_factory=list)
    cache: Any = None                          # 每请求独立 KV 缓存
    created_at: str = field(default_factory=_now_iso)
    completed_at: str = ""
    error: str = ""
    step_count: int = 0
    forward_calls: int = 0
    cache_hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("cache", None)
        return d


class ContinuousBatcher:
    """连续批处理调度器

    功能:
    - 动态拼 batch: 新请求加入当前 batch, 已完成的移出
    - 请求队列: FIFO + 优先级
    - padding 管理: batch 内不同长度对齐
    - 迭代调度: schedule() -> 当前 batch 执行一步
    - 请求状态: waiting / running / completed
    - 吞吐量优化: 尽量填满 batch
    - 统计: avg_batch_size / throughput / queue_length
    """

    def __init__(self,
                 engine: Optional[InferenceEngine] = None,
                 max_batch_size: int = 8,
                 max_seq_len: int = 2048):
        self.engine = engine if engine is not None else InferenceEngine()
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        # 等待队列 (优先级排序)
        self._waiting: List[BatchRequest] = []
        # 正在运行的请求
        self._running: List[BatchRequest] = []
        # 已完成请求 (保留最近一批)
        self._completed: deque = deque(maxlen=200)
        # 全部请求索引
        self._requests: Dict[str, BatchRequest] = {}
        self._lock = threading.Lock()
        # 统计
        self._total_steps = 0
        self._total_tokens_generated = 0
        self._total_batch_slots_used = 0   # 每步 batch 内的请求数之和
        self._total_schedules = 0
        self._start_time = time.time()

    # ---------- 请求提交 ----------

    def submit(self, input_ids: List[int], **kwargs) -> str:
        """提交一个推理请求, 返回 request_id"""
        req = BatchRequest(
            request_id=str(uuid.uuid4())[:12],
            input_ids=list(input_ids),
            **kwargs,
        )
        with self._lock:
            self._waiting.append(req)
            self._requests[req.request_id] = req
        return req.request_id

    def submit_request(self, request: BatchRequest) -> str:
        """提交一个已构造的 BatchRequest"""
        with self._lock:
            self._waiting.append(request)
            self._requests[request.request_id] = request
        return request.request_id

    # ---------- 批次管理 ----------

    def _fill_batch(self):
        """从等待队列填充运行批次 (按优先级降序)"""
        if not self._waiting:
            return
        # 按优先级排序 (高优先级先入)
        self._waiting.sort(key=lambda r: r.priority, reverse=True)
        while self._waiting and len(self._running) < self.max_batch_size:
            req = self._waiting.pop(0)
            req.status = "running"
            self._running.append(req)

    def _pad_batch(self, sequences: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """对 batch 内序列进行 padding, 返回 (padded, attention_mask)

        pad_token_id 固定为 0, attention_mask: 1=有效, 0=padding。
        """
        if not sequences:
            return [], []
        max_len = max(len(s) for s in sequences)
        padded, masks = [], []
        for s in sequences:
            pad_len = max_len - len(s)
            padded.append(list(s) + [0] * pad_len)
            masks.append([1] * len(s) + [0] * pad_len)
        return padded, masks

    def _step_request(self, req: BatchRequest) -> bool:
        """对单个请求执行一步推理, 返回是否完成"""
        if req.status != "running":
            return req.status == "completed"
        # 首步: 处理完整 prompt
        if req.step_count == 0:
            full_ids = list(req.input_ids)
            logits, req.cache, hit = self.engine._get_logits(
                full_ids, req.cache, True)
            req.forward_calls += 1
            if hit:
                req.cache_hits += 1
        else:
            # 增量: 只传入上一个生成的 token
            last_token = req.output_ids[-1] if req.output_ids else req.input_ids[-1]
            logits, req.cache, hit = self.engine._get_logits(
                [last_token], req.cache, True)
            req.forward_calls += 1
            if hit:
                req.cache_hits += 1

        # 采样
        generated = list(req.input_ids) + list(req.output_ids)
        last_logits = logits[-1] if logits else [0.0]
        next_id = self.engine._sample(
            last_logits, req.temperature, req.top_k, req.top_p,
            req.repetition_penalty, generated)
        req.output_ids.append(next_id)
        req.step_count += 1

        # 检查停止条件
        all_ids = list(req.input_ids) + list(req.output_ids)
        stopped, trim_len = self.engine._check_stop(
            all_ids, req.eos_token_id, req.stop_words, self.engine.decode_fn)

        if stopped or req.step_count >= req.max_new_tokens:
            req.status = "completed"
            req.completed_at = _now_iso()
            # 截断停止序列
            if stopped and trim_len < len(all_ids):
                removed = len(all_ids) - trim_len
                req.output_ids = req.output_ids[:max(0, len(req.output_ids) - removed)]
            return True
        return False

    # ---------- 调度 ----------

    def schedule(self) -> Dict[str, Any]:
        """执行一步调度: 填充 batch, 对运行中请求各推理一步, 移出已完成请求

        Returns:
            本次调度信息 (batch_size, completed_ids 等)
        """
        with self._lock:
            self._fill_batch()

            # 记录 padding 信息 (用于统计)
            if self._running:
                seqs = [list(r.input_ids) + list(r.output_ids) for r in self._running]
                padded, masks = self._pad_batch(seqs)
                padding_tokens = sum(m.count(0) for m in masks)
            else:
                padding_tokens = 0

            batch_size = len(self._running)
            completed_ids: List[str] = []

            # 对每个运行请求执行一步
            still_running: List[BatchRequest] = []
            for req in self._running:
                done = self._step_request(req)
                if done:
                    req.status = "completed"
                    self._completed.append(req)
                    completed_ids.append(req.request_id)
                else:
                    still_running.append(req)
            self._running = still_running

            # 更新统计
            self._total_steps += 1
            self._total_schedules += 1
            self._total_batch_slots_used += batch_size
            self._total_tokens_generated += batch_size  # 每个请求生成1个token

            return {
                "schedule_id": self._total_schedules,
                "batch_size": batch_size,
                "queue_length": len(self._waiting),
                "completed_request_ids": completed_ids,
                "padding_tokens": padding_tokens,
                "running_ids": [r.request_id for r in self._running],
            }

    def run_until_complete(self, max_iterations: int = 10000) -> Dict[str, Any]:
        """持续调度直到所有请求完成"""
        iterations = 0
        while iterations < max_iterations:
            if not self._waiting and not self._running:
                break
            self.schedule()
            iterations += 1
        return {
            "iterations": iterations,
            "total_completed": len(self._completed),
            "queue_remaining": len(self._waiting),
            "running_remaining": len(self._running),
        }

    # ---------- 查询 ----------

    def get_request(self, request_id: str) -> Optional[BatchRequest]:
        """获取请求"""
        return self._requests.get(request_id)

    def get_request_status(self, request_id: str) -> Optional[str]:
        req = self._requests.get(request_id)
        return req.status if req else None

    def get_result(self, request_id: str) -> Optional[List[int]]:
        """获取已完成请求的输出"""
        req = self._requests.get(request_id)
        if req and req.status == "completed":
            return list(req.output_ids)
        return None

    # ---------- 统计与仪表盘 ----------

    def get_stats(self) -> Dict[str, Any]:
        """批处理统计"""
        elapsed = time.time() - self._start_time
        avg_batch = self._total_batch_slots_used / max(self._total_schedules, 1)
        throughput = self._total_tokens_generated / max(elapsed, 1e-6)
        return {
            "max_batch_size": self.max_batch_size,
            "queue_length": len(self._waiting),
            "running_count": len(self._running),
            "completed_count": len(self._completed),
            "total_requests": len(self._requests),
            "total_steps": self._total_steps,
            "total_schedules": self._total_schedules,
            "avg_batch_size": round(avg_batch, 2),
            "throughput_tokens_per_sec": round(throughput, 2),
            "total_tokens_generated": self._total_tokens_generated,
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """批处理仪表盘"""
        running_info = []
        for r in self._running:
            running_info.append({
                "request_id": r.request_id,
                "step": r.step_count,
                "max_new_tokens": r.max_new_tokens,
                "progress": round(r.step_count / max(r.max_new_tokens, 1), 2),
                "priority": r.priority,
            })
        completed_info = []
        for r in list(self._completed)[-10:]:
            completed_info.append({
                "request_id": r.request_id,
                "output_len": len(r.output_ids),
                "step_count": r.step_count,
                "forward_calls": r.forward_calls,
                "cache_hits": r.cache_hits,
            })
        return {
            "batcher": "ContinuousBatcher",
            "max_batch_size": self.max_batch_size,
            "stats": self.get_stats(),
            "running_requests": running_info,
            "recent_completed": completed_info,
            "engine_model": type(self.engine.model).__name__ if self.engine else None,
        }


# ============================================================
# 3. StreamingOutput [流式输出]
# ============================================================

@dataclass
class StreamConfig:
    """流式输出配置"""
    format: str = "sse"             # sse / websocket / raw
    buffer_size: int = 1            # 每次刷新累积的 token 数
    flush_interval: float = 0.0     # 刷新间隔秒数 (0=无延迟)
    eos_token_id: Optional[int] = None
    send_usage: bool = True         # 是否在末尾发送用量统计
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StreamingOutput:
    """流式输出管理器

    功能:
    - SSE 格式: data: {json}\\n\\n
    - 逐 token 生成: stream_generate(input_ids) -> generator
    - 缓冲管理: 控制输出速率
    - 断流处理: 客户端断开检测
    - 支持回调: on_token(token), on_complete(), on_error(err)
    - WebSocket 消息格式: {type:"token", content:"xxx"}
    """

    def __init__(self,
                 engine: Optional[InferenceEngine] = None,
                 config: Optional[StreamConfig] = None,
                 decode_fn: Optional[Callable[[List[int]], str]] = None):
        self.engine = engine if engine is not None else InferenceEngine()
        self.config = config or StreamConfig()
        self.decode_fn = decode_fn or self.engine.decode_fn
        # 回调
        self._on_token: Optional[Callable] = None
        self._on_complete: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        # 断开标志 (外部设置 True 表示客户端断开)
        self._disconnected = False
        self._lock = threading.Lock()
        # 统计
        self._stream_count = 0
        self._total_tokens_streamed = 0
        self._total_disconnects = 0
        self._total_errors = 0

    # ---------- 回调设置 ----------

    def set_callbacks(self,
                      on_token: Optional[Callable] = None,
                      on_complete: Optional[Callable] = None,
                      on_error: Optional[Callable] = None):
        """设置流式回调"""
        if on_token is not None:
            self._on_token = on_token
        if on_complete is not None:
            self._on_complete = on_complete
        if on_error is not None:
            self._on_error = on_error

    # ---------- 断开检测 ----------

    def disconnect(self):
        """标记客户端已断开"""
        with self._lock:
            self._disconnected = True

    def is_disconnected(self) -> bool:
        """检查客户端是否已断开"""
        return self._disconnected

    def _reset_disconnect(self):
        with self._lock:
            self._disconnected = False

    # ---------- 格式化 ----------

    def _format_sse(self, data: Dict[str, Any]) -> str:
        """SSE 格式: data: {json}\\n\\n"""
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _format_ws(self, data: Dict[str, Any]) -> str:
        """WebSocket 消息格式: {type:..., content:...}"""
        return json.dumps(data, ensure_ascii=False)

    def _format_raw(self, token_id: int, text: str = "") -> str:
        """原始格式"""
        return text if text else str(token_id)

    def _format_chunk(self, token_id: int, text: str, is_final: bool = False) -> str:
        """根据配置格式化输出块"""
        fmt = self.config.format
        if fmt == "sse":
            data = {"type": "token", "content": text, "token_id": token_id}
            if is_final:
                data["type"] = "done"
            return self._format_sse(data)
        elif fmt == "websocket":
            data = {"type": "done" if is_final else "token",
                     "content": text, "token_id": token_id}
            return self._format_ws(data)
        else:  # raw
            if is_final:
                return ""
            return self._format_raw(token_id, text)

    # ---------- 核心流式生成 ----------

    def stream_generate(self,
                        input_ids: List[int],
                        max_new_tokens: int = 128,
                        temperature: float = 1.0,
                        top_k: int = 0,
                        top_p: float = 1.0,
                        repetition_penalty: float = 1.0,
                        eos_token_id: Optional[int] = None,
                        use_cache: bool = True,
                        decode_fn: Optional[Callable] = None) -> Generator[str, None, List[int]]:
        """逐 token 流式生成

        Yields:
            格式化的输出块 (SSE / WebSocket / raw)

        Returns:
            完整的生成 token id 列表 (generator 结束后获取)
        """
        self._reset_disconnect()
        _decode = decode_fn if decode_fn is not None else self.decode_fn
        _eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        generated: List[int] = list(input_ids)
        new_tokens: List[int] = []
        cache: Any = None
        buffer: List[int] = []
        last_flush = time.time()
        stream_start = time.time()
        error_msg = ""

        try:
            # 首次前向
            logits, cache, _ = self.engine._get_logits(generated, cache, use_cache)

            for step in range(max_new_tokens):
                # 断开检测
                if self.is_disconnected():
                    self._total_disconnects += 1
                    break

                last_logits = logits[-1] if logits else [0.0]
                next_id = self.engine._sample(
                    last_logits, temperature, top_k, top_p,
                    repetition_penalty, generated)
                new_tokens.append(next_id)
                generated.append(next_id)
                buffer.append(next_id)

                # EOS 检测
                if _eos is not None and next_id == _eos:
                    break

                # 缓冲刷新
                should_flush = (len(buffer) >= self.config.buffer_size or
                                (self.config.flush_interval > 0 and
                                 time.time() - last_flush >= self.config.flush_interval))
                if should_flush:
                    for tid in buffer:
                        text = _decode([tid]) if _decode else ""
                        chunk = self._format_chunk(tid, text, is_final=False)
                        if self._on_token:
                            ok, err = _safe_call(self._on_token, tid, text)
                            if not ok:
                                error_msg = err
                                raise RuntimeError(err)
                        yield chunk
                    buffer = []
                    last_flush = time.time()

                # 下一轮前向
                if step < max_new_tokens - 1:
                    logits, cache, _ = self.engine._get_logits(
                        [next_id], cache, use_cache)

            # 刷新剩余缓冲
            for tid in buffer:
                text = _decode([tid]) if _decode else ""
                chunk = self._format_chunk(tid, text, is_final=False)
                if self._on_token:
                    _safe_call(self._on_token, tid, text)
                yield chunk
            buffer = []

            # 发送结束标记
            elapsed = time.time() - stream_start
            if self.config.send_usage:
                usage = {
                    "type": "usage",
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": len(new_tokens),
                    "total_tokens": len(input_ids) + len(new_tokens),
                    "elapsed_sec": round(elapsed, 4),
                    "tokens_per_sec": round(len(new_tokens) / max(elapsed, 1e-6), 2),
                }
                if self.config.format == "sse":
                    yield self._format_sse(usage)
                elif self.config.format == "websocket":
                    yield self._format_ws(usage)

            yield self._format_chunk(0, "", is_final=True)

        except Exception as e:
            error_msg = str(e)
            self._total_errors += 1
            if self._on_error:
                _safe_call(self._on_error, error_msg)
            err_data = {"type": "error", "content": error_msg}
            if self.config.format == "sse":
                yield self._format_sse(err_data)
            elif self.config.format == "websocket":
                yield self._format_ws(err_data)
        finally:
            with self._lock:
                self._stream_count += 1
                self._total_tokens_streamed += len(new_tokens)
            if self._on_complete:
                _safe_call(self._on_complete)

        return new_tokens

    def stream(self, input_ids: List[int], **kwargs) -> Generator[str, None, List[int]]:
        """stream_generate 的别名"""
        return self.stream_generate(input_ids, **kwargs)

    # ---------- 统计与仪表盘 ----------

    def get_stats(self) -> Dict[str, Any]:
        """流式输出统计"""
        return {
            "stream_count": self._stream_count,
            "total_tokens_streamed": self._total_tokens_streamed,
            "total_disconnects": self._total_disconnects,
            "total_errors": self._total_errors,
            "format": self.config.format,
            "buffer_size": self.config.buffer_size,
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """流式输出仪表盘"""
        return {
            "streaming": "StreamingOutput",
            "config": self.config.to_dict(),
            "stats": self.get_stats(),
            "has_callbacks": {
                "on_token": self._on_token is not None,
                "on_complete": self._on_complete is not None,
                "on_error": self._on_error is not None,
            },
            "is_disconnected": self.is_disconnected(),
            "engine_model": type(self.engine.model).__name__ if self.engine else None,
        }


# ============================================================
# 4. InferenceCache [推理结果缓存]
# ============================================================

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str                       # 精确匹配键 (prompt 哈希)
    prompt: str                    # 原始 prompt 文本
    response: Any                  # 缓存的响应 (token ids 或文本)
    created_at: float              # 创建时间戳
    last_access: float             # 最后访问时间戳
    access_count: int = 0          # 访问次数
    token_count: int = 0           # 响应 token 数
    semantic_fingerprint: str = "" # 语义指纹 (用于语义缓存)
    prompt_length: int = 0         # prompt 长度
    ttl: float = 3600.0            # 有效期秒数

    def is_expired(self) -> bool:
        """是否过期"""
        if self.ttl <= 0:
            return False
        return time.time() - self.created_at > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class InferenceCache:
    """推理结果缓存

    功能:
    - LRU 缓存: 最近使用的 (prompt -> response)
    - 精确匹配: prompt 完全相同直接返回
    - 语义缓存: 相似度 > 阈值则返回 (简化版: 字符 n-gram Jaccard + 长度比)
    - TTL 过期: 可配置缓存有效期
    - 缓存统计: hit_rate / size / memory_usage
    - 手动失效: invalidate(pattern)
    - 持久化: save / load 缓存到磁盘
    """

    def __init__(self,
                 max_size: int = 1000,
                 ttl: float = 3600.0,
                 semantic_threshold: float = 0.85,
                 enable_semantic: bool = True,
                 cache_dir: Optional[str] = None):
        self.max_size = max_size
        self.ttl = ttl
        self.semantic_threshold = semantic_threshold
        self.enable_semantic = enable_semantic
        self.cache_dir = cache_dir or INFERENCE_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        # OrderedDict: key -> CacheEntry (按访问顺序, 末尾为最近使用)
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        # 统计
        self._hit_count = 0
        self._miss_count = 0
        self._semantic_hit_count = 0
        self._eviction_count = 0
        self._cache_file = os.path.join(self.cache_dir, "inference_cache.json")

    # ---------- 指纹与相似度 ----------

    @staticmethod
    def _ngram_set(text: str, n: int = 2) -> set:
        """字符 n-gram 集合"""
        if len(text) < n:
            return {text} if text else set()
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    @staticmethod
    def _semantic_fingerprint(text: str) -> str:
        """生成语义指纹 (基于字符 bigram 的哈希摘要)"""
        ngrams = InferenceCache._ngram_set(text, n=2)
        if not ngrams:
            return "empty"
        # 对 ngram 排序后哈希, 保证一致性
        sig = "|".join(sorted(ngrams))
        return _stable_hash(sig)[:16]

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """计算两个文本的相似度 (简化版: n-gram Jaccard * 0.7 + 长度比 * 0.3)"""
        sa = InferenceCache._ngram_set(a, n=2)
        sb = InferenceCache._ngram_set(b, n=2)
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        jaccard = inter / union if union else 0.0
        len_ratio = min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0
        return jaccard * 0.7 + len_ratio * 0.3

    # ---------- 内部清理 ----------

    def _cleanup_expired(self):
        """清理过期条目"""
        expired_keys = [k for k, v in self._store.items() if v.is_expired()]
        for k in expired_keys:
            del self._store[k]
            self._eviction_count += 1

    def _evict(self):
        """LRU 驱逐 (移除最久未使用)"""
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)  # 移除头部 (最旧)
            self._eviction_count += 1

    # ---------- 精确匹配 ----------

    def get(self, prompt: str) -> Optional[Any]:
        """精确匹配获取缓存

        Args:
            prompt: 完整 prompt 文本

        Returns:
            缓存的响应, 未命中返回 None
        """
        key = _stable_hash(prompt)
        with self._lock:
            self._cleanup_expired()
            if key in self._store:
                entry = self._store[key]
                # LRU: 移到末尾
                self._store.move_to_end(key)
                entry.last_access = time.time()
                entry.access_count += 1
                self._hit_count += 1
                return entry.response
            self._miss_count += 1
            return None

    # ---------- 语义匹配 ----------

    def get_semantic(self, prompt: str) -> Optional[Tuple[Any, float]]:
        """语义匹配获取缓存

        Returns:
            (response, similarity) 或 None
        """
        if not self.enable_semantic:
            return None
        with self._lock:
            self._cleanup_expired()
            best_entry: Optional[CacheEntry] = None
            best_sim = 0.0
            for entry in self._store.values():
                sim = self._similarity(prompt, entry.prompt)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
            if best_entry is not None and best_sim >= self.semantic_threshold:
                # 更新访问
                self._store.move_to_end(best_entry.key)
                best_entry.last_access = time.time()
                best_entry.access_count += 1
                self._hit_count += 1
                self._semantic_hit_count += 1
                return best_entry.response, round(best_sim, 4)
            self._miss_count += 1
            return None

    # ---------- 写入 ----------

    def put(self, prompt: str, response: Any, token_count: int = 0,
            ttl: Optional[float] = None):
        """写入缓存

        Args:
            prompt: 完整 prompt 文本
            response: 响应内容 (token ids 或文本)
            token_count: 响应 token 数
            ttl: 自定义有效期 (None 取默认)
        """
        key = _stable_hash(prompt)
        now = time.time()
        entry = CacheEntry(
            key=key,
            prompt=prompt,
            response=response,
            created_at=now,
            last_access=now,
            access_count=0,
            token_count=token_count,
            semantic_fingerprint=self._semantic_fingerprint(prompt),
            prompt_length=len(prompt),
            ttl=ttl if ttl is not None else self.ttl,
        )
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = entry
            self._evict()

    # ---------- 失效 ----------

    def invalidate(self, pattern: str) -> int:
        """按正则模式失效缓存

        Args:
            pattern: 匹配 prompt 的正则表达式

        Returns:
            失效的条目数
        """
        with self._lock:
            try:
                regex = re.compile(pattern)
            except re.error:
                return 0
            keys_to_remove = [k for k, v in self._store.items() if regex.search(v.prompt)]
            for k in keys_to_remove:
                del self._store[k]
            return len(keys_to_remove)

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    # ---------- 持久化 ----------

    def save(self, path: Optional[str] = None) -> str:
        """保存缓存到磁盘"""
        path = path or self._cache_file
        with self._lock:
            data = []
            for entry in self._store.values():
                d = entry.to_dict()
                # response 可能是 list, 确保可序列化
                d["response"] = entry.response
                data.append(d)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"entries": data, "saved_at": _now_iso()},
                      f, ensure_ascii=False, indent=2)
        return path

    def load(self, path: Optional[str] = None) -> int:
        """从磁盘加载缓存, 返回加载条目数"""
        path = path or self._cache_file
        if not os.path.exists(path):
            return 0
        with self._lock:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                return 0
            count = 0
            for d in data.get("entries", []):
                key = d.get("key", _stable_hash(d.get("prompt", "")))
                entry = CacheEntry(
                    key=key,
                    prompt=d.get("prompt", ""),
                    response=d.get("response"),
                    created_at=d.get("created_at", time.time()),
                    last_access=d.get("last_access", time.time()),
                    access_count=d.get("access_count", 0),
                    token_count=d.get("token_count", 0),
                    semantic_fingerprint=d.get("semantic_fingerprint", ""),
                    prompt_length=d.get("prompt_length", 0),
                    ttl=d.get("ttl", self.ttl),
                )
                if not entry.is_expired():
                    self._store[key] = entry
                    count += 1
            self._evict()
            return count

    # ---------- 统计与仪表盘 ----------

    def get_stats(self) -> Dict[str, Any]:
        """缓存统计"""
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / max(total, 1)
        # 估算内存使用 (粗略: prompt + response 的字符数 * 2 bytes)
        mem_chars = 0
        for entry in self._store.values():
            mem_chars += entry.prompt_length + entry.token_count * 4
        mem_kb = mem_chars * 2 / 1024
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 4),
            "semantic_hit_count": self._semantic_hit_count,
            "eviction_count": self._eviction_count,
            "memory_usage_kb": round(mem_kb, 2),
            "ttl": self.ttl,
            "semantic_threshold": self.semantic_threshold,
            "semantic_enabled": self.enable_semantic,
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """缓存仪表盘"""
        entries_info = []
        for entry in list(self._store.values())[-10:]:
            entries_info.append({
                "key": entry.key[:8],
                "prompt_preview": entry.prompt[:50],
                "prompt_length": entry.prompt_length,
                "token_count": entry.token_count,
                "access_count": entry.access_count,
                "is_expired": entry.is_expired(),
            })
        return {
            "cache": "InferenceCache",
            "stats": self.get_stats(),
            "cache_file": self._cache_file,
            "recent_entries": entries_info,
        }


# ============================================================
# 5. FunctionCaller [Function Calling]
# ============================================================

@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]             # JSON Schema 格式参数定义
    handler: Callable[..., Any]
    category: str = "general"
    enabled: bool = True
    keywords: List[str] = field(default_factory=list)    # 意图检测关键词
    patterns: List[str] = field(default_factory=list)    # 意图检测正则模式
    extractors: Dict[str, Callable] = field(default_factory=dict)  # 参数提取器

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("handler", None)
        d.pop("extractors", None)
        return d

    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class ToolCall:
    """工具调用"""
    call_id: str
    name: str
    arguments: Dict[str, Any]
    result: Any = None
    error: str = ""
    status: str = "pending"     # pending / running / success / failed
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """工具执行结果"""
    call_id: str
    name: str
    success: bool
    result: Any
    error: str = ""
    formatted: str = ""
    elapsed: float = 0.0
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FunctionCaller:
    """Function Calling 系统

    功能:
    - 工具注册: register_tool(name, description, parameters_schema, handler)
    - 工具列表生成: 生成 function calling 格式的工具描述
    - 意图检测: 从用户输入检测是否需要调用工具 (关键词匹配 + 模式匹配)
    - 参数提取: 从自然语言提取工具参数 (简化版正则)
    - 工具执行: execute_tool(name, args) -> result
    - 结果格式化: 将工具结果格式化为对话上下文
    - 多工具链: 一个请求可调用多个工具
    - 内置工具: calculator, datetime, search, code_executor (模拟)
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._lock = threading.Lock()
        # 统计
        self._call_count = 0
        self._success_count = 0
        self._fail_count = 0
        self._total_elapsed = 0.0
        self._call_history: deque = deque(maxlen=200)
        self._tool_call_counts: Dict[str, int] = {}
        # 注册内置工具
        self._register_builtin_tools()

    # ---------- 工具注册 ----------

    def register_tool(self, name: str, description: str,
                      parameters: Dict[str, Any],
                      handler: Callable,
                      category: str = "general",
                      keywords: Optional[List[str]] = None,
                      patterns: Optional[List[str]] = None,
                      extractors: Optional[Dict[str, Callable]] = None,
                      enabled: bool = True) -> ToolDefinition:
        """注册一个工具"""
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            category=category,
            enabled=enabled,
            keywords=keywords or [],
            patterns=patterns or [],
            extractors=extractors or {},
        )
        with self._lock:
            self._tools[name] = tool
        return tool

    def unregister_tool(self, name: str) -> bool:
        """注销工具"""
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                return True
            return False

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[ToolDefinition]:
        """列出所有工具"""
        if category:
            return [t for t in self._tools.values() if t.category == category]
        return list(self._tools.values())

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """生成 function calling 格式的工具描述列表 (OpenAI 兼容)"""
        return [t.to_openai_format() for t in self._tools.values() if t.enabled]

    # ---------- 意图检测 ----------

    def detect_intent(self, user_input: str) -> List[str]:
        """从用户输入检测需要调用的工具

        通过关键词匹配 + 正则模式匹配检测意图。

        Returns:
            匹配到的工具名列表 (按匹配度排序)
        """
        matched: Dict[str, int] = {}
        input_lower = user_input.lower()
        for tool in self._tools.values():
            if not tool.enabled:
                continue
            score = 0
            # 关键词匹配
            for kw in tool.keywords:
                if kw.lower() in input_lower:
                    score += 2
            # 正则模式匹配
            for pat in tool.patterns:
                try:
                    if re.search(pat, user_input, re.IGNORECASE):
                        score += 3
                except re.error:
                    pass
            if score > 0:
                matched[tool.name] = score
        # 按分数降序
        return [name for name, _ in sorted(matched.items(), key=lambda x: -x[1])]

    # ---------- 参数提取 ----------

    def extract_arguments(self, user_input: str, tool_name: str) -> Dict[str, Any]:
        """从自然语言提取工具参数 (简化版正则)

        优先使用工具自定义的提取器, 否则使用内置工具的默认提取逻辑。
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return {}
        args: Dict[str, Any] = {}
        # 使用工具自定义提取器
        for param_name, extractor in tool.extractors.items():
            try:
                val = extractor(user_input)
                if val is not None:
                    args[param_name] = val
            except Exception:
                pass
        # 内置工具默认提取
        if not args:
            args = self._builtin_extract(user_input, tool_name)
        return args

    def _builtin_extract(self, user_input: str, tool_name: str) -> Dict[str, Any]:
        """内置工具的参数提取"""
        if tool_name == "calculator":
            # 提取数学表达式: 数字和运算符
            expr = re.search(r"[-+]?\d+(?:\.\d+)?(?:\s*[-+*/^%]\s*[-+]?\d+(?:\.\d+)?)+",
                             user_input)
            if expr:
                return {"expression": expr.group().replace("^", "**").replace(" ", "")}
            # 尝试提取 "计算 ..." 后面的内容
            m = re.search(r"计算\s*(.+)", user_input)
            if m:
                return {"expression": m.group(1).strip().replace("^", "**")}
            return {}
        elif tool_name == "datetime":
            return {"query": user_input}
        elif tool_name == "search":
            # 提取搜索关键词: "搜索/查一下/搜索一下" 后面的内容
            m = re.search(r"(?:搜索|搜索一下|查一下|查询|search)\s*(.+)", user_input,
                          re.IGNORECASE)
            if m:
                return {"query": m.group(1).strip()}
            return {"query": user_input}
        elif tool_name == "code_executor":
            # 提取代码块 ```...``` 或行内代码
            m = re.search(r"```(?:\w+)?\s*\n(.*?)```", user_input, re.DOTALL)
            if m:
                return {"code": m.group(1).strip()}
            m = re.search(r"执行\s*(.+)", user_input, re.DOTALL)
            if m:
                return {"code": m.group(1).strip()}
            return {"code": user_input}
        return {}

    # ---------- 工具执行 ----------

    def execute_tool(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """执行单个工具

        Args:
            name: 工具名
            args: 参数字典

        Returns:
            ToolResult
        """
        call_id = str(uuid.uuid4())[:12]
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                call_id=call_id, name=name, success=False,
                result=None, error=f"工具 '{name}' 不存在",
                arguments=args,
            )
        if not tool.enabled:
            return ToolResult(
                call_id=call_id, name=name, success=False,
                result=None, error=f"工具 '{name}' 已禁用",
                arguments=args,
            )

        start = time.time()
        started = _now_iso()
        ok, result_or_err = _safe_call(tool.handler, **args)
        elapsed = time.time() - start
        finished = _now_iso()

        # 统计
        with self._lock:
            self._call_count += 1
            self._total_elapsed += elapsed
            self._tool_call_counts[name] = self._tool_call_counts.get(name, 0) + 1
            if ok:
                self._success_count += 1
            else:
                self._fail_count += 1
            self._call_history.append(ToolCall(
                call_id=call_id, name=name, arguments=args,
                result=result_or_err if ok else None,
                error="" if ok else str(result_or_err),
                status="success" if ok else "failed",
                started_at=started, finished_at=finished,
            ))

        if ok:
            formatted = self.format_result_for_context(
                ToolResult(call_id=call_id, name=name, success=True,
                           result=result_or_err, arguments=args))
            return ToolResult(
                call_id=call_id, name=name, success=True,
                result=result_or_err, formatted=formatted,
                elapsed=round(elapsed, 4), arguments=args,
            )
        else:
            return ToolResult(
                call_id=call_id, name=name, success=False,
                result=None, error=str(result_or_err),
                elapsed=round(elapsed, 4), arguments=args,
            )

    # ---------- 多工具链 ----------

    def execute_chain(self, user_input: str,
                      max_tools: int = 5) -> List[ToolResult]:
        """多工具链: 一个请求可调用多个工具

        流程: 意图检测 -> 参数提取 -> 依次执行

        Args:
            user_input: 用户输入
            max_tools: 最多调用工具数

        Returns:
            工具执行结果列表
        """
        results: List[ToolResult] = []
        tool_names = self.detect_intent(user_input)[:max_tools]
        for name in tool_names:
            args = self.extract_arguments(user_input, name)
            if not args:
                # 无法提取参数, 跳过
                continue
            result = self.execute_tool(name, args)
            results.append(result)
            # 如果工具失败, 可选择终止链
            if not result.success:
                break
        return results

    # ---------- 结果格式化 ----------

    def format_result_for_context(self, result: ToolResult) -> str:
        """将工具结果格式化为对话上下文"""
        if result.success:
            content = result.result
            if isinstance(content, (dict, list)):
                content_str = json.dumps(content, ensure_ascii=False, indent=2)
            else:
                content_str = str(content)
            return (f"[工具调用结果] 工具: {result.name}\n"
                    f"参数: {json.dumps(result.arguments, ensure_ascii=False)}\n"
                    f"结果: {content_str}")
        else:
            return (f"[工具调用失败] 工具: {result.name}\n"
                    f"参数: {json.dumps(result.arguments, ensure_ascii=False)}\n"
                    f"错误: {result.error}")

    # ---------- 内置工具 ----------

    def _register_builtin_tools(self):
        """注册内置工具: calculator, datetime, search, code_executor"""

        # 1. 计算器
        def _calculator(expression: str = "") -> Any:
            """安全计算数学表达式"""
            # 仅允许数字和基本运算符
            if not expression:
                return {"error": "空表达式"}
            # 白名单过滤
            allowed = set("0123456789+-*/.()% ")
            if not all(c in allowed for c in expression):
                return {"error": "表达式包含非法字符"}
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return {"expression": expression, "result": result}
            except Exception as e:
                return {"error": str(e)}

        self.register_tool(
            name="calculator",
            description="数学计算器, 支持加减乘除、括号、取模等基本运算",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式, 如 '2+3*4'",
                    }
                },
                "required": ["expression"],
            },
            handler=_calculator,
            category="math",
            keywords=["计算", "等于", "加", "减", "乘", "除", "calculate", "compute"],
            patterns=[r"[-+]?\d+.*[-+*/].*\d", r"计算"],
        )

        # 2. 日期时间
        def _datetime(query: str = "") -> Any:
            """获取当前日期时间信息"""
            now = datetime.now()
            return {
                "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
                "timestamp": int(now.timestamp()),
                "iso": now.isoformat(),
                "query": query,
            }

        self.register_tool(
            name="datetime",
            description="获取当前日期、时间、星期等信息",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的日期时间查询",
                    }
                },
                "required": ["query"],
            },
            handler=_datetime,
            category="utility",
            keywords=["时间", "日期", "今天", "星期", "几点", "date", "time", "today"],
            patterns=[r"今天|现在|当前|几点|星期|日期|时间"],
        )

        # 3. 搜索 (模拟)
        def _search(query: str = "") -> Any:
            """模拟搜索引擎"""
            if not query:
                return {"error": "搜索关键词为空"}
            # 模拟搜索结果
            results = []
            for i in range(1, 4):
                results.append({
                    "title": f"{query} - 相关结果 {i}",
                    "url": f"https://search.lingyuan.ai/result?q={query}&page={i}",
                    "snippet": f"关于「{query}」的模拟搜索结果第 {i} 条, "
                               f"包含相关概述信息。",
                    "score": round(1.0 / i, 2),
                })
            return {"query": query, "results": results, "total": len(results)}

        self.register_tool(
            name="search",
            description="网络搜索工具, 根据关键词返回搜索结果",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["query"],
            },
            handler=_search,
            category="information",
            keywords=["搜索", "查找", "查一下", "search", "find", "look up"],
            patterns=[r"搜索|查找|查一下|搜一下|search"],
        )

        # 4. 代码执行器 (模拟)
        def _code_executor(code: str = "", language: str = "python") -> Any:
            """模拟代码执行"""
            if not code:
                return {"error": "代码为空"}
            # 模拟执行: 解析 print 语句
            output_lines = []
            for line in code.strip().split("\n"):
                line = line.strip()
                m = re.match(r"print\s*\((.*)\)", line)
                if m:
                    inner = m.group(1).strip()
                    # 简单处理字符串字面量
                    if inner.startswith(("'", '"')) and inner.endswith(("'", '"')):
                        output_lines.append(inner[1:-1])
                    else:
                        try:
                            val = eval(inner, {"__builtins__": {}}, {})
                            output_lines.append(str(val))
                        except Exception:
                            output_lines.append(f"<{inner}>")
            return {
                "code": code,
                "language": language,
                "output": "\n".join(output_lines) if output_lines else "(无输出)",
                "exit_code": 0,
            }

        self.register_tool(
            name="code_executor",
            description="代码执行器, 模拟运行 Python 代码并返回输出",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的代码",
                    },
                    "language": {
                        "type": "string",
                        "description": "编程语言",
                        "default": "python",
                    }
                },
                "required": ["code"],
            },
            handler=_code_executor,
            category="development",
            keywords=["执行", "运行", "代码", "code", "run", "python", "print"],
            patterns=[r"```", r"执行.*代码", r"运行.*代码"],
        )

    # ---------- 统计与仪表盘 ----------

    def get_stats(self) -> Dict[str, Any]:
        """Function Calling 统计"""
        avg_elapsed = self._total_elapsed / max(self._call_count, 1)
        success_rate = self._success_count / max(self._call_count, 1)
        return {
            "tool_count": len(self._tools),
            "call_count": self._call_count,
            "success_count": self._success_count,
            "fail_count": self._fail_count,
            "success_rate": round(success_rate, 4),
            "avg_elapsed_sec": round(avg_elapsed, 4),
            "tool_call_counts": dict(self._tool_call_counts),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """Function Calling 仪表盘"""
        tools_info = []
        for tool in self._tools.values():
            tools_info.append({
                "name": tool.name,
                "description": tool.description[:60],
                "category": tool.category,
                "enabled": tool.enabled,
                "keywords": tool.keywords,
                "call_count": self._tool_call_counts.get(tool.name, 0),
            })
        recent_calls = [c.to_dict() for c in list(self._call_history)[-10:]]
        return {
            "function_caller": "FunctionCaller",
            "stats": self.get_stats(),
            "tools": tools_info,
            "recent_calls": recent_calls,
        }


# ============================================================
# 6. ChatTemplateManager [对话模板系统]
# ============================================================

@dataclass
class ChatMessage:
    """对话消息"""
    role: str            # system / user / assistant / tool
    content: str
    name: str = ""       # 可选: 发送者名称
    tool_call_id: str = ""  # tool 角色对应的调用 ID

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatTemplate:
    """对话模板定义"""
    name: str
    system_prefix: str       # system 消息前缀
    user_prefix: str         # user 消息前缀
    assistant_prefix: str    # assistant 消息前缀
    tool_prefix: str         # tool 消息前缀
    system_suffix: str       # system 消息后缀
    separator: str           # 消息间分隔符
    eos: str = ""            # 结束标记
    add_bos: bool = False    # 是否添加序列开始标记
    bos: str = ""            # 序列开始标记
    # system 放置策略: "first" (开头) 或 "embed" (嵌入每条 user 前)
    system_strategy: str = "first"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChatTemplateManager:
    """对话模板系统

    功能:
    - 支持模板: ChatML, Llama-2-Chat, Alpaca, Vicuna, Qwen, 通用灵元格式
    - 模板渲染: format_messages(messages, template_name) -> formatted_text
    - 消息角色: system, user, assistant, tool
    - 模板定义: 每个模板有 system_prefix, user_prefix, assistant_prefix, system_suffix
    - 自动检测: detect_template(model_name) -> template_name
    - 自定义模板: register_template(name, config)
    - 多轮对话格式化: 保持对话历史上下文
    - Token 计数: 估算格式化后的 token 数
    """

    def __init__(self):
        self._templates: Dict[str, ChatTemplate] = {}
        self._register_builtin_templates()
        # 统计
        self._format_count = 0
        self._total_tokens_formatted = 0
        self._template_usage: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ---------- 内置模板注册 ----------

    def _register_builtin_templates(self):
        """注册内置对话模板"""

        # 1. ChatML (OpenAI 风格)
        self._templates["chatml"] = ChatTemplate(
            name="chatml",
            system_prefix="<|im_start|>system\n",
            user_prefix="<|im_start|>user\n",
            assistant_prefix="<|im_start|>assistant\n",
            tool_prefix="<|im_start|>tool\n",
            system_suffix="<|im_end|>\n",
            separator="",
            eos="<|im_end|>",
            add_bos=False,
            system_strategy="first",
        )

        # 2. Llama-2-Chat
        self._templates["llama2"] = ChatTemplate(
            name="llama2",
            system_prefix="<<SYS>>\n",
            user_prefix="[INST] ",
            assistant_prefix="",
            tool_prefix="[TOOL] ",
            system_suffix="\n<</SYS>>\n\n",
            separator=" ",
            eos="</s>",
            add_bos=True,
            bos="<s>",
            system_strategy="first",
        )

        # 3. Alpaca
        self._templates["alpaca"] = ChatTemplate(
            name="alpaca",
            system_prefix="### System:\n",
            user_prefix="### Instruction:\n",
            assistant_prefix="### Response:\n",
            tool_prefix="### Tool:\n",
            system_suffix="\n\n",
            separator="\n\n",
            eos="",
            add_bos=False,
            system_strategy="first",
        )

        # 4. Vicuna
        self._templates["vicuna"] = ChatTemplate(
            name="vicuna",
            system_prefix="",
            user_prefix="USER: ",
            assistant_prefix="ASSISTANT: ",
            tool_prefix="TOOL: ",
            system_suffix="",
            separator="",
            eos="</s>",
            add_bos=False,
            system_strategy="first",
        )

        # 5. Qwen (与 ChatML 类似但有细微差异)
        self._templates["qwen"] = ChatTemplate(
            name="qwen",
            system_prefix="<|im_start|>system\n",
            user_prefix="<|im_start|>user\n",
            assistant_prefix="<|im_start|>assistant\n",
            tool_prefix="<|im_start|>system\n",   # Qwen 中 tool 结果以 system 角色返回
            system_suffix="<|im_end|>\n",
            separator="",
            eos="<|im_end|>",
            add_bos=False,
            system_strategy="first",
        )

        # 6. 通用灵元格式
        self._templates["lingyuan"] = ChatTemplate(
            name="lingyuan",
            system_prefix="[系统] ",
            user_prefix="[用户] ",
            assistant_prefix="[灵元] ",
            tool_prefix="[工具] ",
            system_suffix="\n",
            separator="\n",
            eos="[结束]",
            add_bos=True,
            bos="[开始]\n",
            system_strategy="first",
        )

    # ---------- 模板管理 ----------

    def register_template(self, name: str, config: Union[Dict[str, Any], ChatTemplate]) -> ChatTemplate:
        """注册自定义模板"""
        if isinstance(config, ChatTemplate):
            tpl = config
            tpl.name = name
        else:
            tpl = ChatTemplate(name=name, **config)
        with self._lock:
            self._templates[name] = tpl
        return tpl

    def get_template(self, name: str) -> Optional[ChatTemplate]:
        """获取模板"""
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """列出所有模板名"""
        return list(self._templates.keys())

    def list_templates_detail(self) -> List[Dict[str, Any]]:
        """列出所有模板详情"""
        return [t.to_dict() for t in self._templates.values()]

    # ---------- 自动检测 ----------

    def detect_template(self, model_name: str) -> str:
        """根据模型名自动检测合适的模板

        Args:
            model_name: 模型名称

        Returns:
            模板名 (未匹配则返回 "chatml")
        """
        name_lower = model_name.lower()
        # Qwen 系列
        if "qwen" in name_lower:
            return "qwen"
        # Llama 系列
        if "llama" in name_lower:
            return "llama2"
        # Alpaca
        if "alpaca" in name_lower:
            return "alpaca"
        # Vicuna
        if "vicuna" in name_lower:
            return "vicuna"
        # ChatML
        if "chatml" in name_lower or "gpt" in name_lower or "chat" in name_lower:
            return "chatml"
        # 灵元
        if "lingyuan" in name_lower or "灵元" in model_name:
            return "lingyuan"
        # 默认
        return "chatml"

    # ---------- 模板渲染 ----------

    def format_messages(self,
                        messages: List[Union[ChatMessage, Dict[str, Any]]],
                        template_name: str = "chatml") -> str:
        """将消息列表渲染为格式化文本

        Args:
            messages: 消息列表 (ChatMessage 或 dict)
            template_name: 模板名

        Returns:
            格式化后的完整文本
        """
        tpl = self._templates.get(template_name)
        if tpl is None:
            raise ValueError(f"未知模板: {template_name}, 可用: {self.list_templates()}")

        # 统一转为 ChatMessage
        msgs: List[ChatMessage] = []
        for m in messages:
            if isinstance(m, ChatMessage):
                msgs.append(m)
            elif isinstance(m, dict):
                msgs.append(ChatMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    name=m.get("name", ""),
                    tool_call_id=m.get("tool_call_id", ""),
                ))

        parts: List[str] = []
        if tpl.add_bos and tpl.bos:
            parts.append(tpl.bos)

        # 分离 system 消息
        system_msgs = [m for m in msgs if m.role == "system"]
        other_msgs = [m for m in msgs if m.role != "system"]

        # 处理 system 消息 (放在开头)
        if system_msgs and tpl.system_strategy == "first":
            sys_content = "\n".join(m.content for m in system_msgs)
            parts.append(f"{tpl.system_prefix}{sys_content}{tpl.system_suffix}")

        # 处理其他消息
        for m in other_msgs:
            if m.role == "user":
                prefix = tpl.user_prefix
                suffix = tpl.system_suffix  # 复用作为结束标记
            elif m.role == "assistant":
                prefix = tpl.assistant_prefix
                suffix = tpl.eos
            elif m.role == "tool":
                prefix = tpl.tool_prefix
                suffix = tpl.system_suffix
            else:
                prefix = tpl.user_prefix
                suffix = ""

            content = m.content
            if tpl.separator and parts:
                parts.append(tpl.separator)
            parts.append(f"{prefix}{content}{suffix}")

        result = "".join(parts)

        # 统计
        with self._lock:
            self._format_count += 1
            self._template_usage[template_name] = self._template_usage.get(template_name, 0) + 1
            self._total_tokens_formatted += _estimate_tokens(result)

        return result

    def format_single(self, role: str, content: str,
                      template_name: str = "chatml") -> str:
        """格式化单条消息"""
        return self.format_messages([ChatMessage(role=role, content=content)],
                                    template_name)

    # ---------- Token 计数 ----------

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数"""
        return _estimate_tokens(text)

    def count_tokens(self,
                     messages: List[Union[ChatMessage, Dict[str, Any]]],
                     template_name: str = "chatml") -> int:
        """估算格式化后的 token 数 (不实际格式化)"""
        tpl = self._templates.get(template_name)
        if tpl is None:
            tpl = self._templates.get("chatml")
        # 粗略估算: 各消息内容 + 模板标记开销
        total = 0
        if tpl and tpl.add_bos:
            total += _estimate_tokens(tpl.bos)
        for m in messages:
            content = m.content if isinstance(m, ChatMessage) else m.get("content", "")
            total += _estimate_tokens(content)
            total += 4  # 每条消息的模板标记开销
        return total

    # ---------- 统计与仪表盘 ----------

    def get_stats(self) -> Dict[str, Any]:
        """模板系统统计"""
        return {
            "template_count": len(self._templates),
            "format_count": self._format_count,
            "total_tokens_formatted": self._total_tokens_formatted,
            "template_usage": dict(self._template_usage),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """模板系统仪表盘"""
        return {
            "manager": "ChatTemplateManager",
            "available_templates": self.list_templates(),
            "templates_detail": self.list_templates_detail(),
            "stats": self.get_stats(),
        }
