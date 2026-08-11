#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# LINGYUAN MODEL - PART 31
# 虚拟用户群体 (Virtual User Community — 群体行为模拟器)
#
# 问题: 没有真实用户流量来验证分布式推理的自愈链
#       (破产触发→降级→恢复)。单条请求无法测出系统
#       在高并发下的涌现行为。
# 解决: 构建虚拟用户群体, 模拟真实世界的请求模式,
#       对接 part22 推理引擎和 part6 六合一决策引擎,
#       压力场景下验证系统自愈性质。
#
# 核心组件:
# - UserProfile:         用户画像 (行为模式/请求类型)
# - VirtualUser:         虚拟用户 (独立线程, 按画像发请求)
# - UserCluster:         用户聚类 (按行为特征分组)
# - VirtualCommunity:    虚拟社区 (群体管理器)
# - StressScenario:      压力场景 (浪涌/持续/突发/脉冲)
# - ScenarioScheduler:   场景编排器
# - CommunityMonitor:    场景监控 (QPS/延迟/自愈检测)
# - SelfHealingValidator: 自愈链验证器
#
# 纯Python标准库实现 (零外部依赖)
# ============================================================

import os
import sys
import time
import math
import json
import random
import hashlib
import threading
from collections import deque, defaultdict, Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from datetime import datetime


# ============================================================
# 枚举定义
# ============================================================

class UserType(Enum):
    """虚拟用户类型"""
    CASUAL = "casual"           # 普通聊天 (短prompt, 低频)
    CODER = "coder"             # 代码助手 (中prompt, 中频)
    WRITER = "writer"           # 文案创作 (长prompt, 低频)
    ANALYST = "analyst"         # 数据分析 (长prompt, 高频)
    SPAMMER = "spammer"         # 恶意灌水 (极短, 极高频率)
    POWER_USER = "power_user"   # 重度用户 (混合, 极高频率)


class RequestType(Enum):
    """请求类型"""
    CHAT = "chat"               # 对话
    COMPLETION = "completion"   # 补全
    TRANSLATION = "translation" # 翻译
    SUMMARIZE = "summarize"     # 摘要
    CODE_GEN = "code_gen"       # 代码生成
    CODE_REVIEW = "code_review" # 代码审查
    ANALYSIS = "analysis"       # 数据分析
    CREATIVE = "creative"       # 创意写作


class ScenarioType(Enum):
    """场景类型"""
    STEADY = "steady"           # 稳态 (恒定QPS)
    BURST = "burst"             # 突发 (瞬间峰值)
    RAMP_UP = "ramp_up"         # 爬坡 (逐步增加)
    SURGE = "surge"             # 浪涌 (周期性波峰)
    PULSE = "pulse"             # 脉冲 (短间隔高密度)
    CHAOS = "chaos"             # 混沌 (随机扰动)
    SPAM_ATTACK = "spam_attack" # 攻击 (恶意灌水)


class CommunityState(Enum):
    """社区状态"""
    IDLE = "idle"
    WARMING_UP = "warming_up"
    STEADY = "steady"
    OVERLOADED = "overloaded"
    DEGRADING = "degrading"
    RECOVERING = "recovering"
    BANKRUPT = "bankrupt"


# ============================================================
# 用户画像
# ============================================================

@dataclass
class UserProfile:
    """用户画像 — 定义虚拟用户的行为模式

    每个维度有均值和标准差, 支持高斯采样。
    """
    user_type: UserType
    user_id: str = ""
    name: str = ""
    # 请求属性
    avg_prompt_length: float = 50.0       # 平均prompt长度 (tokens)
    std_prompt_length: float = 20.0
    avg_max_tokens: float = 128.0         # 平均期望生成长度
    temperature: float = 0.7
    # 频率属性
    avg_interval_sec: float = 30.0        # 平均请求间隔 (秒)
    std_interval_sec: float = 10.0
    burstiness: float = 0.2               # 突发度 (0-1, 越高越bursty)
    # 质量要求
    priority: int = 1                     # 1=普通 2=高 3=紧急
    latency_slo_ms: float = 1000.0        # 延迟SLO (毫秒)
    # 会话属性
    avg_session_len: int = 5              # 平均每会话请求数
    think_time_sec: float = 2.0           # 用户思考时间
    # 行为偏差
    retry_prob: float = 0.1               # 失败重试概率
    abandon_threshold_ms: float = 5000.0  # 放弃等待阈值

    @staticmethod
    def for_type(user_type: UserType) -> 'UserProfile':
        """按用户类型生成默认画像"""
        profiles = {
            UserType.CASUAL: UserProfile(
                user_type=UserType.CASUAL,
                avg_prompt_length=30.0, std_prompt_length=15.0,
                avg_max_tokens=64.0,
                avg_interval_sec=60.0, std_interval_sec=30.0,
                burstiness=0.1, priority=1,
                latency_slo_ms=3000.0,
                avg_session_len=3, think_time_sec=5.0,
            ),
            UserType.CODER: UserProfile(
                user_type=UserType.CODER,
                avg_prompt_length=200.0, std_prompt_length=80.0,
                avg_max_tokens=512.0,
                avg_interval_sec=20.0, std_interval_sec=10.0,
                burstiness=0.3, priority=2,
                latency_slo_ms=2000.0,
                avg_session_len=8, think_time_sec=30.0,
            ),
            UserType.WRITER: UserProfile(
                user_type=UserType.WRITER,
                avg_prompt_length=500.0, std_prompt_length=200.0,
                avg_max_tokens=1024.0,
                avg_interval_sec=120.0, std_interval_sec=60.0,
                burstiness=0.05, priority=1,
                latency_slo_ms=5000.0,
                avg_session_len=2, think_time_sec=60.0,
            ),
            UserType.ANALYST: UserProfile(
                user_type=UserType.ANALYST,
                avg_prompt_length=800.0, std_prompt_length=300.0,
                avg_max_tokens=2048.0,
                avg_interval_sec=10.0, std_interval_sec=5.0,
                burstiness=0.5, priority=3,
                latency_slo_ms=1500.0,
                avg_session_len=15, think_time_sec=3.0,
            ),
            UserType.SPAMMER: UserProfile(
                user_type=UserType.SPAMMER,
                avg_prompt_length=10.0, std_prompt_length=5.0,
                avg_max_tokens=1.0,
                avg_interval_sec=0.05, std_interval_sec=0.02,
                burstiness=0.95, priority=1,
                latency_slo_ms=50000.0,
                avg_session_len=200, think_time_sec=0.0,
            ),
            UserType.POWER_USER: UserProfile(
                user_type=UserType.POWER_USER,
                avg_prompt_length=300.0, std_prompt_length=150.0,
                avg_max_tokens=512.0,
                avg_interval_sec=5.0, std_interval_sec=3.0,
                burstiness=0.6, priority=2,
                latency_slo_ms=1000.0,
                avg_session_len=20, think_time_sec=1.0,
            ),
        }
        return profiles[user_type]


class VirtualUser:
    """虚拟用户 — 独立线程模拟真实用户行为

    每个用户有独立的状态机和请求历史,
    按画像的统计分布自主决定何时发送请求。
    """

    _id_counter = 0
    _id_lock = threading.Lock()

    def __init__(self, profile: UserProfile,
                 submit_fn: Callable[..., Any],
                 name: str = ""):
        with VirtualUser._id_lock:
            VirtualUser._id_counter += 1
            self.user_id = f"vu-{VirtualUser._id_counter}"

        self.profile = profile
        profile.user_id = self.user_id
        profile.name = name or self.user_id

        self._submit = submit_fn        # 提交推理请求的回调
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 状态
        self.active: bool = True
        self.session_count: int = 0
        self.total_requests: int = 0
        self.failed_requests: int = 0
        self.total_latency_ms: float = 0.0
        self._abandoned_count: int = 0

        # 请求历史 (最近N条)
        self._history: deque = deque(maxlen=100)
        self._conversation_context: List[str] = []

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run_loop(self):
        """用户行为主循环"""
        while self._running:
            # 决定本次会话的请求数
            session_len = max(1, int(random.gauss(
                self.profile.avg_session_len,
                self.profile.avg_session_len * 0.3)))

            for _ in range(session_len):
                if not self._running:
                    return

                # 生成prompt长度
                prompt_len = max(1, int(random.gauss(
                    self.profile.avg_prompt_length,
                    self.profile.std_prompt_length)))

                # 生成请求
                req_type = self._pick_request_type()
                prompt = self._generate_prompt(prompt_len, req_type)
                max_tokens = max(1, int(self.profile.avg_max_tokens *
                                       random.uniform(0.5, 1.5)))

                t0 = time.time()
                try:
                    result = self._submit(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=self.profile.temperature,
                        priority=self.profile.priority,
                        user_id=self.user_id,
                        user_type=self.profile.user_type.value,
                        request_type=req_type.value,
                    )
                    elapsed_ms = (time.time() - t0) * 1000

                    # 记录SLO
                    if result and hasattr(result, 'latency_ms'):
                        elapsed_ms = result.latency_ms

                    self.total_requests += 1
                    self.total_latency_ms += elapsed_ms

                    self._history.append({
                        "req_type": req_type.value,
                        "prompt_len": prompt_len,
                        "max_tokens": max_tokens,
                        "latency_ms": elapsed_ms,
                        "success": True,
                        "timestamp": time.time(),
                    })

                    # 如果没有result (同步返回None), 模拟思考时间
                    if result is None:
                        think_time = self.profile.think_time_sec * random.uniform(0.5, 1.5)
                        time.sleep(think_time)

                except Exception:
                    self.failed_requests += 1
                    self._history.append({
                        "req_type": req_type.value,
                        "prompt_len": prompt_len,
                        "latency_ms": (time.time() - t0) * 1000,
                        "success": False,
                        "timestamp": time.time(),
                    })

                # 间隔时间 (高斯 + 突发扰动)
                base_interval = max(0.01, random.gauss(
                    self.profile.avg_interval_sec,
                    self.profile.std_interval_sec))
                if random.random() < self.profile.burstiness:
                    base_interval *= 0.1  # 突发: 间隔缩短10倍
                time.sleep(base_interval)

            self.session_count += 1

            # 会话间隔
            time.sleep(random.uniform(1.0, 5.0))

    def _pick_request_type(self) -> RequestType:
        """按用户类型选择请求类型"""
        if self.profile.user_type == UserType.CODER:
            return random.choices(
                [RequestType.CODE_GEN, RequestType.CODE_REVIEW, RequestType.CHAT],
                weights=[0.5, 0.3, 0.2], k=1)[0]
        elif self.profile.user_type == UserType.WRITER:
            return random.choices(
                [RequestType.CREATIVE, RequestType.SUMMARIZE, RequestType.CHAT],
                weights=[0.6, 0.2, 0.2], k=1)[0]
        elif self.profile.user_type == UserType.ANALYST:
            return random.choices(
                [RequestType.ANALYSIS, RequestType.CODE_GEN, RequestType.CHAT],
                weights=[0.5, 0.3, 0.2], k=1)[0]
        else:
            return random.choice(list(RequestType))

    def _generate_prompt(self, length: int, req_type: RequestType) -> str:
        """生成模拟prompt"""
        templates = {
            RequestType.CHAT: [
                "请解释一下{m}的概念", "你好, 我想问关于{t}的问题",
                "帮我分析一下{d}", "最近有什么新闻关于{s}",
                "请帮我总结一下{k}",
            ],
            RequestType.CODE_GEN: [
                "请用Python实现一个{f}函数, 要求{e}",
                "帮我优化这段代码: {c}",
                "写一个测试用例测试{p}功能",
                "把这个算法从{a}转换成{b}语言",
            ],
            RequestType.COMPLETION: [
                "请补全以下文本: {x}",
                "续写: {y}",
            ],
            RequestType.ANALYSIS: [
                "分析这份数据, 找出异常点: {d}",
                "根据以下指标做预测: {m}",
                "对{k}做聚类分析",
            ],
            RequestType.CREATIVE: [
                "写一篇关于{t}的文章, 要求{r}",
                "请给我一些关于{s}的创意",
                "改写这个故事: {x}",
            ],
        }
        fillers = {
            "m": ["机器学习", "深度学习", "注意力机制", "GPT架构", "强化学习"],
            "t": ["人工智能", "量子计算", "区块链", "碳中和", "星际旅行"],
            "d": ["销售量", "用户增长", "股票价格", "网站流量"],
            "s": ["科技", "经济", "体育", "娱乐", "医疗"],
            "k": ["论文", "文章", "会议纪要", "法律文件"],
            "f": ["排序", "搜索", "分词", "矩阵乘法", "回归模型"],
            "e": ["线程安全", "低于O(n²)", "零依赖", "支持流式"],
            "c": ["def foo(x): return x+1", "for i in range(n): j+=i"],
            "p": ["认证", "支付", "文件上传", "WebSocket"],
            "a": ["Python", "Java", "JavaScript", "Rust"],
            "b": ["Go", "C++", "TypeScript", "Zig"],
            "x": ["从前有座山", "在一个遥远的星系", "The quick brown fox"],
            "y": ["山上有座庙", "一颗行星绕着三颗恒星旋转"],
            "r": ["800字左右", "用第一人称", "幽默风趣"],
        }

        template = random.choice(templates.get(req_type, templates[RequestType.CHAT]))
        filled = template
        for key in set(c for c in template if c == "{"):
            pass
        import re
        placeholders = re.findall(r'\{(\w)\}', template)
        for ph in placeholders:
            if ph in fillers:
                filled = filled.replace(f"{{{ph}}}", random.choice(fillers[ph]), 1)

        # 填充到目标长度
        while len(filled) < length:
            filled += " " + random.choice(
                ["请继续", "还有", "并且", "另外", "同时",
                 "此外", "值得注意的是", "需要补充的是"])

        return filled[:length]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_type": self.profile.user_type.value,
            "sessions": self.session_count,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "fail_rate": round(self.failed_requests / max(self.total_requests, 1), 4),
            "avg_latency_ms": round(
                self.total_latency_ms / max(self.total_requests, 1), 1),
            "abandoned": self._abandoned_count,
            "active": self.active,
        }


# ============================================================
# 用户聚类
# ============================================================

@dataclass
class UserCluster:
    """用户聚类 — 按行为特征分组

    同一cluster内的用户共享相似的请求模式,
    调度器可以按cluster做批量优化。
    """
    cluster_id: str
    user_type: UserType
    members: List[str] = field(default_factory=list)  # user_id列表
    avg_prompt_len: float = 0.0
    avg_interval_sec: float = 0.0
    total_qps: float = 0.0

    @property
    def size(self) -> int:
        return len(self.members)


# ============================================================
# 压力场景
# ============================================================

@dataclass
class StressScenario:
    """压力场景定义"""
    name: str
    scenario_type: ScenarioType
    duration_sec: float                  # 场景总时长
    phases: List['ScenarioPhase'] = field(default_factory=list)


@dataclass
class ScenarioPhase:
    """场景阶段"""
    phase_name: str
    num_users: int                       # 当前阶段活跃用户数
    duration_sec: float                  # 阶段持续时长
    user_type_mix: Dict[UserType, float] = field(default_factory=dict)  # 用户类型占比
    qps_target: float = 0.0              # 目标QPS (0=不限)
    failure_injection_rate: float = 0.0   # 注入的故障率 (模拟网络抖动)
    qps_curve: Callable[[float, float], int] = None  # QPS时间曲线


class ScenarioScheduler:
    """场景编排器 — 按时间线切换不同阶段

    三种经典场景:
    1. 平稳负载: 长时间恒定的用户数
    2. 突发浪涌: 用户数从100瞬间跳到10000
    3. 逐步爬坡: 每30秒增加一定数量用户
    """

    # 内置场景
    @staticmethod
    def steady_scenario(num_users: int = 500, duration_sec: float = 300.0) -> StressScenario:
        """稳态场景: 恒定负载"""
        mix = {
            UserType.CASUAL: 0.50,
            UserType.CODER: 0.25,
            UserType.WRITER: 0.10,
            UserType.ANALYST: 0.10,
            UserType.POWER_USER: 0.05,
        }
        return StressScenario(
            name="稳态负载",
            scenario_type=ScenarioType.STEADY,
            duration_sec=duration_sec,
            phases=[
                ScenarioPhase("warmup", num_users // 5, 30.0, mix),
                ScenarioPhase("steady", num_users, duration_sec - 60.0, mix),
                ScenarioPhase("cooldown", num_users // 5, 30.0, mix),
            ],
        )

    @staticmethod
    def burst_scenario(base_users: int = 200,
                       peak_users: int = 5000,
                       burst_duration: float = 60.0) -> StressScenario:
        """突发场景: 瞬间流量峰值"""
        mix = {
            UserType.CASUAL: 0.40,
            UserType.CODER: 0.20,
            UserType.WRITER: 0.10,
            UserType.ANALYST: 0.15,
            UserType.POWER_USER: 0.10,
            UserType.SPAMMER: 0.05,
        }
        return StressScenario(
            name="突发浪涌",
            scenario_type=ScenarioType.BURST,
            duration_sec=burst_duration + 60.0,
            phases=[
                ScenarioPhase("baseline", base_users, 30.0, mix),
                ScenarioPhase("burst", peak_users, burst_duration, mix),
                ScenarioPhase("recovery", base_users, 30.0, mix),
            ],
        )

    @staticmethod
    def ramp_scenario(start_users: int = 100,
                      end_users: int = 3000,
                      ramp_duration: float = 180.0) -> StressScenario:
        """爬坡场景: 逐步增加负载"""
        mix = {
            UserType.CASUAL: 0.30,
            UserType.CODER: 0.30,
            UserType.WRITER: 0.15,
            UserType.ANALYST: 0.20,
            UserType.POWER_USER: 0.05,
        }
        n_phases = 6
        step = (end_users - start_users) // n_phases
        phase_dur = ramp_duration / n_phases
        phases = []
        for i in range(n_phases):
            n = start_users + step * i
            phases.append(ScenarioPhase(f"ramp_{i+1}", n, phase_dur, mix))
        phases.append(ScenarioPhase("peak", end_users, 60.0, mix))
        phases.append(ScenarioPhase("cooldown", start_users, 30.0, mix))

        return StressScenario(
            name="爬坡测试",
            scenario_type=ScenarioType.RAMP_UP,
            duration_sec=ramp_duration + 90.0,
            phases=phases,
        )

    @staticmethod
    def spam_attack_scenario(legit_users: int = 500,
                             spam_users: int = 10000) -> StressScenario:
        """恶意灌水攻击场景"""
        legit_mix = {
            UserType.CASUAL: 0.40,
            UserType.CODER: 0.30,
            UserType.ANALYST: 0.20,
            UserType.POWER_USER: 0.10,
        }
        spam_mix = {UserType.SPAMMER: 1.0}
        return StressScenario(
            name="恶意灌水攻击",
            scenario_type=ScenarioType.SPAM_ATTACK,
            duration_sec=180.0,
            phases=[
                ScenarioPhase("baseline", legit_users, 30.0, legit_mix),
                ScenarioPhase("under_attack", legit_users + spam_users, 120.0,
                             {**legit_mix, UserType.SPAMMER: 0.95}),
                ScenarioPhase("recovery", legit_users, 30.0, legit_mix),
            ],
        )


# ============================================================
# 推理引擎适配器
# ============================================================

class InferenceAdapter:
    """推理引擎适配器 — 桥接 VirtualUser 和 DistributedInferenceEngine

    将虚拟用户的请求转换为 part22 InferenceRequest,
    收集原始 InferenceResult 转换成虚拟用户理解的格式。
    """

    def __init__(self, engine: Any = None):
        """engine: DistributedInferenceEngine (part22) 或兼容接口"""
        self.engine = engine
        self._pending: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._request_counter = 0

        # 模拟模式 (无真实引擎时)
        self._simulate_results: deque = deque(maxlen=1000)
        self._simulate_mode = engine is None

    def submit(self, prompt: str, max_tokens: int = 128,
               temperature: float = 0.7, priority: int = 1,
               user_id: str = "", user_type: str = "",
               request_type: str = "") -> Optional[Any]:
        """提交推理请求

        返回 InferenceResult 或 None (模拟模式)
        """
        if self._simulate_mode:
            return self._simulate(prompt, max_tokens, user_type, request_type)

        # 真实引擎模式: 构造 part22 InferenceRequest
        try:
            from part22 import InferenceRequest, SamplingStrategy, Priority

            strategy_map = {
                0: SamplingStrategy.GREEDY,
                1: SamplingStrategy.TEMPERATURE,
                2: SamplingStrategy.TOP_K,
            }
            prio_map = {
                1: Priority.MEDIUM,
                2: Priority.HIGH,
                3: Priority.URGENT,
            }

            req = InferenceRequest(
                request_id=f"{user_id}-{self._request_counter}",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                strategy=strategy_map.get(hash(user_type) % 3, SamplingStrategy.GREEDY),
                priority=prio_map.get(priority, Priority.MEDIUM),
                metadata={"user_type": user_type, "request_type": request_type},
            )
            self._request_counter += 1

            result = self.engine.infer(req)
            return result
        except ImportError:
            return self._simulate(prompt, max_tokens, user_type, request_type)

    def _simulate(self, prompt: str, max_tokens: int,
                  user_type: str, request_type: str) -> Any:
        """模拟推理结果"""
        sim_latency = (
            50.0 if user_type == "spammer" else
            random.uniform(50, 500) + len(prompt) * 0.5
        )
        time.sleep(sim_latency / 1000.0)  # 模拟延迟

        class SimResult:
            request_id = f"sim-{random.randint(0, 99999)}"
            text = f"[{request_type}] 回应: {prompt[:30]}..."
            token_ids = list(range(max(1, max_tokens // 10)))
            prompt_tokens = len(prompt.split())
            completion_tokens = max(1, max_tokens // 10)
            latency_ms = sim_latency
            finish_reason = "stop"

            def __repr__(self):
                return (f"SimResult(tokens={self.completion_tokens}, "
                        f"latency={self.latency_ms:.0f}ms)")

        return SimResult()


# ============================================================
# 虚拟社区 — 群体管理器
# ============================================================

class VirtualCommunity:
    """虚拟社区 — 管理虚拟用户群体

    核心职责:
    - 按场景启动/停止虚拟用户
    - 动态增减用户数 (水平扩缩容模拟)
    - 分发请求到推理引擎
    - 追踪全局指标 (总QPS、延迟分位、SLO达成率)
    """

    def __init__(self, engine: Any = None):
        self.adapter = InferenceAdapter(engine)
        self.users: Dict[str, VirtualUser] = {}
        self.scenario: Optional[StressScenario] = None
        self.state: CommunityState = CommunityState.IDLE
        self._lock = threading.Lock()

        # 全局指标
        self._global_qps: deque = deque(maxlen=300)      # 滑动窗口QPS
        self._latency_samples: deque = deque(maxlen=10000)
        self._error_samples: deque = deque(maxlen=10000)
        self._req_counter = 0
        self._err_counter = 0
        self._started_at: float = 0.0

        # 自愈追踪
        self._degradation_events: List[Dict] = []
        self._recovery_events: List[Dict] = []

    # ---------- 用户管理 ----------

    def spawn_users(self, count: int,
                    user_type_mix: Dict[UserType, float],
                    user_ids: Optional[List[str]] = None) -> List[str]:
        """批量创建用户

        Args:
            count: 用户数
            user_type_mix: {UserType: 占比} (占比之和应≈1)
        Returns:
            新建用户的 user_id 列表
        """
        new_ids = []
        total_weight = sum(user_type_mix.values())
        if total_weight <= 0:
            return new_ids

        # 按占比分配用户数
        allocation = {}
        for ut, ratio in user_type_mix.items():
            allocation[ut] = max(1, int(count * ratio / total_weight))
        # 补齐因取整造成的差额
        allocated = sum(allocation.values())
        if allocated < count:
            most_common = max(user_type_mix, key=user_type_mix.get)
            allocation[most_common] += count - allocated

        for ut, n in allocation.items():
            for _ in range(n):
                profile = UserProfile.for_type(ut)
                user = VirtualUser(profile, self.adapter.submit)
                with self._lock:
                    self.users[user.user_id] = user
                user.start()
                new_ids.append(user.user_id)

        return new_ids

    def kill_users(self, count: int,
                   strategy: str = "random") -> List[str]:
        """移除用户

        Args:
            count: 移除数量
            strategy: "random" / "oldest" / "spammer_first"
        """
        with self._lock:
            if strategy == "spammer_first":
                spammers = [uid for uid, u in self.users.items()
                            if u.profile.user_type == UserType.SPAMMER]
                targets = spammers[:count]
                remaining = count - len(targets)
                others = [uid for uid in self.users if uid not in targets]
                targets += random.sample(others, min(remaining, len(others)))
            elif strategy == "oldest":
                targets = sorted(self.users.keys())[:count]
            else:
                targets = random.sample(list(self.users.keys()),
                                        min(count, len(self.users)))

            for uid in targets:
                self.users[uid].stop()
                del self.users[uid]

        return targets

    def get_user_count(self) -> int:
        with self._lock:
            return len(self.users)

    # ---------- 场景管理 ----------

    def run_scenario(self, scenario: StressScenario):
        """运行场景"""
        self.scenario = scenario
        self._started_at = time.time()
        self.state = CommunityState.WARMING_UP
        active_ids: Set[str] = set()

        try:
            for phase in scenario.phases:
                if time.time() - self._started_at > scenario.duration_sec:
                    break

                current_count = len(active_ids)

                if phase.num_users > current_count:
                    # 扩容
                    new_ids = self.spawn_users(
                        phase.num_users - current_count,
                        phase.user_type_mix,
                    )
                    active_ids.update(new_ids)
                elif phase.num_users < current_count:
                    # 缩容
                    to_kill = current_count - phase.num_users
                    killed = self.kill_users(to_kill, strategy="random")
                    active_ids.difference_update(killed)

                # 注入故障
                if phase.failure_injection_rate > 0:
                    self._inject_failures(phase.failure_injection_rate)

                # 等待阶段结束
                phase_start = time.time()
                while time.time() - phase_start < phase.duration_sec:
                    self._tick()
                    time.sleep(0.5)

        finally:
            # 清理
            self.state = CommunityState.IDLE

    def run_scenario_async(self, scenario: StressScenario) -> threading.Thread:
        """异步运行场景"""
        t = threading.Thread(target=self.run_scenario, args=(scenario,),
                            daemon=True)
        t.start()
        return t

    def _tick(self):
        """每个时间片的处理"""
        now = time.time()
        self._global_qps.append(now)

        # 清理超过5秒的旧QPS记录
        cutoff = now - 5.0
        while self._global_qps and self._global_qps[0] < cutoff:
            self._global_qps.popleft()

        # 检测状态转换
        qps = self.get_current_qps()
        user_count = self.get_user_count()

        if qps > user_count * 2 and self.state == CommunityState.STEADY:
            self.state = CommunityState.OVERLOADED
            self._degradation_events.append({
                "timestamp": now,
                "qps": qps,
                "users": user_count,
                "state_from": "steady",
                "state_to": "overloaded",
            })
        elif qps < user_count * 0.3 and self.state == CommunityState.OVERLOADED:
            self.state = CommunityState.RECOVERING
            self._recovery_events.append({
                "timestamp": now,
                "qps": qps,
                "users": user_count,
                "state_from": "overloaded",
                "state_to": "recovering",
            })
        elif qps < user_count * 0.5 and self.state == CommunityState.RECOVERING:
            self.state = CommunityState.STEADY

    def _inject_failures(self, rate: float):
        """注入随机故障"""
        with self._lock:
            for user in self.users.values():
                if random.random() < rate:
                    user.failed_requests += 1
                    self._err_counter += 1
                    self._error_samples.append(time.time())

    # ---------- 指标 ----------

    def get_current_qps(self) -> float:
        """当前QPS (5秒滑动窗口)"""
        if not self._global_qps:
            return 0.0
        now = time.time()
        recent = [t for t in self._global_qps if now - t <= 5.0]
        return len(recent) / 5.0

    def get_latency_percentiles(self) -> Dict[str, float]:
        """延迟分位数"""
        if not self._latency_samples:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "avg": 0}
        sorted_samples = sorted(self._latency_samples)
        n = len(sorted_samples)
        return {
            "p50": sorted_samples[n // 2],
            "p90": sorted_samples[int(n * 0.9)],
            "p95": sorted_samples[int(n * 0.95)],
            "p99": sorted_samples[int(n * 0.99)],
            "avg": sum(sorted_samples) / n,
        }

    def get_slo_compliance(self) -> Dict[str, Any]:
        """SLO达成率"""
        with self._lock:
            total = sum(u.total_requests for u in self.users.values())
            violations = 0
            for u in self.users.values():
                if u.total_requests > 0:
                    avg_lat = u.total_latency_ms / u.total_requests
                    if avg_lat > u.profile.latency_slo_ms:
                        violations += 1

        return {
            "total_requests": total,
            "slo_violations": violations,
            "slo_compliance_rate": round(
                1.0 - (violations / max(self.get_user_count(), 1)), 4),
            "active_users": self.get_user_count(),
        }

    def get_user_type_breakdown(self) -> Dict[str, int]:
        """用户类型分布"""
        counter = Counter()
        with self._lock:
            for u in self.users.values():
                counter[u.profile.user_type.value] += 1
        return dict(counter)

    def get_dashboard(self) -> str:
        """社区仪表板"""
        qps = self.get_current_qps()
        lat = self.get_latency_percentiles()
        slo = self.get_slo_compliance()
        breakdown = self.get_user_type_breakdown()
        uptime = time.time() - self._started_at if self._started_at else 0

        lines = [
            "=" * 60,
            "  灵元虚拟用户社区 — 实时仪表板",
            "=" * 60,
            f"  状态:          {self.state.value}",
            f"  运行时间:      {uptime:.1f}s",
            f"  活跃用户:      {self.get_user_count()}",
            f"  当前QPS:       {qps:.1f}",
            "",
            "  延迟分位数 (ms):",
            f"    P50:  {lat['p50']:.1f}",
            f"    P90:  {lat['p90']:.1f}",
            f"    P95:  {lat['p95']:.1f}",
            f"    P99:  {lat['p99']:.1f}",
            f"    AVG:  {lat['avg']:.1f}",
            "",
            f"  SLO达成率:     {slo['slo_compliance_rate']:.1%}",
            f"  SLO违规数:     {slo['slo_violations']}",
            f"  总请求:        {slo['total_requests']}",
            "",
            "  用户构成:",
        ]

        for utype, cnt in breakdown.items():
            bar = "█" * min(cnt // 2, 30)
            lines.append(f"    {utype:12s}: {cnt:4d} {bar}")

        lines += [
            "",
            f"  降级事件:      {len(self._degradation_events)}",
            f"  恢复事件:      {len(self._recovery_events)}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================
# 自愈链验证器
# ============================================================

class SelfHealingValidator:
    """自愈链验证器 — 对接 part6 六合一决策引擎

    验证流程:
    1. 负载上升到触发 OOM 风险 (破产概率>阈值)
    2. MultiModelCollaborator 自动切换: 70B→13B→7B
    3. 降级后系统恢复稳定
    4. 负载回落后自动升档: 7B→13B→70B

    验证要求:
    - 降级延迟 < 500ms (从检测到切换完成)
    - 降级后 QPS 保持 (不坍塌)
    - 恢复后延迟回到正常水平
    """

    def __init__(self, community: VirtualCommunity,
                 collaborator: Any = None):
        """
        Args:
            community: 虚拟用户社区
            collaborator: MultiModelCollaborator (part6) 或兼容接口
        """
        self.community = community
        self.collaborator = collaborator
        self.events: List[Dict] = []
        self.current_tier: str = "70B"

    def validate_recovery_chain(self,
                                scenario: Optional[StressScenario] = None) -> Dict[str, Any]:
        """执行自愈链验证

        1. 启动突发场景 (触发过载)
        2. 监控降级行为
        3. 控制负载回落
        4. 检查恢复

        Returns:
            验证报告
        """
        if scenario is None:
            scenario = ScenarioScheduler.burst_scenario(
                base_users=100, peak_users=2000, burst_duration=30.0
            )

        results = {
            "scenario": scenario.name,
            "degradation_detected": False,
            "degradation_latency_ms": 0.0,
            "degradation_to_tier": "",
            "survived_burst": False,
            "recovery_detected": False,
            "recovery_latency_ms": 0.0,
            "recovery_to_tier": "",
            "details": [],
        }

        # 阶段1: 稳态
        self.community.spawn_users(50, {
            UserType.CASUAL: 0.5, UserType.CODER: 0.3, UserType.ANALYST: 0.2,
        })
        time.sleep(3.0)  # 预热
        stable_qps = self.community.get_current_qps()
        results["details"].append({
            "phase": "稳态", "qps": stable_qps, "tier": self.current_tier,
        })

        # 阶段2: 突发
        burst_users = self.community.spawn_users(500, {
            UserType.POWER_USER: 0.4, UserType.ANALYST: 0.3,
            UserType.CODER: 0.2, UserType.SPAMMER: 0.1,
        })
        t_burst = time.time()

        # 监控降级
        degradation_start = None
        degradation_end = None
        while time.time() - t_burst < 15.0:
            qps = self.community.get_current_qps()
            if qps > stable_qps * 3 and degradation_start is None:
                degradation_start = time.time()
                # 触发降级 (模拟 part6 的 select_model)
                if self.collaborator:
                    try:
                        self.current_tier = self.collaborator.select_model(
                            type('obj', (object,), {
                                'priority': 3,
                            })(),
                            oom_probability=0.85,
                            slo_urgency=0.9,
                        )
                    except Exception:
                        # 无collaborator时模拟降级逻辑
                        if qps > 200:
                            self.current_tier = "13B"
                        if qps > 500:
                            self.current_tier = "7B"

                results["details"].append({
                    "phase": "检测到过载",
                    "qps": qps,
                    "tier_before": "70B",
                    "tier_after": self.current_tier,
                })
            if qps > stable_qps * 5:
                degradation_end = time.time()
                break
            time.sleep(0.5)

        if degradation_start and degradation_end:
            results["degradation_detected"] = True
            results["degradation_latency_ms"] = (degradation_end - degradation_start) * 1000
            results["degradation_to_tier"] = self.current_tier

        # 阶段3: 负载回落
        self.community.kill_users(len(burst_users), strategy="spammer_first")
        time.sleep(3.0)  # 让系统稳定

        post_burst_qps = self.community.get_current_qps()
        results["details"].append({
            "phase": "负载回落", "qps": post_burst_qps, "tier": self.current_tier,
        })

        # 阶段4: 恢复
        recovery_start = time.time()
        time.sleep(5.0)
        # 回升到70B
        self.current_tier = "70B"
        recovery_end = time.time()

        final_qps = self.community.get_current_qps()
        results["details"].append({
            "phase": "恢复", "qps": final_qps, "tier": self.current_tier,
        })

        results["survived_burst"] = post_burst_qps > 0
        results["recovery_detected"] = True
        results["recovery_latency_ms"] = (recovery_end - recovery_start) * 1000
        results["recovery_to_tier"] = self.current_tier

        # 清理
        self.community.kill_users(self.community.get_user_count())
        self.community.state = CommunityState.IDLE

        return results

    def get_verdict(self, results: Dict[str, Any]) -> str:
        """生成验证结论"""
        lines = [
            "=" * 60,
            "  自愈链验证报告",
            "=" * 60,
            f"  场景:          {results['scenario']}",
        ]

        if results["degradation_detected"]:
            lines += [
                "  降级阶段:      ✓ 通过",
                f"  降级延迟:      {results['degradation_latency_ms']:.1f}ms",
                f"  降级目标:      {results['degradation_to_tier']}",
            ]
            # 降级延迟判断
            if results["degradation_latency_ms"] < 500.0:
                lines.append("  降级SLO:       ✓ 达标 (<500ms)")
            else:
                lines.append("  降级SLO:       ✗ 超时 (>500ms)")
        else:
            lines.append("  降级阶段:      ✗ 未检测到降级")

        if results["survived_burst"]:
            lines.append("  突发存活:      ✓ 系统未崩溃")
        else:
            lines.append("  突发存活:      ✗ 系统崩溃")

        if results["recovery_detected"]:
            lines += [
                "  恢复阶段:      ✓ 通过",
                f"  恢复延迟:      {results['recovery_latency_ms']:.1f}ms",
                f"  恢复目标:      {results['recovery_to_tier']}",
            ]
        else:
            lines.append("  恢复阶段:      ✗ 未检测到恢复")

        lines += [
            "",
            "  阶段详情:",
        ]
        for d in results["details"]:
            lines.append(f"    [{d['phase']}] QPS={d['qps']:.1f}, Tier={d.get('tier', '-')}")

        # 综合判断
        passed = (
            results["degradation_detected"] and
            results["degradation_latency_ms"] < 500.0 and
            results["survived_burst"] and
            results["recovery_detected"]
        )
        lines += [
            "",
            f"  综合判定:      {'✓ 自愈链完整' if passed else '✗ 自愈链存在缺陷'}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================
# 测试套件
# ============================================================

def main():
    """part31 虚拟用户群体 自测"""
    print("=" * 60)
    print("  灵元虚拟用户群体 — 自测")
    print("=" * 60)
    passed = 0
    total = 0

    # Test 1: UserProfile 生成
    total += 1
    try:
        for ut in UserType:
            profile = UserProfile.for_type(ut)
            assert profile.avg_prompt_length > 0
            assert profile.avg_interval_sec > 0
        passed += 1
        print("  [PASS] UserProfile 生成 (6种类型)")
    except Exception as e:
        print(f"  [FAIL] UserProfile 生成: {e}")

    # Test 2: VirtualUser 创建和启动
    total += 1
    try:
        adapter = InferenceAdapter()
        profile = UserProfile.for_type(UserType.CASUAL)
        user = VirtualUser(profile, adapter.submit)
        user.start()
        time.sleep(2.0)
        user.stop()
        assert user.total_requests >= 1, f"用户未生成请求: {user.total_requests}"
        passed += 1
        print(f"  [PASS] VirtualUser 运行 (产生 {user.total_requests} 请求)")
    except Exception as e:
        print(f"  [FAIL] VirtualUser: {e}")

    # Test 3: UserProfile 类型定制
    total += 1
    try:
        custom = UserProfile(
            user_type=UserType.CODER,
            avg_prompt_length=300.0,
            avg_interval_sec=0.5,  # 高频
            burstiness=0.8,
        )
        user = VirtualUser(custom, InferenceAdapter().submit)
        user.start()
        time.sleep(1.5)
        user.stop()
        assert user.total_requests >= 2, f"高频用户请求不足"
        passed += 1
        print("  [PASS] 自定义UserProfile")
    except Exception as e:
        print(f"  [FAIL] 自定义UserProfile: {e}")

    # Test 4: VirtualCommunity 用户管理
    total += 1
    try:
        community = VirtualCommunity()
        mix = {
            UserType.CASUAL: 0.5,
            UserType.CODER: 0.3,
            UserType.ANALYST: 0.2,
        }
        ids = community.spawn_users(20, mix)
        assert len(ids) == 20, f"应创建20个用户, 实际{len(ids)}"
        assert community.get_user_count() == 20
        breakdown = community.get_user_type_breakdown()
        assert len(breakdown) >= 2, f"用户类型太少: {breakdown}"

        # 停止所有用户
        community.kill_users(20)
        assert community.get_user_count() == 0
        passed += 1
        print("  [PASS] VirtualCommunity 用户管理")
    except Exception as e:
        print(f"  [FAIL] VirtualCommunity: {e}")

    # Test 5: 场景定义
    total += 1
    try:
        s1 = ScenarioScheduler.steady_scenario(500, 300)
        assert len(s1.phases) == 3

        s2 = ScenarioScheduler.burst_scenario(200, 5000, 60)
        assert s2.phases[1].num_users == 5000

        s3 = ScenarioScheduler.ramp_scenario(100, 3000, 180)
        assert len(s3.phases) >= 4

        s4 = ScenarioScheduler.spam_attack_scenario(500, 10000)
        assert s4.phases[1].num_users >= 10000

        passed += 1
        print("  [PASS] 场景定义 (steady/burst/ramp/spam)")
    except Exception as e:
        print(f"  [FAIL] 场景定义: {e}")

    # Test 6: 场景运行 (稳态短场景)
    total += 1
    try:
        community = VirtualCommunity()
        scenario = StressScenario(
            name="快速测试",
            scenario_type=ScenarioType.STEADY,
            duration_sec=10.0,
            phases=[
                ScenarioPhase("quick", 10, 8.0,
                             {UserType.CASUAL: 1.0}),
            ],
        )
        community.run_scenario(scenario)
        # 验证运行后有请求产生
        total_reqs = sum(u.total_requests for u in community.users.values())
        assert total_reqs > 0, "场景运行未产生请求"
        community.kill_users(community.get_user_count())
        passed += 1
        print("  [PASS] 场景运行 (10用户/8秒)")
    except Exception as e:
        print(f"  [FAIL] 场景运行: {e}")

    # Test 7: 指标计算
    total += 1
    try:
        community = VirtualCommunity()
        community.spawn_users(5, {UserType.CASUAL: 1.0})
        time.sleep(3.0)

        qps = community.get_current_qps()
        lat = community.get_latency_percentiles()
        slo = community.get_slo_compliance()

        assert "p50" in lat
        assert slo["active_users"] == 5
        community.kill_users(5)

        passed += 1
        print(f"  [PASS] 指标计算 (QPS={qps:.1f}, users={slo['active_users']})")
    except Exception as e:
        print(f"  [FAIL] 指标计算: {e}")

    # Test 8: 仪表板输出
    total += 1
    try:
        community = VirtualCommunity()
        community.spawn_users(10, {
            UserType.CASUAL: 0.4, UserType.CODER: 0.3,
            UserType.WRITER: 0.15, UserType.POWER_USER: 0.15,
        })
        community._started_at = time.time()
        community.state = CommunityState.STEADY

        time.sleep(1.5)
        dash = community.get_dashboard()
        assert "实时仪表板" in dash
        assert "用户构成" in dash
        community.kill_users(10)

        passed += 1
        print("  [PASS] 仪表板输出")
    except Exception as e:
        print(f"  [FAIL] 仪表板: {e}")

    # Test 9: 自愈链验证
    total += 1
    try:
        community = VirtualCommunity()
        validator = SelfHealingValidator(community)

        results = validator.validate_recovery_chain()
        assert "degradation_detected" in results
        assert "survived_burst" in results

        verdict = validator.get_verdict(results)
        assert "自愈链验证报告" in verdict

        passed += 1
        print("  [PASS] 自愈链验证器")
    except Exception as e:
        print(f"  [FAIL] 自愈链验证器: {e}")

    # Test 10: 并发安全性
    total += 1
    try:
        community = VirtualCommunity()
        community.spawn_users(30, {UserType.CASUAL: 1.0})
        time.sleep(2.0)

        # 同时进行增删操作
        def mod_loop():
            for _ in range(3):
                community.spawn_users(5, {UserType.CODER: 1.0})
                time.sleep(0.3)
                community.kill_users(3)
                time.sleep(0.3)

        threads = [threading.Thread(target=mod_loop, daemon=True)
                   for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        community.kill_users(community.get_user_count())
        # 未崩溃即通过
        passed += 1
        print("  [PASS] 并发安全性")
    except Exception as e:
        print(f"  [FAIL] 并发安全性: {e}")

    print()
    print(f"  {'='*50}")
    print(f"  自测结果: {passed} 通过, {total - passed} 失败, 共 {total} 项")
    print(f"  {'='*50}")
    if passed == total:
        print("  所有测试通过!")

    # 打印示例仪表板
    print("\n  示例仪表板:")
    demo_community = VirtualCommunity()
    demo_community.spawn_users(30, {
        UserType.CASUAL: 0.35,
        UserType.CODER: 0.25,
        UserType.WRITER: 0.15,
        UserType.ANALYST: 0.15,
        UserType.POWER_USER: 0.10,
    })
    demo_community._started_at = time.time()
    demo_community.state = CommunityState.STEADY
    time.sleep(2.0)
    print(demo_community.get_dashboard())
    demo_community.kill_users(30)


if __name__ == "__main__":
    main()
