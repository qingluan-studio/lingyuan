
# ============================================================
# LINGYUAN MODEL - PART 6
# 六层级递进融合决策引擎 (Six-Layer Recursive Fusion Decision Engine)
#
# 从马拉松配速到模型推理的通用框架
# 核心理念: 资源受限下的最优决策 —— 数学同构映射
#
# 架构:
#   算法0: 熵增三联监测 (Entropy Triple Monitor)
#   算法1: 后悔最小化配速器 (Regret Minimization Pacer)
#   算法2: 心率-坡度耦合递推 (HR-Slope Coupling Recursion)
#   算法3: 能量破产概率前馈 (Energy Bankruptcy Probability)
#   算法4: 最优痛苦分布策略 (Optimal Pain Distribution)
#   算法5: 参赛者群体动力学 (Participant Group Dynamics)
#
#   第一级: 算法2+3 → OOM破产预警器
#   第二级: 第一级 + 算法1+4 → 双层SLO优化器
#   第三级: 第二级 + 算法5 → 多模型协同推理
# ============================================================


# ============================================================
# ALGORITHM 0: 熵增三联监测 (内存熵三联)
# ============================================================

@dataclass
class EntropyTripleState:
    """熵增三联状态"""
    fragmentation: float = 0.0        # 内存碎片率 (0-1)
    swap_jitter: float = 0.0          # swap频率抖动 (相对基线的倍数)
    vram_decline_rate: float = 0.0    # 可用显存下降速率 (GB/s)

    # 阈值触发标记
    frag_triggered: bool = False      # 碎片率越线
    swap_triggered: bool = False      # swap抖动越线
    vram_triggered: bool = False      # 显存衰减越线

    triggered_count: int = 0          # 三联触发计数
    drift_multiplier: float = 1.0     # 漂移项修正系数
    consecutive_minutes: int = 0      # 连续触发分钟数
    force_shrink: bool = False        # 是否强制收缩可行域


class EntropyTripleMonitor:
    """熵增三联监测器 (算法0)

    推理版三联指标:
    - 碎片率 → 连续上升触发KV cache预淘汰
    - swap频率 → 抖动超过基线2倍触发降batch size
    - 可用显存下降速率 → 加速度超过阈值触发模型切换

    融合方式:
    - 单指标触发 → 漂移项 ×1.2
    - 双指标触发 → 漂移项 ×1.5
    - 三联全触发 → 漂移项 ×2.0 + 强制可行域收缩

    内层软触发: 任一项连续3分钟越线 → 强制收缩优化空间
    """

    # 触发阈值 (相对基线的变化幅度)
    FRAG_THRESHOLD = 0.15          # 碎片率绝对阈值
    SWAP_JITTER_THRESHOLD = 2.0   # swap抖动倍数阈值
    VRAM_DECLINE_THRESHOLD = 0.5  # 显存下降速率阈值 (GB/s)

    def __init__(self):
        self.baseline_frag: float = 0.05      # 碎片率基线
        self.baseline_swap: float = 1.0        # swap频率基线
        self.history: List[EntropyTripleState] = []
        self.current_state = EntropyTripleState()
        self._consecutive_trigger_count = 0   # 连续触发计数器

    def update(self, fragmentation: float, swap_freq_ratio: float,
               vram_decline_rate: float) -> EntropyTripleState:
        """更新三联状态

        Args:
            fragmentation: 当前内存碎片率 (0-1)
            swap_freq_ratio: 当前swap频率 / 基线swap频率
            vram_decline_rate: 可用显存下降速率 (GB/s)
        """
        state = EntropyTripleState(
            fragmentation=fragmentation,
            swap_jitter=swap_freq_ratio,
            vram_decline_rate=vram_decline_rate,
        )

        # 逐项检查阈值
        state.frag_triggered = fragmentation > self.FRAG_THRESHOLD
        state.swap_triggered = swap_freq_ratio > self.SWAP_JITTER_THRESHOLD
        state.vram_triggered = vram_decline_rate > self.VRAM_DECLINE_THRESHOLD

        state.triggered_count = sum([
            state.frag_triggered, state.swap_triggered, state.vram_triggered
        ])

        # 漂移项修正系数
        if state.triggered_count == 0:
            state.drift_multiplier = 1.0
            self._consecutive_trigger_count = 0
        elif state.triggered_count == 1:
            state.drift_multiplier = 1.2
        elif state.triggered_count == 2:
            state.drift_multiplier = 1.5
        else:  # 三联全触发
            state.drift_multiplier = 2.0
            state.force_shrink = True

        # 连续触发检测 (内层软触发)
        if state.triggered_count > 0:
            self._consecutive_trigger_count += 1
            state.consecutive_minutes = self._consecutive_trigger_count
            # 连续3次(模拟3分钟)越线 → 强制收缩
            if self._consecutive_trigger_count >= 3:
                state.force_shrink = True
        else:
            self._consecutive_trigger_count = 0

        self.current_state = state
        self.history.append(state)
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return state

    def get_drift_correction(self) -> float:
        """获取漂移项修正系数 (供算法3使用)"""
        return self.current_state.drift_multiplier

    def should_force_shrink(self) -> bool:
        """是否应强制收缩可行域"""
        return self.current_state.force_shrink

    def get_status(self) -> Dict:
        """获取三联状态摘要"""
        s = self.current_state
        return {
            "fragmentation": round(s.fragmentation, 4),
            "swap_jitter": round(s.swap_jitter, 2),
            "vram_decline_rate": round(s.vram_decline_rate, 4),
            "triggered_count": s.triggered_count,
            "drift_multiplier": s.drift_multiplier,
            "force_shrink": s.force_shrink,
            "consecutive_minutes": s.consecutive_minutes,
            "indicators": {
                "frag": "⚠" if s.frag_triggered else "✓",
                "swap": "⚠" if s.swap_triggered else "✓",
                "vram": "⚠" if s.vram_triggered else "✓",
            },
        }


# ============================================================
# ALGORITHM 1: 后悔最小化配速器 (在线凸优化)
# ============================================================

class RegretMinimizer:
    """后悔最小化配速器 (算法1)

    推理映射: per-token/per-batch 调节生成速率，追回SLO偏差

    核心思路:
    - 维护历史决策的累积后悔值
    - 每轮选择使期望后悔最小的策略
    - 在线凸优化框架，简化版FTL (Follow The Leader)
    """

    def __init__(self, learning_rate: float = 0.1):
        self.lr = learning_rate
        self.accumulated_regret: float = 0.0
        self.decision_history: List[Dict] = []
        self.current_rate: float = 1.0  # 当前生成速率倍率

    def update(self, actual_rate: float, target_rate: float,
               oom_probability: float = 0.0,
               force_shrink: bool = False) -> float:
        """更新决策

        Args:
            actual_rate: 上一轮实际生成速率
            target_rate: 目标生成速率 (来自痛苦曲线)
            oom_probability: 当前OOM概率
            force_shrink: 是否被熵增三联强制收缩
        Returns:
            调整后的生成速率倍率
        """
        # 计算瞬时后悔 (实际与目标的偏差)
        instant_regret = (actual_rate - target_rate) ** 2
        self.accumulated_regret += instant_regret

        # 在线梯度下降调整速率
        gradient = 2 * (self.current_rate - target_rate)
        self.current_rate -= self.lr * gradient

        # OOM约束: 概率越高，越保守
        if oom_probability > 0.2:
            self.current_rate *= (1 - oom_probability * 0.5)

        # 熵增三联强制收缩
        if force_shrink:
            self.current_rate *= 0.7

        # 速率限制在合理范围 [0.3, 2.0]
        self.current_rate = max(0.3, min(2.0, self.current_rate))

        self.decision_history.append({
            "actual_rate": actual_rate,
            "target_rate": target_rate,
            "adjusted_rate": self.current_rate,
            "instant_regret": round(instant_regret, 6),
            "accumulated_regret": round(self.accumulated_regret, 6),
            "oom_probability": oom_probability,
            "force_shrink": force_shrink,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.decision_history) > 500:
            self.decision_history = self.decision_history[-500:]

        return self.current_rate

    def get_stats(self) -> Dict:
        return {
            "current_rate": round(self.current_rate, 4),
            "accumulated_regret": round(self.accumulated_regret, 6),
            "decisions": len(self.decision_history),
        }


# ============================================================
# ALGORITHM 2: 心率-坡度耦合递推 (KV命中率-序列长度耦合)
# ============================================================

@dataclass
class CouplingState:
    """耦合状态"""
    kv_hit_rate: float = 0.0       # KV cache命中率
    seq_length: int = 0            # 当前序列长度
    batch_size: int = 1            # 当前batch大小
    coupling_factor: float = 1.0   # 耦合系数 (消耗速率)
    stress_level: float = 0.0      # 压力等级 (0-1)


class HRSlopeCoupling:
    """心率-坡度耦合递推 (算法2)

    推理映射: KV命中率-序列长度耦合
    - 心率 → KV cache命中率 (反映缓存压力)
    - 坡度 → 序列长度 (反映请求复杂度)
    - 耦合方程给出此刻资源消耗速率

    状态方程:
    d(HR)/dt = α · HR + β · slope + γ · batch_size
    映射:
    d(KV_hit)/dt = α · KV_hit + β · seq_len + γ · batch_size
    """

    def __init__(self):
        self.state = CouplingState()
        self.history: List[Dict] = []
        # 耦合方程系数
        self.alpha = -0.02   # 自衰减 (KV命中率自然下降)
        self.beta = 0.001    # 序列长度影响 (越长命中率越降)
        self.gamma = 0.005   # batch影响 (越大压力越大)

    def update(self, kv_hit_rate: float, seq_length: int,
               batch_size: int) -> CouplingState:
        """更新耦合状态"""
        self.state.kv_hit_rate = kv_hit_rate
        self.state.seq_length = seq_length
        self.state.batch_size = batch_size

        # 耦合方程: 计算消耗速率
        # coupling_factor > 1 表示资源消耗加速
        base_consumption = 1.0
        seq_penalty = self.beta * max(seq_length - 512, 0)  # 超过512 token开始惩罚
        batch_penalty = self.gamma * max(batch_size - 1, 0)
        self.state.coupling_factor = base_consumption + seq_penalty + batch_penalty

        # 压力等级 (0-1): 综合KV命中率和耦合因子
        kv_stress = max(0, 1 - kv_hit_rate)  # 命中率越低压力越大
        coupling_stress = min(1, (self.state.coupling_factor - 1) / 2)
        self.state.stress_level = round(kv_stress * 0.6 + coupling_stress * 0.4, 4)

        self.history.append({
            "kv_hit_rate": kv_hit_rate,
            "seq_length": seq_length,
            "batch_size": batch_size,
            "coupling_factor": round(self.state.coupling_factor, 4),
            "stress_level": self.state.stress_level,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return self.state

    def get_consumption_rate(self) -> float:
        """获取当前资源消耗速率"""
        return self.state.coupling_factor

    def get_stress_level(self) -> float:
        """获取压力等级"""
        return self.state.stress_level

    def get_status(self) -> Dict:
        s = self.state
        return {
            "kv_hit_rate": round(s.kv_hit_rate, 4),
            "seq_length": s.seq_length,
            "batch_size": s.batch_size,
            "coupling_factor": round(s.coupling_factor, 4),
            "stress_level": s.stress_level,
        }


# ============================================================
# ALGORITHM 3: 能量破产概率前馈 (OOM破产概率)
# ============================================================

class OOMBancruptcyPredictor:
    """能量破产概率前馈 (算法3)

    推理映射: OOM破产概率预测

    核心模型: 非平稳随机过程首达时间
    dE = μ(s) dt + σ(s) dW_t

    其中:
    - E: 剩余资源 (显存/KV cache)
    - μ(s): 漂移项 (资源消耗速率), 由算法2耦合方程给出
    - σ(s): 扩散项 (随机负载波动)
    - W_t: 维纳过程

    漂移项修正: μ(s) = μ₀ · exp(γ · s / S_total) · drift_multiplier
    其中 drift_multiplier 来自算法0熵增三联
    """

    def __init__(self):
        self.entropy_monitor = EntropyTripleMonitor()
        self.coupling = HRSlopeCoupling()
        self.history: List[Dict] = []

        # 随机过程参数
        self.mu_0: float = 0.5        # 基础漂移 (GB/s)
        self.gamma: float = 0.1        # 疲劳加速度系数
        self.sigma_base: float = 0.1   # 基础扩散
        self.total_resource: float = 80.0  # 总资源量 (GB, 模拟80GB显存)

        # 蒙特卡洛参数
        self.mc_samples: int = 200     # 蒙特卡洛模拟次数
        self.dt: float = 0.1           # 时间步长 (秒)

    def predict(self, remaining_resource: float, consumption_rate: float,
                time_horizon: float = 10.0,
                fragmentation: float = 0.05,
                swap_ratio: float = 1.0,
                vram_decline: float = 0.0) -> Dict:
        """预测OOM破产概率

        Args:
            remaining_resource: 当前剩余资源 (GB)
            consumption_rate: 当前消耗速率 (来自算法2)
            time_horizon: 预测时间窗口 (秒)
            fragmentation: 内存碎片率 (算法0输入)
            swap_ratio: swap频率比 (算法0输入)
            vram_decline: 显存下降速率 (算法0输入)
        Returns:
            {
                "bankruptcy_prob": 破产概率,
                "expected_time_to_oom": 预期OOM时间 (秒),
                "drift_multiplier": 漂移修正系数,
                "entropy_state": 熵增三联状态,
            }
        """
        # 1. 更新熵增三联
        entropy_state = self.entropy_monitor.update(
            fragmentation, swap_ratio, vram_decline
        )
        drift_multiplier = entropy_state.drift_multiplier

        # 2. 计算非平稳漂移项
        # μ(s) = μ₀ · exp(γ · s / S_total) · drift_multiplier
        s_consumed = self.total_resource - remaining_resource
        mu = self.mu_0 * math.exp(
            self.gamma * s_consumed / max(self.total_resource, 1)
        ) * drift_multiplier

        # 消耗速率来自算法2耦合方程
        mu *= consumption_rate

        # 3. 扩散项 (压力越大波动越大)
        stress = self.coupling.get_stress_level()
        sigma = self.sigma_base * (1 + stress * 3)

        # 4. 蒙特卡洛模拟首达时间
        bankruptcy_count = 0
        first_passage_times = []

        for _ in range(self.mc_samples):
            E = remaining_resource
            t = 0.0
            hit_zero = False

            while t < time_horizon and E > 0:
                # 欧拉-丸山离散化
                dW = random.gauss(0, math.sqrt(self.dt))
                dE = mu * self.dt + sigma * dW * math.sqrt(self.dt)
                E += dE
                E = min(E, self.total_resource)  # 不能超过总量
                t += self.dt

                if E <= 0:
                    hit_zero = True
                    first_passage_times.append(t)
                    break

            if hit_zero:
                bankruptcy_count += 1
            else:
                first_passage_times.append(time_horizon)

        prob = bankruptcy_count / self.mc_samples
        expected_time = sum(first_passage_times) / len(first_passage_times)

        result = {
            "bankruptcy_prob": round(prob, 4),
            "expected_time_to_oom": round(expected_time, 2),
            "drift_multiplier": drift_multiplier,
            "mu": round(mu, 4),
            "sigma": round(sigma, 4),
            "entropy_state": self.entropy_monitor.get_status(),
            "time_horizon": time_horizon,
            "remaining_resource": round(remaining_resource, 2),
            "force_shrink": entropy_state.force_shrink,
        }

        self.history.append({
            **result,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return result

    def is_critical(self, threshold: float = 0.3) -> bool:
        """当前破产概率是否超过阈值"""
        if self.history:
            return self.history[-1]["bankruptcy_prob"] >= threshold
        return False

    def get_latest(self) -> Dict:
        if self.history:
            return self.history[-1]
        return {}


# ============================================================
# ALGORITHM 4: 最优痛苦分布策略 (延迟SLO曲线)
# ============================================================

class PainCurveOptimizer:
    """最优痛苦分布策略 (算法4)

    推理映射: 最优延迟SLO分布曲线

    核心模型: 变分法求最优痛苦分布
    min ∫₀ᵀ δᵗ · p(t) dt

    其中:
    - p(t): t时刻的痛苦 (延迟偏离SLO的量)
    - δ: 痛苦贴现因子 (<1, 远端痛苦权重更小)
    - T: 总时间窗口

    非对称修正: 人性偏好推迟痛苦 → 算法反向补偿
    → 最优策略比对称版本更"前慢" (推理: 前期保守留余量)

    简化实现: 离散化 + 梯度下降
    """

    def __init__(self, discount_factor: float = 0.95):
        self.discount = discount_factor    # 痛苦贴现因子
        self.optimal_curve: List[float] = []  # 最优痛苦曲线
        self.total_budget: float = 0.0     # 总痛苦预算
        self.time_horizon: int = 0         # 时间步数
        self.negotiation_history: List[Dict] = []

    def solve_optimal_curve(self, total_budget: float, time_steps: int = 60) -> List[float]:
        """求解最优痛苦分布曲线

        Args:
            total_budget: 总痛苦预算 (总允许延迟偏差)
            time_steps: 离散化步数
        Returns:
            最优痛苦曲线 (每个时间步的痛苦分配)
        """
        self.total_budget = total_budget
        self.time_horizon = time_steps

        # 闭式解: p_t ∝ δ^t (贴现分布), 归一化到预算
        # 人性偏好反向补偿: 把10%的后期痛苦前移
        raw = [self.discount ** t for t in range(time_steps)]
        raw_sum = sum(raw)
        curve = [total_budget * r / raw_sum for r in raw]

        # 人性偏好反向补偿: 把10%的后期痛苦前移
        compensation = curve[-1] * 0.1
        curve[-1] -= compensation
        curve[0] += compensation

        self.optimal_curve = [round(p, 6) for p in curve]
        return self.optimal_curve

    def get_target_at(self, step: int) -> float:
        """获取指定时间步的目标痛苦值"""
        if 0 <= step < len(self.optimal_curve):
            return self.optimal_curve[step]
        return 0.0

    def negotiate_pain_futures(self, current_pain: float, target_pain: float,
                               action_cost: float) -> Dict:
        """痛苦期货协商

        推理映射: 延迟期货 — 某时刻延迟略超SLO但换来后续大段低延迟

        判断: 当下额外痛苦(action_cost)能否换来更低全程总痛苦

        Args:
            current_pain: 当前实际痛苦
            target_pain: 目标痛苦 (来自曲线)
            action_cost: 执行某操作(如提前加速/切模型)的额外痛苦
        Returns:
            {
                "should_execute": 是否执行,
                "net_benefit": 净收益 (正=值得),
                "borrowed_pain": 借入的痛苦,
                "future_relief": 未来减轻的痛苦,
            }
        """
        # 借入痛苦: 当下多承担action_cost
        borrowed = action_cost
        # 未来减负: 假设操作能减少后续20%的痛苦 (简化模型)
        future_steps = max(len(self.optimal_curve) - 1, 1)
        future_total = sum(self.optimal_curve[1:])
        future_relief = future_total * 0.2

        # 净收益 = 未来减负 - 借入痛苦 (都需贴现)
        discounted_borrowed = borrowed  # 当下不贴现
        discounted_relief = future_relief * self.discount
        net_benefit = discounted_relief - discounted_borrowed

        should_execute = net_benefit > 0

        result = {
            "should_execute": should_execute,
            "net_benefit": round(net_benefit, 6),
            "borrowed_pain": round(borrowed, 6),
            "future_relief": round(future_relief, 6),
            "discounted_relief": round(discounted_relief, 6),
            "current_pain": round(current_pain, 6),
            "target_pain": round(target_pain, 6),
        }

        self.negotiation_history.append({
            **result,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.negotiation_history) > 200:
            self.negotiation_history = self.negotiation_history[-200:]

        return result

    def get_status(self) -> Dict:
        return {
            "discount_factor": self.discount,
            "total_budget": round(self.total_budget, 4),
            "time_horizon": self.time_horizon,
            "curve_length": len(self.optimal_curve),
            "curve_preview": self.optimal_curve[:5] + ["..."] + self.optimal_curve[-5:] if len(self.optimal_curve) > 10 else self.optimal_curve,
            "negotiations": len(self.negotiation_history),
        }


# ============================================================
# ALGORITHM 5: 参赛者群体动力学 (多模型协同推理)
# ============================================================

@dataclass
class ModelTier:
    """模型层级定义"""
    tier_id: str             # 70B / 13B / 7B
    params_billion: float    # 参数量(十亿)
    vram_requirement: float  # 显存需求(GB)
    tokens_per_sec: float    # 生成速率
    quality_score: float     # 质量评分(0-1)
    cost_per_token: float    # 每token成本


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    prompt: str
    seq_length: int = 512
    priority: int = 1         # 1=普通, 2=高, 3=紧急
    prefix_hash: str = ""     # prompt前缀哈希(用于batch共享)
    assigned_tier: str = ""   # 分配的模型层级
    draft_tokens: List[str] = field(default_factory=list)  # 投机解码draft


class MultiModelCollaborator:
    """参赛者群体动力学 (算法5)

    推理映射: 多模型协同推理

    核心能力:
    - 投机解码: 小模型快速生成draft, 大模型验证
    - 动态切换: OOM风险升高时大模型自动切小模型
    - batch内共享prefix: 同batch请求自动编入共享prefix组
    - 请求重排博弈: 对即将OOM的相邻实例主动让出KV cache

    涌现属性:
    - 自愈式降级: 破产概率触及阈值前自动切换策略
    - 请求编队: 同类prefix请求自动编入同一batch
    - 延迟期货: 某时刻延迟略超SLO但换来后续低延迟
    - 异构模型博弈: 70B/13B/7B三档实时决定每个请求用哪档
    """

    # 默认模型层级
    DEFAULT_TIERS = [
        ModelTier("70B", 70.0, 140.0, 50.0, 0.95, 0.014),
        ModelTier("13B", 13.0, 26.0, 120.0, 0.85, 0.003),
        ModelTier("7B", 7.0, 14.0, 200.0, 0.75, 0.001),
    ]

    def __init__(self, tiers: List[ModelTier] = None):
        self.tiers: Dict[str, ModelTier] = {}
        for tier in (tiers or self.DEFAULT_TIERS):
            self.tiers[tier.tier_id] = tier

        self.current_tier: str = "70B"    # 当前主力模型
        self.formation_history: List[Dict] = []  # 编队历史
        self.switch_history: List[Dict] = []     # 切换历史
        self.draft_accept_rate: float = 0.0      # 投机解码接受率

    def select_model(self, request: InferenceRequest,
                     oom_probability: float,
                     slo_urgency: float = 0.5) -> str:
        """异构模型博弈: 为请求选择最优模型层级

        Args:
            request: 推理请求
            oom_probability: OOM概率
            slo_urgency: SLO紧迫度 (0-1, 越高越急)
        Returns:
            选择的模型层级ID
        """
        scores = {}

        for tier_id, tier in self.tiers.items():
            # 质量分 (越高越好)
            quality = tier.quality_score

            # 速度分 (越快越好, 归一化)
            speed = min(tier.tokens_per_sec / 200.0, 1.0)

            # OOM风险分 (显存需求越小风险越低)
            oom_safety = 1.0 - min(tier.vram_requirement / 140.0, 1.0)
            oom_penalty = oom_probability * (1 - oom_safety) * 2

            # 成本分 (越便宜越好)
            cost_score = 1.0 - min(tier.cost_per_token / 0.014, 1.0)

            # SLO紧迫度: 越急越偏向快模型
            if slo_urgency > 0.7:
                weighted = speed * 0.5 + quality * 0.2 + oom_safety * 0.2 + cost_score * 0.1
            else:
                weighted = quality * 0.4 + speed * 0.2 + oom_safety * 0.2 + cost_score * 0.2

            # OOM高时强制偏向小模型
            if oom_probability > 0.3:
                weighted *= oom_safety

            scores[tier_id] = round(weighted - oom_penalty, 4)

        # 选择得分最高的
        best_tier = max(scores, key=scores.get)

        # 记录切换
        if best_tier != self.current_tier:
            self.switch_history.append({
                "from": self.current_tier,
                "to": best_tier,
                "reason": f"OOM={oom_probability:.2f}, SLO={slo_urgency:.2f}",
                "scores": scores,
                "timestamp": datetime.now().isoformat(),
            })
            self.current_tier = best_tier

        request.assigned_tier = best_tier
        return best_tier

    def speculative_decode(self, request: InferenceRequest,
                           draft_tier: str = "7B",
                           verify_tier: str = "70B") -> Dict:
        """投机解码: 小模型生成draft, 大模型验证

        推理映射: 跟跑 → 投机解码
        跟跑双重验证: 不光看速度匹配, 还检查小模型内存熵状态

        Args:
            draft_tier: 生成draft的小模型
            verify_tier: 验证的大模型
        Returns:
            投机解码结果
        """
        draft_model = self.tiers.get(draft_tier)
        verify_model = self.tiers.get(verify_tier)

        if not draft_model or not verify_model:
            return {"success": False, "error": "模型层级不存在"}

        # 模拟draft token生成 (小模型快)
        draft_count = random.randint(4, 12)
        draft_tokens = [f"tok_{i}" for i in range(draft_count)]

        # 接受率: 与质量差异相关
        quality_gap = verify_model.quality_score - draft_model.quality_score
        accept_rate = max(0.5, 1.0 - quality_gap * 2 + random.uniform(-0.1, 0.1))
        accept_rate = min(0.95, accept_rate)
        accepted = int(draft_count * accept_rate)

        # 节省时间: draft模型生成快, 大模型只需验证
        time_draft = draft_count / draft_model.tokens_per_sec
        time_verify = (draft_count - accepted) / verify_model.tokens_per_sec + 0.01
        time_naive = draft_count / verify_model.tokens_per_sec
        speedup = time_naive / max(time_draft + time_verify, 0.001)

        # 跟跑双重验证: 检查小模型状态是否健康
        draft_healthy = random.random() > 0.1  # 90%概率健康
        if not draft_healthy:
            accept_rate *= 0.5
            accepted = int(draft_count * accept_rate)

        self.draft_accept_rate = accept_rate

        result = {
            "success": True,
            "draft_tier": draft_tier,
            "verify_tier": verify_tier,
            "draft_count": draft_count,
            "accepted": accepted,
            "rejected": draft_count - accepted,
            "accept_rate": round(accept_rate, 4),
            "speedup": round(speedup, 2),
            "draft_healthy": draft_healthy,
            "time_draft": round(time_draft, 6),
            "time_verify": round(time_verify, 6),
            "time_naive": round(time_naive, 6),
        }

        return result

    def batch_optimize(self, requests: List[InferenceRequest]) -> Dict:
        """batch内共享prefix优化

        推理映射: 编队能量银行 → batch共享prefix缓存
        同prefix请求编入同一batch, 节省KV cache

        Returns:
            编队优化结果
        """
        if not requests:
            return {"success": False, "error": "无请求"}

        # 按prefix_hash分组
        prefix_groups: Dict[str, List[InferenceRequest]] = {}
        for req in requests:
            key = req.prefix_hash or "no_prefix"
            if key not in prefix_groups:
                prefix_groups[key] = []
            prefix_groups[key].append(req)

        # 编入batch: 同prefix的请求尽量放一起
        batches = []
        for prefix, group in prefix_groups.items():
            batch_size = min(len(group), 8)  # 单batch最多8个
            for i in range(0, len(group), batch_size):
                batch = group[i:i + batch_size]
                batches.append({
                    "batch_id": f"batch_{len(batches)}",
                    "prefix_hash": prefix,
                    "request_count": len(batch),
                    "request_ids": [r.request_id for r in batch],
                    "shared_prefix": prefix != "no_prefix",
                    "kv_cache_saved": len(batch) * 0.8 if prefix != "no_prefix" else 0,  # GB
                })

        total_kv_saved = sum(b["kv_cache_saved"] for b in batches)
        total_requests = sum(b["request_count"] for b in batches)
        shared_ratio = sum(1 for b in batches if b["shared_prefix"]) / max(len(batches), 1)

        result = {
            "success": True,
            "total_requests": total_requests,
            "total_batches": len(batches),
            "prefix_groups": len(prefix_groups),
            "shared_prefix_ratio": round(shared_ratio, 4),
            "total_kv_cache_saved_gb": round(total_kv_saved, 2),
            "batches": batches,
        }

        self.formation_history.append({
            **{k: v for k, v in result.items() if k != "batches"},
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.formation_history) > 200:
            self.formation_history = self.formation_history[-200:]

        return result

    def request_reorder(self, requests: List[InferenceRequest],
                         oom_risk: float) -> Dict:
        """请求重排博弈

        推理映射: 领跑分裂 → 主动调度/请求重排
        对即将OOM的相邻实例主动让出KV cache空间换取对方更快完成

        Args:
            oom_risk: 当前OOM风险 (0-1)
        Returns:
            重排结果
        """
        if not requests:
            return {"success": False, "error": "无请求"}

        # 按优先级和序列长度重排
        # OOM风险高时: 短请求优先 (快速释放资源)
        # OOM风险低时: 高优先级优先
        if oom_risk > 0.3:
            # 危急模式: 短请求优先 + 高优先级
            reordered = sorted(requests,
                               key=lambda r: (r.seq_length, -r.priority))
            strategy = "short_first_oom_defensive"
        else:
            # 正常模式: 按优先级 + prefix分组
            reordered = sorted(requests,
                               key=lambda r: (-r.priority, r.prefix_hash))
            strategy = "priority_prefix_grouped"

        result = {
            "success": True,
            "strategy": strategy,
            "oom_risk": round(oom_risk, 4),
            "original_order": [r.request_id for r in requests],
            "reordered": [r.request_id for r in reordered],
            "description": "OOM防御: 短请求优先快速释放资源" if oom_risk > 0.3 else "正常: 优先级+prefix分组",
        }

        return result

    def get_status(self) -> Dict:
        return {
            "current_tier": self.current_tier,
            "available_tiers": list(self.tiers.keys()),
            "tier_details": {
                tid: {
                    "params_billion": t.params_billion,
                    "vram_gb": t.vram_requirement,
                    "tokens_per_sec": t.tokens_per_sec,
                    "quality": t.quality_score,
                } for tid, t in self.tiers.items()
            },
            "switches": len(self.switch_history),
            "formations": len(self.formation_history),
            "draft_accept_rate": round(self.draft_accept_rate, 4),
            "last_switch": self.switch_history[-1] if self.switch_history else None,
        }


# ============================================================
# 第二级: 双层SLO优化器 (第一级 + 算法1 + 算法4)
# ============================================================

class DualLayerSLOOptimizer:
    """双层SLO优化器

    架构:
    - 外层 (变分法): 给定总QPS目标, 解出最优延迟分布曲线 (含延迟贴现因子)
    - 内层 (后悔最小化): per-token调节生成速率, 追回SLO偏差
    - 硬约束: OOM概率 < 30% 且 内存熵三联未全触发

    最终公式:
    min |p_actual(k) - p*(k)|² + λ₁·E_save(g_{k+1}) + λ₂·C_lead(a_{k+1})
    s.t. P_破产 < 0.3, Φ_entropy < 2
    """

    # 约束阈值
    OOM_THRESHOLD = 0.3        # 破产概率阈值
    ENTROPY_TRIGGER_LIMIT = 2  # 三联触发数上限

    # 目标函数权重
    LAMBDA_1 = 0.3   # 能量节省权重
    LAMBDA_2 = 0.2   # 领跑消耗惩罚权重

    def __init__(self):
        self.oom_predictor = OOMBancruptcyPredictor()
        self.outer = PainCurveOptimizer(discount_factor=0.95)
        self.inner = RegretMinimizer(learning_rate=0.1)
        self.optimization_history: List[Dict] = []
        self.current_step: int = 0

    def initialize(self, total_slo_budget: float, time_steps: int = 60):
        """初始化外层优化 (赛前/部署时一次)"""
        self.outer.solve_optimal_curve(total_slo_budget, time_steps)
        self.current_step = 0
        print(f"[双层SLO] 外层优化完成 | 总预算: {total_slo_budget} | 步数: {time_steps}")

    def optimize(self, remaining_resource: float, consumption_rate: float,
                 actual_rate: float, kv_hit_rate: float = 0.8,
                 seq_length: int = 512, batch_size: int = 1,
                 fragmentation: float = 0.05, swap_ratio: float = 1.0,
                 vram_decline: float = 0.0) -> Dict:
        """执行一轮双层优化

        Returns:
            {
                "adjusted_rate": 调整后速率,
                "oom_prediction": OOM预测,
                "target_pain": 目标痛苦值,
                "constraints_active": 约束是否激活,
                "decision": 决策描述,
            }
        """
        # 1. 更新耦合状态 (算法2)
        self.oom_predictor.coupling.update(kv_hit_rate, seq_length, batch_size)

        # 2. OOM破产概率预测 (算法3, 含算法0修正)
        oom_result = self.oom_predictor.predict(
            remaining_resource=remaining_resource,
            consumption_rate=consumption_rate,
            time_horizon=10.0,
            fragmentation=fragmentation,
            swap_ratio=swap_ratio,
            vram_decline=vram_decline,
        )

        # 3. 获取目标痛苦值 (算法4外层)
        target_pain = self.outer.get_target_at(self.current_step)
        if target_pain == 0:
            target_pain = 0.1  # 默认目标

        # 4. 检查硬约束
        oom_prob = oom_result["bankruptcy_prob"]
        force_shrink = oom_result.get("force_shrink", False)
        entropy_triggered = oom_result["entropy_state"]["triggered_count"]

        constraints_active = (oom_prob >= self.OOM_THRESHOLD or
                              entropy_triggered >= self.ENTROPY_TRIGGER_LIMIT or
                              force_shrink)

        # 5. 内层优化 (算法1后悔最小化)
        adjusted_rate = self.inner.update(
            actual_rate=actual_rate,
            target_rate=1.0 / target_pain if target_pain > 0 else 1.0,
            oom_probability=oom_prob,
            force_shrink=force_shrink,
        )

        # 6. 生成决策描述
        if constraints_active:
            if oom_prob >= self.OOM_THRESHOLD:
                decision = f"⚠ OOM概率{oom_prob:.0%}超阈值, 强制降速至{adjusted_rate:.2f}x"
            elif force_shrink:
                decision = f"⚠ 熵增三联触发(force_shrink), 收缩可行域, 速率{adjusted_rate:.2f}x"
            else:
                decision = f"⚠ 三联触发{entropy_triggered}项, 收缩可行域"
        else:
            decision = f"✓ 正常优化 | 目标痛苦: {target_pain:.4f} | 速率: {adjusted_rate:.2f}x"

        self.current_step += 1

        result = {
            "step": self.current_step,
            "adjusted_rate": round(adjusted_rate, 4),
            "target_pain": round(target_pain, 6),
            "oom_prediction": oom_result,
            "constraints_active": constraints_active,
            "force_shrink": force_shrink,
            "decision": decision,
            "inner_stats": self.inner.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }

        self.optimization_history.append(result)
        if len(self.optimization_history) > 500:
            self.optimization_history = self.optimization_history[-500:]

        return result

    def get_status(self) -> Dict:
        return {
            "current_step": self.current_step,
            "outer": self.outer.get_status(),
            "inner": self.inner.get_stats(),
            "oom_latest": self.oom_predictor.get_latest(),
            "total_optimizations": len(self.optimization_history),
            "constraints_active_count": sum(
                1 for h in self.optimization_history if h["constraints_active"]
            ),
        }


# ============================================================
# 第三级: 六合一融合决策引擎
# ============================================================

class FusionDecisionEngine:
    """六层级递进融合决策引擎 (总引擎)

    融合全部6个算法:
    - 算法0: 熵增三联监测 → 提前修正漂移项
    - 算法1: 后悔最小化 → 内层实时纠偏
    - 算法2: KV-序列耦合 → 资源消耗速率
    - 算法3: OOM破产概率 → 硬约束
    - 算法4: 最优痛苦分布 → 外层全局最优
    - 算法5: 多模型协同 → 投机解码/编队/博弈

    决策变量: 配速v_{k+1} + 跟跑目标g_{k+1} + 主动行为a_{k+1}
    映射: 生成速率 + 模型层级选择 + 调度策略

    最终公式:
    min |p_actual(k) - p*(k)|² + λ₁·E_save(g_{k+1}|HR,θ) + λ₂·C_lead(a_{k+1})
    s.t. P_破产 < 0.3, Φ_entropy < 2
    """

    def __init__(self):
        self.slo_optimizer = DualLayerSLOOptimizer()
        self.multi_model = MultiModelCollaborator()
        self.pain_negotiator = PainCurveOptimizer(discount_factor=0.95)
        self.decision_history: List[Dict] = []
        self.initialized: bool = False

        # 系统状态缓存
        self.last_decision: Dict = {}

    def initialize(self, total_slo_budget: float = 5.0, time_steps: int = 60):
        """初始化引擎 (部署时一次)"""
        self.slo_optimizer.initialize(total_slo_budget, time_steps)
        self.initialized = True
        print(f"[融合引擎] 初始化完成 | SLO预算: {total_slo_budget}")

    def decide(self, system_state: Dict) -> Dict:
        """执行完整六层融合决策

        Args:
            system_state: {
                "remaining_vram": 剩余显存(GB),
                "consumption_rate": 消耗速率,
                "actual_gen_rate": 实际生成速率,
                "kv_hit_rate": KV命中率,
                "seq_length": 序列长度,
                "batch_size": batch大小,
                "fragmentation": 碎片率,
                "swap_ratio": swap频率比,
                "vram_decline": 显存下降速率,
                "requests": 推理请求列表,
                "slo_urgency": SLO紧迫度,
            }
        Returns:
            完整决策结果
        """
        if not self.initialized:
            self.initialize()

        # === 第一级 + 第二级: 双层SLO优化 ===
        slo_result = self.slo_optimizer.optimize(
            remaining_resource=system_state.get("remaining_vram", 40.0),
            consumption_rate=system_state.get("consumption_rate", 1.0),
            actual_rate=system_state.get("actual_gen_rate", 1.0),
            kv_hit_rate=system_state.get("kv_hit_rate", 0.8),
            seq_length=system_state.get("seq_length", 512),
            batch_size=system_state.get("batch_size", 1),
            fragmentation=system_state.get("fragmentation", 0.05),
            swap_ratio=system_state.get("swap_ratio", 1.0),
            vram_decline=system_state.get("vram_decline", 0.0),
        )

        oom_prob = slo_result["oom_prediction"]["bankruptcy_prob"]
        force_shrink = slo_result["force_shrink"]

        # === 第三级: 多模型协同 ===
        requests = system_state.get("requests", [])
        slo_urgency = system_state.get("slo_urgency", 0.5)

        # 异构模型博弈: 选择模型层级
        model_choice = "70B"
        if requests:
            # 为第一个请求选择模型 (简化)
            model_choice = self.multi_model.select_model(
                requests[0] if isinstance(requests[0], InferenceRequest)
                else InferenceRequest(request_id="auto", prompt=""),
                oom_probability=oom_prob,
                slo_urgency=slo_urgency,
            )

        # 投机解码决策
        should_spec_decode = oom_prob < 0.3 and slo_urgency > 0.3
        spec_result = None
        if should_spec_decode:
            spec_result = self.multi_model.speculative_decode(
                InferenceRequest(request_id="spec", prompt=""),
                draft_tier="7B",
                verify_tier=model_choice,
            )

        # batch优化
        batch_result = None
        if requests:
            req_objects = []
            for r in requests:
                if isinstance(r, InferenceRequest):
                    req_objects.append(r)
                elif isinstance(r, dict):
                    req_objects.append(InferenceRequest(
                        request_id=r.get("request_id", f"req_{len(req_objects)}"),
                        prompt=r.get("prompt", ""),
                        seq_length=r.get("seq_length", 512),
                        priority=r.get("priority", 1),
                        prefix_hash=r.get("prefix_hash", ""),
                    ))
            batch_result = self.multi_model.batch_optimize(req_objects)

            # 请求重排博弈
            reorder_result = self.multi_model.request_reorder(req_objects, oom_prob)
        else:
            reorder_result = None

        # === 痛苦期货协商 ===
        # 如果需要切模型或有额外操作成本, 评估是否值得
        pain_negotiation = None
        if force_shrink or oom_prob > 0.2:
            action_cost = 0.3 if force_shrink else 0.15
            pain_negotiation = self.pain_negotiator.negotiate_pain_futures(
                current_pain=slo_result["target_pain"],
                target_pain=slo_result["target_pain"],
                action_cost=action_cost,
            )

        # === 生成决策摘要 ===
        actions = []
        if slo_result["constraints_active"]:
            actions.append("约束激活: 降速保护")
        if model_choice != "70B":
            actions.append(f"模型切换: → {model_choice}")
        if spec_result and spec_result["success"]:
            actions.append(f"投机解码: 接受率{spec_result['accept_rate']:.0%}")
        if batch_result and batch_result["total_kv_cache_saved_gb"] > 0:
            actions.append(f"prefix共享: 节省{batch_result['total_kv_cache_saved_gb']}GB KV cache")
        if pain_negotiation and pain_negotiation["should_execute"]:
            actions.append(f"痛苦期货: 净收益+{pain_negotiation['net_benefit']:.4f}")

        if not actions:
            actions.append("维持当前策略")

        decision = {
            "decision_id": f"fd_{int(time.time())}_{random.randint(1000, 9999)}",
            "timestamp": datetime.now().isoformat(),
            "slo_optimization": slo_result,
            "model_choice": model_choice,
            "speculative_decode": spec_result,
            "batch_optimization": batch_result,
            "request_reorder": reorder_result,
            "pain_negotiation": pain_negotiation,
            "oom_probability": oom_prob,
            "adjusted_rate": slo_result["adjusted_rate"],
            "actions": actions,
            "decision_summary": " | ".join(actions),
            "multi_model_status": self.multi_model.get_status(),
            "slo_status": self.slo_optimizer.get_status(),
        }

        self.last_decision = decision
        self.decision_history.append({
            "decision_id": decision["decision_id"],
            "timestamp": decision["timestamp"],
            "oom_probability": oom_prob,
            "adjusted_rate": slo_result["adjusted_rate"],
            "model_choice": model_choice,
            "actions": actions,
            "decision_summary": decision["decision_summary"],
        })
        if len(self.decision_history) > 500:
            self.decision_history = self.decision_history[-500:]

        return decision

    def get_dashboard(self) -> Dict:
        """获取引擎仪表盘"""
        return {
            "initialized": self.initialized,
            "total_decisions": len(self.decision_history),
            "last_decision": self.last_decision.get("decision_summary", "无") if self.last_decision else "无决策",
            "slo_optimizer": self.slo_optimizer.get_status(),
            "multi_model": self.multi_model.get_status(),
            "entropy_monitor": self.slo_optimizer.oom_predictor.entropy_monitor.get_status(),
            "coupling": self.slo_optimizer.oom_predictor.coupling.get_status(),
            "recent_decisions": self.decision_history[-10:],
        }

    def quick_assess(self, remaining_vram: float, consumption_rate: float,
                     fragmentation: float = 0.05, swap_ratio: float = 1.0,
                     vram_decline: float = 0.0) -> Dict:
        """快速评估 (简化接口, 用于闭环引擎调用)"""
        return self.decide({
            "remaining_vram": remaining_vram,
            "consumption_rate": consumption_rate,
            "actual_gen_rate": 1.0,
            "kv_hit_rate": 0.8,
            "seq_length": 512,
            "batch_size": 1,
            "fragmentation": fragmentation,
            "swap_ratio": swap_ratio,
            "vram_decline": vram_decline,
            "requests": [],
            "slo_urgency": 0.5,
        })
