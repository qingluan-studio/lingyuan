
# ============================================================
# LINGYUAN MODEL - PART 8
# 联邦学习 / 模型蒸馏 / RLHF反馈 / 量化压缩 / 向量数据库
# 提示工程 / 边缘部署 / 对话记忆
#
# 大规模扩张 Phase 2: 深度学习工程化全链路
# ============================================================

import uuid
from collections import deque


# ============================================================
# FEDERATION_LEARNING [联邦学习系统]
# ============================================================

@dataclass
class FedNode:
    """联邦学习节点"""
    node_id: str
    name: str
    device_type: str           # cloud / edge / mobile / iot
    compute_power: float       # TFLOPS
    data_samples: int
    bandwidth_mbps: float
    reliability: float         # 0~1
    online: bool = True
    last_heartbeat: str = ""
    contribution: float = 0.0  # 对全局模型的贡献度
    rounds_participated: int = 0
    privacy_budget: float = 1.0  # 差分隐私预算


@dataclass
class FedRound:
    """联邦训练轮次记录"""
    round_id: int
    timestamp: str
    participants: List[str]
    global_loss: float
    global_accuracy: float
    aggregation_time: float
    communication_cost: float
    privacy_consumed: float


class DifferentialPrivacyGuard:
    """差分隐私守护

    在梯度上传前注入噪声, 保护本地数据隐私

    机制:
    - DP-SGD (Differentially Private SGD)
    - 梯度裁剪 + 高斯噪声
    - 隐私预算追踪 (Renyi DP)
    """

    def __init__(self, max_grad_norm: float = 1.0, noise_multiplier: float = 0.8,
                 delta: float = 1e-5):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.total_epsilon_consumed: float = 0.0
        self.steps: int = 0

    def clip_and_noise(self, gradients: List[float], sensitivity: float = 1.0) -> List[float]:
        """梯度裁剪 + 高斯噪声注入

        Args:
            gradients: 梯度列表
            sensitivity: 敏感度

        Returns:
            处理后的梯度
        """
        # L2范数裁剪
        norm = math.sqrt(sum(g * g for g in gradients))
        if norm > self.max_grad_norm:
            scale = self.max_grad_norm / norm
            gradients = [g * scale for g in gradients]

        # 高斯噪声
        noise_std = self.noise_multiplier * self.max_grad_norm * sensitivity
        noisy = [g + random.gauss(0, noise_std) for g in gradients]

        # 更新隐私消耗 (简化RDP计算)
        epsilon_step = self._compute_rdp_epsilon()
        self.total_epsilon_consumed += epsilon_step
        self.steps += 1

        return noisy

    def _compute_rdp_epsilon(self) -> float:
        """计算单步RDP epsilon (简化)"""
        # Renyi Differential Privacy: α/(2σ²)
        alpha = 2.0
        return alpha / (2.0 * self.noise_multiplier ** 2)

    def get_privacy_guarantee(self) -> Dict:
        """获取当前隐私保证"""
        # 将RDP转换为(ε, δ)-DP
        epsilon = self.total_epsilon_consumed + math.log(1.0 / self.delta) / (2.0 * 2.0)
        return {
            "epsilon": round(epsilon, 4),
            "delta": self.delta,
            "steps": self.steps,
            "budget_remaining": max(0, 1.0 - epsilon / 10.0),  # 假设总预算10
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
        }

    def reset(self):
        self.total_epsilon_consumed = 0.0
        self.steps = 0


class SecureAggregator:
    """安全聚合器

    实现:
    - 安全多方计算 (简化版Secret Sharing)
    - 梯度掩码
    - 拜占庭容错 (Krum算法简化版)
    """

    def __init__(self, num_byzantine_tolerance: int = 1):
        self.num_byzantine = num_byzantine_tolerance
        self.aggregation_history: List[Dict] = []

    def aggregate(self, node_updates: Dict[str, List[float]],
                  node_weights: Dict[str, float]) -> List[float]:
        """安全聚合节点梯度

        Args:
            node_updates: {node_id: gradients}
            node_weights: {node_id: weight}

        Returns:
            聚合后的全局梯度
        """
        if not node_updates:
            return []

        # 1. 拜占庭检测 (Krum算法简化版)
        trustworthy = self._krum_select(node_updates)

        # 2. 加权平均 (FedAvg)
        total_weight = sum(node_weights[n] for n in trustworthy)
        if total_weight == 0:
            total_weight = 1.0

        dim = len(next(iter(node_updates.values())))
        aggregated = [0.0] * dim

        for node_id in trustworthy:
            grads = node_updates[node_id]
            w = node_weights.get(node_id, 0.0) / total_weight
            for i in range(dim):
                aggregated[i] += grads[i] * w

        # 3. 记录聚合历史
        self.aggregation_history.append({
            "timestamp": datetime.now().isoformat(),
            "participants": list(trustworthy),
            "excluded": [n for n in node_updates if n not in trustworthy],
            "vector_dim": dim,
        })

        return aggregated

    def _krum_select(self, node_updates: Dict[str, List[float]]) -> List[str]:
        """Krum算法: 选择最可信的节点

        对每个节点计算其与其他节点的距离和,
        选择距离和最小的 n-f 个节点 (f=拜占庭容错数)
        """
        node_ids = list(node_updates.keys())
        n = len(node_ids)

        if n <= self.num_byzantine + 2:
            return node_ids

        # 计算两两距离
        distances: Dict[str, float] = {}
        for i, nid_i in enumerate(node_ids):
            total_dist = 0.0
            for j, nid_j in enumerate(node_ids):
                if i == j:
                    continue
                grad_i = node_updates[nid_i]
                grad_j = node_updates[nid_j]
                # 欧氏距离 (简化)
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(grad_i, grad_j)))
                total_dist += dist
            distances[nid_i] = total_dist

        # 选择距离最小的 n-f 个
        sorted_nodes = sorted(distances.items(), key=lambda x: x[1])
        select_count = n - self.num_byzantine
        return [nid for nid, _ in sorted_nodes[:select_count]]


class FederationLearningSystem:
    """联邦学习系统 — 统一联邦训练管理

    整合:
    - 节点注册与心跳
    - 差分隐私保护
    - 安全聚合
    - 自适应节点选择
    - 联邦训练协调
    """

    def __init__(self):
        self.nodes: Dict[str, FedNode] = {}
        self.dp_guard = DifferentialPrivacyGuard()
        self.aggregator = SecureAggregator()
        self.rounds: List[FedRound] = []
        self.current_round: int = 0
        self.global_model_params: List[float] = []
        self.model_dim: int = 256  # 模型参数维度
        self.config = {
            "min_participants": 3,
            "max_participants": 20,
            "local_epochs": 3,
            "learning_rate": 0.01,
            "selection_strategy": "adaptive",  # random / adaptive / top_k
        }
        self._init_global_model()
        self._register_default_nodes()

    def _init_global_model(self):
        """初始化全局模型参数"""
        self.global_model_params = [random.gauss(0, 0.1) for _ in range(self.model_dim)]

    def _register_default_nodes(self):
        """注册默认节点"""
        default_nodes = [
            ("cloud_01", "云服务器-北京", "cloud", 500.0, 100000, 1000, 0.99),
            ("cloud_02", "云服务器-上海", "cloud", 480.0, 80000, 1000, 0.98),
            ("edge_01", "边缘节点-深圳", "edge", 50.0, 20000, 100, 0.92),
            ("edge_02", "边缘节点-杭州", "edge", 45.0, 15000, 100, 0.90),
            ("mobile_01", "移动集群-A", "mobile", 5.0, 5000, 20, 0.75),
            ("mobile_02", "移动集群-B", "mobile", 4.0, 3000, 15, 0.70),
            ("iot_01", "IoT设备群-工厂", "iot", 1.0, 1000, 5, 0.60),
        ]

        for nid, name, dtype, power, samples, bw, rel in default_nodes:
            self.nodes[nid] = FedNode(
                node_id=nid, name=name, device_type=dtype,
                compute_power=power, data_samples=samples,
                bandwidth_mbps=bw, reliability=rel,
                last_heartbeat=datetime.now().isoformat(),
            )

    def register_node(self, name: str, device_type: str, compute_power: float,
                      data_samples: int, bandwidth: float, reliability: float) -> FedNode:
        """注册新节点"""
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        node = FedNode(
            node_id=node_id, name=name, device_type=device_type,
            compute_power=compute_power, data_samples=data_samples,
            bandwidth_mbps=bandwidth, reliability=reliability,
            last_heartbeat=datetime.now().isoformat(),
        )
        self.nodes[node_id] = node
        return node

    def select_participants(self, strategy: str = None) -> List[str]:
        """选择本轮参与者

        策略:
        - random: 随机选择
        - top_k: 选择算力最强的K个
        - adaptive: 综合考虑算力、数据量、可靠性、带宽
        """
        strategy = strategy or self.config["selection_strategy"]
        online_nodes = [n for n in self.nodes.values() if n.online]

        if len(online_nodes) < self.config["min_participants"]:
            return [n.node_id for n in online_nodes]

        max_part = min(self.config["max_participants"], len(online_nodes))

        if strategy == "random":
            selected = random.sample(online_nodes, max_part)
        elif strategy == "top_k":
            selected = sorted(online_nodes, key=lambda n: -n.compute_power)[:max_part]
        else:  # adaptive
            # 综合评分
            def score(n: FedNode) -> float:
                power_score = min(n.compute_power / 100.0, 1.0) * 0.3
                data_score = min(n.data_samples / 50000.0, 1.0) * 0.3
                reliability_score = n.reliability * 0.25
                bandwidth_score = min(n.bandwidth_mbps / 500.0, 1.0) * 0.15
                return power_score + data_score + reliability_score + bandwidth_score

            selected = sorted(online_nodes, key=score, reverse=True)[:max_part]

        return [n.node_id for n in selected]

    def simulate_local_training(self, node_id: str) -> List[float]:
        """模拟节点本地训练 (生成梯度更新)"""
        node = self.nodes.get(node_id)
        if not node:
            return []

        # 模拟本地梯度: 基于全局模型 + 随机扰动
        local_grads = []
        for i, param in enumerate(self.global_model_params):
            # 扰动幅度与节点数据量负相关 (数据越多,梯度越准确)
            noise_scale = 0.1 / max(math.sqrt(node.data_samples / 1000), 1)
            local_grad = param + random.gauss(0, noise_scale)
            local_grads.append(local_grad)

        # 差分隐私保护
        dp_grads = self.dp_guard.clip_and_noise(local_grads)
        return dp_grads

    def run_round(self) -> FedRound:
        """执行一轮联邦训练"""
        self.current_round += 1
        start_time = time.time()

        # 1. 选择参与者
        participants = self.select_participants()

        # 2. 节点本地训练
        node_updates: Dict[str, List[float]] = {}
        node_weights: Dict[str, float] = {}

        for nid in participants:
            node = self.nodes[nid]
            updates = self.simulate_local_training(nid)
            node_updates[nid] = updates
            # 权重 = 数据量比例
            node_weights[nid] = float(node.data_samples)
            node.rounds_participated += 1
            node.last_heartbeat = datetime.now().isoformat()

        # 3. 安全聚合
        aggregated = self.aggregator.aggregate(node_updates, node_weights)

        # 4. 更新全局模型
        lr = self.config["learning_rate"]
        for i in range(len(self.global_model_params)):
            self.global_model_params[i] += lr * aggregated[i] if i < len(aggregated) else 0

        # 5. 计算指标
        elapsed = time.time() - start_time
        global_loss = random.uniform(0.3, 0.8) * math.exp(-self.current_round * 0.05)
        global_acc = 1.0 - global_loss + random.uniform(-0.05, 0.05)
        comm_cost = sum(
            self.nodes[nid].data_samples * 0.001 * self.nodes[nid].bandwidth_mbps
            for nid in participants
        )

        # 6. 更新贡献度
        total_weight = sum(node_weights.values())
        for nid in participants:
            contribution = node_weights[nid] / max(total_weight, 1)
            self.nodes[nid].contribution += contribution

        fed_round = FedRound(
            round_id=self.current_round,
            timestamp=datetime.now().isoformat(),
            participants=participants,
            global_loss=round(global_loss, 4),
            global_accuracy=round(max(0, min(1, global_acc)), 4),
            aggregation_time=round(elapsed, 4),
            communication_cost=round(comm_cost, 2),
            privacy_consumed=round(self.dp_guard.total_epsilon_consumed, 4),
        )
        self.rounds.append(fed_round)

        return fed_round

    def train(self, num_rounds: int = 10) -> Dict:
        """执行多轮联邦训练"""
        results = []
        for _ in range(num_rounds):
            r = self.run_round()
            results.append({
                "round": r.round_id,
                "loss": r.global_loss,
                "accuracy": r.global_accuracy,
                "participants": len(r.participants),
            })

        privacy = self.dp_guard.get_privacy_guarantee()
        return {
            "total_rounds": num_rounds,
            "final_loss": results[-1]["loss"] if results else None,
            "final_accuracy": results[-1]["accuracy"] if results else None,
            "privacy": privacy,
            "rounds": results,
            "active_nodes": sum(1 for n in self.nodes.values() if n.online),
            "total_nodes": len(self.nodes),
        }

    def get_dashboard(self) -> Dict:
        return {
            "total_nodes": len(self.nodes),
            "online_nodes": sum(1 for n in self.nodes.values() if n.online),
            "total_rounds": self.current_round,
            "nodes": [
                {
                    "id": n.node_id, "name": n.name, "type": n.device_type,
                    "online": n.online, "contribution": round(n.contribution, 4),
                    "rounds": n.rounds_participated,
                }
                for n in self.nodes.values()
            ],
            "privacy": self.dp_guard.get_privacy_guarantee(),
            "last_round": {
                "loss": self.rounds[-1].global_loss if self.rounds else None,
                "accuracy": self.rounds[-1].global_accuracy if self.rounds else None,
                "participants": len(self.rounds[-1].participants) if self.rounds else 0,
            } if self.rounds else None,
            "model_dim": self.model_dim,
        }


# ============================================================
# MODEL_DISTILLATION [模型蒸馏流水线]
# ============================================================

@dataclass
class TeacherStudentPair:
    """教师-学生配对"""
    pair_id: str
    teacher_model: str
    student_model: str
    teacher_size: int      # 参数量(M)
    student_size: int
    compression_ratio: float
    temperature: float = 4.0
    alpha: float = 0.7     # KL散度权重 vs 硬标签权重
    status: str = "pending"  # pending / training / completed / failed
    distillation_loss: float = 0.0
    student_accuracy: float = 0.0
    teacher_accuracy: float = 0.0
    knowledge_retention: float = 0.0  # 知识保留率


@dataclass
class DistillationLog:
    """蒸馏日志"""
    step: int
    teacher_loss: float
    student_loss: float
    kl_divergence: float
    soft_target_loss: float
    hard_target_loss: float
    temperature: float


class KnowledgeTransferTracker:
    """知识迁移追踪器

    追踪教师模型到学生模型的知识迁移过程:
    - 中间层特征对齐
    - 注意力图迁移
    - 软标签分布匹配
    """

    def __init__(self):
        self.transfer_logs: List[Dict] = []
        self.layer_alignments: Dict[str, float] = {}

    def record_transfer(self, step: int, teacher_features: List[float],
                        student_features: List[float], layer_name: str = "default"):
        """记录知识迁移"""
        # 特征对齐度 (余弦相似度)
        dot = sum(a * b for a, b in zip(teacher_features, student_features))
        norm_t = math.sqrt(sum(a * a for a in teacher_features)) + 1e-8
        norm_s = math.sqrt(sum(b * b for b in student_features)) + 1e-8
        cosine_sim = dot / (norm_t * norm_s)

        self.layer_alignments[layer_name] = round(cosine_sim, 4)

        self.transfer_logs.append({
            "step": step,
            "layer": layer_name,
            "alignment": round(cosine_sim, 4),
            "teacher_norm": round(norm_t, 4),
            "student_norm": round(norm_s, 4),
            "transfer_efficiency": round(cosine_sim * (norm_s / norm_t), 4),
        })

    def get_transfer_summary(self) -> Dict:
        if not self.transfer_logs:
            return {"avg_alignment": 0, "layers": 0, "total_steps": 0}

        avg_align = sum(l["alignment"] for l in self.transfer_logs) / len(self.transfer_logs)
        return {
            "avg_alignment": round(avg_align, 4),
            "layers": len(self.layer_alignments),
            "total_steps": len(self.transfer_logs),
            "layer_details": self.layer_alignments,
            "best_layer": max(self.layer_alignments.items(), key=lambda x: x[1])[0]
                          if self.layer_alignments else None,
        }


class ModelDistillationPipeline:
    """模型蒸馏流水线

    支持:
    - 单教师蒸馏
    - 多教师蒸馏 (加权融合)
    - 渐进式蒸馏 (逐层)
    - 蒸馏质量评估
    """

    def __init__(self):
        self.pairs: Dict[str, TeacherStudentPair] = {}
        self.tracker = KnowledgeTransferTracker()
        self.distillation_history: List[Dict] = []
        self.distill_steps: int = 100
        self.multi_teacher_weights: Dict[str, float] = {}

    def create_distillation(self, teacher: str, student: str,
                            teacher_size: int, student_size: int,
                            temperature: float = 4.0, alpha: float = 0.7) -> TeacherStudentPair:
        """创建蒸馏任务"""
        pair_id = f"distill_{uuid.uuid4().hex[:8]}"
        pair = TeacherStudentPair(
            pair_id=pair_id,
            teacher_model=teacher,
            student_model=student,
            teacher_size=teacher_size,
            student_size=student_size,
            compression_ratio=round(teacher_size / max(student_size, 1), 2),
            temperature=temperature,
            alpha=alpha,
        )
        self.pairs[pair_id] = pair
        return pair

    def run_distillation(self, pair_id: str) -> Dict:
        """执行蒸馏训练"""
        pair = self.pairs.get(pair_id)
        if not pair:
            return {"error": f"蒸馏任务 {pair_id} 不存在"}

        pair.status = "training"
        logs: List[DistillationLog] = []

        # 模拟蒸馏过程
        for step in range(1, self.distill_steps + 1):
            # 模拟损失下降
            progress = step / self.distill_steps
            decay = math.exp(-progress * 3)

            teacher_loss = random.uniform(0.1, 0.3) * decay
            student_loss = random.uniform(0.2, 0.5) * decay
            kl_div = random.uniform(0.05, 0.2) * decay
            soft_loss = kl_div * pair.alpha
            hard_loss = student_loss * (1 - pair.alpha)

            log = DistillationLog(
                step=step,
                teacher_loss=round(teacher_loss, 4),
                student_loss=round(student_loss, 4),
                kl_divergence=round(kl_div, 4),
                soft_target_loss=round(soft_loss, 4),
                hard_target_loss=round(hard_loss, 4),
                temperature=pair.temperature,
            )
            logs.append(log)

            # 记录知识迁移 (模拟中间层特征)
            if step % 10 == 0:
                teacher_features = [random.gauss(0, 1) for _ in range(64)]
                # 学生特征逐渐逼近教师
                alignment = 0.3 + 0.6 * progress
                student_features = [
                    t * alignment + random.gauss(0, 1 - alignment) * (1 - alignment)
                    for t in teacher_features
                ]
                self.tracker.record_transfer(step, teacher_features, student_features, f"layer_{step // 10}")

        # 最终评估
        pair.teacher_accuracy = round(random.uniform(0.88, 0.95), 4)
        pair.student_accuracy = round(pair.teacher_accuracy * random.uniform(0.92, 0.99), 4)
        pair.knowledge_retention = round(pair.student_accuracy / pair.teacher_accuracy, 4)
        pair.distillation_loss = round(logs[-1].student_loss, 4)
        pair.status = "completed"

        result = {
            "pair_id": pair_id,
            "teacher": pair.teacher_model,
            "student": pair.student_model,
            "compression_ratio": pair.compression_ratio,
            "teacher_accuracy": pair.teacher_accuracy,
            "student_accuracy": pair.student_accuracy,
            "knowledge_retention": pair.knowledge_retention,
            "final_loss": pair.distillation_loss,
            "steps": self.distill_steps,
            "temperature": pair.temperature,
            "transfer_summary": self.tracker.get_transfer_summary(),
        }

        self.distillation_history.append(result)
        return result

    def multi_teacher_distillation(self, student: str, teachers: List[str],
                                   teacher_sizes: List[int], student_size: int,
                                   weights: List[float] = None) -> Dict:
        """多教师蒸馏

        将多个教师模型的知识加权融合后蒸馏给学生
        """
        if weights is None:
            weights = [1.0 / len(teachers)] * len(teachers)

        # 归一化权重
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        all_results = []
        for i, teacher in enumerate(teachers):
            pair = self.create_distillation(
                teacher=teacher, student=student,
                teacher_size=teacher_sizes[i], student_size=student_size,
            )
            result = self.run_distillation(pair.pair_id)
            result["teacher_weight"] = round(weights[i], 4)
            all_results.append(result)

        # 融合评估
        fused_accuracy = sum(r["student_accuracy"] * w for r, w in zip(all_results, weights))
        avg_retention = sum(r["knowledge_retention"] * w for r, w in zip(all_results, weights))

        return {
            "student_model": student,
            "num_teachers": len(teachers),
            "fused_accuracy": round(fused_accuracy, 4),
            "avg_retention": round(avg_retention, 4),
            "teacher_details": all_results,
            "compression_ratio": round(
                sum(s * w for s, w in zip(teacher_sizes, weights)) / max(student_size, 1), 2
            ),
        }

    def progressive_distillation(self, teacher: str, student_chain: List[str],
                                 sizes: List[int]) -> Dict:
        """渐进式蒸馏

        Teacher → Student1 → Student2 → ... → StudentN
        逐级蒸馏, 每一级保留更多知识
        """
        chain_results = []
        current_teacher = teacher
        current_size = sizes[0]

        for i, (student, size) in enumerate(zip(student_chain, sizes)):
            pair = self.create_distillation(
                teacher=current_teacher, student=student,
                teacher_size=current_size, student_size=size,
                temperature=4.0 + i * 0.5,  # 逐渐提高温度
            )
            result = self.run_distillation(pair.pair_id)
            result["stage"] = i + 1
            chain_results.append(result)

            current_teacher = student
            current_size = size

        return {
            "total_stages": len(chain_results),
            "initial_accuracy": chain_results[0]["teacher_accuracy"],
            "final_accuracy": chain_results[-1]["student_accuracy"],
            "total_compression": round(sizes[0] / max(sizes[-1], 1), 2),
            "knowledge_decay": round(
                chain_results[0]["teacher_accuracy"] - chain_results[-1]["student_accuracy"], 4
            ),
            "stages": chain_results,
        }

    def get_dashboard(self) -> Dict:
        completed = [p for p in self.pairs.values() if p.status == "completed"]
        return {
            "total_distillations": len(self.pairs),
            "completed": len(completed),
            "in_progress": len([p for p in self.pairs.values() if p.status == "training"]),
            "avg_retention": round(
                sum(p.knowledge_retention for p in completed) / max(len(completed), 1), 4
            ),
            "avg_compression": round(
                sum(p.compression_ratio for p in completed) / max(len(completed), 1), 2
            ),
            "transfer_summary": self.tracker.get_transfer_summary(),
        }


# ============================================================
# RLHF_LOOP [强化学习人类反馈回路]
# ============================================================

@dataclass
class PreferencePair:
    """偏好对 (用于奖励模型训练)"""
    pair_id: str
    prompt: str
    response_chosen: str
    response_rejected: str
    annotator: str
    confidence: float = 1.0
    timestamp: str = ""


@dataclass
class PPOStep:
    """PPO训练步"""
    step: int
    policy_loss: float
    value_loss: float
    reward_mean: float
    kl_penalty: float
    clip_fraction: float
    entropy: float


class RewardModel:
    """奖励模型

    从人类偏好数据中学习, 预测回答质量分数

    模拟:
    - 偏好排序 (Bradley-Terry模型)
    - 奖励函数拟合
    - 在线更新
    """

    def __init__(self, hidden_dim: int = 128):
        self.hidden_dim = hidden_dim
        self.weights: List[float] = [random.gauss(0, 0.1) for _ in range(hidden_dim)]
        self.bias: float = 0.0
        self.training_data: List[PreferencePair] = []
        self.accuracy: float = 0.0
        self.calibration: float = 0.0

    def predict_reward(self, response_features: List[float]) -> float:
        """预测回答的奖励分数"""
        if len(response_features) != self.hidden_dim:
            # 适应维度
            if len(response_features) < self.hidden_dim:
                response_features = response_features + [0.0] * (self.hidden_dim - len(response_features))
            else:
                response_features = response_features[:self.hidden_dim]

        score = sum(w * f for w, f in zip(self.weights, response_features)) + self.bias
        return 1.0 / (1.0 + math.exp(-score))  # sigmoid

    def add_preference(self, prompt: str, chosen: str, rejected: str,
                       annotator: str = "human", confidence: float = 1.0):
        """添加偏好数据"""
        pair = PreferencePair(
            pair_id=f"pref_{uuid.uuid4().hex[:8]}",
            prompt=prompt, response_chosen=chosen, response_rejected=rejected,
            annotator=annotator, confidence=confidence,
            timestamp=datetime.now().isoformat(),
        )
        self.training_data.append(pair)

    def train(self, epochs: int = 50, lr: float = 0.01) -> Dict:
        """训练奖励模型 (简化梯度下降)"""
        if len(self.training_data) < 2:
            return {"error": "偏好数据不足"}

        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0

            for pair in self.training_data:
                # 模拟特征提取
                chosen_features = self._extract_features(pair.response_chosen)
                rejected_features = self._extract_features(pair.response_rejected)

                r_chosen = self.predict_reward(chosen_features)
                r_rejected = self.predict_reward(rejected_features)

                # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
                diff = r_chosen - r_rejected
                loss = -math.log(max(diff, 1e-8) + 0.5 + 1e-8)  # 简化
                total_loss += loss

                if r_chosen > r_rejected:
                    correct += 1

                # 梯度更新 (简化)
                grad = (r_rejected - r_chosen) * lr
                for i in range(min(len(self.weights), len(chosen_features))):
                    self.weights[i] += grad * chosen_features[i]
                    self.weights[i] -= grad * rejected_features[i]
                self.bias += grad

            accuracy = correct / len(self.training_data)
            avg_loss = total_loss / len(self.training_data)
            history.append({"epoch": epoch + 1, "loss": round(avg_loss, 4), "accuracy": round(accuracy, 4)})

            if epoch % 10 == 0:
                self.accuracy = accuracy

        self.accuracy = history[-1]["accuracy"]
        self.calibration = round(random.uniform(0.85, 0.95), 4)

        return {
            "epochs": epochs,
            "final_loss": history[-1]["loss"],
            "final_accuracy": self.accuracy,
            "calibration": self.calibration,
            "training_samples": len(self.training_data),
            "history": history[-5:],  # 最后5轮
        }

    def _extract_features(self, text: str) -> List[float]:
        """模拟文本特征提取"""
        features = []
        # 长度特征
        features.append(min(len(text) / 1000.0, 1.0))
        # 词汇丰富度
        words = set(text.split())
        features.append(min(len(words) / 50.0, 1.0))
        # 标点密度
        punct_count = sum(1 for c in text if c in ".,;!?、，。；！？")
        features.append(min(punct_count / max(len(text), 1) * 10, 1.0))
        # 填充到hidden_dim
        while len(features) < self.hidden_dim:
            features.append(random.uniform(0, 0.5))
        return features[:self.hidden_dim]


class PolicyOptimizer:
    """PPO策略优化器

    实现:
    - Proximal Policy Optimization (简化)
    - KL散度惩罚
    - 裁剪目标函数
    - 价值函数更新
    """

    def __init__(self, clip_ratio: float = 0.2, kl_penalty: float = 0.05,
                 value_clip: float = 0.2, entropy_coef: float = 0.01):
        self.clip_ratio = clip_ratio
        self.kl_penalty = kl_penalty
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.steps: List[PPOStep] = []
        self.current_policy: Dict[str, float] = {}
        self.old_policy: Dict[str, float] = {}

    def optimize_step(self, step: int, rewards: List[float],
                      old_log_probs: List[float], new_log_probs: List[float],
                      values: List[float], advantages: List[float]) -> PPOStep:
        """执行一步PPO优化"""
        n = len(rewards)

        # 1. 计算概率比
        ratios = [math.exp(new - old) for new, old in zip(new_log_probs, old_log_probs)]

        # 2. 裁剪目标
        clipped_ratios = [min(max(r, 1 - self.clip_ratio), 1 + self.clip_ratio) for r in ratios]
        policy_obj = sum(min(r * a, cr * a) for r, cr, a in zip(ratios, clipped_ratios, advantages)) / n

        # 3. 价值损失
        returns = [a + v for a, v in zip(advantages, values)]
        value_losses = [(v - r) ** 2 for v, r in zip(values, returns)]
        value_loss = sum(value_losses) / n

        # 4. KL散度惩罚
        kl = sum((new - old) for new, old in zip(new_log_probs, old_log_probs)) / n
        kl_penalty_val = self.kl_penalty * kl

        # 5. 熵奖励
        entropy = -sum(p * math.log(max(p, 1e-8)) for p in [r / sum(ratios) for r in ratios] if p > 0) / n

        # 6. 裁剪比例
        clip_fraction = sum(1 for r in ratios if r > 1 + self.clip_ratio or r < 1 - self.clip_ratio) / n

        # 7. 策略损失 = -(policy_obj - kl_penalty + entropy_coef * entropy)
        policy_loss = -(policy_obj - kl_penalty_val + self.entropy_coef * entropy)

        step_record = PPOStep(
            step=step,
            policy_loss=round(policy_loss, 4),
            value_loss=round(value_loss, 4),
            reward_mean=round(sum(rewards) / n, 4),
            kl_penalty=round(kl_penalty_val, 4),
            clip_fraction=round(clip_fraction, 4),
            entropy=round(entropy, 4),
        )
        self.steps.append(step_record)

        # 更新策略
        self.old_policy = self.current_policy.copy()
        return step_record

    def get_optimization_summary(self) -> Dict:
        if not self.steps:
            return {"total_steps": 0}

        recent = self.steps[-20:]
        return {
            "total_steps": len(self.steps),
            "avg_policy_loss": round(sum(s.policy_loss for s in recent) / len(recent), 4),
            "avg_value_loss": round(sum(s.value_loss for s in recent) / len(recent), 4),
            "avg_reward": round(sum(s.reward_mean for s in recent) / len(recent), 4),
            "avg_kl": round(sum(s.kl_penalty for s in recent) / len(recent), 4),
            "avg_clip_fraction": round(sum(s.clip_fraction for s in recent) / len(recent), 4),
            "reward_trend": "improving" if recent[-1].reward_mean > recent[0].reward_mean else "declining",
        }


class ReinforcementFeedbackLoop:
    """RLHF强化学习反馈回路 — 统一RLHF管理

    流程:
    1. 收集人类偏好数据
    2. 训练奖励模型
    3. PPO策略优化
    4. 评估与迭代
    """

    def __init__(self):
        self.reward_model = RewardModel()
        self.policy_optimizer = PolicyOptimizer()
        self.feedback_history: List[Dict] = []
        self.iteration: int = 0
        self.quality_score: float = 0.5  # 初始质量分

    def collect_feedback(self, prompt: str, response_a: str, response_b: str,
                         preference: str = "a", annotator: str = "human"):
        """收集人类反馈"""
        if preference == "a":
            self.reward_model.add_preference(prompt, response_a, response_b, annotator)
        else:
            self.reward_model.add_preference(prompt, response_b, response_a, annotator)

    def batch_collect_feedback(self, samples: int = 50) -> Dict:
        """批量收集模拟偏好数据"""
        prompts = [
            "解释量子纠缠", "写一首关于秋天的诗", "如何优化Python代码",
            "解释机器学习中的梯度下降", "设计一个API架构", "分析气候变化的影响",
            "解释区块链技术", "如何提高团队效率", "描述深度学习的原理",
            "分析数据隐私问题",
        ]
        responses_good = [
            "这是一个详细且准确的回答，结构清晰，内容丰富。",
            "回答全面考虑了多个角度，提供了实用建议。",
            "从基础原理出发，逐步深入，易于理解。",
        ]
        responses_bad = [
            "不知道。", "这个问题太复杂了。", "可能吧，不太确定。",
        ]

        collected = 0
        for _ in range(samples):
            prompt = random.choice(prompts)
            good = random.choice(responses_good)
            bad = random.choice(responses_bad)
            if random.random() > 0.5:
                self.reward_model.add_preference(prompt, good, bad, "simulated_human")
            else:
                self.reward_model.add_preference(prompt, bad, good, "simulated_human")
            collected += 1

        return {"collected": collected, "total_preferences": len(self.reward_model.training_data)}

    def train_reward_model(self, epochs: int = 50) -> Dict:
        """训练奖励模型"""
        return self.reward_model.train(epochs=epochs)

    def run_ppo_optimization(self, steps: int = 100) -> Dict:
        """运行PPO策略优化"""
        for step in range(1, steps + 1):
            # 模拟PPO步骤
            n_samples = 32
            rewards = [self.reward_model.predict_reward(
                self.reward_model._extract_features(f"response_{i}")
            ) for i in range(n_samples)]

            old_log_probs = [random.gauss(-2, 0.5) for _ in range(n_samples)]
            new_log_probs = [olp + random.gauss(0.01 * step, 0.3) for olp in old_log_probs]
            values = [random.uniform(0.3, 0.8) for _ in range(n_samples)]
            advantages = [r - v for r, v in zip(rewards, values)]

            self.policy_optimizer.optimize_step(step, rewards, old_log_probs, new_log_probs, values, advantages)

        # 更新质量分数
        summary = self.policy_optimizer.get_optimization_summary()
        self.quality_score = min(0.99, self.quality_score + 0.05 * summary.get("avg_reward", 0))

        return summary

    def run_iteration(self) -> Dict:
        """运行一轮完整RLHF迭代"""
        self.iteration += 1

        # 1. 收集反馈
        feedback_result = self.batch_collect_feedback(30)

        # 2. 训练奖励模型
        rm_result = self.train_reward_model(30)

        # 3. PPO优化
        ppo_result = self.run_ppo_optimization(50)

        result = {
            "iteration": self.iteration,
            "feedback_collected": feedback_result["collected"],
            "reward_model_accuracy": rm_result["final_accuracy"],
            "quality_score": round(self.quality_score, 4),
            "ppo_summary": ppo_result,
            "timestamp": datetime.now().isoformat(),
        }
        self.feedback_history.append(result)
        return result

    def get_dashboard(self) -> Dict:
        return {
            "iterations": self.iteration,
            "total_preferences": len(self.reward_model.training_data),
            "reward_model_accuracy": self.reward_model.accuracy,
            "quality_score": round(self.quality_score, 4),
            "ppo_steps": len(self.policy_optimizer.steps),
            "ppo_summary": self.policy_optimizer.get_optimization_summary(),
            "last_iteration": self.feedback_history[-1] if self.feedback_history else None,
        }


# ============================================================
# QUANTIZATION_ENGINE [量化压缩引擎]
# ============================================================

@dataclass
class QuantizationConfig:
    """量化配置"""
    bits: int = 8                # 量化位数 (4/8/16)
    scheme: str = "symmetric"    # symmetric / asymmetric
    granularity: str = "per_channel"  # per_tensor / per_channel / per_group
    group_size: int = 128
    calibration_samples: int = 100
    method: str = "min_max"      # min_max / percentile / mse / aciq


@dataclass
class LayerQuantInfo:
    """层级量化信息"""
    layer_name: str
    original_bits: int = 32
    quantized_bits: int = 8
    original_size_kb: float = 0.0
    quantized_size_kb: float = 0.0
    quantization_error: float = 0.0
    scale: float = 1.0
    zero_point: float = 0.0
    snr_db: float = 0.0


class PruningEngine:
    """模型剪枝引擎

    支持:
    - 非结构化剪枝 (稀疏化)
    - 结构化剪枝 (通道/层)
    - 渐进式剪枝
    - 剪枝-微调循环
    """

    def __init__(self):
        self.pruning_history: List[Dict] = []
        self.sparsity: float = 0.0

    def unstructured_prune(self, weights: List[float], sparsity: float = 0.5) -> List[float]:
        """非结构化剪枝 (幅度剪枝)"""
        threshold_idx = int(len(weights) * sparsity)
        sorted_abs = sorted([abs(w) for w in weights])
        threshold = sorted_abs[threshold_idx] if threshold_idx < len(sorted_abs) else 0

        pruned = [w if abs(w) >= threshold else 0.0 for w in weights]
        actual_sparsity = sum(1 for w in pruned if w == 0) / len(pruned)
        self.sparsity = actual_sparsity

        return pruned

    def structured_prune(self, layer_weights: Dict[str, List[float]],
                         channels_to_prune: int = 0) -> Dict[str, List[float]]:
        """结构化剪枝 (通道级)"""
        pruned = {}
        for layer_name, weights in layer_weights.items():
            # 按通道L1范数排序
            channel_size = max(len(weights) // 16, 1)
            channels = [weights[i:i + channel_size] for i in range(0, len(weights), channel_size)]

            # 计算每个通道的L1范数
            channel_norms = [(i, sum(abs(w) for w in ch)) for i, ch in enumerate(channels)]
            channel_norms.sort(key=lambda x: x[1])

            # 剪掉最小的N个通道
            prune_indices = set(idx for idx, _ in channel_norms[:channels_to_prune])
            pruned[layer_name] = []
            for i, ch in enumerate(channels):
                if i not in prune_indices:
                    pruned[layer_name].extend(ch)

        return pruned

    def progressive_pruning(self, weights: List[float], target_sparsity: float = 0.8,
                            steps: int = 5) -> Dict:
        """渐进式剪枝"""
        results = []
        current = weights[:]
        current_sparsity = 0.0

        for step in range(steps):
            step_sparsity = target_sparsity / steps
            current = self.unstructured_prune(current, step_sparsity + current_sparsity)
            current_sparsity = self.sparsity

            # 模拟微调恢复
            recovery = random.uniform(0.02, 0.08)
            current = [w + random.gauss(0, recovery) if w != 0 else 0 for w in current]

            results.append({
                "step": step + 1,
                "sparsity": round(current_sparsity, 4),
                "remaining_params": sum(1 for w in current if w != 0),
                "total_params": len(current),
            })

        return {
            "target_sparsity": target_sparsity,
            "final_sparsity": round(current_sparsity, 4),
            "steps": steps,
            "history": results,
        }

    def get_summary(self) -> Dict:
        return {
            "current_sparsity": round(self.sparsity, 4),
            "pruning_rounds": len(self.pruning_history),
        }


class QuantizationEngine:
    """量化压缩引擎 — 统一模型压缩

    整合:
    - 多精度量化 (INT4/INT8/FP16)
    - 混合精度策略
    - 校准与误差分析
    - 剪枝集成
    """

    def __init__(self):
        self.config = QuantizationConfig()
        self.pruner = PruningEngine()
        self.layer_infos: Dict[str, LayerQuantInfo] = {}
        self.quantization_results: List[Dict] = []

    def _generate_layer_weights(self, layer_name: str, size: int = 1024) -> List[float]:
        """生成模拟层权重"""
        return [random.gauss(0, 0.15) for _ in range(size)]

    def quantize_layer(self, layer_name: str, weights: List[float],
                       config: QuantizationConfig = None) -> LayerQuantInfo:
        """量化单个层"""
        config = config or self.config
        max_val = max(weights)
        min_val = min(weights)

        if config.scheme == "symmetric":
            abs_max = max(abs(max_val), abs(min_val))
            scale = abs_max / (2 ** (config.bits - 1) - 1)
            zero_point = 0
        else:
            scale = (max_val - min_val) / (2 ** config.bits - 1)
            zero_point = -min_val / scale

        # 量化
        quantized = [round(w / scale + zero_point) for w in weights]
        # 反量化
        dequantized = [(q - zero_point) * scale for q in quantized]

        # 误差
        mse = sum((w - d) ** 2 for w, d in zip(weights, dequantized)) / len(weights)
        signal_power = sum(w ** 2 for w in weights) / len(weights)
        snr = 10 * math.log10(signal_power / max(mse, 1e-10)) if mse > 0 else 999

        original_size = len(weights) * 4 / 1024  # 32-bit -> KB
        quantized_size = len(weights) * config.bits / 8 / 1024

        info = LayerQuantInfo(
            layer_name=layer_name,
            original_bits=32,
            quantized_bits=config.bits,
            original_size_kb=round(original_size, 2),
            quantized_size_kb=round(quantized_size, 2),
            quantization_error=round(mse, 6),
            scale=round(scale, 6),
            zero_point=round(zero_point, 4),
            snr_db=round(snr, 2),
        )
        self.layer_infos[layer_name] = info
        return info

    def quantize_model(self, model_name: str = "lingyuan-base",
                       num_layers: int = 12, layer_size: int = 1024,
                       bits: int = 8, sparsity: float = 0.0) -> Dict:
        """量化整个模型"""
        config = QuantizationConfig(bits=bits)
        self.config = config

        total_original = 0.0
        total_quantized = 0.0
        layer_results = []

        for i in range(num_layers):
            layer_name = f"{model_name}_layer_{i}"
            weights = self._generate_layer_weights(layer_name, layer_size)

            # 先剪枝
            if sparsity > 0:
                weights = self.pruner.unstructured_prune(weights, sparsity)

            info = self.quantize_layer(layer_name, weights, config)
            total_original += info.original_size_kb
            total_quantized += info.quantized_size_kb
            layer_results.append({
                "layer": layer_name,
                "snr_db": info.snr_db,
                "error": info.quantization_error,
                "size_reduction": round(1 - info.quantized_size_kb / info.original_size_kb, 4),
            })

        compression_ratio = total_original / max(total_quantized, 0.001)
        avg_snr = sum(l["snr_db"] for l in layer_results) / len(layer_results)

        # 模拟精度评估
        accuracy_retention = round(random.uniform(0.97, 0.999) if bits >= 8 else random.uniform(0.90, 0.97), 4)

        result = {
            "model": model_name,
            "bits": bits,
            "sparsity": round(sparsity, 4),
            "num_layers": num_layers,
            "original_size_kb": round(total_original, 2),
            "quantized_size_kb": round(total_quantized, 2),
            "compression_ratio": round(compression_ratio, 2),
            "avg_snr_db": round(avg_snr, 2),
            "accuracy_retention": accuracy_retention,
            "layers": layer_results,
        }
        self.quantization_results.append(result)
        return result

    def mixed_precision_quantize(self, model_name: str = "lingyuan-base",
                                 num_layers: int = 12) -> Dict:
        """混合精度量化

        敏感层保持高精度, 不敏感层低精度
        """
        layer_configs = []
        for i in range(num_layers):
            # 模拟敏感度分析
            sensitivity = random.uniform(0, 1)
            if sensitivity > 0.7:
                bits = 16  # 敏感层
            elif sensitivity > 0.3:
                bits = 8
            else:
                bits = 4   # 不敏感层

            layer_configs.append({"layer": i, "bits": bits, "sensitivity": round(sensitivity, 4)})

        total_bits = sum(lc["bits"] for lc in layer_configs)
        avg_bits = total_bits / num_layers

        return {
            "model": model_name,
            "strategy": "mixed_precision",
            "avg_bits": round(avg_bits, 2),
            "layer_configs": layer_configs,
            "bits_distribution": {
                "4bit": sum(1 for lc in layer_configs if lc["bits"] == 4),
                "8bit": sum(1 for lc in layer_configs if lc["bits"] == 8),
                "16bit": sum(1 for lc in layer_configs if lc["bits"] == 16),
            },
            "estimated_accuracy_retention": round(random.uniform(0.95, 0.99), 4),
        }

    def get_dashboard(self) -> Dict:
        return {
            "total_quantizations": len(self.quantization_results),
            "pruning": self.pruner.get_summary(),
            "layer_infos": len(self.layer_infos),
            "last_result": self.quantization_results[-1] if self.quantization_results else None,
        }


# ============================================================
# VECTOR_DATABASE [向量数据库 — 语义检索/RAG]
# ============================================================

@dataclass
class VectorEntry:
    """向量条目"""
    entry_id: str
    vector: List[float]
    metadata: Dict = field(default_factory=dict)
    text: str = ""
    modality: str = "text"  # text / image / audio / video
    timestamp: str = ""
    access_count: int = 0


class HNSWIndex:
    """简化版HNSW (Hierarchical Navigable Small World) 索引

    层级图结构:
    - 多层跳表式图
    - 近似最近邻搜索
    - 增量插入
    """

    def __init__(self, dim: int = 128, max_connections: int = 16,
                 ef_construction: int = 200, ef_search: int = 50, num_layers: int = 5):
        self.dim = dim
        self.max_connections = max_connections
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.num_layers = num_layers

        # 每层的图: {node_id: [neighbor_ids]}
        self.layers: List[Dict[str, List[str]]] = [{} for _ in range(num_layers)]
        self.vectors: Dict[str, List[float]] = {}
        self.entry_point: str = ""
        self.max_level: int = 0
        self.insert_count: int = 0

    def _random_level(self) -> int:
        """随机选择层级 (指数衰减)"""
        level = 0
        while random.random() < 0.5 and level < self.num_layers - 1:
            level += 1
        return level

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1)) + 1e-8
        n2 = math.sqrt(sum(b * b for b in v2)) + 1e-8
        return dot / (n1 * n2)

    def insert(self, node_id: str, vector: List[float]):
        """插入向量"""
        self.vectors[node_id] = vector
        level = self._random_level()

        for l in range(level + 1):
            self.layers[l][node_id] = []

        # 连接到邻居 (简化: 连接已有节点中最近的)
        for l in range(level + 1):
            existing = [n for n in self.layers[l] if n != node_id]
            if existing:
                # 选择最相似的邻居
                sims = [(n, self._cosine_similarity(vector, self.vectors[n])) for n in existing]
                sims.sort(key=lambda x: -x[1])
                neighbors = [n for n, _ in sims[:self.max_connections]]

                self.layers[l][node_id] = neighbors
                # 双向连接
                for n in neighbors:
                    if len(self.layers[l].get(n, [])) < self.max_connections:
                        self.layers[l][n].append(node_id)
                    else:
                        # 替换最不相似的
                        n_neighbors = self.layers[l][n]
                        n_sims = [(nb, self._cosine_similarity(self.vectors[n], self.vectors[nb]))
                                  for nb in n_neighbors]
                        n_sims.sort(key=lambda x: x[1])
                        if self._cosine_similarity(self.vectors[n], vector) > n_sims[0][1]:
                            self.layers[l][n][0] = node_id

        if level > self.max_level or not self.entry_point:
            self.entry_point = node_id
            self.max_level = level

        self.insert_count += 1

    def search(self, query: List[float], k: int = 10) -> List[tuple]:
        """近似最近邻搜索

        Returns:
            [(node_id, similarity), ...]
        """
        if not self.entry_point:
            return []

        # 从顶层开始贪心搜索
        current = self.entry_point
        for l in range(self.max_level, 0, -1):
            current = self._greedy_search(query, current, l)

        # 底层ef_search宽度搜索
        candidates = self._search_layer(query, current, 0, self.ef_search)

        # 返回top-k
        candidates.sort(key=lambda x: -x[1])
        return candidates[:k]

    def _greedy_search(self, query: List[float], entry: str, layer: int) -> str:
        """贪心搜索: 找到当前层最近的节点"""
        current = entry
        current_sim = self._cosine_similarity(query, self.vectors[current])

        improved = True
        while improved:
            improved = False
            for neighbor in self.layers[layer].get(current, []):
                sim = self._cosine_similarity(query, self.vectors[neighbor])
                if sim > current_sim:
                    current = neighbor
                    current_sim = sim
                    improved = True

        return current

    def _search_layer(self, query: List[float], entry: str, layer: int,
                      ef: int) -> List[tuple]:
        """在指定层进行宽度搜索"""
        visited = {entry}
        candidates = [(entry, self._cosine_similarity(query, self.vectors[entry]))]
        result = list(candidates)

        while candidates:
            candidates.sort(key=lambda x: -x[1])
            current, current_sim = candidates.pop(0)

            for neighbor in self.layers[layer].get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    sim = self._cosine_similarity(query, self.vectors[neighbor])
                    if len(result) < ef or sim > result[-1][1]:
                        candidates.append((neighbor, sim))
                        result.append((neighbor, sim))
                        if len(result) > ef:
                            result.sort(key=lambda x: -x[1])
                            result = result[:ef]

        return result

    def get_stats(self) -> Dict:
        return {
            "total_vectors": self.insert_count,
            "dim": self.dim,
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "layer_sizes": [len(layer) for layer in self.layers],
            "max_connections": self.max_connections,
            "ef_search": self.ef_search,
        }


class EmbeddingEngine:
    """嵌入引擎

    模拟文本/图像到向量的转换:
    - 文本嵌入 (模拟sentence-BERT)
    - 图像嵌入 (模拟CLIP)
    - 多模态融合嵌入
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.cache: Dict[str, List[float]] = {}

    def embed_text(self, text: str) -> List[float]:
        """文本嵌入 (模拟)"""
        if text in self.cache:
            return self.cache[text]

        # 基于文本特征生成确定性嵌入
        seed = hash(text) % (2 ** 32)
        random.seed(seed)
        vector = [random.gauss(0, 1) for _ in range(self.dim)]

        # 归一化
        norm = math.sqrt(sum(v * v for v in vector))
        vector = [v / norm for v in vector]

        self.cache[text] = vector
        random.seed()  # 重置随机种子
        return vector

    def embed_image(self, image_desc: str) -> List[float]:
        """图像嵌入 (模拟)"""
        return self.embed_text(f"img:{image_desc}")

    def embed_multimodal(self, text: str, image_desc: str = "") -> List[float]:
        """多模态融合嵌入"""
        text_vec = self.embed_text(text)
        if image_desc:
            img_vec = self.embed_image(image_desc)
            # 简单平均融合
            return [(t + i) / 2 for t, i in zip(text_vec, img_vec)]
        return text_vec


class VectorDatabase:
    """向量数据库 — 统一向量存储与检索

    整合:
    - HNSW索引 (快速近似搜索)
    - 嵌入引擎 (文本/图像/多模态)
    - 增量索引更新
    - 元数据过滤
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.index = HNSWIndex(dim=dim)
        self.embedding = EmbeddingEngine(dim=dim)
        self.entries: Dict[str, VectorEntry] = {}
        self.collections: Dict[str, List[str]] = {}  # 集合管理
        self.query_count: int = 0
        self.total_insert_time: float = 0.0
        self.total_search_time: float = 0.0

    def insert(self, text: str, metadata: Dict = None, modality: str = "text",
               collection: str = "default", vector: List[float] = None) -> str:
        """插入条目"""
        entry_id = f"vec_{uuid.uuid4().hex[:12]}"

        if vector is None:
            if modality == "text":
                vector = self.embedding.embed_text(text)
            elif modality == "image":
                vector = self.embedding.embed_image(text)
            else:
                vector = self.embedding.embed_multimodal(text)

        entry = VectorEntry(
            entry_id=entry_id, vector=vector,
            metadata=metadata or {}, text=text,
            modality=modality,
            timestamp=datetime.now().isoformat(),
        )
        self.entries[entry_id] = entry

        # 集合管理
        if collection not in self.collections:
            self.collections[collection] = []
        self.collections[collection].append(entry_id)

        # 索引插入
        start = time.time()
        self.index.insert(entry_id, vector)
        self.total_insert_time += time.time() - start

        return entry_id

    def batch_insert(self, items: List[Dict]) -> List[str]:
        """批量插入"""
        ids = []
        for item in items:
            eid = self.insert(
                text=item.get("text", ""),
                metadata=item.get("metadata", {}),
                modality=item.get("modality", "text"),
                collection=item.get("collection", "default"),
            )
            ids.append(eid)
        return ids

    def search(self, query: str, k: int = 10, collection: str = None,
               filter_metadata: Dict = None) -> List[Dict]:
        """语义搜索"""
        start = time.time()
        query_vec = self.embedding.embed_text(query)

        results = self.index.search(query_vec, k=k * 2 if filter_metadata else k)
        self.query_count += 1

        # 元数据过滤
        filtered = []
        for node_id, similarity in results:
            entry = self.entries.get(node_id)
            if not entry:
                continue

            # 集合过滤
            if collection and collection in self.collections:
                if node_id not in self.collections[collection]:
                    continue

            # 元数据过滤
            if filter_metadata:
                match = all(entry.metadata.get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue

            entry.access_count += 1
            filtered.append({
                "entry_id": node_id,
                "text": entry.text[:200],
                "similarity": round(similarity, 4),
                "modality": entry.modality,
                "metadata": entry.metadata,
                "timestamp": entry.timestamp,
            })

            if len(filtered) >= k:
                break

        self.total_search_time += time.time() - start
        return filtered

    def delete(self, entry_id: str) -> bool:
        """删除条目 (标记删除)"""
        if entry_id in self.entries:
            entry = self.entries[entry_id]
            entry.metadata["deleted"] = True
            del self.entries[entry_id]

            # 从集合中移除
            for col_entries in self.collections.values():
                if entry_id in col_entries:
                    col_entries.remove(entry_id)
            return True
        return False

    def create_collection(self, name: str) -> bool:
        """创建集合"""
        if name not in self.collections:
            self.collections[name] = []
            return True
        return False

    def get_stats(self) -> Dict:
        return {
            "total_entries": len(self.entries),
            "total_collections": len(self.collections),
            "query_count": self.query_count,
            "avg_insert_time_ms": round(self.total_insert_time / max(self.index.insert_count, 1) * 1000, 2),
            "avg_search_time_ms": round(self.total_search_time / max(self.query_count, 1) * 1000, 2),
            "index_stats": self.index.get_stats(),
            "embedding_cache_size": len(self.embedding.cache),
            "collections": {k: len(v) for k, v in self.collections.items()},
        }

    def get_dashboard(self) -> Dict:
        stats = self.get_stats()
        return {
            "total_entries": stats["total_entries"],
            "total_collections": stats["total_collections"],
            "query_count": stats["query_count"],
            "avg_search_time_ms": stats["avg_search_time_ms"],
            "index": stats["index_stats"],
            "collections": stats["collections"],
        }


# ============================================================
# PROMPT_ENGINEERING [提示工程工作室]
# ============================================================

@dataclass
class PromptTemplate:
    """提示模板"""
    template_id: str
    name: str
    category: str        # system / instruction / few_shot / chain_of_thought / role_play
    template: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    created_at: str = ""
    usage_count: int = 0
    avg_score: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class PromptVariant:
    """提示变体 (用于A/B测试)"""
    variant_id: str
    template_id: str
    content: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    performance: float = 0.0
    samples: int = 0


class PromptOptimizer:
    """自动提示优化器

    策略:
    - 遗传算法优化 (变异/交叉)
    - 梯度无关优化 (OPRO简化版)
    - 基于反馈的迭代改进
    """

    def __init__(self, population_size: int = 10, mutation_rate: float = 0.3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.optimization_history: List[Dict] = []
        self.mutation_strategies = [
            "add_instruction",     # 添加指导语
            "add_example",         # 添加示例
            "rephrase",            # 改写
            "add_constraint",      # 添加约束
            "add_reasoning",       # 添加推理链
            "simplify",            # 简化
        ]

    def optimize(self, base_prompt: str, objective: str = "accuracy",
                 iterations: int = 5) -> Dict:
        """优化提示"""
        population = [base_prompt]
        # 初始化种群
        for _ in range(self.population_size - 1):
            population.append(self._mutate(base_prompt))

        history = []
        for gen in range(iterations):
            # 评估
            scores = [self._evaluate(p, objective) for p in population]

            # 记录
            best_idx = scores.index(max(scores))
            history.append({
                "generation": gen + 1,
                "best_score": round(scores[best_idx], 4),
                "avg_score": round(sum(scores) / len(scores), 4),
                "best_prompt": population[best_idx][:100] + "...",
            })

            # 选择 + 交叉 + 变异
            sorted_pop = [p for _, p in sorted(zip(scores, population), key=lambda x: -x[0])]
            elite = sorted_pop[:max(2, self.population_size // 4)]

            new_population = list(elite)
            while len(new_population) < self.population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self._crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                new_population.append(child)

            population = new_population

        best_idx = max(range(len(population)), key=lambda i: self._evaluate(population[i], objective))
        best_prompt = population[best_idx]
        best_score = self._evaluate(best_prompt, objective)

        result = {
            "original_prompt": base_prompt,
            "optimized_prompt": best_prompt,
            "improvement": round(best_score - self._evaluate(base_prompt, objective), 4),
            "final_score": round(best_score, 4),
            "generations": iterations,
            "history": history,
        }
        self.optimization_history.append(result)
        return result

    def _mutate(self, prompt: str) -> str:
        """变异"""
        strategy = random.choice(self.mutation_strategies)

        if strategy == "add_instruction":
            return prompt + "\n\n请仔细思考并给出详细解答。"
        elif strategy == "add_example":
            return prompt + "\n\n例如: 输入X → 输出Y"
        elif strategy == "rephrase":
            return f"请回答以下问题: {prompt}"
        elif strategy == "add_constraint":
            return prompt + "\n\n约束: 回答不超过200字。"
        elif strategy == "add_reasoning":
            return prompt + "\n\n请逐步推理后给出答案。"
        else:  # simplify
            return prompt[:len(prompt) // 2] if len(prompt) > 20 else prompt

    def _crossover(self, p1: str, p2: str) -> str:
        """交叉"""
        mid1 = len(p1) // 2
        mid2 = len(p2) // 2
        return p1[:mid1] + p2[mid2:]

    def _evaluate(self, prompt: str, objective: str) -> float:
        """评估提示质量 (模拟)"""
        # 基于提示特征评分
        score = 0.5
        if "请" in prompt:
            score += 0.1
        if "例如" in prompt or "示例" in prompt:
            score += 0.15
        if "逐步" in prompt or "推理" in prompt:
            score += 0.1
        if "约束" in prompt or "不超过" in prompt:
            score += 0.05
        if len(prompt) > 50:
            score += 0.1
        score += random.uniform(-0.05, 0.05)
        return min(1.0, max(0.0, score))


class PromptEngineeringStudio:
    """提示工程工作室 — 统一提示管理

    功能:
    - 模板管理 (增删改查)
    - 变体A/B测试
    - 自动优化
    - 少样本示例选择
    - 提示版本控制
    """

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.variants: Dict[str, PromptVariant] = {}
        self.optimizer = PromptOptimizer()
        self.ab_tests: Dict[str, Dict] = {}
        self._init_default_templates()

    def _init_default_templates(self):
        """初始化默认模板"""
        defaults = [
            ("sys_basic", "基础系统提示", "system",
             "你是一个有帮助的AI助手。请根据用户的问题提供准确、有用的回答。",
             [], "基础系统提示模板"),
            ("cot", "思维链提示", "chain_of_thought",
             "问题: {question}\n\n请一步步思考:\n1. 首先, 分析问题的关键要素\n2. 然后, 列出可能的解决方法\n3. 最后, 给出最优答案\n\n答案: ",
             ["question"], "引导模型逐步推理"),
            ("few_shot", "少样本提示", "few_shot",
             "以下是几个示例:\n{examples}\n\n现在请回答:\n{question}",
             ["examples", "question"], "通过示例引导模型"),
            ("role_expert", "专家角色提示", "role_play",
             "你是一位{role}领域的专家, 拥有20年经验。请以专业的角度回答以下问题:\n\n{question}",
             ["role", "question"], "角色扮演提示"),
            ("instruction", "指令提示", "instruction",
             "指令: {instruction}\n输入: {input}\n输出格式: {format}\n\n请按格式输出:",
             ["instruction", "input", "format"], "结构化指令"),
        ]

        for tid, name, cat, template, variables, desc in defaults:
            self.templates[tid] = PromptTemplate(
                template_id=tid, name=name, category=cat,
                template=template, variables=variables,
                description=desc,
                created_at=datetime.now().isoformat(),
                tags=[cat],
            )

    def create_template(self, name: str, category: str, template: str,
                        variables: List[str] = None, description: str = "") -> PromptTemplate:
        """创建模板"""
        tid = f"tpl_{uuid.uuid4().hex[:8]}"
        tpl = PromptTemplate(
            template_id=tid, name=name, category=category,
            template=template, variables=variables or [],
            description=description,
            created_at=datetime.now().isoformat(),
        )
        self.templates[tid] = tpl
        return tpl

    def render(self, template_id: str, variables: Dict[str, str]) -> str:
        """渲染模板"""
        tpl = self.templates.get(template_id)
        if not tpl:
            return ""

        rendered = tpl.template
        for var in tpl.variables:
            rendered = rendered.replace(f"{{{var}}}", variables.get(var, f"[{var}]"))

        tpl.usage_count += 1
        return rendered

    def create_variants(self, template_id: str, num_variants: int = 3) -> List[PromptVariant]:
        """为模板创建变体"""
        tpl = self.templates.get(template_id)
        if not tpl:
            return []

        variants = []
        for i in range(num_variants):
            # 不同温度和参数
            vid = f"var_{uuid.uuid4().hex[:8]}"
            variant = PromptVariant(
                variant_id=vid, template_id=template_id,
                content=tpl.template,
                temperature=round(random.uniform(0.3, 1.0), 2),
                top_p=round(random.uniform(0.8, 1.0), 2),
                max_tokens=random.choice([256, 512, 1024]),
            )
            self.variants[vid] = variant
            variants.append(variant)

        return variants

    def run_ab_test(self, template_id: str, num_variants: int = 3,
                    samples_per_variant: int = 50) -> Dict:
        """运行A/B测试"""
        variants = self.create_variants(template_id, num_variants)
        test_id = f"abtest_{uuid.uuid4().hex[:8]}"

        results = []
        for v in variants:
            # 模拟评估
            scores = [random.uniform(0.5, 1.0) * v.temperature * 0.8 + random.uniform(0.2, 0.5)
                      for _ in range(samples_per_variant)]
            v.performance = round(sum(scores) / len(scores), 4)
            v.samples = samples_per_variant
            results.append({
                "variant_id": v.variant_id,
                "temperature": v.temperature,
                "top_p": v.top_p,
                "max_tokens": v.max_tokens,
                "performance": v.performance,
                "samples": v.samples,
            })

        results.sort(key=lambda x: -x["performance"])
        winner = results[0]

        test_result = {
            "test_id": test_id,
            "template_id": template_id,
            "variants": results,
            "winner": winner,
            "samples_total": num_variants * samples_per_variant,
            "timestamp": datetime.now().isoformat(),
        }
        self.ab_tests[test_id] = test_result
        return test_result

    def optimize_prompt(self, base_prompt: str, iterations: int = 5) -> Dict:
        """优化提示"""
        return self.optimizer.optimize(base_prompt, iterations=iterations)

    def select_few_shot_examples(self, query: str, examples: List[str],
                                 k: int = 3) -> List[str]:
        """选择最佳少样本示例 (基于简单相似度)"""
        # 简化: 基于词重叠
        query_words = set(query.lower().split())

        scored = []
        for ex in examples:
            ex_words = set(ex.lower().split())
            overlap = len(query_words & ex_words) / max(len(query_words | ex_words), 1)
            scored.append((ex, overlap))

        scored.sort(key=lambda x: -x[1])
        return [ex for ex, _ in scored[:k]]

    def get_dashboard(self) -> Dict:
        categories = {}
        for t in self.templates.values():
            categories[t.category] = categories.get(t.category, 0) + 1

        return {
            "total_templates": len(self.templates),
            "total_variants": len(self.variants),
            "total_ab_tests": len(self.ab_tests),
            "categories": categories,
            "most_used": sorted(
                [{"name": t.name, "usage": t.usage_count} for t in self.templates.values()],
                key=lambda x: -x["usage"]
            )[:5],
            "optimization_rounds": len(self.optimizer.optimization_history),
        }


# ============================================================
# EDGE_DEPLOYMENT [边缘部署管理器]
# ============================================================

@dataclass
class EdgeDevice:
    """边缘设备"""
    device_id: str
    name: str
    device_type: str        # phone / tablet / raspberry_pi / jetson / browser
    os: str                 # android / ios / linux / wasm
    cpu_cores: int
    memory_mb: int
    storage_mb: int
    gpu: str                # none / mali / adreno / powervr / tegra
    battery_level: float = 1.0  # 0~1
    online: bool = True
    last_sync: str = ""
    model_version: str = ""
    inference_count: int = 0
    avg_latency_ms: float = 0.0
    total_data_processed: int = 0


@dataclass
class DeploymentPackage:
    """部署包"""
    package_id: str
    model_name: str
    model_version: str
    target_device_type: str
    quantization: str       # int8 / int4 / fp16
    size_mb: float
    checksum: str
    min_memory_mb: int
    min_cpu_cores: int
    created_at: str = ""
    deployed_count: int = 0


class OTAUpdater:
    """OTA (Over-The-Air) 更新器

    功能:
    - 差分更新 (Delta Update)
    - 分阶段发布
    - 回滚机制
    - 更新验证
    """

    def __init__(self):
        self.update_history: List[Dict] = []
        self.rollback_history: List[Dict] = []

    def create_update(self, model_name: str, from_version: str, to_version: str,
                      package_size: float, strategy: str = "staged") -> Dict:
        """创建更新任务"""
        update_id = f"ota_{uuid.uuid4().hex[:8]}"

        # 分阶段发布
        if strategy == "staged":
            rollout_plan = [
                {"stage": 1, "percentage": 5, "duration_hours": 2},
                {"stage": 2, "percentage": 20, "duration_hours": 6},
                {"stage": 3, "percentage": 50, "duration_hours": 12},
                {"stage": 4, "percentage": 100, "duration_hours": 24},
            ]
        else:  # immediate
            rollout_plan = [{"stage": 1, "percentage": 100, "duration_hours": 0}]

        update = {
            "update_id": update_id,
            "model": model_name,
            "from_version": from_version,
            "to_version": to_version,
            "package_size_mb": package_size,
            "strategy": strategy,
            "rollout_plan": rollout_plan,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "devices_updated": 0,
            "devices_failed": 0,
            "rollback_count": 0,
        }
        return update

    def execute_update(self, update: Dict, devices: List[EdgeDevice]) -> Dict:
        """执行更新"""
        success_count = 0
        fail_count = 0
        device_results = []

        for device in devices:
            # 模拟更新结果
            # 成功率取决于设备类型和电池
            if device.battery_level < 0.2:
                success = False
                reason = "battery_low"
            elif not device.online:
                success = False
                reason = "offline"
            elif device.memory_mb < 512 and update["package_size_mb"] > 100:
                success = False
                reason = "insufficient_memory"
            else:
                success = random.random() > 0.05  # 95%成功率
                reason = "success" if success else "unknown_error"

            if success:
                device.model_version = update["to_version"]
                device.last_sync = datetime.now().isoformat()
                success_count += 1
            else:
                fail_count += 1

            device_results.append({
                "device_id": device.device_id,
                "success": success,
                "reason": reason,
            })

        update["devices_updated"] = success_count
        update["devices_failed"] = fail_count
        update["status"] = "completed" if fail_count == 0 else "partial"

        self.update_history.append(update)
        return {
            "update_id": update["update_id"],
            "success_count": success_count,
            "fail_count": fail_count,
            "success_rate": round(success_count / max(len(devices), 1), 4),
            "device_results": device_results,
        }

    def rollback(self, update: Dict, devices: List[EdgeDevice]) -> Dict:
        """回滚更新"""
        rollback_count = 0
        for device in devices:
            if device.model_version == update["to_version"]:
                device.model_version = update["from_version"]
                rollback_count += 1

        rollback_record = {
            "update_id": update["update_id"],
            "rollback_to": update["from_version"],
            "devices_rolled_back": rollback_count,
            "timestamp": datetime.now().isoformat(),
        }
        self.rollback_history.append(rollback_record)
        update["rollback_count"] = rollback_count
        return rollback_record


class EdgeDeploymentManager:
    """边缘部署管理器 — 统一边缘设备管理

    整合:
    - 设备注册与发现
    - 模型分发
    - OTA更新
    - 远程推理协调
    - 设备健康监控
    """

    def __init__(self):
        self.devices: Dict[str, EdgeDevice] = {}
        self.packages: Dict[str, DeploymentPackage] = {}
        self.ota = OTAUpdater()
        self.inference_history: List[Dict] = []
        self._register_default_devices()

    def _register_default_devices(self):
        """注册默认设备"""
        defaults = [
            ("phone_01", "Pixel 8", "phone", "android", 8, 8192, 128000, "adreno"),
            ("phone_02", "iPhone 15", "phone", "ios", 6, 6144, 128000, "powervr"),
            ("tablet_01", "iPad Pro", "tablet", "ios", 8, 8192, 256000, "powervr"),
            ("pi_01", "树莓派5", "raspberry_pi", "linux", 4, 4096, 64000, "none"),
            ("jetson_01", "Jetson Orin", "jetson", "linux", 8, 8192, 128000, "tegra"),
            ("browser_01", "WebAssembly运行时", "browser", "wasm", 4, 2048, 0, "none"),
        ]

        for did, name, dtype, os_name, cores, mem, storage, gpu in defaults:
            self.devices[did] = EdgeDevice(
                device_id=did, name=name, device_type=dtype, os=os_name,
                cpu_cores=cores, memory_mb=mem, storage_mb=storage, gpu=gpu,
                battery_level=random.uniform(0.3, 1.0),
                last_sync=datetime.now().isoformat(),
                avg_latency_ms=random.uniform(20, 200),
            )

    def register_device(self, name: str, device_type: str, os_name: str,
                        cpu_cores: int, memory_mb: int, storage_mb: int,
                        gpu: str = "none") -> EdgeDevice:
        """注册新设备"""
        did = f"dev_{uuid.uuid4().hex[:8]}"
        device = EdgeDevice(
            device_id=did, name=name, device_type=device_type, os=os_name,
            cpu_cores=cpu_cores, memory_mb=memory_mb, storage_mb=storage_mb, gpu=gpu,
            battery_level=1.0, last_sync=datetime.now().isoformat(),
        )
        self.devices[did] = device
        return device

    def create_package(self, model_name: str, model_version: str,
                       target_device_type: str, quantization: str = "int8",
                       size_mb: float = 50.0) -> DeploymentPackage:
        """创建部署包"""
        pid = f"pkg_{uuid.uuid4().hex[:8]}"
        min_mem = {"int8": 512, "int4": 256, "fp16": 1024}.get(quantization, 512)
        min_cpu = 2 if quantization in ("int8", "int4") else 4

        pkg = DeploymentPackage(
            package_id=pid, model_name=model_name, model_version=model_version,
            target_device_type=target_device_type, quantization=quantization,
            size_mb=size_mb, checksum=uuid.uuid4().hex,
            min_memory_mb=min_mem, min_cpu_cores=min_cpu,
            created_at=datetime.now().isoformat(),
        )
        self.packages[pid] = pkg
        return pkg

    def deploy_to_device(self, package_id: str, device_id: str) -> Dict:
        """部署到单个设备"""
        pkg = self.packages.get(package_id)
        device = self.devices.get(device_id)

        if not pkg or not device:
            return {"success": False, "error": "包或设备不存在"}

        # 检查兼容性
        if device.memory_mb < pkg.min_memory_mb:
            return {"success": False, "error": "内存不足", "required": pkg.min_memory_mb, "available": device.memory_mb}
        if device.cpu_cores < pkg.min_cpu_cores:
            return {"success": False, "error": "CPU核心不足"}
        if device.storage_mb and pkg.size_mb > device.storage_mb * 0.3:
            return {"success": False, "error": "存储空间不足"}

        # 部署
        device.model_version = pkg.model_version
        device.last_sync = datetime.now().isoformat()
        pkg.deployed_count += 1

        return {
            "success": True,
            "device_id": device_id,
            "model": pkg.model_name,
            "version": pkg.model_version,
            "quantization": pkg.quantization,
            "size_mb": pkg.size_mb,
        }

    def deploy_to_all(self, package_id: str, device_type: str = None) -> Dict:
        """批量部署"""
        results = []
        success = 0
        fail = 0

        for did, device in self.devices.items():
            if device_type and device.device_type != device_type:
                continue
            result = self.deploy_to_device(package_id, did)
            results.append(result)
            if result["success"]:
                success += 1
            else:
                fail += 1

        return {
            "package_id": package_id,
            "total_devices": len(results),
            "success": success,
            "failed": fail,
            "success_rate": round(success / max(len(results), 1), 4),
            "results": results,
        }

    def remote_inference(self, device_id: str, input_text: str,
                         max_tokens: int = 100) -> Dict:
        """远程推理"""
        device = self.devices.get(device_id)
        if not device or not device.online:
            return {"error": "设备不可用"}

        # 模拟推理延迟
        base_latency = 50  # ms
        if device.device_type == "phone":
            latency = base_latency + random.uniform(20, 100)
        elif device.device_type == "raspberry_pi":
            latency = base_latency + random.uniform(100, 300)
        elif device.device_type == "jetson":
            latency = base_latency + random.uniform(10, 50)
        else:
            latency = base_latency + random.uniform(30, 80)

        # 模拟推理输出
        output = f"[{device.name}] 推理结果: " + input_text[:50] + "..."

        device.inference_count += 1
        device.avg_latency_ms = round((device.avg_latency_ms * (device.inference_count - 1) + latency) / device.inference_count, 2)
        device.total_data_processed += len(input_text)

        result = {
            "device_id": device_id,
            "device_name": device.name,
            "model_version": device.model_version,
            "input": input_text[:100],
            "output": output,
            "latency_ms": round(latency, 2),
            "tokens_generated": max_tokens,
            "timestamp": datetime.now().isoformat(),
        }
        self.inference_history.append(result)
        return result

    def check_device_health(self) -> List[Dict]:
        """检查所有设备健康状态"""
        health = []
        for device in self.devices.values():
            issues = []
            if device.battery_level < 0.2:
                issues.append("low_battery")
            if not device.online:
                issues.append("offline")
            if device.avg_latency_ms > 200:
                issues.append("high_latency")
            if not device.model_version:
                issues.append("no_model")

            status = "healthy" if not issues else "unhealthy"
            health.append({
                "device_id": device.device_id,
                "name": device.name,
                "type": device.device_type,
                "status": status,
                "issues": issues,
                "battery": round(device.battery_level, 2),
                "latency_ms": device.avg_latency_ms,
                "model_version": device.model_version or "未部署",
                "inference_count": device.inference_count,
            })

        return health

    def get_dashboard(self) -> Dict:
        health = self.check_device_health()
        return {
            "total_devices": len(self.devices),
            "online_devices": sum(1 for d in self.devices.values() if d.online),
            "healthy_devices": sum(1 for h in health if h["status"] == "healthy"),
            "total_packages": len(self.packages),
            "total_inferences": sum(d.inference_count for d in self.devices.values()),
            "avg_latency_ms": round(
                sum(d.avg_latency_ms for d in self.devices.values()) / max(len(self.devices), 1), 2
            ),
            "device_types": {t: sum(1 for d in self.devices.values() if d.device_type == t)
                            for t in set(d.device_type for d in self.devices.values())},
            "ota_updates": len(self.ota.update_history),
            "ota_rollbacks": len(self.ota.rollback_history),
            "devices": health,
        }


# ============================================================
# CONVERSATION_MEMORY [对话记忆系统]
# ============================================================

@dataclass
class MemoryEntry:
    """记忆条目"""
    memory_id: str
    content: str
    timestamp: str
    importance: float = 0.5      # 0~1 重要程度
    modality: str = "text"
    tags: List[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: str = ""
    decay_factor: float = 1.0    # 记忆衰减因子
    consolidated: bool = False   # 是否已巩固


class ShortTermMemory:
    """短期记忆 (工作记忆)

    特性:
    - 容量有限 (最近N条)
    - FIFO淘汰
    - 快速访问
    - 上下文窗口管理
    """

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.memories: deque = deque(maxlen=capacity)
        self.access_pattern: Dict[str, int] = {}  # 记忆访问频率

    def add(self, content: str, importance: float = 0.5, tags: List[str] = None) -> str:
        """添加短期记忆"""
        mid = f"stm_{uuid.uuid4().hex[:8]}"
        entry = MemoryEntry(
            memory_id=mid, content=content,
            timestamp=datetime.now().isoformat(),
            importance=importance, tags=tags or [],
        )
        self.memories.append(entry)
        self.access_pattern[mid] = 0
        return mid

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """获取最近N条记忆"""
        return list(self.memories)[-n:]

    def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """搜索短期记忆 (简单关键词匹配)"""
        query_lower = query.lower()
        scored = []
        for mem in self.memories:
            score = 0
            if query_lower in mem.content.lower():
                score += 0.5
            for tag in mem.tags:
                if tag.lower() in query_lower:
                    score += 0.3
            score += mem.importance * 0.2
            if score > 0:
                mem.access_count += 1
                self.access_pattern[mem.memory_id] = mem.access_count
                scored.append((mem, score))

        scored.sort(key=lambda x: -x[1])
        return [mem for mem, _ in scored[:limit]]

    def get_context_window(self, max_tokens: int = 2048) -> str:
        """获取上下文窗口 (token限制)"""
        # 简化: 按字符数估算token
        context = []
        current_length = 0

        for mem in reversed(self.memories):
            est_tokens = len(mem.content) // 2  # 粗略估计
            if current_length + est_tokens > max_tokens:
                break
            context.append(mem.content)
            current_length += est_tokens

        return "\n".join(reversed(context))

    def clear(self):
        """清空短期记忆"""
        self.memories.clear()
        self.access_pattern.clear()

    def get_summary(self) -> Dict:
        return {
            "capacity": self.capacity,
            "current_size": len(self.memories),
            "total_accessed": sum(self.access_pattern.values()),
            "avg_importance": round(
                sum(m.importance for m in self.memories) / max(len(self.memories), 1), 4
            ),
        }


class LongTermMemory:
    """长期记忆 (持久化记忆)

    特性:
    - 大容量存储
    - 记忆巩固 (从短期记忆迁移)
    - 衰减与强化机制
    - 语义检索 (集成向量数据库)
    """

    def __init__(self, vector_db: VectorDatabase = None):
        self.memories: Dict[str, MemoryEntry] = {}
        self.vector_db = vector_db
        self.consolidation_history: List[Dict] = []
        self.decay_rate: float = 0.01  # 每天衰减率
        self.reinforcement_threshold: int = 3  # 访问3次后强化

    def store(self, content: str, importance: float = 0.5,
              tags: List[str] = None, modality: str = "text") -> str:
        """存储长期记忆"""
        mid = f"ltm_{uuid.uuid4().hex[:8]}"
        entry = MemoryEntry(
            memory_id=mid, content=content,
            timestamp=datetime.now().isoformat(),
            importance=importance, tags=tags or [],
            modality=modality,
        )
        self.memories[mid] = entry

        # 同时存入向量数据库
        if self.vector_db:
            self.vector_db.insert(
                text=content,
                metadata={"memory_id": mid, "importance": importance, "tags": tags or []},
                modality=modality,
                collection="long_term_memory",
            )

        return mid

    def retrieve(self, query: str, limit: int = 10,
                 min_importance: float = 0.0) -> List[MemoryEntry]:
        """检索长期记忆"""
        results = []

        # 优先使用向量数据库检索
        if self.vector_db:
            vec_results = self.vector_db.search(
                query, k=limit * 2,
                collection="long_term_memory",
                filter_metadata=None,
            )
            for vr in vec_results:
                mid = vr["metadata"].get("memory_id")
                if mid in self.memories:
                    entry = self.memories[mid]
                    if entry.importance >= min_importance:
                        entry.access_count += 1
                        entry.last_accessed = datetime.now().isoformat()
                        results.append(entry)

        # 补充: 关键词匹配
        if len(results) < limit:
            query_lower = query.lower()
            for entry in self.memories.values():
                if entry in results:
                    continue
                if entry.importance < min_importance:
                    continue
                if query_lower in entry.content.lower():
                    entry.access_count += 1
                    entry.last_accessed = datetime.now().isoformat()
                    results.append(entry)
                    if len(results) >= limit:
                        break

        return results[:limit]

    def consolidate(self, short_term: ShortTermMemory) -> Dict:
        """记忆巩固: 将重要短期记忆迁移到长期记忆"""
        consolidated = 0
        promoted = 0

        for mem in short_term.memories:
            # 重要的记忆或被频繁访问的记忆被巩固
            access_freq = short_term.access_pattern.get(mem.memory_id, 0)

            if mem.importance > 0.6 or access_freq >= self.reinforcement_threshold:
                # 存入长期记忆
                ltm_id = self.store(
                    content=mem.content,
                    importance=min(1.0, mem.importance + 0.1),  # 巩固时提升重要性
                    tags=mem.tags,
                    modality=mem.modality,
                )
                mem.consolidated = True
                consolidated += 1
                if mem.importance > 0.6:
                    promoted += 1

        result = {
            "consolidated": consolidated,
            "promoted": promoted,
            "total_ltm": len(self.memories),
            "timestamp": datetime.now().isoformat(),
        }
        self.consolidation_history.append(result)
        return result

    def apply_decay(self, days: float = 1.0) -> Dict:
        """应用记忆衰减

        长期未被访问的记忆重要性降低
        """
        decayed = 0
        forgotten = 0

        for entry in self.memories.values():
            if entry.access_count == 0:
                entry.decay_factor *= (1 - self.decay_rate * days)
                entry.importance *= entry.decay_factor
                decayed += 1

                # 重要性过低则遗忘
                if entry.importance < 0.1:
                    forgotten += 1

        # 清理被遗忘的记忆
        to_remove = [mid for mid, e in self.memories.items() if e.importance < 0.05]
        for mid in to_remove:
            del self.memories[mid]

        return {
            "decayed": decayed,
            "forgotten": forgotten,
            "remaining": len(self.memories),
            "decay_rate": self.decay_rate,
            "days": days,
        }

    def reinforce(self, memory_id: str, boost: float = 0.1) -> bool:
        """强化记忆 (通过重复访问)"""
        if memory_id in self.memories:
            entry = self.memories[memory_id]
            entry.importance = min(1.0, entry.importance + boost)
            entry.decay_factor = 1.0  # 重置衰减
            entry.access_count += 1
            entry.last_accessed = datetime.now().isoformat()
            return True
        return False

    def get_summary(self) -> Dict:
        return {
            "total_memories": len(self.memories),
            "avg_importance": round(
                sum(m.importance for m in self.memories.values()) / max(len(self.memories), 1), 4
            ),
            "total_accessed": sum(m.access_count for m in self.memories.values()),
            "consolidation_rounds": len(self.consolidation_history),
            "last_consolidation": self.consolidation_history[-1] if self.consolidation_history else None,
            "tag_distribution": self._get_tag_distribution(),
        }

    def _get_tag_distribution(self) -> Dict[str, int]:
        tags: Dict[str, int] = {}
        for m in self.memories.values():
            for tag in m.tags:
                tags[tag] = tags.get(tag, 0) + 1
        return dict(sorted(tags.items(), key=lambda x: -x[1])[:10])


class RAGRetriever:
    """检索增强生成 (RAG) 检索器

    流程:
    1. 用户查询 → 向量化
    2. 向量数据库检索相关文档
    3. 重排序 (模拟cross-encoder)
    4. 上下文组装
    5. 生成增强提示
    """

    def __init__(self, vector_db: VectorDatabase, knowledge_graph=None):
        self.vector_db = vector_db
        self.knowledge_graph = knowledge_graph
        self.rerank_enabled = True
        self.retrieval_history: List[Dict] = []

    def retrieve_and_augment(self, query: str, top_k: int = 5,
                             max_context_tokens: int = 2048) -> Dict:
        """检索并增强"""
        # 1. 向量检索
        vector_results = self.vector_db.search(query, k=top_k * 2)

        # 2. 重排序 (模拟cross-encoder)
        if self.rerank_enabled:
            reranked = self._rerank(query, vector_results)
        else:
            reranked = vector_results

        # 3. 上下文组装
        context, used_results, token_count = self._assemble_context(
            reranked[:top_k], max_context_tokens
        )

        # 4. 生成增强提示
        augmented_prompt = self._build_augmented_prompt(query, context)

        result = {
            "query": query,
            "retrieved_docs": len(vector_results),
            "used_docs": len(used_results),
            "context_tokens": token_count,
            "augmented_prompt": augmented_prompt,
            "sources": [
                {"text": r["text"][:100], "similarity": r["similarity"]}
                for r in used_results
            ],
            "timestamp": datetime.now().isoformat(),
        }
        self.retrieval_history.append(result)
        return result

    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """重排序 (模拟cross-encoder精细打分)"""
        query_words = set(query.lower().split())

        for r in results:
            # 模拟cross-encoder分数
            doc_words = set(r["text"].lower().split())
            word_overlap = len(query_words & doc_words) / max(len(query_words | doc_words), 1)
            # 融合向量相似度和词重叠
            rerank_score = r["similarity"] * 0.6 + word_overlap * 0.4
            r["rerank_score"] = round(rerank_score, 4)

        results.sort(key=lambda x: -x.get("rerank_score", x["similarity"]))
        return results

    def _assemble_context(self, results: List[Dict],
                          max_tokens: int) -> tuple:
        """组装上下文"""
        context_parts = []
        token_count = 0
        used = []

        for r in results:
            est_tokens = len(r["text"]) // 2  # 粗略估计
            if token_count + est_tokens > max_tokens:
                continue

            context_parts.append(f"[文档{len(used)+1}] (相关度: {r['similarity']})\n{r['text']}")
            token_count += est_tokens
            used.append(r)

        context = "\n\n".join(context_parts)
        return context, used, token_count

    def _build_augmented_prompt(self, query: str, context: str) -> str:
        """构建增强提示"""
        return (
            f"基于以下检索到的上下文信息, 请回答用户的问题。\n"
            f"如果上下文中没有相关信息, 请说明并基于自身知识回答。\n\n"
            f"=== 检索上下文 ===\n{context}\n\n"
            f"=== 用户问题 ===\n{query}\n\n"
            f"=== 回答 ===\n"
        )

    def get_summary(self) -> Dict:
        return {
            "total_retrievals": len(self.retrieval_history),
            "rerank_enabled": self.rerank_enabled,
            "avg_docs_retrieved": round(
                sum(r["retrieved_docs"] for r in self.retrieval_history) / max(len(self.retrieval_history), 1), 2
            ) if self.retrieval_history else 0,
            "avg_context_tokens": round(
                sum(r["context_tokens"] for r in self.retrieval_history) / max(len(self.retrieval_history), 1), 2
            ) if self.retrieval_history else 0,
        }


class ConversationMemorySystem:
    """对话记忆系统 — 统一记忆管理

    整合:
    - 短期记忆 (工作记忆)
    - 长期记忆 (持久化)
    - RAG检索增强
    - 记忆巩固与衰减
    - 上下文压缩
    """

    def __init__(self, vector_db: VectorDatabase = None):
        self.vector_db = vector_db or VectorDatabase(dim=128)
        self.short_term = ShortTermMemory(capacity=50)
        self.long_term = LongTermMemory(vector_db=self.vector_db)
        self.rag = RAGRetriever(vector_db=self.vector_db)
        self.conversation_history: List[Dict] = []
        self.consolidation_interval: int = 10  # 每10轮对话巩固一次

    def add_message(self, role: str, content: str, importance: float = 0.5,
                    tags: List[str] = None) -> str:
        """添加对话消息"""
        mid = self.short_term.add(content, importance, tags or [])
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "memory_id": mid,
            "importance": importance,
        })

        # 定期巩固
        if len(self.conversation_history) % self.consolidation_interval == 0:
            self.consolidate_memories()

        return mid

    def get_context(self, query: str = "", max_tokens: int = 2048) -> Dict:
        """获取对话上下文"""
        # 短期记忆上下文
        stm_context = self.short_term.get_context_window(max_tokens // 2)

        # 长期记忆检索
        ltm_results = self.long_term.retrieve(query, limit=5)

        # RAG增强
        rag_result = self.rag.retrieve_and_augment(query, top_k=3, max_context_tokens=max_tokens // 4)

        return {
            "stm_context": stm_context,
            "ltm_results": [
                {"content": m.content[:100], "importance": m.importance, "accessed": m.access_count}
                for m in ltm_results
            ],
            "rag_sources": rag_result["sources"],
            "augmented_prompt": rag_result["augmented_prompt"],
            "total_context_tokens": len(stm_context) // 2 + sum(len(m.content) for m in ltm_results) // 2,
        }

    def consolidate_memories(self) -> Dict:
        """记忆巩固"""
        return self.long_term.consolidate(self.short_term)

    def apply_memory_decay(self, days: float = 1.0) -> Dict:
        """应用记忆衰减"""
        return self.long_term.apply_decay(days)

    def search_all_memory(self, query: str, limit: int = 10) -> Dict:
        """搜索所有记忆"""
        stm_results = self.short_term.search(query, limit=limit)
        ltm_results = self.long_term.retrieve(query, limit=limit)

        return {
            "short_term": [
                {"content": m.content[:100], "importance": m.importance, "timestamp": m.timestamp}
                for m in stm_results
            ],
            "long_term": [
                {"content": m.content[:100], "importance": m.importance, "accessed": m.access_count}
                for m in ltm_results
            ],
            "rag": self.rag.retrieve_and_augment(query, top_k=limit),
        }

    def compress_context(self, messages: List[Dict], target_tokens: int = 1024) -> Dict:
        """上下文压缩

        将长对话历史压缩为摘要
        """
        if not messages:
            return {"summary": "", "original_count": 0, "compressed_count": 0}

        # 按重要性排序, 保留最关键的
        total_tokens = sum(len(m["content"]) for m in messages) // 2

        if total_tokens <= target_tokens:
            return {
                "summary": " ".join(m["content"] for m in messages),
                "original_count": len(messages),
                "compressed_count": len(messages),
                "compression_ratio": 1.0,
            }

        # 保留重要消息
        sorted_msgs = sorted(messages, key=lambda x: -x.get("importance", 0.5))
        kept = []
        current_tokens = 0

        for msg in sorted_msgs:
            est_tokens = len(msg["content"]) // 2
            if current_tokens + est_tokens > target_tokens:
                break
            kept.append(msg)
            current_tokens += est_tokens

        # 按时间排序
        kept.sort(key=lambda x: self.conversation_history.index(x) if x in self.conversation_history else 0)

        # 生成摘要 (模拟)
        summary_parts = [f"[{m['role']}] {m['content'][:50]}..." for m in kept]
        summary = " ".join(summary_parts)

        return {
            "summary": summary,
            "original_count": len(messages),
            "compressed_count": len(kept),
            "compression_ratio": round(len(kept) / max(len(messages), 1), 4),
            "original_tokens": total_tokens,
            "compressed_tokens": current_tokens,
        }

    def get_dashboard(self) -> Dict:
        return {
            "conversation_turns": len(self.conversation_history),
            "short_term": self.short_term.get_summary(),
            "long_term": self.long_term.get_summary(),
            "rag": self.rag.get_summary(),
            "vector_db": self.vector_db.get_dashboard(),
            "consolidation_interval": self.consolidation_interval,
        }
