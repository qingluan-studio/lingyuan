
# ============================================================
# LINGYUAN MODEL - PART 26
# 自适应学习系统 (Adaptive Learning System)
#
# 个性化学习路径规划与自适应难度调整
# 基于知识空间理论 + 间隔重复 + 学习者画像
#
# 核心组件:
# - 学习者画像: 认知风格、知识水平、学习偏好建模
# - 知识状态追踪: 知识空间理论, 前驱/后继关系
# - 学习路径生成: 拓扑排序 + 难度梯度 + 个性化偏好
# - 难度自适应: 项目反应理论(IRT), 实时调整
# - 间隔重复: 艾宾浩斯遗忘曲线, 最优复习时机
# - 学习分析: 多维度分析, 可视化报告
# ============================================================

import math
import time
import json
import random
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Callable


# ============================================================
# 枚举定义
# ============================================================

class CognitiveStyle(Enum):
    """认知风格 (VARK模型)"""
    VISUAL = "visual"        # 视觉型
    AUDITORY = "auditory"    # 听觉型
    READING = "reading"      # 读写型
    KINESTHETIC = "kinesthetic"  # 动觉型


class DifficultyLevel(Enum):
    """难度等级"""
    BEGINNER = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


class MasteryLevel(Enum):
    """掌握程度"""
    UNKNOWN = 0.0
    INTRODUCED = 0.2
    FAMILIAR = 0.4
    PROFICIENT = 0.6
    MASTERED = 0.8
    EXPERT = 1.0


class LearningObjective(Enum):
    """学习目标类型"""
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class ContentFormat(Enum):
    """内容格式"""
    TEXT = "text"
    VIDEO = "video"
    INTERACTIVE = "interactive"
    EXERCISE = "exercise"
    QUIZ = "quiz"
    PROJECT = "project"


# ============================================================
# 知识点与知识空间
# ============================================================

@dataclass
class KnowledgePoint:
    """知识点

    知识空间理论中的原子知识单元
    """
    kp_id: str
    name: str
    description: str = ""
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    estimated_time: float = 30.0  # 预估学习时间(分钟)
    prerequisites: List[str] = field(default_factory=list)  # 前驱知识点ID
    objectives: List[LearningObjective] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    content_formats: List[ContentFormat] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeRelation:
    """知识点间关系"""
    source: str  # 前驱知识点ID
    target: str  # 后继知识点ID
    relation_type: str = "prerequisite"  # prerequisite, related, extends
    strength: float = 1.0  # 关联强度


class KnowledgeSpace:
    """知识空间

    基于知识空间理论(KST), 管理知识点及其依赖关系

    核心概念:
    - 知识状态: 学习者已掌握的知识点集合
    - 知识空间: 所有可能的合法知识状态集合
    - 外延: 从当前状态可达的知识点

    一个知识状态是"合法的"当且仅当它满足所有前驱关系:
    如果掌握了知识点X, 则必须掌握X的所有前驱知识点
    """

    def __init__(self):
        self.points: Dict[str, KnowledgePoint] = {}
        self.relations: Dict[str, List[KnowledgeRelation]] = defaultdict(list)
        self._adjacency: Dict[str, List[str]] = defaultdict(list)  # 前驱→后继
        self._reverse_adj: Dict[str, List[str]] = defaultdict(list)  # 后继→前驱

    def add_point(self, point: KnowledgePoint) -> None:
        """添加知识点"""
        self.points[point.kp_id] = point
        for prereq in point.prerequisites:
            self.add_relation(prereq, point.kp_id)

    def add_relation(self, source: str, target: str,
                     relation_type: str = "prerequisite",
                     strength: float = 1.0) -> None:
        """添加知识点关系"""
        rel = KnowledgeRelation(source, target, relation_type, strength)
        self.relations[source].append(rel)
        self._adjacency[source].append(target)
        self._reverse_adj[target].append(source)

    def get_prerequisites(self, kp_id: str) -> Set[str]:
        """获取直接前驱"""
        return set(self._reverse_adj.get(kp_id, []))

    def get_successors(self, kp_id: str) -> Set[str]:
        """获取直接后继"""
        return set(self._adjacency.get(kp_id, []))

    def get_all_prerequisites(self, kp_id: str) -> Set[str]:
        """获取所有前驱 (递归)"""
        visited = set()
        queue = deque([kp_id])
        while queue:
            current = queue.popleft()
            for prereq in self._reverse_adj.get(current, []):
                if prereq not in visited:
                    visited.add(prereq)
                    queue.append(prereq)
        return visited

    def is_learnable(self, kp_id: str, mastered: Set[str]) -> bool:
        """检查知识点是否可学习 (所有前驱已掌握)"""
        prereqs = self.get_prerequisites(kp_id)
        return prereqs.issubset(mastered)

    def get_frontier(self, mastered: Set[str]) -> List[str]:
        """获取学习前沿 — 可学但未学的知识点

        前沿知识点: 所有前驱已掌握, 但自身未掌握
        """
        frontier = []
        for kp_id in self.points:
            if kp_id not in mastered and self.is_learnable(kp_id, mastered):
                frontier.append(kp_id)
        return frontier

    def topological_sort(self) -> List[str]:
        """拓扑排序 — 知识点的学习顺序"""
        in_degree = {kp_id: len(self._reverse_adj.get(kp_id, []))
                     for kp_id in self.points}
        queue = deque([kp for kp, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)
            for successor in self._adjacency.get(current, []):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        return result

    def shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """最短学习路径 (BFS)"""
        if start == end:
            return [start]

        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            for successor in self._adjacency.get(current, []):
                if successor == end:
                    return path + [successor]
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, path + [successor]))

        return None


# ============================================================
# 学习者画像
# ============================================================

@dataclass
class LearnerProfile:
    """学习者画像

    多维度建模学习者特征:
    - 认知风格: VARK模型
    - 知识水平: 各知识点掌握程度
    - 学习偏好: 内容格式偏好
    - 行为模式: 学习时间、频率、节奏
    - 情感状态: 动机、挫败感、信心
    """

    learner_id: str
    name: str = ""
    cognitive_style: CognitiveStyle = CognitiveStyle.VISUAL
    created_at: float = field(default_factory=time.time)

    # 知识掌握: kp_id -> mastery (0.0~1.0)
    knowledge_mastery: Dict[str, float] = field(default_factory=dict)

    # 学习偏好: 内容格式 -> 偏好权重
    format_preferences: Dict[ContentFormat, float] = field(default_factory=dict)

    # 行为统计
    total_study_time: float = 0.0  # 总学习时间(分钟)
    total_sessions: int = 0
    total_items_attempted: int = 0
    total_items_correct: int = 0
    avg_session_length: float = 0.0  # 平均会话长度(分钟)
    study_streak: int = 0  # 连续学习天数
    last_study_time: float = 0.0

    # IRT参数 (项目反应理论)
    # 能力θ值: 标准正态分布, 通常[-3, 3]
    ability_theta: float = 0.0

    # 情感状态
    motivation: float = 0.5  # 0~1
    frustration: float = 0.0  # 0~1
    confidence: float = 0.5  # 0~1

    # 学习历史
    history: List[Dict[str, Any]] = field(default_factory=list)

    # 间隔重复队列
    review_queue: Dict[str, float] = field(default_factory=dict)  # kp_id -> next_review_time

    @property
    def accuracy(self) -> float:
        """正确率"""
        if self.total_items_attempted == 0:
            return 0.0
        return self.total_items_correct / self.total_items_attempted

    @property
    def mastered_count(self) -> int:
        """已掌握知识点数"""
        return sum(1 for m in self.knowledge_mastery.values()
                   if m >= MasteryLevel.PROFICIENT.value)

    def update_mastery(self, kp_id: str, delta: float,
                       decay: float = 0.1) -> None:
        """更新知识掌握度

        使用指数移动平均(EMA)平滑更新
        """
        current = self.knowledge_mastery.get(kp_id, 0.0)
        # EMA: new = old + learning_rate * (target - old)
        # 这里delta是增量, 不是目标值
        new_mastery = current + delta * (1.0 - decay)
        new_mastery = max(0.0, min(1.0, new_mastery))
        self.knowledge_mastery[kp_id] = new_mastery

        # 更新能力θ值
        self._update_theta()

    def _update_theta(self) -> None:
        """更新IRT能力θ值 (基于平均掌握度)"""
        if self.knowledge_mastery:
            avg_mastery = sum(self.knowledge_mastery.values()) / len(self.knowledge_mastery)
            # 映射 [0, 1] -> [-3, 3]
            self.ability_theta = (avg_mastery - 0.5) * 6.0

    def update_emotion(self, success: bool) -> None:
        """更新情感状态"""
        if success:
            self.motivation = min(1.0, self.motivation + 0.05)
            self.confidence = min(1.0, self.confidence + 0.03)
            self.frustration = max(0.0, self.frustration - 0.1)
        else:
            self.frustration = min(1.0, self.frustration + 0.08)
            self.confidence = max(0.0, self.confidence - 0.05)
            self.motivation = max(0.0, self.motivation - 0.02)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "name": self.name,
            "cognitive_style": self.cognitive_style.value,
            "knowledge_mastery": self.knowledge_mastery,
            "format_preferences": {k.value: v for k, v in self.format_preferences.items()},
            "total_study_time": self.total_study_time,
            "total_sessions": self.total_sessions,
            "accuracy": self.accuracy,
            "mastered_count": self.mastered_count,
            "ability_theta": self.ability_theta,
            "motivation": self.motivation,
            "frustration": self.frustration,
            "confidence": self.confidence,
            "study_streak": self.study_streak,
        }


# ============================================================
# 项目反应理论 (IRT) 难度自适应
# ============================================================

class IRTModel:
    """三参数逻辑斯蒂模型 (3PL)

    P(θ) = c + (1 - c) / (1 + exp(-a * (θ - b)))

    参数:
    - a: 区分度 (discrimination)
    - b: 难度 (difficulty)
    - c: 猜测系数 (guessing)
    - θ: 学习者能力

    用于:
    1. 预测答题正确率
    2. 选择最合适难度的题目
    3. 估计学习者能力
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, c: float = 0.0):
        self.a = a
        self.b = b
        self.c = c

    def probability(self, theta: float) -> float:
        """计算答对概率 P(θ)"""
        return self.c + (1.0 - self.c) / (1.0 + math.exp(-self.a * (theta - self.b)))

    def information(self, theta: float) -> float:
        """计算信息量 I(θ)

        信息量越大, 该题目对能力估计的贡献越大
        最大信息量出现在 θ = b 时
        """
        p = self.probability(theta)
        q = 1.0 - p
        if p == 0 or q == 0:
            return 0.0
        num = self.a ** 2 * (p - self.c) ** 2
        den = (1.0 - self.c) ** 2 * p * q
        if den == 0:
            return 0.0
        return num / den

    @staticmethod
    def estimate_ability(responses: List[Tuple[float, bool]],
                         max_iter: int = 50,
                         tol: float = 1e-4) -> float:
        """估计学习者能力 (最大似然估计)

        Args:
            responses: [(难度b, 是否答对), ...]
            max_iter: 最大迭代次数
            tol: 收敛阈值

        Returns:
            估计的能力θ值
        """
        theta = 0.0
        for _ in range(max_iter):
            old_theta = theta
            numerator = 0.0
            denominator = 0.0
            for b, correct in responses:
                model = IRTModel(a=1.0, b=b, c=0.0)
                p = model.probability(theta)
                q = 1.0 - p
                if p == 0 or q == 0:
                    continue
                info = model.information(theta)
                numerator += (1.0 if correct else 0.0 - p) * model.a
                denominator += info

            if abs(denominator) < 1e-10:
                break

            theta += numerator / denominator
            theta = max(-3.0, min(3.0, theta))

            if abs(theta - old_theta) < tol:
                break

        return theta


class DifficultyAdapter:
    """难度自适应器

    基于IRT模型, 实时调整学习内容难度

    策略:
    - 答对 → 提高难度
    - 答错 → 降低难度
    - 连续答对 → 加速提升
    - 连续答错 → 加速降低

    目标: 保持正确率在70-85%之间 (最近发展区)
    """

    def __init__(self, target_accuracy: float = 0.75):
        self.target_accuracy = target_accuracy
        self.irt = IRTModel()
        self.recent_results: deque = deque(maxlen=10)
        self.current_difficulty: DifficultyLevel = DifficultyLevel.MEDIUM

    def adjust(self, correct: bool, response_time: float = 0.0) -> DifficultyLevel:
        """根据答题结果调整难度

        Args:
            correct: 是否答对
            response_time: 响应时间(秒), 用于辅助判断

        Returns:
            调整后的难度等级
        """
        self.recent_results.append(correct)

        # 连续正确/错误数
        streak = 0
        positive = correct
        for r in reversed(self.recent_results):
            if r == positive:
                streak += 1
            else:
                break

        current_val = self.current_difficulty.value

        if correct:
            if streak >= 3:
                # 连续答对3次以上, 跳级提升
                current_val = min(5, current_val + 2)
            else:
                current_val = min(5, current_val + 1)
        else:
            if streak >= 3:
                # 连续答错3次以上, 跳级降低
                current_val = max(1, current_val - 2)
            else:
                current_val = max(1, current_val - 1)

        # 检查最近10题的正确率
        if len(self.recent_results) >= 5:
            recent_acc = sum(self.recent_results) / len(self.recent_results)
            if recent_acc > 0.9:
                current_val = min(5, current_val + 1)
            elif recent_acc < 0.4:
                current_val = max(1, current_val - 1)

        self.current_difficulty = DifficultyLevel(current_val)
        return self.current_difficulty

    def select_difficulty_for_theta(self, theta: float) -> DifficultyLevel:
        """根据能力θ值选择最合适难度"""
        # θ ∈ [-3, 3] → 难度 [1, 5]
        level = int(round((theta + 3.0) / 6.0 * 4.0 + 1.0))
        level = max(1, min(5, level))
        self.current_difficulty = DifficultyLevel(level)
        return self.current_difficulty

    def get_optimal_item_difficulty(self, theta: float) -> float:
        """获取最优题目难度b值 (信息量最大化)"""
        # 3PL模型中, 最大信息量出现在 θ ≈ b 附近
        return theta


# ============================================================
# 间隔重复系统 (Spaced Repetition)
# ============================================================

class SpacedRepetitionSystem:
    """间隔重复系统

    基于艾宾浩斯遗忘曲线和SM-2算法

    遗忘曲线: R = exp(-t / S)
    - R: 记忆保持率
    - t: 经过的时间
    - S: 记忆强度

    SM-2算法:
    - 根据回忆质量(0-5分)调整间隔
    - 质量越高, 下次复习间隔越长
    - 质量低于3分, 重置间隔

    最优复习时机: 当记忆保持率降到约70%时
    """

    def __init__(self, target_retention: float = 0.7):
        self.target_retention = target_retention
        # kp_id -> {interval, repetitions, ease_factor, last_review}
        self.schedule: Dict[str, Dict[str, Any]] = {}

    def review(self, kp_id: str, quality: int,
               current_time: Optional[float] = None) -> Dict[str, Any]:
        """记录复习并计算下次复习时间

        Args:
            kp_id: 知识点ID
            quality: 回忆质量 0-5
                0-2: 完全忘记
                3: 勉强回忆, 有困难
                4: 正确回忆, 有一些迟疑
                5: 完美回忆
            current_time: 当前时间戳

        Returns:
            {interval, repetitions, ease_factor, next_review}
        """
        current_time = current_time or time.time()
        quality = max(0, min(5, quality))

        if kp_id not in self.schedule:
            self.schedule[kp_id] = {
                "interval": 1.0,  # 天
                "repetitions": 0,
                "ease_factor": 2.5,
                "last_review": current_time,
            }

        entry = self.schedule[kp_id]

        # SM-2算法
        if quality < 3:
            # 回忆失败, 重置
            entry["repetitions"] = 0
            entry["interval"] = 1.0
        else:
            entry["repetitions"] += 1
            if entry["repetitions"] == 1:
                entry["interval"] = 1.0
            elif entry["repetitions"] == 2:
                entry["interval"] = 6.0
            else:
                entry["interval"] *= entry["ease_factor"]

        # 更新ease_factor
        old_ef = entry["ease_factor"]
        new_ef = old_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        entry["ease_factor"] = max(1.3, new_ef)

        entry["last_review"] = current_time

        # 计算下次复习时间
        next_review = current_time + entry["interval"] * 86400.0  # 天→秒

        return {
            "interval_days": entry["interval"],
            "repetitions": entry["repetitions"],
            "ease_factor": entry["ease_factor"],
            "next_review": next_review,
        }

    def get_due_items(self, current_time: Optional[float] = None) -> List[str]:
        """获取到期需要复习的知识点"""
        current_time = current_time or time.time()
        due = []
        for kp_id, entry in self.schedule.items():
            next_review = entry["last_review"] + entry["interval"] * 86400.0
            if current_time >= next_review:
                due.append(kp_id)
        return due

    def get_retention(self, kp_id: str,
                      current_time: Optional[float] = None) -> float:
        """计算当前记忆保持率

        R = exp(-t / S)
        """
        current_time = current_time or time.time()
        if kp_id not in self.schedule:
            return 0.0

        entry = self.schedule[kp_id]
        elapsed = (current_time - entry["last_review"]) / 86400.0  # 秒→天
        # 记忆强度S ≈ interval * ease_factor
        S = max(0.1, entry["interval"] * entry["ease_factor"])
        retention = math.exp(-elapsed / S)
        return max(0.0, min(1.0, retention))

    def get_optimal_review_time(self, kp_id: str) -> Optional[float]:
        """计算最优复习时间 (保持率降到目标值时)"""
        if kp_id not in self.schedule:
            return None

        entry = self.schedule[kp_id]
        S = max(0.1, entry["interval"] * entry["ease_factor"])
        # R = exp(-t/S) = target_retention
        # t = -S * ln(target_retention)
        t_days = -S * math.log(self.target_retention)
        return entry["last_review"] + t_days * 86400.0


# ============================================================
# 学习路径生成器
# ============================================================

class LearningPathGenerator:
    """学习路径生成器

    基于知识空间理论生成个性化学习路径

    算法:
    1. 获取学习前沿 (可学但未学的知识点)
    2. 根据学习目标筛选
    3. 根据难度梯度排序
    4. 考虑学习者偏好和认知风格
    5. 生成最优学习路径

    路径优化目标:
    - 最小化总学习时间
    - 最大化学习收益
    - 保持难度梯度平滑 (最近发展区)
    - 匹配学习者偏好
    """

    def __init__(self, knowledge_space: KnowledgeSpace):
        self.ks = knowledge_space

    def generate_path(self, profile: LearnerProfile,
                      target_kp: Optional[str] = None,
                      max_length: int = 20) -> List[str]:
        """生成个性化学习路径

        Args:
            profile: 学习者画像
            target_kp: 目标知识点 (如果指定, 生成到达目标的路径)
            max_length: 路径最大长度

        Returns:
            知识点ID列表 (学习顺序)
        """
        mastered = {kp for kp, m in profile.knowledge_mastery.items()
                    if m >= MasteryLevel.PROFICIENT.value}

        if target_kp:
            return self._path_to_target(profile, mastered, target_kp, max_length)
        else:
            return self._path_frontier(profile, mastered, max_length)

    def _path_to_target(self, profile: LearnerProfile,
                        mastered: Set[str], target_kp: str,
                        max_length: int) -> List[str]:
        """生成到达目标知识点的路径"""
        if target_kp in mastered:
            return []

        # 获取所有前驱
        all_prereqs = self.ks.get_all_prerequisites(target_kp)
        needed = all_prereqs | {target_kp}
        unlearned = needed - mastered

        if not unlearned:
            return [target_kp]

        # 拓扑排序
        topo = self.ks.topological_sort()
        path = [kp for kp in topo if kp in unlearned]

        # 按难度和偏好排序
        path = self._sort_by_difficulty_and_preference(path, profile)

        return path[:max_length]

    def _path_frontier(self, profile: LearnerProfile,
                       mastered: Set[str], max_length: int) -> List[str]:
        """基于学习前沿生成路径"""
        frontier = self.ks.get_frontier(mastered)

        if not frontier:
            return []

        # 按难度和偏好排序
        path = self._sort_by_difficulty_and_preference(frontier, profile)

        # 逐步扩展: 学完一个知识点后, 获取新的前沿
        result = []
        current_mastered = set(mastered)

        while len(result) < max_length and path:
            # 取最优的知识点
            best = path.pop(0)
            result.append(best)
            current_mastered.add(best)

            # 获取新的前沿
            new_frontier = self.ks.get_frontier(current_mastered)
            # 排除已在路径中的
            new_frontier = [kp for kp in new_frontier if kp not in result]
            # 合并并排序
            remaining = path + new_frontier
            path = self._sort_by_difficulty_and_preference(remaining, profile)

        return result

    def _sort_by_difficulty_and_preference(self, kp_ids: List[str],
                                           profile: LearnerProfile) -> List[str]:
        """按难度和偏好排序

        排序策略:
        1. 难度匹配: 选择略高于当前水平的知识点 (最近发展区)
        2. 格式偏好: 优先选择偏好的内容格式
        3. 预估时间: 短时间优先 (快速正反馈)
        """

        def score_kp(kp_id: str) -> float:
            kp = self.ks.points.get(kp_id)
            if not kp:
                return float('inf')

            # 难度匹配分: 越接近当前能力越好
            target_difficulty = profile.ability_theta
            kp_difficulty = (kp.difficulty.value - 3) * 1.0  # [-2, 2]
            diff_penalty = abs(kp_difficulty - target_difficulty)

            # 格式偏好分
            format_score = 0.0
            if kp.content_formats and profile.format_preferences:
                for fmt in kp.content_formats:
                    format_score += profile.format_preferences.get(fmt, 0.0)
                format_score /= len(kp.content_formats)

            # 时间分: 短时间优先
            time_score = 1.0 / (1.0 + kp.estimated_time / 60.0)

            # 情感调节: 挫败感高时, 降低难度权重
            frustration_penalty = profile.frustration * diff_penalty * 2.0

            # 综合分 (越低越好)
            total = diff_penalty + frustration_penalty - format_score * 0.5 - time_score * 0.3
            return total

        return sorted(kp_ids, key=score_kp)

    def generate_study_plan(self, profile: LearnerProfile,
                            available_time: float = 60.0,
                            target_kp: Optional[str] = None) -> List[Dict[str, Any]]:
        """生成学习计划

        Args:
            profile: 学习者画像
            available_time: 可用时间(分钟)
            target_kp: 目标知识点

        Returns:
            学习计划 [{kp_id, name, estimated_time, difficulty, ...}, ...]
        """
        path = self.generate_path(profile, target_kp)

        plan = []
        remaining_time = available_time

        for kp_id in path:
            kp = self.ks.points.get(kp_id)
            if not kp:
                continue
            if kp.estimated_time > remaining_time:
                break

            plan.append({
                "kp_id": kp_id,
                "name": kp.name,
                "estimated_time": kp.estimated_time,
                "difficulty": kp.difficulty.name,
                "objectives": [obj.value for obj in kp.objectives],
                "content_formats": [fmt.value for fmt in kp.content_formats],
            })
            remaining_time -= kp.estimated_time

        return plan


# ============================================================
# 内容推荐器
# ============================================================

class ContentRecommender:
    """个性化内容推荐器

    基于协同过滤 + 内容过滤 + 知识图谱

    推荐策略:
    1. 基于知识状态: 推荐前沿知识点
    2. 基于认知风格: 匹配内容格式
    3. 基于难度: 最近发展区
    4. 基于间隔重复: 到期复习项
    5. 基于兴趣: 标签匹配
    """

    def __init__(self, knowledge_space: KnowledgeSpace,
                 srs: SpacedRepetitionSystem):
        self.ks = knowledge_space
        self.srs = srs
        self.content_pool: Dict[str, Dict[str, Any]] = {}  # content_id -> content

    def add_content(self, content_id: str, kp_id: str,
                    format: ContentFormat, title: str = "",
                    body: str = "", tags: Optional[List[str]] = None) -> None:
        """添加学习内容"""
        self.content_pool[content_id] = {
            "content_id": content_id,
            "kp_id": kp_id,
            "format": format,
            "title": title,
            "body": body,
            "tags": tags or [],
        }

    def recommend(self, profile: LearnerProfile,
                  top_k: int = 5,
                  include_review: bool = True) -> List[Dict[str, Any]]:
        """生成推荐内容列表

        Args:
            profile: 学习者画像
            top_k: 推荐数量
            include_review: 是否包含复习内容

        Returns:
            推荐内容列表
        """
        recommendations = []

        # 1. 到期复习项
        if include_review:
            due_items = self.srs.get_due_items()
            for kp_id in due_items:
                retention = self.srs.get_retention(kp_id)
                contents = self._get_contents_for_kp(kp_id, profile)
                for content in contents[:1]:  # 每个知识点推荐1个内容
                    recommendations.append({
                        **content,
                        "reason": "复习巩固",
                        "priority": 1.0 - retention,  # 保持率越低优先级越高
                        "retention": retention,
                    })

        # 2. 新知识学习
        mastered = {kp for kp, m in profile.knowledge_mastery.items()
                    if m >= MasteryLevel.PROFICIENT.value}
        frontier = self.ks.get_frontier(mastered)

        for kp_id in frontier:
            contents = self._get_contents_for_kp(kp_id, profile)
            for content in contents[:2]:  # 每个知识点推荐2个内容
                kp = self.ks.points.get(kp_id)
                difficulty_match = self._difficulty_match(kp, profile)
                recommendations.append({
                    **content,
                    "reason": "新知识学习",
                    "priority": 0.5 + difficulty_match * 0.3,
                    "difficulty": kp.difficulty.name if kp else "UNKNOWN",
                })

        # 3. 按优先级排序
        recommendations.sort(key=lambda x: x.get("priority", 0.0), reverse=True)

        return recommendations[:top_k]

    def _get_contents_for_kp(self, kp_id: str,
                             profile: LearnerProfile) -> List[Dict[str, Any]]:
        """获取知识点的学习内容 (按偏好排序)"""
        contents = [c for c in self.content_pool.values()
                    if c["kp_id"] == kp_id]

        # 按认知风格偏好排序
        def content_score(content):
            fmt = content["format"]
            pref = profile.format_preferences.get(fmt, 0.0)
            return pref

        contents.sort(key=content_score, reverse=True)
        return contents

    def _difficulty_match(self, kp: Optional[KnowledgePoint],
                          profile: LearnerProfile) -> float:
        """计算难度匹配度 (0~1)"""
        if not kp:
            return 0.0
        target_level = profile.ability_theta
        kp_level = (kp.difficulty.value - 3) * 1.0
        diff = abs(kp_level - target_level)
        return max(0.0, 1.0 - diff / 4.0)


# ============================================================
# 学习分析引擎
# ============================================================

class LearningAnalytics:
    """学习分析引擎

    多维度分析学习行为, 生成洞察报告

    分析维度:
    1. 知识图谱分析: 覆盖率、缺口、优势
    2. 学习效率: 时间投入 vs 产出
    3. 进步趋势: 掌握度变化曲线
    4. 行为模式: 最佳学习时段、节奏
    5. 风险预警: 挫败感、停滞、遗忘
    """

    def __init__(self, knowledge_space: KnowledgeSpace):
        self.ks = knowledge_space

    def analyze(self, profile: LearnerProfile) -> Dict[str, Any]:
        """生成完整学习分析报告"""
        return {
            "knowledge_analysis": self._analyze_knowledge(profile),
            "efficiency_analysis": self._analyze_efficiency(profile),
            "progress_analysis": self._analyze_progress(profile),
            "behavior_analysis": self._analyze_behavior(profile),
            "risk_assessment": self._assess_risk(profile),
            "recommendations": self._generate_insights(profile),
        }

    def _analyze_knowledge(self, profile: LearnerProfile) -> Dict[str, Any]:
        """知识图谱分析"""
        total_kps = len(self.ks.points)
        mastered = profile.mastered_count
        learning = sum(1 for m in profile.knowledge_mastery.values()
                       if 0 < m < MasteryLevel.PROFICIENT.value)

        # 知识缺口: 前驱已掌握但自身未掌握
        mastered_set = {kp for kp, m in profile.knowledge_mastery.items()
                        if m >= MasteryLevel.PROFICIENT.value}
        gaps = []
        for kp_id in self.ks.points:
            if kp_id not in mastered_set:
                prereqs = self.ks.get_prerequisites(kp_id)
                if prereqs.issubset(mastered_set):
                    gaps.append(kp_id)

        # 优势领域 (掌握度高的标签)
        tag_mastery = defaultdict(list)
        for kp_id, mastery in profile.knowledge_mastery.items():
            kp = self.ks.points.get(kp_id)
            if kp:
                for tag in kp.tags:
                    tag_mastery[tag].append(mastery)

        strengths = []
        for tag, masteries in tag_mastery.items():
            avg = sum(masteries) / len(masteries)
            if avg >= MasteryLevel.PROFICIENT.value:
                strengths.append({"tag": tag, "avg_mastery": avg})

        strengths.sort(key=lambda x: x["avg_mastery"], reverse=True)

        return {
            "total_knowledge_points": total_kps,
            "mastered_count": mastered,
            "learning_count": learning,
            "coverage_rate": mastered / max(1, total_kps),
            "knowledge_gaps": gaps,
            "strengths": strengths[:5],
        }

    def _analyze_efficiency(self, profile: LearnerProfile) -> Dict[str, Any]:
        """学习效率分析"""
        # 掌握知识点数 / 总学习时间
        mastery_per_hour = (profile.mastered_count /
                            max(0.1, profile.total_study_time / 60.0))

        # 正确率
        accuracy = profile.accuracy

        # 平均每知识点学习时间
        avg_time_per_kp = (profile.total_study_time /
                           max(1, len(profile.knowledge_mastery)))

        return {
            "mastery_per_hour": mastery_per_hour,
            "accuracy": accuracy,
            "avg_time_per_kp_minutes": avg_time_per_kp,
            "total_study_hours": profile.total_study_time / 60.0,
            "efficiency_score": mastery_per_hour * accuracy * 10,
        }

    def _analyze_progress(self, profile: LearnerProfile) -> Dict[str, Any]:
        """进步趋势分析"""
        if not profile.history:
            return {"trend": "no_data"}

        # 按时间排序的历史记录
        history = sorted(profile.history, key=lambda x: x.get("timestamp", 0))

        # 计算掌握度变化
        mastery_progression = []
        cumulative_mastered = 0
        for entry in history:
            if entry.get("mastered", False):
                cumulative_mastered += 1
            mastery_progression.append({
                "timestamp": entry.get("timestamp"),
                "cumulative_mastered": cumulative_mastered,
                "mastery": entry.get("mastery", 0.0),
            })

        # 计算学习速度 (最近7条 vs 之前7条)
        recent = mastery_progression[-7:]
        earlier = mastery_progression[:-7] if len(mastery_progression) > 7 else []

        recent_rate = sum(1 for r in recent if r.get("mastered")) / max(1, len(recent))
        earlier_rate = (sum(1 for r in earlier if r.get("mastered")) /
                        max(1, len(earlier))) if earlier else 0

        if recent_rate > earlier_rate:
            trend = "accelerating"
        elif recent_rate < earlier_rate:
            trend = "decelerating"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "recent_mastery_rate": recent_rate,
            "earlier_mastery_rate": earlier_rate,
            "progression_length": len(mastery_progression),
            "current_ability_theta": profile.ability_theta,
        }

    def _analyze_behavior(self, profile: LearnerProfile) -> Dict[str, Any]:
        """行为模式分析"""
        return {
            "total_sessions": profile.total_sessions,
            "avg_session_length": profile.avg_session_length,
            "study_streak_days": profile.study_streak,
            "preferred_format": self._get_preferred_format(profile),
            "cognitive_style": profile.cognitive_style.value,
            "motivation": profile.motivation,
            "confidence": profile.confidence,
        }

    def _get_preferred_format(self, profile: LearnerProfile) -> str:
        """获取偏好的内容格式"""
        if not profile.format_preferences:
            return "unknown"
        best = max(profile.format_preferences.items(), key=lambda x: x[1])
        return best[0].value

    def _assess_risk(self, profile: LearnerProfile) -> Dict[str, Any]:
        """风险预警"""
        risks = []

        # 高挫败感
        if profile.frustration > 0.6:
            risks.append({
                "type": "high_frustration",
                "severity": "high",
                "message": "挫败感较高, 建议降低难度或休息",
            })

        # 低动机
        if profile.motivation < 0.3:
            risks.append({
                "type": "low_motivation",
                "severity": "high",
                "message": "学习动机低, 建议设置小目标获得正反馈",
            })

        # 学习停滞
        if profile.study_streak == 0 and profile.total_sessions > 5:
            risks.append({
                "type": "learning_stagnation",
                "severity": "medium",
                "message": "最近没有学习活动, 建议恢复学习节奏",
            })

        # 低正确率
        if profile.total_items_attempted > 10 and profile.accuracy < 0.4:
            risks.append({
                "type": "low_accuracy",
                "severity": "high",
                "message": "正确率偏低, 当前难度可能过高",
            })

        # 能力θ值过低
        if profile.ability_theta < -1.5:
            risks.append({
                "type": "low_ability",
                "severity": "medium",
                "message": "整体能力偏低, 建议从基础知识点开始",
            })

        return {
            "risks": risks,
            "risk_count": len(risks),
            "high_risk_count": sum(1 for r in risks if r["severity"] == "high"),
            "overall_risk_level": "high" if any(r["severity"] == "high" for r in risks)
            else ("medium" if risks else "low"),
        }

    def _generate_insights(self, profile: LearnerProfile) -> List[str]:
        """生成学习建议"""
        insights = []
        analysis = self._analyze_knowledge(profile)
        risk = self._assess_risk(profile)

        # 基于知识缺口的建议
        if analysis["knowledge_gaps"]:
            insights.append(f"发现{len(analysis['knowledge_gaps'])}个知识缺口, "
                            f"建议优先补齐前置知识")

        # 基于优势的建议
        if analysis["strengths"]:
            top_strength = analysis["strengths"][0]
            insights.append(f"在'{top_strength['tag']}'领域表现突出, "
                            f"可尝试更高难度内容")

        # 基于风险的建议
        for r in risk["risks"]:
            insights.append(r["message"])

        # 基于效率的建议
        eff = self._analyze_efficiency(profile)
        if eff["accuracy"] > 0.85 and profile.total_items_attempted > 20:
            insights.append("正确率很高, 可以尝试挑战更高难度的内容")
        elif eff["accuracy"] < 0.5 and profile.total_items_attempted > 20:
            insights.append("正确率偏低, 建议复习基础知识或降低难度")

        if not insights:
            insights.append("学习状态良好, 继续保持!")

        return insights


# ============================================================
# 自适应学习系统 (主系统)
# ============================================================

class AdaptiveLearningSystem:
    """自适应学习系统 — 端到端

    整合所有组件, 提供完整的自适应学习体验:

    流程:
    1. 注册学习者 → 创建画像
    2. 评估初始水平 → 初始化知识状态
    3. 生成学习路径 → 个性化规划
    4. 执行学习活动 → 自适应难度调整
    5. 记录学习行为 → 更新画像
    6. 间隔重复复习 → 巩固记忆
    7. 学习分析报告 → 洞察与建议
    """

    def __init__(self, knowledge_space: Optional[KnowledgeSpace] = None):
        self.ks = knowledge_space or KnowledgeSpace()
        self.srs = SpacedRepetitionSystem()
        self.difficulty_adapter = DifficultyAdapter()
        self.path_generator = LearningPathGenerator(self.ks)
        self.recommender = ContentRecommender(self.ks, self.srs)
        self.analytics = LearningAnalytics(self.ks)
        self.learners: Dict[str, LearnerProfile] = {}

        # 全局统计
        self.total_learners = 0
        self.total_learning_activities = 0

    def register_learner(self, learner_id: str, name: str = "",
                         cognitive_style: CognitiveStyle = CognitiveStyle.VISUAL) -> LearnerProfile:
        """注册新学习者"""
        profile = LearnerProfile(
            learner_id=learner_id,
            name=name,
            cognitive_style=cognitive_style,
        )
        # 初始化格式偏好
        style_prefs = {
            CognitiveStyle.VISUAL: {ContentFormat.VIDEO: 0.8, ContentFormat.INTERACTIVE: 0.6},
            CognitiveStyle.AUDITORY: {ContentFormat.VIDEO: 0.7, ContentFormat.TEXT: 0.4},
            CognitiveStyle.READING: {ContentFormat.TEXT: 0.9, ContentFormat.EXERCISE: 0.6},
            CognitiveStyle.KINESTHETIC: {ContentFormat.INTERACTIVE: 0.9, ContentFormat.PROJECT: 0.7},
        }
        profile.format_preferences = style_prefs.get(cognitive_style, {})

        self.learners[learner_id] = profile
        self.total_learners += 1
        return profile

    def assess_initial_level(self, learner_id: str,
                             assessments: List[Tuple[str, int]]) -> Dict[str, Any]:
        """初始水平评估

        Args:
            learner_id: 学习者ID
            assessments: [(知识点ID, 质量分0-5), ...]

        Returns:
            评估结果
        """
        profile = self.learners.get(learner_id)
        if not profile:
            return {"error": "学习者未注册"}

        responses = []
        for kp_id, quality in assessments:
            kp = self.ks.points.get(kp_id)
            if not kp:
                continue
            b = (kp.difficulty.value - 3) * 1.0  # 难度b值
            correct = quality >= 3
            responses.append((b, correct))

            # 更新掌握度
            mastery_delta = (quality - 3) * 0.15
            profile.update_mastery(kp_id, mastery_delta)

            # 更新间隔重复
            self.srs.review(kp_id, quality)

        # 估计能力θ值
        if responses:
            theta = IRTModel.estimate_ability(responses)
            profile.ability_theta = theta

        return {
            "ability_theta": profile.ability_theta,
            "mastered_count": profile.mastered_count,
            "initial_difficulty": self.difficulty_adapter.select_difficulty_for_theta(
                profile.ability_theta
            ).name,
        }

    def get_learning_path(self, learner_id: str,
                          target_kp: Optional[str] = None,
                          max_length: int = 20) -> List[Dict[str, Any]]:
        """获取个性化学习路径"""
        profile = self.learners.get(learner_id)
        if not profile:
            return []

        path = self.path_generator.generate_path(profile, target_kp, max_length)

        result = []
        for kp_id in path:
            kp = self.ks.points.get(kp_id)
            if kp:
                result.append({
                    "kp_id": kp_id,
                    "name": kp.name,
                    "difficulty": kp.difficulty.name,
                    "estimated_time": kp.estimated_time,
                    "objectives": [obj.value for obj in kp.objectives],
                    "is_learnable": self.ks.is_learnable(
                        kp_id,
                        {k for k, m in profile.knowledge_mastery.items()
                         if m >= MasteryLevel.PROFICIENT.value}
                    ),
                })
        return result

    def record_activity(self, learner_id: str, kp_id: str,
                        correct: bool, quality: int = 3,
                        time_spent: float = 0.0,
                        response_time: float = 0.0) -> Dict[str, Any]:
        """记录学习活动

        Args:
            learner_id: 学习者ID
            kp_id: 知识点ID
            correct: 是否答对
            quality: 回忆质量 0-5
            time_spent: 学习时间(分钟)
            response_time: 响应时间(秒)

        Returns:
            活动反馈
        """
        profile = self.learners.get(learner_id)
        if not profile:
            return {"error": "学习者未注册"}

        self.total_learning_activities += 1

        # 更新知识掌握度
        mastery_delta = (quality - 3) * 0.15
        if correct:
            mastery_delta = max(mastery_delta, 0.1)
        else:
            mastery_delta = min(mastery_delta, -0.1)
        profile.update_mastery(kp_id, mastery_delta)

        # 更新行为统计
        profile.total_study_time += time_spent
        profile.total_sessions += 1
        profile.total_items_attempted += 1
        if correct:
            profile.total_items_correct += 1
        profile.avg_session_length = (
            profile.avg_session_length * 0.9 + time_spent * 0.1
        )
        profile.last_study_time = time.time()

        # 更新情感状态
        profile.update_emotion(correct)

        # 记录历史
        profile.history.append({
            "timestamp": time.time(),
            "kp_id": kp_id,
            "correct": correct,
            "quality": quality,
            "time_spent": time_spent,
            "mastery": profile.knowledge_mastery.get(kp_id, 0.0),
            "mastered": profile.knowledge_mastery.get(kp_id, 0.0) >= MasteryLevel.PROFICIENT.value,
        })

        # 间隔重复
        srs_result = self.srs.review(kp_id, quality)

        # 难度调整
        new_difficulty = self.difficulty_adapter.adjust(correct, response_time)

        # 生成反馈
        feedback = self._generate_feedback(profile, kp_id, correct, new_difficulty)

        return {
            "correct": correct,
            "new_mastery": profile.knowledge_mastery.get(kp_id, 0.0),
            "new_difficulty": new_difficulty.name,
            "next_review": srs_result["next_review"],
            "ability_theta": profile.ability_theta,
            "feedback": feedback,
        }

    def _generate_feedback(self, profile: LearnerProfile, kp_id: str,
                           correct: bool,
                           new_difficulty: DifficultyLevel) -> str:
        """生成个性化反馈"""
        kp = self.ks.points.get(kp_id)
        kp_name = kp.name if kp else kp_id

        if correct:
            if profile.frustration > 0.5:
                return f"做得好! '{kp_name}'已掌握, 你正在进步, 继续加油!"
            else:
                return f"正确! '{kp_name}'掌握度提升, 难度将提升到{new_difficulty.name}"
        else:
            if profile.frustration > 0.7:
                return f"没关系, '{kp_name}'有点难, 让我们降低难度重新尝试"
            else:
                prereqs = self.ks.get_prerequisites(kp_id)
                if prereqs:
                    prereq_names = [self.ks.points.get(p, KnowledgePoint(p, p)).name
                                    for p in prereqs]
                    return (f"'{kp_name}'需要先掌握: {', '.join(prereq_names)}. "
                            f"难度调整为{new_difficulty.name}")
                return f"'{kp_name}'还需要多练习, 难度调整为{new_difficulty.name}"

    def get_study_plan(self, learner_id: str,
                       available_time: float = 60.0,
                       target_kp: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取学习计划"""
        profile = self.learners.get(learner_id)
        if not profile:
            return []
        return self.path_generator.generate_study_plan(
            profile, available_time, target_kp
        )

    def get_recommendations(self, learner_id: str,
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """获取个性化推荐"""
        profile = self.learners.get(learner_id)
        if not profile:
            return []
        return self.recommender.recommend(profile, top_k)

    def get_analytics_report(self, learner_id: str) -> Dict[str, Any]:
        """获取学习分析报告"""
        profile = self.learners.get(learner_id)
        if not profile:
            return {"error": "学习者未注册"}
        return self.analytics.analyze(profile)

    def get_due_reviews(self, learner_id: str) -> List[Dict[str, Any]]:
        """获取到期复习项"""
        profile = self.learners.get(learner_id)
        if not profile:
            return []

        due_items = self.srs.get_due_items()
        result = []
        for kp_id in due_items:
            kp = self.ks.points.get(kp_id)
            retention = self.srs.get_retention(kp_id)
            result.append({
                "kp_id": kp_id,
                "name": kp.name if kp else kp_id,
                "current_mastery": profile.knowledge_mastery.get(kp_id, 0.0),
                "estimated_retention": retention,
                "priority": 1.0 - retention,
            })

        result.sort(key=lambda x: x["priority"], reverse=True)
        return result

    def build_curriculum(self, curriculum_data: List[Dict[str, Any]]) -> int:
        """从数据构建课程知识空间

        Args:
            curriculum_data: [{kp_id, name, description, difficulty, prerequisites, ...}, ...]

        Returns:
            添加的知识点数
        """
        count = 0
        for item in curriculum_data:
            difficulty_str = item.get("difficulty", "MEDIUM")
            try:
                difficulty = DifficultyLevel[difficulty_str]
            except KeyError:
                difficulty = DifficultyLevel.MEDIUM

            objectives = []
            for obj_str in item.get("objectives", []):
                try:
                    objectives.append(LearningObjective(obj_str))
                except ValueError:
                    pass

            formats = []
            for fmt_str in item.get("content_formats", ["text"]):
                try:
                    formats.append(ContentFormat(fmt_str))
                except ValueError:
                    formats.append(ContentFormat.TEXT)

            kp = KnowledgePoint(
                kp_id=item["kp_id"],
                name=item.get("name", item["kp_id"]),
                description=item.get("description", ""),
                difficulty=difficulty,
                estimated_time=item.get("estimated_time", 30.0),
                prerequisites=item.get("prerequisites", []),
                objectives=objectives,
                tags=item.get("tags", []),
                content_formats=formats,
            )
            self.ks.add_point(kp)
            count += 1

        return count

    def get_system_stats(self) -> Dict[str, Any]:
        """系统全局统计"""
        return {
            "total_learners": self.total_learners,
            "active_learners": len(self.learners),
            "total_knowledge_points": len(self.ks.points),
            "total_learning_activities": self.total_learning_activities,
            "total_content": len(self.recommender.content_pool),
        }


# ============================================================
# 预设课程
# ============================================================

def create_python_curriculum() -> List[Dict[str, Any]]:
    """创建Python编程课程知识空间"""
    return [
        {"kp_id": "py_var", "name": "变量与数据类型", "difficulty": "BEGINNER",
         "estimated_time": 20, "prerequisites": [], "tags": ["基础", "变量"],
         "objectives": ["understand", "apply"], "content_formats": ["text", "interactive"]},
        {"kp_id": "py_str", "name": "字符串操作", "difficulty": "BEGINNER",
         "estimated_time": 25, "prerequisites": ["py_var"], "tags": ["基础", "字符串"],
         "objectives": ["understand", "apply"], "content_formats": ["text", "exercise"]},
        {"kp_id": "py_list", "name": "列表与元组", "difficulty": "EASY",
         "estimated_time": 30, "prerequisites": ["py_var"], "tags": ["基础", "数据结构"],
         "objectives": ["understand", "apply"], "content_formats": ["text", "interactive", "exercise"]},
        {"kp_id": "py_dict", "name": "字典与集合", "difficulty": "EASY",
         "estimated_time": 30, "prerequisites": ["py_list"], "tags": ["基础", "数据结构"],
         "objectives": ["understand", "apply"], "content_formats": ["text", "exercise"]},
        {"kp_id": "py_control", "name": "条件与循环", "difficulty": "EASY",
         "estimated_time": 35, "prerequisites": ["py_var"], "tags": ["基础", "控制流"],
         "objectives": ["understand", "apply"], "content_formats": ["text", "interactive", "exercise"]},
        {"kp_id": "py_func", "name": "函数定义与调用", "difficulty": "MEDIUM",
         "estimated_time": 40, "prerequisites": ["py_control", "py_list"],
         "tags": ["基础", "函数"], "objectives": ["understand", "apply", "analyze"],
         "content_formats": ["text", "exercise", "project"]},
        {"kp_id": "py_oop", "name": "面向对象编程", "difficulty": "HARD",
         "estimated_time": 60, "prerequisites": ["py_func", "py_dict"],
         "tags": ["进阶", "OOP"], "objectives": ["understand", "apply", "analyze", "create"],
         "content_formats": ["text", "interactive", "project"]},
        {"kp_id": "py_exception", "name": "异常处理", "difficulty": "MEDIUM",
         "estimated_time": 30, "prerequisites": ["py_func"],
         "tags": ["进阶", "异常"], "objectives": ["understand", "apply"],
         "content_formats": ["text", "exercise"]},
        {"kp_id": "py_file", "name": "文件操作", "difficulty": "MEDIUM",
         "estimated_time": 30, "prerequisites": ["py_str", "py_exception"],
         "tags": ["进阶", "IO"], "objectives": ["understand", "apply"],
         "content_formats": ["text", "exercise"]},
        {"kp_id": "py_decorator", "name": "装饰器", "difficulty": "HARD",
         "estimated_time": 45, "prerequisites": ["py_func", "py_oop"],
         "tags": ["高级", "装饰器"], "objectives": ["understand", "apply", "analyze", "create"],
         "content_formats": ["text", "interactive", "project"]},
        {"kp_id": "py_generator", "name": "生成器与迭代器", "difficulty": "HARD",
         "estimated_time": 40, "prerequisites": ["py_func", "py_oop"],
         "tags": ["高级", "生成器"], "objectives": ["understand", "apply", "analyze"],
         "content_formats": ["text", "exercise", "project"]},
        {"kp_id": "py_async", "name": "异步编程", "difficulty": "EXPERT",
         "estimated_time": 50, "prerequisites": ["py_decorator", "py_generator"],
         "tags": ["高级", "异步"], "objectives": ["understand", "apply", "analyze", "evaluate"],
         "content_formats": ["text", "interactive", "project"]},
    ]


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Part 26 — 自适应学习系统 自测")
    print("=" * 60)

    # 1. 构建知识空间 + 课程
    print("\n[1] 构建Python课程知识空间...")
    curriculum = create_python_curriculum()
    system = AdaptiveLearningSystem()
    n = system.build_curriculum(curriculum)
    print(f"    知识点: {n}个, 关系: {len(system.ks.relations)}条")

    # 2. 注册学习者
    print("\n[2] 注册学习者...")
    profile = system.register_learner("learner_001", "小明",
                                       CognitiveStyle.VISUAL)
    print(f"    学习者: {profile.name} ({profile.learner_id})")
    print(f"    认知风格: {profile.cognitive_style.value}")

    # 3. 初始评估
    print("\n[3] 初始水平评估...")
    assessments = [("py_var", 3), ("py_str", 2), ("py_func", 1)]
    result = system.assess_initial_level("learner_001", assessments)
    print(f"    评估结果: 掌握{len(result.get('mastered', []))}个知识点")
    if 'initial_level' in result:
        print(f"    初始水平: {result['initial_level']}")

    # 4. 学习路径
    print("\n[4] 生成学习路径...")
    path = system.get_learning_path("learner_001", max_length=5)
    print(f"    路径长度: {len(path)}步")
    for i, step in enumerate(path[:3]):
        print(f"    {i+1}. {step['name']} ({step['difficulty']})")

    # 5. 记录学习活动
    print("\n[5] 记录学习活动...")
    for _ in range(3):
        activity = system.record_activity("learner_001", "py_var",
                                           correct=True, quality=4)
    print(f"    活动记录: 已完成{system.total_learning_activities}次")

    # 6. 推荐内容
    print("\n[6] 内容推荐...")
    recs = system.get_recommendations("learner_001")
    print(f"    推荐数: {len(recs)}")

    # 7. 分析报告
    print("\n[7] 学习分析报告...")
    report = system.get_analytics_report("learner_001")
    if isinstance(report, dict):
        for k, v in list(report.items())[:5]:
            print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("Part 26 自测完成")
    print("=" * 60)
