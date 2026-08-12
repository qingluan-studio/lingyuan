
# ============================================================
# LINGYUAN MODEL - PART 15
# MLOps & 实验管理
#
# 实验追踪 / 训练任务队列 / GPU资源调度 / 训练实时监控 / 模型对比
# 对应52项清单 #42-46
# ============================================================

import uuid
import math
import random
import json
import os
import time
import threading
from collections import deque, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime


# ============================================================
# 实验追踪 - EXPERIMENT_TRACKER [清单 #42]
# ============================================================

@dataclass
class Experiment:
    """实验数据模型

    记录一次完整的训练实验，包含参数、指标、产物和状态。
    """
    experiment_id: str
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)  # 指标名 -> [(step, value), ...]
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "running"  # running / completed / failed / stopped
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    started_at: str = ""
    ended_at: str = ""
    owner: str = ""
    notes: str = ""
    parent_id: str = ""  # 父实验ID (用于实验血缘追踪)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "config": self.config,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "status": self.status,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "owner": self.owner,
            "notes": self.notes,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experiment":
        """从字典反序列化"""
        return cls(
            experiment_id=d.get("experiment_id", ""),
            name=d.get("name", ""),
            config=d.get("config", {}),
            params=d.get("params", {}),
            metrics=d.get("metrics", {}),
            artifacts=d.get("artifacts", []),
            status=d.get("status", "running"),
            tags=d.get("tags", []),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            started_at=d.get("started_at", ""),
            ended_at=d.get("ended_at", ""),
            owner=d.get("owner", ""),
            notes=d.get("notes", ""),
            parent_id=d.get("parent_id", ""),
        )


class ExperimentTracker:
    """实验追踪系统

    提供实验全生命周期管理:
    - 创建实验并记录配置
    - 追踪参数和指标
    - 管理产物 (artifacts)
    - 查询/搜索/对比实验
    - 持久化到 JSON
    """

    # 合法的实验状态
    VALID_STATUS = {"running", "completed", "failed", "stopped"}

    def __init__(self, storage_path: str = ""):
        """初始化实验追踪器

        Args:
            storage_path: 持久化文件路径, 默认使用 DATA_DIR 下的 experiments.json
        """
        # 若未指定存储路径，使用全局 DATA_DIR
        if not storage_path:
            data_dir = globals().get("DATA_DIR", "/tmp/lingyuan_data")
            os.makedirs(data_dir, exist_ok=True)
            storage_path = os.path.join(data_dir, "experiments.json")
        self.storage_path = storage_path
        self._experiments: Dict[str, Experiment] = {}
        self._lock = threading.RLock()
        # 启动时尝试加载已有数据
        self.load()

    # ----------------------------------------------------------
    # 实验生命周期
    # ----------------------------------------------------------

    def create_experiment(self, name: str, config: Dict[str, Any],
                          tags: List[str] = None, owner: str = "",
                          parent_id: str = "") -> str:
        """创建新实验

        Args:
            name: 实验名称
            config: 实验配置 (模型架构/超参/数据集等)
            tags: 标签列表
            owner: 创建者
            parent_id: 父实验ID

        Returns:
            experiment_id
        """
        now = datetime.now().isoformat()
        exp_id = f"exp_{uuid.uuid4().hex[:12]}"
        exp = Experiment(
            experiment_id=exp_id,
            name=name,
            config=config if config else {},
            tags=tags if tags else [],
            owner=owner,
            parent_id=parent_id,
            created_at=now,
            updated_at=now,
            started_at=now,
        )
        with self._lock:
            self._experiments[exp_id] = exp
        return exp_id

    def set_status(self, experiment_id: str, status: str) -> bool:
        """设置实验状态

        Args:
            experiment_id: 实验ID
            status: 新状态 (running/completed/failed/stopped)

        Returns:
            是否设置成功
        """
        if status not in self.VALID_STATUS:
            return False
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            exp.status = status
            exp.updated_at = datetime.now().isoformat()
            if status in ("completed", "failed", "stopped"):
                exp.ended_at = exp.updated_at
        return True

    def add_tags(self, experiment_id: str, tags: List[str]) -> bool:
        """添加标签"""
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            for t in tags:
                if t not in exp.tags:
                    exp.tags.append(t)
            exp.updated_at = datetime.now().isoformat()
        return True

    def set_notes(self, experiment_id: str, notes: str) -> bool:
        """设置备注"""
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            exp.notes = notes
            exp.updated_at = datetime.now().isoformat()
        return True

    # ----------------------------------------------------------
    # 参数与指标记录
    # ----------------------------------------------------------

    def log_params(self, experiment_id: str, params: Dict[str, Any]) -> bool:
        """记录实验参数

        可多次调用，后续参数会合并到已有参数中。

        Args:
            experiment_id: 实验ID
            params: 参数字典 (如 learning_rate, batch_size, model_name 等)

        Returns:
            是否记录成功
        """
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            exp.params.update(params)
            exp.updated_at = datetime.now().isoformat()
        return True

    def log_metrics(self, experiment_id: str, metrics: Dict[str, float],
                    step: int) -> bool:
        """记录指标 (带步数)

        指标按名称分组存储为 (step, value) 序列，可用于绘制 loss 曲线。

        Args:
            experiment_id: 实验ID
            metrics: 指标字典 (如 {"loss": 0.32, "accuracy": 0.88})
            step: 训练步数

        Returns:
            是否记录成功
        """
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            for k, v in metrics.items():
                if k not in exp.metrics:
                    exp.metrics[k] = []
                # 尝试转换为 float
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    fv = v
                exp.metrics[k].append((step, fv))
            exp.updated_at = datetime.now().isoformat()
        return True

    def log_artifact(self, experiment_id: str, name: str, path: str,
                     artifact_type: str = "file", metadata: Dict = None) -> bool:
        """记录产物 (artifact)

        Args:
            experiment_id: 实验ID
            name: 产物名称
            path: 产物文件路径
            artifact_type: 产物类型 (file/model/checkpoint/log/figure)
            metadata: 额外元数据

        Returns:
            是否记录成功
        """
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            artifact = {
                "name": name,
                "path": path,
                "type": artifact_type,
                "metadata": metadata if metadata else {},
                "logged_at": datetime.now().isoformat(),
            }
            exp.artifacts.append(artifact)
            exp.updated_at = datetime.now().isoformat()
        return True

    # ----------------------------------------------------------
    # 查询与搜索
    # ----------------------------------------------------------

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """获取单个实验"""
        with self._lock:
            return self._experiments.get(experiment_id)

    def list_experiments(self, status: str = None, tag: str = None,
                         owner: str = None, limit: int = 100,
                         sort_by: str = "created_at",
                         reverse: bool = True) -> List[Experiment]:
        """列出实验 (支持过滤)

        Args:
            status: 按状态过滤
            tag: 按标签过滤
            owner: 按创建者过滤
            limit: 返回最大数量
            sort_by: 排序字段 (created_at/updated_at/name)
            reverse: 是否降序

        Returns:
            实验列表
        """
        with self._lock:
            results = list(self._experiments.values())
        # 过滤
        if status:
            results = [e for e in results if e.status == status]
        if tag:
            results = [e for e in results if tag in e.tags]
        if owner:
            results = [e for e in results if e.owner == owner]
        # 排序
        sort_key_map = {
            "created_at": lambda e: e.created_at,
            "updated_at": lambda e: e.updated_at,
            "name": lambda e: e.name,
        }
        key_func = sort_key_map.get(sort_by, lambda e: e.created_at)
        results.sort(key=key_func, reverse=reverse)
        return results[:limit]

    def search_experiments(self, query: str = "", name: str = "",
                           param_filter: Dict[str, Any] = None,
                           metric_name: str = None,
                           metric_min: float = None,
                           metric_max: float = None) -> List[Experiment]:
        """搜索实验

        支持按名称/参数/指标范围搜索。

        Args:
            query: 通用搜索关键词 (匹配名称和标签)
            name: 精确/包含匹配实验名称
            param_filter: 参数过滤 (如 {"model": "gpt2"} 匹配参数中 model=gpt2 的实验)
            metric_name: 指标名称
            metric_min: 指标最小值 (最后一条记录)
            metric_max: 指标最大值

        Returns:
            匹配的实验列表
        """
        with self._lock:
            results = list(self._experiments.values())
        # 通用关键词搜索
        if query:
            q = query.lower()
            results = [
                e for e in results
                if q in e.name.lower() or any(q in t.lower() for t in e.tags)
                or q in e.notes.lower()
            ]
        # 名称匹配
        if name:
            n = name.lower()
            results = [e for e in results if n in e.name.lower()]
        # 参数过滤
        if param_filter:
            def _param_match(exp):
                for pk, pv in param_filter.items():
                    if exp.params.get(pk) != pv:
                        return False
                return True
            results = [e for e in results if _param_match(e)]
        # 指标范围过滤
        if metric_name:
            filtered = []
            for e in results:
                series = e.metrics.get(metric_name, [])
                if not series:
                    continue
                last_val = series[-1][1]
                if isinstance(last_val, (int, float)):
                    if metric_min is not None and last_val < metric_min:
                        continue
                    if metric_max is not None and last_val > metric_max:
                        continue
                filtered.append(e)
            results = filtered
        return results

    def get_best_experiment(self, metric: str, mode: str = "min",
                            status: str = "completed") -> Optional[Experiment]:
        """按指标找最优实验

        Args:
            metric: 指标名称 (如 "loss" 或 "accuracy")
            mode: "min" 取最小值最优, "max" 取最大值最优
            status: 限定实验状态

        Returns:
            最优实验, 无则 None
        """
        candidates = self.list_experiments(status=status, limit=10000)
        best = None
        best_val = None
        for exp in candidates:
            series = exp.metrics.get(metric, [])
            if not series:
                continue
            last_val = series[-1][1]
            if not isinstance(last_val, (int, float)):
                continue
            if best_val is None:
                best_val = last_val
                best = exp
            elif mode == "min" and last_val < best_val:
                best_val = last_val
                best = exp
            elif mode == "max" and last_val > best_val:
                best_val = last_val
                best = exp
        return best

    # ----------------------------------------------------------
    # 对比与分析
    # ----------------------------------------------------------

    def compare_experiments(self, id1: str, id2: str) -> Dict[str, Any]:
        """对比两个实验

        生成差异报告，包含参数差异、指标差异、产物差异。

        Args:
            id1: 实验1 ID
            id2: 实验2 ID

        Returns:
            差异报告字典
        """
        exp1 = self.get_experiment(id1)
        exp2 = self.get_experiment(id2)
        if not exp1 or not exp2:
            return {"error": "实验不存在", "id1": id1, "id2": id2}

        # 参数差异
        all_param_keys = set(exp1.params.keys()) | set(exp2.params.keys())
        param_diff = {}
        for k in all_param_keys:
            v1 = exp1.params.get(k)
            v2 = exp2.params.get(k)
            if v1 != v2:
                param_diff[k] = {"exp1": v1, "exp2": v2}

        # 指标差异 (取最后一条记录对比)
        all_metric_keys = set(exp1.metrics.keys()) | set(exp2.metrics.keys())
        metric_diff = {}
        for k in all_metric_keys:
            s1 = exp1.metrics.get(k, [])
            s2 = exp2.metrics.get(k, [])
            v1 = s1[-1][1] if s1 else None
            v2 = s2[-1][1] if s2 else None
            delta = None
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                delta = v2 - v1
            metric_diff[k] = {"exp1": v1, "exp2": v2, "delta": delta}

        # 产物差异
        arts1 = {a["name"] for a in exp1.artifacts}
        arts2 = {a["name"] for a in exp2.artifacts}
        artifact_diff = {
            "only_in_exp1": list(arts1 - arts2),
            "only_in_exp2": list(arts2 - arts1),
            "common": list(arts1 & arts2),
        }

        return {
            "experiment_1": {"id": id1, "name": exp1.name, "status": exp1.status},
            "experiment_2": {"id": id2, "name": exp2.name, "status": exp2.status},
            "param_differences": param_diff,
            "metric_differences": metric_diff,
            "artifact_differences": artifact_diff,
            "summary": {
                "params_changed": len(param_diff),
                "metrics_compared": len(all_metric_keys),
                "artifacts_diff": len(artifact_diff["only_in_exp1"]) + len(artifact_diff["only_in_exp2"]),
            },
        }

    def get_loss_curve(self, experiment_id: str,
                      metric: str = "loss") -> List[Tuple[int, float]]:
        """获取 loss 曲线数据

        Args:
            experiment_id: 实验ID
            metric: 指标名 (默认 "loss")

        Returns:
            [(step, value), ...] 序列
        """
        exp = self.get_experiment(experiment_id)
        if not exp:
            return []
        return list(exp.metrics.get(metric, []))

    def get_metric_summary(self, experiment_id: str,
                           metric: str) -> Dict[str, float]:
        """获取指标统计摘要

        Returns:
            {min, max, mean, last, count}
        """
        exp = self.get_experiment(experiment_id)
        if not exp:
            return {}
        series = exp.metrics.get(metric, [])
        values = [v for _, v in series if isinstance(v, (int, float))]
        if not values:
            return {}
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "last": values[-1],
            "count": len(values),
        }

    # ----------------------------------------------------------
    # 持久化
    # ----------------------------------------------------------

    def save(self, path: str = None) -> bool:
        """保存实验数据到 JSON

        Args:
            path: 目标路径, 默认使用初始化时的 storage_path

        Returns:
            是否保存成功
        """
        target = path or self.storage_path
        if not target:
            return False
        with self._lock:
            data = {eid: exp.to_dict() for eid, exp in self._experiments.items()}
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except (OSError, IOError):
            return False

    def load(self, path: str = None) -> bool:
        """从 JSON 加载实验数据

        Args:
            path: 来源路径, 默认使用初始化时的 storage_path

        Returns:
            是否加载成功
        """
        source = path or self.storage_path
        if not source or not os.path.exists(source):
            return False
        try:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._experiments = {
                    eid: Experiment.from_dict(d) for eid, d in data.items()
                }
            return True
        except (OSError, IOError, json.JSONDecodeError):
            return False

    def __len__(self) -> int:
        """返回实验总数"""
        with self._lock:
            return len(self._experiments)

    def __repr__(self) -> str:
        return f"<ExperimentTracker experiments={len(self)}>"


# ============================================================
# 训练任务队列 - TRAINING_JOB_QUEUE [清单 #43]
# ============================================================

@dataclass
class TrainingJob:
    """训练任务定义

    描述一个待执行/执行中/已完成的训练任务。
    """
    job_id: str
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 数值越大优先级越高
    status: str = "pending"  # pending / running / completed / failed / cancelled
    dependencies: List[str] = field(default_factory=list)  # 依赖的 job_id 列表
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: int = 0  # 0 表示不限时
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    result: Any = None
    error: str = ""
    resource_requirements: Dict[str, Any] = field(default_factory=dict)  # 如 {"gpu_memory_mb": 8000, "num_gpus": 1}
    owner: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "config": self.config,
            "priority": self.priority,
            "status": self.status,
            "dependencies": self.dependencies,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "resource_requirements": self.resource_requirements,
            "owner": self.owner,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingJob":
        return cls(
            job_id=d.get("job_id", ""),
            name=d.get("name", ""),
            config=d.get("config", {}),
            priority=d.get("priority", 0),
            status=d.get("status", "pending"),
            dependencies=d.get("dependencies", []),
            max_retries=d.get("max_retries", 3),
            retry_count=d.get("retry_count", 0),
            timeout_seconds=d.get("timeout_seconds", 0),
            created_at=d.get("created_at", ""),
            started_at=d.get("started_at", ""),
            completed_at=d.get("completed_at", ""),
            result=d.get("result"),
            error=d.get("error", ""),
            resource_requirements=d.get("resource_requirements", {}),
            owner=d.get("owner", ""),
            tags=d.get("tags", []),
        )


class TrainingJobQueue:
    """训练任务队列

    功能:
    - 优先级调度 (高优先级先执行, 同优先级 FIFO)
    - 并发控制 (max_concurrent_jobs)
    - 任务依赖 (依赖任务完成后才执行)
    - 失败重试 (可配次数)
    - 超时自动取消
    - 状态变更回调
    - 持久化保存/恢复
    """

    # 合法状态
    VALID_STATUS = {"pending", "running", "completed", "failed", "cancelled"}

    def __init__(self, max_concurrent_jobs: int = 4,
                 storage_path: str = "",
                 resource_checker: Callable[[Dict[str, Any]], bool] = None):
        """初始化任务队列

        Args:
            max_concurrent_jobs: 最大并发任务数
            storage_path: 持久化文件路径
            resource_checker: 资源检查回调, 接收 resource_requirements 返回是否有足够资源
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        if not storage_path:
            data_dir = globals().get("DATA_DIR", "/tmp/lingyuan_data")
            os.makedirs(data_dir, exist_ok=True)
            storage_path = os.path.join(data_dir, "training_jobs.json")
        self.storage_path = storage_path
        self._jobs: Dict[str, TrainingJob] = {}
        self._callbacks: List[Callable[[str, str, str], None]] = []  # (job_id, old_status, new_status)
        self._resource_checker = resource_checker
        self._lock = threading.RLock()
        self._running_count = 0
        self._start_times: Dict[str, float] = {}  # job_id -> wall clock for timeout
        self.load()

    # ----------------------------------------------------------
    # 回调注册
    # ----------------------------------------------------------

    def register_callback(self, callback: Callable[[str, str, str], None]):
        """注册状态变更回调

        回调签名: callback(job_id, old_status, new_status)
        """
        self._callbacks.append(callback)

    def _notify(self, job_id: str, old_status: str, new_status: str):
        """触发所有回调"""
        for cb in self._callbacks:
            try:
                cb(job_id, old_status, new_status)
            except Exception:
                pass

    # ----------------------------------------------------------
    # 任务管理
    # ----------------------------------------------------------

    def submit(self, name: str, config: Dict[str, Any] = None,
               priority: int = 0, dependencies: List[str] = None,
               max_retries: int = 3, timeout_seconds: int = 0,
               resource_requirements: Dict[str, Any] = None,
               owner: str = "", tags: List[str] = None) -> str:
        """提交新任务

        Args:
            name: 任务名称
            config: 任务配置
            priority: 优先级 (数值越大越优先)
            dependencies: 依赖任务ID列表
            max_retries: 最大重试次数
            timeout_seconds: 超时秒数 (0=不限)
            resource_requirements: 资源需求
            owner: 创建者
            tags: 标签

        Returns:
            job_id
        """
        now = datetime.now().isoformat()
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = TrainingJob(
            job_id=job_id,
            name=name,
            config=config if config else {},
            priority=priority,
            dependencies=dependencies if dependencies else [],
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            resource_requirements=resource_requirements if resource_requirements else {},
            owner=owner,
            tags=tags if tags else [],
            created_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._notify(job_id, "", "pending")
        return job_id

    def cancel(self, job_id: str) -> bool:
        """取消任务 (仅 pending/running 可取消)"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status not in ("pending", "running"):
                return False
            old = job.status
            job.status = "cancelled"
            job.completed_at = datetime.now().isoformat()
            if old == "running":
                self._running_count -= 1
                self._start_times.pop(job_id, None)
        self._notify(job_id, old, "cancelled")
        return True

    def _mark_completed(self, job_id: str, result: Any = None):
        """标记任务完成"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            old = job.status
            job.status = "completed"
            job.result = result
            job.completed_at = datetime.now().isoformat()
            self._running_count -= 1
            self._start_times.pop(job_id, None)
        self._notify(job_id, old, "completed")

    def _mark_failed(self, job_id: str, error: str):
        """标记任务失败, 并在可重试时重新入队"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            old = job.status
            self._running_count -= 1
            self._start_times.pop(job_id, None)
            job.error = error
            # 判断是否可以重试
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = "pending"
            else:
                job.status = "failed"
                job.completed_at = datetime.now().isoformat()
        new_status = self._jobs[job_id].status
        self._notify(job_id, old, new_status)

    def update_result(self, job_id: str, result: Any) -> bool:
        """更新任务结果并标记完成"""
        job = self._jobs.get(job_id)
        if not job or job.status != "running":
            return False
        self._mark_completed(job_id, result)
        return True

    def report_failure(self, job_id: str, error: str) -> bool:
        """报告任务执行失败"""
        job = self._jobs.get(job_id)
        if not job or job.status != "running":
            return False
        self._mark_failed(job_id, error)
        return True

    # ----------------------------------------------------------
    # 调度
    # ----------------------------------------------------------

    def _dependencies_met(self, job: TrainingJob) -> bool:
        """检查任务依赖是否满足"""
        for dep_id in job.dependencies:
            dep = self._jobs.get(dep_id)
            if not dep or dep.status != "completed":
                return False
        return True

    def _check_timeout(self):
        """检查运行中任务是否超时"""
        now = time.time()
        timed_out = []
        with self._lock:
            for job_id, start in list(self._start_times.items()):
                job = self._jobs.get(job_id)
                if not job:
                    continue
                if job.timeout_seconds > 0 and (now - start) > job.timeout_seconds:
                    timed_out.append(job_id)
        for jid in timed_out:
            self._mark_failed(jid, f"任务超时 ({self._jobs[jid].timeout_seconds}s)")

    def next_job(self) -> Optional[TrainingJob]:
        """获取下一个可执行的任务

        调度策略:
        1. 先检查超时
        2. 检查并发数限制
        3. 从 pending 任务中按优先级 + FIFO 选取
        4. 检查依赖是否完成
        5. 检查资源是否充足 (通过 resource_checker)
        6. 将任务标记为 running

        Returns:
            可执行的任务, 无则 None
        """
        self._check_timeout()
        with self._lock:
            if self._running_count >= self.max_concurrent_jobs:
                return None
            # 筛选 pending 且依赖满足的任务
            candidates = [
                job for job in self._jobs.values()
                if job.status == "pending" and self._dependencies_met(job)
            ]
            if not candidates:
                return None
            # 按优先级降序, 同优先级按创建时间升序 (FIFO)
            candidates.sort(key=lambda j: (-j.priority, j.created_at))
            for job in candidates:
                # 资源检查
                if self._resource_checker and job.resource_requirements:
                    if not self._resource_checker(job.resource_requirements):
                        continue
                # 选中此任务
                old = job.status
                job.status = "running"
                job.started_at = datetime.now().isoformat()
                self._running_count += 1
                self._start_times[job.job_id] = time.time()
                self._notify(job.job_id, old, "running")
                return job
        return None

    def drain(self, executor: Callable[[TrainingJob], Any] = None) -> int:
        """排空队列: 不断调度执行任务直到没有可执行任务

        Args:
            executor: 任务执行回调, 接收 TrainingJob 返回结果或抛异常。
                      若不提供, 仅做调度标记不实际执行。

        Returns:
            本次排空执行的任务数
        """
        executed = 0
        while True:
            job = self.next_job()
            if not job:
                break
            executed += 1
            if executor:
                try:
                    result = executor(job)
                    self._mark_completed(job.job_id, result)
                except Exception as e:
                    self._mark_failed(job.job_id, str(e))
        return executed

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """获取任务"""
        return self._jobs.get(job_id)

    def list_jobs(self, status: str = None, owner: str = None,
                  limit: int = 100) -> List[TrainingJob]:
        """列出任务"""
        with self._lock:
            results = list(self._jobs.values())
        if status:
            results = [j for j in results if j.status == status]
        if owner:
            results = [j for j in results if j.owner == owner]
        results.sort(key=lambda j: j.created_at, reverse=True)
        return results[:limit]

    def queue_status(self) -> Dict[str, Any]:
        """获取队列状态摘要"""
        with self._lock:
            counter = Counter(j.status for j in self._jobs.values())
        return {
            "total": len(self._jobs),
            "pending": counter.get("pending", 0),
            "running": counter.get("running", 0),
            "completed": counter.get("completed", 0),
            "failed": counter.get("failed", 0),
            "cancelled": counter.get("cancelled", 0),
            "max_concurrent": self.max_concurrent_jobs,
            "available_slots": max(0, self.max_concurrent_jobs - self._running_count),
        }

    # ----------------------------------------------------------
    # 持久化
    # ----------------------------------------------------------

    def save(self, path: str = None) -> bool:
        """保存队列状态到 JSON"""
        target = path or self.storage_path
        if not target:
            return False
        with self._lock:
            data = {jid: job.to_dict() for jid, job in self._jobs.items()}
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except (OSError, IOError):
            return False

    def load(self, path: str = None) -> bool:
        """从 JSON 恢复队列状态"""
        source = path or self.storage_path
        if not source or not os.path.exists(source):
            return False
        try:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._jobs = {jid: TrainingJob.from_dict(d) for jid, d in data.items()}
                # 恢复后, 所有 running 状态重置为 pending (进程重启)
                self._running_count = 0
                self._start_times = {}
                for job in self._jobs.values():
                    if job.status == "running":
                        job.status = "pending"
            return True
        except (OSError, IOError, json.JSONDecodeError):
            return False

    def __len__(self) -> int:
        return len(self._jobs)

    def __repr__(self) -> str:
        s = self.queue_status()
        return (f"<TrainingJobQueue total={s['total']} "
                f"running={s['running']} pending={s['pending']}>")


# ============================================================
# GPU资源调度 - GPU_SCHEDULER [清单 #44]
# ============================================================

@dataclass
class GPUDevice:
    """GPU 设备

    描述一张 GPU 的基本信息和显存状态。
    """
    device_id: str
    name: str
    total_memory_mb: int
    compute_capability: str = "8.0"
    allocated_memory_mb: int = 0
    utilization_history: deque = field(default_factory=lambda: deque(maxlen=120))
    reserved_for: str = ""  # 为特定用户/任务预留
    reserved_until: float = 0.0  # 预留截止时间戳

    @property
    def available_memory_mb(self) -> int:
        """可用显存"""
        return self.total_memory_mb - self.allocated_memory_mb

    @property
    def is_reserved(self) -> bool:
        """是否处于预留状态"""
        return bool(self.reserved_for) and time.time() < self.reserved_until

    @property
    def utilization_pct(self) -> float:
        """最新利用率"""
        if not self.utilization_history:
            return 0.0
        return self.utilization_history[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "total_memory_mb": self.total_memory_mb,
            "compute_capability": self.compute_capability,
            "allocated_memory_mb": self.allocated_memory_mb,
            "available_memory_mb": self.available_memory_mb,
            "reserved_for": self.reserved_for,
            "reserved_until": self.reserved_until,
            "utilization_pct": self.utilization_pct,
        }


@dataclass
class GPUAllocation:
    """GPU 分配记录"""
    allocation_id: str
    device_id: str
    task_id: str
    memory_mb: int
    strategy: str = "single"  # single / data_parallel / model_parallel
    allocated_at: float = 0.0
    priority: int = 0


class GPUScheduler:
    """GPU 资源调度器

    功能:
    - GPU 设备注册与管理
    - 显存分配/释放/查询
    - 按显存需求分配任务到合适的 GPU
    - 多卡策略 (single / data_parallel / model_parallel)
    - 抢占机制 (高优先级任务可抢占低优先级)
    - 显存碎片分析
    - GPU 利用率追踪
    - 等待队列 (资源不足时排队)
    - 预留机制 (为特定用户/任务预留)
    """

    # 多卡策略
    STRATEGY_SINGLE = "single"
    STRATEGY_DATA_PARALLEL = "data_parallel"
    STRATEGY_MODEL_PARALLEL = "model_parallel"

    def __init__(self):
        self._devices: Dict[str, GPUDevice] = {}
        self._allocations: Dict[str, GPUAllocation] = {}  # allocation_id -> allocation
        self._device_allocations: Dict[str, List[str]] = {}  # device_id -> [allocation_id, ...]
        self._wait_queue: deque = deque()  # 等待队列: (task_id, memory_mb, num_gpus, strategy, priority, callback)
        self._lock = threading.RLock()

    @property
    def devices(self) -> Dict[str, 'GPUDevice']:
        """已注册的GPU设备字典"""
        return self._devices

    # ----------------------------------------------------------
    # 设备管理
    # ----------------------------------------------------------

    def register_gpu(self, device_id: str, name: str, memory_mb: int,
                     compute_capability: str = "8.0") -> bool:
        """注册 GPU 设备

        Args:
            device_id: 设备标识 (如 "gpu0")
            name: 设备名称 (如 "A100-SXM4-80GB")
            memory_mb: 显存总量 (MB)
            compute_capability: 计算能力版本

        Returns:
            是否注册成功 (重复注册返回 False)
        """
        with self._lock:
            if device_id in self._devices:
                return False
            self._devices[device_id] = GPUDevice(
                device_id=device_id,
                name=name,
                total_memory_mb=memory_mb,
                compute_capability=compute_capability,
            )
            self._device_allocations[device_id] = []
            return True

    def unregister_gpu(self, device_id: str) -> bool:
        """注销 GPU 设备 (仅在无分配时允许)"""
        with self._lock:
            if device_id not in self._devices:
                return False
            allocs = self._device_allocations.get(device_id, [])
            if allocs:
                return False  # 仍有任务占用
            del self._devices[device_id]
            self._device_allocations.pop(device_id, None)
            return True

    def list_gpus(self) -> List[Dict[str, Any]]:
        """列出所有 GPU 设备状态"""
        with self._lock:
            return [dev.to_dict() for dev in self._devices.values()]

    def get_gpu(self, device_id: str) -> Optional[GPUDevice]:
        """获取 GPU 设备"""
        return self._devices.get(device_id)

    # ----------------------------------------------------------
    # 显存管理
    # ----------------------------------------------------------

    def allocate(self, device_id: str, memory_mb: int, task_id: str,
                 priority: int = 0,
                 strategy: str = STRATEGY_SINGLE) -> Optional[str]:
        """在指定 GPU 上分配显存

        Args:
            device_id: 目标 GPU
            memory_mb: 需要的显存 (MB)
            task_id: 任务ID
            priority: 优先级
            strategy: 多卡策略

        Returns:
            allocation_id, 失败返回 None
        """
        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return None
            # 检查预留
            if dev.is_reserved and dev.reserved_for != task_id:
                return None
            if dev.available_memory_mb < memory_mb:
                return None
            alloc_id = f"alloc_{uuid.uuid4().hex[:10]}"
            alloc = GPUAllocation(
                allocation_id=alloc_id,
                device_id=device_id,
                task_id=task_id,
                memory_mb=memory_mb,
                strategy=strategy,
                allocated_at=time.time(),
                priority=priority,
            )
            dev.allocated_memory_mb += memory_mb
            self._allocations[alloc_id] = alloc
            self._device_allocations[device_id].append(alloc_id)
            return alloc_id

    def release(self, allocation_id: str) -> bool:
        """释放显存分配"""
        with self._lock:
            alloc = self._allocations.get(allocation_id)
            if not alloc:
                return False
            dev = self._devices.get(alloc.device_id)
            if dev:
                dev.allocated_memory_mb = max(0, dev.allocated_memory_mb - alloc.memory_mb)
            self._device_allocations.get(alloc.device_id, []).remove(allocation_id)
            del self._allocations[allocation_id]
            # 尝试处理等待队列
            self._process_wait_queue()
            return True

    def release_task(self, task_id: str) -> int:
        """释放某任务的所有分配

        Returns:
            释放的分配数量
        """
        with self._lock:
            to_release = [
                aid for aid, alloc in self._allocations.items()
                if alloc.task_id == task_id
            ]
        count = 0
        for aid in to_release:
            if self.release(aid):
                count += 1
        return count

    def available_memory(self, device_id: str) -> int:
        """查询指定 GPU 可用显存"""
        dev = self._devices.get(device_id)
        if not dev:
            return 0
        return dev.available_memory_mb

    # ----------------------------------------------------------
    # 任务分配 (自动选卡)
    # ----------------------------------------------------------

    def assign_task(self, task_id: str, memory_mb: int, num_gpus: int = 1,
                    strategy: str = STRATEGY_SINGLE,
                    priority: int = 0,
                    callback: Callable = None) -> Optional[List[str]]:
        """为任务分配 GPU (自动选择)

        分配策略:
        - single: 选择一张显存足够的 GPU
        - data_parallel: 选择多张显存足够的 GPU
        - model_parallel: 选择多张 GPU, 显存可跨卡切分

        若资源不足且提供了 callback, 任务进入等待队列。

        Args:
            task_id: 任务ID
            memory_mb: 每张卡需要的显存 (MB)
            num_gpus: 需要的 GPU 数量
            strategy: 多卡策略
            priority: 优先级
            callback: 资源就绪时的回调

        Returns:
            allocation_id 列表, 失败返回 None
        """
        with self._lock:
            alloc_ids = self._try_assign(task_id, memory_mb, num_gpus,
                                          strategy, priority)
            if alloc_ids:
                return alloc_ids
            # 资源不足, 尝试抢占
            preempted = self._try_preempt(task_id, memory_mb, num_gpus, priority)
            if preempted:
                alloc_ids = self._try_assign(task_id, memory_mb, num_gpus,
                                              strategy, priority)
                if alloc_ids:
                    return alloc_ids
            # 仍不足, 进入等待队列
            if callback is not None:
                self._wait_queue.append((
                    task_id, memory_mb, num_gpus, strategy, priority, callback
                ))
            return None

    def _try_assign(self, task_id: str, memory_mb: int, num_gpus: int,
                    strategy: str, priority: int) -> List[str]:
        """尝试分配 (不加锁, 调用者持有锁)"""
        # 找到可用且非预留(或为自己预留)的 GPU
        available_devs = []
        for dev in self._devices.values():
            if dev.is_reserved and dev.reserved_for != task_id:
                continue
            if dev.available_memory_mb >= memory_mb:
                available_devs.append(dev)
        if len(available_devs) < num_gpus:
            return []
        # 按可用显存降序选择 (best-fit: 选可用显存最少的满足条件的卡, 减少碎片)
        # 这里使用 first-fit 降序以简化逻辑
        available_devs.sort(key=lambda d: d.available_memory_mb, reverse=True)
        selected = available_devs[:num_gpus]
        alloc_ids = []
        for dev in selected:
            alloc_id = f"alloc_{uuid.uuid4().hex[:10]}"
            alloc = GPUAllocation(
                allocation_id=alloc_id,
                device_id=dev.device_id,
                task_id=task_id,
                memory_mb=memory_mb,
                strategy=strategy,
                allocated_at=time.time(),
                priority=priority,
            )
            dev.allocated_memory_mb += memory_mb
            self._allocations[alloc_id] = alloc
            self._device_allocations[dev.device_id].append(alloc_id)
            alloc_ids.append(alloc_id)
        return alloc_ids

    def _try_preempt(self, task_id: str, memory_mb: int,
                     num_gpus: int, priority: int) -> bool:
        """尝试抢占低优先级任务 (调用者持有锁)

        策略: 找到优先级低于当前任务且能释放足够显存的分配, 释放之。
        """
        if priority <= 0:
            return False
        # 需要抢占的分配数
        needed = num_gpus
        # 候选: 优先级比当前低的分配
        candidates = [
            alloc for alloc in self._allocations.values()
            if alloc.priority < priority and alloc.task_id != task_id
        ]
        # 按优先级升序 (先抢优先级最低的)
        candidates.sort(key=lambda a: a.priority)
        to_free = []
        for alloc in candidates:
            if needed <= 0:
                break
            to_free.append(alloc.allocation_id)
            needed -= 1
        if needed > 0:
            return False
        # 执行抢占
        for aid in to_free:
            alloc = self._allocations.get(aid)
            if alloc:
                dev = self._devices.get(alloc.device_id)
                if dev:
                    dev.allocated_memory_mb = max(0, dev.allocated_memory_mb - alloc.memory_mb)
                self._device_allocations.get(alloc.device_id, []).remove(aid)
                del self._allocations[aid]
        return True

    def _process_wait_queue(self):
        """处理等待队列 (调用者持有锁)"""
        if not self._wait_queue:
            return
        remaining = deque()
        while self._wait_queue:
            item = self._wait_queue.popleft()
            task_id, memory_mb, num_gpus, strategy, priority, callback = item
            alloc_ids = self._try_assign(task_id, memory_mb, num_gpus,
                                          strategy, priority)
            if alloc_ids:
                # 资源就绪, 触发回调
                try:
                    callback(task_id, alloc_ids)
                except Exception:
                    pass
            else:
                remaining.append(item)
        self._wait_queue = remaining

    # ----------------------------------------------------------
    # 预留机制
    # ----------------------------------------------------------

    def reserve_gpu(self, device_id: str, for_task: str,
                    duration_seconds: int = 3600) -> bool:
        """为特定任务/用户预留 GPU

        Args:
            device_id: GPU 设备ID
            for_task: 预留目标 (任务ID或用户名)
            duration_seconds: 预留时长 (秒)

        Returns:
            是否预留成功
        """
        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return False
            if dev.is_reserved:
                return False
            dev.reserved_for = for_task
            dev.reserved_until = time.time() + duration_seconds
            return True

    def release_reservation(self, device_id: str) -> bool:
        """释放预留"""
        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return False
            dev.reserved_for = ""
            dev.reserved_until = 0.0
            return True

    # ----------------------------------------------------------
    # 利用率追踪
    # ----------------------------------------------------------

    def record_utilization(self, device_id: str, utilization_pct: float):
        """记录 GPU 利用率

        Args:
            device_id: GPU 设备ID
            utilization_pct: 利用率 (0-100)
        """
        with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return
            dev.utilization_history.append(max(0.0, min(100.0, float(utilization_pct))))

    def get_utilization_history(self, device_id: str,
                                last_n: int = 60) -> List[float]:
        """获取利用率历史"""
        dev = self._devices.get(device_id)
        if not dev:
            return []
        history = list(dev.utilization_history)
        return history[-last_n:] if last_n else history

    # ----------------------------------------------------------
    # 显存碎片分析
    # ----------------------------------------------------------

    def fragmentation_analysis(self) -> Dict[str, Any]:
        """显存碎片分析

        分析所有 GPU 的显存碎片情况。

        Returns:
            {
                "devices": 每张卡的碎片信息,
                "total_fragmentation": 整体碎片率,
                "total_available_mb": 总可用显存,
                "max_contiguous_mb": 最大连续可用块,
            }
        """
        with self._lock:
            devices_info = []
            total_alloc = 0
            total_avail = 0
            total_total = 0
            for dev in self._devices.values():
                avail = dev.available_memory_mb
                alloc = dev.allocated_memory_mb
                total = dev.total_memory_mb
                # 碎片率 = 1 - (最大连续可用块 / 总可用)
                # 简化: 单卡的可用即视为连续块 (无虚拟地址碎片)
                # 碎片率定义为: 已分配显存的离散程度
                frag_rate = 0.0
                if total > 0 and avail < total:
                    frag_rate = 1.0 - (avail / total) if alloc > 0 else 0.0
                devices_info.append({
                    "device_id": dev.device_id,
                    "total_mb": total,
                    "allocated_mb": alloc,
                    "available_mb": avail,
                    "fragmentation_rate": round(frag_rate, 4),
                })
                total_alloc += alloc
                total_avail += avail
                total_total += total
            # 整体碎片率
            overall_frag = 0.0
            if total_total > 0:
                overall_frag = 1.0 - (total_avail / total_total)
            # 最大连续可用块 = 单卡最大可用显存
            max_contiguous = max(
                (d["available_mb"] for d in devices_info),
                default=0,
            )
            return {
                "devices": devices_info,
                "total_fragmentation": round(overall_frag, 4),
                "total_allocated_mb": total_alloc,
                "total_available_mb": total_avail,
                "total_memory_mb": total_total,
                "max_contiguous_mb": max_contiguous,
                "device_count": len(self._devices),
            }

    # ----------------------------------------------------------
    # 状态查询
    # ----------------------------------------------------------

    def cluster_status(self) -> Dict[str, Any]:
        """获取集群整体状态"""
        frag = self.fragmentation_analysis()
        with self._lock:
            return {
                "total_gpus": len(self._devices),
                "active_allocations": len(self._allocations),
                "waiting_tasks": len(self._wait_queue),
                "total_allocated_mb": frag["total_allocated_mb"],
                "total_available_mb": frag["total_available_mb"],
                "total_memory_mb": frag["total_memory_mb"],
                "fragmentation": frag["total_fragmentation"],
                "devices": frag["devices"],
            }

    def __repr__(self) -> str:
        return (f"<GPUScheduler devices={len(self._devices)} "
                f"allocations={len(self._allocations)} "
                f"waiting={len(self._wait_queue)}>")


# ============================================================
# 训练实时监控 - TRAINING_MONITOR [清单 #45]
# ============================================================

class RingBuffer:
    """环形缓冲区

    存储最近 N 个数据点, 满后自动覆盖最旧数据。
    """

    def __init__(self, capacity: int = 1000):
        self._capacity = max(1, capacity)
        self._buffer: deque = deque(maxlen=self._capacity)

    def append(self, item: Any):
        """追加数据点"""
        self._buffer.append(item)

    def extend(self, items):
        """批量追加"""
        for item in items:
            self._buffer.append(item)

    def latest(self, n: int = 1) -> List[Any]:
        """获取最近 n 个数据点"""
        n = max(0, n)
        items = list(self._buffer)
        return items[-n:] if n > 0 else items

    def all(self) -> List[Any]:
        """获取全部数据"""
        return list(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


@dataclass
class AlertEvent:
    """告警事件"""
    alert_id: str
    monitor_id: str
    alert_type: str  # threshold / trend / nan / grad_explosion / loss_spike
    severity: str    # info / warning / critical
    message: str
    metric: str
    value: float
    threshold: float
    step: int
    timestamp: str = ""


class TrainingMonitor:
    """训练实时监控系统

    功能:
    - 实时指标采集 (loss, learning_rate, grad_norm, throughput 等)
    - 可配置采集频率 (每步/每N步)
    - 环形缓冲区存储最近 N 个数据点
    - 异常检测 (loss 突增 / NaN / 梯度爆炸)
    - 阈值告警与趋势告警
    - 生成可视化图表数据点
    - 训练统计 (ETA / 已完成步数 / 总步数 / 速度)
    - 检查点保存建议
    - 多实验同时监控
    """

    # 内置指标名
    METRIC_LOSS = "loss"
    METRIC_LR = "learning_rate"
    METRIC_GRAD_NORM = "grad_norm"
    METRIC_THROUGHPUT = "throughput"

    def __init__(self, buffer_size: int = 1000,
                 collect_every: int = 1):
        """初始化监控器

        Args:
            buffer_size: 环形缓冲区大小
            collect_every: 采集频率 (每 N 步采集一次)
        """
        self.buffer_size = buffer_size
        self.collect_every = max(1, collect_every)
        # 每个 monitor_id 的数据
        self._monitors: Dict[str, Dict[str, Any]] = {}
        # 告警回调
        self._alert_callbacks: List[Callable[[AlertEvent], None]] = []
        # 告警历史
        self._alerts: deque = deque(maxlen=500)
        self._lock = threading.RLock()

    # ----------------------------------------------------------
    # 监控会话管理
    # ----------------------------------------------------------

    def start_monitoring(self, monitor_id: str, total_steps: int = 0,
                         config: Dict[str, Any] = None,
                         thresholds: Dict[str, Dict[str, float]] = None) -> bool:
        """开始监控一个训练过程

        Args:
            monitor_id: 监控ID (通常为 experiment_id 或 job_id)
            total_steps: 总训练步数 (用于计算 ETA)
            config: 训练配置
            thresholds: 告警阈值, 格式如
                {"loss": {"max": 10.0, "min": 0.0},
                 "grad_norm": {"max": 100.0}}

        Returns:
            是否成功启动 (重复启动返回 False)
        """
        with self._lock:
            if monitor_id in self._monitors:
                return False
            self._monitors[monitor_id] = {
                "total_steps": total_steps,
                "current_step": 0,
                "config": config if config else {},
                "thresholds": thresholds if thresholds else {},
                "metrics": {},  # metric_name -> RingBuffer
                "started_at": time.time(),
                "last_update": time.time(),
                "step_times": deque(maxlen=100),  # 最近100步的时间戳
                "checkpoint_steps": [],  # 建议保存检查点的步数
                "last_checkpoint_step": 0,
            }
            return True

    def stop_monitoring(self, monitor_id: str) -> bool:
        """停止监控"""
        with self._lock:
            return self._monitors.pop(monitor_id, None) is not None

    def list_monitors(self) -> List[str]:
        """列出所有正在监控的ID"""
        with self._lock:
            return list(self._monitors.keys())

    # ----------------------------------------------------------
    # 指标采集
    # ----------------------------------------------------------

    def record(self, monitor_id: str, step: int,
               metrics: Dict[str, float]):
        """记录指标数据点

        根据 collect_every 决定是否实际采集。

        Args:
            monitor_id: 监控ID
            step: 当前步数
            metrics: 指标字典
        """
        with self._lock:
            mon = self._monitors.get(monitor_id)
            if not mon:
                return
            # 采集频率过滤
            if step > 0 and step % self.collect_every != 0:
                # 仍更新步数
                mon["current_step"] = step
                mon["last_update"] = time.time()
                return
            now = time.time()
            mon["current_step"] = step
            mon["last_update"] = now
            mon["step_times"].append(now)
            # 存储指标
            for name, value in metrics.items():
                if name not in mon["metrics"]:
                    mon["metrics"][name] = RingBuffer(self.buffer_size)
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    fv = float("nan")
                mon["metrics"][name].append((step, fv, now))
            # 异常检测
            self._detect_anomalies(monitor_id, mon, step, metrics)

    def _detect_anomalies(self, monitor_id: str, mon: Dict, step: int,
                          metrics: Dict[str, float]):
        """异常检测 (调用者持有锁)"""
        for name, value in metrics.items():
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            # NaN 检测
            if math.isnan(fv) or math.isinf(fv):
                self._raise_alert(monitor_id, "nan", "critical",
                                  f"指标 {name} 出现 NaN/Inf (step={step})",
                                  name, fv, 0.0, step)
                continue
            # 阈值检测
            thresholds = mon["thresholds"].get(name, {})
            if "max" in thresholds and fv > thresholds["max"]:
                self._raise_alert(monitor_id, "threshold", "warning",
                                  f"指标 {name}={fv:.4f} 超过上限 {thresholds['max']}",
                                  name, fv, thresholds["max"], step)
            if "min" in thresholds and fv < thresholds["min"]:
                self._raise_alert(monitor_id, "threshold", "warning",
                                  f"指标 {name}={fv:.4f} 低于下限 {thresholds['min']}",
                                  name, fv, thresholds["min"], step)
            # loss 突增检测
            if name == self.METRIC_LOSS:
                buf = mon["metrics"].get(name)
                if buf and len(buf) >= 2:
                    prev_val = buf.all()[-2][1]
                    if prev_val > 0 and fv > prev_val * 2:
                        self._raise_alert(monitor_id, "loss_spike", "critical",
                                          f"loss 突增: {prev_val:.4f} -> {fv:.4f} (step={step})",
                                          name, fv, prev_val, step)
            # 梯度爆炸检测
            if name == self.METRIC_GRAD_NORM:
                if fv > 1000.0:
                    self._raise_alert(monitor_id, "grad_explosion", "critical",
                                      f"梯度爆炸: grad_norm={fv:.4f} (step={step})",
                                      name, fv, 1000.0, step)
            # 趋势检测: 连续上升的 loss
            if name == self.METRIC_LOSS:
                buf = mon["metrics"].get(name)
                if buf and len(buf) >= 5:
                    recent = buf.all()[-5:]
                    vals = [v[1] for v in recent]
                    if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
                        self._raise_alert(monitor_id, "trend", "warning",
                                          f"loss 连续5步上升, 可能发散 (step={step})",
                                          name, fv, 0.0, step)

    def _raise_alert(self, monitor_id: str, alert_type: str,
                     severity: str, message: str, metric: str,
                     value: float, threshold: float, step: int):
        """产生告警 (调用者持有锁)"""
        alert = AlertEvent(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            monitor_id=monitor_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold,
            step=step,
            timestamp=datetime.now().isoformat(),
        )
        self._alerts.append(alert)
        for cb in self._alert_callbacks:
            try:
                cb(alert)
            except Exception:
                pass

    # ----------------------------------------------------------
    # 告警管理
    # ----------------------------------------------------------

    def register_alert_callback(self, callback: Callable[[AlertEvent], None]):
        """注册告警回调"""
        self._alert_callbacks.append(callback)

    def get_alerts(self, monitor_id: str = None,
                   severity: str = None,
                   limit: int = 100) -> List[AlertEvent]:
        """获取告警历史"""
        with self._lock:
            alerts = list(self._alerts)
        if monitor_id:
            alerts = [a for a in alerts if a.monitor_id == monitor_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:] if limit else alerts

    # ----------------------------------------------------------
    # 数据查询与可视化
    # ----------------------------------------------------------

    def get_metric_series(self, monitor_id: str, metric: str,
                          last_n: int = 0) -> List[Tuple[int, float]]:
        """获取指标时序数据 (用于绘图)

        Returns:
            [(step, value), ...]
        """
        with self._lock:
            mon = self._monitors.get(monitor_id)
            if not mon:
                return []
            buf = mon["metrics"].get(metric)
            if not buf:
                return []
            data = [(s, v) for s, v, _ in buf.all()]
        return data[-last_n:] if last_n else data

    def get_chart_data(self, monitor_id: str, metric: str,
                       last_n: int = 200) -> Dict[str, Any]:
        """生成图表数据点 (可用于前端绘图)

        Returns:
            {
                "metric": 指标名,
                "steps": [步数列表],
                "values": [数值列表],
                "x_range": [min_step, max_step],
                "y_range": [min_val, max_val],
                "point_count": 数据点数,
            }
        """
        series = self.get_metric_series(monitor_id, metric, last_n)
        if not series:
            return {
                "metric": metric,
                "steps": [],
                "values": [],
                "x_range": [0, 0],
                "y_range": [0, 0],
                "point_count": 0,
            }
        steps = [s for s, _ in series]
        values = [v for _, v in series]
        return {
            "metric": metric,
            "steps": steps,
            "values": values,
            "x_range": [min(steps), max(steps)],
            "y_range": [min(values), max(values)],
            "point_count": len(series),
        }

    def get_latest_metrics(self, monitor_id: str) -> Dict[str, float]:
        """获取最新一组指标"""
        with self._lock:
            mon = self._monitors.get(monitor_id)
            if not mon:
                return {}
            result = {}
            for name, buf in mon["metrics"].items():
                data = buf.all()
                if data:
                    result[name] = data[-1][1]
            return result

    # ----------------------------------------------------------
    # 训练统计
    # ----------------------------------------------------------

    def get_training_stats(self, monitor_id: str) -> Dict[str, Any]:
        """获取训练统计信息

        Returns:
            {
                "current_step": 当前步数,
                "total_steps": 总步数,
                "progress_pct": 进度百分比,
                "speed_steps_per_sec": 每秒步数,
                "elapsed_seconds": 已运行秒数,
                "eta_seconds": 预计剩余秒数,
                "eta_human": 人类可读ETA,
            }
        """
        with self._lock:
            mon = self._monitors.get(monitor_id)
            if not mon:
                return {}
            current = mon["current_step"]
            total = mon["total_steps"]
            elapsed = time.time() - mon["started_at"]
            # 计算速度
            speed = 0.0
            step_times = list(mon["step_times"])
            if len(step_times) >= 2:
                time_span = step_times[-1] - step_times[0]
                step_span = len(step_times) - 1
                if time_span > 0:
                    speed = step_span / time_span
            # ETA
            eta = 0.0
            if speed > 0 and total > current:
                eta = (total - current) / speed
            progress = (current / total * 100) if total > 0 else 0.0
            return {
                "current_step": current,
                "total_steps": total,
                "progress_pct": round(progress, 2),
                "speed_steps_per_sec": round(speed, 4),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta, 1),
                "eta_human": self._format_duration(eta),
            }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """格式化时长为人类可读"""
        if seconds <= 0:
            return "未知"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h{minutes}m{secs}s"
        if minutes > 0:
            return f"{minutes}m{secs}s"
        return f"{secs}s"

    # ----------------------------------------------------------
    # 检查点建议
    # ----------------------------------------------------------

    def suggest_checkpoint(self, monitor_id: str,
                           interval: int = 1000,
                           metric: str = "loss",
                           mode: str = "min") -> Optional[int]:
        """建议保存检查点的时机

        策略:
        1. 每隔 interval 步建议保存
        2. 当指标达到新的最优值时建议保存

        Args:
            monitor_id: 监控ID
            interval: 步数间隔
            metric: 参考指标
            mode: "min" 或 "max"

        Returns:
            建议保存检查点的步数, 无建议返回 None
        """
        with self._lock:
            mon = self._monitors.get(monitor_id)
            if not mon:
                return None
            current = mon["current_step"]
            last_cp = mon["last_checkpoint_step"]
            # 按间隔
            if current > 0 and current - last_cp >= interval:
                mon["checkpoint_steps"].append(current)
                mon["last_checkpoint_step"] = current
                return current
            # 按指标最优
            buf = mon["metrics"].get(metric)
            if buf and len(buf) >= 2:
                series = [(s, v) for s, v, _ in buf.all()]
                vals = [v for _, v in series]
                if len(vals) >= 2:
                    best_so_far = min(vals[:-1]) if mode == "min" else max(vals[:-1])
                    current_val = vals[-1]
                    is_better = (
                        current_val < best_so_far if mode == "min"
                        else current_val > best_so_far
                    )
                    if is_better and current - last_cp >= interval // 4:
                        mon["checkpoint_steps"].append(current)
                        mon["last_checkpoint_step"] = current
                        return current
            return None

    def get_checkpoint_history(self, monitor_id: str) -> List[int]:
        """获取检查点建议历史"""
        with self._lock:
            mon = self._monitors.get(monitor_id)
            if not mon:
                return []
            return list(mon["checkpoint_steps"])

    # ----------------------------------------------------------
    # 多实验对比
    # ----------------------------------------------------------

    def compare_monitors(self, monitor_ids: List[str],
                         metric: str = "loss") -> Dict[str, Any]:
        """同时对比多个训练的指标

        Args:
            monitor_ids: 监控ID列表
            metric: 对比指标

        Returns:
            {
                "metric": 指标名,
                "series": {monitor_id: [(step, value), ...]},
                "latest": {monitor_id: 最新值},
                "best": {monitor_id: 最优值},
                "ranking": 按最优值排序的列表,
            }
        """
        with self._lock:
            series_data = {}
            latest = {}
            best = {}
            for mid in monitor_ids:
                mon = self._monitors.get(mid)
                if not mon:
                    continue
                buf = mon["metrics"].get(metric)
                if not buf:
                    continue
                data = [(s, v) for s, v, _ in buf.all()]
                series_data[mid] = data
                if data:
                    latest[mid] = data[-1][1]
                    vals = [v for _, v in data]
                    best[mid] = min(vals) if metric == "loss" else max(vals)
            # 排名
            ranking = []
            for mid, val in sorted(best.items(),
                                    key=lambda x: x[1],
                                    reverse=(metric != "loss")):
                ranking.append({"monitor_id": mid, "best_value": val})
            return {
                "metric": metric,
                "series": series_data,
                "latest": latest,
                "best": best,
                "ranking": ranking,
            }

    def __repr__(self) -> str:
        with self._lock:
            return f"<TrainingMonitor monitors={len(self._monitors)} alerts={len(self._alerts)}>"


# ============================================================
# 模型对比工具 - MODEL_COMPARATOR [清单 #46]
# ============================================================

@dataclass
class ModelInfo:
    """模型信息

    描述一个待对比的模型版本。
    """
    model_id: str
    name: str
    version: str
    metrics: Dict[str, float] = field(default_factory=dict)  # accuracy, loss, latency_ms, size_mb 等
    params: Dict[str, Any] = field(default_factory=dict)      # 参数量/层数/隐藏维度等
    config: Dict[str, Any] = field(default_factory=dict)      # 训练配置
    created_at: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "metrics": self.metrics,
            "params": self.params,
            "config": self.config,
            "created_at": self.created_at,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelInfo":
        return cls(
            model_id=d.get("model_id", ""),
            name=d.get("name", ""),
            version=d.get("version", ""),
            metrics=d.get("metrics", {}),
            params=d.get("params", {}),
            config=d.get("config", {}),
            created_at=d.get("created_at", ""),
            tags=d.get("tags", []),
        )


class ModelComparator:
    """模型对比工具

    功能:
    - 多模型并排对比
    - 指标并排表格 (accuracy / loss / latency / size)
    - 差异分析 (性能差异 / 参数差异 / 配置差异)
    - 简化 t 检验 (判断指标差异是否显著)
    - 基于对比结果推荐最优模型
    - 同一模型不同版本的指标趋势
    - 导出 JSON 对比报告
    """

    # 常用对比指标
    METRIC_ACCURACY = "accuracy"
    METRIC_LOSS = "loss"
    METRIC_LATENCY = "latency_ms"
    METRIC_SIZE = "size_mb"
    METRIC_F1 = "f1"

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._lock = threading.RLock()

    # ----------------------------------------------------------
    # 模型注册
    # ----------------------------------------------------------

    def register_model(self, model_id: str, name: str, version: str,
                       metrics: Dict[str, float] = None,
                       params: Dict[str, Any] = None,
                       config: Dict[str, Any] = None,
                       tags: List[str] = None,
                       created_at: str = "") -> str:
        """注册模型

        Args:
            model_id: 模型ID
            name: 模型名称
            version: 版本号
            metrics: 指标字典
            params: 参数信息
            config: 配置信息
            tags: 标签
            created_at: 创建时间

        Returns:
            model_id
        """
        if not created_at:
            created_at = datetime.now().isoformat()
        model = ModelInfo(
            model_id=model_id,
            name=name,
            version=version,
            metrics=metrics if metrics else {},
            params=params if params else {},
            config=config if config else {},
            tags=tags if tags else [],
            created_at=created_at,
        )
        with self._lock:
            self._models[model_id] = model
        return model_id

    def update_metrics(self, model_id: str,
                       metrics: Dict[str, float]) -> bool:
        """更新模型指标"""
        with self._lock:
            model = self._models.get(model_id)
            if not model:
                return False
            model.metrics.update(metrics)
            return True

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型"""
        return self._models.get(model_id)

    def list_models(self, name: str = None) -> List[ModelInfo]:
        """列出模型"""
        with self._lock:
            results = list(self._models.values())
        if name:
            results = [m for m in results if m.name == name]
        return results

    # ----------------------------------------------------------
    # 并排对比
    # ----------------------------------------------------------

    def compare(self, model_ids: List[str]) -> Dict[str, Any]:
        """多模型并排对比

        Args:
            model_ids: 待对比的模型ID列表

        Returns:
            {
                "models": 模型基本信息,
                "metrics_table": 指标并排表格,
                "metric_differences": 指标差异,
                "param_differences": 参数差异,
                "config_differences": 配置差异,
                "recommendation": 推荐建议,
            }
        """
        with self._lock:
            models = [self._models.get(mid) for mid in model_ids]
            models = [m for m in models if m is not None]
        if not models:
            return {"error": "无有效模型", "models": []}

        # 模型基本信息
        models_info = [
            {"model_id": m.model_id, "name": m.name, "version": m.version,
             "created_at": m.created_at}
            for m in models
        ]

        # 指标并排表格
        all_metrics = set()
        for m in models:
            all_metrics.update(m.metrics.keys())
        metrics_table = {}
        for metric in sorted(all_metrics):
            row = {}
            for m in models:
                row[m.model_id] = m.metrics.get(metric)
            metrics_table[metric] = row

        # 指标差异 (相对第一个模型)
        base = models[0]
        metric_diffs = {}
        for metric in sorted(all_metrics):
            base_val = base.metrics.get(metric)
            diff_row = {}
            for m in models[1:]:
                cur_val = m.metrics.get(metric)
                delta = None
                if isinstance(base_val, (int, float)) and isinstance(cur_val, (int, float)):
                    delta = cur_val - base_val
                diff_row[m.model_id] = {"value": cur_val, "delta": delta}
            metric_diffs[metric] = diff_row

        # 参数差异
        all_params = set()
        for m in models:
            all_params.update(m.params.keys())
        param_diffs = {}
        for pk in sorted(all_params):
            row = {m.model_id: m.params.get(pk) for m in models}
            values = list(row.values())
            if len(set(str(v) for v in values)) > 1:
                param_diffs[pk] = row

        # 配置差异
        all_config = set()
        for m in models:
            all_config.update(m.config.keys())
        config_diffs = {}
        for ck in sorted(all_config):
            row = {m.model_id: m.config.get(ck) for m in models}
            values = list(row.values())
            if len(set(str(v) for v in values)) > 1:
                config_diffs[ck] = row

        # 显著性检验
        significance = self._significance_test(models)

        # 推荐
        recommendation = self._recommend(models)

        return {
            "models": models_info,
            "metrics_table": metrics_table,
            "metric_differences": metric_diffs,
            "param_differences": param_diffs,
            "config_differences": config_diffs,
            "significance_test": significance,
            "recommendation": recommendation,
        }

    def _significance_test(self, models: List[ModelInfo]) -> Dict[str, Any]:
        """简化 t 检验

        对成对模型的 accuracy 指标做简化 t 检验。
        由于实际场景中通常只有单点指标 (非多次实验),
        这里使用指标差值与经验阈值比较来判断显著性。

        Returns:
            {metric: {pair: {"significant": bool, "delta": float, "note": str}}}
        """
        results = {}
        # 对 accuracy 和 loss 做检验
        test_metrics = [self.METRIC_ACCURACY, self.METRIC_LOSS]
        for metric in test_metrics:
            pair_results = {}
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    v1 = models[i].metrics.get(metric)
                    v2 = models[j].metrics.get(metric)
                    if not (isinstance(v1, (int, float)) and isinstance(v2, (int, float))):
                        continue
                    delta = v2 - v1
                    # 经验阈值: accuracy 差异 > 0.5% 视为显著
                    # loss 差异 > 1% 相对差异视为显著
                    threshold = 0.005 if metric == self.METRIC_ACCURACY else 0.01
                    if metric == self.METRIC_LOSS and v1 != 0:
                        rel_diff = abs(delta) / abs(v1)
                        significant = rel_diff > threshold
                    else:
                        significant = abs(delta) > threshold
                    pair_key = f"{models[i].model_id} vs {models[j].model_id}"
                    pair_results[pair_key] = {
                        "delta": round(delta, 6),
                        "significant": significant,
                        "note": "差异显著" if significant else "差异不显著",
                    }
            if pair_results:
                results[metric] = pair_results
        return results

    def _recommend(self, models: List[ModelInfo]) -> Dict[str, Any]:
        """基于对比结果推荐最优模型

        评分策略 (加权综合):
        - accuracy 越高越好 (权重 0.4)
        - loss 越低越好 (权重 0.3)
        - latency 越低越好 (权重 0.2)
        - size 越小越好 (权重 0.1)

        Returns:
            {"best_model_id": ..., "reason": ..., "ranking": [...]}
        """
        scored = []
        for m in models:
            acc = m.metrics.get(self.METRIC_ACCURACY, 0.0)
            loss = m.metrics.get(self.METRIC_LOSS, float("inf"))
            latency = m.metrics.get(self.METRIC_LATENCY, 0.0)
            size = m.metrics.get(self.METRIC_SIZE, 0.0)
            # 归一化评分 (0-1)
            score = 0.0
            if isinstance(acc, (int, float)):
                score += acc * 0.4
            if isinstance(loss, (int, float)) and loss > 0:
                score += (1.0 / (1.0 + loss)) * 0.3
            if isinstance(latency, (int, float)) and latency > 0:
                score += (1.0 / (1.0 + latency / 100.0)) * 0.2
            if isinstance(size, (int, float)) and size > 0:
                score += (1.0 / (1.0 + size / 1000.0)) * 0.1
            scored.append({
                "model_id": m.model_id,
                "name": m.name,
                "version": m.version,
                "score": round(score, 4),
                "accuracy": acc,
                "loss": loss,
                "latency_ms": latency,
                "size_mb": size,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0] if scored else None
        reason = ""
        if best:
            reason = (f"综合评分最高 ({best['score']}), "
                      f"accuracy={best['accuracy']}, loss={best['loss']}, "
                      f"latency={best['latency_ms']}ms, size={best['size_mb']}MB")
        return {
            "best_model_id": best["model_id"] if best else None,
            "best_name": best["name"] if best else None,
            "reason": reason,
            "ranking": scored,
        }

    # ----------------------------------------------------------
    # 历史趋势
    # ----------------------------------------------------------

    def version_trend(self, name: str, metric: str) -> Dict[str, Any]:
        """同一模型不同版本的指标趋势

        Args:
            name: 模型名称
            metric: 指标名

        Returns:
            {
                "name": 模型名,
                "metric": 指标名,
                "versions": [{version, value}, ...],
                "trend": "improving" / "declining" / "stable",
                "best_version": 最优版本,
            }
        """
        with self._lock:
            versions = [m for m in self._models.values() if m.name == name]
        if not versions:
            return {"name": name, "metric": metric, "versions": [], "trend": "unknown"}
        # 按 created_at 排序
        versions.sort(key=lambda m: m.created_at)
        trend_data = []
        for m in versions:
            val = m.metrics.get(metric)
            if isinstance(val, (int, float)):
                trend_data.append({"version": m.version, "value": val,
                                   "model_id": m.model_id})
        if not trend_data:
            return {"name": name, "metric": metric, "versions": [], "trend": "unknown"}
        # 趋势判断
        values = [d["value"] for d in trend_data]
        if len(values) < 2:
            trend = "stable"
        else:
            # accuracy 类指标: 越大越好; loss 类指标: 越小越好
            improving = metric in (self.METRIC_ACCURACY, self.METRIC_F1)
            first, last = values[0], values[-1]
            if improving:
                if last > first * 1.01:
                    trend = "improving"
                elif last < first * 0.99:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                if last < first * 0.99:
                    trend = "improving"
                elif last > first * 1.01:
                    trend = "declining"
                else:
                    trend = "stable"
        # 最优版本
        improving = metric in (self.METRIC_ACCURACY, self.METRIC_F1)
        best = max(trend_data, key=lambda d: d["value"]) if improving \
            else min(trend_data, key=lambda d: d["value"])
        return {
            "name": name,
            "metric": metric,
            "versions": trend_data,
            "trend": trend,
            "best_version": best["version"],
            "best_value": best["value"],
        }

    # ----------------------------------------------------------
    # 导出报告
    # ----------------------------------------------------------

    def export_report(self, model_ids: List[str],
                      path: str = None) -> Dict[str, Any]:
        """导出 JSON 对比报告

        Args:
            model_ids: 待对比的模型ID列表
            path: 导出文件路径 (不提供则仅返回字典)

        Returns:
            对比报告字典
        """
        report = self.compare(model_ids)
        report["exported_at"] = datetime.now().isoformat()
        report["model_count"] = len(model_ids)
        if path:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            except (OSError, IOError):
                pass
        return report

    # ----------------------------------------------------------
    # 持久化
    # ----------------------------------------------------------

    def save(self, path: str) -> bool:
        """保存模型库到 JSON"""
        with self._lock:
            data = {mid: m.to_dict() for mid, m in self._models.items()}
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except (OSError, IOError):
            return False

    def load(self, path: str) -> bool:
        """从 JSON 加载模型库"""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._models = {mid: ModelInfo.from_dict(d) for mid, d in data.items()}
            return True
        except (OSError, IOError, json.JSONDecodeError):
            return False

    def __len__(self) -> int:
        return len(self._models)

    def __repr__(self) -> str:
        return f"<ModelComparator models={len(self._models)}>"
