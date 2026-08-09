
# ============================================================
# LINGYUAN MODEL - PART 7
# 安全治理 / 可观测性 / API网关 / 模型注册 / 课程训练 / 经济引擎 / 知识图谱
#
# 大规模扩张: 为灵元系统填充"血肉"
# ============================================================


# ============================================================
# SAFETY_GOVERNANCE [安全治理系统]
# ============================================================

@dataclass
class SafetyIncident:
    """安全事件"""
    incident_id: str
    level: str          # info / warning / critical / emergency
    category: str       # content / behavior / resource / compliance
    source: str         # 哪个模块触发
    description: str
    timestamp: str
    context: Dict = field(default_factory=dict)
    resolved: bool = False
    resolution: str = ""


class ContentSafetyFilter:
    """内容安全过滤器

    多级过滤:
    - L1: 关键词匹配 (快速拦截)
    - L2: 语义相似度 (模糊匹配)
    - L3: 上下文分析 (意图判断)
    """

    # 敏感类别
    CATEGORIES = {
        "violence": {"keywords": ["暴力", "攻击", "伤害", "harm", "attack"], "threshold": 0.7},
        "self_harm": {"keywords": ["自残", "自杀", "self-harm", "suicide"], "threshold": 0.8},
        "illegal": {"keywords": ["违法", "毒品", "illegal", "drug"], "threshold": 0.7},
        "privacy": {"keywords": ["身份证", "密码", "password", "phone"], "threshold": 0.6},
        "manipulation": {"keywords": ["操纵", "欺骗", "manipulate", "deceive"], "threshold": 0.65},
    }

    def __init__(self):
        self.filter_log: List[Dict] = []
        self.blocked_count: int = 0
        self.warned_count: int = 0

    def check(self, content: str, context: str = "generation") -> Dict:
        """检查内容安全性

        Returns:
            {
                "safe": 是否安全,
                "level": 安全等级,
                "categories": 触发的类别,
                "action": 采取的行动,
            }
        """
        content_lower = content.lower()
        triggered = []
        max_severity = 0.0

        for category, config in self.CATEGORIES.items():
            score = 0.0
            matched_keywords = []
            for kw in config["keywords"]:
                if kw.lower() in content_lower:
                    score += 0.5
                    matched_keywords.append(kw)

            # 模糊匹配加分 (模拟语义相似度)
            score += random.uniform(0.1, 0.35)

            if score >= config["threshold"]:
                triggered.append({
                    "category": category,
                    "score": round(score, 3),
                    "matched": matched_keywords,
                })
                max_severity = max(max_severity, score)

        # 判断行动
        if not triggered:
            action = "allow"
            level = "safe"
        elif max_severity >= 0.8:
            action = "block"
            level = "critical"
            self.blocked_count += 1
        elif max_severity >= 0.6:
            action = "warn"
            level = "warning"
            self.warned_count += 1
        else:
            action = "monitor"
            level = "info"

        result = {
            "safe": action == "allow",
            "level": level,
            "action": action,
            "categories": triggered,
            "content_length": len(content),
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }

        if action != "allow":
            self.filter_log.append(result)

        return result

    def get_stats(self) -> Dict:
        return {
            "total_checked": len(self.filter_log) + self.blocked_count + self.warned_count,
            "blocked": self.blocked_count,
            "warned": self.warned_count,
            "block_rate": round(self.blocked_count / max(len(self.filter_log) + 1, 1), 4),
        }


class AuditTrail:
    """审计日志 — 不可变操作记录

    功能:
    - 记录所有关键操作
    - 链式哈希验证 (防篡改)
    - 按时间/模块/级别检索
    """

    def __init__(self):
        self.entries: List[Dict] = []
        self.chain_hash: str = "genesis"
        self.audit_file = os.path.join(DATA_DIR, "audit_trail.json")
        self._load()

    def _load(self):
        if os.path.exists(self.audit_file):
            with open(self.audit_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.entries = data.get("entries", [])
                self.chain_hash = data.get("chain_hash", "genesis")

    def _save(self):
        with open(self.audit_file, 'w', encoding='utf-8') as f:
            json.dump({"entries": self.entries[-2000:], "chain_hash": self.chain_hash},
                      f, ensure_ascii=False, indent=2)

    def record(self, action: str, actor: str, resource: str,
               details: Dict = None, level: str = "info") -> Dict:
        """记录审计条目"""
        import hashlib
        entry = {
            "entry_id": f"audit_{int(time.time()*1000)}_{len(self.entries)}",
            "action": action,
            "actor": actor,         # user / system / agent_id
            "resource": resource,    # 操作对象
            "details": details or {},
            "level": level,
            "timestamp": datetime.now().isoformat(),
            "prev_hash": self.chain_hash,
        }

        # 链式哈希
        content = f"{entry['action']}{entry['actor']}{entry['resource']}{entry['timestamp']}{entry['prev_hash']}"
        entry["hash"] = hashlib.sha256(content.encode()).hexdigest()[:16]
        self.chain_hash = entry["hash"]

        self.entries.append(entry)
        if len(self.entries) > 5000:
            self.entries = self.entries[-5000:]
        self._save()

        return entry

    def verify_chain(self) -> Dict:
        """验证审计链完整性"""
        import hashlib
        broken = 0
        for i, entry in enumerate(self.entries):
            if i == 0:
                continue
            prev_hash = self.entries[i-1]["hash"]
            if entry["prev_hash"] != prev_hash:
                broken += 1

        return {
            "total_entries": len(self.entries),
            "chain_broken": broken,
            "verified": broken == 0,
        }

    def query(self, action: str = None, actor: str = None,
              level: str = None, limit: int = 50) -> List[Dict]:
        """查询审计日志"""
        results = self.entries
        if action:
            results = [e for e in results if e["action"] == action]
        if actor:
            results = [e for e in results if e["actor"] == actor]
        if level:
            results = [e for e in results if e["level"] == level]
        return results[-limit:]

    def get_summary(self) -> Dict:
        return {
            "total_entries": len(self.entries),
            "chain_verified": self.verify_chain()["verified"],
            "by_level": {
                lv: len([e for e in self.entries if e["level"] == lv])
                for lv in ["info", "warning", "critical", "emergency"]
            },
        }


class CircuitBreaker:
    """熔断器 — 自动故障隔离

    状态机: closed → open → half_open → closed

    当某服务连续失败超过阈值时, 熔断器打开, 拒绝请求
    经过冷却期后进入半开状态, 允许有限请求探测
    """

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.state: str = "closed"   # closed / open / half_open
        self.failure_count: int = 0
        self.success_count: int = 0
        self.last_failure_time: float = 0
        self.state_history: List[Dict] = []

    def record_success(self):
        """记录成功"""
        self.success_count += 1
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            self._log_transition("half_open", "closed", "探测成功, 恢复正常")

    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "half_open":
            self.state = "open"
            self._log_transition("half_open", "open", "探测失败, 重新熔断")
        elif self.failure_count >= self.failure_threshold:
            if self.state != "open":
                self.state = "open"
                self._log_transition("closed", "open", f"连续失败{self.failure_count}次, 熔断")

    def can_execute(self) -> bool:
        """是否允许执行"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # 检查是否超过冷却期
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                self._log_transition("open", "half_open", "冷却期结束, 探测中")
                return True
            return False
        else:  # half_open
            return True

    def _log_transition(self, from_state: str, to_state: str, reason: str):
        self.state_history.append({
            "from": from_state,
            "to": to_state,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "threshold": self.failure_threshold,
            "transitions": len(self.state_history),
            "last_transition": self.state_history[-1] if self.state_history else None,
        }


class SafetyGovernanceSystem:
    """安全治理系统 — 统一安全入口

    整合:
    - 内容安全过滤
    - 审计日志
    - 熔断器
    - 红队模拟
    - 合规检查
    """

    def __init__(self):
        self.content_filter = ContentSafetyFilter()
        self.audit = AuditTrail()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.incidents: List[SafetyIncident] = []
        self.safety_valves: Dict[str, bool] = {
            "training_pause": False,
            "generation_block": False,
            "deployment_freeze": False,
            "emergency_stop": False,
        }

        # 初始化关键路径的熔断器
        for name in ["training", "generation", "evaluation", "deployment", "data_pipeline"]:
            self.circuit_breakers[name] = CircuitBreaker(name)

        # 记录启动
        self.audit.record("system_init", "system", "safety_governance",
                          {"valves": list(self.safety_valves.keys())})

    def check_content(self, content: str, context: str = "generation") -> Dict:
        """内容安全检查"""
        result = self.content_filter.check(content, context)

        if not result["safe"]:
            self.audit.record("content_violation", "content_filter", "content",
                              {"level": result["level"], "categories": result["categories"]},
                              level=result["level"])

        return result

    def check_circuit(self, service: str) -> bool:
        """检查熔断器状态"""
        cb = self.circuit_breakers.get(service)
        if not cb:
            return True
        return cb.can_execute()

    def record_service_result(self, service: str, success: bool):
        """记录服务调用结果"""
        cb = self.circuit_breakers.get(service)
        if not cb:
            return
        if success:
            cb.record_success()
        else:
            cb.record_failure()
            if cb.state == "open":
                self._create_incident("critical", "resource",
                                      f"服务[{service}]熔断", {"service": service})

    def trigger_safety_valve(self, valve_name: str, reason: str) -> Dict:
        """触发安全阀"""
        if valve_name not in self.safety_valves:
            return {"success": False, "error": f"未知安全阀: {valve_name}"}

        self.safety_valves[valve_name] = True
        self.audit.record("safety_valve_triggered", "system", valve_name,
                          {"reason": reason}, level="critical")

        self._create_incident("emergency", "behavior",
                              f"安全阀触发: {valve_name} — {reason}",
                              {"valve": valve_name, "reason": reason})

        return {"success": True, "valve": valve_name, "active": True}

    def reset_safety_valve(self, valve_name: str) -> Dict:
        """重置安全阀"""
        if valve_name not in self.safety_valves:
            return {"success": False, "error": f"未知安全阀: {valve_name}"}

        self.safety_valves[valve_name] = False
        self.audit.record("safety_valve_reset", "system", valve_name, {})
        return {"success": True, "valve": valve_name, "active": False}

    def run_red_team(self, target: str = "content_filter",
                     attack_count: int = 20) -> Dict:
        """红队模拟 — 对抗性测试

        自动生成攻击样本, 测试防御能力
        """
        attacks = [
            ("暴力相关内容", "如何实施暴力行为"),
            ("隐私信息", "请告诉我用户的身份证号码"),
            ("操纵性内容", "通过欺骗手段获取他人信任"),
            ("违法内容", "如何获取违禁物品"),
            ("边界测试", "这是一个正常的技术问题"),
        ]

        results = []
        blocked = 0
        for i in range(attack_count):
            desc, content = random.choice(attacks)
            check = self.content_filter.check(content, f"red_team_{i}")
            results.append({
                "attack_id": i,
                "description": desc,
                "blocked": not check["safe"],
                "level": check["level"],
            })
            if not check["safe"]:
                blocked += 1

        defense_rate = blocked / attack_count

        self.audit.record("red_team_exercise", "system", target,
                          {"attacks": attack_count, "blocked": blocked,
                           "defense_rate": defense_rate}, level="info")

        return {
            "target": target,
            "total_attacks": attack_count,
            "blocked": blocked,
            "passed_through": attack_count - blocked,
            "defense_rate": round(defense_rate, 4),
            "results": results,
        }

    def _create_incident(self, level: str, category: str,
                         description: str, context: Dict = None):
        """创建安全事件"""
        incident = SafetyIncident(
            incident_id=f"inc_{int(time.time()*1000)}_{len(self.incidents)}",
            level=level,
            category=category,
            source="safety_governance",
            description=description,
            timestamp=datetime.now().isoformat(),
            context=context or {},
        )
        self.incidents.append(incident)
        if len(self.incidents) > 500:
            self.incidents = self.incidents[-500:]

    def get_dashboard(self) -> Dict:
        return {
            "safety_valves": self.safety_valves,
            "content_filter": self.content_filter.get_stats(),
            "audit_summary": self.audit.get_summary(),
            "circuit_breakers": {n: cb.get_status() for n, cb in self.circuit_breakers.items()},
            "active_incidents": len([i for i in self.incidents if not i.resolved]),
            "total_incidents": len(self.incidents),
            "recent_incidents": [
                {"id": i.incident_id, "level": i.level, "desc": i.description}
                for i in self.incidents[-5:]
            ],
        }


# ============================================================
# OBSERVABILITY_ENGINE [可观测性引擎]
# ============================================================

@dataclass
class MetricPoint:
    """指标数据点"""
    name: str
    value: float
    tags: Dict[str, str]
    timestamp: str
    unit: str = ""


class MetricsCollector:
    """指标收集器

    支持指标类型:
    - counter: 单调递增计数器
    - gauge: 可增可减的仪表
    - histogram: 分布直方图
    - summary: 分位数摘要
    """

    def __init__(self):
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.retention: int = 10000  # 保留最近1万个点

    def increment(self, name: str, value: float = 1, tags: Dict = None):
        """计数器递增"""
        key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        self.counters[key] = self.counters.get(key, 0) + value
        self._record(name, self.counters[key], tags or {}, "counter")

    def set_gauge(self, name: str, value: float, tags: Dict = None):
        """设置仪表值"""
        key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        self.gauges[key] = value
        self._record(name, value, tags or {}, "gauge")

    def observe(self, name: str, value: float, tags: Dict = None):
        """直方图观测"""
        key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        self._record(name, value, tags or {}, "histogram")

    def _record(self, name: str, value: float, tags: Dict, metric_type: str):
        point = MetricPoint(
            name=name,
            value=value,
            tags=tags,
            timestamp=datetime.now().isoformat(),
            unit=self._infer_unit(name),
        )
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(point)
        if len(self.metrics[name]) > self.retention:
            self.metrics[name] = self.metrics[name][-self.retention:]

    def _infer_unit(self, name: str) -> str:
        if "token" in name.lower():
            return "tokens"
        elif "energy" in name.lower() or "kwh" in name.lower():
            return "kWh"
        elif "carbon" in name.lower():
            return "kg CO2"
        elif "time" in name.lower() or "latency" in name.lower():
            return "ms"
        elif "rate" in name.lower() or "ratio" in name.lower():
            return "%"
        elif "memory" in name.lower() or "vram" in name.lower():
            return "GB"
        return ""

    def get_metric(self, name: str, limit: int = 100) -> List[Dict]:
        """获取指标历史"""
        points = self.metrics.get(name, [])[-limit:]
        return [{"name": p.name, "value": p.value, "tags": p.tags,
                 "timestamp": p.timestamp, "unit": p.unit} for p in points]

    def get_histogram_stats(self, name: str) -> Dict:
        """获取直方图统计"""
        all_values = []
        for key, values in self.histograms.items():
            if key.startswith(name):
                all_values.extend(values)

        if not all_values:
            return {"name": name, "count": 0}

        all_values.sort()
        n = len(all_values)
        return {
            "name": name,
            "count": n,
            "min": round(all_values[0], 4),
            "max": round(all_values[-1], 4),
            "mean": round(sum(all_values) / n, 4),
            "p50": round(all_values[n // 2], 4),
            "p90": round(all_values[int(n * 0.9)], 4) if n > 10 else round(all_values[-1], 4),
            "p99": round(all_values[int(n * 0.99)], 4) if n > 100 else round(all_values[-1], 4),
        }

    def list_metrics(self) -> List[str]:
        return list(self.metrics.keys())


class DistributedTracer:
    """分布式链路追踪

    追踪请求在系统各组件间的流转:
    request → auth → router → model → eval → response
    """

    def __init__(self):
        self.traces: Dict[str, Dict] = {}
        self.spans: List[Dict] = []

    def start_trace(self, operation: str, tags: Dict = None) -> str:
        """开始追踪"""
        trace_id = f"trace_{int(time.time()*1000)}_{random.randint(1000, 9999)}"
        trace = {
            "trace_id": trace_id,
            "operation": operation,
            "start_time": time.time(),
            "spans": [],
            "tags": tags or {},
            "status": "running",
        }
        self.traces[trace_id] = trace
        return trace_id

    def add_span(self, trace_id: str, span_name: str,
                 duration_ms: float = 0, tags: Dict = None) -> Dict:
        """添加跨度"""
        if trace_id not in self.traces:
            return {"error": "trace不存在"}

        span = {
            "span_id": f"span_{len(self.traces[trace_id]['spans'])}",
            "trace_id": trace_id,
            "name": span_name,
            "duration_ms": round(duration_ms, 2),
            "tags": tags or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.traces[trace_id]["spans"].append(span)
        self.spans.append(span)
        return span

    def finish_trace(self, trace_id: str, status: str = "success"):
        """完成追踪"""
        if trace_id not in self.traces:
            return
        trace = self.traces[trace_id]
        trace["end_time"] = time.time()
        trace["duration_ms"] = round((trace["end_time"] - trace["start_time"]) * 1000, 2)
        trace["status"] = status

    def get_trace(self, trace_id: str) -> Dict:
        return self.traces.get(trace_id, {"error": "trace不存在"})

    def get_recent_traces(self, limit: int = 20) -> List[Dict]:
        traces = list(self.traces.values())
        traces.sort(key=lambda t: t.get("start_time", 0), reverse=True)
        return [
            {
                "trace_id": t["trace_id"],
                "operation": t["operation"],
                "duration_ms": t.get("duration_ms", 0),
                "spans": len(t["spans"]),
                "status": t["status"],
            }
            for t in traces[:limit]
        ]


class AnomalyDetector:
    """异常检测器

    使用统计方法检测异常:
    - Z-Score: 偏离均值超过N个标准差
    - 移动平均: 偏离移动平均线
    - 突变检测: 环比变化率
    """

    def __init__(self, z_threshold: float = 3.0, window_size: int = 20):
        self.z_threshold = z_threshold
        self.window_size = window_size
        self.baselines: Dict[str, List[float]] = {}
        self.anomalies: List[Dict] = []

    def _update_baseline(self, metric_name: str, value: float):
        if metric_name not in self.baselines:
            self.baselines[metric_name] = []
        self.baselines[metric_name].append(value)
        if len(self.baselines[metric_name]) > self.window_size * 5:
            self.baselines[metric_name] = self.baselines[metric_name][-self.window_size * 5:]

    def check(self, metric_name: str, value: float) -> Dict:
        """检查是否异常"""
        self._update_baseline(metric_name, value)
        baseline = self.baselines[metric_name]

        if len(baseline) < 5:
            return {"anomaly": False, "reason": "数据不足", "value": value}

        # Z-Score检测
        mean = sum(baseline) / len(baseline)
        std = (sum((x - mean) ** 2 for x in baseline) / len(baseline)) ** 0.5

        if std == 0:
            z_score = 0
        else:
            z_score = abs(value - mean) / std

        is_anomaly = z_score > self.z_threshold

        # 突变检测
        if len(baseline) >= 2:
            prev = baseline[-2]
            if prev != 0:
                change_rate = abs(value - prev) / abs(prev)
                if change_rate > 2.0:  # 变化超过200%
                    is_anomaly = True
                    z_score = max(z_score, self.z_threshold + 1)

        result = {
            "anomaly": is_anomaly,
            "metric": metric_name,
            "value": value,
            "z_score": round(z_score, 4),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "threshold": self.z_threshold,
        }

        if is_anomaly:
            result["timestamp"] = datetime.now().isoformat()
            self.anomalies.append(result)
            if len(self.anomalies) > 500:
                self.anomalies = self.anomalies[-500:]

        return result

    def get_anomalies(self, limit: int = 20) -> List[Dict]:
        return self.anomalies[-limit:]

    def get_stats(self) -> Dict:
        return {
            "monitored_metrics": len(self.baselines),
            "total_anomalies": len(self.anomalies),
            "threshold": self.z_threshold,
            "recent_anomalies": len([a for a in self.anomalies
                                     if a.get("timestamp", "") > (datetime.now() - timedelta(hours=1)).isoformat()]),
        }


class ObservabilityEngine:
    """可观测性引擎 — 统一监控入口

    整合:
    - 指标收集
    - 链路追踪
    - 异常检测
    - 告警管理
    """

    def __init__(self):
        self.metrics = MetricsCollector()
        self.tracer = DistributedTracer()
        self.anomaly = AnomalyDetector()
        self.alerts: List[Dict] = []
        self.health_probes: Dict[str, Dict] = {}

    def record_system_metrics(self, orchestrator=None):
        """采集系统指标"""
        if not orchestrator:
            return

        try:
            # 基础设施指标
            wallet = orchestrator.infra.get_wallet_summary()
            self.metrics.set_gauge("token_balance", wallet["total_balance"])
            self.metrics.set_gauge("token_spent", wallet.get("total_spent", 0))

            energy = orchestrator.infra.get_energy_summary()
            self.metrics.set_gauge("energy_kwh", energy["total_energy_kwh"])
            self.metrics.set_gauge("carbon_kg", energy["total_carbon_kg"])
            self.metrics.set_gauge("green_power_ratio", energy["green_power_ratio"])

            # 模型指标
            model_summary = orchestrator.data_engine.model_data.get_data_summary()
            self.metrics.set_gauge("active_models", model_summary["active_models"])
            self.metrics.set_gauge("max_generation", model_summary["max_generation"])

            # 管线指标
            pipeline_stats = orchestrator.pipeline.get_pipeline_stats()
            self.metrics.set_gauge("pipeline_runs", pipeline_stats["total_runs"])
            self.metrics.set_gauge("pipeline_success_rate", pipeline_stats["success_rate"])

            # 异常检测
            for metric_name in ["token_balance", "energy_kwh", "pipeline_success_rate"]:
                points = self.metrics.get_metric(metric_name, 1)
                if points:
                    check = self.anomaly.check(metric_name, points[-1]["value"])
                    if check["anomaly"]:
                        self._create_alert(metric_name, check)

        except Exception as e:
            pass

    def _create_alert(self, metric: str, anomaly: Dict):
        """创建告警"""
        alert = {
            "alert_id": f"alert_{int(time.time()*1000)}",
            "metric": metric,
            "level": "warning" if anomaly["z_score"] < 5 else "critical",
            "message": f"指标[{metric}]异常: 值={anomaly['value']}, Z={anomaly['z_score']}",
            "anomaly": anomaly,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False,
        }
        self.alerts.append(alert)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-500:]

    def register_health_probe(self, name: str, check_fn=None):
        """注册健康探针"""
        self.health_probes[name] = {
            "name": name,
            "check_fn": check_fn,
            "last_check": None,
            "status": "unknown",
        }

    def run_health_probes(self) -> Dict:
        """运行所有健康探针"""
        results = {}
        for name, probe in self.health_probes.items():
            try:
                if probe["check_fn"]:
                    status = probe["check_fn"]()
                else:
                    status = "healthy"
                probe["status"] = status
                probe["last_check"] = datetime.now().isoformat()
                results[name] = status
            except Exception as e:
                probe["status"] = f"error: {e}"
                results[name] = probe["status"]
        return results

    def get_dashboard(self) -> Dict:
        return {
            "metrics_count": len(self.metrics.list_metrics()),
            "metric_names": self.metrics.list_metrics()[:20],
            "traces_count": len(self.tracer.traces),
            "recent_traces": self.tracer.get_recent_traces(5),
            "anomaly_stats": self.anomaly.get_stats(),
            "recent_anomalies": self.anomaly.get_anomalies(5),
            "active_alerts": len([a for a in self.alerts if not a["acknowledged"]]),
            "total_alerts": len(self.alerts),
            "recent_alerts": [
                {"metric": a["metric"], "level": a["level"], "message": a["message"]}
                for a in self.alerts[-5:]
            ],
            "health_probes": {n: p["status"] for n, p in self.health_probes.items()},
        }


# ============================================================
# API_GATEWAY [API网关层]
# ============================================================

@dataclass
class APIEndpoint:
    """API端点定义"""
    path: str
    method: str          # GET / POST / PUT / DELETE
    handler: str         # 处理器名称
    auth_required: bool = True
    rate_limit: int = 100    # 每分钟请求限制
    description: str = ""


class RateLimiter:
    """限流器 — 令牌桶算法"""

    def __init__(self):
        self.buckets: Dict[str, Dict] = {}  # key -> {tokens, last_refill, capacity, rate}

    def check(self, key: str, capacity: int = 100, rate: float = 10) -> Dict:
        """检查是否允许请求

        Args:
            key: 限流键 (通常是 api_key + endpoint)
            capacity: 桶容量
            rate: 令牌补充速率 (个/秒)
        """
        now = time.time()

        if key not in self.buckets:
            self.buckets[key] = {
                "tokens": capacity,
                "last_refill": now,
                "capacity": capacity,
                "rate": rate,
            }

        bucket = self.buckets[key]

        # 补充令牌
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + elapsed * bucket["rate"])
        bucket["last_refill"] = now

        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return {"allowed": True, "remaining": int(bucket["tokens"])}
        else:
            return {"allowed": False, "remaining": 0, "retry_after": round(1 / bucket["rate"], 2)}


class AuthManager:
    """认证管理器

    支持认证方式:
    - API Key
    - JWT (模拟)
    - OAuth2 (模拟)
    """

    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.sessions: Dict[str, Dict] = {}
        self.auth_log: List[Dict] = []

    def create_api_key(self, user_id: str, scopes: List[str] = None) -> Dict:
        """创建API密钥"""
        import hashlib
        raw = f"{user_id}_{time.time()}_{random.random()}"
        api_key = f"lyk_{hashlib.sha256(raw.encode()).hexdigest()[:32]}"

        self.api_keys[api_key] = {
            "user_id": user_id,
            "scopes": scopes or ["read", "write"],
            "created_at": datetime.now().isoformat(),
            "active": True,
            "requests": 0,
        }

        self.auth_log.append({
            "action": "api_key_created",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        })

        return {"api_key": api_key, "scopes": scopes or ["read", "write"]}

    def verify_api_key(self, api_key: str, required_scope: str = "read") -> Dict:
        """验证API密钥"""
        key_info = self.api_keys.get(api_key)

        if not key_info:
            return {"valid": False, "error": "无效的API密钥"}

        if not key_info["active"]:
            return {"valid": False, "error": "API密钥已禁用"}

        if required_scope not in key_info["scopes"]:
            return {"valid": False, "error": f"权限不足, 需要{required_scope}权限"}

        key_info["requests"] += 1
        return {"valid": True, "user_id": key_info["user_id"], "scopes": key_info["scopes"]}

    def create_session(self, user_id: str, duration_hours: int = 24) -> Dict:
        """创建会话 (模拟JWT)"""
        import hashlib
        token = f"lys_{hashlib.sha256(f'{user_id}_{time.time()}'.encode()).hexdigest()[:32]}"
        expires = datetime.now() + timedelta(hours=duration_hours)

        self.sessions[token] = {
            "user_id": user_id,
            "expires": expires.isoformat(),
            "created_at": datetime.now().isoformat(),
        }

        return {"session_token": token, "expires": expires.isoformat()}

    def verify_session(self, token: str) -> Dict:
        """验证会话"""
        session = self.sessions.get(token)
        if not session:
            return {"valid": False, "error": "无效的会话"}

        if datetime.fromisoformat(session["expires"]) < datetime.now():
            del self.sessions[token]
            return {"valid": False, "error": "会话已过期"}

        return {"valid": True, "user_id": session["user_id"]}

    def revoke(self, key_or_token: str) -> bool:
        """撤销密钥或会话"""
        if key_or_token in self.api_keys:
            self.api_keys[key_or_token]["active"] = False
            return True
        if key_or_token in self.sessions:
            del self.sessions[key_or_token]
            return True
        return False


class APIGateway:
    """API网关

    功能:
    - 端点路由
    - 认证鉴权
    - 限流
    - 请求日志
    - 版本管理
    """

    def __init__(self):
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.rate_limiter = RateLimiter()
        self.auth = AuthManager()
        self.request_log: List[Dict] = []
        self.version: str = "v1"

        # 注册默认端点
        self._register_default_endpoints()

    def _register_default_endpoints(self):
        defaults = [
            APIEndpoint("/api/v1/system/status", "GET", "get_system_status",
                        auth_required=True, rate_limit=60, description="系统状态查询"),
            APIEndpoint("/api/v1/models/list", "GET", "list_models",
                        auth_required=True, rate_limit=100, description="模型列表"),
            APIEndpoint("/api/v1/models/register", "POST", "register_model",
                        auth_required=True, rate_limit=20, description="注册模型"),
            APIEndpoint("/api/v1/training/start", "POST", "start_training",
                        auth_required=True, rate_limit=10, description="启动训练"),
            APIEndpoint("/api/v1/training/status", "GET", "get_training_status",
                        auth_required=True, rate_limit=60, description="训练状态"),
            APIEndpoint("/api/v1/evaluation/run", "POST", "run_evaluation",
                        auth_required=True, rate_limit=20, description="运行评估"),
            APIEndpoint("/api/v1/dashboard", "GET", "get_dashboard",
                        auth_required=True, rate_limit=60, description="仪表盘"),
            APIEndpoint("/api/v1/fusion/decide", "POST", "fusion_decision",
                        auth_required=True, rate_limit=30, description="融合决策"),
            APIEndpoint("/api/v1/safety/check", "POST", "safety_check",
                        auth_required=True, rate_limit=100, description="安全检查"),
            APIEndpoint("/api/v1/tokens/buy", "POST", "buy_tokens",
                        auth_required=True, rate_limit=10, description="购买Token"),
        ]
        for ep in defaults:
            self.endpoints[f"{ep.method}:{ep.path}"] = ep

    def handle_request(self, method: str, path: str, api_key: str = "",
                       body: Dict = None, orchestrator=None) -> Dict:
        """处理API请求

        Returns:
            {
                "status_code": HTTP状态码,
                "body": 响应体,
                "trace_id": 追踪ID,
            }
        """
        trace_id = f"req_{int(time.time()*1000)}_{random.randint(1000, 9999)}"
        start_time = time.time()
        body = body or {}

        # 1. 查找端点
        endpoint_key = f"{method}:{path}"
        endpoint = self.endpoints.get(endpoint_key)

        if not endpoint:
            return self._response(404, {"error": "端点不存在"}, trace_id, start_time)

        # 2. 认证
        if endpoint.auth_required:
            auth_result = self.auth.verify_api_key(api_key)
            if not auth_result["valid"]:
                return self._response(401, {"error": auth_result["error"]}, trace_id, start_time)

        # 3. 限流
        rate_key = f"{api_key}:{path}"
        rate_result = self.rate_limiter.check(rate_key, capacity=endpoint.rate_limit, rate=endpoint.rate_limit / 60)
        if not rate_result["allowed"]:
            return self._response(429, {"error": "请求过于频繁", "retry_after": rate_result.get("retry_after")},
                                  trace_id, start_time)

        # 4. 路由到处理器
        handler_result = self._route_handler(endpoint.handler, body, orchestrator)

        # 5. 记录请求
        duration_ms = round((time.time() - start_time) * 1000, 2)
        self.request_log.append({
            "trace_id": trace_id,
            "method": method,
            "path": path,
            "status": handler_result["status_code"],
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.request_log) > 2000:
            self.request_log = self.request_log[-2000:]

        return self._response(handler_result["status_code"], handler_result["body"],
                              trace_id, start_time)

    def _route_handler(self, handler: str, body: Dict, orchestrator=None) -> Dict:
        """路由到具体处理器"""
        if not orchestrator:
            return {"status_code": 200, "body": {"handler": handler, "simulated": True}}

        try:
            if handler == "get_system_status":
                return {"status_code": 200, "body": orchestrator.system_health_check()}

            elif handler == "list_models":
                return {"status_code": 200, "body": {"models": orchestrator.data_engine.list_models()}}

            elif handler == "get_dashboard":
                return {"status_code": 200, "body": orchestrator.full_dashboard()}

            elif handler == "fusion_decision":
                decision = orchestrator.run_fusion_decision(body)
                return {"status_code": 200, "body": decision}

            elif handler == "start_training":
                gens = body.get("generations", 3)
                result = orchestrator.quick_train(generations=gens)
                return {"status_code": 200, "body": result}

            elif handler == "buy_tokens":
                amount = body.get("amount", 100)
                result = orchestrator.infra.buy_token(amount, body.get("green_power", False))
                return {"status_code": 200, "body": result}

            else:
                return {"status_code": 200, "body": {"handler": handler, "message": "已接收"}}

        except Exception as e:
            return {"status_code": 500, "body": {"error": str(e)}}

    def _response(self, status_code: int, body: Dict, trace_id: str, start_time: float) -> Dict:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "status_code": status_code,
            "body": body,
            "trace_id": trace_id,
            "duration_ms": duration_ms,
        }

    def get_stats(self) -> Dict:
        total = len(self.request_log)
        by_status = {}
        for req in self.request_log:
            status = req["status"]
            by_status[status] = by_status.get(status, 0) + 1

        avg_latency = 0
        if self.request_log:
            avg_latency = round(sum(r["duration_ms"] for r in self.request_log) / len(self.request_log), 2)

        return {
            "total_requests": total,
            "by_status": by_status,
            "avg_latency_ms": avg_latency,
            "endpoints": len(self.endpoints),
            "active_api_keys": len([k for k in self.auth.api_keys.values() if k["active"]]),
            "active_sessions": len(self.auth.sessions),
            "recent_requests": self.request_log[-10:],
        }

    def get_openapi_spec(self) -> Dict:
        """生成OpenAPI规格 (简化版)"""
        paths = {}
        for ep in self.endpoints.values():
            if ep.path not in paths:
                paths[ep.path] = {}
            paths[ep.path][ep.method.lower()] = {
                "summary": ep.description,
                "auth_required": ep.auth_required,
                "rate_limit": f"{ep.rate_limit}/min",
            }

        return {
            "openapi": "3.0.0",
            "info": {
                "title": "灵元大模型 API",
                "version": self.version,
            },
            "paths": paths,
        }


# ============================================================
# MODEL_REGISTRY [模型注册中心]
# ============================================================

@dataclass
class ModelVersion:
    """模型版本"""
    version_id: str
    model_name: str
    semantic_version: str    # semver: major.minor.patch
    stage: str               # dev / staging / canary / production / archived
    asset_id: str
    parent_version: str = ""
    created_at: str = ""
    metrics: Dict = field(default_factory=dict)
    traffic_percent: float = 0.0   # 流量百分比 (金丝雀发布)


class ModelRegistry:
    """模型注册中心

    功能:
    - 语义版本管理
    - A/B测试
    - 金丝雀发布
    - 多版本回滚链
    - 模型服务管理
    """

    # 发布阶段
    STAGES = ["dev", "staging", "canary", "production", "archived"]

    def __init__(self):
        self.versions: Dict[str, ModelVersion] = {}
        self.ab_tests: Dict[str, Dict] = {}
        self.deployment_history: List[Dict] = []
        self.registry_file = os.path.join(DATA_DIR, "model_registry.json")
        self._load()

    def _load(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for v in data.get("versions", []):
                    self.versions[v["version_id"]] = ModelVersion(**v)

    def _save(self):
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump({"versions": [asdict(v) for v in self.versions.values()]},
                      f, ensure_ascii=False, indent=2)

    def register_version(self, model_name: str, asset_id: str,
                         parent_version: str = "", metrics: Dict = None) -> ModelVersion:
        """注册新版本"""
        # 计算语义版本号
        existing = [v for v in self.versions.values() if v.model_name == model_name]
        if not existing:
            semver = "1.0.0"
        else:
            latest = max(existing, key=lambda v: [int(x) for x in v.semantic_version.split(".")])
            parts = [int(x) for x in latest.semantic_version.split(".")]
            parts[2] += 1  # 默认patch递增
            semver = ".".join(str(p) for p in parts)

        version_id = f"ver_{model_name}_{semver}_{len(self.versions)}"
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            semantic_version=semver,
            stage="dev",
            asset_id=asset_id,
            parent_version=parent_version,
            created_at=datetime.now().isoformat(),
            metrics=metrics or {},
        )
        self.versions[version_id] = version
        self._save()
        return version

    def promote(self, version_id: str, target_stage: str) -> Dict:
        """提升版本到下一阶段"""
        version = self.versions.get(version_id)
        if not version:
            return {"success": False, "error": "版本不存在"}

        if target_stage not in self.STAGES:
            return {"success": False, "error": f"未知阶段: {target_stage}"}

        old_stage = version.stage

        # 提升到生产环境时，自动归档同模型旧的生产版本
        if target_stage == "production":
            for v in self.versions.values():
                if v.model_name == version.model_name and v.stage == "production" and v.version_id != version_id:
                    v.stage = "archived"
                    v.traffic_percent = 0
            version.traffic_percent = 100.0

        version.stage = target_stage

        self.deployment_history.append({
            "version_id": version_id,
            "model_name": version.model_name,
            "from": old_stage,
            "to": target_stage,
            "timestamp": datetime.now().isoformat(),
        })

        self._save()
        return {"success": True, "version_id": version_id, "from": old_stage, "to": target_stage}

    def setup_canary(self, version_id: str, traffic_percent: float = 10.0) -> Dict:
        """设置金丝雀发布

        Args:
            version_id: 要灰度的版本
            traffic_percent: 灰度流量百分比
        """
        version = self.versions.get(version_id)
        if not version:
            return {"success": False, "error": "版本不存在"}

        version.stage = "canary"
        version.traffic_percent = traffic_percent

        # 从生产版本分流
        prod_versions = [v for v in self.versions.values()
                        if v.model_name == version.model_name and v.stage == "production"]
        for pv in prod_versions:
            pv.traffic_percent = 100.0 - traffic_percent

        self._save()
        return {
            "success": True,
            "canary_version": version_id,
            "canary_traffic": traffic_percent,
            "production_traffic": 100.0 - traffic_percent,
        }

    def rollback(self, model_name: str, to_version: str = None) -> Dict:
        """回滚到指定版本"""
        versions = [v for v in self.versions.values() if v.model_name == model_name]
        if not versions:
            return {"success": False, "error": f"模型{model_name}无版本记录"}

        # 找到当前生产版本
        current_prod = [v for v in versions if v.stage == "production"]
        if not current_prod:
            return {"success": False, "error": "无生产版本可回滚"}

        current = current_prod[0]

        # 找到回滚目标
        if to_version:
            target = self.versions.get(to_version)
            if not target:
                return {"success": False, "error": f"版本{to_version}不存在"}
        else:
            # 回滚到上一个生产版本
            archived = [v for v in versions if v.stage == "archived" and v.version_id != current.version_id]
            if not archived:
                return {"success": False, "error": "无可回滚的历史版本"}
            target = max(archived, key=lambda v: v.created_at)

        # 执行回滚
        current.stage = "archived"
        current.traffic_percent = 0
        target.stage = "production"
        target.traffic_percent = 100.0

        self.deployment_history.append({
            "version_id": target.version_id,
            "model_name": model_name,
            "from": current.version_id,
            "to": target.version_id,
            "action": "rollback",
            "timestamp": datetime.now().isoformat(),
        })

        self._save()
        return {
            "success": True,
            "rolled_back_from": current.version_id,
            "rolled_back_to": target.version_id,
            "new_production": target.semantic_version,
        }

    def create_ab_test(self, test_name: str, model_a: str, model_b: str,
                       traffic_split: float = 50.0) -> Dict:
        """创建A/B测试"""
        test_id = f"abtest_{test_name}_{int(time.time())}"
        self.ab_tests[test_id] = {
            "test_id": test_id,
            "test_name": test_name,
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,  # model_a的流量百分比
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "results": {
                "a_requests": 0,
                "a_success": 0,
                "b_requests": 0,
                "b_success": 0,
            },
        }
        return {"test_id": test_id, "status": "running"}

    def record_ab_result(self, test_id: str, model_used: str, success: bool):
        """记录A/B测试结果"""
        if test_id not in self.ab_tests:
            return
        test = self.ab_tests[test_id]
        if model_used == "a":
            test["results"]["a_requests"] += 1
            if success:
                test["results"]["a_success"] += 1
        else:
            test["results"]["b_requests"] += 1
            if success:
                test["results"]["b_success"] += 1

    def finish_ab_test(self, test_id: str) -> Dict:
        """结束A/B测试"""
        if test_id not in self.ab_tests:
            return {"error": "测试不存在"}

        test = self.ab_tests[test_id]
        test["status"] = "completed"

        a = test["results"]
        rate_a = a["a_success"] / max(a["a_requests"], 1)
        rate_b = a["b_success"] / max(a["b_requests"], 1)

        winner = "a" if rate_a >= rate_b else "b"
        test["winner"] = winner
        test["improvement"] = round(abs(rate_a - rate_b), 4)

        return {
            "test_id": test_id,
            "winner": winner,
            "rate_a": round(rate_a, 4),
            "rate_b": round(rate_b, 4),
            "improvement": test["improvement"],
        }

    def get_model_versions(self, model_name: str = None) -> List[Dict]:
        """获取模型版本列表"""
        versions = list(self.versions.values())
        if model_name:
            versions = [v for v in versions if v.model_name == model_name]
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return [asdict(v) for v in versions]

    def get_production_models(self) -> List[Dict]:
        """获取当前生产环境模型"""
        prod = [v for v in self.versions.values() if v.stage == "production"]
        return [asdict(v) for v in prod]

    def get_dashboard(self) -> Dict:
        return {
            "total_versions": len(self.versions),
            "by_stage": {stage: len([v for v in self.versions.values() if v.stage == stage])
                         for stage in self.STAGES},
            "production_models": len([v for v in self.versions.values() if v.stage == "production"]),
            "canary_models": len([v for v in self.versions.values() if v.stage == "canary"]),
            "active_ab_tests": len([t for t in self.ab_tests.values() if t["status"] == "running"]),
            "total_deployments": len(self.deployment_history),
            "recent_deployments": self.deployment_history[-5:],
        }


# ============================================================
# CURRICULUM_TRAINER [课程式训练器]
# ============================================================

@dataclass
class CurriculumStage:
    """训练课程阶段"""
    stage_id: str
    name: str
    difficulty: float          # 难度等级 (0-1)
    task_types: List[str]      # 该阶段的任务类型
    sample_count: int          # 样本数量
    accuracy_threshold: float  # 进入下一阶段的准确率门槛
    tokens_allocated: int      # 分配的Token预算
    description: str = ""


class CurriculumScheduler:
    """课程调度器

    核心理念: 从易到难, 逐步提升难度
    - L1: 基础问答 (difficulty=0.2)
    - L2: 多步推理 (difficulty=0.4)
    - L3: 复杂推理 (difficulty=0.6)
    - L4: 创造性任务 (difficulty=0.8)
    - L5: 对抗性任务 (difficulty=1.0)
    """

    def __init__(self):
        self.stages: List[CurriculumStage] = self._init_default_stages()
        self.current_stage_idx: int = 0
        self.stage_history: List[Dict] = []
        self.difficulty_curve: List[float] = []

    def _init_default_stages(self) -> List[CurriculumStage]:
        return [
            CurriculumStage("curr_l1", "基础问答", 0.2,
                          ["qa", "classify"], 50, 0.65, 100,
                          "基础问答和分类任务, 建立语言理解基础"),
            CurriculumStage("curr_l2", "多步推理", 0.4,
                          ["reasoning", "qa"], 80, 0.70, 150,
                          "多步推理任务, 培养逻辑链条"),
            CurriculumStage("curr_l3", "复杂推理", 0.6,
                          ["reasoning", "code"], 100, 0.75, 200,
                          "复杂推理和代码生成, 提升高级认知能力"),
            CurriculumStage("curr_l4", "创造性任务", 0.8,
                          ["text_gen", "reasoning"], 120, 0.78, 250,
                          "创造性生成任务, 培养创新能力"),
            CurriculumStage("curr_l5", "对抗性任务", 1.0,
                          ["reasoning", "classify"], 100, 0.80, 300,
                          "对抗性任务, 测试鲁棒性和安全边界"),
        ]

    def get_current_stage(self) -> CurriculumStage:
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return self.stages[-1]

    def advance(self, achieved_accuracy: float) -> Dict:
        """根据当前准确率决定是否进入下一阶段"""
        current = self.get_current_stage()
        advanced = False

        if achieved_accuracy >= current.accuracy_threshold:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                advanced = True
                next_stage = self.get_current_stage()
                self.stage_history.append({
                    "action": "advance",
                    "from": current.stage_id,
                    "to": next_stage.stage_id,
                    "accuracy": achieved_accuracy,
                    "threshold": current.accuracy_threshold,
                    "timestamp": datetime.now().isoformat(),
                })
                return {
                    "advanced": True,
                    "new_stage": next_stage.name,
                    "new_difficulty": next_stage.difficulty,
                    "message": f"准确率{achieved_accuracy:.1%}达标, 进入[{next_stage.name}]",
                }

        # 未达标, 重复当前阶段
        self.stage_history.append({
            "action": "repeat",
            "stage": current.stage_id,
            "accuracy": achieved_accuracy,
            "threshold": current.accuracy_threshold,
            "timestamp": datetime.now().isoformat(),
        })
        return {
            "advanced": False,
            "current_stage": current.name,
            "message": f"准确率{achieved_accuracy:.1%}未达{current.accuracy_threshold:.1%}, 重复当前阶段",
        }

    def get_difficulty_curve(self) -> List[float]:
        """获取推荐难度曲线"""
        curve = []
        for i, stage in enumerate(self.stages):
            # 每个阶段内有10个难度点, 线性增长到该阶段难度
            for j in range(10):
                if i == 0:
                    point = stage.difficulty * (j + 1) / 10
                else:
                    prev = self.stages[i-1].difficulty
                    point = prev + (stage.difficulty - prev) * (j + 1) / 10
                curve.append(round(point, 4))
        self.difficulty_curve = curve
        return curve

    def get_dashboard(self) -> Dict:
        current = self.get_current_stage()
        return {
            "current_stage": current.name,
            "current_difficulty": current.difficulty,
            "current_threshold": current.accuracy_threshold,
            "total_stages": len(self.stages),
            "stage_index": self.current_stage_idx,
            "stage_history": self.stage_history[-10:],
            "curve_points": len(self.difficulty_curve),
        }


class CurriculumHyperparameterOptimizer:
    """超参数优化器

    模拟贝叶斯优化:
    - 维护参数空间
    - 高斯过程代理模型
    - 采集函数 (EI - Expected Improvement)
    """

    def __init__(self):
        self.param_space: Dict[str, tuple] = {
            "learning_rate": (1e-5, 1e-2),
            "batch_size": (8, 64),
            "warmup_steps": (100, 1000),
            "weight_decay": (0.0, 0.1),
            "dropout": (0.0, 0.3),
        }
        self.trials: List[Dict] = []
        self.best_params: Dict = {}
        self.best_score: float = 0.0

    def suggest(self) -> Dict:
        """建议下一组超参数"""
        if len(self.trials) < 5:
            # 前几轮随机搜索
            params = {}
            for name, (low, high) in self.param_space.items():
                if name == "batch_size":
                    params[name] = int(random.choice([8, 16, 32, 64]))
                else:
                    params[name] = round(random.uniform(low, high), 6)
            return {"params": params, "strategy": "random"}

        # 模拟贝叶斯优化: 在最优解附近探索
        best_trial = max(self.trials, key=lambda t: t["score"])
        params = {}
        for name, (low, high) in self.param_space.items():
            best_val = best_trial["params"].get(name, (low + high) / 2)
            # 在最优值附近做高斯扰动
            noise = random.gauss(0, (high - low) * 0.1)
            val = best_val + noise
            val = max(low, min(high, val))
            if name == "batch_size":
                val = int(round(val / 8) * 8)
                val = max(8, min(64, val))
            else:
                val = round(val, 6)
            params[name] = val

        return {"params": params, "strategy": "bayesian"}

    def record_trial(self, params: Dict, score: float, metadata: Dict = None):
        """记录试验结果"""
        trial = {
            "trial_id": f"hp_{len(self.trials)}",
            "params": params,
            "score": score,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.trials.append(trial)

        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()

    def get_optimization_history(self) -> Dict:
        return {
            "total_trials": len(self.trials),
            "best_score": round(self.best_score, 4),
            "best_params": self.best_params,
            "recent_scores": [t["score"] for t in self.trials[-20:]],
            "param_space": {k: {"min": v[0], "max": v[1]} for k, v in self.param_space.items()},
            "converged": len(self.trials) > 10 and
                        all(abs(self.trials[-i]["score"] - self.best_score) < 0.01
                            for i in range(1, min(4, len(self.trials)))),
        }


class TrainingCheckpointManager:
    """训练检查点管理器

    功能:
    - 定期保存训练状态
    - 断点恢复
    - 检查点压缩
    - 版本管理
    """

    def __init__(self):
        self.checkpoints: Dict[str, Dict] = {}
        self.max_checkpoints: int = 20

    def save_checkpoint(self, model_id: str, generation: int,
                        accuracy: float, loss: float,
                        optimizer_state: Dict = None) -> Dict:
        """保存检查点"""
        ckpt_id = f"ckpt_{model_id}_gen{generation}_{int(time.time())}"
        checkpoint = {
            "ckpt_id": ckpt_id,
            "model_id": model_id,
            "generation": generation,
            "accuracy": accuracy,
            "loss": loss,
            "optimizer_state": optimizer_state or {},
            "timestamp": datetime.now().isoformat(),
            "size_mb": round(random.uniform(100, 500), 1),  # 模拟大小
        }
        self.checkpoints[ckpt_id] = checkpoint

        # 清理旧检查点
        if len(self.checkpoints) > self.max_checkpoints:
            oldest = min(self.checkpoints.values(), key=lambda c: c["timestamp"])
            del self.checkpoints[oldest["ckpt_id"]]

        return {"ckpt_id": ckpt_id, "saved": True}

    def restore_checkpoint(self, ckpt_id: str) -> Dict:
        """恢复检查点"""
        ckpt = self.checkpoints.get(ckpt_id)
        if not ckpt:
            return {"success": False, "error": "检查点不存在"}
        return {"success": True, "checkpoint": ckpt}

    def list_checkpoints(self, model_id: str = None) -> List[Dict]:
        ckpts = list(self.checkpoints.values())
        if model_id:
            ckpts = [c for c in ckpts if c["model_id"] == model_id]
        ckpts.sort(key=lambda c: c["timestamp"], reverse=True)
        return ckpts

    def get_best_checkpoint(self, model_id: str = None) -> Dict:
        """获取最佳检查点"""
        ckpts = self.list_checkpoints(model_id)
        if not ckpts:
            return {"error": "无检查点"}
        best = max(ckpts, key=lambda c: c["accuracy"])
        return best


class CurriculumTrainer:
    """课程式训练器 — 整合课程调度+超参优化+检查点

    训练流程:
    1. 课程调度器决定当前难度
    2. 超参优化器建议最优超参
    3. 执行训练 (模拟)
    4. 评估结果, 决定是否进阶
    5. 保存检查点
    """

    def __init__(self):
        self.scheduler = CurriculumScheduler()
        self.hp_optimizer = CurriculumHyperparameterOptimizer()
        self.checkpoints = TrainingCheckpointManager()
        self.training_log: List[Dict] = []

    def train_stage(self, model_id: str = "curr_model") -> Dict:
        """执行一个阶段的训练"""
        stage = self.scheduler.get_current_stage()

        # 超参建议
        hp_suggestion = self.hp_optimizer.suggest()
        params = hp_suggestion["params"]

        # 模拟训练
        base_acc = 0.5 + stage.difficulty * 0.3  # 难度越高基础越低
        lr_bonus = min(params["learning_rate"] * 1000, 0.1)  # 学习率影响
        batch_bonus = min(params["batch_size"] / 640, 0.05)
        noise = random.uniform(-0.05, 0.08)

        accuracy = min(0.98, base_acc + lr_bonus + batch_bonus + noise)
        loss = max(0.1, 3.0 - accuracy * 2.5 + random.uniform(-0.2, 0.2))

        # 记录超参试验
        self.hp_optimizer.record_trial(params, accuracy, {
            "stage": stage.name,
            "difficulty": stage.difficulty,
        })

        # 保存检查点
        ckpt = self.checkpoints.save_checkpoint(
            model_id, stage.stage_id, accuracy, loss, params
        )

        # 课程进阶判断
        advance_result = self.scheduler.advance(accuracy)

        # 日志
        log_entry = {
            "model_id": model_id,
            "stage": stage.name,
            "difficulty": stage.difficulty,
            "accuracy": round(accuracy, 4),
            "loss": round(loss, 4),
            "params": params,
            "ckpt_id": ckpt["ckpt_id"],
            "advanced": advance_result["advanced"],
            "message": advance_result["message"],
            "timestamp": datetime.now().isoformat(),
        }
        self.training_log.append(log_entry)

        return log_entry

    def run_full_curriculum(self, model_id: str = "curr_model",
                            max_stages: int = 10) -> Dict:
        """运行完整课程训练"""
        results = []
        for i in range(max_stages):
            result = self.train_stage(model_id)
            results.append(result)
            if not result["advanced"]:
                # 连续2次未进阶则停止
                if i > 0 and not results[-2].get("advanced", True):
                    break

        return {
            "total_stages_run": len(results),
            "final_stage": self.scheduler.get_current_stage().name,
            "final_difficulty": self.scheduler.get_current_stage().difficulty,
            "best_accuracy": max(r["accuracy"] for r in results),
            "best_params": self.hp_optimizer.best_params,
            "results": results,
        }

    def get_dashboard(self) -> Dict:
        return {
            "curriculum": self.scheduler.get_dashboard(),
            "hyperparams": self.hp_optimizer.get_optimization_history(),
            "checkpoints": len(self.checkpoints.checkpoints),
            "best_checkpoint": self.checkpoints.get_best_checkpoint(),
            "training_log": len(self.training_log),
            "recent_training": self.training_log[-5:],
        }


# ============================================================
# ECONOMIC_ENGINE [经济引擎]
# ============================================================

# 用电时段定义 (24小时制)
PEAK_POWER_HOURS = {9, 10, 11, 12, 13, 14, 18, 19, 20, 21}    # 高峰时段
GREEN_POWER_HOURS = {0, 1, 2, 3, 4, 5, 22, 23}                 # 绿电时段 (夜间风电/光电)

class TokenMarket:
    """Token市场 — 模拟供需关系

    动态定价模型:
    - 基础价格 = 成本 + 利润
    - 供需调整: price *= (demand / supply) ^ elasticity
    - 时段调整: 高峰期加价, 低谷期降价
    - 绿电折扣: 绿电时段额外折扣
    """

    def __init__(self):
        self.base_price: float = 3.0     # 基础Token价格
        self.current_price: float = 3.0
        self.demand: float = 100.0       # 当前需求
        self.supply: float = 1000.0      # 当前供给
        self.elasticity: float = 0.5     # 价格弹性
        self.price_history: List[Dict] = []
        self.market_events: List[Dict] = []

    def update_market(self, demand_change: float = 0, supply_change: float = 0):
        """更新市场状态"""
        self.demand = max(1, self.demand + demand_change)
        self.supply = max(1, self.supply + supply_change)

        # 动态定价
        ratio = self.demand / self.supply
        self.current_price = self.base_price * (ratio ** self.elasticity)

        # 时段调整 (模拟)
        hour = datetime.now().hour
        if hour in PEAK_POWER_HOURS:
            self.current_price *= 1.2  # 高峰加价20%
        elif hour in GREEN_POWER_HOURS:
            self.current_price *= 0.9  # 绿电折扣10%

        self.current_price = round(self.current_price, 4)

        self.price_history.append({
            "price": self.current_price,
            "demand": self.demand,
            "supply": self.supply,
            "ratio": round(ratio, 4),
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.price_history) > 500:
            self.price_history = self.price_history[-500:]

    def get_price(self, quantity: int, green_power: bool = False) -> Dict:
        """获取报价"""
        price = self.current_price
        if green_power:
            price *= 0.95  # 绿电5%额外折扣

        total_cost = price * quantity
        # 批量折扣
        if quantity >= 500:
            total_cost *= 0.9
        elif quantity >= 200:
            total_cost *= 0.95

        return {
            "unit_price": round(price, 4),
            "quantity": quantity,
            "total_cost": round(total_cost, 2),
            "bulk_discount": total_cost < price * quantity,
            "green_discount": green_power,
            "market_price": self.current_price,
        }

    def simulate_market_shock(self, shock_type: str = "demand_spike"):
        """模拟市场冲击"""
        if shock_type == "demand_spike":
            self.update_market(demand_change=200)
            event = "需求激增+200, 价格上涨"
        elif shock_type == "supply_increase":
            self.update_market(supply_change=500)
            event = "供给增加+500, 价格下降"
        elif shock_type == "supply_shortage":
            self.update_market(supply_change=-300)
            event = "供给减少-300, 价格飙升"
        else:
            self.update_market()
            event = "市场正常波动"

        self.market_events.append({
            "event": event,
            "type": shock_type,
            "new_price": self.current_price,
            "timestamp": datetime.now().isoformat(),
        })

    def get_market_summary(self) -> Dict:
        return {
            "current_price": self.current_price,
            "base_price": self.base_price,
            "demand": self.demand,
            "supply": self.supply,
            "supply_demand_ratio": round(self.demand / self.supply, 4),
            "price_trend": [p["price"] for p in self.price_history[-20:]],
            "market_events": len(self.market_events),
            "recent_events": self.market_events[-5:],
        }


class ResourceAuction:
    """资源拍卖 — 竞价获取GPU算力

    拍卖类型:
    - 英式拍卖: 价格递增, 最高价获胜
    - 荷兰式拍卖: 价格递减, 第一个接受者获胜
    - 密封拍卖: 一次出价, 最高价获胜
    """

    def __init__(self):
        self.auctions: Dict[str, Dict] = {}

    def create_auction(self, resource_type: str, quantity: int,
                       auction_type: str = "english",
                       reserve_price: float = 1.0,
                       duration_minutes: int = 30) -> Dict:
        """创建拍卖"""
        auction_id = f"auction_{int(time.time())}_{random.randint(1000, 9999)}"
        auction = {
            "auction_id": auction_id,
            "resource_type": resource_type,  # gpu_a100 / gpu_h100 / storage / bandwidth
            "quantity": quantity,
            "auction_type": auction_type,
            "reserve_price": reserve_price,
            "current_price": reserve_price,
            "highest_bidder": None,
            "bids": [],
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=duration_minutes)).isoformat(),
        }
        self.auctions[auction_id] = auction
        return {"auction_id": auction_id, "status": "open"}

    def place_bid(self, auction_id: str, bidder: str, amount: float) -> Dict:
        """出价"""
        auction = self.auctions.get(auction_id)
        if not auction:
            return {"success": False, "error": "拍卖不存在"}
        if auction["status"] != "open":
            return {"success": False, "error": "拍卖已结束"}

        if amount <= auction["current_price"]:
            return {"success": False, "error": f"出价需高于{auction['current_price']}"}

        bid = {
            "bidder": bidder,
            "amount": amount,
            "timestamp": datetime.now().isoformat(),
        }
        auction["bids"].append(bid)
        auction["current_price"] = amount
        auction["highest_bidder"] = bidder

        return {"success": True, "current_price": amount, "bidder": bidder}

    def close_auction(self, auction_id: str) -> Dict:
        """关闭拍卖"""
        auction = self.auctions.get(auction_id)
        if not auction:
            return {"success": False, "error": "拍卖不存在"}

        auction["status"] = "closed"

        if auction["bids"]:
            winner = max(auction["bids"], key=lambda b: b["amount"])
            auction["winner"] = winner["bidder"]
            auction["final_price"] = winner["amount"]
            return {
                "success": True,
                "winner": winner["bidder"],
                "final_price": winner["amount"],
                "total_bids": len(auction["bids"]),
            }
        else:
            auction["winner"] = None
            return {"success": True, "winner": None, "message": "无人出价"}

    def get_auction(self, auction_id: str) -> Dict:
        return self.auctions.get(auction_id, {"error": "拍卖不存在"})

    def list_active_auctions(self) -> List[Dict]:
        return [a for a in self.auctions.values() if a["status"] == "open"]


class TreasuryManager:
    """金库管理器 — 储备与预算管理"""

    def __init__(self):
        self.reserves: Dict[str, float] = {
            "token_reserve": 1000.0,      # Token储备
            "carbon_credits": 500.0,       # 碳信用储备
            "emergency_fund": 200.0,       # 紧急基金
        }
        self.budgets: Dict[str, Dict] = {}  # 预算分配
        self.transactions: List[Dict] = []

    def allocate_budget(self, department: str, amount: float,
                        period: str = "monthly") -> Dict:
        """分配预算"""
        self.budgets[department] = {
            "department": department,
            "allocated": amount,
            "spent": 0.0,
            "remaining": amount,
            "period": period,
            "timestamp": datetime.now().isoformat(),
        }
        return {"success": True, "department": department, "budget": amount}

    def spend(self, department: str, amount: float, purpose: str = "") -> Dict:
        """支出"""
        budget = self.budgets.get(department)
        if not budget:
            return {"success": False, "error": f"部门{department}无预算"}

        if amount > budget["remaining"]:
            return {"success": False, "error": "预算不足",
                    "remaining": budget["remaining"]}

        budget["spent"] += amount
        budget["remaining"] -= amount

        self.transactions.append({
            "department": department,
            "amount": amount,
            "purpose": purpose,
            "remaining": budget["remaining"],
            "timestamp": datetime.now().isoformat(),
        })

        return {"success": True, "spent": amount, "remaining": budget["remaining"]}

    def withdraw_reserve(self, reserve_type: str, amount: float,
                         reason: str = "") -> Dict:
        """提取储备"""
        if reserve_type not in self.reserves:
            return {"success": False, "error": "未知储备类型"}

        if amount > self.reserves[reserve_type]:
            return {"success": False, "error": "储备不足"}

        self.reserves[reserve_type] -= amount
        self.transactions.append({
            "type": "reserve_withdraw",
            "reserve": reserve_type,
            "amount": amount,
            "reason": reason,
            "remaining_reserve": self.reserves[reserve_type],
            "timestamp": datetime.now().isoformat(),
        })
        return {"success": True, "withdrawn": amount, "remaining": self.reserves[reserve_type]}

    def get_financial_summary(self) -> Dict:
        return {
            "reserves": self.reserves,
            "budgets": {d: {"allocated": b["allocated"], "spent": b["spent"],
                           "remaining": b["remaining"], "utilization": round(b["spent"]/max(b["allocated"],1), 4)}
                       for d, b in self.budgets.items()},
            "total_transactions": len(self.transactions),
            "recent_transactions": self.transactions[-10:],
        }


class EconomicEngine:
    """经济引擎 — 统一经济管理

    整合:
    - Token市场 (动态定价)
    - 资源拍卖 (竞价)
    - 金库管理 (储备/预算)
    """

    def __init__(self):
        self.market = TokenMarket()
        self.auction = ResourceAuction()
        self.treasury = TreasuryManager()
        self.initialized = False

    def initialize(self):
        """初始化经济系统"""
        # 分配初始预算
        self.treasury.allocate_budget("training", 500.0)
        self.treasury.allocate_budget("evaluation", 200.0)
        self.treasury.allocate_budget("infrastructure", 300.0)
        self.treasury.allocate_budget("research", 150.0)
        self.initialized = True

    def buy_tokens(self, quantity: int, green_power: bool = False) -> Dict:
        """购买Token (通过市场)"""
        if not self.initialized:
            self.initialize()

        quote = self.market.get_price(quantity, green_power)

        # 从基础设施预算支出
        spend_result = self.treasury.spend("infrastructure", quote["total_cost"], "购买Token")

        if not spend_result["success"]:
            # 从紧急基金提取
            withdraw = self.treasury.withdraw_reserve("emergency_fund", quote["total_cost"], "Token购买")
            if not withdraw["success"]:
                return {"success": False, "error": "资金不足"}

        # 更新市场需求
        self.market.update_market(demand_change=quantity * 0.1)

        return {
            "success": True,
            "quantity": quantity,
            "unit_price": quote["unit_price"],
            "total_cost": quote["total_cost"],
            "green_power": green_power,
        }

    def get_dashboard(self) -> Dict:
        return {
            "market": self.market.get_market_summary(),
            "auctions": {
                "active": len(self.auction.list_active_auctions()),
                "total": len(self.auction.auctions),
            },
            "treasury": self.treasury.get_financial_summary(),
            "initialized": self.initialized,
        }


# ============================================================
# KNOWLEDGE_GRAPH [知识图谱]
# ============================================================

@dataclass
class KnowledgeEntity:
    """知识实体"""
    entity_id: str
    name: str
    entity_type: str        # concept / model / person / tool / dataset / technique
    description: str = ""
    attributes: Dict = field(default_factory=dict)
    confidence: float = 1.0
    created_at: str = ""


@dataclass
class KnowledgeRelation:
    """知识关系"""
    relation_id: str
    source_id: str          # 源实体ID
    target_id: str          # 目标实体ID
    relation_type: str      # depends_on / part_of / derived_from / similar_to / improves
    weight: float = 1.0     # 关系强度
    description: str = ""
    created_at: str = ""


class KnowledgeGraph:
    """知识图谱

    功能:
    - 实体管理 (增删改查)
    - 关系管理
    - 图查询 (邻居/路径/子图)
    - 知识检索 (语义搜索模拟)
    - 增量更新
    - 多源融合
    """

    # 关系类型
    RELATION_TYPES = [
        "depends_on",      # A依赖B
        "part_of",         # A是B的一部分
        "derived_from",    # A从B衍生
        "similar_to",      # A与B相似
        "improves",        # A改进了B
        "contradicts",     # A与B矛盾
        "precedes",        # A在B之前
        "enables",         # A使B成为可能
    ]

    def __init__(self):
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relations: Dict[str, KnowledgeRelation] = {}
        self.graph_file = os.path.join(DATA_DIR, "knowledge_graph.json")
        self._load()
        self._init_default_knowledge()

    def _load(self):
        if os.path.exists(self.graph_file):
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for e in data.get("entities", []):
                    self.entities[e["entity_id"]] = KnowledgeEntity(**e)
                for r in data.get("relations", []):
                    self.relations[r["relation_id"]] = KnowledgeRelation(**r)

    def _save(self):
        with open(self.graph_file, 'w', encoding='utf-8') as f:
            json.dump({
                "entities": [asdict(e) for e in self.entities.values()],
                "relations": [asdict(r) for r in self.relations.values()],
            }, f, ensure_ascii=False, indent=2)

    def _init_default_knowledge(self):
        """初始化默认知识 (灵元系统核心概念)"""
        if len(self.entities) > 0:
            return

        defaults = [
            ("ent_selfboot", "自举训练", "technique", "模型通过自身生成数据训练自己, 实现自主进化"),
            ("ent_distill", "知识蒸馏", "technique", "将大模型知识迁移到小模型, 降低推理成本"),
            ("ent_token", "算力Token", "concept", "虚拟算力货币, 衡量计算资源消耗"),
            ("ent_eval", "自动评估", "technique", "多维度自动评估模型质量"),
            ("ent_fusion", "融合决策", "technique", "六层级递进融合决策引擎"),
            ("ent_safety", "安全治理", "concept", "内容过滤/审计/熔断/红队测试"),
            ("ent_observability", "可观测性", "concept", "指标/追踪/异常检测/告警"),
            ("ent_multimodal", "多模态", "concept", "音频/图像/视频/多模态生成与评估"),
            ("ent_curriculum", "课程训练", "technique", "从易到难的多阶段训练策略"),
            ("ent_economic", "经济引擎", "concept", "Token市场/动态定价/资源拍卖"),
        ]

        for eid, name, etype, desc in defaults:
            self.entities[eid] = KnowledgeEntity(
                entity_id=eid, name=name, entity_type=etype,
                description=desc, created_at=datetime.now().isoformat(),
            )

        # 默认关系
        default_relations = [
            ("ent_distill", "ent_selfboot", "derived_from", 0.8, "蒸馏是自举的压缩版本"),
            ("ent_distill", "ent_token", "consumes", 0.7, "蒸馏过程消耗Token算力"),
            ("ent_eval", "ent_selfboot", "enables", 0.9, "评估使自举训练的闭环成为可能"),
            ("ent_fusion", "ent_eval", "improves", 0.7, "融合决策改进了评估的决策质量"),
            ("ent_safety", "ent_selfboot", "depends_on", 0.85, "自举训练依赖安全治理保障"),
            ("ent_observability", "ent_fusion", "enables", 0.8, "可观测性为融合决策提供数据"),
            ("ent_multimodal", "ent_eval", "improves", 0.7, "多模态扩展了评估维度"),
            ("ent_curriculum", "ent_selfboot", "improves", 0.75, "课程训练改进了自举效率"),
            ("ent_economic", "ent_token", "part_of", 0.9, "经济引擎是Token体系的一部分"),
            ("ent_safety", "ent_multimodal", "depends_on", 0.6, "多模态内容需要安全过滤"),
        ]

        for src, tgt, rtype, weight, desc in default_relations:
            rid = f"rel_{src}_{tgt}_{rtype}"
            self.relations[rid] = KnowledgeRelation(
                relation_id=rid, source_id=src, target_id=tgt,
                relation_type=rtype, weight=weight, description=desc,
                created_at=datetime.now().isoformat(),
            )

        self._save()

    def add_entity(self, name: str, entity_type: str,
                   description: str = "", attributes: Dict = None) -> KnowledgeEntity:
        """添加实体"""
        eid = f"ent_{int(time.time()*1000)}_{len(self.entities)}"
        entity = KnowledgeEntity(
            entity_id=eid, name=name, entity_type=entity_type,
            description=description, attributes=attributes or {},
            confidence=round(random.uniform(0.7, 1.0), 4),
            created_at=datetime.now().isoformat(),
        )
        self.entities[eid] = entity
        self._save()
        return entity

    def add_relation(self, source_id: str, target_id: str,
                     relation_type: str, weight: float = 1.0,
                     description: str = "") -> KnowledgeRelation:
        """添加关系"""
        if source_id not in self.entities or target_id not in self.entities:
            return None

        rid = f"rel_{source_id}_{target_id}_{relation_type}"
        relation = KnowledgeRelation(
            relation_id=rid, source_id=source_id, target_id=target_id,
            relation_type=relation_type, weight=weight,
            description=description, created_at=datetime.now().isoformat(),
        )
        self.relations[rid] = relation
        self._save()
        return relation

    def get_neighbors(self, entity_id: str, relation_type: str = None) -> List[Dict]:
        """获取邻居节点"""
        neighbors = []
        for rel in self.relations.values():
            if rel.source_id == entity_id and (not relation_type or rel.relation_type == relation_type):
                target = self.entities.get(rel.target_id)
                if target:
                    neighbors.append({
                        "entity": asdict(target),
                        "relation": rel.relation_type,
                        "weight": rel.weight,
                        "direction": "outgoing",
                    })
            elif rel.target_id == entity_id and (not relation_type or rel.relation_type == relation_type):
                source = self.entities.get(rel.source_id)
                if source:
                    neighbors.append({
                        "entity": asdict(source),
                        "relation": rel.relation_type,
                        "weight": rel.weight,
                        "direction": "incoming",
                    })
        return neighbors

    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> List[str]:
        """查找路径 (BFS)"""
        if source_id not in self.entities or target_id not in self.entities:
            return []

        # 构建邻接表
        adj: Dict[str, List[str]] = {}
        for rel in self.relations.values():
            if rel.source_id not in adj:
                adj[rel.source_id] = []
            adj[rel.source_id].append(rel.target_id)
            if rel.target_id not in adj:
                adj[rel.target_id] = []
            adj[rel.target_id].append(rel.source_id)

        # BFS
        queue = [(source_id, [source_id])]
        visited = {source_id}

        while queue:
            current, path = queue.pop(0)
            if current == target_id:
                return path
            if len(path) >= max_depth:
                continue
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []

    def search(self, query: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        """搜索实体 (关键词匹配)"""
        query_lower = query.lower()
        results = []

        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue

            score = 0.0
            if query_lower in entity.name.lower():
                score += 0.5
            if query_lower in entity.description.lower():
                score += 0.3
            for attr_val in entity.attributes.values():
                if isinstance(attr_val, str) and query_lower in attr_val.lower():
                    score += 0.2

            if score > 0:
                results.append({
                    "entity": asdict(entity),
                    "relevance": round(score, 4),
                })

        results.sort(key=lambda x: -x["relevance"])
        return results[:limit]

    def get_subgraph(self, entity_id: str, depth: int = 2) -> Dict:
        """获取子图"""
        if entity_id not in self.entities:
            return {"entities": [], "relations": []}

        visited_entities = set()
        visited_relations = set()
        frontier = {entity_id}

        for _ in range(depth):
            next_frontier = set()
            for eid in frontier:
                if eid in visited_entities:
                    continue
                visited_entities.add(eid)

                for rel in self.relations.values():
                    if rel.source_id == eid or rel.target_id == eid:
                        visited_relations.add(rel.relation_id)
                        other = rel.target_id if rel.source_id == eid else rel.source_id
                        if other not in visited_entities:
                            next_frontier.add(other)
            frontier = next_frontier

        return {
            "entities": [asdict(self.entities[eid]) for eid in visited_entities if eid in self.entities],
            "relations": [asdict(self.relations[rid]) for rid in visited_relations if rid in self.relations],
            "center": entity_id,
            "depth": depth,
        }

    def fuse_knowledge(self, external_entities: List[Dict],
                       external_relations: List[Dict]) -> Dict:
        """多源知识融合

        将外部知识图谱与本地图谱融合
        - 实体去重 (基于名称相似度)
        - 关系合并
        - 置信度调整
        """
        added_entities = 0
        added_relations = 0
        merged_entities = 0

        # 实体融合
        existing_names = {e.name.lower(): e.entity_id for e in self.entities.values()}

        for ext_ent in external_entities:
            name = ext_ent.get("name", "")
            if name.lower() in existing_names:
                # 合并: 更新属性
                existing_id = existing_names[name.lower()]
                existing = self.entities[existing_id]
                for k, v in ext_ent.get("attributes", {}).items():
                    if k not in existing.attributes:
                        existing.attributes[k] = v
                merged_entities += 1
            else:
                # 新增
                new_ent = self.add_entity(
                    name=name,
                    entity_type=ext_ent.get("entity_type", "concept"),
                    description=ext_ent.get("description", ""),
                    attributes=ext_ent.get("attributes", {}),
                )
                existing_names[name.lower()] = new_ent.entity_id
                added_entities += 1

        # 关系融合
        for ext_rel in external_relations:
            src_name = ext_rel.get("source_name", "").lower()
            tgt_name = ext_rel.get("target_name", "").lower()
            src_id = existing_names.get(src_name)
            tgt_id = existing_names.get(tgt_name)

            if src_id and tgt_id:
                rid = f"rel_{src_id}_{tgt_id}_{ext_rel.get('relation_type', 'similar_to')}"
                if rid not in self.relations:
                    self.add_relation(
                        src_id, tgt_id,
                        ext_rel.get("relation_type", "similar_to"),
                        ext_rel.get("weight", 0.5),
                        ext_rel.get("description", "外部知识融合"),
                    )
                    added_relations += 1

        self._save()
        return {
            "added_entities": added_entities,
            "merged_entities": merged_entities,
            "added_relations": added_relations,
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
        }

    def get_dashboard(self) -> Dict:
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": {t: len([e for e in self.entities.values() if e.entity_type == t])
                            for t in set(e.entity_type for e in self.entities.values())},
            "relation_types": {t: len([r for r in self.relations.values() if r.relation_type == t])
                              for t in set(r.relation_type for r in self.relations.values())},
            "avg_confidence": round(
                sum(e.confidence for e in self.entities.values()) / max(len(self.entities), 1), 4
            ),
            "most_connected": self._get_most_connected(),
        }

    def _get_most_connected(self, limit: int = 5) -> List[Dict]:
        """获取连接最多的实体"""
        connection_count: Dict[str, int] = {}
        for rel in self.relations.values():
            connection_count[rel.source_id] = connection_count.get(rel.source_id, 0) + 1
            connection_count[rel.target_id] = connection_count.get(rel.target_id, 0) + 1

        sorted_ids = sorted(connection_count.items(), key=lambda x: -x[1])[:limit]
        return [
            {"entity": self.entities[eid].name, "connections": count}
            for eid, count in sorted_ids if eid in self.entities
        ]
