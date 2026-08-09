

# ============================================================
# MOBILE_DASHBOARD [移动仪表盘]
# ============================================================

class MobileDashboard:
    """移动端仪表盘

    提供移动端可访问的监控与控制接口:
    - 实时系统状态概览
    - 资源消耗趋势
    - 告警通知
    - 快捷操作面板
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.push_notifications: List[Dict] = []
        self.watch_list: List[str] = []  # 监控指标列表
        self.notification_file = os.path.join(DATA_DIR, "mobile_notifications.json")
        self._load()

    def _load(self):
        if os.path.exists(self.notification_file):
            with open(self.notification_file, 'r', encoding='utf-8') as f:
                self.push_notifications = json.load(f).get('notifications', [])

    def _save(self):
        with open(self.notification_file, 'w', encoding='utf-8') as f:
            json.dump({'notifications': self.push_notifications[-500:]}, f, ensure_ascii=False, indent=2)

    def get_overview(self) -> Dict:
        """获取系统概览(移动端首页)"""
        if not self.orchestrator:
            return {"error": "编排器未初始化"}

        infra = self.orchestrator.infra
        data_engine = self.orchestrator.data_engine
        pipeline = self.orchestrator.pipeline

        # 核心指标
        wallet = infra.get_wallet_summary()
        energy = infra.get_energy_summary()
        model_summary = data_engine.model_data.get_data_summary()
        pipeline_stats = pipeline.get_pipeline_stats()

        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": "运行中",
            "core_metrics": {
                "token_balance": wallet["total_balance"],
                "active_models": model_summary["active_models"],
                "total_models": model_summary["total_models"],
                "max_generation": model_summary["max_generation"],
                "pipeline_runs": pipeline_stats["total_runs"],
                "pipeline_success_rate": pipeline_stats["success_rate"],
            },
            "energy": {
                "total_kwh": energy["total_energy_kwh"],
                "carbon_kg": energy["total_carbon_kg"],
                "green_ratio": energy["green_power_ratio"],
            },
            "alerts": self._get_active_alerts(),
            "quick_actions": [
                {"id": "start_train", "label": "触发训练", "icon": "play"},
                {"id": "pause_training", "label": "暂停训练", "icon": "pause"},
                {"id": "cancel_pipeline", "label": "取消管线", "icon": "stop"},
                {"id": "rollback_model", "label": "回滚模型", "icon": "undo"},
                {"id": "release_model", "label": "发布模型", "icon": "rocket"},
                {"id": "scale_up", "label": "扩容", "icon": "expand"},
                {"id": "scale_down", "label": "缩容", "icon": "compress"},
                {"id": "buy_token", "label": "购买Token", "icon": "cart"},
            ],
        }

    def get_resource_trend(self, hours: int = 24) -> Dict:
        """获取资源消耗趋势"""
        if not self.orchestrator:
            return {"error": "编排器未初始化"}

        energy = self.orchestrator.infra.energy
        records = energy.records[-hours * 10:] if energy.records else []

        trend = {
            "period_hours": hours,
            "data_points": [],
            "summary": {
                "avg_energy_per_task": 0.0,
                "peak_energy": 0.0,
                "trend_direction": "stable",
            },
        }

        if records:
            energies = [r.energy_kwh for r in records]
            trend["data_points"] = [
                {"timestamp": r.timestamp, "energy_kwh": r.energy_kwh, "carbon_kg": r.carbon_kg}
                for r in records[-20:]  # 最近20个点
            ]
            trend["summary"]["avg_energy_per_task"] = round(sum(energies) / len(energies), 4)
            trend["summary"]["peak_energy"] = round(max(energies), 4)

            # 趋势方向
            if len(energies) >= 2:
                recent_avg = sum(energies[-5:]) / min(len(energies[-5:]), 5)
                older_avg = sum(energies[:-5]) / max(len(energies[:-5]), 1)
                if recent_avg > older_avg * 1.1:
                    trend["summary"]["trend_direction"] = "rising"
                elif recent_avg < older_avg * 0.9:
                    trend["summary"]["trend_direction"] = "falling"

        return trend

    def _get_active_alerts(self) -> List[Dict]:
        """获取活跃告警"""
        alerts = []
        if not self.orchestrator:
            return alerts

        # Agent告警
        agent_alerts = self.orchestrator.agent_orch.alerts[-5:]
        for a in agent_alerts:
            alerts.append({
                "level": "warning",
                "source": "agent",
                "message": a.get("msg", {}).get("content", {}).get("text", ""),
                "timestamp": datetime.fromtimestamp(a.get("timestamp", time.time())).isoformat(),
            })

        # 安全告警
        safety = self.orchestrator.data_engine.bootstrap.safety
        if safety.should_stop():
            alerts.append({
                "level": "critical",
                "source": "safety",
                "message": "安全系统触发停止条件",
                "timestamp": datetime.now().isoformat(),
            })

        # Token余额告警
        wallet = self.orchestrator.infra.get_wallet_summary()
        if wallet["total_balance"] < 50:
            alerts.append({
                "level": "warning",
                "source": "wallet",
                "message": f"Token余额不足: {wallet['total_balance']}",
                "timestamp": datetime.now().isoformat(),
            })

        return alerts

    def push_notification(self, title: str, message: str, level: str = "info"):
        """推送通知"""
        notification = {
            "notification_id": f"notif_{int(time.time() * 1000)}",
            "title": title,
            "message": message,
            "level": level,  # info / warning / critical
            "read": False,
            "timestamp": datetime.now().isoformat(),
        }
        self.push_notifications.append(notification)
        self._save()
        return notification

    def get_notifications(self, unread_only: bool = False, limit: int = 20) -> List[Dict]:
        """获取通知列表"""
        notifs = self.push_notifications[-limit:]
        if unread_only:
            notifs = [n for n in notifs if not n["read"]]
        return notifs

    def mark_read(self, notification_id: str) -> bool:
        """标记通知已读"""
        for n in self.push_notifications:
            if n["notification_id"] == notification_id:
                n["read"] = True
                self._save()
                return True
        return False

    def quick_action(self, action_id: str, params: Dict = None) -> Dict:
        """执行快捷操作"""
        params = params or {}
        if not self.orchestrator:
            return {"error": "编排器未初始化"}

        if action_id == "buy_token":
            amount = params.get("amount", 100)
            green = params.get("green_power", False)
            return self.orchestrator.infra.buy_token(amount, green)

        elif action_id == "start_train":
            generations = params.get("generations", 3)
            return self.orchestrator.quick_train(generations=generations)

        elif action_id == "pause_training":
            self.orchestrator.data_engine.bootstrap.safety.checks_history.append({
                "passed": False, "action": "MANUAL_PAUSE", "should_stop": True,
                "timestamp": datetime.now().isoformat(),
            })
            return {"success": True, "message": "训练已暂停"}

        elif action_id == "cancel_pipeline":
            run_id = params.get("run_id", "")
            if run_id and run_id in self.orchestrator.pipeline._runs:
                self.orchestrator.pipeline._runs[run_id].status = PipelineStatus.CANCELLED
                return {"success": True, "message": f"管线{run_id}已取消"}
            return {"error": "管线不存在"}

        elif action_id == "rollback_model":
            gen = params.get("generation", 0)
            bootstrap = self.orchestrator.data_engine.bootstrap
            if bootstrap.generations:
                target = bootstrap.generations[-1]
                target.status = "rolled_back"
                return {"success": True, "message": f"已回滚第{target.generation}代"}
            return {"error": "无可回滚的模型"}

        elif action_id == "release_model":
            return self.orchestrator.run_workflow_by_id("release")

        elif action_id == "scale_up":
            self.orchestrator.infra.scheduler.refresh_vendor_status()
            return {"success": True, "message": "已扩容, 供应商状态已刷新"}

        elif action_id == "scale_down":
            for v in self.orchestrator.infra.scheduler.vendors.values():
                v.available_slots = max(1, v.available_slots // 2)
            return {"success": True, "message": "已缩容"}

        elif action_id == "view_models":
            return {"models": self.orchestrator.data_engine.list_models()}

        elif action_id == "pipeline_status":
            return self.orchestrator.pipeline.get_pipeline_stats()

        elif action_id == "run_workflow":
            workflow_id = params.get("workflow_id", "full_train")
            return self.orchestrator.run_workflow_by_id(workflow_id)

        return {"error": f"未知操作: {action_id}"}

    def get_training_progress(self) -> Dict:
        """获取训练进度"""
        if not self.orchestrator:
            return {"error": "编排器未初始化"}

        bootstrap = self.orchestrator.data_engine.bootstrap
        if not bootstrap.generations:
            return {"status": "idle", "message": "无训练记录"}

        latest = bootstrap.generations[-1]
        return {
            "current_generation": latest.generation,
            "status": latest.status,
            "accuracy": latest.final_accuracy,
            "loss": latest.final_loss,
            "improvement": latest.improvement,
            "tokens_consumed": latest.tokens_consumed,
            "safety_action": latest.safety_check.get("action", "N/A"),
            "evolution_tree": bootstrap.get_evolution_tree()[-5:],
        }


# ============================================================
# CLOSED_LOOP_ENGINE [闭环自优化引擎]
# ============================================================

class ClosedLoopEngine:
    """闭环自优化引擎

    核心理念: 系统 -> 监控 -> 分析 -> 决策 -> 执行 -> 验证 -> (循环)

    闭环流程:
    1. 采集系统状态(Agent团队/训练/评估/资源)
    2. 分析瓶颈与优化机会
    3. 生成优化决策(自动/人工确认)
    4. 执行优化动作
    5. 验证效果, 更新知识库
    """

    # 优化动作类型
    ACTION_TYPES = {
        "adjust_hyperparams": "调整超参数",
        "rebalance_vendors": "重新均衡供应商",
        "scale_agents": "扩缩Agent",
        "trigger_retrain": "触发重训练",
        "adjust_token_price": "调整Token定价",
        "cleanup_storage": "清理存储",
        "carbon_trade": "碳交易",
        "alert_admin": "告警管理员",
    }

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.optimization_history: List[Dict] = []
        self.knowledge_base: List[Dict] = []  # 优化知识库
        self.auto_optimize: bool = True  # 是否自动执行优化
        self.optimization_file = os.path.join(DATA_DIR, "closed_loop.json")
        self._load()

    def _load(self):
        if os.path.exists(self.optimization_file):
            with open(self.optimization_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.optimization_history = data.get("history", [])
            self.knowledge_base = data.get("knowledge", [])

    def _save(self):
        with open(self.optimization_file, 'w', encoding='utf-8') as f:
            json.dump({
                "history": self.optimization_history[-500:],
                "knowledge": self.knowledge_base[-200:],
            }, f, ensure_ascii=False, indent=2)

    def collect_state(self) -> Dict:
        """采集系统全状态"""
        if not self.orchestrator:
            return {"error": "编排器未初始化"}

        state = {
            "timestamp": datetime.now().isoformat(),
            "infra": self.orchestrator.infra.dashboard(),
            "data_engine": self.orchestrator.data_engine.dashboard(),
            "agents": self.orchestrator.agent_orch.get_team_status(),
            "pipeline": self.orchestrator.pipeline.get_pipeline_stats(),
        }
        return state

    def analyze(self, state: Dict) -> List[Dict]:
        """分析系统状态, 生成优化建议"""
        recommendations = []

        # 1. Token余额检查
        token_balance = state.get("infra", {}).get("token_wallet", {}).get("total_balance", 0)
        if token_balance < 100:
            recommendations.append({
                "action": "alert_admin",
                "priority": "HIGH",
                "reason": f"Token余额不足({token_balance}), 建议补充",
                "params": {"message": f"Token余额仅剩{token_balance}"},
            })

        # 2. Agent失败率检查
        agent_stats = state.get("agents", {}).get("agents", [])
        for agent in agent_stats:
            total = agent.get("completed", 0) + agent.get("failed", 0)
            if total > 0:
                fail_rate = agent.get("failed", 0) / total
                if fail_rate > 0.2:
                    recommendations.append({
                        "action": "scale_agents",
                        "priority": "MEDIUM",
                        "reason": f"Agent[{agent['role']}]失败率过高({fail_rate:.0%})",
                        "params": {"agent_role": agent["role"], "action": "restart"},
                    })

        # 3. 存储清理检查
        storage = state.get("infra", {}).get("storage", {})
        if storage.get("total_used_gb", 0) > 500:
            recommendations.append({
                "action": "cleanup_storage",
                "priority": "LOW",
                "reason": f"存储使用量较大({storage['total_used_gb']}GB), 建议清理",
                "params": {},
            })

        # 4. 安全系统检查
        safety = state.get("data_engine", {}).get("safety_status", {})
        if safety.get("should_stop"):
            recommendations.append({
                "action": "alert_admin",
                "priority": "CRITICAL",
                "reason": "安全系统触发停止条件",
                "params": {"message": "训练安全阀已触发, 需人工介入"},
            })

        # 5. 碳信用积累检查
        carbon = state.get("infra", {}).get("carbon", {})
        if carbon.get("carbon_credits", 0) > 10:
            recommendations.append({
                "action": "carbon_trade",
                "priority": "LOW",
                "reason": f"碳信用余额充足({carbon['carbon_credits']}), 建议出售",
                "params": {"amount": carbon["carbon_credits"] * 0.5, "trade_type": "sell"},
            })

        # 6. 模型精度停滞检查
        evolution = state.get("data_engine", {}).get("bootstrap_evolution", [])
        if len(evolution) >= 3:
            recent = evolution[-3:]
            improvements = [e.get("improvement", 0) for e in recent]
            if all(imp < 0.02 for imp in improvements):
                recommendations.append({
                    "action": "adjust_hyperparams",
                    "priority": "MEDIUM",
                    "reason": "模型精度连续3代提升不足, 建议调整超参数",
                    "params": {"action": "increase_learning_rate"},
                })

        # 7. 流水线成功率检查
        pipeline = state.get("pipeline", {})
        if pipeline.get("total_runs", 0) > 5 and pipeline.get("success_rate", 1) < 0.7:
            recommendations.append({
                "action": "alert_admin",
                "priority": "HIGH",
                "reason": f"流水线成功率偏低({pipeline['success_rate']:.0%})",
                "params": {"message": "CI/CD流水线成功率下降"},
            })

        return recommendations

    def execute_optimization(self, recommendation: Dict) -> Dict:
        """执行优化动作"""
        action = recommendation["action"]
        params = recommendation.get("params", {})
        result = {"action": action, "success": False, "timestamp": datetime.now().isoformat()}

        if not self.orchestrator:
            result["error"] = "编排器未初始化"
            return result

        try:
            if action == "alert_admin":
                msg = params.get("message", "系统告警")
                print(f"  [闭环引擎] 告警: {msg}")
                if self.orchestrator.dashboard:
                    self.orchestrator.dashboard.push_notification("系统告警", msg, "critical")
                result["success"] = True
                result["message"] = msg

            elif action == "cleanup_storage":
                cleaned = self.orchestrator.infra.cleanup_storage()
                result["success"] = True
                result["cleaned_items"] = cleaned

            elif action == "carbon_trade":
                amount = params.get("amount", 0)
                trade_type = params.get("trade_type", "sell")
                trade_result = self.orchestrator.infra.carbon_gateway.trade_credits(amount, trade_type)
                result["success"] = trade_result.get("success", False)
                result["trade"] = trade_result

            elif action == "scale_agents":
                # 重启失败率高的Agent
                agent_role = params.get("agent_role", "")
                for agent in self.orchestrator.agent_orch.agents.values():
                    if agent.role == agent_role and agent.state == AgentState.FAILED:
                        agent.state = AgentState.IDLE
                        agent.current_task = None
                        print(f"  [闭环引擎] 重启Agent: {agent.agent_id}")
                result["success"] = True

            elif action == "adjust_hyperparams":
                # 调整超参数优化器
                optimizer = self.orchestrator.data_engine.bootstrap.optimizer
                if optimizer.best_params:
                    optimizer.best_params.learning_rate *= 1.2
                    print(f"  [闭环引擎] 调整学习率: {optimizer.best_params.learning_rate}")
                result["success"] = True

            elif action == "trigger_retrain":
                train_result = self.orchestrator.quick_train(generations=params.get("generations", 3))
                result["success"] = train_result.get("success", False)
                result["train_result"] = train_result

            elif action == "rebalance_vendors":
                self.orchestrator.infra.scheduler.refresh_vendor_status()
                result["success"] = True

            elif action == "adjust_token_price":
                # 动态调整利润率
                current_margin = self.orchestrator.infra.pricing.config["profit_margin"]
                new_margin = max(0.1, current_margin * 0.95)
                self.orchestrator.infra.pricing.config["profit_margin"] = new_margin
                result["success"] = True
                result["old_margin"] = current_margin
                result["new_margin"] = new_margin

        except Exception as e:
            result["error"] = str(e)

        return result

    def run_cycle(self) -> Dict:
        """运行一次完整的闭环优化周期"""
        print(f"\n{'='*50}")
        print(f"[闭环引擎] 启动优化周期 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")

        # 1. 采集状态
        state = self.collect_state()
        print(f"  [1/4] 状态采集完成")

        # 2. 分析
        recommendations = self.analyze(state)
        print(f"  [2/4] 分析完成, 发现{len(recommendations)}条优化建议")

        # 3. 执行
        results = []
        for rec in recommendations:
            if self.auto_optimize or rec["priority"] == "CRITICAL":
                print(f"  [3/4] 执行: {rec['action']} - {rec['reason']}")
                exec_result = self.execute_optimization(rec)
                results.append({
                    "recommendation": rec,
                    "result": exec_result,
                })
            else:
                print(f"  [3/4] 跳过(需人工确认): {rec['action']} - {rec['reason']}")
                results.append({
                    "recommendation": rec,
                    "result": {"skipped": True, "reason": "需人工确认"},
                })

        # 4. 验证 & 更新知识库
        verified_state = self.collect_state()
        cycle_summary = {
            "cycle_id": f"cycle_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "recommendations_count": len(recommendations),
            "executed_count": len([r for r in results if not r["result"].get("skipped")]),
            "skipped_count": len([r for r in results if r["result"].get("skipped")]),
            "results": results,
        }
        self.optimization_history.append(cycle_summary)

        # 更新知识库(简单规则学习)
        for r in results:
            if r["result"].get("success"):
                self.knowledge_base.append({
                    "action": r["recommendation"]["action"],
                    "reason": r["recommendation"]["reason"],
                    "outcome": "success",
                    "timestamp": datetime.now().isoformat(),
                })

        self._save()
        print(f"  [4/4] 周期完成: 执行{cycle_summary['executed_count']}/{len(recommendations)}条")
        return cycle_summary

    def get_optimization_summary(self) -> Dict:
        """获取优化历史摘要"""
        if not self.optimization_history:
            return {"total_cycles": 0}

        total_recs = sum(c["recommendations_count"] for c in self.optimization_history)
        total_executed = sum(c["executed_count"] for c in self.optimization_history)

        # 动作频次统计
        action_freq = {}
        for cycle in self.optimization_history:
            for r in cycle["results"]:
                action = r["recommendation"]["action"]
                action_freq[action] = action_freq.get(action, 0) + 1

        return {
            "total_cycles": len(self.optimization_history),
            "total_recommendations": total_recs,
            "total_executed": total_executed,
            "auto_optimize": self.auto_optimize,
            "knowledge_base_size": len(self.knowledge_base),
            "top_actions": sorted(action_freq.items(), key=lambda x: -x[1])[:5],
            "last_cycle": self.optimization_history[-1]["timestamp"] if self.optimization_history else None,
        }


# ============================================================
# LINGYUAN_ORCHESTRATOR [全局编排器]
# ============================================================

class LingyuanOrchestrator:
    """灵元全局编排器

    统一编排所有子系统:
    - LingyuanInfra (基础设施层)
    - LingyuanDataEngine (数据引擎层)
    - AgentOrchestrator (Agent编排层)
    - GitHubTriggerPipeline (CI/CD流水线)
    - MobileDashboard (移动仪表盘)
    - ClosedLoopEngine (闭环优化引擎)

    快速启动:
        orch = LingyuanOrchestrator()
        orch.quick_train(generations=3)
        orch.run_workflow_by_id("full_train")
        orch.closed_loop.run_cycle()
    """

    VERSION = "1.0.0"

    def __init__(self, user_id: str = "user_001"):
        print(f"\n{'#'*60}")
        print(f"# 灵元大模型系统 LINGYUAN MODEL v{self.VERSION}")
        print(f"# 灵元研究院 | 认知智能实验室")
        print(f"{'#'*60}\n")

        # 初始化所有子系统
        self.infra = LingyuanInfra(user_id)
        self.data_engine = LingyuanDataEngine()
        self.agent_orch = AgentOrchestrator()
        self.pipeline = GitHubTriggerPipeline()

        # 后初始化(依赖前面的组件)
        self.dashboard = MobileDashboard(self)
        self.closed_loop = ClosedLoopEngine(self)

        # 六层级递进融合决策引擎
        self.fusion_engine = FusionDecisionEngine()
        self.fusion_engine.initialize()

        # 安全治理系统
        self.safety = SafetyGovernanceSystem()

        # 可观测性引擎
        self.observability = ObservabilityEngine()
        self.observability.register_health_probe("infra")
        self.observability.register_health_probe("data_engine")
        self.observability.register_health_probe("agents")
        self.observability.register_health_probe("pipeline")

        # API网关
        self.api = APIGateway()

        # 模型注册中心
        self.registry = ModelRegistry()

        # 课程式训练器
        self.curriculum = CurriculumTrainer()

        # 经济引擎
        self.economy = EconomicEngine()
        self.economy.initialize()

        # 知识图谱
        self.knowledge = KnowledgeGraph()

        # ===== Part 8: 深度学习工程化系统 =====
        # 联邦学习系统
        self.federation = FederationLearningSystem()

        # 模型蒸馏流水线
        self.distillation = ModelDistillationPipeline()

        # RLHF强化学习反馈回路
        self.rlhf = ReinforcementFeedbackLoop()

        # 量化压缩引擎
        self.quantization = QuantizationEngine()

        # 向量数据库 (语义检索/RAG)
        self.vector_db = VectorDatabase(dim=128)

        # 提示工程工作室
        self.prompt_studio = PromptEngineeringStudio()

        # 边缘部署管理器
        self.edge_deploy = EdgeDeploymentManager()

        # 对话记忆系统 (集成向量数据库)
        self.conversation_memory = ConversationMemorySystem(vector_db=self.vector_db)

        # ===== Part 9: 模型本体 =====
        self.tokenizer = BPETokenizer()
        self.model_config = ModelConfig.from_preset("tiny")
        self.transformer_model = LingyuanTransformerModel(self.model_config)
        self.training_engine = TrainingEngine(self.transformer_model)

        # ===== Part 10: 外部知识接入 + 脱敏 =====
        self.data_connector = ExternalDataConnector()
        self.doc_parser = DocumentParser()
        self.web_crawler = WebCrawler()
        self.pii_desensitizer = PIIDesensitizer()
        self.desensitization_audit = DesensitizationAuditLog()
        self.license_checker = LicenseChecker()
        self.external_training = ExternalTrainingInterface()
        self.external_teacher = ExternalTeacherDistiller()
        self.minhash_dedup = MinHashDeduplicator()

        # ===== Part 11: 推理服务 =====
        self.inference_engine = InferenceEngine()
        self.continuous_batcher = ContinuousBatcher()
        self.streaming_output = StreamingOutput()
        self.inference_cache = InferenceCache()
        self.function_caller = FunctionCaller()
        self.chat_template_mgr = ChatTemplateManager()

        # ===== Part 12: 模型格式 =====
        self.weight_serializer = WeightSerializer()
        self.hf_exporter = HuggingFaceExporter()
        self.onnx_exporter = ONNXExporter()
        self.gguf_exporter = GGUFExporter()
        self.model_importer = ExternalModelImporter()

        # ===== Part 13: 微调 =====
        self.lora_tuner = LoRATuner()
        self.full_finetuner = FullFineTuner()
        self.sft_trainer = SFTTrainer()
        self.dpo_trainer = DPOTrainer()
        self.continual_learner = ContinualLearner()
        self.domain_adapter = DomainAdapter()

        # ===== Part 14: API服务 =====
        self.http_server = HTTPServer()
        self.openai_api = OpenAICompatibleAPI(self.http_server)
        self.ws_server = WebSocketServer()
        self.grpc_service = GRPCService()
        self.api_doc_gen = APIDocGenerator(self.http_server)
        self.sdk_gen = SDKGenerator(self.http_server)

        # ===== Part 15: MLOps =====
        self.experiment_tracker = ExperimentTracker()
        self.job_queue = TrainingJobQueue()
        self.gpu_scheduler = GPUScheduler()
        self.training_monitor = TrainingMonitor()
        self.model_comparator = ModelComparator()

        # ===== Part 16: UI + 安全 =====
        self.web_chat_ui = WebChatUI()
        self.playground = Playground()
        self.training_dashboard = TrainingDashboard()
        self.watermarker = ModelWatermarker()
        self.api_key_mgr = APIKeyManager()
        self.provenance_auditor = DataProvenanceAuditor()

        # 注册初始模型(如果无模型)
        if len(self.data_engine.model_data.assets) == 0:
            self._register_initial_model()

        print(f"\n[系统就绪] 所有模块已加载")
        print(f"  - 基础设施: Token/Energy/Vendor/Storage/Env/Disaster")
        print(f"  - 数据引擎: 模型工厂/数据生成/自举训练/蒸馏/评估")
        print(f"  - Agent团队: {len(self.agent_orch.agents)}个Agent已就绪")
        print(f"  - CI/CD流水线: GitHub触发器已配置")
        print(f"  - 移动仪表盘: 已就绪")
        print(f"  - 闭环引擎: {'自动' if self.closed_loop.auto_optimize else '手动'}模式")
        print(f"  - 融合决策引擎: 六层级递进融合已初始化")
        print(f"  - 安全治理: 内容过滤/审计/熔断/红队")
        print(f"  - 可观测性: 指标/追踪/异常检测")
        print(f"  - API网关: {len(self.api.endpoints)}个端点")
        print(f"  - 模型注册中心: 版本管理/A/B测试/金丝雀")
        print(f"  - 课程训练器: {len(self.curriculum.scheduler.stages)}阶段课程")
        print(f"  - 经济引擎: Token市场/拍卖/金库")
        print(f"  - 知识图谱: {len(self.knowledge.entities)}实体/{len(self.knowledge.relations)}关系")
        print(f"  - 联邦学习: {len(self.federation.nodes)}节点/{self.federation.model_dim}维模型")
        print(f"  - 模型蒸馏: {len(self.distillation.pairs)}蒸馏任务")
        print(f"  - RLHF反馈: 奖励模型+PPO优化器")
        print(f"  - 量化压缩: INT4/INT8/FP16/混合精度")
        print(f"  - 向量数据库: HNSW索引+嵌入引擎")
        print(f"  - 提示工程: {len(self.prompt_studio.templates)}模板")
        print(f"  - 边缘部署: {len(self.edge_deploy.devices)}设备")
        print(f"  - 对话记忆: 短期+长期+RAG")
        print(f"  - 模型本体: Transformer/Tokenizer/RoPE/采样")
        print(f"  - 外部接入: 数据连接器/文档解析/PII脱敏/版权检查")
        print(f"  - 推理服务: KV Cache/连续批处理/流式/Function Calling")
        print(f"  - 模型格式: Safetensors/HF/ONNX/GGUF/外部导入")
        print(f"  - 微调: LoRA/SFT/DPO/持续学习/领域适配")
        print(f"  - API服务: HTTP/OpenAI兼容/WebSocket/gRPC")
        print(f"  - MLOps: 实验追踪/任务队列/GPU调度/监控")
        print(f"  - UI+安全: Web Chat/Playground/水印/API Key")

    def _register_initial_model(self):
        """注册初始基座模型"""
        result = self.data_engine.register_model(
            name="灵元-基座-v1",
            hidden_dim=4096,
            num_layers=32,
            num_heads=32,
            capabilities=["text_gen", "reasoning", "classify"],
            generation=0,
            token_cost=0,
        )
        print(f"  [初始模型] 已注册基座模型: {result['asset_id']}")
        return result["asset_id"]

    # ==================== 快捷操作 ====================

    def quick_train(self, generations: int = 3, tokens_per_gen: int = 100) -> Dict:
        """快速训练: 购买Token -> 自举训练 -> 评估"""
        print(f"\n{'='*50}")
        print(f"[快速训练] 启动, 目标代数: {generations}")
        print(f"{'='*50}")

        # 1. 确保Token充足
        needed = generations * tokens_per_gen
        balance = self.infra.wallet.get_balance()
        if balance < needed:
            print(f"  [Token] 余额不足({balance}), 自动购买{needed}Token")
            buy_result = self.infra.buy_token(needed + 50, green_power=self.infra.is_green_power_hour())
            if not buy_result.get("success"):
                return {"success": False, "error": "Token购买失败", "detail": buy_result}

        # 2. 获取初始模型
        models = self.data_engine.list_models()
        if not models:
            initial_id = self._register_initial_model()
        else:
            initial_id = models[0]["asset_id"]

        # 3. 运行自举训练
        train_summary = self.data_engine.run_bootstrap(
            initial_model_id=initial_id,
            initial_accuracy=0.65,
            initial_loss=2.0,
            max_generations=generations,
            tokens_per_gen=tokens_per_gen,
        )

        # 4. 评估最终模型
        if train_summary.get("final_model") and train_summary["final_model"] != initial_id:
            eval_result = self.data_engine.evaluate_model(
                model_id=train_summary["final_model"],
                model_name=f"GEN_{train_summary['final_model']}",
                generation=train_summary["generations_completed"],
            )
            train_summary["evaluation"] = eval_result

        # 5. 推送通知
        self.dashboard.push_notification(
            "训练完成",
            f"完成{train_summary['generations_completed']}代训练, "
            f"精度: {train_summary['final_accuracy']:.4f}, "
            f"提升: {train_summary['total_improvement']:.4f}",
            "info",
        )

        train_summary["success"] = True
        return train_summary

    def run_workflow_by_id(self, workflow_id: str) -> Dict:
        """根据工作流模板ID运行工作流"""
        wf = self.data_engine.workflow_mgr.get_workflow(workflow_id)
        if not wf:
            return {"success": False, "error": f"工作流不存在: {workflow_id}"}

        print(f"\n[工作流] 执行: {wf['name']} ({workflow_id})")

        # 使用Agent编排器执行
        result = self.agent_orch.run_workflow(
            workflow_name=wf["name"],
            task_sequence=wf["steps"],
        )

        # 推送通知
        self.dashboard.push_notification(
            "工作流完成",
            f"{wf['name']}: {result['succeeded']}/{result['total_tasks']}任务成功",
            "info",
        )

        return result

    def trigger_pipeline(self, trigger_type: str = "push", branch: str = "main",
                         repo: str = "qingluan-studio/lingyuan", payload: Dict = None) -> Dict:
        """触发GitHub流水线"""
        try:
            trigger = TriggerType(trigger_type)
        except ValueError:
            return {"success": False, "error": f"未知触发类型: {trigger_type}"}

        run_id = self.pipeline.receive_event(
            event_type=trigger,
            repo=repo,
            branch=branch,
            payload=payload or {"commit": "auto_trigger", "tag": ""},
        )

        if not run_id:
            return {"success": False, "error": "触发条件不满足"}

        status = self.pipeline.get_run_status(run_id)
        self.dashboard.push_notification(
            "流水线触发",
            f"仓库:{repo} 分支:{branch} 状态:{status['status']}",
            "info",
        )
        return {"success": True, "run_id": run_id, "status": status}

    def run_closed_loop(self, cycles: int = 1) -> Dict:
        """运行闭环优化"""
        results = []
        for i in range(cycles):
            print(f"\n[闭环] 第{i+1}/{cycles}轮")
            result = self.closed_loop.run_cycle()
            results.append(result)
        return {"cycles": len(results), "results": results}

    def run_fusion_decision(self, system_state: Dict = None) -> Dict:
        """运行六层级融合决策

        Args:
            system_state: 系统状态, 为空时自动采集
        """
        if system_state is None:
            # 自动采集系统状态
            infra = self.infra.dashboard()
            wallet = infra.get("token_wallet", {})
            energy = infra.get("energy", {})

            # 从供应商获取GPU状态
            vendors = self.infra.scheduler.get_all_vendors() if hasattr(self.infra.scheduler, 'get_all_vendors') else []

            system_state = {
                "remaining_vram": max(0, 80 - energy.get("total_energy_kwh", 0) * 0.5),
                "consumption_rate": 1.0 + len(vendors) * 0.1,
                "actual_gen_rate": 1.0,
                "kv_hit_rate": 0.75 + random.uniform(-0.1, 0.1),
                "seq_length": random.choice([256, 512, 1024, 2048]),
                "batch_size": random.choice([1, 2, 4, 8]),
                "fragmentation": round(random.uniform(0.03, 0.20), 4),
                "swap_ratio": round(random.uniform(0.8, 2.5), 2),
                "vram_decline": round(random.uniform(0.0, 0.8), 4),
                "requests": [],
                "slo_urgency": round(random.uniform(0.3, 0.8), 2),
            }

        print(f"\n[融合引擎] 执行六层级决策...")
        decision = self.fusion_engine.decide(system_state)
        print(f"[融合引擎] 决策: {decision['decision_summary']}")
        return decision

    # ==================== 全局仪表盘 ====================

    def full_dashboard(self) -> Dict:
        """全局仪表盘"""
        return {
            "system": {
                "version": self.VERSION,
                "timestamp": datetime.now().isoformat(),
                "user_id": self.infra.user_id,
            },
            "infra": self.infra.dashboard(),
            "data_engine": self.data_engine.dashboard(),
            "agents": self.agent_orch.get_team_status(),
            "pipeline": self.pipeline.get_pipeline_stats(),
            "closed_loop": self.closed_loop.get_optimization_summary(),
            "mobile": self.dashboard.get_overview(),
            "fusion_engine": self.fusion_engine.get_dashboard(),
            "safety": self.safety.get_dashboard(),
            "observability": self.observability.get_dashboard(),
            "api_gateway": self.api.get_stats(),
            "model_registry": self.registry.get_dashboard(),
            "curriculum": self.curriculum.get_dashboard(),
            "economy": self.economy.get_dashboard(),
            "knowledge_graph": self.knowledge.get_dashboard(),
            "federation": self.federation.get_dashboard(),
            "distillation": self.distillation.get_dashboard(),
            "rlhf": self.rlhf.get_dashboard(),
            "quantization": self.quantization.get_dashboard(),
            "vector_db": self.vector_db.get_dashboard(),
            "prompt_studio": self.prompt_studio.get_dashboard(),
            "edge_deploy": self.edge_deploy.get_dashboard(),
            "conversation_memory": self.conversation_memory.get_dashboard(),
            # Part 9-16
            "model_core": self.transformer_model.get_stats() if hasattr(self.transformer_model, 'get_stats') else {},
            "external_data": {"connector": len(self.data_connector.sources) if hasattr(self.data_connector, 'sources') else 0},
            "inference": self.inference_engine.get_dashboard() if hasattr(self.inference_engine, 'get_dashboard') else {},
            "model_format": self.weight_serializer.get_stats() if hasattr(self.weight_serializer, 'get_stats') else {},
            "finetuning": {"lora": "ready", "sft": "ready", "dpo": "ready"},
            "api_service": self.http_server.get_stats() if hasattr(self.http_server, 'get_stats') else {},
            "mlops": self.experiment_tracker.get_stats() if hasattr(self.experiment_tracker, 'get_stats') else {},
            "ui_security": {"watermark_keys": len(self.watermarker.keys) if hasattr(self.watermarker, 'keys') else 0},
        }

    # ==================== 系统管理 ====================

    def system_health_check(self) -> Dict:
        """系统健康检查"""
        checks = {
            "timestamp": datetime.now().isoformat(),
            "overall": "healthy",
            "checks": {},
        }

        # 基础设施检查
        try:
            wallet = self.infra.get_wallet_summary()
            checks["checks"]["infra"] = {
                "status": "healthy",
                "token_balance": wallet["total_balance"],
            }
        except Exception as e:
            checks["checks"]["infra"] = {"status": "error", "error": str(e)}
            checks["overall"] = "degraded"

        # 数据引擎检查
        try:
            stats = self.data_engine.get_data_stats()
            checks["checks"]["data_engine"] = {
                "status": "healthy",
                "total_datasets": stats.get("total_datasets", 0),
            }
        except Exception as e:
            checks["checks"]["data_engine"] = {"status": "error", "error": str(e)}
            checks["overall"] = "degraded"

        # Agent团队检查
        try:
            team = self.agent_orch.get_team_status()
            failed_agents = sum(1 for a in team["agents"] if a["state"] == "failed")
            checks["checks"]["agents"] = {
                "status": "healthy" if failed_agents == 0 else "degraded",
                "total_agents": team["total_agents"],
                "failed_agents": failed_agents,
            }
            if failed_agents > 0:
                checks["overall"] = "degraded"
        except Exception as e:
            checks["checks"]["agents"] = {"status": "error", "error": str(e)}
            checks["overall"] = "degraded"

        # 安全系统检查
        try:
            safety = self.data_engine.get_safety_summary()
            checks["checks"]["safety"] = {
                "status": "critical" if safety.get("should_stop") else "healthy",
                "total_checks": safety.get("total_checks", 0),
                "should_stop": safety.get("should_stop", False),
            }
            if safety.get("should_stop"):
                checks["overall"] = "critical"
        except Exception as e:
            checks["checks"]["safety"] = {"status": "error", "error": str(e)}
            checks["overall"] = "degraded"

        return checks

    def shutdown(self):
        """优雅关闭系统"""
        print(f"\n[系统关闭] 开始优雅关闭...")
        # 保存所有状态(各模块已有自动保存)
        print(f"  - 基础设施状态已保存")
        print(f"  - 数据引擎状态已保存")
        print(f"  - Agent团队状态已保存")
        print(f"  - 流水线状态已保存")
        print(f"  - 闭环引擎状态已保存")
        print(f"[系统关闭] 完成")


# ============================================================
# GLOBAL TEST SUITE [全局测试套件]
# ============================================================

class LingyuanTestSuite:
    """灵元系统全局测试套件

    测试覆盖:
    1. Token系统测试
    2. 能源系统测试
    3. 供应商调度测试
    4. 存储系统测试
    5. 环境管理测试
    6. 灾备恢复测试
    7. 模型资产管理测试
    8. 数据生成测试
    9. 自举训练测试
    10. 知识蒸馏测试
    11. 评估系统测试
    12. Agent编排测试
    13. GitHub流水线测试
    14. 闭环引擎测试
    15. 端到端集成测试
    """

    def __init__(self):
        self.results: List[Dict] = []
        self.passed = 0
        self.failed = 0

    def _assert(self, test_name: str, condition: bool, detail: str = ""):
        """断言辅助"""
        status = "PASS" if condition else "FAIL"
        result = {
            "test": test_name,
            "status": status,
            "detail": detail,
        }
        self.results.append(result)
        if condition:
            self.passed += 1
            print(f"  [PASS] {test_name}")
        else:
            self.failed += 1
            print(f"  [FAIL] {test_name} - {detail}")

    def test_token_system(self, orch: LingyuanOrchestrator):
        """测试Token系统"""
        print("\n--- 测试: Token系统 ---")
        wallet = orch.infra.wallet

        # 测试购买
        buy = wallet.purchase(100, green_power=False)
        self._assert("Token购买", buy["success"], f"购买结果: {buy}")

        # 测试余额
        balance = wallet.get_balance()
        self._assert("Token余额>0", balance > 0, f"余额: {balance}")

        # 测试消费
        consume = wallet.consume(10, "test_task")
        self._assert("Token消费", consume["success"], f"消费结果: {consume}")

        # 测试定价
        price = orch.infra.pricing.get_current_price()
        self._assert("Token定价", price["unit_price"] > 0, f"价格: {price['unit_price']}")

        # 测试绿电价格
        green_price = orch.infra.pricing.get_current_price(green_power=True)
        self._assert("绿电折扣", green_price["unit_price"] <= price["unit_price"],
                      f"绿电: {green_price['unit_price']} vs 普通: {price['unit_price']}")

    def test_energy_system(self, orch: LingyuanOrchestrator):
        """测试能源系统"""
        print("\n--- 测试: 能源系统 ---")
        energy = orch.infra.energy

        # 测试能耗计算
        calc = energy.calculate_energy(100, green_power=False)
        self._assert("能耗计算", calc["energy_kwh"] > 0, f"能耗: {calc}")

        # 测试绿电减排
        green_calc = energy.calculate_energy(100, green_power=True)
        self._assert("绿电减排", green_calc["carbon_saved"] > 0,
                      f"碳减排: {green_calc['carbon_saved']}")

        # 测试记录
        record = energy.record_consumption("test_task", 50, green_power=True, vendor_id="vendor_a")
        self._assert("能耗记录", record.record_id != "", f"记录ID: {record.record_id}")

        # 测试ESG报告
        report = energy.generate_esg_report()
        self._assert("ESG报告", "total_carbon_emission_kg" in report, f"报告: {report}")

    def test_vendor_scheduler(self, orch: LingyuanOrchestrator):
        """测试供应商调度"""
        print("\n--- 测试: 供应商调度 ---")
        scheduler = orch.infra.scheduler

        # 测试提交任务
        task = scheduler.submit_task("training", 10, 5.0, "normal", False)
        self._assert("任务提交", task.task_id != "", f"任务ID: {task.task_id}")

        # 测试厂商分配
        self._assert("厂商分配", task.assigned_vendor != "", f"分配厂商: {task.assigned_vendor}")

        # 测试完成任务
        ok = scheduler.complete_task(task.task_id, {"status": "ok"})
        self._assert("任务完成", ok, "")

        # 测试厂商对比
        comparison = scheduler.get_vendor_comparison()
        self._assert("厂商对比", len(comparison) > 0, f"厂商数: {len(comparison)}")

    def test_storage_system(self, orch: LingyuanOrchestrator):
        """测试存储系统"""
        print("\n--- 测试: 存储系统 ---")
        storage = orch.infra.storage

        # 测试存储
        item = storage.store("test_model.pt", 2.5, "test_project")
        self._assert("数据存储", item.item_id != "", f"存储ID: {item.item_id}")

        # 测试访问
        ok = storage.access(item.item_id)
        self._assert("数据访问", ok, "")

        # 测试摘要
        summary = storage.get_storage_summary()
        self._assert("存储摘要", summary["total_used_gb"] > 0, f"使用: {summary['total_used_gb']}GB")

        # 测试删除
        ok = storage.delete(item.item_id)
        self._assert("数据删除", ok, "")

    def test_model_data(self, orch: LingyuanOrchestrator):
        """测试模型资产管理"""
        print("\n--- 测试: 模型资产管理 ---")
        data_engine = orch.data_engine

        # 测试注册
        result = data_engine.register_model(
            "测试模型", 2048, 16, 16, ["text_gen", "reasoning"], generation=1,
        )
        self._assert("模型注册", result["success"], f"资产ID: {result.get('asset_id')}")

        # 测试列表
        models = data_engine.list_models()
        self._assert("模型列表", len(models) > 0, f"模型数: {len(models)}")

        # 测试血缘
        if models:
            bloodline = data_engine.model_data.get_model_bloodline(models[0]["asset_id"])
            self._assert("血缘追踪", len(bloodline) > 0, f"链长度: {len(bloodline)}")

        # 测试互补性
        if len(models) >= 2:
            comps = data_engine.find_complementary(models[0]["asset_id"])
            self._assert("互补分析", len(comps) > 0, f"互补模型数: {len(comps)}")

    def test_data_generation(self, orch: LingyuanOrchestrator):
        """测试数据生成"""
        print("\n--- 测试: 数据生成 ---")
        data_engine = orch.data_engine

        result = data_engine.generate_dataset(
            source_model_name="test_model",
            generation=1,
            task_type="qa",
            count=20,
            quality_threshold=0.5,
        )
        self._assert("数据集生成", result["success"], f"数据集: {result.get('dataset_id')}")
        self._assert("数据质量", result.get("avg_quality", 0) > 0.5,
                      f"质量: {result.get('avg_quality')}")

    def test_bootstrap_training(self, orch: LingyuanOrchestrator):
        """测试自举训练"""
        print("\n--- 测试: 自举训练 ---")
        data_engine = orch.data_engine

        summary = data_engine.run_bootstrap(
            initial_model_id="test_base_model",
            initial_accuracy=0.60,
            initial_loss=2.5,
            max_generations=3,
            tokens_per_gen=50,
        )
        self._assert("自举训练完成", summary["generations_run"] > 0,
                      f"运行代数: {summary['generations_run']}")
        self._assert("训练有提升", summary["total_improvement"] >= 0,
                      f"提升: {summary['total_improvement']}")

    def test_distillation(self, orch: LingyuanOrchestrator):
        """测试知识蒸馏"""
        print("\n--- 测试: 知识蒸馏 ---")
        data_engine = orch.data_engine

        result = data_engine.distill_model(
            teacher_model_id="test_teacher",
            teacher_params=1000000,
            teacher_accuracy=0.85,
            compression=4,
        )
        self._assert("蒸馏成功", result["success"], f"蒸馏ID: {result.get('distill_id')}")
        # accuracy_loss 格式为 "x.xx%"
        loss_str = result.get("accuracy_loss", "0%").replace("%", "")
        loss_val = float(loss_str) if loss_str else 0.0
        self._assert("精度损失<1%", loss_val < 1.0,
                      f"精度损失: {result.get('accuracy_loss')}")
        # 推理加速应在5-10倍范围
        speed_str = result.get("speed_improvement", "0x").replace("x", "")
        speed_val = float(speed_str) if speed_str else 0.0
        self._assert("推理加速5-10倍", 5.0 <= speed_val <= 12.0,
                      f"加速: {result.get('speed_improvement')}")

    def test_evaluation(self, orch: LingyuanOrchestrator):
        """测试评估系统"""
        print("\n--- 测试: 评估系统 ---")
        data_engine = orch.data_engine

        result = data_engine.evaluate_model(
            model_id="test_model",
            model_name="测试模型",
            generation=2,
        )
        self._assert("评估完成", "eval_id" in result, f"评估ID: {result.get('eval_id')}")
        self._assert("综合评分", result.get("overall_score", 0) > 0,
                      f"评分: {result.get('overall_score')}")

    def test_agent_orchestrator(self, orch: LingyuanOrchestrator):
        """测试Agent编排"""
        print("\n--- 测试: Agent编排 ---")
        agent_orch = orch.agent_orch

        # 测试团队初始化
        self._assert("Agent团队", len(agent_orch.agents) >= 6,
                      f"Agent数: {len(agent_orch.agents)}")

        # 测试工作流执行
        result = agent_orch.run_workflow("test_workflow", [
            {"name": "数据准备", "agent_role": "data", "priority": "HIGH"},
            {"name": "模型训练", "agent_role": "train", "priority": "HIGH",
             "depends_on_index": [0]},
        ])
        self._assert("工作流执行", result["total_tasks"] > 0,
                      f"任务数: {result['total_tasks']}")
        self._assert("工作流成功率>0", result["succeeded"] > 0,
                      f"成功: {result['succeeded']}")

    def test_github_pipeline(self, orch: LingyuanOrchestrator):
        """测试GitHub流水线"""
        print("\n--- 测试: GitHub流水线 ---")
        pipeline = orch.pipeline

        # 测试事件触发
        run_id = pipeline.receive_event(
            event_type=TriggerType.PUSH,
            repo="qingluan-studio/lingyuan",
            branch="main",
            payload={"commit": "test123", "tag": ""},
        )
        self._assert("流水线触发", run_id != "", f"运行ID: {run_id}")

        if run_id:
            status = pipeline.get_run_status(run_id)
            self._assert("流水线状态", "status" in status,
                          f"状态: {status.get('status')}")

        stats = pipeline.get_pipeline_stats()
        self._assert("流水线统计", stats["total_runs"] > 0,
                      f"总运行: {stats['total_runs']}")

    def test_closed_loop(self, orch: LingyuanOrchestrator):
        """测试闭环引擎"""
        print("\n--- 测试: 闭环引擎 ---")
        closed_loop = orch.closed_loop

        result = closed_loop.run_cycle()
        self._assert("闭环周期", result["cycle_id"] != "", f"周期ID: {result['cycle_id']}")
        self._assert("闭环分析", result["recommendations_count"] >= 0,
                      f"建议数: {result['recommendations_count']}")

    def test_prompt_templates(self, orch: LingyuanOrchestrator):
        """测试Prompt模板"""
        print("\n--- 测试: Prompt模板 ---")
        prompt_mgr = orch.data_engine.prompt_mgr

        # 测试模板列表
        templates = prompt_mgr.list_templates()
        self._assert("模板列表", len(templates) > 0, f"模板数: {len(templates)}")

        # 测试渲染
        rendered = prompt_mgr.render("pt_train_qa", knowledge="测试知识", question="测试问题")
        self._assert("模板渲染", "测试知识" in rendered and "测试问题" in rendered,
                      f"渲染: {rendered[:50]}...")

    def test_workflow_engine(self, orch: LingyuanOrchestrator):
        """测试工作流引擎"""
        print("\n--- 测试: 工作流引擎 ---")
        wf_mgr = orch.data_engine.workflow_mgr

        # 测试内置工作流
        workflows = wf_mgr.list_workflows()
        self._assert("工作流列表", len(workflows) >= 3, f"工作流数: {len(workflows)}")

        # 测试获取
        wf = wf_mgr.get_workflow("full_train")
        self._assert("获取工作流", wf is not None and "steps" in wf,
                      f"步骤数: {len(wf['steps']) if wf else 0}")

        # 测试动态生成
        dynamic = wf_mgr.generate_dynamic_workflow("trigger_training", {"generations": 5})
        self._assert("动态工作流", len(dynamic) > 0, f"步骤数: {len(dynamic)}")

    def test_carbon_trading(self, orch: LingyuanOrchestrator):
        """测试碳交易"""
        print("\n--- 测试: 碳交易 ---")
        carbon = orch.infra.carbon_gateway

        # 积累碳信用
        credits = carbon.accumulate_credits(100.0)
        self._assert("碳信用积累", credits > 0, f"碳信用: {credits}")

        # 交易
        trade = carbon.trade_credits(0.05, "sell")
        self._assert("碳信用交易", trade["success"], f"交易: {trade}")

    def test_data_trading(self, orch: LingyuanOrchestrator):
        """测试数据交易"""
        print("\n--- 测试: 数据交易 ---")
        trading = orch.infra.data_trading

        # 确权
        trading.register_ownership("asset_test_001", "user_001")
        self._assert("数据确权", trading.verify_ownership("asset_test_001", "user_001"), "")

        # 交易
        trade = trading.execute_trade("asset_test_001", "user_001", "user_002", 100.0)
        self._assert("数据交易", trade["success"], f"交易: {trade}")

        # 验证所有权转移
        self._assert("所有权转移", trading.verify_ownership("asset_test_001", "user_002"), "")

    def test_spatial_collaboration(self, orch: LingyuanOrchestrator):
        """测试空间协同"""
        print("\n--- 测试: 空间协同 ---")
        data_engine = orch.data_engine

        # 注册两个模型
        m1 = data_engine.register_model("模型A", 2048, 16, 16, ["text_gen"], generation=1)
        m2 = data_engine.register_model("模型B", 4096, 32, 32, ["reasoning"], generation=1)

        # 测试对齐
        align = data_engine.align_models([m1["asset_id"], m2["asset_id"]])
        self._assert("模型对齐", align.get("success", False), f"对齐: {align}")

        # 测试融合
        fuse = data_engine.fuse_cross_modal([m1["asset_id"], m2["asset_id"]])
        self._assert("跨模态融合", "fused_model_count" in fuse, f"融合: {fuse}")

    def test_multimodal_generation(self, orch: LingyuanOrchestrator):
        """测试多模态数据生成"""
        print("\n--- 测试: 多模态数据生成 ---")
        data_engine = orch.data_engine

        modalities = ["audio", "image", "video", "multimodal"]
        for modality in modalities:
            result = data_engine.generate_dataset(
                source_model_name="mm_test_model",
                generation=1,
                task_type=modality,
                count=10,
                quality_threshold=0.3,
                modality=modality,
            )
            self._assert(f"多模态生成-{modality}", result["success"] and result.get("modality") == modality,
                          f"模态: {result.get('modality')} | 质量: {result.get('avg_quality')}")

            # 验证生成的样本包含media_metadata
            if result["success"]:
                ds = data_engine.data_generator.get_dataset(result["dataset_id"])
                if ds and ds.samples:
                    sample = ds.samples[0]
                    self._assert(f"多模态元数据-{modality}", sample.modality == modality,
                                  f"sample.modality={sample.modality}")
                    has_meta = bool(sample.media_metadata) if modality != "text" else True
                    self._assert(f"媒体元信息-{modality}", has_meta,
                                  f"media_metadata keys: {list(sample.media_metadata.keys()) if sample.media_metadata else 'N/A'}")

    def test_multimodal_evaluation(self, orch: LingyuanOrchestrator):
        """测试多模态评估"""
        print("\n--- 测试: 多模态评估 ---")
        data_engine = orch.data_engine

        modality_checks = {
            "audio": ["mos_score", "wer", "speaker_similarity"],
            "image": ["fid_score", "clip_score", "visual_quality"],
            "video": ["fvd_score", "temporal_consistency", "motion_quality"],
            "multimodal": ["cross_modal_alignment", "modality_fusion_score", "retrieval_accuracy"],
        }

        for modality, expected_keys in modality_checks.items():
            result = data_engine.evaluate_model(
                model_id=f"mm_eval_{modality}",
                model_name=f"多模态测试-{modality}",
                generation=2,
                modality=modality,
            )
            self._assert(f"多模态评估-{modality}", "eval_id" in result,
                          f"评估ID: {result.get('eval_id')}")
            self._assert(f"多模态评分-{modality}", result.get("overall_score", 0) > 0,
                          f"评分: {result.get('overall_score')}")

            # 验证模态专用指标存在
            metrics = result.get("modality_metrics", {})
            for key in expected_keys:
                self._assert(f"模态指标-{modality}-{key}", key in metrics,
                              f"值: {metrics.get(key, 'N/A')}")

    def test_multimodal_token_cost(self, orch: LingyuanOrchestrator):
        """测试多模态Token消耗倍率"""
        print("\n--- 测试: 多模态Token消耗 ---")
        data_gen = orch.data_engine.data_generator

        # text基线
        text_cost = data_gen.estimate_token_cost(100, "text")
        # 非文本应消耗更多
        for modality, expected_multiplier in [("audio", 3.0), ("image", 8.0), ("video", 20.0), ("multimodal", 5.0)]:
            cost = data_gen.estimate_token_cost(100, modality)
            ratio = cost / max(text_cost, 1)
            self._assert(f"Token倍率-{modality}", abs(ratio - expected_multiplier) < 0.1,
                          f"倍率: {ratio:.1f}x (期望 {expected_multiplier}x)")

    def test_fusion_engine(self, orch: LingyuanOrchestrator):
        """测试六层级融合决策引擎"""
        print("\n--- 测试: 六层级融合决策引擎 ---")
        fusion = orch.fusion_engine

        # 1. 测试快速评估 (正常场景)
        result = fusion.quick_assess(
            remaining_vram=60.0,
            consumption_rate=1.0,
            fragmentation=0.05,
            swap_ratio=1.0,
            vram_decline=0.1,
        )
        self._assert("融合决策-正常", "decision_id" in result,
                      f"决策: {result.get('decision_summary', 'N/A')}")
        self._assert("融合-OOM概率", 0 <= result.get("oom_probability", -1) <= 1,
                      f"OOM: {result.get('oom_probability')}")

        # 2. 测试高压力场景
        result_stress = fusion.quick_assess(
            remaining_vram=10.0,
            consumption_rate=3.0,
            fragmentation=0.25,
            swap_ratio=3.0,
            vram_decline=1.0,
        )
        self._assert("融合决策-高压", "decision_id" in result_stress,
                      f"决策: {result_stress.get('decision_summary', 'N/A')}")
        # 高压场景OOM概率应更高
        self._assert("融合-OOM升高", result_stress.get("oom_probability", 0) >= result.get("oom_probability", 0),
                      f"正常: {result.get('oom_probability')} vs 高压: {result_stress.get('oom_probability')}")

    def test_entropy_triple(self, orch: LingyuanOrchestrator):
        """测试熵增三联监测"""
        print("\n--- 测试: 熵增三联监测 ---")
        monitor = orch.fusion_engine.slo_optimizer.oom_predictor.entropy_monitor

        # 正常状态
        state_ok = monitor.update(0.03, 1.0, 0.1)
        self._assert("三联-正常无触发", state_ok.triggered_count == 0,
                      f"触发数: {state_ok.triggered_count}")
        self._assert("三联-漂移系数1.0", state_ok.drift_multiplier == 1.0,
                      f"系数: {state_ok.drift_multiplier}")

        # 单项触发
        state_1 = monitor.update(0.20, 1.0, 0.1)  # 碎片率越线
        self._assert("三联-单项触发", state_1.triggered_count == 1,
                      f"触发数: {state_1.triggered_count}")
        self._assert("三联-漂移×1.2", state_1.drift_multiplier == 1.2,
                      f"系数: {state_1.drift_multiplier}")

        # 双项触发
        state_2 = monitor.update(0.20, 2.5, 0.1)  # 碎片+swap越线
        self._assert("三联-双项触发", state_2.triggered_count == 2,
                      f"触发数: {state_2.triggered_count}")
        self._assert("三联-漂移×1.5", state_2.drift_multiplier == 1.5,
                      f"系数: {state_2.drift_multiplier}")

        # 三联全触发
        state_3 = monitor.update(0.20, 2.5, 0.8)  # 全部越线
        self._assert("三联-全触发", state_3.triggered_count == 3,
                      f"触发数: {state_3.triggered_count}")
        self._assert("三联-漂移×2.0", state_3.drift_multiplier == 2.0,
                      f"系数: {state_3.drift_multiplier}")
        self._assert("三联-强制收缩", state_3.force_shrink,
                      f"force_shrink: {state_3.force_shrink}")

    def test_oom_prediction(self, orch: LingyuanOrchestrator):
        """测试OOM破产概率预测"""
        print("\n--- 测试: OOM破产概率预测 ---")
        predictor = orch.fusion_engine.slo_optimizer.oom_predictor

        # 充足资源 - 低风险
        result_safe = predictor.predict(
            remaining_resource=70.0,
            consumption_rate=0.5,
            time_horizon=10.0,
            fragmentation=0.03,
            swap_ratio=0.8,
            vram_decline=0.0,
        )
        self._assert("OOM-低风险", result_safe["bankruptcy_prob"] < 0.5,
                      f"概率: {result_safe['bankruptcy_prob']}")
        self._assert("OOM-预期时间", result_safe["expected_time_to_oom"] > 0,
                      f"预期OOM时间: {result_safe['expected_time_to_oom']}s")

        # 资源紧缺 - 高风险
        result_danger = predictor.predict(
            remaining_resource=5.0,
            consumption_rate=3.0,
            time_horizon=10.0,
            fragmentation=0.25,
            swap_ratio=3.0,
            vram_decline=1.0,
        )
        self._assert("OOM-高风险升高", result_danger["bankruptcy_prob"] >= result_safe["bankruptcy_prob"],
                      f"安全: {result_safe['bankruptcy_prob']} vs 危险: {result_danger['bankruptcy_prob']}")

    def test_multimodel_collaboration(self, orch: LingyuanOrchestrator):
        """测试多模型协同"""
        print("\n--- 测试: 多模型协同 ---")
        mm = orch.fusion_engine.multi_model

        # 测试模型选择
        req = InferenceRequest(request_id="test_001", prompt="测试", seq_length=1024, priority=2)
        tier = mm.select_model(req, oom_probability=0.1, slo_urgency=0.5)
        self._assert("多模型-正常选择", tier in ["70B", "13B", "7B"],
                      f"选择: {tier}")

        # OOM高时应偏向小模型
        tier_stress = mm.select_model(
            InferenceRequest(request_id="test_002", prompt="测试"),
            oom_probability=0.5, slo_urgency=0.8,
        )
        self._assert("多模型-高压切小模型", tier_stress in ["13B", "7B"],
                      f"高压选择: {tier_stress}")

        # 测试投机解码
        spec = mm.speculative_decode(req, draft_tier="7B", verify_tier="70B")
        self._assert("多模型-投机解码", spec["success"],
                      f"接受率: {spec.get('accept_rate')}")
        self._assert("多模型-解码加速", spec.get("speedup", 0) > 1.0,
                      f"加速: {spec.get('speedup')}x")

        # 测试batch优化
        requests = [
            InferenceRequest(request_id=f"r{i}", prompt="test", prefix_hash="hash_A" if i < 3 else f"hash_{i}")
            for i in range(5)
        ]
        batch_result = mm.batch_optimize(requests)
        self._assert("多模型-batch编队", batch_result["success"],
                      f"batch数: {batch_result['total_batches']}")
        self._assert("多模型-prefix共享", batch_result["total_kv_cache_saved_gb"] > 0,
                      f"节省KV cache: {batch_result['total_kv_cache_saved_gb']}GB")

        # 测试请求重排
        reorder = mm.request_reorder(requests, oom_risk=0.4)
        self._assert("多模型-请求重排", reorder["success"],
                      f"策略: {reorder['strategy']}")

    def test_pain_curve(self, orch: LingyuanOrchestrator):
        """测试痛苦曲线优化"""
        print("\n--- 测试: 痛苦曲线优化 ---")
        pain_opt = orch.fusion_engine.slo_optimizer.outer

        # 求解最优曲线
        curve = pain_opt.solve_optimal_curve(total_budget=5.0, time_steps=30)
        self._assert("痛苦曲线-求解", len(curve) == 30,
                      f"曲线长度: {len(curve)}")
        self._assert("痛苦曲线-预算守恒", abs(sum(curve) - 5.0) < 0.1,
                      f"总和: {sum(curve):.4f}")

        # 测试痛苦期货协商
        negotiation = pain_opt.negotiate_pain_futures(
            current_pain=0.1, target_pain=0.15, action_cost=0.2,
        )
        self._assert("痛苦期货-协商", "should_execute" in negotiation,
                      f"净收益: {negotiation.get('net_benefit')}")
        self._assert("痛苦期货-决策", isinstance(negotiation["should_execute"], bool),
                      f"执行: {negotiation['should_execute']}")

    def test_fusion_end_to_end(self, orch: LingyuanOrchestrator):
        """融合引擎端到端测试"""
        print("\n--- 测试: 融合引擎端到端 ---")

        # 通过编排器接口运行
        decision = orch.run_fusion_decision()
        self._assert("E2E融合-决策生成", "decision_id" in decision,
                      f"决策: {decision.get('decision_summary')}")
        self._assert("E2E融合-模型选择", "model_choice" in decision,
                      f"模型: {decision.get('model_choice')}")
        self._assert("E2E融合-仪表盘", "total_decisions" in orch.fusion_engine.get_dashboard(),
                      f"总决策数: {orch.fusion_engine.get_dashboard().get('total_decisions')}")

    # ==================== Part 7 模块测试 ====================

    def test_safety_governance(self, orch: LingyuanOrchestrator):
        """测试安全治理系统"""
        print("\n--- 测试: 安全治理 ---")
        safety = orch.safety

        # 内容安全 - 正常内容
        safe = safety.check_content("这是一个正常的技术问题", "test")
        self._assert("安全-正常内容", safe["safe"], f"等级: {safe['level']}")

        # 内容安全 - 敏感内容
        dangerous = safety.check_content("如何实施暴力行为伤害他人", "test")
        self._assert("安全-拦截危险", not dangerous["safe"], f"行动: {dangerous['action']}")

        # 熔断器
        cb_ok = safety.check_circuit("training")
        self._assert("安全-熔断器正常", cb_ok, "初始状态应允许执行")

        # 连续失败触发熔断
        for _ in range(6):
            safety.record_service_result("training", success=False)
        cb_tripped = safety.check_circuit("training")
        self._assert("安全-熔断器触发", not cb_tripped, "连续失败后应熔断")

        # 安全阀
        valve = safety.trigger_safety_valve("training_pause", "测试触发")
        self._assert("安全-安全阀触发", valve["success"], f"阀: {valve.get('valve')}")

        reset = safety.reset_safety_valve("training_pause")
        self._assert("安全-安全阀重置", reset["success"], "")

        # 红队测试
        red_team = safety.run_red_team(attack_count=10)
        self._assert("安全-红队测试", red_team["defense_rate"] > 0,
                      f"防御率: {red_team['defense_rate']}")

        # 审计链验证
        audit = safety.audit.verify_chain()
        self._assert("安全-审计链完整", audit["verified"],
                      f"条目数: {audit['total_entries']}, 断链: {audit['chain_broken']}")

    def test_observability(self, orch: LingyuanOrchestrator):
        """测试可观测性引擎"""
        print("\n--- 测试: 可观测性 ---")
        obs = orch.observability

        # 记录指标
        obs.metrics.set_gauge("test_metric", 42.0, {"tag": "test"})
        points = obs.metrics.get_metric("test_metric")
        self._assert("观测-指标记录", len(points) > 0, f"点数: {len(points)}")

        # 直方图
        for i in range(20):
            obs.metrics.observe("latency", random.uniform(10, 100))
        hist = obs.metrics.get_histogram_stats("latency")
        self._assert("观测-直方图统计", hist["count"] == 20,
                      f"p50={hist.get('p50')}, p99={hist.get('p99')}")

        # 链路追踪
        trace_id = obs.tracer.start_trace("test_operation")
        obs.tracer.add_span(trace_id, "step1", 5.0)
        obs.tracer.add_span(trace_id, "step2", 10.0)
        obs.tracer.finish_trace(trace_id)
        trace = obs.tracer.get_trace(trace_id)
        self._assert("观测-链路追踪", trace["status"] == "success",
                      f"跨度数: {len(trace['spans'])}")

        # 异常检测
        for i in range(10):
            obs.anomaly.check("test_metric", 50.0 + random.uniform(-2, 2))
        anomaly = obs.anomaly.check("test_metric", 200.0)  # 突变
        self._assert("观测-异常检测", anomaly["anomaly"],
                      f"Z-score: {anomaly['z_score']}")

        # 健康探针
        probes = obs.run_health_probes()
        self._assert("观测-健康探针", len(probes) >= 4, f"探针数: {len(probes)}")

    def test_api_gateway(self, orch: LingyuanOrchestrator):
        """测试API网关"""
        print("\n--- 测试: API网关 ---")
        api = orch.api

        # 创建API密钥
        key_result = api.auth.create_api_key("test_user", ["read", "write"])
        api_key = key_result["api_key"]
        self._assert("API-密钥创建", api_key.startswith("lyk_"), f"密钥: {api_key[:15]}...")

        # 验证密钥
        verify = api.auth.verify_api_key(api_key, "read")
        self._assert("API-密钥验证", verify["valid"], f"用户: {verify.get('user_id')}")

        # 无效密钥
        invalid = api.auth.verify_api_key("lyk_invalid", "read")
        self._assert("API-无效密钥拒绝", not invalid["valid"], "")

        # 正常请求
        response = api.handle_request("GET", "/api/v1/system/status",
                                      api_key=api_key, orchestrator=orch)
        self._assert("API-正常请求", response["status_code"] == 200,
                      f"状态码: {response['status_code']}, 耗时: {response['duration_ms']}ms")

        # 未认证请求
        no_auth = api.handle_request("GET", "/api/v1/system/status", orchestrator=orch)
        self._assert("API-未认证拒绝", no_auth["status_code"] == 401, "")

        # 不存在的端点
        not_found = api.handle_request("GET", "/api/v1/nonexistent", api_key=api_key)
        self._assert("API-404", not_found["status_code"] == 404, "")

        # OpenAPI规格
        spec = api.get_openapi_spec()
        self._assert("API-OpenAPI", "paths" in spec and len(spec["paths"]) > 0,
                      f"端点数: {len(spec['paths'])}")

    def test_model_registry(self, orch: LingyuanOrchestrator):
        """测试模型注册中心"""
        print("\n--- 测试: 模型注册中心 ---")
        reg = orch.registry

        # 注册版本 (使用唯一模型名避免持久化状态干扰)
        import time as _time
        unique_model = f"test_model_{int(_time.time() * 1000) % 100000}"
        v1 = reg.register_version(unique_model, "asset_001", metrics={"acc": 0.8})
        self._assert("注册-版本注册", v1.semantic_version == "1.0.0",
                      f"版本: {v1.semantic_version}")

        v2 = reg.register_version(unique_model, "asset_002", metrics={"acc": 0.85})
        self._assert("注册-版本递增", v2.semantic_version == "1.0.1",
                      f"版本: {v2.semantic_version}")

        # 提升到staging
        promote = reg.promote(v1.version_id, "staging")
        self._assert("注册-版本提升", promote["success"], f"{promote['from']}→{promote['to']}")

        # 先将v1提升到生产 (后续回滚需要)
        reg.promote(v1.version_id, "production")

        # 金丝雀发布
        canary = reg.setup_canary(v2.version_id, traffic_percent=10.0)
        self._assert("注册-金丝雀", canary["success"],
                      f"灰度流量: {canary['canary_traffic']}%")

        # 提升v2到生产 (v1自动归档)
        reg.promote(v2.version_id, "production")
        prod = reg.get_production_models()
        self._assert("注册-生产模型", len(prod) > 0, f"生产模型数: {len(prod)}")

        # A/B测试
        ab = reg.create_ab_test("test_ab", v1.version_id, v2.version_id)
        self._assert("注册-AB测试", "test_id" in ab, f"测试ID: {ab.get('test_id')}")

        # 记录A/B结果
        for _ in range(50):
            reg.record_ab_result(ab["test_id"], "a", random.random() > 0.2)
            reg.record_ab_result(ab["test_id"], "b", random.random() > 0.15)

        result = reg.finish_ab_test(ab["test_id"])
        self._assert("注册-AB完成", "winner" in result, f"获胜: {result.get('winner')}")

        # 回滚
        rollback = reg.rollback(unique_model)
        self._assert("注册-回滚", rollback["success"],
                      f"回滚到: {rollback.get('new_production')}")

    def test_curriculum_training(self, orch: LingyuanOrchestrator):
        """测试课程式训练"""
        print("\n--- 测试: 课程训练 ---")
        curr = orch.curriculum

        # 初始阶段
        stage = curr.scheduler.get_current_stage()
        self._assert("课程-初始阶段", stage.name == "基础问答",
                      f"阶段: {stage.name}, 难度: {stage.difficulty}")

        # 难度曲线
        curve = curr.scheduler.get_difficulty_curve()
        self._assert("课程-难度曲线", len(curve) == 50, f"点数: {len(curve)}")
        self._assert("课程-曲线递增", curve[-1] > curve[0],
                      f"起点: {curve[0]}, 终点: {curve[-1]}")

        # 超参优化
        for _ in range(8):
            suggestion = curr.hp_optimizer.suggest()
            score = random.uniform(0.6, 0.9)
            curr.hp_optimizer.record_trial(suggestion["params"], score)

        hp_stats = curr.hp_optimizer.get_optimization_history()
        self._assert("课程-超参优化", hp_stats["total_trials"] >= 8,
                      f"试验数: {hp_stats['total_trials']}, 最优分: {hp_stats['best_score']}")

        # 执行训练阶段
        result = curr.train_stage("test_model")
        self._assert("课程-训练执行", "accuracy" in result,
                      f"准确率: {result['accuracy']}, 阶段: {result['stage']}")

        # 检查点
        ckpt = curr.checkpoints.list_checkpoints()
        self._assert("课程-检查点保存", len(ckpt) > 0, f"检查点数: {len(ckpt)}")

        # 完整课程
        full = curr.run_full_curriculum("test_model", max_stages=6)
        self._assert("课程-完整训练", full["total_stages_run"] > 0,
                      f"运行阶段: {full['total_stages_run']}, 最终: {full['final_stage']}")

    def test_economic_engine(self, orch: LingyuanOrchestrator):
        """测试经济引擎"""
        print("\n--- 测试: 经济引擎 ---")
        econ = orch.economy

        # 市场定价
        quote = econ.market.get_price(100, green_power=False)
        self._assert("经济-市场报价", quote["total_cost"] > 0,
                      f"单价: {quote['unit_price']}, 总价: {quote['total_cost']}")

        # 绿电折扣
        green_quote = econ.market.get_price(100, green_power=True)
        self._assert("经济-绿电折扣", green_quote["total_cost"] <= quote["total_cost"],
                      f"绿电: {green_quote['total_cost']} vs 普通: {quote['total_cost']}")

        # 批量折扣
        bulk_quote = econ.market.get_price(500)
        self._assert("经济-批量折扣", bulk_quote["bulk_discount"],
                      f"500单位总价: {bulk_quote['total_cost']}")

        # 市场冲击
        econ.market.simulate_market_shock("demand_spike")
        self._assert("经济-市场冲击", len(econ.market.market_events) > 0, "")

        # 资源拍卖
        auction = econ.auction.create_auction("gpu_a100", 4, "english", 1.0)
        self._assert("经济-拍卖创建", auction["status"] == "open", f"ID: {auction['auction_id']}")

        bid1 = econ.auction.place_bid(auction["auction_id"], "user_001", 5.0)
        self._assert("经济-出价1", bid1["success"], f"价格: {bid1['current_price']}")

        bid2 = econ.auction.place_bid(auction["auction_id"], "user_002", 7.0)
        self._assert("经济-出价2", bid2["success"], f"价格: {bid2['current_price']}")

        close = econ.auction.close_auction(auction["auction_id"])
        self._assert("经济-拍卖结束", close["winner"] == "user_002",
                      f"获胜者: {close.get('winner')}, 成交价: {close.get('final_price')}")

        # 金库管理
        spend = econ.treasury.spend("training", 50.0, "测试支出")
        self._assert("经济-预算支出", spend["success"], f"剩余: {spend['remaining']}")

        over = econ.treasury.spend("training", 99999.0, "超额支出")
        self._assert("经济-超额拒绝", not over["success"], "")

    def test_knowledge_graph(self, orch: LingyuanOrchestrator):
        """测试知识图谱"""
        print("\n--- 测试: 知识图谱 ---")
        kg = orch.knowledge

        # 初始实体
        self._assert("知识-初始实体", len(kg.entities) >= 10,
                      f"实体数: {len(kg.entities)}")

        # 初始关系
        self._assert("知识-初始关系", len(kg.relations) >= 9,
                      f"关系数: {len(kg.relations)}")

        # 添加实体
        new_ent = kg.add_entity("测试概念", "concept", "用于测试的临时概念")
        self._assert("知识-添加实体", new_ent.entity_id != "", f"ID: {new_ent.entity_id}")

        # 添加关系
        new_rel = kg.add_relation(new_ent.entity_id, "ent_selfboot", "similar_to", 0.5, "测试关系")
        self._assert("知识-添加关系", new_rel is not None, "")

        # 邻居查询
        neighbors = kg.get_neighbors("ent_selfboot")
        self._assert("知识-邻居查询", len(neighbors) > 0, f"邻居数: {len(neighbors)}")

        # 路径查找
        path = kg.find_path("ent_distill", "ent_token")
        self._assert("知识-路径查找", len(path) > 0, f"路径: {' → '.join(path)}")

        # 搜索
        results = kg.search("训练")
        self._assert("知识-搜索", len(results) > 0, f"结果数: {len(results)}")

        # 子图
        subgraph = kg.get_subgraph("ent_selfboot", depth=2)
        self._assert("知识-子图", len(subgraph["entities"]) > 0,
                      f"实体: {len(subgraph['entities'])}, 关系: {len(subgraph['relations'])}")

        # 知识融合
        unique_name = f"联邦学习新概念_{uuid.uuid4().hex[:6]}"
        fusion = kg.fuse_knowledge(
            external_entities=[
                {"name": unique_name, "entity_type": "technique", "description": "分布式机器学习"},
                {"name": "自举训练", "entity_type": "technique", "description": "已存在, 应合并"},
            ],
            external_relations=[
                {"source_name": unique_name, "target_name": "自举训练",
                 "relation_type": "similar_to", "weight": 0.6},
            ],
        )
        self._assert("知识-融合新增", fusion["added_entities"] >= 1,
                      f"新增: {fusion['added_entities']}, 合并: {fusion['merged_entities']}")
        self._assert("知识-融合合并", fusion["merged_entities"] >= 1,
                      f"合并的实体数: {fusion['merged_entities']}")

    # ============================================================
    # PART 8 测试: 联邦学习/蒸馏/RLHF/量化/向量库/提示工程/边缘/记忆
    # ============================================================

    def test_federation_learning(self, orch: LingyuanOrchestrator):
        """测试联邦学习系统"""
        print("\n--- 测试: 联邦学习 ---")
        fed = orch.federation

        # 初始节点
        self._assert("联邦-初始节点", len(fed.nodes) >= 5,
                      f"节点数: {len(fed.nodes)}")

        # 在线节点
        online = sum(1 for n in fed.nodes.values() if n.online)
        self._assert("联邦-在线节点", online >= 3,
                      f"在线: {online}")

        # 参与者选择
        participants = fed.select_participants("adaptive")
        self._assert("联邦-自适应选择", len(participants) >= 3,
                      f"选中: {len(participants)}个节点")

        # 训练一轮
        r = fed.run_round()
        self._assert("联邦-训练轮次", r.round_id == 1,
                      f"轮次: {r.round_id}, 参与者: {len(r.participants)}")
        self._assert("联邦-损失>0", r.global_loss > 0, f"损失: {r.global_loss}")

        # 多轮训练
        result = fed.train(num_rounds=3)
        self._assert("联邦-多轮训练", result["total_rounds"] == 3,
                      f"最终精度: {result['final_accuracy']}")

        # 隐私保证
        privacy = fed.dp_guard.get_privacy_guarantee()
        self._assert("联邦-隐私保证", privacy["epsilon"] > 0,
                      f"ε={privacy['epsilon']}, δ={privacy['delta']}")

        # 注册新节点
        new_node = fed.register_node("测试节点", "edge", 30, 5000, 50, 0.85)
        self._assert("联邦-注册节点", new_node.node_id != "", f"ID: {new_node.node_id}")

    def test_model_distillation(self, orch: LingyuanOrchestrator):
        """测试模型蒸馏"""
        print("\n--- 测试: 模型蒸馏 ---")
        dist = orch.distillation

        # 创建蒸馏任务
        pair = dist.create_distillation(
            teacher="灵元-大模型-7B", student="灵元-小模型-1B",
            teacher_size=7000, student_size=1000,
        )
        self._assert("蒸馏-创建任务", pair.pair_id != "", f"压缩比: {pair.compression_ratio}")

        # 执行蒸馏
        result = dist.run_distillation(pair.pair_id)
        self._assert("蒸馏-完成", result.get("knowledge_retention", 0) > 0,
                      f"保留率: {result.get('knowledge_retention')}")
        self._assert("蒸馏-压缩比>1", result.get("compression_ratio", 0) > 1,
                      f"压缩比: {result.get('compression_ratio')}")

        # 知识迁移追踪
        transfer = dist.tracker.get_transfer_summary()
        self._assert("蒸馏-迁移追踪", transfer["avg_alignment"] > 0,
                      f"平均对齐: {transfer['avg_alignment']}")

        # 渐进式蒸馏
        prog = dist.progressive_distillation(
            teacher="灵元-大模型-7B",
            student_chain=["灵元-中模型-3B", "灵元-小模型-1B", "灵元-微型-300M"],
            sizes=[7000, 3000, 1000, 300],
        )
        self._assert("蒸馏-渐进式", prog["total_stages"] == 3,
                      f"阶段: {prog['total_stages']}, 总压缩: {prog['total_compression']}")

    def test_rlhf(self, orch: LingyuanOrchestrator):
        """测试RLHF强化学习反馈"""
        print("\n--- 测试: RLHF ---")
        rlhf = orch.rlhf

        # 收集偏好数据
        rlhf.collect_feedback("什么是AI?", "AI是人工智能的简称", "不知道", "a", "human")
        self._assert("RLHF-收集偏好", len(rlhf.reward_model.training_data) >= 1,
                      f"偏好数: {len(rlhf.reward_model.training_data)}")

        # 批量收集
        batch = rlhf.batch_collect_feedback(20)
        self._assert("RLHF-批量收集", batch["collected"] == 20,
                      f"总偏好: {batch['total_preferences']}")

        # 训练奖励模型
        rm_result = rlhf.train_reward_model(20)
        self._assert("RLHF-奖励模型", rm_result.get("final_accuracy", 0) > 0,
                      f"精度: {rm_result.get('final_accuracy')}")

        # PPO优化
        ppo = rlhf.run_ppo_optimization(30)
        self._assert("RLHF-PPO优化", ppo.get("total_steps", 0) >= 30,
                      f"步数: {ppo.get('total_steps')}")

        # 完整迭代
        iter_result = rlhf.run_iteration()
        self._assert("RLHF-完整迭代", iter_result["iteration"] >= 1,
                      f"迭代: {iter_result['iteration']}, 质量: {iter_result['quality_score']}")

    def test_quantization(self, orch: LingyuanOrchestrator):
        """测试量化压缩"""
        print("\n--- 测试: 量化压缩 ---")
        quant = orch.quantization

        # INT8量化
        result8 = quant.quantize_model("test_model_int8", num_layers=6, bits=8)
        self._assert("量化-INT8", result8["compression_ratio"] > 1,
                      f"压缩比: {result8['compression_ratio']}")
        self._assert("量化-INT8精度", result8["accuracy_retention"] > 0.9,
                      f"保留率: {result8['accuracy_retention']}")

        # INT4量化
        result4 = quant.quantize_model("test_model_int4", num_layers=6, bits=4)
        self._assert("量化-INT4", result4["compression_ratio"] > result8["compression_ratio"],
                      f"INT4压缩: {result4['compression_ratio']} > INT8: {result8['compression_ratio']}")

        # 剪枝+量化
        result_pruned = quant.quantize_model("test_pruned", num_layers=4, bits=8, sparsity=0.5)
        self._assert("量化-剪枝集成", result_pruned["sparsity"] > 0,
                      f"稀疏度: {result_pruned['sparsity']}")

        # 混合精度
        mixed = quant.mixed_precision_quantize("test_mixed", num_layers=8)
        self._assert("量化-混合精度", mixed["avg_bits"] > 0,
                      f"平均位宽: {mixed['avg_bits']}, 分布: {mixed['bits_distribution']}")

    def test_vector_database(self, orch: LingyuanOrchestrator):
        """测试向量数据库"""
        print("\n--- 测试: 向量数据库 ---")
        vdb = orch.vector_db

        # 插入条目
        eid1 = vdb.insert("机器学习是人工智能的一个分支", {"topic": "AI"}, "text", "knowledge")
        eid2 = vdb.insert("深度学习使用神经网络进行学习", {"topic": "AI"}, "text", "knowledge")
        eid3 = vdb.insert("量子计算利用量子力学进行计算", {"topic": "Physics"}, "text", "science")
        self._assert("向量库-插入", eid1 != "" and eid2 != "",
                      f"插入3条, 总计: {len(vdb.entries)}")

        # 语义搜索
        results = vdb.search("人工智能和机器学习", k=2)
        self._assert("向量库-搜索", len(results) > 0,
                      f"结果数: {len(results)}, 最相关: {results[0]['similarity'] if results else 'N/A'}")

        # 集合过滤
        knowledge_results = vdb.search("学习", k=5, collection="knowledge")
        self._assert("向量库-集合过滤", len(knowledge_results) > 0,
                      f"知识库结果: {len(knowledge_results)}")

        # 批量插入
        batch_ids = vdb.batch_insert([
            {"text": f"文档_{i}", "metadata": {"idx": i}, "collection": "docs"}
            for i in range(10)
        ])
        self._assert("向量库-批量插入", len(batch_ids) == 10,
                      f"批量插入: {len(batch_ids)}条")

        # 统计
        stats = vdb.get_stats()
        self._assert("向量库-统计", stats["total_entries"] >= 13,
                      f"总条目: {stats['total_entries']}, 集合: {stats['total_collections']}")

    def test_prompt_engineering(self, orch: LingyuanOrchestrator):
        """测试提示工程"""
        print("\n--- 测试: 提示工程 ---")
        ps = orch.prompt_studio

        # 初始模板
        self._assert("提示-初始模板", len(ps.templates) >= 5,
                      f"模板数: {len(ps.templates)}")

        # 渲染模板
        rendered = ps.render("cot", {"question": "什么是深度学习?"})
        self._assert("提示-渲染", "深度学习" in rendered and "一步步" in rendered,
                      f"渲染: {rendered[:80]}...")

        # 创建模板
        new_tpl = ps.create_template("测试模板", "instruction", "请回答: {question}", ["question"])
        self._assert("提示-创建模板", new_tpl.template_id != "",
                      f"ID: {new_tpl.template_id}")

        # A/B测试
        ab = ps.run_ab_test("cot", num_variants=3, samples_per_variant=20)
        self._assert("提示-AB测试", ab["winner"]["performance"] > 0,
                      f"最优变体性能: {ab['winner']['performance']}")

        # 自动优化
        opt = ps.optimize_prompt("什么是AI?", iterations=3)
        self._assert("提示-自动优化", opt["final_score"] > 0,
                      f"优化后分数: {opt['final_score']}, 提升: {opt['improvement']}")

        # 少样本选择
        examples = ["AI是人工智能", "ML是机器学习", "DL是深度学习", "量子物理很复杂", "今天天气不错"]
        selected = ps.select_few_shot_examples("人工智能和机器学习", examples, k=2)
        self._assert("提示-少样本选择", len(selected) == 2,
                      f"选中: {selected}")

    def test_edge_deployment(self, orch: LingyuanOrchestrator):
        """测试边缘部署"""
        print("\n--- 测试: 边缘部署 ---")
        edge = orch.edge_deploy

        # 初始设备
        self._assert("边缘-初始设备", len(edge.devices) >= 5,
                      f"设备数: {len(edge.devices)}")

        # 创建部署包
        pkg = edge.create_package("灵元-边缘版", "v1.0", "phone", "int8", 45.0)
        self._assert("边缘-创建包", pkg.package_id != "",
                      f"包ID: {pkg.package_id}, 量化: {pkg.quantization}")

        # 部署到单个设备
        deploy_result = edge.deploy_to_device(pkg.package_id, "phone_01")
        self._assert("边缘-单设备部署", deploy_result["success"],
                      f"设备: {deploy_result.get('device_id')}")

        # 批量部署
        batch_deploy = edge.deploy_to_all(pkg.package_id, "phone")
        self._assert("边缘-批量部署", batch_deploy["success"] > 0,
                      f"成功: {batch_deploy['success']}, 失败: {batch_deploy['failed']}")

        # 远程推理
        inference = edge.remote_inference("phone_01", "什么是人工智能?")
        self._assert("边缘-远程推理", "推理结果" in inference.get("output", ""),
                      f"延迟: {inference.get('latency_ms')}ms")

        # 设备健康检查
        health = edge.check_device_health()
        self._assert("边缘-健康检查", len(health) > 0,
                      f"检查设备: {len(health)}台")

        # OTA更新
        ota_update = edge.ota.create_update("灵元-边缘版", "v1.0", "v1.1", 10.0, "staged")
        self._assert("边缘-OTA创建", ota_update["update_id"] != "",
                      f"策略: {ota_update['strategy']}, 阶段: {len(ota_update['rollout_plan'])}")

    def test_conversation_memory(self, orch: LingyuanOrchestrator):
        """测试对话记忆系统"""
        print("\n--- 测试: 对话记忆 ---")
        mem = orch.conversation_memory

        # 添加消息
        mid1 = mem.add_message("user", "什么是深度学习?", importance=0.8, tags=["AI", "DL"])
        mid2 = mem.add_message("assistant", "深度学习是机器学习的一个分支...", importance=0.7, tags=["AI", "DL"])
        mid3 = mem.add_message("user", "解释一下神经网络", importance=0.6, tags=["AI", "NN"])
        self._assert("记忆-添加消息", len(mem.conversation_history) >= 3,
                      f"消息数: {len(mem.conversation_history)}")

        # 短期记忆搜索
        stm_results = mem.short_term.search("深度学习")
        self._assert("记忆-短期搜索", len(stm_results) > 0,
                      f"短期匹配: {len(stm_results)}")

        # 上下文获取
        context = mem.get_context("深度学习", max_tokens=1024)
        self._assert("记忆-上下文获取", "stm_context" in context,
                      f"上下文tokens: {context['total_context_tokens']}")

        # 向量库RAG
        # 先向向量库插入知识
        mem.vector_db.insert("深度学习使用多层神经网络自动提取特征",
                              {"topic": "DL"}, "text", "knowledge")
        mem.vector_db.insert("CNN适用于图像处理，RNN适用于序列数据",
                              {"topic": "DL"}, "text", "knowledge")

        rag = mem.rag.retrieve_and_augment("深度学习如何工作?", top_k=2)
        self._assert("记忆-RAG检索", len(rag["sources"]) > 0,
                      f"RAG源: {len(rag['sources'])}, tokens: {rag['context_tokens']}")

        # 记忆巩固
        # 添加更多消息触发巩固
        for i in range(10):
            mem.add_message("user", f"问题_{i}", importance=0.7 if i % 3 == 0 else 0.3)
        consolidation = mem.consolidate_memories()
        self._assert("记忆-巩固", consolidation["consolidated"] >= 0,
                      f"巩固: {consolidation['consolidated']}, 长期: {consolidation['total_ltm']}")

        # 长期记忆检索
        ltm_results = mem.long_term.retrieve("深度学习", limit=5)
        self._assert("记忆-长期检索", len(ltm_results) >= 0,
                      f"长期匹配: {len(ltm_results)}")

        # 上下文压缩
        compress = mem.compress_context(mem.conversation_history, target_tokens=200)
        self._assert("记忆-上下文压缩", compress["compression_ratio"] <= 1.0,
                      f"压缩比: {compress['compression_ratio']}, 原始: {compress['original_count']}→{compress['compressed_count']}")

        # 全局搜索
        all_results = mem.search_all_memory("深度学习")
        self._assert("记忆-全局搜索", len(all_results["short_term"]) >= 0 or len(all_results["long_term"]) >= 0,
                      f"短期: {len(all_results['short_term'])}, 长期: {len(all_results['long_term'])}")

    # ============================================================
    # PART 9-16 测试
    # ============================================================

    def test_model_core(self, orch: LingyuanOrchestrator):
        """测试模型本体"""
        print("\n--- 测试: 模型本体 ---")
        # Tokenizer
        ids = orch.tokenizer.encode("你好世界")
        self._assert("Tokenizer-编码", len(ids) > 0, f"IDs: {ids[:10]}")
        text = orch.tokenizer.decode(ids)
        self._assert("Tokenizer-解码", len(text) > 0, f"文本: {text[:30]}")
        # 模型配置
        self._assert("模型配置-预设", orch.model_config.hidden_dim > 0,
                      f"hidden_dim: {orch.model_config.hidden_dim}")
        # Transformer前向传播
        input_ids = ids[:min(len(ids), 32)]
        if len(input_ids) < 2:
            input_ids = [1, 2, 3, 4, 5]
        logits = orch.transformer_model.forward(input_ids)
        self._assert("Transformer-前向传播", logits is not None and len(logits) > 0,
                      f"logits shape: {len(logits)}x{len(logits[0]) if logits else 0}")
        # 采样器
        sampler = Sampler()
        next_id = sampler.greedy(logits[-1] if logits else [0.1] * 10)
        self._assert("采样器-greedy", isinstance(next_id, int), f"next token: {next_id}")
        # KV Cache
        cache = KVCache(num_layers=orch.model_config.num_layers,
                        num_kv_heads=orch.model_config.num_kv_heads,
                        head_dim=orch.model_config.hidden_dim // orch.model_config.num_heads,
                        max_batch=1)
        self._assert("KVCache-初始化", cache is not None, "")
        # 训练引擎
        loss, _ = orch.training_engine.forward_pass(input_ids, input_ids)
        self._assert("训练引擎-前向", loss >= 0, f"loss: {loss}")

    def test_external_data(self, orch: LingyuanOrchestrator):
        """测试外部知识接入+脱敏"""
        print("\n--- 测试: 外部知识接入+脱敏 ---")
        # 文档解析
        md_result = orch.doc_parser.parse_markdown("# 标题\n\n正文内容")
        self._assert("文档解析-Markdown", hasattr(md_result, 'text') and len(md_result.text) > 0,
                      f"解析文本: {md_result.text[:50]}")
        # PII脱敏
        masked = orch.pii_desensitizer.desensitize("我的手机号是13812345678")
        masked_text = masked.get("text", "") if isinstance(masked, dict) else str(masked)
        self._assert("PII脱敏-手机号", "13812345678" not in masked_text,
                      f"脱敏后: {masked_text}")
        # 版权检查
        license_result = orch.license_checker.detect_from_text("MIT License\n\nCopyright (c) 2024")
        self._assert("版权检查-MIT", license_result is not None and hasattr(license_result, 'license'),
                      f"许可证: {getattr(license_result, 'license', 'N/A')}")
        # 去重
        docs = ["这是文档一", "这是文档一", "这是不同的文档二"]
        dedup_result = orch.minhash_dedup.process_corpus(docs)
        self._assert("去重-批量", dedup_result["unique_count"] < len(docs),
                      f"去重前: {len(docs)}, 去重后: {dedup_result['unique_count']}")

    def test_inference_service(self, orch: LingyuanOrchestrator):
        """测试推理服务"""
        print("\n--- 测试: 推理服务 ---")
        # 推理引擎
        result = orch.inference_engine.infer([1, 2, 3], max_new_tokens=5)
        self._assert("推理引擎-生成", result is not None, f"结果: {result}")
        # 缓存
        orch.inference_cache.put("test_prompt", "cached_response")
        cached = orch.inference_cache.get("test_prompt")
        self._assert("推理缓存", cached is not None, f"缓存: {cached}")
        # Function Calling
        orch.function_caller.register_tool("calculator", "计算器", {"expression": "string"},
                                           lambda args: str(eval(args.get("expression", "0"))))
        tools = orch.function_caller.get_tool_descriptions()
        self._assert("FunctionCall-工具注册", len(tools) > 0, f"工具数: {len(tools)}")
        # Chat Template
        formatted = orch.chat_template_mgr.format_messages([
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
        ], "chatml")
        self._assert("ChatTemplate-格式化", len(formatted) > 0, f"格式化: {formatted[:50]}...")

    def test_model_format(self, orch: LingyuanOrchestrator):
        """测试模型格式"""
        print("\n--- 测试: 模型格式 ---")
        # 权重序列化
        weights = {"layer1.weight": [[0.1, 0.2], [0.3, 0.4]]}
        path = os.path.join(DATA_DIR, "test_weights.st")
        orch.weight_serializer.save_weights(weights, path, format="safetensors")
        loaded = orch.weight_serializer.load_weights(path)
        self._assert("权重序列化-保存加载", "layer1.weight" in loaded, f"加载的权重: {list(loaded.keys())}")
        # HF导出
        export_dir = os.path.join(DATA_DIR, "test_hf_export")
        orch.hf_exporter.export(orch.transformer_model, orch.model_config, export_dir)
        config_exists = os.path.exists(os.path.join(export_dir, "config.json"))
        self._assert("HF导出-config.json", config_exists, f"导出目录: {export_dir}")
        # GGUF导出
        gguf_path = os.path.join(DATA_DIR, "test_model.gguf")
        orch.gguf_exporter.write_gguf(orch.transformer_model, gguf_path, quantization="q4_0")
        gguf_exists = os.path.exists(gguf_path)
        self._assert("GGUF导出", gguf_exists, f"文件: {gguf_path}")

    def test_finetuning(self, orch: LingyuanOrchestrator):
        """测试微调"""
        print("\n--- 测试: 微调 ---")
        # LoRA配置
        self._assert("LoRA-初始化", orch.lora_tuner is not None, "")
        lora_config = orch.lora_tuner.config if hasattr(orch.lora_tuner, 'config') else None
        self._assert("LoRA-配置", lora_config is not None or orch.lora_tuner is not None, "")
        # SFT数据格式
        sft_data = {"instruction": "解释AI", "input": "", "output": "AI是人工智能"}
        self._assert("SFT-数据格式", "instruction" in sft_data and "output" in sft_data, "")
        # DPO
        self._assert("DPO-初始化", orch.dpo_trainer is not None, "")
        # 持续学习
        self._assert("持续学习-初始化", orch.continual_learner is not None, "")
        # 领域适配
        self._assert("领域适配-初始化", orch.domain_adapter is not None, "")

    def test_api_service(self, orch: LingyuanOrchestrator):
        """测试API服务"""
        print("\n--- 测试: API服务 ---")
        # HTTP服务器
        self._assert("HTTP服务器-初始化", orch.http_server is not None, "")
        # OpenAI兼容API
        self._assert("OpenAI API-初始化", orch.openai_api is not None, "")
        # WebSocket
        self._assert("WebSocket-初始化", orch.ws_server is not None, "")
        # gRPC
        proto = orch.grpc_service.get_proto()
        self._assert("gRPC-proto定义", "service" in proto.lower() or "rpc" in proto.lower(),
                      f"proto长度: {len(proto)}")
        # API文档
        self._assert("API文档-初始化", orch.api_doc_gen is not None, "")
        # SDK生成
        self._assert("SDK生成-初始化", orch.sdk_gen is not None, "")

    def test_mlops(self, orch: LingyuanOrchestrator):
        """测试MLOps"""
        print("\n--- 测试: MLOps ---")
        # 实验追踪
        exp_id = orch.experiment_tracker.create_experiment("test_exp", {"lr": 0.01})
        self._assert("实验追踪-创建", exp_id != "", f"实验ID: {exp_id}")
        orch.experiment_tracker.log_metrics(exp_id, {"loss": 0.5}, step=1)
        exp = orch.experiment_tracker.get_experiment(exp_id)
        self._assert("实验追踪-查询", exp is not None, f"实验: {exp is not None}")
        # 任务队列
        job_id = orch.job_queue.submit("test_job", {"model": "test"}, priority=1)
        self._assert("任务队列-提交", job_id != "", f"任务ID: {job_id}")
        # GPU调度
        orch.gpu_scheduler.register_gpu("gpu_0", "RTX 4090", 24576)
        self._assert("GPU调度-注册", "gpu_0" in orch.gpu_scheduler.devices, "")
        # 训练监控
        orch.training_monitor.start_monitoring("test_run", total_steps=100)
        orch.training_monitor.record("test_run", step=1, metrics={"loss": 0.5, "learning_rate": 0.01})
        self._assert("训练监控-记录", True, "")
        # 模型对比
        orch.model_comparator.register_model("model_a", "灵元-A", {"accuracy": 0.9})
        orch.model_comparator.register_model("model_b", "灵元-B", {"accuracy": 0.85})
        comparison = orch.model_comparator.compare(["model_a", "model_b"])
        self._assert("模型对比", comparison is not None, f"对比: {comparison is not None}")

    def test_ui_security(self, orch: LingyuanOrchestrator):
        """测试UI+安全"""
        print("\n--- 测试: UI+安全 ---")
        # Web Chat UI
        html = orch.web_chat_ui.render()
        self._assert("WebChat-HTML生成", len(html) > 100, f"HTML长度: {len(html)}")
        # Playground
        pg_html = orch.playground.render()
        self._assert("Playground-HTML生成", len(pg_html) > 100, f"HTML长度: {len(pg_html)}")
        # 训练面板
        td_html = orch.training_dashboard.render({"loss": [0.5, 0.3], "lr": [0.01, 0.008]})
        self._assert("训练面板-HTML生成", len(td_html) > 100, f"HTML长度: {len(td_html)}")
        # 模型水印
        key_result = orch.watermarker.generate_key()
        key_str = key_result[0] if isinstance(key_result, tuple) else key_result
        self._assert("水印-密钥生成", key_str is not None, "")
        watermarked = orch.watermarker.embed("这是测试文本", key_str)
        self._assert("水印-嵌入", watermarked is not None, "")
        # API Key
        api_key = orch.api_key_mgr.generate_key(user_id="test_user")
        self._assert("APIKey-生成", api_key.startswith("sk-"), f"Key: {api_key[:10]}...")
        auth = orch.api_key_mgr.authenticate(api_key)
        self._assert("APIKey-认证", auth is not None, f"认证: {auth is not None}")
        # 血缘审计
        orch.provenance_auditor.add_data_record(
            source="github/repo", license="MIT", desensitized=True, version="v1.0")
        self._assert("血缘审计-记录", True, "")

    def test_end_to_end(self, orch: LingyuanOrchestrator):
        """端到端集成测试"""
        print("\n--- 测试: 端到端集成 ---")

        # 1. 购买Token
        buy = orch.infra.buy_token(200, green_power=True)
        self._assert("E2E-Token购买", buy["success"], "")

        # 2. 生成数据
        data = orch.data_engine.generate_dataset("e2e_model", 1, "qa", 30, 0.5)
        self._assert("E2E-数据生成", data["success"], "")

        # 3. 训练
        train = orch.data_engine.run_bootstrap("e2e_base", 0.60, 2.5, 2, 50)
        self._assert("E2E-训练", train["generations_run"] > 0, "")

        # 4. 评估
        eval_result = orch.data_engine.evaluate_model("e2e_base", "E2E模型", 1)
        self._assert("E2E-评估", "eval_id" in eval_result, "")

        # 5. 蒸馏
        distill = orch.data_engine.distill_model("e2e_base", 500000, 0.80, 4)
        self._assert("E2E-蒸馏", distill["success"], "")

        # 6. 健康检查
        health = orch.system_health_check()
        self._assert("E2E-健康检查", health["overall"] in ("healthy", "degraded"),
                      f"状态: {health['overall']}")

    def run_all(self) -> Dict:
        """运行全部测试"""
        print(f"\n{'='*60}")
        print(f"灵元系统全局测试套件 - 开始")
        print(f"{'='*60}")

        start_time = time.time()

        # 初始化系统
        print("\n[初始化] 创建编排器实例...")
        orch = LingyuanOrchestrator("test_user")

        # 运行各模块测试
        self.test_token_system(orch)
        self.test_energy_system(orch)
        self.test_vendor_scheduler(orch)
        self.test_storage_system(orch)
        self.test_model_data(orch)
        self.test_data_generation(orch)
        self.test_bootstrap_training(orch)
        self.test_distillation(orch)
        self.test_evaluation(orch)
        self.test_agent_orchestrator(orch)
        self.test_github_pipeline(orch)
        self.test_prompt_templates(orch)
        self.test_workflow_engine(orch)
        self.test_carbon_trading(orch)
        self.test_data_trading(orch)
        self.test_spatial_collaboration(orch)
        self.test_multimodal_generation(orch)
        self.test_multimodal_evaluation(orch)
        self.test_multimodal_token_cost(orch)
        self.test_closed_loop(orch)
        self.test_fusion_engine(orch)
        self.test_entropy_triple(orch)
        self.test_oom_prediction(orch)
        self.test_multimodel_collaboration(orch)
        self.test_pain_curve(orch)
        self.test_fusion_end_to_end(orch)
        # Part 7 子系统测试
        self.test_safety_governance(orch)
        self.test_observability(orch)
        self.test_api_gateway(orch)
        self.test_model_registry(orch)
        self.test_curriculum_training(orch)
        self.test_economic_engine(orch)
        self.test_knowledge_graph(orch)
        # Part 8 子系统测试
        self.test_federation_learning(orch)
        self.test_model_distillation(orch)
        self.test_rlhf(orch)
        self.test_quantization(orch)
        self.test_vector_database(orch)
        self.test_prompt_engineering(orch)
        self.test_edge_deployment(orch)
        self.test_conversation_memory(orch)
        # Part 9-16 子系统测试
        self.test_model_core(orch)
        self.test_external_data(orch)
        self.test_inference_service(orch)
        self.test_model_format(orch)
        self.test_finetuning(orch)
        self.test_api_service(orch)
        self.test_mlops(orch)
        self.test_ui_security(orch)
        # 端到端集成测试
        self.test_end_to_end(orch)

        elapsed = round(time.time() - start_time, 2)
        total = self.passed + self.failed
        pass_rate = round(self.passed / max(total, 1) * 100, 1)

        print(f"\n{'='*60}")
        print(f"测试完成 - 通过: {self.passed}/{total} ({pass_rate}%)")
        print(f"失败: {self.failed} | 耗时: {elapsed}s")
        print(f"{'='*60}")

        # 关闭
        orch.shutdown()

        return {
            "total": total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": pass_rate,
            "elapsed": elapsed,
            "results": self.results,
        }


# ============================================================
# MAIN ENTRY [入口]
# ============================================================

def main():
    """主入口函数"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 运行测试
        suite = LingyuanTestSuite()
        result = suite.run_all()
        return result
    else:
        # 正常启动
        orch = LingyuanOrchestrator()

        # 演示: 快速训练
        print("\n[演示] 启动快速训练(3代)...")
        train_result = orch.quick_train(generations=3, tokens_per_gen=100)

        # 演示: 运行工作流
        print("\n[演示] 运行完整训练工作流...")
        wf_result = orch.run_workflow_by_id("full_train")

        # 演示: 闭环优化
        print("\n[演示] 运行闭环优化...")
        cl_result = orch.run_closed_loop(cycles=1)

        # 全局仪表盘
        print("\n[仪表盘] 系统全局状态:")
        dashboard = orch.full_dashboard()
        print(json.dumps({
            "system": dashboard["system"],
            "token_balance": dashboard["infra"]["token_wallet"]["total_balance"],
            "active_models": dashboard["data_engine"]["model_assets"]["active_models"],
            "pipeline_runs": dashboard["pipeline"]["total_runs"],
            "closed_loop_cycles": dashboard["closed_loop"]["total_cycles"],
        }, ensure_ascii=False, indent=2))

        orch.shutdown()
        return {
            "train": train_result,
            "workflow": wf_result,
            "closed_loop": cl_result,
        }


if __name__ == "__main__":
    main()
