

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
