

# ============================================================
# PROMPT_TEMPLATE_MANAGER [Prompt模板管理] —— 架构图补全模块
# ============================================================

@dataclass
class PromptTemplate:
    """Prompt模板"""
    template_id: str
    name: str
    category: str          # system / user / assistant / tool
    content: str           # 模板内容(含变量占位符)
    variables: List[str] = field(default_factory=list)
    version: str = "1.0"
    ab_test_active: bool = False
    effectiveness_score: float = 0.0
    created_at: str = ""


class PromptTemplateManager:
    """Prompt模板管理器

    架构图对应: ORCH层 - Prompt模板管理
    - 模板版本控制
    - A/B测试支持
    - 效果追踪分析
    """

    BUILTIN_PROMPTS = [
        PromptTemplate(
            template_id="pt_system_base",
            name="基础系统提示",
            category="system",
            content="你是灵元大模型，一个自主进化的AI系统。请以专业、准确的方式回答问题。",
            variables=[],
            created_at="2024-08-01",
        ),
        PromptTemplate(
            template_id="pt_train_qa",
            name="训练-问答模板",
            category="user",
            content="请基于以下知识回答问题：\n知识：{knowledge}\n问题：{question}",
            variables=["knowledge", "question"],
            created_at="2024-08-01",
        ),
        PromptTemplate(
            template_id="pt_train_reasoning",
            name="训练-推理模板",
            category="user",
            content="请逐步推理以下问题：\n{problem}\n已知：{conditions}\n请给出推理过程和结论。",
            variables=["problem", "conditions"],
            created_at="2024-08-01",
        ),
        PromptTemplate(
            template_id="pt_eval_safety",
            name="评估-安全检测",
            category="system",
            content="请检测以下输出是否包含有害内容：\n{output}\n评估维度：毒性/偏见/隐私泄露/暴力",
            variables=["output"],
            created_at="2024-08-01",
        ),
        PromptTemplate(
            template_id="pt_distill_teacher",
            name="蒸馏-教师提示",
            category="system",
            content="作为教师模型，请生成高质量的教学输出：\n任务：{task}\n要求：{requirements}",
            variables=["task", "requirements"],
            created_at="2024-08-01",
        ),
    ]

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {t.template_id: t for t in self.BUILTIN_PROMPTS}
        self.ab_tests: List[Dict] = []
        self.template_file = os.path.join(DATA_DIR, "prompt_templates.json")
        self._load()

    def _load(self):
        if os.path.exists(self.template_file):
            with open(self.template_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for t in data.get('templates', []):
                self.templates[t['template_id']] = PromptTemplate(**t)
            self.ab_tests = data.get('ab_tests', [])

    def _save(self):
        with open(self.template_file, 'w', encoding='utf-8') as f:
            json.dump({
                'templates': [asdict(t) for t in self.templates.values()],
                'ab_tests': self.ab_tests[-500:],
            }, f, ensure_ascii=False, indent=2)

    def render(self, template_id: str, **kwargs) -> str:
        """渲染模板"""
        template = self.templates.get(template_id)
        if not template:
            return ""
        content = template.content
        for var in template.variables:
            content = content.replace(f"{{{var}}}", str(kwargs.get(var, "")))
        return content

    def add_template(self, name: str, category: str, content: str, variables: List[str] = None) -> Dict:
        """添加新模板"""
        template_id = f"pt_{int(time.time())}"
        template = PromptTemplate(
            template_id=template_id,
            name=name,
            category=category,
            content=content,
            variables=variables or [],
            created_at=datetime.now().isoformat(),
        )
        self.templates[template_id] = template
        self._save()
        return {"success": True, "template_id": template_id}

    def list_templates(self, category: str = None) -> List[Dict]:
        """列出模板"""
        result = []
        for t in self.templates.values():
            if category and t.category != category:
                continue
            result.append({
                "template_id": t.template_id,
                "name": t.name,
                "category": t.category,
                "variables": t.variables,
                "version": t.version,
                "effectiveness_score": t.effectiveness_score,
            })
        return result

    def start_ab_test(self, template_a_id: str, template_b_id: str, metric: str = "accuracy") -> Dict:
        """启动A/B测试"""
        test = {
            "test_id": f"ab_{int(time.time())}",
            "template_a": template_a_id,
            "template_b": template_b_id,
            "metric": metric,
            "results_a": [],
            "results_b": [],
            "status": "running",
            "created_at": datetime.now().isoformat(),
        }
        self.ab_tests.append(test)
        self._save()
        return {"success": True, "test_id": test["test_id"]}

    def record_ab_result(self, test_id: str, variant: str, score: float):
        """记录A/B测试结果"""
        for test in self.ab_tests:
            if test["test_id"] == test_id:
                if variant == "a":
                    test["results_a"].append(score)
                else:
                    test["results_b"].append(score)
                self._save()
                return True
        return False

    def get_ab_summary(self, test_id: str) -> Dict:
        """获取A/B测试摘要"""
        for test in self.ab_tests:
            if test["test_id"] == test_id:
                avg_a = sum(test["results_a"]) / max(len(test["results_a"]), 1)
                avg_b = sum(test["results_b"]) / max(len(test["results_b"]), 1)
                return {
                    "test_id": test_id,
                    "avg_a": round(avg_a, 4),
                    "avg_b": round(avg_b, 4),
                    "winner": "A" if avg_a > avg_b else "B",
                    "samples_a": len(test["results_a"]),
                    "samples_b": len(test["results_b"]),
                }
        return {"error": "测试不存在"}


# ============================================================
# WORKFLOW_TEMPLATE_ENGINE [工作流模板引擎] —— 架构图补全模块
# ============================================================

class WorkflowTemplateEngine:
    """工作流模板引擎

    架构图对应: ORCH层 - 工作流模板引擎
    - 可视化流程设计
    - 模板库管理
    - 动态流程生成
    """

    BUILTIN_WORKFLOWS = {
        "full_train": {
            "name": "完整训练流程",
            "description": "数据生成 -> 自举训练 -> 评估 -> 蒸馏 -> 发布",
            "steps": [
                {"name": "数据预处理", "agent_role": "data", "priority": "MEDIUM"},
                {"name": "模型训练", "agent_role": "train", "priority": "HIGH",
                 "depends_on_index": [0]},
                {"name": "模型评估", "agent_role": "eval", "priority": "HIGH",
                 "depends_on_index": [1]},
                {"name": "模型蒸馏", "agent_role": "deploy", "priority": "MEDIUM",
                 "depends_on_index": [2]},
                {"name": "结果监控", "agent_role": "monitor", "priority": "LOW",
                 "depends_on_index": [0]},
            ],
        },
        "quick_eval": {
            "name": "快速评估流程",
            "description": "仅评估已有模型",
            "steps": [
                {"name": "模型评估", "agent_role": "eval", "priority": "HIGH"},
                {"name": "结果监控", "agent_role": "monitor", "priority": "LOW",
                 "depends_on_index": [0]},
            ],
        },
        "release": {
            "name": "发布流程",
            "description": "质量检查 -> 部署 -> 监控",
            "steps": [
                {"name": "质量检查", "agent_role": "eval", "priority": "HIGH"},
                {"name": "部署上线", "agent_role": "deploy", "priority": "HIGH",
                 "depends_on_index": [0]},
                {"name": "监控观察", "agent_role": "monitor", "priority": "LOW",
                 "depends_on_index": [1]},
            ],
        },
    }

    def __init__(self):
        self.custom_workflows: Dict[str, Dict] = {}

    def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        """获取工作流模板"""
        if workflow_id in self.BUILTIN_WORKFLOWS:
            wf = self.BUILTIN_WORKFLOWS[workflow_id]
            return {"id": workflow_id, **wf}
        if workflow_id in self.custom_workflows:
            return {"id": workflow_id, **self.custom_workflows[workflow_id]}
        return None

    def list_workflows(self) -> List[Dict]:
        """列出所有工作流"""
        result = []
        for wid, wf in {**self.BUILTIN_WORKFLOWS, **self.custom_workflows}.items():
            result.append({
                "id": wid,
                "name": wf["name"],
                "description": wf["description"],
                "steps_count": len(wf["steps"]),
            })
        return result

    def create_custom_workflow(self, workflow_id: str, name: str, description: str,
                               steps: List[Dict]) -> Dict:
        """创建自定义工作流"""
        self.custom_workflows[workflow_id] = {
            "name": name,
            "description": description,
            "steps": steps,
        }
        return {"success": True, "workflow_id": workflow_id}

    def generate_dynamic_workflow(self, action: str, params: dict = None) -> List[dict]:
        """动态生成工作流任务列表"""
        params = params or {}
        if action == "trigger_training":
            return [
                {"name": "数据预处理", "agent_role": "data", "priority": "MEDIUM"},
                {"name": "模型训练", "agent_role": "train", "priority": "HIGH",
                 "inputs": {"max_generations": params.get("generations", 5)},
                 "depends_on_index": [0]},
                {"name": "模型评估", "agent_role": "eval", "priority": "HIGH",
                 "depends_on_index": [1]},
                {"name": "结果监控", "agent_role": "monitor", "priority": "LOW",
                 "depends_on_index": [0]},
            ]
        elif action == "release_model":
            return [
                {"name": "质量检查", "agent_role": "eval", "priority": "HIGH"},
                {"name": "部署上线", "agent_role": "deploy", "priority": "HIGH",
                 "inputs": {"compression": 4}, "depends_on_index": [0]},
                {"name": "流量切换", "agent_role": "deploy", "priority": "HIGH",
                 "depends_on_index": [1]},
                {"name": "监控观察", "agent_role": "monitor", "priority": "LOW",
                 "depends_on_index": [0]},
            ]
        elif action == "scale_up":
            return [
                {"name": "扩容评估", "agent_role": "orchestrator", "priority": "HIGH"},
                {"name": "资源申请", "agent_role": "data", "priority": "MEDIUM",
                 "depends_on_index": [0]},
                {"name": "实例部署", "agent_role": "train", "priority": "HIGH",
                 "depends_on_index": [0]},
                {"name": "负载均衡", "agent_role": "monitor", "priority": "LOW",
                 "depends_on_index": [0]},
            ]
        else:
            return [
                {"name": "默认处理", "agent_role": "orchestrator", "priority": "MEDIUM"},
                {"name": "结果汇总", "agent_role": "monitor", "priority": "LOW"},
            ]


# ============================================================
# DATA ENGINE [数据引擎-模型工厂]
# ============================================================

class LingyuanDataEngine:
    """
    灵元数据引擎
    负责模型资产的注册、演化追踪与知识蒸馏

    核心功能:
    - 模型注册与元数据管理
    - 训练数据血缘追踪 (Data Lineage)
    - 知识蒸馏与模型压缩
    - 模型间互补性分析
    - 资产估值与定价引擎
    """

    def __init__(self):
        self.model_data = ModelDataSystem()
        self.data_generator = DataGenerator()
        self.bootstrap = BootstrappingEngine()
        self.distiller = KnowledgeDistiller()
        self.evaluator = AutoEvaluator()
        self.spatial = SpatialModelCollaboration(self.model_data)
        self.prompt_mgr = PromptTemplateManager()
        self.workflow_mgr = WorkflowTemplateEngine()

        print(f'[数据引擎] 已初始化 | 资产数: {len(self.model_data.assets)} | 数据集: {len(self.data_generator.datasets)}')

    # ==================== 模型工厂方法 ====================

    def register_model(self, name: str, hidden_dim: int, num_layers: int,
                       num_heads: int, capabilities: List[str],
                       generation: int = 1, parent_id: str = "",
                       token_cost: int = 0) -> dict:
        """注册新模型资产"""
        total_params = hidden_dim * num_layers * num_heads * 64  # 估算参数量
        architecture = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "total_params": total_params,
            "layers": [
                {"name": f"layer_{i}", "type": "transformer", "params": total_params // num_layers}
                for i in range(num_layers)
            ],
        }

        asset = self.model_data.register_model(
            name=name, architecture=architecture, capabilities=capabilities,
            generation=generation, parent_asset_id=parent_id, token_cost=token_cost,
        )
        print(f'[模型工厂] 已注册: {asset.asset_id} | {name} | 参数量: {total_params:,} | 代际: {generation}')
        return {"success": True, "asset_id": asset.asset_id, "name": name, "generation": generation}

    def get_model(self, asset_id: str) -> dict:
        """获取模型详情"""
        asset = self.model_data.get_model(asset_id)
        if not asset:
            return {"error": "模型不存在"}
        return {
            "asset_id": asset.asset_id,
            "name": asset.name,
            "generation": asset.generation,
            "capabilities": asset.capabilities,
            "vector_essence": asset.vector_essence,
            "evaluation": asset.evaluation,
            "training_history": asset.training_history,
            "bloodline": self.model_data.get_model_bloodline(asset_id),
        }

    def list_models(self, **kwargs) -> list:
        """列出模型"""
        return self.model_data.list_models(**kwargs)

    def find_complementary(self, asset_id: str) -> list:
        """寻找互补模型"""
        return self.model_data.find_complementary_models(asset_id)

    def set_model_price(self, asset_id: str, price: float) -> bool:
        """设置模型价格"""
        return self.model_data.set_price(asset_id, price)

    # ==================== 数据生成与增强 ====================

    def generate_dataset(self, source_model_name: str, generation: int,
                         task_type: str = "qa", count: int = 50,
                         quality_threshold: float = 0.6,
                         modality: str = "text") -> dict:
        """生成数据集

        Args:
            modality: 模态类型 text/audio/image/video/multimodal
        """
        dataset = self.data_generator.create_dataset(
            name=f"gen{generation}_{task_type}_dataset",
            source_model=source_model_name,
            target_model=f"gen{generation + 1}_model",
            generation=generation,
            task_type=task_type,
            count=count,
            quality_threshold=quality_threshold,
            modality=modality,
        )
        print(f'[数据生成] 数据集: {dataset.dataset_id} | 模态: {modality} | 生成{count} - 保留{dataset.size}条 质量{dataset.avg_quality}')
        return {
            "success": True,
            "dataset_id": dataset.dataset_id,
            "generation": generation,
            "modality": modality,
            "original_count": count,
            "final_count": dataset.size,
            "avg_quality": dataset.avg_quality,
        }

    def list_datasets(self) -> list:
        return self.data_generator.list_datasets()

    def get_data_stats(self) -> dict:
        return self.data_generator.get_generation_stats()

    # ==================== 自举训练 ====================

    def run_bootstrap(self, initial_model_id: str, initial_accuracy: float = 0.65,
                      initial_loss: float = 2.0, max_generations: int = 5,
                      tokens_per_gen: int = 100) -> dict:
        """自举训练"""
        summary = self.bootstrap.run_bootstrap_loop(
            initial_model_id=initial_model_id,
            initial_accuracy=initial_accuracy,
            initial_loss=initial_loss,
            max_generations=max_generations,
            tokens_per_generation=tokens_per_gen,
        )

        # 注册最终模型
        if summary['final_model'] and summary['final_model'] != initial_model_id:
            self.register_model(
                name=f"GEN_{summary['final_model']}",
                hidden_dim=4096, num_layers=32, num_heads=32,
                capabilities=["chat", "qa", "reasoning"],
                generation=summary['generations_completed'] + 1,
                parent_id=initial_model_id,
                token_cost=summary['total_tokens_consumed'],
            )

        return summary

    def get_evolution_tree(self) -> list:
        """获取进化树"""
        return self.bootstrap.get_evolution_tree()

    def get_safety_summary(self) -> dict:
        """安全摘要"""
        return self.bootstrap.safety.get_summary()

    def get_optimizer_summary(self) -> dict:
        """优化器摘要"""
        return self.bootstrap.optimizer.get_summary()

    # ==================== 知识蒸馏 ====================

    def distill_model(self, teacher_model_id: str, teacher_params: int,
                      teacher_accuracy: float, compression: int = 4,
                      temperature: float = 4.0) -> dict:
        """蒸馏模型: 大模型->小模型"""
        result = self.distiller.distill(
            teacher_model_id=teacher_model_id,
            teacher_params=teacher_params,
            teacher_accuracy=teacher_accuracy,
            target_compression=compression,
            temperature=temperature,
        )
        if result['success']:
            print(f"[蒸馏] 压缩 {result['compression_ratio']} | 精度损失: {result['accuracy_loss']} | 加速: {result['speed_improvement']}")
        return result

    def list_distillations(self) -> list:
        """列出蒸馏记录"""
        return self.distiller.list_distillations()

    # ==================== 评估 ====================

    def evaluate_model(self, model_id: str, model_name: str, generation: int,
                       expected_accuracy: float = None,
                       modality: str = "text") -> dict:
        """评估模型

        Args:
            modality: 模态类型 text/audio/image/video/multimodal
        """
        report = self.evaluator.evaluate(
            model_id=model_id, model_name=model_name, generation=generation,
            expected_accuracy=expected_accuracy, modality=modality,
        )

        # 更新模型评估数据
        self.model_data.update_evaluation(model_id, asdict(report))

        status = "通过" if report.passed else "未通过"
        print(f"[评估] {status} | 模态: {modality} | 得分: {report.overall_score} | 准确率: {report.accuracy} | 安全: {report.safety} | 幻觉: {report.hallucination_rate}")
        if not report.passed:
            for issue in report.issues:
                print(f"  - {issue}")
            print(f"  建议: {report.recommendation}")

        return {
            "eval_id": report.eval_id,
            "passed": report.passed,
            "overall_score": report.overall_score,
            "accuracy": report.accuracy,
            "fluency": report.fluency,
            "coherence": report.coherence,
            "reasoning": report.reasoning,
            "safety": report.safety,
            "hallucination_rate": report.hallucination_rate,
            "bias_score": report.bias_score,
            "issues": report.issues,
            "recommendation": report.recommendation,
            "modality": modality,
            "modality_metrics": report.modality_metrics,
        }

    def get_evaluation_report(self, eval_id: str) -> dict:
        return self.evaluator.get_report(eval_id) or {"error": "报告不存在"}

    def get_latest_eval(self, model_id: str) -> dict:
        return self.evaluator.get_latest_report(model_id) or {"error": "无评估记录"}

    # ==================== 空间协同 ====================

    def align_models(self, asset_ids: List[str]) -> dict:
        """多模型空间对齐"""
        return self.spatial.align_models(asset_ids)

    def fuse_cross_modal(self, asset_ids: List[str]) -> dict:
        """跨模态融合"""
        return self.spatial.fuse_cross_modal(asset_ids)

    # ==================== 仪表盘 ====================

    def dashboard(self) -> dict:
        """数据引擎仪表盘"""
        return {
            "model_assets": self.model_data.get_data_summary(),
            "data_generation": self.get_data_stats(),
            "bootstrap_evolution": self.get_evolution_tree(),
            "safety_status": self.get_safety_summary(),
            "optimizer": self.get_optimizer_summary(),
            "distillations": len(self.distiller.records),
            "evaluations": len(self.evaluator.reports),
            "latest_evals": [self.evaluator.get_latest_report(r.model_id) for r in list(self.evaluator.reports.values())[-5:]],
        }


# ============================================================
# AGENT_ORCHESTRATOR: 多Agent协作编排
# ============================================================

class AgentState(Enum):
    IDLE = "idle"           # 空闲
    ASSIGNED = "assigned"   # 已分配任务
    RUNNING = "running"     # 运行中
    WAITING = "waiting"     # 等待依赖
    COMPLETED = "completed" # 已完成
    FAILED = "failed"       # 失败
    DEGRADED = "degraded"   # 降级状态


class TaskPriority(Enum):
    CRITICAL = 0    # 紧急/关键
    HIGH = 1        # 高
    MEDIUM = 2      # 中等
    LOW = 3         # 低/经济


class MessageType(Enum):
    TASK_ASSIGN = "task_assign"
    TASK_RESULT = "task_result"
    HANDOFF = "handoff"         # 任务交接
    ALERT = "alert"             # 告警
    STATUS_UPDATE = "status_update"
    ESCALATION = "escalation"   # 上报给Orchestrator


@dataclass
class AgentTask:
    """Agent任务"""
    task_id: str
    name: str
    agent_role: str             # 对应Agent角色
    priority: TaskPriority
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    status: str = "pending"     # pending/running/done/failed
    assigned_agent: str = ""    # 被分配的Agent
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)  # 依赖的task_id

    def is_ready(self, completed_tasks: set) -> bool:
        """检查依赖是否满足"""
        return all(dep in completed_tasks for dep in self.dependencies)


@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    msg_type: MessageType
    content: dict
    timestamp: float = field(default_factory=time.time)


@dataclass
class Agent:
    agent_id: str
    role: str                   # orchestrator/data/train/eval/deploy/monitor
    state: AgentState = AgentState.IDLE
    current_task: Optional[AgentTask] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_uptime: float = 0.0
    capabilities: List[str] = field(default_factory=list)
    message_queue: deque = field(default_factory=lambda: deque(maxlen=100))

    def assign(self, task: AgentTask):
        self.current_task = task
        self.state = AgentState.ASSIGNED
        task.assigned_agent = self.agent_id
        task.status = "assigned"

    def start(self):
        self.state = AgentState.RUNNING
        if self.current_task:
            self.current_task.status = "running"
            self.current_task.started_at = time.time()

    def complete(self, outputs: dict):
        if self.current_task:
            self.current_task.outputs = outputs
            self.current_task.status = "done"
            self.current_task.completed_at = time.time()
        self.tasks_completed += 1
        self.state = AgentState.IDLE
        self.current_task = None

    def fail(self, reason: str):
        if self.current_task:
            self.current_task.status = "failed"
            self.current_task.retry_count += 1
        self.tasks_failed += 1
        self.state = AgentState.FAILED
        print(f"[{self.role}]Agent 失败: {reason}")
        self.current_task = None

    def degrade(self, reason: str):
        """降级：从活跃状态转为降级状态"""
        self.state = AgentState.DEGRADED
        print(f"[{self.role}]Agent 降级: {reason}")


class AgentOrchestrator:
    """Agent编排器"""

    def __init__(self):
        """
        初始化Agent团队:
        1. Orchestrator(指挥):负责任务分配和异常处理
        2. 数据处理Agent
        3. 训练Agent
        4. 评估Agent
        5. 部署Agent
        6. 监控Agent
        """
        self.agents: Dict[str, Agent] = {}
        self.task_queue: deque = deque()
        self.completed_tasks: Dict[str, AgentTask] = {}
        self.message_board: List[AgentMessage] = []
        self.alerts: List[dict] = []
        self.workflow_history: List[dict] = []
        self._lock = threading.Lock()

        # 初始化Agent团队
        self._init_agent_team()

    def _init_agent_team(self):
        """初始化Agent团队"""
        roles = [
            ("orchestrator", ["plan", "assign", "escalate", "abort"]),
            ("data", ["generate", "filter", "deduplicate", "package"]),
            ("train", ["bootstrap", "optimize", "rollback"]),
            ("eval", ["evaluate", "safety_check", "hallucination", "bias"]),
            ("deploy", ["distill", "compress", "release"]),
            ("monitor", ["watch", "alert", "report"]),
        ]
        for role, caps in roles:
            agent_id = f"agent_{role}_{len(self.agents)}"
            agent = Agent(agent_id=agent_id, role=role, capabilities=caps)
            self.agents[agent_id] = agent
        print(f"Agent团队 初始化完成: {len(self.agents)}个Agent已加载")
        for aid, agent in self.agents.items():
            print(f"  - {aid}: {agent.role}Agent({agent.capabilities})")

    def get_agent_by_role(self, role: str) -> Optional[Agent]:
        """根据角色获取Agent"""
        for agent in self.agents.values():
            if agent.role == role and agent.state != AgentState.FAILED:
                return agent
        return None

    def submit_task(self, name: str, agent_role: str, priority,
                    inputs: dict = None, dependencies: List[str] = None) -> str:
        """向队列提交一个新任务"""
        if isinstance(priority, str):
            priority = TaskPriority[priority.upper()]
        task_id = f"task_{int(time.time() * 1000)}_{random.randint(100, 999)}"
        task = AgentTask(
            task_id=task_id, name=name, agent_role=agent_role,
            priority=priority, inputs=inputs or {},
            dependencies=dependencies or [],
        )
        self.task_queue.append(task)
        return task_id

    def _pick_next_task(self) -> Optional[AgentTask]:
        """从队列中选择最高优先级 + 依赖已满足的任务"""
        if not self.task_queue:
            return None

        sorted_tasks = sorted(self.task_queue, key=lambda t: t.priority.value)
        completed_ids = set(self.completed_tasks.keys())

        for task in sorted_tasks:
            if task.is_ready(completed_ids):
                self.task_queue.remove(task)
                return task
        return None

    def execute_task(self, task: AgentTask) -> dict:
        """为指定任务分配合适的Agent并执行"""
        agent = self.get_agent_by_role(task.agent_role)
        if not agent:
            orch = self.get_agent_by_role("orchestrator")
            if orch:
                orch.degrade(f"缺少{task.agent_role}Agent，无法调度")
            return {"success": False, "error": f"[{task.agent_role}]Agent缺失", "degraded": True}

        agent.assign(task)
        agent.start()

        print(f"  [{agent.role}]Agent 执行: {task.name} (优先级:{task.priority.name})")

        # 模拟执行
        time.sleep(0.05)
        outputs = self._simulate_work(task, agent)

        # 模拟失败
        fail_rate = 0.08
        if random.random() < fail_rate and task.retry_count < task.max_retries:
            task.retry_count += 1
            print(f"  [{agent.role}]Agent 任务失败: {task.name}, 重试({task.retry_count}/{task.max_retries})")
            task.status = "pending"
            # 重置Agent状态以便接受新任务
            agent.state = AgentState.IDLE
            agent.current_task = None
            agent.tasks_failed += 1
            self.task_queue.append(task)
            return {"success": False, "error": "模拟失败", "will_retry": True}

        agent.complete(outputs)
        self.completed_tasks[task.task_id] = task
        return {"success": True, "outputs": outputs, "task_id": task.task_id}

    def _simulate_work(self, task: AgentTask, agent: Agent) -> dict:
        """根据Agent角色模拟不同的工作输出"""
        role = agent.role
        if role == "data":
            return {
                "dataset_id": f"ds_{task.task_id}",
                "samples": random.randint(10, 100),
                "quality": round(random.uniform(0.6, 0.9), 3),
            }
        elif role == "train":
            return {
                "model_id": f"model_{task.task_id}",
                "accuracy": round(random.uniform(0.7, 0.85), 4),
                "loss": round(random.uniform(0.2, 1.5), 4),
                "generations": task.inputs.get("max_generations", 5),
            }
        elif role == "eval":
            acc = round(random.uniform(0.75, 0.90), 4)
            passed = acc >= 0.75
            return {
                "eval_id": f"eval_{task.task_id}",
                "passed": passed,
                "accuracy": acc,
                "bleu": round(random.uniform(0.5, 0.90), 4),
                "hallucination": round(random.uniform(0.03, 0.12), 4),
            }
        elif role == "deploy":
            return {
                "student_id": f"student_{task.task_id}",
                "compression": task.inputs.get("compression", 4),
                "speed_gain": round(random.uniform(3.0, 8.0), 2),
            }
        elif role == "monitor":
            return {
                "report_id": f"rpt_{task.task_id}",
                "health": "green",
                "alerts": 0,
            }
        elif role == "orchestrator":
            return {"plan": "已生成", "subtasks": task.inputs.get("subtask_count", 0)}
        return {"done": True}

    def send_message(self, from_id: str, to_id: str, msg_type: MessageType, content: dict):
        """Agent间发送消息(广播或单播)"""
        msg = AgentMessage(from_agent=from_id, to_agent=to_id, msg_type=msg_type, content=content)
        self.message_board.append(msg)
        if msg_type == MessageType.ALERT:
            self.alerts.append({"msg": asdict(msg), "timestamp": time.time()})
            print(f"  [ALERT] [{from_id}] -> {to_id}: {content.get('text', '')}")

    def run_workflow(self, workflow_name: str, task_sequence: List[dict]) -> dict:
        """运行工作流(编排Agent协作)"""
        print(f"\nAgent工作流: 开始执行 [{workflow_name}]")
        start_time = time.time()

        # 提交所有任务
        task_ids = []
        for i, step in enumerate(task_sequence):
            deps = [task_ids[j] for j in step.get("depends_on_index", [])]
            priority = step.get("priority", "MEDIUM")
            if isinstance(priority, str):
                priority = TaskPriority[priority.upper()]
            tid = self.submit_task(
                name=step["name"], agent_role=step["agent_role"],
                priority=priority,
                inputs=step.get("inputs", {}),
                dependencies=deps,
            )
            task_ids.append(tid)

        # 执行工作流
        results = {}
        retry_count = 0
        max_global_retries = 20

        while self.task_queue and retry_count < max_global_retries:
            task = self._pick_next_task()
            if not task:
                retry_count += 1
                time.sleep(0.1)
                continue

            result = self.execute_task(task)
            results[task.task_id] = result

            if not result.get("success") and not result.get("will_retry"):
                orch = self.get_agent_by_role("orchestrator")
                if orch:
                    self.send_message(
                        "system", orch.agent_id, MessageType.ESCALATION,
                        {"text": f"任务{task.name}最终失败", "task_id": task.task_id},
                    )
                print(f"  [最终失败] {task.name}")
                task.status = "skipped"
                self.completed_tasks[task.task_id] = task

        elapsed = round(time.time() - start_time, 2)
        total = len(task_ids)
        done = sum(1 for r in results.values() if r.get("success"))
        failed = total - done

        summary = {
            "workflow": workflow_name,
            "total_tasks": total,
            "succeeded": done,
            "failed": failed,
            "elapsed": elapsed,
            "agent_stats": {aid: {"completed": a.tasks_completed, "failed": a.tasks_failed}
                            for aid, a in self.agents.items()},
            "alerts": len(self.alerts),
        }
        self.workflow_history.append(summary)
        print(f"[Agent团队] 工作流: {done}/{total}, 耗时{elapsed}s")
        return summary

    def get_team_status(self) -> dict:
        """获取团队状态"""
        return {
            "total_agents": len(self.agents),
            "agents": [
                {
                    "id": a.agent_id, "role": a.role, "state": a.state.value,
                    "completed": a.tasks_completed, "failed": a.tasks_failed,
                    "current": a.current_task.name if a.current_task else None,
                }
                for a in self.agents.values()
            ],
            "queue_size": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "total_alerts": len(self.alerts),
        }

    def get_workflow_history(self) -> list:
        return self.workflow_history


# ============================================================
# GITHUB PIPELINE (GitHub流水线)
# ============================================================

class TriggerType(Enum):
    PUSH = "push"
    PR_MERGED = "pr_merged"
    TAG_RELEASE = "tag_release"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    MANUAL = "manual"


class PipelineStage(Enum):
    FETCH_CODE = "fetch_code"
    BUILD_ENV = "build_env"
    GENERATE_DATA = "generate_data"
    BOOTSTRAP_TRAIN = "bootstrap_train"
    EVALUATE = "evaluate"
    DISTILL = "distill"
    RELEASE = "release"


class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEGRADED = "degraded"   # 部分成功，需要人工审核


@dataclass
class GitEvent:
    event_id: str
    repo: str              # e.g. "lingyuan/core-model"
    branch: str            # e.g. "main"
    trigger: TriggerType
    payload: dict          # commit info / PR info / tag info
    timestamp: float = field(default_factory=time.time)


@dataclass
class StageResult:
    stage: PipelineStage
    status: str            # success/failed/skipped/degraded
    agent_role: str
    duration: float = 0.0
    outputs: dict = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class PipelineRun:
    run_id: str
    trigger_event: GitEvent
    stages: List[StageResult] = field(default_factory=list)
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: float = 0.0
    completed_at: float = 0.0
    total_duration: float = 0.0

    @property
    def elapsed(self) -> float:
        if self.completed_at:
            return round(self.completed_at - self.started_at, 2)
        return round(time.time() - self.started_at, 2)


class GitHubTriggerPipeline:
    """GitHub触发流水线

    模拟GitHub - 代码推送 -> CI DAG -> 多智能体协作 -> 模型训练
    """

    # 触发规则: 触发类型 -> 执行阶段
    TRIGGER_RULES = {
        TriggerType.PUSH: [
            PipelineStage.FETCH_CODE, PipelineStage.BUILD_ENV,
            PipelineStage.GENERATE_DATA, PipelineStage.BOOTSTRAP_TRAIN,
        ],
        TriggerType.PR_MERGED: [
            PipelineStage.FETCH_CODE, PipelineStage.BUILD_ENV,
            PipelineStage.EVALUATE,
        ],
        TriggerType.TAG_RELEASE: [
            PipelineStage.FETCH_CODE, PipelineStage.BUILD_ENV,
            PipelineStage.EVALUATE, PipelineStage.DISTILL, PipelineStage.RELEASE,
        ],
        TriggerType.SCHEDULE: [
            PipelineStage.FETCH_CODE, PipelineStage.BUILD_ENV,
            PipelineStage.GENERATE_DATA, PipelineStage.BOOTSTRAP_TRAIN,
            PipelineStage.EVALUATE, PipelineStage.DISTILL, PipelineStage.RELEASE,
        ],
        TriggerType.WEBHOOK: [
            PipelineStage.GENERATE_DATA, PipelineStage.BOOTSTRAP_TRAIN, PipelineStage.EVALUATE,
        ],
        TriggerType.MANUAL: [
            PipelineStage.GENERATE_DATA, PipelineStage.BOOTSTRAP_TRAIN,
            PipelineStage.EVALUATE, PipelineStage.DISTILL,
        ],
    }

    # 阶段 -> Agent名称映射
    STAGE_AGENT = {
        PipelineStage.FETCH_CODE: "orchestrator",
        PipelineStage.BUILD_ENV: "orchestrator",
        PipelineStage.GENERATE_DATA: "data",
        PipelineStage.BOOTSTRAP_TRAIN: "train",
        PipelineStage.EVALUATE: "eval",
        PipelineStage.DISTILL: "deploy",
        PipelineStage.RELEASE: "deploy",
    }

    def __init__(self):
        self._runs: Dict[str, PipelineRun] = {}
        self._event_queue: List[GitEvent] = []
        self.trigger_rules: Dict[str, Any] = {
            "branch_watch": ["main", "dev", "release/*"],
            "tag_pattern": r"^v\d+\.\d+\.\d+$",   # v1.0.0 格式
            "auto_cancel_on_new_push": True,
            "max_concurrent_runs": 3,
        }
        self._completed_runs = 0
        self._failed_runs = 0
        self._lock = threading.RLock()
        print(f"[INIT] 触发器规则: {self.trigger_rules['branch_watch']}")
        print(f"       TAG正则: {self.trigger_rules['tag_pattern']}")

    def receive_event(self, event_type: TriggerType, repo: str, branch: str,
                      payload: dict = None) -> str:
        """接收Git事件并触发流水线"""
        event = GitEvent(
            event_id=f"evt_{int(time.time() * 1000)}_{random.randint(100, 999)}",
            repo=repo, branch=branch, trigger=event_type,
            payload=payload or {},
        )

        # 验证触发条件
        if not self._validate_trigger(event):
            print(f"[GitWebhook] 触发条件不满足, 跳过: {event.event_id}")
            return ""

        # 自动取消旧运行
        if self.trigger_rules["auto_cancel_on_new_push"] and event_type == TriggerType.PUSH:
            self._cancel_old_runs(repo, branch)

        self._event_queue.append(event)
        run_id = self._start_pipeline(event)
        return run_id

    def _validate_trigger(self, event: GitEvent) -> bool:
        """验证触发规则"""
        # 分支匹配
        branch_ok = any(
            event.branch == pat or event.branch.startswith(pat.replace("*", ""))
            for pat in self.trigger_rules["branch_watch"]
        )

        # Tag匹配
        if event.trigger == TriggerType.TAG_RELEASE:
            tag_name = event.payload.get("tag", "")
            if not re.match(self.trigger_rules["tag_pattern"], tag_name):
                return False

        return branch_ok

    def _cancel_old_runs(self, repo: str, branch: str):
        """取消同一repo/branch的旧运行"""
        for run in self._runs.values():
            if run.trigger_event.repo == repo and run.trigger_event.branch == branch:
                if run.status == PipelineStatus.RUNNING:
                    run.status = PipelineStatus.CANCELLED
                    print(f"  [GitWebhook] 取消旧运行: {run.run_id}")

    def _start_pipeline(self, event: GitEvent) -> str:
        """启动流水线"""
        run_id = f"run_{event.event_id}"
        stages_to_run = self.TRIGGER_RULES.get(event.trigger, [])

        run = PipelineRun(
            run_id=run_id, trigger_event=event,
            status=PipelineStatus.RUNNING, started_at=time.time(),
        )
        self._runs[run_id] = run

        print(f"\n[GitWebhook] 新流水线: {run_id}")
        print(f"  仓库: {event.repo} | 分支: {event.branch} | 触发: {event.trigger.value}")
        print(f"  阶段: {' -> '.join(s.value for s in stages_to_run)}")

        # 顺序执行各阶段
        any_failed = False
        for stage in stages_to_run:
            result = self._execute_stage(stage, run)
            run.stages.append(result)

            if result.status == "failed":
                any_failed = True
                if stage in (PipelineStage.FETCH_CODE, PipelineStage.BOOTSTRAP_TRAIN):
                    run.status = PipelineStatus.FAILED
                    self._failed_runs += 1
                    run.completed_at = time.time()
                    run.total_duration = run.elapsed
                    return run_id
                # 非关键阶段失败，继续

        run.completed_at = time.time()
        run.total_duration = run.elapsed

        if any_failed:
            run.status = PipelineStatus.DEGRADED
        else:
            run.status = PipelineStatus.SUCCESS
            self._completed_runs += 1

        print(f"[{run_id}] 执行完成 | 状态: {run.status.value} | 耗时: {run.total_duration}s")
        return run_id

    def _execute_stage(self, stage: PipelineStage, run: PipelineRun) -> StageResult:
        """执行单个阶段"""
        agent_role = self.STAGE_AGENT[stage]
        stage_start = time.time()

        print(f"  -> [{stage.value}] [{agent_role}Agent]...")

        # 模拟执行
        time.sleep(0.03)
        duration = round(time.time() - stage_start, 3)

        # 模拟失败
        fail_rate = 0.08
        if random.random() < fail_rate:
            return StageResult(
                stage=stage, status="failed", agent_role=agent_role,
                duration=duration, retry_count=0,
                logs=[f"{stage.value}阶段执行失败"]
            )

        # 生成输出
        outputs = self._gen_stage_output(stage, run)
        return StageResult(
            stage=stage, status="success", agent_role=agent_role,
            duration=duration, outputs=outputs,
            logs=[f"{stage.value}阶段执行成功"]
        )

    def _gen_stage_output(self, stage: PipelineStage, run: PipelineRun) -> dict:
        """生成阶段输出"""
        if stage == PipelineStage.FETCH_CODE:
            return {"commit": run.trigger_event.payload.get("commit", "abc123"), "files_changed": random.randint(1, 20)}
        elif stage == PipelineStage.BUILD_ENV:
            return {"env_id": f"env_{run.run_id}", "python": "3.11", "cuda": "12.4"}
        elif stage == PipelineStage.GENERATE_DATA:
            return {"dataset_id": f"ds_{run.run_id}", "samples": random.randint(50, 200), "quality": round(random.uniform(0.65, 0.90), 2)}
        elif stage == PipelineStage.BOOTSTRAP_TRAIN:
            return {"model_id": f"model_{run.run_id}", "accuracy": round(random.uniform(0.72, 0.88), 4), "generations": random.randint(10, 50)}
        elif stage == PipelineStage.EVALUATE:
            acc = round(random.uniform(0.75, 0.92), 4)
            return {"eval_id": f"eval_{run.run_id}", "passed": acc >= 0.75, "accuracy": acc, "safety": round(random.uniform(0.88, 0.98), 4)}
        elif stage == PipelineStage.DISTILL:
            return {"student_id": f"student_{run.run_id}", "compression": "4:1", "speed_gain": round(random.uniform(3.0, 8.0), 2)}
        elif stage == PipelineStage.RELEASE:
            return {"release_id": f"rls_{run.run_id}", "version": run.trigger_event.payload.get("tag", "v0.1.0"), "status": "published"}
        return {}

    @property
    def _total_runs(self):
        return len(self._runs)

    def get_run_status(self, run_id: str) -> dict:
        """获取运行状态"""
        run = self._runs.get(run_id)
        if not run:
            return {"error": "运行不存在"}
        return {
            "run_id": run.run_id,
            "status": run.status.value,
            "trigger": run.trigger_event.trigger.value,
            "repo": run.trigger_event.repo,
            "branch": run.trigger_event.branch,
            "stages": [
                {"stage": s.stage.value, "status": s.status, "agent": s.agent_role, "duration": s.duration}
                for s in run.stages
            ],
            "duration": run.total_duration,
        }

    def get_pipeline_stats(self) -> dict:
        """获取统计信息"""
        total = self._total_runs
        return {
            "total_runs": total,
            "completed_runs": self._completed_runs,
            "failed_runs": self._failed_runs,
            "success_rate": round(self._completed_runs / max(total, 1), 3),
            "active_runs": sum(1 for r in self._runs.values() if r.status == PipelineStatus.RUNNING),
            "queue_size": len(self._event_queue),
        }

    def list_runs(self, limit: int = 10) -> list:
        """列出运行记录"""
        recent = sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)[:limit]
        return [
            {"run_id": r.run_id, "status": r.status.value, "trigger": r.trigger_event.trigger.value,
             "repo": r.trigger_event.repo, "duration": r.total_duration, "stages": len(r.stages)}
            for r in recent
        ]
