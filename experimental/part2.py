

# ============================================================
# LOCAL_STORAGE 本地存储适配器
# ============================================================

class ChannelSystem:
    """渠道管理系统

    配置:
        - 带宽类型: fast(10Gbps) / standard(1Gbps) / economy(100Mbps)
        - 延迟要求: 低延迟(<10ms) / 标准(<50ms) / 宽松(<200ms)
        - 可靠性等级
    """

    def __init__(self, config=CHANNEL_CONFIG):
        self.config = config
        self.usage_file = os.path.join(DATA_DIR, "channel_usage.json")
        self.usage_records: List[Dict] = []
        self._load()

    def _load(self):
        """加载使用记录"""
        if os.path.exists(self.usage_file):
            with open(self.usage_file, 'r', encoding='utf-8') as f:
                self.usage_records = json.load(f).get('records', [])

    def _save(self):
        """保存使用记录"""
        with open(self.usage_file, 'w', encoding='utf-8') as f:
            json.dump({'records': self.usage_records[-2000:]}, f, ensure_ascii=False, indent=2)

    def estimate_transfer_cost(self, data_gb: float, tier: str = None) -> Dict:
        """估算传输成本"""
        tier = tier or self.config["default_tier"]
        tier_config = self.config["tiers"].get(tier, self.config["tiers"]["standard"])
        price_per_gb = tier_config["price_per_gb"]
        cost = data_gb * price_per_gb
        return {
            "tier": tier,
            "data_gb": data_gb,
            "bandwidth": tier_config["bandwidth"],
            "cost": round(cost, 2),
            "estimated_time": self._estimate_time(data_gb, tier_config["bandwidth"]),
        }

    def _estimate_time(self, data_gb: float, bandwidth: str) -> str:
        """估算传输时间"""
        bw_map = {"100Mbps": 0.0125, "1Gbps": 0.125, "5Gbps": 0.625, "10Gbps": 1.25}  # GB/s
        bw_gb_s = bw_map.get(bandwidth, 0.125)
        seconds = data_gb / bw_gb_s
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"

    def record_transfer(self, data_gb: float, tier: str = None, task_id: str = "") -> Dict:
        """记录传输"""
        cost_info = self.estimate_transfer_cost(data_gb, tier)
        record = {
            "record_id": f"transfer_{int(time.time())}",
            "task_id": task_id,
            "data_gb": data_gb,
            "tier": cost_info["tier"],
            "cost": cost_info["cost"],
            "bandwidth": cost_info["bandwidth"],
            "timestamp": datetime.now().isoformat(),
        }
        self.usage_records.append(record)
        self._save()
        return record

    def get_usage_summary(self, days=30) -> Dict:
        """获取使用摘要"""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [r for r in self.usage_records if datetime.fromisoformat(r["timestamp"]) > cutoff]
        total_gb = sum(r["data_gb"] for r in recent)
        total_cost = sum(r["cost"] for r in recent)
        return {
            "period_days": days,
            "total_transfers": len(recent),
            "total_data_gb": round(total_gb, 2),
            "total_cost": round(total_cost, 2),
            "avg_cost_per_gb": round(total_cost / max(total_gb, 0.01), 4),
        }


@dataclass
class StorageItem:
    """存储项"""
    item_id: str
    name: str
    size_gb: float
    tier: str          # hot / cold
    last_access: str
    project: str = ""
    auto_cleanup: bool = True


class StorageSystem:
    """存储系统

    功能：
    - 存储/读取
    - 自动分层（热/冷数据）
    - 生命周期管理
    - 成本优化
    """

    def __init__(self, config=STORAGE_CONFIG):
        self.config = config
        self.storage_file = os.path.join(DATA_DIR, "storage_items.json")
        self.items: List[StorageItem] = []
        self._load()

    def _load(self):
        """加载存储项"""
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.items = [StorageItem(**item) for item in data.get("items", [])]

    def _save(self):
        """保存存储项"""
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump({"items": [asdict(i) for i in self.items]}, f, ensure_ascii=False, indent=2)

    def store(self, name: str, size_gb: float, project: str = "", auto_cleanup: bool = True) -> StorageItem:
        """存储文件"""
        now = datetime.now().isoformat()
        tier = "hot" if size_gb < self.config["hot_tier_threshold_gb"] else "cold"
        item = StorageItem(
            item_id=f"store_{int(time.time())}",
            name=name,
            size_gb=size_gb,
            tier=tier,
            last_access=now,
            project=project,
            auto_cleanup=auto_cleanup,
        )
        self.items.append(item)
        self._save()
        return item

    def access(self, item_id: str) -> bool:
        """访问数据项（自动分层，冷数据变热）"""
        for item in self.items:
            if item.item_id == item_id:
                item.last_access = datetime.now().isoformat()
                if item.tier == "cold":
                    item.tier = "hot"
                    self._save()
                return True
        return False

    def delete(self, item_id: str) -> bool:
        """删除数据项"""
        before = len(self.items)
        self.items = [i for i in self.items if i.item_id != item_id]
        if len(self.items) < before:
            self._save()
            return True
        return False

    def auto_cleanup(self) -> int:
        """自动清理过期数据"""
        cutoff = datetime.now() - timedelta(days=self.config["auto_cleanup_days"])
        before = len(self.items)
        self.items = [
            i for i in self.items
            if not i.auto_cleanup or datetime.fromisoformat(i.last_access) > cutoff
        ]
        cleaned = before - len(self.items)
        if cleaned > 0:
            self._save()
        return cleaned

    def get_storage_summary(self) -> Dict:
        """获取存储使用统计（用于成本估算）"""
        total_gb = sum(i.size_gb for i in self.items)
        hot_gb = sum(i.size_gb for i in self.items if i.tier == "hot")
        cold_gb = sum(i.size_gb for i in self.items if i.tier == "cold")
        monthly_cost = sum(
            i.size_gb * (self.config["price_per_gb_month"] if i.tier == "hot"
            else self.config["price_per_gb_month"] * self.config["cold_tier_discount"])
            for i in self.items
        )
        # 分类统计
        categories = {}
        for item in self.items:
            cat = item.project or "其他"
            if cat not in categories:
                categories[cat] = {"size_gb": 0, "items": 0}
            categories[cat]["size_gb"] += item.size_gb
            categories[cat]["items"] += 1

        return {
            "total_used_gb": round(total_gb, 2),
            "hot_tier_gb": round(hot_gb, 2),
            "cold_tier_gb": round(cold_gb, 2),
            "total_items": len(self.items),
            "monthly_cost": round(monthly_cost, 2),
            "categories": {k: {"size_gb": round(v["size_gb"], 2), "items": v["items"]} for k, v in categories.items()},
            "price_per_gb_hot": self.config["price_per_gb_month"],
            "price_per_gb_cold": round(self.config["price_per_gb_month"] * self.config["cold_tier_discount"], 4),
        }


# ============================================================
# DATA_TRADING [数据交易协议] —— 架构图补全模块
# ============================================================

class DataTradingProtocol:
    """数据交易协议

    架构图对应: DATA层 - 数据交易协议
    - 数据确权机制
    - 隐私计算支持
    - 价值分配智能合约
    """

    def __init__(self):
        self.trades: List[Dict] = []
        self.ownership: Dict[str, str] = {}  # asset_id -> owner
        self.trade_file = os.path.join(DATA_DIR, "data_trades.json")
        self._load()

    def _load(self):
        if os.path.exists(self.trade_file):
            with open(self.trade_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.trades = data.get("trades", [])
            self.ownership = data.get("ownership", {})

    def _save(self):
        with open(self.trade_file, 'w', encoding='utf-8') as f:
            json.dump({"trades": self.trades[-1000:], "ownership": self.ownership}, f, ensure_ascii=False, indent=2)

    def register_ownership(self, asset_id: str, owner: str) -> bool:
        """数据确权"""
        self.ownership[asset_id] = owner
        self._save()
        return True

    def verify_ownership(self, asset_id: str, claimed_owner: str) -> bool:
        """验证所有权"""
        return self.ownership.get(asset_id) == claimed_owner

    def execute_trade(self, asset_id: str, seller: str, buyer: str, price: float) -> Dict:
        """执行数据交易"""
        if not self.verify_ownership(asset_id, seller):
            return {"success": False, "error": "所有权验证失败"}

        trade = {
            "trade_id": f"dtrade_{int(time.time())}",
            "asset_id": asset_id,
            "seller": seller,
            "buyer": buyer,
            "price": price,
            "timestamp": datetime.now().isoformat(),
        }
        self.trades.append(trade)
        self.ownership[asset_id] = buyer  # 转移所有权
        self._save()
        return {"success": True, "trade": trade}

    def get_trade_history(self, asset_id: str = None) -> List[Dict]:
        if asset_id:
            return [t for t in self.trades if t["asset_id"] == asset_id]
        return self.trades

    def get_summary(self) -> Dict:
        return {
            "total_trades": len(self.trades),
            "total_volume": round(sum(t["price"] for t in self.trades), 2),
            "registered_assets": len(self.ownership),
        }


# ============================================================
# ENVIRONMENT_MANAGER [环境管理器]
# ============================================================

@dataclass
class EnvironmentTemplate:
    """环境模板"""
    template_id: str
    name: str
    description: str
    framework: str      # pytorch / tensorflow / jax
    cuda_version: str
    python_version: str
    preinstalled_libs: List[str]  # 预装库列表
    estimated_setup_time: int  # 估计启动时间(秒)
    gpu_required: bool


@dataclass
class VirtualEnvironment:
    """虚拟环境实例"""
    env_id: str
    template_id: str
    template_name: str
    status: str         # creating / ready / running / stopped / destroyed
    created_at: str
    last_active: str
    gpu_type: str = ""
    snapshot_id: str = ""


# 预定义模板
BUILTIN_TEMPLATES = [
    EnvironmentTemplate(
        template_id="tpl_pytorch_train",
        name="PyTorch训练环境",
        description="PyTorch 2.4 + CUDA 12.4 + DeepSpeed + Megatron-LM, 高性能训练环境",
        framework="pytorch",
        cuda_version="12.4",
        python_version="3.11",
        preinstalled_libs=["torch", "deepspeed", "megatron-lm", "transformers", "datasets", "accelerate", "wandb"],
        estimated_setup_time=25,
        gpu_required=True,
    ),
    EnvironmentTemplate(
        template_id="tpl_pytorch_infer",
        name="PyTorch推理",
        description="标准PyTorch + vLLM + TensorRT, 高性能推理环境",
        framework="pytorch",
        cuda_version="12.4",
        python_version="3.11",
        preinstalled_libs=["torch", "vllm", "tensorrt", "fastapi", "uvicorn"],
        estimated_setup_time=20,
        gpu_required=True,
    ),
    EnvironmentTemplate(
        template_id="tpl_data_prep",
        name="数据预处理",
        description="数据预处理专用环境, 支持数据清洗/转换/标注",
        framework="none",
        cuda_version="",
        python_version="3.11",
        preinstalled_libs=["pandas", "numpy", "scikit-learn", "datasets", "tokenizers", "jieba"],
        estimated_setup_time=15,
        gpu_required=False,
    ),
    EnvironmentTemplate(
        template_id="tpl_lingyuan_full",
        name="灵元全功能",
        description="灵元模型完整开发环境, 支持训练/推理/数据处理全流程",
        framework="pytorch",
        cuda_version="12.4",
        python_version="3.11",
        preinstalled_libs=["torch", "deepspeed", "transformers", "lingyuan-core", "edge-tts", "ffmpeg"],
        estimated_setup_time=30,
        gpu_required=True,
    ),
]


class EnvironmentManager:
    """环境管理器

    职责:
    - 管理环境模板
    - 创建/销毁虚拟环境
    - 环境状态监控与自动清理
    - 环境依赖解析与冲突检测
    """

    def __init__(self):
        self.templates: Dict[str, EnvironmentTemplate] = {t.template_id: t for t in BUILTIN_TEMPLATES}
        self.environments: Dict[str, VirtualEnvironment] = {}
        self.env_file = os.path.join(DATA_DIR, "environments.json")
        self._load()

    def _load(self):
        if os.path.exists(self.env_file):
            with open(self.env_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for d in data.get('environments', []):
                self.environments[d['env_id']] = VirtualEnvironment(**d)

    def _save(self):
        with open(self.env_file, 'w', encoding='utf-8') as f:
            json.dump({'environments': [asdict(e) for e in self.environments.values()]}, f, ensure_ascii=False, indent=2)

    def list_templates(self) -> List[Dict]:
        """列出所有可用模板"""
        return [
            {
                'template_id': t.template_id,
                'name': t.name,
                'description': t.description,
                'framework': t.framework,
                'cuda_version': t.cuda_version,
                'python_version': t.python_version,
                'preinstalled': t.preinstalled_libs,
                'estimated_setup_time': t.estimated_setup_time,
                'gpu_required': t.gpu_required,
            }
            for t in self.templates.values()
        ]

    def create_environment(self, template_id: str, gpu_type: str = 'auto') -> Dict:
        """创建新环境"""
        if template_id not in self.templates:
            return {'success': False, 'error': f'模板不存在: {template_id}'}

        template = self.templates[template_id]
        env_id = f"env_{int(time.time())}_{template_id}"
        now = datetime.now().isoformat()

        env = VirtualEnvironment(
            env_id=env_id,
            template_id=template_id,
            template_name=template.name,
            status='creating',
            created_at=now,
            last_active=now,
            gpu_type=gpu_type,
        )
        self.environments[env_id] = env
        self._save()

        # 模拟异步初始化
        env.status = 'ready'
        env.last_active = datetime.now().isoformat()
        self._save()

        return {
            'success': True,
            'env_id': env_id,
            'template_name': template.name,
            'status': 'ready',
            'setup_time': template.estimated_setup_time,
            'framework': template.framework,
            'cuda_version': template.cuda_version,
            'preinstalled': template.preinstalled_libs,
            'gpu_type': gpu_type,
        }

    def start_environment(self, env_id: str) -> bool:
        """启动环境"""
        if env_id not in self.environments:
            return False
        self.environments[env_id].status = "running"
        self.environments[env_id].last_active = datetime.now().isoformat()
        self._save()
        return True

    def stop_environment(self, env_id: str) -> bool:
        """停止环境"""
        if env_id not in self.environments:
            return False
        self.environments[env_id].status = "stopped"
        self._save()
        return True

    def destroy_environment(self, env_id: str) -> bool:
        """销毁环境"""
        if env_id not in self.environments:
            return False
        self.environments[env_id].status = "destroyed"
        self._save()
        return True

    def snapshot(self, env_id: str) -> Dict:
        """创建快照"""
        if env_id not in self.environments:
            return {"success": False, "error": "环境不存在"}
        snapshot_id = f"snap_{int(time.time())}"
        self.environments[env_id].snapshot_id = snapshot_id
        self._save()
        return {"success": True, "snapshot_id": snapshot_id, "env_id": env_id}

    def get_environment_status(self, env_id: str) -> Optional[Dict]:
        """获取环境状态"""
        if env_id not in self.environments:
            return None
        env = self.environments[env_id]
        template = self.templates.get(env.template_id, None)
        return {
            "env_id": env.env_id,
            "template_name": env.template_name,
            "status": env.status,
            "created_at": env.created_at,
            "last_active": env.last_active,
            "gpu_type": env.gpu_type,
            "snapshot_id": env.snapshot_id,
            "framework": template.framework if template else "unknown",
        }

    def list_environments(self, status_filter=None) -> List[Dict]:
        """列出所有环境"""
        result = []
        for env in self.environments.values():
            if status_filter and env.status != status_filter:
                continue
            result.append(self.get_environment_status(env.env_id))
        return result


# ============================================================
# DISASTER RECOVERY (灾难恢复机制)
# ============================================================

@dataclass
class Checkpoint:
    """检查点数据类"""
    checkpoint_id: str
    task_id: str
    step: int           # 训练步数
    loss: float         # 损失值
    accuracy: float     # 准确率
    model_state_path: str  # 模型状态路径
    timestamp: str
    size_gb: float


class DisasterRecovery:
    """灾难恢复管理器"""

    def __init__(self):
        """初始化检查点管理"""
        self.config = DISASTER_CONFIG
        self.checkpoints: Dict[str, List[Checkpoint]] = {}  # task_id -> checkpoints
        self.recovery_log: List[Dict] = []
        self.cp_file = os.path.join(DATA_DIR, "checkpoints.json")
        self._load()

    def _load(self):
        """加载检查点数据"""
        if os.path.exists(self.cp_file):
            with open(self.cp_file, "r", encoding='utf-8') as f:
                data = json.load(f)
                for tid, cps in data.get("checkpoints", {}).items():
                    self.checkpoints[tid] = [Checkpoint(**cp) for cp in cps]
                self.recovery_log = data.get("recovery_log", [])

    def _save(self):
        data = {
            "checkpoints": {tid: [asdict(cp) for cp in cps] for tid, cps in self.checkpoints.items()},
            "recovery_log": self.recovery_log[-500:],
        }
        with open(self.cp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_checkpoint(self, task_id: str, step: int, loss: float,
                        accuracy: float, model_state_path: str, size_gb: float = 0.5) -> Checkpoint:
        """保存模型checkpoint"""
        cp = Checkpoint(
            checkpoint_id=f"cp_{int(time.time())}_{step}",
            task_id=task_id,
            step=step,
            loss=round(loss, 6),
            accuracy=round(accuracy, 4),
            model_state_path=model_state_path,
            timestamp=datetime.now().isoformat(),
            size_gb=size_gb,
        )
        if task_id not in self.checkpoints:
            self.checkpoints[task_id] = []
        self.checkpoints[task_id].append(cp)
        self._save()
        return cp

    def get_latest_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
        """获取最新的checkpoint"""
        if task_id not in self.checkpoints or not self.checkpoints[task_id]:
            return None
        return self.checkpoints[task_id][-1]

    def should_checkpoint(self, task_id: str, current_step: int) -> bool:
        """判断是否需要创建checkpoint"""
        latest = self.get_latest_checkpoint(task_id)
        if not latest:
            return current_step >= self.config["checkpoint_interval"]
        return (current_step - latest.step) >= self.config["checkpoint_interval"]

    def resume_from_checkpoint(self, task_id: str) -> Dict:
        """从checkpoint恢复训练"""
        cp = self.get_latest_checkpoint(task_id)
        if not cp:
            return {"success": False, "error": "无可用checkpoint"}

        log_entry = {
            "type": "resume",
            "task_id": task_id,
            "checkpoint_id": cp.checkpoint_id,
            "resumed_step": cp.step,
            "timestamp": datetime.now().isoformat(),
        }
        self.recovery_log.append(log_entry)
        self._save()

        return {
            "success": True,
            "task_id": task_id,
            "resume_step": cp.step,
            "checkpoint_loss": cp.loss,
            "checkpoint_accuracy": cp.accuracy,
            "model_state_path": cp.model_state_path,
        }

    def handle_vendor_failure(self, task_id: str, old_vendor: str, new_vendor: str) -> Dict:
        """处理供应商故障转移"""
        log_entry = {
            "type": "vendor_failover",
            "task_id": task_id,
            "old_vendor": old_vendor,
            "new_vendor": new_vendor,
            "timestamp": datetime.now().isoformat(),
        }
        self.recovery_log.append(log_entry)
        self._save()

        # 尝试从checkpoint恢复
        resume_info = self.resume_from_checkpoint(task_id)
        return {
            "success": True,
            "failover": f"{old_vendor} -> {new_vendor}",
            "resume_info": resume_info,
            "user_notification": f"检测到算力厂商异常，已自动迁移，训练从断点恢复",
        }

    def get_recovery_log(self, task_id: str = None, limit: int = 20) -> List[Dict]:
        """获取恢复日志"""
        logs = self.recovery_log[-limit:]
        if task_id:
            logs = [l for l in logs if l.get("task_id") == task_id]
        return logs

    def get_checkpoint_summary(self, task_id: str) -> Dict:
        """获取checkpoint摘要"""
        cps = self.checkpoints.get(task_id, [])
        if not cps:
            return {"task_id": task_id, "total_checkpoints": 0}
        latest = cps[-1]
        return {
            "task_id": task_id,
            "total_checkpoints": len(cps),
            "latest_step": latest.step,
            "latest_loss": latest.loss,
            "latest_accuracy": latest.accuracy,
            "latest_timestamp": latest.timestamp,
            "total_storage_gb": round(sum(cp.size_gb for cp in cps), 2),
        }


# ============================================================
# INFRA 基础设施层
# ============================================================

class LingyuanInfra:
    """灵元基础设施管理类

    聚合: TokenSystem + EnergySystem + VendorScheduler + ChannelSystem
          + StorageSystem + EnvironmentManager + DisasterRecovery
          + CarbonTradingGateway + DataTradingProtocol
    """

    def __init__(self, user_id: str = "user_001"):
        self.user_id = user_id

        # 初始化各模块
        self.wallet = TokenSystem(user_id)
        self.pricing = TokenPricingEngine()
        self.energy = EnergySystem()
        self.scheduler = VendorScheduler()
        self.channel = ChannelSystem()
        self.storage = StorageSystem()
        self.env_manager = EnvironmentManager()
        self.recovery = DisasterRecovery()
        self.carbon_gateway = CarbonTradingGateway()
        self.data_trading = DataTradingProtocol()

        print(f"[INFRA] 用户 {user_id} 系统初始化完成")
        print(f"  Token余额: {self.wallet.get_balance()} | 单价: {self.pricing.get_current_price()['unit_price']}")

    # ========== Token操作 ==========
    def buy_token(self, amount: int, green_power: bool = False) -> dict:
        """购买Token"""
        result = self.wallet.purchase(amount, green_power)
        if result['success']:
            print(f"  Token购买成功: {amount} Token, 成本 {result['total_cost']}")
        else:
            print(f"  购买失败: {result['error']}")
        return result

    def get_wallet_summary(self) -> dict:
        """获取钱包信息"""
        return self.wallet.get_wallet_summary()

    def transfer_tokens(self, to_user: str, amount: int) -> dict:
        """转账Token"""
        return self.wallet.transfer(to_user, amount)

    def get_token_price(self) -> dict:
        """获取Token价格"""
        return self.pricing.get_current_price()

    def estimate_cost(self, gpu_hours: int, green_power: bool = False) -> dict:
        """预估计算成本"""
        return self.pricing.estimate_cost(gpu_hours, green_power)

    # ========== 能源管理 ==========
    def get_energy_summary(self, days: int = 30) -> dict:
        """获取能源使用摘要"""
        return self.energy.get_energy_summary(days)

    def generate_esg_report(self, days: int = 30) -> dict:
        """生成ESG报告"""
        return self.energy.generate_esg_report(days)

    def is_green_power_hour(self) -> bool:
        """判断是否为绿色电力时段"""
        return self.energy.is_green_power_hour()

    def get_carbon_summary(self) -> dict:
        """获取碳交易摘要"""
        return self.carbon_gateway.get_summary()

    # ========== 任务调度 ==========
    def create_training_task(self, epochs: int, green_power: bool = False) -> dict:
        """提交训练任务 [Token+绿色计算+Checkpoint]"""
        # 1. 检查Token余额
        balance = self.wallet.get_balance()
        if balance < epochs:
            return {"success": False, "error": f"Token不足: 需要{epochs}, 当前{balance}"}

        # 2. 提交任务到调度器
        task = self.scheduler.submit_task(
            task_type="training",
            gpu_hours=epochs,
            data_size_gb=0,
            priority="normal",
            green_power=green_power
        )

        if task.status == "failed":
            return {"success": False, "error": task.result.get("error", "任务提交失败")}

        # 3. 扣除Token
        consume_result = self.wallet.consume(epochs, task.task_id)
        if not consume_result['success']:
            return {"success": False, "error": "Token扣除失败"}

        # 4. 记录能源
        vendor_id = task.assigned_vendor
        energy_record = self.energy.record_consumption(task.task_id, epochs, green_power, vendor_id)

        # 5. 积累碳信用
        if green_power and energy_record.carbon_kg < energy_record.energy_kwh * self.energy.config["carbon_per_kwh"]:
            carbon_saved = energy_record.energy_kwh * self.energy.config["carbon_per_kwh"] * 0.8
            self.carbon_gateway.accumulate_credits(carbon_saved)

        print(f"  任务提交成功: {task.task_id} | 厂商: {vendor_id} | GPU: {epochs}h")
        return {
            "success": True,
            "task_id": task.task_id,
            "assigned_vendor": vendor_id,
            "gpu_hours": epochs,
            "token_cost": epochs,
            "remaining_balance": self.wallet.get_balance()
        }

    def get_vendor_comparison(self) -> list:
        """获取厂商对比"""
        return self.scheduler.get_vendor_comparison()

    def get_task_status(self, task_id: str) -> dict:
        """获取任务状态"""
        return self.scheduler.get_task_status(task_id)

    def complete_task(self, task_id: str, result: dict = None) -> bool:
        """完成任务"""
        return self.scheduler.complete_task(task_id, result)

    def failover_task(self, task_id: str) -> dict:
        """故障转移"""
        task = self.scheduler.failover(task_id)
        if task:
            self.recovery.handle_vendor_failure(task_id, "", task.assigned_vendor)
            return {"success": True, "task_id": task_id, "new_vendor": task.assigned_vendor}
        return {"success": False, "error": "无法转移"}

    # ========== 存储管理 ==========
    def estimate_transfer_cost(self, data_gb: float, tier: str = "standard") -> dict:
        """估算传输成本"""
        return self.channel.estimate_transfer_cost(data_gb, tier)

    def get_channel_summary(self, days: int = 30) -> dict:
        """获取通道摘要"""
        return self.channel.get_usage_summary(days)

    def store_data(self, name: str, size_gb: float, project: str = "") -> dict:
        """存储数据"""
        item = self.storage.store(name, size_gb, project)
        return {"success": True, "item_id": item.item_id, "name": name, "size_gb": size_gb}

    def get_storage_summary(self) -> dict:
        """获取存储摘要"""
        return self.storage.get_storage_summary()

    def cleanup_storage(self) -> int:
        """清理存储"""
        return self.storage.auto_cleanup()

    # ========== 环境管理 ==========
    def list_env_templates(self) -> list:
        """列出环境模板"""
        return self.env_manager.list_templates()

    def create_environment(self, template_id: str, gpu_type: str = "auto") -> dict:
        """创建环境"""
        result = self.env_manager.create_environment(template_id, gpu_type)
        if result.get("success"):
            print(f"  环境已创建: {result['env_id']} | 模板: {result['template_name']}")
        return result

    def get_env_status(self, env_id: str) -> dict:
        """获取环境状态"""
        return self.env_manager.get_environment_status(env_id)

    def list_environments(self, status_filter=None) -> list:
        """列出环境"""
        return self.env_manager.list_environments(status_filter)

    # ========== 检查点与恢复 ==========
    def save_checkpoint(self, task_id: str, step: int, loss: float,
                        accuracy: float, model_path: str) -> dict:
        """保存checkpoint"""
        cp = self.recovery.save_checkpoint(task_id, step, loss, accuracy, model_path)
        return {"success": True, "checkpoint_id": cp.checkpoint_id, "step": step}

    def resume_training(self, task_id: str) -> dict:
        """从checkpoint恢复"""
        return self.recovery.resume_from_checkpoint(task_id)

    def get_recovery_log(self, task_id: str = None) -> list:
        """获取恢复日志"""
        return self.recovery.get_recovery_log(task_id)

    # ========== 仪表盘 ==========
    def dashboard(self) -> dict:
        """系统仪表盘"""
        return {
            "user_id": self.user_id,
            "token_wallet": self.get_wallet_summary(),
            "energy": self.get_energy_summary(),
            "carbon": self.get_carbon_summary(),
            "vendors": self.get_vendor_comparison(),
            "channel": self.get_channel_summary(),
            "storage": self.get_storage_summary(),
            "environments": self.list_environments(),
            "active_tasks": self.scheduler.get_all_tasks("running"),
            "recovery_log": self.get_recovery_log(),
            "green_power_available": self.is_green_power_hour(),
        }
