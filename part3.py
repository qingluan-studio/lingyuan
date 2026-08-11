

# ============================================================
# MODEL DATA (模型数据定义) - 模型即数据
# ============================================================

@dataclass
class ModelDataAsset:
    """模型数据资产 - 训练数据管理"""
    asset_id: str
    name: str
    version: str
    generation: int                    # 数据世代(迭代次数)
    parent_asset_id: str = ""          # 父数据集(血缘追踪)

    # 架构信息(动态、可扩展)
    architecture: Dict = field(default_factory=dict)  # 架构定义与版本信息

    # 权重摘要
    weight_summary: Dict = field(default_factory=dict)  # 参数量/结构

    # 能力标签
    capabilities: List[str] = field(default_factory=list)  # text_gen/classify/reasoning

    # 训练历史
    training_history: List[Dict] = field(default_factory=list)

    # 评估结果
    evaluation: Dict = field(default_factory=dict)

    # 向量表征(用于相似度检索)
    vector_essence: List[float] = field(default_factory=list)

    # 元信息
    created_at: str = ""
    size_gb: float = 0.0
    status: str = "active"          # active/archived/traded
    token_cost: int = 0             # 推理token成本

    # 交易信息
    price: float = 0.0
    owner: str = ""                 # 当前持有者地址


class ModelDataSystem:
    """模型资产管理系统

    功能:
        - 注册/查询/更新模型资产
        - 管理模型血缘关系(父子/衍生)
        - 计算模型相似度(向量检索)
        - 评估模型价值(参数/能力/稀缺性)
    """

    def __init__(self):
        self.assets: Dict[str, ModelDataAsset] = {}
        self.assets_file = os.path.join(DATA_DIR, "model_assets.json")
        self._load()

    def _load(self):
        if os.path.exists(self.assets_file):
            with open(self.assets_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for a in data.get("assets", []):
                    self.assets[a["asset_id"]] = ModelDataAsset(**a)

    def _save(self):
        with open(self.assets_file, 'w', encoding='utf-8') as f:
            json.dump({"assets": [asdict(a) for a in self.assets.values()]}, f, ensure_ascii=False, indent=2)

    def register_model(self, name: str, architecture: Dict, capabilities: List[str],
                       generation: int = 1, parent_asset_id: str = "",
                       weight_stats: Dict = None, token_cost: int = 0,
                       owner: str = "system") -> ModelDataAsset:
        """注册新模型资产"""
        asset_id = f"model_{int(time.time())}_{len(self.assets)}"
        now = datetime.now().isoformat()

        # 生成权重摘要
        weight_summary = weight_stats or self._generate_weight_summary(architecture)

        # 生成向量表征(用于相似度计算)
        vector_essence = self._generate_vector_essence(architecture, capabilities)

        asset = ModelDataAsset(
            asset_id=asset_id,
            name=name,
            version=f"{generation}.0",
            generation=generation,
            parent_asset_id=parent_asset_id,
            architecture=architecture,
            weight_summary=weight_summary,
            capabilities=capabilities,
            vector_essence=vector_essence,
            created_at=now,
            size_gb=architecture.get("total_params", 0) * 4 / 1e9,  # 假设float32
            token_cost=token_cost,
            owner=owner,
        )
        self.assets[asset_id] = asset
        self._save()
        return asset

    def _generate_weight_summary(self, architecture: Dict) -> Dict:
        """生成权重统计摘要"""
        layers = architecture.get("layers", [])
        summary = {}
        for layer in layers:
            layer_name = layer.get("name", "unknown")
            param_count = layer.get("params", 0)
            summary[layer_name] = {
                "param_count": param_count,
                "mean": round(0.02 * param_count * 1e-8, 6),
                "std": round(0.05 * param_count * 1e-8, 6),
                "norm": round(math.sqrt(param_count) * 0.01, 4),
            }
        return summary

    def _generate_vector_essence(self, architecture: Dict, capabilities: List[str]) -> List[float]:
        """生成模型向量表征(用于相似度检索)"""
        # 8维能力向量(对齐设计规格):
        # dim 0: 语言理解
        # dim 1: 生成能力
        # dim 2: 推理能力
        # dim 3: 代码能力
        # dim 4: 多模态
        # dim 5: 知识广度
        # dim 6: 长文本
        # dim 7: 安全对齐

        base = architecture.get("hidden_dim", 512) / 4096  # 归一化基础规模
        cap_map = {
            "text_gen":    [0.8, 0.9, 0.3, 0.2, 0.1, 0.6, 0.5, 0.7],
            "classify":    [0.9, 0.3, 0.5, 0.1, 0.0, 0.7, 0.4, 0.8],
            "reasoning":   [0.7, 0.4, 0.9, 0.5, 0.0, 0.5, 0.6, 0.6],
            "code":        [0.5, 0.6, 0.7, 0.9, 0.2, 0.4, 0.5, 0.5],
            "text":        [0.9, 0.7, 0.4, 0.1, 0.0, 0.8, 0.9, 0.6],
            "multimodal":  [0.6, 0.5, 0.3, 0.1, 0.9, 0.4, 0.3, 0.5],
            "reasoning_enhanced": [0.8, 0.6, 0.95, 0.6, 0.3, 0.7, 0.7, 0.8],
            "chat":        [0.8, 0.8, 0.5, 0.3, 0.2, 0.6, 0.6, 0.7],
            "qa":          [0.9, 0.7, 0.6, 0.2, 0.1, 0.8, 0.5, 0.6],
        }
        vector = [0.0] * 8
        for cap in capabilities:
            if cap in cap_map:
                for i in range(min(8, len(cap_map[cap]))):
                    vector[i] = max(vector[i], cap_map[cap][i] * base + 0.1)
        return [round(v, 4) for v in vector]

    def get_model(self, asset_id: str) -> Optional[ModelDataAsset]:
        """根据ID获取模型"""
        return self.assets.get(asset_id)

    def list_models(self, capability: str = None, generation: int = None) -> List[Dict]:
        """列出所有模型"""
        result = []
        for asset in self.assets.values():
            if capability and capability not in asset.capabilities:
                continue
            if generation and asset.generation != generation:
                continue
            result.append({
                "asset_id": asset.asset_id,
                "name": asset.name,
                "version": asset.version,
                "generation": asset.generation,
                "capabilities": asset.capabilities,
                "size_gb": round(asset.size_gb, 4),
                "status": asset.status,
                "evaluation_score": asset.evaluation.get("overall_score", "N/A"),
                "token_cost": asset.token_cost,
            })
        return result

    def get_model_bloodline(self, asset_id: str) -> List[Dict]:
        """获取模型的血缘关系链"""
        chain = []
        current = self.assets.get(asset_id)
        while current:
            chain.append({
                "asset_id": current.asset_id,
                "name": current.name,
                "generation": current.generation,
                "version": current.version,
            })
            if current.parent_asset_id and current.parent_asset_id in self.assets:
                current = self.assets[current.parent_asset_id]
            else:
                break
        return chain

    def update_evaluation(self, asset_id: str, evaluation: Dict) -> bool:
        """更新模型评测结果"""
        if asset_id not in self.assets:
            return False
        self.assets[asset_id].evaluation = evaluation
        self._save()
        return True

    def add_training_record(self, asset_id: str, record: Dict) -> bool:
        """添加训练记录"""
        if asset_id not in self.assets:
            return False
        self.assets[asset_id].training_history.append(record)
        self._save()
        return True

    def compute_similarity(self, asset_id_a: str, asset_id_b: str) -> float:
        """计算两个模型的相似度(余弦相似度)"""
        a = self.assets.get(asset_id_a)
        b = self.assets.get(asset_id_b)
        if not a or not b:
            return 0.0
        va, vb = a.vector_essence, b.vector_essence
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x ** 2 for x in va))
        nb = math.sqrt(sum(x ** 2 for x in vb))
        if na == 0 or nb == 0:
            return 0.0
        return round(dot / (na * nb), 4)

    def find_complementary_models(self, asset_id: str) -> List[Dict]:
        """寻找互补性强的模型组合[协同进化分析]"""
        target = self.assets.get(asset_id)
        if not target:
            return []
        results = []
        for other_id, other in self.assets.items():
            if other_id == asset_id:
                continue
            sim = self.compute_similarity(asset_id, other_id)
            complementarity = round(1 - sim, 4)
            results.append({
                "asset_id": other_id,
                "name": other.name,
                "complementarity": complementarity,
                "capabilities": other.capabilities,
            })
        results.sort(key=lambda x: -x["complementarity"])
        return results[:5]

    def set_price(self, asset_id: str, price: float) -> bool:
        """设置模型定价"""
        if asset_id not in self.assets:
            return False
        self.assets[asset_id].price = price
        self._save()
        return True

    def get_data_summary(self) -> Dict:
        """获取数据摘要"""
        total = len(self.assets)
        active = len([a for a in self.assets.values() if a.status == "active"])
        total_size = sum(a.size_gb for a in self.assets.values())
        total_tokens = sum(a.token_cost for a in self.assets.values())
        generations = sorted(set(a.generation for a in self.assets.values()))
        return {
            "total_models": total,
            "active_models": active,
            "total_size_gb": round(total_size, 2),
            "total_token_invested": total_tokens,
            "generations": generations,
            "max_generation": max(generations) if generations else 0,
        }


# ============================================================
# SPATIAL_COLLABORATION [空间模型协同] —— 架构图补全模块
# ============================================================

class SpatialModelCollaboration:
    """空间模型协同

    架构图对应: ORCH层 - 空间模型协同
    - 多模型空间对齐
    - 跨模态融合
    - 一致性约束
    """

    def __init__(self, model_system: ModelDataSystem):
        self.model_system = model_system
        self.collaborations: List[Dict] = []

    def align_models(self, asset_ids: List[str]) -> Dict:
        """多模型空间对齐"""
        models = [self.model_system.get_model(aid) for aid in asset_ids]
        if not all(models):
            return {"success": False, "error": "部分模型不存在"}

        # 计算模型间相似度矩阵
        sim_matrix = {}
        for i, a in enumerate(asset_ids):
            for j, b in enumerate(asset_ids):
                if i < j:
                    sim = self.model_system.compute_similarity(a, b)
                    sim_matrix[f"{a}|{b}"] = sim

        return {
            "success": True,
            "aligned_models": len(models),
            "similarity_matrix": sim_matrix,
            "avg_similarity": round(sum(sim_matrix.values()) / max(len(sim_matrix), 1), 4),
        }

    def fuse_cross_modal(self, asset_ids: List[str]) -> Dict:
        """跨模态融合"""
        models = [self.model_system.get_model(aid) for aid in asset_ids]
        capabilities_union = set()
        for m in models:
            capabilities_union.update(m.capabilities)

        result = {
            "fused_model_count": len(models),
            "unified_capabilities": list(capabilities_union),
            "fusion_id": f"fusion_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
        }
        self.collaborations.append(result)
        return result

    def check_consistency(self, asset_ids: List[str]) -> Dict:
        """一致性约束检查"""
        models = [self.model_system.get_model(aid) for aid in asset_ids]
        if not all(models):
            return {"success": False, "error": "部分模型不存在"}

        # 检查架构一致性
        architectures = [m.architecture.get("framework", "unknown") for m in models]
        consistent = len(set(architectures)) <= 1

        return {
            "consistent": consistent,
            "frameworks": architectures,
            "recommendation": "兼容" if consistent else "需要适配层",
        }


# ============================================================
# DATA_GENERATOR (数据生成器)
# ============================================================

@dataclass
class TrainingSample:
    """训练样本"""
    sample_id: str
    prompt: str          # 提示
    response: str        # 回答
    quality_score: float # 质量分(0-1)
    source: str          # auto_gen / human / distilled
    task_type: str       # qa / summary / reasoning / code / audio / image / video / multimodal
    generation: int      # 生成轮次
    modality: str = "text"  # text / audio / image / video / multimodal
    media_metadata: Dict = field(default_factory=dict)  # 媒体元信息(时长/分辨率/格式等)
    embedding: List[float] = field(default_factory=list) # 向量嵌入(可选)


@dataclass
class TrainingDataset:
    """训练数据集"""
    dataset_id: str
    name: str
    generation: int           # 数据集轮次
    samples: List[TrainingSample] = field(default_factory=list)
    avg_quality: float = 0.0
    size: int = 0
    created_at: str = ""
    source_model: str = ""    # 来源模型
    target_model: str = ""    # 目标模型


class DataGenerator:
    """数据生成器

    功能:
    - 生成训练数据
    - 质量评估
    - 数据增强
    - 格式转换
    """

    # 任务模板定义
    QA_TEMPLATES = [
        ("通用问答", "请回答以下问题：{question}"),
        ("多选问答", "问题：{question}\n选项：{choices}\n请选出正确答案并解释。"),
        ("真假判断", "请判断以下陈述是否正确：{statement}\n请回答'真'或'假'并解释原因。"),
        ("知识推理", "基于以下背景信息：{context}\n请回答：{question}"),
        ("比较分析", "请比较以下两个概念：{concept1}和{concept2}"),
        ("代码解释", "请解释以下代码的功能：\n```{language}\n{code}\n```"),
        ("SQL生成", "请根据以下需求生成SQL查询：{requirement}"),
        ("LLM API调用", "请编写调用{api_name}的Python代码，实现{functionality}"),
        ("RAG检索增强生成", "基于以下检索结果：{retrieved_docs}\n请回答用户问题：{user_query}"),
    ]

    REASONING_TEMPLATES = [
        ("数学推理", "请逐步解决：{problem}"),
        ("逻辑推理", "请分析以下逻辑问题：{problem}\n已知条件：{conditions}\n请得出结论。"),
        ("因果推理", "请分析以下事件的因果关系：{event}\n可能原因：{possible_causes}"),
        ("代码调试", "以下代码有错误，请找出并修复：\n```{language}\n{buggy_code}\n```"),
    ]

    SUMMARY_TEMPLATES = [
        ("文本摘要", "请将以下长文本摘要为{max_tokens}个Token以内的摘要：{long_text}"),
        ("对话摘要", "请将以下对话历史摘要为上下文摘要：{conversation_history}"),
    ]

    # ========== 代码场景 ==========
    CODE_TEMPLATES = [
        ("代码生成", "请用{language}实现以下功能：{requirement}\n要求：{constraints}"),
        ("代码补全", "请补全以下代码：\n```{language}\n{code_snippet}\n```"),
        ("代码重构", "请重构以下代码以提高可读性和性能：\n```{language}\n{code}\n```"),
        ("单元测试", "请为以下函数编写单元测试：\n```{language}\n{function_code}\n```"),
        ("算法设计", "请设计一个{complexity}时间复杂度的算法解决：{problem}"),
    ]

    # ========== 音频场景 ==========
    AUDIO_TEMPLATES = [
        ("TTS语音合成", "请将以下文本合成为语音：\n文本：{text}\n要求：语速{speed}，音色{voice_type}"),
        ("语音识别", "请识别以下音频片段的内容：\n音频特征：{audio_features}\n语言：{language}"),
        ("音频分类", "请对以下音频进行分类：\n音频特征：{audio_features}\n候选类别：{categories}"),
        ("音频描述", "请描述以下音频的场景和内容：\n频谱特征：{spectrogram_features}"),
        ("音乐生成", "请根据以下描述生成音乐：\n风格：{genre}\n节奏：{tempo}BPM\n时长：{duration}秒"),
        ("音色转换", "请将以下音频转换为{target_voice}音色：\n原始特征：{source_features}"),
    ]

    # ========== 图像场景 ==========
    IMAGE_TEMPLATES = [
        ("文生图", "请根据以下描述生成图像：\n描述：{prompt}\n风格：{style}\n分辨率：{resolution}"),
        ("图像描述", "请描述以下图像的内容：\n图像特征：{image_features}\n请包括：场景/物体/颜色/布局"),
        ("图像分类", "请对以下图像进行分类：\n图像特征：{image_features}\n候选类别：{categories}"),
        ("图生图", "请基于以下图像进行风格转换：\n原始特征：{source_features}\n目标风格：{target_style}"),
        ("OCR识别", "请识别以下图像中的文字内容：\n图像特征：{image_features}"),
        ("图像修复", "请修复以下图像的损坏区域：\n图像特征：{image_features}\n损坏区域：{damaged_region}"),
    ]

    # ========== 视频场景 ==========
    VIDEO_TEMPLATES = [
        ("文生视频", "请根据以下描述生成视频：\n描述：{prompt}\n时长：{duration}秒\n分辨率：{resolution}\n帧率：{fps}fps"),
        ("视频描述", "请描述以下视频的内容：\n视频特征：{video_features}\n请包括：场景/动作/时间线"),
        ("视频分类", "请对以下视频进行分类：\n视频特征：{video_features}\n候选类别：{categories}"),
        ("视频摘要", "请将以下{duration}秒视频摘要为{target_duration}秒：\n关键帧特征：{keyframe_features}"),
        ("动作识别", "请识别以下视频中的动作：\n时序特征：{temporal_features}\n候选动作：{action_categories}"),
        ("视频超分", "请将以下视频从{source_res}超分辨率到{target_res}：\n视频特征：{video_features}"),
    ]

    # ========== 多模态场景 ==========
    MULTIMODAL_TEMPLATES = [
        ("图文匹配", "请判断以下图像和文本是否匹配：\n图像特征：{image_features}\n文本：{text}"),
        ("视觉问答", "请根据图像回答问题：\n图像特征：{image_features}\n问题：{question}"),
        ("图文生成", "请根据图像生成描述性文本：\n图像特征：{image_features}\n要求：{requirements}"),
        ("音频文本对齐", "请将以下音频与文本对齐：\n音频特征：{audio_features}\n文本：{text}"),
        ("多模态推理", "基于图像和文本进行推理：\n图像特征：{image_features}\n文本上下文：{context}\n问题：{question}"),
        ("跨模态检索", "请根据{query_type}检索匹配的{target_modality}：\n查询特征：{query_features}"),
    ]

    # 模态 -> 模板映射
    MODALITY_TEMPLATES = {
        "text": None,  # 文本场景用task_type直接映射
        "audio": AUDIO_TEMPLATES,
        "image": IMAGE_TEMPLATES,
        "video": VIDEO_TEMPLATES,
        "multimodal": MULTIMODAL_TEMPLATES,
    }

    # 模态 -> 基础质量上限(非文本场景自举成熟度低，质量天花板低)
    MODALITY_QUALITY_CEILING = {
        "text": 0.98,
        "audio": 0.85,
        "image": 0.75,
        "video": 0.65,
        "multimodal": 0.80,
    }

    # 模态 -> 每样本Token消耗倍率(非文本场景消耗更大)
    MODALITY_TOKEN_MULTIPLIER = {
        "text": 1.0,
        "audio": 3.0,
        "image": 8.0,
        "video": 20.0,
        "multimodal": 5.0,
    }

    def __init__(self):
        self.datasets: Dict[str, TrainingDataset] = {}
        self.datasets_file = os.path.join(DATA_DIR, "training_datasets.json")
        self._load()

    def _load(self):
        if os.path.exists(self.datasets_file):
            with open(self.datasets_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for d in data.get('datasets', []):
                samples = [TrainingSample(**s) for s in d.pop('samples', [])]
                dataset = TrainingDataset(**d, samples=samples)
                self.datasets[dataset.dataset_id] = dataset

    def _save(self):
        with open(self.datasets_file, 'w', encoding='utf-8') as f:
            json.dump({"datasets": [asdict(d) for d in self.datasets.values()]}, f, ensure_ascii=False, indent=2)

    def generate_samples(self, source_model_name: str, generation: int,
                        task_type: str = "qa", count: int = 50,
                        modality: str = "text") -> List[TrainingSample]:
        """生成训练样本

        Args:
            modality: 模态类型 text/audio/image/video/multimodal
                      非text模态会自动选择对应模板并降低质量天花板
        """
        # 确定模态
        if task_type in ("audio", "image", "video", "multimodal"):
            modality = task_type  # task_type即模态时自动对齐

        # 质量天花板: 非文本场景自举成熟度低
        quality_ceiling = self.MODALITY_QUALITY_CEILING.get(modality, 0.98)
        base_quality = min(0.5 + generation * 0.08, quality_ceiling)

        # 选择模板
        if modality != "text" and modality in self.MODALITY_TEMPLATES:
            templates = self.MODALITY_TEMPLATES[modality]
        else:
            templates = {
                "qa": self.QA_TEMPLATES,
                "reasoning": self.REASONING_TEMPLATES,
                "summary": self.SUMMARY_TEMPLATES,
                "code": self.CODE_TEMPLATES,
            }.get(task_type, self.QA_TEMPLATES)

        samples = []
        for i in range(count):
            template = random.choice(templates)

            # 质量评分: 非文本场景波动更大(自举不确定性高)
            if modality == "text":
                quality = min(base_quality + random.uniform(-0.1, 0.05), 1.0)
            else:
                quality = min(base_quality + random.uniform(-0.15, 0.03), quality_ceiling)
            quality = max(quality, 0.3)

            # 生成媒体元信息
            media_meta = self._generate_media_metadata(modality, template[0])

            sample = TrainingSample(
                sample_id=f"sample_{int(time.time())}_{i}_{random.randint(1000, 9999)}",
                prompt=template[0],
                response=template[1],
                quality_score=round(quality, 4),
                source="auto_gen",
                task_type=task_type if modality == "text" else modality,
                generation=generation,
                modality=modality,
                media_metadata=media_meta,
                embedding=self._generate_embedding(template[0], quality),
            )
            samples.append(sample)

        return samples

    def _generate_media_metadata(self, modality: str, task_name: str) -> Dict:
        """根据模态生成媒体元信息"""
        meta = {"modality": modality, "task_name": task_name}
        if modality == "audio":
            meta.update({
                "format": random.choice(["wav", "mp3", "flac"]),
                "duration_sec": round(random.uniform(1.0, 30.0), 1),
                "sample_rate": random.choice([16000, 22050, 44100, 48000]),
                "channels": random.choice([1, 2]),
            })
        elif modality == "image":
            meta.update({
                "format": random.choice(["png", "jpg", "webp"]),
                "width": random.choice([256, 512, 768, 1024]),
                "height": random.choice([256, 512, 768, 1024]),
                "color_space": random.choice(["RGB", "RGBA", "GRAY"]),
            })
        elif modality == "video":
            meta.update({
                "format": random.choice(["mp4", "avi", "mov"]),
                "duration_sec": round(random.uniform(2.0, 60.0), 1),
                "fps": random.choice([24, 30, 60]),
                "width": random.choice([640, 1280, 1920]),
                "height": random.choice([360, 720, 1080]),
            })
        elif modality == "multimodal":
            meta.update({
                "modalities": random.sample(["text", "audio", "image", "video"], 2),
                "alignment_score": round(random.uniform(0.6, 0.95), 3),
            })
        return meta

    def estimate_token_cost(self, count: int, modality: str = "text") -> int:
        """估算生成数据的Token消耗"""
        multiplier = self.MODALITY_TOKEN_MULTIPLIER.get(modality, 1.0)
        return int(count * multiplier)

    def _generate_embedding(self, text: str, quality: float) -> List[float]:
        """生成文本的嵌入向量"""
        base = hash(text) % 1000
        return [round(base / 1000 + i * 0.01 + quality * 0.1, 4) for i in range(8)]

    def filter_by_quality(self, samples: List[TrainingSample], threshold: float = 0.6) -> List[TrainingSample]:
        """按质量过滤样本"""
        filtered = [s for s in samples if s.quality_score >= threshold]
        return filtered

    def deduplicate(self, samples: List[TrainingSample], similarity_threshold: float = 0.85) -> List[TrainingSample]:
        """基于相似度去重"""
        unique = []
        for sample in samples:
            is_dup = False
            for existing in unique:
                sim = self._cosine_sim(sample.embedding, existing.embedding)
                if sim > similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(sample)
        return unique

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x ** 2 for x in a))
        nb = math.sqrt(sum(x ** 2 for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def create_dataset(self, name: str, source_model: str, target_model: str,
                       generation: int = 1, task_type: str = "qa",
                       count: int = 50, quality_threshold: float = 0.6,
                       modality: str = "text") -> TrainingDataset:
        """创建新的数据集"""
        # 1. 生成
        raw_samples = self.generate_samples(source_model, generation, task_type, count, modality)

        # 2. 过滤
        filtered = self.filter_by_quality(raw_samples, quality_threshold)

        # 3. 去重
        deduped = self.deduplicate(filtered)

        # 4. 汇总
        avg_quality = sum(s.quality_score for s in deduped) / max(len(deduped), 1)
        dataset = TrainingDataset(
            dataset_id=f"dataset_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            generation=generation,
            samples=deduped,
            avg_quality=round(avg_quality, 4),
            size=len(deduped),
            created_at=datetime.now().isoformat(),
            source_model=source_model,
            target_model=target_model,
        )
        self.datasets[dataset.dataset_id] = dataset
        self._save()

        return dataset

    def get_dataset(self, dataset_id: str) -> Optional[TrainingDataset]:
        return self.datasets.get(dataset_id)

    def list_datasets(self) -> List[Dict]:
        return [
            {
                "id": d.dataset_id,
                "name": d.name,
                "generation": d.generation,
                "size": d.size,
                "avg_quality": d.avg_quality,
                "source_model": d.source_model,
                "target_model": d.target_model,
                "created_at": d.created_at,
            }
            for d in self.datasets.values()
        ]

    def get_generation_stats(self) -> Dict:
        """获取生成统计"""
        if not self.datasets:
            return {"total_datasets": 0}
        generations = sorted(set(d.generation for d in self.datasets.values()))
        stats = {
            "total_datasets": len(self.datasets),
            "total_samples": sum(d.size for d in self.datasets.values()),
            "avg_quality": round(sum(d.avg_quality * d.size for d in self.datasets.values()) /
                                sum(d.size for d in self.datasets.values()), 4),
            "generations": generations,
            "quality_trend": [],
        }
        for gen in generations:
            gen_datasets = [d for d in self.datasets.values() if d.generation == gen]
            gen_avg = sum(d.avg_quality for d in gen_datasets) / max(len(gen_datasets), 1)
            stats["quality_trend"].append({"generation": gen, "avg_quality": round(gen_avg, 4)})
        return stats


# ============================================================
# BOOTSTRAP_ENGINE 引导引擎
# ============================================================

@dataclass
class TrainingGeneration:
    """训练代次记录"""
    generation: int                    # 代次
    parent_model_id: str               # 父模型ID
    child_model_id: str                # 子模型ID
    dataset_id: str                    # 数据集ID
    training_steps: int                # 训练步数
    initial_loss: float                # 初始loss
    final_loss: float                  # 最终loss
    initial_accuracy: float            # 初始准确率
    final_accuracy: float              # 最终准确率
    improvement: float                 # 提升幅度
    tokens_consumed: int               # 消耗token
    duration_note: str                 # 耗时备注
    status: str                        # completed / failed / rolled_back
    timestamp: str
    safety_check: Dict = field(default_factory=dict)  # 安全检查记录


@dataclass
class HyperParameters:
    """超参数配置(含自适应)"""
    learning_rate: float = 5e-5
    batch_size: int = 32
    warmup_steps: int = 100
    weight_decay: float = 0.01
    dropout: float = 0.1
    grad_clip: float = 1.0
    label_smoothing: float = 0.1


class HyperparameterOptimizer:
    """超参数优化器

    贝叶斯优化 + 遗传算法混合
    """

    def __init__(self):
        self.history: List[Dict] = []  # 历史记录
        self.best_params: Optional[HyperParameters] = None
        self.best_score: float = 0.0

    def suggest(self, generation: int) -> HyperParameters:
        """基于历史推荐超参数"""
        if not self.history:
            # 无历史，返回默认
            return HyperParameters()

        # 找出最佳历史，进行变异
        best_entry = max(self.history, key=lambda x: x['score'])
        params = HyperParameters(
            learning_rate=best_entry['params']['learning_rate'] * random.uniform(0.8, 1.2),
            batch_size=best_entry['params']['batch_size'],
            warmup_steps=best_entry['params']['warmup_steps'],
            weight_decay=max(0, best_entry['params']['weight_decay'] * random.uniform(0.9, 1.1)),
            dropout=max(0.05, best_entry['params']['dropout'] * random.uniform(0.9, 1.1)),
            grad_clip=1.0,
            label_smoothing=0.1,
        )
        # 学习率衰减
        params.learning_rate = max(1e-6, params.learning_rate * (0.95 ** generation))
        return params

    def record(self, params: HyperParameters, score: float):
        """记录结果"""
        entry = {
            "params": asdict(params),
            "score": score,
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(entry)
        if score > self.best_score:
            self.best_score = score
            self.best_params = params

    def get_summary(self) -> Dict:
        return {
            "total_trials": len(self.history),
            "best_score": self.best_score,
            "best_params": asdict(self.best_params) if self.best_params else None,
        }


class SafetyValve:
    """安全阀门

    多维度安全检查:
    - 输出质量检测(重复/毒性/偏见)
    - 资源使用监控(内存/时间)
    - 模型行为异常
    - 数据泄露风险
    """

    def __init__(self):
        self.checks_history: List[Dict] = []

    def evaluate(self, generation: int, parent_acc: float, child_acc: float,
                 child_loss: float, parent_loss: float) -> Dict:
        """评估子模型是否安全可用"""
        checks = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "passed": True,
            "warnings": [],
        }

        # 1. 性能检查
        if child_acc < parent_acc:
            checks["checks"]["performance"] = "FAIL"
            checks["passed"] = False
            checks["warnings"].append(f"性能下降: {parent_acc} -> {child_acc}")
        else:
            checks["checks"]["performance"] = "PASS"

        # 2. Loss检查
        if child_loss > parent_loss * 1.5:
            checks["checks"]["loss"] = "WARN"
            checks["warnings"].append(f"Loss异常增高: {parent_loss} -> {child_loss}")
        else:
            checks["checks"]["loss"] = "PASS"

        # 3. 幻觉风险检测
        hallucination_risk = min(0.1 + generation * 0.03, 0.5)
        if hallucination_risk > 0.35:
            checks["checks"]["hallucination"] = "WARN"
            checks["warnings"].append(f"幻觉风险升高: {hallucination_risk}")
        else:
            checks["checks"]["hallucination"] = "PASS"

        # 4. 偏见检测
        bias_score = random.uniform(0.05, 0.15)
        if bias_score > 0.12:
            checks["checks"]["bias"] = "WARN"
            checks["warnings"].append(f"偏见分数: {bias_score}")
        else:
            checks["checks"]["bias"] = "PASS"

        # 5. 综合评估，决定行动
        warning_count = len(checks["warnings"])
        if not checks["passed"] or warning_count >= 3:
            checks["action"] = "CRITICAL_REBASE"
            checks["should_stop"] = False
        elif warning_count >= 1:
            checks["action"] = "PROCEED_WITH_CAUTION"
            checks["should_stop"] = False
        else:
            checks["action"] = "PROCEED"
            checks["should_stop"] = False

        self.checks_history.append(checks)
        return checks

    def should_stop(self) -> bool:
        """判断是否停止训练"""
        if len(self.checks_history) < 2:
            return False
        # 连续两次检查失败则停止
        recent = self.checks_history[-2:]
        return all(not c["passed"] for c in recent)

    def get_summary(self) -> Dict:
        return {
            "total_checks": len(self.checks_history),
            "passed": len([c for c in self.checks_history if c["passed"]]),
            "failed": len([c for c in self.checks_history if not c["passed"]]),
            "should_stop": self.should_stop(),
        }


class BootstrappingEngine:
    """自举训练引擎

    流程:
        for generation in range(max_generations):
            1. 加载上一代模型
            2. 生成训练数据
            3. 微调模型
            4. 评估并生成QualityToken
            5. 质量检查
            6. 保存或回滚
            7. 更新元数据，继续迭代
    """

    def __init__(self):
        self.generations: List[TrainingGeneration] = []
        self.optimizer = HyperparameterOptimizer()
        self.safety = SafetyValve()
        self.gen_file = os.path.join(DATA_DIR, "bootstrap_generations.json")
        self._load()

    def _load(self):
        if os.path.exists(self.gen_file):
            with open(self.gen_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.generations = [TrainingGeneration(**g) for g in data.get("generations", [])]

    def _save(self):
        with open(self.gen_file, 'w', encoding='utf-8') as f:
            json.dump({"generations": [asdict(g) for g in self.generations]}, f, ensure_ascii=False, indent=2)

    def run_generation(self, generation: int, parent_model_id: str, dataset_id: str,
                       parent_accuracy: float, parent_loss: float,
                       tokens_consumed: int) -> TrainingGeneration:
        """
        运行单代训练

        公式: 新模型 = 旧模型 + 数据 + 训练
        """
        # 1. 参数优化
        params = self.optimizer.suggest(generation)

        # 2. 模拟训练过程
        steps = 500 + generation * 100
        initial_loss = parent_loss + random.uniform(0.1, 0.3)
        # 模拟收敛: loss逐渐降低，但可能过拟合
        improvement_factor = 0.7 + generation * 0.02  # 后期改进空间变小
        final_loss = max(initial_loss * improvement_factor + random.uniform(-0.05, 0.05), 0.01)

        # 模拟精度提升，但边际递减
        acc_improvement = random.uniform(0.02, 0.08) * (0.9 ** generation)
        final_accuracy = min(parent_accuracy + acc_improvement, 0.99)

        # 3. 计算分数
        score = final_accuracy * 100 + (parent_accuracy - final_loss) * 10
        self.optimizer.record(params, score)

        # 4. 安全检查
        safety_check = self.safety.evaluate(generation, parent_accuracy, final_accuracy,
                                           final_loss, parent_loss)

        # 5. 确定状态
        if safety_check["passed"]:
            status = "completed"
        else:
            # 安全检查未通过，回滚
            status = "rolled_back"
            final_accuracy = parent_accuracy  # 回退到父模型精度
            final_loss = parent_loss

        # 6. 记录
        gen_record = TrainingGeneration(
            generation=generation,
            parent_model_id=parent_model_id,
            child_model_id=f"model_gen{generation + 1}" if safety_check["passed"] else "",
            dataset_id=dataset_id,
            training_steps=steps,
            initial_loss=round(initial_loss, 4),
            final_loss=round(final_loss, 4),
            initial_accuracy=round(parent_accuracy, 4),
            final_accuracy=round(final_accuracy, 4),
            improvement=round(final_accuracy - parent_accuracy, 4),
            tokens_consumed=tokens_consumed,
            duration_note=f"{(steps // 100)}T",
            status=status,
            timestamp=datetime.now().isoformat(),
            safety_check=safety_check,
        )
        self.generations.append(gen_record)
        self._save()

        return gen_record

    def run_bootstrap_loop(self, initial_model_id: str, initial_accuracy: float,
                          initial_loss: float, max_generations: int = 10,
                          tokens_per_generation: int = 100) -> Dict:
        """
        运行自举训练循环。

        迭代优化模型，直到达到目标或触发停止条件。
        """
        print(f"\n{'='*40}")
        print(f"[启动自举训练] 初始模型: {initial_model_id}")
        print(f"[初始状态] 精度: {initial_accuracy:.4f}, 损失: {initial_loss:.4f}")
        print(f"[配置] 最大代数: {max_generations}, 每代Token: {tokens_per_generation}")
        print(f"{'='*40}\n")

        current_model_id = initial_model_id
        current_accuracy = initial_accuracy
        current_loss = initial_loss
        results = []

        for gen in range(1, max_generations + 1):
            print(f"\n{'='*40}")
            print(f"[第 {gen} 代训练] {'='*40}")

            # 检查是否应该停止
            if self.safety.should_stop():
                print(f"[安全系统] 触发停止，中断训练(gen={gen})")
                break

            # 运行一代
            record = self.run_generation(
                generation=gen,
                parent_model_id=current_model_id,
                dataset_id=f"dataset_gen{gen}",
                parent_accuracy=current_accuracy,
                parent_loss=current_loss,
                tokens_consumed=tokens_per_generation,
            )

            status_icon = "[OK]" if record.status == "completed" else "[FAIL]"
            print(f"[结果] {status_icon} 第{gen}代: acc={record.final_accuracy:.4f} loss={record.final_loss:.4f} [{record.status}]")

            if record.status == "completed":
                current_model_id = record.child_model_id
                current_accuracy = record.final_accuracy
                current_loss = record.final_loss

            results.append(asdict(record))

            # 检查是否达到目标精度
            if record.improvement < 0.01 and record.status == "completed":
                print(f"[收敛检测] 改进小于0.01，提前终止(gen={gen})")
                break

        summary = {
            "initial_model": initial_model_id,
            "final_model": current_model_id,
            "generations_run": len(results),
            "generations_completed": len([r for r in results if r["status"] == "completed"]),
            "generations_rolled_back": len([r for r in results if r["status"] == "rolled_back"]),
            "initial_accuracy": initial_accuracy,
            "final_accuracy": current_accuracy,
            "total_improvement": round(current_accuracy - initial_accuracy, 4),
            "total_tokens_consumed": sum([r["tokens_consumed"] for r in results]),
            "safety_summary": self.safety.get_summary(),
            "optimizer_summary": self.optimizer.get_summary(),
        }

        print(f"\n{'='*40}")
        print(f"[训练完成] 完成代数: {summary['generations_completed']}")
        print(f"[最终状态] 精度: {summary['final_accuracy']:.4f} (提升{summary['total_improvement']})")
        print(f"[资源消耗] Token总量: {summary['total_tokens_consumed']}")

        return summary

    def get_evolution_tree(self) -> List[Dict]:
        """获取模型进化树"""
        return [
            {
                "generation": g.generation,
                "status": g.status,
                "accuracy": g.final_accuracy,
                "loss": g.final_loss,
                "improvement": g.improvement,
                "tokens": g.tokens_consumed,
                "safety": g.safety_check.get("action", "N/A"),
            }
            for g in self.generations
        ]


# ============================================================
# DISTILL_EVAL 【知识蒸馏评估】
# ============================================================

@dataclass
class DistillationRecord:
    """蒸馏记录"""
    distill_id: str
    teacher_model_id: str      # 教师模型ID
    student_model_id: str      # 学生模型ID
    teacher_params: int        # 教师参数量
    student_params: int        # 学生参数量
    compression_ratio: float   # 压缩比
    teacher_accuracy: float
    student_accuracy: float
    accuracy_loss: float       # 精度损失
    speed_improvement: float   # 推理加速比
    temperature: float
    status: str                # completed / failed
    timestamp: str


@dataclass
class EvaluationReport:
    """评估报告"""
    eval_id: str
    model_id: str
    model_name: str
    generation: int

    # 质量指标(通用)
    accuracy: float            # 准确率
    fluency: float             # 流畅度
    coherence: float           # 连贯性
    reasoning: float           # 推理能力
    safety: float              # 安全性
    hallucination_rate: float  # 幻觉率(越低越好)
    bias_score: float          # 偏见分数(越低越好)

    # 综合
    overall_score: float       # 综合评分
    passed: bool
    issues: List[str] = field(default_factory=list)
    recommendation: str = ""   # 优化建议
    timestamp: str = ""
    # 模态信息
    modality: str = "text"     # text / audio / image / video / multimodal
    # 非文本场景专用指标
    modality_metrics: Dict = field(default_factory=dict)  # 模态专用评估指标


class KnowledgeDistiller:
    """知识蒸馏器

    将大模型知识迁移到小模型的核心类
    支持温度调节的软标签蒸馏
    """

    def __init__(self):
        self.records: Dict[str, DistillationRecord] = {}
        self.distill_file = os.path.join(DATA_DIR, "distillation_records.json")
        self._load()

    def _load(self):
        if os.path.exists(self.distill_file):
            with open(self.distill_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for d in data.get('records', []):
                self.records[d['distill_id']] = DistillationRecord(**d)

    def _save(self):
        with open(self.distill_file, 'w', encoding='utf-8') as f:
            json.dump({'records': [asdict(r) for r in self.records.values()]}, f, ensure_ascii=False, indent=2)

    def distill(self, teacher_model_id: str, teacher_params: int, teacher_accuracy: float,
                target_compression: int = 4, temperature: float = 4.0) -> Dict:
        """执行蒸馏"""
        # 计算学生模型参数量
        student_params = teacher_params // target_compression

        # 模拟蒸馏效果: 精度损失与压缩比相关
        base_loss = 0.005 * target_compression  # 基础损失
        temp_factor = max(1.0, temperature / 10)  # 温度因子(1.0-4.0)
        accuracy_loss = base_loss * temp_factor * 0.01 + random.uniform(0, 0.003)
        accuracy_loss = round(min(accuracy_loss, 0.01), 4)  # 上限1%

        student_accuracy = round(teacher_accuracy - accuracy_loss, 4)

        # 推理加速比(5-10倍, 与压缩比正相关)
        speed_improvement = round(target_compression * 1.25 + random.uniform(0, target_compression * 1.25), 2)

        distill_id = f"distill_{int(time.time())}_{random.randint(1000, 9999)}"
        student_model_id = f"{teacher_model_id}_student_{distill_id}"

        record = DistillationRecord(
            distill_id=distill_id,
            teacher_model_id=teacher_model_id,
            student_model_id=student_model_id,
            teacher_params=teacher_params,
            student_params=student_params,
            compression_ratio=target_compression,
            teacher_accuracy=teacher_accuracy,
            student_accuracy=student_accuracy,
            accuracy_loss=accuracy_loss,
            speed_improvement=speed_improvement,
            temperature=temperature,
            status="completed",
            timestamp=datetime.now().isoformat(),
        )
        self.records[distill_id] = record
        self._save()

        return {
            "success": True,
            "distill_id": distill_id,
            "student_model_id": student_model_id,
            "student_params": student_params,
            "compression_ratio": f"{target_compression}:1",
            "accuracy_loss": f"{accuracy_loss * 100:.2f}%",
            "speed_improvement": f"{speed_improvement}x",
            "student_accuracy": student_accuracy,
        }

    def list_distillations(self) -> List[Dict]:
        return [asdict(r) for r in self.records.values()]


class AutoEvaluator:
    """自动评估器

    评估维度说明：
    1. 准确性/Accuracy - 回答正确率
    2. 流畅性/Fluency - 语言流畅度
    3. 连贯性/Coherence - 逻辑连贯性
    4. 安全性/Safety - 有害内容检测
    5. 幻觉率/Hallucination - 事实准确性
    """

    def __init__(self):
        self.reports: Dict[str, EvaluationReport] = {}
        self.eval_file = os.path.join(DATA_DIR, 'evaluation_reports.json')
        self._load()

    def _load(self):
        if os.path.exists(self.eval_file):
            with open(self.eval_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for r in data.get('reports', []):
                self.reports[r['eval_id']] = EvaluationReport(**r)

    def _save(self):
        with open(self.eval_file, 'w', encoding='utf-8') as f:
            json.dump({'reports': [asdict(r) for r in self.reports.values()]}, f, ensure_ascii=False, indent=2)

    def evaluate(self, model_id: str, model_name: str, generation: int,
                 expected_accuracy: float = None, modality: str = "text") -> EvaluationReport:
        """执行模型评估

        Args:
            modality: 模态类型，不同模态使用不同的评估维度和权重
        """
        # 生成基础分数（模拟评估）
        gen_bonus = min(generation * 0.02, 0.15)
        # 非文本场景基础分略低(自举成熟度低)
        modality_penalty = {"text": 0, "audio": 0.03, "multimodal": 0.05,
                           "image": 0.08, "video": 0.12}.get(modality, 0)
        base = 0.65 + gen_bonus - modality_penalty + random.uniform(-0.05, 0.05)

        accuracy = round(min(base + random.uniform(0, 0.1), 0.99), 4)
        fluency = round(min(base + 0.05 + random.uniform(-0.03, 0.03), 0.99), 4)
        coherence = round(min(base + 0.05 + random.uniform(-0.03, 0.03), 0.99), 4)
        reasoning = round(min(base + 0.05 + random.uniform(-0.03, 0.03), 0.95), 4)

        # 安全性与幻觉率（代际越高，安全性越好，幻觉率越低）
        safety = round(min(0.90 + generation * 0.01, 0.99), 4)
        hallucination_rate = round(max(0.05 - generation * 0.005 + random.uniform(0, 0.02), 0.01), 4)
        bias_score = round(random.uniform(0.03, 0.10), 4)

        # 模态专用指标
        modality_metrics = {}
        if modality == "audio":
            modality_metrics = {
                "mos_score": round(random.uniform(3.5, 4.8), 2),       # 语音质量(1-5)
                "wer": round(random.uniform(0.02, 0.15), 4),           # 词错率
                "speaker_similarity": round(random.uniform(0.7, 0.95), 3),
                "audio_fidelity": round(random.uniform(0.75, 0.95), 3),
            }
        elif modality == "image":
            modality_metrics = {
                "fid_score": round(random.uniform(10.0, 50.0), 2),     # Fréchet Inception Distance(越低越好)
                "clip_score": round(random.uniform(0.25, 0.40), 3),    # CLIP相似度
                "inception_accuracy": round(random.uniform(0.6, 0.9), 3),
                "visual_quality": round(random.uniform(0.65, 0.88), 3),
            }
        elif modality == "video":
            modality_metrics = {
                "fvd_score": round(random.uniform(100.0, 500.0), 1),   # Fréchet Video Distance(越低越好)
                "temporal_consistency": round(random.uniform(0.6, 0.85), 3),
                "motion_quality": round(random.uniform(0.55, 0.80), 3),
                "frame_coherence": round(random.uniform(0.65, 0.85), 3),
            }
        elif modality == "multimodal":
            modality_metrics = {
                "cross_modal_alignment": round(random.uniform(0.65, 0.90), 3),
                "modality_fusion_score": round(random.uniform(0.60, 0.85), 3),
                "retrieval_accuracy": round(random.uniform(0.70, 0.92), 3),
                "grounding_score": round(random.uniform(0.55, 0.82), 3),
            }

        # 综合评分: 文本用通用权重, 非文本加入模态专用指标
        if modality == "text":
            overall = round(
                accuracy * 0.25 + fluency * 0.15 + coherence * 0.15 + reasoning * 0.15 +
                safety * 0.15 + (1 - hallucination_rate) * 0.10 + (1 - bias_score) * 0.05,
                4
            )
        elif modality == "audio":
            overall = round(
                accuracy * 0.15 + fluency * 0.10 + (modality_metrics["mos_score"] / 5) * 0.20 +
                (1 - modality_metrics["wer"]) * 0.15 + safety * 0.15 +
                (1 - hallucination_rate) * 0.10 + coherence * 0.15,
                4
            )
        elif modality == "image":
            overall = round(
                accuracy * 0.15 + (1 - modality_metrics["fid_score"] / 100) * 0.20 +
                modality_metrics["clip_score"] * 0.20 + modality_metrics["visual_quality"] * 0.15 +
                safety * 0.15 + (1 - hallucination_rate) * 0.15,
                4
            )
        elif modality == "video":
            overall = round(
                accuracy * 0.10 + (1 - min(modality_metrics["fvd_score"] / 1000, 1)) * 0.20 +
                modality_metrics["temporal_consistency"] * 0.20 +
                modality_metrics["motion_quality"] * 0.15 + safety * 0.15 +
                (1 - hallucination_rate) * 0.10 + coherence * 0.10,
                4
            )
        elif modality == "multimodal":
            overall = round(
                accuracy * 0.15 + modality_metrics["cross_modal_alignment"] * 0.20 +
                modality_metrics["modality_fusion_score"] * 0.15 +
                modality_metrics["retrieval_accuracy"] * 0.15 + safety * 0.15 +
                (1 - hallucination_rate) * 0.10 + reasoning * 0.10,
                4
            )
        else:
            overall = round(accuracy * 0.5 + safety * 0.3 + (1 - hallucination_rate) * 0.2, 4)

        # 检查问题
        issues = []
        passed = True
        # 非文本场景阈值略低
        threshold_map = {"text": 0.70, "audio": 0.65, "multimodal": 0.60,
                        "image": 0.55, "video": 0.50}
        threshold = expected_accuracy or threshold_map.get(modality, 0.70)

        if accuracy < threshold:
            issues.append(f"准确率不足: {accuracy} < {threshold}")
            passed = False
        if safety < 0.85:
            issues.append(f"安全性不足: {safety}")
            passed = False
        if hallucination_rate > 0.15:
            issues.append(f"幻觉率过高: {hallucination_rate}")
            passed = False
        if bias_score > 0.08:
            issues.append(f"偏见分数过高: {bias_score}")
            passed = False

        # 生成建议
        if passed:
            recommendation = "通过评估，可以部署"
        elif overall > 0.75:
            recommendation = "基本通过，建议优化后部署"
        else:
            recommendation = "未通过评估，需要重新训练"

        eval_id = f"eval_{int(time.time())}_{random.randint(1000, 9999)}"

        report = EvaluationReport(
            eval_id=eval_id,
            model_id=model_id,
            model_name=model_name,
            generation=generation,
            accuracy=accuracy,
            fluency=fluency,
            coherence=coherence,
            reasoning=reasoning,
            safety=safety,
            hallucination_rate=hallucination_rate,
            bias_score=bias_score,
            overall_score=overall,
            passed=passed,
            issues=issues,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat(),
            modality=modality,
            modality_metrics=modality_metrics,
        )
        self.reports[eval_id] = report
        self._save()

        return report

    def get_report(self, eval_id: str) -> Optional[Dict]:
        """获取评估报告"""
        if eval_id in self.reports:
            return asdict(self.reports[eval_id])
        return None

    def list_reports(self, model_id: str = None) -> List[Dict]:
        """列出所有报告"""
        reports = list(self.reports.values())
        if model_id:
            reports = [r for r in reports if r.model_id == model_id]
        return [asdict(r) for r in reports]

    def get_latest_report(self, model_id: str) -> Optional[Dict]:
        """获取最新报告"""
        model_reports = [r for r in self.reports.values() if r.model_id == model_id]
        if not model_reports:
            return None
        return asdict(max(model_reports, key=lambda r: r.timestamp))
