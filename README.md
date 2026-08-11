# 灵元 Lingyuan Model

> V7.0 ULTRA — DeepNorm + GQA + MoE + RoPE/ALiBi 混合位置编码
> 零外部依赖 · 纯Python标准库实现

## 项目结构

```
lingyuan/
├── lingyuan_train/          # 诗词语言模型训练
│   ├── lingyuan_v7.py            # V7.0 ULTRA 核心架构
│   ├── lingyuan_v7_trained.het   # 训练模型 (1M参数)
│   ├── training_report.json      # 训练报告
│   ├── training_results.html     # 可视化报告
│   ├── consolidated_report.json  # 综合报告
│   ├── enhanced_training_report.json
│   ├── v2_training_report.json
│   ├── external_training_report.json
│   ├── model_唐诗_enhanced.het
│   ├── model_论语_enhanced.het
│   ├── model_莎士比亚_enhanced.het
│   └── train_enhanced.py
│
├── lingyuan_code/           # 代码模型训练
│   ├── lingyuan_code_v7.py       # 代码训练系统
│   ├── lingyuan_code_v7.het      # 代码模型 (1M参数)
│   ├── code_training_report.json
│   ├── code_generations.json
│   ├── code_training_results.html
│   └── data/
│       ├── humaneval.jsonl       # HumanEval 数据集
│       └── mbpp.jsonl            # MBPP 数据集
│
├── .gitignore
└── README.md
```

## 架构特性

| 特性 | 说明 |
|------|------|
| DeepNorm | 深层网络稳定训练, α=(2L)^0.25 |
| GQA | 分组查询注意力, KV头压缩至1/4 |
| MoE | 4专家Top-K=2路由 + SwiGLU |
| RoPE | 旋转位置编码, θ=10000 |
| ALiBi | 线性偏置注意力, 支持长序列外推 |
| Sliding Window | 滑动窗口注意力, 高效长上下文 |

## 模型配置

- 参数量: 1,027,968 (基础: 241,536 + MoE: 786,432)
- 层数: 4
- 隐藏维度: 64
- 注意力头: 4 (KV头: 1)
- FFN维度: 256
- 专家数: 4 (激活: 2)

## 训练结果

### 诗词模型
- 10轮 400步, loss 6.3358 → 6.2209
- 续训验证: loss → 6.1032

### 代码模型
- 数据: HumanEval(164) + MBPP(974) + 内置(34)
- 10轮 400步, loss 6.5550 → 6.2320
- 最低单步 loss: 5.9717

## 开源数据

- [HumanEval](https://huggingface.co/datasets/openai_humaneval) — OpenAI
- [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) — Google

## License

MIT
