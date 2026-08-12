# experimental/ — 实验模块存档（未验证）

> **异负得正，异虚为实。** 把虚拟的标成虚拟，真实的才算真实。

本目录存放灵元项目早期按"52项清单"生成的大批企业级概念模块。
**它们不是项目核心，未经过验证，不应被当作可用的功能引用。**

## 为什么被移到这里

2026-08-12 全项目审计的结论：

1. **核心训练机制是真实的**（见根目录 README「真实性验证」），
   但本目录的模块大多停留在概念演示阶段。
2. 部分模块存在**用随机数伪造指标**的代码，例如：
   - `part16.py`: `loss_data = [3.5 - i*0.02 + random.uniform(...) ...]`（伪造训练曲线）
   - `part4.py` / `part8.py`: 用 `random.uniform(0.95, 0.99)` 伪造精度保持率
   - 多处用 `time.sleep` 伪装计算耗时
3. 旧的 `run_tests.py` 通过 `exec` 把这些模块加载进同一个命名空间，
   类名互相覆盖（如 `ModelConfig`），**测试根本无法跑通**。
4. 部分脚本硬编码了 `/workspace/...`、`/data/user/...` 等
   只在特定 AI IDE 环境存在的路径，换环境即失效。

## 目录内容

| 内容 | 说明 |
|------|------|
| `part2.py` ~ `part31.py` | 52项清单模块：存储/Agent/决策引擎/安全/虚拟GPU/训练引擎/外部知识/推理服务/格式导出/微调/API/MLOps/UI/数据工厂/自进化等 |
| `lingyuan_full.py`、`lingyuan_enterprise.py` | 早期聚合入口 |
| `train_enhanced.py`、`train_external.py`、`preprocess.py` | 旧训练/预处理脚本（含外部路径依赖） |
| `legacy_models/` | 假反向传播时代训练出的 `.het` 模型（已废弃，仅作存档） |
| `reports/`、`reports_training/`、`legacy_models_trained/`、`*_report*.json` | 旧时代的训练报告，其中"enhanced/v2/external"系列的指标不可信 |

## 如果想复活其中某个模块

1. 删除其中伪造指标的代码，换成真实计算
2. 去掉硬编码路径，改为相对路径
3. 为它写独立的、能跑通的测试，放进根目录 `tests/`
4. 通过验证后再移回主目录

在此之前，请把它们当作设计草稿，而不是可用组件。
