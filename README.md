# 灵元大模型 Lingyuan Model

> 自主进化 AI 系统 —— 自举训练 · 知识蒸馏 · 多供应商调度 · 多模态评估

[![CI](https://github.com/qingluan-studio/lingyuan/actions/workflows/ci.yml/badge.svg)](https://github.com/qingluan-studio/lingyuan/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib)-success.svg)](#环境要求)

灵元大模型（Lingyuan Model，代号"灵元"）是一套以"自主进化"为核心目标的端到端 AI 系统框架。系统将算力经济、数据生成、自举训练、自动评估、Agent 编排、六层级递进融合决策与自优化闭环融为一体，构建出可持续自我迭代的智能体底座。

灵元坚持"模型即数据"理念：通过血缘追踪与向量表征，将模型权重、训练数据与评估结果统一为可流转、可交易、可审计的数据资产。整个系统以纯 Python 标准库实现，零外部依赖即可运行与测试。

## 目录

- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [测试](#测试)
- [持续集成](#持续集成)
- [环境要求](#环境要求)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 核心特性

- **自举训练（Bootstrap Training）**：模型自主生成训练数据、过滤质量并完成自我迭代，形成无需人工标注的训练闭环。
- **知识蒸馏**：将大模型能力蒸馏至轻量模型，在保持效果的同时降低推理成本。
- **多供应商算力调度**：跨多个算力厂商智能调度，内置 Token 经济模型与能源碳排放追踪。
- **多模态生成与评估**：覆盖音频、图像、视频等多模态的生成能力，并配备模态专用评估指标。
- **六层级递进融合决策引擎**：分层级递进融合多源信号，输出可解释的决策结果。
- **Agent 团队编排**：多 Agent 协同与空间模型协同，由工作流引擎驱动复杂任务编排。
- **自优化闭环引擎**：基于知识库持续优化系统行为，实现"评估—反馈—改进"闭环。
- **模型即数据 + 血缘追踪**：模型权重与数据资产统一管理，全链路血缘可追溯。
- **冷热分层存储与数据交易协议**：按访问热度分层存储，支持数据资产的合规交易。
- **GitHub CI/CD 管线**：事件触发的自动化构建、测试与部署管线。
- **移动端仪表盘**：提供仪表盘、快捷操作与推送通知，随时掌握系统状态。
- **零外部依赖**：纯 Python 标准库实现，开箱即用。

## 系统架构

灵元采用 14 层分层架构，自底向上覆盖从算力基础设施到移动端交互的完整链路：

| #   | 层级           | 核心能力                                                       |
| --- | -------------- | -------------------------------------------------------------- |
| 1   | 基础设施层     | 算力 Token 经济 / 能源碳追踪 / 多厂商调度                      |
| 2   | 安全与治理层   | 权限控制 / 审计日志 / 合规策略                                 |
| 3   | 存储层         | 冷热分层 / 数据交易协议                                        |
| 4   | 模型数据层     | 模型即数据 / 血缘追踪 / 向量表征                               |
| 5   | 数据生成层     | 自举数据生成 / 质量过滤 / 多模态支持                           |
| 6   | 训练层         | 自举训练循环 / 安全阀 / 知识蒸馏                               |
| 7   | 评估层         | 自动评估器 / 模态专用指标                                      |
| 8   | 多模态层       | 音频 / 图像 / 视频 / 多模态生成与评估                          |
| 9   | 编排层         | Agent 团队 / 空间模型协同 / 工作流引擎                         |
| 10  | 决策层         | 六层级递进融合决策引擎                                         |
| 11  | 闭环层         | 自优化引擎 / 知识库                                            |
| 12  | 管线层         | GitHub CI/CD / 事件触发                                        |
| 13  | 接口层         | CLI / REST API / SDK / Web 控制台                              |
| 14  | 移动层         | 仪表盘 / 快捷操作 / 推送通知                                   |

## 文件结构

| 文件              | 规模          | 说明                                                                       |
| ----------------- | ------------- | -------------------------------------------------------------------------- |
| `lingyuan_full.py` | 960 行        | 主入口与核心运行时，提供 CLI 与测试入口（`python lingyuan_full.py test`）  |
| `part2.py`        | 861 行        | 基础设施、存储与模型数据层实现                                              |
| `part3.py`        | 1378 行       | 数据生成层与训练层实现（自举数据生成、质量过滤、自举训练循环、知识蒸馏）   |
| `part4.py`        | 1226 行       | 评估层与多模态层实现（自动评估器、模态专用指标、多模态生成与评估）         |
| `part5.py`        | 1410 行       | 编排层、闭环层与管线层实现（Agent 团队、工作流引擎、自优化引擎、知识库）   |
| `part6.py`        | 六层决策引擎  | 六层级递进融合决策引擎实现                                                 |

## 快速开始

灵元仅需 Python 3.11+ 与标准库，无需安装任何第三方依赖。

```bash
# 1. 克隆仓库
git clone https://github.com/qingluan-studio/lingyuan.git
cd lingyuan

# 2. 运行测试套件
python lingyuan_full.py test
```

测试通过即表示环境就绪。也可作为库引入：

```python
from lingyuan_full import LingyuanOrchestrator

orch = LingyuanOrchestrator()
orch.quick_train(generations=3)
```

## 测试

项目内置测试入口，直接通过主程序运行：

```bash
python lingyuan_full.py test
```

该命令会执行完整测试套件并输出结果。CI 环境同样使用此命令。

## 持续集成

项目通过 GitHub Actions 实现持续集成，配置见 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)：

- **触发条件**：向 `main` 分支推送，或针对 `main` 发起 Pull Request
- **运行环境**：Ubuntu + Python 3.11
- **流程**：检出代码 → 配置 Python 3.11 → 安装依赖 → 运行 `python lingyuan_full.py test`
- **产物**：测试过程中的运行时数据与日志将作为 Artifact 上传，便于排查

## 环境要求

- **Python**：3.11 及以上
- **依赖**：无外部 pip 依赖（纯标准库实现）

`requirements.txt` 仅声明可选项，默认为空即可运行。

## 贡献指南

欢迎通过 Pull Request 贡献代码。提交前请确保：

1. `python lingyuan_full.py test` 全部通过
2. 新增能力配备相应测试
3. 遵循现有代码风格

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 链接

- 仓库地址：<https://github.com/qingluan-studio/lingyuan>
