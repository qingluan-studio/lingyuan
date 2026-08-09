#!/usr/bin/env python3
"""灵元大模型 — 测试运行器

统一执行全部模块的测试套件。
用法: python run_tests.py
"""

import sys
import os

# 确保工作目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """加载所有模块并运行测试"""
    print("=" * 60)
    print("灵元大模型 — 全局测试运行器")
    print("=" * 60)

    # 按依赖顺序加载所有模块
    modules = [
        "lingyuan_full.py",
        "part2.py",
        "part3.py",
        "part4.py",
        "part6.py",   # 融合决策引擎 (part5依赖其FusionDecisionEngine)
        "part7.py",   # 安全/可观测/API/注册/课程/经济/知识图谱
        "part8.py",   # 联邦学习/蒸馏/RLHF/量化/向量库/提示工程/边缘/记忆
        "part9.py",   # 模型本体: Transformer/Tokenizer/位置编码/采样/KVCache/训练引擎
        "part10.py",  # 外部知识接入: 连接器/解析/爬虫/脱敏/版权/训练接口/教师/去重
        "part11.py",  # 推理服务: 引擎/批处理/流式/缓存/FunctionCall/ChatTemplate
        "part12.py",  # 模型格式: 序列化/HF导出/ONNX/GGUF/外部导入
        "part13.py",  # 微调: LoRA/全参数/SFT/DPO/持续学习/领域适配
        "part14.py",  # API服务: HTTP/OpenAI兼容/WebSocket/gRPC/文档/SDK
        "part15.py",  # MLOps: 实验追踪/任务队列/GPU调度/监控/对比
        "part16.py",  # UI+安全: WebChat/Playground/训练面板/水印/APIKey/血缘
        "part5.py",
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for mod_file in modules:
        path = os.path.join(base_dir, mod_file)
        if os.path.exists(path):
            print(f"  [加载] {mod_file}")
            with open(path, 'r', encoding='utf-8') as f:
                exec(f.read(), globals())
        else:
            print(f"  [跳过] {mod_file} (不存在)")

    # 运行测试
    sys.argv = [sys.argv[0], "test"]
    if 'main' in globals() and callable(globals()['main']):
        result = globals()['main']()
        if result and isinstance(result, dict):
            failed = result.get("failed", 0)
            sys.exit(1 if failed > 0 else 0)
        sys.exit(0)
    else:
        print("[错误] 未找到 main() 入口")
        sys.exit(1)


if __name__ == "__main__":
    main()
