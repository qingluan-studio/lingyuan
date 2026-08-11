#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
外部数据集批量训练脚本 — 零外部依赖
分别用三个开源数据集训练灵元模型，记录loss变化并生成报告
"""

import os
import sys
import json
import time
import shutil

# 设置路径
LINGYUAN_DIR = "/data/user/work/lingyuan"
DATA_DIR = "/data/user/work/external_datasets/processed"
OUTPUT_DIR = "/data/user/work/external_datasets/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, LINGYUAN_DIR)

# 导入灵元组件
from lingyuan_enterprise import (
    CharTokenizer, TextDataLoader, TrainingEngine,
    ExperimentTracker, CheckpointManager, EnterpriseLogger, log
)
from part17_hetero_enterprise import HeteroGPU, HeteroConfig

# 设置日志级别为INFO，显示训练进度
log.level = 20  # INFO级别

# 三个数据集配置
DATASETS = [
    {
        "name": "唐诗",
        "source": "chinese-poetry/chinese-poetry (GitHub)",
        "file": os.path.join(DATA_DIR, "tang_poems.txt"),
        "prompt": "春眠不觉晓",
        "desc": "5000首唐诗，417K字符",
    },
    {
        "name": "论语",
        "source": "chinese-poetry/chinese-poetry (GitHub)",
        "file": os.path.join(DATA_DIR, "lunyu.txt"),
        "prompt": "子曰学而",
        "desc": "20章512段，22K字符",
    },
    {
        "name": "莎士比亚",
        "source": "karpathy/char-rnn (GitHub)",
        "file": os.path.join(DATA_DIR, "shakespeare.txt"),
        "prompt": "First Citizen",
        "desc": "100K字符英文戏剧",
    },
]


def train_one_dataset(dataset, epochs=10, steps_per_epoch=20):
    """用单个数据集训练模型"""
    name = dataset["name"]
    data_file = dataset["file"]

    print(f"\n{'='*60}")
    print(f"  训练数据集: {name}")
    print(f"  来源: {dataset['source']}")
    print(f"  描述: {dataset['desc']}")
    print(f"  文件: {data_file}")
    print(f"{'='*60}")

    # 清空checkpoint目录（避免冲突）
    ckpt_dir = os.path.join(LINGYUAN_DIR, "lingyuan_enterprise_data", "checkpoints")
    if os.path.exists(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            os.remove(os.path.join(ckpt_dir, f))

    # 配置 — tiny级，纯Python友好
    config = HeteroConfig.tiny()
    config.vocab_size = 256
    config.hidden_dim = 32
    config.num_heads = 2
    config.num_layers = 2
    config.ffn_dim = 64
    config.max_seq_len = 32
    config.learning_rate = 0.01
    config.bootstrap_buffer_size = 256

    # 分词器
    tokenizer = CharTokenizer(vocab_size=256)
    loader = TextDataLoader(tokenizer, seq_len=config.max_seq_len, batch_size=4)

    # 加载数据
    loader.load_file(data_file)
    print(f"  词表大小: {len(tokenizer.char2id)}")
    print(f"  训练序列数: {len(loader._data)}")

    # 模型
    gpu = HeteroGPU(config)
    stats = gpu.stats()
    print(f"  模型参数量: {stats['config']['params']}")

    # 训练组件
    tracker = ExperimentTracker()
    ckpt_mgr = CheckpointManager()
    engine = TrainingEngine(gpu, tokenizer, loader, tracker, ckpt_mgr)

    # 训练
    start_time = time.time()
    result = engine.train(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        early_stop_patience=10,  # 不太激进早停
        log_interval=10,
        resume=False,  # 不恢复，从头训练
    )
    total_time = time.time() - start_time

    # 生成测试
    prompt = dataset["prompt"]
    prompt_ids = tokenizer.encode(prompt)
    if prompt_ids:
        generated = gpu.generate(prompt_ids, max_new=48, temperature=0.7)
        generated_text = tokenizer.decode(generated)
    else:
        generated_text = "(无法编码提示)"

    # 保存模型
    model_path = os.path.join(OUTPUT_DIR, f"model_{name}.het")
    gpu.save(model_path)

    # 记录结果
    metrics = result.get("metrics", [])
    loss_history = [m["loss"] for m in metrics]
    first_loss = loss_history[0] if loss_history else 0
    last_loss = loss_history[-1] if loss_history else 0
    best_loss = result.get("best_loss", 0)
    improvement = ((first_loss - last_loss) / first_loss * 100) if first_loss > 0 else 0

    summary = {
        "dataset": name,
        "source": dataset["source"],
        "description": dataset["desc"],
        "vocab_size": len(tokenizer.char2id),
        "num_sequences": len(loader._data),
        "model_params": stats["config"]["params"],
        "epochs_completed": result.get("epochs_completed", 0),
        "total_steps": result.get("total_steps", 0),
        "total_time": f"{total_time:.1f}s",
        "first_loss": round(first_loss, 4),
        "last_loss": round(last_loss, 4),
        "best_loss": round(best_loss, 4),
        "improvement_pct": round(improvement, 1),
        "loss_history": [round(l, 4) for l in loss_history],
        "prompt": prompt,
        "generated": generated_text,
        "model_path": model_path,
    }

    print(f"\n  --- {name} 训练结果 ---")
    print(f"  Epochs: {summary['epochs_completed']}")
    print(f"  Steps: {summary['total_steps']}")
    print(f"  Time: {summary['total_time']}")
    print(f"  Loss: {first_loss:.4f} -> {last_loss:.4f} (best: {best_loss:.4f})")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  生成测试 (prompt='{prompt}'):")
    print(f"  {generated_text}")

    return summary


def main():
    print("=" * 60)
    print("  灵元模型 — 外部数据集训练实验")
    print("  三个开源仓库 · 零外部依赖 · 纯Python标准库")
    print("=" * 60)

    all_results = []
    for dataset in DATASETS:
        if not os.path.exists(dataset["file"]):
            print(f"\n  [跳过] 数据文件不存在: {dataset['file']}")
            continue
        result = train_one_dataset(dataset, epochs=10, steps_per_epoch=20)
        all_results.append(result)

    # 保存汇总报告
    report_path = os.path.join(OUTPUT_DIR, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print("\n" + "=" * 60)
    print("  训练汇总报告")
    print("=" * 60)
    print(f"{'数据集':<10} {'Epochs':>7} {'Steps':>7} {'Time':>8} "
          f"{'初始Loss':>10} {'最终Loss':>10} {'改善率':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['dataset']:<10} {r['epochs_completed']:>7} "
              f"{r['total_steps']:>7} {r['total_time']:>8} "
              f"{r['first_loss']:>10.4f} {r['last_loss']:>10.4f} "
              f"{r['improvement_pct']:>7.1f}%")

    print(f"\n  报告已保存: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
