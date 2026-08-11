#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灵元模型 — 增强版外部数据集训练
配置升级: small模型, 512词表, 64维, 4层, seq_len=64
"""

import os
import sys
import json
import time

LINGYUAN_DIR = "/data/user/work/lingyuan"
DATA_DIR = "/data/user/work/external_datasets/processed"
OUTPUT_DIR = "/data/user/work/external_datasets/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, LINGYUAN_DIR)

from lingyuan_enterprise import (
    CharTokenizer, TextDataLoader, TrainingEngine,
    ExperimentTracker, CheckpointManager, log
)
from part17_hetero_enterprise import HeteroGPU, HeteroConfig

log.level = 20  # INFO

DATASETS = [
    {
        "name": "唐诗",
        "source": "chinese-poetry/chinese-poetry",
        "file": os.path.join(DATA_DIR, "tang_poems.txt"),
        "prompt": "春眠不觉晓",
        "desc": "5000首唐诗，417K字符",
    },
    {
        "name": "论语",
        "source": "chinese-poetry/chinese-poetry",
        "file": os.path.join(DATA_DIR, "lunyu.txt"),
        "prompt": "子曰学而时习之",
        "desc": "20章512段，22K字符",
    },
    {
        "name": "莎士比亚",
        "source": "karpathy/char-rnn",
        "file": os.path.join(DATA_DIR, "shakespeare.txt"),
        "prompt": "First Citizen",
        "desc": "200K字符英文戏剧",
    },
]


def train_one_dataset(dataset, epochs=12, steps_per_epoch=25):
    """增强配置训练"""
    name = dataset["name"]
    data_file = dataset["file"]

    print(f"\n{'='*60}")
    print(f"  训练数据集: {name} (增强配置)")
    print(f"  来源: {dataset['source']}")
    print(f"  描述: {dataset['desc']}")
    print(f"{'='*60}")

    # 清空旧checkpoint
    ckpt_dir = os.path.join(LINGYUAN_DIR, "lingyuan_enterprise_data", "checkpoints")
    if os.path.exists(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            os.remove(os.path.join(ckpt_dir, f))

    # 增强配置 — small级别
    config = HeteroConfig.small()
    config.vocab_size = 512
    config.hidden_dim = 48
    config.num_heads = 4
    config.num_layers = 3
    config.ffn_dim = 192
    config.max_seq_len = 48
    config.learning_rate = 0.008
    config.bootstrap_buffer_size = 512

    # 分词器 — 512词表覆盖更多字符
    tokenizer = CharTokenizer(vocab_size=512)
    loader = TextDataLoader(tokenizer, seq_len=config.max_seq_len, batch_size=4)

    # 加载数据
    loader.load_file(data_file)
    print(f"  词表大小: {len(tokenizer.char2id)}")
    print(f"  训练序列数: {len(loader._data)}")

    # 模型
    gpu = HeteroGPU(config)
    stats = gpu.stats()
    print(f"  模型参数量: {stats['config']['params']}")
    print(f"  配置: dim={config.hidden_dim} heads={config.num_heads} "
          f"layers={config.num_layers} seq_len={config.max_seq_len}")

    # 训练组件
    tracker = ExperimentTracker()
    ckpt_mgr = CheckpointManager()
    engine = TrainingEngine(gpu, tokenizer, loader, tracker, ckpt_mgr)

    # 训练
    start_time = time.time()
    result = engine.train(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        early_stop_patience=10,
        log_interval=10,
        resume=False,
    )
    total_time = time.time() - start_time

    # 生成测试 — 多个温度
    prompt = dataset["prompt"]
    prompt_ids = tokenizer.encode(prompt)
    generations = {}
    if prompt_ids:
        for temp in [0.5, 0.8, 1.0]:
            generated = gpu.generate(prompt_ids, max_new=64, temperature=temp)
            generations[f"temp_{temp}"] = tokenizer.decode(generated)
    else:
        generations["error"] = "无法编码提示"

    # 保存模型
    model_path = os.path.join(OUTPUT_DIR, f"model_{name}_enhanced.het")
    gpu.save(model_path)
    print(f"  模型已保存: {model_path}")

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
        "config": {
            "vocab_size": 512,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_layers": 4,
            "ffn_dim": 256,
            "max_seq_len": 64,
            "learning_rate": 0.005,
        },
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
        "generations": generations,
        "model_path": model_path,
    }

    print(f"\n  --- {name} 增强训练结果 ---")
    print(f"  Epochs: {summary['epochs_completed']}")
    print(f"  Steps: {summary['total_steps']}")
    print(f"  Time: {summary['total_time']}")
    print(f"  Loss: {first_loss:.4f} -> {last_loss:.4f} (best: {best_loss:.4f})")
    print(f"  Improvement: {improvement:.1f}%")
    for temp_key, gen_text in generations.items():
        print(f"  生成({temp_key}, prompt='{prompt}'):")
        print(f"    {gen_text[:100]}")

    return summary


def main():
    print("=" * 60)
    print("  灵元模型 — 增强版外部数据集训练")
    print("  配置: 512词表, 64维, 4层, 4头, seq_len=64")
    print("  零外部依赖 · 纯Python标准库")
    print("=" * 60)

    all_results = []
    for dataset in DATASETS:
        if not os.path.exists(dataset["file"]):
            print(f"\n  [跳过] 数据文件不存在: {dataset['file']}")
            continue
        result = train_one_dataset(dataset, epochs=12, steps_per_epoch=25)
        all_results.append(result)

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "enhanced_training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 汇总
    print("\n" + "=" * 60)
    print("  增强训练汇总报告")
    print("=" * 60)
    print(f"{'数据集':<10} {'Epochs':>7} {'Steps':>7} {'Time':>8} "
          f"{'初始Loss':>10} {'最终Loss':>10} {'改善率':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['dataset']:<10} {r['epochs_completed']:>7} "
              f"{r['total_steps']:>7} {r['total_time']:>8} "
              f"{r['first_loss']:>10.4f} {r['last_loss']:>10.4f} "
              f"{r['improvement_pct']:>7.1f}%")

    print(f"\n  报告: {report_path}")
    print("=" * 60)

    # 同时保存到workspace
    ws_report = "/workspace/enhanced_training_report.json"
    with open(ws_report, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  报告已推送到: {ws_report}")


if __name__ == "__main__":
    main()
