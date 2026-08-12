#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练真实性测试 — 损失必须真实下降，且训练前后模型参数确实被更新"""
import os, sys, math, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lingyuan_train"))
from lingyuan_v7 import (LingyuanModel, ModelConfig, CharTokenizer,
                         TextDataLoader, BUILTIN_CORPUS)


def test_training_loss_decreases():
    """在内置古诗词语料上真实训练30步，loss必须显著下降"""
    random.seed(123)
    cfg = ModelConfig.tiny()
    model = LingyuanModel(cfg)

    tokenizer = CharTokenizer(vocab_size=cfg.vocab_size)
    loader = TextDataLoader(tokenizer, seq_len=cfg.max_seq_len, batch_size=2)
    loader.load_text(BUILTIN_CORPUS)
    assert len(loader._data) > 0, "语料加载失败"

    first_losses, last_losses = [], []
    steps = 30
    for step in range(steps):
        inputs, targets = loader.sample_batch()
        step_losses = []
        for inp, tgt in zip(inputs, targets):
            loss = model.train_step(inp, tgt, lr=0.003)
            assert math.isfinite(loss), f"第{step}步loss非有限值: {loss}"
            step_losses.append(loss)
        if step < 3:
            first_losses.extend(step_losses)
        elif step >= steps - 3:
            last_losses.extend(step_losses)

    first_avg = sum(first_losses) / len(first_losses)
    last_avg = sum(last_losses) / len(last_losses)
    assert last_avg < first_avg, f"loss未下降: 初始{first_avg:.4f} -> 最终{last_avg:.4f}"
    return f"30步真实训练: loss {first_avg:.4f} -> {last_avg:.4f}"


def test_params_actually_updated():
    """训练一步后参数值必须发生变化（排除'假更新'）"""
    random.seed(5)
    cfg = ModelConfig.tiny()
    model = LingyuanModel(cfg)

    snapshot = {}
    for name, p in [("head", model.head), ("embed", model.embed),
                    ("final_ln_g", model.final_ln_g)]:
        snapshot[name] = [row[:] for row in p.data]

    S, V = cfg.max_seq_len, cfg.vocab_size
    inp = [random.randrange(V) for _ in range(S)]
    tgt = [random.randrange(V) for _ in range(S)]
    model.train_step(inp, tgt, lr=0.01)

    changed = 0
    for name, p in [("head", model.head), ("embed", model.embed),
                    ("final_ln_g", model.final_ln_g)]:
        diff = sum(1 for i in range(p.rows) for j in range(p.cols)
                   if p.data[i][j] != snapshot[name][i][j])
        assert diff > 0, f"{name} 训练后未更新（假训练）"
        changed += diff
    return f"训练1步后 {changed} 个参数值真实变化"


if __name__ == "__main__":
    print(test_training_loss_decreases())
    print(test_params_actually_updated())
