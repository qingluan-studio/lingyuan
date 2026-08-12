#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""代码模型冒烟测试 — CodeTokenizer/CodeDataLoader/训练全链路真实可运行"""
import os, sys, math, random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "lingyuan_train"))
sys.path.insert(0, os.path.join(_HERE, "..", "lingyuan_code"))

from lingyuan_v7 import LingyuanModel, ModelConfig
from lingyuan_code_v7 import CodeTokenizer, CodeDataLoader


def test_code_pipeline_smoke():
    random.seed(11)
    cfg = ModelConfig(
        vocab_size=256, hidden_dim=32, num_heads=2, num_kv_heads=1,
        num_layers=2, ffn_dim=64, max_seq_len=32,
    )
    model = LingyuanModel(cfg)

    tokenizer = CodeTokenizer(vocab_size=cfg.vocab_size)
    loader = CodeDataLoader(tokenizer, seq_len=cfg.max_seq_len, batch_size=2)
    loader.load_builtin_corpus()

    # 若存在 HumanEval/MBPP 数据则一并加载（真实数据集）
    data_dir = os.path.join(_HERE, "..", "lingyuan_code", "data")
    he = os.path.join(data_dir, "humaneval.jsonl")
    mbpp = os.path.join(data_dir, "mbpp.jsonl")
    if os.path.exists(he):
        loader.load_humaneval(he)
    if os.path.exists(mbpp):
        loader.load_mbpp(mbpp)
    loader.prepare()
    assert len(loader._sequences) > 0, "代码语料准备失败"

    first, last = [], []
    steps = 15
    for step in range(steps):
        inputs, targets = loader.sample_batch()
        sl = []
        for inp, tgt in zip(inputs, targets):
            loss = model.train_step(inp, tgt, lr=0.003)
            assert math.isfinite(loss)
            sl.append(loss)
        if step < 2:
            first.extend(sl)
        elif step >= steps - 2:
            last.extend(sl)

    f, l = sum(first) / len(first), sum(last) / len(last)
    assert l < f, f"代码模型loss未下降: {f:.4f} -> {l:.4f}"
    return f"代码模型15步真实训练: loss {f:.4f} -> {l:.4f} (序列{len(loader._sequences)}条)"


if __name__ == "__main__":
    print(test_code_pipeline_smoke())
