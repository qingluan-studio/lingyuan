#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""保存/加载往返测试 — .het 模型文件必须完整保真"""
import os, sys, tempfile, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lingyuan_train"))
from lingyuan_v7 import LingyuanModel, ModelConfig


def test_save_load_roundtrip():
    random.seed(99)
    cfg = ModelConfig.tiny()
    model = LingyuanModel(cfg)

    S, V = 12, cfg.vocab_size
    ids = [random.randrange(V) for _ in range(S)]

    logits_before = model.forward(ids, training=False)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "roundtrip.het")
        model.save(path)
        assert os.path.exists(path) and os.path.getsize(path) > 0
        loaded = LingyuanModel.load(path)

    logits_after = loaded.forward(ids, training=False)

    assert logits_before.rows == logits_after.rows
    assert logits_before.cols == logits_after.cols
    max_diff = 0.0
    for i in range(logits_before.rows):
        for j in range(logits_before.cols):
            d = abs(logits_before.data[i][j] - logits_after.data[i][j])
            max_diff = max(max_diff, d)
    assert max_diff < 1e-6, f"保存/加载后输出不一致, 最大差异={max_diff}"
    return f"保存/加载往返一致, 最大logit差异={max_diff:.2e}"


if __name__ == "__main__":
    print(test_save_load_roundtrip())
