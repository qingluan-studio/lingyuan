#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成能力测试 — 自回归生成必须输出合法token序列，tokenizer编解码可往返"""
import os, sys, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lingyuan_train"))
from lingyuan_v7 import LingyuanModel, ModelConfig, CharTokenizer, BUILTIN_CORPUS


def test_generation_valid_tokens():
    random.seed(2026)
    cfg = ModelConfig.tiny()
    model = LingyuanModel(cfg)
    tokenizer = CharTokenizer(vocab_size=cfg.vocab_size)
    tokenizer.fit_on_text(BUILTIN_CORPUS)

    prompt = tokenizer.encode("春眠不觉晓")
    assert len(prompt) > 0, "提示词编码失败"

    out = model.generate(prompt, max_new=24, temperature=0.8, top_k=10)
    assert len(out) == len(prompt) + 24, f"生成长度错误: {len(out)}"
    for t in out:
        assert 0 <= t < cfg.vocab_size, f"生成非法token: {t}"

    text = tokenizer.decode(out)
    assert isinstance(text, str) and len(text) > 0
    return f"生成24个token全部合法, 解码文本长度={len(text)}"


def test_tokenizer_roundtrip():
    tokenizer = CharTokenizer(vocab_size=300)
    tokenizer.fit_on_text(BUILTIN_CORPUS)
    sample = "春眠不觉晓，处处闻啼鸟。"
    ids = tokenizer.encode(sample)
    assert len(ids) > 0
    back = tokenizer.decode(ids)
    # 词表覆盖的字符必须无损往返
    covered = [ch for ch in sample if ch in tokenizer.char2id]
    assert all(ch in back for ch in covered), f"编解码丢失字符: {back}"
    return f"编解码往返正常 ({len(ids)} tokens)"


if __name__ == "__main__":
    print(test_generation_valid_tokens())
    print(test_tokenizer_roundtrip())
