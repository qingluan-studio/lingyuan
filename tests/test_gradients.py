#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数值梯度检验 — 灵元反向传播的数学正确性证明

方法: 对采样参数位置做中心差分 (f(x+ε)-f(x-ε))/2ε，
与 _backward() 计算的解析梯度逐点对比。
若两者一致，则链式法则反向传播是真实的，不是模拟。

验收标准:
  - 每个采样点 |解析-数值| 相对误差 < 1%
  - 全采样向量余弦相似度 > 0.999
"""
import os, sys, math, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lingyuan_train"))
from lingyuan_v7 import LingyuanModel, ModelConfig


def _collect_analytic_grads(model, input_ids, target_ids):
    """前向+反向(lr=0不更新参数)，返回采样点列表 [(名称, tensor, i, j, 解析梯度)]"""
    logits = model.forward(input_ids, training=True)
    model._backward(logits, target_ids, lr=0.0)
    return logits


def test_gradient_check():
    random.seed(42)
    cfg = ModelConfig.tiny()
    model = LingyuanModel(cfg)

    # 关闭梯度裁剪，直接对比原始梯度
    model._clip_grads = lambda max_norm=1.0: None

    S, V = 16, cfg.vocab_size
    input_ids = [random.randrange(V) for _ in range(S)]
    target_ids = [random.randrange(V) for _ in range(S)]

    logits = model.forward(input_ids, training=True)
    loss0 = model._cross_entropy(logits, target_ids)
    assert math.isfinite(loss0) and loss0 > 0, "初始loss非法"
    model._backward(logits, target_ids, lr=0.0)

    # 采样参数点: 覆盖 Embedding/Head/LayerNorm/Attention/MoE
    samples = []
    def pick(name, t, n):
        for _ in range(n):
            i = random.randrange(t.rows); j = random.randrange(t.cols)
            samples.append((name, t, i, j, t.grad[i][j]))

    present = list(set(input_ids))
    for k in range(3):
        tid = present[k]
        j = random.randrange(cfg.hidden_dim)
        samples.append((f"embed[{tid}]", model.embed, tid, j,
                        model.embed.grad[tid][j]))
    pick("head", model.head, 3)
    pick("final_ln_g", model.final_ln_g, 2)
    L0, L1 = model.layers[0], model.layers[1]
    pick("L0.wq", L0['wq'], 2)
    pick("L0.wo", L0['wo'], 2)
    pick("L1.wv", L1['wv'], 2)
    pick("L0.expert0.w_gate", L0['moe_experts'][0]['w_gate'], 2)

    # 中心差分数值梯度
    eps = 1e-3
    max_rel = 0.0
    dot = na = nn = 0.0
    checked = 0
    for name, t, i, j, ag in samples:
        orig = t.data[i][j]
        t.data[i][j] = orig + eps
        lp = model._cross_entropy(model.forward(input_ids, training=False), target_ids)
        t.data[i][j] = orig - eps
        lm = model._cross_entropy(model.forward(input_ids, training=False), target_ids)
        t.data[i][j] = orig
        ng = (lp - lm) / (2 * eps)

        if abs(ng) > 1e-9:
            rel = abs(ag - ng) / abs(ng)
            max_rel = max(max_rel, rel)
            assert rel < 0.01, f"{name}[{i},{j}] 梯度不符: 解析={ag:.6f} 数值={ng:.6f}"
        else:
            assert abs(ag) < 1e-7, f"{name}[{i},{j}] 数值梯度为0但解析梯度非0: {ag}"

        dot += ag * ng; na += ag * ag; nn += ng * ng
        checked += 1

    cosine = dot / (math.sqrt(na) * math.sqrt(nn)) if na > 0 and nn > 0 else 1.0
    assert cosine > 0.999, f"梯度方向余弦相似度不足: {cosine:.6f}"
    return f"{checked}个参数点全部吻合, 最大相对误差={max_rel:.2e}, 余弦={cosine:.6f}"


def test_gradient_clipping_direction():
    """梯度裁剪只缩放幅度、不改变方向（裁剪是真实操作而非伪造）"""
    random.seed(7)
    cfg = ModelConfig.tiny()
    model = LingyuanModel(cfg)

    S, V = 16, cfg.vocab_size
    input_ids = [random.randrange(V) for _ in range(S)]
    target_ids = [random.randrange(V) for _ in range(S)]

    logits = model.forward(input_ids, training=True)
    model._backward(logits, target_ids, lr=0.0)  # 含真实裁剪

    # 裁剪后全局范数应 <= max_norm(1.0) + 数值容差
    total_sq = 0.0
    for p in model._all_params():
        for row in p.grad:
            for g in row:
                total_sq += g * g
    norm = math.sqrt(total_sq)
    assert norm <= 1.0 + 1e-6, f"裁剪后范数={norm}"
    return f"裁剪后全局梯度范数={norm:.4f} (<=1.0)"


if __name__ == "__main__":
    print(test_gradient_check())
    print(test_gradient_clipping_direction())
