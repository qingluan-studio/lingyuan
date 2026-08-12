#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""灵元大模型 — 测试运行器（零第三方依赖）

运行 tests/ 下全部真实测试：
  - test_gradients   数值梯度检验（解析梯度 vs 中心差分，证明反向传播真实）
  - test_training    训练真实性（loss 下降、参数确实更新）
  - test_save_load   模型保存/加载往返一致性
  - test_generation  自回归生成与 tokenizer 往返
  - test_code_model  代码模型全链路冒烟

用法: python run_tests.py
退出码: 0=全部通过, 1=存在失败
"""
import os
import sys
import time
import traceback
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(HERE, "tests")


def load_module(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("=" * 62)
    print("灵元大模型 — 测试套件（全部为真实行为验证，无模拟指标）")
    print("=" * 62)

    test_files = sorted(
        f for f in os.listdir(TESTS_DIR)
        if f.startswith("test_") and f.endswith(".py")
    )

    total, passed, failed = 0, 0, []
    t_start = time.time()

    for fname in test_files:
        path = os.path.join(TESTS_DIR, fname)
        print(f"\n▶ {fname}")
        try:
            mod = load_module(path)
        except Exception:
            print(f"  ✗ 模块加载失败")
            traceback.print_exc()
            failed.append((fname, "<import>"))
            total += 1
            continue

        test_fns = [(n, fn) for n, fn in vars(mod).items()
                    if n.startswith("test_") and callable(fn)]
        for name, fn in test_fns:
            total += 1
            t0 = time.time()
            try:
                detail = fn()
                dt = time.time() - t0
                passed += 1
                print(f"  ✓ {name} ({dt:.1f}s)")
                if detail:
                    print(f"      {detail}")
            except Exception as e:
                dt = time.time() - t0
                failed.append((fname, name))
                print(f"  ✗ {name} ({dt:.1f}s): {e}")
                traceback.print_exc()

    dt_all = time.time() - t_start
    print("\n" + "=" * 62)
    print(f"结果: {passed}/{total} 通过, 用时 {dt_all:.1f}s")
    if failed:
        print("失败项:")
        for f, n in failed:
            print(f"  - {f}::{n}")
        print("=" * 62)
        sys.exit(1)
    print("全部通过 — 核心模型的训练/推理/保存均为真实实现")
    print("=" * 62)


if __name__ == "__main__":
    main()
