#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
外部数据集预处理脚本 — 零外部依赖
将三个开源仓库的数据转为纯文本，用于灵元模型训练
"""

import json
import glob
import os

BASE = "/data/user/work/external_datasets"
OUTPUT = "/data/user/work/external_datasets/processed"
os.makedirs(OUTPUT, exist_ok=True)


def process_tang_poems(max_files=5):
    """数据源1: chinese-poetry/全唐诗 (chinese-poetry/chinese-poetry)
    
    JSON格式: {author, paragraphs[], title, id}
    转换为: 每首诗一行，格式: 标题 作者 内容
    """
    repo = os.path.join(BASE, "chinese-poetry/repo/全唐诗")
    files = sorted(glob.glob(os.path.join(repo, "poet.tang.*.json")))[:max_files]
    
    lines = []
    total_poems = 0
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            poems = json.load(f)
        for p in poems:
            title = p.get("title", "")
            author = p.get("author", "")
            paragraphs = p.get("paragraphs", [])
            content = "".join(paragraphs)
            line = f"{title} {author}\n{content}\n"
            lines.append(line)
            total_poems += 1
    
    text = "\n".join(lines)
    out_path = os.path.join(OUTPUT, "tang_poems.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"[唐诗] 文件数: {len(files)}, 诗歌数: {total_poems}, "
          f"文本大小: {len(text)} 字符 -> {out_path}")
    return out_path


def process_lunyu():
    """数据源2: chinese-poetry/论语 (chinese-poetry/chinese-poetry)
    
    JSON格式: [{chapter, paragraphs[]}]
    转换为: 每章一段，格式: 章节名 + 段落内容
    """
    fpath = os.path.join(BASE, "chinese-poetry/repo/论语/lunyu.json")
    with open(fpath, "r", encoding="utf-8") as f:
        chapters = json.load(f)
    
    lines = []
    total_paragraphs = 0
    for ch in chapters:
        chapter = ch.get("chapter", "")
        paragraphs = ch.get("paragraphs", [])
        lines.append(f"《{chapter}》")
        for p in paragraphs:
            lines.append(p)
            total_paragraphs += 1
        lines.append("")
    
    text = "\n".join(lines)
    out_path = os.path.join(OUTPUT, "lunyu.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"[论语] 章节数: {len(chapters)}, 段落数: {total_paragraphs}, "
          f"文本大小: {len(text)} 字符 -> {out_path}")
    return out_path


def process_shakespeare(max_chars=100000):
    """数据源3: tinyshakespeare (karpathy/char-rnn)
    
    已是纯文本，截取前 max_chars 字符用于轻量训练
    """
    fpath = os.path.join(BASE, "tinyshakespeare/input.txt")
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 截取
    text = text[:max_chars]
    
    out_path = os.path.join(OUTPUT, "shakespeare.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"[莎士比亚] 原始大小: {os.path.getsize(fpath)} 字符, "
          f"截取后: {len(text)} 字符 -> {out_path}")
    return out_path


def main():
    print("=" * 60)
    print("外部数据集预处理 — 三个开源仓库")
    print("=" * 60)
    
    p1 = process_tang_poems(max_files=5)
    p2 = process_lunyu()
    p3 = process_shakespeare(max_chars=100000)
    
    print("\n" + "=" * 60)
    print("预处理完成！三个训练数据文件:")
    print(f"  1. 唐诗 (chinese-poetry/chinese-poetry): {p1}")
    print(f"  2. 论语 (chinese-poetry/chinese-poetry): {p2}")
    print(f"  3. 莎士比亚 (karpathy/char-rnn):        {p3}")
    print("=" * 60)
    
    # 打印每个文件的预览
    for name, path in [("唐诗", p1), ("论语", p2), ("莎士比亚", p3)]:
        print(f"\n--- {name}预览 ---")
        with open(path, "r", encoding="utf-8") as f:
            preview = f.read(300)
        print(preview)


if __name__ == "__main__":
    main()
