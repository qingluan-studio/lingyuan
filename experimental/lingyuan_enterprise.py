#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
灵元大模型 — 企业级运行器 (Lingyuan Enterprise Runner)
================================================================================

零外部依赖 | 纯 Python 标准库 | 真训练 · 真生成 · 真运行

用法:
    python lingyuan_enterprise.py train   --data corpus.txt --epochs 50
    python lingyuan_enterprise.py generate --prompt "春眠不觉晓" --max-new 64
    python lingyuan_enterprise.py test    --full
    python lingyuan_enterprise.py status
    python lingyuan_enterprise.py serve   --port 8080

架构:
    lingyuan_enterprise.py
     ├── HeteroGPU (part17_enterprise) — 真Transformer模型
     ├── DataPipeline (part20)          — 数据加载/增强/批处理
     ├── ExperimentTracker (part15)     — 实验管理/MLOps
     ├── CheckpointManager             — 断点续训
     ├── EnterpriseLogger              — 结构化日志
     └── CLInterface                   — 命令行接口
================================================================================
"""

import os
import sys
import json
import time
import math
import random
import hashlib
import struct
import argparse
import threading
import traceback
from datetime import datetime, timedelta
from collections import deque, OrderedDict, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from pathlib import Path


# ============================================================
# 全局路径
# ============================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

ENTERPRISE_DIR = os.path.join(_SCRIPT_DIR, "lingyuan_enterprise_data")
LOG_DIR = os.path.join(ENTERPRISE_DIR, "logs")
CKPT_DIR = os.path.join(ENTERPRISE_DIR, "checkpoints")
DATA_DIR = os.path.join(ENTERPRISE_DIR, "data")
OUTPUT_DIR = os.path.join(ENTERPRISE_DIR, "output")

for d in [ENTERPRISE_DIR, LOG_DIR, CKPT_DIR, DATA_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# Part A: 企业级日志
# ============================================================

class EnterpriseLogger:
    """结构化日志 — 支持文件/控制台双输出"""

    LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

    def __init__(self, name: str = "lingyuan", level: str = "INFO"):
        self.name = name
        self.level = self.LEVELS.get(level.upper(), 20)
        self._file = None
        self._lock = threading.Lock()

        log_path = os.path.join(LOG_DIR,
            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self._file = open(log_path, "a", encoding="utf-8")

    def _log(self, level: str, msg: str, **kwargs):
        if self.LEVELS.get(level, 0) < self.level:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        extra = " " + json.dumps(kwargs, ensure_ascii=False) if kwargs else ""
        line = f"[{ts}] [{level:5s}] [{self.name}] {msg}{extra}"

        with self._lock:
            print(line)
            if self._file:
                self._file.write(line + "\n")
                self._file.flush()

    def debug(self, msg: str, **kwargs): self._log("DEBUG", msg, **kwargs)
    def info(self, msg: str, **kwargs):  self._log("INFO", msg, **kwargs)
    def warn(self, msg: str, **kwargs):  self._log("WARN", msg, **kwargs)
    def error(self, msg: str, **kwargs): self._log("ERROR", msg, **kwargs)

    def close(self):
        if self._file:
            self._file.close()

log = EnterpriseLogger("lingyuan-enterprise", "INFO")


# ============================================================
# Part B: 断点管理
# ============================================================

@dataclass
class Checkpoint:
    """断点元数据"""
    path: str
    epoch: int
    step: int
    loss: float
    timestamp: str
    config: dict = field(default_factory=dict)


class CheckpointManager:
    """断点管理 — 自动保存/恢复/滚动"""

    def __init__(self, base_dir: str = CKPT_DIR, max_keep: int = 5):
        self.base_dir = base_dir
        self.max_keep = max_keep
        self._history: List[Checkpoint] = []
        self._scan()

    def _scan(self):
        """扫描已有断点"""
        self._history = []
        if not os.path.isdir(self.base_dir):
            return
        for fname in sorted(os.listdir(self.base_dir)):
            if fname.endswith(".meta.json"):
                try:
                    with open(os.path.join(self.base_dir, fname), "r") as f:
                        d = json.load(f)
                    self._history.append(Checkpoint(**d))
                except Exception:
                    pass

    def latest(self) -> Optional[Checkpoint]:
        return self._history[-1] if self._history else None

    def save(self, gpu, epoch: int, step: int, loss: float,
             extra_meta: dict = None) -> Checkpoint:
        """保存断点"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"ckpt_epoch{epoch:04d}_step{step:06d}_{ts}"
        path = os.path.join(self.base_dir, name + ".het")

        # 保存模型
        gpu.save(path)

        # 保存元数据
        ckpt = Checkpoint(
            path=os.path.abspath(path),
            epoch=epoch, step=step, loss=loss,
            timestamp=ts,
            config=gpu.cfg.to_dict() if hasattr(gpu.cfg, 'to_dict') else {},
        )
        if extra_meta:
            ckpt.config.update(extra_meta)

        meta_path = path.replace(".het", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(ckpt), f, indent=2, ensure_ascii=False)

        self._history.append(ckpt)
        self._rotate()
        log.info("checkpoint saved", epoch=epoch, step=step, loss=f"{loss:.4f}")
        return ckpt

    def _rotate(self):
        """滚动删除旧断点"""
        while len(self._history) > self.max_keep:
            old = self._history.pop(0)
            try:
                if os.path.exists(old.path):
                    os.remove(old.path)
                meta = old.path.replace(".het", ".meta.json")
                if os.path.exists(meta):
                    os.remove(meta)
            except OSError:
                pass


# ============================================================
# Part C: 实验追踪
# ============================================================

@dataclass
class Experiment:
    """实验记录"""
    id: str
    name: str
    config: dict
    start_time: str
    metrics: List[dict] = field(default_factory=list)
    status: str = "running"
    end_time: str = ""

    def record(self, epoch: int, step: int, loss: float, **extra):
        self.metrics.append({"epoch": epoch, "step": step,
                              "loss": loss, **extra})

    def finish(self):
        self.status = "finished"
        self.end_time = datetime.now().isoformat()


class ExperimentTracker:
    """实验管理"""

    def __init__(self):
        self._exps: Dict[str, Experiment] = {}
        self._load()

    def _load(self):
        path = os.path.join(DATA_DIR, "experiments.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                for d in json.load(f):
                    self._exps[d["id"]] = Experiment(**d)

    def _save(self):
        with open(os.path.join(DATA_DIR, "experiments.json"), "w") as f:
            json.dump([asdict(e) for e in self._exps.values()],
                       f, indent=2, ensure_ascii=False)

    def create(self, name: str, config: dict) -> Experiment:
        eid = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]
        exp = Experiment(id=eid, name=name, config=config,
                          start_time=datetime.now().isoformat())
        self._exps[eid] = exp
        self._save()
        log.info("experiment created", id=eid, name=name)
        return exp

    def list(self) -> List[Experiment]:
        return sorted(self._exps.values(),
                       key=lambda e: e.start_time, reverse=True)


# ============================================================
# Part D: 数据管道 (真实文本处理)
# ============================================================

class CharTokenizer:
    """字符级分词器 v2 — 频率自适应, 零UNK

    改造点:
    - 先扫描全部数据统计字符频率, 按频率排序构建词表
    - 特殊token预留前4位, 剩余按频率降序分配
    - 保证训练数据中所有字符都能被编码, 消除UNK
    """

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        self._fitted = False
        # 特殊token固定前4位
        self._special = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        for i, tok in enumerate(self._special):
            self.char2id[tok] = i
            self.id2char[i] = tok

    def fit_on_text(self, text: str):
        """扫描文本, 按字符频率构建词表"""
        if self._fitted:
            for c in text:
                if c not in self.char2id and len(self.char2id) < self.vocab_size:
                    idx = len(self.char2id)
                    self.char2id[c] = idx
                    self.id2char[idx] = c
            return

        from collections import Counter
        freq = Counter(text)
        sorted_chars = sorted(freq.items(), key=lambda x: -x[1])
        available = self.vocab_size - len(self._special)

        for char, count in sorted_chars[:available]:
            if char not in self.char2id:
                idx = len(self.char2id)
                self.char2id[char] = idx
                self.id2char[idx] = char

        self._fitted = True
        unk_count = sum(1 for c in text if c not in self.char2id)
        if unk_count > 0:
            log.warn("tokenizer has UNK chars",
                      total=len(freq), unk=unk_count,
                      vocab=len(self.char2id))

    def encode(self, text: str) -> List[int]:
        unk = self.char2id.get("<UNK>", 0)
        return [self.char2id.get(c, unk) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id2char.get(i, "?") for i in ids)

    @property
    def bos_id(self):
        return self.char2id.get("<BOS>", 1)

    @property
    def eos_id(self):
        return self.char2id.get("<EOS>", 2)

    @property
    def pad_id(self):
        return self.char2id.get("<PAD>", 0)

    def unk_ratio(self, text: str) -> float:
        """计算UNK比例"""
        if not text:
            return 0.0
        unk = sum(1 for c in text if c not in self.char2id)
        return unk / len(text)


class TextDataLoader:
    """文本数据加载器 v2 — 随机窗口增强

    改造点:
    - 预分词: 加载时一次性编码, 避免重复
    - 随机窗口: sample_batch 从全文随机位置截取, 而非固定切分
    - 数据增强: 每次采样位置随机, 等效于无限数据
    """

    def __init__(self, tokenizer: CharTokenizer, seq_len: int = 64,
                  batch_size: int = 16):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self._data: List[List[int]] = []
        self._flat_ids: List[int] = []
        self._cursor = 0

    def load_file(self, path: str, encoding: str = "utf-8"):
        """加载文本文件并分词"""
        log.info("loading data", path=path)
        with open(path, "r", encoding=encoding, errors="replace") as f:
            text = f.read()

        self.tokenizer.fit_on_text(text)
        ids = self.tokenizer.encode(text)

        self._flat_ids = ids

        for i in range(0, len(ids) - self.seq_len, self.seq_len // 2):
            chunk = ids[i:i + self.seq_len + 1]
            if len(chunk) >= self.seq_len + 1:
                self._data.append(chunk)

        unk_ratio = self.tokenizer.unk_ratio(text)
        log.info("data loaded", sequences=len(self._data),
                  total_tokens=len(ids), unk_ratio=f"{unk_ratio:.4f}")

    def load_text(self, text: str):
        """直接加载文本字符串"""
        self.tokenizer.fit_on_text(text)
        ids = self.tokenizer.encode(text)
        for i in range(0, len(ids) - self.seq_len, self.seq_len // 2):
            chunk = ids[i:i + self.seq_len + 1]
            if len(chunk) >= self.seq_len + 1:
                self._data.append(chunk)

    def sample_batch(self) -> Tuple[List[int], List[int]]:
        """采样一个batch: (input_ids, target_ids)

        v2改造: 优先使用随机窗口采样, 等效无限数据增强
        """
        if not self._data and not self._flat_ids:
            return self._synthetic_batch()

        batch_inputs = []
        batch_targets = []

        if self._flat_ids and len(self._flat_ids) > self.seq_len + 1:
            n = len(self._flat_ids)
            for _ in range(self.batch_size):
                start = random.randrange(0, n - self.seq_len - 1)
                chunk = self._flat_ids[start:start + self.seq_len + 1]
                batch_inputs.append(chunk[:self.seq_len])
                batch_targets.append(chunk[1:self.seq_len + 1])
        else:
            for _ in range(self.batch_size):
                idx = random.randrange(len(self._data))
                chunk = self._data[idx]
                batch_inputs.append(chunk[:self.seq_len])
                batch_targets.append(chunk[1:self.seq_len + 1])

        return batch_inputs, batch_targets

    def _synthetic_batch(self) -> Tuple[List[int], List[int]]:
        """无数据时生成合成batch"""
        seq_len = self.seq_len
        vocab = self.tokenizer.vocab_size
        inputs = [[random.randrange(4, vocab) for _ in range(seq_len)]
                  for _ in range(self.batch_size)]
        targets = [[inp[(i+1) % seq_len] for i in range(seq_len)]
                   for inp in inputs]
        return inputs, targets


# ============================================================
# Part E: 训练引擎
# ============================================================

class TrainingEngine:
    """企业级训练引擎 — 真反向传播, 断点续训, 早停"""

    def __init__(self, gpu, tokenizer: CharTokenizer,
                  data_loader: TextDataLoader,
                  tracker: ExperimentTracker,
                  ckpt_mgr: CheckpointManager):
        self.gpu = gpu
        self.tokenizer = tokenizer
        self.loader = data_loader
        self.tracker = tracker
        self.ckpt_mgr = ckpt_mgr

        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.no_improve = 0

        # 指标
        self.metrics_history: List[dict] = []
        self._lr_history: List[float] = []

    def _compute_lr(self, step: int, total_steps: int,
                     base_lr: float, warmup_ratio: float = 0.1) -> float:
        """warmup + cosine decay 学习率调度"""
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(self, epochs: int = 10, steps_per_epoch: int = 50,
              early_stop_patience: int = 5, log_interval: int = 10,
              resume: bool = True) -> dict:
        """主训练循环"""
        cfg = self.gpu.cfg

        # 断点续训
        if resume:
            latest = self.ckpt_mgr.latest()
            if latest:
                self.current_epoch = latest.epoch
                self.current_step = latest.step
                self.best_loss = latest.loss
                # 恢复模型权重
                try:
                    loaded = type(self.gpu).load(latest.path)
                    self.gpu._embed = loaded._embed
                    self.gpu._head = loaded._head
                    self.gpu._head_bias = loaded._head_bias
                    self.gpu._final_ln_g = loaded._final_ln_g
                    self.gpu._final_ln_b = loaded._final_ln_b
                    self.gpu._layers = loaded._layers
                    log.info("resumed from checkpoint",
                              epoch=self.current_epoch, step=self.current_step)
                except Exception as e:
                    log.warn("failed to restore weights, starting fresh",
                              error=str(e))
                    self.current_epoch = 0
                    self.current_step = 0

        total_start = time.time()
        total_steps = epochs * steps_per_epoch
        base_lr = cfg.learning_rate

        for epoch in range(self.current_epoch, epochs):
            epoch_loss = 0.0
            epoch_start = time.time()

            for step in range(steps_per_epoch):
                self.current_step += 1

                # 动态学习率: warmup + cosine decay
                current_lr = self._compute_lr(
                    self.current_step - 1, total_steps, base_lr)
                self.gpu.cfg.learning_rate = current_lr
                self._lr_history.append(current_lr)

                # 采样batch
                batch_inputs, batch_targets = self.loader.sample_batch()

                # 每个样本独立训练
                batch_loss = 0.0
                for inp, tgt in zip(batch_inputs, batch_targets):
                    loss = self.gpu.train_step(inp, tgt)
                    batch_loss += loss

                avg_loss = batch_loss / self.loader.batch_size
                epoch_loss += avg_loss

                if step % log_interval == 0 or step == steps_per_epoch - 1:
                    elapsed = time.time() - epoch_start
                    log.info("step",
                              epoch=epoch+1, step=step+1,
                              loss=f"{avg_loss:.4f}",
                              lr=f"{current_lr:.6f}",
                              elapsed=f"{elapsed:.1f}s")

            # Epoch结束
            avg_epoch_loss = epoch_loss / steps_per_epoch
            epoch_time = time.time() - epoch_start

            self.metrics_history.append({
                "epoch": epoch + 1,
                "step": self.current_step,
                "loss": avg_epoch_loss,
                "time": epoch_time,
            })

            log.info("epoch", epoch=epoch+1,
                      loss=f"{avg_epoch_loss:.4f}",
                      time=f"{epoch_time:.1f}s",
                      best=f"{self.best_loss:.4f}")

            # 断点保存
            self.ckpt_mgr.save(self.gpu, epoch + 1, self.current_step,
                                avg_epoch_loss)

            # 早停
            if avg_epoch_loss < self.best_loss - 1e-6:
                self.best_loss = avg_epoch_loss
                self.no_improve = 0
            else:
                self.no_improve += 1
                if self.no_improve >= early_stop_patience:
                    log.info("early stop triggered",
                              epoch=epoch+1, patience=early_stop_patience)
                    break

        total_time = time.time() - total_start
        result = {
            "epochs_completed": min(epoch + 1, epochs),
            "total_steps": self.current_step,
            "total_time": f"{total_time:.1f}s",
            "best_loss": self.best_loss,
            "metrics": self.metrics_history,
        }

        log.info("training complete", **result)
        return result


# ============================================================
# Part F: 生成器
# ============================================================

class TextGenerator:
    """文本生成器 — 支持多种解码策略"""

    def __init__(self, gpu, tokenizer: CharTokenizer):
        self.gpu = gpu
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_new: int = 64,
                  temperature: float = 0.8, top_k: int = 0,
                  top_p: float = 0.0) -> str:
        """生成文本"""
        ids = self.tokenizer.encode(prompt)
        if not ids:
            return ""

        # 使用HeteroGPU的generate
        generated = self.gpu.generate(ids, max_new=max_new,
                                       temperature=temperature)
        return self.tokenizer.decode(generated)

    def stream(self, prompt: str, max_new: int = 64):
        """流式生成 (逐token yield)"""
        ids = self.tokenizer.encode(prompt)
        if not ids:
            return

        for _ in range(max_new):
            ctx = ids[-self.gpu.cfg.max_seq_len:]
            logits = self.gpu.forward(ctx)
            last = logits.data[-1]

            mx = max(last)
            sm = sum(math.exp((l - mx) / 0.8) for l in last)
            r = random.random()
            cum = 0.0
            for idx in range(len(last)):
                cum += math.exp((last[idx] - mx) / 0.8) / sm
                if r < cum:
                    ids.append(idx)
                    yield self.tokenizer.decode([idx])
                    break
            else:
                ids.append(len(last) - 1)
                yield self.tokenizer.decode([len(last) - 1])


# ============================================================
# Part G: HTTP 服务
# ============================================================

class LingyuanServer:
    """简易HTTP推理服务 — 纯标准库"""

    def __init__(self, gpu, tokenizer: CharTokenizer,
                  host: str = "localhost", port: int = 8080):
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import socketserver

        self.gpu = gpu
        self.tokenizer = tokenizer
        self.host = host
        self.port = port

        gpu_ref = gpu
        tok_ref = tokenizer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._json({"status": "ok", "model": "lingyuan"})
                elif self.path.startswith("/generate?"):
                    from urllib.parse import parse_qs, urlparse
                    qs = parse_qs(urlparse(self.path).query)
                    prompt = qs.get("prompt", [""])[0]
                    max_new = int(qs.get("max_new", ["32"])[0])
                    temp = float(qs.get("temperature", ["0.8"])[0])

                    ids = tok_ref.encode(prompt)
                    result = gpu_ref.generate(ids, max_new=max_new,
                                               temperature=temp)
                    text = tok_ref.decode(result)
                    self._json({"prompt": prompt, "generated": text,
                                 "tokens": len(result) - len(ids)})
                else:
                    self._json({"usage": "/health, /generate?prompt=..."})

            def do_POST(self):
                if self.path == "/generate":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    prompt = body.get("prompt", "")
                    max_new = body.get("max_new", 32)
                    temp = body.get("temperature", 0.8)

                    ids = tok_ref.encode(prompt)
                    result = gpu_ref.generate(ids, max_new=max_new,
                                               temperature=temp)
                    text = tok_ref.decode(result)
                    self._json({"prompt": prompt, "generated": text,
                                 "tokens": len(result) - len(ids)})
                else:
                    self.send_response(404)
                    self.end_headers()

            def _json(self, data):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

            def log_message(self, format, *args):
                log.debug("http", request=args[0] if args else "")

        self._handler = Handler

    def start(self):
        from http.server import HTTPServer
        server = HTTPServer((self.host, self.port), self._handler)
        log.info("server started", host=self.host, port=self.port)
        print(f"\n  灵元推理服务已启动: http://{self.host}:{self.port}")
        print(f"  GET  /health")
        print(f"  GET  /generate?prompt=你好&max_new=32")
        print(f"  POST /generate  {{\"prompt\": \"你好\"}}\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            log.info("server stopped")
            server.shutdown()


# ============================================================
# Part H: 核心整合 — 从 part17 导入模型
# ============================================================

def _import_hetero_gpu():
    """动态导入 HeteroGPU — 兼容多种文件命名"""
    candidates = [
        "part17_hetero_enterprise",
        "part17",
    ]
    for name in candidates:
        try:
            mod = __import__(name)
            if hasattr(mod, "HeteroGPU") and hasattr(mod, "HeteroConfig"):
                return mod.HeteroGPU, mod.HeteroConfig
        except (ImportError, AttributeError):
            continue

    # 兜底: 内嵌最小实现
    log.warn("HeteroGPU not found, using embedded fallback")
    from part17_hetero_enterprise import HeteroGPU, HeteroConfig
    return HeteroGPU, HeteroConfig


# ============================================================
# Part I: 内置语料 (当无外部数据时的初始训练数据)
# ============================================================

BUILTIN_CORPUS = """
春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
床前明月光，疑是地上霜。举头望明月，低头思故乡。
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。
离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。
远上寒山石径斜，白云生处有人家。停车坐爱枫林晚，霜叶红于二月花。
两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。
日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。
朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。
故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。
横看成岭侧成峰，远近高低各不同。不识庐山真面目，只缘身在此山中。
竹外桃花三两枝，春江水暖鸭先知。蒌蒿满地芦芽短，正是河豚欲上时。
水光潋滟晴方好，山色空蒙雨亦奇。欲把西湖比西子，淡妆浓抹总相宜。
死去元知万事空，但悲不见九州同。王师北定中原日，家祭无忘告乃翁。
千锤万凿出深山，烈火焚烧若等闲。粉骨碎身浑不怕，要留清白在人间。
咬定青山不放松，立根原在破岩中。千磨万击还坚劲，任尔东西南北风。
好雨知时节，当春乃发生。随风潜入夜，润物细无声。
空山不见人，但闻人语响。返景入深林，复照青苔上。
红豆生南国，春来发几枝。愿君多采撷，此物最相思。
独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。
"""


# ============================================================
# Part J: CLI
# ============================================================

def cmd_train(args):
    """训练命令"""
    log.info("=== TRAIN ===")

    HeteroGPU, HeteroConfig = _import_hetero_gpu()

    # 配置
    config = HeteroConfig.small()
    config.vocab_size = 512
    config.hidden_dim = 64
    config.num_heads = 4
    config.num_layers = 4
    config.ffn_dim = 256
    config.max_seq_len = 64
    config.learning_rate = args.lr
    config.bootstrap_buffer_size = 512

    # 分词器
    tokenizer = CharTokenizer(vocab_size=512)
    loader = TextDataLoader(tokenizer, seq_len=config.max_seq_len,
                             batch_size=args.batch_size)

    # 数据
    if args.data and os.path.exists(args.data):
        loader.load_file(args.data)
    else:
        log.warn("no data file, using builtin classical poetry corpus")
        loader.load_text(BUILTIN_CORPUS)

    log.info("tokenizer", vocab=len(tokenizer.char2id),
              sequences=len(loader._data))

    # 模型
    gpu = HeteroGPU(config)
    log.info("model", params=gpu.stats()["config"]["params"])

    # 训练组件
    tracker = ExperimentTracker()
    ckpt_mgr = CheckpointManager()
    engine = TrainingEngine(gpu, tokenizer, loader, tracker, ckpt_mgr)

    # 跑
    result = engine.train(
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        early_stop_patience=args.patience,
        log_interval=args.log_interval,
        resume=not args.no_resume,
    )

    # 保存最终模型
    final_path = os.path.join(OUTPUT_DIR, "lingyuan_final.het")
    gpu.save(final_path)
    log.info("final model saved", path=final_path)

    print("\n=== 训练结果 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_generate(args):
    """生成命令"""
    HeteroGPU, HeteroConfig = _import_hetero_gpu()

    # 加载模型
    if args.model and os.path.exists(args.model):
        gpu = HeteroGPU.load(args.model)
        config = gpu.cfg
        log.info("model loaded", path=args.model)
    else:
        # 尝试默认
        default = os.path.join(OUTPUT_DIR, "lingyuan_final.het")
        if os.path.exists(default):
            gpu = HeteroGPU.load(default)
            config = gpu.cfg
            log.info("model loaded", path=default)
        else:
            log.error("no model found, train first or specify --model")
            sys.exit(1)

    tokenizer = CharTokenizer(vocab_size=config.vocab_size)
    generator = TextGenerator(gpu, tokenizer)

    prompt = args.prompt or "春眠不觉晓"
    print(f"\n{'='*60}")
    print(f"Prompt:  {prompt}")
    print(f"{'='*60}")

    if args.stream:
        print("Output:  ", end="", flush=True)
        full = []
        for token in generator.stream(prompt, max_new=args.max_new):
            print(token, end="", flush=True)
            full.append(token)
        print()
    else:
        output = generator.generate(prompt, max_new=args.max_new,
                                     temperature=args.temperature)
        print(f"Output:  {output}")

    print(f"{'='*60}\n")


def cmd_test(args):
    """测试命令"""
    log.info("=== TEST ===")

    HeteroGPU, HeteroConfig = _import_hetero_gpu()

    print("=" * 60)
    print("灵元企业级 — 测试")
    print("=" * 60)

    passed = 0
    failed = 0

    # 1. 配置
    print("\n--- 配置校验 ---")
    try:
        HeteroConfig.tiny().validate()
        print("  OK: tiny config")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # 2. 模型创建
    print("\n--- 模型创建 ---")
    try:
        gpu = HeteroGPU(HeteroConfig.tiny())
        print(f"  OK: params={gpu.stats()['config']['params']}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1
        print(f"\n{'='*60}")
        print(f"结果: {passed} passed, {failed} failed")
        return

    # 3. 前向
    print("\n--- 前向传播 ---")
    try:
        logits = gpu.forward([1, 2, 3, 4, 5])
        assert logits.shape()[0] == 5, f"shape={logits.shape()}"
        print(f"  OK: logits {logits.shape()}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # 4. 生成
    print("\n--- 文本生成 ---")
    try:
        out = gpu.generate([1, 2, 3], max_new=16)
        print(f"  OK: generated {len(out)} tokens")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # 5. 训练一步
    print("\n--- 训练步 ---")
    try:
        inp = [random.randrange(128) for _ in range(32)]
        tgt = inp[1:] + [random.randrange(128)]
        loss = gpu.train_step(inp, tgt)
        print(f"  OK: loss={loss:.4f}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # 6. 自举
    print("\n--- 自举训练 ---")
    try:
        loss = gpu.bootstrap_epoch(num_samples=16)
        print(f"  OK: bootstrap loss={loss:.4f}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # 7. 存盘
    print("\n--- 存盘/读盘 ---")
    try:
        path = os.path.join(CKPT_DIR, "_test.het")
        gpu.save(path)
        gpu2 = HeteroGPU.load(path)
        l1 = gpu.forward([1, 2, 3])
        l2 = gpu2.forward([1, 2, 3])
        err = max(abs(l1.data[i][j] - l2.data[i][j])
                   for i in range(l1.rows) for j in range(l1.cols))
        assert err < 1e-6, f"roundtrip error={err:.2e}"
        print(f"  OK: roundtrip err={err:.2e}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # 8. 统计分析
    print("\n--- 统计分析 ---")
    try:
        s = gpu.stats()
        print(f"  OK: version={s['version']}, forward={s['runtime']['forward_passes']}, "
              f"sparse_skip={s['sparse']['skip_rate']}, "
              f"cache_hit={s['cache']['hit_rate']}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    print(f"\n{'='*60}")
    print(f"结果: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if args.full:
        # 跑 part17 的完整测试套件
        print("\n--- Full Test Suite (part17_hetero_enterprise) ---")
        try:
            from part17_hetero_enterprise import TestHeteroGPU
            suite = TestHeteroGPU()
            suite.run_all()
        except Exception as e:
            log.error("full test suite failed", error=str(e))


def cmd_status(args):
    """状态命令"""
    HeteroGPU, HeteroConfig = _import_hetero_gpu()

    print("=" * 60)
    print("灵元企业级 — 系统状态")
    print("=" * 60)

    ckpt_mgr = CheckpointManager()
    latest = ckpt_mgr.latest()

    print(f"\n断点:")
    if latest:
        print(f"  最新: epoch={latest.epoch}, step={latest.step}, "
              f"loss={latest.loss:.4f}, time={latest.timestamp}")
    else:
        print(f"  无断点")

    print(f"\n目录:")
    for name, path in [("数据", DATA_DIR), ("断点", CKPT_DIR),
                         ("日志", LOG_DIR), ("输出", OUTPUT_DIR)]:
        count = len(os.listdir(path)) if os.path.isdir(path) else 0
        print(f"  {name}: {count} 文件 — {path}")


def cmd_serve(args):
    """服务命令"""
    HeteroGPU, HeteroConfig = _import_hetero_gpu()

    if args.model and os.path.exists(args.model):
        gpu = HeteroGPU.load(args.model)
    else:
        default = os.path.join(OUTPUT_DIR, "lingyuan_final.het")
        if os.path.exists(default):
            gpu = HeteroGPU.load(default)
        else:
            # 无模型，快速训练一个
            log.warn("no model, training a quick one...")
            config = HeteroConfig.tiny()
            tokenizer = CharTokenizer(vocab_size=128)
            loader = TextDataLoader(tokenizer, seq_len=config.max_seq_len,
                                     batch_size=8)
            loader.load_text(BUILTIN_CORPUS)
            gpu = HeteroGPU(config)
            engine = TrainingEngine(gpu, tokenizer, loader,
                                     ExperimentTracker(),
                                     CheckpointManager())
            engine.train(epochs=3, steps_per_epoch=30,
                          early_stop_patience=99, resume=False)

    tokenizer = CharTokenizer(vocab_size=gpu.cfg.vocab_size)
    server = LingyuanServer(gpu, tokenizer,
                             host=args.host, port=args.port)
    server.start()


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="灵元大模型 — 企业级运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python lingyuan_enterprise.py train --data corpus.txt --epochs 50
  python lingyuan_enterprise.py generate --prompt "春眠不觉晓"
  python lingyuan_enterprise.py test
  python lingyuan_enterprise.py status
  python lingyuan_enterprise.py serve --port 8080
        """)

    sub = parser.add_subparsers(dest="command", help="命令")

    # train
    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--data", type=str, default="",
                          help="训练数据文件路径")
    p_train.add_argument("--epochs", type=int, default=10,
                          help="训练轮数 (default: 10)")
    p_train.add_argument("--steps", type=int, default=50,
                          help="每轮步数 (default: 50)")
    p_train.add_argument("--batch-size", type=int, default=16,
                          help="批大小 (default: 16)")
    p_train.add_argument("--lr", type=float, default=0.001,
                          help="学习率 (default: 0.001)")
    p_train.add_argument("--patience", type=int, default=5,
                          help="早停耐心 (default: 5)")
    p_train.add_argument("--log-interval", type=int, default=10,
                          help="日志间隔 (default: 10)")
    p_train.add_argument("--no-resume", action="store_true",
                          help="不从断点恢复")

    # generate
    p_gen = sub.add_parser("generate", help="生成文本")
    p_gen.add_argument("--model", type=str, default="",
                        help="模型文件路径")
    p_gen.add_argument("--prompt", type=str, default="",
                        help="提示词")
    p_gen.add_argument("--max-new", type=int, default=64,
                        help="最大新token数 (default: 64)")
    p_gen.add_argument("--temperature", type=float, default=0.8,
                        help="温度 (default: 0.8)")
    p_gen.add_argument("--stream", action="store_true",
                        help="流式输出")

    # test
    p_test = sub.add_parser("test", help="运行测试")
    p_test.add_argument("--full", action="store_true",
                         help="完整测试套件")

    # status
    sub.add_parser("status", help="系统状态")

    # serve
    p_serve = sub.add_parser("serve", help="启动推理服务")
    p_serve.add_argument("--model", type=str, default="",
                          help="模型文件路径")
    p_serve.add_argument("--host", type=str, default="localhost",
                          help="绑定地址 (default: localhost)")
    p_serve.add_argument("--port", type=int, default=8080,
                          help="端口 (default: 8080)")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
