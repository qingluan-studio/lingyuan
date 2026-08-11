#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灵元代码模型 — V7.0 ULTRA Code Edition
基于 DeepNorm + GQA + MoE + RoPE/ALiBi 架构
训练数据: HumanEval + MBPP + 内置多语言代码语料
零外部依赖 · 纯Python标准库
"""

import os, sys, json, time, math, random, argparse
from collections import Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# 导入V7核心架构
sys.path.insert(0, "/workspace/lingyuan_train")
from lingyuan_v7 import (
    Tensor, glorot, CharTokenizer, TextDataLoader,
    ModelConfig, LingyuanModel, Trainer
)

# ============================================================
# 代码Tokenizer — 增强字符级，支持代码特殊字符
# ============================================================

class CodeTokenizer(CharTokenizer):
    """代码专用Tokenizer — 字符级，保留缩进和特殊符号"""

    def fit_on_code(self, texts: List[str]):
        """在代码数据上训练词表"""
        all_text = "\n".join(texts)
        freq = Counter(all_text)
        # 特殊token已初始化
        idx = len(self.char2id)
        # 按频率排序，保留所有代码字符
        sorted_chars = sorted(freq.keys(), key=lambda c: -freq[c])
        for ch in sorted_chars:
            if ch not in self.char2id and idx < self.vocab_size:
                self.char2id[ch] = idx
                self.id2char[idx] = ch
                idx += 1
        return self


# ============================================================
# 代码数据加载器
# ============================================================

class CodeDataLoader:
    """代码数据加载器 — 支持HumanEval/MBPP格式"""

    def __init__(self, tokenizer: CodeTokenizer, seq_len: int = 96, batch_size: int = 2):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self._sequences: List[List[int]] = []
        self._code_samples: List[Dict] = []

    def load_humaneval(self, path: str):
        """加载HumanEval数据集"""
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    prompt = item.get("prompt", "")
                    solution = item.get("canonical_solution", "")
                    full_code = prompt + solution
                    if len(full_code) > 10:
                        self._code_samples.append({
                            "type": "humaneval",
                            "task_id": item.get("task_id", ""),
                            "prompt": prompt,
                            "code": full_code,
                            "language": "python",
                        })
                        count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  HumanEval: {count} 个代码样本")
        return count

    def load_mbpp(self, path: str):
        """加载MBPP数据集"""
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    code = item.get("code", "")
                    if len(code) > 10:
                        self._code_samples.append({
                            "type": "mbpp",
                            "task_id": str(item.get("task_id", "")),
                            "prompt": text,
                            "code": code,
                            "language": "python",
                        })
                        count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  MBPP: {count} 个代码样本")
        return count

    def load_builtin_corpus(self):
        """加载内置多语言代码语料"""
        for i, code in enumerate(BUILTIN_CODE_CORPUS):
            self._code_samples.append({
                "type": "builtin",
                "task_id": f"builtin_{i}",
                "prompt": "",
                "code": code["code"],
                "language": code["lang"],
                "desc": code.get("desc", ""),
            })
        print(f"  内置语料: {len(BUILTIN_CODE_CORPUS)} 个代码样本")

    def prepare(self):
        """准备训练序列 — 将代码样本编码为训练序列"""
        all_texts = [s["code"] for s in self._code_samples]
        self.tokenizer.fit_on_code(all_texts)

        for sample in self._code_samples:
            code = sample["code"]
            ids = self.tokenizer.encode(code)
            # 滑动窗口生成训练序列
            step = max(1, self.seq_len // 2)
            for i in range(0, len(ids) - self.seq_len - 1, step):
                self._sequences.append(ids[i:i + self.seq_len + 1])
            # 如果代码短于seq_len，也保留
            if len(ids) <= self.seq_len:
                padded = ids + [0] * (self.seq_len + 1 - len(ids))
                self._sequences.append(padded)

        random.shuffle(self._sequences)
        print(f"  训练序列: {len(self._sequences)} 条 (seq_len={self.seq_len})")

    def sample_batch(self) -> Tuple[List[List[int]], List[List[int]]]:
        if not self._sequences:
            return self._synthetic()
        inputs, targets = [], []
        for _ in range(self.batch_size):
            seq = random.choice(self._sequences)
            inputs.append(seq[:self.seq_len])
            targets.append(seq[1:self.seq_len + 1])
        return inputs, targets

    def _synthetic(self) -> Tuple[List[List[int]], List[List[int]]]:
        inputs, targets = [], []
        for _ in range(self.batch_size):
            seq = [random.randint(0, self.tokenizer.vocab_size - 1) for _ in range(self.seq_len)]
            inputs.append(seq[:-1])
            targets.append(seq[1:])
        return inputs, targets


# ============================================================
# 内置多语言代码语料
# ============================================================

BUILTIN_CODE_CORPUS = [
    # === Python 算法 ===
    {"lang": "python", "desc": "二分查找", "code": """def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
"""},
    {"lang": "python", "desc": "快速排序", "code": """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""},
    {"lang": "python", "desc": "归并排序", "code": """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""},
    {"lang": "python", "desc": "BFS广度优先搜索", "code": """from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result
"""},
    {"lang": "python", "desc": "DFS深度优先搜索", "code": """def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]
    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    return result
"""},
    {"lang": "python", "desc": "Dijkstra最短路径", "code": """import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    while heap:
        current_dist, current = heapq.heappop(heap)
        if current_dist > distances[current]:
            continue
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    return distances
"""},
    {"lang": "python", "desc": "动态规划-背包问题", "code": """def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]
"""},
    {"lang": "python", "desc": "最长公共子序列", "code": """def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
"""},
    {"lang": "python", "desc": "链表反转", "code": """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev
"""},
    {"lang": "python", "desc": "二叉树遍历", "code": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    result = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        result.append(node.val)
        dfs(node.right)
    dfs(root)
    return result
"""},
    # === JavaScript ===
    {"lang": "javascript", "desc": "JS二分查找", "code": """function binarySearch(arr, target) {
    let low = 0, high = arr.length - 1;
    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        if (arr[mid] === target) return mid;
        if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}
"""},
    {"lang": "javascript", "desc": "JS快速排序", "code": """function quicksort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const middle = arr.filter(x => x === pivot);
    const right = arr.filter(x => x > pivot);
    return quicksort(left).concat(middle).concat(quicksort(right));
}
"""},
    {"lang": "javascript", "desc": "JS防抖函数", "code": """function debounce(fn, delay) {
    let timer = null;
    return function(...args) {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => {
            fn.apply(this, args);
        }, delay);
    };
}
"""},
    {"lang": "javascript", "desc": "JS深拷贝", "code": """function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') return obj;
    if (obj instanceof Date) return new Date(obj);
    if (obj instanceof Array) {
        return obj.map(item => deepClone(item));
    }
    const cloned = {};
    for (let key in obj) {
        if (obj.hasOwnProperty(key)) {
            cloned[key] = deepClone(obj[key]);
        }
    }
    return cloned;
}
"""},
    # === C语言 ===
    {"lang": "c", "desc": "C二分查找", "code": """int binary_search(int arr[], int n, int target) {
    int low = 0, high = n - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}
"""},
    {"lang": "c", "desc": "C链表节点", "code": """struct Node {
    int data;
    struct Node* next;
};

struct Node* create_node(int data) {
    struct Node* node = (struct Node*)malloc(sizeof(struct Node));
    node->data = data;
    node->next = NULL;
    return node;
}

void insert_at_head(struct Node** head, int data) {
    struct Node* node = create_node(data);
    node->next = *head;
    *head = node;
}
"""},
    {"lang": "c", "desc": "C字符串反转", "code": """void reverse_string(char* str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = temp;
    }
}
"""},
    # === Java ===
    {"lang": "java", "desc": "Java二分查找", "code": """public class BinarySearch {
    public static int search(int[] arr, int target) {
        int low = 0, high = arr.length - 1;
        while (low <= high) {
            int mid = (low + high) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }
}
"""},
    {"lang": "java", "desc": "Java单例模式", "code": """public class Singleton {
    private static Singleton instance;
    private Singleton() {}

    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
"""},
    # === 更多Python实用函数 ===
    {"lang": "python", "desc": "FizzBuzz", "code": """def fizzbuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
"""},
    {"lang": "python", "desc": "斐波那契数列", "code": """def fibonacci(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)
"""},
    {"lang": "python", "desc": "素数判断", "code": """def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def sieve_of_eratosthenes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]
"""},
    {"lang": "python", "desc": "字符串反转", "code": """def reverse_string(s):
    return s[::-1]

def reverse_words(sentence):
    words = sentence.split()
    return ' '.join(reversed(words))

def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]
"""},
    {"lang": "python", "desc": "LRU缓存", "code": """from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
"""},
    {"lang": "python", "desc": "JSON解析器", "code": """import json

def parse_json(text):
    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None

def to_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)
"""},
    {"lang": "python", "desc": "正则匹配", "code": """import re

def find_emails(text):
    pattern = r'[\w.+-]+@[\w-]+\.[\w.-]+'
    return re.findall(pattern, text)

def validate_phone(phone):
    pattern = r'^1[3-9]\d{9}$'
    return bool(re.match(pattern, phone))
"""},
    {"lang": "python", "desc": "装饰器", "code": """import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} 执行耗时: {elapsed:.4f}s")
        return result
    return wrapper

def retry(max_attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"重试 {attempt + 1}/{max_attempts}: {e}")
        return wrapper
    return decorator
"""},
    {"lang": "python", "desc": "HTTP请求", "code": """import urllib.request
import json

def http_get(url):
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8')
        return json.loads(data)

def http_post(url, data):
    body = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))
"""},
    {"lang": "python", "desc": "文件操作", "code": """import os

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def list_files(directory, ext=None):
    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            if ext is None or name.endswith(ext):
                files.append(path)
    return files
"""},
    {"lang": "python", "desc": "矩阵运算", "code": """def matrix_multiply(a, b):
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result

def matrix_transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
"""},
    {"lang": "python", "desc": "图遍历", "code": """class Graph:
    def __init__(self):
        self.adj = {}

    def add_edge(self, u, v):
        if u not in self.adj:
            self.adj[u] = []
        if v not in self.adj:
            self.adj[v] = []
        self.adj[u].append(v)
        self.adj[v].append(u)

    def get_neighbors(self, node):
        return self.adj.get(node, [])
"""},
    {"lang": "python", "desc": "堆栈实现", "code": """class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
"""},
    {"lang": "python", "desc": "队列实现", "code": """from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("dequeue from empty queue")

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
"""},
    {"lang": "python", "desc": "二叉搜索树", "code": """class BSTNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = BSTNode(val)
            return
        self._insert(self.root, val)

    def _insert(self, node, val):
        if val < node.val:
            if node.left:
                self._insert(node.left, val)
            else:
                node.left = BSTNode(val)
        else:
            if node.right:
                self._insert(node.right, val)
            else:
                node.right = BSTNode(val)

    def search(self, val):
        return self._search(self.root, val)

    def _search(self, node, val):
        if not node:
            return False
        if node.val == val:
            return True
        if val < node.val:
            return self._search(node.left, val)
        return self._search(node.right, val)
"""},
]


# ============================================================
# 代码训练器
# ============================================================

class CodeTrainer:
    """代码训练引擎"""

    def __init__(self, model: LingyuanModel, tokenizer: CodeTokenizer,
                 loader: CodeDataLoader):
        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.history = []
        self.best_loss = float('inf')
        self.no_improve = 0

    def _compute_lr(self, step, total_steps, base_lr):
        """warmup + cosine decay"""
        warmup = max(1, total_steps // 20)
        if step < warmup:
            return base_lr * step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def train(self, epochs=10, steps_per_epoch=50, base_lr=0.001,
              patience=8, log_interval=5) -> dict:
        total_steps = epochs * steps_per_epoch
        current_step = 0
        t0 = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start = time.time()

            for step in range(steps_per_epoch):
                current_step += 1
                lr = self._compute_lr(current_step - 1, total_steps, base_lr)

                inputs, targets = self.loader.sample_batch()
                batch_loss = 0.0
                for inp, tgt in zip(inputs, targets):
                    loss = self.model.train_step(inp, tgt, lr)
                    batch_loss += loss

                avg_loss = batch_loss / max(len(inputs), 1)
                epoch_loss += avg_loss

                if step % log_interval == 0 or step == steps_per_epoch - 1:
                    elapsed = time.time() - epoch_start
                    print(f"  [E{epoch+1} S{step+1}] loss={avg_loss:.4f} "
                          f"lr={lr:.6f} t={elapsed:.0f}s", flush=True)

            avg_epoch = epoch_loss / steps_per_epoch
            epoch_time = time.time() - epoch_start
            self.history.append({
                "epoch": epoch + 1, "loss": avg_epoch, "time": epoch_time
            })

            print(f"  >> Epoch {epoch+1}: loss={avg_epoch:.4f} "
                  f"time={epoch_time:.0f}s best={self.best_loss:.4f}", flush=True)

            if avg_epoch < self.best_loss - 1e-6:
                self.best_loss = avg_epoch
                self.no_improve = 0
            else:
                self.no_improve += 1
                if self.no_improve >= patience:
                    print(f"  !! 早停: {patience}轮无改善", flush=True)
                    break

        total_time = time.time() - t0
        return {
            "epochs": len(self.history),
            "steps": current_step,
            "time": f"{total_time:.0f}s",
            "best_loss": round(self.best_loss, 4),
            "history": self.history,
        }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="灵元代码模型 V7.0 ULTRA Code")
    parser.add_argument("--humaneval", type=str, default="/workspace/lingyuan_code/data/humaneval.jsonl")
    parser.add_argument("--mbpp", type=str, default="/workspace/lingyuan_code/data/mbpp.jsonl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--model", type=str, default="", help="加载已有模型续训")
    parser.add_argument("--output", type=str, default="lingyuan_code_v7.het")
    args = parser.parse_args()

    output_dir = "/workspace/lingyuan_code"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  灵元代码模型 — V7.0 ULTRA Code Edition")
    print("  DeepNorm + GQA + MoE + RoPE/ALiBi")
    print("  训练数据: HumanEval + MBPP + 内置多语言代码语料")
    print("=" * 60)

    # 配置 — 代码需要更长序列
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_dim=64,
        num_heads=4,
        num_kv_heads=1,
        num_layers=4,
        ffn_dim=256,
        max_seq_len=args.seq_len,
    )
    cfg.learning_rate = args.lr

    # Tokenizer
    tokenizer = CodeTokenizer(vocab_size=cfg.vocab_size)

    # 数据加载
    loader = CodeDataLoader(tokenizer, seq_len=cfg.max_seq_len, batch_size=args.batch_size)

    print("\n加载数据:")
    if os.path.exists(args.humaneval):
        loader.load_humaneval(args.humaneval)
    else:
        print(f"  HumanEval 文件不存在: {args.humaneval}")

    if os.path.exists(args.mbpp):
        loader.load_mbpp(args.mbpp)
    else:
        print(f"  MBPP 文件不存在: {args.mbpp}")

    loader.load_builtin_corpus()

    print("\n准备训练数据:")
    loader.prepare()
    print(f"  词表: {tokenizer.actual_size} | 代码样本: {len(loader._code_samples)}")

    # 模型
    if args.model and os.path.exists(args.model):
        model = LingyuanModel.load(args.model)
        print(f"\n模型加载: {args.model} (续训)")
    else:
        model = LingyuanModel(cfg)
        print("\n新建模型")

    stats = model.stats()
    print(f"  版本: {stats['version']}")
    print(f"  参数: {stats['total_params']} (基础: {stats['params']}, MoE: {stats['moe_params']})")
    print(f"  架构: {stats['config']}")

    # 训练
    print(f"\n开始训练: {args.epochs}轮 × {args.steps}步/轮")
    trainer = CodeTrainer(model, tokenizer, loader)
    result = trainer.train(
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        base_lr=args.lr,
        patience=args.patience,
        log_interval=args.log_interval,
    )

    # 保存模型
    model_path = os.path.join(output_dir, args.output)
    model.save(model_path)
    print(f"\n模型保存: {model_path}")

    # 保存训练报告
    report_path = os.path.join(output_dir, "code_training_report.json")
    report = {
        "version": stats["version"],
        "edition": "code_v7_ultra",
        "config": stats["config"],
        "params": stats["total_params"],
        "data_sources": {
            "humaneval": sum(1 for s in loader._code_samples if s["type"] == "humaneval"),
            "mbpp": sum(1 for s in loader._code_samples if s["type"] == "mbpp"),
            "builtin": sum(1 for s in loader._code_samples if s["type"] == "builtin"),
        },
        "vocab_size": tokenizer.actual_size,
        "num_sequences": len(loader._sequences),
        "training": result,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告保存: {report_path}")

    # 代码生成测试
    print("\n" + "=" * 60)
    print("  代码生成测试")
    print("=" * 60)

    test_prompts = [
        "def fibonacci(n):\n    ",
        "def binary_search(arr, target):\n    ",
        "def is_prime(n):\n    ",
        "def quicksort(arr):\n    ",
    ]

    generations = []
    for prompt in test_prompts:
        prompt_ids = tokenizer.encode(prompt)
        if prompt_ids:
            gen = model.generate(prompt_ids, max_new=48, temperature=0.7, top_k=15)
            text = tokenizer.decode(gen)
            print(f"\n  Prompt: {prompt.strip()}")
            print(f"  生成: {text[len(prompt):][:120]}")
            generations.append({"prompt": prompt, "generated": text})

    # 保存生成结果
    gen_path = os.path.join(output_dir, "code_generations.json")
    with open(gen_path, 'w', encoding='utf-8') as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)
    print(f"\n生成结果: {gen_path}")

    print(f"\n{'=' * 60}")
    print(f"训练完成: {result['epochs']}轮 {result['steps']}步")
    print(f"用时: {result['time']} | 最优loss: {result['best_loss']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
