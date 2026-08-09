
# ============================================================
# LINGYUAN MODEL - PART 27
# 安全沙箱与对抗防御系统 (Security Sandbox & Adversarial Defense)
#
# 模型安全防护全栈: 从输入到输出的多层防御
#
# 核心组件:
# - 安全沙箱: 隔离执行不可信代码, 资源限制
# - 输入净化: 提示注入检测, 敏感信息过滤
# - 对抗防御: 对抗样本检测, 输出净化
# - 模型水印: 版权保护, 所有权验证
# - 审计日志: 全链路行为追踪
# - 速率限制: 防止滥用, 公平性保障
# - 异常检测: 行为异常识别, 自动熔断
# ============================================================

import os
import re
import sys
import time
import json
import math
import random
import hashlib
import threading
import traceback
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union


# ============================================================
# 枚举定义
# ============================================================

class ThreatLevel(Enum):
    """威胁等级"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AttackType(Enum):
    """攻击类型"""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    DATA_EXFILTRATION = "data_exfiltration"
    ADVERSARIAL_INPUT = "adversarial_input"
    MODEL_EXTRACTION = "model_extraction"
    DENIAL_OF_SERVICE = "denial_of_service"
    CODE_INJECTION = "code_injection"
    SOCIAL_ENGINEERING = "social_engineering"
    UNKNOWN = "unknown"


class SandboxStatus(Enum):
    """沙箱状态"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"
    BLOCKED = "blocked"


class DefenseAction(Enum):
    """防御动作"""
    ALLOW = "allow"
    SANITIZE = "sanitize"
    BLOCK = "block"
    CHALLENGE = "challenge"  # 要求验证
    LOG_ONLY = "log_only"
    QUARANTINE = "quarantine"


# ============================================================
# 安全事件
# ============================================================

@dataclass
class SecurityEvent:
    """安全事件"""
    event_id: str
    timestamp: float
    event_type: AttackType
    threat_level: ThreatLevel
    source: str  # 来源 (IP, user_id, etc.)
    description: str
    input_data: str = ""
    action_taken: DefenseAction = DefenseAction.LOG_ONLY
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 输入净化器
# ============================================================

class InputSanitizer:
    """输入净化器

    多层过滤:
    1. 提示注入检测: 模式匹配 + 语义分析
    2. 敏感信息过滤: PII, 密钥, 令牌
    3. 恶意内容检测: 恶意URL, XSS, 代码注入
    4. 编码攻击防御: Unicode, Base64, Hex混淆
    """

    # 提示注入模式
    INJECTION_PATTERNS = [
        # 直接指令覆盖
        (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", "指令覆盖"),
        (r"disregard\s+(all\s+)?(previous|prior).*(instruction|prompt)", "指令忽略"),
        (r"forget\s+(everything|all).*(before|prior)", "记忆擦除"),
        # 角色劫持
        (r"you\s+are\s+(now|actually)\s+(a|an)\s+", "角色劫持"),
        (r"pretend\s+(you\s+are|to\s+be)\s+", "角色伪装"),
        (r"act\s+as\s+(if\s+you\s+are\s+)?(a|an)?\s*", "角色扮演注入"),
        # 系统提示泄露
        (r"(show|reveal|print|display|output).*(system|initial|original).*(prompt|instruction)", "系统提示泄露"),
        (r"what.*(are|is).*(your|the).*(system|hidden).*(prompt|instruction|rule)", "系统提示探测"),
        # 越狱模式
        (r"(DAN|do anything now|developer mode|jailbreak)", "越狱关键词"),
        (r"(enable|activate|turn on).*(developer|god|admin|unlimited).*(mode|mode)", "模式激活"),
        # 数据外泄
        (r"(repeat|copy|echo).*(system|hidden|internal).*(prompt|instruction|rule|message)", "数据外泄"),
        # 格式操纵
        (r"\[SYSTEM\]|\[ADMIN\]|\[INST\]|\[/INST\]", "格式伪造"),
        (r"<<(SYS|SYSTEM)>>|<<(SYS|SYSTEM)>>", "系统标记伪造"),
    ]

    # 敏感信息模式
    PII_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),
        (r'\b\d{16,19}\b', "信用卡号"),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "邮箱"),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "电话号码"),
        (r'\b(?:sk-|pk-|Bearer\s)[A-Za-z0-9]{20,}\b', "API密钥"),
        (r'\b[A-Fa-f0-9]{32,64}\b', "哈希值/密钥"),
    ]

    # 恶意URL模式
    MALICIOUS_URL_PATTERNS = [
        (r'https?://[^\s]+\.(tk|ml|ga|cf|gq)[^\s]*', "可疑域名"),
        (r'https?://(?:bit\.ly|tinyurl|t\.co|short\.url)[^\s]*', "短链接"),
        (r'(?:javascript|data):[^;\s]*', "脚本URL"),
    ]

    def __init__(self):
        self.compiled_injection = [(re.compile(p, re.IGNORECASE), desc)
                                   for p, desc in self.INJECTION_PATTERNS]
        self.compiled_pii = [(re.compile(p), desc) for p, desc in self.PII_PATTERNS]
        self.compiled_urls = [(re.compile(p, re.IGNORECASE), desc)
                              for p, desc in self.MALICIOUS_URL_PATTERNS]
        self.blocked_count = 0
        self.sanitized_count = 0

    def scan(self, text: str) -> Dict[str, Any]:
        """扫描输入文本

        Returns:
            {
                threat_level: 威胁等级,
                attack_type: 攻击类型,
                patterns_matched: 匹配的模式,
                sanitized_text: 净化后的文本,
                action: 建议动作
            }
        """
        threats = []
        sanitized = text

        # 1. 检测提示注入
        injection_matches = []
        for pattern, desc in self.compiled_injection:
            matches = pattern.findall(text)
            if matches:
                injection_matches.append({
                    "pattern": desc,
                    "count": len(matches),
                    "samples": matches[:3],
                })
                sanitized = pattern.sub("[REDACTED]", sanitized)

        if injection_matches:
            threats.append({
                "type": AttackType.PROMPT_INJECTION,
                "level": ThreatLevel.HIGH,
                "matches": injection_matches,
            })

        # 2. 检测敏感信息
        pii_matches = []
        for pattern, desc in self.compiled_pii:
            matches = pattern.findall(text)
            if matches:
                pii_matches.append({
                    "type": desc,
                    "count": len(matches),
                })
                sanitized = pattern.sub(f"[{desc}_REDACTED]", sanitized)

        if pii_matches:
            threats.append({
                "type": AttackType.DATA_EXFILTRATION,
                "level": ThreatLevel.MEDIUM,
                "matches": pii_matches,
            })

        # 3. 检测恶意URL
        url_matches = []
        for pattern, desc in self.compiled_urls:
            matches = pattern.findall(text)
            if matches:
                url_matches.append({
                    "type": desc,
                    "urls": matches[:3],
                })
                sanitized = pattern.sub("[URL_BLOCKED]", sanitized)

        if url_matches:
            threats.append({
                "type": AttackType.ADVERSARIAL_INPUT,
                "level": ThreatLevel.MEDIUM,
                "matches": url_matches,
            })

        # 4. 编码混淆检测
        encoding_threats = self._detect_encoding_attacks(text)
        if encoding_threats:
            threats.extend(encoding_threats)

        # 确定总体威胁等级和建议动作
        max_level = ThreatLevel.SAFE
        attack_type = AttackType.UNKNOWN
        for t in threats:
            if t["level"].value > max_level.value:
                max_level = t["level"]
                attack_type = t["type"]

        if max_level.value >= ThreatLevel.HIGH.value:
            action = DefenseAction.BLOCK
            self.blocked_count += 1
        elif max_level.value >= ThreatLevel.MEDIUM.value:
            action = DefenseAction.SANITIZE
            self.sanitized_count += 1
        elif max_level.value >= ThreatLevel.LOW.value:
            action = DefenseAction.LOG_ONLY
        else:
            action = DefenseAction.ALLOW

        return {
            "threat_level": max_level.name,
            "attack_type": attack_type.value,
            "threats": threats,
            "sanitized_text": sanitized if action != DefenseAction.ALLOW else text,
            "action": action.value,
            "was_sanitized": sanitized != text,
        }

    def _detect_encoding_attacks(self, text: str) -> List[Dict[str, Any]]:
        """检测编码混淆攻击"""
        threats = []

        # Unicode零宽字符
        zero_width = ['\u200b', '\u200c', '\u200d', '\u2060', '\ufeff']
        zw_count = sum(text.count(c) for c in zero_width)
        if zw_count > 5:
            threats.append({
                "type": AttackType.ADVERSARIAL_INPUT,
                "level": ThreatLevel.MEDIUM,
                "matches": [{"type": "零宽字符混淆", "count": zw_count}],
            })

        # Base64编码的长字符串 (可能是混淆的指令)
        b64_pattern = re.compile(r'[A-Za-z0-9+/]{40,}={0,2}')
        b64_matches = b64_pattern.findall(text)
        if b64_matches:
            import base64
            for match in b64_matches[:3]:
                try:
                    decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                    # 检查解码后是否包含可疑关键词
                    suspicious = any(kw in decoded.lower() for kw in
                                     ['ignore', 'system', 'instruction', 'prompt', 'admin'])
                    if suspicious:
                        threats.append({
                            "type": AttackType.PROMPT_INJECTION,
                            "level": ThreatLevel.HIGH,
                            "matches": [{"type": "Base64混淆注入", "decoded": decoded[:100]}],
                        })
                        break
                except Exception:
                    pass

        # Hex编码检测
        hex_pattern = re.compile(r'\\x[0-9a-fA-F]{2}')
        hex_count = len(hex_pattern.findall(text))
        if hex_count > 10:
            threats.append({
                "type": AttackType.ADVERSARIAL_INPUT,
                "level": ThreatLevel.LOW,
                "matches": [{"type": "Hex编码混淆", "count": hex_count}],
            })

        return threats


# ============================================================
# 安全沙箱
# ============================================================

class SecuritySandbox:
    """安全沙箱 — 隔离执行不可信代码

    多层隔离:
    1. 模块限制: 禁用危险模块 (os, sys, subprocess等)
    2. 内建限制: 过滤危险内置函数
    3. 资源限制: CPU时间, 内存, 输出大小
    4. 超时控制: 防止死循环
    5. 文件系统隔离: 限制文件访问

    使用AST分析预检查 + 受限exec执行
    """

    # 禁止的模块
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'multiprocessing',
        'ctypes', 'socket', 'http', 'urllib', 'requests',
        'importlib', 'builtins', 'pickle', 'marshal',
        'webbrowser', 'antigravity', 'pty', 'commands',
    }

    # 禁止的内置函数
    BLOCKED_BUILTINS = {
        'eval', 'exec', 'compile', 'open', 'input',
        '__import__', 'globals', 'locals', 'vars',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'breakpoint', 'exit', 'quit', 'help',
    }

    # 允许的模块 (白名单)
    ALLOWED_MODULES = {
        'math', 'random', 'string', 're', 'json',
        'collections', 'itertools', 'functools',
        'datetime', 'decimal', 'fractions', 'statistics',
        'array', 'heapq', 'bisect', 'copy',
    }

    def __init__(self, timeout: float = 5.0,
                 max_memory_mb: int = 128,
                 max_output_chars: int = 10000,
                 max_cpu_ops: int = 10_000_000):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_output_chars = max_output_chars
        self.max_cpu_ops = max_cpu_ops
        self.status = SandboxStatus.IDLE
        self.execution_count = 0
        self.blocked_count = 0
        self._lock = threading.Lock()

    def execute(self, code: str,
                env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """在沙箱中执行代码

        Args:
            code: 要执行的Python代码
            env: 额外的环境变量

        Returns:
            {
                status: 执行状态,
                output: 标准输出,
                error: 错误信息,
                execution_time: 执行时间,
                result: 最后一行表达式的值
            }
        """
        self.execution_count += 1
        start_time = time.time()

        # 1. AST预检查
        ast_check = self._ast_check(code)
        if not ast_check["safe"]:
            self.blocked_count += 1
            self.status = SandboxStatus.BLOCKED
            return {
                "status": SandboxStatus.BLOCKED.value,
                "output": "",
                "error": f"代码被沙箱拦截: {ast_check['reason']}",
                "execution_time": time.time() - start_time,
                "result": None,
            }

        # 2. 构建受限环境
        safe_builtins = self._create_safe_builtins()
        safe_globals = {
            "__builtins__": safe_builtins,
            "__name__": "__sandbox__",
            "range": range,
            "len": len,
            "type": type,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "isinstance": isinstance,
        }

        # 添加允许的模块
        for mod_name in self.ALLOWED_MODULES:
            try:
                safe_globals[mod_name] = __import__(mod_name)
            except ImportError:
                pass

        # 添加用户环境
        if env:
            safe_globals.update(env)

        # 3. 执行代码 (带超时)
        self.status = SandboxStatus.RUNNING
        output_buffer = []
        result = {"output_list": output_buffer}

        def _safe_print(*args, **kwargs):
            output = " ".join(str(a) for a in args)
            output_buffer.append(output)
            if sum(len(s) for s in output_buffer) > self.max_output_chars:
                output_buffer.append("[输出超出限制, 已截断]")
                raise RuntimeError("输出超出限制")

        safe_globals["print"] = _safe_print

        try:
            # 使用线程执行, 设置超时
            exec_result = [None]
            exec_error = [None]

            def _run():
                try:
                    # 先执行全部代码
                    exec(code, safe_globals)
                    # 尝试获取最后一行表达式的值 (作为返回值)
                    lines = code.strip().split('\n')
                    if lines:
                        last_line = lines[-1].strip()
                        if last_line and not last_line.endswith(':') and \
                           not last_line.startswith(('#', ' ', '\t', 'def ', 'class ', 'if ', 'for ', 'while ', 'try', 'except', 'with ')):
                            try:
                                compile(last_line, '<sandbox>', 'eval')
                                exec_result[0] = eval(last_line, safe_globals)
                            except (SyntaxError, NameError):
                                pass  # 不是表达式或变量已消失, 忽略
                except Exception as e:
                    exec_error[0] = e

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            thread.join(timeout=self.timeout)

            if thread.is_alive():
                self.status = SandboxStatus.TIMEOUT
                return {
                    "status": SandboxStatus.TIMEOUT.value,
                    "output": "\n".join(output_buffer),
                    "error": f"执行超时 (>{self.timeout}s)",
                    "execution_time": time.time() - start_time,
                    "result": None,
                }

            if exec_error[0]:
                self.status = SandboxStatus.ERROR
                return {
                    "status": SandboxStatus.ERROR.value,
                    "output": "\n".join(output_buffer),
                    "error": str(exec_error[0]),
                    "execution_time": time.time() - start_time,
                    "result": None,
                }

            self.status = SandboxStatus.COMPLETED
            return {
                "status": SandboxStatus.COMPLETED.value,
                "output": "\n".join(output_buffer),
                "error": None,
                "execution_time": time.time() - start_time,
                "result": exec_result[0],
            }

        except Exception as e:
            self.status = SandboxStatus.ERROR
            return {
                "status": SandboxStatus.ERROR.value,
                "output": "\n".join(output_buffer),
                "error": f"沙箱执行错误: {str(e)}",
                "execution_time": time.time() - start_time,
                "result": None,
            }

    def _ast_check(self, code: str) -> Dict[str, Any]:
        """AST预检查代码安全性"""
        try:
            import ast
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"safe": False, "reason": f"语法错误: {e}"}

        for node in ast.walk(tree):
            # 检查import
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in self.BLOCKED_MODULES:
                        return {"safe": False,
                                "reason": f"禁止导入模块: {alias.name}"}
            # 检查from import
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in self.BLOCKED_MODULES:
                    return {"safe": False,
                            "reason": f"禁止导入模块: {node.module}"}
            # 检查属性访问 (如 __import__)
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith('_') and node.attr not in ('__init__', '__name__'):
                    return {"safe": False,
                            "reason": f"禁止访问私有属性: {node.attr}"}
            # 检查全局声明
            elif isinstance(node, ast.Global):
                return {"safe": False, "reason": "禁止使用global声明"}

        return {"safe": True, "reason": ""}

    def _create_safe_builtins(self) -> Dict[str, Any]:
        """创建安全的内置函数集"""
        safe = {}
        for name in dir(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__):
            if name not in self.BLOCKED_BUILTINS and not name.startswith('_'):
                try:
                    safe[name] = getattr(__builtins__, name) if not isinstance(__builtins__, dict) else __builtins__[name]
                except Exception:
                    pass
        return safe


# ============================================================
# 对抗防御器
# ============================================================

class AdversarialDefense:
    """对抗防御器

    检测和防御对抗性输入/输出

    防御策略:
    1. 文本扰动检测: 检测异常字符、同形字攻击
    2. 统计异常检测: 输入分布偏移检测
    3. 输出过滤: 检测有害输出
    4. 一致性检查: 多次采样一致性验证
    5. 困惑度检测: 异常低困惑度的输入可能是攻击
    """

    def __init__(self):
        self.input_history: deque = deque(maxlen=1000)
        self.output_history: deque = deque(maxlen=1000)
        self.blocked_count = 0
        self.flagged_count = 0

        # 有害输出模式
        self.harmful_patterns = [
            (r'(?i)(how\s+to|ways\s+to|methods?\s+to).*(bomb|weapon|explosive|poison|drug)', "武器制造"),
            (r'(?i)(how\s+to|ways\s+to).*(hack|crack|break\s+into|bypass).*(password|security|firewall)', "黑客攻击"),
            (r'(?i)(create|make|generate).*(malware|virus|trojan|ransomware)', "恶意软件"),
            (r'(?i)(steal|intercept|capture).*(password|credential|token|session)', "凭证窃取"),
        ]
        self.compiled_harmful = [(re.compile(p), d) for p, d in self.harmful_patterns]

        # 同形字映射 (常见攻击字符)
        self.homoglyphs = {
            'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c',
            'у': 'y', 'х': 'x', 'А': 'A', 'В': 'B', 'Е': 'E',
            'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P',
            'С': 'C', 'Т': 'T', 'Х': 'X', 'і': 'i', 'І': 'I',
        }

    def check_input(self, text: str) -> Dict[str, Any]:
        """检查输入是否有对抗性

        Returns:
            {
                is_adversarial: 是否对抗性,
                threat_level: 威胁等级,
                indicators: 检测到的异常指标,
                sanitized: 净化后的文本
            }
        """
        indicators = []
        sanitized = text
        max_level = ThreatLevel.SAFE

        # 1. 同形字检测
        homoglyph_count = sum(1 for c in text if c in self.homoglyphs)
        if homoglyph_count > 3:
            indicators.append({
                "type": "homoglyph_attack",
                "count": homoglyph_count,
                "description": f"检测到{homoglyph_count}个同形字, 可能是混淆攻击",
            })
            max_level = max(max_level, ThreatLevel.MEDIUM, key=lambda x: x.value)
            # 替换同形字
            sanitized = ''.join(self.homoglyphs.get(c, c) for c in sanitized)

        # 2. 异常字符检测
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
        if ascii_ratio < 0.7:
            indicators.append({
                "type": "low_ascii_ratio",
                "ratio": ascii_ratio,
                "description": f"ASCII字符占比仅{ascii_ratio:.1%}, 可能包含混淆字符",
            })
            max_level = max(max_level, ThreatLevel.LOW, key=lambda x: x.value)

        # 3. 重复模式检测 (可能的DoS)
        if len(text) > 100:
            # 检测大量重复
            char_freq = defaultdict(int)
            for c in text:
                char_freq[c] += 1
            max_freq = max(char_freq.values())
            if max_freq > len(text) * 0.5:
                indicators.append({
                    "type": "repetition_attack",
                    "max_char_freq": max_freq / len(text),
                    "description": "检测到大量字符重复, 可能是DoS攻击",
                })
                max_level = max(max_level, ThreatLevel.MEDIUM, key=lambda x: x.value)

        # 4. 统计异常检测 (与历史分布对比)
        if len(self.input_history) > 50:
            avg_len = sum(len(s) for s in self.input_history) / len(self.input_history)
            if len(text) > avg_len * 5:
                indicators.append({
                    "type": "length_anomaly",
                    "input_length": len(text),
                    "avg_length": avg_len,
                    "description": f"输入长度({len(text)})远超平均({avg_len:.0f})",
                })
                max_level = max(max_level, ThreatLevel.LOW, key=lambda x: x.value)

        # 5. 特殊字符密度
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,!?;:\'"()-\n')
        special_ratio = special_chars / max(1, len(text))
        if special_ratio > 0.3:
            indicators.append({
                "type": "high_special_chars",
                "ratio": special_ratio,
                "description": f"特殊字符占比{special_ratio:.1%}, 可能是注入攻击",
            })
            max_level = max(max_level, ThreatLevel.MEDIUM, key=lambda x: x.value)

        # 记录历史
        self.input_history.append(text)

        is_adversarial = max_level.value >= ThreatLevel.MEDIUM.value
        if is_adversarial:
            self.flagged_count += 1

        return {
            "is_adversarial": is_adversarial,
            "threat_level": max_level.name,
            "indicators": indicators,
            "sanitized": sanitized,
        }

    def check_output(self, text: str) -> Dict[str, Any]:
        """检查输出是否有害

        Returns:
            {
                is_harmful: 是否有害,
                threat_level: 威胁等级,
                matches: 匹配的有害模式,
                filtered: 过滤后的输出
            }
        """
        matches = []
        filtered = text
        max_level = ThreatLevel.SAFE

        for pattern, desc in self.compiled_harmful:
            found = pattern.findall(text)
            if found:
                matches.append({
                    "pattern": desc,
                    "count": len(found),
                    "samples": [str(s)[:50] for s in found[:2]],
                })
                filtered = pattern.sub("[内容已过滤]", filtered)
                max_level = max(max_level, ThreatLevel.HIGH, key=lambda x: x.value)

        # 记录历史
        self.output_history.append(text)

        is_harmful = max_level.value >= ThreatLevel.HIGH.value
        if is_harmful:
            self.blocked_count += 1

        return {
            "is_harmful": is_harmful,
            "threat_level": max_level.name,
            "matches": matches,
            "filtered": filtered,
        }

    def consistency_check(self, generate_fn: Callable[[str], str],
                          prompt: str, n: int = 3) -> Dict[str, Any]:
        """一致性检查 — 多次生成, 检测不一致

        对抗样本可能导致模型输出不稳定
        """
        outputs = [generate_fn(prompt) for _ in range(n)]

        # 计算输出间的相似度
        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sim = self._text_similarity(outputs[i], outputs[j])
                similarities.append(sim)

        avg_sim = sum(similarities) / max(1, len(similarities))

        if avg_sim < 0.5:
            return {
                "consistent": False,
                "avg_similarity": avg_sim,
                "warning": "输出不一致, 可能存在对抗输入",
                "outputs": outputs,
            }

        return {
            "consistent": True,
            "avg_similarity": avg_sim,
            "outputs": outputs,
        }

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """计算文本相似度 (Jaccard)"""
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a and not set_b:
            return 1.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0


# ============================================================
# 模型水印
# ============================================================

class ModelWatermark:
    """模型水印 — 版权保护与所有权验证

    水印策略:
    1. 后门水印: 在训练时嵌入触发模式
    2. 指纹水印: 在权重中嵌入统计特征
    3. 输出水印: 在生成文本中嵌入隐蔽标记
    """

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or self._generate_key()
        self.watermark_triggers: Dict[str, str] = {}  # trigger -> response
        self.weight_signatures: Dict[str, float] = {}

    @staticmethod
    def _generate_key() -> str:
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]

    def embed_trigger(self, trigger: str, response: str) -> None:
        """嵌入后门触发器"""
        # 使用密钥对触发器进行哈希
        hashed = hashlib.sha256(
            (trigger + self.secret_key).encode()
        ).hexdigest()
        self.watermark_triggers[hashed] = response

    def verify_trigger(self, trigger: str) -> Tuple[bool, Optional[str]]:
        """验证触发器

        Returns:
            (是否匹配, 预期响应)
        """
        hashed = hashlib.sha256(
            (trigger + self.secret_key).encode()
        ).hexdigest()
        if hashed in self.watermark_triggers:
            return True, self.watermark_triggers[hashed]
        return False, None

    def sign_weights(self, weights: Dict[str, List[List[float]]]) -> str:
        """对模型权重生成签名

        通过统计特征生成权重指纹
        """
        signature_parts = []
        for name, matrix in weights.items():
            if not matrix or not matrix[0]:
                continue
            # 计算权重的统计特征
            flat = [v for row in matrix for v in row]
            if not flat:
                continue
            mean = sum(flat) / len(flat)
            variance = sum((v - mean) ** 2 for v in flat) / len(flat)
            # 嵌入密钥
            signed_mean = mean + int(self.secret_key[:8], 16) * 1e-10
            signature_parts.append(f"{name}:{signed_mean:.15f}:{variance:.15f}")

        signature = hashlib.sha256(
            "|".join(signature_parts).encode()
        ).hexdigest()

        self.weight_signatures[signature] = time.time()
        return signature

    def verify_weights(self, weights: Dict[str, List[List[float]]],
                      expected_signature: str) -> bool:
        """验证权重签名"""
        current_sig = self.sign_weights(weights)
        return current_sig == expected_signature

    def embed_output_watermark(self, text: str) -> str:
        """在输出文本中嵌入隐蔽水印

        使用零宽字符编码水印信息
        """
        # 生成水印位序列
        wm_hash = hashlib.sha256(
            (text[:50] + self.secret_key).encode()
        ).hexdigest()
        wm_bits = ''.join(format(int(c, 16), '04b') for c in wm_hash[:8])

        # 零宽字符映射
        zwc = {
            '0': '\u200b',  # ZWSP
            '1': '\u200c',  # ZWNJ
        }

        # 在文本末尾嵌入水印
        watermark = ''.join(zwc[b] for b in wm_bits)
        return text + watermark

    def extract_output_watermark(self, text: str) -> Optional[str]:
        """提取输出文本中的水印"""
        zwc_reverse = {'\u200b': '0', '\u200c': '1'}

        # 从末尾提取零宽字符
        wm_chars = []
        for c in reversed(text):
            if c in zwc_reverse:
                wm_chars.append(zwc_reverse[c])
            elif c.isalnum() or c in ' .,!?;:\'"()-\n':
                break

        if not wm_chars:
            return None

        wm_bits = ''.join(reversed(wm_chars))
        # 将位序列转回十六进制
        try:
            wm_hex = ''
            for i in range(0, len(wm_bits), 4):
                nibble = wm_bits[i:i+4]
                if len(nibble) == 4:
                    wm_hex += hex(int(nibble, 2))[2:]
            return wm_hex
        except Exception:
            return None


# ============================================================
# 速率限制器
# ============================================================

class RateLimiter:
    """速率限制器

    多策略速率限制:
    1. 固定窗口: 固定时间窗口内的请求上限
    2. 滑动窗口: 更精确的窗口控制
    3. 令牌桶: 允许突发流量
    4. 用户分级: 不同用户不同限制
    """

    def __init__(self, default_limit: int = 60,
                 default_window: float = 60.0):
        self.default_limit = default_limit
        self.default_window = default_window
        self.user_limits: Dict[str, Tuple[int, float]] = {}  # user -> (limit, window)
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.token_buckets: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def set_user_limit(self, user_id: str, limit: int, window: float = 60.0) -> None:
        """设置用户特定的速率限制"""
        self.user_limits[user_id] = (limit, window)

    def check(self, user_id: str) -> Dict[str, Any]:
        """检查是否允许请求 (滑动窗口)

        Returns:
            {
                allowed: 是否允许,
                remaining: 剩余请求数,
                reset_time: 重置时间,
                limit: 限制数
            }
        """
        with self._lock:
            limit, window = self.user_limits.get(
                user_id, (self.default_limit, self.default_window)
            )

            now = time.time()
            user_requests = self.requests[user_id]

            # 清理过期请求
            while user_requests and user_requests[0] < now - window:
                user_requests.popleft()

            if len(user_requests) >= limit:
                reset_time = user_requests[0] + window
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_time": reset_time,
                    "limit": limit,
                }

            user_requests.append(now)
            return {
                "allowed": True,
                "remaining": limit - len(user_requests),
                "reset_time": now + window,
                "limit": limit,
            }

    def check_token_bucket(self, user_id: str,
                           capacity: int = 10,
                           refill_rate: float = 1.0) -> Dict[str, Any]:
        """令牌桶速率限制

        Args:
            user_id: 用户ID
            capacity: 桶容量 (最大突发)
            refill_rate: 令牌补充速率 (个/秒)
        """
        with self._lock:
            now = time.time()

            if user_id not in self.token_buckets:
                self.token_buckets[user_id] = {
                    "tokens": capacity,
                    "last_refill": now,
                }

            bucket = self.token_buckets[user_id]

            # 补充令牌
            elapsed = now - bucket["last_refill"]
            bucket["tokens"] = min(capacity, bucket["tokens"] + elapsed * refill_rate)
            bucket["last_refill"] = now

            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return {
                    "allowed": True,
                    "remaining_tokens": bucket["tokens"],
                    "capacity": capacity,
                }
            else:
                wait_time = (1.0 - bucket["tokens"]) / refill_rate
                return {
                    "allowed": False,
                    "remaining_tokens": bucket["tokens"],
                    "wait_time": wait_time,
                    "capacity": capacity,
                }


# ============================================================
# 异常检测器
# ============================================================

class AnomalyDetector:
    """行为异常检测器

    检测异常使用模式:
    1. 请求频率异常: 突发大量请求
    2. 输入模式异常: 重复相似输入
    3. 错误率异常: 突然高错误率
    4. 行为序列异常: 异常的操作序列
    5. Z-Score异常检测: 统计异常
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.user_activities: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.anomaly_count = 0

    def record_activity(self, user_id: str,
                        activity_type: str = "request",
                        metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """记录用户活动并检测异常

        Returns:
            {
                is_anomalous: 是否异常,
                anomaly_type: 异常类型,
                score: 异常分数 (0~1),
                details: 详情
            }
        """
        now = time.time()
        activity = {
            "timestamp": now,
            "type": activity_type,
            "metadata": metadata or {},
        }

        activities = self.user_activities[user_id]
        activities.append(activity)

        # 如果活动太少, 不检测
        if len(activities) < 10:
            return {"is_anomalous": False, "anomaly_type": None,
                    "score": 0.0, "details": "数据不足"}

        anomalies = []

        # 1. 请求频率异常
        freq_anomaly = self._check_frequency(activities)
        if freq_anomaly:
            anomalies.append(freq_anomaly)

        # 2. 输入相似度异常
        similarity_anomaly = self._check_input_similarity(activities)
        if similarity_anomaly:
            anomalies.append(similarity_anomaly)

        # 3. 错误率异常
        error_anomaly = self._check_error_rate(activities)
        if error_anomaly:
            anomalies.append(error_anomaly)

        # 4. 时间间隔异常
        interval_anomaly = self._check_time_intervals(activities)
        if interval_anomaly:
            anomalies.append(interval_anomaly)

        is_anomalous = len(anomalies) > 0
        max_score = max((a.get("score", 0) for a in anomalies), default=0.0)

        if is_anomalous:
            self.anomaly_count += 1

        return {
            "is_anomalous": is_anomalous,
            "anomaly_type": anomalies[0]["type"] if anomalies else None,
            "score": max_score,
            "details": anomalies,
        }

    def _check_frequency(self, activities: deque) -> Optional[Dict]:
        """检查请求频率异常"""
        if len(activities) < 10:
            return None

        now = time.time()
        recent = [a for a in activities if now - a["timestamp"] < 60.0]
        older = [a for a in activities if 60.0 <= now - a["timestamp"] < 120.0]

        if not older:
            return None

        recent_rate = len(recent) / 60.0
        older_rate = len(older) / 60.0

        if older_rate > 0 and recent_rate > older_rate * 5:
            return {
                "type": "frequency_burst",
                "score": min(1.0, recent_rate / (older_rate * 5) - 1),
                "description": f"请求频率激增: {recent_rate:.1f}/s vs {older_rate:.1f}/s",
            }
        return None

    def _check_input_similarity(self, activities: deque) -> Optional[Dict]:
        """检查输入相似度异常 (可能是自动化攻击)"""
        recent_inputs = [a["metadata"].get("input", "") for a in activities
                         if a["metadata"].get("input")]
        if len(recent_inputs) < 10:
            return None

        # 计算最近10个输入的两两相似度
        recent = recent_inputs[-10:]
        similarities = []
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                sim = AdversarialDefense._text_similarity(recent[i], recent[j])
                similarities.append(sim)

        avg_sim = sum(similarities) / max(1, len(similarities))
        if avg_sim > 0.8:
            return {
                "type": "repetitive_input",
                "score": avg_sim,
                "description": f"输入高度相似 (avg={avg_sim:.2f}), 可能是自动化攻击",
            }
        return None

    def _check_error_rate(self, activities: deque) -> Optional[Dict]:
        """检查错误率异常"""
        recent = list(activities)[-20:]
        errors = sum(1 for a in recent if a["metadata"].get("error"))
        error_rate = errors / max(1, len(recent))

        if error_rate > 0.5 and len(recent) >= 10:
            return {
                "type": "high_error_rate",
                "score": error_rate,
                "description": f"错误率异常高: {error_rate:.1%}",
            }
        return None

    def _check_time_intervals(self, activities: deque) -> Optional[Dict]:
        """检查时间间隔异常 (机器人特征: 间隔过于均匀)"""
        if len(activities) < 10:
            return None

        timestamps = [a["timestamp"] for a in activities][-10:]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if len(intervals) < 5:
            return None

        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return None

        variance = sum((i - mean_interval) ** 2 for i in intervals) / len(intervals)
        cv = (variance ** 0.5) / mean_interval  # 变异系数

        # 变异系数极低说明间隔过于均匀 (机器人特征)
        if cv < 0.05 and mean_interval < 1.0:
            return {
                "type": "robotic_pattern",
                "score": 1.0 - cv,
                "description": f"请求间隔过于均匀 (CV={cv:.4f}), 可能是机器人",
            }
        return None


# ============================================================
# 审计日志
# ============================================================

class AuditLogger:
    """审计日志 — 全链路安全事件追踪

    记录所有安全相关事件, 支持查询和取证
    """

    def __init__(self, max_events: int = 10000):
        self.events: deque = deque(maxlen=max_events)
        self.event_counter = 0
        self._lock = threading.Lock()

    def log(self, event_type: AttackType, threat_level: ThreatLevel,
            source: str, description: str,
            input_data: str = "",
            action: DefenseAction = DefenseAction.LOG_ONLY,
            metadata: Optional[Dict] = None) -> str:
        """记录安全事件

        Returns:
            事件ID
        """
        with self._lock:
            self.event_counter += 1
            event_id = f"SEC-{self.event_counter:06d}"

            event = SecurityEvent(
                event_id=event_id,
                timestamp=time.time(),
                event_type=event_type,
                threat_level=threat_level,
                source=source,
                description=description,
                input_data=input_data[:500],  # 截断
                action_taken=action,
                metadata=metadata or {},
            )
            self.events.append(event)
            return event_id

    def query(self, event_type: Optional[AttackType] = None,
              threat_level: Optional[ThreatLevel] = None,
              source: Optional[str] = None,
              start_time: Optional[float] = None,
              end_time: Optional[float] = None,
              limit: int = 100) -> List[Dict[str, Any]]:
        """查询安全事件"""
        results = []
        for event in reversed(self.events):
            if event_type and event.event_type != event_type:
                continue
            if threat_level and event.threat_level != threat_level:
                continue
            if source and event.source != source:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            results.append({
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.name,
                "source": event.source,
                "description": event.description,
                "action": event.action_taken.value,
                "metadata": event.metadata,
            })

            if len(results) >= limit:
                break

        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取安全统计"""
        total = len(self.events)
        by_type = defaultdict(int)
        by_level = defaultdict(int)
        by_action = defaultdict(int)

        for event in self.events:
            by_type[event.event_type.value] += 1
            by_level[event.threat_level.name] += 1
            by_action[event.action_taken.value] += 1

        return {
            "total_events": total,
            "by_type": dict(by_type),
            "by_level": dict(by_level),
            "by_action": dict(by_action),
        }


# ============================================================
# 安全沙箱系统 (主系统)
# ============================================================

class SecuritySandboxSystem:
    """安全沙箱系统 — 端到端安全防护

    整合所有安全组件, 提供完整的安全防护链:

    防护流程:
    1. 速率限制 → 防止滥用
    2. 输入净化 → 检测注入/敏感信息
    3. 对抗检测 → 检测对抗样本
    4. 异常检测 → 行为异常识别
    5. 沙箱执行 → 安全代码执行
    6. 输出检查 → 有害内容过滤
    7. 审计日志 → 全程记录
    """

    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.sandbox = SecuritySandbox()
        self.adversarial = AdversarialDefense()
        self.watermark = ModelWatermark()
        self.rate_limiter = RateLimiter()
        self.anomaly_detector = AnomalyDetector()
        self.audit = AuditLogger()

        # 全局状态
        self.circuit_breaker_open = False
        self.circuit_breaker_reset_time = 0.0
        self.total_blocked = 0
        self.total_sanitized = 0
        self.total_allowed = 0

    def process_input(self, user_id: str, text: str,
                      metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """处理输入 — 完整安全检查链

        Args:
            user_id: 用户ID
            text: 输入文本
            metadata: 额外元数据

        Returns:
            {
                allowed: 是否允许,
                action: 执行动作,
                sanitized_text: 净化后的文本,
                threats: 检测到的威胁,
                event_id: 审计事件ID
            }
        """
        # 0. 熔断器检查
        if self.circuit_breaker_open:
            if time.time() < self.circuit_breaker_reset_time:
                event_id = self.audit.log(
                    AttackType.DENIAL_OF_SERVICE, ThreatLevel.HIGH,
                    user_id, "熔断器开启, 请求被拒绝",
                    text, DefenseAction.BLOCK, metadata
                )
                return {"allowed": False, "action": "blocked",
                        "reason": "circuit_breaker_open",
                        "sanitized_text": "", "threats": [],
                        "event_id": event_id}
            else:
                self.circuit_breaker_open = False

        # 1. 速率限制
        rate_check = self.rate_limiter.check(user_id)
        if not rate_check["allowed"]:
            event_id = self.audit.log(
                AttackType.DENIAL_OF_SERVICE, ThreatLevel.MEDIUM,
                user_id, f"速率限制触发: {rate_check['limit']}/60s",
                text, DefenseAction.BLOCK, metadata
            )
            return {"allowed": False, "action": "rate_limited",
                    "reason": "rate_limit_exceeded",
                    "sanitized_text": "", "threats": [],
                    "event_id": event_id, "rate_info": rate_check}

        # 2. 输入净化
        sanitize_result = self.sanitizer.scan(text)

        # 3. 对抗检测
        adv_result = self.adversarial.check_input(sanitize_result["sanitized_text"])

        # 4. 异常检测
        anomaly_result = self.anomaly_detector.record_activity(
            user_id, "request", {"input": text[:200], **(metadata or {})}
        )

        # 综合判断
        threats = []
        if sanitize_result["threats"]:
            threats.extend(sanitize_result["threats"])
        if adv_result["indicators"]:
            threats.append({"type": "adversarial", "indicators": adv_result["indicators"]})
        if anomaly_result["is_anomalous"]:
            threats.append({"type": "anomaly", "details": anomaly_result["details"]})

        # 确定最终动作
        threat_level = ThreatLevel.SAFE
        for t in sanitize_result.get("threats", []):
            if isinstance(t.get("level"), ThreatLevel):
                threat_level = max(threat_level, t["level"], key=lambda x: x.value)

        current_threat = ThreatLevel[sanitize_result["threat_level"]]
        if current_threat.value >= ThreatLevel.HIGH.value:
            action = DefenseAction.BLOCK
            self.total_blocked += 1
            allowed = False
            # 更新熔断器
            if self.total_blocked % 10 == 0:
                self.circuit_breaker_open = True
                self.circuit_breaker_reset_time = time.time() + 60.0
        elif current_threat.value >= ThreatLevel.MEDIUM.value:
            action = DefenseAction.SANITIZE
            self.total_sanitized += 1
            allowed = True
        else:
            action = DefenseAction.ALLOW
            self.total_allowed += 1
            allowed = True

        # 记录审计日志
        attack_type = AttackType[sanitize_result["attack_type"].upper()] \
            if sanitize_result["attack_type"] != "unknown" else AttackType.UNKNOWN

        event_id = self.audit.log(
            attack_type, ThreatLevel[sanitize_result["threat_level"]],
            user_id, f"输入安全检查: action={action.value}",
            text, action, {
                "sanitize_result": sanitize_result,
                "adv_result": adv_result,
                "anomaly_result": anomaly_result,
                **(metadata or {}),
            }
        )

        final_text = adv_result["sanitized"] if action != DefenseAction.BLOCK else ""

        return {
            "allowed": allowed,
            "action": action.value,
            "sanitized_text": final_text,
            "threats": threats,
            "event_id": event_id,
            "rate_info": rate_check,
        }

    def execute_code(self, user_id: str, code: str) -> Dict[str, Any]:
        """在安全沙箱中执行代码"""
        # 先进行安全检查
        security_check = self.process_input(user_id, code, {"type": "code_execution"})
        if not security_check["allowed"]:
            return {"status": "blocked", "security": security_check}

        # 沙箱执行
        result = self.sandbox.execute(security_check["sanitized_text"])

        # 输出检查
        if result.get("output"):
            output_check = self.adversarial.check_output(result["output"])
            if output_check["is_harmful"]:
                result["output"] = output_check["filtered"]
                self.audit.log(
                    AttackType.ADVERSARIAL_INPUT, ThreatLevel.HIGH,
                    user_id, "有害输出被过滤",
                    result["output"][:200], DefenseAction.BLOCK,
                    {"patterns": output_check["matches"]}
                )

        self.audit.log(
            AttackType.UNKNOWN, ThreatLevel.SAFE,
            user_id, f"代码执行: {result['status']}",
            code[:200], DefenseAction.ALLOW,
            {"execution_result": result}
        )

        return {"status": result["status"], "result": result, "security": security_check}

    def check_output(self, user_id: str, output: str) -> Dict[str, Any]:
        """检查输出安全性"""
        result = self.adversarial.check_output(output)

        self.audit.log(
            AttackType.ADVERSARIAL_INPUT if result["is_harmful"] else AttackType.UNKNOWN,
            ThreatLevel[result["threat_level"]],
            user_id, f"输出检查: {'有害' if result['is_harmful'] else '安全'}",
            output[:200],
            DefenseAction.BLOCK if result["is_harmful"] else DefenseAction.ALLOW,
            {"matches": result["matches"]}
        )

        return result

    def get_security_report(self) -> Dict[str, Any]:
        """获取安全报告"""
        audit_stats = self.audit.get_stats()
        return {
            "total_requests": (self.total_allowed + self.total_sanitized + self.total_blocked),
            "allowed": self.total_allowed,
            "sanitized": self.total_sanitized,
            "blocked": self.total_blocked,
            "block_rate": self.total_blocked / max(1, self.total_allowed + self.total_sanitized + self.total_blocked),
            "circuit_breaker_open": self.circuit_breaker_open,
            "sandbox_stats": {
                "executions": self.sandbox.execution_count,
                "blocked": self.sandbox.blocked_count,
            },
            "sanitizer_stats": {
                "blocked": self.sanitizer.blocked_count,
                "sanitized": self.sanitizer.sanitized_count,
            },
            "adversarial_stats": {
                "flagged_inputs": self.adversarial.flagged_count,
                "blocked_outputs": self.adversarial.blocked_count,
            },
            "anomaly_stats": {
                "anomalies_detected": self.anomaly_detector.anomaly_count,
            },
            "audit_stats": audit_stats,
        }
