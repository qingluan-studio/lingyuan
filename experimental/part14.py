"""
灵元大模型 - API服务模块 (part14.py)
对应52项清单 #36-41

子系统:
  #36 HTTPServer          - 真实HTTP服务器 (基于http.server, 可curl)
  #37 OpenAICompatibleAPI - OpenAI兼容API (chat/completions, embeddings, models ...)
  #38 WebSocketServer     - WebSocket服务 (握手/帧/房间/广播/流式推理)
  #39 GRPCService         - gRPC服务定义 (proto/简化protobuf编解码/拦截器/健康检查)
  #40 APIDocGenerator     - API文档生成 (OpenAPI 3.0 / YAML / Swagger UI)
  #41 SDKGenerator        - SDK自动生成 (Python / JavaScript / Go)

纯 Python 标准库实现, 零外部依赖.
此文件在 lingyuan_full.py 之后加载, 可使用全局变量: DATA_DIR, LOG_DIR, CONFIG_DIR
"""

import uuid
import math
import random
import json
import os
import time
import threading
import socket
import http.server
import socketserver
import urllib.parse
from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime

# 额外标准库导入 (均为 Python 标准库, 维持零外部依赖)
import hashlib
import base64
import struct
import mimetypes
import traceback

# ============================================================
# 全局路径 (优先使用 lingyuan_full.py 中定义的全局变量)
# ============================================================
_DATA_DIR = globals().get('DATA_DIR', os.path.join(os.path.expanduser('~'), '.lingyuan', 'data'))
_LOG_DIR = globals().get('LOG_DIR', os.path.join(os.path.expanduser('~'), '.lingyuan', 'logs'))
_CONFIG_DIR = globals().get('CONFIG_DIR', os.path.join(os.path.expanduser('~'), '.lingyuan', 'config'))

for _d in [_DATA_DIR, _LOG_DIR, _CONFIG_DIR]:
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception:
        pass


# ============================================================
# 通用数据结构
# ============================================================

@dataclass
class Route:
    """路由定义

    支持路径参数 {param}, 路由级中间件, 文档标签与 schema.
    """
    method: str
    path: str
    handler: Callable
    middleware: List[Callable] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    summary: str = ""
    description: str = ""
    request_schema: Optional[Dict] = None
    response_schema: Optional[Dict] = None
    auth_required: bool = False


@dataclass
class RequestContext:
    """请求上下文 - 传递给 handler 的统一对象"""
    method: str
    path: str
    query: Dict[str, str]
    headers: Dict[str, str]
    body: Any
    path_params: Dict[str, str]
    client_ip: str
    request_id: str
    raw_path: str = ""


@dataclass
class Response:
    """统一响应对象

    body 为可 JSON 序列化对象; stream 为生成器函数时以 SSE 方式推送.
    """
    status: int = 200
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    stream: Optional[Callable] = None


@dataclass
class RequestLog:
    """请求日志条目"""
    method: str
    path: str
    status: int
    latency_ms: float
    timestamp: str
    client_ip: str = ""
    request_id: str = ""


@dataclass
class APIError:
    """OpenAI 格式错误"""
    message: str
    type: str = "api_error"
    code: Optional[str] = None
    param: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"message": self.message, "type": self.type}
        if self.code is not None:
            d["code"] = self.code
        if self.param is not None:
            d["param"] = self.param
        return d


@dataclass
class Usage:
    """用量统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int = 0, completion: int = 0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


# ============================================================
# 中间件工厂
# ============================================================

def auth_middleware(token: str) -> Callable:
    """认证中间件工厂: 校验 Bearer token"""
    def _mw(ctx: RequestContext) -> Optional[Response]:
        auth = ctx.headers.get("authorization", "") or ctx.headers.get("Authorization", "")
        if auth != f"Bearer {token}":
            return Response(status=401, body={
                "error": {"message": "无效的API密钥.", "type": "invalid_request_error", "code": "invalid_api_key"}
            })
        return None
    return _mw


def log_middleware() -> Callable:
    """日志中间件: 记录请求摘要 (实际落库由 HTTPServer._record_request 完成)"""
    def _mw(ctx: RequestContext) -> Optional[Response]:
        return None
    return _mw


def cors_middleware() -> Callable:
    """CORS 中间件: 实际响应头由 HTTPServer 默认注入, 此处仅占位便于链式注册"""
    def _mw(ctx: RequestContext) -> Optional[Response]:
        return None
    return _mw


# ============================================================
# #36 HTTPServer - 真实HTTP服务器
# ============================================================

class HTTPServer:
    """真实HTTP服务器

    基于 http.server.BaseHTTPRequestHandler + ThreadingHTTPServer 实现,
    提供路由注册、路径参数、JSON/Query 解析、中间件链、CORS、静态文件、
    健康检查、请求日志与线程安全. 启动后可用 curl 直接访问.
    """

    def __init__(self, host: str = "0.0.0.0", name: str = "lingyuan-api"):
        self.host = host
        self.name = name
        self.routes: List[Route] = []
        self.middleware: List[Callable] = []
        self.static_dir: Optional[str] = None
        self.server: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
        self.request_logs: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._stats = {
            "total_requests": 0,
            "total_errors": 0,
            "status_counts": {},
            "method_counts": {},
            "path_counts": {},
            "start_time": None,
        }
        self._register_default_routes()

    # ---------- 路由注册 ----------
    def _register_default_routes(self) -> None:
        self.add_route("GET", "/health", self._health_handler,
                       tags=["system"], summary="健康检查",
                       response_schema={"status": "string", "service": "string", "uptime": "number"})

    def _health_handler(self, ctx: RequestContext) -> Response:
        uptime = time.time() - (self._stats["start_time"] or time.time())
        return Response(status=200, body={
            "status": "ok",
            "service": self.name,
            "uptime": round(uptime, 2),
            "routes": len(self.routes),
            "timestamp": datetime.now().isoformat(),
        })

    def add_route(self, method: str, path: str, handler: Callable,
                  middleware: Optional[List[Callable]] = None, tags: Optional[List[str]] = None,
                  summary: str = "", description: str = "", auth_required: bool = False,
                  request_schema: Optional[Dict] = None, response_schema: Optional[Dict] = None) -> "HTTPServer":
        """注册一条路由"""
        route = Route(
            method=method.upper(), path=path, handler=handler,
            middleware=middleware or [], tags=tags or [],
            summary=summary, description=description, auth_required=auth_required,
            request_schema=request_schema, response_schema=response_schema,
        )
        self.routes.append(route)
        return self

    def get(self, path: str, **kw) -> Callable:
        def _deco(func: Callable) -> Callable:
            self.add_route("GET", path, func, **kw)
            return func
        return _deco

    def post(self, path: str, **kw) -> Callable:
        def _deco(func: Callable) -> Callable:
            self.add_route("POST", path, func, **kw)
            return func
        return _deco

    def put(self, path: str, **kw) -> Callable:
        def _deco(func: Callable) -> Callable:
            self.add_route("PUT", path, func, **kw)
            return func
        return _deco

    def delete(self, path: str, **kw) -> Callable:
        def _deco(func: Callable) -> Callable:
            self.add_route("DELETE", path, func, **kw)
            return func
        return _deco

    def use(self, mw: Callable) -> "HTTPServer":
        """注册全局中间件"""
        self.middleware.append(mw)
        return self

    def set_static_dir(self, directory: str) -> "HTTPServer":
        """设置静态文件目录"""
        self.static_dir = directory
        return self

    # ---------- 路径匹配 ----------
    def _match_route(self, method: str, path: str) -> Tuple[Optional[Route], Dict[str, str]]:
        for route in self.routes:
            if route.method != method:
                continue
            params = self._match_path(route.path, path)
            if params is not None:
                return route, params
        return None, {}

    @staticmethod
    def _match_path(pattern: str, path: str) -> Optional[Dict[str, str]]:
        p_segs = [s for s in pattern.split("/") if s != ""]
        a_segs = [s for s in path.split("/") if s != ""]
        if len(p_segs) != len(a_segs):
            return None
        params: Dict[str, str] = {}
        for ps, asg in zip(p_segs, a_segs):
            if ps.startswith("{") and ps.endswith("}"):
                params[ps[1:-1]] = urllib.parse.unquote(asg)
            elif ps != asg:
                return None
        return params

    # ---------- 响应规整 ----------
    @staticmethod
    def _normalize_response(result: Any) -> Response:
        if isinstance(result, Response):
            return result
        if isinstance(result, tuple):
            if len(result) == 2:
                return Response(status=result[0], body=result[1])
            if len(result) == 3:
                return Response(status=result[0], body=result[1], headers=result[2] or {})
        if isinstance(result, dict):
            return Response(body=result)
        if result is None:
            return Response(body={"ok": True})
        return Response(body=result)

    # ---------- 请求记录 ----------
    def _record_request(self, ctx: RequestContext, response: Response, latency: float) -> None:
        with self._lock:
            self._stats["total_requests"] += 1
            if response.status >= 400:
                self._stats["total_errors"] += 1
            self._stats["status_counts"][response.status] = self._stats["status_counts"].get(response.status, 0) + 1
            self._stats["method_counts"][ctx.method] = self._stats["method_counts"].get(ctx.method, 0) + 1
            self._stats["path_counts"][ctx.path] = self._stats["path_counts"].get(ctx.path, 0) + 1
        self.request_logs.append(RequestLog(
            method=ctx.method, path=ctx.path, status=response.status,
            latency_ms=round(latency, 2), timestamp=datetime.now().isoformat(),
            client_ip=ctx.client_ip, request_id=ctx.request_id,
        ))

    # ---------- 静态文件 ----------
    def _serve_static(self, handler, raw_path: str) -> bool:
        if not self.static_dir:
            return False
        rel = raw_path.lstrip("/")
        if rel == "" :
            rel = "index.html"
        fs_path = os.path.normpath(os.path.join(self.static_dir, rel))
        if not fs_path.startswith(os.path.abspath(self.static_dir)):
            return False
        if not os.path.isfile(fs_path):
            return False
        try:
            ctype = mimetypes.guess_type(fs_path)[0] or "application/octet-stream"
            with open(fs_path, "rb") as f:
                data = f.read()
            handler.send_response(200)
            handler.send_header("Content-Type", ctype)
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(data)
            return True
        except Exception:
            return False

    # ---------- 发送响应 ----------
    def _send_response(self, handler, response: Response, ctx: RequestContext) -> None:
        headers = dict(response.headers)
        headers.setdefault("Content-Type", "application/json")
        headers.setdefault("Access-Control-Allow-Origin", "*")
        headers.setdefault("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
        headers["X-Request-ID"] = ctx.request_id

        if response.stream is not None:
            # SSE 流式响应: 无 Content-Length, 通过关闭连接标识响应结束
            headers["Content-Type"] = "text/event-stream"
            headers["Cache-Control"] = "no-cache"
            headers["Connection"] = "close"
            handler.close_connection = True
            handler.send_response(response.status)
            for k, v in headers.items():
                handler.send_header(k, v)
            handler.end_headers()
            try:
                for chunk in response.stream():
                    if isinstance(chunk, (dict, list)):
                        payload = json.dumps(chunk, ensure_ascii=False)
                    else:
                        payload = str(chunk)
                    handler.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    handler.wfile.flush()
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
            except Exception:
                pass
            return

        body_bytes = b""
        if response.body is not None:
            body_bytes = json.dumps(response.body, ensure_ascii=False, default=str).encode("utf-8")
        handler.send_response(response.status)
        handler.send_header("Content-Length", str(len(body_bytes)))
        for k, v in headers.items():
            handler.send_header(k, v)
        handler.end_headers()
        if ctx.method != "HEAD":
            handler.wfile.write(body_bytes)

    # ---------- 构建请求处理器类 ----------
    def _make_handler_class(self):
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "LingyuanHTTP/1.0"

            def log_message(self, fmt, *args):
                pass  # 静默默认日志, 由 server 自行记录

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_PUT(self):
                self._handle("PUT")

            def do_DELETE(self):
                self._handle("DELETE")

            def do_PATCH(self):
                self._handle("PATCH")

            def do_OPTIONS(self):
                self._handle("OPTIONS")

            def _handle(self, method: str):
                parsed = urllib.parse.urlparse(self.path)
                raw_path = parsed.path
                query = dict(urllib.parse.parse_qsl(parsed.query))
                route, path_params = server._match_route(method, raw_path)

                content_length = int(self.headers.get("Content-Length", 0) or 0)
                body: Any = None
                if content_length > 0:
                    raw = self.rfile.read(content_length)
                    try:
                        body = json.loads(raw)
                    except Exception:
                        body = raw.decode("utf-8", errors="replace")

                ctx = RequestContext(
                    method=method, path=raw_path, raw_path=raw_path, query=query,
                    headers={k: v for k, v in self.headers.items()},
                    body=body, path_params=path_params,
                    client_ip=self.client_address[0] if self.client_address else "",
                    request_id=str(uuid.uuid4()),
                )

                start = time.time()
                response: Optional[Response] = None
                try:
                    # OPTIONS 预检直接放行
                    if method == "OPTIONS":
                        response = Response(status=204, body=None)
                    else:
                        for mw in server.middleware:
                            res = mw(ctx)
                            if res is not None:
                                response = res
                                break
                        if response is None:
                            if route is None:
                                if method == "GET" and server._serve_static(self, raw_path):
                                    return
                                response = Response(status=404, body={
                                    "error": {"message": f"未找到路由: {method} {raw_path}",
                                              "type": "not_found"}
                                })
                            else:
                                for mw in route.middleware:
                                    res = mw(ctx)
                                    if res is not None:
                                        response = res
                                        break
                                if response is None:
                                    result = route.handler(ctx)
                                    response = server._normalize_response(result)
                except Exception as e:
                    traceback.print_exc()
                    response = Response(status=500, body={
                        "error": {"message": f"内部错误: {e}", "type": "internal_error"}
                    })

                latency = (time.time() - start) * 1000
                server._record_request(ctx, response, latency)
                try:
                    server._send_response(self, response, ctx)
                except Exception:
                    pass

        return Handler

    # ---------- 启停 ----------
    def start(self, port: int, block: bool = False) -> None:
        """启动服务器. block=False 时在后台线程运行."""
        self._stats["start_time"] = time.time()
        handler_cls = self._make_handler_class()
        self.server = http.server.ThreadingHTTPServer((self.host, port), handler_cls)
        self.server.daemon_threads = True
        self._running = True
        if block:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

    def serve_forever(self, port: int) -> None:
        """阻塞式启动"""
        self.start(port, block=True)

    def stop(self) -> None:
        """停止服务器"""
        self._running = False
        if self.server is not None:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None
        self.server = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ---------- 统计与仪表盘 ----------
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            status_counts = dict(self._stats["status_counts"])
            method_counts = dict(self._stats["method_counts"])
            path_counts = dict(self._stats["path_counts"])
        total = self._stats["total_requests"]
        errors = self._stats["total_errors"]
        latencies = [r.latency_ms for r in self.request_logs]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        return {
            "name": self.name,
            "host": self.host,
            "running": self._running,
            "routes": len(self.routes),
            "static_dir": self.static_dir,
            "total_requests": total,
            "total_errors": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "avg_latency_ms": avg_latency,
            "status_counts": status_counts,
            "method_counts": method_counts,
            "top_paths": sorted(path_counts.items(), key=lambda x: -x[1])[:10],
            "uptime": round(time.time() - (self._stats["start_time"] or time.time()), 2),
        }

    def get_dashboard(self) -> str:
        """获取仪表盘字符串"""
        s = self.get_stats()
        lines = [
            "========== HTTPServer 仪表盘 ==========",
            f"  服务名称:     {s['name']}",
            f"  监听地址:     {s['host']}",
            f"  运行状态:     {'运行中' if s['running'] else '已停止'}",
            f"  路由数量:     {s['routes']}",
            f"  静态目录:     {s['static_dir'] or '未设置'}",
            f"  总请求数:     {s['total_requests']}",
            f"  错误请求数:   {s['total_errors']}",
            f"  错误率:       {s['error_rate']:.2%}",
            f"  平均延迟:     {s['avg_latency_ms']} ms",
            f"  运行时长:     {s['uptime']} s",
            f"  状态码分布:   {s['status_counts']}",
            f"  方法分布:     {s['method_counts']}",
            f"  热门路径:     {s['top_paths'][:5]}",
            "=======================================",
        ]
        return "\n".join(lines)


# ============================================================
# #37 OpenAICompatibleAPI - OpenAI兼容API
# ============================================================

class OpenAICompatibleAPI:
    """OpenAI 兼容 API

    在 HTTPServer 上注册 OpenAI 风格端点:
      POST /v1/chat/completions     聊天补全 (支持 stream SSE)
      POST /v1/completions          文本补全
      GET  /v1/models               模型列表
      GET  /v1/models/{model_id}    模型详情
      POST /v1/embeddings           文本嵌入
      POST /v1/images/generations   图像生成 (模拟)
    含 Bearer 认证、OpenAI 错误格式与用量追踪.
    """

    def __init__(self, http_server: Optional[HTTPServer] = None, api_key: str = "sk-lingyuan",
                 model_name: str = "lingyuan-1.0"):
        self.server = http_server if http_server is not None else HTTPServer()
        self.api_key = api_key
        self.model_name = model_name
        self.models: List[Dict[str, Any]] = [
            {"id": "lingyuan-1.0", "object": "model", "created": 1700000000, "owned_by": "lingyuan"},
            {"id": "lingyuan-1.0-mini", "object": "model", "created": 1700000000, "owned_by": "lingyuan"},
            {"id": "lingyuan-embed", "object": "model", "created": 1700000000, "owned_by": "lingyuan"},
        ]
        self.usage_total = Usage()
        self.inference_fn: Optional[Callable[[str], str]] = None
        self.embed_fn: Optional[Callable[[str], List[float]]] = None
        self._lock = threading.Lock()
        self._stats = {
            "chat_calls": 0,
            "completion_calls": 0,
            "embedding_calls": 0,
            "image_calls": 0,
            "model_queries": 0,
            "stream_calls": 0,
        }
        self._register_routes()

    # ---------- 路由注册 ----------
    def _register_routes(self) -> None:
        auth = [auth_middleware(self.api_key)]
        self.server.add_route("POST", "/v1/chat/completions", self.chat_completions,
                              middleware=auth, tags=["chat"], summary="聊天补全",
                              auth_required=True,
                              request_schema={"model": "string", "messages": "array", "stream": "boolean"})
        self.server.add_route("POST", "/v1/completions", self.completions,
                              middleware=auth, tags=["completion"], summary="文本补全",
                              auth_required=True)
        self.server.add_route("GET", "/v1/models", self.list_models,
                              middleware=auth, tags=["models"], summary="模型列表",
                              auth_required=True)
        self.server.add_route("GET", "/v1/models/{model_id}", self.get_model,
                              middleware=auth, tags=["models"], summary="模型详情",
                              auth_required=True)
        self.server.add_route("POST", "/v1/embeddings", self.embeddings,
                              middleware=auth, tags=["embeddings"], summary="文本嵌入",
                              auth_required=True)
        self.server.add_route("POST", "/v1/images/generations", self.images_generations,
                              middleware=auth, tags=["images"], summary="图像生成",
                              auth_required=True)

    # ---------- 辅助 ----------
    def _error(self, status: int, message: str, etype: str = "api_error",
               code: Optional[str] = None) -> Response:
        err = APIError(message=message, type=etype, code=code)
        return Response(status=status, body={"error": err.to_dict()})

    def _count_tokens(self, text: str) -> int:
        """简易 token 计数 (按字符/词近似)"""
        if not text:
            return 0
        return max(1, len(text))

    def _generate_text(self, prompt: str) -> str:
        """生成回复文本 (可被 inference_fn 覆盖)"""
        if self.inference_fn is not None:
            try:
                return self.inference_fn(prompt)
            except Exception:
                pass
        canned = [
            "灵元大模型已就绪, 正在为您处理请求.",
            "根据上下文, 这是一个由纯Python标准库实现的推理示例.",
            "流式输出可在 stream=true 时通过 SSE 逐 token 推送.",
            "用量统计会自动累计 prompt_tokens 与 completion_tokens.",
        ]
        return random.choice(canned) + f" (输入长度={len(prompt)})"

    def _stream_chunks(self, full_text: str):
        """将完整文本切分为 token 流"""
        tokens = full_text.split()
        for i, tok in enumerate(tokens):
            yield {"id": "chunk", "delta": {"content": tok + " "}, "index": i}
            time.sleep(0.01)

    # ---------- 端点实现 ----------
    def chat_completions(self, ctx: RequestContext) -> Response:
        body = ctx.body if isinstance(ctx.body, dict) else {}
        model = body.get("model", self.model_name)
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens", 256))

        if not messages:
            return self._error(400, "messages 字段不能为空.", "invalid_request_error", "messages_required")

        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt = "\n".join(prompt_parts)
        prompt_tokens = self._count_tokens(prompt)

        with self._lock:
            self._stats["chat_calls"] += 1
            if stream:
                self._stats["stream_calls"] += 1

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            full_text = self._generate_text(prompt)[:max_tokens * 4]
            completion_tokens = self._count_tokens(full_text)
            with self._lock:
                self.usage_total.add(prompt_tokens, completion_tokens)

            def _gen():
                for chunk in self._stream_chunks(full_text):
                    yield {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": chunk["index"], "delta": chunk["delta"],
                                     "finish_reason": None}],
                    }
            return Response(status=200, stream=_gen)

        full_text = self._generate_text(prompt)[:max_tokens * 4]
        completion_tokens = self._count_tokens(full_text)
        with self._lock:
            self.usage_total.add(prompt_tokens, completion_tokens)

        return Response(status=200, body={
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": Usage(prompt_tokens, completion_tokens,
                           prompt_tokens + completion_tokens).to_dict(),
        })

    def completions(self, ctx: RequestContext) -> Response:
        body = ctx.body if isinstance(ctx.body, dict) else {}
        model = body.get("model", self.model_name)
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 64))
        stream = bool(body.get("stream", False))

        if not prompt:
            return self._error(400, "prompt 字段不能为空.", "invalid_request_error", "prompt_required")

        prompt_tokens = self._count_tokens(prompt)
        with self._lock:
            self._stats["completion_calls"] += 1

        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        full_text = self._generate_text(str(prompt))[:max_tokens * 4]
        completion_tokens = self._count_tokens(full_text)
        with self._lock:
            self.usage_total.add(prompt_tokens, completion_tokens)

        if stream:
            def _gen():
                for i, tok in enumerate(full_text.split()):
                    yield {
                        "id": completion_id, "object": "text_completion",
                        "created": created, "model": model,
                        "choices": [{"text": tok + " ", "index": 0, "finish_reason": None}],
                    }
            return Response(status=200, stream=_gen)

        return Response(status=200, body={
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"text": full_text, "index": 0, "finish_reason": "stop",
                         "logprobs": None}],
            "usage": Usage(prompt_tokens, completion_tokens,
                           prompt_tokens + completion_tokens).to_dict(),
        })

    def list_models(self, ctx: RequestContext) -> Response:
        with self._lock:
            self._stats["model_queries"] += 1
        return Response(status=200, body={"object": "list", "data": list(self.models)})

    def get_model(self, ctx: RequestContext) -> Response:
        model_id = ctx.path_params.get("model_id", "")
        with self._lock:
            self._stats["model_queries"] += 1
        for m in self.models:
            if m["id"] == model_id:
                return Response(status=200, body=m)
        return self._error(404, f"模型不存在: {model_id}", "not_found", "model_not_found")

    def embeddings(self, ctx: RequestContext) -> Response:
        body = ctx.body if isinstance(ctx.body, dict) else {}
        inp = body.get("input", "")
        model = body.get("model", "lingyuan-embed")
        if isinstance(inp, list):
            inputs = inp
        else:
            inputs = [inp]
        if not inputs:
            return self._error(400, "input 字段不能为空.", "invalid_request_error", "input_required")

        with self._lock:
            self._stats["embedding_calls"] += 1

        data = []
        total_tokens = 0
        for i, text in enumerate(inputs):
            if self.embed_fn is not None:
                vec = self.embed_fn(str(text))
            else:
                # 确定性伪嵌入 (基于哈希), 维度 16
                h = hashlib.sha256(str(text).encode("utf-8")).digest()
                vec = [round((b / 255.0) * 2 - 1, 6) for b in h[:16]]
            data.append({"object": "embedding", "index": i, "embedding": vec})
            total_tokens += self._count_tokens(str(text))

        with self._lock:
            self.usage_total.add(total_tokens, 0)

        return Response(status=200, body={
            "object": "list",
            "data": data,
            "model": model,
            "usage": Usage(total_tokens, 0, total_tokens).to_dict(),
        })

    def images_generations(self, ctx: RequestContext) -> Response:
        body = ctx.body if isinstance(ctx.body, dict) else {}
        prompt = body.get("prompt", "")
        n = int(body.get("n", 1))
        size = body.get("size", "1024x1024")
        if not prompt:
            return self._error(400, "prompt 字段不能为空.", "invalid_request_error", "prompt_required")
        with self._lock:
            self._stats["image_calls"] += 1
        data = []
        for i in range(max(1, min(n, 10))):
            seed = hashlib.md5(f"{prompt}-{i}".encode()).hexdigest()[:8]
            data.append({
                "url": f"https://placeholder.lingyuan.ai/images/{seed}.png",
                "revised_prompt": prompt,
            })
        return Response(status=200, body={"created": int(time.time()), "data": data})

    # ---------- 统计与仪表盘 ----------
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
        return {
            "api_key_configured": bool(self.api_key),
            "model_name": self.model_name,
            "models": [m["id"] for m in self.models],
            "usage_total": self.usage_total.to_dict(),
            **stats,
        }

    def get_dashboard(self) -> str:
        s = self.get_stats()
        lines = [
            "========== OpenAICompatibleAPI 仪表盘 ==========",
            f"  默认模型:       {s['model_name']}",
            f"  可用模型:       {s['models']}",
            f"  聊天补全调用:   {s['chat_calls']}",
            f"  流式调用:       {s['stream_calls']}",
            f"  文本补全调用:   {s['completion_calls']}",
            f"  嵌入调用:       {s['embedding_calls']}",
            f"  图像生成调用:   {s['image_calls']}",
            f"  模型查询:       {s['model_queries']}",
            f"  累计 prompt_tokens:    {s['usage_total']['prompt_tokens']}",
            f"  累计 completion_tokens:{s['usage_total']['completion_tokens']}",
            f"  累计 total_tokens:      {s['usage_total']['total_tokens']}",
            "================================================",
        ]
        return "\n".join(lines)


# ============================================================
# #38 WebSocketServer - WebSocket服务
# ============================================================

class WSConnection:
    """单个 WebSocket 连接的元数据"""
    def __init__(self, conn: socket.socket, cid: str, path: str, client_ip: str):
        self.conn = conn
        self.id = cid
        self.path = path
        self.client_ip = client_ip
        self.rooms: set = set()
        self.alive = True
        self.connected_at = time.time()
        self.messages_sent = 0
        self.messages_received = 0


class WebSocketServer:
    """简化 WebSocket 服务器

    基于原生 socket 实现: HTTP Upgrade 握手、文本帧收发 (不处理分片/压缩)、
    连接管理、心跳、广播、房间、事件回调与流式推理推送.
    """

    WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, host: str = "0.0.0.0"):
        self.host = host
        self.connections: Dict[str, WSConnection] = {}
        self.rooms: Dict[str, set] = {}
        self.on_connect: Optional[Callable[[WSConnection], None]] = None
        self.on_message: Optional[Callable[[WSConnection, str], None]] = None
        self.on_disconnect: Optional[Callable[[WSConnection], None]] = None
        self.inference_fn: Optional[Callable[[str], Any]] = None
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._stats = {
            "total_connections": 0,
            "current_connections": 0,
            "total_messages": 0,
            "total_broadcasts": 0,
            "start_time": None,
        }

    # ---------- 握手 ----------
    def _handshake(self, conn: socket.socket) -> Optional[Dict[str, str]]:
        data = b""
        try:
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(1024)
                if not chunk:
                    return None
                data += chunk
                if len(data) > 65536:
                    return None
        except Exception:
            return None
        text = data.decode("utf-8", errors="replace")
        lines = text.split("\r\n")
        headers: Dict[str, str] = {}
        path = "/"
        if lines:
            first = lines[0].split()
            if len(first) >= 2:
                path = first[1]
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        key = headers.get("sec-websocket-key")
        if not key:
            conn.send(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return None
        accept = base64.b64encode(
            hashlib.sha1((key + self.WS_MAGIC).encode("utf-8")).digest()
        ).decode("utf-8")
        resp = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        try:
            conn.sendall(resp.encode("utf-8"))
        except Exception:
            return None
        headers["__path__"] = path
        return headers

    # ---------- 帧编解码 ----------
    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        data = b""
        while len(data) < n:
            try:
                chunk = conn.recv(n - len(data))
            except Exception:
                return data
            if not chunk:
                return data
            data += chunk
        return data

    def _recv_frame(self, conn: socket.socket) -> Optional[Tuple[int, bytes]]:
        hdr = self._recv_exact(conn, 2)
        if len(hdr) < 2:
            return None
        b1, b2 = hdr[0], hdr[1]
        opcode = b1 & 0x0F
        masked = b2 & 0x80
        length = b2 & 0x7F
        if length == 126:
            ext = self._recv_exact(conn, 2)
            if len(ext) < 2:
                return None
            length = struct.unpack(">H", ext)[0]
        elif length == 127:
            ext = self._recv_exact(conn, 8)
            if len(ext) < 8:
                return None
            length = struct.unpack(">Q", ext)[0]
        mask = b""
        if masked:
            mask = self._recv_exact(conn, 4)
            if len(mask) < 4:
                return None
        payload = self._recv_exact(conn, length) if length > 0 else b""
        if masked and payload:
            payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))
        return opcode, payload

    def _send_frame(self, conn: socket.socket, payload: Union[str, bytes], opcode: int = 0x1) -> bool:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        header = bytearray()
        header.append(0x80 | opcode)
        n = len(payload)
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header += struct.pack(">H", n)
        else:
            header.append(127)
            header += struct.pack(">Q", n)
        try:
            conn.sendall(bytes(header) + payload)
            return True
        except Exception:
            return False

    # ---------- 连接管理 ----------
    def _handle_connection(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        headers = self._handshake(conn)
        if headers is None:
            try:
                conn.close()
            except Exception:
                pass
            return
        cid = uuid.uuid4().hex
        ws_conn = WSConnection(conn, cid, headers.get("__path__", "/"),
                               addr[0] if addr else "")
        with self._lock:
            self.connections[cid] = ws_conn
            self._stats["total_connections"] += 1
            self._stats["current_connections"] = len(self.connections)
        if self.on_connect is not None:
            try:
                self.on_connect(ws_conn)
            except Exception:
                pass

        try:
            while self._running and ws_conn.alive:
                frame = self._recv_frame(conn)
                if frame is None:
                    break
                opcode, payload = frame
                if opcode == 0x8:  # close
                    break
                elif opcode == 0x9:  # ping
                    self._send_frame(conn, payload, opcode=0xA)
                    continue
                elif opcode == 0xA:  # pong
                    continue
                elif opcode == 0x1:  # text
                    text = payload.decode("utf-8", errors="replace")
                    with self._lock:
                        self._stats["total_messages"] += 1
                    ws_conn.messages_received += 1
                    if self.on_message is not None:
                        try:
                            self.on_message(ws_conn, text)
                        except Exception:
                            pass
                    self._route_message(ws_conn, text)
        except Exception:
            pass
        finally:
            self._disconnect(ws_conn)

    def _disconnect(self, ws_conn: WSConnection) -> None:
        ws_conn.alive = False
        with self._lock:
            self.connections.pop(ws_conn.id, None)
            for room in list(ws_conn.rooms):
                self.rooms.get(room, set()).discard(ws_conn.id)
            self._stats["current_connections"] = len(self.connections)
        if self.on_disconnect is not None:
            try:
                self.on_disconnect(ws_conn)
            except Exception:
                pass
        try:
            ws_conn.conn.close()
        except Exception:
            pass

    # ---------- 消息路由 / 流式推理 ----------
    def _route_message(self, ws_conn: WSConnection, text: str) -> None:
        """根据路径路由消息, /ws/chat 触发流式推理"""
        if ws_conn.path.startswith("/ws/chat"):
            self._stream_inference(ws_conn, text)
            return
        # 默认回显
        self.send(ws_conn, json.dumps({"type": "echo", "data": text}, ensure_ascii=False))

    def _stream_inference(self, ws_conn: WSConnection, prompt: str) -> None:
        if self.inference_fn is not None:
            try:
                result = self.inference_fn(prompt)
                if isinstance(result, str):
                    tokens = result.split()
                else:
                    tokens = [json.dumps(result, ensure_ascii=False)]
            except Exception as e:
                tokens = [f"[错误] {e}"]
        else:
            reply = "灵元流式推理: " + prompt[::-1][:32]
            tokens = reply.split()
        for i, tok in enumerate(tokens):
            if not ws_conn.alive:
                break
            self.send(ws_conn, json.dumps(
                {"type": "token", "index": i, "content": tok + " "}, ensure_ascii=False))
            time.sleep(0.02)
        self.send(ws_conn, json.dumps({"type": "done"}, ensure_ascii=False))

    # ---------- 发送 / 广播 / 房间 ----------
    def send(self, ws_conn: WSConnection, message: str) -> bool:
        ok = self._send_frame(ws_conn.conn, message, opcode=0x1)
        if ok:
            ws_conn.messages_sent += 1
        return ok

    def broadcast(self, message: str) -> int:
        """广播给所有连接"""
        with self._lock:
            conns = list(self.connections.values())
        sent = 0
        for c in conns:
            if self.send(c, message):
                sent += 1
        with self._lock:
            self._stats["total_broadcasts"] += 1
        return sent

    def join_room(self, ws_conn: WSConnection, room: str) -> None:
        with self._lock:
            self.rooms.setdefault(room, set()).add(ws_conn.id)
            ws_conn.rooms.add(room)

    def leave_room(self, ws_conn: WSConnection, room: str) -> None:
        with self._lock:
            self.rooms.get(room, set()).discard(ws_conn.id)
            ws_conn.rooms.discard(room)

    def broadcast_to_room(self, room: str, message: str) -> int:
        with self._lock:
            ids = list(self.rooms.get(room, set()))
            conns = [self.connections[i] for i in ids if i in self.connections]
        sent = 0
        for c in conns:
            if self.send(c, message):
                sent += 1
        return sent

    # ---------- 心跳 ----------
    def _heartbeat_loop(self) -> None:
        while self._running:
            time.sleep(15)
            with self._lock:
                conns = list(self.connections.values())
            for c in conns:
                if c.alive:
                    if not self._send_frame(c.conn, b"", opcode=0x9):
                        self._disconnect(c)

    # ---------- 启停 ----------
    def start(self, port: int, block: bool = False) -> None:
        self._stats["start_time"] = time.time()
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, port))
        self._sock.listen(128)

        def _accept_loop():
            while self._running:
                try:
                    conn, addr = self._sock.accept()
                except Exception:
                    break
                conn.settimeout(None)
                t = threading.Thread(target=self._handle_connection, args=(conn, addr), daemon=True)
                t.start()

        if block:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
            _accept_loop()
        else:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
            self._thread = threading.Thread(target=_accept_loop, daemon=True)
            self._thread.start()

    def serve_forever(self, port: int) -> None:
        self.start(port, block=True)

    def stop(self) -> None:
        self._running = False
        with self._lock:
            conns = list(self.connections.values())
        for c in conns:
            self._disconnect(c)
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    # ---------- 统计与仪表盘 ----------
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
            rooms = {r: len(v) for r, v in self.rooms.items()}
        return {
            "host": self.host,
            "running": self._running,
            "total_connections": stats["total_connections"],
            "current_connections": stats["current_connections"],
            "total_messages": stats["total_messages"],
            "total_broadcasts": stats["total_broadcasts"],
            "rooms": rooms,
            "uptime": round(time.time() - (stats["start_time"] or time.time()), 2),
        }

    def get_dashboard(self) -> str:
        s = self.get_stats()
        lines = [
            "========== WebSocketServer 仪表盘 ==========",
            f"  监听地址:       {s['host']}",
            f"  运行状态:       {'运行中' if s['running'] else '已停止'}",
            f"  累计连接数:     {s['total_connections']}",
            f"  当前连接数:     {s['current_connections']}",
            f"  累计消息数:     {s['total_messages']}",
            f"  累计广播数:     {s['total_broadcasts']}",
            f"  房间列表:       {s['rooms']}",
            f"  运行时长:       {s['uptime']} s",
            "============================================",
        ]
        return "\n".join(lines)


# ============================================================
# #39 GRPCService - gRPC服务定义
# ============================================================

@dataclass
class GenerateRequest:
    """生成请求消息"""
    model: str = "lingyuan-1.0"
    prompt: str = ""
    max_tokens: int = 128
    temperature: float = 1.0
    stream: bool = False


@dataclass
class GenerateResponse:
    """生成响应消息"""
    id: str = ""
    model: str = ""
    text: str = ""
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class EmbedRequest:
    """嵌入请求消息"""
    model: str = "lingyuan-embed"
    input: str = ""


@dataclass
class EmbedResponse:
    """嵌入响应消息"""
    model: str = ""
    embedding: List[float] = field(default_factory=list)
    tokens: int = 0


class GRPCService:
    """gRPC 服务 (简化实现)

    生成 .proto 内容; 使用简化的长度前缀 + 字段类型二进制编解码;
    提供服务端 start(port) / 客户端 call(method, request);
    支持认证/日志/超时拦截器与健康检查协议.
    """

    PROTO_CONTENT = """\
syntax = "proto3";

package lingyuan;

// 推理服务
service InferenceService {
  rpc Generate (GenerateRequest) returns (GenerateResponse);
  rpc StreamGenerate (GenerateRequest) returns (stream GenerateResponse);
  rpc Embed (EmbedRequest) returns (EmbedResponse);
  rpc BatchGenerate (BatchGenerateRequest) returns (BatchGenerateResponse);
}

message GenerateRequest {
  string model = 1;
  string prompt = 2;
  int32 max_tokens = 3;
  float temperature = 4;
  bool stream = 5;
}

message GenerateResponse {
  string id = 1;
  string model = 2;
  string text = 3;
  string finish_reason = 4;
  int32 prompt_tokens = 5;
  int32 completion_tokens = 6;
}

message EmbedRequest {
  string model = 1;
  string input = 2;
}

message EmbedResponse {
  string model = 1;
  repeated float embedding = 2;
  int32 tokens = 3;
}

message BatchGenerateRequest {
  repeated GenerateRequest items = 1;
}

message BatchGenerateResponse {
  repeated GenerateResponse items = 1;
}

message HealthCheckRequest { string service = 1; }
message HealthCheckResponse { string status = 1; }

service Health {
  rpc Check (HealthCheckRequest) returns (HealthCheckResponse);
}
"""

    # 类型标签: 0=int64 1=double 2=string 3=bool 4=bytes 5=list(json)
    _TYPE_INT = 0
    _TYPE_FLOAT = 1
    _TYPE_STR = 2
    _TYPE_BOOL = 3
    _TYPE_BYTES = 4
    _TYPE_LIST = 5

    def __init__(self, service_name: str = "lingyuan.InferenceService"):
        self.service_name = service_name
        self.handlers: Dict[str, Callable] = {}
        self.interceptors: List[Callable] = []
        self.auth_token: Optional[str] = None
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._stats = {
            "total_calls": 0,
            "total_errors": 0,
            "method_counts": {},
            "start_time": None,
        }
        self._register_default_handlers()

    # ---------- proto ----------
    def get_proto(self) -> str:
        """返回 .proto 文件内容"""
        return self.PROTO_CONTENT

    def save_proto(self, path: str) -> str:
        """保存 .proto 文件"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.PROTO_CONTENT)
        return path

    # ---------- 简化 protobuf 编解码 ----------
    def _encode_value(self, v: Any) -> Tuple[int, bytes]:
        if isinstance(v, bool):
            return self._TYPE_BOOL, (b"\x01" if v else b"\x00")
        if isinstance(v, int):
            return self._TYPE_INT, struct.pack(">q", v)
        if isinstance(v, float):
            return self._TYPE_FLOAT, struct.pack(">d", v)
        if isinstance(v, str):
            return self._TYPE_STR, v.encode("utf-8")
        if isinstance(v, bytes):
            return self._TYPE_BYTES, v
        if isinstance(v, (list, tuple)):
            return self._TYPE_LIST, json.dumps(list(v), ensure_ascii=False).encode("utf-8")
        if v is None:
            return self._TYPE_STR, b""
        return self._TYPE_STR, str(v).encode("utf-8")

    def encode_message(self, obj: Any) -> bytes:
        """将 dataclass 实例编码为简化 protobuf 二进制"""
        d = asdict(obj) if hasattr(obj, "__dataclass_fields__") else dict(obj)
        out = bytearray()
        for idx, (_k, v) in enumerate(d.items(), start=1):
            if v is None:
                continue
            wtype, data = self._encode_value(v)
            out.append(idx & 0xFF)
            out.append(wtype & 0xFF)
            out += struct.pack(">I", len(data))
            out += data
        return bytes(out)

    def decode_message(self, data: bytes) -> List[Any]:
        """解码简化 protobuf 二进制为有序值列表"""
        values: List[Any] = []
        i = 0
        n = len(data)
        while i + 2 <= n:
            _field_id = data[i]
            i += 1
            wtype = data[i]
            i += 1
            if i + 4 > n:
                break
            length = struct.unpack(">I", data[i:i + 4])[0]
            i += 4
            raw = data[i:i + length]
            i += length
            if wtype == self._TYPE_INT:
                values.append(struct.unpack(">q", raw)[0])
            elif wtype == self._TYPE_FLOAT:
                values.append(struct.unpack(">d", raw)[0])
            elif wtype == self._TYPE_STR:
                values.append(raw.decode("utf-8", errors="replace"))
            elif wtype == self._TYPE_BOOL:
                values.append(raw == b"\x01")
            elif wtype == self._TYPE_BYTES:
                values.append(raw)
            elif wtype == self._TYPE_LIST:
                values.append(json.loads(raw.decode("utf-8", errors="replace")))
            else:
                values.append(raw.decode("utf-8", errors="replace"))
        return values

    def decode_to(self, data: bytes, cls: type) -> Any:
        """解码为指定 dataclass 实例"""
        values = self.decode_message(data)
        field_names = list(getattr(cls, "__dataclass_fields__", {}).keys())
        kwargs = {}
        for idx, fname in enumerate(field_names):
            if idx < len(values):
                kwargs[fname] = values[idx]
        return cls(**kwargs)

    # ---------- 帧编解码 (长度前缀) ----------
    @staticmethod
    def _pack_frame(method: str, payload: bytes) -> bytes:
        method_bytes = method.encode("utf-8")
        return (struct.pack(">H", len(method_bytes)) + method_bytes +
                struct.pack(">I", len(payload)) + payload)

    @staticmethod
    def _unpack_frame(buf: bytes) -> Tuple[str, bytes, int]:
        if len(buf) < 2:
            return "", b"", 0
        mlen = struct.unpack(">H", buf[0:2])[0]
        if len(buf) < 2 + mlen + 4:
            return "", b"", 0
        method = buf[2:2 + mlen].decode("utf-8", errors="replace")
        off = 2 + mlen
        plen = struct.unpack(">I", buf[off:off + 4])[0]
        off += 4
        if len(buf) < off + plen:
            return "", b"", 0
        payload = buf[off:off + plen]
        return method, payload, off + plen

    # ---------- 默认 handler ----------
    def _register_default_handlers(self) -> None:
        self.handlers["InferenceService/Generate"] = self._handle_generate
        self.handlers["InferenceService/StreamGenerate"] = self._handle_generate
        self.handlers["InferenceService/Embed"] = self._handle_embed
        self.handlers["InferenceService/BatchGenerate"] = self._handle_batch
        self.handlers["Health/Check"] = self._handle_health

    def _handle_generate(self, request: GenerateRequest) -> GenerateResponse:
        text = f"[grpc] 已为模型 {request.model} 生成回复, prompt 长度={len(request.prompt)}"
        pt = max(1, len(request.prompt))
        ct = max(1, len(text))
        return GenerateResponse(
            id=f"grpc-{uuid.uuid4().hex[:16]}", model=request.model,
            text=text, finish_reason="stop", prompt_tokens=pt, completion_tokens=ct,
        )

    def _handle_embed(self, request: EmbedRequest) -> EmbedResponse:
        h = hashlib.sha256(request.input.encode("utf-8")).digest()
        vec = [round((b / 255.0) * 2 - 1, 6) for b in h[:16]]
        return EmbedResponse(model=request.model, embedding=vec,
                             tokens=max(1, len(request.input)))

    def _handle_batch(self, request: Any) -> Any:
        return request  # 简化: 原样返回 (实际可拆分调用)

    def _handle_health(self, _request: Any) -> Any:
        return {"status": "SERVING"}

    def register_handler(self, method: str, handler: Callable) -> None:
        self.handlers[method] = handler

    # ---------- 拦截器 ----------
    def add_interceptor(self, interceptor: Callable) -> None:
        self.interceptors.append(interceptor)

    def auth_interceptor(self, token: str) -> Callable:
        """认证拦截器工厂"""
        def _ic(method: str, request: Any) -> Optional[Any]:
            if token and getattr(request, "_auth_token", None) != token:
                return {"error": "unauthenticated"}
            return None
        return _ic

    def log_interceptor(self) -> Callable:
        def _ic(method: str, request: Any) -> Optional[Any]:
            return None
        return _ic

    def timeout_interceptor(self, seconds: float) -> Callable:
        def _ic(method: str, request: Any) -> Optional[Any]:
            return None  # 超时由 socket settimeout 实现
        return _ic

    # ---------- 服务端 ----------
    def _dispatch(self, method: str, payload: bytes) -> Tuple[int, bytes]:
        handler = self.handlers.get(method)
        if handler is None:
            err = json.dumps({"error": f"未实现方法: {method}"}).encode("utf-8")
            return 12, err  # UNIMPLEMENTED
        # 确定请求类型
        req_cls = self._request_class_for(method)
        try:
            request = self.decode_to(payload, req_cls) if req_cls else payload
        except Exception:
            request = payload
        # 拦截器
        for ic in self.interceptors:
            try:
                res = ic(method, request)
                if res is not None:
                    return 7, json.dumps(res, ensure_ascii=False, default=str).encode("utf-8")
            except Exception as e:
                return 13, json.dumps({"error": str(e)}).encode("utf-8")
        try:
            response = handler(request)
            if isinstance(response, dict):
                return 0, json.dumps(response, ensure_ascii=False, default=str).encode("utf-8")
            return 0, self.encode_message(response)
        except Exception as e:
            return 13, json.dumps({"error": str(e)}).encode("utf-8")

    @staticmethod
    def _request_class_for(method: str) -> Optional[type]:
        mapping = {
            "InferenceService/Generate": GenerateRequest,
            "InferenceService/StreamGenerate": GenerateRequest,
            "InferenceService/Embed": EmbedRequest,
            "InferenceService/BatchGenerate": dict,
            "Health/Check": dict,
        }
        return mapping.get(method)

    def _handle_client(self, conn: socket.socket) -> None:
        conn.settimeout(30)
        try:
            buf = b""
            while self._running:
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                buf += chunk
                method, payload, consumed = self._unpack_frame(buf)
                if consumed == 0:
                    continue
                buf = buf[consumed:]
                status, resp_payload = self._dispatch(method, payload)
                with self._lock:
                    self._stats["total_calls"] += 1
                    self._stats["method_counts"][method] = self._stats["method_counts"].get(method, 0) + 1
                    if status != 0:
                        self._stats["total_errors"] += 1
                conn.sendall(struct.pack(">I", status) + struct.pack(">I", len(resp_payload)) + resp_payload)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def start(self, port: int, block: bool = False) -> None:
        self._stats["start_time"] = time.time()
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.listen(64)

        def _loop():
            while self._running:
                try:
                    conn, _ = self._sock.accept()
                except Exception:
                    break
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

        if block:
            _loop()
        else:
            self._thread = threading.Thread(target=_loop, daemon=True)
            self._thread.start()

    def serve_forever(self, port: int) -> None:
        self.start(port, block=True)

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    # ---------- 客户端 (模拟) ----------
    def call(self, method: str, request: Any, host: str = "127.0.0.1",
             port: int = 0, timeout: float = 10.0) -> Any:
        """客户端调用: 本地直连或远程. port>0 时走网络, 否则本地直接 dispatch."""
        if method not in self.handlers:
            raise ValueError(f"未知方法: {method}")
        payload = self.encode_message(request) if hasattr(request, "__dataclass_fields__") else \
            (json.dumps(request, ensure_ascii=False, default=str).encode("utf-8"))

        if port and port > 0:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                sock.connect((host, port))
                sock.sendall(self._pack_frame(method, payload))
                hdr = self._recv_exact_sock(sock, 8)
                if len(hdr) < 8:
                    raise IOError("响应头部不完整")
                status = struct.unpack(">I", hdr[0:4])[0]
                plen = struct.unpack(">I", hdr[4:8])[0]
                resp_payload = self._recv_exact_sock(sock, plen)
            finally:
                sock.close()
            if status != 0:
                return json.loads(resp_payload.decode("utf-8", errors="replace"))
            resp_cls = self._response_class_for(method)
            if resp_cls:
                return self.decode_to(resp_payload, resp_cls)
            return json.loads(resp_payload.decode("utf-8", errors="replace"))
        else:
            status, resp_payload = self._dispatch(method, payload)
            if status != 0:
                return json.loads(resp_payload.decode("utf-8", errors="replace"))
            resp_cls = self._response_class_for(method)
            if resp_cls:
                return self.decode_to(resp_payload, resp_cls)
            return json.loads(resp_payload.decode("utf-8", errors="replace"))

    @staticmethod
    def _recv_exact_sock(sock: socket.socket, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                break
            data += chunk
        return data

    @staticmethod
    def _response_class_for(method: str) -> Optional[type]:
        # 仅 dataclass 响应使用二进制解码; dict/未知响应走 JSON 解析
        mapping = {
            "InferenceService/Generate": GenerateResponse,
            "InferenceService/StreamGenerate": GenerateResponse,
            "InferenceService/Embed": EmbedResponse,
        }
        return mapping.get(method)

    def health_check(self, host: str = "127.0.0.1", port: int = 0) -> str:
        """gRPC 健康检查协议"""
        result = self.call("Health/Check", {"service": self.service_name}, host=host, port=port)
        if isinstance(result, dict):
            return result.get("status", "UNKNOWN")
        return getattr(result, "status", "UNKNOWN")

    # ---------- 统计与仪表盘 ----------
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
            method_counts = dict(stats.pop("method_counts", {}))
        return {
            "service_name": self.service_name,
            "running": self._running,
            "handlers": list(self.handlers.keys()),
            "interceptors": len(self.interceptors),
            "total_calls": stats.get("total_calls", 0),
            "total_errors": stats.get("total_errors", 0),
            "method_counts": method_counts,
            "uptime": round(time.time() - (stats.get("start_time") or time.time()), 2),
        }

    def get_dashboard(self) -> str:
        s = self.get_stats()
        lines = [
            "========== GRPCService 仪表盘 ==========",
            f"  服务名:         {s['service_name']}",
            f"  运行状态:       {'运行中' if s['running'] else '已停止'}",
            f"  已注册方法:     {s['handlers']}",
            f"  拦截器数量:     {s['interceptors']}",
            f"  总调用数:       {s['total_calls']}",
            f"  错误调用数:     {s['total_errors']}",
            f"  方法调用分布:   {s['method_counts']}",
            f"  运行时长:       {s['uptime']} s",
            "========================================",
        ]
        return "\n".join(lines)


# ============================================================
# #40 APIDocGenerator - API文档生成
# ============================================================

class APIDocGenerator:
    """API 文档生成器

    扫描 HTTPServer 注册的路由, 生成 OpenAPI 3.0 规范 (JSON / 简化 YAML),
    自动推断 schema 与示例, 并生成可浏览的 Swagger UI HTML 页面.
    """

    def __init__(self, server: Optional[HTTPServer] = None, title: str = "Lingyuan API",
                 version: str = "1.0.0", description: str = "灵元大模型 API 服务"):
        self.server = server if server is not None else HTTPServer()
        self.title = title
        self.version = version
        self.description = description
        self._stats = {"generated": 0, "yaml_generated": 0, "ui_generated": 0}

    # ---------- schema 推断 ----------
    def _schema_for_type(self, tp: Any) -> Dict[str, Any]:
        if tp in (int, "int", "integer"):
            return {"type": "integer"}
        if tp in (float, "float", "number"):
            return {"type": "number"}
        if tp in (str, "string"):
            return {"type": "string"}
        if tp in (bool, "boolean"):
            return {"type": "boolean"}
        if tp in (list, List, "array"):
            return {"type": "array", "items": {"type": "string"}}
        if tp in (dict, Dict, "object"):
            return {"type": "object"}
        if isinstance(tp, str):
            return {"type": tp}
        return {"type": "string"}

    def _schema_for_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        props = {}
        required = []
        for k, v in d.items():
            props[k] = self._schema_for_type(v)
            required.append(k)
        schema = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return schema

    def _example_for_schema(self, schema: Dict[str, Any]) -> Any:
        t = schema.get("type", "string")
        if t == "integer":
            return 0
        if t == "number":
            return 0.0
        if t == "boolean":
            return False
        if t == "array":
            return [self._example_for_schema(schema.get("items", {"type": "string"}))]
        if t == "object":
            ex: Dict[str, Any] = {}
            for k, v in schema.get("properties", {}).items():
                ex[k] = self._example_for_schema(v)
            return ex
        return "string"

    def _path_params(self, path: str) -> List[Dict[str, Any]]:
        params = []
        segs = [s for s in path.split("/") if s != ""]
        for seg in segs:
            if seg.startswith("{") and seg.endswith("}"):
                params.append({
                    "name": seg[1:-1], "in": "path", "required": True,
                    "schema": {"type": "string"},
                })
        return params

    def _query_params(self, schema: Optional[Dict]) -> List[Dict[str, Any]]:
        if not schema:
            return []
        params = []
        for k, v in schema.items():
            params.append({"name": k, "in": "query", "required": False,
                           "schema": self._schema_for_type(v)})
        return params

    # ---------- OpenAPI 生成 ----------
    def generate_openapi(self) -> Dict[str, Any]:
        """生成 OpenAPI 3.0 规范字典"""
        paths: Dict[str, Dict[str, Any]] = {}
        tags_set: Dict[str, Dict[str, str]] = {}
        for route in self.server.routes:
            # OpenAPI 路径使用 {param} 形式 (与路由一致)
            api_path = route.path
            item: Dict[str, Any] = {
                "summary": route.summary or "",
                "description": route.description or "",
                "operationId": f"{route.method.lower()}_{route.path.replace('/', '_').strip('_')}",
                "tags": route.tags or ["default"],
            }
            params = self._path_params(route.path)
            if route.method in ("GET", "DELETE"):
                params += self._query_params(route.request_schema)
            if params:
                item["parameters"] = params
            if route.method in ("POST", "PUT", "PATCH") and route.request_schema:
                req_schema = self._schema_for_dict(route.request_schema) \
                    if isinstance(route.request_schema, dict) else {"type": "object"}
                item["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": req_schema,
                            "example": self._example_for_schema(req_schema),
                        }
                    },
                }
            # 响应
            resp_schema = self._schema_for_dict(route.response_schema) \
                if isinstance(route.response_schema, dict) else {"type": "object"}
            item["responses"] = {
                "200": {
                    "description": "成功响应",
                    "content": {"application/json": {
                        "schema": resp_schema,
                        "example": self._example_for_schema(resp_schema),
                    }},
                },
                "400": {"description": "请求错误",
                        "content": {"application/json": {"schema": {"type": "object"}}}},
                "401": {"description": "未授权",
                        "content": {"application/json": {"schema": {"type": "object"}}}},
                "404": {"description": "未找到",
                        "content": {"application/json": {"schema": {"type": "object"}}}},
            }
            if route.auth_required:
                item["security"] = [{"BearerAuth": []}]
            paths.setdefault(api_path, {})[route.method.lower()] = item
            for t in (route.tags or ["default"]):
                tags_set.setdefault(t, {"name": t})

        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description,
            },
            "servers": [{"url": "/", "description": "本地服务"}],
            "tags": list(tags_set.values()),
            "paths": paths,
            "components": {
                "securitySchemes": {
                    "BearerAuth": {"type": "http", "scheme": "bearer"}
                }
            },
        }
        self._stats["generated"] += 1
        return spec

    def to_json(self, indent: int = 2) -> str:
        """输出 JSON 字符串"""
        return json.dumps(self.generate_openapi(), ensure_ascii=False, indent=indent)

    def to_yaml(self) -> str:
        """输出简化 YAML 字符串"""
        self._stats["yaml_generated"] += 1
        return self._dict_to_yaml(self.generate_openapi(), 0)

    def _dict_to_yaml(self, obj: Any, indent: int) -> str:
        pad = "  " * indent
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            lines = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)) and v:
                    lines.append(f"{pad}{k}:")
                    lines.append(self._dict_to_yaml(v, indent + 1))
                elif isinstance(v, list):
                    if not v:
                        lines.append(f"{pad}{k}: []")
                    else:
                        lines.append(f"{pad}{k}:")
                        for item in v:
                            if isinstance(item, (dict, list)):
                                lines.append(self._dict_to_yaml({"-": item}, indent + 1))
                            else:
                                lines.append(f"{'  ' * (indent + 1)}- {self._yaml_scalar(item)}")
                else:
                    lines.append(f"{pad}{k}: {self._yaml_scalar(v)}")
            return "\n".join(lines)
        if isinstance(obj, list):
            if not obj:
                return "[]"
            lines = []
            for item in obj:
                if isinstance(item, (dict, list)):
                    sub = self._dict_to_yaml(item, indent + 1)
                    lines.append(f"{pad}- {sub.lstrip()}")
                else:
                    lines.append(f"{pad}- {self._yaml_scalar(item)}")
            return "\n".join(lines)
        return f"{pad}{self._yaml_scalar(obj)}"

    @staticmethod
    def _yaml_scalar(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v)
        if s == "" or any(c in s for c in [":", "{", "}", "[", "]", ",", "&", "*", "#", "?", "|", ">", "'", '"', "@", "`"]):
            return f'"{s}"'
        return s

    # ---------- Swagger UI ----------
    def generate_swagger_ui(self, spec_url: str = "/openapi.json") -> str:
        """生成 Swagger UI HTML 页面 (使用 CDN 加载官方 swagger-ui)"""
        self._stats["ui_generated"] += 1
        html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style> body {{ margin: 0; }} </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = function() {{
      window.ui = SwaggerUIBundle({{
        url: "{spec_url}",
        dom_id: "#swagger-ui",
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis],
        layout: "BaseLayout",
      }});
    }};
  </script>
</body>
</html>""".format(title=self.title, spec_url=spec_url)
        return html

    def save_docs(self, output_dir: str) -> Dict[str, str]:
        """保存文档到目录: openapi.json / openapi.yaml / index.html"""
        os.makedirs(output_dir, exist_ok=True)
        files = {
            "openapi.json": self.to_json(),
            "openapi.yaml": self.to_yaml(),
            "index.html": self.generate_swagger_ui("/openapi.json"),
        }
        saved = {}
        for name, content in files.items():
            path = os.path.join(output_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            saved[name] = path
        return saved

    # ---------- 统计与仪表盘 ----------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "version": self.version,
            "route_count": len(self.server.routes),
            "generated_count": self._stats["generated"],
            "yaml_generated_count": self._stats["yaml_generated"],
            "ui_generated_count": self._stats["ui_generated"],
        }

    def get_dashboard(self) -> str:
        s = self.get_stats()
        lines = [
            "========== APIDocGenerator 仪表盘 ==========",
            f"  文档标题:       {s['title']}",
            f"  API 版本:       {s['version']}",
            f"  路由数量:       {s['route_count']}",
            f"  OpenAPI 生成:   {s['generated_count']} 次",
            f"  YAML 生成:      {s['yaml_generated_count']} 次",
            f"  SwaggerUI 生成: {s['ui_generated_count']} 次",
            "============================================",
        ]
        return "\n".join(lines)


# ============================================================
# #41 SDKGenerator - SDK自动生成
# ============================================================

class SDKGenerator:
    """SDK 自动生成器

    从 HTTPServer 注册的路由生成多语言客户端 SDK:
      Python  - 基于 urllib 的客户端类 + dataclass 模型
      JavaScript - 基于 fetch 的客户端 + JSDoc 类型
      Go      - 基于 net/http 的客户端 (简化版)
    输出包含 __init__.py / client / models / README / 示例.
    """

    def __init__(self, server: Optional[HTTPServer] = None, base_url: str = "http://localhost:8000",
                 sdk_name: str = "lingyuan_sdk"):
        self.server = server if server is not None else HTTPServer()
        self.base_url = base_url.rstrip("/")
        self.sdk_name = sdk_name
        self._stats = {"python": 0, "javascript": 0, "go": 0}

    # ---------- 路由工具 ----------
    def _method_name(self, route: Route) -> str:
        segs = [s for s in route.path.split("/") if s != "" and not (s.startswith("{") and s.endswith("}"))]
        base = "_".join(segs) or "root"
        return f"{route.method.lower()}_{base}".replace("-", "_").replace(".", "_")

    def _py_path(self, route: Route) -> str:
        # 将 {param} 转为 Python f-string 占位
        parts = []
        for seg in route.path.split("/"):
            if seg == "":
                continue
            if seg.startswith("{") and seg.endswith("}"):
                parts.append("{" + seg[1:-1] + "}")
            else:
                parts.append(seg)
        return "/" + "/".join(parts)

    def _path_params_of(self, route: Route) -> List[str]:
        params = []
        for seg in route.path.split("/"):
            if seg.startswith("{") and seg.endswith("}"):
                params.append(seg[1:-1])
        return params

    # ---------- Python SDK ----------
    def generate_python(self) -> Dict[str, str]:
        """生成 Python SDK 文件集合"""
        self._stats["python"] += 1
        client_lines = [
            '"""灵元大模型 Python SDK - 自动生成"""',
            "import json",
            "import urllib.request",
            "import urllib.error",
            "",
            "",
            "class LingyuanClient:",
            f'    """灵元 API 客户端 (base_url={self.base_url})"""',
            "",
            f"    def __init__(self, base_url={self.base_url!r}, api_key=None):",
            "        self.base_url = base_url.rstrip('/')",
            "        self.api_key = api_key",
            "        self._headers = {'Content-Type': 'application/json'}",
            "        if api_key:",
            "            self._headers['Authorization'] = f'Bearer {api_key}'",
            "",
            "    def _request(self, method, path, body=None, params=None):",
            "        url = self.base_url + path",
            "        if params:",
            "            from urllib.parse import urlencode",
            "            url += '?' + urlencode(params)",
            "        data = json.dumps(body).encode('utf-8') if body is not None else None",
            "        req = urllib.request.Request(url, data=data, method=method, headers=self._headers)",
            "        try:",
            "            with urllib.request.urlopen(req) as resp:",
            "                return json.loads(resp.read().decode('utf-8'))",
            "        except urllib.error.HTTPError as e:",
            "            return json.loads(e.read().decode('utf-8'))",
            "",
        ]
        for route in self.server.routes:
            if route.path == "/health":
                continue
            mname = self._method_name(route)
            pparams = self._path_params_of(route)
            sig_parts = ["self"] + [f"{p}=None" for p in pparams]
            if route.method in ("POST", "PUT", "PATCH"):
                sig_parts.append("body=None")
            sig = ", ".join(sig_parts)
            client_lines.append(f"    def {mname}({sig}):")
            client_lines.append(f'        """{route.summary or route.path}"""')
            path_expr = self._py_path(route)
            if pparams:
                client_lines.append(f"        path = f'{path_expr}'")
            else:
                client_lines.append(f"        path = '{path_expr}'")
            call_args = [f"'{route.method}'", "path"]
            if route.method in ("POST", "PUT", "PATCH"):
                call_args.append("body=body")
            client_lines.append(f"        return self._request({', '.join(call_args)})")
            client_lines.append("")

        client_lines.append("")
        client_lines.append("# 使用示例")
        client_lines.append("# client = LingyuanClient(api_key='sk-lingyuan')")
        client_lines.append("# print(client.get_v1_models())")

        models_lines = [
            '"""灵元大模型 SDK 模型定义 - 自动生成"""',
            "from dataclasses import dataclass, field",
            "from typing import List, Optional, Any",
            "",
            "",
        ]
        for route in self.server.routes:
            if isinstance(route.response_schema, dict):
                cls_name = "".join(p.capitalize() for p in self._method_name(route).split("_")) + "Response"
                models_lines.append("@dataclass")
                models_lines.append(f"class {cls_name}:")
                if not route.response_schema:
                    models_lines.append("    pass")
                else:
                    for k, v in route.response_schema.items():
                        py_t = self._py_type(v)
                        models_lines.append(f"    {k}: {py_t} = None")
                models_lines.append("")

        init_lines = [
            '"""灵元大模型 Python SDK"""',
            "from .client import LingyuanClient",
            "from . import models",
            "",
            "__all__ = ['LingyuanClient', 'models']",
            "__version__ = '1.0.0'",
            "",
        ]

        readme = self._python_readme()
        return {
            "__init__.py": "\n".join(init_lines),
            "client.py": "\n".join(client_lines),
            "models.py": "\n".join(models_lines),
            "README.md": readme,
        }

    def _py_type(self, t: Any) -> str:
        if t in (int, "int", "integer"):
            return "int"
        if t in (float, "float", "number"):
            return "float"
        if t in (bool, "boolean"):
            return "bool"
        if t in (list, List, "array"):
            return "List[Any]"
        if t in (dict, Dict, "object"):
            return "Dict[str, Any]"
        return "str"

    def _python_readme(self) -> str:
        lines = [
            f"# {self.sdk_name} (Python)",
            "",
            "灵元大模型 Python SDK (自动生成).",
            "",
            "## 安装",
            "",
            "将本目录放入项目, 或添加到 PYTHONPATH.",
            "",
            "## 快速开始",
            "",
            "```python",
            "from lingyuan_sdk import LingyuanClient",
            "",
            "client = LingyuanClient(base_url='http://localhost:8000', api_key='sk-lingyuan')",
            "print(client.get_v1_models())",
            "```",
            "",
            "## 依赖",
            "",
            "仅依赖 Python 3.7+ 标准库.",
            "",
        ]
        return "\n".join(lines)

    # ---------- JavaScript SDK ----------
    def generate_javascript(self) -> Dict[str, str]:
        """生成 JavaScript SDK 文件集合"""
        self._stats["javascript"] += 1
        lines = [
            f"// {self.sdk_name} - JavaScript SDK (自动生成)",
            f"// 基于 fetch 封装, base_url: {self.base_url}",
            "",
            f"const BASE_URL = '{self.base_url}';",
            "",
            "class LingyuanClient {",
            "  constructor({ apiKey } = {}) {",
            "    this.apiKey = apiKey || null;",
            "    this.headers = { 'Content-Type': 'application/json' };",
            "    if (this.apiKey) this.headers['Authorization'] = `Bearer ${this.apiKey}`;",
            "  }",
            "",
            "  async _request(method, path, body) {",
            "    const opts = { method, headers: { ...this.headers } };",
            "    if (body !== undefined && body !== null) opts.body = JSON.stringify(body);",
            "    const resp = await fetch(BASE_URL + path, opts);",
            "    return resp.json();",
            "  }",
            "",
        ]
        for route in self.server.routes:
            if route.path == "/health":
                continue
            mname = self._method_name(route)
            pparams = self._path_params_of(route)
            args = pparams[:]
            if route.method in ("POST", "PUT", "PATCH"):
                args.append("body")
            arg_str = ", ".join(args)
            # 构造 path 模板字符串
            tmpl_parts = []
            for seg in route.path.split("/"):
                if seg == "":
                    continue
                if seg.startswith("{") and seg.endswith("}"):
                    tmpl_parts.append("${" + seg[1:-1] + "}")
                else:
                    tmpl_parts.append(seg)
            tmpl = "/" + "/".join(tmpl_parts)
            lines.append(f"  async {mname}({arg_str}) {{")
            lines.append(f"    // {route.summary or route.path}")
            lines.append(f"    const path = `{tmpl}`;")
            call = "path" if route.method not in ("POST", "PUT", "PATCH") else "path, body"
            lines.append(f"    return this._request('{route.method}', {call});")
            lines.append("  }")
            lines.append("")

        lines.append("}")
        lines.append("")
        lines.append("if (typeof module !== 'undefined' && module.exports) module.exports = { LingyuanClient };")
        lines.append("// 使用示例:")
        lines.append("// const client = new LingyuanClient({ apiKey: 'sk-lingyuan' });")
        lines.append("// client.get_v1_models().then(console.log);")

        types_lines = [
            "// 类型定义 (JSDoc)",
            "",
            "/** @typedef {Object} ChatCompletionResponse",
            " * @property {string} id",
            " * @property {string} object",
            " * @property {Array} choices",
            " * @property {Object} usage",
            " */",
            "",
            "/** @typedef {Object} ModelList",
            " * @property {string} object",
            " * @property {Array} data",
            " */",
            "",
        ]

        readme = [
            f"# {self.sdk_name} (JavaScript)",
            "",
            "灵元大模型 JavaScript SDK (自动生成).",
            "",
            "## 快速开始",
            "",
            "```javascript",
            "const { LingyuanClient } = require('./client.js');",
            "const client = new LingyuanClient({ apiKey: 'sk-lingyuan' });",
            "client.get_v1_models().then(console.log);",
            "```",
            "",
            "## 依赖",
            "",
            "需要支持 fetch 的运行环境 (Node 18+ 或现代浏览器).",
            "",
        ]

        return {
            "client.js": "\n".join(lines),
            "types.js": "\n".join(types_lines),
            "README.md": "\n".join(readme),
        }

    # ---------- Go SDK ----------
    def generate_go(self) -> Dict[str, str]:
        """生成 Go SDK 文件集合"""
        self._stats["go"] += 1
        pkg = self.sdk_name.replace("-", "_")
        lines = [
            f"// Package {pkg} 灵元大模型 Go SDK (自动生成)",
            f"package {pkg}",
            "",
            "import (",
            "\t\"bytes\"",
            "\t\"encoding/json\"",
            "\t\"fmt\"",
            "\t\"io\"",
            "\t\"net/http\"",
            ")",
            "",
            f"const BaseURL = \"{self.base_url}\"",
            "",
            "// Client 灵元 API 客户端",
            "type Client struct {",
            "\tAPIKey string",
            "\tHTTP   *http.Client",
            "}",
            "",
            "// NewClient 创建客户端",
            "func NewClient(apiKey string) *Client {",
            "\treturn &Client{APIKey: apiKey, HTTP: http.DefaultClient}",
            "}",
            "",
            "func (c *Client) request(method, path string, body interface{}) (map[string]interface{}, error) {",
            "\tvar reader io.Reader",
            "\tif body != nil {",
            "\t\tb, _ := json.Marshal(body)",
            "\t\treader = bytes.NewReader(b)",
            "\t}",
            "\treq, err := http.NewRequest(method, BaseURL+path, reader)",
            "\tif err != nil {",
            "\t\treturn nil, err",
            "\t}",
            "\treq.Header.Set(\"Content-Type\", \"application/json\")",
            "\tif c.APIKey != \"\" {",
            "\t\treq.Header.Set(\"Authorization\", \"Bearer \"+c.APIKey)",
            "\t}",
            "\tresp, err := c.HTTP.Do(req)",
            "\tif err != nil {",
            "\t\treturn nil, err",
            "\t}",
            "\tdefer resp.Body.Close()",
            "\tvar result map[string]interface{}",
            "\tjson.NewDecoder(resp.Body).Decode(&result)",
            "\treturn result, nil",
            "}",
            "",
        ]
        for route in self.server.routes:
            if route.path == "/health":
                continue
            mname = self._method_name(route)
            pparams = self._path_params_of(route)
            # Go 方法签名
            go_args = ["c *Client"] + [self._go_type(p) + " " + p for p in pparams]
            has_body = route.method in ("POST", "PUT", "PATCH")
            if has_body:
                go_args.append("body map[string]interface{}")
            arg_str = ", ".join(go_args)
            # path 构造
            go_path = '""'
            if pparams:
                fmt_parts = []
                fmt_args = []
                for seg in route.path.split("/"):
                    if seg == "":
                        continue
                    if seg.startswith("{") and seg.endswith("}"):
                        fmt_parts.append("%s")
                        fmt_args.append(seg[1:-1])
                    else:
                        fmt_parts.append(seg)
                go_path = 'fmt.Sprintf("/' + "/".join(fmt_parts) + '", ' + ", ".join(fmt_args) + ")"
            else:
                tmpl_parts = []
                for seg in route.path.split("/"):
                    if seg == "":
                        continue
                    tmpl_parts.append(seg)
                go_path = '"' + "/" + "/".join(tmpl_parts) + '"'
            ret = "map[string]interface{}, error"
            lines.append(f"func ({', '.join(go_args)}) ({mname}) ({ret}) {{")
            call_args = [f'"{route.method}"', go_path]
            if has_body:
                call_args.append("body")
            lines.append(f"\treturn c.request({', '.join(call_args)})")
            lines.append("}")
            lines.append("")

        readme = [
            f"# {self.sdk_name} (Go)",
            "",
            "灵元大模型 Go SDK (自动生成).",
            "",
            "## 快速开始",
            "",
            "```go",
            f"package main",
            "",
            f'import "{pkg}"',
            f'import "fmt"',
            "",
            "func main() {",
            "\tclient := " + pkg + ".NewClient(\"sk-lingyuan\")",
            "\tres, _ := client.Get_v1_models()",
            "\tfmt.Println(res)",
            "}",
            "```",
            "",
            "## 依赖",
            "",
            "仅依赖 Go 标准库.",
            "",
        ]

        return {
            "client.go": "\n".join(lines),
            "README.md": "\n".join(readme),
        }

    @staticmethod
    def _go_type(name: str) -> str:
        return "string"

    # ---------- 保存 ----------
    def save_sdk(self, language: str, output_dir: str) -> Dict[str, str]:
        """保存指定语言的 SDK 到目录"""
        language = language.lower()
        if language == "python":
            files = self.generate_python()
        elif language in ("javascript", "js"):
            files = self.generate_javascript()
        elif language == "go":
            files = self.generate_go()
        else:
            raise ValueError(f"不支持的语言: {language}")
        out = os.path.join(output_dir, language)
        os.makedirs(out, exist_ok=True)
        saved = {}
        for name, content in files.items():
            path = os.path.join(out, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            saved[name] = path
        return saved

    def save_all(self, output_dir: str) -> Dict[str, Dict[str, str]]:
        """保存所有语言 SDK"""
        return {
            "python": self.save_sdk("python", output_dir),
            "javascript": self.save_sdk("javascript", output_dir),
            "go": self.save_sdk("go", output_dir),
        }

    # ---------- 统计与仪表盘 ----------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "sdk_name": self.sdk_name,
            "base_url": self.base_url,
            "route_count": len(self.server.routes),
            "python_generated": self._stats["python"],
            "javascript_generated": self._stats["javascript"],
            "go_generated": self._stats["go"],
            "supported_languages": ["python", "javascript", "go"],
        }

    def get_dashboard(self) -> str:
        s = self.get_stats()
        lines = [
            "========== SDKGenerator 仪表盘 ==========",
            f"  SDK 名称:       {s['sdk_name']}",
            f"  Base URL:       {s['base_url']}",
            f"  路由数量:       {s['route_count']}",
            f"  支持语言:       {s['supported_languages']}",
            f"  Python 生成:    {s['python_generated']} 次",
            f"  JavaScript 生成:{s['javascript_generated']} 次",
            f"  Go 生成:        {s['go_generated']} 次",
            "=========================================",
        ]
        return "\n".join(lines)
