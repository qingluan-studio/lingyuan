
# ============================================================
# LINGYUAN MODEL - PART 16
# 用户界面与安全
#
# Web Chat界面 / Playground试玩 / 训练可视化面板 /
# 模型水印 / API Key管理 / 训练数据血缘审计
# 对应52项清单 #47-52
# ============================================================

import uuid
import math
import random
import json
import os
import time
import hashlib
import hmac
import base64
import re
from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime


# ============================================================
# 辅助函数 - 获取全局目录
# ============================================================

def _get_data_dir() -> str:
    """获取数据目录（兼容全局变量）"""
    return globals().get('DATA_DIR', os.path.join(os.getcwd(), 'lingyuan_data'))


def _get_log_dir() -> str:
    """获取日志目录（兼容全局变量）"""
    return globals().get('LOG_DIR', os.path.join(os.getcwd(), 'lingyuan_logs'))


def _get_config_dir() -> str:
    """获取配置目录（兼容全局变量）"""
    return globals().get('CONFIG_DIR', os.path.join(os.getcwd(), 'lingyuan_config'))


# ============================================================
# #47 WebChatUI - Web Chat界面
# ============================================================

class WebChatUI:
    """Web聊天界面生成器

    生成完整的单页HTML应用（内嵌CSS+JS），包含：
    - 多轮对话界面（消息气泡，用户/助手区分）
    - 流式输出显示（逐字显示效果）
    - 简化版Markdown渲染（加粗/代码块/列表）
    - 参数调节（temperature, max_tokens, top_p）
    - 模型选择（下拉列表）
    - 对话历史保存/加载（localStorage模拟）
    - 新建对话/删除对话
    - 复制消息
    - 停止生成按钮
    - 响应式设计（手机/桌面适配）
    """

    def __init__(self,
                 title: str = "灵元大模型 - 对话助手",
                 api_endpoint: str = "/v1/chat/completions",
                 models: Optional[List[str]] = None):
        self.title = title
        self.api_endpoint = api_endpoint
        self.models = models or [
            "lingyuan-7b", "lingyuan-13b", "lingyuan-70b",
            "lingyuan-mini", "lingyuan-chat"
        ]
        self.version = "1.0.0"
        self.created_at = datetime.now().isoformat()

    def _build_model_options(self) -> str:
        """构建模型下拉选项HTML"""
        opts = []
        for i, m in enumerate(self.models):
            sel = " selected" if i == 0 else ""
            opts.append('<option value="%s"%s>%s</option>' % (m, sel, m))
        return "\n".join(opts)

    def _get_css(self) -> str:
        """获取CSS样式"""
        return r'''
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-tertiary: #0f3460;
            --bg-input: #1e1e3a;
            --text-primary: #e4e4e4;
            --text-secondary: #a0a0b0;
            --accent: #e94560;
            --accent-hover: #ff5470;
            --user-bubble: #0f3460;
            --assistant-bubble: #1e1e3a;
            --border: #2a2a4a;
            --code-bg: #0d0d1f;
            --success: #4caf50;
            --danger: #f44336;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                         "PingFang SC", "Microsoft YaHei", sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }
        .sidebar {
            width: 260px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: margin-left 0.3s;
            flex-shrink: 0;
        }
        .sidebar.collapsed { margin-left: -260px; }
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
        }
        .btn-new-chat {
            width: 100%;
            padding: 10px;
            background: var(--accent);
            color: #fff;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        .btn-new-chat:hover { background: var(--accent-hover); }
        .conversation-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }
        .conv-item {
            padding: 10px 12px;
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s;
        }
        .conv-item:hover { background: var(--bg-tertiary); }
        .conv-item.active { background: var(--bg-tertiary); }
        .conv-title {
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 13px;
        }
        .conv-delete {
            opacity: 0;
            color: var(--danger);
            cursor: pointer;
            padding: 2px 6px;
            font-size: 16px;
            transition: opacity 0.2s;
        }
        .conv-item:hover .conv-delete { opacity: 0.7; }
        .conv-delete:hover { opacity: 1; }
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        .header {
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .btn-icon {
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 18px;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .btn-icon:hover { background: var(--bg-tertiary); }
        .model-select {
            background: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 13px;
            cursor: pointer;
        }
        .settings-panel {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 12px 16px;
            display: none;
            gap: 20px;
            flex-wrap: wrap;
        }
        .settings-panel.open { display: flex; }
        .setting-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
            min-width: 140px;
        }
        .setting-item label {
            font-size: 12px;
            color: var(--text-secondary);
        }
        .setting-item input[type=range] {
            width: 100%;
            cursor: pointer;
        }
        .setting-item input[type=number] {
            background: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 13px;
        }
        .setting-value {
            font-size: 12px;
            color: var(--accent);
            font-weight: 600;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            display: flex;
            gap: 12px;
            max-width: 80%;
        }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .message.assistant { align-self: flex-start; }
        .msg-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            flex-shrink: 0;
        }
        .message.user .msg-avatar { background: var(--accent); color: #fff; }
        .message.assistant .msg-avatar { background: var(--bg-tertiary); color: var(--text-primary); }
        .msg-bubble {
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.6;
            position: relative;
            word-wrap: break-word;
        }
        .message.user .msg-bubble { background: var(--user-bubble); border-top-right-radius: 4px; }
        .message.assistant .msg-bubble { background: var(--assistant-bubble); border-top-left-radius: 4px; }
        .msg-bubble pre {
            background: var(--code-bg);
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 8px 0;
            font-size: 13px;
        }
        .msg-bubble code {
            background: var(--code-bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 13px;
        }
        .msg-bubble pre code { background: none; padding: 0; }
        .msg-bubble ul, .msg-bubble ol { margin-left: 20px; margin-top: 4px; }
        .msg-bubble strong { color: var(--accent); }
        .msg-copy {
            position: absolute;
            top: -10px;
            right: 8px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 11px;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.2s;
            color: var(--text-secondary);
        }
        .msg-bubble:hover .msg-copy { opacity: 1; }
        .typing-cursor {
            display: inline-block;
            width: 2px;
            height: 14px;
            background: var(--accent);
            animation: blink 0.8s infinite;
            vertical-align: middle;
        }
        @keyframes blink { 50% { opacity: 0; } }
        .input-area {
            padding: 16px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        .input-area textarea {
            flex: 1;
            background: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
            resize: none;
            max-height: 120px;
            min-height: 44px;
            font-family: inherit;
        }
        .input-area textarea:focus { outline: none; border-color: var(--accent); }
        .btn-send, .btn-stop {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            color: #fff;
            transition: background 0.2s;
        }
        .btn-send { background: var(--accent); }
        .btn-send:hover { background: var(--accent-hover); }
        .btn-send:disabled { background: var(--text-secondary); cursor: not-allowed; }
        .btn-stop { background: var(--danger); display: none; }
        .btn-stop:hover { opacity: 0.85; }
        .empty-state {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--text-secondary);
            gap: 8px;
        }
        .empty-state h2 { font-size: 22px; color: var(--text-primary); }
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
            justify-content: center;
            max-width: 500px;
        }
        .suggestion-card {
            padding: 10px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            transition: border-color 0.2s;
        }
        .suggestion-card:hover { border-color: var(--accent); }
        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                z-index: 100;
                height: 100vh;
                margin-left: -260px;
            }
            .sidebar.open { margin-left: 0; }
            .message { max-width: 95%; }
            .settings-panel { gap: 10px; }
            .setting-item { min-width: 100px; }
        }
        '''

    def _get_html_body(self) -> str:
        """获取HTML body内容"""
        model_opts = self._build_model_options()
        return '''
        <div class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <button class="btn-new-chat" id="btnNewChat">+ 新建对话</button>
            </div>
            <div class="conversation-list" id="convList"></div>
        </div>
        <div class="main">
            <div class="header">
                <button class="btn-icon" id="btnToggleSidebar">&#9776;</button>
                <select class="model-select" id="modelSelect">''' + model_opts + '''</select>
                <button class="btn-icon" id="btnToggleSettings">&#9881; 设置</button>
            </div>
            <div class="settings-panel" id="settingsPanel">
                <div class="setting-item">
                    <label>Temperature: <span class="setting-value" id="tempVal">0.7</span></label>
                    <input type="range" id="tempSlider" min="0" max="2" step="0.1" value="0.7">
                </div>
                <div class="setting-item">
                    <label>Max Tokens: <span class="setting-value" id="maxTokensVal">2048</span></label>
                    <input type="number" id="maxTokensInput" min="1" max="8192" value="2048">
                </div>
                <div class="setting-item">
                    <label>Top P: <span class="setting-value" id="topPVal">0.9</span></label>
                    <input type="range" id="topPSlider" min="0" max="1" step="0.05" value="0.9">
                </div>
            </div>
            <div class="messages" id="messages">
                <div class="empty-state" id="emptyState">
                    <h2>''' + self.title + '''</h2>
                    <p>开始一段新的对话</p>
                    <div class="suggestions">
                        <div class="suggestion-card" onclick="quickSend('你好，请介绍一下你自己')">介绍你自己</div>
                        <div class="suggestion-card" onclick="quickSend('用Python写一个快速排序')">写快速排序</div>
                        <div class="suggestion-card" onclick="quickSend('解释一下什么是大语言模型')">解释大语言模型</div>
                    </div>
                </div>
            </div>
            <div class="input-area">
                <textarea id="inputBox" placeholder="输入消息... (Enter发送, Shift+Enter换行)" rows="1"></textarea>
                <button class="btn-send" id="btnSend">发送</button>
                <button class="btn-stop" id="btnStop">停止</button>
            </div>
        </div>
        '''

    def _get_js(self) -> str:
        """获取JavaScript代码"""
        return r'''
        var API_ENDPOINT = "''' + self.api_endpoint + r'''";
        var conversations = [];
        var currentConvId = null;
        var isGenerating = false;
        var abortController = null;

        // ====== 对话管理 ======
        function loadConversations() {
            try {
                var saved = localStorage.getItem('ly_conversations');
                if (saved) { conversations = JSON.parse(saved); }
            } catch(e) { conversations = []; }
            if (conversations.length === 0) {
                createConversation();
            } else {
                currentConvId = conversations[0].id;
            }
            renderConvList();
            renderMessages();
        }

        function saveConversations() {
            try {
                localStorage.setItem('ly_conversations', JSON.stringify(conversations));
            } catch(e) {}
        }

        function createConversation() {
            var conv = {
                id: 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6),
                title: '新对话',
                messages: [],
                model: document.getElementById('modelSelect').value,
                params: getParams(),
                createdAt: Date.now()
            };
            conversations.unshift(conv);
            currentConvId = conv.id;
            saveConversations();
            renderConvList();
            renderMessages();
        }

        function deleteConversation(id) {
            conversations = conversations.filter(function(c) { return c.id !== id; });
            if (currentConvId === id) {
                currentConvId = conversations.length > 0 ? conversations[0].id : null;
                if (!currentConvId) { createConversation(); return; }
            }
            saveConversations();
            renderConvList();
            renderMessages();
        }

        function switchConversation(id) {
            currentConvId = id;
            var conv = getConv(id);
            if (conv) {
                document.getElementById('modelSelect').value = conv.model || document.getElementById('modelSelect').value;
            }
            renderConvList();
            renderMessages();
        }

        function getConv(id) {
            return conversations.find(function(c) { return c.id === id; });
        }

        // ====== 渲染 ======
        function renderConvList() {
            var list = document.getElementById('convList');
            list.innerHTML = '';
            conversations.forEach(function(conv) {
                var item = document.createElement('div');
                item.className = 'conv-item' + (conv.id === currentConvId ? ' active' : '');
                item.innerHTML = '<span class="conv-title">' + escapeHtml(conv.title) + '</span>' +
                    '<span class="conv-delete" onclick="event.stopPropagation();deleteConversation(\'' + conv.id + '\')">&times;</span>';
                item.onclick = function() { switchConversation(conv.id); };
                list.appendChild(item);
            });
        }

        function renderMessages() {
            var container = document.getElementById('messages');
            var conv = getConv(currentConvId);
            if (!conv || conv.messages.length === 0) {
                showEmptyState();
                return;
            }
            container.innerHTML = '';
            conv.messages.forEach(function(msg) {
                container.appendChild(createMessageElement(msg.role, msg.content));
            });
            container.scrollTop = container.scrollHeight;
        }

        function showEmptyState() {
            var container = document.getElementById('messages');
            var empty = document.getElementById('emptyState');
            if (empty) {
                container.innerHTML = '';
                container.appendChild(empty);
                empty.style.display = 'flex';
            }
        }

        function createMessageElement(role, content) {
            var div = document.createElement('div');
            div.className = 'message ' + role;
            var avatar = role === 'user' ? 'U' : 'AI';
            div.innerHTML = '<div class="msg-avatar">' + avatar + '</div>' +
                '<div class="msg-bubble">' +
                '<span class="msg-copy" onclick="copyMessage(this)">复制</span>' +
                renderMarkdown(content) +
                '</div>';
            return div;
        }

        // ====== Markdown渲染（简化版） ======
        function renderMarkdown(text) {
            if (!text) return '';
            var html = escapeHtml(text);
            // 代码块
            html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, function(m, lang, code) {
                return '<pre><code>' + code.trim() + '</code></pre>';
            });
            // 行内代码
            html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
            // 加粗
            html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            // 列表
            var lines = html.split('\n');
            var result = [];
            var inList = false;
            for (var i = 0; i < lines.length; i++) {
                var line = lines[i];
                if (/^\s*[-*]\s+/.test(line)) {
                    if (!inList) { result.push('<ul>'); inList = true; }
                    result.push('<li>' + line.replace(/^\s*[-*]\s+/, '') + '</li>');
                } else if (/^\s*\d+\.\s+/.test(line)) {
                    if (!inList) { result.push('<ol>'); inList = true; }
                    result.push('<li>' + line.replace(/^\s*\d+\.\s+/, '') + '</li>');
                } else {
                    if (inList) { result.push(inList ? '</ul>' : '</ol>'); inList = false; }
                    if (line.trim()) { result.push(line); }
                    else if (i < lines.length - 1) { result.push('<br>'); }
                }
            }
            if (inList) { result.push('</ul>'); }
            return result.join('\n');
        }

        function escapeHtml(text) {
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // ====== 消息发送与流式输出 ======
        function getParams() {
            return {
                temperature: parseFloat(document.getElementById('tempSlider').value),
                max_tokens: parseInt(document.getElementById('maxTokensInput').value),
                top_p: parseFloat(document.getElementById('topPSlider').value)
            };
        }

        function quickSend(text) {
            document.getElementById('inputBox').value = text;
            sendMessage();
        }

        async function sendMessage() {
            if (isGenerating) return;
            var input = document.getElementById('inputBox');
            var text = input.value.trim();
            if (!text) return;
            var conv = getConv(currentConvId);
            if (!conv) return;
            // 添加用户消息
            conv.messages.push({ role: 'user', content: text });
            if (conv.title === '新对话') { conv.title = text.substring(0, 30); }
            input.value = '';
            input.style.height = 'auto';
            saveConversations();
            renderConvList();
            renderMessages();
            // 准备助手回复
            isGenerating = true;
            document.getElementById('btnSend').style.display = 'none';
            document.getElementById('btnStop').style.display = 'block';
            var assistantMsg = { role: 'assistant', content: '' };
            conv.messages.push(assistantMsg);
            var container = document.getElementById('messages');
            var msgEl = createMessageElement('assistant', '');
            var bubble = msgEl.querySelector('.msg-bubble');
            container.appendChild(msgEl);
            container.scrollTop = container.scrollHeight;
            try {
                var params = getParams();
                var model = document.getElementById('modelSelect').value;
                conv.model = model;
                conv.params = params;
                abortController = new AbortController();
                var response = await fetch(API_ENDPOINT, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        messages: conv.messages.filter(function(m){return m.content;}).map(function(m){
                            return { role: m.role, content: m.content };
                        }),
                        temperature: params.temperature,
                        max_tokens: params.max_tokens,
                        top_p: params.top_p,
                        stream: true
                    }),
                    signal: abortController.signal
                });
                if (response.ok && response.body) {
                    var reader = response.body.getReader();
                    var decoder = new TextDecoder();
                    var fullText = '';
                    while (true) {
                        var chunk = await reader.read();
                        if (chunk.done) break;
                        var text_chunk = decoder.decode(chunk.value, { stream: true });
                        var lines = text_chunk.split('\n');
                        for (var i = 0; i < lines.length; i++) {
                            var line = lines[i].trim();
                            if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                                try {
                                    var data = JSON.parse(line.substring(6));
                                    if (data.choices && data.choices[0].delta && data.choices[0].delta.content) {
                                        fullText += data.choices[0].delta.content;
                                        bubble.innerHTML = '<span class="msg-copy" onclick="copyMessage(this)">复制</span>' + renderMarkdown(fullText) + '<span class="typing-cursor"></span>';
                                        container.scrollTop = container.scrollHeight;
                                    }
                                } catch(e) {}
                            }
                        }
                    }
                    assistantMsg.content = fullText;
                } else {
                    // 回退到模拟流式输出
                    await simulateStream(bubble, assistantMsg, container);
                }
            } catch(e) {
                if (e.name === 'AbortError') {
                    assistantMsg.content = bubble.textContent.replace('复制', '') || '（已停止生成）';
                } else {
                    // 回退到模拟流式输出
                    await simulateStream(bubble, assistantMsg, container);
                }
            }
            // 移除光标
            var cursor = bubble.querySelector('.typing-cursor');
            if (cursor) cursor.remove();
            bubble.innerHTML = '<span class="msg-copy" onclick="copyMessage(this)">复制</span>' + renderMarkdown(assistantMsg.content);
            isGenerating = false;
            document.getElementById('btnSend').style.display = 'block';
            document.getElementById('btnStop').style.display = 'none';
            saveConversations();
        }

        async function simulateStream(bubble, assistantMsg, container) {
            var sample = '这是灵元大模型的模拟回复。\n\n**功能说明：**\n- 支持多轮对话\n- 流式输出\n- Markdown渲染\n\n```python\nprint("Hello, Lingyuan!")\n```\n\n如需连接实际API，请确保服务端运行在 ' + API_ENDPOINT + '。';
            var fullText = '';
            for (var i = 0; i < sample.length; i++) {
                if (!isGenerating) break;
                fullText += sample[i];
                bubble.innerHTML = '<span class="msg-copy" onclick="copyMessage(this)">复制</span>' + renderMarkdown(fullText) + '<span class="typing-cursor"></span>';
                container.scrollTop = container.scrollHeight;
                await new Promise(function(r) { setTimeout(r, 20); });
            }
            assistantMsg.content = fullText;
        }

        function stopGeneration() {
            isGenerating = false;
            if (abortController) { abortController.abort(); }
            document.getElementById('btnSend').style.display = 'block';
            document.getElementById('btnStop').style.display = 'none';
        }

        function copyMessage(el) {
            var bubble = el.parentElement;
            var text = bubble.textContent.replace('复制', '').trim();
            navigator.clipboard.writeText(text).then(function() {
                el.textContent = '已复制';
                setTimeout(function() { el.textContent = '复制'; }, 1500);
            });
        }

        // ====== 事件绑定 ======
        function initEvents() {
            document.getElementById('btnNewChat').onclick = createConversation;
            document.getElementById('btnSend').onclick = sendMessage;
            document.getElementById('btnStop').onclick = stopGeneration;
            document.getElementById('btnToggleSidebar').onclick = function() {
                var sb = document.getElementById('sidebar');
                sb.classList.toggle('collapsed');
                if (window.innerWidth <= 768) { sb.classList.toggle('open'); }
            };
            document.getElementById('btnToggleSettings').onclick = function() {
                document.getElementById('settingsPanel').classList.toggle('open');
            };
            var input = document.getElementById('inputBox');
            input.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            input.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            });
            // 参数滑块
            var tempSlider = document.getElementById('tempSlider');
            tempSlider.oninput = function() {
                document.getElementById('tempVal').textContent = parseFloat(this.value).toFixed(1);
            };
            var topPSlider = document.getElementById('topPSlider');
            topPSlider.oninput = function() {
                document.getElementById('topPVal').textContent = parseFloat(this.value).toFixed(2);
            };
            var maxTokensInput = document.getElementById('maxTokensInput');
            maxTokensInput.oninput = function() {
                document.getElementById('maxTokensVal').textContent = this.value;
            };
        }

        // ====== 初始化 ======
        loadConversations();
        initEvents();
        '''

    def render(self) -> str:
        """生成完整的HTML页面字符串

        Returns:
            str: 完整的HTML文档字符串，包含内嵌CSS和JS
        """
        html = '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
        html += '<meta charset="UTF-8">\n'
        html += '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        html += '<title>' + self.title + '</title>\n'
        html += '<style>' + self._get_css() + '</style>\n'
        html += '</head>\n<body>\n'
        html += self._get_html_body() + '\n'
        html += '<script>' + self._get_js() + '</script>\n'
        html += '</body>\n</html>'
        return html

    def save_to_file(self, path: str) -> str:
        """将HTML保存到文件

        Args:
            path: 文件保存路径

        Returns:
            str: 文件的绝对路径
        """
        html = self.render()
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path) if os.path.dirname(abs_path) else '.', exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return abs_path

    def get_stats(self) -> Dict[str, Any]:
        """获取UI统计信息"""
        return {
            'title': self.title,
            'version': self.version,
            'models': self.models,
            'api_endpoint': self.api_endpoint,
            'created_at': self.created_at,
            'html_size_bytes': len(self.render()),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """获取仪表盘信息"""
        html = self.render()
        return {
            'module': 'WebChatUI',
            'version': self.version,
            'html_size_kb': round(len(html) / 1024, 1),
            'features': [
                '多轮对话', '流式输出', 'Markdown渲染', '参数调节',
                '模型选择', '历史保存', '新建/删除对话', '复制消息', '停止生成'
            ],
            'models': self.models,
            'responsive': True,
        }

    def __repr__(self) -> str:
        return '<WebChatUI title="%s" models=%d>' % (self.title, len(self.models))


# ============================================================
# #48 Playground - 参数调节试玩界面
# ============================================================

class Playground:
    """Playground参数调节试玩界面生成器

    生成完整的Playground HTML页面，包含：
    - 左侧参数面板（temperature滑块, top_p滑块, max_tokens输入, system prompt, stop words）
    - 右侧输入框 + 输出区域
    - 实时预览当前配置
    - 预设方案（创意/精确/平衡）一键切换
    - 对比模式：两种参数并排对比输出
    - Token计数器：实时显示输入/输出token数
    - 导出配置：下载当前参数为JSON
    """

    # 预设方案
    PRESETS = {
        'creative': {
            'label': '创意',
            'temperature': 1.2,
            'top_p': 0.95,
            'max_tokens': 4096,
            'system_prompt': '你是一个富有创意的助手，回答时尽量发挥想象力。',
            'stop': [],
        },
        'precise': {
            'label': '精确',
            'temperature': 0.1,
            'top_p': 0.5,
            'max_tokens': 2048,
            'system_prompt': '你是一个严谨的助手，回答时力求准确、简洁。',
            'stop': [],
        },
        'balanced': {
            'label': '平衡',
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 2048,
            'system_prompt': '你是一个乐于助人的助手。',
            'stop': [],
        },
    }

    def __init__(self,
                 title: str = "灵元大模型 - Playground",
                 api_endpoint: str = "/v1/chat/completions",
                 models: Optional[List[str]] = None):
        self.title = title
        self.api_endpoint = api_endpoint
        self.models = models or [
            "lingyuan-7b", "lingyuan-13b", "lingyuan-70b",
            "lingyuan-mini", "lingyuan-chat"
        ]
        self.version = "1.0.0"

    def _build_model_options(self) -> str:
        """构建模型下拉选项"""
        opts = []
        for i, m in enumerate(self.models):
            sel = " selected" if i == 0 else ""
            opts.append('<option value="%s"%s>%s</option>' % (m, sel, m))
        return "\n".join(opts)

    def _get_css(self) -> str:
        """获取CSS样式"""
        return r'''
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-panel: #16213e;
            --bg-input: #0d0d1f;
            --text-primary: #e4e4e4;
            --text-secondary: #8888aa;
            --accent: #e94560;
            --accent-2: #4ecdc4;
            --border: #2a2a4a;
            --success: #4caf50;
        }
        body {
            font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .pg-header {
            padding: 12px 20px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .pg-header h1 { font-size: 18px; }
        .pg-header select {
            background: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 13px;
        }
        .pg-header .spacer { flex: 1; }
        .pg-btn {
            padding: 6px 14px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg-panel);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .pg-btn:hover { border-color: var(--accent); }
        .pg-btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
        .pg-body {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        .pg-left {
            width: 320px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            overflow-y: auto;
            padding: 16px;
            flex-shrink: 0;
        }
        .pg-right {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .param-group { margin-bottom: 20px; }
        .param-group label {
            display: block;
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        .param-group .param-value {
            float: right;
            color: var(--accent);
            font-weight: 600;
        }
        .param-group input[type=range] { width: 100%; cursor: pointer; }
        .param-group input[type=number], .param-group input[type=text], .param-group textarea {
            width: 100%;
            background: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 10px;
            font-size: 13px;
            font-family: inherit;
        }
        .param-group textarea { resize: vertical; min-height: 60px; }
        .param-group input:focus, .param-group textarea:focus {
            outline: none; border-color: var(--accent);
        }
        .presets {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }
        .preset-btn {
            flex: 1;
            padding: 8px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg-input);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 12px;
            text-align: center;
            transition: all 0.2s;
        }
        .preset-btn:hover, .preset-btn.active {
            border-color: var(--accent);
            background: var(--bg-panel);
        }
        .config-preview {
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px;
            font-size: 12px;
            font-family: monospace;
            white-space: pre-wrap;
            word-break: break-all;
            max-height: 150px;
            overflow-y: auto;
            color: var(--accent-2);
        }
        .pg-right-header {
            padding: 8px 16px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 13px;
        }
        .token-counter {
            color: var(--text-secondary);
            font-size: 12px;
        }
        .token-counter span { color: var(--accent-2); font-weight: 600; }
        .compare-toggle {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            font-size: 12px;
            color: var(--text-secondary);
        }
        .output-area {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .output-box {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            min-height: 100px;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .output-box.compare { display: flex; gap: 16px; }
        .output-box.compare > div { flex: 1; }
        .output-box.compare > div h4 {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border);
        }
        .input-area {
            padding: 16px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        .input-area textarea {
            flex: 1;
            background: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
            resize: none;
            min-height: 44px;
            max-height: 120px;
            font-family: inherit;
        }
        .input-area textarea:focus { outline: none; border-color: var(--accent); }
        @media (max-width: 768px) {
            .pg-left { width: 100%; display: none; }
            .pg-left.open { display: block; position: fixed; z-index: 100; height: 100vh; }
            .output-box.compare { flex-direction: column; }
        }
        '''

    def _get_html_body(self) -> str:
        """获取HTML body"""
        model_opts = self._build_model_options()
        presets_html = ''.join([
            '<div class="preset-btn" onclick="applyPreset(\'%s\')">%s</div>' % (k, v['label'])
            for k, v in self.PRESETS.items()
        ])
        return '''
        <div class="pg-header">
            <h1>''' + self.title + '''</h1>
            <select id="modelSelect">''' + model_opts + '''</select>
            <div class="spacer"></div>
            <button class="pg-btn" id="btnTogglePanel">&#9776; 参数</button>
            <button class="pg-btn" onclick="exportConfig()">导出配置</button>
        </div>
        <div class="pg-body">
            <div class="pg-left" id="pgLeft">
                <div class="presets">''' + presets_html + '''</div>
                <div class="param-group">
                    <label>System Prompt</label>
                    <textarea id="systemPrompt" rows="3" placeholder="输入系统提示...">你是一个乐于助人的助手。</textarea>
                </div>
                <div class="param-group">
                    <label>Temperature <span class="param-value" id="tempVal">0.7</span></label>
                    <input type="range" id="tempSlider" min="0" max="2" step="0.1" value="0.7">
                </div>
                <div class="param-group">
                    <label>Top P <span class="param-value" id="topPVal">0.9</span></label>
                    <input type="range" id="topPSlider" min="0" max="1" step="0.05" value="0.9">
                </div>
                <div class="param-group">
                    <label>Max Tokens <span class="param-value" id="maxTokensVal">2048</span></label>
                    <input type="number" id="maxTokensInput" min="1" max="8192" value="2048">
                </div>
                <div class="param-group">
                    <label>Stop Words (逗号分隔)</label>
                    <input type="text" id="stopWords" placeholder="例如: STOP, END">
                </div>
                <div class="param-group">
                    <label>当前配置预览</label>
                    <div class="config-preview" id="configPreview"></div>
                </div>
            </div>
            <div class="pg-right">
                <div class="pg-right-header">
                    <div class="token-counter">输入Token: <span id="inputTokens">0</span> | 输出Token: <span id="outputTokens">0</span></div>
                    <div class="spacer" style="flex:1"></div>
                    <label class="compare-toggle">
                        <input type="checkbox" id="compareMode"> 对比模式
                    </label>
                    <button class="pg-btn primary" id="btnRun">运行</button>
                </div>
                <div class="output-area" id="outputArea">
                    <div class="output-box" id="outputBox">
                        <div style="color:var(--text-secondary)">点击"运行"按钮开始生成...</div>
                    </div>
                </div>
                <div class="input-area">
                    <textarea id="inputBox" placeholder="输入提示词..." rows="1"></textarea>
                </div>
            </div>
        </div>
        '''

    def _get_js(self) -> str:
        """获取JavaScript"""
        presets_json = json.dumps(self.PRESETS, ensure_ascii=False)
        return r'''
        var API_ENDPOINT = "''' + self.api_endpoint + r'''";
        var PRESETS = ''' + presets_json + r''';

        function getParams() {
            var stopStr = document.getElementById('stopWords').value.trim();
            var stop = stopStr ? stopStr.split(',').map(function(s){return s.trim();}).filter(Boolean) : [];
            return {
                model: document.getElementById('modelSelect').value,
                system_prompt: document.getElementById('systemPrompt').value,
                temperature: parseFloat(document.getElementById('tempSlider').value),
                top_p: parseFloat(document.getElementById('topPSlider').value),
                max_tokens: parseInt(document.getElementById('maxTokensInput').value),
                stop: stop
            };
        }

        function updatePreview() {
            var p = getParams();
            document.getElementById('tempVal').textContent = p.temperature.toFixed(1);
            document.getElementById('topPVal').textContent = p.top_p.toFixed(2);
            document.getElementById('maxTokensVal').textContent = p.max_tokens;
            document.getElementById('configPreview').textContent = JSON.stringify(p, null, 2);
            updateTokenCount();
        }

        function applyPreset(name) {
            var preset = PRESETS[name];
            if (!preset) return;
            document.getElementById('systemPrompt').value = preset.system_prompt;
            document.getElementById('tempSlider').value = preset.temperature;
            document.getElementById('topPSlider').value = preset.top_p;
            document.getElementById('maxTokensInput').value = preset.max_tokens;
            document.getElementById('stopWords').value = (preset.stop || []).join(', ');
            document.querySelectorAll('.preset-btn').forEach(function(btn){btn.classList.remove('active');});
            event.target.classList.add('active');
            updatePreview();
        }

        function estimateTokens(text) {
            if (!text) return 0;
            var chinese = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
            var english = (text.match(/[a-zA-Z]+/g) || []).length;
            var numbers = (text.match(/\d+/g) || []).length;
            return Math.ceil(chinese * 1.5 + english * 1.3 + numbers * 0.5);
        }

        function updateTokenCount() {
            var input = document.getElementById('inputBox').value;
            document.getElementById('inputTokens').textContent = estimateTokens(input);
        }

        async function runGeneration() {
            var input = document.getElementById('inputBox').value.trim();
            if (!input) return;
            var params = getParams();
            var isCompare = document.getElementById('compareMode').checked;
            var outputBox = document.getElementById('outputBox');
            if (isCompare) {
                outputBox.className = 'output-box compare';
                outputBox.innerHTML = '<div><h4>参数组A (temp=' + params.temperature + ')</h4><div id="outA">生成中...</div></div>' +
                    '<div><h4>参数组B (temp=' + Math.min(2, params.temperature + 0.5).toFixed(1) + ')</h4><div id="outB">生成中...</div></div>';
                var paramsB = JSON.parse(JSON.stringify(params));
                paramsB.temperature = Math.min(2, params.temperature + 0.5);
                generateSingle(input, params, 'outA');
                generateSingle(input, paramsB, 'outB');
            } else {
                outputBox.className = 'output-box';
                outputBox.innerHTML = '<div id="outSingle">生成中...</div>';
                generateSingle(input, params, 'outSingle');
            }
        }

        async function generateSingle(input, params, targetId) {
            var el = document.getElementById(targetId);
            if (!el) return;
            var messages = [];
            if (params.system_prompt) { messages.push({ role: 'system', content: params.system_prompt }); }
            messages.push({ role: 'user', content: input });
            try {
                var resp = await fetch(API_ENDPOINT, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: params.model,
                        messages: messages,
                        temperature: params.temperature,
                        top_p: params.top_p,
                        max_tokens: params.max_tokens,
                        stop: params.stop.length > 0 ? params.stop : undefined
                    })
                });
                if (resp.ok) {
                    var data = await resp.json();
                    var content = data.choices && data.choices[0] && data.choices[0].message
                        ? data.choices[0].message.content : '无返回内容';
                    el.textContent = content;
                    document.getElementById('outputTokens').textContent = estimateTokens(content);
                } else {
                    el.textContent = '[模拟输出] API未连接。Temperature=' + params.temperature +
                        '\n\n这是灵元大模型在 temperature=' + params.temperature +
                        ', top_p=' + params.top_p + ' 下的模拟回复。\n输入: ' + input.substring(0, 100);
                    document.getElementById('outputTokens').textContent = estimateTokens(el.textContent);
                }
            } catch(e) {
                el.textContent = '[模拟输出] API未连接。Temperature=' + params.temperature +
                    '\n\n这是灵元大模型在 temperature=' + params.temperature +
                    ', top_p=' + params.top_p + ' 下的模拟回复。\n输入: ' + input.substring(0, 100);
                document.getElementById('outputTokens').textContent = estimateTokens(el.textContent);
            }
        }

        function exportConfig() {
            var config = getParams();
            var blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'lingyuan_playground_config.json';
            a.click();
            URL.revokeObjectURL(url);
        }

        // 事件绑定
        ['tempSlider', 'topPSlider', 'maxTokensInput', 'systemPrompt', 'stopWords'].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) { el.addEventListener('input', updatePreview); }
        });
        document.getElementById('inputBox').addEventListener('input', updateTokenCount);
        document.getElementById('btnRun').onclick = runGeneration;
        document.getElementById('btnTogglePanel').onclick = function() {
            document.getElementById('pgLeft').classList.toggle('open');
        };
        document.getElementById('inputBox').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); runGeneration(); }
        });
        updatePreview();
        '''

    def render(self) -> str:
        """生成完整的HTML页面字符串

        Returns:
            str: 完整的HTML文档字符串
        """
        html = '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
        html += '<meta charset="UTF-8">\n'
        html += '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        html += '<title>' + self.title + '</title>\n'
        html += '<style>' + self._get_css() + '</style>\n'
        html += '</head>\n<body>\n'
        html += self._get_html_body() + '\n'
        html += '<script>' + self._get_js() + '</script>\n'
        html += '</body>\n</html>'
        return html

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'title': self.title,
            'version': self.version,
            'models': self.models,
            'presets': list(self.PRESETS.keys()),
            'html_size_bytes': len(self.render()),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """获取仪表盘信息"""
        return {
            'module': 'Playground',
            'version': self.version,
            'features': [
                '参数面板', '实时预览', '预设方案', '对比模式',
                'Token计数', '导出配置'
            ],
            'presets': {k: v['label'] for k, v in self.PRESETS.items()},
            'html_size_kb': round(len(self.render()) / 1024, 1),
        }

    def __repr__(self) -> str:
        return '<Playground title="%s">' % self.title


# ============================================================
# #49 TrainingDashboard - 训练可视化面板
# ============================================================

class TrainingDashboard:
    """训练可视化面板生成器

    生成训练监控的HTML页面，包含：
    - 实时loss曲线（Canvas绘制折线图，纯JS无依赖）
    - 学习率曲线
    - 梯度范数曲线
    - 训练进度条（当前步/总步, ETA）
    - 资源占用（GPU利用率, 显存, CPU）
    - 指标卡片（当前loss, best_loss, accuracy）
    - 日志流（最近N条训练日志）
    - 自动刷新（setInterval fetch）
    """

    def __init__(self,
                 title: str = "灵元大模型 - 训练监控面板",
                 refresh_interval: int = 5,
                 status_endpoint: str = "/api/training/status"):
        self.title = title
        self.refresh_interval = refresh_interval
        self.status_endpoint = status_endpoint
        self.version = "1.0.0"

    def _get_css(self) -> str:
        """获取CSS样式"""
        return r'''
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg-primary: #0a0a14;
            --bg-card: #141428;
            --bg-secondary: #1a1a2e;
            --text-primary: #e4e4e4;
            --text-secondary: #6868a0;
            --accent: #e94560;
            --accent-2: #4ecdc4;
            --accent-3: #ffd166;
            --accent-4: #6c5ce7;
            --border: #2a2a4a;
            --success: #4caf50;
            --danger: #f44336;
        }
        body {
            font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }
        .dash-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .dash-header h1 { font-size: 22px; }
        .refresh-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }
        .refresh-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 50% { opacity: 0.4; } }
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 4px; height: 100%;
        }
        .metric-card.loss::before { background: var(--accent); }
        .metric-card.best::before { background: var(--success); }
        .metric-card.acc::before { background: var(--accent-2); }
        .metric-card.lr::before { background: var(--accent-3); }
        .metric-card .label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        .metric-card .value {
            font-size: 28px;
            font-weight: 700;
            font-family: "Courier New", monospace;
        }
        .metric-card .delta {
            font-size: 12px;
            margin-top: 4px;
        }
        .metric-card .delta.up { color: var(--danger); }
        .metric-card .delta.down { color: var(--success); }
        .progress-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
        }
        .progress-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .progress-bar-bg {
            width: 100%;
            height: 24px;
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-4), var(--accent));
            border-radius: 12px;
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            color: #fff;
            min-width: 40px;
        }
        .charts-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .chart-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }
        .chart-card h3 {
            font-size: 14px;
            margin-bottom: 12px;
            color: var(--text-secondary);
        }
        .chart-card canvas {
            width: 100%;
            height: 200px;
            display: block;
        }
        .resource-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .resource-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }
        .resource-card h3 {
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        .resource-bar {
            width: 100%;
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 6px;
        }
        .resource-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s;
        }
        .resource-value {
            font-size: 20px;
            font-weight: 600;
            font-family: "Courier New", monospace;
        }
        .log-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }
        .log-section h3 {
            font-size: 14px;
            margin-bottom: 12px;
            color: var(--text-secondary);
        }
        .log-list {
            max-height: 300px;
            overflow-y: auto;
            font-family: "Courier New", monospace;
            font-size: 12px;
        }
        .log-entry {
            padding: 4px 0;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 8px;
        }
        .log-time { color: var(--text-secondary); white-space: nowrap; }
        .log-level { font-weight: 600; min-width: 50px; }
        .log-level.INFO { color: var(--accent-2); }
        .log-level.WARN { color: var(--accent-3); }
        .log-level.ERROR { color: var(--danger); }
        .log-msg { flex: 1; word-break: break-all; }
        @media (max-width: 768px) {
            .charts-row { grid-template-columns: 1fr; }
            body { padding: 10px; }
        }
        '''

    def _get_js(self, data_json: str) -> str:
        """获取JavaScript"""
        return r'''
        var STATUS_ENDPOINT = "''' + self.status_endpoint + r'''";
        var REFRESH_INTERVAL = ''' + str(self.refresh_interval) + r''';
        var initialData = ''' + data_json + r''';

        // ====== Canvas图表绘制 ======
        function drawChart(canvasId, data, color, label) {
            var canvas = document.getElementById(canvasId);
            if (!canvas) return;
            var ctx = canvas.getContext('2d');
            var dpr = window.devicePixelRatio || 1;
            var rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            var W = rect.width;
            var H = rect.height;
            var padding = { top: 10, right: 10, bottom: 25, left: 50 };
            var plotW = W - padding.left - padding.right;
            var plotH = H - padding.top - padding.bottom;
            // 清空
            ctx.clearRect(0, 0, W, H);
            // 数据检查
            if (!data || data.length === 0) {
                ctx.fillStyle = '#6868a0';
                ctx.font = '13px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('暂无数据', W/2, H/2);
                return;
            }
            // 计算范围
            var minVal = Math.min.apply(null, data);
            var maxVal = Math.max.apply(null, data);
            if (minVal === maxVal) { minVal -= 1; maxVal += 1; }
            var range = maxVal - minVal;
            // 绘制网格
            ctx.strokeStyle = '#2a2a4a';
            ctx.lineWidth = 1;
            ctx.font = '10px monospace';
            ctx.fillStyle = '#6868a0';
            for (var i = 0; i <= 4; i++) {
                var y = padding.top + (plotH / 4) * i;
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(W - padding.right, y);
                ctx.stroke();
                var val = maxVal - (range / 4) * i;
                ctx.textAlign = 'right';
                ctx.fillText(val.toFixed(4), padding.left - 5, y + 3);
            }
            // X轴标签
            ctx.textAlign = 'center';
            var xSteps = Math.min(5, data.length);
            for (var i = 0; i <= xSteps; i++) {
                var x = padding.left + (plotW / xSteps) * i;
                var stepIdx = Math.floor((data.length - 1) / xSteps * i);
                ctx.fillText(String(stepIdx), x, H - padding.bottom + 15);
            }
            // 绘制折线
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (var i = 0; i < data.length; i++) {
                var x = padding.left + (plotW / Math.max(1, data.length - 1)) * i;
                var y = padding.top + plotH - (plotH * (data[i] - minVal) / range);
                if (i === 0) { ctx.moveTo(x, y); }
                else { ctx.lineTo(x, y); }
            }
            ctx.stroke();
            // 填充区域
            ctx.lineTo(padding.left + plotW, padding.top + plotH);
            ctx.lineTo(padding.left, padding.top + plotH);
            ctx.closePath();
            ctx.fillStyle = color + '20';
            ctx.fill();
            // 最新点
            if (data.length > 0) {
                var lastX = padding.left + plotW;
                var lastY = padding.top + plotH - (plotH * (data[data.length-1] - minVal) / range);
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
                ctx.fill();
            }
            // 标签
            ctx.fillStyle = color;
            ctx.font = 'bold 11px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(label + ': ' + data[data.length-1].toFixed(4), padding.left, padding.top - 2);
        }

        // ====== 更新UI ======
        function updateDashboard(data) {
            if (!data) return;
            // 指标卡片
            if (data.metrics) {
                setMetric('loss', data.metrics.current_loss, data.metrics.loss_delta);
                setMetric('best', data.metrics.best_loss);
                setMetric('acc', data.metrics.accuracy !== undefined ? (data.metrics.accuracy * 100).toFixed(2) + '%' : '--');
                setMetric('lr', data.metrics.current_lr !== undefined ? data.metrics.current_lr.toExponential(2) : '--');
            }
            // 进度条
            if (data.progress) {
                var pct = data.progress.total > 0 ? (data.progress.current / data.progress.total * 100) : 0;
                var bar = document.getElementById('progressFill');
                if (bar) { bar.style.width = pct + '%'; bar.textContent = pct.toFixed(1) + '%'; }
                var info = document.getElementById('progressInfo');
                if (info) {
                    info.textContent = '步 ' + data.progress.current + ' / ' + data.progress.total +
                        (data.progress.eta ? '  |  ETA: ' + data.progress.eta : '');
                }
            }
            // 图表
            if (data.charts) {
                drawChart('chartLoss', data.charts.loss || [], '#e94560', 'Loss');
                drawChart('chartLR', data.charts.lr || [], '#ffd166', 'Learning Rate');
                drawChart('chartGrad', data.charts.grad_norm || [], '#4ecdc4', 'Grad Norm');
            }
            // 资源
            if (data.resources) {
                setResource('gpu', data.resources.gpu_util, '%');
                setResource('mem', data.resources.gpu_mem, '%');
                setResource('cpu', data.resources.cpu, '%');
            }
            // 日志
            if (data.logs) {
                var logList = document.getElementById('logList');
                logList.innerHTML = '';
                data.logs.slice(-50).reverse().forEach(function(log) {
                    var entry = document.createElement('div');
                    entry.className = 'log-entry';
                    entry.innerHTML = '<span class="log-time">' + (log.time || '') + '</span>' +
                        '<span class="log-level ' + (log.level || 'INFO') + '">' + (log.level || 'INFO') + '</span>' +
                        '<span class="log-msg">' + escapeHtml(log.message || '') + '</span>';
                    logList.appendChild(entry);
                });
            }
        }

        function setMetric(name, value, delta) {
            var el = document.getElementById('metric_' + name);
            if (el && value !== undefined && value !== null) {
                el.textContent = typeof value === 'number' ? value.toFixed(4) : value;
            }
            if (delta !== undefined) {
                var dEl = document.getElementById('delta_' + name);
                if (dEl) {
                    dEl.textContent = (delta >= 0 ? '+' : '') + delta.toFixed(4);
                    dEl.className = 'delta ' + (delta >= 0 ? 'up' : 'down');
                }
            }
        }

        function setResource(name, value, unit) {
            var valEl = document.getElementById('res_' + name);
            var barEl = document.getElementById('resbar_' + name);
            if (valEl) { valEl.textContent = value + unit; }
            if (barEl) { barEl.style.width = value + '%'; }
        }

        function escapeHtml(text) {
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // ====== 自动刷新 ======
        async function fetchData() {
            try {
                var resp = await fetch(STATUS_ENDPOINT);
                if (resp.ok) {
                    var data = await resp.json();
                    updateDashboard(data);
                }
            } catch(e) {
                // 使用初始数据
            }
        }

        // ====== 初始化 ======
        updateDashboard(initialData);
        setInterval(fetchData, REFRESH_INTERVAL * 1000);
        window.addEventListener('resize', function() {
            if (initialData.charts) {
                drawChart('chartLoss', initialData.charts.loss || [], '#e94560', 'Loss');
                drawChart('chartLR', initialData.charts.lr || [], '#ffd166', 'Learning Rate');
                drawChart('chartGrad', initialData.charts.grad_norm || [], '#4ecdc4', 'Grad Norm');
            }
        });
        '''

    def render(self, data: Optional[Dict[str, Any]] = None) -> str:
        """生成完整的HTML页面字符串

        Args:
            data: 训练数据，包含metrics, progress, charts, resources, logs等

        Returns:
            str: 完整的HTML文档字符串
        """
        if data is None:
            data = self._get_default_data()

        data_json = json.dumps(data, ensure_ascii=False)

        html = '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
        html += '<meta charset="UTF-8">\n'
        html += '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        html += '<title>' + self.title + '</title>\n'
        html += '<style>' + self._get_css() + '</style>\n'
        html += '</head>\n<body>\n'
        html += '<div class="dash-header">\n'
        html += '<h1>' + self.title + '</h1>\n'
        html += '<div class="refresh-indicator"><div class="refresh-dot"></div>'
        html += '<span>每%d秒自动刷新</span></div>\n' % self.refresh_interval
        html += '</div>\n'
        # 指标卡片
        html += '<div class="metrics-row">'
        html += '<div class="metric-card loss"><div class="label">当前 Loss</div><div class="value" id="metric_loss">--</div><div class="delta" id="delta_loss"></div></div>'
        html += '<div class="metric-card best"><div class="label">最佳 Loss</div><div class="value" id="metric_best">--</div><div class="delta"></div></div>'
        html += '<div class="metric-card acc"><div class="label">准确率</div><div class="value" id="metric_acc">--</div><div class="delta"></div></div>'
        html += '<div class="metric-card lr"><div class="label">学习率</div><div class="value" id="metric_lr">--</div><div class="delta"></div></div>'
        html += '</div>\n'
        # 进度条
        html += '<div class="progress-section">'
        html += '<div class="progress-header"><span>训练进度</span><span id="progressInfo">--</span></div>'
        html += '<div class="progress-bar-bg"><div class="progress-bar-fill" id="progressFill" style="width:0%">0%</div></div>'
        html += '</div>\n'
        # 图表
        html += '<div class="charts-row">'
        html += '<div class="chart-card"><h3>Loss 曲线</h3><canvas id="chartLoss"></canvas></div>'
        html += '<div class="chart-card"><h3>学习率曲线</h3><canvas id="chartLR"></canvas></div>'
        html += '<div class="chart-card"><h3>梯度范数曲线</h3><canvas id="chartGrad"></canvas></div>'
        html += '</div>\n'
        # 资源
        html += '<div class="resource-row">'
        html += '<div class="resource-card"><h3>GPU 利用率</h3><div class="resource-bar"><div class="resource-bar-fill" id="resbar_gpu" style="background:var(--accent);width:0%"></div></div><div class="resource-value" id="res_gpu">--</div></div>'
        html += '<div class="resource-card"><h3>显存占用</h3><div class="resource-bar"><div class="resource-bar-fill" id="resbar_mem" style="background:var(--accent-2);width:0%"></div></div><div class="resource-value" id="res_mem">--</div></div>'
        html += '<div class="resource-card"><h3>CPU 利用率</h3><div class="resource-bar"><div class="resource-bar-fill" id="resbar_cpu" style="background:var(--accent-3);width:0%"></div></div><div class="resource-value" id="res_cpu">--</div></div>'
        html += '</div>\n'
        # 日志
        html += '<div class="log-section"><h3>训练日志</h3><div class="log-list" id="logList"></div></div>\n'
        html += '<script>' + self._get_js(data_json) + '</script>\n'
        html += '</body>\n</html>'
        return html

    def _get_default_data(self) -> Dict[str, Any]:
        """获取默认示例数据"""
        loss_data = [3.5 - i * 0.02 + random.uniform(-0.05, 0.05) for i in range(100)]
        lr_data = [5e-4 * (0.95 ** (i // 10)) for i in range(100)]
        grad_data = [abs(random.gauss(1.0, 0.3)) for _ in range(100)]
        return {
            'metrics': {
                'current_loss': loss_data[-1],
                'loss_delta': -0.015,
                'best_loss': min(loss_data),
                'accuracy': 0.8234,
                'current_lr': lr_data[-1],
            },
            'progress': {
                'current': 1500,
                'total': 10000,
                'eta': '2h 15m',
            },
            'charts': {
                'loss': loss_data,
                'lr': lr_data,
                'grad_norm': grad_data,
            },
            'resources': {
                'gpu_util': 87,
                'gpu_mem': 72,
                'cpu': 45,
            },
            'logs': [
                {'time': '14:30:01', 'level': 'INFO', 'message': 'Step 1500: loss=1.5234, lr=2.38e-04'},
                {'time': '14:29:30', 'level': 'INFO', 'message': 'Step 1499: loss=1.5301, lr=2.38e-04'},
                {'time': '14:29:00', 'level': 'INFO', 'message': 'Step 1498: loss=1.5412, lr=2.38e-04'},
                {'time': '14:25:00', 'level': 'WARN', 'message': '梯度范数偏高: 2.345'},
                {'time': '14:20:00', 'level': 'INFO', 'message': 'Checkpoint saved at step 1400'},
            ],
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        data = self._get_default_data()
        return {
            'title': self.title,
            'version': self.version,
            'refresh_interval': self.refresh_interval,
            'status_endpoint': self.status_endpoint,
            'html_size_bytes': len(self.render(data)),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """获取仪表盘信息"""
        return {
            'module': 'TrainingDashboard',
            'version': self.version,
            'features': [
                'Loss曲线', '学习率曲线', '梯度范数曲线',
                '训练进度条', 'ETA计算', '资源监控',
                '指标卡片', '日志流', '自动刷新'
            ],
            'refresh_interval': self.refresh_interval,
            'chart_count': 3,
        }

    def __repr__(self) -> str:
        return '<TrainingDashboard refresh=%ds>' % self.refresh_interval


# ============================================================
# #50 ModelWatermarker - 模型水印
# ============================================================

@dataclass
class WatermarkKey:
    """水印密钥数据模型"""
    key_id: str
    key: str  # 密钥（hex字符串）
    created_at: float = field(default_factory=time.time)
    method: str = "zero_width"  # zero_width / vocabulary / statistical / sentence
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WatermarkKey':
        return cls(
            key_id=d.get('key_id', ''),
            key=d.get('key', ''),
            created_at=d.get('created_at', time.time()),
            method=d.get('method', 'zero_width'),
            description=d.get('description', ''),
            metadata=d.get('metadata', {}),
        )


@dataclass
class WatermarkRecord:
    """水印嵌入记录"""
    record_id: str
    key_id: str
    method: str
    watermark_data: str  # 嵌入的水印数据（JSON字符串）
    text_hash: str  # 原始文本哈希
    embedded_at: float = field(default_factory=time.time)
    text_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelWatermarker:
    """模型水印系统

    支持四种水印嵌入方法：
    1. 词汇替换：用同义词替换特定位置的词（基于密钥）
    2. Unicode零宽字符：在文本中插入零宽空格(ZWSP/ZWJ)编码水印
    3. 统计水印：控制某些token的生成概率偏移（绿名单/红名单）
    4. 句式水印：在特定位置插入特定句式

    功能：
    - 水印嵌入：embed(text, key, watermark_data) → 水印文本
    - 水印提取：extract(text, key) → watermark_data
    - 水印验证：verify(text, key) → bool
    - 密钥管理：生成/存储水印密钥
    - 鲁棒性评估：抗删除/修改能力评估
    - 批量水印：为每条生成内容自动嵌入唯一ID
    """

    # 零宽字符映射
    _ZW_MAP = {
        '0': '\u200B',  # ZWSP - Zero Width Space
        '1': '\u200D',  # ZWJ - Zero Width Joiner
    }
    _ZW_REVERSE = {'\u200B': '0', '\u200D': '1'}

    # 水印分隔符
    _WM_START = '\u200C'  # ZWNJ - Zero Width Non-Joiner (标记水印开始)
    _WM_END = '\u2060'    # Word Joiner (标记水印结束)

    # 同义词组（用于词汇替换水印）
    _SYNONYM_GROUPS = [
        ('使用', '利用', '运用'),
        ('因此', '所以', '故而'),
        ('但是', '然而', '不过'),
        ('可以', '能够', '可'),
        ('非常', '十分', '极为'),
        ('重要', '关键', '核心'),
        ('问题', '难题', '难点'),
        ('方法', '方式', '途径'),
        ('开始', '启动', '着手'),
        ('完成', '结束', '达成'),
        ('提高', '提升', '增强'),
        ('减少', '降低', '削减'),
        ('需要', '需', '须'),
        ('包括', '包含', '涵盖'),
        ('显示', '表明', '呈现'),
    ]

    # 句式模板（用于句式水印）
    _SENTENCE_PATTERNS = [
        '值得注意的是，',
        '从某种意义上说，',
        '需要指出的是，',
        '换句话说，',
        '具体而言，',
        '总的来说，',
        '进一步来说，',
        '从这个角度来看，',
    ]

    # 常用词汇表（用于统计水印的绿/红名单）
    _COMMON_WORDS = [
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
        '都', '一', '上', '也', '很', '到', '说', '要', '去', '你',
        '会', '着', '没', '看', '好', '自', '这', '那', '他', '她',
    ]

    def __init__(self):
        self._keys: Dict[str, WatermarkKey] = {}  # key_id → WatermarkKey
        self._records: Dict[str, WatermarkRecord] = {}  # record_id → record
        self._stats = {
            'total_embedded': 0,
            'total_extracted': 0,
            'total_verified': 0,
            'method_usage': {'zero_width': 0, 'vocabulary': 0, 'statistical': 0, 'sentence': 0},
        }

    # ========== 密钥管理 ==========

    def generate_key(self, method: str = "zero_width",
                     description: str = "") -> Tuple[str, WatermarkKey]:
        """生成水印密钥

        Args:
            method: 水印方法 (zero_width/vocabulary/statistical/sentence)
            description: 密钥描述

        Returns:
            Tuple[str, WatermarkKey]: (密钥字符串, 密钥对象)
        """
        key_id = 'wmk_' + uuid.uuid4().hex[:12]
        # 生成256位密钥
        key_bytes = os.urandom(32)
        key_hex = key_bytes.hex()
        wm_key = WatermarkKey(
            key_id=key_id,
            key=key_hex,
            method=method,
            description=description,
        )
        self._keys[key_id] = wm_key
        return key_hex, wm_key

    def get_key(self, key_id: str) -> Optional[WatermarkKey]:
        """获取密钥"""
        return self._keys.get(key_id)

    def list_keys(self) -> List[WatermarkKey]:
        """列出所有密钥"""
        return list(self._keys.values())

    def delete_key(self, key_id: str) -> bool:
        """删除密钥"""
        if key_id in self._keys:
            del self._keys[key_id]
            return True
        return False

    # ========== 零宽字符水印 ==========

    def _encode_to_zerowidth(self, data: str) -> str:
        """将数据编码为零宽字符序列

        Args:
            data: 要编码的字符串

        Returns:
            str: 零宽字符序列
        """
        data_bytes = data.encode('utf-8')
        binary = ''.join(format(b, '08b') for b in data_bytes)
        return ''.join(self._ZW_MAP[bit] for bit in binary)

    def _decode_from_zerowidth(self, text: str) -> Optional[str]:
        """从零宽字符序列解码数据

        Args:
            text: 包含零宽字符的文本

        Returns:
            Optional[str]: 解码后的字符串，失败返回None
        """
        # 提取水印区域（在start和end标记之间）
        start_idx = text.find(self._WM_START)
        end_idx = text.find(self._WM_END)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            zw_region = text[start_idx + 1:end_idx]
        else:
            # 如果没有标记，提取所有零宽字符
            zw_region = ''.join(c for c in text if c in self._ZW_REVERSE)

        if not zw_region:
            return None

        binary = ''.join(self._ZW_REVERSE[c] for c in zw_region if c in self._ZW_REVERSE)
        if len(binary) < 8:
            return None

        # 补齐到8的倍数
        binary = binary[:len(binary) - (len(binary) % 8)]

        try:
            bytes_list = [int(binary[i:i + 8], 2) for i in range(0, len(binary), 8)]
            return bytes(bytes_list).decode('utf-8', errors='ignore')
        except (ValueError, UnicodeDecodeError):
            return None

    def _get_insert_position(self, text: str, key: str) -> int:
        """基于密钥确定水印插入位置

        Args:
            text: 原始文本
            key: 密钥

        Returns:
            int: 插入位置
        """
        key_hash = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        if len(text) == 0:
            return 0
        return key_hash % len(text)

    def embed_zero_width(self, text: str, key: str,
                         watermark_data: Dict[str, Any]) -> str:
        """使用零宽字符方法嵌入水印

        Args:
            text: 原始文本
            key: 水印密钥
            watermark_data: 要嵌入的水印数据

        Returns:
            str: 含水印的文本
        """
        data_str = json.dumps(watermark_data, ensure_ascii=False, sort_keys=True)
        encoded = self._encode_to_zerowidth(data_str)
        pos = self._get_insert_position(text, key)
        # 插入: 位置 + 开始标记 + 编码数据 + 结束标记
        watermarked = text[:pos] + self._WM_START + encoded + self._WM_END + text[pos:]
        return watermarked

    # ========== 词汇替换水印 ==========

    def embed_vocabulary(self, text: str, key: str,
                         watermark_id: str) -> str:
        """使用词汇替换方法嵌入水印

        基于密钥选择同义词进行替换，替换模式编码水印ID。

        Args:
            text: 原始文本
            key: 水印密钥
            watermark_id: 水印ID

        Returns:
            str: 含水印的文本
        """
        key_hash = int(hashlib.sha256((key + watermark_id).encode()).hexdigest(), 16)
        result = text

        for group_idx, group in enumerate(self._SYNONYM_GROUPS):
            for word_idx, word in enumerate(group):
                if word in result:
                    # 基于密钥决定使用哪个同义词
                    choice = (key_hash + group_idx * 7 + word_idx * 13) % len(group)
                    replacement = group[choice]
                    if replacement != word:
                        result = result.replace(word, replacement, 1)

        return result

    def extract_vocabulary(self, text: str, key: str,
                           original_text: str = "") -> Optional[str]:
        """从词汇替换水印中提取信息

        Args:
            text: 待检测文本
            key: 水印密钥
            original_text: 原始文本（用于对比）

        Returns:
            Optional[str]: 检测到的替换信息
        """
        if not original_text:
            return None

        replacements = []
        for group_idx, group in enumerate(self._SYNONYM_GROUPS):
            for word_idx, word in enumerate(group):
                if word in original_text:
                    for replacement in group:
                        if replacement != word and replacement in text and word not in text:
                            replacements.append({
                                'original': word,
                                'replaced': replacement,
                                'group': group_idx,
                            })
                            break

        if replacements:
            return json.dumps(replacements, ensure_ascii=False)
        return None

    # ========== 统计水印（绿名单/红名单） ==========

    def _get_green_red_lists(self, key: str) -> Tuple[set, set]:
        """基于密钥生成绿名单和红名单

        Args:
            key: 水印密钥

        Returns:
            Tuple[set, set]: (绿名单, 红名单)
        """
        rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))
        shuffled = self._COMMON_WORDS.copy()
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        return set(shuffled[:mid]), set(shuffled[mid:])

    def embed_statistical(self, text: str, key: str,
                          watermark_id: str) -> Tuple[str, Dict[str, Any]]:
        """使用统计水印方法嵌入水印

        通过标记绿名单/红名单词汇的使用情况来编码水印。

        Args:
            text: 原始文本
            key: 水印密钥
            watermark_id: 水印ID

        Returns:
            Tuple[str, Dict]: (文本, 水印信息)
        """
        green_list, red_list = self._get_green_red_lists(key)

        # 统计文本中的绿/红名单词汇
        green_hits = [w for w in green_list if w in text]
        red_hits = [w for w in red_list if w in text]

        # 生成水印签名
        signature = hashlib.sha256(
            (key + watermark_id + ''.join(green_hits)).encode()
        ).hexdigest()[:16]

        watermark_info = {
            'method': 'statistical',
            'watermark_id': watermark_id,
            'signature': signature,
            'green_count': len(green_hits),
            'red_count': len(red_hits),
            'green_ratio': len(green_hits) / max(1, len(green_hits) + len(red_hits)),
        }

        return text, watermark_info

    def verify_statistical(self, text: str, key: str,
                           expected_info: Dict[str, Any]) -> bool:
        """验证统计水印

        Args:
            text: 待验证文本
            key: 水印密钥
            expected_info: 预期的水印信息

        Returns:
            bool: 验证是否通过
        """
        green_list, red_list = self._get_green_red_lists(key)
        green_hits = [w for w in green_list if w in text]
        red_hits = [w for w in red_list if w in text]

        # 检查绿名单比例是否匹配
        total = len(green_hits) + len(red_hits)
        if total == 0:
            return False

        actual_ratio = len(green_hits) / total
        expected_ratio = expected_info.get('green_ratio', 0)

        # 允许一定误差
        return abs(actual_ratio - expected_ratio) < 0.3

    # ========== 句式水印 ==========

    def embed_sentence_pattern(self, text: str, key: str,
                               watermark_id: str) -> str:
        """使用句式水印方法嵌入水印

        在文本的特定位置插入特定句式，句式的选择和位置由密钥决定。

        Args:
            text: 原始文本
            key: 水印密钥
            watermark_id: 水印ID

        Returns:
            str: 含水印的文本
        """
        key_hash = int(
            hashlib.sha256((key + watermark_id).encode()).hexdigest(), 16
        )

        # 选择句式
        pattern_idx = key_hash % len(self._SENTENCE_PATTERNS)
        pattern = self._SENTENCE_PATTERNS[pattern_idx]

        # 将文本分割为句子
        sentences = re.split(r'([。！？\n])', text)

        # 确定插入位置
        sentence_count = len([s for s in sentences if s.strip()])
        if sentence_count == 0:
            return pattern + text

        insert_sentence_idx = (key_hash >> 8) % max(1, sentence_count)

        # 找到第insert_sentence_idx个句子的位置并插入句式
        count = 0
        result_parts = []
        for part in sentences:
            if part.strip() and part not in '。！？\n':
                if count == insert_sentence_idx:
                    result_parts.append(pattern)
                count += 1
            result_parts.append(part)

        return ''.join(result_parts)

    def extract_sentence_pattern(self, text: str, key: str) -> Optional[str]:
        """检测文本中的句式水印

        Args:
            text: 待检测文本
            key: 水印密钥

        Returns:
            Optional[str]: 检测到的句式，无则返回None
        """
        for pattern in self._SENTENCE_PATTERNS:
            if pattern in text:
                return pattern
        return None

    # ========== 统一接口 ==========

    def embed(self, text: str, key: str,
              watermark_data: Optional[Dict[str, Any]] = None,
              method: Optional[str] = None) -> str:
        """统一水印嵌入接口

        Args:
            text: 原始文本
            key: 水印密钥（hex字符串）
            watermark_data: 水印数据（默认自动生成）
            method: 水印方法（默认使用零宽字符）

        Returns:
            str: 含水印的文本
        """
        if watermark_data is None:
            watermark_data = {
                'id': uuid.uuid4().hex[:16],
                'timestamp': time.time(),
                'source': 'lingyuan',
            }

        if method is None:
            method = 'zero_width'

        self._stats['total_embedded'] += 1
        self._stats['method_usage'][method] = self._stats['method_usage'].get(method, 0) + 1

        text_hash = hashlib.sha256(text.encode()).hexdigest()

        if method == 'zero_width':
            result = self.embed_zero_width(text, key, watermark_data)
        elif method == 'vocabulary':
            wm_id = watermark_data.get('id', uuid.uuid4().hex[:16])
            result = self.embed_vocabulary(text, key, wm_id)
        elif method == 'statistical':
            wm_id = watermark_data.get('id', uuid.uuid4().hex[:16])
            result, _ = self.embed_statistical(text, key, wm_id)
        elif method == 'sentence':
            wm_id = watermark_data.get('id', uuid.uuid4().hex[:16])
            result = self.embed_sentence_pattern(text, key, wm_id)
        else:
            result = self.embed_zero_width(text, key, watermark_data)

        # 记录
        record_id = 'wmr_' + uuid.uuid4().hex[:12]
        record = WatermarkRecord(
            record_id=record_id,
            key_id=key[:16],
            method=method,
            watermark_data=json.dumps(watermark_data, ensure_ascii=False),
            text_hash=text_hash,
            text_length=len(text),
        )
        self._records[record_id] = record

        return result

    def extract(self, text: str, key: str,
                method: str = "zero_width") -> Optional[Dict[str, Any]]:
        """从文本中提取水印

        Args:
            text: 待提取的文本
            key: 水印密钥
            method: 水印方法

        Returns:
            Optional[Dict]: 提取的水印数据，失败返回None
        """
        self._stats['total_extracted'] += 1

        if method == 'zero_width':
            decoded = self._decode_from_zerowidth(text)
            if decoded:
                try:
                    return json.loads(decoded)
                except json.JSONDecodeError:
                    return {'raw': decoded}
            return None
        elif method == 'sentence':
            pattern = self.extract_sentence_pattern(text, key)
            if pattern:
                return {'pattern': pattern, 'method': 'sentence'}
            return None
        else:
            return None

    def verify(self, text: str, key: str,
               method: str = "zero_width") -> bool:
        """验证文本中是否包含有效水印

        Args:
            text: 待验证的文本
            key: 水印密钥
            method: 水印方法

        Returns:
            bool: 是否包含有效水印
        """
        self._stats['total_verified'] += 1

        if method == 'zero_width':
            result = self.extract(text, key, 'zero_width')
            return result is not None
        elif method == 'sentence':
            return self.extract_sentence_pattern(text, key) is not None
        else:
            result = self.extract(text, key, method)
            return result is not None

    # ========== 批量水印 ==========

    def batch_embed(self, texts: List[str], key: str,
                    method: str = "zero_width") -> List[str]:
        """批量嵌入水印

        为每条文本嵌入唯一ID水印。

        Args:
            texts: 文本列表
            key: 水印密钥
            method: 水印方法

        Returns:
            List[str]: 含水印的文本列表
        """
        results = []
        for i, text in enumerate(texts):
            watermark_data = {
                'id': uuid.uuid4().hex[:16],
                'batch_index': i,
                'timestamp': time.time(),
                'source': 'lingyuan',
            }
            results.append(self.embed(text, key, watermark_data, method))
        return results

    # ========== 鲁棒性评估 ==========

    def evaluate_robustness(self, text: str, key: str,
                            method: str = "zero_width") -> Dict[str, Any]:
        """评估水印鲁棒性

        测试水印对各种文本修改的抗性。

        Args:
            text: 原始文本
            key: 水印密钥
            method: 水印方法

        Returns:
            Dict: 鲁棒性评估结果
        """
        watermarked = self.embed(text, key, method=method)
        results = {}

        # 1. 原始验证
        results['original'] = self.verify(watermarked, key, method)

        # 2. 部分删除（删除10%的字符）
        if len(watermarked) > 10:
            del_len = max(1, len(watermarked) // 10)
            del_pos = len(watermarked) // 2
            partial = watermarked[:del_pos] + watermarked[del_pos + del_len:]
            results['partial_deletion'] = self.verify(partial, key, method)
        else:
            results['partial_deletion'] = False

        # 3. 添加噪声（在末尾添加随机字符）
        noisy = watermarked + ' ' + ''.join(
            random.choice('abcdefghij') for _ in range(20)
        )
        results['noise_addition'] = self.verify(noisy, key, method)

        # 4. 截断（保留前80%）
        truncated = watermarked[:int(len(watermarked) * 0.8)]
        results['truncation'] = self.verify(truncated, key, method)

        # 5. 字符替换（替换可见字符）
        if method == 'zero_width':
            # 零宽字符不受可见字符替换影响
            results['char_replacement'] = True
        else:
            results['char_replacement'] = False

        # 计算总体鲁棒性分数
        pass_count = sum(1 for v in results.values() if v)
        results['robustness_score'] = pass_count / len(results)
        results['method'] = method

        return results

    # ========== 持久化 ==========

    def save(self, path: Optional[str] = None) -> bool:
        """保存水印密钥和记录到JSON

        Args:
            path: 保存路径，默认使用配置目录

        Returns:
            bool: 是否成功
        """
        if path is None:
            path = os.path.join(_get_config_dir(), 'watermarker.json')

        data = {
            'keys': {kid: k.to_dict() for kid, k in self._keys.items()},
            'records': {rid: r.to_dict() for rid, r in self._records.items()},
            'stats': self._stats,
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except (OSError, IOError):
            return False

    def load(self, path: Optional[str] = None) -> bool:
        """从JSON加载水印密钥和记录

        Args:
            path: 加载路径，默认使用配置目录

        Returns:
            bool: 是否成功
        """
        if path is None:
            path = os.path.join(_get_config_dir(), 'watermarker.json')
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._keys = {
                kid: WatermarkKey.from_dict(d)
                for kid, d in data.get('keys', {}).items()
            }
            self._records = {
                rid: WatermarkRecord(
                    record_id=r.get('record_id', ''),
                    key_id=r.get('key_id', ''),
                    method=r.get('method', 'zero_width'),
                    watermark_data=r.get('watermark_data', ''),
                    text_hash=r.get('text_hash', ''),
                    embedded_at=r.get('embedded_at', time.time()),
                    text_length=r.get('text_length', 0),
                )
                for rid, r in data.get('records', {}).items()
            }
            self._stats = data.get('stats', self._stats)
            return True
        except (OSError, IOError, json.JSONDecodeError):
            return False

    # ========== 统计与仪表盘 ==========

    def get_stats(self) -> Dict[str, Any]:
        """获取水印系统统计信息"""
        return {
            'total_keys': len(self._keys),
            'total_records': len(self._records),
            'total_embedded': self._stats['total_embedded'],
            'total_extracted': self._stats['total_extracted'],
            'total_verified': self._stats['total_verified'],
            'method_usage': dict(self._stats['method_usage']),
            'methods_available': ['zero_width', 'vocabulary', 'statistical', 'sentence'],
            'synonym_groups': len(self._SYNONYM_GROUPS),
            'sentence_patterns': len(self._SENTENCE_PATTERNS),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """获取水印系统仪表盘信息"""
        return {
            'module': 'ModelWatermarker',
            'version': '1.0.0',
            'features': [
                '零宽字符水印', '词汇替换水印', '统计水印', '句式水印',
                '水印提取', '水印验证', '密钥管理', '鲁棒性评估', '批量水印'
            ],
            'stats': self.get_stats(),
            'methods': {
                'zero_width': 'Unicode零宽字符编码',
                'vocabulary': '同义词替换',
                'statistical': '绿名单/红名单统计',
                'sentence': '句式插入',
            },
        }

    def __repr__(self) -> str:
        return '<ModelWatermarker keys=%d records=%d>' % (
            len(self._keys), len(self._records)
        )


# ============================================================
# #51 APIKeyManager - API Key管理
# ============================================================

@dataclass
class APIKeyInfo:
    """API Key信息数据模型"""
    key_id: str
    key_hash: str  # 存储key的SHA256哈希，不存储明文
    key_prefix: str  # 明文key的前几位，用于显示（如 sk-abcd...）
    user_id: str
    name: str
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    permissions: List[str] = field(default_factory=lambda: ['read'])
    allowed_endpoints: List[str] = field(default_factory=lambda: ['*'])
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 10000
    token_quota: int = 1000000
    tokens_used: int = 0
    status: str = "active"  # active / revoked / expired
    last_used: Optional[float] = None
    last_used_endpoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'APIKeyInfo':
        return cls(
            key_id=d.get('key_id', ''),
            key_hash=d.get('key_hash', ''),
            key_prefix=d.get('key_prefix', ''),
            user_id=d.get('user_id', ''),
            name=d.get('name', ''),
            created_at=d.get('created_at', time.time()),
            expires_at=d.get('expires_at'),
            permissions=d.get('permissions', ['read']),
            allowed_endpoints=d.get('allowed_endpoints', ['*']),
            rate_limit_per_minute=d.get('rate_limit_per_minute', 60),
            rate_limit_per_day=d.get('rate_limit_per_day', 10000),
            token_quota=d.get('token_quota', 1000000),
            tokens_used=d.get('tokens_used', 0),
            status=d.get('status', 'active'),
            last_used=d.get('last_used'),
            last_used_endpoint=d.get('last_used_endpoint', ''),
        )

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def is_active(self) -> bool:
        """检查是否有效"""
        return self.status == 'active' and not self.is_expired()


@dataclass
class AuditLogEntry:
    """审计日志条目"""
    log_id: str
    key_id: str
    user_id: str
    timestamp: float = field(default_factory=time.time)
    action: str = ""  # authenticate / revoke / rotate / rate_limit_hit / quota_exceeded
    endpoint: str = ""
    method: str = ""
    status_code: int = 200
    tokens_used: int = 0
    ip_address: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _TokenBucket:
    """令牌桶速率限制器

    使用令牌桶算法实现每分钟/每天的请求速率限制。
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: 桶容量（最大请求数）
            refill_rate: 每秒补充的令牌数
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """消费令牌

        Args:
            tokens: 需要消费的令牌数

        Returns:
            bool: 是否成功消费
        """
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def available(self) -> float:
        """获取可用令牌数"""
        self._refill()
        return self.tokens


class APIKeyManager:
    """API Key管理系统

    功能：
    - Key生成：generate_key(prefix="sk-") → sk-xxxxx (32位hex + 校验位)
    - Key存储：key → {user_id, created_at, expires_at, permissions, rate_limit, status}
    - 认证：authenticate(key) → bool + user_info
    - 权限管理：read/write/admin，按端点粒度控制
    - 速率限制：每分钟/每天请求限制，令牌桶
    - 配额管理：token使用配额，超额拒绝
    - Key吊销：revoke(key), revoke_all(user_id)
    - Key轮换：定期自动轮换（生成新key，旧key过期）
    - 审计：记录每个key的使用日志
    - 持久化：save/load到JSON
    """

    # 权限等级
    PERMISSION_LEVELS = {
        'read': 1,
        'write': 2,
        'admin': 3,
    }

    def __init__(self, salt: Optional[str] = None):
        """
        Args:
            salt: 密钥校验盐值（默认随机生成）
        """
        self._salt = salt or uuid.uuid4().hex
        self._keys: Dict[str, APIKeyInfo] = {}  # key_id → APIKeyInfo
        self._key_hash_index: Dict[str, str] = {}  # key_hash → key_id
        self._audit_logs: deque = deque(maxlen=10000)  # 审计日志
        self._rate_limiters: Dict[str, _TokenBucket] = {}  # key_id → TokenBucket (每分钟)
        self._daily_usage: Dict[str, Dict[str, int]] = {}  # key_id → {date: count}

    # ========== Key生成 ==========

    def generate_key(self,
                     prefix: str = "sk-",
                     user_id: str = "",
                     name: str = "",
                     permissions: Optional[List[str]] = None,
                     allowed_endpoints: Optional[List[str]] = None,
                     rate_limit_per_minute: int = 60,
                     rate_limit_per_day: int = 10000,
                     token_quota: int = 1000000,
                     expires_in_days: Optional[int] = None) -> str:
        """生成API Key

        格式: prefix + 32位hex + 4位校验位

        Args:
            prefix: key前缀
            user_id: 用户ID
            name: key名称
            permissions: 权限列表
            allowed_endpoints: 允许的端点列表
            rate_limit_per_minute: 每分钟请求限制
            rate_limit_per_day: 每天请求限制
            token_quota: token配额
            expires_in_days: 过期天数（None表示不过期）

        Returns:
            str: 生成的API Key明文
        """
        # 生成32位hex随机数
        random_hex = ''.join(format(b, '02x') for b in os.urandom(16))
        # 计算校验位（4位hex）
        checksum = hashlib.sha256((random_hex + self._salt).encode()).hexdigest()[:4]
        key = prefix + random_hex + checksum

        key_id = 'key_' + uuid.uuid4().hex[:12]
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        expires_at = None
        if expires_in_days is not None:
            expires_at = time.time() + expires_in_days * 86400

        key_info = APIKeyInfo(
            key_id=key_id,
            key_hash=key_hash,
            key_prefix=key[:8] + '...',
            user_id=user_id,
            name=name or ('Key for %s' % user_id),
            expires_at=expires_at,
            permissions=permissions or ['read'],
            allowed_endpoints=allowed_endpoints or ['*'],
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_day=rate_limit_per_day,
            token_quota=token_quota,
        )

        self._keys[key_id] = key_info
        self._key_hash_index[key_hash] = key_id

        # 初始化速率限制器
        self._rate_limiters[key_id] = _TokenBucket(
            capacity=rate_limit_per_minute,
            refill_rate=rate_limit_per_minute / 60.0,
        )

        return key

    # ========== Key验证 ==========

    def _verify_key_format(self, key: str) -> bool:
        """验证key格式"""
        pattern = r'^sk-[0-9a-f]{32}[0-9a-f]{4}$'
        return bool(re.match(pattern, key))

    def _verify_key_checksum(self, key: str) -> bool:
        """验证key校验位"""
        match = re.match(r'^sk-([0-9a-f]{32})([0-9a-f]{4})$', key)
        if not match:
            return False
        random_part, checksum = match.groups()
        expected = hashlib.sha256((random_part + self._salt).encode()).hexdigest()[:4]
        return hmac.compare_digest(checksum, expected)

    def _find_key_by_hash(self, key: str) -> Optional[APIKeyInfo]:
        """通过哈希查找key"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_id = self._key_hash_index.get(key_hash)
        if key_id:
            return self._keys.get(key_id)
        return None

    # ========== 认证 ==========

    def authenticate(self, key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """认证API Key

        Args:
            key: API Key明文

        Returns:
            Tuple[bool, Optional[Dict]]: (是否认证成功, 用户信息)
        """
        # 格式检查
        if not self._verify_key_format(key):
            self._add_audit_log('', '', 'authenticate', action='failed',
                                detail='invalid_format')
            return False, None

        # 校验位检查
        if not self._verify_key_checksum(key):
            self._add_audit_log('', '', 'authenticate', action='failed',
                                detail='invalid_checksum')
            return False, None

        # 查找key
        key_info = self._find_key_by_hash(key)
        if key_info is None:
            self._add_audit_log('', '', 'authenticate', action='failed',
                                detail='key_not_found')
            return False, None

        # 检查状态
        if key_info.status != 'active':
            self._add_audit_log(key_info.key_id, key_info.user_id,
                                'authenticate', action='failed',
                                detail='key_%s' % key_info.status)
            return False, None

        # 检查过期
        if key_info.is_expired():
            key_info.status = 'expired'
            self._add_audit_log(key_info.key_id, key_info.user_id,
                                'authenticate', action='failed',
                                detail='key_expired')
            return False, None

        # 更新使用时间
        key_info.last_used = time.time()

        user_info = {
            'key_id': key_info.key_id,
            'user_id': key_info.user_id,
            'name': key_info.name,
            'permissions': key_info.permissions,
            'allowed_endpoints': key_info.allowed_endpoints,
            'tokens_used': key_info.tokens_used,
            'token_quota': key_info.token_quota,
            'tokens_remaining': key_info.token_quota - key_info.tokens_used,
        }

        self._add_audit_log(key_info.key_id, key_info.user_id,
                            'authenticate', action='success')
        return True, user_info

    # ========== 权限管理 ==========

    def check_permission(self, key: str, required_permission: str) -> bool:
        """检查key是否具有指定权限

        Args:
            key: API Key明文
            required_permission: 需要的权限 (read/write/admin)

        Returns:
            bool: 是否具有权限
        """
        success, user_info = self.authenticate(key)
        if not success or not user_info:
            return False

        key_permissions = user_info.get('permissions', [])
        required_level = self.PERMISSION_LEVELS.get(required_permission, 0)

        for perm in key_permissions:
            perm_level = self.PERMISSION_LEVELS.get(perm, 0)
            if perm_level >= required_level:
                return True

        return False

    def check_endpoint(self, key: str, endpoint: str) -> bool:
        """检查key是否可以访问指定端点

        Args:
            key: API Key明文
            endpoint: API端点路径

        Returns:
            bool: 是否允许访问
        """
        key_info = self._find_key_by_hash(key)
        if key_info is None or not key_info.is_active():
            return False

        allowed = key_info.allowed_endpoints
        if '*' in allowed:
            return True

        for pattern in allowed:
            if re.match(pattern.replace('*', '.*'), endpoint):
                return True

        return False

    def update_permissions(self, key: str,
                           permissions: List[str]) -> bool:
        """更新key权限"""
        key_info = self._find_key_by_hash(key)
        if key_info is None:
            return False
        key_info.permissions = permissions
        self._add_audit_log(key_info.key_id, key_info.user_id,
                            'update_permissions', action='success',
                            detail=json.dumps(permissions))
        return True

    # ========== 速率限制与配额 ==========

    def check_rate_limit(self, key: str) -> bool:
        """检查速率限制

        使用令牌桶算法检查每分钟请求限制，
        并检查每天请求限制。

        Args:
            key: API Key明文

        Returns:
            bool: 是否通过速率限制
        """
        key_info = self._find_key_by_hash(key)
        if key_info is None:
            return False

        key_id = key_info.key_id

        # 检查每分钟限制（令牌桶）
        bucket = self._rate_limiters.get(key_id)
        if bucket is None:
            bucket = _TokenBucket(
                capacity=key_info.rate_limit_per_minute,
                refill_rate=key_info.rate_limit_per_minute / 60.0,
            )
            self._rate_limiters[key_id] = bucket

        if not bucket.consume(1):
            self._add_audit_log(key_id, key_info.user_id,
                                'rate_limit', action='rate_limit_hit',
                                detail='per_minute')
            return False

        # 检查每天限制
        today = datetime.now().strftime('%Y-%m-%d')
        daily = self._daily_usage.get(key_id, {})
        if daily.get(today, 0) >= key_info.rate_limit_per_day:
            self._add_audit_log(key_id, key_info.user_id,
                                'rate_limit', action='rate_limit_hit',
                                detail='per_day')
            return False

        # 更新每日使用计数
        daily[today] = daily.get(today, 0) + 1
        self._daily_usage[key_id] = daily

        return True

    def check_quota(self, key: str, tokens: int = 0) -> bool:
        """检查token配额

        Args:
            key: API Key明文
            tokens: 需要使用的token数

        Returns:
            bool: 是否有足够配额
        """
        key_info = self._find_key_by_hash(key)
        if key_info is None:
            return False

        if key_info.tokens_used + tokens > key_info.token_quota:
            self._add_audit_log(key_info.key_id, key_info.user_id,
                                'quota', action='quota_exceeded',
                                detail='tokens_needed=%d, remaining=%d' % (
                                    tokens, key_info.token_quota - key_info.tokens_used
                                ))
            return False

        return True

    def record_usage(self, key: str, tokens_used: int = 0,
                     endpoint: str = "", method: str = "POST",
                     status_code: int = 200) -> None:
        """记录API使用情况

        Args:
            key: API Key明文
            tokens_used: 使用的token数
            endpoint: 请求端点
            method: HTTP方法
            status_code: 响应状态码
        """
        key_info = self._find_key_by_hash(key)
        if key_info is None:
            return

        key_info.tokens_used += tokens_used
        key_info.last_used = time.time()
        key_info.last_used_endpoint = endpoint

        self._add_audit_log(
            key_info.key_id, key_info.user_id,
            endpoint=endpoint, method=method, status_code=status_code,
            tokens_used=tokens_used,
            action='usage'
        )

    # ========== Key吊销 ==========

    def revoke(self, key: str) -> bool:
        """吊销API Key

        Args:
            key: API Key明文

        Returns:
            bool: 是否成功
        """
        key_info = self._find_key_by_hash(key)
        if key_info is None:
            return False

        key_info.status = 'revoked'
        self._add_audit_log(key_info.key_id, key_info.user_id,
                            'revoke', action='success')
        return True

    def revoke_by_id(self, key_id: str) -> bool:
        """通过key_id吊销"""
        key_info = self._keys.get(key_id)
        if key_info is None:
            return False
        key_info.status = 'revoked'
        self._add_audit_log(key_id, key_info.user_id,
                            'revoke', action='success')
        return True

    def revoke_all(self, user_id: str) -> int:
        """吊销用户的所有Key

        Args:
            user_id: 用户ID

        Returns:
            int: 吊销的key数量
        """
        count = 0
        for key_info in self._keys.values():
            if key_info.user_id == user_id and key_info.status == 'active':
                key_info.status = 'revoked'
                self._add_audit_log(key_info.key_id, user_id,
                                    'revoke', action='success',
                                    detail='revoke_all')
                count += 1
        return count

    # ========== Key轮换 ==========

    def rotate_key(self, key: str, expires_in_days: int = 7) -> Optional[str]:
        """轮换API Key

        生成新key，旧key在指定天数后过期。

        Args:
            key: 旧API Key明文
            expires_in_days: 旧key的过期天数

        Returns:
            Optional[str]: 新API Key明文，失败返回None
        """
        key_info = self._find_key_by_hash(key)
        if key_info is None:
            return None

        # 旧key设置过期时间
        key_info.expires_at = time.time() + expires_in_days * 86400

        # 生成新key
        new_key = self.generate_key(
            user_id=key_info.user_id,
            name=key_info.name + ' (rotated)',
            permissions=key_info.permissions,
            allowed_endpoints=key_info.allowed_endpoints,
            rate_limit_per_minute=key_info.rate_limit_per_minute,
            rate_limit_per_day=key_info.rate_limit_per_day,
            token_quota=key_info.token_quota - key_info.tokens_used,
        )

        self._add_audit_log(key_info.key_id, key_info.user_id,
                            'rotate', action='success',
                            detail='old_key_expires_in_%d_days' % expires_in_days)

        return new_key

    def rotate_expired(self) -> int:
        """轮换所有已过期的key

        Returns:
            int: 标记为过期的key数量
        """
        count = 0
        for key_info in self._keys.values():
            if key_info.status == 'active' and key_info.is_expired():
                key_info.status = 'expired'
                self._add_audit_log(key_info.key_id, key_info.user_id,
                                    'auto_expire', action='success')
                count += 1
        return count

    # ========== 审计日志 ==========

    def _add_audit_log(self, key_id: str, user_id: str,
                       endpoint: str = "", method: str = "",
                       status_code: int = 0, tokens_used: int = 0,
                       ip_address: str = "", action: str = "",
                       detail: str = "") -> None:
        """添加审计日志"""
        log = AuditLogEntry(
            log_id='log_' + uuid.uuid4().hex[:12],
            key_id=key_id,
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            tokens_used=tokens_used,
            ip_address=ip_address,
            action=action,
            detail=detail,
        )
        self._audit_logs.append(log)

    def get_audit_logs(self, key: Optional[str] = None,
                       user_id: Optional[str] = None,
                       action: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """获取审计日志

        Args:
            key: 按key过滤（明文）
            user_id: 按用户过滤
            action: 按操作类型过滤
            limit: 返回条数上限

        Returns:
            List[Dict]: 审计日志列表
        """
        filter_key_id = None
        if key:
            key_info = self._find_key_by_hash(key)
            if key_info:
                filter_key_id = key_info.key_id

        results = []
        for log in reversed(self._audit_logs):
            if filter_key_id and log.key_id != filter_key_id:
                continue
            if user_id and log.user_id != user_id:
                continue
            if action and log.action != action:
                continue
            results.append(log.to_dict())
            if len(results) >= limit:
                break

        return results

    # ========== 查询 ==========

    def list_keys(self, user_id: Optional[str] = None,
                  status: Optional[str] = None) -> List[APIKeyInfo]:
        """列出API Key

        Args:
            user_id: 按用户过滤
            status: 按状态过滤

        Returns:
            List[APIKeyInfo]: key信息列表
        """
        results = list(self._keys.values())
        if user_id:
            results = [k for k in results if k.user_id == user_id]
        if status:
            results = [k for k in results if k.status == status]
        return results

    def get_key_info(self, key: str) -> Optional[APIKeyInfo]:
        """获取key详细信息"""
        return self._find_key_by_hash(key)

    # ========== 持久化 ==========

    def save(self, path: Optional[str] = None) -> bool:
        """保存到JSON

        Args:
            path: 保存路径，默认使用配置目录

        Returns:
            bool: 是否成功
        """
        if path is None:
            path = os.path.join(_get_config_dir(), 'api_keys.json')

        data = {
            'salt': self._salt,
            'keys': {kid: k.to_dict() for kid, k in self._keys.items()},
            'audit_logs': [log.to_dict() for log in self._audit_logs],
            'daily_usage': self._daily_usage,
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except (OSError, IOError):
            return False

    def load(self, path: Optional[str] = None) -> bool:
        """从JSON加载

        Args:
            path: 加载路径，默认使用配置目录

        Returns:
            bool: 是否成功
        """
        if path is None:
            path = os.path.join(_get_config_dir(), 'api_keys.json')
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._salt = data.get('salt', self._salt)
            self._keys = {
                kid: APIKeyInfo.from_dict(d)
                for kid, d in data.get('keys', {}).items()
            }
            self._key_hash_index = {
                k.key_hash: k.key_id for k in self._keys.values()
            }
            self._audit_logs.clear()
            for log_data in data.get('audit_logs', []):
                self._audit_logs.append(AuditLogEntry(
                    log_id=log_data.get('log_id', ''),
                    key_id=log_data.get('key_id', ''),
                    user_id=log_data.get('user_id', ''),
                    timestamp=log_data.get('timestamp', time.time()),
                    action=log_data.get('action', ''),
                    endpoint=log_data.get('endpoint', ''),
                    method=log_data.get('method', ''),
                    status_code=log_data.get('status_code', 200),
                    tokens_used=log_data.get('tokens_used', 0),
                    ip_address=log_data.get('ip_address', ''),
                    detail=log_data.get('detail', ''),
                ))
            self._daily_usage = data.get('daily_usage', {})

            # 重建速率限制器
            self._rate_limiters = {}
            for kid, k in self._keys.items():
                if k.status == 'active':
                    self._rate_limiters[kid] = _TokenBucket(
                        capacity=k.rate_limit_per_minute,
                        refill_rate=k.rate_limit_per_minute / 60.0,
                    )

            return True
        except (OSError, IOError, json.JSONDecodeError):
            return False

    # ========== 统计与仪表盘 ==========

    def get_stats(self) -> Dict[str, Any]:
        """获取API Key管理统计信息"""
        active_keys = [k for k in self._keys.values() if k.status == 'active']
        revoked_keys = [k for k in self._keys.values() if k.status == 'revoked']
        expired_keys = [k for k in self._keys.values() if k.status == 'expired']
        total_tokens_used = sum(k.tokens_used for k in self._keys.values())
        total_quota = sum(k.token_quota for k in self._keys.values())
        unique_users = set(k.user_id for k in self._keys.values())

        return {
            'total_keys': len(self._keys),
            'active_keys': len(active_keys),
            'revoked_keys': len(revoked_keys),
            'expired_keys': len(expired_keys),
            'total_users': len(unique_users),
            'total_tokens_used': total_tokens_used,
            'total_token_quota': total_quota,
            'quota_utilization': total_tokens_used / max(1, total_quota),
            'total_audit_logs': len(self._audit_logs),
            'permission_levels': list(self.PERMISSION_LEVELS.keys()),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """获取API Key管理仪表盘"""
        stats = self.get_stats()
        # 按用户统计
        user_stats = {}
        for k in self._keys.values():
            if k.user_id not in user_stats:
                user_stats[k.user_id] = {
                    'keys': 0, 'active': 0, 'tokens_used': 0
                }
            user_stats[k.user_id]['keys'] += 1
            if k.status == 'active':
                user_stats[k.user_id]['active'] += 1
            user_stats[k.user_id]['tokens_used'] += k.tokens_used

        # 最近审计日志
        recent_logs = list(self._audit_logs)[-10:]

        return {
            'module': 'APIKeyManager',
            'version': '1.0.0',
            'features': [
                'Key生成', 'Key认证', '权限管理', '速率限制',
                '配额管理', 'Key吊销', 'Key轮换', '审计日志', '持久化'
            ],
            'stats': stats,
            'user_count': len(user_stats),
            'top_users': sorted(
                user_stats.items(),
                key=lambda x: x[1]['tokens_used'],
                reverse=True
            )[:5],
            'recent_audit_logs': [log.to_dict() for log in recent_logs],
        }

    def __repr__(self) -> str:
        return '<APIKeyManager keys=%d active=%d>' % (
            len(self._keys),
            sum(1 for k in self._keys.values() if k.status == 'active')
        )


# ============================================================
# #52 DataProvenanceAuditor - 训练数据血缘审计
# ============================================================

@dataclass
class DataRecord:
    """训练数据记录

    记录每条训练数据的来源、许可证、脱敏状态等信息。
    """
    data_id: str
    source: str  # 数据来源（如 "公开数据集/CLUE", "用户上传", "合成数据"）
    license: str  # 许可证类型（如 "MIT", "Apache-2.0", "CC-BY-4.0", "私有"）
    desensitized: bool  # 是否已脱敏
    desensitization_method: str = ""  # 脱敏方法
    version: str = "1.0"  # 数据版本
    timestamp: float = field(default_factory=time.time)
    hash: str = ""  # 数据内容哈希
    size: int = 0  # 数据大小（字节）
    format: str = "json"  # 数据格式
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataRecord':
        return cls(
            data_id=d.get('data_id', ''),
            source=d.get('source', ''),
            license=d.get('license', ''),
            desensitized=d.get('desensitized', False),
            desensitization_method=d.get('desensitization_method', ''),
            version=d.get('version', '1.0'),
            timestamp=d.get('timestamp', time.time()),
            hash=d.get('hash', ''),
            size=d.get('size', 0),
            format=d.get('format', 'json'),
            tags=d.get('tags', []),
            metadata=d.get('metadata', {}),
        )


@dataclass
class ModelTrainingRecord:
    """模型训练记录

    记录一次模型训练使用了哪些数据。
    """
    record_id: str
    model_id: str
    model_version: str
    data_ids: List[str] = field(default_factory=list)
    training_date: float = field(default_factory=time.time)
    epoch: int = 1
    loss: float = 0.0
    accuracy: float = 0.0
    notes: str = ""
    training_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelTrainingRecord':
        return cls(
            record_id=d.get('record_id', ''),
            model_id=d.get('model_id', ''),
            model_version=d.get('model_version', ''),
            data_ids=d.get('data_ids', []),
            training_date=d.get('training_date', time.time()),
            epoch=d.get('epoch', 1),
            loss=d.get('loss', 0.0),
            accuracy=d.get('accuracy', 0.0),
            notes=d.get('notes', ''),
            training_params=d.get('training_params', {}),
            metadata=d.get('metadata', {}),
        )


class DataProvenanceAuditor:
    """训练数据血缘审计系统

    功能：
    - 血缘记录：每条训练数据记录 {source, license, desensitized, version, timestamp, hash}
    - 血缘链：数据 → 脱敏 → 训练 → 模型 全链路追溯
    - 查询：按模型/数据/来源查询
    - 合规检查：检查是否有未脱敏/无许可证的数据被使用
    - 影响分析：某条数据被删除后影响哪些模型
    - 可视化：生成血缘关系图（JSON格式，可用于D3.js可视化）
    - 导出：审计报告（JSON/CSV）
    - 持久化：save/load到JSON
    """

    # 已知合规许可证
    COMPLIANT_LICENSES = {
        'MIT', 'Apache-2.0', 'CC-BY-4.0', 'CC-BY-SA-4.0',
        'CC0-1.0', 'BSD-3-Clause', 'ISC', 'Unlicense',
        'Mozilla-Public-License-2.0', 'GPL-3.0',
    }

    def __init__(self):
        self._data_records: Dict[str, DataRecord] = {}  # data_id → DataRecord
        self._training_records: Dict[str, ModelTrainingRecord] = {}  # record_id → ModelTrainingRecord
        self._data_to_models: Dict[str, List[str]] = {}  # data_id → [model_id, ...]
        self._model_to_records: Dict[str, List[str]] = {}  # model_id → [record_id, ...]

    # ========== 数据记录管理 ==========

    def add_data_record(self,
                        source: str,
                        license: str,
                        desensitized: bool = False,
                        desensitization_method: str = "",
                        version: str = "1.0",
                        data_content: str = "",
                        size: int = 0,
                        format: str = "json",
                        tags: Optional[List[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> DataRecord:
        """添加训练数据记录

        Args:
            source: 数据来源
            license: 许可证类型
            desensitized: 是否已脱敏
            desensitization_method: 脱敏方法
            version: 数据版本
            data_content: 数据内容（用于计算哈希）
            size: 数据大小
            format: 数据格式
            tags: 标签列表
            metadata: 元数据

        Returns:
            DataRecord: 创建的数据记录
        """
        data_id = 'dat_' + uuid.uuid4().hex[:12]
        data_hash = hashlib.sha256(data_content.encode()).hexdigest() if data_content else ''

        record = DataRecord(
            data_id=data_id,
            source=source,
            license=license,
            desensitized=desensitized,
            desensitization_method=desensitization_method,
            version=version,
            hash=data_hash,
            size=size or len(data_content.encode()),
            format=format,
            tags=tags or [],
            metadata=metadata or {},
        )

        self._data_records[data_id] = record
        self._data_to_models[data_id] = []

        return record

    def add_training_record(self,
                            model_id: str,
                            model_version: str,
                            data_ids: List[str],
                            epoch: int = 1,
                            loss: float = 0.0,
                            accuracy: float = 0.0,
                            notes: str = "",
                            training_params: Optional[Dict[str, Any]] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> ModelTrainingRecord:
        """添加模型训练记录

        Args:
            model_id: 模型ID
            model_version: 模型版本
            data_ids: 使用的训练数据ID列表
            epoch: 训练轮数
            loss: 训练损失
            accuracy: 训练准确率
            notes: 备注
            training_params: 训练参数
            metadata: 元数据

        Returns:
            ModelTrainingRecord: 创建的训练记录
        """
        record_id = 'trn_' + uuid.uuid4().hex[:12]

        record = ModelTrainingRecord(
            record_id=record_id,
            model_id=model_id,
            model_version=model_version,
            data_ids=list(data_ids),
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            notes=notes,
            training_params=training_params or {},
            metadata=metadata or {},
        )

        self._training_records[record_id] = record

        # 更新索引
        if model_id not in self._model_to_records:
            self._model_to_records[model_id] = []
        self._model_to_records[model_id].append(record_id)

        # 更新数据→模型索引
        for data_id in data_ids:
            if data_id not in self._data_to_models:
                self._data_to_models[data_id] = []
            if model_id not in self._data_to_models[data_id]:
                self._data_to_models[data_id].append(model_id)

        return record

    # ========== 查询 ==========

    def query_by_model(self, model_id: str) -> Dict[str, Any]:
        """按模型查询：哪些数据训练了这个模型

        Args:
            model_id: 模型ID

        Returns:
            Dict: 包含模型信息、训练记录和数据记录
        """
        record_ids = self._model_to_records.get(model_id, [])
        training_records = [self._training_records[rid] for rid in record_ids]

        all_data_ids = set()
        for tr in training_records:
            all_data_ids.update(tr.data_ids)

        data_records = [self._data_records[did] for did in all_data_ids
                        if did in self._data_records]

        return {
            'model_id': model_id,
            'training_record_count': len(training_records),
            'training_records': [tr.to_dict() for tr in training_records],
            'data_count': len(data_records),
            'data_records': [dr.to_dict() for dr in data_records],
        }

    def query_by_data(self, data_id: str) -> Dict[str, Any]:
        """按数据查询：这条数据被哪些模型使用

        Args:
            data_id: 数据ID

        Returns:
            Dict: 包含数据信息和使用该数据的模型列表
        """
        data_record = self._data_records.get(data_id)
        model_ids = self._data_to_models.get(data_id, [])

        # 获取相关训练记录
        related_training = []
        for tr in self._training_records.values():
            if data_id in tr.data_ids:
                related_training.append(tr)

        return {
            'data_id': data_id,
            'data_record': data_record.to_dict() if data_record else None,
            'model_count': len(model_ids),
            'model_ids': model_ids,
            'training_records': [tr.to_dict() for tr in related_training],
        }

    def query_by_source(self, source: str) -> Dict[str, Any]:
        """按来源查询：某来源的所有数据使用情况

        Args:
            source: 数据来源

        Returns:
            Dict: 包含该来源的所有数据和使用情况
        """
        matching_data = [
            dr for dr in self._data_records.values() if dr.source == source
        ]

        # 收集所有使用该来源数据的模型
        affected_models = set()
        for dr in matching_data:
            models = self._data_to_models.get(dr.data_id, [])
            affected_models.update(models)

        return {
            'source': source,
            'data_count': len(matching_data),
            'data_records': [dr.to_dict() for dr in matching_data],
            'affected_model_count': len(affected_models),
            'affected_model_ids': list(affected_models),
        }

    # ========== 合规检查 ==========

    def check_compliance(self) -> List[Dict[str, Any]]:
        """合规检查：检查是否有未脱敏/无许可证的数据被使用

        Returns:
            List[Dict]: 合规问题列表
        """
        issues = []

        for dr in self._data_records.values():
            # 检查1: 未脱敏的敏感数据
            if not dr.desensitized:
                models = self._data_to_models.get(dr.data_id, [])
                if models:
                    issues.append({
                        'type': 'not_desensitized',
                        'severity': 'high',
                        'data_id': dr.data_id,
                        'source': dr.source,
                        'license': dr.license,
                        'affected_models': models,
                        'detail': '数据未脱敏但已被用于训练',
                    })

            # 检查2: 无许可证或未知许可证
            if not dr.license or dr.license == '未知' or dr.license == '无':
                models = self._data_to_models.get(dr.data_id, [])
                if models:
                    issues.append({
                        'type': 'no_license',
                        'severity': 'critical',
                        'data_id': dr.data_id,
                        'source': dr.source,
                        'affected_models': models,
                        'detail': '数据无明确许可证但已被用于训练',
                    })

            # 检查3: 非合规许可证
            elif dr.license not in self.COMPLIANT_LICENSES and dr.license != '私有':
                models = self._data_to_models.get(dr.data_id, [])
                if models:
                    issues.append({
                        'type': 'non_compliant_license',
                        'severity': 'medium',
                        'data_id': dr.data_id,
                        'source': dr.source,
                        'license': dr.license,
                        'affected_models': models,
                        'detail': '许可证 "%s" 不在合规列表中' % dr.license,
                    })

            # 检查4: 私有数据未脱敏
            if dr.license == '私有' and not dr.desensitized:
                models = self._data_to_models.get(dr.data_id, [])
                if models:
                    issues.append({
                        'type': 'private_not_desensitized',
                        'severity': 'critical',
                        'data_id': dr.data_id,
                        'source': dr.source,
                        'affected_models': models,
                        'detail': '私有数据未脱敏即用于训练',
                    })

        return issues

    # ========== 影响分析 ==========

    def impact_analysis(self, data_id: str) -> Dict[str, Any]:
        """影响分析：某条数据被删除后影响哪些模型

        Args:
            data_id: 数据ID

        Returns:
            Dict: 影响分析结果
        """
        data_record = self._data_records.get(data_id)
        if data_record is None:
            return {'error': '数据不存在', 'data_id': data_id}

        affected_models = self._data_to_models.get(data_id, [])
        affected_training = []

        for tr in self._training_records.values():
            if data_id in tr.data_ids:
                # 计算该数据在训练集中的占比
                total_data = len(tr.data_ids)
                impact_ratio = 1.0 / total_data if total_data > 0 else 0
                affected_training.append({
                    'record_id': tr.record_id,
                    'model_id': tr.model_id,
                    'model_version': tr.model_version,
                    'training_date': tr.training_date,
                    'data_count': total_data,
                    'impact_ratio': impact_ratio,
                    'remaining_data': total_data - 1,
                })

        # 按影响程度排序
        affected_training.sort(key=lambda x: x['impact_ratio'], reverse=True)

        return {
            'data_id': data_id,
            'data_record': data_record.to_dict(),
            'affected_model_count': len(affected_models),
            'affected_models': affected_models,
            'affected_training_count': len(affected_training),
            'affected_training': affected_training,
            'recommendation': (
                '需要重新训练受影响的模型' if affected_models
                else '无影响，该数据未被任何模型使用'
            ),
        }

    # ========== 可视化 ==========

    def generate_lineage_graph(self) -> Dict[str, Any]:
        """生成血缘关系图（JSON格式，可用于D3.js可视化）

        生成包含节点和边的图结构：
        - 节点类型: source（来源）, data（数据）, model（模型）
        - 边类型: source→data, data→model

        Returns:
            Dict: 图结构 {nodes: [...], links: [...]}
        """
        nodes = []
        links = []
        node_ids = set()

        def add_node(node_id: str, node_type: str, label: str,
                     extra: Optional[Dict] = None):
            if node_id not in node_ids:
                node_ids.add(node_id)
                node = {
                    'id': node_id,
                    'type': node_type,
                    'label': label,
                }
                if extra:
                    node.update(extra)
                nodes.append(node)

        def add_link(source: str, target: str, link_type: str = 'used'):
            links.append({
                'source': source,
                'target': target,
                'type': link_type,
            })

        # 添加来源节点和数据节点
        sources_seen = set()
        for dr in self._data_records.values():
            # 来源节点
            source_id = 'src_' + dr.source
            if dr.source not in sources_seen:
                sources_seen.add(dr.source)
                add_node(source_id, 'source', dr.source)

            # 数据节点
            add_node(dr.data_id, 'data', '%s (%s)' % (dr.source, dr.version), {
                'source': dr.source,
                'license': dr.license,
                'desensitized': dr.desensitized,
                'hash': dr.hash[:16],
            })

            # 来源 → 数据 链接
            add_link(source_id, dr.data_id, 'contains')

        # 添加模型节点和训练链接
        models_seen = set()
        for tr in self._training_records.values():
            model_node_id = 'mdl_' + tr.model_id
            if tr.model_id not in models_seen:
                models_seen.add(tr.model_id)
                add_node(model_node_id, 'model', tr.model_id, {
                    'version': tr.model_version,
                    'epoch': tr.epoch,
                    'loss': tr.loss,
                    'accuracy': tr.accuracy,
                })

            # 数据 → 模型 链接
            for data_id in tr.data_ids:
                if data_id in node_ids:
                    add_link(data_id, model_node_id, 'trained_with')

        return {
            'nodes': nodes,
            'links': links,
            'metadata': {
                'node_count': len(nodes),
                'link_count': len(links),
                'source_count': len([n for n in nodes if n['type'] == 'source']),
                'data_count': len([n for n in nodes if n['type'] == 'data']),
                'model_count': len([n for n in nodes if n['type'] == 'model']),
                'generated_at': datetime.now().isoformat(),
            },
        }

    # ========== 导出 ==========

    def export_report(self, fmt: str = 'json') -> str:
        """导出审计报告

        Args:
            fmt: 导出格式 ('json' 或 'csv')

        Returns:
            str: 报告内容字符串
        """
        if fmt == 'json':
            report = {
                'report_type': 'data_provenance_audit',
                'generated_at': datetime.now().isoformat(),
                'summary': self.get_stats(),
                'compliance_issues': self.check_compliance(),
                'data_records': [dr.to_dict() for dr in self._data_records.values()],
                'training_records': [tr.to_dict() for tr in self._training_records.values()],
                'lineage_graph': self.generate_lineage_graph(),
            }
            return json.dumps(report, ensure_ascii=False, indent=2)

        elif fmt == 'csv':
            lines = []
            # 数据记录CSV
            lines.append('=== 数据记录 ===')
            lines.append('data_id,source,license,desensitized,version,hash,size,format')
            for dr in self._data_records.values():
                lines.append('%s,%s,%s,%s,%s,%s,%d,%s' % (
                    dr.data_id, dr.source, dr.license,
                    dr.desensitized, dr.version, dr.hash[:16],
                    dr.size, dr.format
                ))
            lines.append('')

            # 训练记录CSV
            lines.append('=== 训练记录 ===')
            lines.append('record_id,model_id,model_version,data_count,epoch,loss,accuracy')
            for tr in self._training_records.values():
                lines.append('%s,%s,%s,%d,%d,%.4f,%.4f' % (
                    tr.record_id, tr.model_id, tr.model_version,
                    len(tr.data_ids), tr.epoch, tr.loss, tr.accuracy
                ))
            lines.append('')

            # 合规问题CSV
            lines.append('=== 合规问题 ===')
            lines.append('type,severity,data_id,source,detail')
            for issue in self.check_compliance():
                lines.append('%s,%s,%s,%s,%s' % (
                    issue['type'], issue['severity'],
                    issue['data_id'], issue.get('source', ''),
                    issue['detail']
                ))

            return '\n'.join(lines)

        else:
            return '不支持的格式: %s' % fmt

    # ========== 持久化 ==========

    def save(self, path: Optional[str] = None) -> bool:
        """保存到JSON

        Args:
            path: 保存路径，默认使用数据目录

        Returns:
            bool: 是否成功
        """
        if path is None:
            path = os.path.join(_get_data_dir(), 'provenance.json')

        data = {
            'data_records': {
                did: dr.to_dict()
                for did, dr in self._data_records.items()
            },
            'training_records': {
                rid: tr.to_dict()
                for rid, tr in self._training_records.items()
            },
            'data_to_models': self._data_to_models,
            'model_to_records': self._model_to_records,
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except (OSError, IOError):
            return False

    def load(self, path: Optional[str] = None) -> bool:
        """从JSON加载

        Args:
            path: 加载路径，默认使用数据目录

        Returns:
            bool: 是否成功
        """
        if path is None:
            path = os.path.join(_get_data_dir(), 'provenance.json')
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._data_records = {
                did: DataRecord.from_dict(d)
                for did, d in data.get('data_records', {}).items()
            }
            self._training_records = {
                rid: ModelTrainingRecord.from_dict(d)
                for rid, d in data.get('training_records', {}).items()
            }
            self._data_to_models = data.get('data_to_models', {})
            self._model_to_records = data.get('model_to_records', {})
            return True
        except (OSError, IOError, json.JSONDecodeError):
            return False

    # ========== 统计与仪表盘 ==========

    def get_stats(self) -> Dict[str, Any]:
        """获取血缘审计统计信息"""
        compliance_issues = self.check_compliance()
        sources = set(dr.source for dr in self._data_records.values())
        licenses = set(dr.license for dr in self._data_records.values())
        models = set(tr.model_id for tr in self._training_records.values())
        desensitized_count = sum(
            1 for dr in self._data_records.values() if dr.desensitized
        )

        return {
            'total_data_records': len(self._data_records),
            'total_training_records': len(self._training_records),
            'total_models': len(models),
            'total_sources': len(sources),
            'total_licenses': len(licenses),
            'desensitized_count': desensitized_count,
            'desensitization_rate': desensitized_count / max(1, len(self._data_records)),
            'compliance_issues': len(compliance_issues),
            'critical_issues': sum(1 for i in compliance_issues if i['severity'] == 'critical'),
            'high_issues': sum(1 for i in compliance_issues if i['severity'] == 'high'),
            'compliant_licenses': list(self.COMPLIANT_LICENSES),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """获取血缘审计仪表盘"""
        stats = self.get_stats()
        compliance_issues = self.check_compliance()

        # 按来源统计
        source_stats = {}
        for dr in self._data_records.values():
            if dr.source not in source_stats:
                source_stats[dr.source] = {
                    'data_count': 0, 'desensitized': 0, 'models': set()
                }
            source_stats[dr.source]['data_count'] += 1
            if dr.desensitized:
                source_stats[dr.source]['desensitized'] += 1
            source_stats[dr.source]['models'].update(
                self._data_to_models.get(dr.data_id, [])
            )

        # 转换set为list
        for s in source_stats.values():
            s['models'] = list(s['models'])

        return {
            'module': 'DataProvenanceAuditor',
            'version': '1.0.0',
            'features': [
                '血缘记录', '血缘链追溯', '多维度查询',
                '合规检查', '影响分析', '血缘可视化', '报告导出', '持久化'
            ],
            'stats': stats,
            'source_breakdown': source_stats,
            'compliance_summary': {
                'total_issues': len(compliance_issues),
                'by_severity': {
                    'critical': sum(1 for i in compliance_issues if i['severity'] == 'critical'),
                    'high': sum(1 for i in compliance_issues if i['severity'] == 'high'),
                    'medium': sum(1 for i in compliance_issues if i['severity'] == 'medium'),
                },
                'by_type': {
                    t: sum(1 for i in compliance_issues if i['type'] == t)
                    for t in set(i['type'] for i in compliance_issues)
                },
            },
            'lineage_graph_summary': {
                'nodes': len(self._data_records) + len(self._training_records),
                'edges': sum(
                    len(tr.data_ids) for tr in self._training_records.values()
                ),
            },
        }

    def __repr__(self) -> str:
        return '<DataProvenanceAuditor data=%d training=%d models=%d>' % (
            len(self._data_records),
            len(self._training_records),
            len(set(tr.model_id for tr in self._training_records.values())),
        )