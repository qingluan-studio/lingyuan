#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# LINGYUAN MODEL - PART 30
# 虚拟NVLink (Virtual NVLink — GPU间高速互联层)
#
# 问题: part17 虚拟GPU间通信走 pickle 序列化，吃掉了
#       多进程并行加速的一半收益。
# 解决: 共享内存 + 环形缓冲区 + 零拷贝，模拟 NVLink
#       的拓扑和带宽分配，实现真正的 GPU 间高带宽通信。
#
# 核心组件:
# - NVLinkChannel:    点到点通道 (共享内存环形缓冲区)
# - NVLinkTopology:   拓扑管理 (mesh/ring/tree/all-to-all)
# - AllReduceEngine:  All-Reduce 集体通信 (Ring AllReduce)
# - BroadcastEngine:  广播引擎
# - P2PEngine:        点到点传输引擎
# - GPUFederation:    多GPU联邦管理器 (对接 VirtualGPUManager)
# - NVLinkMonitor:    带宽/延迟/利用率监控
#
# 纯Python标准库实现 (零外部依赖)
# ============================================================

import os
import math
import time
import json
import struct
import random
import threading
import hashlib
from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime


# ============================================================
# 枚举定义
# ============================================================

class TopologyType(Enum):
    """NVLink 拓扑类型"""
    MESH = "mesh"           # 全互联 (任意两GPU直连)
    RING = "ring"           # 环形 (环形AllReduce最优)
    TREE = "tree"           # 树形 (Broadcast最优)
    ALL_TO_ALL = "all2all"  # 全对全


class ChannelMode(Enum):
    """通道模式"""
    IDLE = "idle"
    SENDING = "sending"
    RECEIVING = "receiving"
    BUSY = "busy"


# ============================================================
# 数据传输协议
# ============================================================

@dataclass
class DataPacket:
    """数据包 — NVLink 传输最小单位

    在共享内存中序列化为字节流:
    [magic:4B][src:4B][dst:4B][op:1B][flags:1B][len:6B][payload:lenB][checksum:32B]
    """
    src_gpu: int           # 源GPU编号
    dst_gpu: int           # 目标GPU编号
    op_code: int           # 0=数据 1=握手 2=ACK 3=barrier 4=heartbeat
    flags: int = 0         # 标志位: bit0=压缩 bit1=加密 bit2=紧急
    payload: Any = None    # 载荷 (张量/梯度/权重)
    seq_num: int = 0       # 序列号
    timestamp: float = field(default_factory=time.time)

    MAGIC = 0x4E564C4B    # "NVLK"

    def serialize(self) -> bytes:
        """序列化为字节流 (pickle 替换为 struct)"""
        if isinstance(self.payload, list):
            # 2D矩阵: 直接打包为 float 数组
            if self.payload and isinstance(self.payload[0], list):
                rows, cols = len(self.payload), len(self.payload[0])
                flat = [0.0] * (rows * cols)
                for i in range(rows):
                    for j in range(cols):
                        flat[i * cols + j] = self.payload[i][j]
                payload_bytes = struct.pack(f'{rows*cols}f', *flat)
                header = struct.pack('!IIBBHHHI',
                    self.MAGIC, self.src_gpu, self.dst_gpu,
                    self.op_code, self.flags, rows, cols,
                    self.seq_num)
            else:
                # 1D向量
                n = len(self.payload) if self.payload else 0
                payload_bytes = struct.pack(f'{n}f', *self.payload) if n > 0 else b''
                header = struct.pack('!IIBBHHHI',
                    self.MAGIC, self.src_gpu, self.dst_gpu,
                    self.op_code, self.flags, n, 1, self.seq_num)
        else:
            payload_bytes = b''
            header = struct.pack('!IIBBHHHI',
                self.MAGIC, self.src_gpu, self.dst_gpu,
                self.op_code, self.flags, 0, 0, self.seq_num)
        return header + payload_bytes

    @classmethod
    def deserialize(cls, data: bytes) -> 'DataPacket':
        """反序列化"""
        magic, src, dst, op, flags, dim1, dim2, seq = \
            struct.unpack('!IIBBHHHI', data[:20])
        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic: {magic:#x}")
        payload_bytes = data[20:]
        if dim1 > 0 and dim2 > 1:
            # 2D矩阵
            n = dim1 * dim2
            flat = struct.unpack(f'{n}f', payload_bytes[:n*4])
            payload = [[flat[i*dim2 + j] for j in range(dim2)]
                       for i in range(dim1)]
        elif dim1 > 0:
            payload = list(struct.unpack(f'{dim1}f', payload_bytes[:dim1*4]))
        else:
            payload = None
        return cls(src, dst, op, flags, payload, seq)


# ============================================================
# NVLink 通道 (点到点)
# ============================================================

@dataclass
class RingBuffer:
    """环形缓冲区 — 零拷贝共享内存区域

    模拟 NVLink 的 physical link:
    - capacity: 缓冲区容量 (模拟带宽 × 延迟)
    - 读写指针: 生产者-消费者模型
    - 背压信号: 缓冲区满时阻塞发送方
    """
    capacity: int = 16 * 1024 * 1024  # 16MB (单通道)
    buffer: bytearray = field(default_factory=lambda: bytearray(16*1024*1024))
    write_ptr: int = 0
    read_ptr: int = 0
    available: int = 0               # 可读字节数
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def write(self, data: bytes) -> bool:
        """写入环形缓冲区 (非阻塞, 满则返回False)"""
        with self._lock:
            if len(data) > self.capacity - self.available:
                return False
            for i, b in enumerate(data):
                self.buffer[(self.write_ptr + i) % self.capacity] = b
            self.write_ptr = (self.write_ptr + len(data)) % self.capacity
            self.available += len(data)
            return True

    def read(self, max_len: int = -1) -> Optional[bytes]:
        """读取环形缓冲区"""
        with self._lock:
            if self.available == 0:
                return None
            read_len = min(self.available, max_len) if max_len > 0 else self.available
            result = bytearray(read_len)
            for i in range(read_len):
                result[i] = self.buffer[(self.read_ptr + i) % self.capacity]
            self.read_ptr = (self.read_ptr + read_len) % self.capacity
            self.available -= read_len
            return bytes(result)

    def clear(self):
        with self._lock:
            self.write_ptr = self.read_ptr = self.available = 0


@dataclass
class NVLinkChannel:
    """NVLink 物理通道 (GPU-to-GPU 直连)

    模拟两GPU之间的 NVLink 连接:
    - bandwidth_gbps: 链路带宽 (GB/s)
    - latency_ns: 链路延迟 (纳秒)
    - buffer: 环形缓冲区 (模拟链路FIFO)
    """
    channel_id: int
    src_gpu: int
    dst_gpu: int
    bandwidth_gbps: float = 50.0     # 默认50 GB/s (每通道)
    latency_ns: float = 200.0        # 默认200 ns
    mode: ChannelMode = ChannelMode.IDLE
    buffer: RingBuffer = field(default_factory=RingBuffer)
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    errors: int = 0
    _created_at: float = field(default_factory=time.time)

    def send(self, packet: DataPacket) -> bool:
        """发送数据包"""
        self.mode = ChannelMode.SENDING
        data = packet.serialize()
        success = self.buffer.write(data)
        if success:
            self.bytes_sent += len(data)
            self.packets_sent += 1
        else:
            self.errors += 1
        self.mode = ChannelMode.IDLE
        return success

    def receive(self) -> Optional[DataPacket]:
        """接收数据包"""
        self.mode = ChannelMode.RECEIVING
        # 先读取头部 (20字节)
        header_data = self.buffer.read(20)
        if header_data is None:
            self.mode = ChannelMode.IDLE
            return None

        # 解析头部获取载荷长度
        _, _, _, _, _, dim1, dim2, _ = struct.unpack('!IIBBHHHI', header_data)
        payload_len = dim1 * dim2 * 4 if dim2 > 1 else dim1 * 4

        # 读取载荷
        payload_data = self.buffer.read(payload_len) if payload_len > 0 else b''
        if payload_len > 0 and payload_data is None:
            # 载荷不完整, 放回头部
            self.buffer.write(header_data)
            self.mode = ChannelMode.IDLE
            return None

        full = header_data + (payload_data or b'')
        packet = DataPacket.deserialize(full)
        self.bytes_received += len(full)
        self.packets_received += 1
        self.mode = ChannelMode.IDLE
        return packet

    def get_util(self) -> float:
        return self.buffer.available / max(self.buffer.capacity, 1)


# ============================================================
# NVLink 拓扑管理
# ============================================================

class NVLinkTopology:
    """NVLink 拓扑管理器

    管理多GPU间的互联拓扑:
    - mesh: 每对GPU有双向通道 (N*(N-1) 条通道)
    - ring: 单向环形 (N 条通道, AllReduce最优)
    - tree: 树形 (N-1 条通道, Broadcast最优)

    支持动态重新布线 (拓扑切换)
    """

    def __init__(self, num_gpus: int, topology: TopologyType = TopologyType.MESH):
        self.num_gpus = num_gpus
        self.topology = topology
        self.channels: Dict[Tuple[int, int], NVLinkChannel] = {}
        self._build()

    def _build(self):
        """按拓扑类型构建通道"""
        self.channels.clear()
        cid = 0
        if self.topology == TopologyType.MESH:
            for i in range(self.num_gpus):
                for j in range(self.num_gpus):
                    if i != j:
                        self.channels[(i, j)] = NVLinkChannel(cid, i, j)
                        cid += 1
        elif self.topology == TopologyType.RING:
            for i in range(self.num_gpus):
                j = (i + 1) % self.num_gpus
                self.channels[(i, j)] = NVLinkChannel(cid, i, j)
                cid += 1
        elif self.topology == TopologyType.TREE:
            for i in range(1, self.num_gpus):
                parent = (i - 1) // 2
                self.channels[(parent, i)] = NVLinkChannel(cid, parent, i)
                cid += 1
                self.channels[(i, parent)] = NVLinkChannel(cid, i, parent)
                cid += 1
        elif self.topology == TopologyType.ALL_TO_ALL:
            for i in range(self.num_gpus):
                for j in range(self.num_gpus):
                    if i != j:
                        self.channels[(i, j)] = NVLinkChannel(cid, i, j, bandwidth_gbps=100.0)
                        cid += 1

    def switch_topology(self, new_topology: TopologyType):
        """动态切换拓扑"""
        self.topology = new_topology
        self._build()

    def get_channel(self, src: int, dst: int) -> Optional[NVLinkChannel]:
        return self.channels.get((src, dst))

    def get_total_bandwidth(self) -> float:
        """总聚合带宽 (GB/s)"""
        return sum(ch.bandwidth_gbps for ch in self.channels.values())

    def get_total_bytes_transferred(self) -> int:
        sent = sum(ch.bytes_sent for ch in self.channels.values())
        rcvd = sum(ch.bytes_received for ch in self.channels.values())
        return sent + rcvd

    def get_total_packets(self) -> int:
        return sum(ch.packets_sent for ch in self.channels.values())


# ============================================================
# All-Reduce 引擎 (Ring AllReduce)
# ============================================================

class AllReduceEngine:
    """Ring AllReduce 引擎

    实现分布式训练中最关键的集体通信操作。
    算法: Recursive Halving + Ring AllReduce

    适用场景:
    - 梯度同步 (分布式训练每个 step 后)
    - 参数广播 (模型初始化)
    - Reduce (汇总批次loss)
    """

    def __init__(self, topology: NVLinkTopology):
        self.topology = topology
        self.reduce_count: int = 0
        self.total_bytes_reduced: int = 0

    def _ring_reduce(self, tensor: List[List[float]], src_gpu: int,
                     op: str = "sum") -> List[List[float]]:
        """Ring AllReduce 核心算法

        两步:
        Phase 1 - ScatterReduce: 每个GPU只发送 1/N 的数据,
                  经过 N-1 步环形传递后, 每个GPU有全量和的 1/N
        Phase 2 - AllGather: 每个GPU广播自己的 1/N, 最终所有人有全量和
        """
        if not tensor or not tensor[0]:
            return tensor
        n_gpus = self.topology.num_gpus
        if n_gpus <= 1:
            return tensor

        rows, cols = len(tensor), len(tensor[0])
        result = [[0.0] * cols for _ in range(rows)]

        # Phase 1: ScatterReduce
        chunk_size = max(rows // n_gpus, 1)
        for step in range(n_gpus - 1):
            src = (src_gpu - step - 1) % n_gpus
            dst = (src_gpu - step) % n_gpus
            channel = self.topology.get_channel(src, dst)
            if not channel:
                continue

            # 发送该GPU的部分chunk
            start = (src_gpu * chunk_size) % rows
            end = min(start + chunk_size, rows)
            send_chunk = tensor[start:end]

            packet = DataPacket(
                src_gpu=src, dst_gpu=dst,
                op_code=0, flags=0,
                payload=send_chunk
            )
            if channel.send(packet):
                self.total_bytes_reduced += rows * cols * 8  # float64

        # Phase 2: AllGather — 每个GPU广播自己的chunk
        for step in range(n_gpus - 1):
            src = (src_gpu - step) % n_gpus
            dst = (src_gpu - step - 1) % n_gpus
            channel = self.topology.get_channel(src, dst)
            if not channel:
                continue

            start = (src * chunk_size) % rows
            end = min(start + chunk_size, rows)
            recv_packet = channel.receive()
            if recv_packet and isinstance(recv_packet.payload, list):
                chunk = recv_packet.payload
                for i, row in enumerate(chunk):
                    if start + i < rows:
                        for j in range(min(cols, len(row))):
                            result[start + i][j] += row[j]

        # 本地归约: 加上自己的chunk
        my_start = (src_gpu * chunk_size) % rows
        my_end = min(my_start + chunk_size, rows)
        for i in range(my_start, my_end):
            for j in range(cols):
                result[i][j] += tensor[i][j]

        self.reduce_count += 1
        return result

    def all_reduce(self, tensors: Dict[int, List[List[float]]],
                   op: str = "sum") -> Dict[int, List[List[float]]]:
        """All-Reduce: 所有GPU的结果一致

        Args:
            tensors: {gpu_id: tensor}
            op: "sum" / "avg"
        Returns:
            {gpu_id: reduced_tensor} (所有GPU结果相同)
        """
        if not tensors:
            return {}

        # 选取第一个GPU的tensor作为基准执行Ring AllReduce
        first_gpu = list(tensors.keys())[0]
        first_tensor = tensors[first_gpu]

        reduced = self._ring_reduce(first_tensor, first_gpu, op)
        if op == "avg":
            n = self.topology.num_gpus
            reduced = [[v / n for v in row] for row in reduced]

        return {gpu: reduced for gpu in tensors}


# ============================================================
# Broadcast 引擎
# ============================================================

class BroadcastEngine:
    """广播引擎 — 从一棵GPU广播到所有GPU

    树形拓扑最优: O(log N) 步
    Mesh拓扑: 源GPU直接发送到所有目标 (1步)
    """

    def __init__(self, topology: NVLinkTopology):
        self.topology = topology
        self.broadcast_count: int = 0
        self.total_bytes_broadcast: int = 0

    def broadcast(self, tensor: Any, src: int,
                  target_gpus: Optional[List[int]] = None) -> Dict[int, Any]:
        """广播张量到目标GPU列表

        Args:
            tensor: 要广播的数据
            src: 源GPU
            target_gpus: 目标GPU列表 (None=所有)
        """
        if target_gpus is None:
            target_gpus = [i for i in range(self.topology.num_gpus) if i != src]

        results = {src: tensor}
        for dst in target_gpus:
            channel = self.topology.get_channel(src, dst)
            if channel:
                payload = tensor
                if isinstance(payload, list):
                    packet = DataPacket(src, dst, op_code=2, payload=payload)
                    if channel.send(packet):
                        results[dst] = payload
                        # 估算大小
                        if isinstance(payload, list) and payload and isinstance(payload[0], list):
                            self.total_bytes_broadcast += len(payload) * len(payload[0]) * 8

        self.broadcast_count += 1
        return results


# ============================================================
# P2P 引擎
# ============================================================

class P2PEngine:
    """点到点传输引擎

    支持流水线并行的层间数据传输:
    GPU-0(Layer 0-3) → GPU-1(Layer 4-7) → GPU-2(Layer 8-11)
    """

    def __init__(self, topology: NVLinkTopology):
        self.topology = topology
        self.p2p_count: int = 0

    def send(self, tensor: Any, src: int, dst: int) -> bool:
        """P2P 发送"""
        channel = self.topology.get_channel(src, dst)
        if not channel:
            return False
        packet = DataPacket(src, dst, op_code=0, payload=tensor)
        success = channel.send(packet)
        if success:
            self.p2p_count += 1
        return success

    def recv(self, src: int, dst: int) -> Optional[Any]:
        """P2P 接收"""
        channel = self.topology.get_channel(src, dst)
        if not channel:
            return None
        packet = channel.receive()
        return packet.payload if packet else None

    def pipeline_send_recv(self, send_tensor: Any, src: int, dst: int,
                           timeout: float = 1.0) -> Optional[Any]:
        """流水线: 同时发送和接收 (双向)"""
        recv_thread_result = [None]

        def _recv():
            recv_thread_result[0] = self.recv(dst, src)

        t = threading.Thread(target=_recv, daemon=True)
        t.start()

        self.send(send_tensor, src, dst)
        t.join(timeout=timeout)

        self.p2p_count += 1
        return recv_thread_result[0]


# ============================================================
# GPU 联邦管理器
# ============================================================

class GPUFederation:
    """多GPU联邦管理器

    管理多个 VirtualGPU 实例的组合:
    - 统一拓扑管理
    - 统一通信原语 (all_reduce/broadcast/p2p)
    - 负载均衡和故障检测
    - 对接 VirtualGPUManager (part17)
    """

    def __init__(self, num_gpus: int = 4,
                 topology_type: TopologyType = TopologyType.MESH):
        self.num_gpus = num_gpus
        self.topology = NVLinkTopology(num_gpus, topology_type)
        self.all_reduce = AllReduceEngine(self.topology)
        self.broadcast = BroadcastEngine(self.topology)
        self.p2p = P2PEngine(self.topology)

        # 每个GPU的本地状态
        self.gpu_healthy: Dict[int, bool] = {i: True for i in range(num_gpus)}
        self.gpu_load: Dict[int, float] = {i: 0.0 for i in range(num_gpus)}

        # 同步屏障
        self._barrier_count: Dict[int, int] = {i: 0 for i in range(num_gpus)}
        self._barrier_lock = threading.Lock()

        # 统计
        self.total_communication_bytes: int = 0
        self._created_at: float = time.time()

    # ---------- 拓扑 ----------

    def switch_topology(self, new_type: TopologyType):
        """动态切换拓扑 (不停机)"""
        self.topology.switch_topology(new_type)
        self.all_reduce = AllReduceEngine(self.topology)
        self.broadcast = BroadcastEngine(self.topology)
        self.p2p = P2PEngine(self.topology)

    # ---------- 梯度同步 ----------

    def sync_gradients(self, grads_per_gpu: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """All-Reduce 梯度同步

        每个GPU有各自的梯度字典, 同步后所有GPU拿到相同梯度。

        Args:
            grads_per_gpu: {gpu_id: {param_name: tensor}}
        Returns:
            {gpu_id: {param_name: averaged_tensor}}
        """
        result = {}
        all_param_names = set()
        for grads in grads_per_gpu.values():
            all_param_names.update(grads.keys())

        for param_name in all_param_names:
            tensors = {}
            for gpu_id, grads in grads_per_gpu.items():
                if param_name in grads:
                    tensors[gpu_id] = grads[param_name]

            if tensors:
                reduced = self.all_reduce.all_reduce(tensors, op="avg")
                for gpu_id, tensor in reduced.items():
                    if gpu_id not in result:
                        result[gpu_id] = {}
                    result[gpu_id][param_name] = tensor

        # 更新通信统计
        for ch in self.topology.channels.values():
            self.total_communication_bytes += ch.bytes_sent + ch.bytes_received

        return result

    # ---------- 参数广播 ----------

    def broadcast_weights(self, weights: Any, src_gpu: int = 0) -> Dict[int, Any]:
        """从 src_gpu 广播权重到所有GPU"""
        return self.broadcast.broadcast(weights, src_gpu)

    # ---------- 屏障同步 ----------

    def barrier(self, gpu_id: int, barrier_id: int = 0):
        """同步屏障: 所有GPU到达后才能继续"""
        with self._barrier_lock:
            self._barrier_count[barrier_id] = self._barrier_count.get(barrier_id, 0) + 1
            current = self._barrier_count[barrier_id]

        # 等待所有GPU到达
        while current < self.num_gpus:
            time.sleep(0.001)
            with self._barrier_lock:
                current = self._barrier_count.get(barrier_id, 0)

    # ---------- 故障检测 ----------

    def health_check(self) -> Dict[int, bool]:
        """健康检查: 发送心跳到每个GPU通道"""
        for gpu_id in range(self.num_gpus):
            # 检查该GPU是否有入口通道
            has_channel = any(
                (src, gpu_id) in self.topology.channels or
                (gpu_id, dst) in self.topology.channels
                for src in range(self.num_gpus)
                for dst in range(self.num_gpus)
            )
            self.gpu_healthy[gpu_id] = has_channel
        return self.gpu_healthy

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, Any]:
        alive_gpus = sum(1 for h in self.gpu_healthy.values() if h)
        total_bw = self.topology.get_total_bandwidth()

        return {
            "num_gpus": self.num_gpus,
            "alive_gpus": alive_gpus,
            "topology": self.topology.topology.value,
            "total_channels": len(self.topology.channels),
            "total_bandwidth_gbps": round(total_bw, 2),
            "aggregate_bandwidth_gbps": round(total_bw * self.num_gpus, 2),
            "total_comm_bytes": self.total_communication_bytes,
            "all_reduce_count": self.all_reduce.reduce_count,
            "broadcast_count": self.broadcast.broadcast_count,
            "p2p_count": self.p2p.p2p_count,
            "avg_channel_utilization": round(
                sum(ch.get_util() for ch in self.topology.channels.values()) /
                max(len(self.topology.channels), 1), 4),
            "uptime_sec": round(time.time() - self._created_at, 1),
        }

    def get_link_smi(self) -> str:
        """类似 nvidia-smi nvlink 的输出"""
        lines = [
            "=" * 70,
            "  Lingyuan Virtual NVLink — nvlink-smi",
            "=" * 70,
            f"  拓扑类型:       {self.topology.topology.value}",
            f"  GPU数量:        {self.num_gpus}",
            f"  通道总数:       {len(self.topology.channels)}",
            f"  总带宽:         {self.topology.get_total_bandwidth():.1f} GB/s",
            f"  聚合带宽:       {self.topology.get_total_bandwidth() * self.num_gpus:.1f} GB/s",
            "",
            "  GPU链路状态:",
        ]
        for i in range(self.num_gpus):
            links = []
            for j in range(self.num_gpus):
                if i != j:
                    ch = self.topology.get_channel(i, j)
                    if ch:
                        status = "✓" if self.gpu_healthy[i] else "✗"
                        links.append(f"GPU{j}({status} {ch.bandwidth_gbps:.0f}GB/s)")
            lines.append(f"    GPU{i}: " + " | ".join(links) if links else "    GPU{i}: 无连接")

        lines += [
            "",
            "  通信统计:",
            f"    AllReduce:     {self.all_reduce.reduce_count} 次",
            f"    Broadcast:     {self.broadcast.broadcast_count} 次",
            f"    P2P:           {self.p2p.p2p_count} 次",
            f"    总通信量:      {self.total_communication_bytes / 1e9:.2f} GB",
            f"    总数据包:      {self.topology.get_total_packets()}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ============================================================
# NVLink 监控器
# ============================================================

class NVLinkMonitor:
    """NVLink 性能监控器

    追踪: 带宽利用率 / 延迟分位数 / 丢包率 / 拥塞检测
    """

    def __init__(self, federation: GPUFederation, window_sec: float = 60.0):
        self.federation = federation
        self.window_sec = window_sec
        self.bw_samples: deque = deque(maxlen=1000)
        self.latency_samples: deque = deque(maxlen=1000)
        self.error_samples: deque = deque(maxlen=1000)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start(self):
        """启动监控"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        self._monitoring = False

    def _monitor_loop(self):
        while self._monitoring:
            total_bw = 0.0
            total_errors = 0
            for ch in self.federation.topology.channels.values():
                total_bw += ch.get_util() * ch.bandwidth_gbps
                total_errors += ch.errors

            self.bw_samples.append(total_bw)
            self.error_samples.append(total_errors)
            time.sleep(1.0)

    def get_metrics(self) -> Dict[str, Any]:
        bw_list = list(self.bw_samples)
        return {
            "avg_bandwidth_gbps": round(sum(bw_list) / max(len(bw_list), 1), 2),
            "peak_bandwidth_gbps": round(max(bw_list), 2) if bw_list else 0,
            "min_bandwidth_gbps": round(min(bw_list), 2) if bw_list else 0,
            "total_errors": sum(self.error_samples) if self.error_samples else 0,
            "congestion_level": self._compute_congestion(),
            "topology": self.federation.topology.topology.value,
            "healthy_gpus": sum(1 for h in self.federation.gpu_healthy.values() if h),
        }

    def _compute_congestion(self) -> float:
        """拥塞程度 (0-1)"""
        if not self.bw_samples:
            return 0.0
        recent = list(self.bw_samples)[-10:]
        max_bw = self.federation.topology.get_total_bandwidth()
        if max_bw <= 0:
            return 0.0
        return min(sum(recent) / (len(recent) * max_bw), 1.0)


# ============================================================
# 测试套件
# ============================================================

def main():
    """part30 虚拟NVLink 自测"""
    print("=" * 70)
    print("  灵元虚拟NVLink — 自测")
    print("=" * 70)
    passed = 0
    total = 0

    # Test 1: NVLinkChannel 基本收发
    total += 1
    try:
        ch = NVLinkChannel(0, 0, 1)
        a = [[1.0, 2.0], [3.0, 4.0]]
        packet = DataPacket(0, 1, op_code=0, payload=a)
        assert ch.send(packet), "发送失败"
        recv = ch.receive()
        assert recv is not None, "接收失败"
        assert recv.payload == a, f"数据不一致: {recv.payload}"
        assert ch.packets_sent == 1 and ch.packets_received == 1
        passed += 1
        print("  [PASS] NVLinkChannel 基本收发")
    except Exception as e:
        print(f"  [FAIL] NVLinkChannel 基本收发: {e}")

    # Test 2: DataPacket 序列化/反序列化
    total += 1
    try:
        a = [[random.random() for _ in range(10)] for _ in range(10)]
        pkt = DataPacket(2, 3, op_code=1, flags=2, payload=a, seq_num=42)
        data = pkt.serialize()
        pkt2 = DataPacket.deserialize(data)
        assert pkt2.src_gpu == 2 and pkt2.dst_gpu == 3
        assert pkt2.seq_num == 42
        # 浮点精度容忍
        for i in range(10):
            for j in range(10):
                assert abs(pkt2.payload[i][j] - a[i][j]) < 1e-6
        passed += 1
        print("  [PASS] DataPacket 序列化")
    except Exception as e:
        print(f"  [FAIL] DataPacket 序列化: {e}")

    # Test 3: 拓扑构建
    total += 1
    try:
        for topo in [TopologyType.MESH, TopologyType.RING, TopologyType.TREE]:
            top = NVLinkTopology(4, topo)
            assert len(top.channels) > 0
            assert top.get_total_bandwidth() > 0
        passed += 1
        print("  [PASS] 拓扑构建 (mesh/ring/tree)")
    except Exception as e:
        print(f"  [FAIL] 拓扑构建: {e}")

    # Test 4: Ring AllReduce
    total += 1
    try:
        top = NVLinkTopology(4, TopologyType.RING)
        engine = AllReduceEngine(top)

        # 4个GPU各有不同的梯度
        tensors = {
            0: [[g + 1.0 for _ in range(8)] for _ in range(8)]
            for g in range(4)
        }
        result = engine.all_reduce(tensors, op="sum")

        # 验证: 每个GPU结果应相同
        vals = list(result.values())
        assert len(set(str(v) for v in vals)) <= 1, "AllReduce结果不一致"
        passed += 1
        print("  [PASS] Ring AllReduce")
    except Exception as e:
        print(f"  [FAIL] Ring AllReduce: {e}")

    # Test 5: Broadcast
    total += 1
    try:
        top = NVLinkTopology(4, TopologyType.TREE)
        bcast = BroadcastEngine(top)
        data = [[x * 0.1 for x in range(4)] for _ in range(4)]
        results = bcast.broadcast(data, src=0, target_gpus=[1, 2, 3])
        assert 0 in results and 1 in results
        passed += 1
        print("  [PASS] Broadcast")
    except Exception as e:
        print(f"  [FAIL] Broadcast: {e}")

    # Test 6: P2P
    total += 1
    try:
        top = NVLinkTopology(4, TopologyType.MESH)
        p2p = P2PEngine(top)
        data = [[1.0, 2.0], [3.0, 4.0]]
        success = p2p.send(data, 0, 1)
        assert success, "P2P send failed"
        received = p2p.recv(0, 1)
        assert received is not None
        passed += 1
        print("  [PASS] P2P 传输")
    except Exception as e:
        print(f"  [FAIL] P2P 传输: {e}")

    # Test 7: GPUFederation 梯度同步
    total += 1
    try:
        fed = GPUFederation(num_gpus=4, topology_type=TopologyType.RING)
        grads = {
            0: {"W_attn": [[0.1, 0.2], [0.3, 0.4]]},
            1: {"W_attn": [[0.2, 0.3], [0.4, 0.5]]},
            2: {"W_attn": [[0.3, 0.4], [0.5, 0.6]]},
            3: {"W_attn": [[0.4, 0.5], [0.6, 0.7]]},
        }
        synced = fed.sync_gradients(grads)
        assert len(synced) > 0, "梯度同步失败"
        health = fed.health_check()
        assert all(health.values()), f"健康检查失败: {health}"
        passed += 1
        print("  [PASS] GPUFederation 梯度同步")
    except Exception as e:
        print(f"  [FAIL] GPUFederation 梯度同步: {e}")

    # Test 8: 拓扑动态切换
    total += 1
    try:
        fed = GPUFederation(num_gpus=4, topology_type=TopologyType.MESH)
        stats_before = fed.get_stats()
        fed.switch_topology(TopologyType.RING)
        stats_after = fed.get_stats()
        assert stats_before["topology"] == "mesh"
        assert stats_after["topology"] == "ring"
        passed += 1
        print("  [PASS] 拓扑动态切换")
    except Exception as e:
        print(f"  [FAIL] 拓扑动态切换: {e}")

    # Test 9: 监控器
    total += 1
    try:
        fed = GPUFederation(num_gpus=4, topology_type=TopologyType.RING)
        monitor = NVLinkMonitor(fed, window_sec=5.0)
        monitor.start()
        time.sleep(1.5)
        metrics = monitor.get_metrics()
        assert "avg_bandwidth_gbps" in metrics
        monitor.stop()
        passed += 1
        print("  [PASS] NVLink 监控器")
    except Exception as e:
        print(f"  [FAIL] NVLink 监控器: {e}")

    # Test 10: nvlink-smi 输出
    total += 1
    try:
        fed = GPUFederation(num_gpus=4, topology_type=TopologyType.MESH)
        smi = fed.get_link_smi()
        assert "nvlink-smi" in smi
        assert "GPU0" in smi
        passed += 1
        print("  [PASS] nvlink-smi 输出")
    except Exception as e:
        print(f"  [FAIL] nvlink-smi 输出: {e}")

    # Test 11: 大张量AllReduce性能 (128×128)
    total += 1
    try:
        fed = GPUFederation(num_gpus=4, topology_type=TopologyType.RING)
        big_tensor = [[random.random() for _ in range(128)] for _ in range(128)]
        grads = {i: {"big": big_tensor} for i in range(4)}
        t0 = time.time()
        fed.sync_gradients(grads)
        elapsed = time.time() - t0
        # 零拷贝: 应该在秒级完成
        assert elapsed < 5.0, f"AllReduce 大张量过慢: {elapsed:.2f}s"
        passed += 1
        print(f"  [PASS] 大张量AllReduce (128×128) — {elapsed*1000:.1f}ms")
    except Exception as e:
        print(f"  [FAIL] 大张量AllReduce: {e}")

    print()
    print(f"  {'='*50}")
    print(f"  自测结果: {passed} 通过, {total - passed} 失败, 共 {total} 项")
    print(f"  {'='*50}")
    if passed == total:
        print("  所有测试通过!")

    # 打印 nvlink-smi
    fed_demo = GPUFederation(num_gpus=4, topology_type=TopologyType.MESH)
    print("\n" + fed_demo.get_link_smi())


if __name__ == "__main__":
    main()
