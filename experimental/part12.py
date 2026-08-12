"""
part12.py - 灵元大模型: 模型格式与互操作模块
对应52项清单 #25-29:
  #25 WeightSerializer      - 权重序列化(多格式支持)
  #26 HuggingFaceExporter   - HuggingFace格式导出
  #27 ONNXExporter          - ONNX导出(简化JSON表示)
  #28 GGUFExporter          - GGUF量化导出
  #29 ExternalModelImporter - 外部模型导入

纯Python标准库实现, 零外部依赖。
此文件在 lingyuan_full.py 之后加载, 可使用全局变量: DATA_DIR, LOG_DIR, CONFIG_DIR
"""

import uuid
import math
import random
import json
import os
import time
import struct
import hashlib
import shutil
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime

# ============================================================================
# 全局配置 (从 lingyuan_full.py 继承)
# ============================================================================

_DATA_DIR = globals().get('DATA_DIR', os.path.join(os.path.expanduser('~'), '.lingyuan', 'data'))
_LOG_DIR = globals().get('LOG_DIR', os.path.join(os.path.expanduser('~'), '.lingyuan', 'logs'))
_CONFIG_DIR = globals().get('CONFIG_DIR', os.path.join(os.path.expanduser('~'), '.lingyuan', 'config'))

# 模型缓存与导出目录
MODEL_CACHE_DIR = os.path.join(str(_DATA_DIR), 'model_cache')
EXPORT_DIR = os.path.join(str(_DATA_DIR), 'exports')


def _ensure_dir(path: str) -> None:
    """确保目录存在, 递归创建"""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _log(msg: str, level: str = "INFO") -> None:
    """日志记录到文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] [part12] {msg}"
    try:
        _ensure_dir(str(_LOG_DIR))
        log_path = os.path.join(str(_LOG_DIR), 'part12.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')
    except Exception:
        pass


# ============================================================================
# 辅助函数
# ============================================================================

def _get_model_weights(model) -> Dict[str, Any]:
    """从模型对象提取权重字典

    支持以下输入:
    1. 直接传入 Dict[str, List[float]]
    2. model.state_dict() 方法
    3. model.get_weights() 方法
    4. model.weights 属性
    """
    if isinstance(model, dict):
        # 如果字典中有 'weights' 或 'state_dict' 键, 提取权重
        if 'weights' in model and isinstance(model['weights'], dict):
            return model['weights']
        if 'state_dict' in model and isinstance(model['state_dict'], dict):
            return model['state_dict']
        return model
    if hasattr(model, 'state_dict'):
        try:
            return model.state_dict()
        except Exception:
            pass
    if hasattr(model, 'get_weights'):
        try:
            return model.get_weights()
        except Exception:
            pass
    if hasattr(model, 'weights'):
        w = model.weights
        if isinstance(w, dict):
            return w
    raise ValueError("无法从模型对象提取权重, 请提供 state_dict() 或 get_weights() 方法")


def _get_model_config(model) -> Dict[str, Any]:
    """从模型对象提取配置"""
    cfg = None
    if isinstance(model, dict) and 'config' in model:
        cfg = model['config']
    elif hasattr(model, 'config'):
        cfg = model.config
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, '__dataclass_fields__'):
        try:
            return asdict(cfg)
        except Exception:
            pass
    if hasattr(cfg, '__dict__'):
        return dict(vars(cfg))
    return {}


def _get_model_tokenizer(model):
    """从模型对象提取分词器"""
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        return model.tokenizer
    if isinstance(model, dict) and 'tokenizer' in model:
        return model['tokenizer']
    return None


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class WeightTensor:
    """权重张量 - 表示单个权重张量"""
    name: str
    data: List[float]
    shape: List[int] = field(default_factory=list)
    dtype: str = "float32"

    def __post_init__(self):
        if not self.shape:
            self.shape = [len(self.data)]

    @property
    def num_elements(self) -> int:
        """元素总数"""
        count = 1
        for d in self.shape:
            count *= d
        return count

    @property
    def nbytes(self) -> int:
        """字节数 (float32)"""
        return self.num_elements * 4

    def checksum(self) -> str:
        """计算SHA256校验和"""
        try:
            packed = struct.pack(f'<{len(self.data)}f', *self.data)
        except struct.error:
            packed = b''
        return hashlib.sha256(packed).hexdigest()

    def metadata(self) -> Dict[str, Any]:
        """返回元数据"""
        return {
            'name': self.name,
            'shape': list(self.shape),
            'dtype': self.dtype,
            'num_elements': self.num_elements,
            'nbytes': self.nbytes,
            'checksum': self.checksum(),
        }


@dataclass
class ShardInfo:
    """分片信息"""
    shard_index: int
    shard_name: str
    shard_path: str
    tensor_names: List[str] = field(default_factory=list)
    shard_size: int = 0


@dataclass
class SerializationStats:
    """序列化统计"""
    total_tensors_saved: int = 0
    total_bytes_saved: int = 0
    total_tensors_loaded: int = 0
    total_bytes_loaded: int = 0
    save_operations: int = 0
    load_operations: int = 0
    shard_count: int = 0
    compressed: bool = False
    compressed_bytes: int = 0
    last_save_format: str = ""
    last_load_format: str = ""
    checksums_verified: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class ExportStats:
    """导出统计"""
    total_exports: int = 0
    total_files_generated: int = 0
    total_bytes_exported: int = 0
    last_export_dir: str = ""
    last_export_time: float = 0.0
    verification_passed: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class ONNXNode:
    """ONNX计算图节点"""
    name: str
    op_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ONNXTensor:
    """ONNX张量描述"""
    name: str
    data_type: int  # ONNX数据类型枚举
    dims: List[int] = field(default_factory=list)
    raw_data_size: int = 0


@dataclass
class ONNXStats:
    """ONNX导出统计"""
    total_exports: int = 0
    total_nodes: int = 0
    total_initializers: int = 0
    total_parameters: int = 0
    last_export_path: str = ""
    last_export_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    supported_ops: List[str] = field(default_factory=list)


@dataclass
class GGUFStats:
    """GGUF导出统计"""
    total_exports: int = 0
    total_tensors: int = 0
    total_bytes: int = 0
    original_bytes: int = 0
    quantization_type: str = ""
    compression_ratio: float = 0.0
    last_export_path: str = ""
    last_export_time: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class ImportStats:
    """导入统计"""
    total_imports: int = 0
    total_downloads: int = 0
    total_bytes_downloaded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    supported_architectures: List[str] = field(default_factory=list)
    last_imported_model: str = ""
    errors: List[str] = field(default_factory=list)


def _infer_shape(data) -> List[int]:
    """递归推断嵌套列表的shape"""
    shape = []
    cur = data
    while isinstance(cur, (list, tuple)) and len(cur) > 0:
        shape.append(len(cur))
        cur = cur[0]
    return shape


def _flatten_nested(data) -> List[float]:
    """递归展平嵌套列表为一维float列表"""
    result = []
    if isinstance(data, (list, tuple)):
        for item in data:
            result.extend(_flatten_nested(item))
    else:
        result.append(float(data))
    return result


def _normalize_weights(weights: Dict[str, Any]) -> List[WeightTensor]:
    """将权重字典规范化为 WeightTensor 列表

    支持以下输入格式:
    - Dict[str, List[float]]: name -> flat data
    - Dict[str, Dict]: name -> {'data': [...], 'shape': [...], 'dtype': '...'}
    - Dict[str, Tuple]: name -> (data, shape)
    - Dict[str, WeightTensor]: name -> WeightTensor
    """
    tensors = []
    for name, value in weights.items():
        if isinstance(value, WeightTensor):
            tensors.append(value)
        elif isinstance(value, dict):
            data = value.get('data', [])
            shape = value.get('shape', [len(data)] if data else [0])
            dtype = value.get('dtype', 'float32')
            tensors.append(WeightTensor(
                name=name, data=list(data),
                shape=list(shape), dtype=dtype
            ))
        elif isinstance(value, (list, tuple)) and len(value) == 2 \
                and isinstance(value[0], (list, tuple)) \
                and isinstance(value[1], (list, tuple)) \
                and all(isinstance(s, int) for s in value[1]):
            # (data, shape) 格式 — shape必须全为整数
            data, shape = value
            tensors.append(WeightTensor(
                name=name, data=list(data), shape=list(shape)
            ))
        elif isinstance(value, (list, tuple)):
            # 检查是否为多维数组 (嵌套列表)
            if value and isinstance(value[0], (list, tuple)):
                # 多维数组: 展平并计算shape
                shape = _infer_shape(value)
                flat_data = _flatten_nested(value)
                tensors.append(WeightTensor(
                    name=name, data=flat_data, shape=shape
                ))
            else:
                # 纯数据, 推断shape为1D
                tensors.append(WeightTensor(
                    name=name, data=list(value), shape=[len(value)]
                ))
        else:
            try:
                data = list(value)
                tensors.append(WeightTensor(
                    name=name, data=data, shape=[len(data)]
                ))
            except Exception:
                _log(f"无法处理权重 '{name}', 类型: {type(value)}", "WARN")
    return tensors


# ============================================================================
# #25 WeightSerializer - 权重序列化
# ============================================================================

class WeightSerializer:
    """权重序列化器 - 支持多种格式的权重保存与加载

    支持格式:
    - safetensors: JSON header + raw float32 binary (类似HuggingFace safetensors)
    - pytorch_bin: pickle模拟(JSON header + binary)
    - numpy_npy: 简化的.npy格式(header + binary)
    - json: 纯JSON(调试用,体积大)

    特性:
    - 权重元数据: {name, shape, dtype, offset, size}
    - 校验和: 每个权重tensor的SHA256
    - 可选zlib压缩
    - 分片保存: 大模型自动分片
    - 加载验证: 校验形状/校验和
    """

    # 支持的格式列表
    SUPPORTED_FORMATS = ['safetensors', 'pytorch_bin', 'numpy_npy', 'json']

    # safetensors dtype 映射
    _DTYPE_MAP = {
        'float32': 'F32',
        'float16': 'F16',
        'int8': 'I8',
        'int32': 'I32',
        'int64': 'I64',
        'bool': 'BOOL',
    }
    _DTYPE_REVERSE = {v: k for k, v in _DTYPE_MAP.items()}

    # dtype 对应的字节数
    _DTYPE_BYTES = {
        'F32': 4, 'F16': 2, 'I8': 1, 'I32': 4, 'I64': 8, 'BOOL': 1,
        'float32': 4, 'float16': 2, 'int8': 1, 'int32': 4, 'int64': 8,
    }

    # 默认分片大小 (2GB, 可配置)
    DEFAULT_SHARD_SIZE = 2 * 1024 * 1024 * 1024

    def __init__(self, compress: bool = False, shard_size: Optional[int] = None):
        """初始化权重序列化器

        Args:
            compress: 是否启用zlib压缩
            shard_size: 分片大小(字节), None表示不分片
        """
        self.compress = compress
        self.shard_size = shard_size
        self.stats = SerializationStats(compressed=compress)

    def save_weights(self, model, path: str, format: str = "safetensors") -> str:
        """保存权重到文件

        Args:
            model: 模型对象或权重字典
            path: 输出文件路径
            format: 保存格式 (safetensors/pytorch_bin/numpy_npy/json)

        Returns:
            实际保存的文件路径(可能因分片而不同)
        """
        start_time = time.time()
        self.stats.save_operations += 1
        self.stats.last_save_format = format

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的格式: {format}, 支持: {self.SUPPORTED_FORMATS}")

        # 提取并规范化权重
        raw_weights = _get_model_weights(model)
        tensors = _normalize_weights(raw_weights)

        if not tensors:
            self.stats.errors.append(f"保存时未找到任何权重: {path}")
            _log(f"警告: 未找到任何权重张量", "WARN")

        # 计算总大小
        total_size = sum(t.nbytes for t in tensors)
        self.stats.total_tensors_saved += len(tensors)
        self.stats.total_bytes_saved += total_size

        # 判断是否需要分片
        use_sharding = (
            self.shard_size is not None
            and total_size > self.shard_size
            and format in ('safetensors', 'pytorch_bin')
        )

        if use_sharding:
            saved_path = self._save_sharded(tensors, path, format)
        else:
            saved_path = self._save_single(tensors, path, format)

        elapsed = time.time() - start_time
        self.stats.save_time = elapsed
        _log(f"权重保存完成: {path}, 格式={format}, 张量数={len(tensors)}, "
             f"大小={total_size} bytes, 耗时={elapsed:.3f}s")

        return saved_path

    def load_weights(self, path: str) -> Dict[str, List[float]]:
        """从文件加载权重

        Args:
            path: 权重文件路径

        Returns:
            Dict[str, List[float]]: 权重名到数据的映射
        """
        start_time = time.time()
        self.stats.load_operations += 1

        # 检查是否是分片文件
        index_path = path.replace('.safetensors', '.index.json') \
            if path.endswith('.safetensors') else \
            path.replace('.bin', '.index.json') if path.endswith('.bin') else None

        if index_path and os.path.exists(index_path):
            weights = self._load_sharded(path)
        else:
            # 自动检测格式
            fmt = self._detect_format(path)
            self.stats.last_load_format = fmt

            if fmt == 'safetensors':
                weights = self._load_safetensors(path)
            elif fmt == 'pytorch_bin':
                weights = self._load_pytorch_bin(path)
            elif fmt == 'numpy_npy':
                weights = self._load_numpy_npy(path)
            elif fmt == 'json':
                weights = self._load_json(path)
            else:
                raise ValueError(f"无法检测文件格式: {path}")

        total_size = sum(len(v) * 4 for v in weights.values())
        self.stats.total_tensors_loaded += len(weights)
        self.stats.total_bytes_loaded += total_size

        elapsed = time.time() - start_time
        self.stats.load_time = elapsed
        _log(f"权重加载完成: {path}, 张量数={len(weights)}, 耗时={elapsed:.3f}s")

        return weights

    def _detect_format(self, path: str) -> str:
        """根据文件扩展名检测格式"""
        lower = path.lower()
        if lower.endswith('.safetensors'):
            return 'safetensors'
        if lower.endswith('.bin') or lower.endswith('.pt'):
            return 'pytorch_bin'
        if lower.endswith('.npy'):
            return 'numpy_npy'
        if lower.endswith('.json'):
            return 'json'
        # 尝试读取magic
        try:
            with open(path, 'rb') as f:
                magic = f.read(8)
            if magic.startswith(b'\x93NUMPY'):
                return 'numpy_npy'
            if magic.startswith(b'{'):
                return 'json'
            # safetensors: 前8字节是header长度
            header_len = struct.unpack('<Q', magic[:8])[0]
            if 0 < header_len < 10 * 1024 * 1024:
                header = json.loads(f.read(header_len).decode('utf-8'))  # noqa
                if isinstance(header, dict):
                    return 'safetensors'
        except Exception:
            pass
        return 'safetensors'  # 默认

    # ------------------------------------------------------------------
    # safetensors 格式
    # ------------------------------------------------------------------

    def _save_safetensors(self, tensors: List[WeightTensor], path: str,
                          shard_info: Optional[Dict] = None) -> str:
        """保存为 safetensors 格式

        格式: [8字节header_length] [JSON header] [binary tensor data]
        """
        header = OrderedDict()
        binary_data = bytearray()
        offset = 0

        for tensor in tensors:
            dtype_str = self._DTYPE_MAP.get(tensor.dtype, 'F32')
            byte_size = tensor.num_elements * self._DTYPE_BYTES.get(dtype_str, 4)

            # 写入元数据
            header[tensor.name] = {
                'dtype': dtype_str,
                'shape': list(tensor.shape),
                'data_offsets': [offset, offset + byte_size],
            }

            # 写入二进制数据
            packed = struct.pack(f'<{len(tensor.data)}f', *tensor.data)
            binary_data.extend(packed)
            offset += byte_size

        # 元数据区
        meta = {
            'format': 'pt',
            'total_size': len(binary_data),
        }
        if shard_info:
            meta['shard_info'] = shard_info
        header['__metadata__'] = meta

        # 添加校验和到元数据
        checksums = {}
        for tensor in tensors:
            checksums[tensor.name] = tensor.checksum()
        header['__checksums__'] = checksums

        header_json = json.dumps(header, ensure_ascii=False)
        header_bytes = header_json.encode('utf-8')

        # 可选压缩
        if self.compress:
            compressed_data = zlib.compress(bytes(binary_data))
            header['__metadata__']['compressed'] = True
            header['__metadata__']['compressed_size'] = len(compressed_data)
            header_json = json.dumps(header, ensure_ascii=False)
            header_bytes = header_json.encode('utf-8')
            data_to_write = compressed_data
            self.stats.compressed_bytes += len(compressed_data)
        else:
            data_to_write = bytes(binary_data)

        with open(path, 'wb') as f:
            f.write(struct.pack('<Q', len(header_bytes)))
            f.write(header_bytes)
            f.write(data_to_write)

        return path

    def _load_safetensors(self, path: str) -> Dict[str, List[float]]:
        """加载 safetensors 格式"""
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len).decode('utf-8'))

            is_compressed = header.get('__metadata__', {}).get('compressed', False)
            checksums = header.get('__checksums__', {})

            if is_compressed:
                compressed = f.read()
                binary_data = zlib.decompress(compressed)
            else:
                binary_data = f.read()

        weights = {}
        for name, info in header.items():
            if name.startswith('__'):
                continue
            dtype = self._DTYPE_REVERSE.get(info['dtype'], 'float32')
            shape = info['shape']
            offsets = info['data_offsets']
            byte_size = offsets[1] - offsets[0]
            elem_count = byte_size // 4  # 默认 float32

            raw = binary_data[offsets[0]:offsets[1]]
            data = list(struct.unpack(f'<{elem_count}f', raw))
            weights[name] = data

            # 校验和验证
            if name in checksums:
                tensor = WeightTensor(name=name, data=data, shape=shape, dtype=dtype)
                if tensor.checksum() == checksums[name]:
                    self.stats.checksums_verified += 1

        return weights

    # ------------------------------------------------------------------
    # pytorch_bin 格式 (pickle模拟)
    # ------------------------------------------------------------------

    def _save_pytorch_bin(self, tensors: List[WeightTensor], path: str,
                          shard_info: Optional[Dict] = None) -> str:
        """保存为 pytorch_bin 格式 (用JSON+binary模拟pickle)

        格式: [8字节 magic] [8字节 header_length] [JSON header] [binary data]
        """
        MAGIC = b'LYPT0001'  # LingYuan PyTorch模拟 (8字节对齐)

        header = {
            'format': 'pytorch_bin',
            'version': '1.0',
            'tensors': {},
            'total_size': 0,
        }
        if shard_info:
            header['shard_info'] = shard_info

        binary_data = bytearray()
        offset = 0

        for tensor in tensors:
            byte_size = tensor.num_elements * 4
            header['tensors'][tensor.name] = {
                'dtype': tensor.dtype,
                'shape': list(tensor.shape),
                'offset': offset,
                'size': byte_size,
                'checksum': tensor.checksum(),
            }
            packed = struct.pack(f'<{len(tensor.data)}f', *tensor.data)
            binary_data.extend(packed)
            offset += byte_size

        header['total_size'] = len(binary_data)

        header_json = json.dumps(header, ensure_ascii=False)
        header_bytes = header_json.encode('utf-8')

        if self.compress:
            binary_data = zlib.compress(bytes(binary_data))
            header['compressed'] = True
            header['compressed_size'] = len(binary_data)
            header_json = json.dumps(header, ensure_ascii=False)
            header_bytes = header_json.encode('utf-8')

        with open(path, 'wb') as f:
            f.write(MAGIC)
            f.write(struct.pack('<Q', len(header_bytes)))
            f.write(header_bytes)
            f.write(binary_data)

        return path

    def _load_pytorch_bin(self, path: str) -> Dict[str, List[float]]:
        """加载 pytorch_bin 格式"""
        with open(path, 'rb') as f:
            magic = f.read(8)
            if magic != b'LYPT0001':
                _log(f"警告: magic不匹配: {magic}", "WARN")
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len).decode('utf-8'))

            is_compressed = header.get('compressed', False)
            if is_compressed:
                binary_data = zlib.decompress(f.read())
            else:
                binary_data = f.read()

        weights = {}
        for name, info in header['tensors'].items():
            offset = info['offset']
            byte_size = info['size']
            elem_count = byte_size // 4
            raw = binary_data[offset:offset + byte_size]
            data = list(struct.unpack(f'<{elem_count}f', raw))
            weights[name] = data

            # 校验
            tensor = WeightTensor(
                name=name, data=data,
                shape=info['shape'], dtype=info.get('dtype', 'float32')
            )
            if 'checksum' in info and tensor.checksum() == info['checksum']:
                self.stats.checksums_verified += 1

        return weights

    # ------------------------------------------------------------------
    # numpy_npy 格式
    # ------------------------------------------------------------------

    def _save_numpy_npy(self, tensors: List[WeightTensor], path: str,
                        shard_info: Optional[Dict] = None) -> str:
        """保存为简化的 .npy 格式

        格式: [magic \x93NUMPY] [version 2B] [header_len 2B] [dict header] [binary]
        注意: .npy 只能保存单个数组, 多个tensor时保存为 .npz (多个.npy打包)
        """
        if len(tensors) == 1:
            # 单个tensor: 标准.npy
            tensor = tensors[0]
            magic = b'\x93NUMPY'
            version = struct.pack('<BB', 1, 0)

            # npy header dict
            npy_header = {
                'descr': '<f4',
                'fortran_order': False,
                'shape': tuple(tensor.shape),
            }
            header_str = repr(npy_header)
            # 补齐到 64 字节对齐
            while (len(header_str) + 10) % 64 != 0:
                header_str += ' '
            header_str += '\n'
            header_bytes = header_str.encode('latin-1')

            packed = struct.pack(f'<{len(tensor.data)}f', *tensor.data)

            if self.compress:
                packed = zlib.compress(packed)

            with open(path, 'wb') as f:
                f.write(magic)
                f.write(version)
                f.write(struct.pack('<H', len(header_bytes)))
                f.write(header_bytes)
                f.write(packed)
        else:
            # 多个tensor: 保存为 .npz (JSON manifest + 多个数据块)
            manifest = {
                'format': 'numpy_npz',
                'version': '1.0',
                'tensors': {},
                'total_size': 0,
                'compressed': self.compress,
            }
            binary_data = bytearray()
            offset = 0

            for tensor in tensors:
                byte_size = tensor.num_elements * 4
                manifest['tensors'][tensor.name] = {
                    'dtype': tensor.dtype,
                    'shape': list(tensor.shape),
                    'offset': offset,
                    'size': byte_size,
                    'checksum': tensor.checksum(),
                }
                packed = struct.pack(f'<{len(tensor.data)}f', *tensor.data)
                binary_data.extend(packed)
                offset += byte_size

            manifest['total_size'] = len(binary_data)

            if self.compress:
                binary_data = zlib.compress(bytes(binary_data))

            manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode('utf-8')

            with open(path, 'wb') as f:
                f.write(b'\x93NUMPZ')  # 自定义magic
                f.write(struct.pack('<Q', len(manifest_bytes)))
                f.write(manifest_bytes)
                f.write(binary_data)

        return path

    def _load_numpy_npy(self, path: str) -> Dict[str, List[float]]:
        """加载 numpy_npy 格式"""
        with open(path, 'rb') as f:
            magic = f.read(6)

        if magic == b'\x93NUMPZ':
            # 多tensor npz格式
            with open(path, 'rb') as f:
                f.read(6)  # skip magic
                manifest_len = struct.unpack('<Q', f.read(8))[0]
                manifest = json.loads(f.read(manifest_len).decode('utf-8'))
                is_compressed = manifest.get('compressed', False)
                binary_data = f.read()
                if is_compressed:
                    binary_data = zlib.decompress(binary_data)

            weights = {}
            for name, info in manifest['tensors'].items():
                offset = info['offset']
                byte_size = info['size']
                elem_count = byte_size // 4
                raw = binary_data[offset:offset + byte_size]
                data = list(struct.unpack(f'<{elem_count}f', raw))
                weights[name] = data
            return weights

        elif magic[:5] == b'\x93NUM':
            # 标准npy单tensor格式
            with open(path, 'rb') as f:
                f.read(6)  # skip magic
                major, minor = struct.unpack('<BB', f.read(2))
                if major == 1:
                    header_len = struct.unpack('<H', f.read(2))[0]
                else:
                    header_len = struct.unpack('<I', f.read(4))[0]
                header_str = f.read(header_len).decode('latin-1')
                # 解析dict (简化版)
                header_str = header_str.strip()
                # 提取shape
                shape_start = header_str.find("'shape':")
                shape = [1]
                if shape_start >= 0:
                    shape_end = header_str.find(')', shape_start)
                    shape_str = header_str[shape_start:shape_end + 1]
                    # 简单解析元组
                    import re
                    nums = re.findall(r'\d+', shape_str)
                    shape = [int(n) for n in nums] if nums else [1]

                binary_data = f.read()
                if self.compress:
                    binary_data = zlib.decompress(binary_data)

                elem_count = len(binary_data) // 4
                data = list(struct.unpack(f'<{elem_count}f', binary_data))

            return {'array': data}
        else:
            raise ValueError(f"未知的numpy格式: {magic}")

    # ------------------------------------------------------------------
    # json 格式
    # ------------------------------------------------------------------

    def _save_json(self, tensors: List[WeightTensor], path: str,
                   shard_info: Optional[Dict] = None) -> str:
        """保存为纯JSON格式 (调试用, 体积大)"""
        output = {
            'format': 'json',
            'version': '1.0',
            'tensors': {},
            'metadata': {
                'total_tensors': len(tensors),
                'total_bytes': sum(t.nbytes for t in tensors),
            },
        }
        if shard_info:
            output['shard_info'] = shard_info

        for tensor in tensors:
            output['tensors'][tensor.name] = {
                'dtype': tensor.dtype,
                'shape': list(tensor.shape),
                'data': list(tensor.data),
                'checksum': tensor.checksum(),
            }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return path

    def _load_json(self, path: str) -> Dict[str, List[float]]:
        """加载JSON格式"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        weights = {}
        for name, info in data.get('tensors', {}).items():
            weights[name] = list(info['data'])

            # 校验
            tensor = WeightTensor(
                name=name, data=info['data'],
                shape=info['shape'], dtype=info.get('dtype', 'float32')
            )
            if 'checksum' in info and tensor.checksum() == info['checksum']:
                self.stats.checksums_verified += 1

        return weights

    # ------------------------------------------------------------------
    # 分片保存与加载
    # ------------------------------------------------------------------

    def _save_single(self, tensors: List[WeightTensor], path: str,
                     format: str) -> str:
        """单文件保存"""
        _ensure_dir(os.path.dirname(path) if os.path.dirname(path) else '.')
        if format == 'safetensors':
            return self._save_safetensors(tensors, path)
        elif format == 'pytorch_bin':
            return self._save_pytorch_bin(tensors, path)
        elif format == 'numpy_npy':
            return self._save_numpy_npy(tensors, path)
        elif format == 'json':
            return self._save_json(tensors, path)
        return path

    def _save_sharded(self, tensors: List[WeightTensor], path: str,
                      format: str) -> str:
        """分片保存大模型权重

        生成:
        - shard_00001.{ext}, shard_00002.{ext}, ...
        - {base_name}.index.json (权重到分片的映射)
        """
        ext_map = {
            'safetensors': '.safetensors',
            'pytorch_bin': '.bin',
            'numpy_npy': '.npy',
            'json': '.json',
        }
        ext = ext_map.get(format, '.bin')
        base_dir = os.path.dirname(path)
        base_name = os.path.basename(path)
        if '.' in base_name:
            base_name = base_name.rsplit('.', 1)[0]

        _ensure_dir(base_dir if base_dir else '.')

        shards: List[ShardInfo] = []
        current_shard_tensors: List[WeightTensor] = []
        current_size = 0
        shard_index = 0
        weight_map: Dict[str, str] = {}
        total_size = 0

        for tensor in tensors:
            tensor_size = tensor.nbytes
            total_size += tensor_size

            # 检查是否需要新分片
            if current_size + tensor_size > self.shard_size and current_shard_tensors:
                shard_index += 1
                shard_name = f"{base_name}-shard_{shard_index:05d}{ext}"
                shard_path = os.path.join(base_dir, shard_name) if base_dir else shard_name

                for t in current_shard_tensors:
                    weight_map[t.name] = shard_name

                shard_info = ShardInfo(
                    shard_index=shard_index,
                    shard_name=shard_name,
                    shard_path=shard_path,
                    tensor_names=[t.name for t in current_shard_tensors],
                    shard_size=current_size,
                )
                shards.append(shard_info)

                self._save_single(current_shard_tensors, shard_path, format)

                current_shard_tensors = []
                current_size = 0

            current_shard_tensors.append(tensor)
            current_size += tensor_size

        # 保存最后一个分片
        if current_shard_tensors:
            shard_index += 1
            shard_name = f"{base_name}-shard_{shard_index:05d}{ext}"
            shard_path = os.path.join(base_dir, shard_name) if base_dir else shard_name

            for t in current_shard_tensors:
                weight_map[t.name] = shard_name

            shard_info = ShardInfo(
                shard_index=shard_index,
                shard_name=shard_name,
                shard_path=shard_path,
                tensor_names=[t.name for t in current_shard_tensors],
                shard_size=current_size,
            )
            shards.append(shard_info)

            self._save_single(current_shard_tensors, shard_path, format)

        # 写入索引文件
        index_data = {
            'metadata': {
                'total_size': total_size,
                'format': format,
                'shard_count': len(shards),
                'compressed': self.compress,
            },
            'weight_map': weight_map,
            'shards': [
                {
                    'name': s.shard_name,
                    'path': s.shard_path,
                    'tensors': s.tensor_names,
                    'size': s.shard_size,
                }
                for s in shards
            ],
        }
        index_path = os.path.join(base_dir, f"{base_name}.index.json") if base_dir \
            else f"{base_name}.index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

        self.stats.shard_count = len(shards)
        _log(f"分片保存完成: {len(shards)}个分片, 索引: {index_path}")

        return index_path

    def _load_sharded(self, path: str) -> Dict[str, List[float]]:
        """加载分片权重

        Args:
            path: 索引文件路径或任一分片路径
        """
        # 找到索引文件
        if path.endswith('.index.json'):
            index_path = path
        else:
            base_dir = os.path.dirname(path)
            base_name = os.path.basename(path)
            if '.' in base_name:
                base_name = base_name.rsplit('.', 1)[0]
            # 去掉-shard_xxxxx
            if '-shard_' in base_name:
                base_name = base_name.split('-shard_')[0]
            index_path = os.path.join(base_dir, f"{base_name}.index.json") if base_dir \
                else f"{base_name}.index.json"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        weight_map = index_data['weight_map']
        fmt = index_data['metadata'].get('format', 'safetensors')
        base_dir = os.path.dirname(index_path)

        # 按分片分组加载
        shard_to_tensors: Dict[str, List[str]] = {}
        for tensor_name, shard_name in weight_map.items():
            if shard_name not in shard_to_tensors:
                shard_to_tensors[shard_name] = []
            shard_to_tensors[shard_name].append(tensor_name)

        all_weights: Dict[str, List[float]] = {}
        ext_map = {
            'safetensors': '.safetensors',
            'pytorch_bin': '.bin',
            'numpy_npy': '.npy',
            'json': '.json',
        }
        ext = ext_map.get(fmt, '.bin')

        for shard_name, tensor_names in shard_to_tensors.items():
            shard_path = os.path.join(base_dir, shard_name)
            if not os.path.exists(shard_path):
                _log(f"警告: 分片文件不存在: {shard_path}", "WARN")
                continue

            shard_weights = self.load_weights(shard_path)
            for name in tensor_names:
                if name in shard_weights:
                    all_weights[name] = shard_weights[name]

        return all_weights

    # ------------------------------------------------------------------
    # 验证与统计
    # ------------------------------------------------------------------

    def verify_weights(self, path: str, expected_metadata: Optional[Dict] = None) -> bool:
        """验证权重文件完整性

        Args:
            path: 权重文件路径
            expected_metadata: 期望的元数据(可选)

        Returns:
            bool: 验证是否通过
        """
        try:
            weights = self.load_weights(path)
            if not weights:
                self.stats.errors.append(f"验证失败: 无权重数据 {path}")
                return False

            # 验证形状
            if expected_metadata:
                for name, meta in expected_metadata.items():
                    if name not in weights:
                        self.stats.errors.append(f"验证失败: 缺少权重 {name}")
                        return False
                    expected_shape = meta.get('shape', [])
                    expected_size = 1
                    for d in expected_shape:
                        expected_size *= d
                    if len(weights[name]) != expected_size:
                        self.stats.errors.append(
                            f"验证失败: 形状不匹配 {name}, "
                            f"期望={expected_size}, 实际={len(weights[name])}"
                        )
                        return False

            _log(f"权重验证通过: {path}, {len(weights)}个张量")
            return True
        except Exception as e:
            self.stats.errors.append(f"验证异常: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取序列化统计"""
        return {
            'total_tensors_saved': self.stats.total_tensors_saved,
            'total_bytes_saved': self.stats.total_bytes_saved,
            'total_tensors_loaded': self.stats.total_tensors_loaded,
            'total_bytes_loaded': self.stats.total_bytes_loaded,
            'save_operations': self.stats.save_operations,
            'load_operations': self.stats.load_operations,
            'shard_count': self.stats.shard_count,
            'compressed': self.stats.compressed,
            'compressed_bytes': self.stats.compressed_bytes,
            'last_save_format': self.stats.last_save_format,
            'last_load_format': self.stats.last_load_format,
            'checksums_verified': self.stats.checksums_verified,
            'error_count': len(self.stats.errors),
            'supported_formats': self.SUPPORTED_FORMATS,
        }

    def get_dashboard(self) -> str:
        """获取仪表盘(格式化字符串)"""
        saved_mb = self.stats.total_bytes_saved / (1024 * 1024)
        loaded_mb = self.stats.total_bytes_loaded / (1024 * 1024)
        comp_mb = self.stats.compressed_bytes / (1024 * 1024)
        lines = [
            "=" * 60,
            "       WeightSerializer 仪表盘",
            "=" * 60,
            f"  保存操作:       {self.stats.save_operations}",
            f"  加载操作:       {self.stats.load_operations}",
            f"  已保存张量数:   {self.stats.total_tensors_saved}",
            f"  已加载张量数:   {self.stats.total_tensors_loaded}",
            f"  已保存大小:     {saved_mb:.2f} MB",
            f"  已加载大小:     {loaded_mb:.2f} MB",
            f"  压缩大小:       {comp_mb:.2f} MB",
            f"  分片数:         {self.stats.shard_count}",
            f"  压缩启用:       {'是' if self.stats.compressed else '否'}",
            f"  最近保存格式:   {self.stats.last_save_format}",
            f"  最近加载格式:   {self.stats.last_load_format}",
            f"  校验通过数:     {self.stats.checksums_verified}",
            f"  错误数:         {len(self.stats.errors)}",
            f"  支持格式:       {', '.join(self.SUPPORTED_FORMATS)}",
            "=" * 60,
        ]
        return '\n'.join(lines)


# ============================================================================
# #26 HuggingFaceExporter - HuggingFace格式导出
# ============================================================================

class HuggingFaceExporter:
    """HuggingFace格式导出器

    生成HuggingFace transformers兼容的模型文件:
    - config.json: 模型配置
    - model.safetensors: 权重文件
    - tokenizer.json: 分词器配置
    - tokenizer_config.json: 分词器特殊token配置
    - special_tokens_map.json: 特殊token映射
    - generation_config.json: 生成配置
    - README.md: 模型卡片
    """

    # 灵元配置 -> HuggingFace配置字段映射
    CONFIG_MAP = {
        'hidden_size': 'hidden_size',
        'num_layers': 'num_hidden_layers',
        'num_heads': 'num_attention_heads',
        'intermediate_size': 'intermediate_size',
        'vocab_size': 'vocab_size',
        'max_seq_len': 'max_position_embeddings',
        'num_kv_heads': 'num_key_value_heads',
        'rms_norm_eps': 'rms_norm_eps',
        'rope_theta': 'rope_theta',
        'attention_bias': 'attention_bias',
        'hidden_act': 'hidden_act',
        'tie_word_embeddings': 'tie_word_embeddings',
        'bos_token_id': 'bos_token_id',
        'eos_token_id': 'eos_token_id',
        'pad_token_id': 'pad_token_id',
        'initializer_range': 'initializer_range',
        'layer_norm_eps': 'layer_norm_eps',
        'use_cache': 'use_cache',
        'pretraining_tp': 'pretraining_tp',
    }

    # 默认特殊token
    DEFAULT_SPECIAL_TOKENS = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'mask_token': '<mask>',
    }

    # 默认生成配置
    DEFAULT_GEN_CONFIG = {
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'max_new_tokens': 512,
        'repetition_penalty': 1.1,
        'do_sample': True,
        'num_beams': 1,
        'length_penalty': 1.0,
        'early_stopping': False,
    }

    def __init__(self, serializer: Optional[WeightSerializer] = None):
        """初始化HuggingFace导出器"""
        self.serializer = serializer or WeightSerializer()
        self.stats = ExportStats()

    def export(self, model, config: Optional[Dict] = None,
               output_dir: str = None) -> str:
        """导出模型为HuggingFace格式

        Args:
            model: 模型对象
            config: 模型配置(可选, 默认从模型提取)
            output_dir: 输出目录

        Returns:
            输出目录路径
        """
        start_time = time.time()
        self.stats.total_exports += 1

        if output_dir is None:
            output_dir = os.path.join(EXPORT_DIR, f'hf_export_{int(time.time())}')
        _ensure_dir(output_dir)

        # 提取配置
        if config is None:
            config = _get_model_config(model)
        # 统一转为dict (支持dataclass对象)
        if not isinstance(config, dict):
            if hasattr(config, '__dataclass_fields__'):
                try:
                    from dataclasses import asdict as _asdict
                    config = _asdict(config)
                except Exception:
                    pass
            if not isinstance(config, dict):
                config = {k: getattr(config, k, None) for k in dir(config)
                          if not k.startswith('_') and not callable(getattr(config, k, None))}

        # 生成各文件
        files_generated = 0

        # 1. config.json
        config_path = os.path.join(output_dir, 'config.json')
        self._generate_config(config, config_path)
        files_generated += 1

        # 2. model.safetensors (权重)
        weights_path = os.path.join(output_dir, 'model.safetensors')
        self._export_weights(model, weights_path)
        files_generated += 1

        # 3. tokenizer.json
        tokenizer_path = os.path.join(output_dir, 'tokenizer.json')
        self._generate_tokenizer(model, tokenizer_path)
        files_generated += 1

        # 4. tokenizer_config.json
        tok_config_path = os.path.join(output_dir, 'tokenizer_config.json')
        self._generate_tokenizer_config(tok_config_path)
        files_generated += 1

        # 5. special_tokens_map.json
        special_tokens_path = os.path.join(output_dir, 'special_tokens_map.json')
        self._generate_special_tokens_map(special_tokens_path)
        files_generated += 1

        # 6. generation_config.json
        gen_config_path = os.path.join(output_dir, 'generation_config.json')
        self._generate_generation_config(config, gen_config_path)
        files_generated += 1

        # 7. README.md (模型卡片)
        readme_path = os.path.join(output_dir, 'README.md')
        self._generate_readme(config, readme_path)
        files_generated += 1

        # 计算总大小
        total_size = 0
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath):
                total_size += os.path.getsize(fpath)

        # 验证导出
        verified = self._verify_export(output_dir)

        elapsed = time.time() - start_time
        self.stats.total_files_generated += files_generated
        self.stats.total_bytes_exported += total_size
        self.stats.last_export_dir = output_dir
        self.stats.last_export_time = elapsed
        self.stats.verification_passed = verified

        _log(f"HF导出完成: {output_dir}, 文件数={files_generated}, "
             f"大小={total_size} bytes, 耗时={elapsed:.3f}s")

        return output_dir

    def _generate_config(self, config, path: str) -> None:
        """生成 config.json (HuggingFace格式)"""
        # 统一转为dict (支持dataclass对象和dict)
        if not isinstance(config, dict):
            config = {k: getattr(config, k, None) for k in dir(config)
                      if not k.startswith('_') and not callable(getattr(config, k, None))}

        # 映射灵元配置到HF格式
        hf_config = OrderedDict()

        # 基础架构信息
        hf_config['architectures'] = ['LingyuanForCausalLM']
        hf_config['model_type'] = 'lingyuan'
        hf_config['torch_dtype'] = 'float32'
        hf_config['transformers_version'] = '4.36.0'

        # 映射配置字段
        for ly_key, hf_key in self.CONFIG_MAP.items():
            if ly_key in config:
                hf_config[hf_key] = config[ly_key]

        # 默认值
        defaults = {
            'hidden_size': 4096,
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'intermediate_size': 11008,
            'vocab_size': 32000,
            'max_position_embeddings': 4096,
            'num_key_value_heads': 32,
            'rms_norm_eps': 1e-6,
            'rope_theta': 10000.0,
            'attention_bias': False,
            'hidden_act': 'silu',
            'tie_word_embeddings': False,
            'bos_token_id': 1,
            'eos_token_id': 2,
            'pad_token_id': 0,
            'use_cache': True,
        }
        for key, val in defaults.items():
            if key not in hf_config:
                hf_config[key] = val

        # 添加额外字段
        hf_config['auto_map'] = {
            'AutoConfig': 'configuration_lingyuan.LingyuanConfig',
            'AutoModel': 'modeling_lingyuan.LingyuanForCausalLM',
            'AutoModelForCausalLM': 'modeling_lingyuan.LingyuanForCausalLM',
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(hf_config, f, ensure_ascii=False, indent=2)

    def _export_weights(self, model, path: str) -> None:
        """导出权重为 model.safetensors"""
        self.serializer.save_weights(model, path, format='safetensors')

    def _generate_tokenizer(self, model, path: str) -> None:
        """生成 tokenizer.json"""
        tokenizer = _get_model_tokenizer(model)

        # 从分词器提取词表
        if tokenizer is not None:
            vocab = getattr(tokenizer, 'vocab', {})
            merges = getattr(tokenizer, 'merges', [])
            if hasattr(vocab, 'items'):
                vocab_dict = {str(k): int(v) for k, v in vocab.items()}
            elif isinstance(vocab, dict):
                vocab_dict = {str(k): int(v) for k, v in vocab.items()}
            else:
                vocab_dict = {}
            merges_list = []
            if isinstance(merges, list):
                for m in merges:
                    if isinstance(m, (list, tuple)) and len(m) == 2:
                        merges_list.append(f"{m[0]} {m[1]}")
                    elif isinstance(m, str):
                        merges_list.append(m)
        else:
            vocab_dict = {}
            merges_list = []

        # 构建added_tokens
        added_tokens = []
        for token_name, token_str in self.DEFAULT_SPECIAL_TOKENS.items():
            added_tokens.append({
                'content': token_str,
                'single_word': False,
                'lstrip': False,
                'rstrip': False,
                'normalized': False,
                'special': True,
            })

        tokenizer_json = {
            'version': '1.0',
            'truncation': None,
            'padding': None,
            'added_tokens': added_tokens,
            'normalizer': {
                'type': 'Sequence',
                'normalizers': [
                    {'type': 'NFD'},
                    {'type': 'Lowercase'},
                    {'type': 'Strip'},
                ],
            },
            'pre_tokenizer': {
                'type': 'ByteLevel',
                'add_prefix_space': False,
            },
            'post_processor': {
                'type': 'TemplateProcessing',
                'single': [
                    {'SpecialToken': {'id': '<s>', 'type_id': 0}},
                    {'Sequence': {'id': 'A', 'type_id': 0}},
                ],
                'pair': [
                    {'SpecialToken': {'id': '<s>', 'type_id': 0}},
                    {'Sequence': {'id': 'A', 'type_id': 0}},
                    {'SpecialToken': {'id': '</s>', 'type_id': 0}},
                    {'Sequence': {'id': 'B', 'type_id': 1}},
                ],
                'special_tokens': {
                    '<s>': {'id': '<s>', 'ids': [1], 'tokens': ['<s>']},
                    '</s>': {'id': '</s>', 'ids': [2], 'tokens': ['</s>']},
                },
            },
            'decoder': {
                'type': 'ByteLevel',
                'trim_offsets': True,
            },
            'model': {
                'type': 'BPE',
                'vocab': vocab_dict if vocab_dict else {'<unk>': 0, '<s>': 1, '</s>': 2},
                'merges': merges_list,
                'byte_fallback': False,
            },
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    def _generate_tokenizer_config(self, path: str) -> None:
        """生成 tokenizer_config.json"""
        config = {
            'tokenizer_class': 'LingyuanTokenizer',
            'model_max_length': 4096,
            'padding_side': 'right',
            'truncation_side': 'right',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'mask_token': '<mask>',
            'add_bos_token': True,
            'add_eos_token': False,
            'clean_up_tokenization_spaces': False,
            'legacy': True,
            'sp_model_kwargs': {},
            'chat_template': (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}{{ '<s>user: ' + message['content'] + '</s>' }}"
                "{% elif message['role'] == 'assistant' %}{{ '<s>assistant: ' + message['content'] + '</s>' }}"
                "{% endif %}{% endfor %}"
            ),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _generate_special_tokens_map(self, path: str) -> None:
        """生成 special_tokens_map.json"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.DEFAULT_SPECIAL_TOKENS, f, ensure_ascii=False, indent=2)

    def _generate_generation_config(self, config, path: str) -> None:
        """生成 generation_config.json"""
        # 统一转为dict
        if not isinstance(config, dict):
            config = {k: getattr(config, k, None) for k in dir(config)
                      if not k.startswith('_') and not callable(getattr(config, k, None))}

        gen_config = dict(self.DEFAULT_GEN_CONFIG)

        # 从模型配置中提取
        if 'bos_token_id' in config:
            gen_config['bos_token_id'] = config['bos_token_id']
        if 'eos_token_id' in config:
            gen_config['eos_token_id'] = config['eos_token_id']
        if 'pad_token_id' in config:
            gen_config['pad_token_id'] = config['pad_token_id']

        gen_config['transformers_version'] = '4.36.0'

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(gen_config, f, ensure_ascii=False, indent=2)

    def _generate_readme(self, config: Dict, path: str) -> None:
        """生成 README.md (模型卡片)"""
        model_name = config.get('model_name', 'Lingyuan')
        hidden_size = config.get('hidden_size', 4096)
        num_layers = config.get('num_layers', config.get('num_hidden_layers', 32))
        num_heads = config.get('num_heads', config.get('num_attention_heads', 32))
        vocab_size = config.get('vocab_size', 32000)
        max_seq = config.get('max_seq_len', config.get('max_position_embeddings', 4096))

        # 估算参数量
        params_b = (num_layers * (
            hidden_size * hidden_size * 4 +
            hidden_size * vocab_size / num_layers
        )) / 1e9

        readme = f"""# {model_name}

## 模型描述

{model_name} 是灵元大模型系列的一员, 基于 Transformer 解码器架构。

## 模型结构

| 配置项 | 值 |
|--------|-----|
| hidden_size | {hidden_size} |
| num_hidden_layers | {num_layers} |
| num_attention_heads | {num_heads} |
| vocab_size | {vocab_size} |
| max_position_embeddings | {max_seq} |
| 估计参数量 | ~{params_b:.2f}B |

## 使用方法

### 使用 transformers 加载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

## 训练详情

- 架构: Decoder-only Transformer
- 激活函数: SiLU
- 位置编码: RoPE
- 归一化: RMSNorm

## 许可证

Apache License 2.0

## 引用

```
@misc{{lingyuan2026,
  title={{Lingyuan: A Large Language Model}},
  author={{Lingyuan Team}},
  year={{2026}},
}}
```
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(readme)

    def _verify_export(self, output_dir: str) -> bool:
        """验证导出文件完整性"""
        expected_files = [
            'config.json',
            'model.safetensors',
            'tokenizer.json',
            'tokenizer_config.json',
            'special_tokens_map.json',
            'generation_config.json',
            'README.md',
        ]

        for fname in expected_files:
            fpath = os.path.join(output_dir, fname)
            if not os.path.exists(fpath):
                self.stats.errors.append(f"导出验证失败: 缺少文件 {fname}")
                return False
            if os.path.getsize(fpath) == 0:
                self.stats.errors.append(f"导出验证失败: 文件为空 {fname}")
                return False

        # 验证config.json可解析
        try:
            config_path = os.path.join(output_dir, 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if 'architectures' not in cfg:
                self.stats.errors.append("导出验证失败: config.json缺少architectures字段")
                return False
        except Exception as e:
            self.stats.errors.append(f"导出验证异常: {e}")
            return False

        _log(f"HF导出验证通过: {output_dir}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取导出统计"""
        return {
            'total_exports': self.stats.total_exports,
            'total_files_generated': self.stats.total_files_generated,
            'total_bytes_exported': self.stats.total_bytes_exported,
            'last_export_dir': self.stats.last_export_dir,
            'last_export_time': self.stats.last_export_time,
            'verification_passed': self.stats.verification_passed,
            'error_count': len(self.stats.errors),
        }

    def get_dashboard(self) -> str:
        """获取仪表盘"""
        export_mb = self.stats.total_bytes_exported / (1024 * 1024)
        lines = [
            "=" * 60,
            "       HuggingFaceExporter 仪表盘",
            "=" * 60,
            f"  导出次数:       {self.stats.total_exports}",
            f"  生成文件数:     {self.stats.total_files_generated}",
            f"  导出大小:       {export_mb:.2f} MB",
            f"  最近导出目录:   {self.stats.last_export_dir}",
            f"  最近导出耗时:   {self.stats.last_export_time:.3f}s",
            f"  验证通过:       {'是' if self.stats.verification_passed else '否'}",
            f"  错误数:         {len(self.stats.errors)}",
            "=" * 60,
        ]
        return '\n'.join(lines)


# ============================================================================
# #27 ONNXExporter - ONNX导出
# ============================================================================

class ONNXExporter:
    """ONNX格式导出器 (简化JSON表示)

    生成ONNX模型的JSON表示, 包含:
    - 计算图: nodes(操作) + initializers(权重) + inputs + outputs
    - 节点映射: 灵元操作 -> ONNX操作
    - 动态batch: 支持动态batch_size维度

    注意: 生成JSON格式的ONNX模型描述, 不生成真正的protobuf, 但结构正确。
    """

    # 灵元操作 -> ONNX操作映射
    OP_MAP = {
        'linear': ['MatMul', 'Add'],
        'matmul': 'MatMul',
        'add': 'Add',
        'layernorm': 'LayerNormalization',
        'rms_norm': 'LayerNormalization',
        'softmax': 'Softmax',
        'gelu': 'Gelu',
        'silu': 'Mul',
        'relu': 'Relu',
        'sigmoid': 'Sigmoid',
        'tanh': 'Tanh',
        'embedding': 'Gather',
        'transpose': 'Transpose',
        'reshape': 'Reshape',
        'concat': 'Concat',
        'slice': 'Slice',
        'reduce_mean': 'ReduceMean',
        'dropout': 'Identity',
        'mul': 'Mul',
        'div': 'Div',
        'sub': 'Sub',
        'pow': 'Pow',
        'sqrt': 'Sqrt',
        'cast': 'Cast',
        'where': 'Where',
        'trilu': 'Trilu',
    }

    # ONNX数据类型枚举
    DATA_TYPES = {
        'float32': 1,
        'uint8': 2,
        'int8': 3,
        'int32': 6,
        'int64': 7,
        'float16': 10,
        'float64': 11,
    }

    def __init__(self):
        """初始化ONNX导出器"""
        self.stats = ONNXStats()
        self.stats.supported_ops = list(self.OP_MAP.keys())

    def export(self, model, input_shape: Tuple[int, ...],
               output_path: str) -> str:
        """导出模型为ONNX格式(JSON表示)

        Args:
            model: 模型对象
            input_shape: 输入形状 (batch_size, seq_len) 或 (seq_len,)
            output_path: 输出文件路径 (.json)

        Returns:
            输出文件路径
        """
        start_time = time.time()
        self.stats.total_exports += 1

        # 提取权重和配置
        raw_weights = _get_model_weights(model)
        config = _get_model_config(model)
        tensors = _normalize_weights(raw_weights)

        # 构建ONNX模型
        onnx_model = self._build_model(tensors, config, input_shape)

        # 统计
        self.stats.total_nodes = len(onnx_model['graph']['node'])
        self.stats.total_initializers = len(onnx_model['graph']['initializer'])
        self.stats.total_parameters = sum(
            init.get('raw_data_size', 0) for init in onnx_model['graph']['initializer']
        )

        # 写入文件
        _ensure_dir(os.path.dirname(output_path) if os.path.dirname(output_path) else '.')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(onnx_model, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        self.stats.last_export_path = output_path
        self.stats.last_export_time = elapsed

        _log(f"ONNX导出完成: {output_path}, 节点数={self.stats.total_nodes}, "
             f"初始化器={self.stats.total_initializers}, 耗时={elapsed:.3f}s")

        return output_path

    def _build_model(self, tensors: List[WeightTensor], config: Dict,
                     input_shape: Tuple[int, ...]) -> Dict:
        """构建ONNX模型结构"""
        hidden_size = config.get('hidden_size', 4096)
        num_layers = config.get('num_layers', config.get('num_hidden_layers', 32))
        num_heads = config.get('num_heads', config.get('num_attention_heads', 32))
        vocab_size = config.get('vocab_size', 32000)
        intermediate_size = config.get('intermediate_size', 11008)
        max_seq = config.get('max_seq_len', config.get('max_position_embeddings', 4096))

        batch_size = input_shape[0] if len(input_shape) >= 2 else 1
        seq_len = input_shape[1] if len(input_shape) >= 2 else input_shape[0]

        # 构建initializers (权重)
        initializers = []
        for tensor in tensors:
            init = {
                'name': tensor.name,
                'data_type': self.DATA_TYPES.get(tensor.dtype, 1),
                'dims': list(tensor.shape),
                'raw_data_size': tensor.num_elements,
            }
            initializers.append(init)

        # 构建inputs
        inputs = [
            {
                'name': 'input_ids',
                'type': {
                    'tensor_type': {
                        'elem_type': 7,  # INT64
                        'shape': {
                            'dim': [
                                {'dim_param': 'batch_size'},
                                {'dim_value': seq_len},
                            ]
                        },
                    }
                },
            },
            {
                'name': 'attention_mask',
                'type': {
                    'tensor_type': {
                        'elem_type': 7,  # INT64
                        'shape': {
                            'dim': [
                                {'dim_param': 'batch_size'},
                                {'dim_value': seq_len},
                            ]
                        },
                    }
                },
            },
        ]

        # 构建outputs
        outputs = [
            {
                'name': 'logits',
                'type': {
                    'tensor_type': {
                        'elem_type': 1,  # FLOAT
                        'shape': {
                            'dim': [
                                {'dim_param': 'batch_size'},
                                {'dim_value': seq_len},
                                {'dim_value': vocab_size},
                            ]
                        },
                    }
                },
            },
        ]

        # 构建计算图节点
        nodes = self._build_graph_nodes(num_layers, hidden_size, num_heads,
                                         intermediate_size, seq_len)

        # 构建完整ONNX模型
        model = {
            'ir_version': 8,
            'producer_name': 'lingyuan',
            'producer_version': '1.0',
            'domain': 'lingyuan',
            'model_version': 1,
            'doc_string': 'Lingyuan Large Language Model (ONNX JSON representation)',
            'opset_import': [
                {'domain': '', 'version': 17},
                {'domain': 'com.lingyuan', 'version': 1},
            ],
            'metadata_props': [
                {'key': 'model_architecture', 'value': 'decoder_only_transformer'},
                {'key': 'hidden_size', 'value': str(hidden_size)},
                {'key': 'num_layers', 'value': str(num_layers)},
                {'key': 'num_heads', 'value': str(num_heads)},
                {'key': 'vocab_size', 'value': str(vocab_size)},
                {'key': 'max_position_embeddings', 'value': str(max_seq)},
                {'key': 'dynamic_batch', 'value': 'true'},
            ],
            'graph': {
                'name': 'lingyuan_main_graph',
                'node': nodes,
                'initializer': initializers,
                'input': inputs,
                'output': outputs,
                'value_info': [],
            },
        }

        return model

    def _build_graph_nodes(self, num_layers: int, hidden_size: int,
                           num_heads: int, intermediate_size: int,
                           seq_len: int) -> List[Dict]:
        """构建计算图节点(Transformer解码器层)"""
        nodes = []
        head_dim = hidden_size // num_heads

        # 1. Embedding 查找
        nodes.append({
            'name': 'embedding_lookup',
            'op_type': 'Gather',
            'input': ['embedding.weight', 'input_ids'],
            'output': ['hidden_states'],
            'attribute': {'axis': 0},
        })

        # 2. 位置编码 (使用RoPE的简化表示)
        nodes.append({
            'name': 'position_embeddings',
            'op_type': 'Mul',
            'input': ['hidden_states', 'position_ids'],
            'output': ['positioned_states'],
            'attribute': {},
        })
        nodes.append({
            'name': 'residual_0',
            'op_type': 'Add',
            'input': ['hidden_states', 'positioned_states'],
            'output': ['layer_input'],
            'attribute': {},
        })

        # 3. 每个Transformer层
        for i in range(num_layers):
            prefix = f'layers.{i}'

            # 3.1 输入LayerNorm
            nodes.append({
                'name': f'{prefix}.input_layernorm',
                'op_type': 'LayerNormalization',
                'input': [f'layer_input', f'{prefix}.ln1.weight'],
                'output': [f'{prefix}.ln1_out'],
                'attribute': {'epsilon': 1e-6, 'axis': -1},
            })

            # 3.2 Q/K/V 投影
            nodes.append({
                'name': f'{prefix}.q_proj',
                'op_type': 'MatMul',
                'input': [f'{prefix}.ln1_out', f'{prefix}.attention.q_proj.weight'],
                'output': [f'{prefix}.q_out'],
                'attribute': {},
            })
            nodes.append({
                'name': f'{prefix}.k_proj',
                'op_type': 'MatMul',
                'input': [f'{prefix}.ln1_out', f'{prefix}.attention.k_proj.weight'],
                'output': [f'{prefix}.k_out'],
                'attribute': {},
            })
            nodes.append({
                'name': f'{prefix}.v_proj',
                'op_type': 'MatMul',
                'input': [f'{prefix}.ln1_out', f'{prefix}.attention.v_proj.weight'],
                'output': [f'{prefix}.v_out'],
                'attribute': {},
            })

            # 3.3 Reshape (多头拆分)
            nodes.append({
                'name': f'{prefix}.q_reshape',
                'op_type': 'Reshape',
                'input': [f'{prefix}.q_out', 'q_shape'],
                'output': [f'{prefix}.q_reshaped'],
                'attribute': {},
            })
            nodes.append({
                'name': f'{prefix}.k_reshape',
                'op_type': 'Reshape',
                'input': [f'{prefix}.k_out', 'k_shape'],
                'output': [f'{prefix}.k_reshaped'],
                'attribute': {},
            })
            nodes.append({
                'name': f'{prefix}.v_reshape',
                'op_type': 'Reshape',
                'input': [f'{prefix}.v_out', 'v_shape'],
                'output': [f'{prefix}.v_reshaped'],
                'attribute': {},
            })

            # 3.4 转置
            nodes.append({
                'name': f'{prefix}.q_transpose',
                'op_type': 'Transpose',
                'input': [f'{prefix}.q_reshaped'],
                'output': [f'{prefix}.q_t'],
                'attribute': {'perm': [0, 2, 1, 3]},
            })
            nodes.append({
                'name': f'{prefix}.k_transpose',
                'op_type': 'Transpose',
                'input': [f'{prefix}.k_reshaped'],
                'output': [f'{prefix}.k_t'],
                'attribute': {'perm': [0, 2, 1, 3]},
            })
            nodes.append({
                'name': f'{prefix}.v_transpose',
                'op_type': 'Transpose',
                'input': [f'{prefix}.v_reshaped'],
                'output': [f'{prefix}.v_t'],
                'attribute': {'perm': [0, 2, 1, 3]},
            })

            # 3.5 注意力分数: Q * K^T
            nodes.append({
                'name': f'{prefix}.attn_scores',
                'op_type': 'MatMul',
                'input': [f'{prefix}.q_t', f'{prefix}.k_t'],
                'output': [f'{prefix}.attn_raw'],
                'attribute': {},
            })

            # 3.6 缩放
            nodes.append({
                'name': f'{prefix}.attn_scale',
                'op_type': 'Mul',
                'input': [f'{prefix}.attn_raw', 'scale_factor'],
                'output': [f'{prefix}.attn_scaled'],
                'attribute': {},
            })

            # 3.7 因果mask
            nodes.append({
                'name': f'{prefix}.causal_mask',
                'op_type': 'Trilu',
                'input': ['mask_shape'],
                'output': [f'{prefix}.mask'],
                'attribute': {'upper': 0},
            })
            nodes.append({
                'name': f'{prefix}.apply_mask',
                'op_type': 'Add',
                'input': [f'{prefix}.attn_scaled', f'{prefix}.mask'],
                'output': [f'{prefix}.attn_masked'],
                'attribute': {},
            })

            # 3.8 Softmax
            nodes.append({
                'name': f'{prefix}.softmax',
                'op_type': 'Softmax',
                'input': [f'{prefix}.attn_masked'],
                'output': [f'{prefix}.attn_probs'],
                'attribute': {'axis': -1},
            })

            # 3.9 注意力输出: probs * V
            nodes.append({
                'name': f'{prefix}.attn_output',
                'op_type': 'MatMul',
                'input': [f'{prefix}.attn_probs', f'{prefix}.v_t'],
                'output': [f'{prefix}.attn_out'],
                'attribute': {},
            })

            # 3.10 输出投影
            nodes.append({
                'name': f'{prefix}.o_proj',
                'op_type': 'MatMul',
                'input': [f'{prefix}.attn_out', f'{prefix}.attention.o_proj.weight'],
                'output': [f'{prefix}.attn_result'],
                'attribute': {},
            })

            # 3.11 残差连接1
            nodes.append({
                'name': f'{prefix}.residual1',
                'op_type': 'Add',
                'input': ['layer_input', f'{prefix}.attn_result'],
                'output': [f'{prefix}.attn_res'],
                'attribute': {},
            })

            # 3.12 后注意力LayerNorm
            nodes.append({
                'name': f'{prefix}.post_attn_layernorm',
                'op_type': 'LayerNormalization',
                'input': [f'{prefix}.attn_res', f'{prefix}.ln2.weight'],
                'output': [f'{prefix}.ln2_out'],
                'attribute': {'epsilon': 1e-6, 'axis': -1},
            })

            # 3.13 MLP: gate_proj
            nodes.append({
                'name': f'{prefix}.gate_proj',
                'op_type': 'MatMul',
                'input': [f'{prefix}.ln2_out', f'{prefix}.mlp.gate_proj.weight'],
                'output': [f'{prefix}.gate_out'],
                'attribute': {},
            })

            # 3.14 MLP: up_proj
            nodes.append({
                'name': f'{prefix}.up_proj',
                'op_type': 'MatMul',
                'input': [f'{prefix}.ln2_out', f'{prefix}.mlp.up_proj.weight'],
                'output': [f'{prefix}.up_out'],
                'attribute': {},
            })

            # 3.15 激活函数 (SiLU: x * sigmoid(x))
            nodes.append({
                'name': f'{prefix}.silu',
                'op_type': 'Sigmoid',
                'input': [f'{prefix}.gate_out'],
                'output': [f'{prefix}.gate_sigmoid'],
                'attribute': {},
            })
            nodes.append({
                'name': f'{prefix}.silu_mul',
                'op_type': 'Mul',
                'input': [f'{prefix}.gate_out', f'{prefix}.gate_sigmoid'],
                'output': [f'{prefix}.act_out'],
                'attribute': {},
            })
            nodes.append({
                'name': f'{prefix}.mlp_mul',
                'op_type': 'Mul',
                'input': [f'{prefix}.act_out', f'{prefix}.up_out'],
                'output': [f'{prefix}.mlp_inter'],
                'attribute': {},
            })

            # 3.16 MLP: down_proj
            nodes.append({
                'name': f'{prefix}.down_proj',
                'op_type': 'MatMul',
                'input': [f'{prefix}.mlp_inter', f'{prefix}.mlp.down_proj.weight'],
                'output': [f'{prefix}.mlp_out'],
                'attribute': {},
            })

            # 3.17 残差连接2
            nodes.append({
                'name': f'{prefix}.residual2',
                'op_type': 'Add',
                'input': [f'{prefix}.attn_res', f'{prefix}.mlp_out'],
                'output': [f'layer_input_{i+1}'],
                'attribute': {},
            })

            # 更新layer_input引用
            if i < num_layers - 1:
                nodes.append({
                    'name': f'{prefix}.update_input',
                    'op_type': 'Identity',
                    'input': [f'layer_input_{i+1}'],
                    'output': ['layer_input'],
                    'attribute': {},
                })

        # 4. 最终LayerNorm
        nodes.append({
            'name': 'final_layernorm',
            'op_type': 'LayerNormalization',
            'input': [f'layer_input_{num_layers}', 'norm.weight'],
            'output': ['final_hidden'],
            'attribute': {'epsilon': 1e-6, 'axis': -1},
        })

        # 5. LM Head
        nodes.append({
            'name': 'lm_head',
            'op_type': 'MatMul',
            'input': ['final_hidden', 'lm_head.weight'],
            'output': ['logits'],
            'attribute': {},
        })

        return nodes

    def get_stats(self) -> Dict[str, Any]:
        """获取导出统计"""
        return {
            'total_exports': self.stats.total_exports,
            'total_nodes': self.stats.total_nodes,
            'total_initializers': self.stats.total_initializers,
            'total_parameters': self.stats.total_parameters,
            'last_export_path': self.stats.last_export_path,
            'last_export_time': self.stats.last_export_time,
            'supported_ops': list(self.OP_MAP.keys()),
            'error_count': len(self.stats.errors),
        }

    def get_dashboard(self) -> str:
        """获取仪表盘"""
        params_m = self.stats.total_parameters / 1e6
        lines = [
            "=" * 60,
            "       ONNXExporter 仪表盘",
            "=" * 60,
            f"  导出次数:       {self.stats.total_exports}",
            f"  节点数:         {self.stats.total_nodes}",
            f"  初始化器数:     {self.stats.total_initializers}",
            f"  参数量:         {params_m:.2f}M",
            f"  最近导出路径:   {self.stats.last_export_path}",
            f"  最近导出耗时:   {self.stats.last_export_time:.3f}s",
            f"  支持操作数:     {len(self.OP_MAP)}",
            f"  错误数:         {len(self.stats.errors)}",
            "=" * 60,
        ]
        return '\n'.join(lines)


# ============================================================================
# #28 GGUFExporter - GGUF量化导出
# ============================================================================

class GGUFExporter:
    """GGUF量化导出器

    支持GGUF文件格式的写入与读取, 兼容llama.cpp。

    GGUF格式:
    - magic: 0x46554747 ("GGUF")
    - version: 3
    - tensor_count, metadata_kv_count
    - metadata: general.architecture, general.name, tokenizer.ggml.model等
    - tensor info: name, n_dims, dimensions, type(f32/f16/q4_0/q8_0), offset
    - tensor data: 量化后的权重数据

    量化选项:
    - f32: 不量化 (float32)
    - f16: 截断精度到float16
    - q4_0: 对称量化到int4 (每32个值一组)
    - q8_0: 对称量化到int8 (每32个值一组)
    """

    # GGUF magic number: "GGUF" in ASCII, little-endian
    GGUF_MAGIC = 0x46554747
    GGUF_VERSION = 3

    # GGUF元数据值类型
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12

    # GGML tensor类型
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9

    # 量化类型 -> GGML类型ID映射
    QUANT_TYPE_MAP = {
        'f32': GGML_TYPE_F32,
        'f16': GGML_TYPE_F16,
        'q4_0': GGML_TYPE_Q4_0,
        'q8_0': GGML_TYPE_Q8_0,
    }

    # 量化类型 -> GGUF file_type映射
    FILE_TYPE_MAP = {
        'f32': 0,   # LLAMA_FTYPE_ALL_F32
        'f16': 1,   # LLAMA_FTYPE_MOSTLY_F16
        'q4_0': 2,  # LLAMA_FTYPE_MOSTLY_Q4_0
        'q8_0': 7,  # LLAMA_FTYPE_MOSTLY_Q8_0
    }

    # 量化块大小
    BLOCK_SIZE = 32

    # 对齐边界 (字节)
    ALIGNMENT = 32

    def __init__(self):
        """初始化GGUF导出器"""
        self.stats = GGUFStats()

    def write_gguf(self, model, output_path: str,
                   quantization: str = "q4_0") -> str:
        """写入GGUF文件

        Args:
            model: 模型对象或权重字典
            output_path: 输出文件路径
            quantization: 量化类型 (f32/f16/q4_0/q8_0)

        Returns:
            输出文件路径
        """
        start_time = time.time()
        self.stats.total_exports += 1

        if quantization not in self.QUANT_TYPE_MAP:
            raise ValueError(f"不支持的量化类型: {quantization}, "
                             f"支持: {list(self.QUANT_TYPE_MAP.keys())}")

        # 提取权重和配置
        raw_weights = _get_model_weights(model)
        config = _get_model_config(model)
        tensors = _normalize_weights(raw_weights)
        tokenizer = _get_model_tokenizer(model)

        # 准备元数据
        metadata = self._prepare_metadata(config, tokenizer, quantization)

        # 量化并收集张量信息
        tensor_infos = []
        tensor_data_blocks = []
        current_offset = 0

        for tensor in tensors:
            # 量化权重数据
            quantized, type_id = self._quantize(tensor.data, quantization)
            quantized_bytes = bytes(quantized)

            # 对齐偏移
            if current_offset % self.ALIGNMENT != 0:
                padding = self.ALIGNMENT - (current_offset % self.ALIGNMENT)
                current_offset += padding

            tensor_info = {
                'name': tensor.name,
                'n_dims': len(tensor.shape),
                'dimensions': list(tensor.shape),
                'type': type_id,
                'offset': current_offset,
            }
            tensor_infos.append(tensor_info)
            tensor_data_blocks.append(quantized_bytes)

            self.stats.original_bytes += tensor.nbytes
            current_offset += len(quantized_bytes)

        # 写入文件
        _ensure_dir(os.path.dirname(output_path) if os.path.dirname(output_path) else '.')

        with open(output_path, 'wb') as f:
            # 写入头部
            self._write_header(f, metadata, len(tensor_infos))

            # 写入张量信息
            for info in tensor_infos:
                self._write_tensor_info(f, info)

            # 对齐到ALIGNMENT边界
            pos = f.tell()
            if pos % self.ALIGNMENT != 0:
                padding = self.ALIGNMENT - (pos % self.ALIGNMENT)
                f.write(b'\x00' * padding)

            # 写入张量数据
            for block in tensor_data_blocks:
                f.write(block)

        # 统计
        total_bytes = os.path.getsize(output_path)
        self.stats.total_tensors = len(tensor_infos)
        self.stats.total_bytes = total_bytes
        self.stats.quantization_type = quantization
        if self.stats.original_bytes > 0:
            self.stats.compression_ratio = total_bytes / self.stats.original_bytes

        elapsed = time.time() - start_time
        self.stats.last_export_path = output_path
        self.stats.last_export_time = elapsed

        _log(f"GGUF写入完成: {output_path}, 量化={quantization}, "
             f"张量数={len(tensor_infos)}, 大小={total_bytes} bytes, "
             f"压缩比={self.stats.compression_ratio:.2f}, 耗时={elapsed:.3f}s")

        return output_path

    def read_gguf_header(self, path: str) -> Dict[str, Any]:
        """读取GGUF文件头部元数据

        Args:
            path: GGUF文件路径

        Returns:
            包含元数据和张量信息的字典
        """
        with open(path, 'rb') as f:
            # 读取magic
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != self.GGUF_MAGIC:
                raise ValueError(f"无效的GGUF magic: 0x{magic:08X}, 期望: 0x{self.GGUF_MAGIC:08X}")

            # 读取版本
            version = struct.unpack('<I', f.read(4))[0]

            # 读取张量数和元数据KV数
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            # 读取元数据KV对
            metadata = {}
            for _ in range(metadata_kv_count):
                key = self._read_gguf_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_gguf_value(f, value_type)
                metadata[key] = value

            # 读取张量信息
            tensor_infos = []
            for _ in range(tensor_count):
                name = self._read_gguf_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dimensions = []
                for _ in range(n_dims):
                    dim = struct.unpack('<Q', f.read(8))[0]
                    dimensions.append(dim)
                tensor_type = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                tensor_infos.append({
                    'name': name,
                    'n_dims': n_dims,
                    'dimensions': dimensions,
                    'type': tensor_type,
                    'offset': offset,
                })

        result = {
            'magic': hex(magic),
            'version': version,
            'tensor_count': tensor_count,
            'metadata_kv_count': metadata_kv_count,
            'metadata': metadata,
            'tensor_infos': tensor_infos,
            'file_size': os.path.getsize(path),
        }

        _log(f"GGUF头读取完成: {path}, 版本={version}, "
             f"张量数={tensor_count}, 元数据项={metadata_kv_count}")

        return result

    # ------------------------------------------------------------------
    # 元数据准备
    # ------------------------------------------------------------------

    def _prepare_metadata(self, config: Dict, tokenizer, quantization: str) -> List[Tuple]:
        """准备GGUF元数据KV对

        返回: [(key, value_type, value), ...]
        """
        arch = 'llama'  # 默认架构
        hidden_size = config.get('hidden_size', 4096)
        num_layers = config.get('num_layers', config.get('num_hidden_layers', 32))
        num_heads = config.get('num_heads', config.get('num_attention_heads', 32))
        num_kv_heads = config.get('num_kv_heads', config.get('num_key_value_heads', num_heads))
        intermediate_size = config.get('intermediate_size', 11008)
        max_seq = config.get('max_seq_len', config.get('max_position_embeddings', 4096))
        rms_eps = config.get('rms_norm_eps', 1e-6)
        rope_theta = config.get('rope_theta', 10000.0)
        vocab_size = config.get('vocab_size', 32000)
        model_name = config.get('model_name', 'Lingyuan')

        head_dim = hidden_size // num_heads
        rope_dim = head_dim

        meta = []

        # general.* 元数据
        meta.append(('general.architecture', self.GGUF_TYPE_STRING, arch))
        meta.append(('general.name', self.GGUF_TYPE_STRING, model_name))
        meta.append(('general.quantization_version', self.GGUF_TYPE_INT32, 2))
        meta.append(('general.file_type', self.GGUF_TYPE_INT32,
                     self.FILE_TYPE_MAP.get(quantization, 2)))
        meta.append(('general.file_type.name', self.GGUF_TYPE_STRING, quantization))

        # tokenizer.ggml.* 元数据
        meta.append(('tokenizer.ggml.model', self.GGUF_TYPE_STRING, 'llama'))
        meta.append(('tokenizer.ggml.bos_token_id', self.GGUF_TYPE_UINT32, 1))
        meta.append(('tokenizer.ggml.eos_token_id', self.GGUF_TYPE_UINT32, 2))
        meta.append(('tokenizer.ggml.unknown_token_id', self.GGUF_TYPE_UINT32, 0))
        meta.append(('tokenizer.ggml.padding_token_id', self.GGUF_TYPE_UINT32, 0))

        # 从分词器提取词表
        if tokenizer is not None:
            vocab = getattr(tokenizer, 'vocab', None)
            if vocab is not None:
                if hasattr(vocab, 'items'):
                    vocab_items = list(vocab.items())
                elif isinstance(vocab, dict):
                    vocab_items = list(vocab.items())
                else:
                    vocab_items = []
                tokens = [k for k, _ in sorted(vocab_items, key=lambda x: x[1])]
                scores = [0.0] * len(tokens)
                token_types = [0] * len(tokens)  # 0=normal, 1=unknown, 2=control
                meta.append(('tokenizer.ggml.tokens', self.GGUF_TYPE_ARRAY,
                             (self.GGUF_TYPE_STRING, tokens)))
                meta.append(('tokenizer.ggml.scores', self.GGUF_TYPE_ARRAY,
                             (self.GGUF_TYPE_FLOAT32, scores)))
                meta.append(('tokenizer.ggml.token_type', self.GGUF_TYPE_ARRAY,
                             (self.GGUF_TYPE_INT32, token_types)))
            else:
                # 默认词表
                tokens = ['<unk>', '<s>', '</s>'] + [f'token_{i}' for i in range(vocab_size - 3)]
                meta.append(('tokenizer.ggml.tokens', self.GGUF_TYPE_ARRAY,
                             (self.GGUF_TYPE_STRING, tokens)))
                meta.append(('tokenizer.ggml.scores', self.GGUF_TYPE_ARRAY,
                             (self.GGUF_TYPE_FLOAT32, [0.0] * len(tokens))))
                meta.append(('tokenizer.ggml.token_type', self.GGUF_TYPE_ARRAY,
                             (self.GGUF_TYPE_INT32, [0] * len(tokens))))
        else:
            # 无分词器, 使用默认
            tokens = ['<unk>', '<s>', '</s>'] + [f'token_{i}' for i in range(max(vocab_size - 3, 0))]
            meta.append(('tokenizer.ggml.tokens', self.GGUF_TYPE_ARRAY,
                         (self.GGUF_TYPE_STRING, tokens)))
            meta.append(('tokenizer.ggml.scores', self.GGUF_TYPE_ARRAY,
                         (self.GGUF_TYPE_FLOAT32, [0.0] * len(tokens))))
            meta.append(('tokenizer.ggml.token_type', self.GGUF_TYPE_ARRAY,
                         (self.GGUF_TYPE_INT32, [0] * len(tokens))))

        # 架构特定元数据 ({arch}.*)
        meta.append((f'{arch}.context_length', self.GGUF_TYPE_UINT32, max_seq))
        meta.append((f'{arch}.embedding_length', self.GGUF_TYPE_UINT32, hidden_size))
        meta.append((f'{arch}.block_count', self.GGUF_TYPE_UINT32, num_layers))
        meta.append((f'{arch}.feed_forward_length', self.GGUF_TYPE_UINT32, intermediate_size))
        meta.append((f'{arch}.attention.head_count', self.GGUF_TYPE_UINT32, num_heads))
        meta.append((f'{arch}.attention.head_count_kv', self.GGUF_TYPE_UINT32, num_kv_heads))
        meta.append((f'{arch}.rope.dimension_count', self.GGUF_TYPE_UINT32, rope_dim))
        meta.append((f'{arch}.attention.layer_norm_rms_epsilon',
                     self.GGUF_TYPE_FLOAT32, float(rms_eps)))
        meta.append((f'{arch}.rope.freq_base', self.GGUF_TYPE_FLOAT32, float(rope_theta)))

        return meta

    # ------------------------------------------------------------------
    # 量化实现
    # ------------------------------------------------------------------

    def _quantize(self, data: List[float], quant_type: str) -> Tuple[bytearray, int]:
        """量化权重数据

        Args:
            data: float32权重数据
            quant_type: 量化类型

        Returns:
            (量化后的字节数据, GGML类型ID)
        """
        if not data:
            return bytearray(), self.QUANT_TYPE_MAP[quant_type]

        if quant_type == 'f32':
            return self._quantize_f32(data), self.GGML_TYPE_F32
        elif quant_type == 'f16':
            return self._quantize_f16(data), self.GGML_TYPE_F16
        elif quant_type == 'q4_0':
            return self._quantize_q4_0(data), self.GGML_TYPE_Q4_0
        elif quant_type == 'q8_0':
            return self._quantize_q8_0(data), self.GGML_TYPE_Q8_0
        else:
            raise ValueError(f"不支持的量化类型: {quant_type}")

    def _quantize_f32(self, data: List[float]) -> bytearray:
        """f32量化 (无量化, 直接存储float32)"""
        result = bytearray()
        result.extend(struct.pack(f'<{len(data)}f', *data))
        return result

    def _quantize_f16(self, data: List[float]) -> bytearray:
        """f16量化: 截断float32到float16 (IEEE 754半精度)"""
        result = bytearray()
        for v in data:
            try:
                # struct 'e' 格式: IEEE 754 binary16
                result.extend(struct.pack('<e', v))
            except (struct.error, OverflowError):
                # 处理溢出
                if math.isinf(v) or math.isnan(v):
                    result.extend(struct.pack('<e', 0.0))
                elif v > 65504.0:
                    result.extend(struct.pack('<e', 65504.0))
                elif v < -65504.0:
                    result.extend(struct.pack('<e', -65504.0))
                else:
                    result.extend(struct.pack('<e', 0.0))
        return result

    def _quantize_q8_0(self, data: List[float]) -> bytearray:
        """q8_0量化: 对称量化到int8, 每32个值一组

        每个block存储: [f16 scale (2字节)] [32 × int8 (32字节)] = 34字节
        量化公式: scale = amax / 127, q = round(value / scale)
        """
        result = bytearray()
        bs = self.BLOCK_SIZE

        # 补齐到block_size的整数倍
        padded = list(data)
        remainder = len(padded) % bs
        if remainder > 0:
            padded.extend([0.0] * (bs - remainder))

        for i in range(0, len(padded), bs):
            block = padded[i:i + bs]

            # 找最大绝对值
            amax = 0.0
            for v in block:
                av = abs(v)
                if av > amax:
                    amax = av

            # 计算缩放因子
            d = amax / 127.0 if amax > 0 else 0.0
            id_ = 1.0 / d if d != 0 else 0.0

            # 存储scale为f16
            try:
                result.extend(struct.pack('<e', d))
            except (struct.error, OverflowError):
                result.extend(struct.pack('<e', 0.0))

            # 量化每个值到int8
            for v in block:
                q = round(v * id_)
                q = max(-128, min(127, int(q)))
                result.extend(struct.pack('<b', q))

        return result

    def _quantize_q4_0(self, data: List[float]) -> bytearray:
        """q4_0量化: 对称量化到int4, 每32个值一组

        每个block存储: [f16 scale (2字节)] [16 × uint8 (32个4bit值打包)] = 18字节
        量化公式: d = max / -8, q = round(value * (1/d)), 存储 q+8 映射到[0,15]
        """
        result = bytearray()
        bs = self.BLOCK_SIZE

        # 补齐到block_size的整数倍
        padded = list(data)
        remainder = len(padded) % bs
        if remainder > 0:
            padded.extend([0.0] * (bs - remainder))

        for i in range(0, len(padded), bs):
            block = padded[i:i + bs]

            # 找绝对值最大的值(保留符号)
            amax = 0.0
            max_val = 0.0
            for v in block:
                av = abs(v)
                if av > amax:
                    amax = av
                    max_val = v

            # 计算缩放因子 (使用max/-8, 与llama.cpp一致)
            d = max_val / -8.0 if max_val != 0 else 0.0
            id_ = 1.0 / d if d != 0 else 0.0

            # 存储scale为f16
            try:
                result.extend(struct.pack('<e', d))
            except (struct.error, OverflowError):
                result.extend(struct.pack('<e', 0.0))

            # 量化并打包为4bit
            nibbles = []
            for v in block:
                q = v * id_
                # 映射到 [0, 15], 加8.5后截断 (与llama.cpp一致)
                xi = int(q + 8.5)
                xi = max(0, min(15, xi))
                nibbles.append(xi)

            # 两个4bit值打包为一个字节
            for j in range(0, len(nibbles), 2):
                byte_val = nibbles[j] & 0x0F
                if j + 1 < len(nibbles):
                    byte_val |= (nibbles[j + 1] & 0x0F) << 4
                result.extend(struct.pack('<B', byte_val))

        return result

    # ------------------------------------------------------------------
    # GGUF二进制写入
    # ------------------------------------------------------------------

    def _write_header(self, f, metadata: List[Tuple], tensor_count: int) -> None:
        """写入GGUF文件头"""
        # magic
        f.write(struct.pack('<I', self.GGUF_MAGIC))
        # version
        f.write(struct.pack('<I', self.GGUF_VERSION))
        # tensor_count
        f.write(struct.pack('<Q', tensor_count))
        # metadata_kv_count
        f.write(struct.pack('<Q', len(metadata)))

        # 写入元数据KV对
        for key, value_type, value in metadata:
            self._write_gguf_string(f, key)
            f.write(struct.pack('<I', value_type))  # 写入值类型
            self._write_gguf_value(f, value_type, value)

    def _write_tensor_info(self, f, info: Dict) -> None:
        """写入张量信息"""
        self._write_gguf_string(f, info['name'])
        f.write(struct.pack('<I', info['n_dims']))
        for dim in info['dimensions']:
            f.write(struct.pack('<Q', int(dim)))
        f.write(struct.pack('<I', info['type']))
        f.write(struct.pack('<Q', info['offset']))

    def _write_gguf_string(self, f, s: str) -> None:
        """写入GGUF字符串 (uint64长度 + UTF-8字节)"""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _read_gguf_string(self, f) -> str:
        """读取GGUF字符串"""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _write_gguf_value(self, f, value_type: int, value: Any) -> None:
        """写入GGUF元数据值"""
        if value_type == self.GGUF_TYPE_UINT8:
            f.write(struct.pack('<B', int(value)))
        elif value_type == self.GGUF_TYPE_INT8:
            f.write(struct.pack('<b', int(value)))
        elif value_type == self.GGUF_TYPE_UINT16:
            f.write(struct.pack('<H', int(value)))
        elif value_type == self.GGUF_TYPE_INT16:
            f.write(struct.pack('<h', int(value)))
        elif value_type == self.GGUF_TYPE_UINT32:
            f.write(struct.pack('<I', int(value)))
        elif value_type == self.GGUF_TYPE_INT32:
            f.write(struct.pack('<i', int(value)))
        elif value_type == self.GGUF_TYPE_FLOAT32:
            f.write(struct.pack('<f', float(value)))
        elif value_type == self.GGUF_TYPE_BOOL:
            f.write(struct.pack('<B', 1 if value else 0))
        elif value_type == self.GGUF_TYPE_STRING:
            self._write_gguf_string(f, str(value))
        elif value_type == self.GGUF_TYPE_UINT64:
            f.write(struct.pack('<Q', int(value)))
        elif value_type == self.GGUF_TYPE_INT64:
            f.write(struct.pack('<q', int(value)))
        elif value_type == self.GGUF_TYPE_FLOAT64:
            f.write(struct.pack('<d', float(value)))
        elif value_type == self.GGUF_TYPE_ARRAY:
            # value = (elem_type, values)
            elem_type, values = value
            f.write(struct.pack('<I', elem_type))
            f.write(struct.pack('<Q', len(values)))
            for v in values:
                self._write_gguf_value(f, elem_type, v)
        else:
            raise ValueError(f"未知的GGUF值类型: {value_type}")

    def _read_gguf_value(self, f, value_type: int) -> Any:
        """读取GGUF元数据值"""
        if value_type == self.GGUF_TYPE_UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == self.GGUF_TYPE_INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == self.GGUF_TYPE_UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == self.GGUF_TYPE_INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == self.GGUF_TYPE_UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == self.GGUF_TYPE_INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == self.GGUF_TYPE_FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == self.GGUF_TYPE_BOOL:
            return bool(struct.unpack('<B', f.read(1))[0])
        elif value_type == self.GGUF_TYPE_STRING:
            return self._read_gguf_string(f)
        elif value_type == self.GGUF_TYPE_UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == self.GGUF_TYPE_INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == self.GGUF_TYPE_FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif value_type == self.GGUF_TYPE_ARRAY:
            elem_type = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<Q', f.read(8))[0]
            values = []
            for _ in range(length):
                values.append(self._read_gguf_value(f, elem_type))
            return values
        else:
            raise ValueError(f"未知的GGUF值类型: {value_type}")

    def get_stats(self) -> Dict[str, Any]:
        """获取导出统计"""
        return {
            'total_exports': self.stats.total_exports,
            'total_tensors': self.stats.total_tensors,
            'total_bytes': self.stats.total_bytes,
            'original_bytes': self.stats.original_bytes,
            'quantization_type': self.stats.quantization_type,
            'compression_ratio': self.stats.compression_ratio,
            'last_export_path': self.stats.last_export_path,
            'last_export_time': self.stats.last_export_time,
            'supported_quant_types': list(self.QUANT_TYPE_MAP.keys()),
            'error_count': len(self.stats.errors),
        }

    def get_dashboard(self) -> str:
        """获取仪表盘"""
        total_mb = self.stats.total_bytes / (1024 * 1024)
        orig_mb = self.stats.original_bytes / (1024 * 1024)
        lines = [
            "=" * 60,
            "       GGUFExporter 仪表盘",
            "=" * 60,
            f"  导出次数:       {self.stats.total_exports}",
            f"  张量数:         {self.stats.total_tensors}",
            f"  原始大小:       {orig_mb:.2f} MB",
            f"  导出大小:       {total_mb:.2f} MB",
            f"  压缩比:         {self.stats.compression_ratio:.2f}",
            f"  量化类型:       {self.stats.quantization_type}",
            f"  最近导出路径:   {self.stats.last_export_path}",
            f"  最近导出耗时:   {self.stats.last_export_time:.3f}s",
            f"  支持量化类型:   {', '.join(self.QUANT_TYPE_MAP.keys())}",
            f"  错误数:         {len(self.stats.errors)}",
            "=" * 60,
        ]
        return '\n'.join(lines)


# ============================================================================
# #29 ExternalModelImporter - 外部模型导入
# ============================================================================

class ExternalModelImporter:
    """外部模型导入器

    支持从HuggingFace导入外部模型, 包括:
    - 模拟从HuggingFace下载
    - 支持的模型: Qwen, Llama, Mistral, ChatGLM, Baichuan, Yi, DeepSeek
    - 配置转换: HF config -> 灵元 ModelConfig
    - 权重转换: HF权重名 -> 灵元权重名映射
    - 分词器导入: HF tokenizer -> 灵元 BPETokenizer
    - 模型注册: 导入后自动注册到灵元 ModelRegistry
    - 缓存管理: 下载的模型缓存到本地
    - 验证: 导入后验证模型完整性
    """

    # 支持的架构列表
    SUPPORTED_ARCHITECTURES = [
        'qwen', 'llama', 'mistral', 'chatglm', 'baichuan', 'yi', 'deepseek'
    ]

    # 架构映射表: 每个架构的层名映射规则
    ARCHITECTURE_MAPPINGS = {
        'llama': {
            'model_type': 'llama',
            'architectures': ['LlamaForCausalLM'],
            'config_map': {
                'hidden_size': 'hidden_size',
                'num_hidden_layers': 'num_layers',
                'num_attention_heads': 'num_heads',
                'intermediate_size': 'intermediate_size',
                'vocab_size': 'vocab_size',
                'max_position_embeddings': 'max_seq_len',
                'num_key_value_heads': 'num_kv_heads',
                'rms_norm_eps': 'rms_norm_eps',
                'rope_theta': 'rope_theta',
                'bos_token_id': 'bos_token_id',
                'eos_token_id': 'eos_token_id',
                'pad_token_id': 'pad_token_id',
                'tie_word_embeddings': 'tie_word_embeddings',
            },
            'weight_map': {
                'model.embed_tokens.weight': 'embedding.weight',
                'model.layers.{n}.input_layernorm.weight': 'layers.{n}.ln1.weight',
                'model.layers.{n}.self_attn.q_proj.weight': 'layers.{n}.attention.q_proj.weight',
                'model.layers.{n}.self_attn.q_proj.bias': 'layers.{n}.attention.q_proj.bias',
                'model.layers.{n}.self_attn.k_proj.weight': 'layers.{n}.attention.k_proj.weight',
                'model.layers.{n}.self_attn.k_proj.bias': 'layers.{n}.attention.k_proj.bias',
                'model.layers.{n}.self_attn.v_proj.weight': 'layers.{n}.attention.v_proj.weight',
                'model.layers.{n}.self_attn.v_proj.bias': 'layers.{n}.attention.v_proj.bias',
                'model.layers.{n}.self_attn.o_proj.weight': 'layers.{n}.attention.o_proj.weight',
                'model.layers.{n}.self_attn.o_proj.bias': 'layers.{n}.attention.o_proj.bias',
                'model.layers.{n}.post_attention_layernorm.weight': 'layers.{n}.ln2.weight',
                'model.layers.{n}.mlp.gate_proj.weight': 'layers.{n}.mlp.gate_proj.weight',
                'model.layers.{n}.mlp.down_proj.weight': 'layers.{n}.mlp.down_proj.weight',
                'model.layers.{n}.mlp.up_proj.weight': 'layers.{n}.mlp.up_proj.weight',
                'model.norm.weight': 'norm.weight',
                'lm_head.weight': 'lm_head.weight',
            },
        },
        'mistral': {
            'model_type': 'mistral',
            'architectures': ['MistralForCausalLM'],
            'config_map': {
                'hidden_size': 'hidden_size',
                'num_hidden_layers': 'num_layers',
                'num_attention_heads': 'num_heads',
                'intermediate_size': 'intermediate_size',
                'vocab_size': 'vocab_size',
                'max_position_embeddings': 'max_seq_len',
                'num_key_value_heads': 'num_kv_heads',
                'rms_norm_eps': 'rms_norm_eps',
                'rope_theta': 'rope_theta',
                'sliding_window': 'sliding_window',
            },
            'weight_map': {
                'model.embed_tokens.weight': 'embedding.weight',
                'model.layers.{n}.input_layernorm.weight': 'layers.{n}.ln1.weight',
                'model.layers.{n}.self_attn.q_proj.weight': 'layers.{n}.attention.q_proj.weight',
                'model.layers.{n}.self_attn.k_proj.weight': 'layers.{n}.attention.k_proj.weight',
                'model.layers.{n}.self_attn.v_proj.weight': 'layers.{n}.attention.v_proj.weight',
                'model.layers.{n}.self_attn.o_proj.weight': 'layers.{n}.attention.o_proj.weight',
                'model.layers.{n}.post_attention_layernorm.weight': 'layers.{n}.ln2.weight',
                'model.layers.{n}.mlp.gate_proj.weight': 'layers.{n}.mlp.gate_proj.weight',
                'model.layers.{n}.mlp.down_proj.weight': 'layers.{n}.mlp.down_proj.weight',
                'model.layers.{n}.mlp.up_proj.weight': 'layers.{n}.mlp.up_proj.weight',
                'model.norm.weight': 'norm.weight',
                'lm_head.weight': 'lm_head.weight',
            },
        },
        'qwen': {
            'model_type': 'qwen',
            'architectures': ['QWenLMHeadModel'],
            'config_map': {
                'hidden_size': 'hidden_size',
                'num_hidden_layers': 'num_layers',
                'num_attention_heads': 'num_heads',
                'intermediate_size': 'intermediate_size',
                'vocab_size': 'vocab_size',
                'max_position_embeddings': 'max_seq_len',
                'layer_norm_epsilon': 'rms_norm_eps',
                'bos_token_id': 'bos_token_id',
                'eos_token_id': 'eos_token_id',
            },
            'weight_map': {
                'transformer.wte.weight': 'embedding.weight',
                'transformer.h.{n}.ln_1.weight': 'layers.{n}.ln1.weight',
                'transformer.h.{n}.ln_2.weight': 'layers.{n}.ln2.weight',
                'transformer.h.{n}.attn.c_attn.weight': 'layers.{n}.attention.qkv.weight',
                'transformer.h.{n}.attn.c_attn.bias': 'layers.{n}.attention.qkv.bias',
                'transformer.h.{n}.attn.c_proj.weight': 'layers.{n}.attention.o_proj.weight',
                'transformer.h.{n}.mlp.w1.weight': 'layers.{n}.mlp.gate_proj.weight',
                'transformer.h.{n}.mlp.w2.weight': 'layers.{n}.mlp.down_proj.weight',
                'transformer.h.{n}.mlp.c_proj.weight': 'layers.{n}.mlp.up_proj.weight',
                'transformer.ln_f.weight': 'norm.weight',
                'lm_head.weight': 'lm_head.weight',
            },
        },
        'chatglm': {
            'model_type': 'chatglm',
            'architectures': ['ChatGLMModel'],
            'config_map': {
                'hidden_size': 'hidden_size',
                'num_layers': 'num_layers',
                'num_attention_heads': 'num_heads',
                'vocab_size': 'vocab_size',
                'max_sequence_length': 'max_seq_len',
                'layernorm_epsilon': 'rms_norm_eps',
            },
            'weight_map': {
                'transformer.word_embeddings.weight': 'embedding.weight',
                'transformer.layers.{n}.input_layernorm.weight': 'layers.{n}.ln1.weight',
                'transformer.layers.{n}.attention.query_key_value.weight': 'layers.{n}.attention.qkv.weight',
                'transformer.layers.{n}.attention.query_key_value.bias': 'layers.{n}.attention.qkv.bias',
                'transformer.layers.{n}.attention.dense.weight': 'layers.{n}.attention.o_proj.weight',
                'transformer.layers.{n}.post_attention_layernorm.weight': 'layers.{n}.ln2.weight',
                'transformer.layers.{n}.mlp.dense_h_to_4h.weight': 'layers.{n}.mlp.gate_proj.weight',
                'transformer.layers.{n}.mlp.dense_4h_to_h.weight': 'layers.{n}.mlp.down_proj.weight',
                'transformer.final_layernorm.weight': 'norm.weight',
                'lm_head.weight': 'lm_head.weight',
            },
        },
        'baichuan': {
            'model_type': 'baichuan',
            'architectures': ['BaichuanForCausalLM'],
            'config_map': {
                'hidden_size': 'hidden_size',
                'num_hidden_layers': 'num_layers',
                'num_attention_heads': 'num_heads',
                'intermediate_size': 'intermediate_size',
                'vocab_size': 'vocab_size',
                'max_position_embeddings': 'max_seq_len',
                'rms_norm_eps': 'rms_norm_eps',
            },
            'weight_map': {
                'model.embed_tokens.weight': 'embedding.weight',
                'model.layers.{n}.input_layernorm.weight': 'layers.{n}.ln1.weight',
                'model.layers.{n}.self_attn.W_pack.weight': 'layers.{n}.attention.qkv.weight',
                'model.layers.{n}.self_attn.o_proj.weight': 'layers.{n}.attention.o_proj.weight',
                'model.layers.{n}.post_attention_layernorm.weight': 'layers.{n}.ln2.weight',
                'model.layers.{n}.mlp.gate_proj.weight': 'layers.{n}.mlp.gate_proj.weight',
                'model.layers.{n}.mlp.down_proj.weight': 'layers.{n}.mlp.down_proj.weight',
                'model.layers.{n}.mlp.up_proj.weight': 'layers.{n}.mlp.up_proj.weight',
                'model.norm.weight': 'norm.weight',
                'lm_head.weight': 'lm_head.weight',
            },
        },
        'yi': {
            'model_type': 'yi',
            'architectures': ['YiForCausalLM'],
            'config_map': {
                'hidden_size': 'hidden_size',
                'num_hidden_layers': 'num_layers',
                'num_attention_heads': 'num_heads',
                'intermediate_size': 'intermediate_size',
                'vocab_size': 'vocab_size',
                'max_position_embeddings': 'max_seq_len',
                'num_key_value_heads': 'num_kv_heads',
                'rms_norm_eps': 'rms_norm_eps',
                'rope_theta': 'rope_theta',
            },
            'weight_map': {
                'model.embed_tokens.weight': 'embedding.weight',
                'model.layers.{n}.input_layernorm.weight': 'layers.{n}.ln1.weight',
                'model.layers.{n}.self_attn.q_proj.weight': 'layers.{n}.attention.q_proj.weight',
                'model.layers.{n}.self_attn.k_proj.weight': 'layers.{n}.attention.k_proj.weight',
                'model.layers.{n}.self_attn.v_proj.weight': 'layers.{n}.attention.v_proj.weight',
                'model.layers.{n}.self_attn.o_proj.weight': 'layers.{n}.attention.o_proj.weight',
                'model.layers.{n}.post_attention_layernorm.weight': 'layers.{n}.ln2.weight',
                'model.layers.{n}.mlp.gate_proj.weight': 'layers.{n}.mlp.gate_proj.weight',
                'model.layers.{n}.mlp.down_proj.weight': 'layers.{n}.mlp.down_proj.weight',
                'model.layers.{n}.mlp.up_proj.weight': 'layers.{n}.mlp.up_proj.weight',
                'model.norm.weight': 'norm.weight',
                'lm_head.weight': 'lm_head.weight',
            },
        },
        'deepseek': {
            'model_type': 'deepseek',
            'architectures': ['DeepseekForCausalLM'],
            'config_map': {
                'hidden_size': 'hidden_size',
                'num_hidden_layers': 'num_layers',
                'num_attention_heads': 'num_heads',
                'intermediate_size': 'intermediate_size',
                'vocab_size': 'vocab_size',
                'max_position_embeddings': 'max_seq_len',
                'num_key_value_heads': 'num_kv_heads',
                'rms_norm_eps': 'rms_norm_eps',
                'rope_theta': 'rope_theta',
            },
            'weight_map': {
                'model.embed_tokens.weight': 'embedding.weight',
                'model.layers.{n}.input_layernorm.weight': 'layers.{n}.ln1.weight',
                'model.layers.{n}.self_attn.q_proj.weight': 'layers.{n}.attention.q_proj.weight',
                'model.layers.{n}.self_attn.k_proj.weight': 'layers.{n}.attention.k_proj.weight',
                'model.layers.{n}.self_attn.v_proj.weight': 'layers.{n}.attention.v_proj.weight',
                'model.layers.{n}.self_attn.o_proj.weight': 'layers.{n}.attention.o_proj.weight',
                'model.layers.{n}.post_attention_layernorm.weight': 'layers.{n}.ln2.weight',
                'model.layers.{n}.mlp.gate_proj.weight': 'layers.{n}.mlp.gate_proj.weight',
                'model.layers.{n}.mlp.down_proj.weight': 'layers.{n}.mlp.down_proj.weight',
                'model.layers.{n}.mlp.up_proj.weight': 'layers.{n}.mlp.up_proj.weight',
                'model.norm.weight': 'norm.weight',
                'lm_head.weight': 'lm_head.weight',
            },
        },
    }

    # repo_id 前缀到架构的映射
    REPO_ARCH_MAP = {
        'qwen': 'qwen',
        'Qwen': 'qwen',
        'llama': 'llama',
        'Llama': 'llama',
        'meta-llama': 'llama',
        'mistral': 'mistral',
        'Mistral': 'mistral',
        'chatglm': 'chatglm',
        'ChatGLM': 'chatglm',
        'THUDM': 'chatglm',
        'baichuan': 'baichuan',
        'Baichuan': 'baichuan',
        'baichuan-inc': 'baichuan',
        'Yi': 'yi',
        '01-ai': 'yi',
        'deepseek': 'deepseek',
        'DeepSeek': 'deepseek',
        'deepseek-ai': 'deepseek',
    }

    def __init__(self, cache_dir: Optional[str] = None,
                 serializer: Optional[WeightSerializer] = None):
        """初始化外部模型导入器

        Args:
            cache_dir: 缓存目录
            serializer: 权重序列化器
        """
        self.cache_dir = cache_dir or MODEL_CACHE_DIR
        _ensure_dir(self.cache_dir)
        self.serializer = serializer or WeightSerializer()
        self.stats = ImportStats()
        self.stats.supported_architectures = list(self.SUPPORTED_ARCHITECTURES)

    def download_model(self, repo_id: str, revision: str = "main") -> str:
        """模拟从HuggingFace下载模型

        Args:
            repo_id: HuggingFace仓库ID (如 "meta-llama/Llama-2-7b-hf")
            revision: 模型版本

        Returns:
            本地缓存路径
        """
        # 检查缓存
        cache_path = self._get_cache_path(repo_id, revision)
        if os.path.exists(cache_path) and os.path.exists(os.path.join(cache_path, 'config.json')):
            self.stats.cache_hits += 1
            _log(f"从缓存加载模型: {repo_id}@{revision}")
            return cache_path

        self.stats.cache_misses += 1
        self.stats.total_downloads += 1

        # 检测架构
        architecture = self._detect_architecture(repo_id)
        _log(f"开始下载模型: {repo_id}@{revision}, 架构: {architecture}")

        # 模拟生成配置
        hf_config = self._simulate_hf_config(repo_id, architecture)

        # 模拟生成权重
        hf_weights = self._simulate_hf_weights(hf_config, architecture)

        # 模拟生成分词器
        hf_tokenizer = self._simulate_hf_tokenizer(architecture, hf_config)

        # 保存到缓存
        _ensure_dir(cache_path)

        # 保存config.json
        config_path = os.path.join(cache_path, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(hf_config, f, ensure_ascii=False, indent=2)

        # 保存权重 (使用safetensors格式)
        weights_path = os.path.join(cache_path, 'model.safetensors')
        self.serializer.save_weights(hf_weights, weights_path, format='safetensors')

        # 保存分词器配置
        tok_config_path = os.path.join(cache_path, 'tokenizer_config.json')
        with open(tok_config_path, 'w', encoding='utf-8') as f:
            json.dump(hf_tokenizer, f, ensure_ascii=False, indent=2)

        # 保存tokenizer.json
        tok_path = os.path.join(cache_path, 'tokenizer.json')
        with open(tok_path, 'w', encoding='utf-8') as f:
            json.dump({
                'version': '1.0',
                'model': {
                    'type': 'BPE',
                    'vocab': {t: i for i, t in enumerate(hf_tokenizer.get('tokens', []))},
                    'merges': [],
                },
            }, f, ensure_ascii=False, indent=2)

        # 计算下载大小
        download_size = 0
        for fname in os.listdir(cache_path):
            fpath = os.path.join(cache_path, fname)
            if os.path.isfile(fpath):
                download_size += os.path.getsize(fpath)

        self.stats.total_bytes_downloaded += download_size
        _log(f"模型下载完成: {repo_id}, 大小: {download_size} bytes, 缓存: {cache_path}")

        return cache_path

    def import_model(self, repo_id: str, revision: str = "main",
                     register: bool = True) -> Dict[str, Any]:
        """导入外部模型到灵元格式

        完整流程: 下载 -> 配置转换 -> 权重转换 -> 分词器导入 -> 注册 -> 验证

        Args:
            repo_id: HuggingFace仓库ID
            revision: 模型版本
            register: 是否注册到ModelRegistry

        Returns:
            导入结果字典
        """
        start_time = time.time()
        self.stats.total_imports += 1
        self.stats.last_imported_model = repo_id

        # 1. 下载模型
        cache_path = self.download_model(repo_id, revision)
        architecture = self._detect_architecture(repo_id)

        # 2. 加载HF配置
        config_path = os.path.join(cache_path, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            hf_config = json.load(f)

        # 3. 配置转换
        ly_config = self._convert_config(hf_config, architecture)

        # 4. 加载并转换权重
        weights_path = os.path.join(cache_path, 'model.safetensors')
        hf_weights = self.serializer.load_weights(weights_path)
        ly_weights = self._convert_weights(hf_weights, architecture)

        # 5. 分词器导入
        ly_tokenizer = self._import_tokenizer(cache_path, architecture)

        # 6. 验证
        verified = self._verify_import(ly_config, ly_weights)

        # 7. 注册到ModelRegistry
        registered = False
        if register and verified:
            registered = self._register_model(repo_id, ly_config, ly_weights, ly_tokenizer)

        elapsed = time.time() - start_time

        result = {
            'repo_id': repo_id,
            'revision': revision,
            'architecture': architecture,
            'config': ly_config,
            'weight_count': len(ly_weights),
            'tokenizer_vocab_size': len(ly_tokenizer.get('vocab', {})) if ly_tokenizer else 0,
            'verified': verified,
            'registered': registered,
            'cache_path': cache_path,
            'import_time': elapsed,
        }

        _log(f"模型导入完成: {repo_id}, 架构={architecture}, "
             f"权重数={len(ly_weights)}, 验证={verified}, 注册={registered}, "
             f"耗时={elapsed:.3f}s")

        return result

    # ------------------------------------------------------------------
    # 架构检测
    # ------------------------------------------------------------------

    def _detect_architecture(self, repo_id: str) -> str:
        """根据repo_id检测模型架构"""
        repo_lower = repo_id.lower()

        # 检查已知前缀
        for key, arch in self.REPO_ARCH_MAP.items():
            if key in repo_id:
                return arch

        # 默认使用llama架构
        return 'llama'

    # ------------------------------------------------------------------
    # 配置转换
    # ------------------------------------------------------------------

    def _convert_config(self, hf_config: Dict, architecture: str) -> Dict[str, Any]:
        """将HF配置转换为灵元配置

        Args:
            hf_config: HuggingFace配置字典
            architecture: 模型架构

        Returns:
            灵元配置字典
        """
        mapping = self.ARCHITECTURE_MAPPINGS.get(architecture, {})
        config_map = mapping.get('config_map', {})

        ly_config = {
            'model_name': hf_config.get('_name_or_path', architecture),
            'architecture': architecture,
            'hf_model_type': hf_config.get('model_type', ''),
            'hf_architectures': hf_config.get('architectures', []),
        }

        # 映射配置字段
        for hf_key, ly_key in config_map.items():
            if hf_key in hf_config:
                ly_config[ly_key] = hf_config[hf_key]

        # 确保必要字段有默认值
        defaults = {
            'hidden_size': 4096,
            'num_layers': 32,
            'num_heads': 32,
            'num_kv_heads': 32,
            'intermediate_size': 11008,
            'vocab_size': 32000,
            'max_seq_len': 4096,
            'rms_norm_eps': 1e-6,
            'rope_theta': 10000.0,
            'bos_token_id': 1,
            'eos_token_id': 2,
            'pad_token_id': 0,
            'tie_word_embeddings': False,
        }
        for key, val in defaults.items():
            if key not in ly_config:
                ly_config[key] = val

        _log(f"配置转换完成: {architecture}, 字段数={len(ly_config)}")
        return ly_config

    # ------------------------------------------------------------------
    # 权重转换
    # ------------------------------------------------------------------

    def _convert_weights(self, hf_weights: Dict[str, Any],
                         architecture: str) -> Dict[str, Any]:
        """将HF权重名转换为灵元权重名

        Args:
            hf_weights: HuggingFace权重字典
            architecture: 模型架构

        Returns:
            转换后的权重字典(灵元命名)
        """
        mapping = self.ARCHITECTURE_MAPPINGS.get(architecture, {})
        weight_map = mapping.get('weight_map', {})

        ly_weights = {}
        unmapped = []

        for hf_name, data in hf_weights.items():
            ly_name = self._convert_weight_name(hf_name, weight_map)
            if ly_name is not None:
                ly_weights[ly_name] = data
            else:
                # 未映射的权重保留原名
                ly_weights[hf_name] = data
                unmapped.append(hf_name)

        if unmapped:
            _log(f"权重转换: {len(unmapped)}个权重未找到映射, 保留原名", "WARN")

        _log(f"权重转换完成: {architecture}, "
             f"HF权重数={len(hf_weights)}, 灵元权重数={len(ly_weights)}")

        return ly_weights

    def _convert_weight_name(self, hf_name: str,
                             weight_map: Dict[str, str]) -> Optional[str]:
        """转换单个权重名

        支持带 {n} 占位符的层名映射

        Args:
            hf_name: HuggingFace权重名
            weight_map: 权重名映射表

        Returns:
            灵元权重名, 或None如果无映射
        """
        # 1. 尝试直接匹配
        if hf_name in weight_map:
            mapped = weight_map[hf_name]
            return mapped.replace('{n}', '0') if '{n}' in mapped else mapped

        # 2. 尝试模式匹配(带层号)
        for hf_pattern, ly_pattern in weight_map.items():
            if '{n}' not in hf_pattern:
                continue

            # 分割模式获取前缀和后缀
            parts = hf_pattern.split('{n}')
            if len(parts) != 2:
                continue
            prefix, suffix = parts

            if hf_name.startswith(prefix) and hf_name.endswith(suffix):
                # 提取层号
                middle = hf_name[len(prefix):len(hf_name) - len(suffix)] if suffix \
                    else hf_name[len(prefix):]
                if middle.isdigit():
                    n = int(middle)
                    return ly_pattern.replace('{n}', str(n))

        # 3. 尝试通配符匹配 (更宽松)
        for hf_pattern, ly_pattern in weight_map.items():
            if '{n}' not in hf_pattern:
                # 尝试部分匹配
                if hf_pattern in hf_name or hf_name in hf_pattern:
                    return ly_pattern.replace('{n}', '0') if '{n}' in ly_pattern else ly_pattern

        return None

    # ------------------------------------------------------------------
    # 分词器导入
    # ------------------------------------------------------------------

    def _import_tokenizer(self, cache_path: str,
                          architecture: str) -> Dict[str, Any]:
        """导入分词器

        Args:
            cache_path: 缓存路径
            architecture: 模型架构

        Returns:
            灵元分词器配置
        """
        ly_tokenizer = {
            'type': 'BPE',
            'vocab': {},
            'merges': [],
            'special_tokens': {},
        }

        # 尝试加载tokenizer.json
        tok_path = os.path.join(cache_path, 'tokenizer.json')
        if os.path.exists(tok_path):
            try:
                with open(tok_path, 'r', encoding='utf-8') as f:
                    tok_data = json.load(f)
                model = tok_data.get('model', {})
                ly_tokenizer['vocab'] = model.get('vocab', {})
                ly_tokenizer['merges'] = model.get('merges', [])

                # 提取added_tokens作为special_tokens
                added = tok_data.get('added_tokens', [])
                for at in added:
                    ly_tokenizer['special_tokens'][at.get('content', '')] = at
            except Exception as e:
                _log(f"加载tokenizer.json失败: {e}", "WARN")

        # 尝试加载tokenizer_config.json
        tok_config_path = os.path.join(cache_path, 'tokenizer_config.json')
        if os.path.exists(tok_config_path):
            try:
                with open(tok_config_path, 'r', encoding='utf-8') as f:
                    tok_config = json.load(f)
                ly_tokenizer['tokenizer_class'] = tok_config.get('tokenizer_class', '')
                ly_tokenizer['model_max_length'] = tok_config.get('model_max_length', 4096)
                ly_tokenizer['bos_token'] = tok_config.get('bos_token', '<s>')
                ly_tokenizer['eos_token'] = tok_config.get('eos_token', '</s>')
                ly_tokenizer['unk_token'] = tok_config.get('unk_token', '<unk>')
                ly_tokenizer['pad_token'] = tok_config.get('pad_token', '<pad>')
            except Exception as e:
                _log(f"加载tokenizer_config.json失败: {e}", "WARN")

        _log(f"分词器导入完成: {architecture}, 词表大小={len(ly_tokenizer['vocab'])}")
        return ly_tokenizer

    # ------------------------------------------------------------------
    # 模型注册
    # ------------------------------------------------------------------

    def _register_model(self, repo_id: str, config: Dict,
                        weights: Dict, tokenizer: Dict) -> bool:
        """注册模型到灵元 ModelRegistry

        尝试使用全局的 ModelRegistry, 如果不存在则创建本地注册表

        Args:
            repo_id: 仓库ID
            config: 灵元配置
            weights: 灵元权重
            tokenizer: 灵元分词器

        Returns:
            是否注册成功
        """
        model_name = config.get('model_name', repo_id)

        # 尝试使用全局ModelRegistry
        model_registry = globals().get('ModelRegistry', None)

        if model_registry is not None:
            try:
                # 如果ModelRegistry有register方法
                if hasattr(model_registry, 'register'):
                    model_registry.register(model_name, {
                        'config': config,
                        'weights': weights,
                        'tokenizer': tokenizer,
                        'source': 'huggingface',
                        'repo_id': repo_id,
                    })
                    _log(f"模型已注册到全局ModelRegistry: {model_name}")
                    return True
                # 如果ModelRegistry有register_model方法
                elif hasattr(model_registry, 'register_model'):
                    model_registry.register_model(model_name, config, weights)
                    _log(f"模型已注册到全局ModelRegistry: {model_name}")
                    return True
            except Exception as e:
                _log(f"注册到全局ModelRegistry失败: {e}", "WARN")

        # 本地注册表
        local_registry_path = os.path.join(self.cache_dir, 'model_registry.json')
        registry = {}
        if os.path.exists(local_registry_path):
            try:
                with open(local_registry_path, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
            except Exception:
                registry = {}

        registry[model_name] = {
            'repo_id': repo_id,
            'architecture': config.get('architecture', ''),
            'config': config,
            'weight_count': len(weights),
            'tokenizer_vocab_size': len(tokenizer.get('vocab', {})) if tokenizer else 0,
            'registered_at': datetime.now().isoformat(),
        }

        try:
            with open(local_registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
            _log(f"模型已注册到本地注册表: {model_name}")
            return True
        except Exception as e:
            _log(f"本地注册失败: {e}", "ERROR")
            self.stats.errors.append(f"注册失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    def _verify_import(self, config: Dict, weights: Dict) -> bool:
        """验证导入的模型完整性

        Args:
            config: 灵元配置
            weights: 灵元权重

        Returns:
            是否验证通过
        """
        # 检查必要配置字段
        required_fields = ['hidden_size', 'num_layers', 'num_heads', 'vocab_size']
        for field_name in required_fields:
            if field_name not in config:
                self.stats.errors.append(f"验证失败: 配置缺少字段 {field_name}")
                return False

        # 检查权重非空
        if not weights:
            self.stats.errors.append("验证失败: 权重为空")
            return False

        # 检查关键权重是否存在
        has_embedding = any('embedding' in name for name in weights.keys())
        if not has_embedding:
            _log("警告: 未找到embedding权重", "WARN")

        # 检查层数
        num_layers = config.get('num_layers', 32)
        layer_count = len(set(
            name.split('.')[1] for name in weights.keys()
            if name.startswith('layers.') and len(name.split('.')) > 1 and name.split('.')[1].isdigit()
        ))
        if layer_count > 0 and layer_count != num_layers:
            _log(f"警告: 权重层数({layer_count})与配置层数({num_layers})不匹配", "WARN")

        _log(f"导入验证通过: {len(weights)}个权重, {num_layers}层")
        return True

    # ------------------------------------------------------------------
    # 缓存管理
    # ------------------------------------------------------------------

    def _get_cache_path(self, repo_id: str, revision: str) -> str:
        """获取模型缓存路径

        将repo_id转换为安全的目录名

        Args:
            repo_id: 仓库ID
            revision: 版本

        Returns:
            缓存路径
        """
        # 将repo_id转换为安全目录名
        safe_name = repo_id.replace('/', '__').replace('\\', '__')
        safe_name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in safe_name)
        return os.path.join(self.cache_dir, f"{safe_name}__{revision}")

    def clear_cache(self, repo_id: Optional[str] = None) -> int:
        """清除模型缓存

        Args:
            repo_id: 指定repo_id清除, None则清除全部

        Returns:
            清除的缓存数量
        """
        cleared = 0
        if not os.path.exists(self.cache_dir):
            return 0

        if repo_id:
            cache_path = self._get_cache_path(repo_id, 'main')
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path, ignore_errors=True)
                cleared = 1
        else:
            for item in os.listdir(self.cache_dir):
                item_path = os.path.join(self.cache_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                    cleared += 1

        _log(f"缓存清除: {cleared}项")
        return cleared

    def list_cached_models(self) -> List[Dict[str, Any]]:
        """列出已缓存的模型

        Returns:
            缓存模型信息列表
        """
        models = []
        if not os.path.exists(self.cache_dir):
            return models

        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            if not os.path.isdir(item_path):
                continue

            # 解析repo_id和revision
            if '__' in item:
                parts = item.rsplit('__', 1)
                repo_id = parts[0].replace('__', '/')
                revision = parts[1] if len(parts) > 1 else 'main'
            else:
                repo_id = item
                revision = 'unknown'

            # 计算大小
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(item_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        total_size += os.path.getsize(fpath)
                        file_count += 1
                    except Exception:
                        pass

            models.append({
                'repo_id': repo_id,
                'revision': revision,
                'cache_path': item_path,
                'size_bytes': total_size,
                'file_count': file_count,
            })

        return models

    # ------------------------------------------------------------------
    # 模拟数据生成
    # ------------------------------------------------------------------

    def _simulate_hf_config(self, repo_id: str, architecture: str) -> Dict[str, Any]:
        """模拟生成HuggingFace配置

        根据repo_id和架构生成合理的配置
        """
        # 根据架构设置默认参数
        arch_defaults = {
            'llama': {
                'hidden_size': 4096, 'num_hidden_layers': 32,
                'num_attention_heads': 32, 'intermediate_size': 11008,
                'vocab_size': 32000, 'max_position_embeddings': 4096,
            },
            'mistral': {
                'hidden_size': 4096, 'num_hidden_layers': 32,
                'num_attention_heads': 32, 'intermediate_size': 14336,
                'vocab_size': 32000, 'max_position_embeddings': 32768,
            },
            'qwen': {
                'hidden_size': 4096, 'num_hidden_layers': 32,
                'num_attention_heads': 32, 'intermediate_size': 11008,
                'vocab_size': 151936, 'max_position_embeddings': 32768,
            },
            'chatglm': {
                'hidden_size': 4096, 'num_hidden_layers': 28,
                'num_attention_heads': 32, 'intermediate_size': 13696,
                'vocab_size': 65024, 'max_position_embeddings': 32768,
            },
            'baichuan': {
                'hidden_size': 4096, 'num_hidden_layers': 32,
                'num_attention_heads': 32, 'intermediate_size': 11008,
                'vocab_size': 64000, 'max_position_embeddings': 4096,
            },
            'yi': {
                'hidden_size': 4096, 'num_hidden_layers': 32,
                'num_attention_heads': 32, 'intermediate_size': 11008,
                'vocab_size': 64000, 'max_position_embeddings': 4096,
            },
            'deepseek': {
                'hidden_size': 4096, 'num_hidden_layers': 32,
                'num_attention_heads': 32, 'intermediate_size': 11008,
                'vocab_size': 32000, 'max_position_embeddings': 4096,
            },
        }

        defaults = arch_defaults.get(architecture, arch_defaults['llama'])

        # 获取架构映射信息
        arch_info = self.ARCHITECTURE_MAPPINGS.get(architecture, {})
        model_type = arch_info.get('model_type', architecture)
        hf_architectures = arch_info.get('architectures', [f'{architecture.capitalize()}ForCausalLM'])

        hf_config = {
            'architectures': hf_architectures,
            'model_type': model_type,
            'torch_dtype': 'float32',
            'transformers_version': '4.36.0',
            'hidden_size': defaults['hidden_size'],
            'num_hidden_layers': defaults['num_hidden_layers'],
            'num_attention_heads': defaults['num_attention_heads'],
            'intermediate_size': defaults['intermediate_size'],
            'vocab_size': defaults['vocab_size'],
            'max_position_embeddings': defaults['max_position_embeddings'],
            'rms_norm_eps': 1e-6,
            'rope_theta': 10000.0,
            'attention_bias': False,
            'hidden_act': 'silu',
            'tie_word_embeddings': False,
            'bos_token_id': 1,
            'eos_token_id': 2,
            'pad_token_id': 0,
            'use_cache': True,
            '_name_or_path': repo_id,
        }

        # 架构特定字段
        if architecture in ('llama', 'mistral', 'yi', 'deepseek'):
            hf_config['num_key_value_heads'] = defaults['num_attention_heads']
        if architecture == 'mistral':
            hf_config['sliding_window'] = 4096
        if architecture == 'chatglm':
            hf_config['max_sequence_length'] = defaults['max_position_embeddings']
            hf_config['layernorm_epsilon'] = 1e-5
        if architecture == 'qwen':
            hf_config['layer_norm_epsilon'] = 1e-6

        return hf_config

    def _simulate_hf_weights(self, hf_config: Dict,
                             architecture: str) -> Dict[str, List[float]]:
        """模拟生成HuggingFace格式权重

        使用架构映射表生成正确命名的权重张量

        Args:
            hf_config: HuggingFace配置
            architecture: 模型架构

        Returns:
            HF格式权重字典
        """
        arch_info = self.ARCHITECTURE_MAPPINGS.get(architecture, {})
        weight_map = arch_info.get('weight_map', {})

        hidden_size = hf_config.get('hidden_size', 4096)
        num_layers = hf_config.get('num_hidden_layers', 32)
        vocab_size = hf_config.get('vocab_size', 32000)
        intermediate_size = hf_config.get('intermediate_size', 11008)
        num_heads = hf_config.get('num_attention_heads', 32)
        head_dim = hidden_size // num_heads

        weights = {}

        # 生成embedding权重
        weights['embedding.weight'] = [random.uniform(-0.02, 0.02)
                                        for _ in range(min(vocab_size * 4, 1000))]

        # 生成每层权重
        for n in range(min(num_layers, 4)):
            weights[f'layers.{n}.ln1.weight'] = [1.0] * min(hidden_size, 64)
            weights[f'layers.{n}.attention.q_proj.weight'] = [0.0] * min(64, 100)
            weights[f'layers.{n}.attention.k_proj.weight'] = [0.0] * min(64, 100)
            weights[f'layers.{n}.attention.v_proj.weight'] = [0.0] * min(64, 100)
            weights[f'layers.{n}.attention.o_proj.weight'] = [0.0] * min(64, 100)
            weights[f'layers.{n}.ln2.weight'] = [1.0] * min(hidden_size, 64)
            weights[f'layers.{n}.mlp.gate_proj.weight'] = [0.0] * min(64, 100)
            weights[f'layers.{n}.mlp.up_proj.weight'] = [0.0] * min(64, 100)
            weights[f'layers.{n}.mlp.down_proj.weight'] = [0.0] * min(64, 100)

        # 最终norm和lm_head
        weights['norm.weight'] = [1.0] * min(hidden_size, 64)
        weights['lm_head.weight'] = [0.0] * min(vocab_size * 4, 1000)

        # 将灵元命名转换为HF命名 (反向映射)
        hf_weights = {}
        reverse_map = {}
        for hf_pattern, ly_pattern in weight_map.items():
            if '{n}' in ly_pattern:
                for n in range(min(num_layers, 4)):
                    hf_name = hf_pattern.replace('{n}', str(n))
                    ly_name = ly_pattern.replace('{n}', str(n))
                    reverse_map[ly_name] = hf_name
            else:
                reverse_map[ly_pattern] = hf_pattern

        for ly_name, data in weights.items():
            if ly_name in reverse_map:
                hf_weights[reverse_map[ly_name]] = data
            else:
                hf_weights[ly_name] = data

        return hf_weights

    def _simulate_hf_tokenizer(self, architecture: str,
                               hf_config: Dict) -> Dict[str, Any]:
        """模拟生成HuggingFace分词器配置

        Args:
            architecture: 模型架构
            hf_config: HuggingFace配置

        Returns:
            分词器配置字典
        """
        vocab_size = hf_config.get('vocab_size', 32000)

        # 生成基础token列表
        tokens = ['<unk>', '<s>', '</s>', '<pad>', '<mask>']
        for i in range(vocab_size - 5):
            tokens.append(f'token_{i}')

        # 架构特定的分词器配置
        arch_tok_config = {
            'llama': {'tokenizer_class': 'LlamaTokenizer', 'model_max_length': 4096},
            'mistral': {'tokenizer_class': 'LlamaTokenizer', 'model_max_length': 32768},
            'qwen': {'tokenizer_class': 'QWenTokenizer', 'model_max_length': 32768},
            'chatglm': {'tokenizer_class': 'ChatGLMTokenizer', 'model_max_length': 32768},
            'baichuan': {'tokenizer_class': 'BaichuanTokenizer', 'model_max_length': 4096},
            'yi': {'tokenizer_class': 'LlamaTokenizer', 'model_max_length': 4096},
            'deepseek': {'tokenizer_class': 'LlamaTokenizer', 'model_max_length': 4096},
        }

        tok_defaults = arch_tok_config.get(architecture, arch_tok_config['llama'])

        tokenizer_config = {
            'tokenizer_class': tok_defaults['tokenizer_class'],
            'model_max_length': tok_defaults['model_max_length'],
            'padding_side': 'right',
            'truncation_side': 'right',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'mask_token': '<mask>',
            'add_bos_token': True,
            'add_eos_token': False,
            'clean_up_tokenization_spaces': False,
            'legacy': True,
            'tokens': tokens[:100],  # 只保存部分用于演示
        }

        return tokenizer_config

    def get_stats(self) -> Dict[str, Any]:
        """获取导入统计"""
        return {
            'total_imports': self.stats.total_imports,
            'total_downloads': self.stats.total_downloads,
            'total_bytes_downloaded': self.stats.total_bytes_downloaded,
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'supported_architectures': self.stats.supported_architectures,
            'last_imported_model': self.stats.last_imported_model,
            'error_count': len(self.stats.errors),
        }

    def get_dashboard(self) -> str:
        """获取仪表盘"""
        download_mb = self.stats.total_bytes_downloaded / (1024 * 1024)
        lines = [
            "=" * 60,
            "       ExternalModelImporter 仪表盘",
            "=" * 60,
            f"  导入次数:       {self.stats.total_imports}",
            f"  下载次数:       {self.stats.total_downloads}",
            f"  下载大小:       {download_mb:.2f} MB",
            f"  缓存命中:       {self.stats.cache_hits}",
            f"  缓存未命中:     {self.stats.cache_misses}",
            f"  支持架构:       {', '.join(self.stats.supported_architectures)}",
            f"  最近导入模型:   {self.stats.last_imported_model}",
            f"  错误数:         {len(self.stats.errors)}",
            "=" * 60,
        ]
        return '\n'.join(lines)