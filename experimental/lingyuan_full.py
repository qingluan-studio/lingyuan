#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# LINGYUAN MODEL - 灵元大模型系统
# v1.0.0 | 日期: 2024-08
# 灵元研究院 | 认知智能实验室
#
# 三大部分:
#   Part 1: 核心架构 (配置/Token/能源/供应商/存储/环境/灾备)
#   Part 2: 训练流程 (数据/自举/蒸馏/评估)
#   Part 3: 部署与API (Agent编排/流水线/仪表盘/闭环)
#
# 快速启动:
#   from lingyuan_full import LingyuanOrchestrator
#   orch = LingyuanOrchestrator()
#   orch.quick_train(generations=3)
# ============================================================

import os
import json
import time
import random
import math
import re
import threading
import copy
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

# ============================================================
# 全局目录配置
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'lingyuan_data')
LOG_DIR = os.path.join(BASE_DIR, 'lingyuan_logs')
CONFIG_DIR = os.path.join(BASE_DIR, 'lingyuan_config')

for _d in [DATA_DIR, LOG_DIR, CONFIG_DIR]:
    os.makedirs(_d, exist_ok=True)

# ============================================================
# 配置区
# ============================================================

# ========== Token计费配置 ==========
TOKEN_CONFIG = {
    "token_unit": "Token",
    "base_price": 2.0,                # 基础价格
    "electricity_embedded": 0.8,      # 电力成本系数
    "cooling_embedded": 0.3,          # 冷却成本系数
    "bandwidth_embedded": 0.2,        # 带宽成本系数
    "gpu_embedded": 0.3,             # GPU成本系数
    "ops_embedded": 0.1,             # 运维成本系数
    "profit_margin": 0.4,            # 利润率

    # 时段定价
    "peak_hours": [9, 10, 11, 14, 15, 16],  # 峰时(9-11点, 14-16点)
    "peak_markup": 1.2,                      # 峰时加价20%
    "off_peak_discount": 0.7,                # 平峰7折
    "night_discount": 0.5,                   # 凌晨5折(0-6时)

    # 绿电
    "green_power_discount": 0.85,    # 绿电折扣

    # 购买限制
    "min_purchase": 1,               # 最小购买量(至少1小时起步)
    "max_wallet_balance": 100000,    # 最大钱包余额
    "transfer_fee": 0.02,            # 转账费率
    "expire_days": 365,              # Token有效期
}

# ========== 供应商配置 ==========
VENDOR_CONFIG = {
    "vendors": [
        {
            "id": "vendor_a",
            "name": "供应商A-A100",
            "gpu_type": "A100",
            "price_multiplier": 0.8,
            "speed_factor": 0.7,
            "reliability": 0.95,
            "max_concurrent": 50,
            "green_power_ratio": 0.3,
        },
        {
            "id": "vendor_b",
            "name": "供应商B-H100",
            "gpu_type": "H100",
            "price_multiplier": 1.5,
            "speed_factor": 1.3,
            "reliability": 0.99,
            "max_concurrent": 30,
            "green_power_ratio": 0.6,
        },
        {
            "id": "vendor_c",
            "name": "供应商C-L40S",
            "gpu_type": "L40S",
            "price_multiplier": 1.0,
            "speed_factor": 1.0,
            "reliability": 0.97,
            "max_concurrent": 40,
            "green_power_ratio": 0.5,
        },
    ],
    "selection_strategy": "cost_optimal",  # cost_optimal / speed_optimal / balanced
    "failover_enabled": True,
    "failover_retry_limit": 3,
}

# ========== 电力配置 ==========
POWER_CONFIG = {
    "energy_per_token": 0.35,        # 每Token能耗(千瓦时)
    "carbon_per_kwh": 0.581,         # 电网碳排放因子(kg CO2)
    "peak_power_hours": [10, 11, 12, 13, 14],
    "green_power_hours": [10, 11, 12, 13],  # 绿电时段(10-14点)
    "carbon_tracking": True,
    "esg_report_enabled": True,
}

# ========== 网络配置 ==========
CHANNEL_CONFIG = {
    "tiers": {
        "fast": {"price_per_gb": 0.5, "bandwidth": "10Gbps"},
        "standard": {"price_per_gb": 0.2, "bandwidth": "5Gbps"},
        "economy": {"price_per_gb": 0.05, "bandwidth": "1Gbps"},
    },
    "default_tier": "standard",
}

# ========== 存储配置 ==========
STORAGE_CONFIG = {
    "price_per_gb_month": 0.1,
    "hot_tier_threshold_gb": 100,
    "cold_tier_discount": 0.3,
    "auto_cleanup_days": 30,
}

# ========== 灾备配置 ==========
DISASTER_CONFIG = {
    "checkpoint_interval": 100,      # 每N步触发checkpoint
    "auto_resume": True,             # 自动故障恢复
    "max_retry": 3,
    "backup_vendors": ["vendor_c"],  # 冷备服务商
}


# ============================================================
# TOKEN_SYSTEM [代币经济系统]
# ============================================================

@dataclass
class TokenBatch:
    """代币批次记录"""
    batch_id: str
    amount: int                    # Token数量
    purchase_price: float          # 购入价格(元/Token)
    purchase_time: str             # 购入时间
    expire_time: str               # 过期时间
    source: str = "purchase"       # purchase/transfer/green_power
    green_power: bool = False      # 是否绿电Token
    used: int = 0                  # 已使用量
    remaining: int = field(init=False)

    def __post_init__(self):
        self.remaining = self.amount - self.used

    def is_expired(self) -> bool:
        expire_dt = datetime.fromisoformat(self.expire_time)
        return datetime.now() > expire_dt

    def use(self, count: int) -> bool:
        """使用Token"""
        if count > self.remaining or self.is_expired():
            return False
        self.used += count
        self.remaining = self.amount - self.used
        return True

    def to_dict(self) -> Dict:
        return asdict(self)


class TokenPricingEngine:
    """Token定价引擎

    支持动态定价策略: 峰谷电价、绿电折扣、批量折扣
    """

    def __init__(self, config=TOKEN_CONFIG):
        self.config = config

    def get_current_price(self, green_power=False) -> Dict:
        """获取实时Token价格"""
        now = datetime.now()
        hour = now.hour

        # 峰谷判断
        time_factor = 1.0
        if hour in self.config["peak_hours"]:
            time_factor = self.config.get("peak_markup", 1.2)  # 高峰时段加价20%
        elif 0 <= hour < 6:
            time_factor = self.config["night_discount"]  # 凌晨5折
        else:
            time_factor = self.config.get("off_peak_discount", 0.7)  # 平峰7折

        # 绿电因子
        green_factor = 1.0
        if green_power:
            green_factor = self.config["green_power_discount"]

        # 计算价格
        base = self.config["base_price"]
        elec = self.config["electricity_embedded"]
        cooling = self.config["cooling_embedded"]
        bandwidth = self.config["bandwidth_embedded"]
        gpu = self.config.get("gpu_embedded", 0.3)
        ops = self.config.get("ops_embedded", 0.1)
        margin = self.config["profit_margin"]

        cost_base = base + elec + cooling + bandwidth + gpu + ops
        final_price = cost_base * time_factor * green_factor * (1 + margin)

        return {
            "unit_price": round(final_price, 4),
            "base_price": base,
            "time_factor": round(time_factor, 2),
            "green_factor": round(green_factor, 2),
            "cost_breakdown": {
                "base": base,
                "electricity": elec,
                "cooling": cooling,
                "bandwidth": bandwidth,
                "gpu": gpu,
                "ops": ops,
                "margin": round(cost_base * margin, 4),
            },
            "timestamp": now.isoformat(),
            "hour": hour,
            "is_peak": hour in self.config["peak_hours"],
            "is_green": green_power,
        }

    def estimate_cost(self, gpu_hours: int, green_power=False) -> Dict:
        """估算成本"""
        pricing = self.get_current_price(green_power)
        total = pricing['unit_price'] * gpu_hours
        return {
            "gpu_hours": gpu_hours,
            "unit_price": pricing['unit_price'],
            "total_cost": round(total, 2),
            "time_needed": gpu_hours,
            "pricing_detail": pricing,
        }


class TokenSystem:
    """Token钱包系统

    - 用户ID和Token余额
    - Token批次管理(先进先出)
    - 交易历史记录
    - 价格引擎集成
    """

    def __init__(self, user_id: str, config=TOKEN_CONFIG):
        self.user_id = user_id
        self.config = config
        self.wallet_file = os.path.join(DATA_DIR, f'wallet_{user_id}.json')
        self.batches: List[TokenBatch] = []
        self.transaction_history: List[Dict] = []
        self.pricing = TokenPricingEngine(config)
        self._load()

    def _load(self):
        """从文件加载钱包数据"""
        if os.path.exists(self.wallet_file):
            with open(self.wallet_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                batches_data = data.get('batches', [])
                self.batches = []
                for b in batches_data:
                    b.pop('remaining', None)  # remaining字段废弃,重新计算
                    self.batches.append(TokenBatch(**b))
                self.transaction_history = data.get('transactions', [])

    def _save(self):
        """保存钱包数据到文件"""
        data = {
            "user_id": self.user_id,
            "batches": [b.to_dict() for b in self.batches],
            "transactions": self.transaction_history[-1000:],  # 只保留最近1000条
        }
        with open(self.wallet_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def purchase(self, amount: int, green_power=False) -> Dict:
        """购买Token"""
        if amount < self.config['min_purchase']:
            return {"success": False, "error": f"最低购买: {self.config['min_purchase']} Token"}

        balance = self.get_balance()
        cost = self.pricing.estimate_cost(amount, green_power)
        total_cost = cost['total_cost']

        if balance + amount > self.config['max_wallet_balance']:
            return {"success": False, "error": f"超出最大限额: {self.config['max_wallet_balance']} Token"}

        now = datetime.now()
        expire = now + timedelta(days=self.config['expire_days'])

        batch = TokenBatch(
            batch_id=f"batch_{int(time.time())}_{len(self.batches)}",
            amount=amount,
            purchase_price=cost['unit_price'],
            purchase_time=now.isoformat(),
            expire_time=expire.isoformat(),
            source='purchase',
            green_power=green_power,
        )
        self.batches.append(batch)

        transaction = {
            "type": "purchase",
            "amount": amount,
            "cost": total_cost,
            "unit_price": cost['unit_price'],
            "batch_id": batch.batch_id,
            "green_power": green_power,
            "timestamp": now.isoformat(),
        }
        self.transaction_history.append(transaction)
        self._save()

        return {
            "success": True,
            "batch_id": batch.batch_id,
            "amount": amount,
            "total_cost": total_cost,
            "unit_price": cost["unit_price"],
            "expire_time": batch.expire_time,
        }

    def consume(self, amount: int, task_id: str = "") -> Dict:
        """消费Token

        逻辑:
        1. 查找可用Token
        2. 按过期时间优先、绿色能源优先
        3. 扣除并记录
        """
        available = [b for b in self.batches if not b.is_expired() and b.remaining > 0]
        if sum(b.remaining for b in available) < amount:
            return {"success": False, "error": "Token不足", "balance": self.get_balance()}

        # 按过期时间升序、绿色能源优先、购买价格升序
        available.sort(key=lambda b: (b.expire_time, not b.green_power, b.purchase_price))

        consumed_detail = []
        remaining_to_consume = amount
        for batch in available:
            if remaining_to_consume <= 0:
                break
            take = min(batch.remaining, remaining_to_consume)
            batch.remaining -= take
            consumed_detail.append({
                "batch_id": batch.batch_id,
                "amount": take,
                "unit_price": batch.purchase_price,
                "green_power": batch.green_power,
            })
            remaining_to_consume -= take

        transaction = {
            "type": "consume",
            "amount": amount,
            "task_id": task_id,
            "detail": consumed_detail,
            "timestamp": datetime.now().isoformat(),
        }
        self.transaction_history.append(transaction)
        self._save()

        return {
            "success": True,
            "consumed": amount,
            "detail": consumed_detail,
            "remaining_balance": self.get_balance(),
        }

    def transfer(self, to_user_id: str, amount: int) -> Dict:
        """转账Token给其他用户

        1. 扣除手续费
        2. 创建转账批次
        """
        if self.get_balance() < amount:
            return {"success": False, "error": "余额不足"}

        fee = int(amount * self.config["transfer_fee"])
        actual_transfer = amount - fee

        # 先扣除发送方
        self.consume(amount, f"transfer_to_{to_user_id}")

        now = datetime.now()
        expire = now + timedelta(days=self.config["expire_days"])

        # 创建接收方批次
        target_wallet_file = os.path.join(DATA_DIR, f"wallet_{to_user_id}.json")
        target_batches = []
        if os.path.exists(target_wallet_file):
            with open(target_wallet_file, 'r', encoding='utf-8') as f:
                target_data = json.load(f)
                target_batches = target_data.get("batches", [])

        # 创建新批次
        new_batch = TokenBatch(
            batch_id=f"transfer_{int(time.time())}",
            amount=actual_transfer,
            purchase_price=0,  # 转账Token无购买价格
            purchase_time=now.isoformat(),
            expire_time=expire.isoformat(),
            source="transfer",
        )
        target_batches.append(new_batch.to_dict())

        with open(target_wallet_file, 'w', encoding='utf-8') as f:
            json.dump({
                "user_id": to_user_id,
                "batches": target_batches,
                "transactions": [],
            }, f, ensure_ascii=False, indent=2)

        transaction = {
            "type": "transfer",
            "to_user": to_user_id,
            "amount": amount,
            "fee": fee,
            "actual_transfer": actual_transfer,
            "timestamp": now.isoformat(),
        }
        self.transaction_history.append(transaction)
        self._save()

        return {
            "success": True,
            "transferred": actual_transfer,
            "fee": fee,
            "to_user": to_user_id,
        }

    def get_balance(self) -> int:
        """获取当前可用余额"""
        return sum(b.remaining for b in self.batches if not b.is_expired())

    def get_wallet_summary(self) -> Dict:
        """获取钱包摘要"""
        active_batches = [b for b in self.batches if not b.is_expired() and b.remaining > 0]
        total = sum(b.remaining for b in active_batches)
        green_total = sum(b.remaining for b in active_batches if b.green_power)
        expiring_soon = sum(
            b.remaining for b in active_batches
            if datetime.fromisoformat(b.expire_time) < datetime.now() + timedelta(days=7)
        )

        return {
            "user_id": self.user_id,
            "total_balance": total,
            "green_power_balance": green_total,
            "expiring_soon": expiring_soon,
            "active_batches": len(active_batches),
            "current_price": self.pricing.get_current_price(),
            "wallet_limit": self.config['max_wallet_balance'],
        }

    def get_transaction_history(self, limit=20) -> List[Dict]:
        """获取交易历史"""
        return self.transaction_history[-limit:]

    def cleanup_expired(self) -> int:
        """清理过期Token"""
        before = len(self.batches)
        self.batches = [b for b in self.batches if not b.is_expired()]
        cleaned = before - len(self.batches)
        if cleaned:
            self._save()
        return cleaned


# ============================================================
# ENERGY_SYSTEM [能源系统模块]
# ============================================================

@dataclass
class EnergyRecord:
    """能源消费记录"""
    record_id: str
    task_id: str
    tokens_consumed: int
    energy_kwh: float          # 消耗的电量(度)
    carbon_kg: float           # 碳排放量(kg)
    green_power: bool          # 是否使用绿色电力
    vendor_id: str             # 供应商ID
    timestamp: str


class EnergySystem:
    """能源消耗追踪系统

    Token能源消耗计算与追踪:
    - 记录每次任务能耗
    - 计算碳足迹
    - ESG碳信用转换
    - 绿色能源折扣
    """

    def __init__(self, config=POWER_CONFIG, token_config=TOKEN_CONFIG):
        self.config = config
        self.token_config = token_config
        self.energy_file = os.path.join(DATA_DIR, "energy_records.json")
        self.records: List[EnergyRecord] = []
        self._load()

    def _load(self):
        if os.path.exists(self.energy_file):
            with open(self.energy_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.records = [EnergyRecord(**r) for r in data.get('records', [])]

    def _save(self):
        data = {
            'records': [asdict(r) for r in self.records[-5000:]],
        }
        with open(self.energy_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def calculate_energy(self, tokens: int, green_power=False, vendor_id="") -> Dict:
        """计算Token消耗对应的能源指标"""
        energy_per_token = self.config['energy_per_token']
        carbon_per_kwh = self.config['carbon_per_kwh']

        energy_kwh = tokens * energy_per_token
        carbon_kg = energy_kwh * carbon_per_kwh

        # 绿色电力折扣计算
        carbon_saved = 0.0
        if green_power:
            # 绿色电力减少80%碳排放
            carbon_saved = carbon_kg * 0.8
            carbon_kg = carbon_kg * 0.2

        return {
            "energy_kwh": round(energy_kwh, 4),
            "carbon_kg": round(carbon_kg, 4),
            "green_power": green_power,
            "carbon_saved": round(carbon_saved, 4),
            "vendor_id": vendor_id,
        }

    def record_consumption(self, task_id: str, tokens: int, green_power=False, vendor_id="") -> EnergyRecord:
        """记录能耗"""
        calc = self.calculate_energy(tokens, green_power, vendor_id)
        record = EnergyRecord(
            record_id=f"energy_{int(datetime.now().timestamp())}_{task_id}",
            task_id=task_id,
            tokens_consumed=tokens,
            energy_kwh=calc["energy_kwh"],
            carbon_kg=calc["carbon_kg"],
            green_power=green_power,
            vendor_id=vendor_id,
            timestamp=datetime.now().isoformat(),
        )
        self.records.append(record)
        self._save()
        return record

    def is_green_power_hour(self) -> bool:
        """判断是否绿电时段"""
        hour = datetime.now().hour
        return hour in self.config.get("green_power_hours", [])

    def get_energy_summary(self, days=30) -> Dict:
        """能源汇总"""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [r for r in self.records if datetime.fromisoformat(r.timestamp) > cutoff]

        total_energy = sum(r.energy_kwh for r in recent)
        total_carbon = sum(r.carbon_kg for r in recent)
        total_saved = sum(r.energy_kwh * self.config["carbon_per_kwh"] * 0.8 for r in recent if r.green_power)
        green_ratio = len([r for r in recent if r.green_power]) / max(len(recent), 1)

        return {
            "period_days": days,
            "total_tasks": len(recent),
            "total_energy_kwh": round(total_energy, 2),
            "total_carbon_kg": round(total_carbon, 2),
            "carbon_saved_kg": round(total_saved, 2),
            "green_power_ratio": round(green_ratio, 2),
            "avg_energy_per_task": round(total_energy / max(len(recent), 1), 4),
            "esg_compliant": self.config.get("esg_report_enabled", True),
        }

    def generate_esg_report(self, days=30) -> Dict:
        """生成ESG报告"""
        summary = self.get_energy_summary(days)
        return {
            "report_type": "ESG碳足迹",
            "report_period": f"{days}天",
            "generated_at": datetime.now().isoformat(),
            "total_carbon_emission_kg": summary["total_carbon_kg"],
            "carbon_saved_kg": summary["carbon_saved_kg"],
            "green_power_ratio": summary["green_power_ratio"],
            "total_energy_kwh": summary["total_energy_kwh"],
            "total_tasks": summary["total_tasks"],
            "compliance": "符合绿色计算标准",
            "recommendation": "建议继续优化模型推理效率以降低能耗",
        }

    def get_vendor_green_ratio(self, vendor_id: str) -> float:
        """获取供应商绿电比例"""
        for v in VENDOR_CONFIG["vendors"]:
            if v["id"] == vendor_id:
                return v["green_power_ratio"]
        return 0.0


# ============================================================
# CARBON_TRADING [碳交易网关] —— 架构图补全模块
# ============================================================

class CarbonTradingGateway:
    """碳交易网关

    架构图对应: vPOWER层 - 碳交易网关
    - 碳配额管理
    - 碳交易撮合
    - 跨境碳结算
    """

    def __init__(self, config=POWER_CONFIG):
        self.config = config
        self.carbon_credits: float = 0.0  # 碳信用余额
        self.trades: List[Dict] = []
        self.trade_file = os.path.join(DATA_DIR, "carbon_trades.json")
        self._load()

    def _load(self):
        if os.path.exists(self.trade_file):
            with open(self.trade_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.carbon_credits = data.get("carbon_credits", 0.0)
            self.trades = data.get("trades", [])

    def _save(self):
        with open(self.trade_file, 'w', encoding='utf-8') as f:
            json.dump({
                "carbon_credits": self.carbon_credits,
                "trades": self.trades[-1000:],
            }, f, ensure_ascii=False, indent=2)

    def accumulate_credits(self, carbon_saved_kg: float) -> float:
        """积累碳信用(碳减排量转碳信用)"""
        # 1kg CO2 = 0.001 碳信用
        credits = carbon_saved_kg * 0.001
        self.carbon_credits += credits
        self._save()
        return round(self.carbon_credits, 4)

    def trade_credits(self, amount: float, trade_type: str = "sell") -> Dict:
        """交易碳信用"""
        if trade_type == "sell":
            if amount > self.carbon_credits:
                return {"success": False, "error": "碳信用不足"}
            self.carbon_credits -= amount
            price_per_credit = 50.0  # 50元/碳信用
            revenue = amount * price_per_credit
        else:  # buy
            self.carbon_credits += amount
            price_per_credit = 52.0  # 买入稍贵
            revenue = -(amount * price_per_credit)

        trade = {
            "trade_id": f"carbon_{int(time.time())}",
            "type": trade_type,
            "amount": amount,
            "price_per_credit": price_per_credit,
            "total": round(revenue, 2),
            "timestamp": datetime.now().isoformat(),
        }
        self.trades.append(trade)
        self._save()
        return {"success": True, "trade": trade, "remaining_credits": round(self.carbon_credits, 4)}

    def get_summary(self) -> Dict:
        return {
            "carbon_credits": round(self.carbon_credits, 4),
            "total_trades": len(self.trades),
            "total_revenue": round(sum(t["total"] for t in self.trades if t["type"] == "sell"), 2),
            "total_spent": round(sum(abs(t["total"]) for t in self.trades if t["type"] == "buy"), 2),
        }


# ============================================================
# VENDOR_SCHEDULER [供应商调度器]
# ============================================================

@dataclass
class VendorStatus:
    """供应商状态"""
    vendor_id: str
    name: str
    gpu_type: str
    available_slots: int      # 可用槽位
    current_price: float      # 当前价格
    speed_factor: float       # 速度系数
    reliability: float        # 可靠性
    green_power_ratio: float  # 绿电比例
    is_healthy: bool          # 健康状态
    last_check: str           # 最后检查


@dataclass
class ComputeTask:
    """计算任务"""
    task_id: str
    task_type: str            # training / inference / evaluation
    gpu_hours: int            # 所需GPU时
    data_size_gb: float       # 数据量
    priority: str = "normal"  # urgent / normal / economy
    green_power: bool = False # 是否要求绿电
    assigned_vendor: str = ""
    status: str = "pending"   # pending / running / completed / failed
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    result: Dict = None


class VendorScheduler:
    """供应商调度器"""

    def __init__(self, config=VENDOR_CONFIG):
        self.config = config
        self.vendors: Dict[str, VendorStatus] = {}
        self.tasks: Dict[str, ComputeTask] = {}
        self.tasks_file = os.path.join(DATA_DIR, "compute_tasks.json")
        self._init_vendors()
        self._load_tasks()

    def _init_vendors(self):
        """初始化厂商"""
        for v in self.config["vendors"]:
            self.vendors[v["id"]] = VendorStatus(
                vendor_id=v["id"],
                name=v["name"],
                gpu_type=v["gpu_type"],
                available_slots=v["max_concurrent"],
                current_price=v["price_multiplier"],
                speed_factor=v["speed_factor"],
                reliability=v["reliability"],
                green_power_ratio=v["green_power_ratio"],
                is_healthy=True,
                last_check=datetime.now().isoformat(),
            )

    def _load_tasks(self):
        if os.path.exists(self.tasks_file):
            with open(self.tasks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for d in data.get("tasks", []):
                task = ComputeTask(**d)
                self.tasks[task.task_id] = task

    def _save_tasks(self):
        data = {
            "tasks": [asdict(t) for t in self.tasks.values()],
        }
        with open(self.tasks_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def refresh_vendor_status(self):
        """刷新厂商状态"""
        for vid, status in self.vendors.items():
            base = next(v["max_concurrent"] for v in self.config["vendors"] if v["id"] == vid)
            status.available_slots = random.randint(int(base * 0.3), base)
            status.is_healthy = random.random() < status.reliability
            status.last_check = datetime.now().isoformat()

    def select_vendor(self, task: ComputeTask) -> str:
        """
        选择厂商
        cost_optimal: 成本最优
        speed_optimal: 速度最优
        balanced: 均衡
        """
        strategy = self.config["selection_strategy"]
        candidates = [
            v for v in self.vendors.values()
            if v.is_healthy and v.available_slots > 0
        ]

        if not candidates:
            return ""

        if task.green_power:
            candidates.sort(key=lambda v: -v.green_power_ratio)

        if strategy == "cost_optimal" or task.priority == "economy":
            candidates.sort(key=lambda v: (v.current_price, -v.green_power_ratio if task.green_power else 0))
        elif strategy == "speed_optimal" or task.priority == "urgent":
            candidates.sort(key=lambda v: -v.speed_factor)
        else:
            # 均衡策略：综合评分
            candidates.sort(key=lambda v: (-v.speed_factor / v.current_price * v.reliability))

        return candidates[0].vendor_id if candidates else ""

    def submit_task(self, task_type: str, gpu_hours: int, data_size_gb: float = 0,
                    priority: str = "normal", green_power: bool = False) -> ComputeTask:
        """提交计算任务"""
        task_id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        task = ComputeTask(
            task_id=task_id,
            task_type=task_type,
            gpu_hours=gpu_hours,
            data_size_gb=data_size_gb,
            priority=priority,
            green_power=green_power,
            created_at=datetime.now().isoformat(),
        )

        vendor_id = self.select_vendor(task)
        if not vendor_id:
            task.status = "failed"
            task.result = {"error": "无可用厂商"}
            self.tasks[task_id] = task
            self._save_tasks()
            return task

        task.assigned_vendor = vendor_id
        task.status = "running"
        task.started_at = datetime.now().isoformat()

        # 占用槽位
        self.vendors[vendor_id].available_slots -= 1

        self.tasks[task_id] = task
        self._save_tasks()
        return task

    def complete_task(self, task_id: str, result: Dict = None) -> bool:
        """完成任务"""
        if task_id not in self.tasks:
            return False
        task = self.tasks[task_id]
        task.status = "completed"
        task.completed_at = datetime.now().isoformat()
        task.result = result or {"status": "ok"}

        # 释放资源
        if task.assigned_vendor and task.assigned_vendor in self.vendors:
            self.vendors[task.assigned_vendor].available_slots += 1

        self._save_tasks()
        return True

    def failover(self, task_id: str) -> Optional[ComputeTask]:
        """任务故障转移"""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        if task.status != "failed" and task.status != "running":
            return None

        retry_count = 0
        max_retry = self.config["failover_retry_limit"]

        while retry_count < max_retry:
            retry_count += 1

            old_vendor = task.assigned_vendor
            available = [
                v for v in self.vendors.values()
                if v.vendor_id != old_vendor and v.is_healthy and v.available_slots > 0
            ]

            if not available:
                continue

            available.sort(key=lambda x: x.reliability)
            new_vendor = available[0].vendor_id

            task.assigned_vendor = new_vendor
            task.status = "running"
            task.started_at = datetime.now().isoformat()
            self.vendors[new_vendor].available_slots -= 1
            self._save_tasks()
            return task

        task.status = "failed"
        task.result = {"error": f"故障转移失败，已达最大重试次数({max_retry})"}
        self._save_tasks()
        return task

    def get_vendor_comparison(self) -> List[Dict]:
        """获取供应商对比数据"""
        comparison = []
        for vid, status in self.vendors.items():
            comparison.append({
                "vendor_id": vid,
                "name": status.name,
                "gpu_type": status.gpu_type,
                "price_multiplier": status.current_price,
                "speed_factor": status.speed_factor,
                "reliability": status.reliability,
                "available_slots": status.available_slots,
                "green_power_ratio": status.green_power_ratio,
                "is_healthy": status.is_healthy,
                "cost_effectiveness": round(status.speed_factor / status.current_price * status.reliability, 4),
            })

        comparison.sort(key=lambda x: x["cost_effectiveness"], reverse=True)
        return comparison

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        if task_id not in self.tasks:
            return None
        task = self.tasks[task_id]
        vendor = self.vendors.get(task.assigned_vendor, None)
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "assigned_vendor": task.assigned_vendor,
            "vendor_name": vendor.name if vendor else "N/A",
            "gpu_hours": task.gpu_hours,
            "priority": task.priority,
            "green_power": task.green_power,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "result": task.result,
        }

    def get_all_tasks(self, status_filter=None) -> List[Dict]:
        """获取所有任务"""
        tasks = []
        for task in self.tasks.values():
            if status_filter and task.status != status_filter:
                continue
            tasks.append(self.get_task_status(task.task_id))
        return tasks


# ============================================================
# ENTRY POINT [入口]
# ============================================================

def main():
    """主入口函数 - 委托给完整实现"""
    import sys
    # 检查是否是其他模块已经定义了main
    # (part5.py 中定义了完整的 main 函数和测试入口)
    try:
        # 尝试调用已加载的 main 函数
        if 'main' in globals() and callable(globals()['main']) and globals()['main'] is not main:
            return globals()['main']()
        elif 'main' in dir():
            return main.__wrapped__() if hasattr(main, '__wrapped__') else None
    except Exception:
        pass

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("请运行: python part5.py test")
        print("(lingyuan_full.py 仅包含基础设施层, 完整测试入口在 part5.py)")
        return None
    else:
        print("灵元大模型 - 基础设施模块")
        print("完整系统启动请运行: python part5.py")
        return None


if __name__ == "__main__":
    main()
