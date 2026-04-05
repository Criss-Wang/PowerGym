"""Shared state objects for the EV public charging environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


"""Shared state objects for the EV public charging environment."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

@dataclass
class ChargerState:
    """物理桩的内部模拟状态"""
    p_max_kw: float = 150.0
    occupied_or_not: int = 0
    step_energy_delivered_kwh: float = 0.0
    ev: Optional[Any] = None  # 存储具体的 EV 对象

@dataclass
class EnvState:
    """环境总揽状态"""
    time_s: float = 0.0
    dt: float = 300.0
    lmp: float = 0.25  # 实时电价
    charger_states: Dict[str, ChargerState] = field(default_factory=dict)
