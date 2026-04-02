"""Common data structures for EV charging environment simulation."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# common.py
@dataclass
class ChargerState:
    p_max_kw: float = 150.0
    occupied_or_not: int = 0
    ev: Optional[Any] = None  # Internal reference only - DO NOT serialize to global_state!
    last_received_price: float = 0.25
    p_kw: float = 0.0
    revenue: float = 0.0

@dataclass
class EnvState:
    """Simulation state exchanged between global_state ↔ env ↔ run_simulation."""
    charger_states: Dict[str, ChargerState] = field(default_factory=dict)
    station_prices: Dict[str, float] = field(default_factory=dict)
    # Map charger_agent_id -> station_id for reverse lookup
    charger_agent_to_station: Dict[str, str] = field(default_factory=dict)
    # Market info
    lmp: float = 0.20
    time_s: float = 0.0
    dt: float = 300.0
    new_arrivals: int = 0
    # Station aggregated metrics
    station_power: Dict[str, float] = field(default_factory=dict)  # Current power output per station (kW)
    station_capacity: Dict[str, float] = field(default_factory=dict)  # Total capacity per station (kW)
    station_revenue: Dict[str, float] = field(default_factory=dict)  # Cumulative revenue per station
