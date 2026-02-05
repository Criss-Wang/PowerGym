# ev_charging/entities/charging_station.py
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from ..config import StationConfig
from .ev import EV


@dataclass
class ChargingStation:
    station_id: int
    name: str

    dt: int
    action_period: int
    cfg: StationConfig

    # mutable prices (set by env / action)
    charging_price: float = 0.25       # $/kWh
    electricity_price: float = 0.25    # $/kWh (LMP)
    parking_fee: float = 3.0           # $/h

    ts: int = 0
    busy_list: List[EV] = field(default_factory=list)

    energy_delivered: float = 0.0
    energy_consumption: float = 0.0
    revenue: float = 0.0
    cost: float = 0.0
    profit: float = 0.0

    profit_acc: float = 0.0
    revenue_acc: float = 0.0
    cost_acc: float = 0.0
    num_arrive_acc: int = 0

    num_arrive_window: deque = field(default_factory=deque)
    occupy_r_window: deque = field(default_factory=deque)
    profit_window: deque = field(default_factory=deque)
    cost_window: deque = field(default_factory=deque)

    def __post_init__(self):
        window_len = int(self.action_period / self.dt)
        self.num_arrive_window = deque(maxlen=window_len)
        self.occupy_r_window = deque(maxlen=window_len)
        self.profit_window = deque(maxlen=window_len)
        self.cost_window = deque(maxlen=window_len)

        # init from cfg
        self.parking_fee = float(getattr(self.cfg, "parking_fee", self.parking_fee))
        self.charging_price = float(getattr(self.cfg, "init_price", self.charging_price))

    @property
    def num_chargers(self) -> int:
        return int(self.cfg.num_chargers)

    @property
    def charging_power(self) -> float:
        return float(self.cfg.charging_power)

    def set_prices(self, charging_price: float, electricity_price: float) -> None:
        self.charging_price = float(charging_price)
        self.electricity_price = float(electricity_price)

    def leaving(self) -> None:
        leave_idxs = []
        for i, ev in enumerate(self.busy_list):
            if ev.get_demand() <= 0:
                leave_idxs.append(i)
        for i in reversed(leave_idxs):
            del self.busy_list[i]

    def arrive(self, ev_list: List[EV]) -> None:
        self.busy_list += list(ev_list)
        self.num_arrive_acc += len(ev_list)
        self.num_arrive_window.append(len(ev_list))
        self.occupy_r_window.append(len(self.busy_list) / max(self.num_chargers, 1))

    def step(self) -> Dict[str, float]:
        # reset step totals
        self.energy_delivered = 0.0
        self.revenue = 0.0

        for ev in self.busy_list:
            ev.price = self.charging_price
            ev.parking_fee = self.parking_fee
            e, pay = ev.charge_and_pay(self.charging_power, self.dt)
            self.energy_delivered += e
            self.revenue += pay

        # cost & profit (same structure as old)
        eff = float(getattr(self.cfg, "charging_efficiency", 0.9))
        cost_kwh = float(getattr(self.cfg, "cost_kwh", 0.01))
        cost_hour = float(getattr(self.cfg, "cost_hour", (1000 / 30 / 24)))

        self.energy_consumption = self.energy_delivered / max(eff, 1e-9)
        self.cost = self.energy_consumption * (self.electricity_price + cost_kwh) + (self.dt / 3600.0) * cost_hour
        self.profit = self.revenue - self.cost

        self.cost_window.append(self.cost)
        self.profit_window.append(self.profit)

        self.revenue_acc += self.revenue
        self.cost_acc += self.cost
        self.profit_acc += self.profit

        self.ts += self.dt
        return {f"{self.name}/power": self.energy_consumption / (self.dt / 3600.0)}
