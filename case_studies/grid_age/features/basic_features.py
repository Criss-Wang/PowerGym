"""Basic feature providers for device properties.

This module defines low-level feature providers for device attributes:
- Power (active and reactive)
- State of Charge (SOC)
- Capacity
- Unit Commitment (on/off status)
- Pricing
- Network state
"""

from typing import Any
import numpy as np
from heron.core.feature import FeatureProvider


class PowerFeature(FeatureProvider):
    """Active and reactive power feature.

    Used by all devices that inject/consume power.
    """
    visibility = ["public"]

    P: float = 0.0  # Active power (MW)
    Q: float = 0.0  # Reactive power (MVAr)

    # Power limits
    min_p: float = -1.0
    max_p: float = 1.0
    min_q: float = -1.0
    max_q: float = 1.0

    def set_values(self, **kwargs: Any) -> None:
        """Update power values with constraints."""
        if "P" in kwargs:
            self.P = np.clip(kwargs["P"], self.min_p, self.max_p)
        if "Q" in kwargs:
            self.Q = np.clip(kwargs["Q"], self.min_q, self.max_q)


class SOCFeature(FeatureProvider):
    """State of Charge feature for energy storage devices."""
    visibility = ["public"]

    soc: float = 0.5  # State of charge [0, 1]
    capacity: float = 1.0  # Energy capacity (MWh)

    # SOC limits
    min_soc: float = 0.1
    max_soc: float = 0.9

    # Efficiency
    ch_eff: float = 0.95   # Charging efficiency
    dsc_eff: float = 0.95  # Discharging efficiency

    def set_values(self, **kwargs: Any) -> None:
        """Update SOC values with constraints."""
        if "soc" in kwargs:
            self.soc = np.clip(kwargs["soc"], 0.0, 1.0)
        if "capacity" in kwargs:
            self.capacity = max(0.0, kwargs["capacity"])

    def update_soc(self, power: float, dt: float) -> None:
        """Update SOC based on power and time step.

        Args:
            power: Power in MW (positive = charge, negative = discharge)
            dt: Time step in hours
        """
        if power > 0:  # Charging
            delta_soc = power * self.ch_eff * dt / self.capacity
        else:  # Discharging
            delta_soc = power / self.dsc_eff * dt / self.capacity

        self.soc = np.clip(self.soc + delta_soc, 0.0, 1.0)


class UnitCommitmentFeature(FeatureProvider):
    """Unit commitment status feature for generators."""
    visibility = ["public"]

    on: int = 1  # 0=off, 1=on

    # Startup/shutdown dynamics
    startup_time: int = 0
    shutdown_time: int = 0
    starting: int = 0
    shutting: int = 0

    def set_values(self, **kwargs: Any) -> None:
        """Update unit commitment state."""
        if "on" in kwargs:
            self.on = int(kwargs["on"])


class AvailabilityFeature(FeatureProvider):
    """Availability feature for renewable energy sources."""
    visibility = ["public"]

    availability: float = 1.0  # Fraction [0, 1]
    max_capacity: float = 1.0  # MW

    def set_values(self, **kwargs: Any) -> None:
        """Update availability."""
        if "availability" in kwargs:
            self.availability = np.clip(kwargs["availability"], 0.0, 1.0)
        if "max_capacity" in kwargs:
            self.max_capacity = max(0.0, kwargs["max_capacity"])

    def get_available_power(self) -> float:
        """Get available power based on current availability.

        Returns:
            Available power in MW
        """
        return self.availability * self.max_capacity


class PriceFeature(FeatureProvider):
    """Electricity price feature."""
    visibility = ["public"]

    price: float = 50.0  # $/MWh
    sell_discount: float = 0.9  # Multiplier for selling

    def set_values(self, **kwargs: Any) -> None:
        """Update price."""
        if "price" in kwargs:
            self.price = max(0.0, kwargs["price"])
        if "sell_discount" in kwargs:
            self.sell_discount = np.clip(kwargs["sell_discount"], 0.0, 1.0)


class CostFeature(FeatureProvider):
    """Cost accumulation feature for devices."""
    visibility = ["owner"]  # Private to agent

    total_cost: float = 0.0  # Accumulated cost ($)

    # Cost coefficients (for fuel cost: a*P^2 + b*P + c)
    cost_a: float = 0.0
    cost_b: float = 0.0
    cost_c: float = 0.0

    def set_values(self, **kwargs: Any) -> None:
        """Update cost."""
        if "total_cost" in kwargs:
            self.total_cost = kwargs["total_cost"]

    def compute_cost(self, power: float, dt: float) -> float:
        """Compute operational cost.

        Args:
            power: Power output (MW)
            dt: Time step (hours)

        Returns:
            Cost in $
        """
        cost_per_hour = self.cost_a * power**2 + self.cost_b * power + self.cost_c
        return cost_per_hour * dt

    def add_cost(self, cost: float) -> None:
        """Add cost to total."""
        self.total_cost += cost


class VoltageFeature(FeatureProvider):
    """Voltage measurement feature for network buses."""
    visibility = ["public"]

    voltage: float = 1.0  # Per-unit voltage
    min_voltage: float = 0.95
    max_voltage: float = 1.05

    def set_values(self, **kwargs: Any) -> None:
        """Update voltage."""
        if "voltage" in kwargs:
            self.voltage = kwargs["voltage"]

    def get_violation(self) -> float:
        """Get voltage violation magnitude.

        Returns:
            Violation magnitude (0 if within limits)
        """
        if self.voltage < self.min_voltage:
            return self.min_voltage - self.voltage
        elif self.voltage > self.max_voltage:
            return self.voltage - self.max_voltage
        return 0.0
