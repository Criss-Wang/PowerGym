"""Device feature providers for GridAges microgrid components.

This module defines feature providers for:
- Energy Storage System (ESS)
- Distributed Generator (DG)
- Renewable Energy Source (RES) - Solar PV and Wind
- Grid Connection
"""

from typing import Any
import numpy as np
from heron.core.feature import FeatureProvider


class ESSFeature(FeatureProvider):
    """Energy Storage System feature with SOC dynamics.

    Models a battery energy storage system with:
    - Active/reactive power control
    - State of charge (SOC) tracking
    - Charge/discharge efficiency
    - Capacity constraints
    """
    visibility = ["public"]

    # Power (MW, MVAr)
    P: float = 0.0
    Q: float = 0.0

    # State of charge
    soc: float = 0.5  # Fraction [0, 1]

    # Capacity and constraints
    capacity: float = 2.0  # MWh
    min_p: float = -0.5   # MW (discharge)
    max_p: float = 0.5    # MW (charge)
    min_soc: float = 0.2  # Minimum SOC
    max_soc: float = 0.9  # Maximum SOC

    # Efficiency
    ch_eff: float = 0.95   # Charging efficiency
    dsc_eff: float = 0.95  # Discharging efficiency

    def set_values(self, **kwargs: Any) -> None:
        """Update ESS state values with constraints."""
        if "P" in kwargs:
            self.P = np.clip(kwargs["P"], self.min_p, self.max_p)
        if "Q" in kwargs:
            self.Q = kwargs["Q"]
        if "soc" in kwargs:
            self.soc = np.clip(kwargs["soc"], 0.0, 1.0)
        if "capacity" in kwargs:
            self.capacity = kwargs["capacity"]

    def update_soc(self, dt: float) -> None:
        """Update SOC based on power flow and time step.

        Args:
            dt: Time step in hours
        """
        if self.P > 0:  # Charging
            delta_soc = self.P * self.ch_eff * dt / self.capacity
        else:  # Discharging
            delta_soc = self.P / self.dsc_eff * dt / self.capacity

        self.soc = np.clip(self.soc + delta_soc, 0.0, 1.0)

    def get_feasible_power_range(self) -> tuple[float, float]:
        """Get feasible power range based on current SOC.

        Returns:
            (min_power, max_power) tuple in MW
        """
        # Maximum discharge limited by available energy above min_soc
        available_energy = (self.soc - self.min_soc) * self.capacity
        max_discharge = -min(abs(self.min_p), available_energy / (0.001 / self.dsc_eff))  # Negative

        # Maximum charge limited by spare capacity below max_soc
        spare_capacity = (self.max_soc - self.soc) * self.capacity
        max_charge = min(self.max_p, spare_capacity / (0.001 * self.ch_eff))

        return (max_discharge, max_charge)


class DGFeature(FeatureProvider):
    """Distributed Generator (diesel/gas) feature.

    Models a controllable generator with:
    - Active/reactive power control
    - Unit commitment (on/off)
    - Power limits and ramp rates
    - Startup/shutdown dynamics (optional)
    """
    visibility = ["public"]

    # Power (MW, MVAr)
    P: float = 0.0
    Q: float = 0.0

    # Unit commitment
    on: int = 1  # 0=off, 1=on

    # Power limits
    max_p: float = 0.66  # MW
    min_p: float = 0.1   # MW (minimum stable generation)
    max_q: float = 0.33  # MVAr
    min_q: float = -0.33

    # Unit commitment dynamics (optional)
    startup_time: int = 0   # Steps required to start
    shutdown_time: int = 0  # Steps required to shut down
    starting: int = 0       # Current startup counter
    shutting: int = 0       # Current shutdown counter

    # Fuel cost coefficients (quadratic: a*P^2 + b*P + c)
    fuel_cost_a: float = 10.0
    fuel_cost_b: float = 5.0
    fuel_cost_c: float = 1.0

    def set_values(self, **kwargs: Any) -> None:
        """Update DG state values with constraints."""
        if "P" in kwargs:
            if self.on:
                self.P = np.clip(kwargs["P"], self.min_p, self.max_p)
            else:
                self.P = 0.0
        if "Q" in kwargs:
            self.Q = np.clip(kwargs["Q"], self.min_q, self.max_q)
        if "on" in kwargs:
            self.on = int(kwargs["on"])

    def compute_fuel_cost(self, dt: float) -> float:
        """Compute fuel cost for current power output.

        Args:
            dt: Time step in hours

        Returns:
            Fuel cost in $
        """
        if not self.on or self.P <= 0:
            return 0.0

        # Quadratic cost function
        cost_per_hour = (self.fuel_cost_a * self.P ** 2 +
                         self.fuel_cost_b * self.P +
                         self.fuel_cost_c)
        return cost_per_hour * dt


class RESFeature(FeatureProvider):
    """Renewable Energy Source feature (Solar PV or Wind).

    Models non-dispatchable renewable generation with:
    - Active power set externally (based on availability)
    - Reactive power control (for voltage support)
    - Curtailment capability
    """
    visibility = ["public"]

    # Power (MW, MVAr)
    P: float = 0.0  # Set externally based on weather/availability
    Q: float = 0.0  # Controllable for voltage support

    # Limits
    max_p: float = 0.1   # MW (rated capacity)
    max_q: float = 0.05  # MVAr

    # Availability (external input)
    availability: float = 1.0  # Fraction [0, 1]

    # Type (not included in vector - use separate field or metadata)
    # res_type: str = "PV"  # "PV" or "Wind"

    def set_values(self, **kwargs: Any) -> None:
        """Update RES state values with constraints."""
        if "P" in kwargs:
            # P is typically set externally but can be curtailed
            self.P = np.clip(kwargs["P"], 0.0, self.max_p * self.availability)
        if "Q" in kwargs:
            self.Q = np.clip(kwargs["Q"], -self.max_q, self.max_q)
        if "availability" in kwargs:
            self.availability = np.clip(kwargs["availability"], 0.0, 1.0)

    def set_availability(self, availability: float) -> None:
        """Update availability based on weather/time of day.

        Args:
            availability: Fraction of rated capacity available [0, 1]
        """
        self.availability = np.clip(availability, 0.0, 1.0)
        # Cap current output if exceeds new availability
        max_available = self.max_p * self.availability
        if self.P > max_available:
            self.P = max_available


class GridFeature(FeatureProvider):
    """Grid connection point feature (DSO interface).

    Models the connection to the main distribution grid with:
    - Power exchange (buy/sell)
    - Dynamic electricity pricing
    - Convention: P > 0 means buying from grid, P < 0 means selling
    """
    visibility = ["public"]

    # Power exchange (MW, MVAr)
    P: float = 0.0  # Positive = buy, Negative = sell
    Q: float = 0.0

    # Pricing
    price: float = 50.0  # $/MWh
    sell_discount: float = 0.9  # Multiplier for sell price

    # Capacity (optional constraint)
    max_import: float = 5.0  # MW
    max_export: float = 5.0  # MW

    def set_values(self, **kwargs: Any) -> None:
        """Update grid state values."""
        if "P" in kwargs:
            self.P = np.clip(kwargs["P"], -self.max_export, self.max_import)
        if "Q" in kwargs:
            self.Q = kwargs["Q"]
        if "price" in kwargs:
            self.price = max(0.0, kwargs["price"])

    def compute_energy_cost(self, dt: float) -> float:
        """Compute cost of energy exchange with grid.

        Args:
            dt: Time step in hours

        Returns:
            Energy cost in $ (positive = cost, negative = revenue)
        """
        if self.P > 0:  # Buying from grid
            return self.P * self.price * dt
        else:  # Selling to grid
            return self.P * self.price * self.sell_discount * dt

    def set_price(self, price: float) -> None:
        """Update electricity price.

        Args:
            price: New price in $/MWh
        """
        self.price = max(0.0, price)


class NetworkFeature(FeatureProvider):
    """Network state feature (voltages, line flows).

    Stores power flow results from simulation:
    - Bus voltages
    - Line loading
    - Violations
    """
    visibility = ["owner"]  # Only visible to owning agent

    # Bus voltages (per unit)
    voltage_min: float = 1.0
    voltage_max: float = 1.0
    voltage_avg: float = 1.0

    # Line loading (max percentage)
    max_line_loading: float = 0.0  # Percentage

    # Violations
    voltage_violations: int = 0      # Number of buses outside limits
    overload_violations: int = 0     # Number of overloaded lines

    def set_values(self, **kwargs: Any) -> None:
        """Update network state from power flow results."""
        if "voltage_min" in kwargs:
            self.voltage_min = kwargs["voltage_min"]
        if "voltage_max" in kwargs:
            self.voltage_max = kwargs["voltage_max"]
        if "voltage_avg" in kwargs:
            self.voltage_avg = kwargs["voltage_avg"]
        if "max_line_loading" in kwargs:
            self.max_line_loading = kwargs["max_line_loading"]
        if "voltage_violations" in kwargs:
            self.voltage_violations = int(kwargs["voltage_violations"])
        if "overload_violations" in kwargs:
            self.overload_violations = int(kwargs["overload_violations"])

    def compute_safety_penalty(self, voltage_limit: float = 0.05,
                                loading_limit: float = 1.0) -> float:
        """Compute safety penalty based on violations.

        Args:
            voltage_limit: Allowable voltage deviation (pu)
            loading_limit: Maximum line loading (fraction)

        Returns:
            Safety penalty (higher = worse)
        """
        penalty = 0.0

        # Voltage violations
        if self.voltage_min < (1.0 - voltage_limit):
            penalty += (1.0 - voltage_limit - self.voltage_min) * 10
        if self.voltage_max > (1.0 + voltage_limit):
            penalty += (self.voltage_max - 1.0 - voltage_limit) * 10

        # Line overload violations
        if self.max_line_loading > loading_limit:
            penalty += (self.max_line_loading - loading_limit) * 20

        return penalty
