"""Microgrid coordinator agent.

This module implements a coordinator agent that manages multiple devices
within a single microgrid.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.core.feature import FeatureProvider
from heron.core.action import Action
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig

from case_studies.grid_age.agents.device_agents import (
    ESSFieldAgent,
    DGFieldAgent,
    RESFieldAgent,
)
from case_studies.grid_age.features.basic_features import (
    PowerFeature,
    PriceFeature,
    VoltageFeature,
)


class MicrogridCoordinatorAgent(CoordinatorAgent):
    """Coordinator agent for a microgrid managing multiple devices.

    Manages:
    - ESS (Energy Storage)
    - DG (Distributed Generator)
    - PV (Solar)
    - Wind (Wind turbine)

    The coordinator can either:
    1. Let devices self-direct (each has own policy)
    2. Centrally coordinate via protocol (compute joint action)

    Action (if coordinating): 4D continuous [P_ess, P_dg, Q_pv, Q_wind]
    """

    def __init__(
        self,
        agent_id: str,
        subordinates: Dict[str, FieldAgent],  # REQUIRED: Always pass pre-initialized devices
        # Hierarchy parameters
        upstream_id: Optional[str] = None,
        env_id: Optional[str] = None,
        # Timing config
        tick_config: Optional[TickConfig] = None,
        # Coordinator policy (for centralized mode)
        policy: Optional[Policy] = None,
        # Protocol (for centralized coordination)
        protocol: Optional[Protocol] = None,
    ):
        """Initialize microgrid coordinator.

        Args:
            agent_id: Microgrid ID (e.g., "MG1")
            subordinates: Pre-initialized device agents (REQUIRED)
                         Must include ESS, DG, and RES agents
            upstream_id: Parent agent ID
            env_id: Environment ID
            tick_config: Timing configuration
            policy: Coordinator policy (for centralized control)
            protocol: Coordination protocol

        Example:
            devices = {
                "MG1_ESS": ESSFieldAgent(...),
                "MG1_DG": DGFieldAgent(...),
                "MG1_PV": RESFieldAgent(...),
                "MG1_Wind": RESFieldAgent(...),
            }
            coordinator = MicrogridCoordinatorAgent(
                agent_id="MG1",
                subordinates=devices
            )
        """
        # Store parameters
        self.dt = 1.0

        if subordinates is None or len(subordinates) == 0:
            raise ValueError(
                f"MicrogridCoordinatorAgent requires subordinates. "
                f"Create device agents externally and pass as subordinates dict."
            )

        # Coordinator-level features (grid connection, voltage)
        features = [
            PowerFeature(P=0.0, Q=0.0),  # Net power exchange with grid
            PriceFeature(price=50.0),
            VoltageFeature(voltage=1.0),
        ]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            subordinates=subordinates,  # Use provided subordinates directly
            env_id=env_id,
            tick_config=tick_config,
            policy=policy,
            protocol=protocol,
        )

    def set_state(self, price: Optional[float] = None, voltage: Optional[float] = None, P: Optional[float] = None, Q: Optional[float] = None, **kwargs) -> None:
        """Update coordinator state.

        Args:
            price: Electricity price ($/MWh)
            voltage: Voltage at coordination point (pu)
            P: Net power exchange with grid (MW)
            Q: Net reactive power exchange (MVAr)
            **kwargs: Additional state parameters
        """
        if price is not None:
            self.state.features["PriceFeature"].set_values(price=price)

        if voltage is not None:
            self.state.features["VoltageFeature"].set_values(voltage=voltage)

        if P is not None or Q is not None:
            update_dict = {}
            if P is not None:
                update_dict['P'] = P
            if Q is not None:
                update_dict['Q'] = Q
            self.state.features["PowerFeature"].set_values(**update_dict)

    def set_grid_price(self, price: float) -> None:
        """Set electricity price.

        Args:
            price: Price in $/MWh
        """
        self.set_state(price=price)

    def set_renewable_availability(self, pv_avail: float, wind_avail: float) -> None:
        """Set renewable availability for subordinate devices.

        Note: This updates subordinate agents' states, not the coordinator's own state.

        Args:
            pv_avail: PV availability [0, 1]
            wind_avail: Wind availability [0, 1]
        """
        # Update subordinate RES agents (they will use their own set_state)
        pv_agent = self.subordinates.get(f"{self.agent_id}_PV")
        wind_agent = self.subordinates.get(f"{self.agent_id}_Wind")

        if pv_agent:
            pv_agent.set_availability(pv_avail)
        if wind_agent:
            wind_agent.set_availability(wind_avail)

    def compute_grid_cost(self) -> float:
        """Compute cost of grid energy exchange.

        Returns:
            Grid cost in $
        """
        power_feature = self.state.features["PowerFeature"]
        price_feature = self.state.features["PriceFeature"]

        if power_feature.P > 0:  # Buying
            return power_feature.P * price_feature.price * self.dt
        else:  # Selling
            return power_feature.P * price_feature.price * price_feature.sell_discount * self.dt

    def __repr__(self) -> str:
        return (f"MicrogridCoordinatorAgent(id={self.agent_id}, "
                f"devices={len(self.subordinates)}, "
                f"protocol={self.protocol.__class__.__name__ if self.protocol else 'None'})")
