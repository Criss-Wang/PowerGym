"""Device-level field agents for GridAges microgrids.

This module implements individual field agents for each device type:
- ESSFieldAgent: Energy Storage System
- DGFieldAgent: Distributed Generator
- RESFieldAgent: Renewable Energy Source (PV/Wind)
"""

from typing import Any, List, Optional
import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.core.feature import FeatureProvider
from heron.core.action import Action
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig

from case_studies.grid_age.features.basic_features import (
    PowerFeature,
    SOCFeature,
    UnitCommitmentFeature,
    AvailabilityFeature,
    CostFeature,
)


class ESSFieldAgent(FieldAgent):
    """Energy Storage System field agent.

    Controls a single battery storage unit with:
    - Active power control (charge/discharge)
    - SOC tracking and constraints
    - Cycling cost

    Action: 1D continuous [P_ess] in [-1, 1]
    """

    def __init__(
        self,
        agent_id: str,
        capacity: float = 2.0,
        min_p: float = -0.5,
        max_p: float = 0.5,
        min_soc: float = 0.2,
        max_soc: float = 0.9,
        upstream_id: Optional[str] = None,
        env_id: Optional[str] = None,
        tick_config: Optional[TickConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        self.dt = 1.0  # Time step (hours)
        self.min_p = min_p
        self.max_p = max_p

        features = [
            PowerFeature(min_p=min_p, max_p=max_p, P=0.0, Q=0.0),
            SOCFeature(capacity=capacity, soc=0.5, min_soc=min_soc, max_soc=max_soc),
            CostFeature(cost_a=0.0, cost_b=0.1, cost_c=0.0),  # Cycling cost
        ]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config,
            policy=policy,
            protocol=protocol,
        )

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """1D action: [P_ess]"""
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(c=np.zeros(1))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Set action."""
        if isinstance(action, Action):
            self.action.set_values(c=action.c)
        elif isinstance(action, np.ndarray):
            self.action.set_values(c=action)
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def set_state(self, P: Optional[float] = None, **kwargs) -> None:
        """Update state based on power setpoint and SOC dynamics.

        Args:
            P: Active power setpoint (MW). If None, computed from current action.
            **kwargs: Additional state parameters
        """
        power_feature = self.state.features["PowerFeature"]
        soc_feature = self.state.features["SOCFeature"]

        if P is None:
            # Compute P from action if not provided
            # Denormalize action from [-1, 1] to [min_p, max_p]
            # Get feasible range based on SOC
            available_energy = (soc_feature.soc - soc_feature.min_soc) * soc_feature.capacity
            max_discharge = -min(abs(self.min_p), available_energy / (0.001 / soc_feature.dsc_eff))

            spare_capacity = (soc_feature.max_soc - soc_feature.soc) * soc_feature.capacity
            max_charge = min(self.max_p, spare_capacity / (0.001 * soc_feature.ch_eff))

            # Denormalize
            action_val = self.action.c[0]
            if action_val > 0:
                P = action_val * max_charge
            else:
                P = action_val * abs(max_discharge)

        # Update power
        power_feature.set_values(P=P)

        # Update SOC
        soc_feature.update_soc(power=P, dt=self.dt)

    def apply_action(self):
        """Apply action to update state."""
        self.set_state()

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward based on cycling cost and SOC violations."""
        power_vec = local_state.get("PowerFeature", np.zeros(6))
        soc_vec = local_state.get("SOCFeature", np.zeros(6))

        power = float(power_vec[0])
        soc = float(soc_vec[0])

        # Cycling cost
        cost = abs(power) * 0.1 * self.dt

        # SOC violations
        soc_feature = self.state.features["SOCFeature"]
        safety_penalty = 0.0
        if soc < soc_feature.min_soc:
            safety_penalty = (soc_feature.min_soc - soc) * 100
        elif soc > soc_feature.max_soc:
            safety_penalty = (soc - soc_feature.max_soc) * 100

        return -(cost + 10.0 * safety_penalty)


class DGFieldAgent(FieldAgent):
    """Distributed Generator field agent.

    Controls a diesel/gas generator with:
    - Active power control
    - Unit commitment
    - Fuel cost

    Action: 1D continuous [P_dg] in [-1, 1]
    """

    def __init__(
        self,
        agent_id: str,
        max_p: float = 0.66,
        min_p: float = 0.1,
        fuel_cost_a: float = 10.0,
        fuel_cost_b: float = 5.0,
        fuel_cost_c: float = 1.0,
        upstream_id: Optional[str] = None,
        env_id: Optional[str] = None,
        tick_config: Optional[TickConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        self.dt = 1.0
        self.min_p = min_p
        self.max_p = max_p

        features = [
            PowerFeature(min_p=0.0, max_p=max_p, P=min_p, Q=0.0),
            UnitCommitmentFeature(on=1),
            CostFeature(cost_a=fuel_cost_a, cost_b=fuel_cost_b, cost_c=fuel_cost_c),
        ]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config,
            policy=policy,
            protocol=protocol,
        )

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """1D action: [P_dg]"""
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(c=np.array([0.0]))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Set action."""
        if isinstance(action, Action):
            self.action.set_values(c=action.c)
        elif isinstance(action, np.ndarray):
            self.action.set_values(c=action)
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def set_state(self, P: Optional[float] = None, **kwargs) -> None:
        """Update state based on power setpoint and unit commitment.

        Args:
            P: Active power setpoint (MW). If None, computed from current action.
            **kwargs: Additional state parameters
        """
        power_feature = self.state.features["PowerFeature"]
        uc_feature = self.state.features["UnitCommitmentFeature"]

        if P is None:
            # Compute P from action if not provided
            if uc_feature.on:
                # Denormalize from [-1, 1] to [min_p, max_p]
                P = self.min_p + (self.action.c[0] + 1) / 2 * (self.max_p - self.min_p)
            else:
                P = 0.0

        # Update power
        power_feature.set_values(P=P)

    def apply_action(self):
        """Apply action to update state."""
        self.set_state()

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward based on fuel cost."""
        power_vec = local_state.get("PowerFeature", np.zeros(6))
        uc_vec = local_state.get("UnitCommitmentFeature", np.zeros(4))

        power = float(power_vec[0])
        on = int(uc_vec[0])

        # Fuel cost
        if on and power > 0:
            cost_feature = self.state.features["CostFeature"]
            cost = cost_feature.compute_cost(power, self.dt)
        else:
            cost = 0.0

        return -cost


class RESFieldAgent(FieldAgent):
    """Renewable Energy Source field agent.

    Controls reactive power for voltage support.
    Active power is set externally based on availability.

    Action: 1D continuous [Q_res] in [-1, 1]
    """

    def __init__(
        self,
        agent_id: str,
        max_p: float = 0.1,
        max_q: float = 0.05,
        res_type: str = "PV",
        upstream_id: Optional[str] = None,
        env_id: Optional[str] = None,
        tick_config: Optional[TickConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        self.max_q = max_q
        self.res_type = res_type

        features = [
            PowerFeature(min_p=0.0, max_p=max_p, min_q=-max_q, max_q=max_q, P=0.0, Q=0.0),
            AvailabilityFeature(max_capacity=max_p, availability=1.0),
        ]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config,
            policy=policy,
            protocol=protocol,
        )

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """1D action: [Q_res]"""
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(c=np.zeros(1))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Set action."""
        if isinstance(action, Action):
            self.action.set_values(c=action.c)
        elif isinstance(action, np.ndarray):
            self.action.set_values(c=action)
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def set_state(self, Q: Optional[float] = None, availability: Optional[float] = None, P: Optional[float] = None, **kwargs) -> None:
        """Update state based on reactive power and availability.

        Args:
            Q: Reactive power setpoint (MVAr). If None, computed from current action.
            availability: Renewable availability [0, 1]. If provided, updates P accordingly.
            P: Active power setpoint (MW). If None and availability provided, computed from availability.
            **kwargs: Additional state parameters
        """
        power_feature = self.state.features["PowerFeature"]
        avail_feature = self.state.features["AvailabilityFeature"]

        # Update reactive power
        if Q is None:
            # Compute Q from action if not provided
            Q = self.action.c[0] * self.max_q
        power_feature.set_values(Q=Q)

        # Update availability and active power if provided
        if availability is not None:
            avail_feature.set_values(availability=availability)
            if P is None:
                P = avail_feature.get_available_power()
            power_feature.set_values(P=P)

    def apply_action(self):
        """Apply action to update state."""
        self.set_state()

    def set_availability(self, availability: float) -> None:
        """Set renewable availability and update active power.

        Args:
            availability: Fraction [0, 1]
        """
        self.set_state(availability=availability)

    def compute_local_reward(self, local_state: dict) -> float:
        """Renewable energy has minimal cost."""
        return 0.0  # No fuel cost
