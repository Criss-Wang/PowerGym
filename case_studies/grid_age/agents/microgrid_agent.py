"""Microgrid field agent for GridAges case study.

This module implements a composite field agent that controls multiple devices
within a single microgrid:
- Energy Storage System (ESS)
- Distributed Generator (DG)
- Solar PV
- Wind Turbine
- Grid Connection
"""

from typing import Any, Dict, List, Optional
import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.core.feature import FeatureProvider
from heron.core.state import FieldAgentState
from heron.core.action import Action
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig


class MicrogridFieldAgent(FieldAgent):
    """Composite field agent controlling a microgrid with multiple devices.

    The agent controls:
    - ESS: Active power (charge/discharge)
    - DG: Active power setpoint
    - PV: Reactive power (active power set by availability)
    - Wind: Reactive power (active power set by availability)

    Action space: 4D continuous
        [P_ess, P_dg, Q_pv, Q_wind] normalized to [-1, 1]

    Observation space: ~15-20 dimensions
        - Device states (P, Q, SOC, on, etc.)
        - Network state (voltages, line loading)
        - Grid state (price)
    """

    def __init__(
        self,
        agent_id: str,
        features: List[FeatureProvider],  # REQUIRED: Pass pre-initialized features
        # Hierarchy parameters
        upstream_id: Optional[str] = None,
        env_id: Optional[str] = None,
        # Timing config
        tick_config: Optional[TickConfig] = None,
        # Execution parameters
        policy: Optional[Policy] = None,
        # Coordination parameters
        protocol: Optional[Protocol] = None,
    ):
        """Initialize microgrid field agent.

        IMPORTANT: Features must be created externally before passing to agent.

        Args:
            agent_id: Unique agent identifier (e.g., "MG1", "MG2")
            features: Pre-initialized feature providers (REQUIRED)
                     Expected: [ESSFeature, DGFeature, RESFeature(PV), RESFeature(Wind), GridFeature, NetworkFeature]
            upstream_id: Parent agent ID (if any)
            env_id: Environment ID
            tick_config: Timing configuration for event-driven mode
            policy: Policy for action selection
            protocol: Coordination protocol (if parent-controlled)

        Example:
            features = [
                ESSFeature(capacity=2.0, min_p=-0.5, max_p=0.5, soc=0.5),
                DGFeature(max_p=0.66, min_p=0.1, on=1),
                RESFeature(max_p=0.1),  # PV
                RESFeature(max_p=0.1),  # Wind
                GridFeature(),
                NetworkFeature(),
            ]
            agent = MicrogridFieldAgent(agent_id="MG1", features=features)
        """
        if not features or len(features) < 6:
            raise ValueError(
                f"MicrogridFieldAgent requires 6 features "
                f"(ESS, DG, PV, Wind, Grid, Network). Got {len(features) if features else 0}"
            )

        # Extract denormalization parameters from features
        # These are used in apply_action() to convert normalized actions to physical units
        ess_feature = features[0]  # Assume first is ESS
        dg_feature = features[1]   # Assume second is DG
        pv_feature = features[2]   # Assume third is PV
        wind_feature = features[3] # Assume fourth is Wind

        self.ess_min_p = getattr(ess_feature, 'min_p', -0.5)
        self.ess_max_p = getattr(ess_feature, 'max_p', 0.5)
        self.dg_min_p = getattr(dg_feature, 'min_p', 0.1)
        self.dg_max_p = getattr(dg_feature, 'max_p', 0.66)
        self.pv_max_q = getattr(pv_feature, 'max_q', 0.05)
        self.wind_max_q = getattr(wind_feature, 'max_q', 0.05)

        # Time step for dynamics (1 hour default)
        self.dt = 1.0  # hours

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config,
            policy=policy,
            protocol=protocol,
        )

        # Store feature indices for easy access
        self.ess_idx = 0
        self.dg_idx = 1
        self.pv_idx = 2
        self.wind_idx = 3
        self.grid_idx = 4
        self.network_idx = 5

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """Initialize action space.

        Action space: 4D continuous in [-1, 1]
            [0]: P_ess - ESS power (charge/discharge)
            [1]: P_dg - DG power setpoint
            [2]: Q_pv - PV reactive power
            [3]: Q_wind - Wind reactive power

        Returns:
            Action object with 4D continuous space
        """
        action = Action()
        action.set_specs(
            dim_c=4,
            range=(
                np.array([-1.0, -1.0, -1.0, -1.0]),
                np.array([1.0, 1.0, 1.0, 1.0])
            )
        )
        action.set_values(c=np.zeros(4))
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Set action from policy output or upstream.

        Args:
            action: Action object or numpy array
        """
        if isinstance(action, Action):
            self.action.set_values(c=action.c)
        elif isinstance(action, np.ndarray):
            self.action.set_values(c=action)
        elif isinstance(action, dict) and 'c' in action:
            self.action.set_values(c=action['c'])
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def set_state(
        self,
        P_ess: Optional[float] = None,
        P_dg: Optional[float] = None,
        Q_pv: Optional[float] = None,
        Q_wind: Optional[float] = None,
        voltage_min: Optional[float] = None,
        voltage_max: Optional[float] = None,
        voltage_avg: Optional[float] = None,
        max_line_loading: Optional[float] = None,
        voltage_violations: Optional[int] = None,
        overload_violations: Optional[int] = None,
        pv_availability: Optional[float] = None,
        wind_availability: Optional[float] = None,
        grid_price: Optional[float] = None,
        **kwargs
    ) -> None:
        """Update state for all device features and network conditions.

        This is the ONLY method that should update agent state. All other methods
        (apply_action, update_device_dynamics, set_renewable_availability, etc.)
        should call this method.

        Args:
            P_ess: ESS active power (MW). If None, computed from action.
            P_dg: DG active power (MW). If None, computed from action.
            Q_pv: PV reactive power (MVAr). If None, computed from action.
            Q_wind: Wind reactive power (MVAr). If None, computed from action.
            voltage_min: Minimum bus voltage (pu)
            voltage_max: Maximum bus voltage (pu)
            voltage_avg: Average bus voltage (pu)
            max_line_loading: Maximum line loading percentage
            voltage_violations: Number of voltage violations
            overload_violations: Number of overload violations
            pv_availability: PV availability [0, 1]
            wind_availability: Wind availability [0, 1]
            grid_price: Electricity price ($/MWh)
            **kwargs: Additional state parameters
        """
        # Get device features
        ess_feature = self.state.features[self.ess_idx]
        dg_feature = self.state.features[self.dg_idx]
        pv_feature = self.state.features[self.pv_idx]
        wind_feature = self.state.features[self.wind_idx]
        grid_feature = self.state.features[self.grid_idx]
        network_feature = self.state.features[self.network_idx]

        # Compute power setpoints from actions if not provided
        if P_ess is None or P_dg is None or Q_pv is None or Q_wind is None:
            # Extract normalized actions
            ess_action = self.action.c[0]  # [-1, 1]
            dg_action = self.action.c[1]   # [-1, 1]
            pv_action = self.action.c[2]   # [-1, 1]
            wind_action = self.action.c[3] # [-1, 1]

            if P_ess is None:
                # Denormalize ESS power: [-1, 1] → [min_p, max_p]
                min_p_feasible, max_p_feasible = ess_feature.get_feasible_power_range()
                P_ess = ess_action * (max_p_feasible if ess_action > 0 else abs(min_p_feasible))

            if P_dg is None:
                # Denormalize DG power: [-1, 1] → [min_p, max_p]
                P_dg = self.dg_min_p + (dg_action + 1) / 2 * (self.dg_max_p - self.dg_min_p)

            if Q_pv is None:
                # Denormalize reactive power: [-1, 1] → [-max_q, max_q]
                Q_pv = pv_action * self.pv_max_q

            if Q_wind is None:
                Q_wind = wind_action * self.wind_max_q

        # Update device features with setpoints
        ess_feature.set_values(P=P_ess)
        dg_feature.set_values(P=P_dg)
        pv_feature.set_values(Q=Q_pv)
        wind_feature.set_values(Q=Q_wind)

        # Update ESS SOC dynamics if needed (called after power flow)
        # Note: This is part of state update, not a separate operation
        if hasattr(ess_feature, 'update_soc'):
            ess_feature.update_soc(self.dt)

        # Update network state if provided
        if any(x is not None for x in [voltage_min, voltage_max, voltage_avg, max_line_loading, voltage_violations, overload_violations]):
            update_dict = {}
            if voltage_min is not None:
                update_dict['voltage_min'] = voltage_min
            if voltage_max is not None:
                update_dict['voltage_max'] = voltage_max
            if voltage_avg is not None:
                update_dict['voltage_avg'] = voltage_avg
            if max_line_loading is not None:
                update_dict['max_line_loading'] = max_line_loading
            if voltage_violations is not None:
                update_dict['voltage_violations'] = voltage_violations
            if overload_violations is not None:
                update_dict['overload_violations'] = overload_violations
            network_feature.set_values(**update_dict)

        # Update renewable availability if provided
        if pv_availability is not None:
            pv_feature.set_availability(pv_availability)
            pv_feature.set_values(P=pv_feature.max_p * pv_availability)

        if wind_availability is not None:
            wind_feature.set_availability(wind_availability)
            wind_feature.set_values(P=wind_feature.max_p * wind_availability)

        # Update grid price if provided
        if grid_price is not None:
            grid_feature.set_price(grid_price)

    def apply_action(self):
        """Apply normalized actions to device features."""
        self.set_state()

    def compute_local_reward(self, local_state: dict) -> float:
        """Compute reward based on operational cost and safety violations.

        Reward = -(cost + penalty × safety)

        Cost components:
        - ESS cycling cost
        - DG fuel cost
        - Grid energy purchase cost

        Safety components:
        - Voltage violations
        - Line overloading
        - SOC bound violations

        Args:
            local_state: Dict of feature names to numpy arrays (feature vectors)

        Returns:
            Reward value (higher is better)
        """
        # Extract feature vectors (numpy arrays)
        ess_vec = local_state.get("ESSFeature", np.zeros(11))
        dg_vec = local_state.get("DGFeature", np.zeros(13))
        grid_vec = local_state.get("GridFeature", np.zeros(5))
        network_vec = local_state.get("NetworkFeature", np.zeros(6))

        # Parse ESS vector: [P, Q, soc, capacity, min_p, max_p, min_soc, max_soc, ch_eff, dsc_eff]
        ess_power = float(ess_vec[0])
        ess_soc = float(ess_vec[2]) if len(ess_vec) > 2 else 0.5

        # Parse DG vector: [P, Q, on, max_p, min_p, max_q, min_q, startup, shutdown, starting, shutting, fuel_a, fuel_b, fuel_c]
        dg_power = float(dg_vec[0])
        dg_on = int(dg_vec[2]) if len(dg_vec) > 2 else 1

        # Parse Grid vector: [P, Q, price, sell_discount, max_import, max_export]
        grid_power = float(grid_vec[0])
        grid_price = float(grid_vec[2]) if len(grid_vec) > 2 else 50.0

        # Parse Network vector: [voltage_min, voltage_max, voltage_avg, max_line_loading, voltage_violations, overload_violations]
        # Network violations will be extracted from the feature

        # Compute costs
        # ESS cycling cost (degradation)
        ess_cost = abs(ess_power) * 0.1 * self.dt  # $0.1/MWh cycling cost

        # DG fuel cost
        if dg_on and dg_power > 0:
            dg_feature = self.state.features[self.dg_idx]
            dg_cost = dg_feature.compute_fuel_cost(self.dt)
        else:
            dg_cost = 0.0

        # Grid energy cost
        grid_feature = self.state.features[self.grid_idx]
        grid_cost = grid_feature.compute_energy_cost(self.dt)

        total_cost = ess_cost + dg_cost + grid_cost

        # Compute safety violations
        network_feature = self.state.features[self.network_idx]
        safety_penalty = network_feature.compute_safety_penalty()

        # SOC bound violations
        ess_feature = self.state.features[self.ess_idx]
        if ess_soc < ess_feature.min_soc:
            safety_penalty += (ess_feature.min_soc - ess_soc) * 100
        elif ess_soc > ess_feature.max_soc:
            safety_penalty += (ess_soc - ess_feature.max_soc) * 100

        # Reward = -cost - penalty * safety
        reward = -total_cost - 10.0 * safety_penalty

        return reward

    def update_device_dynamics(self) -> None:
        """Update device dynamics (ESS SOC, DG unit commitment, etc.).

        This is called after power flow simulation to update internal
        device states based on their setpoints and dynamics.
        """
        # Delegate to set_state for all state updates
        self.set_state()

    def set_renewable_availability(self, pv_availability: float,
                                    wind_availability: float) -> None:
        """Set renewable energy availability from external data.

        Args:
            pv_availability: PV availability fraction [0, 1]
            wind_availability: Wind availability fraction [0, 1]
        """
        self.set_state(pv_availability=pv_availability, wind_availability=wind_availability)

    def set_grid_price(self, price: float) -> None:
        """Set electricity price.

        Args:
            price: Electricity price in $/MWh
        """
        self.set_state(grid_price=price)

    def get_device_states(self) -> Dict[str, Dict[str, float]]:
        """Get current device states for simulation.

        Returns:
            Dict mapping device name to state dict
        """
        return {
            "ess": {
                "P": self.state.features[self.ess_idx].P,
                "Q": self.state.features[self.ess_idx].Q,
                "soc": self.state.features[self.ess_idx].soc,
            },
            "dg": {
                "P": self.state.features[self.dg_idx].P,
                "Q": self.state.features[self.dg_idx].Q,
                "on": self.state.features[self.dg_idx].on,
            },
            "pv": {
                "P": self.state.features[self.pv_idx].P,
                "Q": self.state.features[self.pv_idx].Q,
            },
            "wind": {
                "P": self.state.features[self.wind_idx].P,
                "Q": self.state.features[self.wind_idx].Q,
            },
            "grid": {
                "P": self.state.features[self.grid_idx].P,
                "Q": self.state.features[self.grid_idx].Q,
            },
        }

    def update_network_state(self, voltage_min: float, voltage_max: float,
                              voltage_avg: float, max_line_loading: float,
                              voltage_violations: int, overload_violations: int) -> None:
        """Update network state from power flow results.

        Args:
            voltage_min: Minimum bus voltage (pu)
            voltage_max: Maximum bus voltage (pu)
            voltage_avg: Average bus voltage (pu)
            max_line_loading: Maximum line loading percentage
            voltage_violations: Number of voltage violations
            overload_violations: Number of overload violations
        """
        self.set_state(
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            voltage_avg=voltage_avg,
            max_line_loading=max_line_loading,
            voltage_violations=voltage_violations,
            overload_violations=overload_violations,
        )

    def __repr__(self) -> str:
        return (f"MicrogridFieldAgent(id={self.agent_id}, "
                f"ess={self.state.features[self.ess_idx].capacity}MWh, "
                f"dg={self.dg_max_p}MW)")
