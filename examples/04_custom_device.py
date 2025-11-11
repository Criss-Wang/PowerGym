"""
Example 4: Creating a Custom Device
====================================

This example demonstrates how to create your own custom device by subclassing
DeviceAgent. We'll implement a simple Solar Panel device with MPPT control.

What you'll learn:
- Subclassing DeviceAgent to create custom devices
- Implementing device-specific methods (set_action_space, set_device_state, etc.)
- Using feature providers (ElectricalBasePh, PhaseConnection, custom features)
- Integrating custom devices into GridAgent and environments
- Device lifecycle: reset → observe → act → update_state → update_cost_safety

Custom Device: SolarPanel
- Continuous action: Curtailment factor (0=off, 1=max output)
- State features: P, Q (electrical), Irradiance, Temperature
- Physics: P = irradiance * panel_area * efficiency * curtailment
- Cost: Zero generation cost (renewable), but curtailment has opportunity cost

Runtime: ~30 seconds for 24 timesteps
"""

import numpy as np
import pandapower as pp
from dataclasses import dataclass
from typing import Any, Dict, Optional

from powergrid.agents.device_agent import DeviceAgent
from powergrid.agents.grid_agent import PowerGridAgentV2
from powergrid.core.policies import Policy
from powergrid.core.protocols import CentralizedSetpointProtocol, NoProtocol, Protocol
from powergrid.core.state import DeviceState, PhaseModel
from powergrid.devices.storage import ESS
from powergrid.envs.multi_agent.networked_grid_env import NetworkedGridEnv
from powergrid.features.electrical import ElectricalBasePh
from powergrid.features.connection import PhaseConnection
from powergrid.features.base import FeatureProvider
from powergrid.networks.ieee13 import IEEE13Bus


# ============================================================================
# Step 1: Define Custom Feature (if needed)
# ============================================================================

@dataclass
class SolarPanelFeature(FeatureProvider):
    """Custom feature for solar panel state."""

    # Panel characteristics
    panel_area_m2: float = 100.0  # Panel area in m²
    efficiency: float = 0.20  # 20% efficiency
    max_power_kw: float = 20.0  # Maximum rated power

    # Environmental state
    irradiance_w_m2: float = 0.0  # Current solar irradiance (W/m²)
    temperature_c: float = 25.0  # Panel temperature (°C)
    curtailment_factor: float = 1.0  # Curtailment (0-1)

    # Performance tracking
    energy_generated_kwh: float = 0.0  # Cumulative energy
    curtailed_energy_kwh: float = 0.0  # Lost opportunity

    def vector(self) -> np.ndarray:
        """Convert to observation vector."""
        return np.array(
            [
                self.irradiance_w_m2 / 1000.0,  # Normalize to kW/m²
                self.temperature_c / 100.0,  # Normalize
                self.curtailment_factor,
                self.energy_generated_kwh / 100.0,  # Normalize
            ],
            dtype=np.float32,
        )

    def names(self) -> list[str]:
        """Feature names for observation."""
        return [
            "irradiance_kw_m2",
            "temperature_c_norm",
            "curtailment_factor",
            "energy_generated_kwh_norm",
        ]

    def clamp_(self) -> None:
        """Clamp values to valid ranges."""
        self.irradiance_w_m2 = np.clip(self.irradiance_w_m2, 0.0, 1200.0)
        self.temperature_c = np.clip(self.temperature_c, -20.0, 80.0)
        self.curtailment_factor = np.clip(self.curtailment_factor, 0.0, 1.0)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "kind": "SolarPanelFeature",
            "payload": {
                "panel_area_m2": self.panel_area_m2,
                "efficiency": self.efficiency,
                "max_power_kw": self.max_power_kw,
                "irradiance_w_m2": self.irradiance_w_m2,
                "temperature_c": self.temperature_c,
                "curtailment_factor": self.curtailment_factor,
                "energy_generated_kwh": self.energy_generated_kwh,
                "curtailed_energy_kwh": self.curtailed_energy_kwh,
            },
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Deserialize from dictionary."""
        return cls(**d["payload"])


# ============================================================================
# Step 2: Implement Custom Device
# ============================================================================

@dataclass
class SolarPanelConfig:
    """Configuration for solar panel device."""

    bus: str
    panel_area_m2: float = 100.0
    efficiency: float = 0.20
    max_power_kw: float = 20.0
    min_q_mvar: float = -5.0
    max_q_mvar: float = 5.0
    s_rated_mva: float = 0.025  # 25 kVA
    dt_h: float = 1.0  # Timestep in hours
    phase_model: str = "balanced_1ph"


class SolarPanel(DeviceAgent):
    """Custom solar panel device with curtailment control.

    Action space: [curtailment_factor, Q]
    - curtailment_factor ∈ [0, 1]: 0=off, 1=full output
    - Q ∈ [min_q, max_q]: Reactive power

    State features:
    - ElectricalBasePh: P, Q output
    - PhaseConnection: Bus connection
    - SolarPanelFeature: Irradiance, temperature, etc.
    """

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any],
    ):
        """Initialize solar panel device."""
        config_data = device_config.get("device_state_config", {})

        self._solar_config = SolarPanelConfig(
            bus=config_data.get("bus", ""),
            panel_area_m2=config_data.get("panel_area_m2", 100.0),
            efficiency=config_data.get("efficiency", 0.20),
            max_power_kw=config_data.get("max_power_kw", 20.0),
            min_q_mvar=config_data.get("min_q_mvar", -5.0),
            max_q_mvar=config_data.get("max_q_mvar", 5.0),
            s_rated_mva=config_data.get("s_rated_mva", 0.025),
            dt_h=config_data.get("dt_h", 1.0),
            phase_model=config_data.get("phase_model", "balanced_1ph"),
        )

        # Store environmental data source
        self._irradiance_data = None
        self._temperature_data = None
        self._current_timestep = 0

        super().__init__(
            agent_id=agent_id,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    def set_action_space(self) -> None:
        """Define action space: [curtailment, Q]."""
        self.action.set_specs(
            dim_c=2,
            range=(
                np.array([0.0, self._solar_config.min_q_mvar], dtype=np.float32),
                np.array([1.0, self._solar_config.max_q_mvar], dtype=np.float32),
            ),
        )

    def set_device_state(self) -> None:
        """Initialize state with custom features."""
        # Electrical output
        electrical = ElectricalBasePh(
            phase_model=PhaseModel.BALANCED_1PH,
            P_MW=0.0,
            Q_MVAr=0.0,
        )

        # Bus connection
        connection = PhaseConnection(
            phase_model=PhaseModel.BALANCED_1PH,
            connection="ON",
        )

        # Solar-specific features
        solar_feature = SolarPanelFeature(
            panel_area_m2=self._solar_config.panel_area_m2,
            efficiency=self._solar_config.efficiency,
            max_power_kw=self._solar_config.max_power_kw,
        )

        # Aggregate into DeviceState
        self.state = DeviceState(
            phase_model=PhaseModel.BALANCED_1PH,
            features=[electrical, connection, solar_feature],
            prefix_names=False,
        )

        # Store references for easy access
        self.electrical = electrical
        self.solar = solar_feature

    def set_environmental_data(self, irradiance_data: np.ndarray, temperature_data: np.ndarray):
        """Set time-series environmental data (called externally)."""
        self._irradiance_data = irradiance_data
        self._temperature_data = temperature_data

    def reset_device(self, **kwargs) -> None:
        """Reset device to initial state."""
        self.electrical.P_MW = 0.0
        self.electrical.Q_MVAr = 0.0
        self.solar.irradiance_w_m2 = 0.0
        self.solar.temperature_c = 25.0
        self.solar.curtailment_factor = 1.0
        self.solar.energy_generated_kwh = 0.0
        self.solar.curtailed_energy_kwh = 0.0
        self._current_timestep = 0
        self.cost = 0.0
        self.safety = 0.0

    def update_state(self, timestep: Optional[int] = None) -> None:
        """Update solar panel state based on actions and environment."""
        # Get environmental conditions
        if timestep is not None:
            self._current_timestep = timestep

        if self._irradiance_data is not None:
            self.solar.irradiance_w_m2 = float(self._irradiance_data[self._current_timestep])
            self.solar.temperature_c = float(self._temperature_data[self._current_timestep])

        # Get curtailment action
        curtailment = float(self.action.c[0])
        q_setpoint = float(self.action.c[1])

        self.solar.curtailment_factor = curtailment

        # Compute output power (simplified model)
        # P = Irradiance * Area * Efficiency * Curtailment
        max_power_mw = (
            self.solar.irradiance_w_m2
            * self.solar.panel_area_m2
            * self.solar.efficiency
            / 1e6  # Convert W to MW
        )

        # Apply curtailment
        actual_power_mw = max_power_mw * curtailment

        # Temperature derating (simple model: -0.5% per °C above 25°C)
        temp_derate = 1.0 - 0.005 * max(0.0, self.solar.temperature_c - 25.0)
        actual_power_mw *= temp_derate

        # Update electrical state
        self.electrical.P_MW = actual_power_mw
        self.electrical.Q_MVAr = q_setpoint

        # Track energy
        energy_mwh = actual_power_mw * self._solar_config.dt_h
        curtailed_mwh = (max_power_mw - actual_power_mw) * self._solar_config.dt_h

        self.solar.energy_generated_kwh += energy_mwh * 1000.0
        self.solar.curtailed_energy_kwh += curtailed_mwh * 1000.0

        # Clamp features
        self.state.clamp_()

    def update_cost_safety(self) -> None:
        """Compute cost and safety metrics."""
        # Cost: Opportunity cost of curtailment (negative revenue)
        # Assume selling price $50/MWh
        curtailed_mwh = self.solar.curtailed_energy_kwh / 1000.0
        opportunity_cost = curtailed_mwh * 50.0

        self.cost = opportunity_cost

        # Safety: Check if exceeding rated power
        s_mva = np.sqrt(self.electrical.P_MW ** 2 + self.electrical.Q_MVAr ** 2)
        overload = max(0.0, s_mva - self._solar_config.s_rated_mva)

        self.safety = overload * 10.0  # Penalty for overload


# ============================================================================
# Step 3: Use Custom Device in Environment
# ============================================================================

class CustomDeviceEnv(NetworkedGridEnv):
    """Environment demonstrating custom solar panel device."""

    def _build_net(self):
        """Build network with custom solar panel."""
        net = IEEE13Bus("MG1")

        # Create solar panel with our custom class
        solar = SolarPanel(
            agent_id="solar1",
            device_config={
                "name": "solar1",
                "device_state_config": {
                    "bus": "Bus 645",
                    "panel_area_m2": 150.0,  # 150 m² of panels
                    "efficiency": 0.22,  # 22% efficient
                    "max_power_kw": 33.0,  # 33 kW rated
                    "min_q_mvar": -10.0,
                    "max_q_mvar": 10.0,
                    "s_rated_mva": 0.04,  # 40 kVA
                },
            },
        )

        # Create ESS for energy management
        ess = ESS(
            agent_id="ess1",
            device_config={
                "name": "ess1",
                "device_state_config": {
                    "bus": "Bus 634",
                    "capacity_MWh": 40.0,
                    "max_e_MWh": 36.0,
                    "min_e_MWh": 4.0,
                    "max_p_MW": 10.0,
                    "min_p_MW": -10.0,
                    "max_q_MVAr": 5.0,
                    "min_q_MVAr": -5.0,
                    "s_rated_MVA": 12.0,
                    "init_soc": 0.5,
                    "ch_eff": 0.95,
                    "dsc_eff": 0.95,
                },
            },
        )

        # Create GridAgent
        mg_agent = PowerGridAgentV2(
            net=net,
            grid_config={
                "name": "MG1",
                "base_power": 1.0,
                "load_scale": 1.0,
            },
            devices=[solar, ess],
            protocol=CentralizedSetpointProtocol(),
            centralized=True,
        )

        # Create dataset
        dataset = self._create_solar_dataset()
        mg_agent.add_dataset(dataset)

        # Set environmental data for solar panel
        solar.set_environmental_data(
            dataset["irradiance"],
            dataset["temperature"],
        )

        # Set environment attributes
        self.data_size = len(dataset["load"])
        self._total_days = self.data_size // self.max_episode_steps

        # Add devices to network
        mg_agent.add_sgen([solar])
        mg_agent.add_storage([ess])

        # Store
        self.possible_agents = ["MG1"]
        self.agent_dict = {"MG1": mg_agent}
        self.net = net

        return net

    def _create_solar_dataset(self):
        """Create dataset with solar irradiance patterns."""
        days = 10
        hours = days * 24

        # Daily load pattern
        load_pattern = np.array([0.5, 0.45, 0.4, 0.4, 0.45, 0.55,
                                 0.7, 0.85, 0.95, 1.0, 1.05, 1.1,
                                 1.1, 1.05, 1.0, 1.05, 1.15, 1.2,
                                 1.15, 1.05, 0.95, 0.85, 0.75, 0.65])
        load = np.tile(load_pattern, days) + np.random.normal(0, 0.05, hours)
        load = np.clip(load, 0.3, 1.5)

        # Solar irradiance (W/m²) - realistic daily curve
        irrad_pattern = np.array([0, 0, 0, 0, 0, 50,          # Night + dawn
                                  200, 400, 650, 850, 950, 1000,  # Morning ramp
                                  1000, 950, 850, 650, 400, 200,  # Afternoon decline
                                  50, 0, 0, 0, 0, 0])             # Sunset + night
        irradiance = np.tile(irrad_pattern, days)
        irradiance += np.random.normal(0, 50, hours)  # Cloud variability
        irradiance = np.clip(irradiance, 0, 1200)

        # Temperature (°C) - correlates with solar
        temp_pattern = np.array([15, 14, 13, 13, 14, 16,
                                 20, 25, 30, 35, 38, 40,
                                 40, 38, 35, 32, 28, 24,
                                 20, 18, 17, 16, 15, 15])
        temperature = np.tile(temp_pattern, days)
        temperature += np.random.normal(0, 2, hours)
        temperature = np.clip(temperature, 10, 45)

        return {
            "load": load,
            "solar": np.zeros(hours),  # Not used (we have irradiance instead)
            "wind": np.zeros(hours),
            "price": 50.0 * np.ones(hours),
            "irradiance": irradiance,
            "temperature": temperature,
        }

    def step(self, actions):
        """Override step to update solar panel timestep."""
        # Update solar panel with current timestep before stepping
        solar = self.agent_dict["MG1"].devices["solar1"]
        solar._current_timestep = self._t

        return super().step(actions)

    def _reward_and_safety(self):
        """Compute rewards and safety."""
        rewards = {}
        safety = {}

        for agent_id, agent in self.agent_dict.items():
            rewards[agent_id] = -agent.cost
            safety[agent_id] = agent.safety

        return rewards, safety


def main():
    """Run custom device example."""
    print("=" * 70)
    print("Example 4: Creating a Custom Device (Solar Panel)")
    print("=" * 70)

    # Create environment
    env_config = {
        "max_episode_steps": 24,
        "train": True,
    }

    print("\n[1] Creating environment with custom SolarPanel device...")
    env = CustomDeviceEnv(env_config)

    print(f"    Devices: {list(env.agent_dict['MG1'].devices.keys())}")
    print(f"    Solar panel features:")
    solar = env.agent_dict["MG1"].devices["solar1"]
    print(f"      - Panel area: {solar._solar_config.panel_area_m2} m²")
    print(f"      - Efficiency: {solar._solar_config.efficiency * 100:.1f}%")
    print(f"      - Max power: {solar._solar_config.max_power_kw} kW")

    # Reset
    print("\n[2] Resetting environment...")
    obs, info = env.reset(seed=42)

    # Run simulation
    print("\n[3] Running 24-hour simulation...")
    print(f"    {'Hour':<6} {'Irrad':<10} {'Solar P':<10} {'Curtail':<10} {'ESS P':<10} {'Reward':<10}")
    print("    " + "-" * 65)

    total_reward = 0

    for t in range(24):
        # Sample actions
        actions = {"MG1": env.action_spaces["MG1"].sample()}

        # Step
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # Get solar state
        solar = env.agent_dict["MG1"].devices["solar1"]
        irradiance = solar.solar.irradiance_w_m2
        solar_p = solar.electrical.P_MW
        curtail = solar.solar.curtailment_factor

        # Get ESS state
        ess = env.agent_dict["MG1"].devices["ess1"]
        ess_p = ess.electrical.P_MW

        reward = rewards["MG1"]
        total_reward += reward

        print(
            f"    {t+1:<6} "
            f"{irradiance:>8.0f}  "
            f"{solar_p:>8.3f}  "
            f"{curtail:>8.2f}  "
            f"{ess_p:>8.2f}  "
            f"{reward:>8.2f}"
        )

        if terminateds["__all__"]:
            break

    # Summary
    print("\n[4] Simulation Summary:")
    print(f"    Total reward: {total_reward:.2f}")
    print(f"    Average reward/hour: {total_reward / 24:.2f}")

    solar = env.agent_dict["MG1"].devices["solar1"]
    print(f"\n    Solar Panel Performance:")
    print(f"      Total energy generated: {solar.solar.energy_generated_kwh:.2f} kWh")
    print(f"      Total energy curtailed: {solar.solar.curtailed_energy_kwh:.2f} kWh")
    print(f"      Curtailment rate: {solar.solar.curtailed_energy_kwh / (solar.solar.energy_generated_kwh + solar.solar.curtailed_energy_kwh) * 100:.1f}%")

    print("\n[5] Custom Device Implementation Steps:")
    print("    1. Define custom FeatureProvider (SolarPanelFeature)")
    print("    2. Subclass DeviceAgent (SolarPanel)")
    print("    3. Implement set_action_space() - define control inputs")
    print("    4. Implement set_device_state() - aggregate features")
    print("    5. Implement reset_device() - initialize state")
    print("    6. Implement update_state() - apply physics/actions")
    print("    7. Implement update_cost_safety() - compute metrics")
    print("    8. Integrate into GridAgent and Environment")

    print("\n[6] Key Takeaways:")
    print("    - DeviceAgent is highly extensible")
    print("    - Custom features can encode domain-specific state")
    print("    - Physics updates happen in update_state()")
    print("    - Cost/safety computed separately from physics")
    print("    - Custom devices work seamlessly with existing infrastructure")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
