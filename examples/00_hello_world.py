#!/usr/bin/env python3
"""Hello World Example - Domain-Agnostic HERON Introduction.

This example demonstrates the core HERON concepts without any domain-specific
dependencies. It creates a simple 2-level hierarchy:

    CoordinatorAgent ("zone_controller")
           |
    +------+------+
    |             |
FieldAgent   FieldAgent
("sensor_1") ("sensor_2")

The coordinator aggregates sensor readings and sends setpoint commands.

Run with:
    python examples/00_hello_world.py
"""

import numpy as np
from typing import Any, Dict, Optional

from heron.agents import FieldAgent, CoordinatorAgent
from heron.core.feature import FeatureProvider
from heron.core.observation import Observation
from heron.protocols.vertical import SetpointProtocol


# =============================================================================
# Step 1: Define a custom Feature for agent state
# =============================================================================

class TemperatureFeature(FeatureProvider):
    """Simple temperature feature that sensors can observe.

    Features are the building blocks of agent state. They define:
    - What data the agent tracks (temperature value)
    - Who can see it (visibility tags)
    - How to convert to/from vectors for RL
    """

    # Visibility: owner can see it, coordinator can aggregate it
    visibility = ["owner", "coordinator"]

    def __init__(self, initial_temp: float = 20.0):
        self.temperature = initial_temp

    def vector(self) -> np.ndarray:
        """Convert feature to numpy array for RL algorithms."""
        return np.array([self.temperature], dtype=np.float32)

    def names(self) -> list:
        """Human-readable names for each dimension."""
        return ["temperature"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for saving/loading."""
        return {"temperature": self.temperature}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemperatureFeature":
        """Deserialize from dict."""
        return cls(initial_temp=data.get("temperature", 20.0))

    def set_values(self, **kwargs) -> None:
        """Update feature values."""
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]


# =============================================================================
# Step 2: Define a custom FieldAgent (sensor)
# =============================================================================

class TemperatureSensor(FieldAgent):
    """A simple temperature sensor that can be controlled via setpoints.

    FieldAgents are the leaf nodes in the hierarchy. They:
    - Manage local state (temperature reading)
    - Execute actions (adjust temperature setpoint)
    - Report observations to their coordinator
    """

    def set_state(self) -> None:
        """Initialize the agent's state with features."""
        # Add temperature feature to state
        self.state.features.append(
            TemperatureFeature(initial_temp=20.0 + np.random.randn() * 2)
        )

    def set_action(self) -> None:
        """Define the agent's action space.

        This sensor can receive a temperature setpoint in range [15, 30].
        """
        self.action.set_specs(
            dim_c=1,  # 1 continuous action dimension
            range=(np.array([15.0]), np.array([30.0]))  # Temperature range
        )

    def reset_agent(self, **kwargs) -> None:
        """Reset sensor to initial state."""
        # Randomize initial temperature
        for feature in self.state.features:
            if isinstance(feature, TemperatureFeature):
                feature.temperature = 20.0 + np.random.randn() * 2

    def update_state(self, setpoint: float) -> None:
        """Simulate sensor responding to setpoint.

        In a real application, this would interface with hardware
        or a physics simulation.
        """
        for feature in self.state.features:
            if isinstance(feature, TemperatureFeature):
                # Simple first-order response toward setpoint
                feature.temperature += 0.1 * (setpoint - feature.temperature)


# =============================================================================
# Step 3: Define a custom CoordinatorAgent
# =============================================================================

class ZoneController(CoordinatorAgent):
    """Zone controller that manages multiple temperature sensors.

    CoordinatorAgents sit above FieldAgents in the hierarchy. They:
    - Aggregate observations from subordinates
    - Compute coordination actions (e.g., setpoints)
    - Distribute actions to subordinates via protocols
    """

    def _build_local_observation(
        self,
        subordinate_obs: Dict[str, Observation],
        **kwargs
    ) -> Dict[str, Any]:
        """Build coordinator's local observation from subordinate data.

        This aggregates all sensor readings into a single observation dict.
        """
        # Collect all temperature readings
        temps = []
        for agent_id, obs in subordinate_obs.items():
            if "state" in obs.local:
                temps.append(obs.local["state"][0])  # Temperature is first feature

        return {
            "subordinate_obs": subordinate_obs,
            "avg_temperature": np.mean(temps) if temps else 20.0,
            "num_sensors": len(temps),
        }


# =============================================================================
# Step 4: Build the hierarchy and run a simple simulation
# =============================================================================

def main():
    print("=" * 60)
    print("HERON Hello World Example")
    print("=" * 60)

    # Create field agents (sensors)
    sensor_1 = TemperatureSensor(agent_id="sensor_1")
    sensor_2 = TemperatureSensor(agent_id="sensor_2")

    print(f"\nCreated sensors:")
    print(f"  - {sensor_1}")
    print(f"  - {sensor_2}")

    # Create coordinator with SetpointProtocol for direct control
    coordinator = ZoneController(
        agent_id="zone_controller",
        protocol=SetpointProtocol(),  # Direct setpoint-based control
    )

    # Register subordinates with coordinator
    coordinator.subordinate_agents = {
        "sensor_1": sensor_1,
        "sensor_2": sensor_2,
    }
    # Update the base class subordinates dict as well
    coordinator.subordinates = coordinator.subordinate_agents

    print(f"\nCreated coordinator:")
    print(f"  - {coordinator}")

    # Reset all agents
    coordinator.reset()

    print("\n" + "-" * 60)
    print("Running simulation for 10 steps...")
    print("-" * 60)

    target_temp = 25.0  # Target temperature setpoint

    for step in range(10):
        # 1. Coordinator observes (aggregates from sensors)
        obs = coordinator.observe()
        avg_temp = obs.local.get("avg_temperature", 0)

        # 2. Compute setpoints (simple proportional control)
        # In practice, this would come from an RL policy
        error = target_temp - avg_temp
        setpoint_adjustment = np.clip(error * 0.5, -2.0, 2.0)

        # Create joint action for all subordinates
        # SetpointProtocol will split this among subordinates
        joint_action = np.array([
            target_temp + setpoint_adjustment,  # sensor_1 setpoint
            target_temp + setpoint_adjustment,  # sensor_2 setpoint
        ])

        # 3. Coordinator acts (distributes setpoints via protocol)
        coordinator.act(obs, upstream_action=joint_action)

        # 4. Simulate sensor response (in real env, this is physics)
        for sensor in [sensor_1, sensor_2]:
            setpoint = sensor.action.c[0] if sensor.action.c is not None else target_temp
            sensor.update_state(setpoint)

        # Print status
        t1 = sensor_1.state.features[0].temperature
        t2 = sensor_2.state.features[0].temperature
        print(f"Step {step+1:2d}: avg_temp={avg_temp:.2f}, "
              f"sensor_1={t1:.2f}, sensor_2={t2:.2f}, "
              f"target={target_temp:.1f}")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)

    # Print final state
    print(f"\nFinal temperatures:")
    print(f"  - sensor_1: {sensor_1.state.features[0].temperature:.2f}")
    print(f"  - sensor_2: {sensor_2.state.features[0].temperature:.2f}")
    print(f"  - target:   {target_temp:.1f}")

    print("\n" + "-" * 60)
    print("Key Concepts Demonstrated:")
    print("-" * 60)
    print("1. FeatureProvider - Composable state with visibility control")
    print("2. FieldAgent - Leaf agents managing local state and actions")
    print("3. CoordinatorAgent - Manages subordinates, aggregates observations")
    print("4. SetpointProtocol - Vertical coordination via direct setpoints")
    print("5. Hierarchical observe/act pattern")
    print("\nSee docs/source/key_concepts.md for detailed explanations.")


if __name__ == "__main__":
    main()
