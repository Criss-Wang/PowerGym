#!/usr/bin/env python3
"""
Example 9: Hierarchical Grid System with GridSystemAgent
========================================================

This example demonstrates the full 3-level HERON hierarchy for power grids
using the new HierarchicalGridEnv and GridSystemAgent components.

What you'll learn:
- How GridSystemAgent coordinates multiple PowerGridAgents
- Using HierarchicalGridEnv for training and testing
- CTDE training with system-level coordination (Option A)
- Event-driven testing with heterogeneous tick rates (Option B)

HERON 3-Level Hierarchy for Power Grids:
    Level 3 (System): GridSystemAgent
        - Manages multiple grid areas
        - System-wide frequency regulation
        - Inter-area power flow coordination

    Level 2 (Coordinator): PowerGridAgent
        - Manages devices within a grid/microgrid
        - Local voltage regulation
        - Device coordination via protocols

    Level 1 (Field): DeviceAgent (Generator, ESS, etc.)
        - Individual controllable devices
        - Local state management
        - Executes setpoints from coordinator

Tick Intervals (typical power grid):
    - Devices: 1-10 seconds (fast response)
    - Grid Coordinator: 60 seconds (local optimization)
    - System Agent: 300 seconds (system-wide coordination)

Usage:
    # Run the complete example
    python examples/09_hierarchical_grid_system.py

    # Quick test mode
    python examples/09_hierarchical_grid_system.py --quick

Runtime: ~5-10 seconds
"""

import argparse
import numpy as np
from typing import Any, Dict, List, Optional

from heron.agents.system_agent import SystemAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.field_agent import FieldAgent
from heron.core.feature import FeatureProvider
from heron.core.observation import Observation
from heron.core.state import SystemAgentState
from heron.protocols.vertical import SetpointProtocol, SystemProtocol
from heron.scheduling import EventScheduler, EventType


# =============================================================================
# Power Grid Domain Features
# =============================================================================

class DevicePowerFeature(FeatureProvider):
    """Power output feature for a controllable device."""
    visibility = ["owner", "coordinator"]

    def __init__(self, p_mw: float = 0.0, p_max: float = 10.0, p_min: float = 0.0):
        self.p_mw = p_mw
        self.p_max = p_max
        self.p_min = p_min

    def vector(self) -> np.ndarray:
        return np.array([self.p_mw, self.p_max, self.p_min], dtype=np.float32)

    def names(self) -> List[str]:
        return ["p_mw", "p_max", "p_min"]

    def to_dict(self) -> Dict[str, Any]:
        return {"p_mw": self.p_mw, "p_max": self.p_max, "p_min": self.p_min}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DevicePowerFeature":
        return cls(**data)

    def set_values(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


class GridAggregateFeature(FeatureProvider):
    """Aggregate power metrics for a grid coordinator."""
    visibility = ["owner", "system"]

    def __init__(
        self,
        total_gen_mw: float = 0.0,
        total_load_mw: float = 0.0,
        net_power_mw: float = 0.0
    ):
        self.total_gen_mw = total_gen_mw
        self.total_load_mw = total_load_mw
        self.net_power_mw = net_power_mw

    def vector(self) -> np.ndarray:
        return np.array([
            self.total_gen_mw, self.total_load_mw, self.net_power_mw
        ], dtype=np.float32)

    def names(self) -> List[str]:
        return ["total_gen_mw", "total_load_mw", "net_power_mw"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_gen_mw": self.total_gen_mw,
            "total_load_mw": self.total_load_mw,
            "net_power_mw": self.net_power_mw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridAggregateFeature":
        return cls(**data)

    def set_values(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


class SystemFrequencyFeature(FeatureProvider):
    """System-wide frequency feature."""
    visibility = ["system"]

    def __init__(self, frequency_hz: float = 60.0, nominal_hz: float = 60.0):
        self.frequency_hz = frequency_hz
        self.nominal_hz = nominal_hz

    def vector(self) -> np.ndarray:
        return np.array([self.frequency_hz, self.nominal_hz], dtype=np.float32)

    def names(self) -> List[str]:
        return ["frequency_hz", "nominal_hz"]

    def to_dict(self) -> Dict[str, Any]:
        return {"frequency_hz": self.frequency_hz, "nominal_hz": self.nominal_hz}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemFrequencyFeature":
        return cls(**data)

    def set_values(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


# =============================================================================
# Power Grid Agents (Simplified for demonstration)
# =============================================================================

class SimpleGenerator(FieldAgent):
    """Simplified generator device (L1)."""

    def __init__(
        self,
        agent_id: str,
        p_max: float = 10.0,
        tick_interval: float = 1.0,
        **kwargs
    ):
        # Set attributes BEFORE super().__init__ (which calls set_action)
        self._p_max = p_max
        self._cost_per_mw = np.random.uniform(20, 50)  # $/MWh
        super().__init__(agent_id=agent_id, tick_interval=tick_interval, **kwargs)

    def set_state(self) -> None:
        self.state.features = [
            DevicePowerFeature(p_mw=0.0, p_max=self._p_max, p_min=0.0)
        ]

    def set_action(self) -> None:
        self.action.set_specs(
            dim_c=1,
            range=(np.array([0.0]), np.array([self._p_max]))
        )

    def reset_agent(self, **kwargs) -> None:
        for f in self.state.features:
            if isinstance(f, DevicePowerFeature):
                f.p_mw = 0.0

    @property
    def cost(self) -> float:
        """Generation cost."""
        for f in self.state.features:
            if isinstance(f, DevicePowerFeature):
                return f.p_mw * self._cost_per_mw
        return 0.0

    def tick(self, scheduler, current_time, global_state=None, proxy=None) -> None:
        """Apply setpoint from action."""
        self._timestep = current_time
        if self.action.c is not None:
            setpoint = float(np.clip(self.action.c[0], 0, self._p_max))
            for f in self.state.features:
                if isinstance(f, DevicePowerFeature):
                    # Ramp toward setpoint
                    f.p_mw += 0.3 * (setpoint - f.p_mw)


class SimpleMicrogrid(CoordinatorAgent):
    """Simplified microgrid coordinator (L2)."""

    def __init__(
        self,
        agent_id: str,
        generators: Optional[List[SimpleGenerator]] = None,
        load_mw: float = 15.0,
        tick_interval: float = 60.0,
        **kwargs
    ):
        # Set attributes BEFORE super().__init__ (which calls set_state)
        self._load_mw = load_mw
        self._generators = generators or []

        super().__init__(
            agent_id=agent_id,
            protocol=SetpointProtocol(),
            tick_interval=tick_interval,
            **kwargs
        )

        if generators:
            for gen in generators:
                self.subordinates[gen.agent_id] = gen
                self.subordinate_agents[gen.agent_id] = gen
                gen.upstream_id = self.agent_id

    def set_state(self) -> None:
        self.state.features = [
            GridAggregateFeature(total_gen_mw=0.0, total_load_mw=self._load_mw)
        ]

    def _build_local_observation(
        self,
        subordinate_obs: Dict[str, Observation],
        **kwargs
    ) -> Dict[str, Any]:
        """Aggregate generator outputs."""
        total_gen = 0.0
        for obs in subordinate_obs.values():
            if "state" in obs.local and len(obs.local["state"]) >= 1:
                total_gen += obs.local["state"][0]  # p_mw

        net_power = total_gen - self._load_mw

        for f in self.state.features:
            if isinstance(f, GridAggregateFeature):
                f.total_gen_mw = total_gen
                f.net_power_mw = net_power

        return {
            "subordinate_obs": subordinate_obs,
            "total_gen_mw": total_gen,
            "total_load_mw": self._load_mw,
            "net_power_mw": net_power,
        }

    @property
    def cost(self) -> float:
        """Total generation cost."""
        return sum(
            gen.cost for gen in self.subordinate_agents.values()
            if hasattr(gen, 'cost')
        )

    @property
    def safety(self) -> float:
        """Power imbalance penalty."""
        for f in self.state.features:
            if isinstance(f, GridAggregateFeature):
                return abs(f.net_power_mw) * 10.0  # Penalty for imbalance
        return 0.0


class SimpleGridSystem(SystemAgent):
    """Simplified grid system agent (L3)."""

    def __init__(
        self,
        agent_id: str = "grid_system",
        microgrids: Optional[List[SimpleMicrogrid]] = None,
        tick_interval: float = 300.0,
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            protocol=SystemProtocol(),
            tick_interval=tick_interval,
            **kwargs
        )

        self.state = SystemAgentState(owner_id=self.agent_id, owner_level=3)

        if microgrids:
            for mg in microgrids:
                self.coordinators[mg.agent_id] = mg
                mg.upstream_id = self.agent_id

        self._init_state()
        self.frequency = 60.0

    def set_state(self) -> None:
        self.state.features = [SystemFrequencyFeature(frequency_hz=60.0)]

    def _build_system_observation(
        self,
        coordinator_obs: Dict[str, Observation],
        *args, **kwargs
    ) -> Dict[str, Any]:
        """Aggregate microgrid metrics."""
        total_gen = 0.0
        total_load = 0.0

        for obs in coordinator_obs.values():
            if "total_gen_mw" in obs.local:
                total_gen += obs.local["total_gen_mw"]
            if "total_load_mw" in obs.local:
                total_load += obs.local["total_load_mw"]

        # Simulate frequency response to imbalance
        imbalance = total_gen - total_load
        self.frequency = 60.0 + 0.01 * imbalance  # Simplified droop

        for f in self.state.features:
            if isinstance(f, SystemFrequencyFeature):
                f.frequency_hz = self.frequency

        return {
            "coordinator_obs": coordinator_obs,
            "total_generation": total_gen,
            "total_load": total_load,
            "frequency": self.frequency,
        }

    @property
    def cost(self) -> float:
        return sum(mg.cost for mg in self.coordinators.values() if hasattr(mg, 'cost'))

    @property
    def safety(self) -> float:
        return sum(mg.safety for mg in self.coordinators.values() if hasattr(mg, 'safety'))


# =============================================================================
# Build Hierarchy
# =============================================================================

def build_power_grid_hierarchy():
    """Build a simple 3-level power grid hierarchy.

    Structure:
        GridSystem (L3)
            |
        +---+---+
        |       |
      MG_A    MG_B    (L2)
        |       |
      +---+   +---+
      |   |   |   |
     G1  G2  G3  G4   (L1)

    Returns:
        Tuple of (system, microgrids, generators)
    """
    # Create generators (L1)
    gen_a1 = SimpleGenerator("gen_a1", p_max=10.0, tick_interval=1.0)
    gen_a2 = SimpleGenerator("gen_a2", p_max=8.0, tick_interval=1.0)
    gen_b1 = SimpleGenerator("gen_b1", p_max=12.0, tick_interval=1.0)
    gen_b2 = SimpleGenerator("gen_b2", p_max=6.0, tick_interval=1.0)

    generators = [gen_a1, gen_a2, gen_b1, gen_b2]

    # Create microgrids (L2)
    mg_a = SimpleMicrogrid(
        agent_id="microgrid_a",
        generators=[gen_a1, gen_a2],
        load_mw=12.0,
        tick_interval=60.0,
    )
    mg_b = SimpleMicrogrid(
        agent_id="microgrid_b",
        generators=[gen_b1, gen_b2],
        load_mw=10.0,
        tick_interval=60.0,
    )

    microgrids = [mg_a, mg_b]

    # Create system agent (L3)
    system = SimpleGridSystem(
        agent_id="grid_system",
        microgrids=microgrids,
        tick_interval=300.0,
    )

    return system, microgrids, generators


# =============================================================================
# Option A: CTDE Training (Synchronous)
# =============================================================================

def run_ctde_training(num_steps: int = 30, verbose: bool = True):
    """Run CTDE training with synchronous step.

    This demonstrates Option A where all agents step together
    and rewards can be shared for cooperative learning.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Option A: CTDE Training (Synchronous)")
        print("=" * 60)

    system, microgrids, generators = build_power_grid_hierarchy()

    # Reset all
    system.reset()
    for mg in microgrids:
        mg.reset()
    for gen in generators:
        gen.reset()

    total_cost = 0.0
    total_safety = 0.0
    step_data = []

    for step in range(num_steps):
        # 1. System observes (aggregates up the hierarchy)
        system_obs = system.observe()
        freq = system_obs.local.get("frequency", 60.0)
        total_gen = system_obs.local.get("total_generation", 0)
        total_load = system_obs.local.get("total_load", 0)

        # 2. Compute control action (simple proportional control)
        # Target: match generation to load
        imbalance = total_load - total_gen
        adjustment = np.clip(imbalance * 0.3, -5, 5)

        # Distribute to generators (4 generators)
        target_per_gen = (total_load + adjustment) / 4
        joint_action = np.array([target_per_gen] * 4)

        # 3. System distributes actions down hierarchy
        system.act(system_obs, upstream_action=joint_action)

        # 4. Simulate generator response
        for gen in generators:
            setpoint = gen.action.c[0] if gen.action.c is not None else 0
            for f in gen.state.features:
                if isinstance(f, DevicePowerFeature):
                    f.p_mw += 0.3 * (setpoint - f.p_mw)
                    f.p_mw = np.clip(f.p_mw, 0, f.p_max)

        # 5. Compute metrics
        step_cost = system.cost
        step_safety = system.safety
        total_cost += step_cost
        total_safety += step_safety

        step_data.append({
            "step": step,
            "frequency": freq,
            "total_gen": total_gen,
            "total_load": total_load,
            "cost": step_cost,
            "safety": step_safety,
        })

        if verbose and (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}: freq={freq:.3f}Hz, "
                  f"gen={total_gen:.1f}MW, load={total_load:.1f}MW, "
                  f"cost=${step_cost:.1f}, safety={step_safety:.1f}")

    avg_cost = total_cost / num_steps
    avg_safety = total_safety / num_steps

    if verbose:
        print(f"\nTraining complete:")
        print(f"  Average cost: ${avg_cost:.2f}")
        print(f"  Average safety penalty: {avg_safety:.2f}")

    return {
        "total_cost": total_cost,
        "avg_cost": avg_cost,
        "total_safety": total_safety,
        "avg_safety": avg_safety,
        "step_data": step_data,
    }


# =============================================================================
# Option B: Event-Driven Testing
# =============================================================================

def run_event_driven_testing(t_end: float = 600.0, verbose: bool = True):
    """Run event-driven testing with heterogeneous tick rates.

    This demonstrates Option B where agents tick independently
    at their configured intervals.

    Tick rates:
        - Generators: 1 second (fast device response)
        - Microgrids: 60 seconds (local coordination)
        - System: 300 seconds (system-wide coordination)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Option B: Event-Driven Testing")
        print("=" * 60)

    system, microgrids, generators = build_power_grid_hierarchy()
    all_agents = [system] + microgrids + generators

    # Reset all
    for agent in all_agents:
        agent.reset()

    # Create scheduler
    scheduler = EventScheduler(start_time=0.0)

    # Register agents with tick intervals
    for gen in generators:
        scheduler.register_agent(gen.agent_id, tick_interval=gen.tick_interval)
    for mg in microgrids:
        scheduler.register_agent(mg.agent_id, tick_interval=mg.tick_interval)
    scheduler.register_agent(system.agent_id, tick_interval=system.tick_interval)

    # Track ticks
    tick_counts = {a.agent_id: 0 for a in all_agents}
    agent_lookup = {a.agent_id: a for a in all_agents}

    def tick_handler(event, sched):
        agent_id = event.agent_id
        agent = agent_lookup.get(agent_id)
        if agent:
            tick_counts[agent_id] += 1
            agent.tick(sched, event.timestamp, None, None)

    scheduler.set_handler(EventType.AGENT_TICK, tick_handler)

    if verbose:
        print(f"\nRunning simulation for {t_end}s...")
        print(f"  - Generators tick every 1s")
        print(f"  - Microgrids tick every 60s")
        print(f"  - System ticks every 300s")

    # Run simulation
    events_processed = scheduler.run_until(t_end=t_end)

    if verbose:
        print(f"\nEvents processed: {events_processed}")
        print(f"\nTick counts:")
        for agent in all_agents:
            level = "L1" if isinstance(agent, SimpleGenerator) else \
                    "L2" if isinstance(agent, SimpleMicrogrid) else "L3"
            expected = int(t_end / agent.tick_interval) + 1
            actual = tick_counts[agent.agent_id]
            status = "OK" if actual == expected else "MISMATCH"
            print(f"  {agent.agent_id:15s} ({level}): {actual:4d} ticks "
                  f"(expected {expected}) [{status}]")

    return {
        "events_processed": events_processed,
        "tick_counts": tick_counts,
        "simulation_time": scheduler.current_time,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Example 9: Hierarchical Grid System with GridSystemAgent"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick test with fewer steps"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True,
        help="Verbose output"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Example 9: Hierarchical Grid System")
    print("=" * 60)
    print("\nThis example demonstrates the 3-level HERON hierarchy:")
    print("  L3: GridSystemAgent - System-wide coordination")
    print("  L2: PowerGridAgent (SimpleMicrogrid) - Local optimization")
    print("  L1: DeviceAgent (SimpleGenerator) - Device control")

    # Run training
    num_steps = 10 if args.quick else 30
    training_results = run_ctde_training(num_steps=num_steps, verbose=args.verbose)

    # Run testing
    t_end = 120.0 if args.quick else 600.0
    testing_results = run_event_driven_testing(t_end=t_end, verbose=args.verbose)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nOption A (CTDE Training):")
    print(f"  - Steps: {num_steps}")
    print(f"  - Average cost: ${training_results['avg_cost']:.2f}")
    print(f"  - Average safety: {training_results['avg_safety']:.2f}")

    print(f"\nOption B (Event-Driven Testing):")
    print(f"  - Simulation time: {testing_results['simulation_time']:.1f}s")
    print(f"  - Events processed: {testing_results['events_processed']}")

    print("\n" + "-" * 60)
    print("This pattern is used by HierarchicalGridEnv for full power grids.")
    print("See: powergrid/envs/hierarchical_grid_env.py")
    print("-" * 60)


if __name__ == "__main__":
    main()
