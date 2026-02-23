"""Custom environment — simulation physics, multiple agent types, coordinators.

Builds on 01_quickstart by showing:
  - A simulation function that enforces physical constraints
  - Two zones of agents managed by separate coordinators
  - EnvBuilder for explicit hierarchy wiring
"""

import numpy as np
from heron.shortcuts import Feature, Clipped, SimpleFieldAgent, EnvBuilder

# -- Features ------------------------------------------------------------------

class Battery(Feature):
    charge: float = Clipped(default=0.5, min=0.0, max=1.0)
    capacity: float = 1.0

class Solar(Feature):
    output: float = Clipped(default=0.0, min=0.0, max=1.0)

# -- Agents --------------------------------------------------------------------

class BatteryAgent(SimpleFieldAgent):
    features = [Battery()]
    action_dim = 1
    action_range = (-0.2, 0.2)  # charge/discharge rate

    def reward(self, state):
        charge = state["Battery"]["charge"]
        return -((charge - 0.5) ** 2)  # keep charge balanced

class SolarAgent(SimpleFieldAgent):
    features = [Solar()]
    action_dim = 1
    action_range = (0.0, 1.0)  # curtailment factor

    def reward(self, state):
        return state["Solar"]["output"]  # maximise output

# -- Simulation ----------------------------------------------------------------

def simulate(states: dict) -> dict:
    """Clip battery charge to capacity; solar output is action * irradiance."""
    for aid, s in states.items():
        if "capacity" in s:
            s["charge"] = float(np.clip(s["charge"], 0.0, s["capacity"]))
        if "output" in s:
            irradiance = 0.8  # fixed for this example
            s["output"] = float(np.clip(s["output"] * irradiance, 0.0, 1.0))
    return states

# -- Build with explicit coordinators -----------------------------------------

env = (
    EnvBuilder()
    .add_agents(
        BatteryAgent(agent_id="bat_1"),
        BatteryAgent(agent_id="bat_2"),
        SolarAgent(agent_id="sol_1"),
        SolarAgent(agent_id="sol_2"),
    )
    .add_coordinator("storage_zone", subordinates=["bat_*"])
    .add_coordinator("solar_zone", subordinates=["sol_*"])
    .with_simulation(simulate)
    .build()
)

# -- Run -----------------------------------------------------------------------

obs, _ = env.reset(seed=0)
field_ids = ["bat_1", "bat_2", "sol_1", "sol_2"]
rng = np.random.default_rng(0)

for step in range(5):
    actions = {
        "bat_1": rng.uniform(-0.2, 0.2, size=1).astype(np.float32),
        "bat_2": rng.uniform(-0.2, 0.2, size=1).astype(np.float32),
        "sol_1": rng.uniform(0.0, 1.0, size=1).astype(np.float32),
        "sol_2": rng.uniform(0.0, 1.0, size=1).astype(np.float32),
    }
    obs, rewards, terminated, truncated, info = env.step(actions)

    print(f"step {step+1}  "
          + "  ".join(f"{aid}={rewards[aid]:+.3f}" for aid in field_ids))
