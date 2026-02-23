"""Quickstart — minimum viable HERON environment.

Shows the 3 things you need:
  1. A Feature class (what agents observe and control)
  2. A SimpleFieldAgent subclass (how agents behave)
  3. make_env() to wire it all together
"""

import numpy as np
import heron
from heron.shortcuts import Feature, Clipped, SimpleFieldAgent

# 1. Feature: each agent controls a single "value" in [-1, 1]
class Output(Feature):
    value: float = Clipped(default=0.0, min=-1.0, max=1.0)

# 2. Agent: penalised for deviating from zero
class Device(SimpleFieldAgent):
    features = [Output()]
    action_dim = 1
    action_range = (-1.0, 1.0)

    def reward(self, state):
        return -(state["Output"]["value"] ** 2)

# 3. Build and run
env = heron.make_env(
    Device(agent_id="dev_1"),
    Device(agent_id="dev_2"),
    Device(agent_id="dev_3"),
)

obs, _ = env.reset(seed=42)

for step in range(5):
    # random continuous actions for each device
    actions = {f"dev_{i}": np.random.uniform(-1, 1, size=1).astype(np.float32) for i in range(1, 4)}
    obs, rewards, terminated, truncated, info = env.step(actions)

    print(f"step {step+1}  rewards: "
          + "  ".join(f"{aid}={rewards[aid]:+.3f}" for aid in sorted(rewards) if aid.startswith("dev")))
