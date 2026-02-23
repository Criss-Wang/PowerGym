"""EPyMARL adapter — wrap a HERON env for discrete-action MARL algorithms.

Shows how to:
  - Wrap any HERON env with HeronEPyMARLAdapter
  - Discretise continuous actions into N bins
  - Run an episode with the EPyMARL interface (get_obs, get_state, etc.)
"""

import numpy as np
from heron.shortcuts import Feature, Clipped, SimpleFieldAgent
from heron.shortcuts.quickstart import make_env
from heron.adaptors.epymarl import HeronEPyMARLAdapter

# -- Minimal agent (same as 01_quickstart) -------------------------------------

class Output(Feature):
    value: float = Clipped(default=0.0, min=-1.0, max=1.0)

class Device(SimpleFieldAgent):
    features = [Output()]
    action_dim = 1
    action_range = (-1.0, 1.0)

    def reward(self, state):
        return -(state["Output"]["value"] ** 2)

# -- Wrap with EPyMARL adapter -------------------------------------------------
# The adapter needs a factory because EPyMARL may create multiple env instances.

adapter = HeronEPyMARLAdapter(
    env_creator=lambda _: make_env(
        Device(agent_id="dev_1"),
        Device(agent_id="dev_2"),
    ),
    n_discrete=11,   # 11 bins per continuous action dimension
    max_steps=25,
)

print(f"agents: {adapter.n_agents}  |  obs_size: {adapter.get_obs_size()}  |  "
      f"actions: {adapter.get_total_actions()}  |  state_size: {adapter.get_state_size()}\n")

# -- Run one episode with discrete actions -------------------------------------

obs, _ = adapter.reset()
total_reward = 0.0

for step in range(adapter.episode_limit):
    actions = [np.random.randint(adapter.get_total_actions()) for _ in range(adapter.n_agents)]
    obs, reward, terminated, truncated, info = adapter.step(actions)
    total_reward += reward
    if terminated or truncated:
        break

print(f"episode finished after {step+1} steps  |  total_reward: {total_reward:+.2f}")
