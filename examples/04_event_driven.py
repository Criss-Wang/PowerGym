"""Event-driven mode — agents tick asynchronously with realistic timing.

HERON supports two execution modes:
  - env.step(actions)       — synchronous, actions provided externally (for training)
  - env.run_event_driven()  — asynchronous, agents choose actions via a Policy

This example shows the event-driven mode. The key differences from env.step():
  1. Each agent needs a Policy (to decide actions autonomously)
  2. Each agent gets a TickConfig (to control timing and jitter)
  3. The simulation runs on the SystemAgent's tick cycle (default: every 300s)
"""

import numpy as np
from heron.shortcuts import Feature, Clipped, SimpleFieldAgent, EnvBuilder
from heron.core.policies import Policy
from heron.core.observation import Observation
from heron.scheduling import TickConfig, JitterType
from heron.scheduling.analysis import EventAnalyzer

# -- Minimal random policy -----------------------------------------------------

class RandomPolicy(Policy):
    """Takes random actions — a starting point for event-driven testing."""
    observation_mode = "local"

    def __init__(self, action_dim=1, action_range=(-1.0, 1.0), seed=0):
        self.action_dim = action_dim
        self.action_range = action_range
        self.rng = np.random.default_rng(seed)

    def forward(self, observation: Observation):
        lo, hi = self.action_range
        values = self.rng.uniform(lo, hi, size=self.action_dim).astype(np.float32)
        return self.vec_to_action(values, self.action_dim, self.action_range)

# -- Same minimal agent as 01, but with policy and tick_config -----------------

class Output(Feature):
    value: float = Clipped(default=0.0, min=-1.0, max=1.0)

class Device(SimpleFieldAgent):
    features = [Output()]
    action_dim = 1
    action_range = (-1.0, 1.0)

    def reward(self, state):
        return -(state["Output"]["value"] ** 2)

# -- Build with timing configuration ------------------------------------------

tick_cfg = TickConfig.with_jitter(
    tick_interval=5.0,    # agents tick every ~5s
    obs_delay=0.1,        # 100ms observation latency
    act_delay=0.2,        # 200ms action effect delay
    jitter_type=JitterType.GAUSSIAN,
    jitter_ratio=0.1,     # 10% timing variability
    seed=42,
)

env = (
    EnvBuilder()
    .add_agents(
        Device(agent_id="dev_1", policy=RandomPolicy(seed=1), tick_config=tick_cfg),
        Device(agent_id="dev_2", policy=RandomPolicy(seed=2), tick_config=tick_cfg),
    )
    .build()
)
env.reset(seed=0)

# -- Run event-driven simulation -----------------------------------------------
# SystemAgent ticks every 300s by default, so t_end=1500 gives ~5 simulation cycles.

analyzer = EventAnalyzer(track_data=True)
episode = env.run_event_driven(analyzer, t_end=1500.0)

# -- Results -------------------------------------------------------------------

print(f"events: {episode.num_events}  |  duration: {episode.duration:.1f}s  |  "
      f"action_results: {episode.action_result_count}\n")

for aid, history in sorted(analyzer.get_reward_history().items()):
    if history and aid.startswith("dev"):
        rewards = [r for _, r in history]
        print(f"  {aid}: {len(history)} ticks, mean_reward={np.mean(rewards):.4f}")
