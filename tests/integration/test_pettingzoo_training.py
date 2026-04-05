"""HERON PettingZoo Integration: IPPO Training + Event-Driven Evaluation.

The PettingZoo analogue of ``test_rllib_action_passing.py``.  Demonstrates
the full adaptor pipeline:

1. Wrap a HERON env as a PettingZoo ParallelEnv
2. Train IPPO and shared-parameter PPO through the adaptor
3. Bridge trained policies back into HERON's event-driven simulation
4. Validate that rewards improve and event-driven eval produces results

Uses a custom "target-tracking" env where the reward signal is clear:
each agent observes its state (a scalar) and the optimal action pushes
it toward zero.  This isolates the adaptor's correctness from demo-env
specifics.

Run::

    python tests/integration/test_pettingzoo_training.py
"""

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.envs.builder import EnvBuilder
from heron.adaptors.pettingzoo import PettingZooParallelEnv, pettingzoo_env
from heron.adaptors.pettingzoo_runner import IPPOTrainer, evaluate_event_driven

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Custom "Target-Tracking" Env (clear reward signal for training validation)
# ============================================================================

@dataclass(slots=True)
class TrackerFeature(Feature):
    """Scalar position feature — agent tries to reach zero."""
    visibility: ClassVar[Sequence[str]] = ("public",)
    position: float = 5.0


class TrackerAgent(FieldAgent):
    """Agent that must learn to push its position toward zero.

    Action: continuous scalar in [-1, 1].
    State update: position += action.
    Reward: -|position| (maximized at position=0).
    """

    def __init__(self, agent_id: str, features: Optional[List[Feature]] = None, **kwargs):
        super().__init__(agent_id=agent_id, features=features or [TrackerFeature()], **kwargs)

    def init_action(self, features: Optional[List[Feature]] = None) -> Action:
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        return action

    def set_state(self, *a, **kw) -> None:
        pass

    def set_action(self, action: Any, *a, **kw) -> None:
        self.action.set_values(action)

    def apply_action(self) -> None:
        feat = self.state.features.get("TrackerFeature")
        if feat is not None:
            feat.position += float(self.action.c[0])

    def compute_local_reward(self, local_state: dict, prev=None) -> float:
        # Read directly from internal state (avoids format mismatch)
        feat = self.state.features.get("TrackerFeature")
        return -abs(feat.position) if feat else 0.0


def _noop_sim(agent_states: Dict[str, Dict]) -> Dict[str, Dict]:
    return agent_states


def _build_tracker_env(n_agents: int = 2, max_steps: int = 50, initial_pos: float = 5.0):
    builder = EnvBuilder("tracker")
    for i in range(n_agents):
        builder.add_agent(
            f"agent_{i}", TrackerAgent,
            features=[TrackerFeature(position=initial_pos)],
        )
    return builder.simulation(_noop_sim).termination(max_steps=max_steps).build()


# ============================================================================
# Configuration
# ============================================================================
MAX_STEPS = 50
NUM_TRAIN_EPISODES = 200
HIDDEN_DIM = 64
LR = 3e-4


# ============================================================================
# Sanity Check
# ============================================================================
def sanity_check():
    """Verify the PettingZoo adaptor wraps, resets, steps, and returns valid data."""
    logger.info("--- Sanity Check ---")
    heron_env = _build_tracker_env()
    env = PettingZooParallelEnv(heron_env=heron_env)

    logger.info(f"  Agents: {env.possible_agents}")
    for aid in env.possible_agents:
        logger.info(f"  {aid}: obs={env.observation_space(aid)}, act={env.action_space(aid)}")

    obs, infos = env.reset(seed=42)
    assert set(obs.keys()) == set(env.agents)
    for aid, o in obs.items():
        assert env.observation_space(aid).contains(o), f"{aid} obs not in space"

    actions = {aid: env.action_space(aid).sample() for aid in env.agents}
    obs2, rew, term, trunc, info2 = env.step(actions)
    assert set(obs2.keys()) == set(env.agents)
    for aid in info2:
        assert "is_active" in info2[aid]

    # Verify reward is non-zero and negative (position starts at 5.0)
    for aid, r in rew.items():
        assert r < 0, f"{aid} reward should be negative, got {r}"

    env.close()
    logger.info("  Reset / step / info / reward checks OK")
    logger.info("--- Sanity Check PASSED ---\n")


# ============================================================================
# Training
# ============================================================================
def train_ippo(num_episodes: int = NUM_TRAIN_EPISODES):
    """IPPO: independent per-agent PPO networks."""
    logger.info("=" * 60)
    logger.info(f"IPPO Training: TrackerEnv, {num_episodes} episodes")
    logger.info("=" * 60)

    env = PettingZooParallelEnv(heron_env=_build_tracker_env(max_steps=MAX_STEPS))
    trainer = IPPOTrainer(env, hidden_dim=HIDDEN_DIM, lr=LR, shared_params=False)
    metrics = trainer.train(num_episodes=num_episodes, log_interval=50, verbose=True)

    first = np.mean([m.total_reward for m in metrics[:30]])
    last = np.mean([m.total_reward for m in metrics[-30:]])
    logger.info(f"  First 30 avg reward: {first:.2f}")
    logger.info(f"  Last  30 avg reward: {last:.2f}")
    improved = last > first
    logger.info(f"  Improved: {improved}")

    return {"algorithm": "IPPO", "first": first, "last": last,
            "improved": improved, "trainer": trainer, "env": env}


def train_shared_ppo(num_episodes: int = NUM_TRAIN_EPISODES):
    """Shared-parameter PPO (MAPPO-like)."""
    logger.info("\n" + "=" * 60)
    logger.info(f"Shared PPO Training: TrackerEnv, {num_episodes} episodes")
    logger.info("=" * 60)

    env = PettingZooParallelEnv(heron_env=_build_tracker_env(max_steps=MAX_STEPS))
    trainer = IPPOTrainer(env, hidden_dim=HIDDEN_DIM, lr=LR, shared_params=True)
    metrics = trainer.train(num_episodes=num_episodes, log_interval=50, verbose=True)

    first = np.mean([m.total_reward for m in metrics[:30]])
    last = np.mean([m.total_reward for m in metrics[-30:]])
    logger.info(f"  First 30 avg reward: {first:.2f}")
    logger.info(f"  Last  30 avg reward: {last:.2f}")
    improved = last > first
    logger.info(f"  Improved: {improved}")

    return {"algorithm": "Shared PPO", "first": first, "last": last,
            "improved": improved, "trainer": trainer, "env": env}


# ============================================================================
# Event-Driven Evaluation
# ============================================================================
def run_event_driven_eval(trainer: IPPOTrainer, label: str):
    """Run HERON event-driven eval with trained policies."""
    logger.info(f"\n--- Event-Driven Eval ({label}) ---")

    heron_env = trainer.env._heron_env
    policies = trainer.get_policies()
    results = evaluate_event_driven(
        heron_env=heron_env, policies=policies, t_end=50.0, seed=100,
    )

    logger.info(f"  Terminated: {results['terminated']}")
    logger.info(f"  Truncated:  {results['truncated']}")
    logger.info(f"  Events:     {results['num_events']}")
    logger.info(f"  Rewards:    {results['per_agent_rewards']}")
    logger.info(f"  Total:      {results['total_reward']:.2f}")

    return results


# ============================================================================
# Also verify with the TwoRoomHeating demo env (adaptor+API correctness)
# ============================================================================
def demo_env_api_check():
    """Quick check that the adaptor works with the real demo env end-to-end."""
    import heron.demo_envs  # noqa: F401

    logger.info("\n--- TwoRoomHeating-v0 API Check ---")
    env = pettingzoo_env("TwoRoomHeating-v0", max_steps=20)
    obs, infos = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(25):
        if not env.agents:
            break
        actions = {aid: env.action_space(aid).sample() for aid in env.agents}
        obs, rewards, term, trunc, infos = env.step(actions)
        total_reward += sum(rewards.values())
    env.close()
    logger.info(f"  Ran 20-step episode, total_reward={total_reward:.2f}")
    logger.info("--- TwoRoomHeating-v0 API Check PASSED ---")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("HERON PettingZoo Integration Test")
    logger.info("Env: TrackerEnv (clear reward signal)")
    logger.info("Algorithms: IPPO, Shared PPO")
    logger.info("=" * 60)

    sanity_check()
    demo_env_api_check()

    # Train both algorithms
    ippo = train_ippo()
    shared = train_shared_ppo()

    # Event-driven eval with trained policies
    ippo_eval = run_event_driven_eval(ippo["trainer"], "IPPO")
    shared_eval = run_event_driven_eval(shared["trainer"], "Shared PPO")

    # Cleanup
    ippo["env"].close()
    shared["env"].close()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    for res in [ippo, shared]:
        logger.info(f"  {res['algorithm']:12s}: first_30={res['first']:.2f}, "
                     f"last_30={res['last']:.2f}, improved={res['improved']}")
    logger.info("Event-Driven Evaluation:")
    for label, ev in [("IPPO", ippo_eval), ("Shared PPO", shared_eval)]:
        logger.info(f"  {label:12s}: events={ev['num_events']}, "
                     f"total_reward={ev['total_reward']:.2f}")

    # Assertions
    assert ippo["improved"], "IPPO should show reward improvement"
    assert shared["improved"], "Shared PPO should show reward improvement"
    assert ippo_eval["num_events"] > 0, "Event-driven eval should produce events"
    assert shared_eval["num_events"] > 0, "Event-driven eval should produce events"

    logger.info("\nAll assertions passed. Done.")
