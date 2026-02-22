"""Event-driven deployment for the EV public charging case study.

Workflow:
1. Train pricing policies in synchronous CTDE mode
2. Attach trained policies to station coordinator agents
3. Configure TickConfig with jitter for realistic async timing
4. Run event-driven simulation via env.run_event_driven()

Usage:
    python -m case_studies.power.ev_public_charging_case.run_event_driven
"""

import logging
from typing import Dict

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.scheduling.analysis import EventAnalyzer
from heron.scheduling.tick_config import JitterType, TickConfig

from case_studies.power.ev_public_charging_case.policies import PricingPolicy
from case_studies.power.ev_public_charging_case.train_rllib import create_charging_env, train_simple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def configure_tick_configs(env, seed: int = 42) -> None:
    """Configure TickConfig with jitter for each agent level.

    System (L3): tick every 300s (one simulation step)
    Coordinator (L2): tick every 300s, with obs/act/msg delays
    Field (L1): tick every 300s, with smaller delays

    Args:
        env: ChargingEnv with registered agents
        seed: Base seed for jitter RNG
    """
    system_config = TickConfig.with_jitter(
        tick_interval=300.0,
        obs_delay=1.0,
        act_delay=2.0,
        msg_delay=1.0,
        reward_delay=5.0,  # wait for coordinator rewards (which wait for field agent rewards)
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=seed,
    )
    coordinator_config = TickConfig.with_jitter(
        tick_interval=300.0,
        obs_delay=1.0,
        act_delay=2.0,
        msg_delay=1.0,
        reward_delay=4.0,  # wait for field agent reward round-trips (~3 msg hops)
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=seed + 1,
    )
    field_config = TickConfig.with_jitter(
        tick_interval=300.0,
        obs_delay=0.5,
        act_delay=1.0,
        msg_delay=0.5,
        jitter_type=JitterType.GAUSSIAN,
        jitter_ratio=0.1,
        seed=seed + 2,
    )

    for agent_id, agent in env.registered_agents.items():
        if isinstance(agent, SystemAgent):
            agent.tick_config = system_config
        elif isinstance(agent, CoordinatorAgent):
            agent.tick_config = coordinator_config
        elif isinstance(agent, FieldAgent):
            agent.tick_config = field_config

    # Update scheduler's cached tick configs (cached during attach())
    for agent_id, agent in env.registered_agents.items():
        if hasattr(agent, 'tick_config') and agent.tick_config is not None:
            env.scheduler._agent_tick_configs[agent_id] = agent.tick_config


def deploy_event_driven(
    env,
    policies: Dict[str, PricingPolicy],
    t_end: float = 3600.0,
    seed: int = 100,
) -> None:
    """Deploy trained policies in event-driven mode.

    Args:
        env: ChargingEnv instance
        policies: Dict mapping station_id -> trained PricingPolicy
        t_end: Simulation end time in seconds
        seed: Seed for tick config jitter
    """
    # Attach trained policies to coordinator agents
    logger.info("Attaching trained policies to station coordinators...")
    env.set_agent_policies(policies)

    # Configure tick timing with jitter
    logger.info("Configuring TickConfigs with Gaussian jitter...")
    configure_tick_configs(env, seed=seed)

    # Reset to apply new configs
    env.reset(seed=seed)

    # Run event-driven simulation
    logger.info(f"Running event-driven simulation for {t_end}s...")
    event_analyzer = EventAnalyzer(verbose=False, track_data=True)
    episode = env.run_event_driven(event_analyzer=event_analyzer, t_end=t_end)

    # Print summary
    summary = episode.summary()
    logger.info("Event-driven simulation complete.")
    logger.info(f"  Total events: {summary.get('num_events', 0)}")
    logger.info(f"  Duration: {summary.get('duration', 0):.1f}s")
    logger.info(f"  Observations: {summary.get('observations', 0)}")
    logger.info(f"  State updates: {summary.get('state_updates', 0)}")
    logger.info(f"  Action results: {summary.get('action_results', 0)}")
    logger.info(f"  Event types: {summary.get('event_counts', {})}")
    logger.info(f"  Message types: {summary.get('message_type_counts', {})}")

    reward_history = event_analyzer.get_reward_history()
    if reward_history:
        for agent_id, rewards in reward_history.items():
            if rewards:
                total_r = sum(r for _, r in rewards)
                logger.info(f"  Agent {agent_id}: total_reward={total_r:.2f}, steps={len(rewards)}")

    return episode


def main():
    """Full pipeline: train CTDE -> deploy event-driven."""
    logger.info("=" * 60)
    logger.info("Phase 1: CTDE Training (synchronous)")
    logger.info("=" * 60)
    env, policies, returns = train_simple(num_episodes=50, seed=42)
    env.close()

    logger.info(f"\nTraining returns (last 5): {[round(r, 2) for r in returns[-5:]]}")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Event-Driven Deployment (asynchronous)")
    logger.info("=" * 60)

    # Create fresh env for event-driven deployment
    deploy_env = create_charging_env()
    deploy_event_driven(deploy_env, policies, t_end=3600.0, seed=100)
    deploy_env.close()

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
