"""Quick rollout script for a single-station EV charging environment.

Usage:
    python -m case_studies.power.ev_public_charging_case.run_single_station_rollout
"""

import numpy as np

from heron.agents.system_agent import SystemAgent

from case_studies.power.ev_public_charging_case.agents import ChargingSlot, StationCoordinator
from case_studies.power.ev_public_charging_case.envs.charging_env import ChargingEnv


def main():
    # Create a single station with 5 charger slots
    s_id = "station_0"
    slot_list = [
        ChargingSlot(agent_id=f"{s_id}_slot_{j}", p_max_kw=150.0)
        for j in range(5)
    ]
    station = StationCoordinator(agent_id=s_id)
    system = SystemAgent()

    slot_ids = [f"{s_id}_slot_{j}" for j in range(5)]
    env = ChargingEnv(
        agents=[system, station] + slot_list,
        hierarchy={"system_agent": [s_id], s_id: slot_ids},
        arrival_rate=10.0,
        dt=300.0,
        episode_length=86400.0,
    )

    print("Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"Agents in obs: {list(obs.keys())}")

    total_reward = 0.0
    num_steps = 288  # 1 day at 300s intervals

    for step in range(num_steps):
        # Random pricing action
        actions = {s_id: np.array([np.random.uniform(0.15, 0.40)], dtype=np.float32)}
        obs, rewards, terminated, truncated, infos = env.step(actions)

        step_reward = rewards.get(s_id, 0.0)
        total_reward += step_reward

        if step % 50 == 0:
            print(f"Step {step:3d} | Reward: {step_reward:7.3f} | Cumulative: {total_reward:8.3f}")

        if terminated.get("__all__", False) or truncated.get("__all__", False):
            print(f"Episode ended at step {step}")
            break

    print(f"\nTotal reward over {num_steps} steps: {total_reward:.4f}")
    env.close()


if __name__ == "__main__":
    main()
