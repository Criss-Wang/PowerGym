import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import numpy as np

from case_studies.power.ev_public_charging_case.env.charging_env import SimpleChargingEnv

def env_creator(config):
    return SimpleChargingEnv(
        arrival_rate=config.get("arrival_rate", 1000.0),
        num_stations=config.get("num_stations", 1),
        charger_num=config.get("charger_num", 2),
        seed=config.get("seed", 0),
    )

register_env("ev_public_charging_v1", env_creator)

ray.init(ignore_reinit_error=True)

config = (
    PPOConfig()
    .environment(env="ev_public_charging_v1", env_config={"arrival_rate": 1000.0, "num_stations": 1, "charger_num": 2})
    .framework("torch")
)

algo = config.build()
for _ in range(10):
    result = algo.train()
    print(result["episode_reward_mean"])

# rollout
env = env_creator({"arrival_rate": 1000.0, "num_stations": 1, "charger_num": 2, "seed": 0})
obs, info = env.reset(seed=0)

for t in range(10):
    actions = {}
    for aid, o in obs.items():
        a = algo.compute_single_action(o, policy_id="default_policy")
        actions[aid] = a
    obs, rewards, terminated, truncated, infos = env.step(actions)
    print(t, rewards)