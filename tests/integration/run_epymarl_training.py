"""Unified MARL training with EPyMARL + HERON.

Trains MAPPO, QMIX, or MADDPG on the HERON action-passing environment
using EPyMARL (Extended PyMARL). Bypasses Sacred and calls EPyMARL's
training loop directly.

Usage::

    python tests/integration/run_epymarl_training.py --algo mappo
    python tests/integration/run_epymarl_training.py --algo qmix
    python tests/integration/run_epymarl_training.py --algo maddpg
    python tests/integration/run_epymarl_training.py --algo mappo --timesteps 5000

See also:
  - ``test_rllib_action_passing.py``   — RLlib (MAPPO, IPPO)
  - ``test_qmix_action_passing.py``    — Custom PyTorch QMIX
  - ``test_maddpg_action_passing.py``  — Custom PyTorch MADDPG
"""

import argparse
import logging
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace as SN
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import numpy as np
import torch as th

# ---------------------------------------------------------------------------
#  EPyMARL path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EPYMARL_SRC = os.path.join(_PROJECT_ROOT, "third_party", "epymarl", "src")
if _EPYMARL_SRC not in sys.path:
    sys.path.insert(0, _EPYMARL_SRC)

# ---------------------------------------------------------------------------
#  HERON environment components (same as test_rllib_action_passing.py)
# ---------------------------------------------------------------------------
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.core.feature import FeatureProvider
from heron.core.action import Action
from heron.envs.base import MultiAgentEnv
from heron.protocols.vertical import VerticalProtocol
from heron.adaptors.epymarl import HeronEPyMARLAdapter


@dataclass(slots=True)
class DevicePowerFeature(FeatureProvider):
    """Power state feature for devices."""
    visibility: ClassVar[Sequence[str]] = ["public"]
    power: float = 0.0
    capacity: float = 1.0

    def vector(self) -> np.ndarray:
        return np.array([self.power, self.capacity], dtype=np.float32)

    def set_values(self, **kwargs: Any) -> None:
        if "power" in kwargs:
            self.power = np.clip(kwargs["power"], -self.capacity, self.capacity)
        if "capacity" in kwargs:
            self.capacity = kwargs["capacity"]


class DeviceAgent(FieldAgent):
    """Field agent that directly controls a power device."""

    @property
    def power(self) -> float:
        return self.state.features["DevicePowerFeature"].power

    @property
    def capacity(self) -> float:
        return self.state.features["DevicePowerFeature"].capacity

    def init_action(self, features: List[FeatureProvider] = []):
        action = Action()
        action.set_specs(dim_c=1, range=(np.array([-1.0]), np.array([1.0])))
        action.set_values(np.array([0.0]))
        return action

    def compute_local_reward(self, local_state: dict) -> float:
        if "DevicePowerFeature" in local_state:
            power = float(local_state["DevicePowerFeature"][0])
            return -power ** 2
        return 0.0

    def set_action(self, action: Any) -> None:
        if isinstance(action, Action):
            if len(action.c) != self.action.dim_c:
                self.action.set_values(action.c[: self.action.dim_c])
            else:
                self.action.set_values(c=action.c)
        else:
            self.action.set_values(action)

    def set_state(self) -> None:
        new_power = self.action.c[0] * 0.5
        self.state.features["DevicePowerFeature"].set_values(power=new_power)

    def apply_action(self):
        self.set_state()


class ZoneCoordinator(CoordinatorAgent):
    def compute_local_reward(self, local_state: dict) -> float:
        return sum(local_state.get("subordinate_rewards", {}).values())


class GridSystem(SystemAgent):
    pass


class EnvState:
    def __init__(self, device_powers: Optional[Dict[str, float]] = None):
        self.device_powers = device_powers or {"device_1": 0.0, "device_2": 0.0}


class ActionPassingEnv(MultiAgentEnv):
    """Minimal multi-agent env for testing action passing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        for did in env_state.device_powers:
            env_state.device_powers[did] = np.clip(
                env_state.device_powers[did], -1.0, 1.0,
            )
        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict:
        from heron.agents.constants import FIELD_LEVEL

        agent_states = {}
        for aid, ag in self.registered_agents.items():
            if hasattr(ag, "level") and ag.level == FIELD_LEVEL and "device" in aid:
                agent_states[aid] = {
                    "_owner_id": aid,
                    "_owner_level": ag.level,
                    "_state_type": "FieldAgentState",
                    "features": {
                        "DevicePowerFeature": {
                            "power": env_state.device_powers.get(aid, 0.0),
                            "capacity": 1.0,
                        }
                    },
                }
        return {"agent_states": agent_states}

    def global_state_to_env_state(self, global_state: Dict) -> EnvState:
        agent_states = global_state.get("agent_states", {})
        device_powers = {}
        for aid, sd in agent_states.items():
            if "device" in aid and "features" in sd:
                feat = sd["features"].get("DevicePowerFeature", {})
                device_powers[aid] = feat.get("power", 0.0)
        return EnvState(
            device_powers=device_powers or {"device_1": 0.0, "device_2": 0.0}
        )


# ---------------------------------------------------------------------------
#  Environment factory (module-level for pickling in parallel runner)
# ---------------------------------------------------------------------------

def create_action_passing_env(config: dict) -> ActionPassingEnv:
    """Build and return the HERON action-passing environment."""
    device_1 = DeviceAgent(
        agent_id="device_1",
        features=[DevicePowerFeature(power=0.0, capacity=1.0)],
    )
    device_2 = DeviceAgent(
        agent_id="device_2",
        features=[DevicePowerFeature(power=0.0, capacity=1.0)],
    )
    coordinator = ZoneCoordinator(
        agent_id="coordinator",
        subordinates={"device_1": device_1, "device_2": device_2},
    )
    coordinator.protocol = VerticalProtocol()
    system = GridSystem(
        agent_id="system_agent",
        subordinates={"coordinator": coordinator},
    )
    return ActionPassingEnv(
        system_agent=system,
        scheduler_config={"start_time": 0.0, "time_step": 1.0},
        message_broker_config={"buffer_size": 1000, "max_queue_size": 100},
        simulation_wait_interval=0.01,
    )


# ---------------------------------------------------------------------------
#  EPyMARL env factory (called by REGISTRY)
# ---------------------------------------------------------------------------

def heron_env_fn(common_reward=True, reward_scalarisation="sum", seed=0, **kwargs):
    """EPyMARL-compatible env factory registered in REGISTRY."""
    return HeronEPyMARLAdapter(
        env_creator=create_action_passing_env,
        n_discrete=11,
        max_steps=50,
        common_reward=common_reward,
        reward_scalarisation=reward_scalarisation,
        seed=seed,
    )


# ---------------------------------------------------------------------------
#  Algorithm configs
# ---------------------------------------------------------------------------

ALGO_CONFIGS = {
    "mappo": {
        "action_selector": "soft_policies",
        "mask_before_softmax": True,
        "runner": "episode",  # use episode runner (parallel spawns subprocesses)
        "buffer_size": 10,
        "batch_size_run": 1,
        "batch_size": 10,
        "target_update_interval_or_tau": 0.01,
        "lr": 0.0003,
        "hidden_dim": 128,
        "obs_agent_id": True,
        "obs_last_action": False,
        "obs_individual_obs": False,
        "agent_output_type": "pi_logits",
        "learner": "ppo_learner",
        "entropy_coef": 0.001,
        "use_rnn": True,
        "standardise_returns": False,
        "standardise_rewards": True,
        "q_nstep": 5,
        "critic_type": "cv_critic",
        "epochs": 4,
        "eps_clip": 0.2,
        "name": "mappo",
        "mac": "basic_mac",
        "common_reward": True,
        "reward_scalarisation": "sum",
    },
    "qmix": {
        "action_selector": "epsilon_greedy",
        "epsilon_start": 1.0,
        "epsilon_finish": 0.05,
        "epsilon_anneal_time": 5000,
        "evaluation_epsilon": 0.0,
        "runner": "episode",
        "buffer_size": 5000,
        "target_update_interval_or_tau": 200,
        "obs_agent_id": True,
        "obs_last_action": False,
        "obs_individual_obs": False,
        "standardise_returns": False,
        "standardise_rewards": True,
        "agent_output_type": "q",
        "learner": "q_learner",
        "double_q": True,
        "mixer": "qmix",
        "use_rnn": False,
        "mixing_embed_dim": 32,
        "hypernet_layers": 2,
        "hypernet_embed": 64,
        "name": "qmix",
        "mac": "basic_mac",
        "common_reward": True,
        "reward_scalarisation": "sum",
    },
    "maddpg": {
        "runner": "episode",
        "buffer_size": 5000,
        "target_update_interval_or_tau": 200,
        "obs_agent_id": True,
        "obs_last_action": False,
        "obs_individual_obs": False,
        "mac": "maddpg_mac",
        "reg": 0.001,
        "batch_size": 32,
        "lr": 0.0005,
        "use_rnn": True,
        "standardise_returns": False,
        "standardise_rewards": True,
        "learner": "maddpg_learner",
        "agent_output_type": "pi_logits",
        "hidden_dim": 128,
        "critic_type": "maddpg_critic",
        "name": "maddpg",
        "common_reward": True,
        "reward_scalarisation": "sum",
    },
}

# Base defaults (from EPyMARL's default.yaml)
BASE_CONFIG = {
    "runner": "episode",
    "mac": "basic_mac",
    "env": "heron",
    "common_reward": True,
    "reward_scalarisation": "sum",
    "env_args": {"seed": 0},
    "batch_size_run": 1,
    "test_nepisode": 5,
    "test_interval": 500,
    "test_greedy": True,
    "log_interval": 500,
    "runner_log_interval": 500,
    "learner_log_interval": 500,
    "t_max": 2000,
    "use_cuda": False,
    "buffer_cpu_only": True,
    "use_tensorboard": False,
    "use_wandb": False,
    "wandb_team": None,
    "wandb_project": None,
    "wandb_mode": "offline",
    "wandb_save_model": False,
    "save_model": False,
    "save_model_interval": 50000,
    "checkpoint_path": "",
    "evaluate": False,
    "render": False,
    "load_step": 0,
    "save_replay": False,
    "local_results_path": "results",
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_size": 32,
    "lr": 0.0005,
    "optim_alpha": 0.99,
    "optim_eps": 0.00001,
    "grad_norm_clip": 10,
    "add_value_last_step": True,
    "agent": "rnn",
    "hidden_dim": 64,
    "obs_agent_id": True,
    "obs_last_action": True,
    "obs_individual_obs": False,
    "repeat_id": 1,
    "label": "default_label",
    "hypergroup": None,
}


# ---------------------------------------------------------------------------
#  Sanity check
# ---------------------------------------------------------------------------

def sanity_check():
    """Verify the adapter interface works correctly."""
    print("\n--- Sanity Check ---")
    adapter = HeronEPyMARLAdapter(
        env_creator=create_action_passing_env,
        n_discrete=11,
        max_steps=50,
    )

    info = adapter.get_env_info()
    print(f"  n_agents:      {info['n_agents']}")
    print(f"  obs_shape:     {info['obs_shape']}")
    print(f"  state_shape:   {info['state_shape']}")
    print(f"  n_actions:     {info['n_actions']}")
    print(f"  episode_limit: {info['episode_limit']}")

    obs, _ = adapter.reset()
    assert len(obs) == info["n_agents"], "obs count mismatch"
    assert obs[0].shape[0] == info["obs_shape"], "obs shape mismatch"

    state = adapter.get_state()
    assert state.shape[0] == info["state_shape"], "state shape mismatch"

    avail = adapter.get_avail_actions()
    assert len(avail) == info["n_agents"], "avail actions count mismatch"
    assert len(avail[0]) == info["n_actions"], "avail actions size mismatch"

    # Step with middle action (index 5 of 11 → midpoint of [-1, 1])
    actions = [5, 5]
    obs2, reward, terminated, truncated, info2 = adapter.step(actions)
    assert len(obs2) == info["n_agents"], "step obs count mismatch"
    assert isinstance(reward, float), "Expected scalar reward"

    print("  Reset / step / get_obs / get_state OK")
    print("--- Sanity Check PASSED ---\n")


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def run_training(algo: str, t_max: int = 2000, seed: int = 0):
    """Run EPyMARL training for a given algorithm."""
    # Register HERON env in EPyMARL's REGISTRY
    from envs import REGISTRY as env_REGISTRY
    env_REGISTRY["heron"] = heron_env_fn

    # Build config
    config = deepcopy(BASE_CONFIG)
    algo_config = deepcopy(ALGO_CONFIGS[algo])
    config.update(algo_config)
    config["t_max"] = t_max
    config["seed"] = seed
    config["env_args"]["seed"] = seed

    # Create args namespace
    args = SN(**config)
    args.device = "cuda" if args.use_cuda and th.cuda.is_available() else "cpu"

    # Validate reward config
    from utils.general_reward_support import test_alg_config_supports_reward
    assert test_alg_config_supports_reward(args), (
        f"Algorithm {algo} does not support the reward configuration"
    )

    # Setup logger
    from utils.logging import Logger
    console_logger = logging.getLogger(f"epymarl.{algo}")
    console_logger.setLevel(logging.INFO)
    if not console_logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(levelname)s %(asctime)s] %(message)s", "%H:%M:%S"))
        console_logger.addHandler(ch)
    logger = Logger(console_logger)

    # Run training
    print(f"\n{'=' * 60}")
    print(f"EPyMARL Training: {algo.upper()}")
    print(f"  t_max: {t_max}, seed: {seed}")
    print(f"{'=' * 60}")

    from run import run_sequential
    run_sequential(args=args, logger=logger)

    # Extract results from logger stats
    returns = logger.stats.get("test_return_mean", [])
    if not returns:
        returns = logger.stats.get("return_mean", [])

    if returns:
        final_return = returns[-1][1]
        print(f"\n  {algo.upper()} final mean return: {final_return:.3f}")
    else:
        print(f"\n  {algo.upper()}: no return data logged")

    return {
        "algorithm": algo.upper(),
        "returns": [(t, v) for t, v in returns] if returns else [],
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EPyMARL MARL training on HERON")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["mappo", "qmix", "maddpg", "all"],
        default="all",
        help="Algorithm to train (default: all)",
    )
    parser.add_argument("--timesteps", type=int, default=2000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    th.set_num_threads(1)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    print("=" * 60)
    print("HERON EPyMARL Integration — Action Passing")
    print("=" * 60)
    print("Agents: device_1, device_2  (field agents)")
    print("Protocol: VerticalProtocol")
    print(f"Algorithms: {args.algo}")

    sanity_check()

    algos = ["mappo", "qmix", "maddpg"] if args.algo == "all" else [args.algo]
    all_results = []

    for algo in algos:
        result = run_training(algo, t_max=args.timesteps, seed=args.seed)
        all_results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")
    for res in all_results:
        algo = res["algorithm"]
        returns = res["returns"]
        if returns:
            print(f"  {algo:8s}: final_return = {returns[-1][1]:.3f}")
        else:
            print(f"  {algo:8s}: NO DATA")
    print()
