"""
Unified RLlib training + event-driven runner for EV public charging.
Style: Fully Config-driven via YAML (Supporting per-agent schedule overrides).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import re
import yaml
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Heron & Case Study Imports
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.action import Action
from heron.core.policies import Policy
from heron.scheduling.analysis import EpisodeAnalyzer
from heron.scheduling.schedule_config import JitterType, ScheduleConfig

from case_studies.power.ev_public_charging_case.agents import ChargingSlot, StationCoordinator
from case_studies.power.ev_public_charging_case.envs.charging_env import ChargingEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip())
    return cleaned.strip("-._") or "run"


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    return value


def _config_fingerprint(config: Dict[str, Any]) -> str:
    subset = {
        "env_specs": config.get("env_specs", {}),
        "nondeterminism": config.get("nondeterminism", {}),
        "training": config.get("training", {}),
    }
    canonical = _canonicalize(subset)
    raw = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def _build_run_name(config: Dict[str, Any]) -> str:
    env_specs = config.get("env_specs", {})
    training = config.get("training", {})
    nondeterminism = config.get("nondeterminism", {})
    default_sched = nondeterminism.get("default_config", {})

    label_parts = [
        f"s{int(env_specs.get('num_stations', 0))}",
        f"c{int(env_specs.get('num_chargers', 0))}",
        f"arr{float(env_specs.get('arrival_rate', 0.0)):g}",
        f"dt{float(env_specs.get('dt', 0.0)):g}",
        f"seed{int(training.get('seed', 0))}",
        f"jit{float(default_sched.get('jitter_ratio', 0.0)):g}",
    ]
    base_label = _slugify("_".join(label_parts))
    return f"{base_label}_{_utc_compact_now()}_{_config_fingerprint(config)}"


def resolve_output_dir(config: Dict[str, Any]) -> Path:
    paths_cfg = config.setdefault("paths", {})
    cached = paths_cfg.get("resolved_output_dir")
    if cached:
        return Path(cached)

    base_output_dir = Path(paths_cfg.get("output_dir", "outputs"))
    auto_run_name = bool(paths_cfg.get("auto_run_name", True))
    manual_run_name = paths_cfg.get("run_name")

    if auto_run_name:
        run_name = _slugify(str(manual_run_name)) if manual_run_name else _build_run_name(config)
        resolved = base_output_dir / run_name
    else:
        resolved = base_output_dir

    resolved.mkdir(parents=True, exist_ok=True)
    paths_cfg["resolved_output_dir"] = str(resolved)

    pointer_path = base_output_dir / "latest_run.json"
    _write_json(
        pointer_path,
        {
            "timestamp": _utc_now(),
            "resolved_output_dir": str(resolved),
            "run_name": resolved.name,
            "config_hash": _config_fingerprint(config),
        },
    )
    return resolved

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _checkpoint_to_path(checkpoint_obj: Any) -> str:
    """Normalize RLlib checkpoint return values to a concrete path string."""
    if isinstance(checkpoint_obj, str):
        return checkpoint_obj
    if hasattr(checkpoint_obj, "path"):
        return str(checkpoint_obj.path)
    if hasattr(checkpoint_obj, "checkpoint") and hasattr(checkpoint_obj.checkpoint, "path"):
        return str(checkpoint_obj.checkpoint.path)
    raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint_obj)!r}")

def configure_schedule_configs(env: ChargingEnv, nondeterm_specs: Dict[str, Any], seed: int = 42) -> None:
    """
    根据 YAML 配置为每个 Agent 设置调度参数。
    支持全局默认值与站点特定值的层级覆盖。
    """
    default = nondeterm_specs.get("default_config", {})
    station_specific = nondeterm_specs.get("station_specific", {})

    for agent_id, agent in env.registered_agents.items():
        if isinstance(agent, CoordinatorAgent):
            spec = station_specific.get(agent_id, {})

            conf = ScheduleConfig.with_jitter(
                tick_interval = float(spec.get("tick_interval", default.get("tick_interval", 300.0))),
                obs_delay     = float(spec.get("obs_delay", default.get("obs_delay", 1.0))),
                act_delay     = float(spec.get("act_delay", default.get("act_delay", 2.0))),
                msg_delay     = float(spec.get("msg_delay", default.get("msg_delay", 1.0))),
                jitter_type   = getattr(JitterType, spec.get("jitter_type", default.get("jitter_type", "GAUSSIAN"))),
                jitter_ratio  = float(spec.get("jitter_ratio", default.get("jitter_ratio", 0.1))),
                seed          = seed + hash(agent_id) % 1000
            )
            agent.schedule_config = conf
            logger.info(f"Configured {agent_id}: jitter={conf.jitter_ratio}, obs_delay={conf.obs_delay}")


    for agent_id, agent in env.registered_agents.items():
        if hasattr(agent, "schedule_config") and agent.schedule_config is not None:
            env.scheduler._agent_schedule_configs[agent_id] = agent.schedule_config

def create_charging_env(env_specs: Dict[str, Any]) -> ChargingEnv:
    num_stations = int(env_specs.get("num_stations"))
    num_chargers = int(env_specs.get("num_chargers"))

    coordinators: List[StationCoordinator] = []
    for i in range(num_stations):
        station_id = f"station_{i}"
        slots = {
            f"{station_id}_slot_{j}": ChargingSlot(agent_id=f"{station_id}_slot_{j}", p_max_kw=150.0)
            for j in range(num_chargers)
        }
        coordinators.append(StationCoordinator(agent_id=station_id, subordinates=slots))

    return ChargingEnv(
        coordinator_agents=coordinators,
        arrival_rate=float(env_specs.get("arrival_rate", 10.0)),
        dt=float(env_specs.get("dt", 300.0)),
        episode_length=float(env_specs.get("episode_length", 86400.0)),
    )


def build_rllib_env_config(env_specs: Dict[str, Any], nondeterminism: Dict[str, Any]) -> Dict[str, Any]:
    """Build the flat env_config expected by RLlibBasedHeronEnv."""
    num_stations = int(env_specs.get("num_stations", 2))
    num_chargers = int(env_specs.get("num_chargers", 4))

    default_sched = nondeterminism.get("default_config", {})
    station_specific = nondeterminism.get("station_specific", {})

    def _schedule_from_spec(spec: Dict[str, Any], seed: int) -> ScheduleConfig:
        return ScheduleConfig.with_jitter(
            tick_interval=float(spec.get("tick_interval", default_sched.get("tick_interval", 300.0))),
            obs_delay=float(spec.get("obs_delay", default_sched.get("obs_delay", 1.0))),
            act_delay=float(spec.get("act_delay", default_sched.get("act_delay", 2.0))),
            msg_delay=float(spec.get("msg_delay", default_sched.get("msg_delay", 1.0))),
            jitter_type=getattr(JitterType, spec.get("jitter_type", default_sched.get("jitter_type", "GAUSSIAN"))),
            jitter_ratio=float(spec.get("jitter_ratio", default_sched.get("jitter_ratio", 0.1))),
            seed=seed,
        )

    agents: List[Dict[str, Any]] = []
    coordinators: List[Dict[str, Any]] = []
    trainable_agent_ids: List[str] = []

    for i in range(num_stations):
        station_id = f"station_{i}"
        trainable_agent_ids.append(station_id)
        station_spec = station_specific.get(station_id, {})

        slot_ids: List[str] = []
        for j in range(num_chargers):
            slot_id = f"{station_id}_slot_{j}"
            slot_ids.append(slot_id)
            agents.append(
                {
                    "agent_id": slot_id,
                    "agent_cls": ChargingSlot,
                    "coordinator": station_id,
                    "p_max_kw": float(env_specs.get("slot_p_max_kw", 150.0)),
                }
            )

        coordinators.append(
            {
                "coordinator_id": station_id,
                "agent_cls": StationCoordinator,
                "subordinates": slot_ids,
                "schedule_config": _schedule_from_spec(station_spec, seed=42 + i),
            }
        )

    return {
        "env_class": ChargingEnv,
        "env_kwargs": {
            "arrival_rate": float(env_specs.get("arrival_rate", 10.0)),
            "dt": float(env_specs.get("dt", 300.0)),
            "episode_length": float(env_specs.get("episode_length", 86400.0)),
        },
        "agents": agents,
        "coordinators": coordinators,
        "agent_ids": trainable_agent_ids,
        "max_steps": int(env_specs.get("episode_length", 86400) // env_specs.get("dt", 300)),
    }


def _get_mean_reward(result: Dict[str, Any]) -> float:
    """Extract mean reward across RLlib result schema variants."""
    for accessor in [
        lambda r: r["env_runners"]["episode_return_mean"],
        lambda r: r["env_runners"]["episode_reward_mean"],
        lambda r: r["episode_reward_mean"],
        lambda r: r["sampler_results"]["episode_reward_mean"],
    ]:
        try:
            val = accessor(result)
            if val is not None:
                return float(val)
        except (KeyError, TypeError, ValueError):
            continue
    return float("nan")


class RLlibPPOPolicyBridge(Policy):
    """Use a restored RLlib PPO policy as a HERON Policy in event-driven mode."""

    observation_mode = "local"

    def __init__(self, module: Any, obs_dim: int, action_space: Any) -> None:
        self._module = module
        self.obs_dim = obs_dim
        self._action_template = Action.from_gym_space(action_space)

    def forward(self, observation: Any) -> Action:
        import torch

        obs_vec = self.extract_obs_vector(observation, self.obs_dim).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).float()

        self._module.eval()
        with torch.no_grad():
            out = self._module.forward_inference({"obs": obs_tensor})

        if "actions" in out:
            raw_action = np.asarray(out["actions"].cpu().numpy()[0], dtype=np.float32).reshape(-1)
        elif "action_dist_inputs" in out:
            dist_inputs = np.asarray(out["action_dist_inputs"].cpu().numpy()[0], dtype=np.float32).reshape(-1)
            raw_action = self._parse_dist_inputs(dist_inputs)
        else:
            raise ValueError(f"Unexpected RLModule output keys: {sorted(out.keys())}")

        action = self._action_template.copy()
        action.set_values(raw_action)
        return action

    def _parse_dist_inputs(self, dist_inputs: np.ndarray) -> np.ndarray:
        """Extract deterministic action from RLlib distribution inputs."""
        if self._action_template.dim_c:
            means = dist_inputs[: self._action_template.dim_c]
            return means.astype(np.float32)

        raise ValueError("Only continuous coordinator actions are supported in this bridge.")

def train_rllib_ppo(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from heron.adaptors.rllib import RLlibBasedHeronEnv
    except ImportError:
        raise RuntimeError("Ray RLlib is required. Install with: pip install 'ray[rllib]'")

    env_specs = config.get("env_specs", {})
    train_params = config.get("training", {})
    output_dir = resolve_output_dir(config)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env_config = build_rllib_env_config(
        env_specs=env_specs,
        nondeterminism=config.get("nondeterminism", {}),
    )
    num_stations = int(env_specs.get("num_stations"))
    policy_ids = {f"policy_st_{i}" for i in range(num_stations)}
    ppo_cfg = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .environment(env=RLlibBasedHeronEnv, env_config=env_config)
        .framework("torch")
        .training(lr=1e-4, train_batch_size=4000)
        .env_runners(num_env_runners=2)
        .multi_agent(
            policies=policy_ids,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_st_{agent_id.split('_')[1]}"
        )
    )

    ray.init(ignore_reinit_error=True)
    algo = ppo_cfg.build_algo()

    best_reward = float("-inf")
    best_ckpt_path = None

    for iteration in range(1, train_params.get("num_iterations", 50) + 1):
        result = algo.train()
        reward_mean = _get_mean_reward(result)
        reward_txt = f"{reward_mean:.3f}" if reward_mean == reward_mean else "n/a"
        logger.info(f"Iter {iteration}: reward_mean = {reward_txt}")

        if not math.isnan(reward_mean) and reward_mean > best_reward:
            best_reward = reward_mean
            best_ckpt = algo.save(str(checkpoints_dir / "best"))
            best_ckpt_path = _checkpoint_to_path(best_ckpt)
            logger.info(
                f"New best checkpoint at iter {iteration}: reward_mean={reward_mean:.3f}, path={best_ckpt_path}"
            )

        if iteration % train_params.get("checkpoint_freq", 10) == 0:
            ckpt_path = _checkpoint_to_path(algo.save(str(checkpoints_dir / "periodic")))
            logger.info(f"Saved checkpoint: {ckpt_path}")

    if best_ckpt_path is None:
        best_ckpt_path = _checkpoint_to_path(algo.save(str(checkpoints_dir / "best")))
        logger.info(f"No numeric reward metric found; using final checkpoint as best: {best_ckpt_path}")

    best_checkpoint_meta = {
        "timestamp": _utc_now(),
        "checkpoint_path": best_ckpt_path,
        "best_reward_mean": None if not math.isfinite(best_reward) else best_reward,
    }
    _write_json(output_dir / "best_checkpoint.json", best_checkpoint_meta)

    algo.stop()
    ray.shutdown()
    return {
        "status": "completed",
        "best_checkpoint": best_ckpt_path,
        "best_reward_mean": best_checkpoint_meta["best_reward_mean"],
    }


def _resolve_best_checkpoint(config: Dict[str, Any], explicit_checkpoint: str | None = None) -> str | None:
    """Resolve best checkpoint path from explicit arg or saved metadata."""
    if explicit_checkpoint:
        return explicit_checkpoint

    paths_cfg = config.setdefault("paths", {})
    if paths_cfg.get("resolved_output_dir"):
        output_dir = Path(paths_cfg["resolved_output_dir"])
    else:
        output_dir = Path(paths_cfg.get("output_dir", "outputs"))

    meta_path = output_dir / "best_checkpoint.json"
    if not meta_path.exists():
        latest_run_path = Path(paths_cfg.get("output_dir", "outputs")) / "latest_run.json"
        if latest_run_path.exists():
            try:
                latest_payload = json.loads(latest_run_path.read_text(encoding="utf-8"))
                latest_dir = latest_payload.get("resolved_output_dir")
                if latest_dir:
                    output_dir = Path(latest_dir)
                    paths_cfg["resolved_output_dir"] = str(output_dir)
                    meta_path = output_dir / "best_checkpoint.json"
            except json.JSONDecodeError:
                return None
        if not meta_path.exists():
            return None

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    checkpoint_path = payload.get("checkpoint_path")
    if checkpoint_path and Path(checkpoint_path).exists():
        return checkpoint_path
    return None


def _build_event_driven_policies_from_checkpoint(
    env: ChargingEnv,
    config: Dict[str, Any],
    checkpoint_path: str,
) -> Dict[str, Policy]:
    """Restore PPO checkpoint and create HERON policies for station coordinators."""
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from heron.adaptors.rllib import RLlibBasedHeronEnv
    except ImportError:
        raise RuntimeError("Ray RLlib is required for checkpoint inference. Install with: pip install 'ray[rllib]'")

    env_specs = config.get("env_specs", {})
    env_config = build_rllib_env_config(
        env_specs=env_specs,
        nondeterminism=config.get("nondeterminism", {}),
    )

    should_shutdown = False
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        should_shutdown = True

    algo = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .environment(env=RLlibBasedHeronEnv, env_config=env_config)
        .framework("torch")
        .training(lr=1e-4, train_batch_size=4000)
        .env_runners(num_env_runners=2)
        .multi_agent(
            policies={"station_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "station_policy",
        )
        .build_algo()
    )

    try:
        algo.restore(checkpoint_path)
        policies: Dict[str, Policy] = {}

        for agent_id, agent in env.registered_agents.items():
            if isinstance(agent, StationCoordinator):
                p_id = f"policy_st_{agent_id.split('_')[1]}"
                module = algo.get_module(p_id)

                if module is None:
                    raise RuntimeError(f"Failed to load RLModule for {p_id}")

                obs_dim = int(np.prod(agent.observation_space.shape))
                policies[agent_id] = RLlibPPOPolicyBridge(
                    module=module,
                    obs_dim=obs_dim,
                    action_space=agent.action_space,
                )
        return policies
    finally:
        # Keep algo attached to policy bridges during event-driven run.
        # Caller must release it via _cleanup_event_driven_inference(env).
        env._rllib_algo_for_event_driven = algo
        env._ray_shutdown_after_event_driven = should_shutdown


def _cleanup_event_driven_inference(env: ChargingEnv) -> None:
    algo = getattr(env, "_rllib_algo_for_event_driven", None)
    if algo is not None:
        algo.stop()
        env._rllib_algo_for_event_driven = None

    should_shutdown = bool(getattr(env, "_ray_shutdown_after_event_driven", False))
    if should_shutdown:
        import ray

        if ray.is_initialized():
            ray.shutdown()
        env._ray_shutdown_after_event_driven = False

def run_event_driven_sim(config: Dict[str, Any], checkpoint_path: str | None = None) -> None:
    env_specs = config.get("env_specs", {})
    nondeterm = config.get("nondeterminism", {})
    output_dir = resolve_output_dir(config)

    env = create_charging_env(env_specs)
    try:
        configure_schedule_configs(env, nondeterm, seed=42)

        episode_analyzer = EpisodeAnalyzer(verbose=False, track_data=True)
        t_end_val = float(env_specs.get("episode_length"))
        episode = env.run_event_driven(episode_analyzer=episode_analyzer, t_end=t_end_val)

        reward_history = episode_analyzer.get_reward_history()
        reward_totals = {
            agent_id: float(sum(r for _, r in rewards))
            for agent_id, rewards in reward_history.items()
            if "station" in agent_id and "slot" not in agent_id
        }

        summary = episode.summary()
        payload = {
            "summary": summary,
            "reward_totals": reward_totals,
            "station_jitters": nondeterm.get("station_specific", {}),
            "timestamp": _utc_now()
        }
        _write_json(output_dir / "event_driven_summary.json", payload)

        logger.info(f"Final Rewards: {reward_totals}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Config-driven RL Training & Simulation")

    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to YAML config file (default: charging_config.yaml)"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "event-driven", "both"],
        default="both",
        help="Mode to run (default: both)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for event-driven inference (default: use output_dir/best_checkpoint.json)",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    full_config = load_config(str(config_path))

    train_result: Dict[str, Any] = {}
    if args.mode in {"train", "both"}:
        train_result = train_rllib_ppo(full_config)

    if args.mode in {"event-driven", "both"}:
        checkpoint_for_event_driven = args.checkpoint
        if args.mode == "both" and not checkpoint_for_event_driven:
            checkpoint_for_event_driven = train_result.get("best_checkpoint")
        run_event_driven_sim(full_config, checkpoint_path=checkpoint_for_event_driven)


if __name__ == "__main__":
    main()
