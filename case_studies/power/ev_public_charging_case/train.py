"""
Unified RLlib training + event-driven runner for EV public charging.
Style: Fully Config-driven via YAML (Supporting per-agent schedule overrides).
"""

from __future__ import annotations

import argparse
import csv
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

from case_studies.power.ev_public_charging_case.agents import ChargerAgent, StationCoordinator
from case_studies.power.ev_public_charging_case.envs.charging_env import ChargingEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _to_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        casted = float(value)
        return casted if math.isfinite(casted) else None
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_metric_from_nested(obj: Any, key: str) -> List[float]:
    values: List[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                as_float = _to_optional_float(v)
                if as_float is not None:
                    values.append(as_float)
            values.extend(_extract_metric_from_nested(v, key))
    elif isinstance(obj, list):
        for v in obj:
            values.extend(_extract_metric_from_nested(v, key))
    return values


def _get_episode_len_mean(result: Dict[str, Any]) -> float | None:
    candidates = [
        result.get("env_runners", {}).get("episode_len_mean"),
        result.get("episode_len_mean"),
        result.get("sampler_results", {}).get("episode_len_mean"),
    ]
    for value in candidates:
        as_float = _to_optional_float(value)
        if as_float is not None:
            return as_float
    return None


def _get_loss_means(result: Dict[str, Any]) -> Dict[str, float | None]:
    policy_losses = _extract_metric_from_nested(result, "policy_loss")
    vf_losses = _extract_metric_from_nested(result, "vf_loss")
    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else None,
        "vf_loss": float(np.mean(vf_losses)) if vf_losses else None,
    }


def _plot_training_metrics(train_rows: List[Dict[str, Any]], output_dir: Path) -> Dict[str, str]:
    if not train_rows:
        return {}

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping training plots")
        return {}

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    reward_points = [
        (int(r["iteration"]), float(r["reward_mean"]))
        for r in train_rows
        if _to_optional_float(r.get("reward_mean")) is not None
    ]
    outputs: Dict[str, str] = {}
    if reward_points:
        xs = [x for x, _ in reward_points]
        ys = [y for _, y in reward_points]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker="o", linewidth=1.5)
        plt.title("Training Reward Mean vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Episode Reward Mean")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        reward_fig = plots_dir / "training_reward_curve.png"
        plt.savefig(reward_fig, dpi=150)
        plt.close()
        outputs["training_reward_curve"] = str(reward_fig)

    loss_points = [
        (
            int(r["iteration"]),
            _to_optional_float(r.get("policy_loss")),
            _to_optional_float(r.get("vf_loss")),
        )
        for r in train_rows
    ]
    if any((p is not None) or (v is not None) for _, p, v in loss_points):
        xs = [it for it, _, _ in loss_points]
        p_losses = [np.nan if p is None else p for _, p, _ in loss_points]
        v_losses = [np.nan if v is None else v for _, _, v in loss_points]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, p_losses, marker=".", linewidth=1.3, label="policy_loss")
        plt.plot(xs, v_losses, marker=".", linewidth=1.3, label="vf_loss")
        plt.title("Training Loss Curves")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        loss_fig = plots_dir / "training_loss_curve.png"
        plt.savefig(loss_fig, dpi=150)
        plt.close()
        outputs["training_loss_curve"] = str(loss_fig)

    return outputs


def _save_event_reward_artifacts(
    reward_history: Dict[str, List[Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping event-driven plots")
        plt = None

    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    station_history = {
        agent_id: rewards
        for agent_id, rewards in reward_history.items()
        if "station" in agent_id and "_charger_" not in agent_id
    }

    rows: List[Dict[str, Any]] = []
    reward_totals: Dict[str, float] = {}
    cumulative_history: Dict[str, List[tuple[float, float]]] = {}
    for agent_id, rewards in station_history.items():
        total = 0.0
        trajectory: List[tuple[float, float]] = []
        for timestamp, reward in rewards:
            ts = float(timestamp)
            r = float(reward)
            total += r
            trajectory.append((ts, total))
            rows.append(
                {
                    "agent_id": agent_id,
                    "timestamp": ts,
                    "reward": r,
                    "cumulative_reward": total,
                }
            )
        reward_totals[agent_id] = total
        cumulative_history[agent_id] = trajectory

    csv_path = metrics_dir / "event_reward_timeseries.csv"
    _write_csv(
        csv_path,
        rows=rows,
        fieldnames=["agent_id", "timestamp", "reward", "cumulative_reward"],
    )

    artifacts: Dict[str, Any] = {
        "reward_totals": reward_totals,
        "event_reward_timeseries_csv": str(csv_path),
    }

    if plt is not None and cumulative_history:
        plt.figure(figsize=(9, 4.5))
        for agent_id, series in sorted(cumulative_history.items()):
            if not series:
                continue
            xs = [x for x, _ in series]
            ys = [y for _, y in series]
            plt.plot(xs, ys, linewidth=1.5, label=agent_id)
        plt.title("Event-Driven Cumulative Reward by Station")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Cumulative Reward")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        line_fig = plots_dir / "event_reward_timeseries.png"
        plt.savefig(line_fig, dpi=150)
        plt.close()
        artifacts["event_reward_timeseries_plot"] = str(line_fig)

        names = sorted(reward_totals.keys())
        vals = [reward_totals[name] for name in names]
        plt.figure(figsize=(8, 4.5))
        plt.bar(names, vals)
        plt.title("Event-Driven Total Reward by Station")
        plt.xlabel("Station")
        plt.ylabel("Total Reward")
        plt.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=20)
        plt.tight_layout()
        bar_fig = plots_dir / "event_reward_totals_bar.png"
        plt.savefig(bar_fig, dpi=150)
        plt.close()
        artifacts["event_reward_totals_bar_plot"] = str(bar_fig)

    return artifacts

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
    nondet = config.get("nondeterminism", {})

    parts = [
        f"s{env_specs.get('num_stations', 0)}",
        f"c{env_specs.get('num_chargers', 0)}",
        f"arr{env_specs.get('arrival_rate', 0.0):g}"
    ]

    spec_configs = nondet.get("station_specific", {})
    if spec_configs:
        intervals = [v.get("tick_interval") for v in spec_configs.values()]
        unique_intervals = sorted(list(set(intervals)))

        if len(unique_intervals) > 1:
            parts.append(f"ticks{min(unique_intervals):g}-{max(unique_intervals):g}")
        else:
            parts.append(f"tick{unique_intervals[0]:g}")
    else:
        default_tick = nondet.get("default_config", {}).get("tick_interval", 0)
        parts.append(f"tick{default_tick:g}")

    jit = nondet.get("default_config", {}).get("jitter_ratio", 0.0)
    parts.append(f"jit{jit:g}")

    parts.append(f"seed{training.get('seed', 0)}")
    parts.append(f"num_iterations{training.get('num_iterations', 0)}")

    base_label = "_".join(parts)

    fingerprint = _config_fingerprint(config)[:6]
    return f"{base_label}_{_utc_compact_now()}_{fingerprint}"

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


def _validate_env_specs_strict(env_specs: Dict[str, Any]) -> None:
    """Validate env_specs in strict charger-only mode."""
    required_keys = {
        "num_stations",
        "num_chargers",
        "charger_p_max_kw",
        "arrival_rate",
        "dt",
        "episode_length",
    }
    allowed_keys = set(required_keys)

    present_keys = set(env_specs.keys())
    missing = sorted(required_keys - present_keys)
    unknown = sorted(present_keys - allowed_keys)

    if missing:
        raise ValueError(f"env_specs missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"env_specs has unsupported keys in strict charger-only mode: {', '.join(unknown)}")

def create_charging_env(env_specs: Dict[str, Any]) -> ChargingEnv:
    _validate_env_specs_strict(env_specs)
    num_stations = int(env_specs["num_stations"])
    num_chargers = int(env_specs["num_chargers"])
    charger_p_max_kw = float(env_specs["charger_p_max_kw"])

    coordinators: List[StationCoordinator] = []
    for i in range(num_stations):
        station_id = f"station_{i}"
        charger_agents = {
            f"{station_id}_charger_{j}": ChargerAgent(agent_id=f"{station_id}_charger_{j}", p_max_kw=charger_p_max_kw)
            for j in range(num_chargers)
        }
        coordinators.append(StationCoordinator(agent_id=station_id, subordinates=charger_agents))

    return ChargingEnv(
        coordinator_agents=coordinators,
        arrival_rate=float(env_specs["arrival_rate"]),
        dt=float(env_specs["dt"]),
        episode_length=float(env_specs["episode_length"]),
    )


def build_rllib_env_config(env_specs: Dict[str, Any], nondeterminism: Dict[str, Any]) -> Dict[str, Any]:
    """Build the flat env_config expected by RLlibBasedHeronEnv."""
    _validate_env_specs_strict(env_specs)
    num_stations = int(env_specs["num_stations"])
    num_chargers = int(env_specs["num_chargers"])
    charger_p_max_kw = float(env_specs["charger_p_max_kw"])

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

        charger_agent_ids: List[str] = []
        for j in range(num_chargers):
            charger_agent_id = f"{station_id}_charger_{j}"
            charger_agent_ids.append(charger_agent_id)
            agents.append(
                {
                    "agent_id": charger_agent_id,
                    "agent_cls": ChargerAgent,
                    "coordinator": station_id,
                    "p_max_kw": charger_p_max_kw,
                }
            )

        coordinators.append(
            {
                "coordinator_id": station_id,
                "agent_cls": StationCoordinator,
                "subordinates": charger_agent_ids,
                "schedule_config": _schedule_from_spec(station_spec, seed=42 + i),
            }
        )

    return {
        "env_class": ChargingEnv,
        "env_kwargs": {
            "arrival_rate": float(env_specs["arrival_rate"]),
            "dt": float(env_specs["dt"]),
            "episode_length": float(env_specs["episode_length"]),
        },
        "agents": agents,
        "coordinators": coordinators,
        "agent_ids": trainable_agent_ids,
        "max_steps": int(float(env_specs["episode_length"]) // float(env_specs["dt"])),
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

        ray.init(local_mode=True)

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
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies=policy_ids,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_st_{agent_id.split('_')[1]}"
        )
    )

    ray.init(ignore_reinit_error=True)
    algo = ppo_cfg.build_algo()

    best_reward = float("-inf")
    best_ckpt_path = None
    train_history: List[Dict[str, Any]] = []

    for iteration in range(1, train_params.get("num_iterations", 50) + 1):
        result = algo.train()
        reward_mean = _get_mean_reward(result)
        episode_len_mean = _get_episode_len_mean(result)
        loss_means = _get_loss_means(result)
        timesteps_total = result.get("timesteps_total")

        train_history.append(
            {
                "iteration": iteration,
                "reward_mean": reward_mean if reward_mean == reward_mean else None,
                "episode_len_mean": episode_len_mean,
                "timesteps_total": timesteps_total,
                "policy_loss": loss_means["policy_loss"],
                "vf_loss": loss_means["vf_loss"],
            }
        )

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

    metrics_dir = output_dir / "metrics"
    train_csv = metrics_dir / "train_metrics.csv"
    _write_csv(
        train_csv,
        rows=train_history,
        fieldnames=["iteration", "reward_mean", "episode_len_mean", "timesteps_total", "policy_loss", "vf_loss"],
    )
    plots = _plot_training_metrics(train_history, output_dir)
    _write_json(
        metrics_dir / "train_metrics_meta.json",
        {
            "timestamp": _utc_now(),
            "train_metrics_csv": str(train_csv),
            "plots": plots,
        },
    )

    algo.stop()
    ray.shutdown()
    return {
        "status": "completed",
        "best_checkpoint": best_ckpt_path,
        "best_reward_mean": best_checkpoint_meta["best_reward_mean"],
        "plots": plots,
        "train_metrics_csv": str(train_csv),
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
    num_stations = int(env_specs.get("num_stations", 0))
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
            policies={f"policy_st_{i}" for i in range(num_stations)},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_st_{agent_id.split('_')[1]}",
        )
        .build_algo()
    )

    try:
        algo.restore(checkpoint_path)
        policies: Dict[str, Policy] = {}

        for agent_id, agent in env.registered_agents.items():
            if isinstance(agent, StationCoordinator):
                p_id = f"policy_st_{agent_id.split('_')[1]}"
                module = None
                attempted_ids = [p_id, "station_policy"]
                for candidate in attempted_ids:
                    try:
                        module = algo.get_module(candidate)
                    except Exception:
                        module = None
                    if module is not None:
                        break

                if module is None:
                    raise RuntimeError(
                        f"Failed to load RLModule for station '{agent_id}'. Tried ids: {attempted_ids}"
                    )

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

        resolved_checkpoint = _resolve_best_checkpoint(config, explicit_checkpoint=checkpoint_path)
        if resolved_checkpoint:
            policies = _build_event_driven_policies_from_checkpoint(
                env=env,
                config=config,
                checkpoint_path=resolved_checkpoint,
            )
            env.set_agent_policies(policies)
            logger.info(f"Loaded RLlib checkpoint for event-driven run: {resolved_checkpoint}")
        else:
            logger.warning("No checkpoint found for event-driven run; using default agent policies.")

        episode_analyzer = EpisodeAnalyzer(verbose=True, track_data=True)
        t_end_val = float(env_specs.get("episode_length"))
        episode = env.run_event_driven(episode_analyzer=episode_analyzer, t_end=t_end_val)

        reward_history = episode_analyzer.get_reward_history()
        reward_artifacts = _save_event_reward_artifacts(reward_history, output_dir)
        reward_totals = reward_artifacts.get("reward_totals", {})

        summary = episode.summary()
        payload = {
            "summary": summary,
            "reward_totals": reward_totals,
            "station_jitters": nondeterm.get("station_specific", {}),
            "checkpoint_path": resolved_checkpoint,
            "artifacts": reward_artifacts,
            "timestamp": _utc_now()
        }
        _write_json(output_dir / "event_driven_summary.json", payload)

        logger.info(f"Final Rewards: {reward_totals}")
    finally:
        _cleanup_event_driven_inference(env)
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

# TODO: track each reward event
if __name__ == "__main__":
    main()
