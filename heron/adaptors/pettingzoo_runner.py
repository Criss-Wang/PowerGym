"""Training and event-driven evaluation runner for PettingZoo-wrapped HERON envs.

Provides utilities to bridge the PettingZoo Parallel API with standard
PyTorch training loops (IPPO / shared-parameter PPO) and HERON's
event-driven evaluation mode — the PettingZoo analogue of
``heron.adaptors.rllib_runner.HeronEnvRunner``.

Usage — training::

    from heron.adaptors.pettingzoo import pettingzoo_env
    from heron.adaptors.pettingzoo_runner import IPPOTrainer

    env = pettingzoo_env("TwoRoomHeating-v0", max_steps=100)
    trainer = IPPOTrainer(env, hidden_dim=64, lr=3e-4)
    metrics = trainer.train(num_episodes=200)
    print(metrics[-1])  # last episode metrics

Usage — event-driven evaluation with trained policies::

    from heron.adaptors.pettingzoo_runner import evaluate_event_driven

    results = evaluate_event_driven(
        heron_env=trainer.env._heron_env,
        policies=trainer.get_policies(),
        t_end=100.0,
    )
    print(results)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from heron.core.policies import Policy
from heron.envs.base import BaseEnv
from heron.scheduling.analysis import EpisodeAnalyzer


# =========================================================================
# Lightweight actor-critic for continuous actions
# =========================================================================

class ContinuousActorCritic(nn.Module):
    """Minimal Gaussian actor-critic for continuous action spaces."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor):
        mean = self.actor_mean(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(obs).squeeze(-1)
        return dist, value

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample an action (no grad)."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist, _ = self(obs_t)
            action = dist.sample().squeeze(0).numpy()
        return action


# =========================================================================
# Episode metrics
# =========================================================================

@dataclass
class EpisodeMetrics:
    """Per-episode metrics returned by the trainer."""

    episode: int = 0
    total_reward: float = 0.0
    per_agent_reward: Dict[str, float] = field(default_factory=dict)
    episode_length: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0


# =========================================================================
# IPPO Trainer
# =========================================================================

class IPPOTrainer:
    """Independent PPO trainer for PettingZoo Parallel envs.

    Each agent gets its own ``ContinuousActorCritic`` network (IPPO).
    Set ``shared_params=True`` for a single shared policy (MAPPO-like).

    This is intentionally minimal — enough to validate the adaptor's
    training loop correctness, not a production MARL framework.
    """

    def __init__(
        self,
        env: Any,  # PettingZooParallelEnv
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        shared_params: bool = False,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps

        # Build per-agent networks
        self.models: Dict[str, ContinuousActorCritic] = {}
        self.optimizers: Dict[str, torch.optim.Adam] = {}

        obs, _ = env.reset(seed=0)
        shared_model = None

        for aid in env.possible_agents:
            obs_dim = env.observation_space(aid).shape[0]
            act_dim = env.action_space(aid).shape[0]

            if shared_params:
                if shared_model is None:
                    shared_model = ContinuousActorCritic(obs_dim, act_dim, hidden_dim)
                self.models[aid] = shared_model
            else:
                self.models[aid] = ContinuousActorCritic(obs_dim, act_dim, hidden_dim)

            if aid not in self.optimizers:
                self.optimizers[aid] = torch.optim.Adam(
                    self.models[aid].parameters(), lr=lr,
                )

    def train(
        self,
        num_episodes: int = 100,
        log_interval: int = 20,
        verbose: bool = False,
    ) -> List[EpisodeMetrics]:
        """Train for *num_episodes* and return per-episode metrics."""
        all_metrics: List[EpisodeMetrics] = []

        for ep in range(num_episodes):
            metrics = self._run_episode()
            metrics.episode = ep
            all_metrics.append(metrics)

            if verbose and (ep + 1) % log_interval == 0:
                recent = all_metrics[-log_interval:]
                avg_r = np.mean([m.total_reward for m in recent])
                avg_len = np.mean([m.episode_length for m in recent])
                print(
                    f"  Episode {ep + 1}/{num_episodes}: "
                    f"avg_reward={avg_r:.2f}, avg_len={avg_len:.0f}"
                )

        return all_metrics

    def _run_episode(self) -> EpisodeMetrics:
        """Collect one episode of data and update all policies."""
        obs, _ = self.env.reset()

        # Per-agent trajectory buffers
        buffers: Dict[str, dict] = {
            aid: {"obs": [], "act": [], "logp": [], "rew": [], "val": []}
            for aid in self.env.possible_agents
        }

        total_reward = 0.0
        per_agent_reward = {aid: 0.0 for aid in self.env.possible_agents}
        steps = 0

        while self.env.agents:
            actions = {}
            for aid in self.env.agents:
                obs_t = torch.as_tensor(obs[aid], dtype=torch.float32).unsqueeze(0)
                model = self.models[aid]
                dist, value = model(obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)

                act_np = action.squeeze(0).detach().numpy()
                # Clip to action space bounds
                low = self.env.action_space(aid).low
                high = self.env.action_space(aid).high
                act_np = np.clip(act_np, low, high)

                actions[aid] = act_np
                buffers[aid]["obs"].append(obs_t.squeeze(0))
                buffers[aid]["act"].append(action.squeeze(0))
                buffers[aid]["logp"].append(log_prob.squeeze(0))
                buffers[aid]["val"].append(value.squeeze(0))

            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

            for aid in list(actions.keys()):
                r = rewards.get(aid, 0.0)
                buffers[aid]["rew"].append(r)
                total_reward += r
                per_agent_reward[aid] += r

            obs = next_obs
            steps += 1

        # PPO update for each agent
        total_ploss = 0.0
        total_vloss = 0.0

        for aid in self.env.possible_agents:
            buf = buffers[aid]
            if not buf["obs"]:
                continue

            ploss, vloss = self._ppo_update(aid, buf)
            total_ploss += ploss
            total_vloss += vloss

        n = len(self.env.possible_agents)
        return EpisodeMetrics(
            total_reward=total_reward,
            per_agent_reward=per_agent_reward,
            episode_length=steps,
            policy_loss=total_ploss / max(n, 1),
            value_loss=total_vloss / max(n, 1),
        )

    def _ppo_update(self, aid: str, buf: dict) -> tuple[float, float]:
        """Single PPO update for one agent's trajectory."""
        obs_t = torch.stack(buf["obs"])
        act_t = torch.stack(buf["act"])
        old_logp = torch.stack(buf["logp"]).detach()
        old_val = torch.stack(buf["val"]).detach()
        rewards = buf["rew"]

        # GAE
        T = len(rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = old_val[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + self.gamma * next_val - old_val[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = gae + old_val[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipped objective
        model = self.models[aid]
        dist, value = model(obs_t)
        new_logp = dist.log_prob(act_t).sum(-1)
        ratio = (new_logp - old_logp).exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(value, returns)
        loss = policy_loss + 0.5 * value_loss

        optimizer = self.optimizers[aid]
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        return policy_loss.item(), value_loss.item()

    def get_policies(self) -> Dict[str, "_NNPolicy"]:
        """Return a dict of agent_id -> HERON Policy for event-driven eval.

        Each policy wraps the trained PyTorch model and handles
        observation format conversion between PettingZoo and HERON's
        event-driven mode.
        """
        policies = {}
        for aid, model in self.models.items():
            obs_dim = self.env.observation_space(aid).shape[0]
            act_dim = self.env.action_space(aid).shape[0]
            low = float(self.env.action_space(aid).low[0])
            high = float(self.env.action_space(aid).high[0])

            def make_fn(m, expected_dim):
                def policy_fn(obs: np.ndarray) -> np.ndarray:
                    obs = np.asarray(obs, dtype=np.float32).flatten()
                    # Pad or truncate to match training obs dim
                    if len(obs) < expected_dim:
                        obs = np.pad(obs, (0, expected_dim - len(obs)))
                    elif len(obs) > expected_dim:
                        obs = obs[:expected_dim]
                    return m.get_action(obs)
                return policy_fn

            policy = _NNPolicy(make_fn(model, obs_dim))
            policy.obs_dim = obs_dim
            policy.action_dim = act_dim
            policy.action_range = (low, high)
            policies[aid] = policy
        return policies


# =========================================================================
# Event-driven evaluation
# =========================================================================

class _NNPolicy(Policy):
    """Bridges a trained PyTorch model to HERON's Policy interface for event-driven eval."""

    observation_mode = "local"

    def __init__(self, policy_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._fn = policy_fn
        self.obs_dim = 0  # auto-detected on first call
        self.action_dim = 1
        self.action_range = (-1.0, 1.0)

    def forward(self, observation: Any) -> Any:
        from heron.core.observation import Observation

        obs_vec = self.extract_obs_vector(observation, self.obs_dim) if self.obs_dim > 0 else self._extract_raw(observation)
        action_vec = self._fn(obs_vec)
        return self.vec_to_action(action_vec, len(action_vec), self.action_range)

    def _extract_raw(self, observation: Any) -> np.ndarray:
        """Extract raw vector when obs_dim is not yet known."""
        from heron.core.observation import Observation

        if isinstance(observation, Observation):
            vec = observation.local_vector()
            self.obs_dim = len(vec)
            return vec
        elif isinstance(observation, np.ndarray):
            self.obs_dim = len(observation)
            return observation
        elif isinstance(observation, dict):
            parts = []
            for k in sorted(observation.keys()):
                v = observation[k]
                if isinstance(v, np.ndarray):
                    parts.append(v.flatten())
                elif isinstance(v, dict):
                    for k2 in sorted(v.keys()):
                        arr = np.asarray(v[k2], dtype=np.float32).flatten()
                        parts.append(arr)
            if parts:
                vec = np.concatenate(parts).astype(np.float32)
                self.obs_dim = len(vec)
                return vec
        return np.zeros(2, dtype=np.float32)


def evaluate_event_driven(
    heron_env: BaseEnv,
    policies: Dict[str, Any],
    t_end: float = 100.0,
    seed: Optional[int] = 42,
    jitter_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run event-driven evaluation with trained policies on the HERON env.

    This is the PettingZoo analogue of the RLlib runner's event-driven
    evaluation mode — it bridges trained PyTorch policies back into HERON's
    native event-driven simulation.

    Args:
        heron_env: The underlying HERON BaseEnv (access via ``pz_env._heron_env``).
        policies: Dict of ``{agent_id: Policy or callable}``.
            If from ``IPPOTrainer.get_policies()``, these are already
            ``_NNPolicy`` instances ready for event-driven use.
        t_end: Simulation end time.
        seed: Reset seed.
        jitter_seed: Jitter seed for event-driven scheduling.

    Returns:
        Dict with ``terminated``, ``truncated``, ``num_events``,
        ``per_agent_rewards``, ``total_reward``.
    """
    # Install policies on the HERON agents
    for aid, policy in policies.items():
        agent = heron_env.get_agent(aid)
        if agent is not None:
            if isinstance(policy, Policy):
                agent.policy = policy
            else:
                agent.policy = _NNPolicy(policy)

    # Reset and run event-driven
    heron_env.reset(seed=seed, jitter_seed=jitter_seed)
    analyzer = EpisodeAnalyzer()
    stats = heron_env.run_event_driven(t_end=t_end, episode_analyzer=analyzer)

    # Collect per-agent reward summaries from the analyzer
    reward_history = analyzer.get_reward_history()
    per_agent_rewards = {}
    for aid in policies:
        entries = reward_history.get(aid, [])
        per_agent_rewards[aid] = sum(r for _, r in entries)

    total_reward = sum(per_agent_rewards.values())

    return {
        "terminated": stats.terminated,
        "truncated": stats.truncated,
        "num_events": stats.num_events,
        "per_agent_rewards": per_agent_rewards,
        "total_reward": total_reward,
    }
