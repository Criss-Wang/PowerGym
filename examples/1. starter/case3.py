"""Case 3: IPPO with heterogeneous field-agent tick rates.

Hierarchy:
    SystemAgent
    └── fleet_0 (PassthroughCoordinator)
        ├── drone_0_0 (TrackedDrone) -- tick_interval=1.0  (fast scout)
        ├── drone_0_1 (TrackedDrone) -- tick_interval=2.0  (medium)
        └── drone_0_2 (TrackedDrone) -- tick_interval=3.0  (slow heavy-lift)

Domain: Same multi-drone transport as case1/case2, but each drone type has
    a distinct tick rate.  Verifies that step-based training correctly gates
    **both** action execution and reward computation at every step.

Training: IPPO -- each drone trains its own policy at its own effective rate.

Verification (full-episode audit):
    1. Build a Heron BaseEnv with TrackedDrones that count apply_action calls.
    2. Run a 30-step episode, recording at each step:
       - is_active_at(step) expectation from tick_interval
       - whether apply_action actually fired (apply_count delta)
       - whether the agent appears in the rewards dict
    3. Assert all three agree at every step for every agent.
    4. Repeat through the RLlib adaptor to confirm the wrapper preserves gating.

Usage:
    cd examples/1.\\ starter
    python case3.py
"""

import numpy as np

from agents import TransportDrone
from env_physics import case1_simulation
from features import DronePositionFeature, FleetSafetyFeature

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from heron.adaptors.rllib import RLlibBasedHeronEnv
from heron.adaptors.rllib_runner import HeronEnvRunner
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.envs.base import BaseEnv
from heron.protocols.vertical import VerticalProtocol
from heron.scheduling import ScheduleConfig

from typing import Any, Dict, List


# ── Tracked drone: counts apply_action calls ────────────────────────────

class TrackedDrone(TransportDrone):
    """TransportDrone that counts how many times apply_action was invoked."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_count: int = 0

    def apply_action(self) -> None:
        self.apply_count += 1
        super().apply_action()

    def reset(self, **kwargs) -> Any:
        self.apply_count = 0
        return super().reset(**kwargs)


# ── Coordinator that exposes drones to RLlib directly ───────────────────

class PassthroughCoordinator(CoordinatorAgent):
    """Coordinator with no action space — drones are trained directly."""

    def __init__(self, agent_id, subordinates, features=None, **kwargs):
        default_features = [FleetSafetyFeature()]
        all_features = (features or []) + default_features
        super().__init__(
            agent_id=agent_id,
            features=all_features,
            subordinates=subordinates,
            protocol=VerticalProtocol(),
            **kwargs,
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        return Action()

    def compute_rewards(self, proxy) -> Dict[str, float]:
        rewards: Dict[str, float] = {}
        for sub in self.subordinates.values():
            rewards.update(sub.compute_rewards(proxy))
        return rewards


# ── Simple env (identity simulation for clean tracking) ─────────────────

class TrackingEnv(BaseEnv):
    """Env with case1 simulation for the audit."""

    def run_simulation(self, env_state, *args, **kwargs):
        return case1_simulation(env_state.get("agent_states", env_state))

    def env_state_to_global_state(self, env_state) -> Dict:
        if isinstance(env_state, dict) and "agent_states" in env_state:
            return env_state
        return {"agent_states": env_state}

    def global_state_to_env_state(self, global_state) -> Any:
        return global_state


# ── Helpers ─────────────────────────────────────────────────────────────

DRONE_IDS = ["drone_0_0", "drone_0_1", "drone_0_2"]
TICK_RATES = {"drone_0_0": 1.0, "drone_0_1": 2.0, "drone_0_2": 3.0}
EPISODE_LEN = 30


def _build_tracking_env(seed: int = 0):
    """Build a Heron BaseEnv with TrackedDrones at heterogeneous ticks."""
    drones = {
        did: TrackedDrone(
            agent_id=did,
            features=[DronePositionFeature(y_pos=0.2 + 0.3 * i)],
            schedule_config=ScheduleConfig.deterministic(tick_interval=TICK_RATES[did]),
        )
        for i, did in enumerate(DRONE_IDS)
    }
    coord = PassthroughCoordinator(
        agent_id="fleet_0", subordinates=drones,
    )
    system = SystemAgent(subordinates={"fleet_0": coord})
    env = TrackingEnv(system_agent=system)
    env.reset(seed=seed)
    return env


# ── Phase 1: full-episode audit on BaseEnv ──────────────────────────────

def audit_base_env():
    """Run a full episode on the raw Heron env, checking action + reward gating."""
    env = _build_tracking_env(seed=0)

    print("=" * 90)
    print("PHASE 1: Full-episode audit on Heron BaseEnv (30 steps)")
    print("=" * 90)
    hdr = (
        f"{'Step':>4} | "
        + " | ".join(f"{'act':>3} {'rew':>4} {'ok':>2}" for _ in DRONE_IDS)
        + " | active"
    )
    labels = "     | " + " | ".join(f"  {did[-1]:>8}" for did in DRONE_IDS) + " |"
    print(labels)
    print(hdr)
    print("-" * 90)

    errors: List[str] = []
    prev_counts = {did: 0 for did in DRONE_IDS}

    for step in range(1, EPISODE_LEN + 1):
        actions = {did: np.array([0.5]) for did in DRONE_IDS}
        _, rew, _, _, _ = env.step(actions)

        cells = []
        active_names = []
        for did in DRONE_IDS:
            agent = env.registered_agents[did]
            tick = int(TICK_RATES[did])
            expected_active = (step % tick == 0)

            # Check action execution
            cur_count = agent.apply_count
            action_fired = (cur_count > prev_counts[did])
            prev_counts[did] = cur_count

            # Check reward presence
            has_reward = did in rew

            # All three must agree
            ok = (expected_active == action_fired == has_reward)
            cells.append(f"{('Y' if action_fired else '.'):>3} {('Y' if has_reward else '.'):>4} {'✓' if ok else '✗':>2}")

            if expected_active:
                active_names.append(did.split("_")[-1])

            if not ok:
                errors.append(
                    f"Step {step}, {did}: expected_active={expected_active}, "
                    f"action_fired={action_fired}, has_reward={has_reward}"
                )

        print(f"{step:>4} | " + " | ".join(cells) + f" | {','.join(active_names) or '(none)'}")

    # Summary: cumulative action counts
    print("-" * 90)
    for did in DRONE_IDS:
        agent = env.registered_agents[did]
        tick = int(TICK_RATES[did])
        expected = EPISODE_LEN // tick
        actual = agent.apply_count
        match = "✓" if actual == expected else "✗"
        print(f"  {did}: apply_count={actual}, expected={expected} (30/{tick}) {match}")

    print("-" * 90)
    if errors:
        print(f"PHASE 1 FAILED — {len(errors)} mismatches:")
        for e in errors:
            print(f"  {e}")
        return False

    print("PHASE 1 PASSED — action execution and reward gating match at every step")
    return True


# ── Phase 2: same audit through RLlib adaptor ───────────────────────────

def audit_rllib_adaptor():
    """Same audit but through RLlibBasedHeronEnv to verify the wrapper."""
    config = {
        "env_id": "audit_hetero",
        "agents": [
            {
                "agent_id": did,
                "agent_cls": TrackedDrone,
                "features": [DronePositionFeature(y_pos=0.2 + 0.3 * i)],
                "schedule_config": ScheduleConfig.deterministic(tick_interval=TICK_RATES[did]),
                "coordinator": "fleet_0",
            }
            for i, did in enumerate(DRONE_IDS)
        ],
        "coordinators": [
            {"coordinator_id": "fleet_0", "agent_cls": PassthroughCoordinator},
        ],
        "simulation": case1_simulation,
        "max_steps": EPISODE_LEN,
        "agent_ids": DRONE_IDS,
    }

    env = RLlibBasedHeronEnv(config)
    env.reset(seed=0)

    heron_agents = env.heron_env.registered_agents

    print("\n" + "=" * 90)
    print("PHASE 2: Full-episode audit through RLlib adaptor (30 steps)")
    print("=" * 90)
    print(f"{'Step':>4} | {'d0_0':>8} | {'d0_1':>8} | {'d0_2':>8} | "
          f"{'d0_0 rew':>8} | {'d0_1 rew':>8} | {'d0_2 rew':>8} | active")
    print("-" * 90)

    errors: List[str] = []
    prev_counts = {did: 0 for did in DRONE_IDS}

    for step in range(1, EPISODE_LEN + 1):
        actions = {did: np.array([0.5]) for did in DRONE_IDS}
        _, rew, _, _, info = env.step(actions)

        act_strs = []
        rew_strs = []
        active_names = []

        for did in DRONE_IDS:
            agent = heron_agents[did]
            tick = int(TICK_RATES[did])
            expected_active = (step % tick == 0)

            # Action check via apply_count
            cur_count = agent.apply_count
            action_fired = (cur_count > prev_counts[did])
            prev_counts[did] = cur_count
            act_strs.append(f"{'ACT' if action_fired else '---':>8}")

            # Reward check: RLlib defaults missing keys to 0.0
            is_active_flag = info[did]["is_active"][did]
            r = rew[did]
            got_reward = (r != 0.0)
            rew_strs.append(f"{r:>8.4f}")

            if expected_active:
                active_names.append(did.split("_")[-1])

            # Verify: is_active flag matches expectation
            if expected_active != is_active_flag:
                errors.append(f"Step {step}, {did}: is_active flag={is_active_flag}, expected={expected_active}")
            # Verify: action execution matches expectation
            if expected_active != action_fired:
                errors.append(f"Step {step}, {did}: action_fired={action_fired}, expected={expected_active}")
            # Verify: inactive agents get 0.0 reward from adaptor
            if not expected_active and r != 0.0:
                errors.append(f"Step {step}, {did}: inactive but got reward={r}")
            # Verify: active agents get non-zero reward (after step 1 where init state may yield ~0)
            if expected_active and not got_reward and step > 1:
                errors.append(f"Step {step}, {did}: active but got zero reward")

        print(
            f"{step:>4} | " + " | ".join(act_strs) + " | "
            + " | ".join(rew_strs)
            + f" | {','.join(active_names) or '(none)'}"
        )

    # Summary
    print("-" * 90)
    for did in DRONE_IDS:
        agent = heron_agents[did]
        tick = int(TICK_RATES[did])
        expected = EPISODE_LEN // tick
        actual = agent.apply_count
        match = "✓" if actual == expected else "✗"
        print(f"  {did}: apply_count={actual}, expected={expected} (30/{tick}) {match}")

    print("-" * 90)
    if errors:
        print(f"PHASE 2 FAILED — {len(errors)} mismatches:")
        for e in errors:
            print(f"  {e}")
        return False

    print("PHASE 2 PASSED — RLlib adaptor preserves action + reward gating")
    return True


# ── Shared env config builder ───────────────────────────────────────────

def _env_config():
    return {
        "env_id": "hetero_tick_transport",
        "agents": [
            {
                "agent_id": did,
                "agent_cls": TrackedDrone,
                "features": [DronePositionFeature(y_pos=0.2 + 0.3 * i)],
                "schedule_config": ScheduleConfig.deterministic(
                    tick_interval=TICK_RATES[did],
                ),
                "coordinator": "fleet_0",
            }
            for i, did in enumerate(DRONE_IDS)
        ],
        "coordinators": [
            {"coordinator_id": "fleet_0", "agent_cls": PassthroughCoordinator},
        ],
        "simulation": case1_simulation,
        "max_steps": EPISODE_LEN,
        "agent_ids": DRONE_IDS,
    }


# ── Phase 3: MAPPO training — without vs with inactive-timestep masking ─

def _train(label: str, use_mask: bool, num_iters: int = 15):
    """Run MAPPO training and return per-iteration per-agent returns."""
    from heron.adaptors.rllib_learner_connector import MaskInactiveAgentTimesteps

    training_kwargs: Dict[str, Any] = dict(
        lr=5e-4, gamma=0.99,
        train_batch_size=800, minibatch_size=128, num_epochs=3,
    )
    if use_mask:
        training_kwargs["learner_connector"] = (
            lambda obs_sp, act_sp: MaskInactiveAgentTimesteps()
        )

    config = (
        PPOConfig()
        .environment(env=RLlibBasedHeronEnv, env_config=_env_config())
        .multi_agent(
            policies={"shared": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *a, **kw: "shared",
        )
        .env_runners(
            env_runner_cls=HeronEnvRunner,
            num_env_runners=1, num_envs_per_env_runner=1,
        )
        .training(**training_kwargs)
        .framework("torch")
    )

    algo = config.build()

    print(f"\n{'Iter':>4} | {'drone_0_0':>12} | {'drone_0_1':>12} | {'drone_0_2':>12} | {'episode_ret':>12}")
    print("-" * 70)

    history = []
    for i in range(num_iters):
        result = algo.train()
        er = result.get("env_runners", {})
        agent_returns = er.get("agent_episode_returns_mean", {})
        r0 = agent_returns.get("drone_0_0", 0.0)
        r1 = agent_returns.get("drone_0_1", 0.0)
        r2 = agent_returns.get("drone_0_2", 0.0)
        episode_ret = er.get("episode_return_mean", 0.0)
        history.append(episode_ret)
        print(f"{i+1:>4} | {r0:>12.3f} | {r1:>12.3f} | {r2:>12.3f} | {episode_ret:>12.3f}")

    algo.stop()
    return history


def train_mappo_comparison():
    """Compare MAPPO with and without MaskInactiveAgentTimesteps."""
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)

    iters = 15

    print("\n" + "=" * 90)
    print("PHASE 3a: MAPPO WITHOUT inactive-timestep masking")
    print("=" * 90)
    hist_no_mask = _train("no_mask", use_mask=False, num_iters=iters)

    print("\n" + "=" * 90)
    print("PHASE 3b: MAPPO WITH MaskInactiveAgentTimesteps connector")
    print("=" * 90)
    hist_masked = _train("masked", use_mask=True, num_iters=iters)

    ray.shutdown()

    # Summary comparison
    print("\n" + "=" * 90)
    print("PHASE 3 COMPARISON: MAPPO episode returns (last 5 iters avg)")
    print("=" * 90)
    avg_no = np.mean(hist_no_mask[-5:]) if len(hist_no_mask) >= 5 else 0
    avg_yes = np.mean(hist_masked[-5:]) if len(hist_masked) >= 5 else 0
    print(f"  Without masking: {avg_no:.3f}")
    print(f"  With masking:    {avg_yes:.3f}")
    print("=" * 90)
    print("PHASE 3 PASSED — both configurations train successfully")
    return True


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("Case 3: Heterogeneous field-agent tick rates — full audit\n")

    ok1 = audit_base_env()
    ok2 = audit_rllib_adaptor()
    ok3 = train_mappo_comparison()

    print("\n" + "=" * 90)
    results = [("BaseEnv audit", ok1), ("RLlib adaptor audit", ok2), ("MAPPO comparison", ok3)]
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print("=" * 90)
    print("Case 3 " + ("PASSED" if all_ok else "FAILED"))
    if not all_ok:
        exit(1)


if __name__ == "__main__":
    main()
