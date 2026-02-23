import numpy as np

from case_studies.power.ev_public_charging_case.env.charging_env import SimpleChargingEnv


class EpsilonGreedyPriceBandit:
    """
    Per-station epsilon-greedy bandit over discrete price actions.

    - Maintains Q(a) and N(a) for each station independently.
    - Chooses:
        with prob epsilon: random action
        else: argmax Q(a) (ties broken randomly)
    - Updates Q using incremental mean:
        Q <- Q + (r - Q) / N
    """

    def __init__(
        self,
        price_grid,
        epsilon=0.1,
        seed=0,
        optimistic_init=0.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.price_grid = np.array(price_grid, dtype=np.float32)
        assert self.price_grid.ndim == 1 and len(self.price_grid) > 0

        self.epsilon = float(epsilon)

        # station_id -> Q-values / counts
        self.Q = {}  # dict[str, np.ndarray]
        self.N = {}  # dict[str, np.ndarray]
        self.last_action_idx = {}  # dict[str, int]

        self.optimistic_init = float(optimistic_init)

    def _ensure_station(self, station_id: str):
        if station_id not in self.Q:
            self.Q[station_id] = np.full(
                shape=(len(self.price_grid),),
                fill_value=self.optimistic_init,
                dtype=np.float32,
            )
            self.N[station_id] = np.zeros(
                shape=(len(self.price_grid),),
                dtype=np.int32,
            )
            self.last_action_idx[station_id] = 0

    def act(self, obs_dict):
        """
        obs_dict: dict[agent_id -> obs]
        returns: dict[agent_id -> np.ndarray action]
        """
        actions = {}
        for station_id in obs_dict.keys():
            self._ensure_station(station_id)

            if self.rng.random() < self.epsilon:
                # explore
                a_idx = int(self.rng.integers(0, len(self.price_grid)))
            else:
                # exploit (random tie-break)
                q = self.Q[station_id]
                best = np.flatnonzero(q == q.max())
                a_idx = int(self.rng.choice(best))

            self.last_action_idx[station_id] = a_idx
            price = float(self.price_grid[a_idx])
            actions[station_id] = np.array([price], dtype=np.float32)

        return actions

    def update(self, rewards_dict):
        """
        rewards_dict: dict[agent_id -> float]
        Uses reward as immediate bandit feedback for last chosen action.
        """
        for station_id, r in rewards_dict.items():
            self._ensure_station(station_id)
            a_idx = self.last_action_idx[station_id]

            self.N[station_id][a_idx] += 1
            n = self.N[station_id][a_idx]
            q = self.Q[station_id][a_idx]

            # incremental mean update
            self.Q[station_id][a_idx] = q + (float(r) - float(q)) / float(n)

    def summary(self):
        out = {}
        for sid in self.Q:
            best_idx = int(np.argmax(self.Q[sid]))
            out[sid] = {
                "best_price": float(self.price_grid[best_idx]),
                "best_Q": float(self.Q[sid][best_idx]),
                "counts": self.N[sid].tolist(),
                "Q": [float(x) for x in self.Q[sid]],
            }
        return out


def main():
    # ----- env -----
    env = SimpleChargingEnv(arrival_rate=100.0, num_stations=2, charger_num=2)

    obs, info = env.reset(seed=0)
    station_ids = list(obs.keys())
    print("Stations:", station_ids)

    # ----- epsilon-greedy bandit policy -----
    price_grid = np.linspace(0.1, 1.0, 1000)

    policy = EpsilonGreedyPriceBandit(
        price_grid=price_grid,
        epsilon=0.2,
        seed=0,
        optimistic_init=0.2,
    )

    # ----- rollout -----
    total_rewards = 0.0

    for t in range(1, 2001):
        actions = policy.act(obs)
        obs, rewards, terminated, truncated, infos = env.step(actions)

        policy.update(rewards)

        for sid, r in rewards.items():
            total_rewards += float(r)

        if t % 20 == 0:
            print(f"\nStep {t}")
            print("  last actions:", {k: float(v[0]) for k, v in actions.items()})
            print("  step rewards:", rewards)
            print("  total rewards:", total_rewards)

            print("  policy summary:", policy.summary())

        if any(terminated.values()) or any(truncated.values()):
            print("Episode ended.")
            break

    env.close()


if __name__ == "__main__":
    main()