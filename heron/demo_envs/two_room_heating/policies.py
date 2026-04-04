"""Rule-based policies for TwoRoomHeating event-driven evaluation."""

import numpy as np

from heron.core.policies import Policy, obs_to_vector, vector_to_action


class HeaterPolicy(Policy):
    """Proportional controller for a zone heater.

    Observes ``ZoneTemperatureFeature`` (temperature, target) and outputs
    a heating command proportional to the error, clipped to [-1, 1].
    """

    observation_mode = "local"

    def __init__(self) -> None:
        self.obs_dim = 2  # temperature, target
        self.action_dim = 1
        self.action_range = (-1.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        temp, target = obs_vec[0], obs_vec[1]
        error = target - temp
        action = float(np.clip(error / 5.0, -1.0, 1.0))
        return np.array([action])


class VentPolicy(Policy):
    """Simple rule-based vent controller for event-driven mode.

    Observes ``VentStatusFeature`` (is_open, cooling_power) and outputs a
    mild cooling fraction.  In step-based mode the vent receives its action
    from the coordinator's broadcast; this policy is only used when the
    vent runs autonomously in event-driven mode.
    """

    observation_mode = "local"

    def __init__(self) -> None:
        self.obs_dim = 2  # is_open, cooling_power
        self.action_dim = 1
        self.action_range = (0.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        return np.array([0.3])  # mild cooling when activated


class CoordinatorPolicy(Policy):
    """Rule-based coordinator that outputs a cooling signal.

    Sees subordinate states (full observation mode) and emits a cooling
    command proportional to the maximum temperature overshoot above 28 C.
    """

    observation_mode = "full"

    def __init__(self) -> None:
        self.obs_dim = 4  # ~2 zones x 2 features each
        self.action_dim = 1
        self.action_range = (0.0, 1.0)

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        # Extract temperatures (indices 0 and 2 for the two zones)
        temps = obs_vec[::2]  # every other value is a temperature
        max_overshoot = max(float(np.max(temps)) - 28.0, 0.0)
        signal = float(np.clip(max_overshoot / 5.0, 0.0, 1.0))
        return np.array([signal])
