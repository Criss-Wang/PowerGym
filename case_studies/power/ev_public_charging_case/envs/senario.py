import numpy as np

class EVTrafficDemandModel:
    ARRIVAL_PROFILE = [
        0.5, 0.5, 0.5, 0.5, 1.0, 1.5, 2.5, 4.0,
        5.5, 6.0, 5.5, 5.0, 4.0, 3.0, 2.5, 2.0,
        1.5, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5
    ]

    def __init__(self, arrival_rate_scale: float):
        self.scale = arrival_rate_scale
        self.time_seconds = 0.0

    def step(self, dt: float):
        self.time_seconds += dt

        hour_idx = int((self.time_seconds % 86400) // 3600)
        current_base_rate = self.ARRIVAL_PROFILE[hour_idx]
        actual_rate = current_base_rate * self.scale
        arrivals = np.random.poisson(actual_rate * dt / 3600.0)

        return {
            "t": self.time_seconds,
            "arrivals": arrivals,
        }