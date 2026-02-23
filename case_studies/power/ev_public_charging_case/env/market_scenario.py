"""Market scenario simulation for Collusion Study."""

import numpy as np

class MarketScenario:
    def __init__(self, arrival_rate: float, price_freq: float):
        self.initial_arrival_rate = arrival_rate
        self.price_freq = price_freq
        self.reset()

    def reset(self):
        self.time_seconds = 0.0
        self.last_price_update = -self.price_freq
        self.current_lmp = 0.20

    def step(self, dt: float):
        self.time_seconds += dt

        if self.time_seconds - self.last_price_update >= self.price_freq:
            self.current_lmp = 0.2 + 0.1 * np.sin(2 * np.pi * self.time_seconds / 86400)
            self.last_price_update = self.time_seconds

        peak_factor = 1.0 + 0.5 * np.sin(2 * np.pi * self.time_seconds / 86400 - np.pi/2)
        arrivals = np.random.poisson(self.initial_arrival_rate * peak_factor * dt / 3600.0)

        return {
            "lmp": self.current_lmp,
            "t": self.time_seconds,
            "arrivals": arrivals
        }