import numpy as np

class MarketScenario:
    ARRIVAL_PROFILE = [
        0.5, 0.5, 0.5, 0.5, 1.0, 1.5, 2.5, 4.0,
        5.5, 6.0, 5.5, 5.0, 4.0, 3.0, 2.5, 2.0,
        1.5, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5
    ]

    def __init__(self, arrival_rate_scale: float, price_freq: float):
        self.scale = arrival_rate_scale
        self.price_freq = price_freq
        self.time_seconds = 0.0
        self.current_lmp = 0.20
        
        self.price_scales = [
            # Day 1: normal
            [1.0, 0.90, 0.85, 0.81, 0.80, 0.90, 1.00, 1.25, 1.45, 1.55, 1.56, 1.60, 
             1.62, 1.66, 1.68, 1.75, 1.85, 1.93, 2.00, 1.85, 1.6, 1.5, 1.2, 1.1],
            # normal
            [1.0, 0.90, 0.85, 0.81, 0.80, 0.90, 1.00, 1.25, 1.45, 1.55, 1.56, 1.60,
             1.62, 1.66, 1.68, 1.75, 1.85, 1.93, 2.00, 1.85, 1.6, 1.5, 1.2, 1.1],
            # # Day 2: price spike
            # [1.0, 0.90, 0.85, 0.81, 0.80, 0.90, 1.00, 1.25, 1.45, 1.55, 1.56, 1.60,
            #  1.62, 1.66, 1.68, 1.75, 1.85*4, 1.93*8, 2.00*10, 1.85*8, 1.6*4, 1.5, 1.2, 1.1]
        ]
        
        self.base_price = 0.15

    def step(self, dt: float):
        self.time_seconds += dt

        day_idx = int((self.time_seconds // 86400) % 2)
        hour_idx = int((self.time_seconds % 86400) // 3600)
        

        self.current_lmp = self.base_price * self.price_scales[day_idx][hour_idx]

        current_base_rate = self.ARRIVAL_PROFILE[hour_idx]
        actual_rate = current_base_rate * self.scale
        arrivals = np.random.poisson(actual_rate * dt / 3600.0)
        
        return {
            "lmp": self.current_lmp, 
            "t": self.time_seconds,
            "arrivals": arrivals
        }