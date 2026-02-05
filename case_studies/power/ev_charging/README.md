# EV Public Charging Station (HERON) Case Study — 2-Station Joint Learning
### RL

* **Agents**

  * **L1 (field):** two stations `station_1`, `station_2`
  * **L2 (coordinator):** `coord_1` (optional; aggregates upper-level observations + group metrics)
  * **L3:** omitted for simplicity

* **Action Space (L1)**

  * `a_i ∈ [-1,1]` → price mapped to `[price_min, price_max]`

* **State / Observation (FeatureProviders + visibility)**

  * **public:** `TimeOfDay`, `ElectricityPrice`
  * **owner (+upper_level):** `StationPrice`, `StationOccupancy`, `StationProfitWindow`
  * (optional) **full_obs experiment:** make `StationPrice/Occupancy` public

* **Transition Dynamics**

  * EV arrivals (time-dependent)
  * utility-based station choice
  * capacity-based charging completion (updates occupancy + revenue/cost)
  * electricity price evolution (time-of-day + noise)

* **Reward (team game)**

  * shared reward for both stations:
    `R = normalize(profit_1 + profit_2)`

* **Episode / Time Structure**

  * internal sim step: `dt = 1–5 min`
  * pricing decision interval: `action_period = 15 min`
  * episode horizon: `48h`

## Environment Design
```text
Agent
 ├─ observes State / Observation
 ├─ chooses Action (price)
 │
 ▼
Environment (Transition Dynamics)
 ├─ EV arrivals happen (exogenous)
 ├─ Users choose stations (behavioral model)
 ├─ Charging / queue / departure
 ├─ electricity price evolution
 │
 ▼
Next State + Reward

```


# Structure
