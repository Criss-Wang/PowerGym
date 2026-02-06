
# EV Public Charging Station — HERON Case Study (2-Station Joint Learning)

This case study implements a **public EV charging station pricing environment**
designed for **multi-agent reinforcement learning (MARL)** under the HERON
feature–agent abstraction.

The goal is to study **profit maximization, coordination, and competition**
between charging stations under stochastic demand, capacity constraints, and
time-varying electricity prices.

---

## 1. Problem Overview

We consider a public charging market with:

- **Two charging stations** (`station_1`, `station_2`)
- Each station sets a **charging price** periodically
- EVs arrive stochastically and choose stations based on price and congestion
- Stations earn revenue from charging and pay electricity + fixed costs
- Agents learn to **maximize (shared or individual) profit**

This environment is suitable for studying:
- Joint profit maximization (team game)
- Price-based tacit coordination


## 2. High-Level Architecture

The environment follows the HERON design principle:

```

Feature  →  Agent  →  World (Dynamics)  →  Feature

```

Key separation of concerns:

- **Features** define *what the state is*
- **World knowing dynamics** defines *how state evolves*
- **Agents** decide *when and how to modify owned features*
- **Demand modules** model exogenous arrivals and user behavior
- **Adapters** expose the environment to RL frameworks (PettingZoo)


## 3. Agents

### L1: Station Agents
- `station_1`, `station_2`
- Action: price adjustment `a ∈ [-1, 1]`
- Action is mapped to a price in `[price_min, price_max]`
- Pricing decisions are made every `action_period`

Agents **do not**:
- Generate arrivals
- Control EV choice
- Simulate charging physics

They only control **price**, via their owned feature.

---

## 4. Features (State Representation)

All environment state is expressed via **FeatureProviders**.

### 4.1 Public Features

#### `PublicMarketSignal`
Visible to all agents.

| Field        | Meaning                          |
|-------------|----------------------------------|
| `elec_price` | Wholesale electricity price      |
| `time_sin`   | Sinusoidal time-of-day encoding  |
| `time_cos`   | Cosine time-of-day encoding      |

Purpose:
- Provides shared exogenous context
- Enables time-aware pricing strategies

---

### 4.2 Station-Owned Features

Each station owns its own feature instances.

#### `ChargingStationStatus`
Visibility: `owner`, `upper_level`

| Field        | Meaning                   |
|-------------|---------------------------|
| `occupancy` | Fraction of chargers used |
| `price`     | Current charging price    |

This is the **main controllable state** of each station.

---

#### `StationProfitWindow`
Visibility: `owner`, `upper_level`

| Field          | Meaning                             |
|---------------|-------------------------------------|
| `profit_ema`  | Exponentially-smoothed profit       |
| `revenue_ema` | Smoothed revenue                    |
| `cost_ema`    | Smoothed cost                       |

Purpose:
- Stabilizes learning
- Provides low-variance economic signals
- Enables profit-based reward shaping

---

## 5. Demand Model

Demand is modeled in two stages.

### 5.1 Arrival Process

Implemented in `demand/arrival_process.py`.

- Hourly piecewise-constant Poisson arrivals
- Arrival rates depend on time-of-day
- Arrivals are **exogenous** (not controlled by agents)

---

### 5.2 User Choice Model

Implemented in `demand/user_choice.py`.

For each arriving EV:
- Utility depends on:
  - Station price
  - Station congestion (occupancy)
  - Idiosyncratic noise
- EV selects the station with highest utility
- If no capacity is available → blocked
- Optional give-up behavior can be added

This models **realistic price–congestion trade-offs**.

---

## 6. World Dynamics

Core environment logic is implemented in:

```

env/public_charging_world.py

```

Each internal simulation step (`dt`) executes:

1. Update public market signals
2. Sample EV arrivals
3. Allocate EVs to stations
4. Update charging sessions and departures
5. Compute revenue, cost, and profit
6. Update profit windows
7. Advance time

### Action Timing
- Agents update prices only every `action_period`
- Between price updates, prices are held constant

This reflects realistic pricing cadence.

---

## 7. Reward Design

### Default: Team Reward
All agents receive the same reward:

```

R_t = normalized( profit_1(t) + profit_2(t) )

```

This enables:
- Cooperative learning
- Joint profit maximization
- MAPPO-style training

### Alternatives (Easy to Modify)
- Individual profit rewards (competition)
- Period-level rewards (per pricing decision)
- EMA-based rewards for stability

---

## 8. Observations

### Partial Observability (Default)

Each station observes:

```

[ PublicMarketSignal,
Own ChargingStationStatus,
Own StationProfitWindow ]

```

### Full Observability (Optional)

Stations additionally observe other stations’ status.
This enables controlled experiments on information sharing.

---

## 9. RL Interface

The environment is exposed via the **PettingZoo Parallel API**:

```

adapters/pettingzoo_env.py

````

- Compatible with RLlib, CleanRL, custom MARL trainers
- Agents act simultaneously
- Supports IPPO / MAPPO style training

---

## 10. Running the Example

From the project root:

```bash
python -m case_studies.power.ev_public_charging_case.run_random_rollout
````

This runs a random policy rollout to validate:

* Environment stability
* Reward flow
* Observation/action wiring

---

## 11. Research Use Cases

This case study supports experiments on:

* Pricing under capacity constraints
* Tacit coordination vs. competition
* Partial vs. full observability
* Joint vs. individual reward structures
* Learning under stochastic demand

It is designed to be:

* Modular
* Extensible
* HERON-native

---

## 12. Extending the Case

Easy extensions include:

* More stations
* Heterogeneous charger capacities
* Dynamic electricity tariffs
* Explicit coordination protocols
* Stackelberg (leader–follower) pricing

---

## 13. Key Design Principle

> **Agents do not own the environment.
> Agents own features.
> The world owns dynamics.**