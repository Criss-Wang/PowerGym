# test_world_sim.py
from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

from ..config import ArrivalConfig, EVConfig, StationConfig, WorldConfig
from ..demand.arrival_process import HourlyPoissonArrival
from ..demand.user_choice import UtilityBasedUserChoice
from ..entities.charging_station import ChargingStation


def build_evsts_info(stations: dict[str, ChargingStation]) -> dict:
    """Build the station info dict that EV.choose_station expects."""
    info = {}
    for name, st in stations.items():
        info[name] = {
            "price": st.charging_price,
            "parking_fee": st.parking_fee,
            "charging_power": st.charging_power,
            "num_chargers": st.num_chargers,
            "num_users": len(st.busy_list),
        }
    return info


def test_world_sim_loop():
    """
    Main test function. pytest will automatically identify functions starting with test_.
    This validates world dynamics without RL (fixed pricing).
    """
    seed = 42
    rng = np.random.default_rng(seed)

    # ---- Configs ----
    arrival_cfg = ArrivalConfig(
        hourly_rate=[
            0.5, 0.5, 0.5, 0.5, 1.0, 1.5, 2.5, 4.0,
            5.5, 6.0, 5.5, 5.0, 4.0, 3.0, 2.5, 2.0,
            1.5, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5
        ],
        scale=1.0,
        rate_is_system_total=True,
    )
    ev_cfg = EVConfig()
    st_cfg = StationConfig(
        num_chargers=2,
        charging_power=100.0,
        init_price=0.30,
        parking_fee=3.0,
        charging_efficiency=0.9,
        cost_kwh=0.01,
        cost_hour=1000 / 30 / 24,
    )
    world_cfg = WorldConfig(dt=60, action_period=1800)

    # ---- Demand Modules ----
    arrival_proc = HourlyPoissonArrival(
        num_stations=2,
        rng=rng,
        cfg=arrival_cfg,
    )
    chooser = UtilityBasedUserChoice(
        ev_cfg=ev_cfg,
        rng=rng,
        u_tol=0.0,
    )

    # ---- Entities ----
    stations = {
        "st_1": ChargingStation(
            station_id=1, name="st_1", dt=world_cfg.dt, action_period=world_cfg.action_period, cfg=st_cfg
        ),
        "st_2": ChargingStation(
            station_id=2, name="st_2", dt=world_cfg.dt, action_period=world_cfg.action_period, cfg=st_cfg
        ),
    }

    # Simple fixed pricing policy (NO RL)
    fixed_price = 0.35
    for st in stations.values():
        st.charging_price = fixed_price

    num_chargers_total = sum(st.num_chargers for st in stations.values())

    # ---- Loop Setup ----
    horizon_seconds = 24 * 3600
    steps = int(horizon_seconds // world_cfg.dt)
    t = 0.0

    # Logging
    history = {
        "occ_1": [], "occ_2": [],
        "prof_1": [], "prof_2": [],
        "elec_prices": [], "arrivals": [],
        "giveup": [], "blocked": []
    }

    print(f"\nStarting simulation for {steps} steps...")

    for k in range(steps):
        # 1. Market electricity price fluctuation
        hour = (t % 86400) / 3600
        elec_price = 0.18 + 0.07 * math.sin(2 * math.pi * hour / 24) + float(rng.normal(0, 0.005))
        elec_price = max(0.01, elec_price)

        # 2. Update station prices and process departures
        for st in stations.values():
            st.set_prices(charging_price=st.charging_price, electricity_price=elec_price)
            st.leaving()

        # 3. Sample new EV arrivals
        arrivals = arrival_proc.sample(t=t, dt=world_cfg.dt)
        # Handle case where sample returns count (int) or list of EVs
        num_arrivals = len(arrivals) if isinstance(arrivals, list) else int(arrivals)

        # 4. Choice and Allocation
        evsts_info = build_evsts_info(stations)
        allocation, stats = chooser.allocate(
            t=t,
            arrivals=arrivals,
            evsts_info=evsts_info,
            num_chargers_total=num_chargers_total,
        )

        # 5. Arriving and Step logic
        for name, st in stations.items():
            st.arrive(allocation[name])
            st.step()

        # 6. Logging
        history["occ_1"].append(len(stations["st_1"].busy_list) / stations["st_1"].num_chargers)
        history["occ_2"].append(len(stations["st_2"].busy_list) / stations["st_2"].num_chargers)
        history["prof_1"].append(stations["st_1"].profit)
        history["prof_2"].append(stations["st_2"].profit)
        history["elec_prices"].append(elec_price)
        history["arrivals"].append(num_arrivals)
        history["giveup"].append(stats["giveup"])
        history["blocked"].append(stats["blocked"])

        t += world_cfg.dt

    # ---- Sanity Checks ----
    def is_sane(data):
        return np.all(np.isfinite(data))

    assert is_sane(history["occ_1"]) and is_sane(history["occ_2"]), "Occupancy contains NaNs"
    assert max(history["occ_1"]) <= 1.0 + 1e-6, "Over-capacity detected in st_1"

    # ---- Summary Output ----
    print("\n" + "=" * 30)
    print("      SIMULATION SUMMARY")
    print("=" * 30)
    total_arr = sum(history['arrivals'])
    print(f"Total Arrivals: {total_arr}")
    if total_arr > 0:
        print(f"Total Give-ups: {sum(history['giveup'])} ({sum(history['giveup']) / total_arr * 100:.1f}%)")
    print(f"Total Blocked:  {sum(history['blocked'])}")
    print(f"Station 1 Profit: {stations['st_1'].profit_acc:.2f}")
    print(f"Station 2 Profit: {stations['st_2'].profit_acc:.2f}")
    print(f"Max Occupancy:   {max(history['occ_1']):.2f} / {max(history['occ_2']):.2f}")
    print("-" * 30)
    print("PASS ✅")

    _plot_results(history)


def _plot_results(h):
    """Internal plotting function to visualize 24h dynamics."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    time_x = np.arange(len(h["occ_1"])) / 60  # convert to hours

    # Plot 1: Occupancy & Arrivals
    ax1 = axes[0]
    ax1.plot(time_x, h["occ_1"], label="St 1 Occupancy", color='blue', alpha=0.7)
    ax1.plot(time_x, h["occ_2"], label="St 2 Occupancy", color='cyan', linestyle='--')
    ax1.set_ylabel("Occupancy Rate")
    ax1.legend(loc='upper left')
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(time_x, h["arrivals"], color='gray', alpha=0.2, label="Arrivals")
    ax1_twin.set_ylabel("Arrivals count")

    # Plot 2: Prices
    ax2 = axes[1]
    ax2.plot(time_x, h["elec_prices"], label="Market Elec Price", color='orange')
    ax2.axhline(y=0.35, color='red', linestyle=':', label="Station Fixed Price")
    ax2.set_ylabel("Price ($/kWh)")
    ax2.legend()

    # Plot 3: Cumulative Profit
    ax3 = axes[2]
    ax3.plot(time_x, np.cumsum(h["prof_1"]), label="St 1 Cum. Profit")
    ax3.plot(time_x, np.cumsum(h["prof_2"]), label="St 2 Cum. Profit")
    ax3.set_ylabel("Profit ($)")
    ax3.set_xlabel("Time of Day (Hours)")
    ax3.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_world_sim_loop()