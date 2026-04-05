"""Transport Fleet environment -- multi-vehicle delivery demo for HERON.

A depot coordinator manages N vehicle agents that move, consume fuel, and
complete deliveries. Custom events (new delivery requests, vehicle breakdowns)
are modeled within the simulation step.
"""

import math
from typing import Dict, List, Optional

import numpy as np

from heron.envs.builder import EnvBuilder
from heron.envs.simple import DefaultHeronEnv
from heron.protocols.vertical import BroadcastActionProtocol, VerticalProtocol
from heron.demo_envs.transport_fleet.agents import (
    DepotCoordinator,
    DepotFeature,
    VehicleAgent,
    VehicleFeature,
)

# A vehicle completes a delivery when within this radius of a delivery point.
_DELIVERY_RADIUS = 1.5


def _transport_simulation(
    agent_states: Dict[str, Dict],
    *,
    delivery_points: List[List[float]],
    breakdown_prob: float,
    new_delivery_prob: float,
    rng: np.random.Generator,
) -> Dict[str, Dict]:
    """Physics step for the transport fleet.

    Handles custom events (new deliveries, breakdowns) and delivery completion.

    Args:
        agent_states: ``{agent_id: {feature_name: {field: val}}}``.
        delivery_points: Mutable list of ``[x, y]`` delivery locations.
        breakdown_prob: Per-step probability of a random vehicle breakdown.
        new_delivery_prob: Per-step probability of a new delivery request.
        rng: NumPy random generator for reproducibility.

    Returns:
        Updated agent_states dict.
    """
    depot_id: Optional[str] = None
    vehicle_ids: List[str] = []
    for aid, features in agent_states.items():
        if "DepotFeature" in features:
            depot_id = aid
        if "VehicleFeature" in features:
            vehicle_ids.append(aid)

    # --- Custom event: new delivery request ---
    if rng.random() < new_delivery_prob:
        pt = [float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10))]
        delivery_points.append(pt)

    # --- Custom event: random vehicle breakdown ---
    if vehicle_ids and rng.random() < breakdown_prob:
        victim = rng.choice(vehicle_ids)
        vf = agent_states[victim]["VehicleFeature"]
        if not vf["is_broken"]:
            vf["is_broken"] = True

    # --- Check delivery completions ---
    step_deliveries = 0
    for vid in vehicle_ids:
        vf = agent_states[vid]["VehicleFeature"]
        if vf["is_broken"] or vf["fuel"] <= 0:
            continue

        vx, vy = vf["x"], vf["y"]
        for i, pt in enumerate(delivery_points):
            dx = vx - pt[0]
            dy = vy - pt[1]
            if math.sqrt(dx * dx + dy * dy) <= _DELIVERY_RADIUS:
                delivery_points.pop(i)
                vf["deliveries"] = vf.get("deliveries", 0) + 1
                vf["has_package"] = False
                step_deliveries += 1
                break  # one delivery per vehicle per step

    # --- Update depot aggregate ---
    if depot_id is not None:
        depot_f = agent_states[depot_id]["DepotFeature"]
        depot_f["total_deliveries"] = (
            depot_f.get("total_deliveries", 0) + step_deliveries
        )
        depot_f["pending_requests"] = len(delivery_points)

    return agent_states


def build_transport_fleet_env(
    n_vehicles: int = 3,
    max_steps: int = 100,
    fuel_capacity: float = 100.0,
    breakdown_prob: float = 0.02,
    new_delivery_prob: float = 0.3,
    initial_deliveries: int = 3,
    seed: Optional[int] = None,
) -> DefaultHeronEnv:
    """Build a TransportFleet-v0 environment.

    Args:
        n_vehicles: Number of vehicle agents.
        max_steps: Episode length (truncation).
        fuel_capacity: Starting fuel for each vehicle.
        breakdown_prob: Per-step vehicle breakdown probability.
        new_delivery_prob: Per-step new delivery probability.
        initial_deliveries: Number of delivery points at reset.
        seed: Random seed for simulation RNG.

    Returns:
        Configured ``DefaultHeronEnv``.
    """
    rng = np.random.default_rng(seed)
    delivery_points: List[List[float]] = [
        [float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10))]
        for _ in range(initial_deliveries)
    ]

    def sim_func(agent_states: Dict[str, Dict]) -> Dict[str, Dict]:
        return _transport_simulation(
            agent_states,
            delivery_points=delivery_points,
            breakdown_prob=breakdown_prob,
            new_delivery_prob=new_delivery_prob,
            rng=rng,
        )

    protocol = VerticalProtocol(
        action_protocol=BroadcastActionProtocol(),
    )

    env = (
        EnvBuilder("transport_fleet")
        .add_agents(
            "vehicle",
            VehicleAgent,
            count=n_vehicles,
            features=[VehicleFeature(fuel=fuel_capacity)],
            coordinator="depot",
            fuel_capacity=fuel_capacity,
        )
        .add_coordinator(
            "depot",
            agent_cls=DepotCoordinator,
            features=[DepotFeature(pending_requests=initial_deliveries)],
            protocol=protocol,
            subordinates=["vehicle_*"],
        )
        .simulation(sim_func)
        .termination(max_steps=max_steps)
        .build()
    )
    return env
