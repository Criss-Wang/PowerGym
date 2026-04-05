"""Sensor network environment -- demonstrates visibility + horizontal protocol.

N sensor agents are placed on a graph. Events (signals) propagate across
edges, and agents must detect active signals while avoiding false positives.
Agents share detection states with neighbors via ``HorizontalProtocol``.
"""

import itertools
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from heron.envs.builder import EnvBuilder
from heron.envs.simple import DefaultHeronEnv
from heron.protocols.horizontal import HorizontalProtocol
from heron.demo_envs.sensor_network.agents import SensorAgent, SensorFeature


def _generate_random_graph(
    n: int,
    connectivity: float,
    seed: int | None = None,
) -> Dict[str, List[str]]:
    """Generate a random undirected graph as an adjacency dict.

    Args:
        n: Number of nodes.
        connectivity: Probability of an edge between any two nodes.
        seed: Optional random seed for reproducibility.

    Returns:
        Adjacency dict ``{node_id: [neighbor_ids]}``.
    """
    rng = random.Random(seed)
    ids = [f"sensor_{i}" for i in range(n)]
    adjacency: Dict[str, List[str]] = {nid: [] for nid in ids}

    for a, b in itertools.combinations(ids, 2):
        if rng.random() < connectivity:
            adjacency[a].append(b)
            adjacency[b].append(a)

    # Ensure every node has at least one neighbor (connect isolates to a random peer)
    for nid in ids:
        if not adjacency[nid]:
            candidates = [other for other in ids if other != nid]
            peer = rng.choice(candidates)
            adjacency[nid].append(peer)
            adjacency[peer].append(nid)

    return adjacency


def _sensor_network_simulation(
    agent_states: Dict[str, Dict],
    adjacency: Dict[str, List[str]],
    spread_prob: float = 0.3,
) -> Dict[str, Dict]:
    """Physics step: signal injection, decay, and propagation.

    1. Existing signals decay slightly (multiply by 0.9).
    2. Random new signal injection at 1-2 random nodes.
    3. Signals spread to neighbors with probability ``spread_prob``.

    Args:
        agent_states: ``{agent_id: {feature_name: {field: val}}}``.
        adjacency: Graph adjacency dict.
        spread_prob: Probability of signal spreading to a neighbor.

    Returns:
        Updated agent_states dict.
    """
    sensor_ids = [aid for aid in agent_states if "SensorFeature" in agent_states[aid]]
    if not sensor_ids:
        return agent_states

    # Step 1: Decay existing signals
    for aid in sensor_ids:
        data = agent_states[aid]["SensorFeature"]
        data["signal_strength"] = data["signal_strength"] * 0.9

    # Step 2: Inject new signals at 1-2 random nodes
    n_inject = np.random.randint(1, 3)  # 1 or 2
    inject_nodes = np.random.choice(sensor_ids, size=min(n_inject, len(sensor_ids)), replace=False)
    for aid in inject_nodes:
        agent_states[aid]["SensorFeature"]["signal_strength"] = min(
            agent_states[aid]["SensorFeature"]["signal_strength"] + 1.0, 2.0,
        )

    # Step 3: Propagation -- signals spread to neighbors
    # Collect pre-spread signals to avoid cascading within a single step
    pre_spread = {
        aid: agent_states[aid]["SensorFeature"]["signal_strength"]
        for aid in sensor_ids
    }
    for aid in sensor_ids:
        if pre_spread[aid] > 0.5:
            for neighbor in adjacency.get(aid, []):
                if neighbor in agent_states and np.random.random() < spread_prob:
                    nbr_data = agent_states[neighbor]["SensorFeature"]
                    nbr_data["signal_strength"] = min(
                        nbr_data["signal_strength"] + pre_spread[aid] * 0.5,
                        2.0,
                    )

    # Step 4: Gossip -- compute neighbor_avg_detection
    for aid in sensor_ids:
        neighbors = adjacency.get(aid, [])
        if neighbors:
            detections = [
                agent_states[n]["SensorFeature"].get("detection", 0.0)
                for n in neighbors
                if n in agent_states and "SensorFeature" in agent_states[n]
            ]
            avg = sum(detections) / len(detections) if detections else 0.0
            agent_states[aid]["SensorFeature"]["neighbor_avg_detection"] = avg

    return agent_states


def build_sensor_network_env(
    n_sensors: int = 5,
    connectivity: float = 0.4,
    spread_prob: float = 0.3,
    max_steps: int = 50,
    graph_seed: int | None = None,
) -> DefaultHeronEnv:
    """Build a SensorNetwork-v0 environment.

    Args:
        n_sensors: Number of sensor nodes.
        connectivity: Edge probability for random graph generation.
        spread_prob: Probability of signal spreading to a neighbor per step.
        max_steps: Episode length (truncation).
        graph_seed: Optional seed for reproducible graph topology.

    Returns:
        Configured ``DefaultHeronEnv``.
    """
    adjacency = _generate_random_graph(n_sensors, connectivity, seed=graph_seed)

    # Build the horizontal protocol with the generated topology
    protocol = HorizontalProtocol(
        state_fields=["detection", "neighbor_avg_detection"],
        topology=adjacency,
    )

    def sim_func(agent_states: Dict[str, Dict]) -> Dict[str, Dict]:
        return _sensor_network_simulation(
            agent_states,
            adjacency=adjacency,
            spread_prob=spread_prob,
        )

    builder = (
        EnvBuilder("sensor_network")
        .add_coordinator(
            "sensor_coordinator",
            protocol=protocol,
            subordinates=["sensor_*"],
        )
        .simulation(sim_func)
        .termination(max_steps=max_steps)
    )

    for i in range(n_sensors):
        agent_id = f"sensor_{i}"
        builder = builder.add_agent(
            agent_id,
            SensorAgent,
            features=[SensorFeature()],
            coordinator="sensor_coordinator",
        )

    return builder.build()
