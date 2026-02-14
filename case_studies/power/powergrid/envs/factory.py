"""Factory functions for creating power grid environments with hierarchical agents."""

from typing import Any, Dict, List, Optional

from heron.agents.system_agent import SystemAgent
from powergrid.envs.hierarchical_microgrid_env import HierarchicalMicrogridEnv
from powergrid.agents import (
    PowerGridAgent,
    Generator,
    ESS,
    Transformer,
)
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.storage import StorageBlock
from powergrid.core.features.metrics import CostSafetyMetrics


def create_hierarchical_env(
    microgrid_configs: List[Dict[str, Any]],
    dataset_path: str,
    episode_steps: int = 24,
    dt: float = 1.0,
    scheduler_config: Optional[Dict[str, Any]] = None,
    message_broker_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> HierarchicalMicrogridEnv:
    """Create a hierarchical microgrid environment with automatic agent hierarchy.

    This factory function creates the full agent hierarchy:
        SystemAgent -> PowerGridAgents -> DeviceAgents (Generator, ESS, Transformer)

    Args:
        microgrid_configs: List of microgrid configuration dicts, each containing:
            - name: Microgrid identifier (e.g., "MG1")
            - load_area: Load area for dataset (e.g., "AVA")
            - renew_area: Renewable area for dataset (e.g., "NP15")
            - generators: List of generator configs [{name, min_p, max_p}, ...]
            - storage: List of storage configs [{name, capacity, min_p, max_p}, ...]
            - transformers: List of transformer configs [{name, ...}, ...]
        dataset_path: Path to dataset file
        episode_steps: Episode length in time steps (default: 24)
        dt: Time step duration in hours (default: 1.0)
        scheduler_config: Configuration for event scheduler
        message_broker_config: Configuration for message broker
        **kwargs: Additional arguments for HierarchicalMicrogridEnv

    Returns:
        Initialized HierarchicalMicrogridEnv

    Note:
        Rewards are computed by individual agents via compute_local_reward().
        Configure safety penalties and reward sharing at the agent level.

    Example:
        >>> microgrid_configs = [
        ...     {
        ...         "name": "MG1",
        ...         "load_area": "AVA",
        ...         "renew_area": "NP15",
        ...         "generators": [{"name": "Gen1", "min_p": 0.1, "max_p": 1.0}],
        ...         "storage": [{"name": "ESS1", "capacity": 2.0, "min_p": -0.5, "max_p": 0.5}],
        ...     },
        ... ]
        >>> env = create_hierarchical_env(microgrid_configs, "path/to/data.h5")
        >>> obs, info = env.reset()
    """
    # Create microgrids
    microgrids = {}
    for mg_config in microgrid_configs:
        mg_id = mg_config["name"]

        # Create device agents for this microgrid
        devices = {}

        # Create generator agents
        for gen_config in mg_config.get("generators", []):
            gen_id = f"{mg_id}_{gen_config['name']}"
            devices[gen_id] = Generator(
                agent_id=gen_id,
                features=[
                    ElectricalBasePh(),
                    CostSafetyMetrics(),
                ],
                min_p=gen_config.get("min_p", 0.0),
                max_p=gen_config.get("max_p", 1.0),
                min_q=gen_config.get("min_q", -0.5),
                max_q=gen_config.get("max_q", 0.5),
            )

        # Create storage agents
        for ess_config in mg_config.get("storage", []):
            ess_id = f"{mg_id}_{ess_config['name']}"
            devices[ess_id] = ESS(
                agent_id=ess_id,
                features=[
                    ElectricalBasePh(),
                    StorageBlock(capacity=ess_config.get("capacity", 2.0)),
                    CostSafetyMetrics(),
                ],
                capacity=ess_config.get("capacity", 2.0),
                min_p=ess_config.get("min_p", -0.5),
                max_p=ess_config.get("max_p", 0.5),
                min_soc=ess_config.get("min_soc", 0.2),
                max_soc=ess_config.get("max_soc", 0.9),
            )

        # Create transformer agents
        for xfmr_config in mg_config.get("transformers", []):
            xfmr_id = f"{mg_id}_{xfmr_config['name']}"
            devices[xfmr_id] = Transformer(
                agent_id=xfmr_id,
                features=[CostSafetyMetrics()],
                min_tap=xfmr_config.get("min_tap", -16),
                max_tap=xfmr_config.get("max_tap", 16),
            )

        # Create coordinator for this microgrid
        microgrids[mg_id] = PowerGridAgent(
            agent_id=mg_id,
            subordinates=devices,
        )

    # Create system agent
    system_agent = SystemAgent(
        agent_id="system_agent",
        subordinates=microgrids,
    )

    # Create environment
    return HierarchicalMicrogridEnv(
        system_agent=system_agent,
        dataset_path=dataset_path,
        episode_steps=episode_steps,
        dt=dt,
        scheduler_config=scheduler_config,
        message_broker_config=message_broker_config,
        **kwargs,
    )


def create_default_3_microgrid_env(
    dataset_path: str,
    episode_steps: int = 24,
    **kwargs,
) -> HierarchicalMicrogridEnv:
    """Create default 3-microgrid environment matching MultiAgentMicrogrids setup.

    This is a convenience function that creates the standard 3-microgrid
    configuration used in experiments.

    Args:
        dataset_path: Path to dataset file
        episode_steps: Episode length (default: 24)
        **kwargs: Additional arguments for create_hierarchical_env

    Returns:
        Initialized HierarchicalMicrogridEnv with 3 microgrids
    """
    microgrid_configs = [
        {
            "name": "MG1",
            "load_area": "AVA",
            "renew_area": "NP15",
            "generators": [
                {"name": "Gen1", "min_p": 0.1, "max_p": 0.66},
            ],
            "storage": [
                {"name": "ESS1", "capacity": 2.0, "min_p": -0.5, "max_p": 0.5},
            ],
        },
        {
            "name": "MG2",
            "load_area": "BANC",
            "renew_area": "NP15",
            "generators": [
                {"name": "Gen1", "min_p": 0.1, "max_p": 0.60},
            ],
            "storage": [
                {"name": "ESS1", "capacity": 2.0, "min_p": -0.5, "max_p": 0.5},
            ],
        },
        {
            "name": "MG3",
            "load_area": "BANCMID",
            "renew_area": "NP15",
            "generators": [
                {"name": "Gen1", "min_p": 0.1, "max_p": 0.50},
            ],
            "storage": [
                {"name": "ESS1", "capacity": 2.0, "min_p": -0.5, "max_p": 0.5},
            ],
        },
    ]

    return create_hierarchical_env(
        microgrid_configs=microgrid_configs,
        dataset_path=dataset_path,
        episode_steps=episode_steps,
        **kwargs,
    )
