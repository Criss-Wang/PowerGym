"""Factory functions for creating microgrid environments with hierarchical agents."""

from typing import Optional, Dict, Any

from heron.agents.system_agent import SystemAgent
from case_studies.grid_age.envs.hierarchical_microgrid_env import HierarchicalMicrogridEnv
from case_studies.grid_age.agents import (
    MicrogridCoordinatorAgent,
    ESSFieldAgent,
    DGFieldAgent,
    RESFieldAgent,
)


def create_hierarchical_env(
    num_microgrids: int = 3,
    episode_steps: int = 24,
    dt: float = 1.0,
    scheduler_config: Optional[Dict[str, Any]] = None,
    message_broker_config: Optional[Dict[str, Any]] = None,
    simulation_wait_interval: Optional[float] = None,
    **kwargs
) -> HierarchicalMicrogridEnv:
    """Create a hierarchical microgrid environment with automatic agent hierarchy.

    This factory function creates the full agent hierarchy:
        SystemAgent → MicrogridCoordinatorAgents → DeviceFieldAgents

    Each microgrid has:
        - ESSFieldAgent (Energy Storage System)
        - DGFieldAgent (Diesel Generator)
        - 2x RESFieldAgent (Solar PV and Wind)

    Args:
        num_microgrids: Number of microgrids (default: 3)
        episode_steps: Episode length in time steps (default: 24)
        dt: Time step duration in hours (default: 1.0)
        scheduler_config: Configuration for event scheduler
        message_broker_config: Configuration for message broker
        simulation_wait_interval: Wait time for simulation events
        **kwargs: Additional arguments for HierarchicalMicrogridEnv

    Returns:
        Initialized HierarchicalMicrogridEnv

    Example:
        >>> env = create_hierarchical_env(num_microgrids=3, episode_steps=24)
        >>> obs, info = env.reset()
        >>> # Train policies on device agents OR coordinator agents
    """
    # DG capacities matching GridAges paper
    dg_capacities = {1: 0.66, 2: 0.60, 3: 0.50}

    # Create microgrids
    microgrids = {}
    for i in range(1, num_microgrids + 1):
        mg_id = f"MG{i}"

        # Create device agents for this microgrid
        devices = {
            f"{mg_id}_ESS": ESSFieldAgent(
                agent_id=f"{mg_id}_ESS",
                capacity=2.0,
                min_p=-0.5,
                max_p=0.5,
                min_soc=0.2,
                max_soc=0.9,
            ),
            f"{mg_id}_DG": DGFieldAgent(
                agent_id=f"{mg_id}_DG",
                min_p=0.1,
                max_p=dg_capacities.get(i, 0.66),
            ),
            f"{mg_id}_PV": RESFieldAgent(
                agent_id=f"{mg_id}_PV",
                max_p=0.1,
                res_type="solar",
            ),
            f"{mg_id}_Wind": RESFieldAgent(
                agent_id=f"{mg_id}_Wind",
                max_p=0.1,
                res_type="wind",
            ),
        }

        # Create coordinator for this microgrid
        microgrids[mg_id] = MicrogridCoordinatorAgent(
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
        episode_steps=episode_steps,
        dt=dt,
        scheduler_config=scheduler_config,
        message_broker_config=message_broker_config,
        simulation_wait_interval=simulation_wait_interval,
        **kwargs
    )
