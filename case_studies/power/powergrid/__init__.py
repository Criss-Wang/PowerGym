"""PowerGrid: Multi-agent power system simulation using HERON.

This module provides power system components following the grid_age style:
- Agents: Generator, ESS (Energy Storage), Transformer, DeviceAgent
- Grid Coordination: PowerGridAgent
- Environments: HierarchicalMicrogridEnv, EnvState
- Features: Electrical, Storage, Network state providers

Example:
    Basic usage with factory function::

        from powergrid.envs import create_hierarchical_env

        env = create_hierarchical_env(
            microgrid_configs=[...],
            dataset_path="path/to/data.h5",
        )
        obs, info = env.reset()
"""

__version__ = "0.1.0"

# Device Agents
from powergrid.agents.device_agent import DeviceAgent
from powergrid.agents.generator import Generator
from powergrid.agents.storage import ESS
from powergrid.agents.transformer import Transformer
from powergrid.core.features.metrics import CostSafetyMetrics

# Grid Agents
from powergrid.agents.power_grid_agent import PowerGridAgent
from heron.agents.proxy_agent import ProxyAgent, PROXY_LEVEL
from powergrid.agents import POWER_FLOW_CHANNEL_TYPE

# Environments
from powergrid.envs.common import EnvState
from powergrid.envs.hierarchical_microgrid_env import HierarchicalMicrogridEnv
from powergrid.envs.factory import create_hierarchical_env, create_default_3_microgrid_env

# Features
from powergrid.core.features.electrical import ElectricalBasePh
from powergrid.core.features.storage import StorageBlock
from powergrid.core.features.network import BusVoltages, LineFlows, NetworkMetrics
from powergrid.core.features.status import StatusBlock
from powergrid.core.features.power_limits import PowerLimits

# Utilities
from powergrid.utils.phase import PhaseModel, PhaseSpec
from powergrid.utils.safety import SafetySpec, total_safety
from powergrid.utils.cost import quadratic_cost, energy_cost

__all__ = [
    # Version
    "__version__",
    # Device Agents
    "DeviceAgent",
    "CostSafetyMetrics",
    "Generator",
    "ESS",
    "Transformer",
    # Grid Agents
    "PowerGridAgent",
    "ProxyAgent",
    "PROXY_LEVEL",
    "POWER_FLOW_CHANNEL_TYPE",
    # Environments
    "EnvState",
    "HierarchicalMicrogridEnv",
    "create_hierarchical_env",
    "create_default_3_microgrid_env",
    # Features
    "ElectricalBasePh",
    "StorageBlock",
    "BusVoltages",
    "LineFlows",
    "NetworkMetrics",
    "StatusBlock",
    "PowerLimits",
    # Utilities
    "PhaseModel",
    "PhaseSpec",
    "SafetySpec",
    "total_safety",
    "quadratic_cost",
    "energy_cost",
]
