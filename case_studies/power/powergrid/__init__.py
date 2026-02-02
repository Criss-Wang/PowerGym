"""PowerGrid: Multi-agent power system simulation using HERON.

This module provides power system components:
- Agents: Generator, ESS (Energy Storage), Transformer, DeviceAgent
- Grid Coordination: GridAgent, PowerGridAgent
- Environments: NetworkedGridEnv, MultiAgentMicrogrids
- Features: Electrical, Storage, Network state providers

Example:
    Basic usage with a generator agent::

        from powergrid.agents import Generator
        from powergrid.envs import MultiAgentMicrogrids

        env = MultiAgentMicrogrids(env_config)
        obs, info = env.reset()
"""

__version__ = "0.1.0"

# Device Agents
from powergrid.agents.device_agent import DeviceAgent
from powergrid.agents.generator import Generator, GeneratorConfig
from powergrid.agents.storage import ESS, StorageConfig
from powergrid.agents.transformer import Transformer, TransformerConfig

# Grid Agents
from powergrid.agents.power_grid_agent import GridAgent, PowerGridAgent
from powergrid.agents.proxy_agent import ProxyAgent

# Environments
from powergrid.envs.networked_grid_env import NetworkedGridEnv
from powergrid.envs.multi_agent_microgrids import MultiAgentMicrogrids

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
    "Generator",
    "GeneratorConfig",
    "ESS",
    "StorageConfig",
    "Transformer",
    "TransformerConfig",
    # Grid Agents
    "GridAgent",
    "PowerGridAgent",
    "ProxyAgent",
    # Environments
    "NetworkedGridEnv",
    "MultiAgentMicrogrids",
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
