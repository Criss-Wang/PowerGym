"""HERON: Hierarchical Environment for Reinforcement learning with cOordination Networks.

A multi-agent coordination framework supporting:
- Hierarchical agent architectures (Field, Coordinator, System levels)
- Protocol-based communication and coordination
- Event-driven scheduling with configurable timing
- Integration with Gymnasium, PettingZoo, and RLlib

Example:
    Basic usage with a field agent::

        from heron.agents import FieldAgent
        from heron.core import Action, Observation
        from heron.protocols import NoProtocol

        agent = FieldAgent(
            agent_id="device_1",
            protocol=NoProtocol(),
        )
"""

__version__ = "0.1.0"

# Core components
from heron.core.action import Action
from heron.core.observation import Observation
from heron.core.state import State, FieldAgentState, CoordinatorAgentState
from heron.core.feature import FeatureProvider
from heron.core.policies import Policy

# Agents
from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.agents.proxy_agent import ProxyAgent

# Environments
from heron.envs.base import EnvCore, MultiAgentEnv
from heron.envs.adapters import PettingZooParallelEnv

# Protocols
from heron.protocols.base import Protocol, NoProtocol
from heron.protocols.vertical import VerticalProtocol
from heron.protocols.horizontal import HorizontalProtocol

# Messaging
from heron.messaging import Message, MessageType, MessageBroker
from heron.messaging.in_memory_broker import InMemoryBroker

# Scheduling
from heron.scheduling import (
    Event,
    EventType,
    EventScheduler,
    TickConfig,
    JitterType,
)

# Shortcuts
from heron.shortcuts.numeric_feature import NumericFeature
from heron.shortcuts.simple_field_agent import SimpleFieldAgent
from heron.shortcuts.simulation_bridge import SimpleEnv
from heron.shortcuts.env_builder import EnvBuilder
from heron.shortcuts import quickstart
from heron.shortcuts.quickstart import make_env

__all__ = [
    # Version
    "__version__",
    # Core
    "Action",
    "Observation",
    "State",
    "FieldAgentState",
    "CoordinatorAgentState",
    "FeatureProvider",
    "Policy",
    # Agents
    "Agent",
    "FieldAgent",
    "CoordinatorAgent",
    "SystemAgent",
    "ProxyAgent",
    # Environments
    "EnvCore",
    "MultiAgentEnv",
    "PettingZooParallelEnv",
    # Protocols
    "Protocol",
    "NoProtocol",
    "VerticalProtocol",
    "HorizontalProtocol",
    # Messaging
    "Message",
    "MessageType",
    "MessageBroker",
    "InMemoryBroker",
    # Scheduling
    "Event",
    "EventType",
    "EventScheduler",
    "TickConfig",
    "JitterType",
    # Shortcuts
    "NumericFeature",
    "SimpleFieldAgent",
    "SimpleEnv",
    "EnvBuilder",
    "quickstart",
    "make_env",
]
