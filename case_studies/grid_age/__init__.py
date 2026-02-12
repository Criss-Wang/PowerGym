"""GridAges case study: Multi-agent microgrid control.

This package implements the GridAges multi-microgrid environment using the Heron
framework for hierarchical multi-agent reinforcement learning.

Architecture:
    SystemAgent → MicrogridCoordinatorAgent → DeviceFieldAgents

Each microgrid has device agents (ESS, DG, PV, Wind) coordinated by a
MicrogridCoordinatorAgent. Policies can be trained at either:
    - Device level (individual device control)
    - Coordinator level (microgrid-level control with action distribution)

Key components:
- features: Device feature providers (SOC, Power, UnitCommitment, Availability, Voltage)
- agents: Device agents (ESS, DG, RES) and MicrogridCoordinator
- envs: HierarchicalMicrogridEnv with Pandapower integration
- train: Training scripts for CTDE

Example usage:
    >>> from case_studies.grid_age import create_hierarchical_env, train_microgrid_ctde
    >>> env = create_hierarchical_env(num_microgrids=3)
    >>> policies = train_microgrid_ctde(env, num_episodes=100)
"""

from case_studies.grid_age.envs import (
    HierarchicalMicrogridEnv,
    EnvState,
    create_hierarchical_env,
)
from case_studies.grid_age.agents import (
    MicrogridCoordinatorAgent,
    ESSFieldAgent,
    DGFieldAgent,
    RESFieldAgent,
)
from case_studies.grid_age.features import (
    SOCFeature,
    PowerFeature,
    UnitCommitmentFeature,
    AvailabilityFeature,
    VoltageFeature,
)
from case_studies.grid_age.train import train_microgrid_ctde, NeuralPolicy

__all__ = [
    # Environment
    "HierarchicalMicrogridEnv",
    "create_hierarchical_env",
    "EnvState",
    # Agents
    "MicrogridCoordinatorAgent",
    "ESSFieldAgent",
    "DGFieldAgent",
    "RESFieldAgent",
    # Features
    "SOCFeature",
    "PowerFeature",
    "UnitCommitmentFeature",
    "AvailabilityFeature",
    "VoltageFeature",
    # Training
    "train_microgrid_ctde",
    "NeuralPolicy",
]

__version__ = "0.1.0"
