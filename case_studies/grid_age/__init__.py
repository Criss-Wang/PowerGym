"""GridAges case study: Multi-agent microgrid control.

This package implements the GridAges multi-microgrid environment using the Heron
framework for hierarchical multi-agent reinforcement learning.

Key components:
- features: Device feature providers (ESS, DG, RES, Grid, Network)
- agents: MicrogridFieldAgent for composite device control
- envs: MicrogridEnv with Pandapower integration
- train: Training scripts for CTDE

Example usage:
    >>> from case_studies.grid_age import MicrogridEnv, train_microgrid_ctde
    >>> env = MicrogridEnv(num_microgrids=3)
    >>> policies = train_microgrid_ctde(env, num_episodes=100)
"""

from case_studies.grid_age.envs import MicrogridEnv, EnvState
from case_studies.grid_age.agents import MicrogridFieldAgent
from case_studies.grid_age.features import (
    ESSFeature,
    DGFeature,
    RESFeature,
    GridFeature,
    NetworkFeature,
)
from case_studies.grid_age.train import train_microgrid_ctde, NeuralPolicy

__all__ = [
    # Environment
    "MicrogridEnv",
    "EnvState",
    # Agents
    "MicrogridFieldAgent",
    # Features
    "ESSFeature",
    "DGFeature",
    "RESFeature",
    "GridFeature",
    "NetworkFeature",
    # Training
    "train_microgrid_ctde",
    "NeuralPolicy",
]

__version__ = "0.1.0"
