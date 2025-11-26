Overview
========

PowerGrid 2.0 is a multi-agent reinforcement learning framework for smart grid control and optimization.

What is PowerGrid?
------------------

PowerGrid provides:

- **Multi-agent environments** compatible with PettingZoo and RLlib
- **Hierarchical agent architecture** (GridAgents and DeviceAgents)
- **Pluggable coordination protocols** (vertical and horizontal)
- **AC power flow simulation** using PandaPower
- **Distributed or centralized training modes**

Key Features
------------

**Dual-Mode Architecture**

Switch between centralized and distributed control:

- **Centralized**: Single policy controls all devices (fast training)
- **Distributed**: Independent policies per agent (scalable, realistic)

**Flexible Protocols**

Implement coordination without changing agent code:

- **Vertical**: Parent-to-subordinate (price signals, setpoints)
- **Horizontal**: Peer-to-peer (trading, consensus)

**Realistic Physics**

Accurate AC power flow with:

- Voltage magnitude and angle
- Real and reactive power
- Line losses and constraints
- Transformer tap control

**RL-Ready**

Built for reinforcement learning:

- Gymnasium/PettingZoo API
- RLlib integration (MAPPO, PPO, IPPO)
- Configurable observation/action spaces
- Reward shaping for grid objectives

When to Use PowerGrid
---------------------

**Good fit:**

- Multi-agent control of distributed energy resources
- Coordinated optimization with communication
- Research on hierarchical RL for power systems
- Testing control strategies on realistic grids

**Not ideal:**

- Single-agent centralized optimization (use MPC instead)
- Ultra-fast real-time control (< 1 second timesteps)
- Large-scale transmission systems (PowerGrid focuses on distribution)

Architecture at a Glance
------------------------

.. code-block:: text

   ┌─────────────────────────────────────┐
   │         Environment                 │
   │  - Runs power flow                  │
   │  - Manages agents                   │
   │  - Executes protocols               │
   └─────────────────────────────────────┘
            │
            ├─► Horizontal Protocol (P2P)
            │
            ▼
       ┌─────────┬─────────┬─────────┐
       │   MG1   │   MG2   │   MG3   │  GridAgents
       └─────────┴─────────┴─────────┘
            │
            ├─► Vertical Protocol (Prices/Setpoints)
            │
            ▼
       ┌────┬────┬────┬────┐
       │ESS │ DG │ PV │Load│  DeviceAgents
       └────┴────┴────┴────┘

Next Steps
----------

- :doc:`/getting_started` - Quick start tutorial
- :doc:`/user_guide/basic_concepts` - Core concepts
- :doc:`/examples/index` - Example code
