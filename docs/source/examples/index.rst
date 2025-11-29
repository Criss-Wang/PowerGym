Examples
========

This section provides practical examples demonstrating how to use PowerGrid for various multi-agent power system applications.

Basic Examples
--------------

Example 1: Single Microgrid with Centralized Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** :file:`examples/01_single_microgrid_basic.py`

Demonstrates the basic usage of PowerGrid with a single microgrid containing multiple devices controlled by a centralized GridAgent.

**Key concepts:**

- Creating device agents (Generator, ESS)
- Building a GridAgent to coordinate devices
- Using CentralizedSetpointProtocol for direct control
- Creating a simple PettingZoo environment
- Running a simulation loop with random actions

**Runtime:** ~30 seconds for 24 timesteps

Example 2: Multi-Microgrid with Peer-to-Peer Energy Trading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** :file:`examples/02_multi_microgrid_p2p.py`

Demonstrates multi-agent coordination with peer-to-peer (P2P) energy trading between microgrids using a horizontal protocol.

**Key concepts:**

- Creating multiple GridAgents (microgrids)
- Using PeerToPeerTradingProtocol for horizontal coordination
- Environment-owned vs agent-owned protocols
- Market clearing mechanism with bids and offers
- Multi-agent reward computation with trading

**Runtime:** ~45 seconds for 24 timesteps

Example 3: Price-Based Coordination with Vertical Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** :file:`examples/03_price_coordination.py`

Demonstrates hierarchical coordination using price signals. A GridAgent broadcasts electricity prices to its subordinate devices, which respond by adjusting their power output.

**Key concepts:**

- Using PriceSignalProtocol for vertical coordination
- GridAgent → DeviceAgent communication via messages
- How devices respond to price signals
- Difference between vertical (agent-owned) and horizontal (env-owned) protocols
- Price-responsive behavior (economic dispatch)

**Runtime:** ~35 seconds for 24 timesteps

Advanced Examples
-----------------

Example 4: Creating a Custom Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** :file:`examples/04_custom_device.py`

Demonstrates how to create your own custom device by subclassing DeviceAgent. This example implements a simple Solar Panel device with MPPT control.

**Key concepts:**

- Subclassing DeviceAgent to create custom devices
- Implementing device-specific methods (set_action_space, set_device_state, etc.)
- Using feature providers (ElectricalBasePh, PhaseConnection, custom features)
- Integrating custom devices into GridAgent and environments
- Device lifecycle: reset → observe → act → update_state → update_cost_safety

**Runtime:** ~30 seconds for 24 timesteps

Example 5: MAPPO Training for Cooperative Multi-Agent Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** :file:`examples/05_mappo_training.py`

Demonstrates production-ready training of Multi-Agent PPO (MAPPO) or Independent PPO (IPPO) on cooperative multi-agent microgrids with full experiment management capabilities.

**Key concepts:**

- Command-line training script with comprehensive arguments
- MAPPO vs IPPO for cooperative tasks
- Shared rewards to encourage cooperation
- Experiment tracking with Weights & Biases
- Checkpointing and resuming training
- Performance monitoring and logging

**Usage examples:**

.. code-block:: bash

    # Train with shared policy (MAPPO) for cooperative tasks
    python examples/05_mappo_training.py --iterations 100

    # Train with independent policies (IPPO)
    python examples/05_mappo_training.py --iterations 100 --independent-policies

    # Resume from checkpoint
    python examples/05_mappo_training.py --resume /path/to/checkpoint

    # With W&B logging
    python examples/05_mappo_training.py --wandb --wandb-project powergrid-coop

    # Quick test mode (3 iterations for verification)
    python examples/05_mappo_training.py --test --no-cuda

**Requirements:**

.. code-block:: bash

    pip install "ray[rllib]==2.9.0"
    pip install wandb  # Optional, for experiment tracking

**Runtime:** ~30 minutes for 100 iterations (depends on workers and hardware)

Running the Examples
--------------------

All examples can be run directly from the command line:

.. code-block:: bash

    # Run basic single microgrid example
    python examples/01_single_microgrid_basic.py

    # Run multi-microgrid P2P trading example
    python examples/02_multi_microgrid_p2p.py

    # Run price coordination example
    python examples/03_price_coordination.py

    # Run custom device example
    python examples/04_custom_device.py

    # Run MAPPO training (see usage examples above)
    python examples/05_mappo_training.py --test

Next Steps
----------

After exploring these examples, you can:

- Learn more about :doc:`../architecture/index` to understand the framework design
- Explore the :doc:`../api/index` for detailed API documentation
- Check out :doc:`../protocols/index` for information on communication protocols
- Read about :doc:`../devices/index` for available device types
