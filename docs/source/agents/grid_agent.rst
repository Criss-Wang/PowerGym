GridAgent
=========

GridAgent represents a microgrid controller that coordinates subordinate devices.

Overview
--------

A GridAgent:

- **Observes** the global grid state and subordinate device states
- **Coordinates** devices using vertical protocols
- **Participates** in peer coordination via horizontal protocols
- **Optimizes** for microgrid objectives (cost, reliability, etc.)

Configuration
-------------

.. code-block:: python

   from powergrid.agents import GridAgent
   from powergrid.core.protocols import PriceSignalProtocol

   agent = GridAgent(
       agent_id='MG1',
       subordinates=[ess_agent, gen_agent, pv_agent],
       vertical_protocol=PriceSignalProtocol(initial_price=50.0),
       centralized=False
   )

Parameters
~~~~~~~~~~

- **agent_id**: Unique identifier
- **subordinates**: List of DeviceAgents to manage
- **vertical_protocol**: Protocol for device coordination
- **centralized**: If True, GridAgent directly controls devices

Observation Space
-----------------

GridAgent observes:

**Local State:**

- Microgrid bus voltages
- Net power import/export
- Subordinate device states
- Time of day, day of week

**Global State (if available):**

- Other GridAgent states
- Main grid price/frequency
- Network topology

**Messages:**

- Horizontal protocol signals (trades, consensus)
- External market signals

Action Space
------------

GridAgent actions depend on the protocol and centralized mode:

**Centralized Mode:**

Direct control of all subordinate devices:

.. code-block:: python

   action = {
       'ESS1': 0.3,   # Charge at 0.3 MW
       'DG1': 0.5,    # Generate 0.5 MW
       'PV1': 0.0     # No curtailment
   }

**Distributed with Price Signal:**

Broadcast price to devices:

.. code-block:: python

   action = 65.0  # $/MWh

**Distributed with Setpoint:**

Send power setpoints to devices:

.. code-block:: python

   action = {
       'ESS1': 0.2,
       'DG1': 0.4
   }

Example Usage
-------------

.. code-block:: python

   from powergrid.agents import GridAgent
   from powergrid.core.protocols import SetpointProtocol

   # Create GridAgent
   grid_agent = GridAgent(
       agent_id='MG1',
       subordinates=device_agents,
       vertical_protocol=SetpointProtocol(),
       centralized=False
   )

   # Get observation
   obs = grid_agent.observe(global_state)

   # Compute action from policy
   action = policy(obs)

   # Coordinate subordinates
   signals = grid_agent.vertical_protocol.coordinate(
       subordinate_observations, action
   )

   # Send signals to devices
   for dev_id, signal in signals.items():
       device_agents[dev_id].receive_message(
           Message(sender='MG1', content=signal)
       )

See Also
--------

- :doc:`device_agent` - Subordinate device agents
- :doc:`/api/core/protocols` - Coordination protocols
- :doc:`/architecture/agents` - Architecture details
