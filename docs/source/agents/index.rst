Agents
======

.. toctree::
   :hidden:

   GridAgent <grid_agent>
   DeviceAgent <device_agent>

PowerGrid uses a hierarchical multi-agent architecture with GridAgents and DeviceAgents.

Agent Types
-----------

- :doc:`grid_agent` - Microgrid controller that coordinates devices
- :doc:`device_agent` - Controls individual distributed energy resources

Agent Hierarchy
---------------

.. code-block:: text

   GridAgent (Microgrid Controller)
   ├── DeviceAgent (Battery)
   ├── DeviceAgent (Generator)
   ├── DeviceAgent (Solar PV)
   └── DeviceAgent (Load)

GridAgent
~~~~~~~~~

Manages a microgrid and coordinates subordinate devices.

**Responsibilities:**

- Observe microgrid state
- Coordinate devices via vertical protocols
- Participate in horizontal protocols with other GridAgents
- Optimize microgrid objectives

**Key Methods:**

- ``observe()``: Get observation from environment
- ``act()``: Compute action using policy
- ``receive_message()``: Handle coordination signals

DeviceAgent
~~~~~~~~~~~

Controls a single distributed energy resource (DER).

**Responsibilities:**

- Observe device state
- Execute control actions
- Respond to GridAgent signals

**Supported Devices:**

- Energy Storage Systems (ESS)
- Diesel Generators
- Solar PV
- Load Controllers
- Transformers
- Capacitor Banks

Creating Custom Agents
-----------------------

Extend the base ``Agent`` class:

.. code-block:: python

   from powergrid.agents import Agent
   from powergrid.core import Observation, Action

   class MyCustomAgent(Agent):
       def observe(self, state):
           # Extract relevant features
           return Observation(local={...})

       def act(self, observation):
           # Compute action from policy
           return Action(c=..., d=...)

       def reset(self):
           # Reset agent state
           pass

Agent Communication
-------------------

Agents communicate via messages:

.. code-block:: python

   from powergrid.core import Message

   # Send message
   message = Message(
       sender='MG1',
       recipient='MG2',
       content={'price': 50.0}
   )
   env.send_message(message)

   # Receive in observation
   obs = agent.observe(state)
   for msg in obs.messages:
       print(f"Received: {msg.content}")

See :doc:`/api/core/protocols` for coordination protocol details.
