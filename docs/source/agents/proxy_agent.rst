ProxyAgent
==========

The ProxyAgent is a specialized agent that manages information distribution in distributed execution mode. It acts as an intermediary between the environment and GridAgents, enforcing information hiding and filtering network state based on visibility rules.

.. note::
   The base ``ProxyAgent`` is defined in ``heron.agents.proxy_agent`` and is domain-agnostic.
   The power grid version (``powergrid.agents.proxy_agent.ProxyAgent``) extends it with
   the ``power_flow`` channel type for PandaPower integration.

.. note::
   ProxyAgent is only used in **distributed mode** (``centralized=False``). In centralized mode, agents directly access the PandaPower network object.

Role and Responsibilities
--------------------------

The ProxyAgent serves as a central hub for network state information in distributed systems:

.. code-block:: text

   Environment (owns PandaPower net)
         │
         │ publishes aggregated network state
         ↓
   ProxyAgent (filters & distributes)
         │
         ├─→ GridAgent MG1 (receives filtered state)
         ├─→ GridAgent MG2 (receives filtered state)
         └─→ GridAgent MG3 (receives filtered state)

**Key Responsibilities:**

1. **Receive**: Consume aggregated network state from the environment
2. **Cache**: Store the most recent network state for all agents
3. **Filter**: Apply visibility rules to determine what each agent can see
4. **Distribute**: Send agent-specific filtered state to each GridAgent

Why ProxyAgent?
----------------

**Information Hiding**
   In real power systems, agents (microgrid operators) don't have complete visibility into the entire network. ProxyAgent enforces this realistic constraint by filtering network information based on ownership and visibility rules.

**Scalability**
   Instead of the environment sending separate messages to each agent, it sends one aggregated message to ProxyAgent, which then efficiently distributes filtered information.

**Flexibility**
   Visibility rules can be customized per agent, allowing for complex information-sharing scenarios (e.g., public vs. private data, different access levels).

**Realism**
   Mimics real-world distributed control systems where a central coordinator (like an ISO) distributes filtered market and network data to participants.

Message Flow
------------

The ProxyAgent participates in the following message sequence during each environment step:

.. code-block:: text

   1. Environment runs power flow on PandaPower network
   2. Environment extracts network state for all agents:
      - Device results (P, Q for each generator/storage)
      - Bus voltages (vm_pu, violations)
      - Line loading (percent, overload)
      - Safety metrics (overvoltage, undervoltage, overloading)

   3. Environment publishes aggregated state to ProxyAgent channel

   4. ProxyAgent receives and caches aggregated state

   5. ProxyAgent iterates through subordinate GridAgents:
      - Extracts agent-specific state from aggregated data
      - Applies visibility filters
      - Publishes filtered state to agent's info channel

   6. GridAgents consume filtered network state from their channels

   7. GridAgents compute cost and safety metrics using filtered data

State Structures
----------------

Aggregated State (Environment → ProxyAgent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       'converged': bool,  # Did power flow converge?
       'agents': {
           'MG1': {
               'converged': bool,
               'device_results': {
                   'gen1': {'p_mw': 1.5, 'q_mvar': 0.3},
                   'ess1': {'p_mw': -0.5, 'q_mvar': 0.1}
               },
               'bus_voltages': {
                   'vm_pu': [1.02, 0.98, 1.01],  # Per-bus voltages
                   'overvoltage': 0.05,           # Total violation (pu)
                   'undervoltage': 0.02
               },
               'line_loading': {
                   'loading_percent': [45.2, 78.9, 32.1],
                   'overloading': 0.0
               }
           },
           'MG2': { ... },
           'MG3': { ... }
       }
   }

Filtered State (ProxyAgent → GridAgent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # What MG1 receives (only its own data)
   {
       'converged': bool,
       'device_results': {
           'gen1': {'p_mw': 1.5, 'q_mvar': 0.3},
           'ess1': {'p_mw': -0.5, 'q_mvar': 0.1}
       },
       'bus_voltages': {
           'vm_pu': [1.02, 0.98, 1.01],  # Only MG1's buses
           'overvoltage': 0.05,
           'undervoltage': 0.02
       },
       'line_loading': {
           'loading_percent': [45.2, 78.9, 32.1],  # Only MG1's lines
           'overloading': 0.0
       }
   }

Key Methods
-----------

receive_network_state_from_environment()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consumes aggregated network state from the environment channel.

.. code-block:: python

   network_state = proxy_agent.receive_network_state_from_environment()
   # Returns: Dict with aggregated state or None if no message available

**Behavior:**
   - Consumes message from ``env_{env_id}__power_flow_results__proxy_agent`` channel
   - Caches state in ``self.network_state_cache``
   - Returns the payload for inspection

distribute_network_state_to_agents()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distributes cached network state to all subordinate GridAgents.

.. code-block:: python

   proxy_agent.distribute_network_state_to_agents()

**Behavior:**
   - Iterates through ``self.subordinate_agents`` list
   - Extracts agent-specific state from ``self.network_state_cache``
   - Applies visibility filtering via ``_filter_state_for_agent()``
   - Publishes to ``env_{env_id}__info__proxy_agent_to_{agent_id}`` channel

_filter_state_for_agent(agent_id, state)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies visibility rules to filter network state for a specific agent.

.. code-block:: python

   filtered_state = proxy_agent._filter_state_for_agent('MG1', agent_state)

**Visibility Rules:**
   - **owner**: Agent can see its own data (default for all agents)
   - **public**: Data marked as public (e.g., market prices, aggregated metrics)
   - Custom rules can be added by extending this method

Usage in Environment
--------------------

ProxyAgent is automatically created and managed by NetworkedGridEnv in distributed mode:

.. code-block:: python

   from powergrid.envs.multi_agent import MultiAgentMicrogrids

   # Create environment in distributed mode
   env = MultiAgentMicrogrids({
       'centralized': False,  # Enable distributed mode
       'microgrid_configs': [
           {'name': 'MG1', 'devices': [...]},
           {'name': 'MG2', 'devices': [...]},
           {'name': 'MG3', 'devices': [...]}
       ]
   })

   # ProxyAgent is created automatically in __init__
   # Environment calls ProxyAgent methods during step():

   # After power flow:
   env._publish_network_state_to_agents()

   # Which internally does:
   # 1. self.proxy_agent.receive_network_state_from_environment()
   # 2. self.proxy_agent.distribute_network_state_to_agents()

Manual Configuration
~~~~~~~~~~~~~~~~~~~~

For advanced use cases, you can customize ProxyAgent behavior:

.. code-block:: python

   # Power grid ProxyAgent (uses "power_flow" channel type)
   from powergrid.agents.proxy_agent import ProxyAgent
   from heron.messaging.memory import InMemoryBroker

   # Create custom ProxyAgent
   message_broker = InMemoryBroker()
   proxy_agent = ProxyAgent(
       agent_id='custom_proxy',
       message_broker=message_broker,
       env_id='env_001',
       subordinate_agents=['MG1', 'MG2', 'MG3']
   )

   # Or use the generic heron ProxyAgent with custom channel type
   from heron.agents.proxy_agent import ProxyAgent as BaseProxyAgent

   generic_proxy = BaseProxyAgent(
       agent_id='custom_proxy',
       message_broker=message_broker,
       env_id='env_001',
       subordinate_agents=['MG1', 'MG2', 'MG3'],
       result_channel_type='my_custom_channel'  # Configurable!
   )

   # Override filtering behavior
   class CustomProxyAgent(ProxyAgent):
       def _filter_state_for_agent(self, agent_id, state):
           # Custom visibility logic
           filtered = super()._filter_state_for_agent(agent_id, state)

           # Example: Add market-wide information
           if 'market_price' in self.network_state_cache:
               filtered['market_price'] = self.network_state_cache['market_price']

           return filtered

Example: Accessing Filtered State in GridAgent
-----------------------------------------------

GridAgents consume filtered network state via ``_consume_network_state()``:

.. code-block:: python

   # In PowerGridAgent.update_cost_safety(net=None)
   # When net=None, agent is in distributed mode

   if net is None:
       # Distributed mode: receive from ProxyAgent
       network_state = self._consume_network_state()

       if network_state and network_state.get('converged', False):
           # Extract pre-computed safety metrics
           bus_voltages = network_state.get('bus_voltages', {})
           line_loading = network_state.get('line_loading', {})

           overvoltage = bus_voltages.get('overvoltage', 0)
           undervoltage = bus_voltages.get('undervoltage', 0)
           overloading = line_loading.get('overloading', 0)

           self.safety += overloading + overvoltage + undervoltage

Message Channels
----------------

ProxyAgent uses the following message channels:

**Input Channel (Environment → ProxyAgent)**
   ``env_{env_id}__power_flow_results__proxy_agent``

   - Type: ``MessageType.POWER_FLOW_RESULT``
   - Sender: ``"environment"``
   - Recipient: ``"proxy_agent"``
   - Payload: Aggregated network state

**Output Channels (ProxyAgent → GridAgents)**
   ``env_{env_id}__info__proxy_agent_to_{agent_id}``

   - Type: ``MessageType.INFO``
   - Sender: ``"proxy_agent"``
   - Recipient: ``{agent_id}`` (e.g., ``"MG1"``)
   - Payload: Filtered agent-specific state

Best Practices
--------------

1. **Always use ProxyAgent in distributed mode**: Never bypass ProxyAgent and send network state directly to agents in distributed mode.

2. **Cache efficiency**: ProxyAgent caches state between ``receive_network_state_from_environment()`` and ``distribute_network_state_to_agents()`` calls, so call them in sequence.

3. **Visibility rules**: Design visibility rules to match real-world information constraints. Overly permissive rules defeat the purpose of distributed mode.

4. **Testing**: When debugging distributed mode, add logging to ProxyAgent methods to trace information flow.

5. **Custom filtering**: If your use case requires special visibility rules (e.g., neighboring agents can see each other's marginal prices), extend ``_filter_state_for_agent()``.

Comparison: Centralized vs Distributed
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Centralized Mode
     - Distributed Mode (ProxyAgent)
   * - Network Access
     - Direct (agents read ``net`` object)
     - Indirect (via filtered messages)
   * - Information
     - Complete visibility
     - Filtered per agent
   * - Realism
     - MARL benchmark setting
     - Real-world distributed control
   * - Scalability
     - All agents query net
     - Single aggregated message
   * - Overhead
     - Minimal (direct access)
     - Message passing + filtering

See Also
--------

- :doc:`grid_agent` - GridAgent that consumes ProxyAgent messages
- :doc:`/core/messaging` - MessageBroker and message handling
- :doc:`/design/distributed_architecture` - Overall distributed mode architecture
- :doc:`/api/agents` - ProxyAgent API reference
