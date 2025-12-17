Protocols
=========

This guide provides an in-depth look at the PowerGrid coordination protocol system, including vertical and horizontal protocols, implementation details, and how to create custom protocols.

Protocol System Overview
------------------------

What are Protocols?
~~~~~~~~~~~~~~~~~~~

Protocols define **how agents coordinate** with each other. In PowerGrid, there are two types:

1. **Vertical Protocols**: Parent → subordinate coordination
2. **Horizontal Protocols**: Peer ↔ peer coordination

Design Philosophy
~~~~~~~~~~~~~~~~~

**Separation of Concerns:**

- **Protocols**: Define *how* to coordinate (the mechanism)

  - **CommunicationProtocol**: Defines coordination messages (WHAT to communicate)
  - **ActionProtocol**: Defines action coordination (HOW to coordinate actions)

- **Policies**: Define *what* to coordinate (the strategy)
- **Agents**: Execute coordination and actions

**Key Benefits:**

- Composable: Mix communication strategies with action strategies
- Mode-agnostic: Works in both centralized and distributed execution
- Plug-and-play: Swap protocols without changing agent code
- Testable: Communication and action protocols are independently testable

Architecture
~~~~~~~~~~~~

.. code-block:: text

   ┌────────────────────────────────────────────────┐
   │           Environment                          │
   │  - Runs horizontal protocols                   │
   │  - Coordinates peer agents                     │
   └────────────────────────────────────────────────┘
            │
            │ Horizontal Coordination
            │ (P2P Trading, Consensus)
            ▼
       ┌────────┬────────┬────────┐
       │  MG1   │  MG2   │  MG3   │  GridAgents
       │        │        │        │  - Own vertical protocols
       └────────┴────────┴────────┘  - Coordinate subordinates
            │        │        │
            │ Vertical Coordination
            │ (Prices, Setpoints)
            ▼
       ┌────┬────┬────┐
       │ESS │ DG │ PV │  DeviceAgents
       └────┴────┴────┘  - Respond to signals

Vertical Protocols
------------------

Purpose
~~~~~~~

Vertical protocols enable **hierarchical control** where a parent agent (GridAgent) coordinates subordinate agents (DeviceAgents).

Ownership
~~~~~~~~~

Each GridAgent **owns** its own vertical protocol. This enables:

- **Decentralized coordination**: Each microgrid independently manages its devices
- **Heterogeneous control**: Different microgrids can use different protocols
- **Privacy**: No need to share subordinate information with other agents

Execution Flow
~~~~~~~~~~~~~~

.. code-block:: python

   # During GridAgent.act():
   def act(self, observation, upstream_action=None):
       # 1. Get coordinator action from policy
       action = self.policy.forward(observation) if self.policy else upstream_action

       # 2. Prepare subordinate states
       device_obs = observation.local.get("device_obs", {})
       subordinate_states = {dev_id: obs.local for dev_id, obs in device_obs.items()}

       # 3. Execute unified coordination (communication + action)
       self.protocol.coordinate(
           coordinator_state=observation.local,
           subordinate_states=subordinate_states,
           coordinator_action=action,
           mode="centralized",  # or "distributed"
           context={
               "subordinates": self.devices,
               "coordinator_id": self.agent_id,
               "timestamp": observation.timestamp
           }
       )

Built-in Vertical Protocols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NoProtocol
^^^^^^^^^^

**Use Case:** Independent device operation (baseline)

**How it works:** No coordination - devices act independently

**Composition:**
- Communication: NoCommunication (no messages)
- Action: NoActionCoordination (no action decomposition)

**Example:**

.. code-block:: python

   from powergrid.core.protocols import NoProtocol

   protocol = NoProtocol()
   # No messages sent, no actions coordinated

**When to use:**

- Benchmarking (compare against coordinated control)
- Fully decentralized control (each device has its own policy)

PriceSignalProtocol
^^^^^^^^^^^^^^^^^^^

**Use Case:** Economic dispatch via marginal price signals

**How it works:**

1. GridAgent broadcasts an electricity price ($/MWh) via messages
2. Devices optimize locally to maximize profit at that price
3. ESS charges when price is low, discharges when high
4. DG generates more when price is high

**Composition:**
- Communication: PriceCommunicationProtocol (broadcasts price messages)
- Action: DecentralizedActionProtocol (devices act independently)

**Parameters:**

- ``initial_price``: Starting price (default: 50.0 $/MWh)

**Example:**

.. code-block:: python

   from powergrid.core.protocols import PriceSignalProtocol
   from powergrid.agents.grid_agent import PowerGridAgent

   protocol = PriceSignalProtocol(initial_price=50.0)

   agent = PowerGridAgent(
       agent_id='MG1',
       protocol=protocol,
       ...
   )

   # Coordinator action: new price from learned policy
   # Protocol broadcasts price to all devices via messages
   agent.act(observation, upstream_action=65.0)

**Implementation Details:**

.. code-block:: python

   class PriceSignalProtocol(VerticalProtocol):
       def __init__(self, initial_price: float = 50.0):
           super().__init__(
               communication_protocol=PriceCommunicationProtocol(initial_price),
               action_protocol=DecentralizedActionProtocol()
           )

       @property
       def price(self):
           return self.communication_protocol.price

**Advantages:**

- Simple, interpretable
- Devices act independently (decentralized)
- Aligns incentives (price = marginal value)
- Works in both centralized and distributed modes

**Disadvantages:**

- Requires price-responsive device policies
- May not satisfy hard constraints

SetpointProtocol
^^^^^^^^^^^^^^^^

**Use Case:** Direct centralized control with power setpoints

**How it works:**

1. GridAgent computes power setpoints for each device
2. Protocol decomposes action into per-device setpoints
3. Devices directly execute their assigned setpoints
4. Useful for model-predictive control (MPC)

**Composition:**
- Communication: SetpointCommunicationProtocol (sends setpoint messages)
- Action: CentralizedActionProtocol (directly applies device actions)

**Example:**

.. code-block:: python

   from powergrid.core.protocols import SetpointProtocol
   from powergrid.agents.grid_agent import PowerGridAgent

   protocol = SetpointProtocol()

   agent = PowerGridAgent(
       agent_id='MG1',
       protocol=protocol,
       ...
   )

   # Coordinator action: dict of per-device setpoints
   parent_action = {
       'ESS1': np.array([0.3]),   # Charge at 0.3 MW
       'DG1': np.array([0.5]),    # Generate 0.5 MW
       'PV1': np.array([0.1])     # Solar at 0.1 MW
   }

   # Or flat array (auto-decomposed by protocol)
   parent_action = np.array([0.3, 0.5, 0.1])

   agent.act(observation, upstream_action=parent_action)

**Implementation Details:**

.. code-block:: python

   class SetpointProtocol(VerticalProtocol):
       def __init__(self):
           super().__init__(
               communication_protocol=SetpointCommunicationProtocol(),
               action_protocol=CentralizedActionProtocol()
           )

**Advantages:**

- Full control over device outputs
- Can enforce hard constraints
- Compatible with MPC, optimization-based control
- Works in both centralized and distributed modes

**Disadvantages:**

- High-dimensional action space (one setpoint per device)
- Less flexible than decentralized coordination

**Note:** ``CentralizedSetpointProtocol`` is now deprecated - use ``SetpointProtocol`` instead.

Horizontal Protocols
--------------------

Purpose
~~~~~~~

Horizontal protocols enable **peer-to-peer coordination** where multiple GridAgents interact as equals.

Ownership
~~~~~~~~~

The **environment** owns and runs horizontal protocols. This is because:

- Requires **global view** of all agents
- Implements **market mechanisms** (auctioneer)
- Ensures **fairness** and **truthfulness**

Execution Flow
~~~~~~~~~~~~~~

.. code-block:: python

   # During environment step (before vertical coordination):

   # 1. Collect observations from all agents
   observations = {aid: agent.observe(global_state) for aid, agent in agents.items()}

   # 2. Run horizontal protocol
   signals = horizontal_protocol.coordinate(agents, observations, topology)

   # 3. Deliver signals to agents (via messages)
   for agent_id, signal in signals.items():
       agents[agent_id].receive_message(Message(sender='MARKET', content=signal))

Built-in Horizontal Protocols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NoHorizontalProtocol
^^^^^^^^^^^^^^^^^^^^

**Use Case:** No peer coordination (baseline)

**How it works:** Returns empty signals for all agents

**Example:**

.. code-block:: python

   from powergrid.agents.protocols import NoHorizontalProtocol

   protocol = NoHorizontalProtocol()
   signals = protocol.coordinate(agents, observations)
   # → {'MG1': {}, 'MG2': {}, 'MG3': {}}

PeerToPeerTradingProtocol
^^^^^^^^^^^^^^^^^^^^^^^^^

**Use Case:** Decentralized energy marketplace

**How it works:**

1. **Agents compute** net demand and marginal cost
2. **Environment collects** bids (buyers) and offers (sellers)
3. **Market clears**: Match bids and offers by price
4. **Trade confirmations** sent back to agents

**Parameters:**

- ``trading_fee``: Transaction fee as fraction of price (default: 0.01)

**Example:**

.. code-block:: python

   from powergrid.agents.protocols import PeerToPeerTradingProtocol

   protocol = PeerToPeerTradingProtocol(trading_fee=0.01)

   # Agents include 'net_demand' and 'marginal_cost' in observations
   observations = {
       'MG1': Observation(local={'net_demand': -0.5, 'marginal_cost': 40}),  # Seller
       'MG2': Observation(local={'net_demand': 0.3, 'marginal_cost': 60}),   # Buyer
       'MG3': Observation(local={'net_demand': 0.2, 'marginal_cost': 55}),   # Buyer
   }

   signals = protocol.coordinate(agents, observations)
   # → {
   #   'MG1': {'trades': [{'counterparty': 'MG2', 'quantity': -0.3, 'price': 50}]},
   #   'MG2': {'trades': [{'counterparty': 'MG1', 'quantity': 0.3, 'price': 50}]},
   #   'MG3': {'trades': [{'counterparty': 'MG1', 'quantity': 0.2, 'price': 48}]}
   # }

ConsensusProtocol
^^^^^^^^^^^^^^^^^

**Use Case:** Distributed frequency or voltage regulation

**How it works:**

1. Agents start with different control values
2. Iteratively **average with neighbors** (gossip)
3. Converge to **global average** (consensus)

**Parameters:**

- ``max_iterations``: Max gossip rounds (default: 10)
- ``tolerance``: Convergence threshold (default: 0.01)

**Example:**

.. code-block:: python

   from powergrid.agents.protocols import ConsensusProtocol

   protocol = ConsensusProtocol(max_iterations=10, tolerance=0.01)

   # Agents have different initial control values
   observations = {
       'MG1': Observation(local={'control_value': 59.9}),  # Hz
       'MG2': Observation(local={'control_value': 60.1}),
       'MG3': Observation(local={'control_value': 60.0}),
   }

   # Optional: custom topology (default is fully connected)
   topology = {
       'adjacency': {
           'MG1': ['MG2'],
           'MG2': ['MG1', 'MG3'],
           'MG3': ['MG2']
       }
   }

   signals = protocol.coordinate(agents, observations, topology)
   # → After convergence:
   # {'MG1': {'consensus_value': 60.0},
   #  'MG2': {'consensus_value': 60.0},
   #  'MG3': {'consensus_value': 60.0}}

Creating Custom Protocols
--------------------------

Custom Vertical Protocol Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from powergrid.agents.protocols import VerticalProtocol
   from powergrid.agents.base import Observation, AgentID
   from typing import Dict, Any, Optional

   class MyVerticalProtocol(VerticalProtocol):
       """My custom vertical protocol."""

       def __init__(self, param1, param2):
           """Initialize with custom parameters."""
           self.param1 = param1
           self.param2 = param2

       def coordinate(
           self,
           subordinate_observations: Dict[AgentID, Observation],
           parent_action: Optional[Any] = None
       ) -> Dict[AgentID, Dict[str, Any]]:
           """
           Compute coordination signals for subordinates.

           Args:
               subordinate_observations: Obs from all subordinates
               parent_action: Optional action from parent's policy

           Returns:
               Dict mapping subordinate_id → signal dict
           """
           signals = {}

           for sub_id, obs in subordinate_observations.items():
               signal = self._compute_signal(obs, parent_action)
               signals[sub_id] = signal

           return signals

       def _compute_signal(self, obs, parent_action):
           """Helper method to compute signal for one subordinate."""
           return {'custom_field': value}

Custom Horizontal Protocol Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from powergrid.agents.protocols import HorizontalProtocol
   from powergrid.agents.base import Agent, Observation, AgentID
   from typing import Dict, Any, Optional

   class MyHorizontalProtocol(HorizontalProtocol):
       """My custom horizontal protocol."""

       def __init__(self, param1, param2):
           self.param1 = param1
           self.param2 = param2

       def coordinate(
           self,
           agents: Dict[AgentID, Agent],
           observations: Dict[AgentID, Observation],
           topology: Optional[Dict] = None
       ) -> Dict[AgentID, Dict[str, Any]]:
           """
           Coordinate peer agents.

           Args:
               agents: All participating agents
               observations: Observations from all agents
               topology: Optional network topology

           Returns:
               Dict mapping agent_id → coordination signal
           """
           signals = {}

           # Step 1: Collect information from all agents
           agent_data = self._collect_data(observations)

           # Step 2: Run global coordination algorithm
           results = self._run_coordination(agent_data, topology)

           # Step 3: Generate signals for each agent
           for agent_id in agents:
               signals[agent_id] = self._generate_signal(agent_id, results)

           return signals

Protocol Comparison
-------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 20 20 20

   * - Protocol
     - Type
     - Ownership
     - Use Case
     - Pros
     - Cons
   * - NoProtocol
     - Vertical
     - Agent
     - Baseline
     - Simple
     - No coordination
   * - PriceSignalProtocol
     - Vertical
     - Agent
     - Economic dispatch
     - Interpretable, fast
     - Needs price-responsive devices
   * - SetpointProtocol
     - Vertical
     - Agent
     - Centralized control
     - Full control
     - High-dimensional action space
   * - NoHorizontalProtocol
     - Horizontal
     - Environment
     - Baseline
     - Simple
     - No peer coordination
   * - PeerToPeerTradingProtocol
     - Horizontal
     - Environment
     - Local energy market
     - Distributed, fair
     - Requires accurate demand estimates
   * - ConsensusProtocol
     - Horizontal
     - Environment
     - Frequency/voltage regulation
     - Robust, distributed
     - Slow convergence

Best Practices
--------------

1. **Start Simple**

   - Begin with ``NoProtocol`` or ``NoHorizontalProtocol``
   - Add coordination only when needed

2. **Test Protocols Independently**

   - Unit test protocols with mock agents
   - Verify market clearing logic before integration

3. **Profile Performance**

   - Horizontal protocols run every timestep (keep fast!)
   - Cache expensive computations

4. **Document Assumptions**

   - Specify what agents must include in observations
   - Example: P2P trading needs ``net_demand``, ``marginal_cost``

5. **Handle Edge Cases**

   - Empty bid/offer lists (no trades)
   - Infeasible setpoints (clip to device limits)
   - Non-convergence (timeout after max iterations)

6. **Validate Results**

   - Check energy balance: sum of trades = 0
   - Verify feasibility: all trades respect device limits
   - Monitor market metrics: price, trade volume, welfare

API Reference
-------------

.. automodule:: powergrid.core.protocols
   :members:
   :undoc-members:
   :show-inheritance:
