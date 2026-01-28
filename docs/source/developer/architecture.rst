Developer Architecture
======================

Internal architecture and design decisions for contributors.

Code Organization
-----------------

.. code-block:: text

   powergrid/
   ├── agents/          # Agent implementations
   │   ├── base.py      # Base Agent class
   │   ├── grid_agent.py
   │   ├── device_agent.py
   │   └── proxy_agent.py  # Extends heron.agents.ProxyAgent
   ├── core/            # Core data structures
   │   ├── action.py
   │   ├── observation.py
   │   ├── state.py
   │   ├── protocols.py
   │   └── policies.py
   ├── devices/         # Device models
   │   ├── base.py
   │   ├── storage.py
   │   ├── generator.py
   │   └── ...
   ├── envs/            # Environments
   │   ├── base_env.py
   │   ├── networked_grid_env.py
   │   └── multi_agent/
   ├── features/        # Feature extraction
   ├── messaging/       # Message broker
   └── utils/           # Utilities

Design Principles
-----------------

**Separation of Concerns**

- Protocols define coordination mechanisms
- Policies define decision strategies
- Agents execute actions
- Devices model physics

**Modularity**

- Swap protocols without changing agents
- Extend device models independently
- Compose features for observations

**Type Safety**

Use dataclasses and type hints:

.. code-block:: python

   from dataclasses import dataclass
   from typing import Dict, Optional

   @dataclass
   class DeviceState:
       p_mw: float
       q_mvar: float
       vm_pu: Optional[float] = None

Key Abstractions
----------------

Agent
~~~~~

Abstract interface for all agents:

.. code-block:: python

   class Agent(ABC):
       @abstractmethod
       def observe(self, state: GlobalState) -> Observation:
           """Extract observation from global state."""
           pass

       @abstractmethod
       def act(self, observation: Observation) -> Action:
           """Compute action from observation."""
           pass

       @abstractmethod
       def reset(self):
           """Reset agent state."""
           pass

Protocol
~~~~~~~~

Coordination mechanism interface:

.. code-block:: python

   class Protocol(ABC):
       @abstractmethod
       def coordinate(self, observations, actions) -> Dict[AgentID, Message]:
           """Coordinate agents and return signals."""
           pass

Device
~~~~~~

Physical device model:

.. code-block:: python

   class Device(ABC):
       @abstractmethod
       def step(self, action: Action, dt: float) -> DeviceState:
           """Simulate one timestep."""
           pass

Adding New Features
-------------------

Adding a New Device
~~~~~~~~~~~~~~~~~~~

1. Create device class in ``powergrid/devices/``
2. Implement ``step()``, ``get_state()``, ``reset()``
3. Register device type in device factory
4. Add tests in ``tests/devices/``
5. Update documentation

Adding a New Protocol
~~~~~~~~~~~~~~~~~~~~~

1. Subclass ``VerticalProtocol`` or ``HorizontalProtocol``
2. Implement ``coordinate()`` method
3. Add to ``powergrid/core/protocols.py``
4. Add tests
5. Document in protocol guide

Adding a New Environment
~~~~~~~~~~~~~~~~~~~~~~~~

1. Create environment in ``powergrid/envs/``
2. Define PandaPower network
3. Configure agents and devices
4. Implement reward function
5. Add tests
6. Add to environment registry

Testing Strategy
----------------

**Unit Tests**

Test individual components in isolation:

.. code-block:: python

   def test_price_signal_protocol():
       protocol = PriceSignalProtocol(initial_price=50.0)
       signals = protocol.coordinate(mock_observations, parent_action=60.0)
       assert all(s['price'] == 60.0 for s in signals.values())

**Integration Tests**

Test component interactions:

.. code-block:: python

   def test_agent_protocol_integration():
       agent = GridAgent(...)
       obs = agent.observe(state)
       action = agent.act(obs)
       signals = agent.vertical_protocol.coordinate(sub_obs, action)
       # Verify signals are valid

**End-to-End Tests**

Test full environment rollouts:

.. code-block:: python

   def test_full_episode():
       env = NetworkedGridEnv(config)
       obs, info = env.reset()

       for _ in range(100):
           actions = {aid: env.action_space(aid).sample() for aid in env.agents}
           obs, rewards, terms, truncs, infos = env.step(actions)

       # Verify episode completed successfully

Performance Optimization
------------------------

**Profiling**

Use ``cProfile`` to identify bottlenecks:

.. code-block:: bash

   python -m cProfile -o profile.stats train_script.py
   python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumtime').print_stats(20)"

**Common Bottlenecks**

- Power flow computation (use sparse matrices)
- Feature extraction (cache when possible)
- Message passing (batch operations)

**Optimization Tips**

- Vectorize operations with NumPy
- Use ``@lru_cache`` for expensive computations
- Avoid deep copying large objects
- Profile before optimizing

Release Process
---------------

1. Update version in ``setup.py`` and ``__init__.py``
2. Update ``CHANGELOG.md``
3. Create release branch: ``git checkout -b release/vX.Y.Z``
4. Run full test suite
5. Build documentation
6. Create GitHub release
7. Publish to PyPI: ``python -m build && twine upload dist/*``

See Also
--------

- :doc:`contributing` - Contribution guidelines
- `GitHub Repository <https://github.com/yourusername/powergrid>`_
