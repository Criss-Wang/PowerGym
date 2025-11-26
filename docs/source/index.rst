:html_theme.sidebar_secondary.remove:

.. title:: Welcome to PowerGrid!

.. toctree::
   :hidden:
   :maxdepth: 1

   Getting Started <getting_started>
   Use Cases <use_cases/index>
   Examples <examples/index>
   Documentation <docs/index>
   API Reference <api/index>

Welcome to PowerGrid
====================

An open-source framework to build and scale multi-agent reinforcement learning for smart grid control.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: sd-text-center

   .. grid-item-card:: 🚀 Get Started with PowerGrid
      :link: getting_started
      :link-type: doc

      Quick start guide to set up PowerGrid and run your first multi-agent simulation

   .. grid-item-card:: 📦 Install PowerGrid
      :link: user_guide/installation
      :link-type: doc

      Install PowerGrid with pip and set up your development environment

   .. grid-item-card:: 📚 PowerGrid Example Gallery
      :link: examples/index
      :link-type: doc

      Explore example networks, training scripts, and use cases

Scale with PowerGrid
--------------------

.. tab-set::

   .. tab-item:: Multi-Agent Control

      .. code-block:: python

         from powergrid.envs import NetworkedGridEnv

         # Create multi-agent environment
         env = NetworkedGridEnv({
             'env_name': 'ieee13_mg',
             'mode': 'distributed',
             'num_microgrids': 2
         })

         # Run episode with multiple GridAgents
         observations, infos = env.reset()
         for step in range(96):  # 24 hours
             actions = {agent: policy(obs) for agent, obs in observations.items()}
             observations, rewards, dones, truncs, infos = env.step(actions)

   .. tab-item:: Hierarchical Coordination

      .. code-block:: python

         from powergrid.agents import GridAgent
         from powergrid.core.protocols import PriceSignalProtocol

         # GridAgent coordinates subordinate devices
         grid_agent = GridAgent(
             agent_id='MG1',
             subordinates=[battery, generator, solar],
             vertical_protocol=PriceSignalProtocol(initial_price=50.0)
         )

   .. tab-item:: Peer-to-Peer Trading

      .. code-block:: python

         from powergrid.core.protocols import PeerToPeerTradingProtocol

         # Enable local energy markets between microgrids
         env = NetworkedGridEnv({
             'env_name': 'ieee13_mg',
             'horizontal_protocol': PeerToPeerTradingProtocol(trading_fee=0.01)
         })

   .. tab-item:: Training with RLlib

      .. code-block:: python

         from ray.rllib.algorithms.ppo import PPOConfig

         config = (
             PPOConfig()
             .environment(NetworkedGridEnv, env_config={'env_name': 'ieee13_mg'})
             .training(lr=3e-4, train_batch_size=4000)
         )
         trainer = config.build()
         trainer.train()

Beyond the Basics
-----------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: ⚡ PowerGrid Core
      :link: core/index
      :link-type: doc

      Scale multi-agent control with hierarchical agents, flexible protocols, and realistic AC power flow physics

   .. grid-item-card:: 🌐 Environments
      :link: environments/index
      :link-type: doc

      Deploy on standard IEEE test feeders (13-bus, 34-bus) and CIGRE networks with distributed energy resources

   .. grid-item-card:: 🤖 Multi-Agent RL
      :link: user_guide/training
      :link-type: doc

      Train coordinated policies with RLlib MAPPO, implement custom protocols, and scale to hundreds of agents
