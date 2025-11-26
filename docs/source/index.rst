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

PowerGrid 2.0
=========

An open-source framework for multi-agent reinforcement learning in smart grid control.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: sd-text-center

   .. grid-item-card:: 🚀 Get Started
      :link: getting_started
      :link-type: doc

      Set up PowerGrid and run your first simulation

   .. grid-item-card:: 📦 Installation
      :link: user_guide/installation
      :link-type: doc

      Install with pip or from source

   .. grid-item-card:: 📚 Examples
      :link: examples/index
      :link-type: doc

      Browse example networks and training scripts

Quick Examples
--------------

.. tab-set::

   .. tab-item:: Multi-Agent Control

      .. code-block:: python

         from powergrid.envs import NetworkedGridEnv

         env = NetworkedGridEnv({
             'env_name': 'ieee13_mg',
             'mode': 'distributed',
             'num_microgrids': 2
         })

         observations, infos = env.reset()
         for step in range(96):
             actions = {agent: policy(obs) for agent, obs in observations.items()}
             observations, rewards, dones, truncs, infos = env.step(actions)

   .. tab-item:: Hierarchical Coordination

      .. code-block:: python

         from powergrid.agents import GridAgent
         from powergrid.core.protocols import PriceSignalProtocol

         grid_agent = GridAgent(
             agent_id='MG1',
             subordinates=[battery, generator, solar],
             vertical_protocol=PriceSignalProtocol(initial_price=50.0)
         )

   .. tab-item:: P2P Trading

      .. code-block:: python

         from powergrid.core.protocols import PeerToPeerTradingProtocol

         env = NetworkedGridEnv({
             'env_name': 'ieee13_mg',
             'horizontal_protocol': PeerToPeerTradingProtocol(trading_fee=0.01)
         })

   .. tab-item:: RLlib Training

      .. code-block:: python

         from ray.rllib.algorithms.ppo import PPOConfig

         config = (
             PPOConfig()
             .environment(NetworkedGridEnv, env_config={'env_name': 'ieee13_mg'})
             .training(lr=3e-4, train_batch_size=4000)
         )
         algo = config.build()
         algo.train()

Key Features
------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: ⚡ PowerGrid Core
      :link: core/index
      :link-type: doc

      Hierarchical agents, flexible protocols, and AC power flow

   .. grid-item-card:: 🌐 Environments
      :link: environments/index
      :link-type: doc

      IEEE test feeders and CIGRE networks with DERs

   .. grid-item-card:: 🤖 Multi-Agent RL
      :link: user_guide/training
      :link-type: doc

      RLlib integration with MAPPO and custom protocols
