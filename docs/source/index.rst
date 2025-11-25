:html_theme.sidebar_secondary.remove:

.. title:: Welcome to PowerGrid!

.. toctree::
   :hidden:

   README
   getting_started
   user_guide/installation
   use_cases/index
   examples/index
   core/index
   environments/index
   agents/index
   devices/index
   protocols/index
   user_guide/index
   api/index
   developer/index

Welcome to PowerGrid
====================

An open-source framework to build and scale multi-agent reinforcement learning for smart grid control.

.. raw:: html

   <div class="container-fluid">
      <div class="row">
         <div class="col-md-4">
            <div class="text-center p-3">
               <h3>🚀 Get Started with PowerGrid</h3>
               <p>Quick start guide to set up PowerGrid and run your first multi-agent simulation.</p>
               <a href="getting_started.html" class="btn btn-primary">Get Started</a>
            </div>
         </div>
         <div class="col-md-4">
            <div class="text-center p-3">
               <h3>📦 Install PowerGrid</h3>
               <p>Install PowerGrid with pip and set up your development environment.</p>
               <a href="user_guide/installation.html" class="btn btn-primary">Install</a>
            </div>
         </div>
         <div class="col-md-4">
            <div class="text-center p-3">
               <h3>📚 PowerGrid Example Gallery</h3>
               <p>Explore example networks, training scripts, and use cases.</p>
               <a href="examples/index.html" class="btn btn-primary">Examples</a>
            </div>
         </div>
      </div>
   </div>

Scale with PowerGrid
--------------------

**Multi-Agent Control**

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

**Hierarchical Coordination**

.. code-block:: python

   from powergrid.agents import GridAgent
   from powergrid.core.protocols import PriceSignalProtocol

   # GridAgent coordinates subordinate devices
   grid_agent = GridAgent(
       agent_id='MG1',
       subordinates=[battery, generator, solar],
       vertical_protocol=PriceSignalProtocol(initial_price=50.0)
   )

**Peer-to-Peer Trading**

.. code-block:: python

   from powergrid.core.protocols import PeerToPeerTradingProtocol

   # Enable local energy markets between microgrids
   env = NetworkedGridEnv({
       'env_name': 'ieee13_mg',
       'horizontal_protocol': PeerToPeerTradingProtocol(trading_fee=0.01)
   })

**Training with RLlib**

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

.. raw:: html

   <div class="container-fluid">
      <div class="row">
         <div class="col-md-4">
            <h4>⚡ PowerGrid Core</h4>
            <p>Scale multi-agent control with hierarchical agents, flexible protocols, and realistic AC power flow physics.</p>
            <a href="core/overview.html">Learn more about PowerGrid Core →</a>
         </div>
         <div class="col-md-4">
            <h4>🌐 Environments</h4>
            <p>Deploy on standard IEEE test feeders (13-bus, 34-bus) and CIGRE networks with distributed energy resources.</p>
            <a href="environments/index.html">Learn more about Environments →</a>
         </div>
         <div class="col-md-4">
            <h4>🤖 Multi-Agent RL</h4>
            <p>Train coordinated policies with RLlib MAPPO, implement custom protocols, and scale to hundreds of agents.</p>
            <a href="user_guide/training.html">Learn more about Training →</a>
         </div>
      </div>
   </div>
