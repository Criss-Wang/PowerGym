Environments
============

PowerGrid environments provide realistic power grid simulations for multi-agent reinforcement learning.

Available Environments
----------------------

.. toctree::
   :maxdepth: 1

   ieee13
   ieee34
   cigre_mv

Environment Interface
---------------------

All PowerGrid environments follow the PettingZoo ParallelEnv API:

.. code-block:: python

   from powergrid.envs import NetworkedGridEnv

   env = NetworkedGridEnv(config)
   observations, infos = env.reset()

   while not done:
       actions = {agent: policy(obs) for agent, obs in observations.items()}
       observations, rewards, terminations, truncations, infos = env.step(actions)

Key Methods
-----------

**reset(seed, options)**

Reset environment to initial state.

- Returns: ``observations, infos``
- ``observations``: Dict[agent_id, Observation]
- ``infos``: Dict[agent_id, dict] with auxiliary info

**step(actions)**

Execute one timestep with agent actions.

- Args: ``actions`` - Dict[agent_id, Action]
- Returns: ``observations, rewards, terminations, truncations, infos``

**observation_space(agent)**

Get observation space for an agent.

**action_space(agent)**

Get action space for an agent.

**agents**

List of agent IDs currently active in the environment.

Configuration
-------------

Environments are configured via YAML or dict:

.. code-block:: yaml

   env_name: ieee13_mg

   microgrids:
     - name: MG1
       devices:
         - type: generator
           bus: 632
           p_mw: 0.5
         - type: storage
           bus: 633
           max_e_mwh: 1.0

   mode: distributed
   timestep_minutes: 15
   max_steps: 96  # 24 hours

See :doc:`/user_guide/configuration` for full configuration options.

Customization
-------------

Create custom environments by:

1. Defining custom grid topology (PandaPower network)
2. Specifying agent assignments
3. Configuring observation/action features
4. Implementing custom reward functions

Example:

.. code-block:: python

   from powergrid.envs import NetworkedGridEnv
   from powergrid.core import RewardFunction

   class CustomReward(RewardFunction):
       def compute(self, state, actions):
           # Your reward logic
           return rewards

   config = {
       'network': my_pandapower_net,
       'agents': agent_config,
       'reward_fn': CustomReward()
   }

   env = NetworkedGridEnv(config)
