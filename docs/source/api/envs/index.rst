Environments
============

Base environment interfaces for HERON.

BaseEnv
-------

.. py:class:: heron.envs.base.BaseEnv

   Abstract base environment class.

   .. py:method:: reset(seed: int = None, options: dict = None) -> tuple
      :abstractmethod:

      Reset environment to initial state.

      :param seed: Random seed
      :param options: Additional options
      :returns: Tuple of (observations, infos)

   .. py:method:: step(actions: dict) -> tuple
      :abstractmethod:

      Execute one environment step.

      :param actions: Dict mapping agent IDs to actions
      :returns: Tuple of (observations, rewards, terminateds, truncateds, infos)

   .. py:attribute:: possible_agents
      :type: list[str]

      List of all possible agent IDs.

   .. py:attribute:: agents
      :type: list[str]

      List of currently active agent IDs.

   .. py:attribute:: observation_spaces
      :type: dict

      Observation space for each agent.

   .. py:attribute:: action_spaces
      :type: dict

      Action space for each agent.

RLlib Integration
-----------------

Wrap HERON environments for RLlib:

.. code-block:: python

   from ray.tune.registry import register_env

   def env_creator(config):
       env = MyHeronEnv(config)
       return env

   register_env("my_heron_env", env_creator)

   # Use in RLlib config
   config = PPOConfig().environment(env="my_heron_env", env_config={...})

Gymnasium Compatibility
-----------------------

For single-agent scenarios or wrapped multi-agent:

.. code-block:: python

   import gymnasium as gym
   from gymnasium import spaces

   class SingleAgentWrapper(gym.Env):
       """Wrap multi-agent env as single-agent."""

       def __init__(self, multi_env, agent_id: str):
           self.env = multi_env
           self.agent_id = agent_id
           self.observation_space = multi_env.observation_space(agent_id)
           self.action_space = multi_env.action_space(agent_id)

       def reset(self, seed=None, options=None):
           obs, info = self.env.reset(seed=seed, options=options)
           return obs[self.agent_id], info.get(self.agent_id, {})

       def step(self, action):
           actions = {self.agent_id: action}
           # Use fixed policy for other agents
           for other in self.env.agents:
               if other != self.agent_id:
                   actions[other] = self.env.action_space(other).sample()

           obs, rewards, terms, truncs, infos = self.env.step(actions)
           return (
               obs[self.agent_id],
               rewards[self.agent_id],
               terms.get(self.agent_id, terms["__all__"]),
               truncs.get(self.agent_id, truncs["__all__"]),
               infos.get(self.agent_id, {})
           )
