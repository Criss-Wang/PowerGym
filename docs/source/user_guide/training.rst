Training
========

Train multi-agent reinforcement learning policies with PowerGrid.

Training with RLlib
-------------------

PowerGrid integrates with Ray RLlib for scalable MARL training.

Basic Training
~~~~~~~~~~~~~~

.. code-block:: python

   from ray import tune
   from ray.rllib.algorithms.ppo import PPOConfig
   from powergrid.envs import NetworkedGridEnv

   config = (
       PPOConfig()
       .environment(NetworkedGridEnv, env_config={
           'env_name': 'ieee13_mg',
           'mode': 'distributed'
       })
       .framework('torch')
       .training(
           lr=3e-4,
           train_batch_size=4000,
           sgd_minibatch_size=128
       )
   )

   tune.run(
       'PPO',
       config=config.to_dict(),
       stop={'training_iteration': 100}
   )

MAPPO Training
~~~~~~~~~~~~~~

Multi-Agent PPO with parameter sharing:

.. code-block:: python

   from ray.rllib.algorithms.ppo import PPOConfig

   config = (
       PPOConfig()
       .environment(NetworkedGridEnv, env_config={...})
       .multi_agent(
           policies={
               'grid_policy': (None, obs_space, act_space, {}),
           },
           policy_mapping_fn=lambda agent_id, *args, **kwargs: 'grid_policy'
       )
   )

Independent Policies
~~~~~~~~~~~~~~~~~~~~

Train separate policies per agent:

.. code-block:: python

   config = (
       PPOConfig()
       .multi_agent(
           policies={
               'MG1_policy': (None, obs_space_mg1, act_space_mg1, {}),
               'MG2_policy': (None, obs_space_mg2, act_space_mg2, {}),
           },
           policy_mapping_fn=lambda agent_id, *args, **kwargs: f'{agent_id}_policy'
       )
   )

Hyperparameters
---------------

Recommended hyperparameters for PowerGrid:

.. code-block:: python

   config = (
       PPOConfig()
       .training(
           lr=3e-4,                    # Learning rate
           train_batch_size=4000,      # Batch size
           sgd_minibatch_size=128,     # SGD minibatch
           num_sgd_iter=10,            # SGD iterations
           gamma=0.99,                 # Discount factor
           lambda_=0.95,               # GAE lambda
           clip_param=0.2,             # PPO clip
           vf_clip_param=10.0,         # Value function clip
           entropy_coeff=0.01,         # Exploration
       )
       .rollouts(
           num_rollout_workers=4,      # Parallel workers
           rollout_fragment_length=200 # Steps per worker
       )
   )

Monitoring Training
-------------------

Use TensorBoard to monitor training:

.. code-block:: bash

   tensorboard --logdir ~/ray_results

Key metrics to track:

- ``episode_reward_mean``: Average episode return
- ``policy_loss``: Policy gradient loss
- ``vf_loss``: Value function loss
- ``entropy``: Policy entropy (exploration)

Custom Metrics
~~~~~~~~~~~~~~

Log custom metrics from environment:

.. code-block:: python

   class CustomEnv(NetworkedGridEnv):
       def step(self, actions):
           obs, rewards, terms, truncs, infos = super().step(actions)

           # Add custom metrics
           for agent_id in self.agents:
               infos[agent_id]['voltage_violations'] = self.count_violations()
               infos[agent_id]['energy_cost'] = self.compute_cost()

           return obs, rewards, terms, truncs, infos

Checkpointing
-------------

Save and load checkpoints:

.. code-block:: python

   # Save checkpoint
   checkpoint_dir = trainer.save()

   # Load checkpoint
   from ray.rllib.algorithms.ppo import PPO
   trainer = PPO.from_checkpoint(checkpoint_dir)

   # Continue training
   for i in range(10):
       result = trainer.train()

Distributed Training
--------------------

Scale to multiple nodes with Ray cluster:

.. code-block:: bash

   # Start Ray cluster
   ray start --head --port=6379

   # On worker nodes
   ray start --address='<head-node-ip>:6379'

.. code-block:: python

   # Connect to cluster
   import ray
   ray.init(address='auto')

   # Training automatically uses cluster resources
   tune.run('PPO', config=config)

See Also
--------

- `RLlib Documentation <https://docs.ray.io/en/latest/rllib/index.html>`_
- :doc:`testing` - Evaluation and testing
- :doc:`configuration` - Environment configuration
