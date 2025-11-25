Testing and Evaluation
======================

Evaluate trained policies and analyze performance.

Rollout Evaluation
------------------

Run policy rollouts to collect episodes:

.. code-block:: python

   from ray.rllib.algorithms.ppo import PPO
   from powergrid.envs import NetworkedGridEnv

   # Load trained policy
   trainer = PPO.from_checkpoint('path/to/checkpoint')

   # Create environment
   env = NetworkedGridEnv(config)

   # Run episodes
   num_episodes = 10
   episode_rewards = []

   for episode in range(num_episodes):
       obs, info = env.reset()
       done = False
       episode_reward = 0

       while not done:
           # Get actions from policy
           actions = {}
           for agent_id, agent_obs in obs.items():
               action = trainer.compute_single_action(agent_obs, policy_id='grid_policy')
               actions[agent_id] = action

           # Step environment
           obs, rewards, terms, truncs, infos = env.step(actions)
           episode_reward += sum(rewards.values())
           done = all(terms.values()) or all(truncs.values())

       episode_rewards.append(episode_reward)

   print(f"Average reward: {np.mean(episode_rewards):.2f}")

Performance Metrics
-------------------

Track key performance indicators:

.. code-block:: python

   metrics = {
       'total_cost': [],
       'voltage_violations': [],
       'line_overloads': [],
       'renewable_curtailment': [],
       'battery_cycles': []
   }

   for episode in range(num_episodes):
       # Run episode and collect metrics
       obs, info = env.reset()
       done = False

       while not done:
           actions = get_actions(obs)
           obs, rewards, terms, truncs, infos = env.step(actions)

           # Collect metrics
           metrics['total_cost'].append(infos['MG1']['cost'])
           metrics['voltage_violations'].append(infos['MG1']['v_violations'])
           # ... more metrics

           done = all(terms.values())

   # Analyze results
   for metric_name, values in metrics.items():
       print(f"{metric_name}: {np.mean(values):.3f} ± {np.std(values):.3f}")

Baseline Comparison
-------------------

Compare against rule-based baselines:

.. code-block:: python

   from powergrid.baselines import RuleBasedController

   # Trained policy
   trained_rewards = evaluate_policy(trained_trainer, env, num_episodes=100)

   # Rule-based baseline
   baseline_controller = RuleBasedController()
   baseline_rewards = evaluate_baseline(baseline_controller, env, num_episodes=100)

   # Compare
   improvement = (np.mean(trained_rewards) - np.mean(baseline_rewards)) / np.abs(np.mean(baseline_rewards)) * 100
   print(f"Improvement over baseline: {improvement:.1f}%")

Visualization
-------------

Plot episode trajectories:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Collect trajectory data
   trajectory = {
       'time': [],
       'voltage': [],
       'power': [],
       'soc': []
   }

   obs, info = env.reset()
   done = False
   t = 0

   while not done:
       actions = get_actions(obs)
       obs, rewards, terms, truncs, infos = env.step(actions)

       # Record state
       trajectory['time'].append(t)
       trajectory['voltage'].append(obs['MG1'].local['vm_pu'])
       trajectory['power'].append(obs['MG1'].local['p_mw'])
       trajectory['soc'].append(obs['MG1'].local['soc'])

       t += 1
       done = all(terms.values())

   # Plot
   fig, axes = plt.subplots(3, 1, figsize=(10, 8))

   axes[0].plot(trajectory['time'], trajectory['voltage'])
   axes[0].set_ylabel('Voltage (pu)')
   axes[0].axhline(0.95, color='r', linestyle='--', label='Limit')
   axes[0].axhline(1.05, color='r', linestyle='--')

   axes[1].plot(trajectory['time'], trajectory['power'])
   axes[1].set_ylabel('Power (MW)')

   axes[2].plot(trajectory['time'], trajectory['soc'])
   axes[2].set_ylabel('SoC')
   axes[2].set_xlabel('Time (steps)')

   plt.tight_layout()
   plt.savefig('trajectory.png')

Stress Testing
--------------

Test policy robustness under extreme conditions:

.. code-block:: python

   # High load scenario
   env_config = {
       'env_name': 'ieee13_mg',
       'load_scaling': 1.5,  # 150% peak load
       'renewable_penetration': 0.8
   }
   high_load_rewards = evaluate_policy(trainer, NetworkedGridEnv(env_config))

   # Equipment failure
   env_config['failed_devices'] = ['DG1']  # Generator fails
   failure_rewards = evaluate_policy(trainer, NetworkedGridEnv(env_config))

   # Price volatility
   env_config['price_volatility'] = 'high'
   volatile_rewards = evaluate_policy(trainer, NetworkedGridEnv(env_config))

Statistical Analysis
--------------------

Perform statistical tests:

.. code-block:: python

   from scipy import stats

   # Compare two policies
   policy_a_rewards = evaluate_policy(trainer_a, env, num_episodes=100)
   policy_b_rewards = evaluate_policy(trainer_b, env, num_episodes=100)

   # T-test
   t_stat, p_value = stats.ttest_ind(policy_a_rewards, policy_b_rewards)
   print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")

   if p_value < 0.05:
       print("Policies are significantly different (p < 0.05)")

Export Results
--------------

Save evaluation results:

.. code-block:: python

   import json
   import pandas as pd

   # Save metrics to JSON
   results = {
       'mean_reward': float(np.mean(episode_rewards)),
       'std_reward': float(np.std(episode_rewards)),
       'metrics': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                   for k, v in metrics.items()}
   }

   with open('evaluation_results.json', 'w') as f:
       json.dump(results, f, indent=2)

   # Save trajectories to CSV
   df = pd.DataFrame(trajectory)
   df.to_csv('trajectory.csv', index=False)

See Also
--------

- :doc:`training` - Training policies
- :doc:`configuration` - Environment configuration
