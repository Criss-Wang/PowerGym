"""Test reset and step behavior."""
import numpy as np
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.envs.configs.config_loader import load_config

env_config = load_config('ieee34_ieee13')
env_config['train'] = True
env_config['max_episode_steps'] = 24

env = MultiAgentMicrogrids(env_config)
print('Initial net.converged:', env.net.get('converged'))

# Reset
obs, info = env.reset(seed=42)
print('After reset net.converged:', env.net.get('converged'))
print('Observations have NaN:', {k: ('Yes' if np.isnan(v).any() else 'No') for k, v in obs.items()})

# Sample actions
actions = {name: space.sample() for name, space in env.action_spaces.items()}
print('\nSampled actions:')
for name, action in actions.items():
    print(f'  {name}: {action}')

# Step
obs_next, rewards, terminateds, truncateds, infos = env.step(actions)
print('\nAfter step net.converged:', env.net.get('converged'))
print('Rewards:', rewards)
print('Terminateds:', terminateds)
print('Truncateds:', truncateds)
