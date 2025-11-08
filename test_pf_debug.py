"""Debug power flow convergence in detail."""
import pandapower as pp
import numpy as np
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.envs.configs.config_loader import load_config

env_config = load_config('ieee34_ieee13')
env_config['train'] = True
env_config['max_episode_steps'] = 24

env = MultiAgentMicrogrids(env_config)
print('Initial power flow converged:', env.net.converged)

# Reset
obs, info = env.reset(seed=42)
print('\nAfter reset converged:', env.net.converged)

# Check total generation vs load
total_load = env.net.load[env.net.load.in_service].p_mw.sum()
total_sgen = env.net.sgen[env.net.sgen.in_service].p_mw.sum()
total_storage = env.net.storage[env.net.storage.in_service].p_mw.sum()

print(f'\nPower balance:')
print(f'  Total load: {total_load:.4f} MW')
print(f'  Total sgen: {total_sgen:.4f} MW')
print(f'  Total storage: {total_storage:.4f} MW')
print(f'  Net generation: {total_sgen + total_storage:.4f} MW')
print(f'  Balance (gen - load): {total_sgen + total_storage - total_load:.4f} MW')

# Check if any buses have issues
print(f'\nBus voltage initialization:')
print(f'  Min: {env.net.bus.vn_kv.min():.4f} kV')
print(f'  Max: {env.net.bus.vn_kv.max():.4f} kV')
print(f'  Any NaN: {env.net.bus.vn_kv.isna().any()}')

# Try power flow with more iterations
print('\nTrying power flow with 50 iterations...')
try:
    pp.runpp(env.net, max_iteration=50, calculate_voltage_angles=True)
    print(f'Converged: {env.net.converged}')
    if env.net.converged:
        print('Success!')
except Exception as e:
    print(f'Failed: {e}')

# Try with init='results' (warm start from previous solution)
print('\nTrying with init="dc" (DC power flow initialization)...')
try:
    pp.runpp(env.net, init='dc', max_iteration=50)
    print(f'Converged: {env.net.converged}')
    if env.net.converged:
        print('Success with DC init!')
except Exception as e:
    print(f'Failed: {e}')

# Check if problem is with reactive power
print('\nChecking reactive power settings...')
print('Sgen Q control:')
for idx, row in env.net.sgen.head(6).iterrows():
    print(f'  {row["name"]}: q_mvar={row.q_mvar}, max_q={row.max_q_mvar}, min_q={row.min_q_mvar}')

print('\nStorage Q control:')
for idx, row in env.net.storage.iterrows():
    print(f'  {row["name"]}: q_mvar={row.q_mvar}, max_q={row.max_q_mvar}, min_q={row.min_q_mvar}')
