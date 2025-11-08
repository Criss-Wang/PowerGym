"""Detailed debug of reset behavior."""
import pandapower as pp
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.envs.configs.config_loader import load_config

env_config = load_config('ieee34_ieee13')
env_config['train'] = True
env_config['max_episode_steps'] = 24

env = MultiAgentMicrogrids(env_config)
print('Initial net.converged:', env.net.get('converged'))

# Check storage before reset
print('\nStorage before reset:')
print(env.net.storage[['name', 'p_mw', 'sn_mva', 'in_service']].head())

# Check sgen before reset
print('\nSgen before reset:')
print(env.net.sgen[['name', 'p_mw', 'sn_mva', 'in_service']].head())

# Reset
print('\n' + '='*70)
print('Calling reset...')
print('='*70)
obs, info = env.reset(seed=42)

# Check storage after reset
print('\nStorage after reset:')
print(env.net.storage[['name', 'p_mw', 'sn_mva', 'soc_percent', 'in_service']].head())

# Check sgen after reset
print('\nSgen after reset:')
print(env.net.sgen[['name', 'p_mw', 'sn_mva', 'in_service']].head())

print('\nAfter reset net.converged:', env.net.get('converged'))

# Try manual power flow
print('\n' + '='*70)
print('Running manual power flow...')
print('='*70)
try:
    pp.runpp(env.net, max_iteration=10)
    print('Manual power flow converged:', env.net.converged)
except Exception as e:
    print(f'Manual power flow failed: {e}')
    import traceback
    traceback.print_exc()

# Check for any NaN or problematic values
print('\n' + '='*70)
print('Checking for problematic values...')
print('='*70)
print(f'Storage NaN p_mw: {env.net.storage.p_mw.isna().sum()}')
print(f'Storage NaN sn_mva: {env.net.storage.sn_mva.isna().sum()}')
print(f'Sgen NaN p_mw: {env.net.sgen.p_mw.isna().sum()}')
print(f'Sgen NaN sn_mva: {env.net.sgen.sn_mva.isna().sum()}')
print(f'Load NaN p_mw: {env.net.load.p_mw.isna().sum()}')
