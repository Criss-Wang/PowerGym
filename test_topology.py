"""Check network topology and connectivity."""
import pandapower as pp
import pandapower.topology as top
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.envs.configs.config_loader import load_config

env_config = load_config('ieee34_ieee13')
env_config['train'] = True
env_config['max_episode_steps'] = 24

env = MultiAgentMicrogrids(env_config)

# Check active external grids
print('External grids:')
print(env.net.ext_grid[['name', 'bus', 'in_service']])

# Check for isolated buses
isolated = top.unsupplied_buses(env.net)
print(f'\nIsolated/unsupplied buses: {isolated}')

# Check connected components
mg = top.create_nxgraph(env.net)
import networkx as nx
n_components = nx.number_connected_components(mg.to_undirected())
print(f'Number of connected components: {n_components}')

if n_components > 1:
    components = list(nx.connected_components(mg.to_undirected()))
    for i, comp in enumerate(components):
        print(f'\nComponent {i+1}: {len(comp)} buses')
        if len(comp) < 10:
            print(f'  Buses: {sorted(comp)}')

# Check ext_grid buses
print('\nExt grid buses and their in_service status:')
for idx, row in env.net.ext_grid.iterrows():
    bus_name = env.net.bus.loc[row.bus, 'name']
    print(f'  Bus {row.bus} ({bus_name}): in_service={row.in_service}')

# Reset and check again
print('\n' + '='*70)
print('After reset...')
print('='*70)
obs, info = env.reset(seed=42)

print('\nExternal grids after reset:')
print(env.net.ext_grid[['name', 'bus', 'in_service']])

isolated_after = top.unsupplied_buses(env.net)
print(f'\nIsolated/unsupplied buses after reset: {isolated_after}')
