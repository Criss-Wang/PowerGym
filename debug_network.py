"""Debug script to examine the network configuration."""

import pandapower as pp
from powergrid.envs.multi_agent.multi_agent_microgrids import MultiAgentMicrogrids
from powergrid.envs.configs.config_loader import load_config


def main():
    print("=" * 70)
    print("Network Configuration Debug")
    print("=" * 70)

    # Load config
    env_config = load_config('ieee34_ieee13')
    env_config['train'] = True
    env_config['max_episode_steps'] = 24

    # Create environment
    print("\nCreating environment...")
    env = MultiAgentMicrogrids(env_config)

    # Print network info
    net = env.net
    print(f"\nNetwork: {net.name}")
    print(f"Converged: {net.get('converged', 'N/A')}")

    print(f"\nBuses: {len(net.bus)}")
    print(f"Loads: {len(net.load)}")
    print(f"Static generators: {len(net.sgen)}")
    print(f"Storage: {len(net.storage)}")
    print(f"External grids: {len(net.ext_grid)}")
    print(f"Lines: {len(net.line)}")
    print(f"Transformers: {len(net.trafo)}")

    # Check for isolated buses or issues
    print("\n" + "-" * 70)
    print("External Grids:")
    print(net.ext_grid[['name', 'bus', 'in_service']])

    print("\n" + "-" * 70)
    print("Static Generators:")
    if len(net.sgen) > 0:
        print(net.sgen[['name', 'bus', 'p_mw', 'sn_mva', 'in_service']])

    print("\n" + "-" * 70)
    print("Storage Systems:")
    if len(net.storage) > 0:
        print(net.storage[['name', 'bus', 'p_mw', 'sn_mva', 'in_service']])

    print("\n" + "-" * 70)
    print("Loads:")
    if len(net.load) > 0:
        print(net.load[['name', 'bus', 'p_mw', 'in_service']].head(10))

    # Try running power flow
    print("\n" + "=" * 70)
    print("Running Power Flow...")
    print("=" * 70)
    try:
        pp.runpp(net, max_iteration=10)
        if net.converged:
            print("✓ Power flow converged!")
            print(f"  Max P loss: {net.res_line.pl_mw.max():.4f} MW")
            print(f"  Max Q loss: {net.res_line.ql_mvar.max():.4f} MVAr")
        else:
            print("✗ Power flow did NOT converge")
    except Exception as e:
        print(f"✗ Power flow failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Check for NaN or inf values in bus voltages
    print("\n" + "-" * 70)
    print("Bus Voltage Check:")
    if hasattr(net, 'res_bus') and len(net.res_bus) > 0:
        print(f"  NaN voltages: {net.res_bus.vm_pu.isna().sum()}")
        print(f"  Inf voltages: {(net.res_bus.vm_pu == float('inf')).sum()}")
        if net.res_bus.vm_pu.isna().sum() == 0:
            print(f"  Voltage range: [{net.res_bus.vm_pu.min():.4f}, {net.res_bus.vm_pu.max():.4f}] p.u.")
    else:
        print("  No bus results available")


if __name__ == '__main__':
    main()
