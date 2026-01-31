"""NetworkedGridEnv: Multi-agent environment for networked microgrids using agent classes.

This is a modernized version of the legacy NetworkedGridEnv that replaces GridEnv
with PowerGridAgent while maintaining identical environment logic and API.

Inherits from HERON's PettingZooParallelEnv adapter for proper integration with
the HERON framework while maintaining PettingZoo compatibility.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandapower as pp
from gymnasium.spaces import Box, Dict as SpaceDict, Discrete, MultiDiscrete

from powergrid.agents.power_grid_agent import PowerGridAgent
from powergrid.agents.proxy_agent import ProxyAgent
from heron.envs.adapters import PettingZooParallelEnv
from heron.protocols.base import NoProtocol, Protocol
from heron.messaging.base import ChannelManager, Message, MessageBroker, MessageType
from heron.messaging.in_memory_broker import InMemoryBroker
from heron.utils.typing import AgentID


class NetworkedGridEnv(PettingZooParallelEnv):
    """Base environment for networked power grids with multi-agent control.

    This environment supports both centralized and distributed execution modes:

    Centralized Mode (centralized=True):
    - Agents directly access and modify the PandaPower network object
    - All state updates happen synchronously through direct method calls
    - Traditional multi-agent RL setup with full observability
    - No message broker required

    Distributed Mode (centralized=False):
    - Agents communicate via message broker, never accessing net directly
    - Environment publishes network state (voltages, line loading) to agents via messages
    - Agents publish state updates (P, Q, status) to environment via messages
    - Mimics realistic distributed control systems with limited communication

    Inherits from HERON's PettingZooParallelEnv adapter, which provides:
    - HeronEnvCore functionality (agent management, message broker integration)
    - PettingZoo ParallelEnv interface compatibility

    Attributes:
        env_config: Configuration dictionary with keys:
            - centralized: bool, execution mode (default: False)
            - max_episode_steps: int, episode length (default: 24)
            - train: bool, training mode flag (default: True)
            - message_broker: str, broker type for distributed mode (default: 'in_memory')
        centralized: bool, whether to use centralized or distributed execution
        message_broker: MessageBroker instance for distributed communication, or None
        proxy_agent: ProxyAgent instance for managing network state distribution in distributed mode
        agent_dict: Dictionary of all grid agents in the environment
        net: PandaPower network object containing the complete power system
    """

    def __init__(self, env_config):
        # Extract distributed mode from config (centralized=True means distributed=False)
        centralized = env_config.get('centralized', False)

        # Build message broker first (needed for parent init)
        broker = self._create_message_broker(env_config, centralized)

        # Initialize HERON's PettingZooParallelEnv adapter
        super().__init__(
            env_id=env_config.get('env_id'),
            message_broker=broker,
            distributed=not centralized,
        )

        self._name = "NetworkedGridEnv"
        self.data_size: int = 0
        self._t: int = 0  # current timestep
        self._total_days: int = 0  # total number of days in the dataset

        self.env_config = env_config
        self.max_episode_steps = env_config.get('max_episode_steps', 24)
        self.centralized = centralized
        self.train = env_config.get('train', True)

        # Build environment components
        self.agent_dict = self._build_agents()
        self.net = self._build_net()
        self.proxy_agent = self._build_proxy_agent()

        # Initialize spaces and PettingZoo attributes
        self._init_space()
        self._set_agent_ids(list(self.agent_dict.keys()))

    @staticmethod
    def _create_message_broker(env_config: Dict, centralized: bool) -> Optional[MessageBroker]:
        """Create message broker based on config.

        Static method to allow calling before super().__init__().

        Args:
            env_config: Environment configuration
            centralized: Whether running in centralized mode

        Returns:
            MessageBroker instance or None if centralized
        """
        if centralized:
            return None

        broker_type = env_config.get('message_broker', 'in_memory')
        if broker_type == 'in_memory':
            return InMemoryBroker()
        elif broker_type is None:
            return None
        else:
            raise ValueError(f"Unsupported message broker type: {broker_type}")

    @property
    def actionable_agents(self):
        """Get agents that have actionable devices."""
        return {
            n: a for n, a in self.agent_dict.items()
            if len(a.get_device_action_spaces()) > 0
        }

    @abstractmethod
    def _build_net(self):
        """Build the PandaPower network.

        Returns:
            pandapower.Network: The constructed power network
        """
        pass

    @abstractmethod
    def _reward_and_safety(self):
        """Compute rewards and safety metrics.

        Returns:
            Tuple of (rewards_dict, safety_dict)
        """
        pass

    @abstractmethod
    def _build_agents(self) -> Dict[AgentID, PowerGridAgent]:
        """Build and return the agent dictionary.

        Returns:
            Dictionary mapping agent IDs to PowerGridAgent instances
        """
        pass

    def _build_proxy_agent(self) -> Optional[ProxyAgent]:
        """Build ProxyAgent for managing network state distribution.

        Returns:
            ProxyAgent instance in distributed mode, None in centralized mode
        """
        # If centralized mode, no proxy agent needed
        if self.centralized or self.message_broker is None:
            return None

        # Get list of all agent IDs that will receive network state
        subordinate_agent_ids = list(self.agent_dict.keys())

        # Create proxy agent
        proxy = ProxyAgent(
            agent_id="proxy_agent",
            message_broker=self.message_broker,
            env_id=self.env_id,
            subordinate_agents=subordinate_agent_ids,
            visibility_rules=None,  # Default: all agents see all state
        )

        return proxy

    def _send_actions_to_agent(self, agent_id: AgentID, action: Any):
        """Send action to agent via message broker.

        Args:
            agent_id: Target agent ID
            action: Action to send
        """
        if self.message_broker is None:
            raise RuntimeError("Message broker is not initialized.")
        channel = ChannelManager.action_channel(
            self._name,
            agent_id,
            self.env_id
        )
        message = Message(
            env_id=self.env_id,
            sender_id=self._name,
            recipient_id=agent_id,
            timestamp=self._t,
            message_type=MessageType.ACTION,
            payload={'action': action}
        )
        self.message_broker.publish(channel, message)

    def _consume_all_state_updates(self) -> List[Dict[str, Any]]:
        """Consume all state updates from devices via message broker.

        Returns:
            List of state update payloads from all devices
        """
        if not self.message_broker:
            raise RuntimeError("Message broker required for distributed mode")

        channel = ChannelManager.state_update_channel(self.env_id)
        messages = self.message_broker.consume(
            channel,
            recipient_id="environment",
            env_id=self.env_id,
            clear=True
        )
        return [msg.payload for msg in messages]

    def _apply_state_updates_to_net(self, updates: List[Dict[str, Any]]) -> None:
        """Apply device state updates to pandapower network.

        Args:
            updates: List of state update payloads from devices
        """
        for update in updates:
            agent_id = update.get('agent_id')
            device_type = update.get('device_type')

            if not agent_id or not device_type:
                continue

            # Find the grid agent that owns this device
            grid_agent = None
            for agent in self.agent_dict.values():
                if isinstance(agent, PowerGridAgent):
                    if agent_id in agent.devices:
                        grid_agent = agent
                        break

            if not grid_agent:
                continue

            # Construct pandapower element name (follows PowerGridAgent._add_sgen convention)
            element_name = f"{grid_agent.name} {agent_id}"

            # Get pandapower element index
            try:
                element_idx = pp.get_element_index(self.net, device_type, element_name)
            except (KeyError, IndexError, UserWarning):
                # Element not found in network, skip
                continue

            # Update pandapower network with device state
            self.net[device_type].loc[element_idx, 'p_mw'] = update.get('P_MW', 0.0)
            if 'Q_MVAr' in update:
                self.net[device_type].loc[element_idx, 'q_mvar'] = update.get('Q_MVAr', 0.0)
            self.net[device_type].loc[element_idx, 'in_service'] = update.get('in_service', True)

    def _update_net(self):
        """Update network state based on agent actions."""
        if self.centralized:
            for agent in self.agent_dict.values():
                agent.update_state(self.net, self._t)
        else:
            # Update load scaling for all agents
            self._update_loads_distributed()
            # Consume all state updates from devices at once
            state_updates = self._consume_all_state_updates()
            # Apply state updates to pandapower network
            self._apply_state_updates_to_net(state_updates)

    def _publish_network_state_to_agents(self):
        """Publish network state to agents via ProxyAgent.

        In distributed mode, the environment sends network state to the ProxyAgent,
        which then distributes it to individual agents based on visibility rules.
        This ensures agents never directly access the network object.
        """
        if not self.message_broker or not self.proxy_agent:
            return

        # Collect aggregated network state for all agents
        aggregated_network_state = {
            'converged': self.net.get('converged', False),
            'agents': {}
        }

        for agent in self.agent_dict.values():
            if not isinstance(agent, PowerGridAgent):
                continue

            # Extract network state relevant to this agent
            agent_network_state = {
                'converged': self.net.get('converged', False),
                'device_results': {},
                'bus_voltages': {},
                'line_loading': {}
            }

            # Device results (sgen outputs)
            for device_name in agent.sgen.keys():
                element_name = f"{agent.name} {device_name}"
                try:
                    idx = pp.get_element_index(self.net, 'sgen', element_name)
                    agent_network_state['device_results'][device_name] = {
                        'p_mw': float(self.net.res_sgen.loc[idx, 'p_mw']),
                        'q_mvar': float(self.net.res_sgen.loc[idx, 'q_mvar'])
                    }
                except (KeyError, IndexError, UserWarning):
                    pass

            # Bus voltages for this agent's buses
            if agent_network_state['converged']:
                try:
                    local_bus_ids = pp.get_element_index(self.net, 'bus', agent.name, False)
                    bus_voltages = self.net.res_bus.loc[local_bus_ids, 'vm_pu'].values
                    agent_network_state['bus_voltages'] = {
                        'vm_pu': bus_voltages.tolist(),
                        'overvoltage': float(np.maximum(bus_voltages - 1.05, 0).sum()),
                        'undervoltage': float(np.maximum(0.95 - bus_voltages, 0).sum())
                    }
                except (KeyError, UserWarning):
                    pass

                # Line loading for this agent's lines
                try:
                    local_line_ids = pp.get_element_index(self.net, 'line', agent.name, False)
                    line_loading = self.net.res_line.loc[local_line_ids, 'loading_percent'].values
                    agent_network_state['line_loading'] = {
                        'loading_percent': line_loading.tolist(),
                        'overloading': float(np.maximum(line_loading - 100, 0).sum() * 0.01)
                    }
                except (KeyError, UserWarning):
                    pass

            # Store agent-specific state in aggregated state
            aggregated_network_state['agents'][agent.agent_id] = agent_network_state

        # Send aggregated network state to ProxyAgent
        channel = ChannelManager.custom_channel("power_flow", self.env_id, "proxy_agent")
        message = Message(
            env_id=self.env_id,
            sender_id="environment",
            recipient_id="proxy_agent",
            timestamp=self._t,
            message_type=MessageType.RESULT,
            payload=aggregated_network_state
        )
        self.message_broker.publish(channel, message)

        # ProxyAgent receives and distributes to individual agents
        self.proxy_agent.receive_network_state_from_environment()
        self.proxy_agent.distribute_network_state_to_agents()

    def _update_loads_distributed(self):
        """Update load scaling for all agents in distributed mode.

        In distributed mode, the environment directly updates the network
        without passing net to agents.
        """
        for agent in self.agent_dict.values():
            if not hasattr(agent, 'dataset') or agent.dataset is None:
                continue

            # Get load scaling from agent's dataset
            load_scaling = agent.dataset['load'][self._t]

            # Update network loads for this agent (environment does this, not agent)
            try:
                local_ids = pp.get_element_index(self.net, 'load', agent.name, False)
                self.net.load.loc[local_ids, 'scaling'] = load_scaling
                # Apply load rescaling directly (don't pass net to agent)
                self.net.load.loc[local_ids, 'scaling'] *= agent.load_scale
            except (KeyError, UserWarning):
                # No loads for this agent, skip
                continue

    def step(self, action_n: Dict[str, Any]):
        """Execute one environment step.

        For centralized case: Collective agent state update
        - Actions are computed and set on devices by each agent.
        - Agent state updates are done after actions are set and during _update_net.

        For decentralized case: Per agent state update
        - Actions are sent to agents via message broker.
        - Each agent steps its local devices and updates its local state.
        - Agent state updates are done before _update_net.

        Args:
            action_n: Dictionary mapping agent IDs to actions

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
        """
        # Set action for each agent
        if self.centralized:
            # note that action can be None here -> decentralized action computation per grid agent
            for name, action in action_n.items():
                if name in self.actionable_agents:
                    # Get observation for this agent
                    obs = self.actionable_agents[name].observe()
                    # Compute and set actions on devices
                    self.actionable_agents[name].act(obs, upstream_action=action)
        else:
            # Distributed mode: run agent steps concurrently
            import asyncio

            async def run_distributed_steps():
                tasks = []
                for agent_id, action in action_n.items():
                    if agent_id in self.actionable_agents:
                        self._send_actions_to_agent(agent_id, action)
                        tasks.append(self.actionable_agents[agent_id].step_distributed())
                await asyncio.gather(*tasks)

            asyncio.run(run_distributed_steps())

        # Update network states based on agent actions
        self._update_net()

        # Run power flow for the whole network
        try:
            pp.runpp(self.net)
        except Exception:
            self.net['converged'] = False

        # ==== Sync states to all agents ====
        if self.centralized:
            # Centralized: agents access net directly
            for agent_id, agent in self.agent_dict.items():
                agent.sync_global_state(self.net, self._t)
        else:
            # Distributed: send network state to agents via messages
            self._publish_network_state_to_agents()

        # ==== Compute rewards and safety ====
        if self.centralized:
            # Centralized: agents compute from net directly
            for agent in self.agent_dict.values():
                agent.update_cost_safety(self.net)
        else:
            # Distributed: agents compute from locally received network state (via messages)
            for agent in self.agent_dict.values():
                agent.update_cost_safety(None)

        # Get rewards and safety from subclass
        rewards, safety = self._reward_and_safety()

        # Share rewards if configured
        if self.env_config.get('share_reward'):
            shared_reward = np.mean(list(rewards.values()))
            rewards = {name: shared_reward for name in self.agent_dict}

        # ==== Update environment and prepare outputs ====
        # Increment episode step counter
        self._episode_step += 1

        # Timestep counter (for dataset indexing)
        self._t = self._t + 1 if self._t < self.data_size else 0

        # Done when we've completed max_episode_steps
        done = self._episode_step >= self.max_episode_steps
        terminateds = {"__all__": done}
        truncateds = {"__all__": done}

        # Info - wrap safety values in dicts for Ray compatibility
        infos = {agent_id: {"safety": safety_val} for agent_id, safety_val in safety.items()}

        return self._get_obs(), rewards, terminateds, truncateds, infos

    def reset(self, seed=None, options=None):
        """Reset environment and all agents.

        Args:
            seed: Random seed
            options: Reset options (unused)

        Returns:
            Tuple of (observations_dict, info_dict)
        """
        # Initialize RNG
        if seed is not None:
            np.random.seed(seed)

        # Reset episode step counter
        self._episode_step = 0

        # Reset all agents
        if self.train:
            self._day = np.random.randint(0, self._total_days - 1)
            self._t = self._day * self.max_episode_steps
            for agent in self.agent_dict.values():
                agent.reset(seed=seed)
        else:
            if hasattr(self, '_day'):
                self._day += 1
                self._t = self._day * self.max_episode_steps
            else:
                self._t, self._day = 0, 0
            for agent in self.agent_dict.values():
                agent.reset(seed=seed)

        # Update initial states
        for agent in self.agent_dict.values():
            agent.update_state(self.net, self._t)

        # Initial power flow
        try:
            pp.runpp(self.net)
        except Exception:
            self.net['converged'] = False

        # In distributed mode, publish initial network state to agents via ProxyAgent
        if not self.centralized and self.proxy_agent is not None:
            self._publish_network_state_to_agents()

        info = {}

        return self._get_obs(), info

    def _get_obs(self):
        """Get observations for all agents.

        In centralized mode, agents observe with direct network access.
        In distributed mode, agents observe without network access (they use
        ProxyAgent state received via messages).

        Returns:
            Dict mapping agent_id to observation array
        """
        obs_dict = {}
        for agent_id, agent in self.agent_dict.items():
            if self.centralized:
                # Centralized mode: pass network directly
                obs = agent.observe(net=self.net)
            else:
                # Distributed mode: no network access
                obs = agent.observe(net=None)
            # Extract state from observation
            obs_dict[agent_id] = obs.local['state']

        return obs_dict

    def _init_space(self):
        """Initialize action and observation spaces for all agents.

        In centralized mode, observation space includes network load information.
        In distributed mode, observation space excludes load information (agents
        don't have direct network access).
        """
        ac_spaces = {}
        ob_spaces = {}

        for name, agent in self.agent_dict.items():
            ac_spaces[name] = agent.get_grid_action_space()
            # Pass net only in centralized mode
            net_for_obs = self.net if self.centralized else None
            ob_spaces[name] = agent.get_grid_observation_space(net_for_obs)

        # Use adapter's _init_spaces method to set up spaces
        self._init_spaces(ac_spaces, ob_spaces)
