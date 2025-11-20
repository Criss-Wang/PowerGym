"""
NetworkedGridEnv: Multi-agent environment for networked microgrids using agent classes.

This is a modernized version of the legacy NetworkedGridEnv that replaces GridEnv
with PowerGridAgent while maintaining identical environment logic and API.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

import gymnasium.utils.seeding as seeding
import numpy as np
import pandapower as pp
from gymnasium.spaces import Box, Dict as SpaceDict, Discrete, MultiDiscrete
from pettingzoo import ParallelEnv

from powergrid.agents.grid_agent import PowerGridAgent
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.utils.typing import AgentID
from powergrid.messaging.base import MessageBroker, ChannelManager, Message, MessageType
from powergrid.messaging.memory import InMemoryBroker
from powergrid.utils.helpers import gen_uuid

class NetworkedGridEnv(ParallelEnv):
    def __init__(self, env_config):
        super().__init__()
        self._env_id = gen_uuid()
        self._name = "NetworkedGridEnv"
        self.data_size: int = 0
        self._t: int = 0  # current timestep
        self._total_days: int = 0  # total number of days in the dataset

        self.env_config = env_config
        self.max_episode_steps = env_config.get('max_episode_steps', 24)
        self.centralized = env_config.get('centralized', False)
        self.train = env_config.get('train', True)

        self.message_broker = self._build_message_broker()
        self.agent_dict = self._build_agents()
        self.net = self._build_net()
        self._init_space()

    @property
    def actionable_agents(self):
        """Get agents that have actionable devices."""
        return {
            n: a for n, a in self.agent_dict.items()
            if len(a.get_device_action_spaces()) > 0
        }

    @abstractmethod
    def _build_net(self):
        pass

    @abstractmethod
    def _reward_and_safety(self):
        pass

    @abstractmethod
    def _build_agents(self) -> Dict[AgentID, PowerGridAgent]:
        pass

    def _build_message_broker(self) -> Optional[MessageBroker]:
        broker_type = self.env_config.get('message_broker', None)
        if broker_type == 'in_memory':
            return InMemoryBroker()
        elif broker_type is None:
            return None
        else:
            raise ValueError(f"Unsupported message broker type: {broker_type}")
        

    def _send_actions_to_agent(self, agent_id: AgentID, action: Any):
        if self.message_broker is None:
            raise RuntimeError("Message broker is not initialized.")
        channel = ChannelManager.action_channel(
            self._name,
            agent_id,
            self._env_id
        )
        message = Message(
            env_id=self._env_id,
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

        channel = ChannelManager.state_update_channel(self._env_id)
        messages = self.message_broker.consume(
            channel,
            recipient_id="environment",
            env_id=self._env_id,
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
            except (KeyError, IndexError):
                # Element not found in network, skip
                continue

            # Update pandapower network with device state
            self.net[device_type].loc[element_idx, 'p_mw'] = update.get('P_MW', 0.0)
            if 'Q_MVAr' in update:
                self.net[device_type].loc[element_idx, 'q_mvar'] = update.get('Q_MVAr', 0.0)
            self.net[device_type].loc[element_idx, 'in_service'] = update.get('in_service', True)

    def _update_net(self):
        if self.centralized:
            for agent in self.agent_dict.values():
                agent.update_state(self.net, self._t)
        else:
            # Consume all state updates from devices at once
            state_updates = self._consume_all_state_updates()
            # Apply state updates to pandapower network
            self._apply_state_updates_to_net(state_updates)

    def step(self, action_n: Dict[str, Any]):
        """
        For centralized case: Collective agent state update
        - Actions are computed and set on devices by each agent.
        - Agent state updates are done after actions are set and during _update_net.

        For decentralized case: Per agent state update
        - Actions are sent to agents via message broker.
        - Each agent steps its local devices and updates its local state.
        - Agent state updates are done before _update_net.
        """
        # Set action for each agent
        if self.centralized:
            # note that action can be None here -> decentralized action computation per grid agent
            for name, action in action_n.items():
                if name in self.actionable_agents:
                    # Get observation for this agent
                    obs = self.actionable_agents[name].observe()
                    # Compute and set actions on devices
                    self.actionable_agents[name].act(obs, given_action=action)
        else:
            for agent_id, action in action_n.items():
                if agent_id in self.actionable_agents:
                    self._send_actions_to_agent(agent_id, action)
                    self.actionable_agents[agent_id].step_distributed()

        # Update network states based on agent actions
        self._update_net()

        # Run power flow for the whole network
        try:
            pp.runpp(self.net)
        except:
            self.net['converged'] = False

        # ==== Sync states to all agents ====
        # TODO: Think about whether message passing is needed for decentralized case
        for agent_id, agent in self.agent_dict.items():
            agent.sync_global_state(self.net, self._t)

        # ==== Compute rewards and safety ====
        # Update costs and safety for all agents
        for agent in self.agent_dict.values():
            agent.update_cost_safety(self.net)

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
        """
        Reset environment and all agents.

        Args:
            seed: Random seed
            options: Reset options (unused)

        Returns:
            observations: Dict mapping agent_id → observation
            info: Empty info dict
        """
        # Initialize RNG
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        elif not hasattr(self, 'np_random'):
            self.np_random, _ = seeding.np_random(None)

        # Reset episode step counter
        self._episode_step = 0

        # Reset all agents
        if self.train:
            self._day = self.np_random.integers(self._total_days - 1)
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
        except:
            self.net['converged'] = False

        info = {}

        return self._get_obs(), info

    def _get_obs(self):
        """
        Get observations for all agents.

        Returns:
            Dict mapping agent_id → observation array
        """
        obs_dict = {}
        for agent_id, agent in self.agent_dict.items():
            obs = agent.observe(net=self.net)
            # Extract state from observation
            obs_dict[agent_id] = obs.local['state']

        return obs_dict

    def _init_space(self):
        """Initialize action and observation spaces for all agents."""
        ac_spaces = {}
        ob_spaces = {}

        for name, agent in self.agent_dict.items():
            ac_spaces[name] = agent.get_grid_action_space()
            ob_spaces[name] = agent.get_grid_observation_space(self.net)

        self.action_spaces = ac_spaces
        self.observation_spaces = ob_spaces
