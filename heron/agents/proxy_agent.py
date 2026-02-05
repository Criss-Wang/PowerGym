"""Proxy Agent for managing state distribution in MARL.

The ProxyAgent acts as an intermediary between the environment and other agents,
managing state updates and controlling information visibility.

Key responsibilities:
1. Cache state from the environment
2. Apply visibility rules to filter state for each agent
3. Provide state on-demand to requesting agents (with optional time delay)

Usage Pattern (Option A - Synchronous):
    proxy = ProxyAgent(
        env_id="env_1",
        registered_agents=["sensor_1", "controller_1"],
        visibility_rules={
            "sensor_1": ["reading", "status"],
            "controller_1": ["measurement"],
        }
    )

    # In env.step():
    proxy.update_state(env_state)  # Cache latest state

    # Agents request state through proxy
    state = proxy.get_state_for_agent("sensor_1", requestor_level=1)

Usage Pattern (Option B - Event-Driven):
    # Agents call get_state_for_agent() in their tick() method
    # ProxyAgent applies obs_delay by returning historical state via at_time param
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from heron.agents.base import Agent
from heron.core.observation import Observation
from heron.utils.typing import AgentID

if TYPE_CHECKING:
    from heron.core.state import State


PROXY_LEVEL = 0  # Proxy is not part of the agent hierarchy (L1-L3)


class ProxyAgent(Agent):
    """Proxy agent that manages state distribution.

    The ProxyAgent sits between the environment and other agents, acting as the
    single source of truth for state information. All agents should retrieve
    state through the ProxyAgent rather than directly accessing the environment.

    This enables:
    - Visibility filtering based on agent level and configuration
    - Historical state access for observation delays (Option B)
    - Centralized state management

    Attributes:
        state_cache: Latest state received from environment
        state_history: Historical states for delayed observations (Option B)
        visibility_rules: Dict mapping agent IDs to allowed state keys
        registered_agents: List of agent IDs that can request state from this proxy
    """

    def __init__(
        self,
        agent_id: AgentID = "proxy_agent",
        env_id: Optional[str] = None,
        registered_agents: Optional[List[AgentID]] = None,
        visibility_rules: Optional[Dict[AgentID, List[str]]] = None,
        history_length: int = 100,
    ):
        """Initialize proxy agent.

        Args:
            agent_id: Unique identifier for this proxy agent
            env_id: Environment ID for multi-environment isolation
            registered_agents: List of agent IDs that can request state
            visibility_rules: Dict mapping agent IDs to allowed state keys.
                If None, all agents see all state by default.
            history_length: Number of timesteps of history to maintain
        """
        super().__init__(
            agent_id=agent_id,
            level=PROXY_LEVEL,
            upstream_id=None,
            env_id=env_id,
            subordinates={},
        )

        self.registered_agents: List[AgentID] = registered_agents or []
        self.visibility_rules: Dict[AgentID, List[str]] = visibility_rules or {}
        self.state_cache: Dict[str, Any] = {}
        self.state_history: List[Dict[str, Any]] = []
        self.history_length = history_length

    # ============================================
    # State Management
    # ============================================

    def update_state(self, state: Dict[str, Any]) -> None:
        """Update cached state from environment.

        Should be called by the environment after each step.

        Args:
            state: Current environment state
        """
        self.state_cache = state.copy()

        # Add to history for delayed observations
        self.state_history.append({
            'timestamp': self._timestep,
            'state': state.copy()
        })

        # Trim history if exceeds limit
        if len(self.state_history) > self.history_length:
            self.state_history = self.state_history[-self.history_length:]

    def get_state_at_time(self, target_time: float) -> Dict[str, Any]:
        """Get state from a specific time for delayed observations.

        Used in Option B to simulate observation delays.

        Args:
            target_time: Timestamp to retrieve state for

        Returns:
            State at or before target_time, or current cache if not available
        """
        if not self.state_history:
            return self.state_cache

        # Find the most recent state at or before target_time
        for entry in reversed(self.state_history):
            if entry['timestamp'] <= target_time:
                return entry['state']

        # If target_time is before all history, return oldest
        return self.state_history[0]['state'] if self.state_history else self.state_cache

    # ============================================
    # State Filtering
    # ============================================

    def _filter_state_by_keys(
        self,
        agent_id: AgentID,
        agent_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Filter state based on visibility rules for a specific agent.

        Args:
            agent_id: Agent requesting the state
            agent_state: Agent-specific state to filter

        Returns:
            Filtered state dict containing only allowed keys
        """
        if agent_id not in self.visibility_rules:
            # No specific rules, return full agent state
            return agent_state.copy() if agent_state else {}

        allowed_keys = self.visibility_rules[agent_id]
        return {key: agent_state[key] for key in allowed_keys if key in agent_state}

    def get_state_for_agent(
        self,
        agent_id: AgentID,
        requestor_level: int,
        owner_id: Optional[AgentID] = None,
        owner_level: Optional[int] = None,
        state: Optional["State"] = None,
        at_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get filtered state respecting visibility rules.

        This is the primary method for agents to access state. It integrates
        with FeatureProvider.is_observable_by() for fine-grained visibility control.

        Args:
            agent_id: ID of the agent requesting state
            requestor_level: Hierarchy level of requesting agent (1=field, 2=coord, 3=system)
            owner_id: ID of the agent whose state is being requested (defaults to agent_id)
            owner_level: Hierarchy level of owner (defaults to requestor_level)
            state: Optional State object with FeatureProviders for visibility checking
            at_time: Optional timestamp for delayed observations (Option B)

        Returns:
            Filtered state dict based on visibility rules
        """
        owner_id = owner_id or agent_id
        owner_level = owner_level or requestor_level

        # Get base state (from history if time specified, otherwise current)
        base_state = self.get_state_at_time(at_time) if at_time is not None else self.state_cache

        agents_state = base_state.get('agents', {})
        agent_state = agents_state.get(owner_id, {})

        # Apply key-based visibility rules
        filtered = self._filter_state_by_keys(agent_id, agent_state)

        # If State object provided, apply FeatureProvider visibility rules
        if state is not None:
            return {
                feature.feature_name: filtered[feature.feature_name]
                for feature in state.features
                if feature.is_observable_by(agent_id, requestor_level, owner_id, owner_level)
                and feature.feature_name in filtered
            }

        return filtered

    def get_latest_state_for_agent(self, agent_id: AgentID) -> Dict[str, Any]:
        """Get the latest cached state for a specific agent.

        Convenience method that calls get_state_for_agent with minimal params.

        Args:
            agent_id: Agent requesting the state

        Returns:
            Filtered state dict
        """
        agents_state = self.state_cache.get('agents', {})
        agent_state = agents_state.get(agent_id, {})
        return self._filter_state_by_keys(agent_id, agent_state)

    def get_observable_features(
        self,
        requestor_id: AgentID,
        requestor_level: int,
        owner_id: AgentID,
        owner_level: int,
        state: "State",
    ) -> List[str]:
        """Get list of feature names observable by the requesting agent.

        Args:
            requestor_id: ID of agent requesting observation
            requestor_level: Hierarchy level of requestor
            owner_id: ID of agent that owns the state
            owner_level: Hierarchy level of owner
            state: State object containing FeatureProviders

        Returns:
            List of feature names the requestor can observe
        """
        return [
            feature.feature_name
            for feature in state.features
            if feature.is_observable_by(requestor_id, requestor_level, owner_id, owner_level)
        ]

    # ============================================
    # Agent Registration
    # ============================================

    def register_agent(self, agent_id: AgentID) -> None:
        """Register a new agent that can request state.

        Args:
            agent_id: Agent ID to register
        """
        if agent_id not in self.registered_agents:
            self.registered_agents.append(agent_id)

    def set_visibility_rules(
        self,
        agent_id: AgentID,
        allowed_keys: List[str],
    ) -> None:
        """Set visibility rules for an agent.

        Args:
            agent_id: Agent to set rules for
            allowed_keys: List of state keys the agent can access
        """
        self.visibility_rules[agent_id] = allowed_keys

    # ============================================
    # Required Agent Interface Methods
    # ============================================

    def observe(
        self,
        global_state: Optional[Dict[str, Any]] = None,
        proxy: Optional[Agent] = None,
        **kwargs,
    ) -> Observation:
        """ProxyAgent doesn't observe in the traditional sense.

        It receives state updates via update_state() instead.

        Returns:
            Empty observation with current timestamp
        """
        return Observation(timestamp=self._timestep)

    def act(self, observation: Observation, upstream_action: Any = None) -> None:
        """ProxyAgent doesn't take actions.

        It manages state distribution instead.
        """
        pass

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset proxy agent state.

        Args:
            seed: Random seed
            **kwargs: Additional reset parameters
        """
        super().reset(seed=seed)
        self.state_cache = {}
        self.state_history = []

    # ============================================
    # Utility Methods
    # ============================================

    def __repr__(self) -> str:
        num_registered = len(self.registered_agents)
        return f"ProxyAgent(id={self.agent_id}, registered_agents={num_registered})"
