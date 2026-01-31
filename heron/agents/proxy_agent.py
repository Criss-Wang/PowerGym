"""Proxy Agent for managing state distribution in MARL.

The ProxyAgent acts as an intermediary between the environment and other agents,
managing state updates and controlling information visibility.

Key responsibilities:
1. Cache state from the environment
2. Apply visibility rules to filter state for each agent
3. Provide state on-demand to requesting agents

Usage Pattern (Option A - Synchronous):
    # Setup
    proxy = ProxyAgent(
        env_id="env_1",
        subordinate_agents=["battery_1", "solar_1"],
        visibility_rules={
            "battery_1": ["SoC", "Power"],
            "solar_1": ["Irradiance"],
        }
    )

    # In env.step():
    proxy.update_state(env_state)  # Cache latest state

    # Agents request state through proxy
    state = proxy.get_state_for_agent("battery_1", requestor_level=1)

Usage Pattern (Option B - Event-Driven):
    # Agents call get_state_for_agent() in their tick() method
    # ProxyAgent applies obs_delay by returning historical state
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from heron.agents.base import Agent
from heron.core.observation import Observation
from heron.utils.typing import AgentID

if TYPE_CHECKING:
    from heron.core.state import State


PROXY_LEVEL = 3  # Level identifier for proxy-level agents


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
        subordinate_agents: List of agent IDs that can request state from this proxy
    """

    def __init__(
        self,
        agent_id: AgentID = "proxy_agent",
        env_id: Optional[str] = None,
        subordinate_agents: Optional[List[AgentID]] = None,
        visibility_rules: Optional[Dict[AgentID, List[str]]] = None,
        history_length: int = 100,  # Number of timesteps to keep in history
    ):
        """Initialize proxy agent.

        Args:
            agent_id: Unique identifier for this proxy agent
            env_id: Environment ID for multi-environment isolation
            subordinate_agents: List of agent IDs that can request state
            visibility_rules: Dict mapping agent IDs to allowed state keys.
                            If None, all agents see all state by default.
            history_length: Number of timesteps of history to maintain
        """
        super().__init__(
            agent_id=agent_id,
            level=PROXY_LEVEL,
            upstream_id=None,  # Proxy has no upstream
            env_id=env_id,
            subordinates={},  # Proxy doesn't manage subordinates hierarchically
        )

        self.subordinate_agents = subordinate_agents or []
        self.visibility_rules = visibility_rules or {}
        self.state_cache: Dict[str, Any] = {}
        self.state_history: List[Dict[str, Any]] = []
        self.history_length = history_length

    # ============================================
    # State Management (Both Modes)
    # ============================================

    def update_state(self, state: Dict[str, Any]) -> None:
        """Update cached state from environment. [Both Modes]

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

        # Trim history if needed
        if len(self.state_history) > self.history_length:
            self.state_history = self.state_history[-self.history_length:]

    # ============================================
    # Event-Driven Execution (Option B - Testing)
    # ============================================

    def get_state_at_time(self, target_time: float) -> Optional[Dict[str, Any]]:
        """Get state from a specific time for delayed observations. [Testing Only]

        Used in Option B to simulate observation delays.

        Args:
            target_time: Timestamp to retrieve state for

        Returns:
            State at or before target_time, or None if not available
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
    # State Filtering (Both Modes)
    # ============================================

    def _filter_state_for_agent(
        self, agent_id: AgentID, agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Filter state based on visibility rules for a specific agent. [Both Modes]

        Args:
            agent_id: Agent requesting the state
            agent_state: Agent-specific state extracted from aggregated state

        Returns:
            Filtered state dict
        """
        if agent_id not in self.visibility_rules:
            # No specific rules, return full agent state
            return agent_state.copy() if agent_state else {}

        allowed_keys = self.visibility_rules[agent_id]
        filtered_state = {}

        for key in allowed_keys:
            if key in agent_state:
                filtered_state[key] = agent_state[key]

        return filtered_state

    def get_latest_state_for_agent(self, agent_id: AgentID) -> Dict[str, Any]:
        """Get the latest cached state for a specific agent. [Both Modes]

        Args:
            agent_id: Agent requesting the state

        Returns:
            Filtered state dict
        """
        agents_state = self.state_cache.get('agents', {})
        agent_state = agents_state.get(agent_id, {})
        return self._filter_state_for_agent(agent_id, agent_state)

    def get_state_for_agent(
        self,
        agent_id: AgentID,
        requestor_level: int,
        owner_id: Optional[AgentID] = None,
        owner_level: Optional[int] = None,
        state: Optional["State"] = None,
        at_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get filtered state respecting FeatureProvider visibility rules. [Both Modes]

        This is the recommended method for agents to access state. It integrates
        with FeatureProvider.is_observable_by() for fine-grained visibility control.

        Args:
            agent_id: ID of the agent requesting state
            requestor_level: Hierarchy level of requesting agent (1=field, 2=coord, 3=system)
            owner_id: ID of the agent whose state is being requested (defaults to agent_id)
            owner_level: Hierarchy level of owner (defaults to requestor_level)
            state: Optional State object with FeatureProviders for visibility checking
            at_time: Optional timestamp for delayed observations (Option B - Testing)

        Returns:
            Filtered state dict based on visibility rules
        """
        if owner_id is None:
            owner_id = agent_id
        if owner_level is None:
            owner_level = requestor_level

        # Get base state (from history if time specified, otherwise current)
        if at_time is not None:
            base_state = self.get_state_at_time(at_time) or {}
        else:
            base_state = self.state_cache

        agents_state = base_state.get('agents', {})
        agent_state = agents_state.get(owner_id, {})

        # Apply key-based visibility rules first
        filtered = self._filter_state_for_agent(agent_id, agent_state)

        # If State object provided, apply FeatureProvider visibility rules
        if state is not None:
            feature_filtered = {}
            for feature in state.features:
                if feature.is_observable_by(
                    agent_id, requestor_level, owner_id, owner_level
                ):
                    feature_name = feature.feature_name
                    if feature_name in filtered:
                        feature_filtered[feature_name] = filtered[feature_name]
            return feature_filtered

        return filtered

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
        observable = []
        for feature in state.features:
            if feature.is_observable_by(
                requestor_id, requestor_level, owner_id, owner_level
            ):
                observable.append(feature.feature_name)
        return observable

    # ============================================
    # Agent Registration (Both Modes)
    # ============================================

    def register_subordinate(self, agent_id: AgentID) -> None:
        """Register a new subordinate agent. [Both Modes]

        Args:
            agent_id: Agent ID to register
        """
        if agent_id not in self.subordinate_agents:
            self.subordinate_agents.append(agent_id)

    def set_visibility_rules(
        self,
        agent_id: AgentID,
        allowed_keys: List[str],
    ) -> None:
        """Set visibility rules for an agent. [Both Modes]

        Args:
            agent_id: Agent to set rules for
            allowed_keys: List of state keys the agent can access
        """
        self.visibility_rules[agent_id] = allowed_keys

    # ============================================
    # Required Abstract Methods (Both Modes)
    # ============================================

    def observe(
        self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs
    ) -> Observation:
        """ProxyAgent doesn't observe in the traditional sense. [Both Modes]

        It receives state updates via update_state() instead.

        Returns:
            Empty observation
        """
        return Observation(timestamp=self._timestep)

    def act(self, observation: Observation, *args, **kwargs) -> None:
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

    def __repr__(self) -> str:
        num_subordinates = len(self.subordinate_agents)
        return f"ProxyAgent(id={self.agent_id}, subordinates={num_subordinates})"
