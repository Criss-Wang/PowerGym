"""Tests for ProxyAgent module."""

import pytest
import numpy as np

from heron.agents.proxy_agent import ProxyAgent, PROXY_LEVEL
from heron.core.observation import Observation
from heron.core.state import FieldAgentState
from heron.core.feature import FeatureProvider


class MockFeature(FeatureProvider):
    """Mock feature for testing visibility."""

    visibility = ["public"]

    def __init__(self, value: float = 1.0):
        self.value = value

    def vector(self):
        return np.array([self.value], dtype=np.float32)

    def names(self):
        return ["value"]

    def to_dict(self):
        return {"value": self.value}

    @classmethod
    def from_dict(cls, d):
        return cls(value=d.get("value", 1.0))

    def set_values(self, **kwargs):
        if "value" in kwargs:
            self.value = kwargs["value"]


class OwnerOnlyFeature(FeatureProvider):
    """Feature visible only to owner."""

    visibility = ["owner"]

    def __init__(self, secret: float = 42.0):
        self.secret = secret

    def vector(self):
        return np.array([self.secret], dtype=np.float32)

    def names(self):
        return ["secret"]

    def to_dict(self):
        return {"secret": self.secret}

    @classmethod
    def from_dict(cls, d):
        return cls(secret=d.get("secret", 42.0))

    def set_values(self, **kwargs):
        if "secret" in kwargs:
            self.secret = kwargs["secret"]


class TestProxyAgentInitialization:
    """Test ProxyAgent initialization."""

    def test_basic_initialization(self):
        """Test basic proxy agent initialization."""
        proxy = ProxyAgent()

        assert proxy.agent_id == "proxy_agent"
        assert proxy.level == PROXY_LEVEL
        assert proxy.subordinate_agents == []
        assert proxy.visibility_rules == {}
        assert proxy.state_cache == {}
        assert proxy.state_history == []

    def test_initialization_with_subordinates(self):
        """Test initialization with subordinate agents."""
        proxy = ProxyAgent(
            agent_id="proxy_1",
            subordinate_agents=["agent_1", "agent_2", "agent_3"],
        )

        assert len(proxy.subordinate_agents) == 3
        assert "agent_1" in proxy.subordinate_agents

    def test_initialization_with_visibility_rules(self):
        """Test initialization with visibility rules."""
        visibility_rules = {
            "agent_1": ["power", "voltage"],
            "agent_2": ["current"],
        }
        proxy = ProxyAgent(
            agent_id="proxy_1",
            visibility_rules=visibility_rules,
        )

        assert proxy.visibility_rules == visibility_rules

    def test_initialization_with_history_length(self):
        """Test initialization with custom history length."""
        proxy = ProxyAgent(history_length=50)

        assert proxy.history_length == 50


class TestProxyAgentStateManagement:
    """Test state management functionality."""

    def test_update_state(self):
        """Test updating cached state."""
        proxy = ProxyAgent()
        state = {"agents": {"agent_1": {"power": 100}}}

        proxy.update_state(state)

        assert proxy.state_cache == state
        assert len(proxy.state_history) == 1

    def test_update_state_adds_to_history(self):
        """Test that state updates add to history."""
        proxy = ProxyAgent()

        for i in range(5):
            proxy._timestep = float(i)
            proxy.update_state({"value": i})

        assert len(proxy.state_history) == 5

    def test_update_state_trims_history(self):
        """Test that history is trimmed to max length."""
        proxy = ProxyAgent(history_length=3)

        for i in range(10):
            proxy._timestep = float(i)
            proxy.update_state({"value": i})

        assert len(proxy.state_history) == 3
        assert proxy.state_history[-1]["state"]["value"] == 9


class TestProxyAgentStateAtTime:
    """Test historical state retrieval."""

    def test_get_state_at_time_empty_history(self):
        """Test getting state with empty history."""
        proxy = ProxyAgent()
        proxy.state_cache = {"default": True}

        state = proxy.get_state_at_time(5.0)

        assert state == {"default": True}

    def test_get_state_at_time_exact_match(self):
        """Test getting state at exact timestamp."""
        proxy = ProxyAgent()
        proxy.state_history = [
            {"timestamp": 1.0, "state": {"v": 1}},
            {"timestamp": 2.0, "state": {"v": 2}},
            {"timestamp": 3.0, "state": {"v": 3}},
        ]

        state = proxy.get_state_at_time(2.0)

        assert state == {"v": 2}

    def test_get_state_at_time_before_target(self):
        """Test getting state for time between entries."""
        proxy = ProxyAgent()
        proxy.state_history = [
            {"timestamp": 1.0, "state": {"v": 1}},
            {"timestamp": 3.0, "state": {"v": 3}},
            {"timestamp": 5.0, "state": {"v": 5}},
        ]

        state = proxy.get_state_at_time(4.0)

        assert state == {"v": 3}

    def test_get_state_at_time_before_all_history(self):
        """Test getting state for time before all history."""
        proxy = ProxyAgent()
        proxy.state_history = [
            {"timestamp": 5.0, "state": {"v": 5}},
            {"timestamp": 10.0, "state": {"v": 10}},
        ]

        state = proxy.get_state_at_time(1.0)

        assert state == {"v": 5}


class TestProxyAgentStateFiltering:
    """Test state filtering functionality."""

    def test_filter_state_no_rules(self):
        """Test filtering with no visibility rules."""
        proxy = ProxyAgent()
        agent_state = {"power": 100, "voltage": 1.0}

        filtered = proxy._filter_state_for_agent("agent_1", agent_state)

        assert filtered == agent_state

    def test_filter_state_with_rules(self):
        """Test filtering with visibility rules."""
        proxy = ProxyAgent(
            visibility_rules={"agent_1": ["power"]}
        )
        agent_state = {"power": 100, "voltage": 1.0, "current": 50}

        filtered = proxy._filter_state_for_agent("agent_1", agent_state)

        assert filtered == {"power": 100}
        assert "voltage" not in filtered

    def test_filter_state_empty_rules(self):
        """Test filtering with empty rules list."""
        proxy = ProxyAgent(
            visibility_rules={"agent_1": []}
        )
        agent_state = {"power": 100}

        filtered = proxy._filter_state_for_agent("agent_1", agent_state)

        assert filtered == {}


class TestProxyAgentGetLatestState:
    """Test getting latest state for agent."""

    def test_get_latest_state_basic(self):
        """Test getting latest state."""
        proxy = ProxyAgent()
        proxy.state_cache = {
            "agents": {
                "agent_1": {"power": 100, "voltage": 1.0}
            }
        }

        state = proxy.get_latest_state_for_agent("agent_1")

        assert state == {"power": 100, "voltage": 1.0}

    def test_get_latest_state_missing_agent(self):
        """Test getting state for missing agent."""
        proxy = ProxyAgent()
        proxy.state_cache = {"agents": {}}

        state = proxy.get_latest_state_for_agent("unknown_agent")

        assert state == {}


class TestProxyAgentGetStateForAgent:
    """Test get_state_for_agent method."""

    def test_get_state_for_agent_basic(self):
        """Test basic state retrieval."""
        proxy = ProxyAgent()
        proxy.state_cache = {
            "agents": {
                "agent_1": {"power": 100}
            }
        }

        state = proxy.get_state_for_agent(
            agent_id="agent_1",
            requestor_level=1,
        )

        assert state == {"power": 100}

    def test_get_state_for_agent_with_time(self):
        """Test state retrieval at specific time."""
        proxy = ProxyAgent()
        proxy.state_history = [
            {"timestamp": 1.0, "state": {"agents": {"agent_1": {"v": 1}}}},
            {"timestamp": 2.0, "state": {"agents": {"agent_1": {"v": 2}}}},
        ]

        state = proxy.get_state_for_agent(
            agent_id="agent_1",
            requestor_level=1,
            at_time=1.5,
        )

        assert state == {"v": 1}


class TestProxyAgentRegistration:
    """Test agent registration functionality."""

    def test_register_subordinate(self):
        """Test registering a subordinate."""
        proxy = ProxyAgent()

        proxy.register_subordinate("new_agent")

        assert "new_agent" in proxy.subordinate_agents

    def test_register_subordinate_duplicate(self):
        """Test registering duplicate subordinate."""
        proxy = ProxyAgent(subordinate_agents=["agent_1"])

        proxy.register_subordinate("agent_1")

        assert proxy.subordinate_agents.count("agent_1") == 1

    def test_set_visibility_rules(self):
        """Test setting visibility rules."""
        proxy = ProxyAgent()

        proxy.set_visibility_rules("agent_1", ["power", "voltage"])

        assert proxy.visibility_rules["agent_1"] == ["power", "voltage"]


class TestProxyAgentAbstractMethods:
    """Test required abstract method implementations."""

    def test_observe_returns_empty(self):
        """Test observe returns empty observation."""
        proxy = ProxyAgent()
        proxy._timestep = 5.0

        obs = proxy.observe()

        assert isinstance(obs, Observation)
        assert obs.timestamp == 5.0

    def test_act_does_nothing(self):
        """Test act does nothing."""
        proxy = ProxyAgent()
        obs = Observation()

        # Should not raise
        proxy.act(obs)

    def test_reset_clears_state(self):
        """Test reset clears state."""
        proxy = ProxyAgent()
        proxy.state_cache = {"key": "value"}
        proxy.state_history = [{"timestamp": 1.0, "state": {}}]

        proxy.reset()

        assert proxy.state_cache == {}
        assert proxy.state_history == []


class TestProxyAgentRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__."""
        proxy = ProxyAgent(
            agent_id="proxy_1",
            subordinate_agents=["a1", "a2"],
        )

        repr_str = repr(proxy)

        assert "ProxyAgent" in repr_str
        assert "proxy_1" in repr_str
        assert "subordinates=2" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
