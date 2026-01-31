"""Tests for heron.utils.typing module."""

import pytest

from heron.utils.typing import AgentID, float_if_not_none


class TestAgentID:
    """Test AgentID type alias."""

    def test_agent_id_is_string(self):
        """Test that AgentID is a string type alias."""
        agent_id: AgentID = "agent_1"

        assert isinstance(agent_id, str)

    def test_agent_id_accepts_any_string(self):
        """Test AgentID accepts any string value."""
        agent_ids: list[AgentID] = ["a", "agent_123", "coordinator_main", ""]

        for aid in agent_ids:
            assert isinstance(aid, str)


class TestFloatIfNotNone:
    """Test float_if_not_none function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        result = float_if_not_none(None)

        assert result is None

    def test_int_converts_to_float(self):
        """Test that int is converted to float."""
        result = float_if_not_none(5)

        assert isinstance(result, float)
        assert result == 5.0

    def test_float_stays_float(self):
        """Test that float stays float."""
        result = float_if_not_none(3.14)

        assert isinstance(result, float)
        assert result == 3.14

    def test_string_numeric(self):
        """Test that numeric string converts to float."""
        result = float_if_not_none("2.5")

        assert isinstance(result, float)
        assert result == 2.5

    def test_negative_value(self):
        """Test negative value conversion."""
        result = float_if_not_none(-10)

        assert isinstance(result, float)
        assert result == -10.0

    def test_zero_converts(self):
        """Test zero converts correctly."""
        result = float_if_not_none(0)

        assert isinstance(result, float)
        assert result == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
