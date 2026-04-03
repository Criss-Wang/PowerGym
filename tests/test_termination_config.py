"""Unit tests for termination/truncation configuration types.

Tests the following components:
  - ``TerminationConfig`` — dataclass with validation and factory classmethods
  - ``AllSemantics`` — enum for ``__all__`` combination logic
  - ``EnvContext`` — frozen dataclass providing read-only env snapshots
  - ``compute_all_done`` — shared utility for combining per-agent flags
"""

import pytest
from dataclasses import FrozenInstanceError

from heron.envs.termination import TerminationConfig, AllSemantics, compute_all_done
from heron.core.env_context import EnvContext


# =============================================================================
# TerminationConfig
# =============================================================================

class TestTerminationConfig:
    """Tests for TerminationConfig defaults, validation, and factories."""

    def test_default_config(self):
        """TerminationConfig.default() has no limits and ALL semantics."""
        cfg = TerminationConfig.default()
        assert cfg.max_steps is None
        assert cfg.max_sim_time is None
        assert cfg.all_semantics == AllSemantics.ALL

    def test_max_steps_validation_zero(self):
        """max_steps=0 should raise ValueError (must be >= 1)."""
        with pytest.raises(ValueError, match="max_steps must be >= 1"):
            TerminationConfig(max_steps=0)

    def test_max_steps_validation_negative(self):
        """max_steps=-5 should raise ValueError."""
        with pytest.raises(ValueError, match="max_steps must be >= 1"):
            TerminationConfig(max_steps=-5)

    def test_max_sim_time_validation_negative(self):
        """max_sim_time=-1 should raise ValueError (must be > 0)."""
        with pytest.raises(ValueError, match="max_sim_time must be > 0"):
            TerminationConfig(max_sim_time=-1.0)

    def test_max_sim_time_validation_zero(self):
        """max_sim_time=0 should raise ValueError (must be > 0)."""
        with pytest.raises(ValueError, match="max_sim_time must be > 0"):
            TerminationConfig(max_sim_time=0.0)

    def test_with_max_steps_factory(self):
        """Factory classmethod sets max_steps and all_semantics correctly."""
        cfg = TerminationConfig.with_max_steps(50, all_semantics="any")
        assert cfg.max_steps == 50
        assert cfg.max_sim_time is None
        assert cfg.all_semantics == "any"

    def test_with_max_steps_factory_default_semantics(self):
        """Factory classmethod defaults to ALL semantics."""
        cfg = TerminationConfig.with_max_steps(10)
        assert cfg.max_steps == 10
        assert cfg.all_semantics == AllSemantics.ALL

    def test_with_max_sim_time_factory(self):
        """Factory classmethod sets max_sim_time and all_semantics correctly."""
        cfg = TerminationConfig.with_max_sim_time(100.0, all_semantics="any")
        assert cfg.max_sim_time == 100.0
        assert cfg.max_steps is None
        assert cfg.all_semantics == "any"

    def test_with_max_sim_time_factory_default_semantics(self):
        """Factory classmethod defaults to ALL semantics."""
        cfg = TerminationConfig.with_max_sim_time(50.0)
        assert cfg.max_sim_time == 50.0
        assert cfg.all_semantics == AllSemantics.ALL


# =============================================================================
# AllSemantics enum
# =============================================================================

class TestAllSemantics:
    """Tests for the AllSemantics enum values."""

    def test_all_semantics_enum_values(self):
        """ALL.value == 'all' and ANY.value == 'any'."""
        assert AllSemantics.ALL.value == "all"
        assert AllSemantics.ANY.value == "any"

    def test_all_semantics_from_string(self):
        """Enum can be constructed from string values."""
        assert AllSemantics("all") == AllSemantics.ALL
        assert AllSemantics("any") == AllSemantics.ANY


# =============================================================================
# EnvContext (frozen dataclass)
# =============================================================================

class TestEnvContext:
    """Tests for the EnvContext frozen dataclass."""

    def test_env_context_frozen(self):
        """Assigning to a field on a frozen EnvContext raises FrozenInstanceError."""
        ctx = EnvContext(step_count=5, sim_time=10.0)
        with pytest.raises(FrozenInstanceError):
            ctx.step_count = 99

    def test_env_context_defaults(self):
        """Default EnvContext has zeroed fields and None limits."""
        ctx = EnvContext()
        assert ctx.step_count == 0
        assert ctx.sim_time == 0.0
        assert ctx.max_steps is None
        assert ctx.max_sim_time is None
        assert ctx.all_semantics == "all"

    def test_env_context_custom_values(self):
        """EnvContext stores all provided values."""
        ctx = EnvContext(
            step_count=10,
            sim_time=42.5,
            max_steps=100,
            max_sim_time=500.0,
            all_semantics="any",
        )
        assert ctx.step_count == 10
        assert ctx.sim_time == 42.5
        assert ctx.max_steps == 100
        assert ctx.max_sim_time == 500.0
        assert ctx.all_semantics == "any"


# =============================================================================
# compute_all_done
# =============================================================================

class TestComputeAllDone:
    """Tests for the compute_all_done shared utility."""

    def test_all_semantics_all_true(self):
        """ALL semantics: all agents True -> True."""
        flags = {"a": True, "b": True, "c": True}
        expected = {"a", "b", "c"}
        assert compute_all_done(flags, expected, "all") is True

    def test_all_semantics_one_false(self):
        """ALL semantics: one agent False -> False."""
        flags = {"a": True, "b": False, "c": True}
        expected = {"a", "b", "c"}
        assert compute_all_done(flags, expected, "all") is False

    def test_any_semantics_one_true(self):
        """ANY semantics: one agent True -> True."""
        flags = {"a": False, "b": True, "c": False}
        expected = {"a", "b", "c"}
        assert compute_all_done(flags, expected, "any") is True

    def test_any_semantics_all_false(self):
        """ANY semantics: all agents False -> False."""
        flags = {"a": False, "b": False, "c": False}
        expected = {"a", "b", "c"}
        assert compute_all_done(flags, expected, "any") is False

    def test_empty_agents(self):
        """Empty expected_agents set -> False (guards all([]) == True)."""
        assert compute_all_done({}, set(), "all") is False
        assert compute_all_done({}, set(), "any") is False

    def test_missing_agents_default_false(self):
        """Agents in expected_agents but not in flags default to False."""
        flags = {"a": True}
        expected = {"a", "b", "c"}
        # b and c default to False, so ALL semantics -> False
        assert compute_all_done(flags, expected, "all") is False
        # but ANY sees a=True -> True
        assert compute_all_done(flags, expected, "any") is True

    def test_extra_flags_ignored(self):
        """Flags for agents not in expected_agents are ignored."""
        flags = {"a": True, "b": True, "extra": False}
        expected = {"a", "b"}
        assert compute_all_done(flags, expected, "all") is True
