"""Tests for TickConfig class."""

import numpy as np
import pytest

from heron.scheduling.tick_config import JitterType, TickConfig


class TestTickConfigBasic:
    """Test basic TickConfig functionality."""

    def test_default_initialization(self):
        """Test default config values."""
        config = TickConfig()

        assert config.tick_interval == 1.0
        assert config.obs_delay == 0.0
        assert config.act_delay == 0.0
        assert config.msg_delay == 0.0
        assert config.jitter_type == JitterType.NONE
        assert config.jitter_ratio == 0.1
        assert config.min_delay == 0.0
        assert config.rng is None

    def test_deterministic_factory(self):
        """Test deterministic factory method."""
        config = TickConfig.deterministic(
            tick_interval=2.0,
            obs_delay=0.5,
            act_delay=0.3,
            msg_delay=0.1,
        )

        assert config.tick_interval == 2.0
        assert config.obs_delay == 0.5
        assert config.act_delay == 0.3
        assert config.msg_delay == 0.1
        assert config.jitter_type == JitterType.NONE

    def test_with_jitter_factory(self):
        """Test with_jitter factory method."""
        config = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.5,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.2,
            seed=42,
        )

        assert config.tick_interval == 1.0
        assert config.obs_delay == 0.5
        assert config.jitter_type == JitterType.GAUSSIAN
        assert config.jitter_ratio == 0.2
        assert config.rng is not None

    def test_deterministic_no_jitter(self):
        """Test deterministic config returns exact values."""
        config = TickConfig.deterministic(
            tick_interval=1.0,
            obs_delay=0.5,
            act_delay=0.2,
            msg_delay=0.1,
        )

        # Should return exact values (no jitter)
        for _ in range(10):
            assert config.get_tick_interval() == 1.0
            assert config.get_obs_delay() == 0.5
            assert config.get_act_delay() == 0.2
            assert config.get_msg_delay() == 0.1

    def test_zero_delay_no_jitter(self):
        """Test that zero delays remain zero even with jitter enabled."""
        config = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.0,  # Zero
            act_delay=0.0,  # Zero
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.5,
            seed=42,
        )

        # Zero delays should stay zero
        for _ in range(10):
            assert config.get_obs_delay() == 0.0
            assert config.get_act_delay() == 0.0


class TestTickConfigJitter:
    """Test jitter functionality."""

    def test_uniform_jitter_bounds(self):
        """Test uniform jitter stays within bounds."""
        config = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=1.0,
            jitter_type=JitterType.UNIFORM,
            jitter_ratio=0.1,  # +/- 10%
            seed=42,
        )

        for _ in range(100):
            delay = config.get_obs_delay()
            assert 0.9 <= delay <= 1.1  # Within +/- 10%

    def test_uniform_tick_interval_bounds(self):
        """Test uniform jitter on tick_interval stays within bounds."""
        config = TickConfig.with_jitter(
            tick_interval=1.0,
            jitter_type=JitterType.UNIFORM,
            jitter_ratio=0.1,
            seed=42,
        )

        for _ in range(100):
            interval = config.get_tick_interval()
            assert 0.9 <= interval <= 1.1

    def test_gaussian_jitter_distribution(self):
        """Test gaussian jitter approximately follows distribution."""
        config = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=1.0,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,
            seed=42,
        )

        samples = [config.get_obs_delay() for _ in range(1000)]
        mean = np.mean(samples)
        std = np.std(samples)

        # Mean should be close to base value
        assert abs(mean - 1.0) < 0.05
        # Std should be close to jitter_ratio * base
        assert abs(std - 0.1) < 0.02

    def test_negative_clamped_to_min_delay(self):
        """Test negative results are clamped to min_delay."""
        config = TickConfig.with_jitter(
            tick_interval=0.1,
            obs_delay=0.1,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.9,  # 90% - can go negative
            seed=42,
            min_delay=0.01,
        )

        for _ in range(100):
            assert config.get_obs_delay() >= 0.01

    def test_tick_interval_minimum_floor(self):
        """Test tick_interval has minimum floor of 0.001."""
        config = TickConfig.with_jitter(
            tick_interval=0.01,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.99,  # Can go very negative
            seed=42,
        )

        for _ in range(100):
            assert config.get_tick_interval() >= 0.001

    def test_jitter_varies_values(self):
        """Test that jittered values actually vary."""
        config = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.5,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,
            seed=42,
        )

        intervals = [config.get_tick_interval() for _ in range(10)]
        delays = [config.get_obs_delay() for _ in range(10)]

        # Should have some variation (not all identical)
        assert len(set(intervals)) > 1
        assert len(set(delays)) > 1


class TestTickConfigReproducibility:
    """Test reproducibility with RNG."""

    def test_same_seed_same_results(self):
        """Test same seed produces same jitter sequence."""
        config1 = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.5,
            jitter_type=JitterType.GAUSSIAN,
            seed=42,
        )
        config2 = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.5,
            jitter_type=JitterType.GAUSSIAN,
            seed=42,
        )

        for _ in range(10):
            assert config1.get_obs_delay() == config2.get_obs_delay()

    def test_different_seeds_different_results(self):
        """Test different seeds produce different sequences."""
        config1 = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.5,
            jitter_type=JitterType.GAUSSIAN,
            seed=42,
        )
        config2 = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.5,
            jitter_type=JitterType.GAUSSIAN,
            seed=123,
        )

        # Very unlikely to match
        results1 = [config1.get_obs_delay() for _ in range(10)]
        results2 = [config2.get_obs_delay() for _ in range(10)]
        assert results1 != results2

    def test_seed_method(self):
        """Test seed() method resets RNG."""
        config = TickConfig.with_jitter(
            tick_interval=1.0,
            obs_delay=0.5,
            jitter_type=JitterType.GAUSSIAN,
            seed=42,
        )

        # Get some values
        values1 = [config.get_obs_delay() for _ in range(5)]

        # Re-seed and get values again
        config.seed(42)
        values2 = [config.get_obs_delay() for _ in range(5)]

        assert values1 == values2


class TestTickConfigValidation:
    """Test validation logic."""

    def test_invalid_tick_interval_zero(self):
        """Test that zero tick_interval raises error."""
        with pytest.raises(ValueError, match="tick_interval must be positive"):
            TickConfig(tick_interval=0.0)

    def test_invalid_tick_interval_negative(self):
        """Test that negative tick_interval raises error."""
        with pytest.raises(ValueError, match="tick_interval must be positive"):
            TickConfig(tick_interval=-1.0)

    def test_invalid_jitter_ratio_negative(self):
        """Test that negative jitter_ratio raises error."""
        with pytest.raises(ValueError, match="jitter_ratio must be non-negative"):
            TickConfig(jitter_ratio=-0.1)

    def test_invalid_min_delay_negative(self):
        """Test that negative min_delay raises error."""
        with pytest.raises(ValueError, match="min_delay must be non-negative"):
            TickConfig(min_delay=-0.1)

    def test_zero_jitter_ratio_valid(self):
        """Test that zero jitter_ratio is valid."""
        config = TickConfig(jitter_ratio=0.0)
        assert config.jitter_ratio == 0.0


class TestTickConfigIntegrationWithScheduler:
    """Test TickConfig integration with EventScheduler."""

    def test_register_agent_with_tick_config(self):
        """Test registering agent with TickConfig."""
        from heron.scheduling.scheduler import EventScheduler

        scheduler = EventScheduler()
        config = TickConfig.with_jitter(
            tick_interval=2.0,
            act_delay=0.5,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,
            seed=42,
        )

        scheduler.register_agent("agent_1", tick_config=config)

        assert scheduler.agent_tick_configs["agent_1"] is config
        assert scheduler.agent_intervals["agent_1"] == 2.0
        assert scheduler.agent_act_delays["agent_1"] == 0.5

    def test_backward_compatible_registration(self):
        """Test backward-compatible registration without TickConfig."""
        from heron.scheduling.scheduler import EventScheduler

        scheduler = EventScheduler()
        scheduler.register_agent("agent_1", tick_interval=3.0, obs_delay=0.2)

        assert scheduler.agent_intervals["agent_1"] == 3.0
        assert scheduler.agent_obs_delays["agent_1"] == 0.2
        assert "agent_1" in scheduler.agent_tick_configs

    def test_jittered_ticks_vary(self):
        """Test that jittered tick intervals actually vary."""
        from heron.scheduling.scheduler import EventScheduler

        scheduler = EventScheduler()
        config = TickConfig.with_jitter(
            tick_interval=1.0,
            jitter_type=JitterType.UNIFORM,
            jitter_ratio=0.2,
            seed=42,
        )

        scheduler.register_agent("agent_1", tick_config=config)

        # Collect scheduled timestamps
        timestamps = []
        for _ in range(10):
            event = scheduler.pop()
            if event:
                timestamps.append(event.timestamp)
                scheduler.current_time = event.timestamp
                scheduler.schedule_agent_tick("agent_1")

        # Calculate intervals
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        # Should have some variation
        assert len(set(round(i, 6) for i in intervals)) > 1


class TestTickConfigIntegrationWithAgent:
    """Test TickConfig integration with Agent base class."""

    def _make_test_agent(self, **kwargs):
        """Create a test agent with proper state/action setup."""
        from heron.agents.field_agent import FieldAgent
        from heron.core.action import Action
        from heron.core.feature import FeatureProvider

        class MockFeature(FeatureProvider):
            visibility = ["public"]

            def vector(self):
                return np.array([1.0], dtype=np.float32)

            def names(self):
                return ["value"]

            def to_dict(self):
                return {"value": 1.0}

            @classmethod
            def from_dict(cls, d):
                return cls()

            def set_values(self, **kw):
                pass

        class TestAgent(FieldAgent):
            def set_action(self):
                self.action.set_specs(
                    dim_c=1,
                    range=(np.array([-1.0]), np.array([1.0]))
                )

            def set_state(self):
                self.state.features = [MockFeature()]

        return TestAgent(
            agent_id="test_agent",
            config={"name": "test"},
            **kwargs
        )

    def test_agent_with_tick_config(self):
        """Test creating agent with TickConfig."""
        tick_cfg = TickConfig.with_jitter(
            tick_interval=2.0,
            obs_delay=0.1,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,
            seed=42,
        )

        agent = self._make_test_agent(tick_config=tick_cfg)

        assert agent.tick_config is tick_cfg
        assert agent.tick_interval == 2.0
        assert agent.obs_delay == 0.1

    def test_agent_backward_compatible(self):
        """Test agent creation with legacy timing params."""
        agent = self._make_test_agent(
            tick_interval=3.0,
            obs_delay=0.2,
        )

        assert agent.tick_interval == 3.0
        assert agent.obs_delay == 0.2
        assert agent.tick_config.jitter_type == JitterType.NONE

    def test_enable_jitter(self):
        """Test enable_jitter method."""
        agent = self._make_test_agent(
            tick_interval=1.0,
            obs_delay=0.5,
        )

        # Initially deterministic
        assert agent.tick_config.jitter_type == JitterType.NONE

        # Enable jitter
        agent.enable_jitter(jitter_ratio=0.1, seed=42)

        assert agent.tick_config.jitter_type == JitterType.GAUSSIAN
        assert agent.tick_config.jitter_ratio == 0.1
        # Base values preserved
        assert agent.tick_interval == 1.0
        assert agent.obs_delay == 0.5

    def test_disable_jitter(self):
        """Test disable_jitter method."""
        tick_cfg = TickConfig.with_jitter(
            tick_interval=1.0,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=0.1,
            seed=42,
        )

        agent = self._make_test_agent(tick_config=tick_cfg)

        # Initially jittered
        assert agent.tick_config.jitter_type == JitterType.GAUSSIAN

        # Disable jitter
        agent.disable_jitter()

        assert agent.tick_config.jitter_type == JitterType.NONE
        # Base value preserved
        assert agent.tick_interval == 1.0
