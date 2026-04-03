"""Unit tests for Disturbance and DisturbanceSchedule."""

import numpy as np
import pytest

from heron.agents.constants import SYSTEM_AGENT_ID
from heron.scheduling.disturbance import Disturbance, DisturbanceSchedule
from heron.scheduling.event import EventType
from heron.scheduling.scheduler import EventScheduler


# =============================================================================
# Disturbance dataclass
# =============================================================================

class TestDisturbance:
    def test_defaults(self):
        d = Disturbance(timestamp=5.0, disturbance_type="line_fault")
        assert d.timestamp == 5.0
        assert d.disturbance_type == "line_fault"
        assert d.payload == {}
        assert d.requires_physics is True

    def test_custom_payload(self):
        d = Disturbance(
            timestamp=10.0,
            disturbance_type="load_spike",
            payload={"bus": 5, "delta_kw": 100.0},
            requires_physics=False,
        )
        assert d.payload["bus"] == 5
        assert d.payload["delta_kw"] == 100.0
        assert d.requires_physics is False


# =============================================================================
# DisturbanceSchedule
# =============================================================================

class TestDisturbanceSchedule:
    def test_sorted_by_timestamp(self):
        sched = DisturbanceSchedule([
            Disturbance(timestamp=25.0, disturbance_type="spike"),
            Disturbance(timestamp=5.0, disturbance_type="fault"),
            Disturbance(timestamp=15.0, disturbance_type="drop"),
        ])
        assert len(sched) == 3
        assert sched.disturbances[0].timestamp == 5.0
        assert sched.disturbances[1].timestamp == 15.0
        assert sched.disturbances[2].timestamp == 25.0

    def test_empty_schedule(self):
        sched = DisturbanceSchedule([])
        assert len(sched) == 0
        assert repr(sched) == "DisturbanceSchedule(n=0)"

    def test_none_defaults_to_empty(self):
        sched = DisturbanceSchedule()
        assert len(sched) == 0

    def test_from_list(self):
        specs = [
            {"t": 25.0, "type": "load_spike", "bus": 5, "delta_kw": 100.0},
            {"t": 12.3, "type": "line_fault", "element": "line_7_8"},
        ]
        sched = DisturbanceSchedule.from_list(specs)
        assert len(sched) == 2
        # Sorted by timestamp
        assert sched.disturbances[0].timestamp == 12.3
        assert sched.disturbances[0].disturbance_type == "line_fault"
        assert sched.disturbances[0].payload == {"element": "line_7_8"}
        assert sched.disturbances[1].timestamp == 25.0
        assert sched.disturbances[1].payload == {"bus": 5, "delta_kw": 100.0}

    def test_from_list_requires_physics_override(self):
        specs = [{"t": 1.0, "type": "noise", "requires_physics": False}]
        sched = DisturbanceSchedule.from_list(specs)
        assert sched.disturbances[0].requires_physics is False

    def test_poisson_seeded(self):
        rng = np.random.default_rng(42)
        sched = DisturbanceSchedule.poisson(
            rate=1.0,
            disturbance_types=["fault", "spike"],
            t_end=100.0,
            rng=rng,
        )
        assert len(sched) > 0
        # All timestamps within (0, 100)
        for d in sched.disturbances:
            assert 0 < d.timestamp < 100.0
        # Sorted
        timestamps = [d.timestamp for d in sched.disturbances]
        assert timestamps == sorted(timestamps)

    def test_poisson_reproducible(self):
        s1 = DisturbanceSchedule.poisson(
            rate=0.5, disturbance_types=["fault"], t_end=50.0,
            rng=np.random.default_rng(123),
        )
        s2 = DisturbanceSchedule.poisson(
            rate=0.5, disturbance_types=["fault"], t_end=50.0,
            rng=np.random.default_rng(123),
        )
        assert len(s1) == len(s2)
        for d1, d2 in zip(s1.disturbances, s2.disturbances):
            assert d1.timestamp == d2.timestamp

    def test_poisson_default_payload(self):
        sched = DisturbanceSchedule.poisson(
            rate=2.0, disturbance_types=["fault"], t_end=10.0,
            rng=np.random.default_rng(0),
            default_payload={"severity": "high"},
        )
        for d in sched.disturbances:
            assert d.payload == {"severity": "high"}

    def test_enqueue(self):
        sched = DisturbanceSchedule([
            Disturbance(timestamp=5.0, disturbance_type="fault"),
            Disturbance(timestamp=10.0, disturbance_type="spike"),
        ])
        scheduler = EventScheduler(start_time=0.0)
        sched.enqueue(scheduler)
        assert scheduler.pending_count == 2

        e1 = scheduler.pop()
        assert e1.event_type == EventType.ENV_UPDATE
        assert e1.timestamp == 5.0
        assert e1.priority == 0
        assert e1.agent_id == SYSTEM_AGENT_ID
        assert e1.payload["disturbance"].disturbance_type == "fault"

        e2 = scheduler.pop()
        assert e2.timestamp == 10.0
        assert e2.payload["disturbance"].disturbance_type == "spike"
