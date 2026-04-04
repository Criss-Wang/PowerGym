"""Tests for heron.registry — environment registration and instantiation."""

import pytest

from heron.registry import register, make, unregister, list_envs, spec, _registry


class _DummyEnv:
    """Minimal env stub for registry tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a clean registry."""
    saved = dict(_registry)
    _registry.clear()
    yield
    _registry.clear()
    _registry.update(saved)


class TestRegister:
    def test_register_callable(self):
        register("Test-v0", entry_point=_DummyEnv, kwargs={"a": 1})
        s = spec("Test-v0")
        assert s.id == "Test-v0"
        assert s.kwargs == {"a": 1}

    def test_register_string_entry_point(self):
        register("Test-v1", entry_point="tests.test_registry:_DummyEnv")
        s = spec("Test-v1")
        assert s.entry_point == "tests.test_registry:_DummyEnv"

    def test_duplicate_id_raises(self):
        register("Dup-v0", entry_point=_DummyEnv)
        with pytest.raises(ValueError, match="already registered"):
            register("Dup-v0", entry_point=_DummyEnv)


class TestMake:
    def test_make_from_callable(self):
        register("Test-v0", entry_point=_DummyEnv, kwargs={"x": 10})
        env = make("Test-v0")
        assert isinstance(env, _DummyEnv)
        assert env.kwargs == {"x": 10}

    def test_make_override_kwargs(self):
        register("Test-v0", entry_point=_DummyEnv, kwargs={"x": 10, "y": 20})
        env = make("Test-v0", x=99)
        assert env.kwargs["x"] == 99
        assert env.kwargs["y"] == 20

    def test_make_from_string(self):
        register("Test-v0", entry_point="tests.test_registry:_DummyEnv")
        env = make("Test-v0", foo="bar")
        assert isinstance(env, _DummyEnv)
        assert env.kwargs == {"foo": "bar"}

    def test_make_unknown_id_raises(self):
        with pytest.raises(KeyError, match="not found in registry"):
            make("NonExistent-v0")

    def test_make_bad_entry_point_string(self):
        register("Bad-v0", entry_point="no_colon_here")
        with pytest.raises(ValueError, match="module.path:ClassName"):
            make("Bad-v0")


class TestUnregister:
    def test_unregister(self):
        register("Test-v0", entry_point=_DummyEnv)
        unregister("Test-v0")
        assert "Test-v0" not in list_envs()

    def test_unregister_unknown_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            unregister("Ghost-v0")


class TestListAndSpec:
    def test_list_envs_returns_copy(self):
        register("A-v0", entry_point=_DummyEnv)
        envs = list_envs()
        envs.clear()
        assert "A-v0" in list_envs()

    def test_spec_unknown_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            spec("Ghost-v0")
