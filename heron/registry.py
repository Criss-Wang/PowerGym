"""Environment registry for HERON.

Mirrors Gymnasium's ``gym.make()`` pattern — register environments by ID
and instantiate them with ``heron.make(env_id, **kwargs)``.

Example::

    import heron

    # Built-in demo envs are auto-registered on import
    env = heron.make("Thermostat-v0")

    # Override default kwargs
    env = heron.make("Thermostat-v0", max_steps=200)

    # Register a custom env
    heron.register(
        id="MyEnv-v0",
        entry_point="my_package.envs:MyEnv",
        kwargs={"default_param": 42},
    )
    env = heron.make("MyEnv-v0")
"""

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type, Union


@dataclass(frozen=True)
class EnvSpec:
    """Specification for a registered environment.

    Attributes:
        id: Unique environment identifier (e.g. ``"Thermostat-v0"``).
        entry_point: Either a callable that returns an env, or a string
            ``"module.path:ClassName"`` that will be lazily imported.
        kwargs: Default keyword arguments passed to the entry point.
    """

    id: str
    entry_point: Union[str, Callable, Type]
    kwargs: Dict[str, Any] = field(default_factory=dict)


# Global registry mapping env ID -> EnvSpec
_registry: Dict[str, EnvSpec] = {}


def register(
    id: str,
    entry_point: Union[str, Callable, Type],
    kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Register an environment.

    Args:
        id: Unique environment identifier.
        entry_point: Callable or import path ``"module:Class"``.
        kwargs: Default keyword arguments for the entry point.

    Raises:
        ValueError: If *id* is already registered.
    """
    if id in _registry:
        raise ValueError(
            f"Environment '{id}' is already registered. "
            f"Use a different ID or call unregister('{id}') first."
        )
    _registry[id] = EnvSpec(
        id=id,
        entry_point=entry_point,
        kwargs=dict(kwargs or {}),
    )


def make(id: str, **override_kwargs: Any) -> Any:
    """Create an environment instance by ID.

    Default kwargs from ``register()`` are merged with *override_kwargs*
    (overrides take precedence).

    Args:
        id: Registered environment identifier.
        **override_kwargs: Keyword arguments that override defaults.

    Returns:
        Environment instance.

    Raises:
        KeyError: If *id* is not registered.
    """
    if id not in _registry:
        available = ", ".join(sorted(_registry.keys())) or "(none)"
        raise KeyError(
            f"Environment '{id}' not found in registry. "
            f"Available: {available}"
        )
    spec = _registry[id]
    entry_point = spec.entry_point

    # Lazy import from string path
    if isinstance(entry_point, str):
        module_path, _, attr_name = entry_point.partition(":")
        if not attr_name:
            raise ValueError(
                f"entry_point string must be 'module.path:ClassName', "
                f"got '{entry_point}'"
            )
        module = importlib.import_module(module_path)
        entry_point = getattr(module, attr_name)

    merged_kwargs = {**spec.kwargs, **override_kwargs}
    return entry_point(**merged_kwargs)


def unregister(id: str) -> None:
    """Remove an environment from the registry.

    Args:
        id: Environment identifier to remove.

    Raises:
        KeyError: If *id* is not registered.
    """
    if id not in _registry:
        raise KeyError(f"Environment '{id}' is not registered.")
    del _registry[id]


def list_envs() -> Dict[str, EnvSpec]:
    """Return a copy of the current registry."""
    return dict(_registry)


def spec(id: str) -> EnvSpec:
    """Return the EnvSpec for a registered environment.

    Raises:
        KeyError: If *id* is not registered.
    """
    if id not in _registry:
        raise KeyError(f"Environment '{id}' is not registered.")
    return _registry[id]
