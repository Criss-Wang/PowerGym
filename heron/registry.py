"""Environment registry for HERON.

Mirrors Gymnasium's ``gym.make()`` pattern — register environments by ID
and instantiate them with ``heron.make(env_id, **kwargs)``.

Example::

    import heron
    import heron.demo_envs  # auto-registers built-in demo envs

    env = heron.make("Thermostat-v0")

    # Override default kwargs
    env = heron.make("Thermostat-v0", max_steps=200)

    # Register a custom env
    heron.register(
        env_id="MyEnv-v0",
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
    env_id: str,
    entry_point: Union[str, Callable, Type],
    kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Register an environment.

    Args:
        env_id: Unique environment identifier (e.g. ``"Thermostat-v0"``).
        entry_point: Callable or import path ``"module:Class"``.
        kwargs: Default keyword arguments for the entry point.

    Raises:
        ValueError: If *env_id* is already registered.
    """
    if env_id in _registry:
        raise ValueError(
            f"Environment '{env_id}' is already registered. "
            f"Use a different ID or call unregister('{env_id}') first."
        )
    _registry[env_id] = EnvSpec(
        id=env_id,
        entry_point=entry_point,
        kwargs=dict(kwargs or {}),
    )


def make(env_id: str, **override_kwargs: Any) -> Any:
    """Create an environment instance by ID.

    Default kwargs from ``register()`` are merged with *override_kwargs*
    (overrides take precedence).

    Args:
        env_id: Registered environment identifier.
        **override_kwargs: Keyword arguments that override defaults.

    Returns:
        Environment instance.

    Raises:
        KeyError: If *env_id* is not registered.
        ValueError: If the entry_point string cannot be resolved.
    """
    if env_id not in _registry:
        available = ", ".join(sorted(_registry.keys())) or "(none)"
        raise KeyError(
            f"Environment '{env_id}' not found in registry. "
            f"Available: {available}"
        )
    env_spec = _registry[env_id]
    entry_point = env_spec.entry_point

    # Lazy import from string path
    if isinstance(entry_point, str):
        module_path, _, attr_name = entry_point.partition(":")
        if not attr_name:
            raise ValueError(
                f"entry_point string must be 'module.path:ClassName', "
                f"got '{entry_point}'"
            )
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"Failed to import module '{module_path}' for env "
                f"'{env_id}'. Check the entry_point: '{env_spec.entry_point}'"
            ) from exc
        if not hasattr(module, attr_name):
            raise ValueError(
                f"Module '{module_path}' has no attribute '{attr_name}' "
                f"(entry_point='{env_spec.entry_point}' for env '{env_id}')."
            )
        entry_point = getattr(module, attr_name)

    merged_kwargs = {**env_spec.kwargs, **override_kwargs}
    return entry_point(**merged_kwargs)


def unregister(env_id: str) -> None:
    """Remove an environment from the registry.

    Args:
        env_id: Environment identifier to remove.

    Raises:
        KeyError: If *env_id* is not registered.
    """
    if env_id not in _registry:
        raise KeyError(f"Environment '{env_id}' is not registered.")
    del _registry[env_id]


def list_envs() -> Dict[str, EnvSpec]:
    """Return a snapshot of the current registry.

    Returns:
        A dict mapping env ID strings to their ``EnvSpec`` objects.
        The returned dict is a shallow copy — mutating it does not
        affect the registry.
    """
    return dict(_registry)


def spec(env_id: str) -> EnvSpec:
    """Return the ``EnvSpec`` for a registered environment.

    Args:
        env_id: Registered environment identifier.

    Raises:
        KeyError: If *env_id* is not registered.
    """
    if env_id not in _registry:
        raise KeyError(f"Environment '{env_id}' is not registered.")
    return _registry[env_id]
