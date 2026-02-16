from dataclasses import fields, asdict, MISSING
from typing import Any, ClassVar, Dict, List, Sequence, Type, TypeVar
import numpy as np

T = TypeVar("T", bound="FeatureProvider")

# Global feature registry for State.from_dict() reconstruction
_FEATURE_REGISTRY: Dict[str, Type["FeatureProvider"]] = {}


class FeatureMeta(type):
    """Metaclass that auto-registers FeatureProvider subclasses."""

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Auto-register all FeatureProvider subclasses (but not FeatureProvider itself)
        # Check if the class being created (name) is not FeatureProvider AND has FeatureProvider as a base
        if name != 'FeatureProvider' and bases:
            _FEATURE_REGISTRY[name] = cls

        return cls


class FeatureProvider(metaclass=FeatureMeta):
    """Base class for feature providers.

    Subclasses should:
    1. Use @dataclass(slots=True) decorator for clarity and memory optimization
    2. Define visibility as ClassVar[Sequence[str]]
    3. Override set_values() for validation if needed

    Example:
        @dataclass(slots=True)
        class MyFeature(FeatureProvider):
            visibility: ClassVar[Sequence[str]] = ["public"]
            value: float = 0.0
    """

    visibility: ClassVar[Sequence[str]]
    _class_feature_name: ClassVar[str]  # Class-level default name
    _instance_feature_name: str = None  # Instance-level override (set via set_feature_name)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._class_feature_name = cls.__name__
        # Subclasses should use @dataclass(slots=True) explicitly

    @property
    def feature_name(self) -> str:
        """Get feature name. Instance name takes precedence over class name."""
        if self._instance_feature_name is not None:
            return self._instance_feature_name
        return self._class_feature_name

    def set_feature_name(self, name: str) -> "FeatureProvider":
        """Set a custom instance-level feature name. Returns self for chaining."""
        self._instance_feature_name = name
        return self

    def vector(self) -> np.ndarray:
        return np.array(
            [getattr(self, f.name) for f in fields(self)],
            dtype=np.float32
        )

    def names(self) -> List[str]:
        return [f.name for f in fields(self)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], d: Dict[str, Any]) -> T:
        kwargs = {}
        for f in fields(cls):
            if f.name in d:
                kwargs[f.name] = d[f.name]
            elif f.default is not MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not MISSING:
                kwargs[f.name] = f.default_factory()
        return cls(**kwargs)

    def set_values(self, **kwargs: Any) -> None:
        """Override this method to add validation logic."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def reset(self: T, **overrides: Any) -> T:
        for f in fields(self):
            if f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())
        self.set_values(**overrides)
        return self

    def is_observable_by(
        self,
        requestor_id: str,
        requestor_level: int,
        owner_id: str,
        owner_level: int
    ) -> bool:
        if "public" in self.visibility:
            return True
        if "owner" in self.visibility and requestor_id == owner_id:
            return True
        if "system" in self.visibility and requestor_level >= 3:
            return True
        if "upper_level" in self.visibility and requestor_level == owner_level + 1:
            return True
        return False


# Registry access functions
def get_feature_class(feature_name: str) -> Type[FeatureProvider]:
    """Get feature class by name from registry.

    Args:
        feature_name: Name of the feature class

    Returns:
        Feature class type

    Raises:
        ValueError: If feature not registered
    """
    if feature_name not in _FEATURE_REGISTRY:
        raise ValueError(f"Feature '{feature_name}' not found in registry. Available: {list(_FEATURE_REGISTRY.keys())}")
    return _FEATURE_REGISTRY[feature_name]


def get_all_registered_features() -> Dict[str, Type[FeatureProvider]]:
    """Get all registered feature classes.

    Returns:
        Dict mapping feature names to feature classes
    """
    return _FEATURE_REGISTRY.copy()
