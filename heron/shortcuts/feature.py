"""Declarative feature definition for HERON shortcuts.

Provides ``Feature`` base class and ``Clipped`` descriptor for defining
features as plain Python classes instead of factory calls.

Usage::

    class Output(Feature):
        value: float = Clipped(default=0.0, min=-1.0, max=1.0)

    class Battery(Feature):
        charge: float = Clipped(default=0.5, min=0.0, max=1.0)
        capacity: float = 1.0
"""

import dataclasses
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from heron.core.feature import FeatureProvider


class Clipped:
    """Declare a float field with clipping bounds.

    Args:
        default: Default value for the field (default 0.0).
        min: Minimum allowed value (default -inf).
        max: Maximum allowed value (default +inf).

    Example::

        class Output(Feature):
            value: float = Clipped(default=0.0, min=-1.0, max=1.0)
    """

    def __init__(self, default: float = 0.0, *, min: float = -float("inf"), max: float = float("inf")):
        self.default = default
        self.min = min
        self.max = max


class Feature(FeatureProvider):
    """Declarative base for features — define fields as type annotations.

    Subclasses are automatically converted to dataclasses. Each field can use:
    - A plain default value: ``capacity: float = 1.0``
    - A ``Clipped`` descriptor: ``charge: float = Clipped(0.5, min=0.0, max=1.0)``

    Example::

        class Battery(Feature):
            charge: float = Clipped(default=0.5, min=0.0, max=1.0)
            capacity: float = 1.0

        b = Battery()           # Battery(charge=0.5, capacity=1.0)
        b.vector()              # array([0.5, 1.0])
        b.set_values(charge=2)  # clipped to 1.0
    """

    visibility: Sequence[str] = ("public",)

    def __init_subclass__(cls, visibility=None, **kwargs):
        super().__init_subclass__(**kwargs)

        if visibility is not None:
            cls.visibility = visibility

        # Only process classes that define their own fields
        own_annotations = cls.__dict__.get("__annotations__", {})
        if not own_annotations:
            return

        # Process Clipped descriptors before @dataclass
        clips: Dict[str, Tuple[float, float]] = {}
        for attr_name in list(own_annotations.keys()):
            val = cls.__dict__.get(attr_name)
            if isinstance(val, Clipped):
                clips[attr_name] = (val.min, val.max)
                # Replace Clipped with plain default for dataclass processing
                setattr(cls, attr_name, val.default)

        cls._feature_clips = clips

        # Ensure @dataclass only sees this class's own annotations
        # (prevents inheriting _instance_feature_name etc. from FeatureProvider)
        cls.__annotations__ = own_annotations.copy()

        # Apply @dataclass (slots=False to avoid class recreation)
        dataclasses.dataclass(cls)

        # Generate clipped set_values if any fields have bounds
        if clips:
            _clips = dict(clips)

            def _set_values(self, **kwargs: Any) -> None:
                for key, value in kwargs.items():
                    if hasattr(self, key):
                        if key in _clips:
                            lo, hi = _clips[key]
                            value = float(np.clip(value, lo, hi))
                        setattr(self, key, float(value))

            cls.set_values = _set_values
