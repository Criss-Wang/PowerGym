"""Factory for creating simple numeric FeatureProvider subclasses.

Reduces the boilerplate of defining a ``@dataclass(slots=True)`` FeatureProvider
subclass when all fields are plain floats with optional clipping.

Usage::

    PowerFeature = NumericFeature("PowerFeature", "power", "capacity", visibility=("public",))
    f = PowerFeature(power=0.5, capacity=1.0)
    f.vector()  # array([0.5, 1.0])
"""

from dataclasses import field, make_dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import numpy as np

from heron.core.feature import FeatureProvider


def NumericFeature(
    name: str,
    *field_names: str,
    visibility: Sequence[str] = ("public",),
    defaults: Optional[Dict[str, float]] = None,
    clips: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Type[FeatureProvider]:
    """Create a FeatureProvider subclass whose fields are all floats.

    Args:
        name: Class name for the generated feature (also used as registry key).
        *field_names: Names of the float fields.
        visibility: Visibility tags (default ``("public",)``).
        defaults: Optional mapping of field name -> default value. Fields not
            listed default to ``0.0``.
        clips: Optional mapping of field name -> ``(min, max)`` bounds enforced
            in ``set_values``.

    Returns:
        A new ``FeatureProvider`` subclass registered in the feature registry.

    Example::

        PowerFeature = NumericFeature(
            "PowerFeature", "power", "capacity",
            defaults={"capacity": 1.0},
            clips={"power": (-1.0, 1.0)},
        )
    """
    if not field_names:
        raise ValueError("NumericFeature requires at least one field name")

    defaults = defaults or {}
    clips = clips or {}

    # Build dataclass fields: (name, type, Field) tuples
    dc_fields = []
    for fn in field_names:
        default_val = defaults.get(fn, 0.0)
        dc_fields.append((fn, float, field(default=default_val)))

    # Capture in closure for the generated methods
    _field_names = tuple(field_names)
    _clips = dict(clips)
    _visibility = tuple(visibility)

    def _vector(self) -> np.ndarray:
        return np.array([getattr(self, fn) for fn in _field_names], dtype=np.float32)

    def _set_values(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key in _clips:
                    lo, hi = _clips[key]
                    value = float(np.clip(value, lo, hi))
                setattr(self, key, float(value))

    # Create the dataclass subclass via make_dataclass
    cls = make_dataclass(
        name,
        dc_fields,
        bases=(FeatureProvider,),
        slots=True,
        namespace={
            "vector": _vector,
            "set_values": _set_values,
        },
    )

    # Set class-level visibility (ClassVar — not a dataclass field)
    cls.visibility = _visibility

    return cls
