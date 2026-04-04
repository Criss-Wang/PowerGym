"""Shared features for the TwoRoomHeating demo environments."""

from dataclasses import dataclass
from typing import ClassVar, Sequence

from heron.core.feature import Feature


@dataclass(slots=True)
class ZoneTemperatureFeature(Feature):
    """Temperature state for a single zone.

    Attributes:
        temperature: Current zone temperature (Celsius).
        target: Target setpoint temperature (Celsius).
    """

    visibility: ClassVar[Sequence[str]] = ("owner", "upper_level")
    temperature: float = 18.0
    target: float = 22.0


@dataclass(slots=True)
class VentStatusFeature(Feature):
    """Status of the emergency ventilation system (used from v2+).

    Attributes:
        is_open: Vent opening fraction (0.0=closed, 1.0=fully open).
        cooling_power: Current cooling effect being applied.
    """

    visibility: ClassVar[Sequence[str]] = ("owner", "upper_level")
    is_open: float = 0.0
    cooling_power: float = 0.0
