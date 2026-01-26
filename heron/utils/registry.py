"""Feature provider registry for the HERON framework."""

from typing import Dict, Optional, Type


class ProviderRegistry:
    """Registry for feature providers."""
    _types: Dict[str, Type] = {}

    @classmethod
    def register(cls, typ: Type, name: Optional[str] = None) -> None:
        """Register a provider type."""
        key = name or typ.__name__
        cls._types[key] = typ

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get a provider by name."""
        return cls._types.get(name)

    @classmethod
    def all(cls) -> Dict[str, Type]:
        """Get all registered providers."""
        return dict(cls._types)


def provider(name: Optional[str] = None):
    """Decorator to auto-register provider."""
    def deco(typ):
        ProviderRegistry.register(typ, name)
        return typ
    return deco
