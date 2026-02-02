"""Tests for heron.utils.registry module."""

import pytest

from heron.utils.registry import ProviderRegistry, provider


class TestProviderRegistry:
    """Test ProviderRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        ProviderRegistry._types.clear()

    def test_register_with_class_name(self):
        """Test registering a type using its class name."""

        class MyProvider:
            pass

        ProviderRegistry.register(MyProvider)

        assert "MyProvider" in ProviderRegistry._types
        assert ProviderRegistry._types["MyProvider"] is MyProvider

    def test_register_with_custom_name(self):
        """Test registering a type with custom name."""

        class MyProvider:
            pass

        ProviderRegistry.register(MyProvider, name="custom_name")

        assert "custom_name" in ProviderRegistry._types
        assert "MyProvider" not in ProviderRegistry._types

    def test_get_registered_type(self):
        """Test getting a registered type."""

        class MyProvider:
            pass

        ProviderRegistry.register(MyProvider, name="test_provider")
        result = ProviderRegistry.get("test_provider")

        assert result is MyProvider

    def test_get_unregistered_returns_none(self):
        """Test getting unregistered type returns None."""
        result = ProviderRegistry.get("nonexistent")

        assert result is None

    def test_all_returns_copy(self):
        """Test all() returns a copy of registered types."""

        class Provider1:
            pass

        class Provider2:
            pass

        ProviderRegistry.register(Provider1)
        ProviderRegistry.register(Provider2)

        all_types = ProviderRegistry.all()

        assert len(all_types) == 2
        assert "Provider1" in all_types
        assert "Provider2" in all_types
        # Verify it's a copy
        all_types["new"] = str
        assert "new" not in ProviderRegistry._types

    def test_register_overwrites_existing(self):
        """Test that registering with same name overwrites."""

        class Provider1:
            pass

        class Provider2:
            pass

        ProviderRegistry.register(Provider1, name="same_name")
        ProviderRegistry.register(Provider2, name="same_name")

        assert ProviderRegistry.get("same_name") is Provider2


class TestProviderDecorator:
    """Test provider decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        ProviderRegistry._types.clear()

    def test_decorator_without_name(self):
        """Test decorator without custom name uses class name."""

        @provider()
        class DecoratedProvider:
            pass

        assert "DecoratedProvider" in ProviderRegistry._types
        assert ProviderRegistry.get("DecoratedProvider") is DecoratedProvider

    def test_decorator_with_name(self):
        """Test decorator with custom name."""

        @provider(name="custom")
        class DecoratedProvider:
            pass

        assert "custom" in ProviderRegistry._types
        assert "DecoratedProvider" not in ProviderRegistry._types

    def test_decorator_returns_original_class(self):
        """Test that decorator returns the original class."""

        @provider()
        class DecoratedProvider:
            def method(self):
                return "test"

        instance = DecoratedProvider()
        assert instance.method() == "test"

    def test_multiple_decorated_classes(self):
        """Test multiple decorated classes."""

        @provider()
        class Provider1:
            pass

        @provider()
        class Provider2:
            pass

        @provider(name="third")
        class Provider3:
            pass

        all_types = ProviderRegistry.all()
        assert len(all_types) == 3
        assert "Provider1" in all_types
        assert "Provider2" in all_types
        assert "third" in all_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
