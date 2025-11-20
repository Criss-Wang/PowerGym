"""Utility functions for message broker infrastructure."""

from powergrid.messaging.base import MessageBroker
from powergrid.messaging.memory import InMemoryBroker


def create_message_broker(broker_type: str = "memory", **config) -> MessageBroker:
    """Factory to create message brokers.

    Args:
        broker_type: Type of broker - 'memory'
        **config: Broker-specific configuration

    Returns:
        MessageBroker instance

    Examples:
        >>> # In-memory broker (default, for testing/development)
        >>> broker = create_message_broker("memory")
    Raises:
        ValueError: If broker_type is not recognized
    """
    if broker_type == "memory":
        return InMemoryBroker()
    # TODO: Implement other broker types, e.g., KafkaBroker, RedisBroker
    else:
        raise ValueError(
            f"Unknown broker type: {broker_type}. "
            f"Supported types: 'memory', 'kafka', 'redis'"
        )
