"""Messaging infrastructure for HERON agent communication.

This module provides pub/sub messaging:
- Message: Message container with type, sender, recipient, payload
- MessageType: Message type enumeration
- MessageBroker: Abstract message broker interface
- InMemoryBroker: In-memory message broker implementation
- ChannelManager: Channel lifecycle management
"""

from heron.messaging.messages import (
    Message,
    MessageType,
)
from heron.messaging.broker_base import MessageBroker
from heron.messaging.channels import ChannelManager
from heron.messaging.in_memory_broker import InMemoryBroker

__all__ = [
    "Message",
    "MessageType",
    "MessageBroker",
    "InMemoryBroker",
    "ChannelManager",
]
