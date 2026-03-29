"""Message types and structures for agent communication.

This module defines the core message format and type system used across
all broker implementations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class MessageType(Enum):
    """Generic message types for agent communication.

    Domains can use MessageType.CUSTOM with a 'custom_type' key in the payload
    for domain-specific message types.
    """
    ACTION = "action"
    INFO = "info"
    BROADCAST = "broadcast"
    STATE_UPDATE = "state_update"
    RESULT = "result"  # Generic result message
    # TODO: When domains start using CUSTOM message types in practice,
    # re-introduce MessageTypeRegistry to catalog and validate custom types.
    CUSTOM = "custom"  # For domain-specific message types


@dataclass
class Message:
    """Generic message structure for agent communication.

    This message format is implementation-agnostic and works with any broker backend.

    Attributes:
        env_id: Environment/rollout identifier for multi-environment isolation
        sender_id: Sender agent ID
        recipient_id: Recipient agent ID (or "broadcast" for broadcasts)
        timestamp: Message timestamp
        message_type: Type of message (action, info, etc.)
        payload: Arbitrary message data as dict
    """
    env_id: str
    sender_id: str
    recipient_id: str
    timestamp: float
    message_type: MessageType
    payload: Dict[str, Any]
