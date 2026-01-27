"""Base message broker interface and utilities.

This module defines the abstract MessageBroker interface that all broker
implementations must follow, along with message structures and channel management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MessageType(Enum):
    """Generic message types for agent communication.

    Domains can register additional message types via MessageTypeRegistry.
    """
    ACTION = "action"
    INFO = "info"
    BROADCAST = "broadcast"
    STATE_UPDATE = "state_update"
    RESULT = "result"  # Generic result message
    CUSTOM = "custom"  # For domain-specific message types


class MessageTypeRegistry:
    """Registry for domain-specific message types.

    Allows domains to register custom message type identifiers that can be
    used in the payload's 'custom_type' field when MessageType.CUSTOM is used.

    Example:
        # In powergrid domain initialization
        MessageTypeRegistry.register("power_flow_result")
        MessageTypeRegistry.register("voltage_update")

        # When creating a message
        msg = Message(
            ...,
            message_type=MessageType.CUSTOM,
            payload={"custom_type": "power_flow_result", "data": {...}}
        )
    """
    _registered_types: Dict[str, str] = {}

    @classmethod
    def register(cls, type_name: str, description: str = "") -> None:
        """Register a domain-specific message type.

        Args:
            type_name: Unique identifier for the custom type
            description: Optional description of the message type
        """
        cls._registered_types[type_name] = description

    @classmethod
    def is_registered(cls, type_name: str) -> bool:
        """Check if a custom type is registered."""
        return type_name in cls._registered_types

    @classmethod
    def get_all(cls) -> Dict[str, str]:
        """Get all registered custom types."""
        return cls._registered_types.copy()


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


class MessageBroker(ABC):
    """Abstract message broker interface.

    This interface defines the contract for any message broker implementation.
    Implementations can use Kafka, RabbitMQ, Redis, in-memory queues, etc.

    The broker provides a pub/sub model where agents publish messages to channels
    and consume messages from channels based on recipient filtering.
    """

    @abstractmethod
    def create_channel(self, channel_name: str) -> None:
        """Create a new message channel.

        Channels are named communication pathways. Depending on the implementation,
        these might map to Kafka topics, Redis pub/sub channels, RabbitMQ queues, etc.

        Args:
            channel_name: Unique identifier for the channel
        """
        pass

    @abstractmethod
    def publish(self, channel: str, message: Message) -> None:
        """Publish a message to a channel.

        Args:
            channel: Channel name to publish to
            message: Message to publish
        """
        pass

    @abstractmethod
    def consume(
        self,
        channel: str,
        recipient_id: str,
        env_id: str,
        clear: bool = True
    ) -> List[Message]:
        """Consume messages from a channel.

        Args:
            channel: Channel name to consume from
            recipient_id: Filter messages for this recipient
            env_id: Filter messages for this environment
            clear: If True, remove consumed messages from channel (default: True)

        Returns:
            List of messages matching the filters
        """
        pass

    @abstractmethod
    def subscribe(self, channel: str, callback: Callable[[Message], None]) -> None:
        """Subscribe to a channel with a callback.

        When messages are published to the channel, the callback will be invoked.
        This is for asynchronous/reactive message handling.

        Args:
            channel: Channel name to subscribe to
            callback: Function to call when messages arrive
        """
        pass

    @abstractmethod
    def clear_environment(self, env_id: str) -> None:
        """Clear all messages for a specific environment.

        Useful for resetting environments in vectorized rollouts.

        Args:
            env_id: Environment identifier
        """
        pass

    @abstractmethod
    def get_environment_channels(self, env_id: str) -> List[str]:
        """Get all channels associated with an environment.

        Args:
            env_id: Environment identifier

        Returns:
            List of channel names for this environment
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the broker and clean up resources.

        Should be called when done using the broker to properly release
        connections, threads, etc.
        """
        pass


class ChannelRegistry:
    """Registry for domain-specific channel types.

    Allows domains to document their custom channel types.

    Example:
        # In powergrid domain initialization
        ChannelRegistry.register("power_flow", "Power flow results from environment")

        # When creating a channel
        channel = ChannelManager.custom_channel("power_flow", env_id, agent_id)
    """
    _registered_types: Dict[str, str] = {}

    @classmethod
    def register(cls, channel_type: str, description: str = "") -> None:
        """Register a domain-specific channel type.

        Args:
            channel_type: Unique identifier for the channel type
            description: Description of the channel's purpose
        """
        cls._registered_types[channel_type] = description

    @classmethod
    def is_registered(cls, channel_type: str) -> bool:
        """Check if a channel type is registered."""
        return channel_type in cls._registered_types

    @classmethod
    def get_all(cls) -> Dict[str, str]:
        """Get all registered channel types."""
        return cls._registered_types.copy()


class ChannelManager:
    """Channel name management for agent communication.

    This class generates channel names following a consistent naming convention.
    It's independent of the broker implementation.

    Core channel naming convention:
    - env_{env_id}__action__{upstream_id}_to_{node_id}
    - env_{env_id}__info__{node_id}_to_{upstream_id}
    - env_{env_id}__broadcast__{agent_id}
    - env_{env_id}__state_updates
    - env_{env_id}__results__{agent_id}

    Custom channels (domain-specific):
    - env_{env_id}__{channel_type}__{agent_id}

    Domains can use custom_channel() for domain-specific communication patterns.
    """

    @staticmethod
    def action_channel(upstream_id: str, node_id: str, env_id: str = "default") -> str:
        """Generate action channel name for parent->child communication.

        Args:
            upstream_id: Parent agent ID
            node_id: Child agent ID
            env_id: Environment ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__action__{upstream_id}_to_{node_id}"

    @staticmethod
    def info_channel(node_id: str, upstream_id: str, env_id: str = "default") -> str:
        """Generate info channel name for child->parent communication.

        Args:
            node_id: Child agent ID
            upstream_id: Parent agent ID
            env_id: Environment ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__info__{node_id}_to_{upstream_id}"

    @staticmethod
    def broadcast_channel(agent_id: str, env_id: str = "default") -> str:
        """Generate broadcast channel name for agent broadcasts.

        Args:
            agent_id: Broadcasting agent ID
            env_id: Environment ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__broadcast__{agent_id}"

    @staticmethod
    def state_update_channel(env_id: str = "default") -> str:
        """Generate state update channel for agent->environment state updates.

        Args:
            env_id: Environment ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__state_updates"

    @staticmethod
    def result_channel(env_id: str, agent_id: str) -> str:
        """Generate result channel for environment->agent results.

        Args:
            env_id: Environment ID
            agent_id: Agent ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__results__{agent_id}"

    @staticmethod
    def custom_channel(channel_type: str, env_id: str, agent_id: str) -> str:
        """Generate a custom channel for domain-specific communication.

        This is a generic method for domains to create their own channel types.
        The channel_type should be registered via ChannelRegistry for documentation.

        Args:
            channel_type: Domain-specific channel type identifier
            env_id: Environment ID
            agent_id: Agent ID

        Returns:
            Channel name string
        """
        return f"env_{env_id}__{channel_type}__{agent_id}"

    @staticmethod
    def agent_channels(
        agent_id: str,
        upstream_id: Optional[str],
        subordinate_ids: List[str],
        env_id: str = "default"
    ) -> Dict[str, List[str]]:
        """Get all channels for an agent (subscribe and publish).

        Args:
            agent_id: The agent's ID
            upstream_id: Parent agent ID (if any)
            subordinate_ids: List of subordinate agent IDs
            env_id: Environment identifier

        Returns:
            Dict with 'subscribe' and 'publish' channel lists
        """
        subscribe_channels = []
        publish_channels = []

        # Subscribe to actions from parent
        if upstream_id:
            subscribe_channels.append(
                ChannelManager.action_channel(upstream_id, agent_id, env_id)
            )

        # Subscribe to info from subordinates
        for sub_id in subordinate_ids:
            subscribe_channels.append(
                ChannelManager.info_channel(sub_id, agent_id, env_id)
            )

        # Publish info to parent
        if upstream_id:
            publish_channels.append(
                ChannelManager.info_channel(agent_id, upstream_id, env_id)
            )

        # Publish actions to subordinates
        for sub_id in subordinate_ids:
            publish_channels.append(
                ChannelManager.action_channel(agent_id, sub_id, env_id)
            )

        return {
            'subscribe': subscribe_channels,
            'publish': publish_channels
        }
