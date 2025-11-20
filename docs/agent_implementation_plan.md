# Message-Based Hierarchical Agent Implementation

## Status: ✅ **IMPLEMENTED** (2025-11-20)

## Overview

PowerGrid 2.0 now supports message-based hierarchical multi-agent control with distributed execution. The architecture uses an abstract MessageBroker interface with an InMemoryBroker implementation, providing a foundation for future Kafka integration.

**Implemented Features**:
- ✅ Message broker abstraction (MessageBroker interface)
- ✅ InMemoryBroker implementation for local/single-process execution
- ✅ Distributed step execution with message passing
- ✅ Environment-agent communication (actions, network state)
- ✅ Device-environment communication (state updates)
- ✅ Centralized vs Distributed execution modes

**See**: [distributed_architecture.md](design/distributed_architecture.md) for full documentation

---

## What Was Implemented

### 1. Message Broker System

**Files**:
- `powergrid/messaging/base.py` - MessageBroker interface, Message dataclass, ChannelManager
- `powergrid/messaging/memory.py` - InMemoryBroker implementation
- `powergrid/messaging/utils.py` - Helper utilities

**Key Classes**:
```python
class MessageBroker(ABC):
    """Abstract message broker interface."""
    def create_channel(channel_name: str)
    def publish(channel: str, message: Message)
    def consume(channel: str, recipient_id: str, env_id: str, clear: bool)
    def subscribe(channel: str, callback: Callable)
    def clear_environment(env_id: str)

class InMemoryBroker(MessageBroker):
    """In-memory implementation using Dict[str, List[Message]]."""

class ChannelManager:
    """Channel naming conventions."""
    @staticmethod
    def action_channel(upstream_id, node_id, env_id)
    @staticmethod
    def state_update_channel(env_id)
    @staticmethod
    def power_flow_result_channel(env_id, agent_id)
```

### 2. Agent Distributed Execution

**Updated Files**:
- `powergrid/agents/base.py` - Added `step_distributed()` method
- `powergrid/agents/grid_agent.py` - Implemented action decomposition, network state consumption
- `powergrid/agents/device_agent.py` - Device action execution

**Key Methods**:
```python
class Agent:
    def step_distributed(self):
        """Execute one step with message-based communication."""
        # 1. Receive action from upstream
        # 2. Derive and send actions to subordinates
        # 3. Execute subordinate steps recursively
        # 4. Execute own action
        # 5. Publish state updates
```

### 3. Environment-Side Communication

**Updated Files**:
- `powergrid/envs/multi_agent/networked_grid_env.py` - Message-based environment

**Key Methods**:
```python
class NetworkedGridEnv:
    def _send_actions_to_agent(agent_id, action):
        """Send action via message broker."""

    def _consume_all_state_updates():
        """Consume device state updates."""

    def _apply_state_updates_to_net(updates):
        """Apply updates to PandaPower network."""

    def _publish_network_state_to_agents():
        """Send power flow results to agents."""
```

### 4. Device State Publishing

**Updated Files**:
- `powergrid/devices/generator.py` - State update publishing

**Key Methods**:
```python
class Generator(DeviceAgent):
    def _publish_state_updates(self):
        """Publish P, Q, status to environment."""
        message = Message(
            payload={
                'agent_id': self.agent_id,
                'device_type': 'sgen',
                'P_MW': self.electrical.P_MW,
                'Q_MVAr': self.electrical.Q_MVAr,
                'in_service': self.status.in_service
            }
        )
        self.message_broker.publish(channel, message)
```

### 5. Execution Modes

**Centralized Mode** (`centralized: true`):
- Agents directly access PandaPower network
- No message broker required
- Traditional multi-agent RL setup

**Distributed Mode** (`centralized: false`):
- Agents communicate only via message broker
- Environment has exclusive access to network
- Realistic distributed control simulation

---

## Remaining Work for Kafka Integration

The current implementation provides the foundation for Kafka integration. To add Kafka support:

### To Implement:

1. **KafkaBroker class** (`powergrid/messaging/kafka.py`):
   ```python
   class KafkaBroker(MessageBroker):
       """Kafka-based message broker."""
       def __init__(self, bootstrap_servers: List[str]):
           self.producer = KafkaProducer(...)
           self.consumer = KafkaConsumer(...)

       def publish(self, channel: str, message: Message):
           self.producer.send(channel, message.to_json())

       def consume(self, channel: str, ...):
           records = self.consumer.poll(...)
           return [Message.from_json(r) for r in records]
   ```

2. **Configuration**:
   ```yaml
   centralized: false
   message_broker: 'kafka'
   kafka_config:
     bootstrap_servers: ['localhost:9092']
     topic_prefix: 'powergrid'
   ```

3. **Deployment Changes**:
   - Run agents in separate processes/containers
   - Connect all to same Kafka cluster
   - Environment orchestrates via Kafka topics

### Benefits of Kafka (Future):
- True distributed training across multiple machines
- Scalability to hundreds of agents
- Message persistence and replay
- Production-ready deployment

---

## Previous Design (Historical)
   - Simple mailbox-based messaging
   - Abstract `observe()` and `act()` methods
   - No explicit hierarchy management
   - No async execution support

2. **DeviceAgent** ([powergrid/agents/device_agent.py](../powergrid/agents/device_agent.py))
   - Wraps individual devices (DG, ESS, RES)
   - Level 1 in hierarchy
   - Direct device state management

3. **GridAgent** ([powergrid/agents/grid_agent.py](../powergrid/agents/grid_agent.py))
   - Coordinates multiple DeviceAgents
   - Level 2 in hierarchy
   - Uses Protocol for coordination
   - Two variants: PowerGridAgent and PowerGridAgentV2

4. **Protocols** ([powergrid/core/protocols.py](../powergrid/core/protocols.py))
   - Vertical protocols (parent → child): NoProtocol, PriceSignal, Setpoint
   - Horizontal protocols (peer ↔ peer): P2P Trading, Consensus
   - Currently uses direct method calls

### Limitations

- **Synchronous execution**: Agents call subordinates directly
- **Direct coupling**: Parent agents directly access subordinate methods
- **No true async support**: Cannot run subordinate steps in parallel
- **Limited observability**: No centralized message broker for monitoring
- **Scalability**: Hard to distribute agents across processes/machines

## Proposed Architecture

### Key Design Principles

1. **Kafka-centric communication**: All inter-agent messages flow through Kafka
2. **Recursive step execution**: Each agent follows the same step flow regardless of level
3. **Async/sync flexibility**: Support both synchronous and asynchronous subordinate execution
4. **Decoupled agents**: Agents only communicate via Kafka topics
5. **Information aggregation**: Each agent builds compiled info from self + subordinates

### Component Updates

## 1. Kafka Broker Infrastructure

### File: `powergrid/messaging/kafka_broker.py`

```python
"""Kafka-based message broker for agent communication."""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
import asyncio
from enum import Enum

class MessageType(Enum):
    ACTION = "action"
    INFO = "info"
    BROADCAST = "broadcast"

@dataclass
class KafkaMessage:
    """Message structure for Kafka topics."""
    sender_id: str
    recipient_id: str
    timestamp: float
    message_type: MessageType
    payload: Dict[str, Any]

class KafkaBroker:
    """In-memory Kafka broker simulation for agent messaging.

    For production, this would interface with actual Kafka (kafka-python).
    For simulation/testing, we implement a simple in-memory message queue.
    """

    def __init__(self, mode: str = "memory"):
        """Initialize broker.

        Args:
            mode: 'memory' for in-memory or 'kafka' for actual Kafka
        """
        self.mode = mode
        self.topics: Dict[str, List[KafkaMessage]] = {}
        self.subscribers: Dict[str, List[Callable]] = {}

    def create_topic(self, topic_name: str) -> None:
        """Create a new topic."""
        if topic_name not in self.topics:
            self.topics[topic_name] = []
            self.subscribers[topic_name] = []

    def publish(self, topic: str, message: KafkaMessage) -> None:
        """Publish message to topic."""
        if topic not in self.topics:
            self.create_topic(topic)
        self.topics[topic].append(message)

        # Notify subscribers
        for callback in self.subscribers[topic]:
            callback(message)

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to topic with callback."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def consume(self, topic: str, agent_id: str, clear: bool = True) -> List[KafkaMessage]:
        """Consume messages for a specific agent from topic."""
        if topic not in self.topics:
            return []

        # Filter messages for this agent
        messages = [
            msg for msg in self.topics[topic]
            if msg.recipient_id == agent_id
        ]

        if clear:
            # Remove consumed messages
            self.topics[topic] = [
                msg for msg in self.topics[topic]
                if msg.recipient_id != agent_id
            ]

        return messages

    def compile_messages(self, topics: List[str], agent_id: str) -> Dict[str, List[KafkaMessage]]:
        """Compile all messages sent to agent from multiple topics."""
        compiled = {}
        for topic in topics:
            messages = self.consume(topic, agent_id)
            if messages:
                compiled[topic] = messages
        return compiled

    def clear_topic(self, topic: str) -> None:
        """Clear all messages from topic."""
        if topic in self.topics:
            self.topics[topic] = []
```

### File: `powergrid/messaging/topics.py`

```python
"""Kafka topic naming conventions and management."""

from typing import List

class TopicManager:
    """Manages Kafka topic naming conventions."""

    @staticmethod
    def action_topic(parent_id: str, child_id: str) -> str:
        """Generate action topic name for parent → child."""
        return f"{parent_id}_to_{child_id}_action"

    @staticmethod
    def info_topic(child_id: str, parent_id: str) -> str:
        """Generate info topic name for child → parent."""
        return f"{child_id}_to_{parent_id}_info"

    @staticmethod
    def broadcast_topic(agent_id: str) -> str:
        """Generate broadcast topic name for agent."""
        return f"{agent_id}_broadcast"

    @staticmethod
    def agent_topics(agent_id: str, parent_id: Optional[str], subordinate_ids: List[str]) -> Dict[str, List[str]]:
        """Get all topics relevant to an agent.

        Returns:
            {
                'subscribe': [list of topics to subscribe to],
                'publish': [list of topics to publish to]
            }
        """
        subscribe_topics = []
        publish_topics = []

        # Subscribe to action from parent
        if parent_id:
            subscribe_topics.append(TopicManager.action_topic(parent_id, agent_id))

        # Subscribe to info from subordinates
        for sub_id in subordinate_ids:
            subscribe_topics.append(TopicManager.info_topic(sub_id, agent_id))

        # Publish info to parent
        if parent_id:
            publish_topics.append(TopicManager.info_topic(agent_id, parent_id))

        # Publish actions to subordinates
        for sub_id in subordinate_ids:
            publish_topics.append(TopicManager.action_topic(agent_id, sub_id))

        return {
            'subscribe': subscribe_topics,
            'publish': publish_topics
        }
```

## 2. Updated Base Agent Class

### File: `powergrid/agents/base.py` (Major Update)

Key changes:
1. Add `subordinates` and `parent_id` attributes
2. Add `kafka_broker` reference
3. Implement `step()` method with recursive execution
4. Add methods for Kafka communication
5. Add action derivation capabilities

```python
"""Updated base agent with Kafka support and recursive step execution."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import asyncio
import gymnasium as gym

from powergrid.core.observation import AgentID, Message, Observation
from powergrid.messaging.kafka_broker import KafkaBroker, KafkaMessage, MessageType
from powergrid.messaging.topics import TopicManager

class ExecutionMode(Enum):
    """Execution mode for subordinate coordination."""
    SYNC = "sync"
    ASYNC = "async"

class Agent(ABC):
    """Abstract base class for Kafka-enabled hierarchical agents."""

    def __init__(
        self,
        agent_id: AgentID,
        level: int = 1,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        kafka_broker: Optional[KafkaBroker] = None,
        parent_id: Optional[AgentID] = None,
        subordinates: Optional[Dict[AgentID, 'Agent']] = None,
        execution_mode: ExecutionMode = ExecutionMode.ASYNC,
    ):
        """Initialize Kafka-enabled agent.

        Args:
            agent_id: Unique identifier
            level: Hierarchy level
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            kafka_broker: Shared Kafka broker instance
            parent_id: Parent agent ID (None for root)
            subordinates: Dictionary of subordinate agents
            execution_mode: How to execute subordinates (sync/async)
        """
        self.agent_id = agent_id
        self.level = level
        self.observation_space = observation_space
        self.action_space = action_space
        self.kafka_broker = kafka_broker
        self.parent_id = parent_id
        self.subordinates = subordinates or {}
        self.execution_mode = execution_mode

        # Internal state
        self.mailbox: List[Message] = []
        self._timestep = 0.0
        self._compiled_info: Dict[str, Any] = {}
        self._last_observation: Optional[Observation] = None
        self._last_action: Optional[Any] = None

        # Setup Kafka topics if broker available
        if self.kafka_broker:
            self._setup_kafka_topics()

    def _setup_kafka_topics(self) -> None:
        """Create and subscribe to relevant Kafka topics."""
        subordinate_ids = list(self.subordinates.keys())
        topics = TopicManager.agent_topics(
            self.agent_id,
            self.parent_id,
            subordinate_ids
        )

        # Subscribe to topics
        for topic in topics['subscribe']:
            self.kafka_broker.create_topic(topic)

        # Create publish topics
        for topic in topics['publish']:
            self.kafka_broker.create_topic(topic)

    # ========================================================================
    # RECURSIVE STEP EXECUTION (Core of new design)
    # ========================================================================

    def step(self, global_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute one step of agent behavior with recursive subordinate execution.

        This implements the recursive step flow from the design:
        1. Get action from Kafka (sent by parent)
        2. Compile messages from Kafka
        3. Identify parent agent
        4. If has subordinates:
           - Derive subordinate actions
           - Send actions to subordinates via Kafka
           - Execute subordinate steps (async/sync)
           - Collect info from subordinates via Kafka
        5. Derive own action
        6. Execute own action
        7. Collate partial observations and update state
        8. Build compiled info
        9. Send info to parent via Kafka

        Args:
            global_state: Global environment state

        Returns:
            Compiled information including own state and subordinate info
        """
        # Step 1: Get action from parent via Kafka
        parent_action = self._receive_action_from_parent()

        # Step 2 & 3: Compile messages and identify parent
        self._compile_kafka_messages()

        # Step 4: Handle subordinates (if any)
        subordinate_info = {}
        if self.subordinates:
            # Derive actions for subordinates
            subordinate_actions = self.derive_subordinate_actions(
                parent_action=parent_action,
                global_state=global_state
            )

            # Send actions to subordinates via Kafka
            self._send_actions_to_subordinates(subordinate_actions)

            # Execute subordinate steps (recursive call)
            subordinate_info = self._execute_subordinate_steps(
                global_state=global_state
            )

        # Step 5: Derive own action from parent action
        own_action = self.derive_own_action(
            parent_action=parent_action,
            subordinate_info=subordinate_info,
            global_state=global_state
        )

        # Step 6: Execute own action
        self.execute_own_action(own_action, global_state)

        # Step 7: Extract observations and update state
        observation = self.observe(global_state)
        self._last_observation = observation
        self._last_action = own_action

        # Step 8: Build compiled information
        compiled_info = self.build_compiled_info(
            observation=observation,
            subordinate_info=subordinate_info,
            global_state=global_state
        )
        self._compiled_info = compiled_info

        # Step 9: Send info to parent via Kafka
        if self.parent_id:
            self._send_info_to_parent(compiled_info)

        return compiled_info

    def _receive_action_from_parent(self) -> Optional[Any]:
        """Receive action from parent via Kafka."""
        if not self.parent_id or not self.kafka_broker:
            return None

        topic = TopicManager.action_topic(self.parent_id, self.agent_id)
        messages = self.kafka_broker.consume(topic, self.agent_id)

        if messages:
            # Get most recent action
            latest_msg = messages[-1]
            return latest_msg.payload.get('action')

        return None

    def _compile_kafka_messages(self) -> None:
        """Compile all messages sent to this agent via Kafka."""
        if not self.kafka_broker:
            return

        # Collect from all relevant topics
        subordinate_ids = list(self.subordinates.keys())
        topics = []

        # Info topics from subordinates
        for sub_id in subordinate_ids:
            topics.append(TopicManager.info_topic(sub_id, self.agent_id))

        # Broadcast topics (if implemented)
        # topics.append(TopicManager.broadcast_topic(self.agent_id))

        compiled = self.kafka_broker.compile_messages(topics, self.agent_id)

        # Store in mailbox for compatibility
        for topic, messages in compiled.items():
            for kafka_msg in messages:
                # Convert to legacy Message format
                msg = Message(
                    sender=kafka_msg.sender_id,
                    content=kafka_msg.payload,
                    recipient=kafka_msg.recipient_id,
                    timestamp=kafka_msg.timestamp
                )
                self.mailbox.append(msg)

    def _send_actions_to_subordinates(self, actions: Dict[AgentID, Any]) -> None:
        """Send actions to subordinates via Kafka."""
        if not self.kafka_broker:
            return

        for sub_id, action in actions.items():
            topic = TopicManager.action_topic(self.agent_id, sub_id)
            message = KafkaMessage(
                sender_id=self.agent_id,
                recipient_id=sub_id,
                timestamp=self._timestep,
                message_type=MessageType.ACTION,
                payload={'action': action}
            )
            self.kafka_broker.publish(topic, message)

    def _execute_subordinate_steps(
        self,
        global_state: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Execute subordinate step() methods and collect results.

        Args:
            global_state: Global environment state

        Returns:
            Dictionary mapping subordinate IDs to their compiled info
        """
        if self.execution_mode == ExecutionMode.SYNC:
            return self._execute_subordinates_sync(global_state)
        else:
            return self._execute_subordinates_async(global_state)

    def _execute_subordinates_sync(
        self,
        global_state: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Execute subordinates synchronously."""
        results = {}
        for sub_id, subordinate in self.subordinates.items():
            # Recursive call to subordinate's step()
            compiled_info = subordinate.step(global_state)
            results[sub_id] = compiled_info
        return results

    def _execute_subordinates_async(
        self,
        global_state: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Dict[str, Any]]:
        """Execute subordinates asynchronously."""
        # For now, implement as sync (async requires event loop)
        # TODO: Implement true async with asyncio
        return self._execute_subordinates_sync(global_state)

    def _send_info_to_parent(self, compiled_info: Dict[str, Any]) -> None:
        """Send compiled info to parent via Kafka."""
        if not self.parent_id or not self.kafka_broker:
            return

        topic = TopicManager.info_topic(self.agent_id, self.parent_id)
        message = KafkaMessage(
            sender_id=self.agent_id,
            recipient_id=self.parent_id,
            timestamp=self._timestep,
            message_type=MessageType.INFO,
            payload=compiled_info
        )
        self.kafka_broker.publish(topic, message)

    # ========================================================================
    # ACTION DERIVATION (Option 1: Model-based, Option 2: Manual)
    # ========================================================================

    def derive_subordinate_actions(
        self,
        parent_action: Optional[Any],
        global_state: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        """Derive actions for subordinates from parent action.

        This can be implemented in two ways:
        - Option 1 (Model-based): Use a model to decompose parent action
        - Option 2 (Manual): Manually split/assign actions

        Subclasses should override this method with specific logic.

        Args:
            parent_action: Action received from parent
            global_state: Global environment state

        Returns:
            Dictionary mapping subordinate IDs to their derived actions
        """
        # Default: return empty actions (subordinates act independently)
        return {sub_id: None for sub_id in self.subordinates}

    def derive_own_action(
        self,
        parent_action: Optional[Any],
        subordinate_info: Dict[AgentID, Dict[str, Any]],
        global_state: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Derive this agent's own action.

        Args:
            parent_action: Action received from parent
            subordinate_info: Compiled info from subordinates
            global_state: Global environment state

        Returns:
            Agent's own action
        """
        # Default: use parent action or compute from observation
        if parent_action is not None:
            return parent_action

        # Fall back to observe() and act()
        observation = self.observe(global_state)
        return self.act(observation)

    def execute_own_action(
        self,
        action: Any,
        global_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Execute this agent's own action.

        Subclasses should implement device-specific execution logic.

        Args:
            action: Action to execute
            global_state: Global environment state
        """
        pass  # No-op by default

    def build_compiled_info(
        self,
        observation: Observation,
        subordinate_info: Dict[AgentID, Dict[str, Any]],
        global_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build compiled information to send to parent.

        Args:
            observation: This agent's observation
            subordinate_info: Info collected from subordinates
            global_state: Global environment state

        Returns:
            Compiled information including own state and subordinate info
        """
        # Extract reward/metrics
        reward_dict = self.get_reward() if hasattr(self, 'get_reward') else {}

        return {
            'agent_id': self.agent_id,
            'observation': observation.local if observation else {},
            'reward': reward_dict.get('cost', 0.0),
            'cost': reward_dict.get('cost', 0.0),
            'safety': reward_dict.get('safety', 0.0),
            'status': 'active',
            'subordinate_info': subordinate_info,
            'timestamp': self._timestep
        }

    # ========================================================================
    # LEGACY METHODS (for backward compatibility)
    # ========================================================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> None:
        """Reset agent to initial state."""
        self.mailbox.clear()
        self._timestep = 0.0
        self._compiled_info = {}
        self._last_observation = None
        self._last_action = None

        # Reset subordinates
        for subordinate in self.subordinates.values():
            subordinate.reset(seed=seed, **kwargs)

    @abstractmethod
    def observe(self, global_state: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Observation:
        """Extract relevant observations from global state."""
        pass

    @abstractmethod
    def act(self, observation: Observation, *args, **kwargs) -> Any:
        """Compute action from observation."""
        pass

    # Communication methods (legacy, now supplemented by Kafka)
    def send_message(
        self,
        content: Dict[str, Any],
        recipients: Optional[Union[AgentID, List[AgentID]]] = None,
    ) -> Message:
        """Create a message to send to other agents (legacy)."""
        return Message(
            sender=self.agent_id,
            content=content,
            recipient=recipients,
            timestamp=self._timestep,
        )

    def receive_message(self, message: Message) -> None:
        """Handle incoming communication (legacy)."""
        self.mailbox.append(message)

    def clear_mailbox(self) -> List[Message]:
        """Clear and return all messages from mailbox."""
        messages = self.mailbox.copy()
        self.mailbox.clear()
        return messages

    def update_timestep(self, timestep: float) -> None:
        """Update internal timestep counter."""
        self._timestep = timestep

        # Update subordinates
        for subordinate in self.subordinates.values():
            subordinate.update_timestep(timestep)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, level={self.level}, subs={len(self.subordinates)})"
```

## 3. Updated DeviceAgent

Key changes:
- Add Kafka broker to init
- Implement `execute_own_action()` instead of relying on `act()`
- Override `derive_own_action()` to use policy

```python
class DeviceAgent(Agent):
    """Updated DeviceAgent with Kafka support."""

    def __init__(
        self,
        agent_id: Optional[str] = None,
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any] = {},
        kafka_broker: Optional[KafkaBroker] = None,
        parent_id: Optional[AgentID] = None,
    ):
        # ... existing device setup ...

        super().__init__(
            agent_id=agent_id or self.config.name,
            level=DEVICE_LEVEL,
            action_space=self._get_action_space(),
            observation_space=self._get_observation_space(),
            kafka_broker=kafka_broker,
            parent_id=parent_id,
            subordinates={},  # Devices have no subordinates
        )

    def derive_own_action(
        self,
        parent_action: Optional[Any],
        subordinate_info: Dict[AgentID, Dict[str, Any]],
        global_state: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Derive action using policy or use parent's action."""
        if parent_action is not None:
            return parent_action

        # Use policy to compute action
        observation = self.observe(global_state)
        if self.policy:
            return self.policy.forward(observation)

        return None

    def execute_own_action(
        self,
        action: Any,
        global_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Execute action on device."""
        if action is not None:
            self._set_device_action(action)
```

## 4. Updated GridAgent

Key changes:
- Add Kafka broker and subordinate management
- Implement `derive_subordinate_actions()` with protocol
- Support both model-based and protocol-based action derivation

```python
class GridAgent(Agent):
    """Updated GridAgent with Kafka-based coordination."""

    def __init__(
        self,
        agent_id: AgentID,
        devices: List[DeviceAgent] = [],
        protocol: Protocol = NoProtocol(),
        policy: Optional[Policy] = None,
        centralized: bool = True,
        kafka_broker: Optional[KafkaBroker] = None,
        parent_id: Optional[AgentID] = None,
        execution_mode: ExecutionMode = ExecutionMode.ASYNC,
    ):
        subordinates = {agent.agent_id: agent for agent in devices}

        # Set parent reference in subordinates
        for device in devices:
            device.parent_id = agent_id
            device.kafka_broker = kafka_broker

        super().__init__(
            agent_id=agent_id,
            level=GRID_LEVEL,
            kafka_broker=kafka_broker,
            parent_id=parent_id,
            subordinates=subordinates,
            execution_mode=execution_mode,
        )

        self.protocol = protocol
        self.policy = policy
        self.centralized = centralized

    def derive_subordinate_actions(
        self,
        parent_action: Optional[Any],
        global_state: Optional[Dict[str, Any]] = None
    ) -> Dict[AgentID, Any]:
        """Derive device actions using protocol or policy.

        Option 1 (Model-based): Use policy to compute joint action, then split
        Option 2 (Manual/Protocol): Use protocol to derive actions
        """
        if self.centralized and self.policy:
            # Option 1: Model-based derivation
            observation = self.observe(global_state)
            joint_action = self.policy.forward(observation)

            # Split joint action among devices
            return self._split_action_to_devices(joint_action)
        else:
            # Option 2: Protocol-based derivation
            device_observations = {
                sub_id: sub.observe(global_state)
                for sub_id, sub in self.subordinates.items()
            }

            # Use protocol to derive coordination signals
            # This could be prices, setpoints, etc.
            coordination_signals = self.protocol.coordinate(
                device_observations,
                parent_action
            )

            return coordination_signals

    def _split_action_to_devices(self, joint_action: Any) -> Dict[AgentID, Any]:
        """Split joint action vector into per-device actions."""
        import numpy as np
        action = np.asarray(joint_action)

        device_actions = {}
        offset = 0

        for sub_id, subordinate in self.subordinates.items():
            # Get device action size
            action_size = subordinate.action.dim_c + subordinate.action.dim_d
            device_action = action[offset:offset + action_size]
            device_actions[sub_id] = device_action
            offset += action_size

        return device_actions
```

## 5. Environment Integration

### File: `powergrid/envs/kafka_multi_agent_env.py`

```python
"""Kafka-based multi-agent environment."""

import gymnasium as gym
from typing import Dict, Any, List
from powergrid.messaging.kafka_broker import KafkaBroker
from powergrid.agents.base import Agent, AgentID

class KafkaMultiAgentEnv(gym.Env):
    """Multi-agent environment with Kafka-based communication."""

    def __init__(
        self,
        agents: Dict[AgentID, Agent],
        root_agents: List[AgentID],
        kafka_mode: str = "memory"
    ):
        """Initialize Kafka-based environment.

        Args:
            agents: All agents in the system
            root_agents: IDs of root-level agents (no parents)
            kafka_mode: 'memory' or 'kafka'
        """
        self.agents = agents
        self.root_agents = root_agents

        # Create shared Kafka broker
        self.kafka_broker = KafkaBroker(mode=kafka_mode)

        # Inject broker into all agents
        for agent in agents.values():
            agent.kafka_broker = self.kafka_broker
            agent._setup_kafka_topics()

    def step(self, actions: Dict[AgentID, Any]) -> tuple:
        """Execute environment step using recursive agent execution.

        Args:
            actions: Actions for root agents only

        Returns:
            observations, rewards, dones, infos
        """
        # Send actions to root agents via Kafka
        for agent_id, action in actions.items():
            if agent_id in self.root_agents:
                # Publish action to agent
                topic = f"env_to_{agent_id}_action"
                self.kafka_broker.publish(topic, KafkaMessage(
                    sender_id="environment",
                    recipient_id=agent_id,
                    timestamp=self.timestep,
                    message_type=MessageType.ACTION,
                    payload={'action': action}
                ))

        # Execute root agent steps (triggers recursive execution)
        all_info = {}
        for agent_id in self.root_agents:
            agent = self.agents[agent_id]
            compiled_info = agent.step(global_state=self.get_global_state())
            all_info[agent_id] = compiled_info

        # Collect observations, rewards from compiled info
        observations = self._extract_observations(all_info)
        rewards = self._extract_rewards(all_info)
        dones = {agent_id: False for agent_id in self.root_agents}
        infos = all_info

        self.timestep += 1
        return observations, rewards, dones, infos

    def _extract_observations(self, compiled_info: Dict) -> Dict:
        """Extract observations from compiled info."""
        return {
            agent_id: info['observation']
            for agent_id, info in compiled_info.items()
        }

    def _extract_rewards(self, compiled_info: Dict) -> Dict:
        """Extract rewards from compiled info."""
        return {
            agent_id: -info['cost'] - info['safety']
            for agent_id, info in compiled_info.items()
        }
```

## 6. Migration Strategy

### Phase 1: Infrastructure Setup
1. Implement Kafka broker (in-memory mode)
2. Implement topic management
3. Add unit tests for messaging

### Phase 2: Base Agent Updates
1. Update Agent base class with Kafka support
2. Implement `step()` method with recursive execution
3. Add backward compatibility layer
4. Test with simple 2-level hierarchy

### Phase 3: Device/Grid Agent Updates
1. Update DeviceAgent with Kafka
2. Update GridAgent with Kafka
3. Migrate existing protocols to Kafka
4. Test with existing environments

### Phase 4: Environment Integration
1. Create KafkaMultiAgentEnv
2. Migrate existing multi-agent environments
3. Add async execution support
4. Performance benchmarking

### Phase 5: Advanced Features
1. Add model-based action derivation
2. Implement true async execution (asyncio)
3. Add monitoring/visualization tools
4. Add support for real Kafka (optional)

## 7. Backward Compatibility

To maintain compatibility with existing code:

1. **Legacy `act()` method**: Keep existing `act()` method for single-step execution
2. **Direct method calls**: Support both Kafka and direct calls initially
3. **Optional Kafka**: Make Kafka broker optional (fallback to direct calls)
4. **Gradual migration**: Deprecate old methods over multiple versions

Example compatibility layer:

```python
class Agent(ABC):
    def act(self, observation: Observation, *args, **kwargs) -> Any:
        """Legacy act method (for backward compatibility)."""
        # If Kafka is enabled, this is called from step()
        # If not, this is called directly by environment
        if self.kafka_broker:
            warnings.warn("Using legacy act() with Kafka enabled. Use step() instead.")

        return self._compute_action(observation)

    @abstractmethod
    def _compute_action(self, observation: Observation) -> Any:
        """Actual action computation (to be implemented by subclasses)."""
        pass
```

## 8. Testing Strategy

### Unit Tests
- Kafka broker message delivery
- Topic management
- Agent step() execution
- Action derivation

### Integration Tests
- 2-level hierarchy (Grid → Devices)
- 3-level hierarchy (System → Grids → Devices)
- Sync vs async execution
- Protocol coordination via Kafka

### Performance Tests
- Message throughput
- Latency of recursive execution
- Scaling with number of agents
- Memory usage

## 9. Documentation Updates

- Update architecture diagrams
- Add Kafka communication guide
- Update protocol documentation
- Add migration guide for existing code
- Add examples for new agent creation

## 10. Future Extensions

### Real Kafka Integration
- Add kafka-python backend
- Distributed agent execution
- Persistence and replay

### Advanced Coordination
- Multi-level action derivation models
- Learned coordination strategies
- Adaptive execution modes

### Monitoring & Debugging
- Message flow visualization
- Agent interaction graphs
- Performance profiling
- Distributed tracing

## Summary

This implementation plan provides a comprehensive path to update the PowerGrid agent architecture to support Kafka-based hierarchical communication with recursive execution. The design maintains backward compatibility while enabling new capabilities like async execution, distributed agents, and centralized monitoring.

The key innovation is the `step()` method that recursively executes the agent hierarchy, with all communication flowing through Kafka topics. This enables true decoupling of agents and opens the door to distributed multi-agent systems.
