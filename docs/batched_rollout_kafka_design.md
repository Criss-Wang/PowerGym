# Batched Rollout Kafka Design: Environment Isolation

## Problem Statement

When running batched rollouts for training (e.g., 128 parallel environments), a single shared Kafka broker could mix messages between different environment instances. We need to ensure that:

1. Messages sent by Agent A in Environment 1 only reach Agent A in Environment 1
2. Messages from Environment 2 don't leak into Environment 1
3. Each rollout maintains complete isolation while sharing the broker infrastructure

## Solution: Environment ID Namespacing

### Approach 1: Per-Environment Broker (Simple but Wasteful)

```python
class VectorizedEnv:
    def __init__(self, num_envs=128):
        # Create separate broker for each environment
        self.brokers = [KafkaBroker() for _ in range(num_envs)]
        self.envs = [
            KafkaMultiAgentEnv(agents, broker=self.brokers[i])
            for i in range(num_envs)
        ]
```

**Pros**: Complete isolation, simple implementation
**Cons**: Memory overhead (128 broker instances), no message monitoring across envs

### Approach 2: Single Broker with Environment ID (Recommended)

Add `env_id` to all messages and topics to namespace by environment.

## Updated Implementation

### 1. Updated Message Structure

```python
@dataclass
class KafkaMessage:
    """Message structure with environment ID for batched rollouts."""
    env_id: str              # NEW: Environment/rollout identifier
    sender_id: str
    recipient_id: str
    timestamp: float
    message_type: MessageType
    payload: Dict[str, Any]
```

### 2. Updated Topic Naming

```python
class TopicManager:
    """Topic management with environment namespacing."""

    @staticmethod
    def action_topic(parent_id: str, child_id: str, env_id: str = "default") -> str:
        """Generate action topic name with environment namespace."""
        return f"env_{env_id}__{parent_id}_to_{child_id}_action"

    @staticmethod
    def info_topic(child_id: str, parent_id: str, env_id: str = "default") -> str:
        """Generate info topic name with environment namespace."""
        return f"env_{env_id}__{child_id}_to_{parent_id}_info"

    @staticmethod
    def broadcast_topic(agent_id: str, env_id: str = "default") -> str:
        """Generate broadcast topic name with environment namespace."""
        return f"env_{env_id}__{agent_id}_broadcast"

    @staticmethod
    def agent_topics(
        agent_id: str,
        parent_id: Optional[str],
        subordinate_ids: List[str],
        env_id: str = "default"
    ) -> Dict[str, List[str]]:
        """Get all topics for an agent in a specific environment."""
        subscribe_topics = []
        publish_topics = []

        # Subscribe to action from parent
        if parent_id:
            subscribe_topics.append(
                TopicManager.action_topic(parent_id, agent_id, env_id)
            )

        # Subscribe to info from subordinates
        for sub_id in subordinate_ids:
            subscribe_topics.append(
                TopicManager.info_topic(sub_id, agent_id, env_id)
            )

        # Publish info to parent
        if parent_id:
            publish_topics.append(
                TopicManager.info_topic(agent_id, parent_id, env_id)
            )

        # Publish actions to subordinates
        for sub_id in subordinate_ids:
            publish_topics.append(
                TopicManager.action_topic(agent_id, sub_id, env_id)
            )

        return {
            'subscribe': subscribe_topics,
            'publish': publish_topics
        }
```

### 3. Updated Kafka Broker

```python
class KafkaBroker:
    """Kafka broker with environment isolation support."""

    def __init__(self, mode: str = "memory"):
        self.mode = mode
        self.topics: Dict[str, List[KafkaMessage]] = {}
        self.subscribers: Dict[str, List[Callable]] = {}

    def publish(self, topic: str, message: KafkaMessage) -> None:
        """Publish message to topic.

        Topic name already includes env_id, so messages are automatically isolated.
        """
        if topic not in self.topics:
            self.create_topic(topic)
        self.topics[topic].append(message)

        # Notify subscribers
        for callback in self.subscribers[topic]:
            callback(message)

    def consume(
        self,
        topic: str,
        agent_id: str,
        env_id: str,
        clear: bool = True
    ) -> List[KafkaMessage]:
        """Consume messages for a specific agent in a specific environment."""
        if topic not in self.topics:
            return []

        # Filter messages for this agent AND environment
        messages = [
            msg for msg in self.topics[topic]
            if msg.recipient_id == agent_id and msg.env_id == env_id
        ]

        if clear:
            # Remove consumed messages
            self.topics[topic] = [
                msg for msg in self.topics[topic]
                if not (msg.recipient_id == agent_id and msg.env_id == env_id)
            ]

        return messages

    def clear_environment(self, env_id: str) -> None:
        """Clear all messages for a specific environment (useful for reset)."""
        for topic in self.topics:
            if topic.startswith(f"env_{env_id}__"):
                self.topics[topic] = []

    def get_environment_topics(self, env_id: str) -> List[str]:
        """Get all topics for a specific environment."""
        return [
            topic for topic in self.topics
            if topic.startswith(f"env_{env_id}__")
        ]
```

### 4. Updated Agent Base Class

```python
class Agent(ABC):
    """Agent with environment ID for batched rollouts."""

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
        env_id: str = "default",  # NEW: Environment identifier
    ):
        self.agent_id = agent_id
        self.level = level
        self.observation_space = observation_space
        self.action_space = action_space
        self.kafka_broker = kafka_broker
        self.parent_id = parent_id
        self.subordinates = subordinates or {}
        self.execution_mode = execution_mode
        self.env_id = env_id  # NEW

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
        """Create and subscribe to topics with environment namespace."""
        subordinate_ids = list(self.subordinates.keys())
        topics = TopicManager.agent_topics(
            self.agent_id,
            self.parent_id,
            subordinate_ids,
            self.env_id  # Pass environment ID
        )

        # Subscribe to topics
        for topic in topics['subscribe']:
            self.kafka_broker.create_topic(topic)

        # Create publish topics
        for topic in topics['publish']:
            self.kafka_broker.create_topic(topic)

    def _receive_action_from_parent(self) -> Optional[Any]:
        """Receive action from parent in this environment."""
        if not self.parent_id or not self.kafka_broker:
            return None

        topic = TopicManager.action_topic(
            self.parent_id,
            self.agent_id,
            self.env_id  # Use environment ID
        )
        messages = self.kafka_broker.consume(
            topic,
            self.agent_id,
            self.env_id  # Filter by environment ID
        )

        if messages:
            latest_msg = messages[-1]
            return latest_msg.payload.get('action')

        return None

    def _send_actions_to_subordinates(self, actions: Dict[AgentID, Any]) -> None:
        """Send actions to subordinates in this environment."""
        if not self.kafka_broker:
            return

        for sub_id, action in actions.items():
            topic = TopicManager.action_topic(
                self.agent_id,
                sub_id,
                self.env_id  # Use environment ID
            )
            message = KafkaMessage(
                env_id=self.env_id,  # Include environment ID
                sender_id=self.agent_id,
                recipient_id=sub_id,
                timestamp=self._timestep,
                message_type=MessageType.ACTION,
                payload={'action': action}
            )
            self.kafka_broker.publish(topic, message)

    def _send_info_to_parent(self, compiled_info: Dict[str, Any]) -> None:
        """Send compiled info to parent in this environment."""
        if not self.parent_id or not self.kafka_broker:
            return

        topic = TopicManager.info_topic(
            self.agent_id,
            self.parent_id,
            self.env_id  # Use environment ID
        )
        message = KafkaMessage(
            env_id=self.env_id,  # Include environment ID
            sender_id=self.agent_id,
            recipient_id=self.parent_id,
            timestamp=self._timestep,
            message_type=MessageType.INFO,
            payload=compiled_info
        )
        self.kafka_broker.publish(topic, message)

    def set_env_id(self, env_id: str) -> None:
        """Set environment ID (useful for env reuse in vectorized envs).

        This allows the same agent instance to be reused across different
        environment instances by changing its env_id.
        """
        old_env_id = self.env_id
        self.env_id = env_id

        # Update subordinates
        for subordinate in self.subordinates.values():
            subordinate.set_env_id(env_id)

        # Re-setup topics with new env_id
        if self.kafka_broker:
            self._setup_kafka_topics()
```

### 5. Vectorized Environment with Shared Broker

```python
class VectorizedKafkaEnv:
    """Vectorized environment with shared Kafka broker and environment isolation."""

    def __init__(
        self,
        agent_factory: Callable[[str], Dict[AgentID, Agent]],
        root_agent_ids: List[AgentID],
        num_envs: int = 128,
        kafka_mode: str = "memory"
    ):
        """Initialize vectorized environment with shared broker.

        Args:
            agent_factory: Function that creates agents for one environment
            root_agent_ids: IDs of root agents
            num_envs: Number of parallel environments
            kafka_mode: 'memory' or 'kafka'
        """
        self.num_envs = num_envs
        self.root_agent_ids = root_agent_ids

        # Create single shared Kafka broker
        self.kafka_broker = KafkaBroker(mode=kafka_mode)

        # Create environments with unique env_ids
        self.envs = []
        for i in range(num_envs):
            env_id = f"rollout_{i}"

            # Create agents for this environment
            agents = agent_factory(env_id)

            # Inject shared broker and env_id into all agents
            self._inject_broker_and_env_id(agents, self.kafka_broker, env_id)

            # Create environment
            env = KafkaMultiAgentEnv(
                agents=agents,
                root_agents=root_agent_ids,
                kafka_broker=self.kafka_broker,
                env_id=env_id
            )
            self.envs.append(env)

    def _inject_broker_and_env_id(
        self,
        agents: Dict[AgentID, Agent],
        broker: KafkaBroker,
        env_id: str
    ) -> None:
        """Recursively inject broker and env_id into all agents."""
        for agent in agents.values():
            agent.kafka_broker = broker
            agent.set_env_id(env_id)

    def step(self, actions: List[Dict[AgentID, Any]]) -> tuple:
        """Execute parallel steps across all environments.

        Args:
            actions: List of action dicts, one per environment

        Returns:
            observations, rewards, dones, infos (all batched)
        """
        assert len(actions) == self.num_envs

        # Execute steps in parallel (each env handles its own namespace)
        results = [
            self.envs[i].step(actions[i])
            for i in range(self.num_envs)
        ]

        # Batch results
        obs_batch = [r[0] for r in results]
        reward_batch = [r[1] for r in results]
        done_batch = [r[2] for r in results]
        info_batch = [r[3] for r in results]

        return obs_batch, reward_batch, done_batch, info_batch

    def reset(self, env_idx: Optional[List[int]] = None) -> List[Dict]:
        """Reset specified environments (or all if None)."""
        if env_idx is None:
            env_idx = list(range(self.num_envs))

        obs_batch = []
        for i in env_idx:
            # Clear Kafka messages for this environment
            self.kafka_broker.clear_environment(self.envs[i].env_id)

            # Reset environment
            obs = self.envs[i].reset()
            obs_batch.append(obs)

        return obs_batch
```

### 6. Updated Environment with env_id

```python
class KafkaMultiAgentEnv(gym.Env):
    """Multi-agent environment with Kafka communication and environment ID."""

    def __init__(
        self,
        agents: Dict[AgentID, Agent],
        root_agents: List[AgentID],
        kafka_broker: KafkaBroker,
        env_id: str = "default"
    ):
        """Initialize environment.

        Args:
            agents: All agents in the system
            root_agents: IDs of root-level agents
            kafka_broker: Shared Kafka broker instance
            env_id: Unique identifier for this environment instance
        """
        self.agents = agents
        self.root_agents = root_agents
        self.kafka_broker = kafka_broker
        self.env_id = env_id
        self.timestep = 0

    def step(self, actions: Dict[AgentID, Any]) -> tuple:
        """Execute environment step.

        All messages are automatically namespaced by env_id in the agents.
        """
        # Send actions to root agents via Kafka
        for agent_id, action in actions.items():
            if agent_id in self.root_agents:
                topic = TopicManager.action_topic(
                    "environment",
                    agent_id,
                    self.env_id  # Use environment ID
                )
                self.kafka_broker.publish(topic, KafkaMessage(
                    env_id=self.env_id,  # Include environment ID
                    sender_id="environment",
                    recipient_id=agent_id,
                    timestamp=self.timestep,
                    message_type=MessageType.ACTION,
                    payload={'action': action}
                ))

        # Execute root agent steps (recursive, all namespaced by env_id)
        all_info = {}
        for agent_id in self.root_agents:
            agent = self.agents[agent_id]
            compiled_info = agent.step(global_state=self.get_global_state())
            all_info[agent_id] = compiled_info

        # Extract results
        observations = self._extract_observations(all_info)
        rewards = self._extract_rewards(all_info)
        dones = {agent_id: False for agent_id in self.root_agents}
        infos = all_info

        self.timestep += 1
        return observations, rewards, dones, infos

    def reset(self) -> Dict[AgentID, Any]:
        """Reset environment."""
        self.timestep = 0

        # Clear all messages for this environment
        self.kafka_broker.clear_environment(self.env_id)

        # Reset all agents
        for agent in self.agents.values():
            agent.reset()

        # Return initial observations
        observations = {}
        for agent_id in self.root_agents:
            obs = self.agents[agent_id].observe(self.get_global_state())
            observations[agent_id] = obs

        return observations
```

## Visualization: Batched Rollout Message Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SHARED KAFKA BROKER                             │
│                                                                     │
│  Topics grouped by environment:                                    │
│                                                                     │
│  env_rollout_0__grid1_to_device1_action                            │
│  env_rollout_0__device1_to_grid1_info                              │
│  env_rollout_0__grid1_to_device2_action                            │
│  env_rollout_0__device2_to_grid1_info                              │
│                                                                     │
│  env_rollout_1__grid1_to_device1_action                            │
│  env_rollout_1__device1_to_grid1_info                              │
│  env_rollout_1__grid1_to_device2_action                            │
│  env_rollout_1__device2_to_grid1_info                              │
│                                                                     │
│  ...                                                               │
│                                                                     │
│  env_rollout_127__grid1_to_device1_action                          │
│  env_rollout_127__device1_to_grid1_info                            │
└─────────────────────────────────────────────────────────────────────┘
         │                                  │
         │                                  │
    ┌────▼─────────┐                  ┌────▼─────────┐
    │ Environment 0│                  │Environment 127│
    │ env_id:      │                  │ env_id:      │
    │ "rollout_0"  │                  │ "rollout_127"│
    │              │                  │              │
    │  Grid1       │                  │  Grid1       │
    │   ├─Device1  │                  │   ├─Device1  │
    │   └─Device2  │                  │   └─Device2  │
    └──────────────┘                  └──────────────┘

Messages with env_id="rollout_0" only affect Environment 0
Messages with env_id="rollout_127" only affect Environment 127
```

## Example Usage

```python
# Define agent factory
def create_agents(env_id: str) -> Dict[AgentID, Agent]:
    """Create agents for one environment instance."""
    # Create devices
    device1 = DeviceAgent(
        agent_id="device1",
        policy=device_policy,
        env_id=env_id  # Pass env_id
    )
    device2 = DeviceAgent(
        agent_id="device2",
        policy=device_policy,
        env_id=env_id  # Pass env_id
    )

    # Create grid
    grid = GridAgent(
        agent_id="grid1",
        devices=[device1, device2],
        policy=grid_policy,
        env_id=env_id  # Pass env_id
    )

    return {
        "grid1": grid,
        "device1": device1,
        "device2": device2,
    }

# Create vectorized environment with 128 parallel rollouts
vec_env = VectorizedKafkaEnv(
    agent_factory=create_agents,
    root_agent_ids=["grid1"],
    num_envs=128,
    kafka_mode="memory"
)

# Training loop
for episode in range(1000):
    obs_batch = vec_env.reset()

    for step in range(episode_length):
        # Compute actions for each environment
        actions_batch = [
            policy.compute_actions(obs_batch[i])
            for i in range(128)
        ]

        # Execute parallel steps (isolated by env_id)
        obs_batch, rewards, dones, infos = vec_env.step(actions_batch)

        # Update policy...
```

## Performance Considerations

### Memory Overhead
- **Topic Count**: `num_envs × num_agents × num_relationships × 2 (action + info)`
  - Example: 128 envs × 10 agents × 5 relationships × 2 = 12,800 topics
  - In-memory: ~100 bytes per topic = ~1.3 MB (negligible)

### Message Filtering
- Each `consume()` call filters by both `recipient_id` and `env_id`
- O(n) filtering where n = messages in topic
- Optimized by clearing consumed messages

### Optimization: Topic-per-Environment
Instead of filtering in consume, use env-specific topics:
```python
# Already doing this via topic naming!
topic = f"env_{env_id}__{parent_id}_to_{child_id}_action"
```

This ensures O(1) topic lookup and no filtering needed.

## Summary

The key insight is to **namespace all topics and messages by environment ID**:

1. **Topic naming**: `env_{env_id}__{agent_communication}`
2. **Message filtering**: Include `env_id` in every message
3. **Shared broker**: Single broker handles all environments efficiently
4. **Complete isolation**: No cross-contamination between rollouts

This design enables:
- ✅ Efficient batched rollouts with shared infrastructure
- ✅ Complete message isolation between environments
- ✅ Easy monitoring (filter by env_id)
- ✅ Scalable to hundreds of parallel environments
- ✅ Clean reset logic (clear by env_id)
