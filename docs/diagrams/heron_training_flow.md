# HERON Training Flow (Option A - Synchronous)

This diagram shows the synchronous execution flow used during training with Centralized Training, Decentralized Execution (CTDE).

```mermaid
sequenceDiagram
    autonumber
    participant Env as MultiAgentEnv
    participant SA as SystemAgent (L3)
    participant CA as CoordinatorAgent (L2)
    participant FA as FieldAgent (L1)
    participant Proto as Protocol
    participant Policy as Policy
    participant State as State
    participant MB as MessageBroker
    participant Proxy as ProxyAgent

    Note over Env,Proxy: === RESET PHASE ===

    Env->>Env: reset(seed, options)
    Env->>SA: reset(seed)
    SA->>CA: reset(seed)
    CA->>FA: reset(seed)
    FA->>State: reset()
    Env->>Proxy: reset()
    Proxy->>Proxy: clear state_cache, state_history

    Note over Env,Proxy: === STEP PHASE (repeated) ===

    rect rgb(240, 248, 255)
        Note over Env,Proxy: 1. OBSERVATION COLLECTION (Bottom-up)

        Env->>Env: step(actions)
        Env->>Proxy: update_proxy_state(env_state)
        Proxy->>Proxy: state_cache = env_state

        Env->>SA: observe(global_state)
        SA->>CA: observe(global_state)

        loop For each FieldAgent
            CA->>FA: observe(global_state)
            FA->>State: vector()
            State-->>FA: state_vector
            FA->>FA: _get_obs()
            FA-->>CA: Observation(local, timestamp)
        end

        CA->>CA: _build_local_observation(subordinate_obs)
        CA-->>SA: Observation(local={subordinate_obs}, global_info)
        SA-->>Env: Observation(coordinator_obs)
    end

    rect rgb(255, 248, 240)
        Note over Env,Proxy: 2. ACTION COMPUTATION (Centralized Policy)

        Env->>Policy: forward(joint_observation)
        Policy-->>Env: joint_action
    end

    rect rgb(240, 255, 240)
        Note over Env,Proxy: 3. ACTION DISTRIBUTION (Top-down via act())

        Env->>SA: act(observation, upstream_action)
        SA->>CA: act(observation, upstream_action)

        CA->>CA: coordinate_subordinates(obs, action)
        CA->>Proto: coordinate(coord_state, sub_states, action, context)

        Proto->>Proto: compute_coordination_messages()
        Proto->>Proto: compute_action_coordination()
        Proto-->>CA: (messages, actions)

        CA->>CA: _apply_coordination(messages, actions)

        opt If MessageBroker enabled
            CA->>MB: send_message(message, recipient_id)
            MB->>MB: publish(channel, msg)
        end

        loop For each FieldAgent
            CA->>FA: act(observation, upstream_action)
            FA->>FA: _handle_coordinator_action(upstream_action)
            FA->>FA: action.set_values(action)
        end
    end

    rect rgb(255, 240, 255)
        Note over Env,Proxy: 4. ENVIRONMENT PHYSICS UPDATE

        Env->>Env: _apply_actions_to_physics()
        Env->>Env: _simulate_physics()
        Env->>Env: _compute_rewards()
        Env-->>Env: (obs, rewards, terminated, truncated, info)
    end
```

## Key Components

### Agent Hierarchy (L1 → L2 → L3)

```mermaid
graph TB
    subgraph "Agent Hierarchy"
        SA[SystemAgent L3]
        CA1[CoordinatorAgent L2]
        CA2[CoordinatorAgent L2]
        FA1[FieldAgent L1]
        FA2[FieldAgent L1]
        FA3[FieldAgent L1]
        FA4[FieldAgent L1]

        SA --> CA1
        SA --> CA2
        CA1 --> FA1
        CA1 --> FA2
        CA2 --> FA3
        CA2 --> FA4
    end

    subgraph "State Flow"
        direction LR
        S1[FieldAgentState] --> S2[CoordinatorAgentState]
        S2 --> S3[SystemState]
    end
```

### Data Structures

```mermaid
classDiagram
    class Observation {
        +Dict local
        +Dict global_info
        +float timestamp
        +vector() np.ndarray
    }

    class Action {
        +int dim_c
        +int dim_d
        +np.ndarray c
        +np.ndarray d
        +set_values(values)
        +vector() np.ndarray
        +Space space
    }

    class State {
        +str owner_id
        +int owner_level
        +List~FeatureProvider~ features
        +vector() np.ndarray
        +observed_by(requestor_id, level) Dict
        +update(**kwargs)
    }

    class Protocol {
        +CommunicationProtocol communication_protocol
        +ActionProtocol action_protocol
        +coordinate() Tuple~messages, actions~
    }

    Observation --> Action : used by Policy
    State --> Observation : builds
    Protocol --> Action : distributes
```

### Message Broker Channels

```mermaid
graph LR
    subgraph "Channel Naming Convention"
        AC[action channel<br/>env_{id}__action__{upstream}_to_{node}]
        IC[info channel<br/>env_{id}__info__{node}_to_{upstream}]
        BC[broadcast channel<br/>env_{id}__broadcast__{agent}]
        SC[state_update channel<br/>env_{id}__state_updates]
    end

    subgraph "Flow Direction"
        Parent -->|ACTION| Child
        Child -->|INFO| Parent
        Any -->|BROADCAST| All
        Env -->|STATE_UPDATE| Agents
    end
```
