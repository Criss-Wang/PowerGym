# HERON Testing Flow (Option B - Event-Driven)

This diagram shows the event-driven execution flow used during testing, where agents operate asynchronously with realistic timing delays.

```mermaid
sequenceDiagram
    autonumber
    participant Env as MultiAgentEnv
    participant Sched as EventScheduler
    participant SA as SystemAgent (L3)
    participant CA as CoordinatorAgent (L2)
    participant FA as FieldAgent (L1)
    participant Proxy as ProxyAgent
    participant MB as MessageBroker
    participant Policy as Policy

    Note over Env,Policy: === SETUP PHASE ===

    Env->>Env: setup_event_driven()
    Env->>Sched: EventScheduler(start_time=0)

    loop For each Agent
        Env->>Sched: register_agent(id, tick_interval, obs_delay, act_delay)
        Sched->>Sched: schedule(AGENT_TICK at t=0)
    end

    Env->>Env: setup_default_handlers()
    Env->>Sched: set_handler(AGENT_TICK, tick_handler)
    Env->>Sched: set_handler(ACTION_EFFECT, action_handler)
    Env->>Sched: set_handler(MESSAGE_DELIVERY, msg_handler)

    Note over Env,Policy: === EVENT LOOP (run_event_driven) ===

    loop while current_time < t_end
        Sched->>Sched: pop() next event from heap

        alt Event: AGENT_TICK (FieldAgent)
            rect rgb(240, 255, 240)
                Note over Sched,FA: FieldAgent Tick (t=current_time)

                Sched->>FA: tick(scheduler, current_time, global_state, proxy)
                FA->>FA: _timestep = current_time

                opt Has upstream messages
                    FA->>MB: receive_action_messages()
                    MB-->>FA: [upstream_action]
                end

                alt obs_delay > 0 and proxy exists
                    FA->>Proxy: request_state_from_proxy(at_time=t-obs_delay)
                    Proxy->>Proxy: get_state_at_time(target_time)
                    Proxy-->>FA: delayed_state
                    FA->>FA: _build_observation_from_proxy(proxy_state)
                else No delay
                    FA->>FA: observe(global_state)
                end

                alt Has upstream_action
                    FA->>FA: _handle_coordinator_action(upstream_action)
                else Has policy
                    FA->>Policy: forward(observation)
                    Policy-->>FA: action
                    FA->>FA: _handle_local_action(action)
                end

                FA->>FA: action.set_values(action)

                opt act_delay > 0
                    FA->>Sched: schedule_action_effect(agent_id, action, delay)
                    Sched->>Sched: schedule(ACTION_EFFECT at t+act_delay)
                end

                Sched->>Sched: schedule_agent_tick(t + tick_interval)
            end

        else Event: AGENT_TICK (CoordinatorAgent)
            rect rgb(255, 248, 240)
                Note over Sched,CA: CoordinatorAgent Tick (t=current_time)

                Sched->>CA: tick(scheduler, current_time, global_state, proxy)
                CA->>CA: _timestep = current_time

                opt Has upstream messages from SystemAgent
                    CA->>MB: receive_action_messages()
                    MB-->>CA: [upstream_action]
                end

                CA->>CA: observe(global_state)
                Note over CA: Collects from subordinates (may be stale)

                alt Has upstream_action
                    CA->>CA: use upstream_action
                else Has policy
                    CA->>Policy: forward(observation)
                    Policy-->>CA: joint_action
                end

                loop For each subordinate
                    CA->>Sched: schedule_message_delivery(sender, recipient, action, msg_delay)
                    Sched->>Sched: schedule(MESSAGE_DELIVERY at t+msg_delay)
                end

                Sched->>Sched: schedule_agent_tick(t + tick_interval)
            end

        else Event: AGENT_TICK (SystemAgent)
            rect rgb(240, 248, 255)
                Note over Sched,SA: SystemAgent Tick (t=current_time)

                Sched->>SA: tick(scheduler, current_time, global_state, proxy)
                SA->>SA: observe(global_state)

                opt Has external action
                    SA->>MB: receive_action_messages()
                end

                opt Has action to distribute
                    loop For each coordinator
                        SA->>Sched: schedule_message_delivery(sender, recipient, action, msg_delay)
                    end
                end

                Sched->>Sched: schedule_agent_tick(t + tick_interval)
            end

        else Event: ACTION_EFFECT
            rect rgb(255, 240, 255)
                Note over Sched,Env: Delayed Action Takes Effect

                Sched->>Env: on_action_effect(agent_id, action)
                Env->>Env: Apply action to physics
            end

        else Event: MESSAGE_DELIVERY
            rect rgb(255, 255, 240)
                Note over Sched,MB: Delayed Message Delivery

                Sched->>MB: publish_action(sender, recipient, action)
                MB->>MB: store in recipient's channel
                Note over MB: Available on next tick
            end
        end
    end
```

## Event Priority Queue

```mermaid
graph TB
    subgraph "EventScheduler"
        direction TB
        Q[Priority Queue<br/>Min-Heap by timestamp]

        E1[Event t=0.0<br/>AGENT_TICK FA1]
        E2[Event t=0.0<br/>AGENT_TICK FA2]
        E3[Event t=0.5<br/>ACTION_EFFECT FA1]
        E4[Event t=1.0<br/>AGENT_TICK FA1]
        E5[Event t=5.0<br/>AGENT_TICK CA1]
        E6[Event t=60.0<br/>AGENT_TICK SA]

        Q --> E1
        Q --> E2
        Q --> E3
        Q --> E4
        Q --> E5
        Q --> E6
    end

    subgraph "Event Types"
        TICK[AGENT_TICK<br/>Agent observe-act cycle]
        ACTION[ACTION_EFFECT<br/>Delayed action applies]
        MSG[MESSAGE_DELIVERY<br/>Delayed message arrives]
        OBS[OBSERVATION_READY<br/>Delayed obs available]
    end
```

## Timing Parameters

```mermaid
graph LR
    subgraph "Agent Timing Parameters"
        FA[FieldAgent<br/>tick_interval=1s<br/>obs_delay=0.1s<br/>act_delay=0.2s<br/>msg_delay=0.05s]

        CA[CoordinatorAgent<br/>tick_interval=60s<br/>obs_delay=0s<br/>act_delay=0s<br/>msg_delay=1s]

        SA[SystemAgent<br/>tick_interval=300s<br/>obs_delay=0s<br/>act_delay=0s<br/>msg_delay=5s]
    end

    subgraph "Delay Effects"
        OD[obs_delay<br/>Agent sees state from t-delay]
        AD[act_delay<br/>Action takes effect at t+delay]
        MD[msg_delay<br/>Message arrives at t+delay]
    end
```

## ProxyAgent State History

```mermaid
sequenceDiagram
    participant Env as Environment
    participant Proxy as ProxyAgent
    participant FA as FieldAgent

    Note over Env,FA: ProxyAgent maintains state history for delayed observations

    Env->>Proxy: update_state(state @ t=0)
    Proxy->>Proxy: state_history.append({t=0, state})

    Env->>Proxy: update_state(state @ t=1)
    Proxy->>Proxy: state_history.append({t=1, state})

    Env->>Proxy: update_state(state @ t=2)
    Proxy->>Proxy: state_history.append({t=2, state})

    Note over FA: FieldAgent ticks at t=2.5 with obs_delay=1.0

    FA->>Proxy: get_state_for_agent(at_time=1.5)
    Proxy->>Proxy: Find state where t <= 1.5
    Proxy-->>FA: state @ t=1 (delayed observation)

    FA->>FA: Build observation from delayed state
    FA->>FA: Compute action based on stale info
```

## Message Broker Flow in Event-Driven Mode

```mermaid
graph TB
    subgraph "Coordinator Tick (t=60)"
        CT[CA.tick] --> COMPUTE[Compute joint action]
        COMPUTE --> SCHED_MSG[Schedule MESSAGE_DELIVERY<br/>for each subordinate<br/>at t=60+msg_delay]
    end

    subgraph "Message Delivery Event (t=61)"
        MDE[MESSAGE_DELIVERY] --> PUBLISH[broker.publish<br/>action_channel]
        PUBLISH --> STORE[Store in channel queue]
    end

    subgraph "FieldAgent Tick (t=62)"
        FAT[FA.tick] --> RECEIVE[receive_action_messages]
        RECEIVE --> CONSUME[broker.consume<br/>action_channel]
        CONSUME --> ACT[Execute received action]
    end

    CT -.->|schedules| MDE
    MDE -.->|enables| FAT
```

## Complete Event-Driven Architecture

```mermaid
graph TB
    subgraph "HeronEnvCore"
        ENV[MultiAgentEnv]
        PROXY[ProxyAgent]
        MB[MessageBroker]
    end

    subgraph "EventScheduler"
        QUEUE[Event Priority Queue]
        HANDLERS[Event Handlers]
    end

    subgraph "Agent Hierarchy"
        SA[SystemAgent<br/>tick_interval=300s]
        CA[CoordinatorAgent<br/>tick_interval=60s]
        FA1[FieldAgent<br/>tick_interval=1s]
        FA2[FieldAgent<br/>tick_interval=1s]
    end

    ENV -->|setup_event_driven| QUEUE
    ENV -->|setup_default_handlers| HANDLERS
    ENV -->|update_proxy_state| PROXY

    QUEUE -->|AGENT_TICK| SA
    QUEUE -->|AGENT_TICK| CA
    QUEUE -->|AGENT_TICK| FA1
    QUEUE -->|AGENT_TICK| FA2
    QUEUE -->|ACTION_EFFECT| ENV
    QUEUE -->|MESSAGE_DELIVERY| MB

    SA -->|schedule_message_delivery| QUEUE
    CA -->|schedule_message_delivery| QUEUE
    FA1 -->|schedule_action_effect| QUEUE
    FA2 -->|schedule_action_effect| QUEUE

    FA1 -->|receive_action_messages| MB
    FA2 -->|receive_action_messages| MB
    CA -->|receive_action_messages| MB

    FA1 -->|request_state_from_proxy| PROXY
    FA2 -->|request_state_from_proxy| PROXY
```
