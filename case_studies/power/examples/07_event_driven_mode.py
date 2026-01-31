"""
Example 7: Event-Driven Mode (Option B)
=======================================

This example demonstrates the event-driven execution mode of HERON, which is used
for testing policy robustness under realistic timing constraints.

What you'll learn:
- How to setup event-driven execution with EventScheduler
- Configuring agent tick intervals and delays
- How hierarchical coordination works in async mode
- Differences between Option A (sync) and Option B (event-driven)

Key Concepts:
- Option A (Synchronous): All agents step together in env.step()
  - Used for training with CTDE pattern
  - Coordinator waits for subordinate observations

- Option B (Event-Driven): Agents tick independently via EventScheduler
  - Used for testing policy robustness
  - Configurable delays (obs_delay, act_delay, msg_delay)
  - Coordinator sends messages that subordinates receive on their next tick

Runtime: ~5 seconds
"""

import numpy as np
from typing import Any, Dict

from gymnasium.spaces import Box

from heron.scheduling import EventScheduler, EventType, Event
from heron.agents.field_agent import FieldAgent
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation
from heron.core.policies import Policy


class SimplePolicy(Policy):
    """Simple policy that returns random actions."""

    def __init__(self, action_dim: int = 1):
        self.action_dim = action_dim

    def forward(self, observation: Observation) -> np.ndarray:
        return np.random.uniform(-1, 1, self.action_dim)

    def reset(self) -> None:
        pass


class SimpleFieldAgent(FieldAgent):
    """Simple field agent for demonstration."""

    def __init__(self, agent_id: str, tick_interval: float = 1.0, act_delay: float = 0.0):
        self._value = 0.0  # Initialize before super().__init__
        super().__init__(
            agent_id=agent_id,
            config={"name": agent_id},
            tick_interval=tick_interval,
            act_delay=act_delay,
        )

    def set_state(self) -> None:
        pass  # Use default empty state

    def set_action(self) -> None:
        self.action.set_specs(
            dim_c=1,
            range=(np.array([-1.0]), np.array([1.0]))
        )

    def _get_obs(self) -> np.ndarray:
        """Override to return simple observation."""
        return np.array([self._value], dtype=np.float32)

    def _get_observation_space(self):
        """Override to return simple observation space."""
        return Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def update_state(self, delta: float = 0.0) -> None:
        self._value += delta

    def update_cost_safety(self) -> None:
        self.cost = abs(self._value)
        self.safety = max(0, abs(self._value) - 5.0)


class SimpleCoordinator(CoordinatorAgent):
    """Simple coordinator for demonstration."""

    def __init__(
        self,
        agent_id: str,
        subordinates: Dict[str, FieldAgent],
        tick_interval: float = 5.0,
        msg_delay: float = 0.5,
    ):
        # Set up subordinates before super().__init__
        self._init_subordinates = subordinates

        super().__init__(
            agent_id=agent_id,
            policy=SimplePolicy(action_dim=len(subordinates)),
            tick_interval=tick_interval,
            msg_delay=msg_delay,
        )

        # Set subordinates
        self.subordinate_agents = subordinates
        for sub_id, sub in subordinates.items():
            sub.upstream_id = agent_id


def run_event_driven_example():
    """Run event-driven simulation example."""

    print("=" * 80)
    print("Example 7: Event-Driven Mode (Option B)")
    print("=" * 80)
    print()

    # ============================================================
    # Step 1: Create Agents with Timing Parameters
    # ============================================================
    print("Step 1: Creating agents with timing parameters...")
    print("-" * 80)

    # Create field agents with 1-second tick interval
    field_1 = SimpleFieldAgent("field_1", tick_interval=1.0, act_delay=0.1)
    field_2 = SimpleFieldAgent("field_2", tick_interval=1.0, act_delay=0.1)

    # Create coordinator with 5-second tick interval and 0.5s message delay
    coordinator = SimpleCoordinator(
        "coordinator",
        subordinates={"field_1": field_1, "field_2": field_2},
        tick_interval=5.0,
        msg_delay=0.5,
    )

    print(f"Created coordinator: tick_interval={coordinator.tick_interval}s, msg_delay={coordinator.msg_delay}s")
    print(f"Created field_1: tick_interval={field_1.tick_interval}s, act_delay={field_1.act_delay}s")
    print(f"Created field_2: tick_interval={field_2.tick_interval}s, act_delay={field_2.act_delay}s")
    print()

    # ============================================================
    # Step 2: Setup EventScheduler
    # ============================================================
    print("Step 2: Setting up EventScheduler...")
    print("-" * 80)

    scheduler = EventScheduler(start_time=0.0)

    # Register agents with their timing parameters
    scheduler.register_agent(
        agent_id="coordinator",
        tick_interval=coordinator.tick_interval,
        obs_delay=coordinator.obs_delay,
        act_delay=coordinator.act_delay,
    )
    scheduler.register_agent(
        agent_id="field_1",
        tick_interval=field_1.tick_interval,
        obs_delay=field_1.obs_delay,
        act_delay=field_1.act_delay,
    )
    scheduler.register_agent(
        agent_id="field_2",
        tick_interval=field_2.tick_interval,
        obs_delay=field_2.obs_delay,
        act_delay=field_2.act_delay,
    )

    print(f"Registered {len(scheduler.agent_intervals)} agents with scheduler")
    print()

    # ============================================================
    # Step 3: Setup Event Handlers
    # ============================================================
    print("Step 3: Setting up event handlers...")
    print("-" * 80)

    agents = {
        "coordinator": coordinator,
        "field_1": field_1,
        "field_2": field_2,
    }

    tick_count = {"coordinator": 0, "field_1": 0, "field_2": 0}
    action_count = {"field_1": 0, "field_2": 0}

    def on_agent_tick(event: Event, sched: EventScheduler) -> None:
        """Handle AGENT_TICK events."""
        agent = agents.get(event.agent_id)
        if agent is not None:
            tick_count[event.agent_id] += 1
            agent.tick(sched, event.timestamp, global_state=None)
            print(f"  t={event.timestamp:5.1f}: {event.agent_id} tick #{tick_count[event.agent_id]}")

    def on_action_effect(event: Event, sched: EventScheduler) -> None:
        """Handle ACTION_EFFECT events."""
        agent_id = event.agent_id
        action = event.payload.get("action")
        if agent_id in ["field_1", "field_2"]:
            action_count[agent_id] += 1
            # Apply action to field agent
            agent = agents[agent_id]
            if action is not None and hasattr(action, '__len__'):
                agent.update_state(float(action[0]) if len(action) > 0 else 0.0)
            print(f"  t={event.timestamp:5.1f}: {agent_id} action effect #{action_count[agent_id]}")

    def on_message_delivery(event: Event, sched: EventScheduler) -> None:
        """Handle MESSAGE_DELIVERY events via message broker."""
        recipient_id = event.agent_id
        sender = event.payload.get("sender")
        message = event.payload.get("message", {})

        # Publish message via message broker if available
        if env.message_broker is not None and sender is not None:
            if "action" in message:
                env.publish_action(sender_id=sender, recipient_id=recipient_id, action=message.get("action"))
            else:
                env.publish_info(sender_id=sender, recipient_id=recipient_id, info=message)
        print(f"  t={event.timestamp:5.1f}: Message delivered to {recipient_id} from {sender}")

    scheduler.set_handler(EventType.AGENT_TICK, on_agent_tick)
    scheduler.set_handler(EventType.ACTION_EFFECT, on_action_effect)
    scheduler.set_handler(EventType.MESSAGE_DELIVERY, on_message_delivery)

    print("Event handlers configured")
    print()

    # ============================================================
    # Step 4: Run Simulation
    # ============================================================
    print("Step 4: Running event-driven simulation for 15 seconds...")
    print("-" * 80)

    events_processed = scheduler.run_until(t_end=15.0)

    print()
    print(f"Processed {events_processed} events")
    print(f"Final simulation time: {scheduler.current_time:.1f}s")
    print()

    # ============================================================
    # Step 5: Summary
    # ============================================================
    print("Step 5: Summary")
    print("-" * 80)

    print("Tick counts:")
    for agent_id, count in tick_count.items():
        print(f"  {agent_id}: {count} ticks")

    print("\nAction effect counts:")
    for agent_id, count in action_count.items():
        print(f"  {agent_id}: {count} actions applied")

    print("\nField agent final states:")
    print(f"  field_1 value: {field_1._value:.3f}")
    print(f"  field_2 value: {field_2._value:.3f}")

    print()


def compare_option_a_vs_b():
    """Compare synchronous (Option A) vs event-driven (Option B)."""

    print("=" * 80)
    print("Comparison: Option A (Sync) vs Option B (Event-Driven)")
    print("=" * 80)
    print()

    print("Option A - Synchronous (Training):")
    print("-" * 40)
    print("1. All agents step together in env.step()")
    print("2. Coordinator.observe() calls subordinate.observe()")
    print("3. Coordinator.act() computes joint action")
    print("4. Actions applied immediately to subordinates")
    print("5. No delays - perfect synchronization")
    print()

    print("Option B - Event-Driven (Testing):")
    print("-" * 40)
    print("1. Each agent ticks at its own interval")
    print("2. Coordinator sends messages (with msg_delay)")
    print("3. Subordinates receive messages on their next tick")
    print("4. Actions take effect after act_delay")
    print("5. Tests policy robustness to async timing")
    print()

    print("Key Timing Parameters:")
    print("-" * 40)
    print("tick_interval: How often an agent ticks (steps)")
    print("obs_delay:     Agent sees state from t - obs_delay")
    print("act_delay:     Action takes effect at t + act_delay")
    print("msg_delay:     Messages arrive after msg_delay")
    print()


def main():
    """Run all demonstrations."""

    # Main event-driven example
    run_event_driven_example()

    # Comparison explanation
    compare_option_a_vs_b()

    print("=" * 80)
    print("Example 7 Complete!")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("1. Option B uses EventScheduler for realistic async execution")
    print("2. Agents tick independently at their own intervals")
    print("3. Coordinator sends messages that arrive with delay")
    print("4. Field agents apply actions with act_delay")
    print("5. Use Option A for training, Option B for testing robustness")
    print()


if __name__ == "__main__":
    main()
