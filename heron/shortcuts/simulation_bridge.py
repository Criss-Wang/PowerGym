"""SimpleEnv — a MultiAgentEnv that requires only one user function.

Instead of implementing three abstract methods (``run_simulation``,
``env_state_to_global_state``, ``global_state_to_env_state``), the user
provides a single ``simulate`` callable (or ``None`` for pass-through).

The bridge automatically converts between the HERON global-state dict
and a flat ``{agent_id: {field_name: value, ...}}`` representation.

Usage::

    def simulate(agent_states: dict) -> dict:
        for aid, s in agent_states.items():
            s["power"] = np.clip(s["power"], -1.0, 1.0)
        return agent_states

    env = SimpleEnv(
        system_agent=system,
        simulate_fn=simulate,
    )
"""

from typing import Any, Callable, Dict, List, Optional

from heron.agents.constants import FIELD_LEVEL
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.agents.system_agent import SystemAgent
from heron.envs.base import MultiAgentEnv


class SimpleEnv(MultiAgentEnv):
    """MultiAgentEnv with automatic state conversion.

    Args:
        simulate_fn: ``(flat_states) -> flat_states`` or ``None`` for pass-through.
            *flat_states* is ``{agent_id: {field_name: value, ...}}``.
        **kwargs: Forwarded to ``MultiAgentEnv.__init__``.
    """

    def __init__(
        self,
        simulate_fn: Optional[Callable[[Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Any]]]] = None,
        **kwargs: Any,
    ):
        self._simulate_fn = simulate_fn
        super().__init__(**kwargs)

    # -- Abstract method implementations --------------------------------------

    def run_simulation(self, env_state: Any, *args, **kwargs) -> Any:
        if self._simulate_fn is None:
            return env_state
        return self._simulate_fn(env_state)

    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> Any:
        """Flatten HERON global state to ``{agent_id: {field: value}}``."""
        agent_states = global_state.get("agent_states", {})
        flat: Dict[str, Dict[str, Any]] = {}
        for aid, state_dict in agent_states.items():
            features = state_dict.get("features", state_dict)
            merged: Dict[str, Any] = {}
            for feat_name, feat_fields in features.items():
                if feat_name.startswith("_"):
                    continue
                if isinstance(feat_fields, dict):
                    merged.update(feat_fields)
                else:
                    merged[feat_name] = feat_fields
            flat[aid] = merged
        return flat

    def env_state_to_global_state(self, env_state: Any) -> Dict[str, Any]:
        """Repack flat dict back to HERON global-state format."""
        agent_states: Dict[str, Any] = {}
        for aid, flat_fields in env_state.items():
            agent = self.registered_agents.get(aid)
            if agent is None or not hasattr(agent, "level"):
                continue
            if agent.level != FIELD_LEVEL:
                continue

            # Rebuild per-feature dicts from the agent's current state
            feature_dict: Dict[str, Dict[str, Any]] = {}
            for feat_name, feat_obj in agent.state.features.items():
                feat_fields_for_name: Dict[str, Any] = {}
                for fn in feat_obj.names():
                    if fn in flat_fields:
                        feat_fields_for_name[fn] = flat_fields[fn]
                    else:
                        feat_fields_for_name[fn] = getattr(feat_obj, fn)
                feature_dict[feat_name] = feat_fields_for_name

            agent_states[aid] = {
                "_owner_id": aid,
                "_owner_level": agent.level,
                "_state_type": "FieldAgentState",
                "features": feature_dict,
            }

        return {"agent_states": agent_states}
