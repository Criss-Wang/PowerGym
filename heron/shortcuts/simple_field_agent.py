"""SimpleFieldAgent — a FieldAgent with sensible defaults.

Only ``reward()`` must be overridden. Action setup, state management,
and action application are derived from class-level declarations.

Usage::

    class DeviceAgent(SimpleFieldAgent):
        features = [Battery()]
        action_dim = 1
        action_range = (-0.2, 0.2)

        def reward(self, state) -> float:
            charge = state["Battery"]["charge"]
            return -((charge - 0.5) ** 2)
"""

from copy import deepcopy
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import FeatureProvider
from heron.core.state import FieldAgentState, State
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.tick_config import TickConfig
from heron.utils.typing import AgentID


class SimpleFieldAgent(FieldAgent):
    """FieldAgent subclass with auto-generated boilerplate.

    Class-level attributes (override in subclass):
        features: List of Feature instances (deep-copied per agent instance).
        action_dim: Number of continuous action dimensions (default 1).
        action_range: ``(low, high)`` bounds for actions (default ``(-1.0, 1.0)``).

    Only ``reward(state) -> float`` must be overridden.

    The ``state`` dict passed to ``reward()`` uses named access::

        state["FeatureName"]["field_name"]  # e.g. state["Battery"]["charge"]
    """

    # -- Class-level declarations (override in subclass) ----------------------
    features: ClassVar[List[FeatureProvider]] = []
    action_dim: ClassVar[int] = 1
    action_range: ClassVar[Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]] = (-1.0, 1.0)

    def __init__(
        self,
        agent_id: AgentID,
        features: Optional[List[FeatureProvider]] = None,
        *,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        tick_config: Optional[TickConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        # Deep-copy class-level feature templates so each instance is independent
        if features is None:
            features = [deepcopy(f) for f in self.__class__.features]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            tick_config=tick_config,
            policy=policy,
            protocol=protocol,
        )

    # -- Auto-generated lifecycle methods -------------------------------------

    def init_action(self, features: List[FeatureProvider] = []) -> Action:
        """Auto-generate Action from ``action_dim`` and ``action_range``."""
        action = Action()
        dim = self.__class__.action_dim
        r = self.__class__.action_range

        if isinstance(r[0], np.ndarray):
            lo, hi = r
        else:
            lo = np.full(dim, float(r[0]), dtype=np.float32)
            hi = np.full(dim, float(r[1]), dtype=np.float32)

        action.set_specs(dim_c=dim, range=(lo, hi))
        action.reset()
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        """Delegate to ``Action.set_values``."""
        if isinstance(action, Action):
            self.action.set_values(action)
        else:
            self.action.set_values(action)

    def apply_action(self) -> None:
        """Write ``action.c[i]`` into the first feature's *i*-th field."""
        if not self.state.features:
            return
        first_feature = next(iter(self.state.features.values()))
        field_names = first_feature.names()
        updates = {}
        for i, fn in enumerate(field_names):
            if i < self.action.dim_c:
                updates[fn] = float(self.action.c[i])
        if updates:
            first_feature.set_values(**updates)

    def set_state(self, *args, **kwargs) -> None:
        """No-op — state is updated via ``apply_action``."""
        pass

    # -- Reward (user overrides this) -----------------------------------------

    def compute_local_reward(self, local_state: dict) -> float:
        """Framework hook — converts raw state to named dict, then calls reward().

        Do NOT override this. Override ``reward()`` instead.
        """
        named_state = self._to_named_state(local_state)
        return self.reward(named_state)

    def reward(self, state: dict) -> float:
        """Compute reward from named state dict.

        Override this in your subclass. The ``state`` dict maps feature names
        to field-name dicts::

            state["Output"]["value"]        # single-field feature
            state["Battery"]["charge"]      # multi-field feature
            state["Battery"]["capacity"]

        Args:
            state: ``{feature_name: {field_name: float_value, ...}, ...}``

        Returns:
            Scalar reward value.
        """
        raise NotImplementedError(
            "SimpleFieldAgent subclasses must implement reward()"
        )

    def _to_named_state(self, local_state: dict) -> Dict[str, Any]:
        """Convert ``{feat_name: np.array}`` to ``{feat_name: {field: value}}``."""
        named: Dict[str, Any] = {}
        for feat_name, vec in local_state.items():
            # Pass through non-feature entries (e.g. subordinate_rewards)
            if not isinstance(vec, np.ndarray):
                named[feat_name] = vec
                continue
            # Look up field names from the agent's own features
            if feat_name in self.state.features:
                feature = self.state.features[feat_name]
                field_names = feature.names()
                named[feat_name] = {
                    fn: float(vec[i])
                    for i, fn in enumerate(field_names)
                    if i < len(vec)
                }
            else:
                named[feat_name] = vec
        return named
