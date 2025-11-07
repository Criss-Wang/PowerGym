from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from powergrid.core.policies import Policy
from powergrid.features.step_state import StepState
from powergrid.utils.registry import provider
from powergrid.agents.device_agent import DeviceAgent
from powergrid.core.protocols import NoProtocol, Protocol
from powergrid.core.state import DeviceState, PhaseModel


@dataclass
class ShuntConfig:
    """Configuration for a switched shunt (capacitor/reactor bank)."""
    bus: str
    q_mvar: float
    max_step: int
    switching_cost: float



class Shunt(DeviceAgent):
    """Switched shunt (capacitor/reactor bank) — **controllable**, not passive.

    Discrete action selects number of steps (0..max_step). Optional switching
    cost applies when the step changes.
    """
    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        policy: Optional[Policy] = None,
        protocol: Protocol = NoProtocol(),
        device_config: Dict[str, Any],
    ) -> None:
        config = device_config.get("device_state_config", {})
        self._shunt_config = ShuntConfig(
            bus=config.get("bus", ""),
            q_mvar=float(config.get("q_mvar", 0.0)),
            max_step=int(config.get("max_step", 0)),
            switching_cost=float(config.get("switching_cost", 0.0)),
        )
        self.type = "SCB"
        self._last_step = 0

        super().__init__(
            agent_id=agent_id,
            policy=policy,
            protocol=protocol,
            device_config=device_config,
        )

    def set_action_space(self) -> None:
        # discrete steps: 0..max_step
        self.action.set_specs(
            dim_c=0,
            dim_d=1,
            ncats=self._shunt_config.max_step + 1,
            range=None,
            masks=[np.ones(self._shunt_config.max_step + 1, dtype=bool)]
        )
        self.action.sample()

    def set_device_state(self) -> None:
        # Create step state feature
        step_state = StepState(
            max_step=self._shunt_config.max_step,
            step=np.zeros(self._shunt_config.max_step + 1, dtype=np.float32),
        )
        self.state = DeviceState(
            phase_model=PhaseModel.BALANCED_1PH,
            phase_spec=None,
            features=[step_state],
            prefix_names=False
        )

    def update_state(self) -> None:
        step_state = self._get_step_state()
        step = int(self.action.d[0]) if self.action.d.size else 0
        step_state.step = np.zeros(self._shunt_config.max_step + 1, dtype=np.float32)
        step_state.step[step] = 1.0
        self._current_step = step  # for cost calculation

    def update_cost_safety(self) -> None:
        changed = int(getattr(self, "_current_step", 0) != getattr(self, "_last_step", 0))
        self.cost = float(self._shunt_config.switching_cost * changed)
        self.safety = 0.0
        self._last_step = getattr(self, "_current_step", self._last_step)

    def reset_device(self, rnd=None) -> None:
        step_state = self._get_step_state()
        step_state.step = np.zeros(self._shunt_config.max_step + 1, dtype=np.float32)
        self._last_step = 0
        self.cost = 0.0
        self.safety = 0.0

    def _get_step_state(self) -> StepState:
        """Get the StepState feature from state."""
        for feature in self.state.features:
            if isinstance(feature, StepState):
                return feature
        raise ValueError("StepState feature not found in state")

    @property
    def bus(self) -> str:
        return self._shunt_config.bus

    def __repr__(self) -> str:
        return f"Shunt(name={self.agent_id}, bus={self._shunt_config.bus}, max_step={self._shunt_config.max_step})"