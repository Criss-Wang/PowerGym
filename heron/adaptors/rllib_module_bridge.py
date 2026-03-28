from __future__ import annotations

from typing import Optional

import gymnasium as gym

from ray.rllib.core.rl_module.rl_module import RLModule

from heron.core.action import Action
from heron.core.observation import Observation
from heron.core.policies import Policy


class RLlibModuleBridge(Policy):
    """Wraps an RLlib RLModule (new API stack) as a HERON ``Policy``.

    Parameters
    ----------
    rl_module : ray.rllib.core.rl_module.RLModule
        An RLModule obtained via ``algo.get_module(module_id)``.
    agent_id : str
        The HERON agent ID this policy is attached to.
    action_space : gymnasium.Space, optional
        Action space for building HERON Actions. If *None*, falls back to
        ``rl_module.config.action_space``.
    """

    observation_mode: str = "full"

    def __init__(
        self,
        rl_module: RLModule,
        agent_id: str,
        action_space: gym.Space | None = None,
    ) -> None:
        self._module = rl_module
        self._agent_id = agent_id
        space = action_space or rl_module.config.action_space
        self._action_template = Action.from_gym_space(space)

    def forward(self, observation: Observation) -> Optional[Action]:
        import torch

        obs_vec = observation.vector()
        obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).float()

        self._module.eval()
        with torch.no_grad():
            output = self._module.forward_inference({"obs": obs_tensor})

        # Extract action from distribution inputs
        if "actions" in output:
            raw = output["actions"].cpu().numpy()[0]
            action = self._action_template.copy()
            action.set_values(raw)
        elif "action_dist_inputs" in output:
            dist_inputs = output["action_dist_inputs"].cpu().numpy()[0]
            action = self._action_template.copy()
            action.set_values(
                **self._parse_dist_inputs(dist_inputs, self._action_template)
            )
        else:
            raise ValueError(f"Unexpected RLModule output keys: {output.keys()}")
        return action

    @staticmethod
    def _parse_dist_inputs(
        dist_inputs: "np.ndarray", template: Action
    ) -> dict:
        """Extract continuous means and discrete argmax from dist_inputs.

        Layout produced by RLlib:
          - Gaussian (continuous): [means (dim_c), log_stds (dim_c)]
          - Categorical (discrete): [logits_head_0, logits_head_1, ...]
          - Mixed: Gaussian params first, then categorical logits
        """
        import numpy as np

        offset = 0
        result: dict = {}

        # Continuous: first dim_c values are means, next dim_c are log_stds
        if template.dim_c:
            result["c"] = dist_inputs[offset : offset + template.dim_c]
            offset += template.dim_c * 2  # skip means + log_stds

        # Discrete: argmax over each head's logits
        if template.dim_d:
            d_vals = np.empty(template.dim_d, dtype=np.int32)
            for i, n in enumerate(template.ncats):
                logits = dist_inputs[offset : offset + n]
                d_vals[i] = int(np.argmax(logits))
                offset += n
            result["d"] = d_vals

        return result

    def reset(self) -> None:
        pass
