"""Custom RLlib learner connector for HERON heterogeneous tick-rate masking.

``MaskInactiveAgentTimesteps`` modifies the ``LOSS_MASK`` in the learner
pipeline so that timesteps where an agent was inactive (did not act due to
its tick rate) are excluded from the PPO/MAPPO loss and GAE computation.

Without this connector, inactive timesteps produce ``reward=0.0`` and a
stale observation, but RLlib still computes advantages and policy gradients
for them — adding noise that degrades sample efficiency, especially under
MAPPO with a shared policy.

Pipeline position::

    AddOneTsToEpisodesAndTruncate   ← creates LOSS_MASK (bootstrap ts = False)
    **MaskInactiveAgentTimesteps**  ← ANDs activity mask into existing LOSS_MASK
    ...default connectors...
    GeneralAdvantageEstimation      ← uses LOSS_MASK-aware rewards

Usage::

    from heron.adaptors.rllib_learner_connector import MaskInactiveAgentTimesteps

    config = (
        PPOConfig()
        ...
        .training(
            learner_connector=lambda obs_sp, act_sp: MaskInactiveAgentTimesteps(),
        )
    )
"""

from typing import Any, Dict, List, Optional

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.typing import EpisodeType


class MaskInactiveAgentTimesteps(ConnectorV2):
    """Masks inactive-agent timesteps in the LOSS_MASK.

    Reads ``info["is_active"]`` from each single-agent episode and sets
    ``LOSS_MASK = False`` for timesteps where the agent was not active.
    Must run **after** ``AddOneTsToEpisodesAndTruncate`` (which creates
    the initial LOSS_MASK with the bootstrap timestep masked out).

    The ``info["is_active"]`` dict is set by ``RLlibBasedHeronEnv.step()``
    and maps each agent ID to a boolean.  This connector extracts the
    flag for the episode's own agent.
    """

    INFO_KEY = "is_active"

    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        if Columns.LOSS_MASK not in batch:
            return batch

        for sa_episode in self.single_agent_episode_iterator(
            episodes, agents_that_stepped_only=False,
        ):
            agent_id = sa_episode.agent_id

            # Build the sub-key that AddOneTsToEpisodesAndTruncate used
            if agent_id is not None:
                sub_key = (
                    sa_episode.multi_agent_episode_id,
                    agent_id,
                    sa_episode.module_id,
                )
            else:
                sub_key = (sa_episode.id_,)

            mask_data = batch[Columns.LOSS_MASK]
            if sub_key not in mask_data:
                continue

            # mask_data[sub_key] is a flat list of booleans:
            #   [True, True, ..., True, False]
            # produced by AddOneTsToEpisodesAndTruncate.
            # Length = ep_len + 1 (last entry is the bootstrap ts mask).
            mask_list = mask_data[sub_key]
            if not mask_list:
                continue

            # Read per-timestep info dicts from the episode.
            # AddOneTsToEpisodesAndTruncate extended the episode by 1 ts,
            # so len(sa_episode) is now ep_len + 1.  Infos for the real
            # timesteps are at indices 0..ep_len-2 (the original steps);
            # the last info (index ep_len-1) belongs to the bootstrap ts.
            ep_len = len(sa_episode) - 1  # original length before extension
            infos = sa_episode.get_infos(slice(0, ep_len))
            if not isinstance(infos, list):
                infos = [infos]

            for t, info in enumerate(infos):
                if not isinstance(info, dict):
                    continue
                active_flags = info.get(self.INFO_KEY)
                if active_flags is None:
                    continue

                # active_flags is {agent_id: bool, ...} — extract own flag
                if isinstance(active_flags, dict):
                    is_active = active_flags.get(agent_id, True)
                else:
                    is_active = bool(active_flags)

                if not is_active:
                    mask_list[t] = False

        return batch
