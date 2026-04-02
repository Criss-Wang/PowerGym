from typing import Any, List, Optional

import numpy as np

from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.scheduler import Event, EventScheduler
from heron.scheduling.schedule_config import ScheduleConfig
from heron.utils.typing import AgentID

from case_studies.power.ev_public_charging_case.features import ChargerFeature


class ChargerAgent(FieldAgent):
    """A single charging port controlled by station pricing.

    Features:
        ChargerFeature — physical charger state (power, max power, open/closed)

    Action:
        1D continuous [0, 0.8] — station price broadcast from coordinator ($/kWh).
        The charger agent stores the latest received price; env decides EV occupancy/charging.
    """

    def __init__(
        self,
        agent_id: AgentID,
        p_max_kw: float = 150.0,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        self._p_max_kw = p_max_kw

        features: List[Feature] = [ChargerFeature(p_max_kw=p_max_kw)]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            schedule_config=schedule_config,
            policy=policy,
            protocol=protocol,
        )

    def init_action(self, features: Optional[List[Feature]] = None) -> Action:
        action = Action()
        action.set_specs(
            dim_c=1,
            dim_d=0,
            range=(np.array([0.0]), np.array([0.8])),  # price range from coordinator
        )
        action.set_values(c=np.array([0.25], dtype=np.float32))  # default price
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        if isinstance(action, Action):
            self.action = action
            return
        price = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self.action.set_values(c=np.array([price], dtype=np.float32))

    def set_state(self, **kwargs) -> None:
        if 'charging_price' in kwargs:
            self.state.update_feature("ChargerFeature", charging_price=kwargs['charging_price'])
        if 'p_kw' in kwargs:
            self.state.update_feature("ChargerFeature", p_kw=kwargs['p_kw'])
        if 'occupied_or_not' in kwargs:
            self.state.update_feature("ChargerFeature", occupied_or_not=kwargs['occupied_or_not'])
        if 'revenue' in kwargs:
            self.state.update_feature("ChargerFeature", revenue=kwargs['revenue'])

    def apply_action(self) -> None:
        """Apply agent action: broadcast pricing decision to environment.
        
        The agent's action (price command) is propagated to the environment.
        The environment uses this price in charging physics decisions.
        """
        received_price = float(self.action.c[0])
        self.state.update_feature("ChargerFeature", charging_price=received_price)

    def compute_power_delivery(self, local_state: dict, ev_soc: float, ev_battery_capacity: float, 
                              ev_demand_remaining: float, p_max_kw: float) -> float:
        """AGENT ECONOMICS: Determine charging power based on EV demand/utility.
        
        This is an economic decision: should this charger deliver power?
        At what rate?
        
        Args:
            local_state: Local charger state from environment
            ev_soc: EV state of charge (fraction)
            ev_battery_capacity: EV battery capacity (kWh)
            ev_demand_remaining: EV remaining demand to charge (kWh)
            p_max_kw: Maximum available power (kW)
        
        Returns:
            Power command in kW
        """
        # If no demand, deliver zero power
        if ev_demand_remaining <= 1e-6:
            return 0.0
        
        # For now: deliver max power if there's demand
        # In a more sophisticated agent, this could be optimized based on price/utility
        return p_max_kw

    def compute_revenue(self, price_charged: float, lmp: float, energy_kwh: float) -> float:
        """AGENT ECONOMICS: Compute revenue from charging transaction.
        
        Revenue = (price - lmp) * energy_delivered
        This represents the profit margin.
        
        Args:
            price_charged: Price charged to EV ($/kWh)
            lmp: Locational marginal price ($/kWh)
            energy_kwh: Energy delivered (kWh)
        
        Returns:
            Revenue in dollars
        """
        return (price_charged - lmp) * energy_kwh


    @Agent.handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        from heron.agents.proxy_agent import PROXY_AGENT_ID, MSG_SET_STATE, STATE_TYPE_LOCAL
        
        self.apply_action()
        scheduler.schedule_message_delivery(
            sender_id=self.agent_id,
            recipient_id=PROXY_AGENT_ID,
            message={MSG_SET_STATE: STATE_TYPE_LOCAL, "body": self.state.to_dict(include_metadata=True)},
        )

    def compute_local_reward(self, local_state: dict) -> float:
        """
        Reward = true revenue from environment simulation.
        
        ChargerFeature.revenue = (price - lmp) * energy_kwh
        This aligns agent objectives with actual economic profit.
        """
        charger_vec = local_state.get("ChargerFeature")
        if charger_vec is None:
            return 0.0
        if isinstance(charger_vec, np.ndarray) and len(charger_vec) >= 4:
            # Vector format: [p_norm, occupied_or_not, price, revenue]
            revenue = float(charger_vec[3])
        else:
            revenue = 0.0
        return float(revenue) if revenue is not None else 0.0
