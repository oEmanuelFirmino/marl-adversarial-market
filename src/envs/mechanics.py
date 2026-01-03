import numpy as np
from src.core.interfaces import MarketPhysics
from src.core.types import MarketState, StepResult, AgentID


class SimpleMarketMechanics(MarketPhysics):
    def __init__(self):
        self.max_steps = 100
        self.base_price = 100.0

    def reset(self) -> MarketState:
        return MarketState(
            step_count=0,
            global_volatility=np.random.uniform(0.1, 0.3),
            lead_budget=np.random.uniform(80, 150),
            lead_urgency=np.random.uniform(0.0, 1.0),
            broker_cash=1000.0,
            broker_active_deals=0,
            insurer_capital=5000.0,
            insurer_risk_threshold=0.5,
            last_transaction_price=0.0,
            transaction_occurred=False,
        )

    def compute_step(self, state: MarketState, actions: dict) -> StepResult:
        """
        Lógica:
        1. Broker propõe Preço (Baseado na ação: Baixo, Médio, Alto).
        2. Seguradora define Taxa (Baseado na ação: Baixa, Média, Alta).
        3. Lead decide (Comprar ou Esperar) baseado na utilidade (Preço + Urgência).
        """

        act_broker = actions.get(AgentID("broker"), 0)
        act_insurer = actions.get(AgentID("insurer"), 0)
        act_lead = actions.get(AgentID("lead"), 0)

        price_multipliers = {0: 1.0, 1: 0.8, 2: 1.0, 3: 1.2, 4: 1.5}
        offered_price = self.base_price * price_multipliers.get(act_broker, 1.0)

        risk_multipliers = {0: 1.0, 1: 0.9, 2: 1.3}
        insurance_cost = 10.0 * risk_multipliers.get(act_insurer, 1.0)

        final_price = offered_price + insurance_cost

        deal_done = False
        rewards = {
            AgentID("lead"): -0.1,
            AgentID("broker"): -0.1,
            AgentID("insurer"): 0.0,
        }

        if act_lead == 1:
            if final_price <= state.lead_budget:
                deal_done = True

                rewards[AgentID("lead")] = (state.lead_budget - final_price) + (
                    state.lead_urgency * 10
                )
                rewards[AgentID("broker")] = final_price - 50.0
                rewards[AgentID("insurer")] = insurance_cost - (
                    2.0 if state.global_volatility > 0.5 else 0.0
                )
            else:

                rewards[AgentID("lead")] -= 1.0

        elif act_lead == 2:
            rewards[AgentID("lead")] -= 2.0

        new_state = MarketState(
            step_count=state.step_count + 1,
            global_volatility=state.global_volatility,
            lead_budget=state.lead_budget,
            lead_urgency=min(1.0, state.lead_urgency + 0.05),
            broker_cash=state.broker_cash
            + (rewards[AgentID("broker")] if deal_done else 0),
            broker_active_deals=state.broker_active_deals + (1 if deal_done else 0),
            insurer_capital=state.insurer_capital
            + (rewards[AgentID("insurer")] if deal_done else 0),
            insurer_risk_threshold=state.insurer_risk_threshold,
            last_transaction_price=final_price if deal_done else 0.0,
            transaction_occurred=deal_done,
        )

        done = (new_state.step_count >= self.max_steps) or deal_done or (act_lead == 2)

        return StepResult(
            next_state=new_state,
            rewards=rewards,
            done=done,
            info={"price": final_price, "deal": deal_done},
        )
