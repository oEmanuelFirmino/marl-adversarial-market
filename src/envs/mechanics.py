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
            responder_budget=np.random.uniform(80, 150),
            responder_urgency=np.random.uniform(0.0, 1.0),
            proposer_cash=1000.0,
            proposer_active_deals=0,
            regulator_capital=5000.0,
            regulator_risk_threshold=0.5,
            last_transaction_price=0.0,
            transaction_occurred=False,
        )

    def compute_step(self, state: MarketState, actions: dict) -> StepResult:
        """
        Lógica:
        1. Proposer propõe Preço (Baseado na ação: Baixo, Médio, Alto).
        2. Seguradora define Taxa (Baseado na ação: Baixa, Média, Alta).
        3. Responder decide (Comprar ou Esperar) baseado na utilidade (Preço + Urgência).
        """

        act_proposer = actions.get(AgentID("proposer"), 0)
        act_regulator = actions.get(AgentID("regulator"), 0)
        act_responder = actions.get(AgentID("responder"), 0)

        price_multipliers = {0: 1.0, 1: 0.8, 2: 1.0, 3: 1.2, 4: 1.5}
        offered_price = self.base_price * price_multipliers.get(act_proposer, 1.0)

        risk_multipliers = {0: 1.0, 1: 0.9, 2: 1.3}
        insurance_cost = 10.0 * risk_multipliers.get(act_regulator, 1.0)

        final_price = offered_price + insurance_cost

        deal_done = False
        rewards = {
            AgentID("responder"): -0.1,
            AgentID("proposer"): -0.1,
            AgentID("regulator"): 0.0,
        }

        if act_responder == 1:
            if final_price <= state.responder_budget:
                deal_done = True

                rewards[AgentID("responder")] = (state.responder_budget - final_price) + (
                    state.responder_urgency * 10
                )
                rewards[AgentID("proposer")] = final_price - 50.0
                rewards[AgentID("regulator")] = insurance_cost - (
                    2.0 if state.global_volatility > 0.5 else 0.0
                )
            else:

                rewards[AgentID("responder")] -= 1.0

        elif act_responder == 2:
            rewards[AgentID("responder")] -= 2.0

        new_state = MarketState(
            step_count=state.step_count + 1,
            global_volatility=state.global_volatility,
            responder_budget=state.responder_budget,
            responder_urgency=min(1.0, state.responder_urgency + 0.05),
            proposer_cash=state.proposer_cash
            + (rewards[AgentID("proposer")] if deal_done else 0),
            proposer_active_deals=state.proposer_active_deals + (1 if deal_done else 0),
            regulator_capital=state.regulator_capital
            + (rewards[AgentID("regulator")] if deal_done else 0),
            regulator_risk_threshold=state.regulator_risk_threshold,
            last_transaction_price=final_price if deal_done else 0.0,
            transaction_occurred=deal_done,
        )

        done = (new_state.step_count >= self.max_steps) or deal_done or (act_responder == 2)

        return StepResult(
            next_state=new_state,
            rewards=rewards,
            done=done,
            info={"price": final_price, "deal": deal_done},
        )
