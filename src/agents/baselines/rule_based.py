import numpy as np
from src.agents.base import Agent


class FixedBroker(Agent):
    """Corretor Conservador: Sempre mantém o preço base."""

    def act(self, observation: np.ndarray) -> int:
        # Action Space Broker: 0:Wait, 1:Discount, 2:Base, 3:Increase, 4:Abusive
        # Estratégia: Sempre oferta o preço padrão (2)
        return 2


class FixedInsurer(Agent):
    """Seguradora Padrão: Mantém risco moderado."""

    def act(self, observation: np.ndarray) -> int:
        # Action Space Insurer: 0:Standard, 1:LowRisk, 2:HighRisk
        return 0


class ThresholdLead(Agent):
    """
    Lead Racional:
    - Compra se (Orçamento > Último Preço) E (Urgência > Minima).
    - Sai do mercado se o tempo estiver acabando.
    """

    def __init__(self, name, observation_space, action_space, min_urgency=0.2):
        super().__init__(name, observation_space, action_space)
        self.min_urgency = min_urgency

    def act(self, observation: np.ndarray) -> int:
        # Obs Vector: [Volatility, Urgency, Budget_Norm, Last_Price_Norm, Step_Norm]
        urgency = observation[1]
        budget_norm = observation[2]
        last_price_norm = observation[3]
        step_norm = observation[4]

        # Regra 1: Se o jogo está acabando (>90%), sai fora (Action 2: Leave)
        if step_norm > 0.9:
            return 2

        # Regra 2: Decisão de Compra
        # Se tem dinheiro E tem um pouco de pressa
        # Nota: Usamos last_price como proxy do preço atual
        price_acceptable = last_price_norm <= budget_norm

        # Action Space Lead: 0:Wait, 1:Buy, 2:Leave
        if price_acceptable and urgency > self.min_urgency:
            return 1  # Tentar Comprar

        return 0  # Esperar (Wait)
