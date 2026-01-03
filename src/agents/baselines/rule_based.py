import numpy as np
from src.agents.base import Agent


class FixedProposer(Agent):
    """Corretor Conservador: Sempre mantém o preço base."""

    def act(self, observation: np.ndarray) -> int:

        return 2


class FixedRegulator(Agent):
    """Seguradora Padrão: Mantém risco moderado."""

    def act(self, observation: np.ndarray) -> int:

        return 0


class ThresholdResponder(Agent):
    """
    Responder Racional:
    - Compra se (Orçamento > Último Preço) E (Urgência > Minima).
    - Sai do mercado se o tempo estiver acabando.
    """

    def __init__(self, name, observation_space, action_space, min_urgency=0.2):
        super().__init__(name, observation_space, action_space)
        self.min_urgency = min_urgency

    def act(self, observation: np.ndarray) -> int:

        urgency = observation[1]
        budget_norm = observation[2]
        last_price_norm = observation[3]
        step_norm = observation[4]

        if step_norm > 0.9:
            return 2

        price_acceptable = last_price_norm <= budget_norm

        if price_acceptable and urgency > self.min_urgency:
            return 1

        return 0
