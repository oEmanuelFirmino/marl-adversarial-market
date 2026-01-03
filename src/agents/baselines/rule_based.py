import numpy as np
import random
from dataclasses import dataclass


class FixedProposer:
    """
    Agente que propõe sempre a mesma ação.
    Essencial para validação técnica do ambiente e debugging.
    """

    def __init__(self, agent_id, obs_space, action_space):
        self.agent_id = agent_id

    def act(self, obs):

        return 2


class FixedRegulator:
    def __init__(self, agent_id, obs_space, action_space):
        self.agent_id = agent_id

    def act(self, obs):
        volatility = obs[0]
        if volatility > 0.7:
            return 2
        elif volatility > 0.3:
            return 1
        return 0


class ThresholdResponder:
    """Oponente determinístico (Legado/Fácil de Hackear)"""

    def __init__(self, agent_id, obs_space, action_space):
        self.agent_id = agent_id

    def act(self, obs):
        urgency = obs[1]
        budget_norm = obs[2]
        price_norm = obs[3]

        if price_norm <= budget_norm:
            return 1
        if urgency > 0.8:
            return 2
        return 0


class StochasticResponder:
    """Oponente com ruído simples (Nível 1 de Dificuldade)"""

    def __init__(self, agent_id, obs_space, action_space, irrationality=0.1):
        self.agent_id = agent_id
        self.irrationality = irrationality

    def act(self, obs):
        urgency = obs[1]
        budget_norm = obs[2]
        price_norm = obs[3]

        can_afford = price_norm <= budget_norm
        wants_to_buy = False

        if can_afford:
            ratio = price_norm / (budget_norm + 1e-5)
            buy_prob = 1.0 - (ratio**3) + (urgency * 0.4)
            if random.random() < buy_prob:
                wants_to_buy = True

        if random.random() < self.irrationality:
            wants_to_buy = not wants_to_buy

        if wants_to_buy:
            return 1

        leave_prob = 0.05 + (urgency if urgency > 0.6 else 0)
        if random.random() < leave_prob:
            return 2
        return 0


@dataclass
class ResponderProfile:
    name: str
    price_sensitivity: float
    patience: float
    irrationality: float
    urgency_factor: float


class DynamicResponder:
    """
    Oponente Camaleão.
    Sorteia um perfil psicológico diferente a cada episódio.
    Obriga a IA a fazer 'Sondagem' nos primeiros passos.
    """

    def __init__(self, agent_id, obs_space, action_space):
        self.agent_id = agent_id
        self.current_profile = None
        self.profiles = [
            ResponderProfile(
                "Cheapskate",
                price_sensitivity=4.0,
                patience=0.3,
                irrationality=0.05,
                urgency_factor=0.1,
            ),
            ResponderProfile(
                "Desperate",
                price_sensitivity=0.8,
                patience=0.8,
                irrationality=0.15,
                urgency_factor=2.5,
            ),
            ResponderProfile(
                "Trader",
                price_sensitivity=2.5,
                patience=0.6,
                irrationality=0.02,
                urgency_factor=0.5,
            ),
            ResponderProfile(
                "Chaos",
                price_sensitivity=1.0,
                patience=0.5,
                irrationality=0.40,
                urgency_factor=0.0,
            ),
        ]
        self.reset_persona()

    def reset_persona(self):
        """Chamado externamente (ex: no env.reset) para trocar a máscara"""
        self.current_profile = random.choice(self.profiles)
        return self.current_profile.name

    def act(self, obs):
        if self.current_profile is None:
            self.reset_persona()

        p = self.current_profile

        urgency = obs[1]
        budget_norm = obs[2]
        price_norm = obs[3]

        ratio = price_norm / (budget_norm + 1e-5)

        if ratio > 1.05:
            base_prob = 0.0
        else:
            base_prob = 1.0 - (ratio**p.price_sensitivity)

        buy_prob = base_prob + (urgency * p.urgency_factor)

        wants_to_buy = random.random() < buy_prob

        if random.random() < p.irrationality:
            wants_to_buy = not wants_to_buy

        if wants_to_buy:
            return 1

        base_leave = 1.0 - p.patience
        leave_prob = base_leave + (urgency * 0.5)

        if random.random() < leave_prob:
            return 2

        return 0
