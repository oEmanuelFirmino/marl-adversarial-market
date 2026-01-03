import numpy as np
import random


class FixedProposer:
    """
    Agente que propõe sempre a mesma ação (Baseline simples).
    Útil para validação de ambiente.
    """

    def __init__(self, agent_id, obs_space, action_space):
        self.agent_id = agent_id

    def act(self, obs):
        # Retorna ação 2 (Preço Justo/Médio no mapeamento atual)
        return 2


class FixedRegulator:
    def __init__(self, agent_id, obs_space, action_space):
        self.agent_id = agent_id

    def act(self, obs):
        volatility = obs[0]
        if volatility > 0.7:
            return 2  # Taxa Alta
        elif volatility > 0.3:
            return 1  # Taxa Média
        return 0  # Taxa Baixa


class ThresholdResponder:
    """Oponente determinístico antigo (Mantido para legado/comparação)"""

    def __init__(self, agent_id, obs_space, action_space):
        self.agent_id = agent_id

    def act(self, obs):
        urgency = obs[1]
        budget_norm = obs[2]
        offered_price_norm = obs[3]

        # Se preço <= budget, compra
        if offered_price_norm <= budget_norm:
            return 1  # Buy

        # Se urgência alta e não pode pagar, sai
        if urgency > 0.8:
            return 2  # Leave

        return 0  # Wait


class StochasticResponder:
    """
    [NOVO] Um cliente humano e imprevisível.
    Toma decisões baseadas em probabilidade e 'humor'.
    """

    def __init__(self, agent_id, obs_space, action_space, irrationality=0.1):
        self.agent_id = agent_id
        self.irrationality = irrationality  # Chance de agir loucamente

    def act(self, obs):
        # obs vector: [volatility, urgency, budget_norm, price_norm, time, competitor, sentiment]
        urgency = obs[1]
        budget_norm = obs[2]
        offered_price_norm = obs[3]

        # Reconstrutores aproximados
        estimated_budget = budget_norm
        estimated_price = offered_price_norm

        # Lógica Base
        can_afford = estimated_price <= estimated_budget
        wants_to_buy = False

        # Curva de Decisão Probabilística (Sigmoide)
        if can_afford:
            # Se preço é muito menor que o budget, chance muito alta.
            # Se preço está no limite, chance cai.
            ratio = estimated_price / (estimated_budget + 1e-5)
            buy_prob = 1.0 - (ratio**3)  # Curva agressiva

            # Fator Urgência aumenta chance
            buy_prob += urgency * 0.4

            if random.random() < buy_prob:
                wants_to_buy = True

        # --- INJEÇÃO DE CAOS (Irracionalidade) ---
        # Simula erro humano, mau humor ou desespero
        if random.random() < self.irrationality:
            wants_to_buy = not wants_to_buy

        if wants_to_buy:
            return 1  # Buy

        # Se não comprou:
        # Urgência alta ou tédio faz sair
        leave_prob = 0.05 + (urgency if urgency > 0.6 else 0)
        if random.random() < leave_prob:
            return 2  # Leave

        return 0  # Wait
