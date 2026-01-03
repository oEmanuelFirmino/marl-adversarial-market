import torch
import torch.nn as nn
import numpy as np
from src.agents.policy.network import AugmentedActorCritic
from src.agents.belief.predictor import BeliefNetwork
from src.engine.buffer import RolloutBuffer


class BeliefPPOTrainer:
    def __init__(
        self,
        env,
        agent_id="proposer",
        opponent_id="responder",
        lr=0.002,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4,
    ):
        """
        Orquestrador de Treinamento Híbrido (PPO + Belief Learning).

        Args:
            env: Instância do ambiente PettingZoo/Gym.
            agent_id: ID do agente sendo treinado (ex: 'proposer').
            opponent_id: ID do oponente alvo da crença (ex: 'responder').
            lr: Learning Rate compartilhado.
            gamma: Fator de desconto.
            eps_clip: Parâmetro de clip do PPO (estabilidade).
            K_epochs: Quantas vezes reutilizar o batch de dados por update.
        """
        self.env = env
        self.agent_id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.obs_dim = env.observation_space(agent_id).shape[0]
        self.action_dim = env.action_space(agent_id).n

        self.opp_action_dim = env.action_space(opponent_id).n

        self.belief_net = BeliefNetwork(self.obs_dim, self.opp_action_dim)
        self.belief_optimizer = torch.optim.Adam(self.belief_net.parameters(), lr=lr)
        self.belief_loss_fn = nn.CrossEntropyLoss()

        self.policy = AugmentedActorCritic(
            self.obs_dim, self.opp_action_dim, self.action_dim
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = AugmentedActorCritic(
            self.obs_dim, self.opp_action_dim, self.action_dim
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        """
        Executa o forward pass completo para inferência/coleta de dados.
        Retorna:
            action: Ação escolhida (int)
            log_prob: Log probabilidade da ação (para PPO)
            val: Valor do estado V(s) (para GAE)
            belief_probs: Vetor de probabilidades da crença (input para a Policy)
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state)

            belief_probs = self.belief_net.get_probs(state_t)

            action, log_prob = self.policy_old.get_action(state_t, belief_probs)
            val = self.policy_old.get_value(state_t)

        return action.item(), log_prob, val.item(), belief_probs

    def update(self):
        """
        Realiza a atualização dos pesos das duas redes (Belief e Policy).
        """

        old_states, old_actions, old_logprobs, returns = (
            self.buffer.compute_gae_and_returns(0.0, self.gamma)
        )

        b_states, b_probs_old, b_targets = self.buffer.get_belief_data()

        avg_belief_loss = 0
        for _ in range(self.K_epochs):

            predicted_logits = self.belief_net(b_states)

            belief_loss = self.belief_loss_fn(predicted_logits, b_targets)

            self.belief_optimizer.zero_grad()
            belief_loss.backward()
            self.belief_optimizer.step()

            avg_belief_loss += belief_loss.item()

        avg_belief_loss /= self.K_epochs

        for _ in range(self.K_epochs):

            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, b_probs_old, old_actions
            )

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = returns - state_values.detach().squeeze()

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.mse_loss(state_values.squeeze(), returns)
                - 0.01 * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()

        return avg_belief_loss
