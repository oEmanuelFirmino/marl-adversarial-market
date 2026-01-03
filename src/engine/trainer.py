import torch
import torch.nn as nn
from src.agents.policy.network import ActorCritic, AugmentedActorCritic
from src.agents.belief.predictor import BeliefNetwork
from src.engine.buffer import RolloutBuffer


class PPOTrainer:
    def __init__(
        self, env, agent_id="broker", lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2
    ):
        self.env = env
        self.agent_id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.obs_dim = env.observation_space(agent_id).shape[0]
        self.action_dim = env.action_space(agent_id).n

        self.policy = ActorCritic(self.obs_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(self.obs_dim, self.action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, log_prob = self.policy_old.get_action(state)
            val = self.policy_old.get_value(state)
        return action.item(), log_prob, val.item()

    def update(self):
        old_states, old_actions, old_logprobs, returns = (
            self.buffer.compute_gae_and_returns(0.0, self.gamma)
        )

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
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


class BeliefPPOTrainer:
    def __init__(
        self,
        env,
        agent_id="broker",
        opponent_id="lead",
        lr=0.002,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4,
    ):
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
        with torch.no_grad():
            state_t = torch.FloatTensor(state)

            has_batch_dim = state_t.dim() > 1
            if not has_batch_dim:
                state_t = state_t.unsqueeze(0)

            belief_probs = self.belief_net.get_probs(state_t)

            action, log_prob = self.policy_old.get_action(state_t, belief_probs)
            val = self.policy_old.get_value(state_t)

            if not has_batch_dim:
                belief_probs = belief_probs.squeeze(0)

        return action.item(), log_prob, val.item(), belief_probs

    def update(self):

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

    def save_checkpoint(self, path):
        """Salva os pesos da Política e da Rede de Crença."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "belief_state_dict": self.belief_net.state_dict(),
                "optimizer_policy": self.optimizer.state_dict(),
                "optimizer_belief": self.belief_optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path):
        """Carrega os pesos salvos."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        self.belief_net.load_state_dict(checkpoint["belief_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_policy"])
        self.belief_optimizer.load_state_dict(checkpoint["optimizer_belief"])
