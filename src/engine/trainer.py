import torch
import torch.nn as nn
from src.agents.policy.network import ActorCritic, RecurrentActorCritic
from src.agents.belief.predictor import BeliefNetwork, RecurrentBeliefNetwork
from src.engine.buffer import RolloutBuffer


class PPOTrainer:
    """Vanilla PPO (Legado)"""

    def __init__(
        self, env, agent_id="proposer", lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2
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
    """
    Recurrent PPO with TBPTT (Truncated Backpropagation Through Time)
    """

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
        self.env = env
        self.agent_id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.obs_dim = env.observation_space(agent_id).shape[0]
        self.action_dim = env.action_space(agent_id).n
        self.opp_action_dim = env.action_space(opponent_id).n

        self.hidden_dim = 64

        self.belief_net = RecurrentBeliefNetwork(
            self.obs_dim, self.opp_action_dim, self.hidden_dim
        )
        self.policy = RecurrentActorCritic(
            self.obs_dim, self.opp_action_dim, self.action_dim, self.hidden_dim
        )

        self.belief_optimizer = torch.optim.Adam(self.belief_net.parameters(), lr=lr)
        self.belief_loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = RecurrentActorCritic(
            self.obs_dim, self.opp_action_dim, self.action_dim, self.hidden_dim
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()

        self.reset_memory()

    def reset_memory(self):
        self.h_belief = torch.zeros(1, 1, self.hidden_dim)
        self.h_policy = torch.zeros(1, 1, self.hidden_dim)

    def save_checkpoint(self, path):
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
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        self.belief_net.load_state_dict(checkpoint["belief_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_policy"])
        self.belief_optimizer.load_state_dict(checkpoint["optimizer_belief"])

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state)
            has_batch_dim = state_t.dim() > 1
            if not has_batch_dim:
                state_t = state_t.unsqueeze(0)

            belief_probs, self.h_belief = self.belief_net.get_probs(
                state_t, self.h_belief
            )
            action, log_prob, self.h_policy = self.policy_old.get_action(
                state_t, belief_probs, self.h_policy
            )
            val = self.policy_old.get_value(state_t, belief_probs, self.h_policy)

            if not has_batch_dim:
                belief_probs = belief_probs.squeeze(0)

        return action.item(), log_prob, val.item(), belief_probs

    def update(self):

        _, _, _, returns_all = self.buffer.compute_gae_and_returns(0.0, self.gamma)

        seq_len = 10

        avg_belief_loss = 0

        for _ in range(self.K_epochs):
            h_belief = torch.zeros(1, 1, self.hidden_dim)
            epoch_loss = 0
            count = 0

            data_generator = self.buffer.get_sequential_batches(seq_len)

            for batch in data_generator:
                (states, _, _, _, dones, _, opp_actions) = batch

                predicted_logits, h_belief = self.belief_net(states, h_belief)

                loss = self.belief_loss_fn(
                    predicted_logits.view(-1, self.opp_action_dim), opp_actions.view(-1)
                )

                self.belief_optimizer.zero_grad()
                loss.backward()
                self.belief_optimizer.step()

                h_belief = h_belief.detach()

                if torch.any(dones):
                    h_belief = torch.zeros(1, 1, self.hidden_dim)

                epoch_loss += loss.item()
                count += 1

            if count > 0:
                avg_belief_loss += epoch_loss / count

        avg_belief_loss /= self.K_epochs

        for _ in range(self.K_epochs):
            h_policy = torch.zeros(1, 1, self.hidden_dim)
            data_generator = self.buffer.get_sequential_batches(seq_len)
            current_idx = 0

            for batch in data_generator:
                (states, actions, old_logprobs, _, dones, belief_probs, _) = batch

                batch_len = states.shape[1]
                batch_returns = returns_all[current_idx : current_idx + batch_len]
                current_idx += batch_len
                batch_returns = batch_returns.unsqueeze(0).unsqueeze(-1)

                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    states, belief_probs, actions, h_policy
                )

                if old_logprobs.dim() > logprobs.dim():
                    old_logprobs = old_logprobs.squeeze(-1)

                ratios = torch.exp(logprobs - old_logprobs)
                advantages = batch_returns - state_values.detach()

                ratios = ratios.view(-1)
                advantages = advantages.view(-1)
                state_values = state_values.view(-1)
                batch_returns_flat = batch_returns.view(-1)

                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-7
                )

                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )

                loss = (
                    -torch.min(surr1, surr2).mean()
                    + 0.5 * self.mse_loss(state_values, batch_returns_flat)
                    - 0.01 * dist_entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                h_policy = h_policy.detach()

                if torch.any(dones):
                    h_policy = torch.zeros(1, 1, self.hidden_dim)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        self.reset_memory()

        return avg_belief_loss
