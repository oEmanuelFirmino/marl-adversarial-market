import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action(self, obs):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs, action):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs)
        return action_logprobs, state_values, dist_entropy

    def get_value(self, obs):
        return self.critic(obs)


class AugmentedActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, belief_dim: int, action_dim: int, hidden_dim: int = 64
    ):
        super().__init__()
        self.actor_input_dim = obs_dim + belief_dim
        self.actor = nn.Sequential(
            nn.Linear(self.actor_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action(self, obs, belief_probs):
        augmented_input = torch.cat([obs, belief_probs], dim=-1)
        logits = self.actor(augmented_input)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs, belief_probs, action):
        augmented_input = torch.cat([obs, belief_probs], dim=-1)
        logits = self.actor(augmented_input)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs)
        return action_logprobs, state_values, dist_entropy

    def get_value(self, obs):
        return self.critic(obs)


class RecurrentActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, belief_dim: int, action_dim: int, hidden_dim: int = 64
    ):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + belief_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, belief_probs, hidden_state):

        x = torch.cat([obs, belief_probs], dim=-1)
        x = torch.tanh(self.fc1(x))

        is_sequence = x.dim() == 3

        if not is_sequence:
            x = x.unsqueeze(1)

        gru_out, h_n = self.gru(x, hidden_state)

        action_logits = self.actor(gru_out)
        state_values = self.critic(gru_out)

        if not is_sequence:

            action_logits = action_logits[:, -1, :]
            state_values = state_values[:, -1, :]

        return action_logits, state_values, h_n

    def get_action(self, obs, belief_probs, hidden_state):
        logits, _, new_hidden = self.forward(obs, belief_probs, hidden_state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), new_hidden

    def get_value(self, obs, belief_probs, hidden_state):
        _, val, _ = self.forward(obs, belief_probs, hidden_state)
        return val

    def evaluate(self, obs, belief_probs, action, hidden_state):
        """
        Avalia uma sequência inteira de ações.
        """
        logits, val, _ = self.forward(obs, belief_probs, hidden_state)
        dist = Categorical(logits=logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, val, dist_entropy
