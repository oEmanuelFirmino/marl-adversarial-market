import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


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

    def forward(self):
        raise NotImplementedError("Use get_action or get_value")

    def get_action(self, obs):
        """Amostra uma ação e retorna log_prob para o treino."""
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs, action):
        """Avalia uma ação passada (usado no update do PPO)."""
        logits = self.actor(obs)
        dist = Categorical(logits=logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs)

        return action_logprobs, state_values, dist_entropy

    def get_value(self, obs):
        return self.critic(obs)
