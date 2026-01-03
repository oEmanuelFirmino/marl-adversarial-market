import torch
import torch.nn as nn
from torch.distributions import Categorical


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
        """
        obs: [Batch, Obs_Dim]
        belief_probs: [Batch, Belief_Dim] -> Output do BeliefNetwork
        """

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
