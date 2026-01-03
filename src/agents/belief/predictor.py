import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefNetwork(nn.Module):
    def __init__(self, obs_dim: int, opponent_action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, opponent_action_dim),
        )

    def forward(self, obs):
        return self.net(obs)

    def get_probs(self, obs):
        """Retorna probabilidades normalizadas para injetar na Policy."""
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)
