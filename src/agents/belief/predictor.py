import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefNetwork(nn.Module):
    """Legacy Feedforward Network"""

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
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)


class RecurrentBeliefNetwork(nn.Module):
    """TBPTT Ready Network"""

    def __init__(self, obs_dim: int, opponent_action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, opponent_action_dim)

    def forward(self, obs, hidden_state):
        x = torch.relu(self.fc1(obs))

        is_sequence = x.dim() == 3
        if not is_sequence:
            x = x.unsqueeze(1)

        gru_out, h_n = self.gru(x, hidden_state)

        logits = self.fc2(gru_out)

        if not is_sequence:
            logits = logits[:, -1, :]

        return logits, h_n

    def get_probs(self, obs, hidden_state):
        logits, h_n = self.forward(obs, hidden_state)
        return F.softmax(logits, dim=-1), h_n
