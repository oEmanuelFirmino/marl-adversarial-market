import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Versão Antiga ---
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
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)


# --- NOVA VERSÃO (Recorrente) ---
class RecurrentBeliefNetwork(nn.Module):
    def __init__(self, obs_dim: int, opponent_action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, opponent_action_dim)

    def forward(self, obs, hidden_state):
        # 1. Feature Extraction
        x = torch.relu(self.fc1(obs))

        # 2. Sequence Handling
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 3. Memory Update
        gru_out, h_n = self.gru(x, hidden_state)

        # 4. Prediction
        gru_out = gru_out[:, -1, :]
        logits = self.fc2(gru_out)

        return logits, h_n

    def get_probs(self, obs, hidden_state):
        logits, h_n = self.forward(obs, hidden_state)
        return F.softmax(logits, dim=-1), h_n
