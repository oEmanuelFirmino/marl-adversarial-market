from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    """Interface base para qualquer agente do sistema."""

    def __init__(self, name: str, observation_space, action_space):
        self.name = name
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """Recebe observação, retorna índice da ação."""
        pass

    def reset(self):
        """Opcional: Resetar estado interno do agente (memória, LSTM, etc)."""
        pass
