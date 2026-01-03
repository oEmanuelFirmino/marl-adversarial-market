from abc import ABC, abstractmethod
from typing import Dict
from src.core.types import MarketState, StepResult, AgentID


class MarketPhysics(ABC):
    """Contrato para a lógica determinística (Business Logic)."""

    @abstractmethod
    def reset(self) -> MarketState:
        """Reinicia o estado global do mercado."""
        pass

    @abstractmethod
    def compute_step(
        self, current_state: MarketState, actions: Dict[AgentID, int]
    ) -> StepResult:
        """
        Recebe estado + ações conjuntas.
        Retorna novo estado + recompensas (sem dependência de libs de RL).
        """
        pass
