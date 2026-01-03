from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, NewType

AgentID = NewType("AgentID", str)


class AgentType(IntEnum):
    LEAD = 0
    BROKER = 1
    INSURER = 2


class ActionType(IntEnum):
    WAIT = 0
    ACT_LOW = 1
    ACT_MEDIUM = 2
    ACT_HIGH = 3
    REJECT = 4


@dataclass
class MarketState:
    """O Estado Global da Verdade (Oculto para os agentes)."""

    step_count: int
    global_volatility: float

    lead_budget: float
    lead_urgency: float

    broker_cash: float
    broker_active_deals: int

    insurer_capital: float
    insurer_risk_threshold: float

    last_transaction_price: float = 0.0
    transaction_occurred: bool = False


@dataclass
class StepResult:
    """DTO desacoplado do Gym para retorno da Mecânica."""

    next_state: MarketState
    rewards: Dict[AgentID, float]
    done: bool
    info: Dict[str, any]
