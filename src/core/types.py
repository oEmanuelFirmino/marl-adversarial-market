from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, NewType

AgentID = NewType("AgentID", str)


class AgentType(IntEnum):
    RESPONDER = 0
    PROPOSER = 1
    REGULATOR = 2


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

    competitor_intensity: float
    market_sentiment: float

    responder_budget: float
    responder_urgency: float

    proposer_cash: float
    proposer_active_deals: int

    regulator_capital: float
    regulator_risk_threshold: float

    last_transaction_price: float = 0.0
    transaction_occurred: bool = False


@dataclass
class StepResult:
    """DTO desacoplado do Gym para retorno da Mecânica."""

    next_state: MarketState
    rewards: Dict[AgentID, float]
    done: bool
    info: Dict[str, any]
