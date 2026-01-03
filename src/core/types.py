from enum import Enum, auto
from dataclasses import dataclass
from typing import NewType, Dict, Any


class AgentType(Enum):
    PROPOSER = "proposer"
    RESPONDER = "responder"
    REGULATOR = "regulator"


class ActionType(Enum):
    WAIT = 0
    BUY = 1
    LEAVE = 2


AgentID = NewType("AgentID", str)


@dataclass(frozen=True)
class MarketState:
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

    last_transaction_price: float
    transaction_occurred: bool

    last_opponent_action: float


@dataclass
class StepResult:
    next_state: MarketState
    rewards: Dict[AgentID, float]
    done: bool
    info: Dict[AgentID, Any]
