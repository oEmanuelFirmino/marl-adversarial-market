import functools
import gymnasium
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from src.core.types import AgentID, MarketState
from src.envs.mechanics import SimpleMarketMechanics


class MarketAdversarialEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "marl_adversarial_market_v1"}

    def __init__(self):
        self.possible_agents = [
            AgentID("proposer"),
            AgentID("responder"),
            AgentID("regulator"),
        ]
        self.agents = self.possible_agents[:]
        self.mechanics = SimpleMarketMechanics()
        self.state_data = None

        # Obs Space: 7 floats (Volatilidade, Urgência, Budget, Preço, Tempo, Concorrência, Sentimento)
        self._observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self._action_spaces = {
            # [CORREÇÃO] Proposer agora tem 4 ações (0 a 3) em vez de 5
            AgentID("proposer"): spaces.Discrete(4),
            # Responder (0:Wait, 1:Buy, 2:Leave)
            AgentID("responder"): spaces.Discrete(3),
            # Regulator (0:Baixo, 1:Médio, 2:Alto)
            AgentID("regulator"): spaces.Discrete(3),
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.state_data = self.mechanics.reset()

        observations = {
            agent: self._make_obs(self.state_data, agent) for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if not actions:
            return {}, {}, {}, {}, {}

        result = self.mechanics.compute_step(self.state_data, actions)
        self.state_data = result.next_state

        observations = {
            agent: self._make_obs(self.state_data, agent) for agent in self.agents
        }
        rewards = result.rewards

        terminations = {agent: result.done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: result.info for agent in self.agents}

        if result.done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _make_obs(self, state: MarketState, agent_id: AgentID):
        # Normalização simples para [0, 1]
        obs = np.array(
            [
                state.global_volatility,
                state.responder_urgency,
                min(state.responder_budget / 200.0, 1.0),
                min(state.last_transaction_price / 200.0, 1.0),
                state.step_count / 100.0,
                # Novos
                state.competitor_intensity,
                state.market_sentiment / 2.0,
            ],
            dtype=np.float32,
        )
        return obs
