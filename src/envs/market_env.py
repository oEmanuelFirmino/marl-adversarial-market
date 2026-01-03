import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from src.core.types import AgentID
from src.envs.mechanics import SimpleMarketMechanics


class MarketAdversarialEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "market_adversarial_v1"}

    def __init__(self):
        self.mechanics = SimpleMarketMechanics()
        self.possible_agents = ["responder", "proposer", "regulator"]
        self.agents = self.possible_agents[:]

        self._action_spaces = {
            "responder": spaces.Discrete(3),
            "proposer": spaces.Discrete(5),
            "regulator": spaces.Discrete(3),
        }

        self._observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.state_data = None

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.state_data = self.mechanics.reset()

        observations = {
            agent: self._make_obs(self.state_data, agent) for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):

        internal_actions = {AgentID(k): v for k, v in actions.items()}

        result = self.mechanics.compute_step(self.state_data, internal_actions)
        self.state_data = result.next_state

        observations = {
            agent: self._make_obs(self.state_data, agent) for agent in self.agents
        }

        rewards = {
            agent: result.rewards.get(AgentID(agent), 0.0) for agent in self.agents
        }

        terminations = {agent: result.done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: result.info for agent in self.agents}

        if result.done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _make_obs(self, state, agent_id):

        obs = np.array(
            [
                state.global_volatility,
                state.responder_urgency,
                min(state.responder_budget / 200.0, 1.0),
                min(state.last_transaction_price / 200.0, 1.0),
                state.step_count / 100.0,
            ],
            dtype=np.float32,
        )
        return obs

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]
