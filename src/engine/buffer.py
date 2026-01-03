import torch
import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

        self.belief_probs = []
        self.opponent_actions = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.belief_probs[:]
        del self.opponent_actions[:]

    def add(
        self,
        state,
        action,
        logprob,
        reward,
        state_value,
        done,
        belief_prob=None,
        opp_action=None,
    ):
        """
        Adiciona uma transição ao buffer.
        belief_prob e opp_action são opcionais (None para PPO Vanilla, Preenchidos para Belief PPO).
        """

        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.is_terminals.append(done)

        if belief_prob is not None:
            self.belief_probs.append(belief_prob)

        if opp_action is not None:

            if not isinstance(opp_action, torch.Tensor):
                opp_action = torch.tensor(opp_action, dtype=torch.long)
            self.opponent_actions.append(opp_action)

    def compute_gae_and_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Calcula retornos normalizados e vantagens via GAE.
        Retorna tensores prontos para o PPO.
        """
        rewards = self.rewards
        state_values = self.state_values + [last_value]
        is_terminals = self.is_terminals

        returns = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if is_terminals[i]:
                mask = 0
            else:
                mask = 1

            delta = rewards[i] + gamma * state_values[i + 1] * mask - state_values[i]
            gae = delta + gamma * gae_lambda * mask * gae
            returns.insert(0, gae + state_values[i])

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        if len(self.states) > 0 and isinstance(self.states[0], np.ndarray):
            old_states = torch.tensor(np.array(self.states), dtype=torch.float32)
        else:
            old_states = torch.stack(self.states).detach()

        old_actions = torch.stack(self.actions).detach()
        old_logprobs = torch.stack(self.logprobs).detach()

        return old_states, old_actions, old_logprobs, returns

    def get_belief_data(self):
        """Recupera dados específicos para treino da rede de crença."""
        if not self.belief_probs or not self.opponent_actions:
            raise ValueError(
                "Buffer não contém dados de crença. Verifique se belief_prob/opp_action foram passados no .add()"
            )

        if len(self.states) > 0 and isinstance(self.states[0], np.ndarray):
            states_tensor = torch.tensor(np.array(self.states), dtype=torch.float32)
        else:
            states_tensor = torch.stack(self.states).detach()

        return (
            states_tensor,
            torch.stack(self.belief_probs).detach(),
            torch.stack(self.opponent_actions).detach(),
        )
