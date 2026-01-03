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

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, logprob, reward, state_value, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.is_terminals.append(done)

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

        old_states = torch.stack(self.states).detach()
        old_actions = torch.stack(self.actions).detach()
        old_logprobs = torch.stack(self.logprobs).detach()

        return old_states, old_actions, old_logprobs, returns
