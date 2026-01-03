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
        rewards = self.rewards
        state_values = self.state_values + [last_value]
        is_terminals = self.is_terminals

        returns = []
        gae = 0

        for i in reversed(range(len(rewards))):
            mask = 0 if is_terminals[i] else 1
            delta = rewards[i] + gamma * state_values[i + 1] * mask - state_values[i]
            gae = delta + gamma * gae_lambda * mask * gae
            returns.insert(0, gae + state_values[i])

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        if len(self.states) > 0 and isinstance(self.states[0], np.ndarray):
            states_t = torch.tensor(np.array(self.states), dtype=torch.float32)
        else:
            states_t = torch.stack(self.states).detach()

        actions_t = torch.stack(self.actions).detach()
        logprobs_t = torch.stack(self.logprobs).detach()

        return states_t, actions_t, logprobs_t, returns

    def get_sequential_batches(self, seq_len=10):
        """
        Gerador para TBPTT. Produz batches de [Batch=1, Seq_Len, Features].
        """

        if len(self.states) > 0 and isinstance(self.states[0], np.ndarray):
            states_t = torch.tensor(np.array(self.states), dtype=torch.float32)
        else:
            states_t = torch.stack(self.states).detach()

        actions_t = torch.stack(self.actions).detach()
        logprobs_t = torch.stack(self.logprobs).detach()
        values_t = torch.tensor(self.state_values, dtype=torch.float32)
        dones_t = torch.tensor(self.is_terminals, dtype=torch.float32)

        b_probs_t = torch.stack(self.belief_probs).detach()
        opp_act_t = torch.stack(self.opponent_actions).detach()

        total_steps = len(self.states)
        num_sequences = total_steps // seq_len
        cutoff = num_sequences * seq_len

        for i in range(0, cutoff, seq_len):
            end = i + seq_len

            yield (
                states_t[i:end].unsqueeze(0),
                actions_t[i:end].unsqueeze(0),
                logprobs_t[i:end].unsqueeze(0),
                values_t[i:end].unsqueeze(0),
                dones_t[i:end].unsqueeze(0),
                b_probs_t[i:end].unsqueeze(0),
                opp_act_t[i:end].unsqueeze(0),
            )
