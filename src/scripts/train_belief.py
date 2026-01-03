import sys
import os
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.market_env import MarketAdversarialEnv
from src.agents.baselines.rule_based import FixedRegulator, ThresholdResponder
from src.engine.trainer import BeliefPPOTrainer


def train_belief_system():
    print(">>> Iniciando Treinamento Híbrido (PPO + Belief) <<<")

    env = MarketAdversarialEnv()
    responder = ThresholdResponder(
        "responder", env.observation_space("responder"), env.action_space("responder")
    )
    regulator = FixedRegulator(
        "regulator", env.observation_space("regulator"), env.action_space("regulator")
    )

    agent = BeliefPPOTrainer(env, agent_id="proposer", opponent_id="responder")

    history_rewards = []
    history_belief_loss = []

    max_episodes = 2000
    update_timestep = 600
    time_step = 0

    running_loss = 1.0

    for ep in range(max_episodes):
        obs, _ = env.reset()
        ep_reward = 0

        while True:

            act_brk, log_brk, val_brk, belief_probs = agent.select_action(obs["proposer"])

            act_responder = responder.act(obs["responder"])
            act_ins = regulator.act(obs["regulator"])

            actions = {"proposer": act_brk, "responder": act_responder, "regulator": act_ins}
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            done = all(terms.values())

            agent.buffer.add(
                torch.tensor(obs["proposer"], dtype=torch.float32),
                torch.tensor(act_brk),
                log_brk,
                rewards["proposer"],
                torch.tensor(val_brk),
                done,
                belief_probs,
                act_responder,
            )

            obs = next_obs
            ep_reward += rewards["proposer"]
            time_step += 1

            if time_step % update_timestep == 0:
                loss = agent.update()
                history_belief_loss.append(loss)

            if done:
                break

        history_rewards.append(ep_reward)

        if ep % 100 == 0:
            avg_loss = sum(history_belief_loss[-10:]) / 10 if history_belief_loss else 0
            print(f"Ep {ep} | Reward: {ep_reward:.2f} | Belief Loss: {avg_loss:.4f}")

    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Updates")
    ax1.set_ylabel("Belief Loss (Quanto menor, melhor)", color=color)
    ax1.plot(history_belief_loss, color=color, alpha=0.6)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Reward (Quanto maior, melhor)", color=color)

    smooth_rewards = [
        sum(history_rewards[i : i + 50]) / 50
        for i in range(0, len(history_rewards), 50)
    ]
    ax2.plot(range(len(smooth_rewards)), smooth_rewards, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Evolução: Inteligência Estratégica vs Lucro")
    plt.savefig("belief_validation.png")
    print(">>> Resultados salvos em 'belief_validation.png'")


if __name__ == "__main__":
    train_belief_system()
