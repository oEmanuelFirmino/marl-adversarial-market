import sys
import os
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.market_env import MarketAdversarialEnv
from src.agents.baselines.rule_based import FixedInsurer, ThresholdLead
from src.engine.trainer import BeliefPPOTrainer


def train_belief_system():
    print(">>> Iniciando Treinamento Híbrido (PPO + Belief) <<<")

    env = MarketAdversarialEnv()
    lead = ThresholdLead(
        "lead", env.observation_space("lead"), env.action_space("lead")
    )
    insurer = FixedInsurer(
        "insurer", env.observation_space("insurer"), env.action_space("insurer")
    )

    agent = BeliefPPOTrainer(env, agent_id="broker", opponent_id="lead")

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

            act_brk, log_brk, val_brk, belief_probs = agent.select_action(obs["broker"])

            act_lead = lead.act(obs["lead"])
            act_ins = insurer.act(obs["insurer"])

            actions = {"broker": act_brk, "lead": act_lead, "insurer": act_ins}
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            done = all(terms.values())

            agent.buffer.add(
                torch.tensor(obs["broker"], dtype=torch.float32),
                torch.tensor(act_brk),
                log_brk,
                rewards["broker"],
                torch.tensor(val_brk),
                done,
                belief_probs,
                act_lead,
            )

            obs = next_obs
            ep_reward += rewards["broker"]
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
