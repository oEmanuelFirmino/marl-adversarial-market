import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.market_env import MarketAdversarialEnv
from src.agents.baselines.rule_based import FixedInsurer, ThresholdLead
from src.engine.trainer import PPOTrainer


def train():
    print(">>> Iniciando Treinamento PPO (Fase 3) <<<")

    env = MarketAdversarialEnv()

    lead_agent = ThresholdLead(
        "lead", env.observation_space("lead"), env.action_space("lead")
    )
    insurer_agent = FixedInsurer(
        "insurer", env.observation_space("insurer"), env.action_space("insurer")
    )

    ppo_agent = PPOTrainer(env, agent_id="broker", lr=0.002, gamma=0.95)

    max_episodes = 2000
    update_timestep = 500

    time_step = 0
    history_rewards = []
    running_reward = 0

    for i_episode in range(1, max_episodes + 1):
        obs, _ = env.reset()
        current_ep_reward = 0

        while True:

            action_broker, log_prob, val = ppo_agent.select_action(obs["broker"])

            action_lead = lead_agent.act(obs["lead"])
            action_insurer = insurer_agent.act(obs["insurer"])

            actions = {
                "broker": action_broker,
                "lead": action_lead,
                "insurer": action_insurer,
            }

            next_obs, rewards, terms, truncs, _ = env.step(actions)
            done = all(terms.values()) or all(truncs.values())

            ppo_agent.buffer.add(
                torch.tensor(obs["broker"], dtype=torch.float32),
                torch.tensor(action_broker),
                log_prob,
                rewards["broker"],
                torch.tensor(val),
                done,
            )

            obs = next_obs
            current_ep_reward += rewards["broker"]
            time_step += 1

            if time_step % update_timestep == 0:
                ppo_agent.update()

            if done:
                break

        running_reward = 0.05 * current_ep_reward + 0.95 * running_reward
        history_rewards.append(running_reward)

        if i_episode % 100 == 0:
            print(f"Episódio {i_episode}\t Média Reward Móvel: {running_reward:.2f}")

    plt.plot(history_rewards)
    plt.title("Curva de Aprendizado PPO (Corretor)")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa (Média Móvel)")
    plt.grid(True)
    plt.savefig("ppo_training_curve.png")
    print(">>> Treinamento Finalizado. Gráfico salvo.")


if __name__ == "__main__":
    train()
