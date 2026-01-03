import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.market_env import MarketAdversarialEnv
from src.agents.baselines.rule_based import FixedRegulator, ThresholdResponder
from src.engine.trainer import PPOTrainer


def train():
    print(">>> Iniciando Treinamento PPO (Fase 3) <<<")

    env = MarketAdversarialEnv()

    responder_agent = ThresholdResponder(
        "responder", env.observation_space("responder"), env.action_space("responder")
    )
    regulator_agent = FixedRegulator(
        "regulator", env.observation_space("regulator"), env.action_space("regulator")
    )

    ppo_agent = PPOTrainer(env, agent_id="proposer", lr=0.002, gamma=0.95)

    max_episodes = 2000
    update_timestep = 500

    time_step = 0
    history_rewards = []
    running_reward = 0

    for i_episode in range(1, max_episodes + 1):
        obs, _ = env.reset()
        current_ep_reward = 0

        while True:

            action_proposer, log_prob, val = ppo_agent.select_action(obs["proposer"])

            action_responder = responder_agent.act(obs["responder"])
            action_regulator = regulator_agent.act(obs["regulator"])

            actions = {
                "proposer": action_proposer,
                "responder": action_responder,
                "regulator": action_regulator,
            }

            next_obs, rewards, terms, truncs, _ = env.step(actions)
            done = all(terms.values()) or all(truncs.values())

            ppo_agent.buffer.add(
                torch.tensor(obs["proposer"], dtype=torch.float32),
                torch.tensor(action_proposer),
                log_prob,
                rewards["proposer"],
                torch.tensor(val),
                done,
            )

            obs = next_obs
            current_ep_reward += rewards["proposer"]
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
