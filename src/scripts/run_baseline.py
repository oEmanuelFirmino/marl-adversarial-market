import sys
import os
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.market_env import MarketAdversarialEnv
from src.agents.baselines.rule_based import FixedBroker, FixedInsurer, ThresholdLead


def run_simulation(num_episodes=50):
    env = MarketAdversarialEnv()

    agents_map = {
        "lead": ThresholdLead(
            "lead", env.observation_space("lead"), env.action_space("lead")
        ),
        "broker": FixedBroker(
            "broker", env.observation_space("broker"), env.action_space("broker")
        ),
        "insurer": FixedInsurer(
            "insurer", env.observation_space("insurer"), env.action_space("insurer")
        ),
    }

    history = {"broker_rewards": [], "ep_lengths": [], "deals_count": 0}

    print(f">>> Rodando {num_episodes} episódios com Baseline Agents <<<")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward_broker = 0
        steps = 0

        while not done:

            actions = {}
            for agent_id in env.agents:
                agent = agents_map[agent_id]
                actions[agent_id] = agent.act(obs[agent_id])

            obs, rewards, terms, truncs, infos = env.step(actions)

            ep_reward_broker += rewards.get("broker", 0)
            steps += 1

            if infos["lead"].get("deal"):
                history["deals_count"] += 1

            done = all(terms.values()) or all(truncs.values())

        history["broker_rewards"].append(ep_reward_broker)
        history["ep_lengths"].append(steps)

        if (ep + 1) % 10 == 0:
            print(
                f"Episódio {ep+1}: Steps={steps}, Broker Reward={ep_reward_broker:.2f}"
            )

    return history


def plot_results(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history["broker_rewards"], label="Broker Reward", color="blue", alpha=0.7)
    ax1.axhline(
        y=np.mean(history["broker_rewards"]), color="r", linestyle="--", label="Média"
    )
    ax1.set_title("Estabilidade de Lucro (Corretor Fixo)")
    ax1.set_xlabel("Episódio")
    ax1.set_ylabel("Payoff Acumulado")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(history["ep_lengths"], bins=10, color="green", alpha=0.7)
    ax2.set_title("Distribuição da Duração da Negociação")
    ax2.set_xlabel("Steps até Fim")
    ax2.set_ylabel("Frequência")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("baseline_results.png")
    print(f"\n>>> Gráfico salvo em 'baseline_results.png'")
    print(
        f">>> Total de Negócios Fechados: {history['deals_count']} / {len(history['broker_rewards'])}"
    )


if __name__ == "__main__":
    data = run_simulation(num_episodes=100)
    plot_results(data)
