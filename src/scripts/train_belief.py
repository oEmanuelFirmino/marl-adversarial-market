import sys
import os
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.market_env import MarketAdversarialEnv
from src.agents.baselines.rule_based import FixedRegulator, ThresholdResponder
from src.engine.trainer import BeliefPPOTrainer


def train_belief_system():
    print(">>> Iniciando Treinamento Híbrido (PPO Recorrente + Belief) <<<")

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

    max_episodes = 2500
    update_timestep = 800
    time_step = 0

    for ep in range(max_episodes):
        obs, _ = env.reset()
        agent.reset_memory()

        ep_reward = 0

        while True:

            act_prop, log_prop, val_prop, belief_probs = agent.select_action(
                obs["proposer"]
            )

            act_resp = responder.act(obs["responder"])
            act_reg = regulator.act(obs["regulator"])

            actions = {
                "proposer": act_prop,
                "responder": act_resp,
                "regulator": act_reg,
            }
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            done = all(terms.values()) or all(truncs.values())

            agent.buffer.add(
                obs["proposer"],
                torch.tensor(act_prop),
                log_prop,
                rewards["proposer"],
                torch.tensor(val_prop),
                done,
                belief_probs,
                act_resp,
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

        if ep % 50 == 0:
            avg_loss = (
                sum(history_belief_loss[-10:]) / 10 if history_belief_loss else 0.0
            )
            avg_rew = sum(history_rewards[-50:]) / 50
            print(f"Ep {ep} | Avg Reward: {avg_rew:.2f} | Belief Loss: {avg_loss:.4f}")

    os.makedirs("data/models", exist_ok=True)
    model_path = "data/models/belief_agent_v1.pt"
    agent.save_checkpoint(model_path)
    print(f">>> Modelo salvo em '{model_path}'")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Updates")
    ax1.set_ylabel("Belief Loss", color=color)
    ax1.plot(history_belief_loss, color=color, alpha=0.6, label="Loss (Previsão)")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Reward (Média Móvel)", color=color)

    window = 50
    smooth_rewards = [
        sum(history_rewards[i : i + window]) / window
        for i in range(0, len(history_rewards), window)
    ]
    ax2.plot(
        range(0, len(history_rewards), window),
        smooth_rewards,
        color=color,
        label="Reward",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Treinamento: Recurrent PPO + Belief")
    plt.tight_layout()
    plt.savefig("recurrent_training.png")
    print(">>> Gráfico salvo em 'recurrent_training.png'")


if __name__ == "__main__":
    train_belief_system()
