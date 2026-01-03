import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.market_env import MarketAdversarialEnv


def main():
    print(">>> Iniciando Validação da Física do Mundo (Phase 1) <<<")

    env = MarketAdversarialEnv()
    print(f"[OK] Ambiente instanciado. Agentes: {env.possible_agents}")

    obs, info = env.reset()
    print("[OK] Reset executado.")
    print(f"    Obs Shape (Lead): {obs['lead'].shape}")
    print(f"    Exemplo Obs: {obs['lead']}")

    print("\n>>> Rodando Simulação Aleatória (10 steps) <<<")
    total_rewards = {a: 0.0 for a in env.possible_agents}

    for step in range(10):

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        obs, rewards, terms, truncs, infos = env.step(actions)

        for a, r in rewards.items():
            total_rewards[a] += r

        print(f"Step {step+1}:")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {rewards}")
        print(f"  Deal Occurred: {infos['lead'].get('deal')}")

        if all(terms.values()):
            print(">>> Episódio terminou prematuramente (Deal ou Reject) <<<")
            break

    print("\n>>> Resumo da Execução <<<")
    print(f"Total Rewards: {total_rewards}")
    print("Validação concluída sem erros de runtime.")


if __name__ == "__main__":
    main()
