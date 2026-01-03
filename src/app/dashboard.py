import sys
import os
import time
import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.market_env import MarketAdversarialEnv
from src.agents.baselines.rule_based import FixedRegulator, ThresholdResponder
from src.engine.trainer import BeliefPPOTrainer

st.set_page_config(page_title="MARL Market Watch", layout="wide")


st.markdown(
    """
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div.stButton > button { background-color: #00FFAA; color: black; border: none; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 MARL Adversarial Market: War Room")


st.sidebar.header("Parâmetros de Simulação")
volatility = st.sidebar.slider("Volatilidade Global", 0.0, 1.0, 0.2)
responder_urgency = st.sidebar.slider("Urgência do Cliente (Inicial)", 0.0, 1.0, 0.5)
simulation_speed = st.sidebar.slider("Velocidade (sec)", 0.1, 2.0, 0.5)


@st.cache_resource
def load_ai_agent():
    env = MarketAdversarialEnv()

    agent = BeliefPPOTrainer(env, agent_id="proposer", opponent_id="responder")

    model_path = "data/models/belief_agent_v1.pt"
    if os.path.exists(model_path):
        agent.load_checkpoint(model_path)
        print("Modelo carregado com sucesso.")
    else:
        st.warning(f"Modelo não encontrado em {model_path}. Usando pesos aleatórios.")

    agent.policy.eval()
    agent.belief_net.eval()
    return agent


if "history" not in st.session_state:
    st.session_state.history = []
if "env" not in st.session_state:
    st.session_state.env = MarketAdversarialEnv()

env = st.session_state.env
agent = load_ai_agent()


responder = ThresholdResponder(
    "responder", env.observation_space("responder"), env.action_space("responder")
)
regulator = FixedRegulator(
    "regulator", env.observation_space("regulator"), env.action_space("regulator")
)


col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Executar Passo"):

        if env.state_data is None:
            obs, _ = env.reset()

            env.state_data.global_volatility = volatility
            env.state_data.responder_urgency = responder_urgency
        else:

            obs = {a: env._make_obs(env.state_data, a) for a in env.agents}

        act_prop, _, val_prop, belief_probs = agent.select_action(obs["proposer"])

        act_resp = responder.act(obs["responder"])
        act_reg = regulator.act(obs["regulator"])

        actions = {"proposer": act_prop, "responder": act_resp, "regulator": act_reg}

        next_obs, rewards, terms, _, infos = env.step(actions)

        step_data = {
            "Step": env.state_data.step_count,
            "Price": infos["responder"].get("price", 0),
            "IA Reward": rewards["proposer"],
            "Deal": 1 if infos["responder"].get("deal") else 0,
            "Belief (Will Buy?)": belief_probs[1].item() if belief_probs.ndim == 1 else belief_probs[0][1].item(),
            "Actual Action": act_resp,
        }
        st.session_state.history.append(step_data)

        if all(terms.values()):
            env.reset()
            st.toast("Fim do Episódio! Resetando...", icon="⚠️")

with col2:
    if st.button("🗑️ Limpar Histórico"):
        st.session_state.history = []
        env.reset()


if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)

    m1, m2, m3 = st.columns(3)
    m1.metric("Dinheiro em Caixa", f"${env.state_data.proposer_cash:.2f}")
    m2.metric("Último Preço", f"${df['Price'].iloc[-1]:.2f}")
    m3.metric("Negócios Fechados", f"{df['Deal'].sum()}")

    st.subheader("Dinâmica de Preço vs. Crença da IA")

    fig, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(df["Step"], df["Price"], color="cyan", label="Preço Ofertado")
    ax1.set_ylabel("Preço ($)", color="cyan")

    ax2 = ax1.twinx()
    ax2.plot(
        df["Step"],
        df["Belief (Will Buy?)"],
        color="magenta",
        linestyle="--",
        label="Crença (Vai Comprar?)",
    )
    ax2.set_ylabel("Probabilidade Estimada", color="magenta")
    ax2.set_ylim(0, 1)

    st.pyplot(fig)

    st.subheader("Log de Transações")
    st.dataframe(df.tail(10))

else:
    st.info("O mercado está fechado. Clique em 'Executar Passo' para iniciar.")
