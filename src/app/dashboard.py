import sys
import os
import time
import torch
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.market_env import MarketAdversarialEnv
from src.agents.baselines.rule_based import FixedRegulator, ThresholdResponder
from src.engine.trainer import BeliefPPOTrainer
from src.core.types import ActionType

st.set_page_config(
    page_title="MARL War Room",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0E1117; }
    .stMetric { background-color: #262730; padding: 10px; border-radius: 5px; border-left: 5px solid #00FFAA; }
    div.stButton > button { width: 100%; border-radius: 5px; font-weight: bold; }
    h1, h2, h3 { color: #FAFAFA; font-family: 'Segoe UI', sans-serif; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system():
    env = MarketAdversarialEnv()

    agent = BeliefPPOTrainer(env, agent_id="proposer", opponent_id="responder")
    model_path = "data/models/belief_agent_v1.pt"

    if os.path.exists(model_path):
        agent.load_checkpoint(model_path)
        print(f"✅ Modelo carregado: {model_path}")
    else:
        print(f"⚠️ Modelo não encontrado. Usando pesos aleatórios.")

    agent.policy.eval()
    agent.belief_net.eval()

    responder = ThresholdResponder(
        "responder", env.observation_space("responder"), env.action_space("responder")
    )
    regulator = FixedRegulator(
        "regulator", env.observation_space("regulator"), env.action_space("regulator")
    )

    return env, agent, responder, regulator


env, agent, responder, regulator = load_system()

if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False

with st.sidebar:
    st.header("🎛️ Configuração do Cenário")

    volatility = st.slider(
        "Volatilidade de Mercado",
        0.0,
        1.0,
        0.2,
        help="Afeta o risco e custo da seguradora.",
    )
    urgency = st.slider(
        "Urgência do Cliente",
        0.0,
        1.0,
        0.5,
        help="Probabilidade do cliente aceitar preços altos.",
    )

    st.divider()

    col_play, col_stop = st.columns(2)
    if col_play.button("▶️ Auto Play"):
        st.session_state.running = True
    if col_stop.button("II Pause"):
        st.session_state.running = False

    if st.button("⏭️ Passo Único"):
        st.session_state.running = False
        run_step = True
    else:
        run_step = False

    if st.button("🗑️ Resetar Simulação", type="primary"):
        st.session_state.history = []
        env.reset()
        st.session_state.running = False
        st.rerun()


def execute_step():
    if env.state_data is None:
        obs, _ = env.reset()
    else:
        env.state_data.global_volatility = volatility
        env.state_data.responder_urgency = urgency
        obs = {a: env._make_obs(env.state_data, a) for a in env.agents}

    act_prop, _, _, belief_probs = agent.select_action(obs["proposer"])

    act_resp = responder.act(obs["responder"])
    act_reg = regulator.act(obs["regulator"])

    actions = {"proposer": act_prop, "responder": act_resp, "regulator": act_reg}
    next_obs, rewards, terms, _, infos = env.step(actions)

    belief_vector = (
        belief_probs[0].tolist() if belief_probs.dim() > 1 else belief_probs.tolist()
    )

    log_entry = {
        "Step": env.state_data.step_count,
        "Price": infos["responder"].get("price", 0),
        "Budget": env.state_data.responder_budget,
        "Cash": env.state_data.proposer_cash,
        "Deal": 1 if infos["responder"].get("deal") else 0,
        "Reward": rewards["proposer"],
        "Belief_Wait": belief_vector[0],
        "Belief_Buy": belief_vector[1],
        "Belief_Leave": belief_vector[2],
        "Real_Action": act_resp,
        "Proposer_Action": act_prop,
    }
    st.session_state.history.append(log_entry)

    if all(terms.values()):
        env.reset()


if st.session_state.running:
    execute_step()
    time.sleep(0.1)
    st.rerun()
elif run_step:
    execute_step()

st.title("🛡️ MARL Adversarial Market")

if not st.session_state.history:
    st.info("A simulação está parada. Use a Sidebar para iniciar.")
else:
    df = pd.DataFrame(st.session_state.history)
    last_row = df.iloc[-1]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    deals_total = df["Deal"].sum()
    conversion_rate = (deals_total / len(df)) * 100

    kpi1.metric(
        "Caixa (Proposer)",
        f"${last_row['Cash']:.2f}",
        delta=f"{last_row['Reward']:.2f}",
    )
    kpi2.metric("Preço Ofertado", f"${last_row['Price']:.2f}")
    kpi3.metric("Taxa de Conversão", f"{conversion_rate:.1f}%")

    predicted_actions = df[["Belief_Wait", "Belief_Buy", "Belief_Leave"]].values.argmax(
        axis=1
    )
    accuracy = np.mean(predicted_actions == df["Real_Action"].values) * 100
    kpi4.metric("Acurácia da Crença", f"{accuracy:.1f}%")

    tab_market, tab_brain, tab_data = st.tabs(
        ["📈 Dinâmica de Mercado", "🧠 Inspeção da IA", "📄 Dados Brutos"]
    )

    with tab_market:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=df["Step"],
                y=df["Budget"],
                name="Budget do Cliente",
                line=dict(color="gray", dash="dot"),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Step"],
                y=df["Price"],
                name="Preço Ofertado",
                line=dict(color="#00FFAA", width=2),
            ),
            secondary_y=False,
        )

        deals_df = df[df["Deal"] == 1]
        fig.add_trace(
            go.Scatter(
                x=deals_df["Step"],
                y=deals_df["Price"],
                mode="markers",
                name="Venda Fechada",
                marker=dict(color="yellow", size=10, symbol="star"),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df["Step"],
                y=df["Belief_Buy"],
                name="Prob. Compra (IA)",
                line=dict(color="magenta"),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title="Histórico de Negociação",
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_brain:
        col_b1, col_b2 = st.columns([1, 2])

        with col_b1:
            st.markdown("### O que a IA vê?")
            st.markdown(f"**Crença Atual:**")

            belief_data = [
                last_row["Belief_Wait"],
                last_row["Belief_Buy"],
                last_row["Belief_Leave"],
            ]
            labels = ["Wait", "Buy", "Leave"]
            colors = ["#FF5555" if i != 1 else "#55FF55" for i in range(3)]

            fig_bar = go.Figure(
                data=[go.Bar(x=labels, y=belief_data, marker_color=colors)]
            )
            fig_bar.update_layout(
                title="Previsão do Comportamento do Cliente",
                template="plotly_dark",
                height=300,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            real_act_str = (
                ActionType(int(last_row["Real_Action"])).name
                if int(last_row["Real_Action"]) <= 4
                else str(last_row["Real_Action"])
            )
            st.info(f"O Cliente realmente fez: **{real_act_str}**")

        with col_b2:
            st.markdown("### Confiança ao Longo do Tempo")
            fig_stack = go.Figure()
            fig_stack.add_trace(
                go.Scatter(
                    x=df["Step"],
                    y=df["Belief_Wait"],
                    mode="lines",
                    name="Crença: Wait",
                    stackgroup="one",
                )
            )
            fig_stack.add_trace(
                go.Scatter(
                    x=df["Step"],
                    y=df["Belief_Buy"],
                    mode="lines",
                    name="Crença: Buy",
                    stackgroup="one",
                )
            )
            fig_stack.add_trace(
                go.Scatter(
                    x=df["Step"],
                    y=df["Belief_Leave"],
                    mode="lines",
                    name="Crença: Leave",
                    stackgroup="one",
                )
            )

            fig_stack.update_layout(
                title="Evolução das Expectativas da IA",
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig_stack, use_container_width=True)

    with tab_data:
        st.dataframe(
            df.sort_values("Step", ascending=False).style.highlight_max(axis=0)
        )
