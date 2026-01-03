# src/app/dashboard.py
import sys
import os
import time
import torch
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Hack de Path para encontrar módulos src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.market_env import MarketAdversarialEnv
from src.agents.baselines.rule_based import FixedRegulator, ThresholdResponder
from src.engine.trainer import BeliefPPOTrainer
from src.core.types import ActionType

# --- Configuração da Página ---
st.set_page_config(
    page_title="MARL War Room",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Customizado ---
st.markdown(
    """
<style>
    .stApp { background-color: #0E1117; }
    [data-testid="stMetricValue"] { font-family: 'Segoe UI', monospace; }
    div.stButton > button { width: 100%; border-radius: 5px; font-weight: bold; }
    h1, h2, h3 { color: #FAFAFA; font-family: 'Segoe UI', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0E1117; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #262730; border-bottom: 2px solid #00FFAA; }
</style>
""",
    unsafe_allow_html=True,
)


# --- Singleton: Carregar Sistema ---
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

# --- Estado da Sessão ---
if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False

if "sim_params" not in st.session_state:
    st.session_state.sim_params = {
        "volatility": 0.2,
        "urgency": 0.5,
        "competitor": 0.1,
        "sentiment": 1.0,
    }

# --- Sidebar ---
with st.sidebar:
    st.header("🎛️ Configuração do Cenário")

    stochastic_mode = st.checkbox("🎲 Modo Estocástico", value=False)

    if stochastic_mode:
        st.caption("Valores variam automaticamente.")
        drift = st.slider("Intensidade (Drift)", 0.01, 0.2, 0.05)
    else:
        drift = 0.0

    st.divider()

    volatility = st.slider(
        "Volatilidade", 0.0, 1.0, st.session_state.sim_params["volatility"]
    )
    urgency = st.slider("Urgência", 0.0, 1.0, st.session_state.sim_params["urgency"])

    st.markdown("### 🏴‍☠️ Concorrência")
    competitor_intensity = st.slider(
        "Agressividade", 0.0, 1.0, st.session_state.sim_params["competitor"]
    )
    market_sentiment = st.slider(
        "Sentimento", 0.5, 1.5, st.session_state.sim_params["sentiment"]
    )

    st.divider()

    col_play, col_stop = st.columns(2)
    if col_play.button("▶️ Auto Play"):
        st.session_state.running = True
        st.rerun()  # Força refresh para iniciar loop
    if col_stop.button("II Pause"):
        st.session_state.running = False
        st.rerun()

    if st.button("⏭️ Passo Único"):
        st.session_state.running = False
        run_step = True
    else:
        run_step = False

    if st.button("🗑️ Resetar Tudo", type="primary"):
        st.session_state.history = []
        st.session_state.sim_params = {
            "volatility": 0.2,
            "urgency": 0.5,
            "competitor": 0.1,
            "sentiment": 1.0,
        }
        env.reset()
        st.session_state.running = False
        st.rerun()


# --- Função Auxiliar ---
def perturb_value(val, drift, min_val=0.0, max_val=1.0):
    noise = np.random.uniform(-drift, drift)
    return np.clip(val + noise, min_val, max_val)


def execute_step():
    # 0. Atualizar Parâmetros
    if stochastic_mode:
        st.session_state.sim_params["volatility"] = perturb_value(
            st.session_state.sim_params["volatility"], drift
        )
        st.session_state.sim_params["urgency"] = perturb_value(
            st.session_state.sim_params["urgency"], drift
        )
        st.session_state.sim_params["competitor"] = perturb_value(
            st.session_state.sim_params["competitor"], drift
        )
        st.session_state.sim_params["sentiment"] = perturb_value(
            st.session_state.sim_params["sentiment"], drift, 0.5, 1.5
        )
    else:
        st.session_state.sim_params["volatility"] = volatility
        st.session_state.sim_params["urgency"] = urgency
        st.session_state.sim_params["competitor"] = competitor_intensity
        st.session_state.sim_params["sentiment"] = market_sentiment

    if env.state_data is None:
        obs, _ = env.reset()
    else:
        env.state_data.global_volatility = st.session_state.sim_params["volatility"]
        env.state_data.responder_urgency = st.session_state.sim_params["urgency"]
        env.state_data.competitor_intensity = st.session_state.sim_params["competitor"]
        env.state_data.market_sentiment = st.session_state.sim_params["sentiment"]
        obs = {a: env._make_obs(env.state_data, a) for a in env.agents}

    # 1. IA Pensa
    act_prop, _, _, belief_probs = agent.select_action(obs["proposer"])

    # 2. Oponentes Reagem
    act_resp = responder.act(obs["responder"])
    act_reg = regulator.act(obs["regulator"])

    # 3. Física
    actions = {"proposer": act_prop, "responder": act_resp, "regulator": act_reg}
    next_obs, rewards, terms, _, infos = env.step(actions)

    # 4. Logging Completo
    belief_vector = (
        belief_probs[0].tolist() if belief_probs.dim() > 1 else belief_probs.tolist()
    )
    is_snatch = 1 if infos["responder"].get("snatch") else 0
    price = infos["responder"].get("price", 0)

    log_entry = {
        "Step": env.state_data.step_count,
        "Price": round(price, 2),
        "Budget": round(env.state_data.responder_budget, 2),
        "Cash": round(env.state_data.proposer_cash, 2),
        "Deal": 1 if infos["responder"].get("deal") else 0,
        "Snatch": is_snatch,
        "Lost_Revenue": round(price, 2) if is_snatch else 0,
        "Reward": round(rewards["proposer"], 3),
        "Belief_Wait": round(belief_vector[0], 2),
        "Belief_Buy": round(belief_vector[1], 2),
        "Belief_Leave": round(belief_vector[2], 2),
        "Real_Action": act_resp,
        "My_Action": act_prop,
        "P_Vol": round(env.state_data.global_volatility, 2),
        "P_Comp": round(env.state_data.competitor_intensity, 2),
        "P_Sent": round(env.state_data.market_sentiment, 2),
        "P_Urg": round(env.state_data.responder_urgency, 2),
    }
    st.session_state.history.append(log_entry)

    if all(terms.values()):
        env.reset()


# ==============================================================================
# 1. RENDERIZAÇÃO (AGORA VEM ANTES DA EXECUÇÃO PARA APARECER EM TEMPO REAL)
# ==============================================================================

st.title("🛡️ MARL Adversarial Market: War Room")

# Se tivermos histórico (mesmo que antigo ou acabado de gerar), mostramos
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    last_row = df.iloc[-1]

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    deals_total = df["Deal"].sum()
    snatch_total = df["Snatch"].sum() if "Snatch" in df.columns else 0
    lost_revenue_total = df["Lost_Revenue"].sum() if "Lost_Revenue" in df.columns else 0
    total_ops = len(df)
    conv_rate = (deals_total / total_ops) * 100 if total_ops > 0 else 0

    kpi1.metric("Caixa", f"${last_row['Cash']:.0f}", delta=f"{last_row['Reward']:.1f}")
    kpi2.metric(
        "Perdas (Concorrência)",
        f"{snatch_total}",
        f"-${lost_revenue_total:.0f}",
        delta_color="inverse",
    )
    kpi3.metric("Conversão", f"{conv_rate:.1f}%")

    pred_actions = df[["Belief_Wait", "Belief_Buy", "Belief_Leave"]].values.argmax(
        axis=1
    )
    acc = np.mean(pred_actions == df["Real_Action"].values) * 100
    kpi4.metric("Acurácia IA", f"{acc:.1f}%")

    # --- ABAS ---
    tab_market, tab_comp, tab_brain, tab_table = st.tabs(
        [
            "📈 Dinâmica de Mercado",
            "⚔️ Batalha de Market Share",
            "🧠 Inspeção da IA",
            "📋 Tabela de Dados",
        ]
    )

    # 1. Dinâmica de Mercado
    with tab_market:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=df["Step"],
                y=df["Budget"],
                name="Budget (Cliente)",
                line=dict(color="gray", dash="dot"),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Step"],
                y=df["Price"],
                name="Preço Ofertado",
                line=dict(color="#00FFAA"),
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
                marker=dict(color="yellow", size=12, symbol="star"),
            ),
            secondary_y=False,
        )

        if "Snatch" in df.columns:
            snatch_df = df[df["Snatch"] == 1]
            if not snatch_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=snatch_df["Step"],
                        y=snatch_df["Price"],
                        mode="markers",
                        name="Perda p/ Concorrência",
                        marker=dict(color="#FF4444", size=10, symbol="x"),
                    ),
                    secondary_y=False,
                )

        fig.update_layout(
            title="Negociações em Tempo Real",
            xaxis_title="Passos da Simulação",
            yaxis_title="Valor ($)",
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2. Batalha de Market Share
    with tab_comp:
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            fig_share = go.Figure()
            df["Cum_Deals"] = df["Deal"].cumsum()
            df["Cum_Snatch"] = df["Snatch"].cumsum() if "Snatch" in df.columns else 0

            fig_share.add_trace(
                go.Scatter(
                    x=df["Step"],
                    y=df["Cum_Deals"],
                    name="Minhas Vendas",
                    line=dict(color="#00FFAA", width=3),
                    fill="tozeroy",
                )
            )
            fig_share.add_trace(
                go.Scatter(
                    x=df["Step"],
                    y=df["Cum_Snatch"],
                    name="Vendas Concorrência",
                    line=dict(color="#FF4444", width=3),
                    fill="tonexty",
                )
            )

            fig_share.update_layout(
                title="Acumulado: Quem está a dominar o mercado?",
                template="plotly_dark",
                height=400,
                yaxis_title="Volume de Negócios",
            )
            st.plotly_chart(fig_share, use_container_width=True)

        with col_c2:
            st.markdown("### Análise")
            st.markdown("Área Verde = Sucesso. Área Vermelha = Falha Competitiva.")

    # 3. Inspeção da IA
    with tab_brain:
        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            b_data = [
                last_row["Belief_Wait"],
                last_row["Belief_Buy"],
                last_row["Belief_Leave"],
            ]
            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=["Wait", "Buy", "Leave"],
                        y=b_data,
                        marker_color=["#FF5555", "#55FF55", "#FF5555"],
                    )
                ]
            )
            fig_bar.update_layout(
                title="Previsão Atual", template="plotly_dark", height=300
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_b2:
            fig_stack = go.Figure()
            fig_stack.add_trace(
                go.Scatter(
                    x=df["Step"], y=df["Belief_Wait"], stackgroup="one", name="Wait"
                )
            )
            fig_stack.add_trace(
                go.Scatter(
                    x=df["Step"], y=df["Belief_Buy"], stackgroup="one", name="Buy"
                )
            )
            fig_stack.add_trace(
                go.Scatter(
                    x=df["Step"], y=df["Belief_Leave"], stackgroup="one", name="Leave"
                )
            )
            fig_stack.update_layout(
                title="Evolução da Confiança", template="plotly_dark", height=350
            )
            st.plotly_chart(fig_stack, use_container_width=True)

    # 4. Tabela
    with tab_table:
        st.markdown("### 📄 Registro Completo")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar Log (CSV)", data=csv, file_name="marl_log.csv", mime="text/csv"
        )

        def highlight_events(row):
            if row.Deal == 1:
                return ["background-color: rgba(0, 255, 170, 0.2)"] * len(row)
            elif row.Snatch == 1:
                return ["background-color: rgba(255, 68, 68, 0.2)"] * len(row)
            return [""] * len(row)

        df_display = df.sort_values("Step", ascending=False)
        st.dataframe(
            df_display.style.apply(highlight_events, axis=1).format(
                {
                    "Price": "${:.2f}",
                    "Budget": "${:.2f}",
                    "Cash": "${:.2f}",
                    "Lost_Revenue": "${:.2f}",
                    "Reward": "{:.3f}",
                    "Belief_Buy": "{:.1%}",
                }
            ),
            use_container_width=True,
            height=500,
        )

else:
    st.info("A simulação está parada. Clique em '▶️ Auto Play' na sidebar.")

# ==============================================================================
# 2. LÓGICA DE CONTROLE (NO FINAL DO ARQUIVO)
# ==============================================================================

if st.session_state.running:
    # Executa o passo
    execute_step()

    # Pequeno delay para visualização suave
    time.sleep(0.05)

    # Rerun para atualizar a tela (voltando ao topo, onde a renderização ocorre com os dados novos)
    st.rerun()

elif run_step:
    execute_step()
    # No caso de passo único, também damos rerun para mostrar o resultado imediatamente
    st.rerun()
