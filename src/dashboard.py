"""
Dashboard de Predição de Hipotensão Intradiálica
Universidade Federal de São Carlos (UFSCar) — TCC
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import os
from datetime import datetime

try:
    from joblib import load
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

# ─────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predição de Hipotensão Intradiálica",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────
CLINICAL_VARS = ["IWG", "VOL", "KT", "BFR", "HBC", "APR", "VPR",
                 "TMP", "SBP", "DBP", "HRA", "TUF"]

VAR_LABELS = {
    "IWG": "IWG — Ganho de Peso Interdialítico (kg)",
    "VOL": "VOL — Volume de Ultrafiltração Prescrito (L)",
    "KT":  "KT  — KT/V Prescrito",
    "BFR": "BFR — Fluxo Sanguíneo (mL/min)",
    "HBC": "HBC — Hemoglobina (g/dL)",
    "APR": "APR — Pressão Arterial Pré-bomba (mmHg)",
    "VPR": "VPR — Pressão Venosa Pós-filtro (mmHg)",
    "TMP": "TMP — Pressão Transmembrana (mmHg)",
    "SBP": "SBP — Pressão Sistólica (mmHg)",
    "DBP": "DBP — Pressão Diastólica (mmHg)",
    "HRA": "HRA — Frequência Cardíaca (bpm)",
    "TUF": "TUF — Taxa de Ultrafiltração (mL/h)",
}

VAR_DEFAULTS = {
    "IWG": (2.5, 2.8),
    "VOL": (2.5, 2.8),
    "KT":  (1.3, 1.3),
    "BFR": (300.0, 300.0),
    "HBC": (11.0, 10.8),
    "APR": (-120.0, -125.0),
    "VPR": (150.0, 155.0),
    "TMP": (180.0, 185.0),
    "SBP": (140.0, 135.0),
    "DBP": (80.0, 78.0),
    "HRA": (72.0, 74.0),
    "TUF": (600.0, 620.0),
}

# Sigmas calibrados (desvio padrão empírico do ruído Gaussiano por variável)
VAR_SIGMAS = {
    "IWG": 0.05, "VOL": 0.05, "KT": 0.03, "BFR": 5.0,
    "HBC": 0.2,  "APR": 5.0,  "VPR": 8.0, "TMP": 10.0,
    "SBP": 5.0,  "DBP": 3.0,  "HRA": 3.0, "TUF": 30.0,
}

HOURS = [0, 1, 2, 3, 4, 5]


# ─────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelo…")
def load_model(path: str):
    """Carrega pipeline .pkl do disco (com cache)."""
    if not JOBLIB_OK:
        return None
    if not os.path.exists(path):
        return None
    return load(path)


def simulate_session(obs_h0: dict, obs_h1: dict,
                     decay_rate: float, noise_sigma_scale: float,
                     n_sim: int, rng: np.random.Generator):
    """
    Retorna lista de n_sim simulações Monte Carlo.
    Cada simulação é um dict {var: array[H0..H5]}.
    """
    simulations = []
    for _ in range(n_sim):
        sim = {}
        for var in CLINICAL_VARS:
            h0_val = obs_h0[var]
            h1_val = obs_h1[var]
            slope = (h1_val - h0_val) * (1.0 - decay_rate)
            sigma = VAR_SIGMAS[var] * noise_sigma_scale
            series = [h0_val, h1_val]
            for t in range(2, 6):
                pred = h1_val + slope * (t - 1)
                noise = rng.normal(0, sigma)
                series.append(pred + noise)
            sim[var] = np.array(series)
        simulations.append(sim)
    return simulations


def build_feature_row(sex: int, age: int, sim: dict) -> pd.DataFrame:
    """Constrói uma linha de features no formato esperado pelo modelo."""
    row = {"SEX": sex, "AGE": age}
    for var in CLINICAL_VARS:
        for h in HOURS:
            row[f"{var}_H{h}"] = sim[var][h]
    return pd.DataFrame([row])


def run_monte_carlo(model, sex: int, age: int,
                    obs_h0: dict, obs_h1: dict,
                    decay_rate: float, noise_sigma_scale: float,
                    n_sim: int) -> tuple:
    """
    Executa Monte Carlo e retorna (probs_matrix, mean_prob, ci_low, ci_high).
    probs_matrix: shape (n_sim,) — probabilidade de hipotensão por simulação.
    """
    rng = np.random.default_rng(seed=42)
    simulations = simulate_session(obs_h0, obs_h1, decay_rate,
                                   noise_sigma_scale, n_sim, rng)

    # Para cada simulação, constrói X e obtém probabilidade
    probs = []
    for sim in simulations:
        X = build_feature_row(sex, age, sim)
        try:
            p = float(model.predict_proba(X)[0, 1])
        except Exception:
            p = float(model.predict(X)[0])
        probs.append(p)

    probs = np.array(probs)
    mean_p = float(np.mean(probs))
    ci_low = float(np.percentile(probs, 2.5))
    ci_high = float(np.percentile(probs, 97.5))
    return probs, mean_p, ci_low, ci_high, simulations


def compute_trajectory_stats(simulations: list) -> dict:
    """Computa média e IC 95% de cada variável por hora."""
    stats = {}
    for var in CLINICAL_VARS:
        mat = np.array([s[var] for s in simulations])  # (n_sim, 6)
        stats[var] = {
            "mean": mat.mean(axis=0),
            "low":  np.percentile(mat, 2.5, axis=0),
            "high": np.percentile(mat, 97.5, axis=0),
        }
    return stats


# ─────────────────────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0f2537; }
[data-testid="stSidebar"] * { color: #e8f4fc !important; }
.alert-red   { background:#ff4b4b22; border-left:4px solid #ff4b4b;
               padding:12px 16px; border-radius:4px; margin:8px 0; }
.alert-green { background:#00cc8822; border-left:4px solid #00cc88;
               padding:12px 16px; border-radius:4px; margin:8px 0; }
.metric-card { background:#1a2a3a; border-radius:10px;
               padding:16px 20px; text-align:center; margin:6px; }
.metric-label { font-size:0.8rem; color:#90b8d0; }
.metric-value { font-size:1.9rem; font-weight:700; color:#e8f4fc; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-monitor.png", width=60)
    st.title("Hipotensão Intradiálica")
    st.caption("Dashboard preditivo — TCC UFSCar")
    st.divider()

    # ── Modelo ──────────────────────────────────────────────
    st.subheader("⚙️ Configuração do Modelo")
    model_path = st.text_input(
        "Caminho do modelo (.pkl)",
        value=os.path.join("..", "models", "modelo_knn.pkl"),
        help="Caminho relativo ou absoluto para o arquivo .pkl do pipeline treinado.",
    )

    # ── Upload CSV opcional ─────────────────────────────────
    st.subheader("📂 Upload de CSV (opcional)")
    uploaded = st.file_uploader(
        "Carregar dados do paciente (CSV com ';')",
        type=["csv"],
        help="O CSV deve conter colunas: SEX, AGE, IWG_H0, IWG_H1, … para todas as variáveis.",
    )

    st.divider()
    # ── Dados do paciente ───────────────────────────────────
    st.subheader("👤 Dados do Paciente")
    sex_label = st.selectbox("Sexo", ["Masculino (0)", "Feminino (1)"])
    sex_val = 0 if sex_label.startswith("M") else 1
    age_val = st.number_input("Idade (anos)", min_value=1, max_value=120,
                               value=65, step=1)

    st.divider()
    # ── Parâmetros de simulação ─────────────────────────────
    st.subheader("🔬 Parâmetros de Simulação")
    decay_rate = st.slider(
        "Taxa de Decaimento da Tendência",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="0 = tendência linear pura; 1 = sem tendência (constante).",
    )
    noise_scale = st.slider(
        "Escala do Ruído Gaussiano",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Multiplica o sigma calibrado de cada variável.",
    )
    n_simulations = st.slider(
        "Número de Simulações Monte Carlo",
        min_value=10, max_value=500, value=100, step=10,
    )
    run_btn = st.button("▶  Simular e Prever", use_container_width=True,
                         type="primary")

# ─────────────────────────────────────────────────────────────
# MAIN — entrada manual das variáveis clínicas
# ─────────────────────────────────────────────────────────────
st.title("🩺 Predição de Hipotensão Intradiálica")
st.markdown(
    "Insira os sinais clínicos observados nas horas **H0** e **H1**. "
    "O dashboard simulará H2–H5 via Monte Carlo e estimará a probabilidade "
    "de hipotensão intradiálica com o modelo pré-treinado."
)

# Preenche com CSV se enviado
csv_defaults = {}
if uploaded is not None:
    try:
        csv_df = pd.read_csv(uploaded, sep=";", nrows=1)
        for var in CLINICAL_VARS:
            for h in [0, 1]:
                col = f"{var}_H{h}"
                if col in csv_df.columns:
                    csv_defaults[(var, h)] = float(csv_df[col].iloc[0])
        st.success("CSV carregado! Valores preenchidos automaticamente.")
    except Exception as e:
        st.warning(f"Não foi possível ler o CSV: {e}")

st.subheader("📋 Sinais Clínicos Observados (H0 e H1)")

obs_h0, obs_h1 = {}, {}
# Divide em 2 colunas
col_a, col_b = st.columns(2)
for i, var in enumerate(CLINICAL_VARS):
    target_col = col_a if i % 2 == 0 else col_b
    with target_col:
        default_h0, default_h1 = VAR_DEFAULTS[var]
        with st.expander(VAR_LABELS[var], expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                obs_h0[var] = st.number_input(
                    f"H0",
                    value=float(csv_defaults.get((var, 0), default_h0)),
                    step=0.1, format="%.2f", key=f"{var}_h0",
                )
            with c2:
                obs_h1[var] = st.number_input(
                    f"H1",
                    value=float(csv_defaults.get((var, 1), default_h1)),
                    step=0.1, format="%.2f", key=f"{var}_h1",
                )

st.divider()

# ─────────────────────────────────────────────────────────────
# SIMULAÇÃO E PREDIÇÃO
# ─────────────────────────────────────────────────────────────
if run_btn:
    model = load_model(model_path)
    if model is None:
        st.error(
            f"❌ Modelo não encontrado em `{model_path}`. "
            "Verifique o caminho e certifique-se de que o arquivo .pkl existe."
        )
        st.stop()

    with st.spinner("Executando Monte Carlo…"):
        try:
            (probs, mean_prob, ci_low, ci_high,
             simulations) = run_monte_carlo(
                model, sex_val, age_val,
                obs_h0, obs_h1,
                decay_rate, noise_scale, n_simulations,
            )
        except Exception as e:
            st.error(f"Erro durante a simulação: {e}")
            st.stop()

    traj = compute_trajectory_stats(simulations)
    label = "SIM" if mean_prob >= 0.5 else "NÃO"
    alert_class = "alert-red" if label == "SIM" else "alert-green"
    alert_icon = "🔴" if label == "SIM" else "🟢"

    # ── Resumo do Paciente ───────────────────────────────────
    st.subheader("📊 Resultado da Predição")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Probabilidade Média</div>
          <div class="metric-value">{mean_prob:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">IC 95% Inferior</div>
          <div class="metric-value">{ci_low:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">IC 95% Superior</div>
          <div class="metric-value">{ci_high:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with r4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Simulações</div>
          <div class="metric-value">{n_simulations}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div class="{alert_class}">'
        f'<strong>{alert_icon} Hipotensão Prevista: {label}</strong><br>'
        f'Probabilidade média = <strong>{mean_prob:.1%}</strong> '
        f'(IC 95%: [{ci_low:.1%}, {ci_high:.1%}])'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Tabela H0–H5 ─────────────────────────────────────────
    st.subheader("📋 Valores por Hora (Observado + Simulado — média IC 95%)")
    table_rows = []
    for var in CLINICAL_VARS:
        row = {"Variável": var}
        for h in range(6):
            if h <= 1:
                v = obs_h0[var] if h == 0 else obs_h1[var]
                row[f"H{h} (obs)"] = f"{v:.2f}"
            else:
                m = traj[var]["mean"][h]
                lo = traj[var]["low"][h]
                hi = traj[var]["high"][h]
                row[f"H{h} (sim)"] = f"{m:.2f} [{lo:.2f}–{hi:.2f}]"
        table_rows.append(row)
    st.dataframe(pd.DataFrame(table_rows).set_index("Variável"),
                 use_container_width=True)

    # ── Gráficos de trajetórias ──────────────────────────────
    st.subheader("📈 Trajetórias Clínicas (H0–H5)")
    n_vars = len(CLINICAL_VARS)
    n_cols_plot = 3
    n_rows_plot = int(np.ceil(n_vars / n_cols_plot))
    fig_traj = make_subplots(
        rows=n_rows_plot, cols=n_cols_plot,
        subplot_titles=[VAR_LABELS[v][:30] for v in CLINICAL_VARS],
        vertical_spacing=0.08, horizontal_spacing=0.06,
    )
    colors = px.colors.qualitative.Safe
    for idx, var in enumerate(CLINICAL_VARS):
        r = idx // n_cols_plot + 1
        c = idx % n_cols_plot + 1
        color = colors[idx % len(colors)]
        mean_vals = traj[var]["mean"]
        low_vals  = traj[var]["low"]
        high_vals = traj[var]["high"]

        # IC 95% como banda
        fig_traj.add_trace(go.Scatter(
            x=HOURS + HOURS[::-1],
            y=list(high_vals) + list(low_vals[::-1]),
            fill="toself", fillcolor=color,
            opacity=0.15, line_color="rgba(0,0,0,0)",
            name=f"{var} IC95", showlegend=False,
        ), row=r, col=c)

        # Linha de média simulada (H0-H5)
        fig_traj.add_trace(go.Scatter(
            x=HOURS, y=mean_vals,
            mode="lines+markers", name=var,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            showlegend=(r == 1 and c == 1),
        ), row=r, col=c)

        # Pontos observados (H0 e H1) destacados
        fig_traj.add_trace(go.Scatter(
            x=[0, 1],
            y=[obs_h0[var], obs_h1[var]],
            mode="markers",
            marker=dict(color="white", size=10, symbol="circle-open",
                        line=dict(color=color, width=2)),
            name=f"{var} obs", showlegend=False,
        ), row=r, col=c)

    fig_traj.update_layout(
        height=300 * n_rows_plot,
        template="plotly_dark",
        title_text="Trajetórias das Variáveis Clínicas (H0–H5)",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
    )
    fig_traj.update_xaxes(tickvals=HOURS, ticktext=[f"H{h}" for h in HOURS])
    st.plotly_chart(fig_traj, use_container_width=True)

    # ── Distribuição das probabilidades Monte Carlo ──────────
    st.subheader("📊 Distribuição das Probabilidades (Monte Carlo)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=probs, nbinsx=30,
        marker_color="#1f77b4", opacity=0.8,
        name="Probabilidade por Simulação",
    ))
    fig_hist.add_vline(x=mean_prob, line_width=3, line_dash="dash",
                       line_color="#ff7f0e",
                       annotation_text=f"Média: {mean_prob:.1%}",
                       annotation_position="top right")
    fig_hist.add_vline(x=0.5, line_width=2, line_dash="dot",
                       line_color="red",
                       annotation_text="Limiar 50%",
                       annotation_position="top left")
    fig_hist.add_vrect(x0=ci_low, x1=ci_high, fillcolor="orange",
                       opacity=0.1, line_width=0, annotation_text="IC 95%",
                       annotation_position="top left")
    fig_hist.update_layout(
        template="plotly_dark",
        xaxis_title="Probabilidade de Hipotensão",
        yaxis_title="Frequência",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Gauge de probabilidade ───────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=mean_prob * 100,
        number={"suffix": "%", "font": {"size": 40}},
        delta={"reference": 50, "valueformat": ".1f",
               "suffix": "% vs. limiar"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#ff4b4b" if mean_prob >= 0.5 else "#00cc88"},
            "steps": [
                {"range": [0, 30],  "color": "#00663322"},
                {"range": [30, 50], "color": "#cccc0022"},
                {"range": [50, 100],"color": "#cc000022"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.85, "value": 50,
            },
        },
        title={"text": "Probabilidade de Hipotensão"},
    ))
    fig_gauge.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        height=300,
    )
    col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
    with col_g2:
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Exportação CSV ───────────────────────────────────────
    st.subheader("💾 Exportar Resultados")

    # Montar DataFrame de exportação
    export_rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for sim_idx, sim in enumerate(simulations):
        row_exp = {
            "timestamp": ts,
            "sim_index": sim_idx + 1,
            "SEX": sex_val,
            "AGE": age_val,
            "decay_rate": decay_rate,
            "noise_scale": noise_scale,
        }
        for var in CLINICAL_VARS:
            for h in HOURS:
                row_exp[f"{var}_H{h}"] = sim[var][h]
        row_exp["prob_hypotension"] = probs[sim_idx]
        export_rows.append(row_exp)

    df_export = pd.DataFrame(export_rows)
    # Adicionar linha de resumo
    summary = {
        "timestamp": ts, "sim_index": "SUMMARY",
        "SEX": sex_val, "AGE": age_val,
        "decay_rate": decay_rate, "noise_scale": noise_scale,
        "prob_hypotension": mean_prob,
    }
    for var in CLINICAL_VARS:
        for h in HOURS:
            summary[f"{var}_H{h}"] = (obs_h0[var] if h == 0
                                       else obs_h1[var] if h == 1
                                       else traj[var]["mean"][h])
    df_export = pd.concat([
        pd.DataFrame([summary]), df_export
    ], ignore_index=True)

    csv_bytes = df_export.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        label="⬇️  Baixar Relatório CSV",
        data=csv_bytes,
        file_name=f"predicao_hipotensao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Log da predição ──────────────────────────────────────
    with st.expander("🗒️ Log da Predição"):
        st.json({
            "timestamp": ts,
            "model_path": model_path,
            "patient": {"SEX": sex_val, "AGE": age_val},
            "simulation_params": {
                "n_simulations": n_simulations,
                "decay_rate": decay_rate,
                "noise_sigma_scale": noise_scale,
            },
            "result": {
                "mean_probability": round(mean_prob, 4),
                "ci_95_low": round(ci_low, 4),
                "ci_95_high": round(ci_high, 4),
                "label": label,
            },
        })
else:
    st.info("👈 Configure os parâmetros na barra lateral e clique em **▶ Simular e Prever**.")
    st.markdown("""
    **Como usar:**
    1. Informe o **sexo** e a **idade** do paciente na barra lateral.
    2. Expanda cada variável clínica e insira os valores observados em **H0** e **H1**.
    3. Ajuste os parâmetros de simulação (taxa de decaimento, ruído, nº de simulações).
    4. Clique em **▶ Simular e Prever** para gerar as previsões.
    5. Baixe o relatório em CSV ao final.

    **Variáveis clínicas disponíveis:**
    """)
    desc_df = pd.DataFrame([
        {"Sigla": k, "Descrição": v} for k, v in VAR_LABELS.items()
    ])
    st.dataframe(desc_df.set_index("Sigla"), use_container_width=True)