"""
dashboard.py — Hemodialysis Session Risk Prediction Dashboard
Streamlit app that:
  1. Collects patient demographics (SEX, AGE, DIA) and clinical signals (H0–H5)
  2. Simulates missing hours via KNN trajectory with distance weighting
  3. Applies a selected pre-trained .pkl model to predict TARGET (hypotension risk)
  4. Visualises results interactively and exports a CSV report

Usage:
    streamlit run dashboard.py
"""

import io
import os
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HD Risk Simulator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE & INITIALIZATION CONFIGURATION (pt-BR / en-US)
# ─────────────────────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "pt-BR"

if "popup_shown" not in st.session_state:
    st.session_state.popup_shown = False

# Pop-up de alerta inicial (Prova de Conceito)
@st.dialog("⚠️ Aviso Importante / Important Notice")
def show_initial_popup():
    st.markdown(
        "**Este sistema é apenas uma prova de conceito para um preditor de hipotensão intradiálica. "
        "Os resultados apresentados não devem ser considerados em um cenário clínico.**"
    )
    st.markdown("---")
    st.markdown(
        "*This system is only a proof of concept for an intradialytic hypotension predictor. "
        "The results presented should not be considered in a clinical setting.*"
    )
    if st.button("Entendi / I understand", use_container_width=True):
        st.session_state.popup_shown = True
        st.rerun()

if not st.session_state.popup_shown:
    show_initial_popup()

# Dicionário central de localização da interface
LOCALIZATION = {
    "pt-BR": {
        "page_title": "Simulador de Risco de HD",
        "sidebar_title": "## 🩺 Simulador de Risco de HD",
        "sidebar_subtitle": "Avaliação de risco de sessão de hemodiálise via simulação de dados ausentes por KNN",
        "patient_demo": "### 👤 Dados Demográficos",
        "sex": "Sexo",
        "sex_options": ["Masculino (0)", "Feminino (1)"],
        "age": "Idade (anos)",
        "dialyzer": "Dialisador (DIA)",
        "dialyzer_help": "Selecione o dialisador utilizado nesta sessão.",
        "num_code": "Código numérico",
        "pred_model": "### 🤖 Modelo de Predição",
        "select_model": "Selecione o modelo",
        "path": "Caminho",
        "sim_params": "### ⚙️ Parâmetros de Simulação",
        "k_neighbors": "K-Vizinhos (imputação de dados ausentes)",
        "k_neighbors_help": "Número de sessões históricas usadas para estimar horas ausentes.",
        "inc_demo": "Incluir dados demográficos na distância",
        "inc_demo_help": "Usar SEXO e IDADE ao buscar os vizinhos mais próximos.",
        "csv_upload_title": "### 📂 Upload de CSV *(opcional)*",
        "csv_upload_help": "O CSV deve ter o mesmo esquema de colunas que dataset_flat_V2.csv (sem Target).",
        "csv_uploader_label": "Carregar um CSV de sessão pré-preenchido",
        "main_title": "Simulador de Risco de Sessão de Hemodiálise",
        "main_subtitle": "Insira pelo menos as medições clínicas de **H0** abaixo. Qualquer hora deixada em branco será **simulada** via imputação de KNN.",
        "metric_patient": "Paciente",
        "metric_years": "anos",
        "metric_dialyzer": "Dialisador",
        "metric_model": "Modelo",
        "clinical_meas_title": "## 📋 Medições Clínicas (H0 – H5)",
        "clinical_meas_subtitle": "Marque as horas que deseja inserir manualmente. Horas desmarcadas serão simuladas. **H0 é obrigatório.** H1–H5 são opcionais.",
        "variable": "Variável",
        "numeric_params": "#### 🔢 Parâmetros Numéricos",
        "bath_group_title": "#### 🔘 Grupo do Banho",
        "bath_group_subtitle": "O grupo do banho é uma variável categórica codificada como três colunas binárias (`BAT_GROUP_Grupo 1/2/3`). Selecione um grupo por hora ativada. Como o grupo do banho raramente muda no meio da sessão, você pode usar **Aplicar a todas as horas** para propagar a seleção de H0 automaticamente.",
        "hour": "Hora",
        "bath_group_label": "Grupo do Banho",
        "btn_simulate": "▶  Simular e Predizer",
        "err_dataset": "Dataset não encontrado em",
        "err_dataset_msg": "Certifique-se de que `data/dataset_flat_V2.csv` está no diretório de trabalho.",
        "spin_hist": "Carregando dataset histórico…",
        "csv_success": "CSV carregado — valores preenchidos a partir da primeira linha.",
        "csv_error": "Não foi possível ler o CSV enviado",
        "err_h0": "❌ Por favor, insira pelo menos um valor clínico em H0 antes de simular.",
        "spin_knn": "Executando simulação de imputação por KNN…",
        "err_sim": "Erro de simulação",
        "err_model_path": "Modelo não encontrado em",
        "err_model_path_msg": "Verifique se o diretório `models_V2/` está presente.",
        "spin_model": "Aplicando modelo de predição…",
        "err_pred": "Erro de predição do modelo",
        "success_complete": "✅ Simulação e predição concluídas!",
        "results_title": "## 🔬 Resultados",
        "high_risk_title": "RISCO ALTO — Evento Hipotensivo Predito",
        "high_risk_body": "O modelo <b>{model_name}</b> prediz um evento hipotensivo intradiálico (TARGET = 1)",
        "high_risk_body_prob": " com probabilidade <b>{prob:.1%}</b>. Recomenda-se monitoramento contínuo do paciente.",
        "low_risk_title": "RISCO BAIXO — Nenhum Evento Hipotensivo Predito",
        "low_risk_body": "O modelo <b>{model_name}</b> prediz ausência de evento hipotensivo (TARGET = 0)",
        "low_risk_body_prob": " com probabilidade <b>{prob:.1%}</b>.",
        "metric_prediction": "Predição",
        "metric_high_risk": "RISCO ALTO",
        "metric_low_risk": "RISCO BAIXO",
        "metric_confidence": "Confiança",
        "session_data_title": "## 📊 Dados da Sessão H0–H5 *(observados + simulados)*",
        "chip_obs": "● Observado",
        "chip_sim": "◌ Simulado via KNN",
        "chip_carried": "→ Carregado adiante",
        "trajectories_title": "## 📈 Trajetórias Clínicas",
        "confidence_title": "## 🎯 Confiança da Predição",
        "interpretation": "### Interpretação",
        "crit_msg": "🔴 **CRÍTICO** — Risco muito alto. Monitoramento imediato necessário.",
        "high_msg": "🟠 **ALTO** — Risco significativo. Vigilância aumentada recomendada.",
        "mod_msg": "🟡 **MODERADO** — Limítrofe. Monitoramento padrão.",
        "low_msg": "🟢 **BAIXO** — Risco baixo. Sessão rotineira esperada.",
        "export_title": "## 💾 Exportar Relatório",
        "btn_download": "⬇  Baixar Relatório CSV",
        "report_caption": "Relatório: {rows} linhas — {obs} observados, {sim} simulados — {vars} variáveis clínicas.",
        "preview_export": "Visualizar dados de exportação",
        "footer_text": "Simulador de Risco de HD · Imputação de dados ausentes por KNN · Desenvolvido com Streamlit & Plotly · Todas as predições são ferramentas de suporte à decisão apenas e devem ser revisadas pela equipe clínica - Isto é apenas uma prova de conceito, não testada em ambiente clínico."
    },
    "en-US": {
        "page_title": "HD Risk Simulator",
        "sidebar_title": "## 🩺 HD Risk Simulator",
        "sidebar_subtitle": "Hemodialysis session risk assessment via KNN imputation",
        "patient_demo": "### 👤 Patient Demographics",
        "sex": "Sex",
        "sex_options": ["Male (0)", "Female (1)"],
        "age": "Age (years)",
        "dialyzer": "Dialyzer (DIA)",
        "dialyzer_help": "Select the dialyzer used in this session.",
        "num_code": "Numeric code",
        "pred_model": "### 🤖 Prediction Model",
        "select_model": "Select model",
        "path": "Path",
        "sim_params": "### ⚙️ Simulation Parameters",
        "k_neighbors": "K-Neighbors (data imputation)",
        "k_neighbors_help": "Number of historical sessions used to estimate missing hours.",
        "inc_demo": "Include demographics in distance",
        "inc_demo_help": "Use SEX and AGE when searching for nearest neighbours.",
        "csv_upload_title": "### 📂 CSV Upload *(optional)*",
        "csv_upload_help": "CSV must have the same column schema as dataset_flat_V2.csv (no Target).",
        "csv_uploader_label": "Upload a pre-filled session CSV",
        "main_title": "Hemodialysis Session Risk Simulator",
        "main_subtitle": "Enter at least the **H0** clinical measurements below. Any hour left blank will be **simulated** via trajectory KNN.",
        "metric_patient": "Patient",
        "metric_years": "yrs",
        "metric_dialyzer": "Dialyzer",
        "metric_model": "Model",
        "clinical_meas_title": "## 📋 Clinical Measurements (H0 – H5)",
        "clinical_meas_subtitle": "Check the hours you want to enter manually. Unchecked hours will be simulated. **H0 is required.** H1–H5 are optional.",
        "variable": "Variable",
        "numeric_params": "#### 🔢 Parameters",
        "bath_group_title": "#### 🔘 Bath Group",
        "bath_group_subtitle": "The bath group is a categorical variable encoded as three binary columns (`BAT_GROUP_Grupo 1/2/3`). Select one group per enabled hour. Because bath group rarely changes mid-session, you can use **Apply to all hours** to propagate the H0 selection automatically.",
        "hour": "Hour",
        "bath_group_label": "Bath Group",
        "btn_simulate": "▶  Simulate & Predict",
        "err_dataset": "Dataset not found at",
        "err_dataset_msg": "Make sure `data/dataset_flat_V2.csv` is in the working directory.",
        "spin_hist": "Loading historical dataset…",
        "csv_success": "CSV loaded — values populated from first row.",
        "csv_error": "Could not parse uploaded CSV",
        "err_h0": "❌ Please enter at least one H0 clinical value before simulating.",
        "spin_knn": "Running KNN trajectory simulation…",
        "err_sim": "Simulation error",
        "err_model_path": "Model not found at",
        "err_model_path_msg": "Check that the `models_V2/` directory is present.",
        "spin_model": "Applying prediction model…",
        "err_pred": "Model prediction error",
        "success_complete": "✅ Simulation and prediction complete!",
        "results_title": "## 🔬 Results",
        "high_risk_title": "HIGH RISK — Hypotensive Event Predicted",
        "high_risk_body": "Model <b>{model_name}</b> predicts an intradialytic hypotensive event (TARGET = 1)",
        "high_risk_body_prob": " with probability <b>{prob:.1%}</b>. Close patient monitoring is recommended.",
        "low_risk_title": "LOW RISK — No Hypotensive Event Predicted",
        "low_risk_body": "Model <b>{model_name}</b> predicts no hypotensive event (TARGET = 0)",
        "low_risk_body_prob": " with probability <b>{prob:.1%}</b>.",
        "metric_prediction": "Prediction",
        "metric_high_risk": "HIGH RISK",
        "metric_low_risk": "LOW RISK",
        "metric_confidence": "Confidence",
        "session_data_title": "## 📊 H0–H5 Session Data *(observed + simulated)*",
        "chip_obs": "● Observed",
        "chip_sim": "◌ KNN-simulated",
        "chip_carried": "→ Carried forward",
        "trajectories_title": "## 📈 Clinical Trajectories",
        "confidence_title": "## 🎯 Prediction Confidence",
        "interpretation": "### Interpretation",
        "crit_msg": "🔴 **CRITICAL** — Very high risk. Immediate monitoring required.",
        "high_msg": "🟠 **HIGH** — Significant risk. Increased vigilance advised.",
        "mod_msg": "🟡 **MODERATE** — Borderline. Standard monitoring.",
        "low_msg": "🟢 **LOW** — Low risk. Routine session expected.",
        "export_title": "## 💾 Export Report",
        "btn_download": "⬇  Download CSV Report",
        "report_caption": "Report: {rows} rows — {obs} observed, {sim} simulated — {vars} clinical variables.",
        "preview_export": "Preview export data",
        "footer_text": "HD Risk Simulator · KNN Trajectory Imputation · Built with Streamlit & Plotly · All predictions are decision-support tools only and must be reviewed by clinical staff - This is just an prove of concept, not tested in clinical setting."
    }
}

L = LOCALIZATION[st.session_state.lang]

# Botão seletor de idioma no topo direito da página
header_left, header_right = st.columns([6, 1])
with header_right:
    if st.session_state.lang == "pt-BR":
        if st.button("![USA](https://flagcdn.com/w20/us.png) English", use_container_width=True, help="Switch interface language to English"):
            st.session_state.lang = "en-US"
            st.rerun()
    else:
        if st.button("![Brasil](https://flagcdn.com/w20/br.png) Portguês", use_container_width=True, help="Mudar o idioma da interface para Português"):
            st.session_state.lang = "pt-BR"
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — must match dataset_flat_V2.csv column schema exactly
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_VARS = [
    "WDR", "IWG",
    "KT", "BFR", "HBC",
    "APR", "VPR", "TMP",
    "SBP", "DBP",
    "TUF",
]

BINARY_VARS = [
    "BAT_GROUP_Grupo 1 - ACF 3A5",
    "BAT_GROUP_Grupo 2 - EuCliD",
    "BAT_GROUP_Grupo 3 - Demais classes",
]

BAT_GROUP_OPTIONS = {
    "Grupo 1 - ACF 3A5":        "BAT_GROUP_Grupo 1 - ACF 3A5",
    "Grupo 2 - EuCliD":         "BAT_GROUP_Grupo 2 - EuCliD",
    "Grupo 3 - Demais classes": "BAT_GROUP_Grupo 3 - Demais classes",
}

CLINICAL_VARS = NUMERIC_VARS + BINARY_VARS
ALL_HOURS = ["H0", "H1", "H2", "H3", "H4", "H5"]

# Display labels internacionalizados
VAR_LABELS = {
    "pt-BR": {
        "WDR": "Peso Seco (Kg)",
        "IWG": "Ganho de Peso Interdialítico (Kg)",
        "KT":  "Depuração de Ureia (L)",
        "BFR": "Fluxo de Sangue (mL/min)",
        "HBC": "Condutividade do Banho (mScm)",
        "APR": "Pressão Arterial (mmHg)",
        "VPR": "Pressão Venosa (mmHg)",
        "TMP": "Pressão Transmembrana (mmHg)",
        "SBP": "Pressão Arterial Sistólica (mmHg)",
        "DBP": "Pressão Arterial Diastólica (mmHg)",
        "TUF": "Ultrafiltração Total (mL)",
        "BAT_GROUP_Grupo 1 - ACF 3A5":        "Grupo do Banho: ACF 3A5 (0/1)",
        "BAT_GROUP_Grupo 2 - EuCliD":         "Grupo do Banho: EuCliD (0/1)",
        "BAT_GROUP_Grupo 3 - Demais classes": "Grupo do Banho: Outras classes (0/1)",
    },
    "en-US": {
        "WDR": "Dry Weight (Kg)",
        "IWG": "Interdialytic Weight Gain (Kg)",
        "KT":  "Urea Clearance (L)",
        "BFR": "Blood Flow Rate (mL/min)",
        "HBC": "Bath Conductivity (mScm)",
        "APR": "Arterial Pressure (mmHg)",
        "VPR": "Venous Pressure (mmHg)",
        "TMP": "Transmembrane Pressure (mmHg)",
        "SBP": "Systolic Blood Pressure (mmHg)",
        "DBP": "Diastolic Blood Pressure (mmHg)",
        "TUF": "Total Ultrafiltration (mL)",
        "BAT_GROUP_Grupo 1 - ACF 3A5":        "Bath Group: ACF 3A5 (0/1)",
        "BAT_GROUP_Grupo 2 - EuCliD":         "Bath Group: EuCliD (0/1)",
        "BAT_GROUP_Grupo 3 - Demais classes": "Bath Group: Other classes (0/1)",
    }
}

VAR_DEFAULTS = {
    "WDR": (65.0,  20.0, 200.0, 0.1),
    "IWG": (2.0,  -5.0,  10.0, 0.1),
    "KT":  (55.0,  20.0,  85.0, 1.0),
    "BFR": (450.0, 50.0, 600.0, 5.0),
    "HBC": (13.8,  13.5,  14.5, 0.1),
    "APR": (-150.0, -300.0, 0.0, 1.0),
    "VPR": (120.0,  0.0, 300.0, 1.0),
    "TMP": (150.0,  0.0, 350.0, 1.0),
    "SBP": (130.0, 60.0, 250.0, 1.0),
    "DBP": (65.0,  40.0, 140.0, 1.0),
    "TUF": (0.5,  0.0, 1.5, 0.1),
}

DIALYZER_MAP = {
    "EuCliD - FX CorDiax 60":    1,
    "FX CorDiax 800":            2,
    "EuCliD - FX CorDiax 600":   3,
    "EuCliD - FX CorDiax 80":    4,
    "ELISIO 210":                5,
    "FX 100":                    6,
    "EuCliD - FX CorDiax 800":   7,
    "FX 80":                     8,
    "Solacea 21H":               9,
    "Evodial":                   10,
    "Sureflux 2.1":              11,
    "EuCliD - HF-80 S":          12,
    "EuCliD - Sureflux-21L":     13,
    "FILTRYZER NF-2.1H":         14,
    "EuCliD - FX60":             15,
    "EuCliD - FX-HDF-600":       16,
    "EuCliD - Sureflux - 190UX": 17,
    "FX 60":                     18,
    "EuCliD - FX80":             19,
    "TorayLight NS-21S":         20,
    "BK-21-F":                   21,
    "EuCliD - FB-190 UGA":       22,
    "EuCliD - FX-HDF-800":       23,
    "Solacea 19H":               24,
    "FX CorDiax 1000":           25,
    "FX CorDiax 600":            26,
}

MODEL_OPTIONS = {
    "K-Nearest Neighbor (KNN)":     "models_V2/modelo_knn.pkl",
    "Random Forest (RF)":           "models_V2/modelo_RF.pkl",
    "Support Vector Machine (SVM)": "models_V2/modelo_svm.pkl",
    "XGBoost":                      "models_V2/modelo_xgboost.pkl",
    "Decision Tree (DT)":           "models_V2/modelo_DT.pkl",
    "Multi-Layer Perceptron (MLP)": "models_V2/modelo_MLP.pkl",
    "Naive Bayes (NB)":             "models_V2/modelo_NB.pkl",
}

DATASET_PATH = "data/dataset_flat_V2.csv"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — clinical dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: #0a0e17; color: #c8d8e8; }

section[data-testid="stSidebar"] {
    background-color: #0d1220 !important;
    border-right: 1px solid #1e2d45;
}

h1 { font-family: 'IBM Plex Mono', monospace; color: #4fc3f7 !important; letter-spacing: -1px; }
h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #81d4fa !important; }

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1a2e 0%, #112240 100%);
    border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px;
}
div[data-testid="metric-container"] label { color: #64b5f6 !important; font-size: 0.75rem !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e3f2fd !important; font-family: 'IBM Plex Mono', monospace;
}

.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: #e3f2fd !important; border: 1px solid #1976d2 !important;
    border-radius: 6px !important; font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important; letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0) !important;
    border-color: #42a5f5 !important; box-shadow: 0 0 12px rgba(66,165,245,0.3) !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
    color: #e8f5e9 !important; border: 1px solid #388e3c !important;
    border-radius: 6px !important; font-family: 'IBM Plex Mono', monospace !important;
}

details { background: #0d1a2e !important; border: 1px solid #1e3a5f !important; border-radius: 8px !important; }

input[type="number"] { background: #0d1a2e !important; color: #c8d8e8 !important; border: 1px solid #1e3a5f !important; }

div[data-baseweb="select"] > div { background: #0d1a2e !important; border-color: #1e3a5f !important; color: #c8d8e8 !important; }

.alert-danger {
    background: linear-gradient(135deg, #3e0000, #5c1a1a);
    border: 1px solid #c62828; border-left: 4px solid #f44336;
    border-radius: 8px; padding: 16px 20px; color: #ffcdd2;
    font-family: 'IBM Plex Mono', monospace;
}
.alert-safe {
    background: linear-gradient(135deg, #003300, #1a3d1a);
    border: 1px solid #2e7d32; border-left: 4px solid #4caf50;
    border-radius: 8px; padding: 16px 20px; color: #c8e6c9;
    font-family: 'IBM Plex Mono', monospace;
}

.chip-observed {
    background: #0d3b5e; color: #4fc3f7; border: 1px solid #1976d2;
    padding: 2px 8px; border-radius: 12px;
    font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace;
}
.chip-simulated {
    background: #1a2700; color: #aed581; border: 1px solid #558b2f;
    padding: 2px 8px; border-radius: 12px;
    font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace;
}
.chip-carried {
    background: #1a1400; color: #ffcc80; border: 1px solid #e65100;
    padding: 2px 8px; border-radius: 12px;
    font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace;
}

hr { border-color: #1e3a5f !important; }
.stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 8px; }
.stTabs [data-baseweb="tab-list"] { background: #0d1a2e; border-bottom: 1px solid #1e3a5f; }
.stTabs [data-baseweb="tab"] { color: #64b5f6 !important; font-family: 'IBM Plex Mono', monospace; }
.stTabs [aria-selected="true"] { color: #4fc3f7 !important; border-bottom: 2px solid #4fc3f7 !important; }
.stSlider > div > div > div { background: #1976d2 !important; }
.stRadio > div { color: #c8d8e8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION CORE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")

def simulate_missing_hours(
    observed: dict,
    sex: int,
    age: int,
    df: pd.DataFrame,
    k: int = 10,
    use_demographics: bool = True,
) -> dict:
    feature_cols: list[str] = []
    query_values: list[float] = []

    if use_demographics:
        feature_cols += ["SEX", "AGE"]
        query_values += [sex, age]

    for var in CLINICAL_VARS:
        for hour in ALL_HOURS:
            col = f"{var}_{hour}"
            if col not in df.columns:
                continue
            val = observed.get(var, {}).get(hour)
            if val is not None:
                feature_cols.append(col)
                query_values.append(float(val))

    if not feature_cols:
        raise ValueError(
            "No observed values found. Provide at least one H0 clinical measurement."
        )

    numeric_output_cols = [
        f"{v}_{h}"
        for v in NUMERIC_VARS
        for h in ALL_HOURS
        if f"{v}_{h}" in df.columns
    ]
    needed_cols = list(dict.fromkeys(feature_cols + numeric_output_cols))
    df_clean = df[needed_cols].dropna()

    X = df_clean[feature_cols].values.astype(float)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    q_sc = scaler.transform([query_values])

    n_neighbors = min(k, len(df_clean))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(X_sc)
    distances, indices = nbrs.kneighbors(q_sc)

    distances = distances[0]
    indices = indices[0]
    eps = 1e-8
    weights = 1.0 / (distances + eps)
    weights /= weights.sum()

    result: dict = {}
    for var in NUMERIC_VARS:
        result[var] = {}
        for hour in ALL_HOURS:
            col = f"{var}_{hour}"
            val_obs = observed.get(var, {}).get(hour)
            if val_obs is not None:
                result[var][hour] = float(val_obs)
            elif col in df_clean.columns:
                neighbor_vals = df_clean.iloc[indices][col].values.astype(float)
                result[var][hour] = float(np.average(neighbor_vals, weights=weights))
            else:
                result[var][hour] = None

    for var in BINARY_VARS:
        result[var] = {}
        last_known: float | None = None
        for hour in ALL_HOURS:
            val_obs = observed.get(var, {}).get(hour)
            if val_obs is not None:
                result[var][hour] = float(val_obs)
                last_known = float(val_obs)
            else:
                result[var][hour] = last_known

    return result

def build_flat_vector(full_result: dict, sex: int, age: int, dia: int) -> pd.DataFrame:
    row = {"SEX": sex, "AGE": age, "DIA": dia}
    for var in CLINICAL_VARS:
        for hour in ALL_HOURS:
            col = f"{var}_{hour}"
            row[col] = full_result.get(var, {}).get(hour)
    return pd.DataFrame([row])

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    errors = {}
    try:
        return joblib.load(model_path)
    except Exception as e:
        errors["joblib"] = str(e)

    try:
        with open(model_path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except Exception as e:
        errors["pickle+latin1"] = str(e)

    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        errors["pickle"] = str(e)

    raise RuntimeError(
        "Could not load model. All strategies failed.\n"
        "Details:\n" + "\n".join(f"  {k}: {v}" for k, v in errors.items())
    )

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#0a0e17",
    plot_bgcolor="#0d1220",
    font=dict(family="IBM Plex Mono, monospace", color="#c8d8e8", size=11),
    xaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45", tickfont=dict(size=10)),
    legend=dict(bgcolor="#0d1a2e", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=40),
    height=280,
)
OBS_COLOR = "#4fc3f7"
SIM_COLOR = "#aed581"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(L["sidebar_title"])
    st.caption(L["sidebar_subtitle"])
    st.divider()

    st.markdown(L["patient_demo"])
    
    sex_label = st.radio(L["sex"], L["sex_options"], horizontal=True)
    sex_val = 0 if "0" in sex_label else 1

    age_val = st.number_input(
        L["age"], min_value=1, max_value=120, value=65, step=1
    )

    dia_label = st.selectbox(
        L["dialyzer"],
        options=list(DIALYZER_MAP.keys()),
        index=0,
        help=L["dialyzer_help"],
    )
    dia_val = DIALYZER_MAP[dia_label]
    st.caption(f"{L['num_code']}: **{dia_val}**")

    st.divider()

    st.markdown(L["pred_model"])
    model_name = st.selectbox(L["select_model"], list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[model_name]
    st.caption(f"{L['path']}: `{model_path}`")

    st.divider()

    st.markdown(L["sim_params"])
    k_neighbors = st.slider(
        L["k_neighbors"],
        min_value=3, max_value=50, value=10, step=1,
        help=L["k_neighbors_help"],
    )
    use_demo = st.toggle(
        L["inc_demo"],
        value=True,
        help=L["inc_demo_help"],
    )

    st.divider()

    st.markdown(L["csv_upload_title"])
    uploaded_csv = st.file_uploader(
        L["csv_uploader_label"],
        type=["csv"],
        help=L["csv_upload_help"],
    )

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.markdown(f"<h1>{L['main_title']}</h1>", unsafe_allow_html=True)
    st.markdown(L["main_subtitle"])
with col_meta:
    gender_symbol = '♂' if sex_val == 0 else '♀'
    st.metric(L["metric_patient"], f"{gender_symbol} · {age_val} {L['metric_years']}")
    st.metric(L["metric_dialyzer"], dia_label.split(" - ")[-1][:20])
    st.metric(L["metric_model"], model_name.split("(")[0].strip())

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL DATA INPUT GRID
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(L["clinical_meas_title"])
st.caption(L["clinical_meas_subtitle"])

hour_enabled = {}
hdr_cols = st.columns([2] + [1] * 6)
hdr_cols[0].markdown(f"**{L['variable']}**")
for i, h in enumerate(ALL_HOURS):
    hour_enabled[h] = hdr_cols[i + 1].checkbox(h, value=(h == "H0"), key=f"hour_{h}")

st.markdown("")

observed: dict = {var: {} for var in CLINICAL_VARS}

# Grupos estruturados por idioma
NUMERIC_GROUPS = {
    "pt-BR": {
        "Peso": ["WDR", "IWG"],
        "Hemodinâmica": ["SBP", "DBP"],
        "Pressões": ["APR", "VPR", "TMP"],
        "Fluxo e Depuração": ["BFR", "KT", "TUF"],
        "Banho": ["HBC"],
    },
    "en-US": {
        "Weight": ["WDR", "IWG"],
        "Haemodynamics": ["SBP", "DBP"],
        "Pressures": ["APR", "VPR", "TMP"],
        "Flow & Clearance": ["BFR", "KT", "TUF"],
        "Bath": ["HBC"],
    }
}

st.markdown(L["numeric_params"])
for group_name, vars_in_group in NUMERIC_GROUPS[st.session_state.lang].items():
    is_expanded = group_name in ["Hemodinâmica", "Haemodynamics", "Peso", "Weight"]
    with st.expander(f"**{group_name}**", expanded=is_expanded):
        for var in vars_in_group:
            default, mn, mx, step = VAR_DEFAULTS[var]
            inp_cols = st.columns([2] + [1] * 6)
            inp_cols[0].markdown(f"*{var}* — {VAR_LABELS[st.session_state.lang][var]}", unsafe_allow_html=False)
            for i, h in enumerate(ALL_HOURS):
                if hour_enabled[h]:
                    val = inp_cols[i + 1].number_input(
                        label=f"{var} {h}",
                        min_value=float(mn),
                        max_value=float(mx),
                        value=float(default),
                        step=float(step),
                        key=f"{var}_{h}",
                        label_visibility="collapsed",
                    )
                    observed[var][h] = val
                else:
                    inp_cols[i + 1].markdown("<span style='color:#2a3a50'>—</span>", unsafe_allow_html=True)

st.markdown(L["bath_group_title"])
st.caption(L["bath_group_subtitle"])

_bat_groups = list(BAT_GROUP_OPTIONS.keys())

bat_hdr = st.columns([2] + [1] * 6)
bat_hdr[0].markdown(f"**{L['hour']}**")
for i, h in enumerate(ALL_HOURS):
    bat_hdr[i + 1].markdown(f"**{h}**" if hour_enabled[h] else f"~~{h}~~")

bat_row = st.columns([2] + [1] * 6)
bat_row[0].markdown(f"*{L['bath_group_label']}*")

bat_selections: dict[str, str | None] = {}
for i, h in enumerate(ALL_HOURS):
    if hour_enabled[h]:
        sel = bat_row[i + 1].selectbox(
            label=f"BAT {h}",
            options=_bat_groups,
            index=0,
            key=f"BAT_GROUP_{h}",
            label_visibility="collapsed",
        )
        bat_selections[h] = sel
    else:
        bat_row[i + 1].markdown("<span style='color:#2a3a50'>—</span>", unsafe_allow_html=True)
        bat_selections[h] = None

for group_label, col_name in BAT_GROUP_OPTIONS.items():
    for h in ALL_HOURS:
        sel = bat_selections.get(h)
        if sel is not None:
            observed[col_name][h] = 1.0 if sel == group_label else 0.0

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATE BUTTON & LOGIC
# ─────────────────────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
run_btn = run_col.button(L["btn_simulate"], use_container_width=True)

if run_btn:
    if not os.path.exists(DATASET_PATH):
        st.error(f"{L['err_dataset']} `{DATASET_PATH}`. {L['err_dataset_msg']}")
        st.stop()

    with st.spinner(L["spin_hist"]):
        df_hist = load_dataset(DATASET_PATH)

    if uploaded_csv is not None:
        try:
            df_up = pd.read_csv(uploaded_csv, sep=";")
            for var in CLINICAL_VARS:
                for h in ALL_HOURS:
                    col = f"{var}_{h}"
                    if col in df_up.columns and not df_up[col].isna().all():
                        observed[var][h] = float(df_up[col].iloc[0])
            if "SEX" in df_up.columns:
                sex_val = int(df_up["SEX"].iloc[0])
            if "AGE" in df_up.columns:
                age_val = int(df_up["AGE"].iloc[0])
            if "DIA" in df_up.columns:
                dia_val = int(df_up["DIA"].iloc[0])
            st.success(L["csv_success"])
        except Exception as e:
            st.warning(f"{L['csv_error']}: {e}")

    any_h0 = any(observed[v].get("H0") is not None for v in CLINICAL_VARS)
    if not any_h0:
        st.error(L["err_h0"])
        st.stop()

    with st.spinner(L["spin_knn"]):
        try:
            full_result = simulate_missing_hours(
                observed=observed,
                sex=sex_val,
                age=age_val,
                df=df_hist,
                k=k_neighbors,
                use_demographics=use_demo,
            )
        except Exception as e:
            st.error(f"{L['err_sim']}: {e}")
            st.stop()

    observed_hours = {
        var: {h for h, v in observed[var].items() if v is not None}
        for var in CLINICAL_VARS
    }

    carried_hours = {
        var: {
            h for h in ALL_HOURS
            if h not in observed_hours[var] and var in BINARY_VARS
               and full_result[var].get(h) is not None
        }
        for var in BINARY_VARS
    }

    X_model = build_flat_vector(full_result, sex=sex_val, age=age_val, dia=dia_val)

    if not os.path.exists(model_path):
        st.error(f"{L['err_model_path']} `{model_path}`. {L['err_model_path_msg']}")
        st.stop()

    with st.spinner(L["spin_model"]):
        try:
            model = load_model(model_path)
            prediction = int(model.predict(X_model)[0])
            prob = (
                float(model.predict_proba(X_model)[0][1])
                if hasattr(model, "predict_proba") else None
            )
        except Exception as e:
            st.error(f"**{L['err_pred']}**")
            st.code(str(e))
            st.stop()

    st.success(L["success_complete"])
    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS — ALERT + METRICS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(L["results_title"])

    if prediction == 1:
        alert_class, alert_icon = "alert-danger", "⚠️"
        alert_title = L["high_risk_title"]
        alert_body = L["high_risk_body"].format(model_name=model_name)
        if prob is not None:
            alert_body += L["high_risk_body_prob"].format(prob=prob)
    else:
        alert_class, alert_icon = "alert-safe", "✅"
        alert_title = L["low_risk_title"]
        alert_body = L["low_risk_body"].format(model_name=model_name)
        if prob is not None:
            alert_body += L["low_risk_body_prob"].format(prob=(1 - prob))

    st.markdown(
        f"""<div class="{alert_class}">
            <div style="font-size:1.3rem;font-weight:600;">{alert_icon} {alert_title}</div>
            <div style="margin-top:8px;font-family:'IBM Plex Sans',sans-serif;font-size:0.95rem;">{alert_body}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(L["metric_prediction"], L["metric_high_risk"] if prediction == 1 else L["metric_low_risk"])
    m2.metric("TARGET", str(prediction))
    m3.metric(L["metric_dialyzer"], dia_label.split(" - ")[-1][:15])
    
    if prob is not None:
        m4.metric("P(TARGET=1)", f"{prob:.1%}")
        m5.metric(L["metric_confidence"], f"{max(prob, 1 - prob):.1%}")
    else:
        m4.metric("P(TARGET=1)", "N/A")
        m5.metric(L["metric_model"], model_name.split("(")[0].strip()[:14])

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(L["session_data_title"])
    st.caption(
        f"<span class='chip-observed'>{L['chip_obs']}</span>&nbsp;&nbsp;"
        f"<span class='chip-simulated'>{L['chip_sim']}</span>&nbsp;&nbsp;"
        f"<span class='chip-carried'>{L['chip_carried']}</span>",
        unsafe_allow_html=True,
    )

    rows = []
    for var in CLINICAL_VARS:
        row = {L["variable"]: f"{var} — {VAR_LABELS[st.session_state.lang][var]}"}
        for h in ALL_HOURS:
            val = full_result[var].get(h)
            is_obs = h in observed_hours[var]
            is_carried = var in BINARY_VARS and h in carried_hours.get(var, set())
            if val is not None:
                tag = "●" if is_obs else ("→" if is_carried else "◌")
                row[h] = f"{tag} {val:.3f}"
            else:
                row[h] = "—"
        rows.append(row)

    df_display = pd.DataFrame(rows).set_index(L["variable"])

    def style_cell(val):
        if isinstance(val, str) and val.startswith("●"):
            return "color: #4fc3f7; background-color: #0d253f;"
        elif isinstance(val, str) and val.startswith("◌"):
            return "color: #aed581; background-color: #162300;"
        elif isinstance(val, str) and val.startswith("→"):
            return "color: #ffcc80; background-color: #1a1200;"
        return ""

    st.dataframe(df_display.style.map(style_cell), use_container_width=True)
    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTIVE CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(L["trajectories_title"])

    VAR_GROUPS = {
        "pt-BR": {
            "Hemodinâmica":    ["SBP", "DBP"],
            "Pressões":        ["APR", "VPR", "TMP"],
            "Fluxo e Depuração": ["BFR", "KT", "TUF"],
            "Peso":           ["WDR", "IWG"],
            "Banho":             ["HBC"],
            "Grupos de Banho":      ["BAT_GROUP_Grupo 1 - ACF 3A5", "BAT_GROUP_Grupo 2 - EuCliD", "BAT_GROUP_Grupo 3 - Demais classes"],
        },
        "en-US": {
            "Haemodynamics":    ["SBP", "DBP"],
            "Pressures":        ["APR", "VPR", "TMP"],
            "Flow & Clearance": ["BFR", "KT", "TUF"],
            "Weight":           ["WDR", "IWG"],
            "Bath":             ["HBC"],
            "Bath Groups":      ["BAT_GROUP_Grupo 1 - ACF 3A5", "BAT_GROUP_Grupo 2 - EuCliD", "BAT_GROUP_Grupo 3 - Demais classes"],
        }
    }

    current_groups = VAR_GROUPS[st.session_state.lang]
    tabs = st.tabs(list(current_groups.keys()))
    for tab, (group_name, vars_in_group) in zip(tabs, current_groups.items()):
        with tab:
            n = len(vars_in_group)
            cols = st.columns(min(n, 3))
            for ci, var in enumerate(vars_in_group):
                col = cols[ci % 3]
                with col:
                    vals = [full_result[var].get(h) for h in ALL_HOURS]
                    obs_set = observed_hours[var]

                    x_obs = [i for i, h in enumerate(ALL_HOURS) if h in obs_set and vals[i] is not None]
                    y_obs = [vals[i] for i in x_obs]
                    x_sim = [i for i, h in enumerate(ALL_HOURS) if h not in obs_set and vals[i] is not None]
                    y_sim = [vals[i] for i in x_sim]

                    fig = go.Figure()
                    x_all = [i for i, v in enumerate(vals) if v is not None]
                    y_all = [v for v in vals if v is not None]
                    fig.add_trace(go.Scatter(
                        x=x_all, y=y_all, mode="lines",
                        line=dict(color="#1e3a5f", width=1.5, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ))
                    if x_obs:
                        fig.add_trace(go.Scatter(
                            x=x_obs, y=y_obs, mode="markers+lines", name=L["chip_obs"].replace("● ", ""),
                            marker=dict(size=9, color=OBS_COLOR, symbol="circle", line=dict(color="#e3f2fd", width=1)),
                            line=dict(color=OBS_COLOR, width=2),
                        ))
                    if x_sim:
                        fig.add_trace(go.Scatter(
                            x=x_sim, y=y_sim, mode="markers+lines", name=L["chip_sim"].replace("◌ ", ""),
                            marker=dict(size=9, color=SIM_COLOR, symbol="diamond", line=dict(color="#f9fbe7", width=1)),
                            line=dict(color=SIM_COLOR, width=2, dash="dash"),
                        ))

                    layout = {**PLOT_LAYOUT}
                    short_label = var.replace("BAT_GROUP_", "")
                    layout["title"] = dict(text=short_label, font=dict(size=13, color="#81d4fa"))
                    layout["xaxis"] = dict(tickmode="array", tickvals=list(range(6)), ticktext=ALL_HOURS, gridcolor="#1e2d45")
                    layout["yaxis"] = dict(title=VAR_LABELS[st.session_state.lang].get(var, "").split("(")[-1].replace(")", ""), gridcolor="#1e2d45")
                    layout["showlegend"] = True
                    layout["legend"] = dict(orientation="h", y=1.12, x=0)
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # PROBABILITY GAUGE
    # ─────────────────────────────────────────────────────────────────────────
    if prob is not None:
        st.markdown(L["confidence_title"])
        gauge_col, text_col = st.columns([1, 1])
        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"color": "#e3f2fd", "family": "IBM Plex Mono", "size": 32}},
                delta={"reference": 50, "increasing": {"color": "#f44336"}, "decreasing": {"color": "#4caf50"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64b5f6"},
                    "bar": {"color": "#f44336" if prob >= 0.5 else "#4caf50"},
                    "bgcolor": "#0d1a2e",
                    "borderwidth": 1, "bordercolor": "#1e3a5f",
                    "steps": [
                        {"range": [0, 30],  "color": "#1b3a2d"},
                        {"range": [30, 60], "color": "#2a3a00"},
                        {"range": [60, 80], "color": "#3e2000"},
                        {"range": [80, 100],"color": "#3e0000"},
                    ],
                    "threshold": {"line": {"color": "#ffeb3b", "width": 3}, "thickness": 0.8, "value": 50},
                },
                title={"text": "P(Hypotension)", "font": {"color": "#81d4fa", "family": "IBM Plex Mono"}},
            ))
            fig_gauge.update_layout(paper_bgcolor="#0a0e17", height=300, margin=dict(l=30, r=30, t=40, b=10), font=dict(color="#c8d8e8"))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with text_col:
            st.markdown(L["interpretation"])
            for thr, msg_key in [
                (0.8, "crit_msg"),
                (0.6, "high_msg"),
                (0.4, "mod_msg"),
                (0.0, "low_msg"),
            ]:
                if prob >= thr:
                    st.markdown(L[msg_key])
                    break

            st.markdown(f"""
| Metric | Value |
|--------|-------|
| P(TARGET=1) | `{prob:.4f}` |
| P(TARGET=0) | `{1 - prob:.4f}` |
| Predicted class | `{prediction}` |
| Model | `{model_name.split("(")[0].strip()}` |
| SEX | `{'Female' if sex_val == 1 else 'Male'}` |
| AGE | `{age_val}` |
| DIA | `{dia_label}` |
""")

        st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(L["export_title"])

    export_rows = []
    source_translation = {
        "pt-BR": {"Observed": "Observado", "Carried forward": "Carregado adiante", "KNN-simulated": "Simulado via KNN"},
        "en-US": {"Observed": "Observed", "Carried forward": "Carried forward", "KNN-simulated": "KNN-simulated"}
    }
    
    for var in CLINICAL_VARS:
        for h in ALL_HOURS:
            val = full_result[var].get(h)
            is_obs = h in observed_hours[var]
            is_carried = var in BINARY_VARS and h in carried_hours.get(var, set())
            
            if is_obs:
                src_key = "Observed"
            elif is_carried:
                src_key = "Carried forward"
            else:
                src_key = "KNN-simulated"
                
            export_rows.append({
                "Variable": var,
                "Hour": h,
                "Value": round(val, 4) if val is not None else None,
                "Source": source_translation[st.session_state.lang][src_key],
                "SEX": sex_val,
                "AGE": age_val,
                "DIA_code": dia_val,
                "DIA_label": dia_label,
                "Prediction_TARGET": prediction,
                "Probability_TARGET1": round(prob, 4) if prob is not None else None,
                "Model": model_name,
                "K_neighbors": k_neighbors,
                "Use_demographics": use_demo,
            })

    df_export = pd.DataFrame(export_rows)
    buf = io.StringIO()
    df_export.to_csv(buf, index=False, sep=";")

    exp_col1, exp_col2 = st.columns([1, 3])
    exp_col1.download_button(
        label=L["btn_download"],
        data=buf.getvalue(),
        file_name=f"hd_session_SEX{sex_val}_AGE{age_val}_DIA{dia_val}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    n_obs = sum(1 for r in export_rows if r["Source"] in ["Observado", "Observed"])
    n_sim = sum(1 for r in export_rows if r["Source"] in ["Simulado via KNN", "KNN-simulated"])
    exp_col2.caption(L["report_caption"].format(rows=len(df_export), obs=n_obs, sim=n_sim, vars=len(CLINICAL_VARS)))

    with st.expander(L["preview_export"]):
        st.dataframe(df_export, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(L["footer_text"])