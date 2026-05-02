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
# CONSTANTS — must match dataset_flat_V2.csv column schema exactly
# ─────────────────────────────────────────────────────────────────────────────

# Numeric clinical variables (one input per hour per variable)
NUMERIC_VARS = [
    "WDR", "WPR", "WPO", "IWG",
    "KT", "BFR", "HBC",
    "APR", "VPR", "TMP",
    "SBP", "DBP",
    "TUF",
]

# Binary/categorical variables encoded as 0/1 in the dataset.
# All three BAT_GROUP columns are present in dataset_flat_V2.csv
# (no drop_first was used in the notebook's get_dummies call).
BINARY_VARS = [
    "BAT_GROUP_Grupo 1 - ACF 3A5",
    "BAT_GROUP_Grupo 2 - EuCliD",
    "BAT_GROUP_Grupo 3 - Demais classes",
]

# Human-readable labels for the bath group radio selector
BAT_GROUP_OPTIONS = {
    "Grupo 1 - ACF 3A5":        "BAT_GROUP_Grupo 1 - ACF 3A5",
    "Grupo 2 - EuCliD":         "BAT_GROUP_Grupo 2 - EuCliD",
    "Grupo 3 - Demais classes": "BAT_GROUP_Grupo 3 - Demais classes",
}

# Full ordered list used by simulate_missing_hours and build_flat_vector
CLINICAL_VARS = NUMERIC_VARS + BINARY_VARS

ALL_HOURS = ["H0", "H1", "H2", "H3", "H4", "H5"]

# ── Variable display labels ──────────────────────────────────────────────────
VAR_LABELS = {
    "WDR": "Dry Weight (Kg)",
    "WPR": "Pre-dialysis Weight (Kg)",
    "WPO": "Post-dialysis Weight (Kg)",
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

# (default, min, max, step) for numeric inputs
VAR_DEFAULTS = {
    "WDR": (65.0,  20.0, 200.0, 0.1),
    "WPR": (67.0,  20.0, 200.0, 0.1),
    "WPO": (65.0,  20.0, 200.0, 0.1),
    "IWG": (2.0,  -5.0,  10.0, 0.1),
    "KT":  (5.0,   0.0,  80.0, 0.1),
    "BFR": (350.0, 50.0, 500.0, 1.0),
    "HBC": (14.0,  8.0,  20.0, 0.1),
    "APR": (-150.0, -300.0, 0.0, 1.0),
    "VPR": (120.0,  0.0, 300.0, 1.0),
    "TMP": (200.0,  0.0, 600.0, 1.0),
    "SBP": (130.0, 60.0, 250.0, 1.0),
    "DBP": (80.0,  40.0, 140.0, 1.0),
    "TUF": (500.0,  0.0, 5000.0, 10.0),
}

# ── Dialyzer lookup ──────────────────────────────────────────────────────────
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

# ── Model options ─────────────────────────────────────────────────────────────
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
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HD Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    """
    Simulates missing hourly clinical measurements using trajectory KNN
    with inverse-distance weighting.

    Parameters
    ----------
    observed : dict
        {variable: {hour: value | None}}
        Hours with non-None values are treated as observed.
    sex, age : int
        Patient demographics (used in distance if use_demographics=True).
    df : pd.DataFrame
        Historical session dataset (dataset_flat_V2.csv).
    k : int
        Number of nearest neighbours.
    use_demographics : bool
        Include SEX/AGE in the feature vector for neighbour search.

    Returns
    -------
    dict  {variable: {hour: float | None}}
    """
    feature_cols = []
    query_values = []

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

    all_clinical_cols = [
        f"{v}_{h}"
        for v in CLINICAL_VARS
        for h in ALL_HOURS
        if f"{v}_{h}" in df.columns
    ]

    needed_cols = list(dict.fromkeys(feature_cols + all_clinical_cols))
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

    result = {}
    for var in CLINICAL_VARS:
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
    return result


def build_flat_vector(full_result: dict, sex: int, age: int, dia: int) -> pd.DataFrame:
    """
    Converts the simulate_missing_hours output dict into a single-row
    DataFrame with columns in the same order as dataset_flat_V2.csv,
    ready to be passed to a pre-trained .pkl model.

    Note: dataset_flat_V2.csv was saved without index=False, so pandas
    wrote the row index as 'Unnamed: 0'. The model was fitted with that
    column present, so we must include it (value 0 is used as a placeholder).
    """
    row = {"SEX": sex, "AGE": age, "DIA": dia}
    for var in CLINICAL_VARS:
        for hour in ALL_HOURS:
            col = f"{var}_{hour}"
            row[col] = full_result.get(var, {}).get(hour)
    return pd.DataFrame([row])


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Robust .pkl loader with three fallback strategies to handle
    cross-version scikit-learn pickle mismatches.
    """
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
        "The .pkl was likely saved with a different scikit-learn / Python version.\n"
        "Re-save with: import joblib; joblib.dump(model, 'path.pkl')\n\n"
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
OBS_COLOR = "#4fc3f7"   # light blue  → observed
SIM_COLOR = "#aed581"   # lime green  → simulated


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 HD Risk Predictor")
    st.caption("Hemodialysis session risk assessment via KNN trajectory simulation")
    st.divider()

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown("### 👤 Patient Demographics")

    sex_label = st.radio("Sex", ["Male (0)", "Female (1)"], horizontal=True)
    sex_val = 0 if sex_label.startswith("Male") else 1

    age_val = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=65, step=1
    )

    dia_label = st.selectbox(
        "Dialyzer (DIA)",
        options=list(DIALYZER_MAP.keys()),
        index=0,
        help="Select the dialyzer used in this session.",
    )
    dia_val = DIALYZER_MAP[dia_label]
    st.caption(f"Numeric code: **{dia_val}**")

    st.divider()

    # ── Model selection ───────────────────────────────────────────────────────
    st.markdown("### 🤖 Prediction Model")
    model_name = st.selectbox("Select model", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[model_name]
    st.caption(f"Path: `{model_path}`")

    st.divider()

    # ── Simulation parameters ─────────────────────────────────────────────────
    st.markdown("### ⚙️ Simulation Parameters")
    k_neighbors = st.slider(
        "K-Neighbors (trajectory imputation)",
        min_value=3, max_value=50, value=10, step=1,
        help="Number of historical sessions used to estimate missing hours.",
    )
    use_demo = st.toggle(
        "Include demographics in distance",
        value=True,
        help="Use SEX and AGE when searching for nearest neighbours.",
    )

    st.divider()

    # ── CSV upload (optional) ─────────────────────────────────────────────────
    st.markdown("### 📂 CSV Upload *(optional)*")
    uploaded_csv = st.file_uploader(
        "Upload a pre-filled session CSV",
        type=["csv"],
        help="CSV must have the same column schema as dataset_flat_V2.csv (no Target).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.markdown("# Hemodialysis Session Risk Predictor")
    st.markdown(
        "Enter at least the **H0** clinical measurements below. "
        "Any hour left blank will be **simulated** via trajectory KNN."
    )
with col_meta:
    st.metric("Patient", f"{'♂' if sex_val == 1 else '♀'} · {age_val} yrs")
    st.metric("Dialyzer", dia_label.split(" - ")[-1][:20])
    st.metric("Model", model_name.split("(")[0].strip())

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL DATA INPUT GRID
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📋 Clinical Measurements (H0 – H5)")
st.caption(
    "Check the hours you want to enter manually. Unchecked hours will be simulated. "
    "**H0 is required.** H1–H5 are optional."
)

# Hour-enable checkboxes
hour_enabled = {}
hdr_cols = st.columns([2] + [1] * 6)
hdr_cols[0].markdown("**Variable**")
for i, h in enumerate(ALL_HOURS):
    hour_enabled[h] = hdr_cols[i + 1].checkbox(h, value=(h == "H0"), key=f"hour_{h}")

st.markdown("")

# Build input dict — None means "simulate this slot"
observed: dict = {var: {} for var in CLINICAL_VARS}

# ── Numeric variables ──────────────────────────────────────────────────────
st.markdown("#### 🔢 Numeric Parameters")
# Group for readability
NUMERIC_GROUPS = {
    "Weight": ["WDR", "WPR", "WPO", "IWG"],
    "Haemodynamics": ["SBP", "DBP"],
    "Pressures": ["APR", "VPR", "TMP"],
    "Flow & Clearance": ["BFR", "KT", "TUF"],
    "Bath": ["HBC"],
}

for group_name, vars_in_group in NUMERIC_GROUPS.items():
    with st.expander(f"**{group_name}**", expanded=(group_name in ["Haemodynamics", "Weight"])):
        for var in vars_in_group:
            default, mn, mx, step = VAR_DEFAULTS[var]
            inp_cols = st.columns([2] + [1] * 6)
            inp_cols[0].markdown(f"*{var}* — {VAR_LABELS[var]}", unsafe_allow_html=False)
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
                    inp_cols[i + 1].markdown(
                        "<span style='color:#2a3a50'>—</span>",
                        unsafe_allow_html=True,
                    )

# ── Bath Group (mutually exclusive — one-hot encoded in the dataset) ──────
st.markdown("#### 🔘 Bath Group")
st.caption(
    "The bath group is a categorical variable encoded as three binary columns "
    "(`BAT_GROUP_Grupo 1/2/3`). Select one group per enabled hour. "
    "Because bath group rarely changes mid-session, you can use **Apply to all hours** "
    "to propagate the H0 selection automatically."
)

# Convenience: copy H0 selection to all enabled hours
_bat_groups = list(BAT_GROUP_OPTIONS.keys())

bat_hdr = st.columns([2] + [1] * 6)
bat_hdr[0].markdown("**Hour**")
for i, h in enumerate(ALL_HOURS):
    bat_hdr[i + 1].markdown(f"**{h}**" if hour_enabled[h] else f"~~{h}~~")

bat_row = st.columns([2] + [1] * 6)
bat_row[0].markdown("*Bath Group*")

bat_selections: dict[str, str | None] = {}   # hour → selected group label
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

# Convert bath group selections → per-variable, per-hour binary values
for group_label, col_name in BAT_GROUP_OPTIONS.items():
    for h in ALL_HOURS:
        sel = bat_selections.get(h)
        if sel is not None:
            observed[col_name][h] = 1.0 if sel == group_label else 0.0

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATE BUTTON
# ─────────────────────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
run_btn = run_col.button("▶  Simulate & Predict", use_container_width=True)

if run_btn:

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        st.error(
            f"Dataset not found at `{DATASET_PATH}`. "
            "Make sure `data/dataset_flat_V2.csv` is in the working directory."
        )
        st.stop()

    with st.spinner("Loading historical dataset…"):
        df_hist = load_dataset(DATASET_PATH)

    # ── Optional CSV override ─────────────────────────────────────────────────
    if uploaded_csv is not None:
        try:
            df_up = pd.read_csv(uploaded_csv, sep=";")
            for var in CLINICAL_VARS:
                for h in ALL_HOURS:
                    col = f"{var}_{h}"
                    if col in df_up.columns and not df_up[col].isna().all():
                        observed[var][h] = float(df_up[col].iloc[0])
            # Override demographics if present
            if "SEX" in df_up.columns:
                sex_val = int(df_up["SEX"].iloc[0])
            if "AGE" in df_up.columns:
                age_val = int(df_up["AGE"].iloc[0])
            if "DIA" in df_up.columns:
                dia_val = int(df_up["DIA"].iloc[0])
            st.success("CSV loaded — values populated from first row.")
        except Exception as e:
            st.warning(f"Could not parse uploaded CSV: {e}")

    # ── Validate H0 presence ──────────────────────────────────────────────────
    any_h0 = any(
        observed[v].get("H0") is not None
        for v in CLINICAL_VARS
    )
    if not any_h0:
        st.error("❌ Please enter at least one H0 clinical value before simulating.")
        st.stop()

    # ── KNN trajectory simulation ─────────────────────────────────────────────
    with st.spinner("Running KNN trajectory simulation…"):
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
            st.error(f"Simulation error: {e}")
            st.stop()

    # Track which hours were observed vs simulated
    observed_hours = {
        var: {h for h, v in observed[var].items() if v is not None}
        for var in CLINICAL_VARS
    }

    # ── Build flat vector → model ─────────────────────────────────────────────
    X_model = build_flat_vector(full_result, sex=sex_val, age=age_val, dia=dia_val)

    # ── Load model and predict ────────────────────────────────────────────────
    if not os.path.exists(model_path):
        st.error(
            f"Model not found at `{model_path}`. "
            "Check that the `models_V2/` directory is present."
        )
        st.stop()

    with st.spinner("Applying prediction model…"):
        try:
            model = load_model(model_path)
            prediction = int(model.predict(X_model)[0])
            prob = (
                float(model.predict_proba(X_model)[0][1])
                if hasattr(model, "predict_proba") else None
            )
        except Exception as e:
            err_msg = str(e)
            st.error("**Model prediction error**")
            if "STACK_GLOBAL requires str" in err_msg or "requires str" in err_msg:
                st.warning(
                    "**Root cause:** The `.pkl` was saved with an older scikit-learn / "
                    "Python version and cannot be deserialised by the current environment.\n\n"
                    "**Fix:** Re-save the model in your training notebook:\n"
                    "```python\n"
                    "import joblib\n"
                    "joblib.dump(model, 'models_V2/modelo_knn.pkl')\n"
                    "```\n"
                    "Then replace the file and restart the dashboard."
                )
            else:
                st.code(err_msg)
            st.stop()

    st.success("✅ Simulation and prediction complete!")
    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS — ALERT + METRICS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("## 🔬 Results")

    if prediction == 1:
        alert_class, alert_icon = "alert-danger", "⚠️"
        alert_title = "HIGH RISK — Hypotensive Event Predicted"
        alert_body = (
            f"Model <b>{model_name}</b> predicts an intradialytic hypotensive event "
            f"(TARGET = 1)"
            + (f" with probability <b>{prob:.1%}</b>." if prob is not None else ".")
            + " Close patient monitoring is recommended."
        )
    else:
        alert_class, alert_icon = "alert-safe", "✅"
        alert_title = "LOW RISK — No Hypotensive Event Predicted"
        alert_body = (
            f"Model <b>{model_name}</b> predicts no hypotensive event "
            f"(TARGET = 0)"
            + (f" with probability <b>{(1 - prob):.1%}</b>." if prob is not None else ".")
        )

    st.markdown(
        f"""<div class="{alert_class}">
            <div style="font-size:1.3rem;font-weight:600;">{alert_icon} {alert_title}</div>
            <div style="margin-top:8px;font-family:'IBM Plex Sans',sans-serif;font-size:0.95rem;">{alert_body}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Prediction", "HIGH RISK" if prediction == 1 else "LOW RISK")
    m2.metric("TARGET", str(prediction))
    m3.metric("Dialyzer", dia_label.split(" - ")[-1][:15])
    if prob is not None:
        m4.metric("P(TARGET=1)", f"{prob:.1%}")
        m5.metric("Confidence", f"{max(prob, 1 - prob):.1%}")
    else:
        m4.metric("P(TARGET=1)", "N/A")
        m5.metric("Model", model_name.split("(")[0].strip()[:14])

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("## 📊 H0–H5 Session Data *(observed + simulated)*")
    st.caption(
        "<span class='chip-observed'>● Observed</span>&nbsp;&nbsp;"
        "<span class='chip-simulated'>◌ Simulated</span>",
        unsafe_allow_html=True,
    )

    rows = []
    for var in CLINICAL_VARS:
        row = {"Variable": f"{var} — {VAR_LABELS[var]}"}
        for h in ALL_HOURS:
            val = full_result[var].get(h)
            is_obs = h in observed_hours[var]
            if val is not None:
                tag = "●" if is_obs else "◌"
                row[h] = f"{tag} {val:.3f}"
            else:
                row[h] = "—"
        rows.append(row)

    df_display = pd.DataFrame(rows).set_index("Variable")

    def style_cell(val):
        if isinstance(val, str) and val.startswith("●"):
            return "color: #4fc3f7; background-color: #0d253f;"
        elif isinstance(val, str) and val.startswith("◌"):
            return "color: #aed581; background-color: #162300;"
        return ""

    st.dataframe(df_display.style.map(style_cell), use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTIVE CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("## 📈 Clinical Trajectories")

    VAR_GROUPS = {
        "Haemodynamics":    ["SBP", "DBP"],
        "Pressures":        ["APR", "VPR", "TMP"],
        "Flow & Clearance": ["BFR", "KT", "TUF"],
        "Weight":           ["WDR", "WPR", "WPO", "IWG"],
        "Bath":             ["HBC"],
        "Bath Groups":      ["BAT_GROUP_Grupo 1 - ACF 3A5", "BAT_GROUP_Grupo 2 - EuCliD", "BAT_GROUP_Grupo 3 - Demais classes"],
    }

    tabs = st.tabs(list(VAR_GROUPS.keys()))
    for tab, (group_name, vars_in_group) in zip(tabs, VAR_GROUPS.items()):
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
                            x=x_obs, y=y_obs, mode="markers+lines", name="Observed",
                            marker=dict(size=9, color=OBS_COLOR, symbol="circle",
                                        line=dict(color="#e3f2fd", width=1)),
                            line=dict(color=OBS_COLOR, width=2),
                        ))
                    if x_sim:
                        fig.add_trace(go.Scatter(
                            x=x_sim, y=y_sim, mode="markers+lines", name="Simulated",
                            marker=dict(size=9, color=SIM_COLOR, symbol="diamond",
                                        line=dict(color="#f9fbe7", width=1)),
                            line=dict(color=SIM_COLOR, width=2, dash="dash"),
                        ))

                    layout = {**PLOT_LAYOUT}
                    short_label = var.replace("BAT_GROUP_", "")
                    layout["title"] = dict(text=short_label, font=dict(size=13, color="#81d4fa"))
                    layout["xaxis"] = dict(
                        tickmode="array", tickvals=list(range(6)),
                        ticktext=ALL_HOURS, gridcolor="#1e2d45",
                    )
                    layout["yaxis"] = dict(
                        title=VAR_LABELS.get(var, "").split("(")[-1].replace(")", ""),
                        gridcolor="#1e2d45",
                    )
                    layout["showlegend"] = True
                    layout["legend"] = dict(orientation="h", y=1.12, x=0)
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # PROBABILITY GAUGE
    # ─────────────────────────────────────────────────────────────────────────
    if prob is not None:
        st.markdown("## 🎯 Prediction Confidence")
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
            fig_gauge.update_layout(
                paper_bgcolor="#0a0e17", height=300,
                margin=dict(l=30, r=30, t=40, b=10),
                font=dict(color="#c8d8e8"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with text_col:
            st.markdown("### Interpretation")
            for thr, msg in [
                (0.8, "🔴 **CRITICAL** — Very high risk. Immediate monitoring required."),
                (0.6, "🟠 **HIGH** — Significant risk. Increased vigilance advised."),
                (0.4, "🟡 **MODERATE** — Borderline. Standard monitoring."),
                (0.0, "🟢 **LOW** — Low risk. Routine session expected."),
            ]:
                if prob >= thr:
                    st.markdown(msg)
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
    st.markdown("## 💾 Export Report")

    export_rows = []
    for var in CLINICAL_VARS:
        for h in ALL_HOURS:
            val = full_result[var].get(h)
            is_obs = h in observed_hours[var]
            export_rows.append({
                "Variable": var,
                "Hour": h,
                "Value": round(val, 4) if val is not None else None,
                "Source": "Observed" if is_obs else "Simulated",
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
        label="⬇  Download CSV Report",
        data=buf.getvalue(),
        file_name=f"hd_session_SEX{sex_val}_AGE{age_val}_DIA{dia_val}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    n_obs = sum(1 for r in export_rows if r["Source"] == "Observed")
    n_sim = sum(1 for r in export_rows if r["Source"] == "Simulated")
    exp_col2.caption(
        f"Report: {len(df_export)} rows — "
        f"{n_obs} observed, {n_sim} simulated — "
        f"{len(CLINICAL_VARS)} clinical variables."
    )

    with st.expander("Preview export data"):
        st.dataframe(df_export, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "HD Risk Predictor · KNN Trajectory Imputation · "
    "Built with Streamlit & Plotly · "
    "All predictions are decision-support tools only and must be reviewed by clinical staff."
)