"""
dashboard.py — Hemodialysis Session Risk Prediction Dashboard
Streamlit app that:
  1. Collects patient demographics (SEX, AGE) and clinical signals (H0–H5)
  2. Simulates missing hours via KNN trajectory with distance weighting
  3. Applies a selected pre-trained .pkl model to predict TARGET (hypotension risk)
  4. Visualises results interactively and exports a CSV report
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
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_VARS = [
    "IWG", "VOL", "KT", "BFR", "HBC",
    "APR", "VPR", "TMP", "SBP", "DBP", "HRA", "TUF",
]
ALL_HOURS = ["H0", "H1", "H2", "H3", "H4", "H5"]

VAR_LABELS = {
    "IWG": "Interdialytic Weight Gain (Kg)",
    "VOL": "Volume Changes (L)",
    "KT":  "Urea Clearance (L)",
    "BFR": "Blood Flow Rate (mL/min)",
    "HBC": "Bath Conductivity (mScm)",
    "APR": "Arterial Pressure (mmHg)",
    "VPR": "Venous Pressure (mmHg)",
    "TMP": "Transmembrane Pressure (mmHg)",
    "SBP": "Systolic Blood Pressure (mmHg)",
    "DBP": "Diastolic Blood Pressure (mmHg)",
    "HRA": "Heart Rate (bpm)",
    "TUF": "Total Ultrafiltration (mL)",
}

VAR_DEFAULTS = {
    "IWG": (0.0, -5.0, 10.0, 0.1),
    "VOL": (0.0, -20.0, 60.0, 0.1),
    "KT":  (5.0,  0.0, 80.0, 0.1),
    "BFR": (350.0, 50.0, 500.0, 1.0),
    "HBC": (14.0,  8.0,  20.0, 0.1),
    "APR": (-150.0, -300.0, 0.0, 1.0),
    "VPR": (120.0,  0.0, 300.0, 1.0),
    "TMP": (200.0,  0.0, 600.0, 1.0),
    "SBP": (130.0, 60.0, 250.0, 1.0),
    "DBP": (80.0,  40.0, 140.0, 1.0),
    "HRA": (75.0,  30.0, 200.0, 1.0),
    "TUF": (0.05, -1.0,   2.0, 0.01),
}

MODEL_OPTIONS = {
    "K-Nearest Neighbor (KNN)":    "models/modelo_knn.pkl",
    "Random Forest (RF)":          "models/modelo_RF.pkl",
    "Support Vector Machine (SVM)":"models/modelo_svm.pkl",
    "XGBoost":                     "models/modelo_xgboost.pkl",
    "Decision Tree (DT)":          "models/modelo_DT.pkl",
    "Multi-Layer Perceptron (MLP)":"models/modelo_MLP.pkl",
    "Naive Bayes (NB)":            "models/modelo_NB.pkl",
}

DATASET_PATH = "data/dataset_flat.csv"

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
# CUSTOM CSS — clinical/medical dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: #0a0e17;
    color: #c8d8e8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0d1220 !important;
    border-right: 1px solid #1e2d45;
}

/* Headers */
h1 { font-family: 'IBM Plex Mono', monospace; color: #4fc3f7 !important; letter-spacing: -1px; }
h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #81d4fa !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1a2e 0%, #112240 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px;
}
div[data-testid="metric-container"] label { color: #64b5f6 !important; font-size: 0.75rem !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e3f2fd !important; font-family: 'IBM Plex Mono', monospace; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: #e3f2fd !important;
    border: 1px solid #1976d2 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0) !important;
    border-color: #42a5f5 !important;
    box-shadow: 0 0 12px rgba(66,165,245,0.3) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
    color: #e8f5e9 !important;
    border: 1px solid #388e3c !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Expanders */
details {
    background: #0d1a2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}

/* Number inputs */
input[type="number"] {
    background: #0d1a2e !important;
    color: #c8d8e8 !important;
    border: 1px solid #1e3a5f !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background: #0d1a2e !important;
    border-color: #1e3a5f !important;
    color: #c8d8e8 !important;
}

/* Alert boxes */
.alert-danger {
    background: linear-gradient(135deg, #3e0000, #5c1a1a);
    border: 1px solid #c62828;
    border-left: 4px solid #f44336;
    border-radius: 8px;
    padding: 16px 20px;
    color: #ffcdd2;
    font-family: 'IBM Plex Mono', monospace;
}
.alert-safe {
    background: linear-gradient(135deg, #003300, #1a3d1a);
    border: 1px solid #2e7d32;
    border-left: 4px solid #4caf50;
    border-radius: 8px;
    padding: 16px 20px;
    color: #c8e6c9;
    font-family: 'IBM Plex Mono', monospace;
}

/* Tag chips */
.chip-observed {
    background: #0d3b5e; color: #4fc3f7;
    border: 1px solid #1976d2;
    padding: 2px 8px; border-radius: 12px;
    font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace;
}
.chip-simulated {
    background: #1a2700; color: #aed581;
    border: 1px solid #558b2f;
    padding: 2px 8px; border-radius: 12px;
    font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace;
}

/* Divider */
hr { border-color: #1e3a5f !important; }

/* Dataframe */
.stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 8px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0d1a2e; border-bottom: 1px solid #1e3a5f; }
.stTabs [data-baseweb="tab"] { color: #64b5f6 !important; font-family: 'IBM Plex Mono', monospace; }
.stTabs [aria-selected="true"] { color: #4fc3f7 !important; border-bottom: 2px solid #4fc3f7 !important; }

/* Slider */
.stSlider > div > div > div { background: #1976d2 !important; }

/* Radio */
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
        raise ValueError("No observed hours found. Provide at least H0 for one clinical variable.")

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


def build_flat_vector(full_result: dict, sex: int, age: int) -> pd.DataFrame:
    row = {"SEX": sex, "AGE": age}
    for var in CLINICAL_VARS:
        for hour in ALL_HOURS:
            col = f"{var}_{hour}"
            row[col] = full_result.get(var, {}).get(hour)
    return pd.DataFrame([row])


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Robust model loader with three fallback strategies.

    The error 'STACK_GLOBAL requires str' is a pickle version mismatch:
    the .pkl was serialised in an older scikit-learn / Python environment
    and is being deserialised by a newer one whose internal class paths
    changed (bytes where str is now required).

    Fix order:
      1. joblib.load          — sklearn's recommended serialiser; handles most mismatches
      2. pickle + latin1      — recovers Python-2-era pickles (bytes → str coercion)
      3. pickle (plain)       — last resort; may still raise on severe mismatches
    """
    errors = {}

    # ── Strategy 1: joblib ────────────────────────────────────────────────────
    try:
        return joblib.load(model_path)
    except Exception as e:
        errors["joblib"] = str(e)

    # ── Strategy 2: pickle with latin-1 encoding ──────────────────────────────
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except Exception as e:
        errors["pickle+latin1"] = str(e)

    # ── Strategy 3: plain pickle ──────────────────────────────────────────────
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        errors["pickle"] = str(e)

    raise RuntimeError(
        "Could not load model. All deserialization strategies failed.\n"
        "This usually means the .pkl was created with a different scikit-learn "
        "or Python version. Try re-saving the model in your current environment.\n\n"
        f"Details:\n" + "\n".join(f"  {k}: {v}" for k, v in errors.items())
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

HOUR_LABELS = ["H0", "H1", "H2", "H3", "H4", "H5"]
OBS_COLOR = "#4fc3f7"   # light blue = observed
SIM_COLOR = "#aed581"   # lime green = simulated


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 HD Risk Predictor")
    st.caption("Hemodialysis session risk assessment via trajectory KNN simulation")
    st.divider()

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown("### 👤 Patient Demographics")
    sex_label = st.radio("Sex", ["Female (0)", "Male (1)"], horizontal=True)
    sex_val = 0 if sex_label.startswith("Female") else 1
    age_val = st.number_input("Age (years)", min_value=1, max_value=120, value=65, step=1)

    st.divider()

    # ── Model selection ───────────────────────────────────────────────────────
    st.markdown("### 🤖 Prediction Model")
    model_name = st.selectbox("Select model", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[model_name]
    st.caption(f"Path: `{model_path}`")

    st.divider()

    # ── Simulation params ─────────────────────────────────────────────────────
    st.markdown("### ⚙️ Simulation Parameters")
    k_neighbors = st.slider("K-Neighbors (KNN imputation)", min_value=3, max_value=50, value=10, step=1)
    use_demo = st.toggle("Include demographics in distance", value=True)

    st.divider()

    # ── CSV upload (optional) ─────────────────────────────────────────────────
    st.markdown("### 📂 CSV Upload *(optional)*")
    uploaded_csv = st.file_uploader(
        "Upload a pre-filled session CSV",
        type=["csv"],
        help="CSV must have the same columns as dataset_flat.csv (without Target).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.markdown("# Hemodialysis Session Risk Predictor")
    st.markdown(
        "Enter the clinical signals for **H0** (mandatory) and optionally H1–H5. "
        "Missing hourly measurements are estimated via **KNN Trajectory Imputation**."
    )
with col_meta:
    st.metric("Patient Sex", "Male" if sex_val == 1 else "Female")
    st.metric("Patient Age", f"{age_val} yrs")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL DATA INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📋 Clinical Data Entry")
st.caption(
    "Enable hours with the checkbox. Leave a variable row unchecked if not measured. "
    "Any unchecked hour will be simulated."
)

# Which hours to show as editable columns
hour_enabled = {}
hour_cols = st.columns([1] + [1] * 6)
hour_cols[0].markdown("**Hour**")
for i, h in enumerate(ALL_HOURS):
    hour_enabled[h] = hour_cols[i + 1].checkbox(h, value=(h == "H0"), key=f"hour_{h}")

st.markdown("")

# Build input grid: rows = variables, cols = hours
observed = {var: {} for var in CLINICAL_VARS}

for var in CLINICAL_VARS:
    with st.expander(f"**{var}** — {VAR_LABELS[var]}", expanded=(var in ["SBP", "DBP", "HRA"])):
        inp_cols = st.columns([1] + [1] * 6)
        inp_cols[0].markdown(f"*{var}*")
        default, mn, mx, step = VAR_DEFAULTS[var]
        for i, h in enumerate(ALL_HOURS):
            if hour_enabled[h]:
                val = inp_cols[i + 1].number_input(
                    label=h,
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

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATE BUTTON
# ─────────────────────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
run_btn = run_col.button("▶  Simulate & Predict", use_container_width=True)


if run_btn:
    # ── Load dataset ──────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found at `{DATASET_PATH}`. Please check your directory.")
        st.stop()

    with st.spinner("Loading historical dataset…"):
        df_hist = load_dataset(DATASET_PATH)

    # ── Override with uploaded CSV if provided ────────────────────────────────
    if uploaded_csv is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_csv, sep=";")
            # Extract first row as observed values
            for var in CLINICAL_VARS:
                for h in ALL_HOURS:
                    col = f"{var}_{h}"
                    if col in df_uploaded.columns and not df_uploaded[col].isna().all():
                        observed[var][h] = float(df_uploaded[col].iloc[0])
                        hour_enabled[h] = True
            st.success("CSV loaded — values populated from first row.")
        except Exception as e:
            st.warning(f"Could not parse uploaded CSV: {e}")

    # ── Validate at least one H0 value ───────────────────────────────────────
    any_h0 = any(observed[v].get("H0") is not None for v in CLINICAL_VARS)
    if not any_h0:
        st.error("Please enter at least one H0 clinical measurement before simulating.")
        st.stop()

    # ── Run KNN trajectory simulation ─────────────────────────────────────────
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

    # Determine which hours were observed vs simulated per variable
    observed_hours = {
        var: set(h for h, v in observed[var].items() if v is not None)
        for var in CLINICAL_VARS
    }

    # ── Build flat vector for model ───────────────────────────────────────────
    X_model = build_flat_vector(full_result, sex=sex_val, age=age_val)

    # ── Load and apply prediction model ──────────────────────────────────────
    if not os.path.exists(model_path):
        st.error(f"Model not found at `{model_path}`.")
        st.stop()

    with st.spinner("Applying prediction model…"):
        try:
            model = load_model(model_path)
            prediction = int(model.predict(X_model)[0])
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X_model)[0][1])
            else:
                prob = None
        except Exception as e:
            err_msg = str(e)
            st.error("**Model prediction error**")
            if "STACK_GLOBAL requires str" in err_msg or "requires str" in err_msg:
                st.warning(
                    "**Root cause:** The `.pkl` file was saved with an older version of "
                    "scikit-learn or Python and cannot be loaded by the current environment. \n\n"
                    "**Fix:** Open your training notebook and re-save the model with:\n"
                    "```python\n"
                    "import joblib\n"
                    "joblib.dump(model, 'models/modelo_knn.pkl')\n"
                    "```\n"
                    "Then replace the `.pkl` file and reload the dashboard."
                )
            else:
                st.code(err_msg)
            st.stop()

    st.success("Simulation and prediction complete!")
    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS: SUMMARY & ALERT
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("## 🔬 Results")

    # Alert card
    if prediction == 1:
        alert_class = "alert-danger"
        alert_icon = "⚠️"
        alert_title = "HIGH RISK — Hypotensive Event Predicted"
        alert_body = (
            f"The model <b>{model_name}</b> predicts an intradialytic hypotensive event "
            f"(TARGET = 1){(' with probability <b>' + f'{prob:.1%}</b>') if prob is not None else ''}. "
            "Close patient monitoring is recommended."
        )
    else:
        alert_class = "alert-safe"
        alert_icon = "✅"
        alert_title = "LOW RISK — No Hypotensive Event Predicted"
        alert_body = (
            f"The model <b>{model_name}</b> predicts no hypotensive event "
            f"(TARGET = 0){(' with probability <b>' + f'{(1-prob):.1%}</b>') if prob is not None else ''}."
        )

    st.markdown(
        f"""
        <div class="{alert_class}">
            <div style="font-size:1.3rem; font-weight:600;">{alert_icon} {alert_title}</div>
            <div style="margin-top:8px; font-family:'IBM Plex Sans',sans-serif; font-size:0.95rem;">{alert_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prediction", "HIGH RISK" if prediction == 1 else "LOW RISK")
    m2.metric("TARGET", str(prediction))
    if prob is not None:
        m3.metric("Prob (TARGET=1)", f"{prob:.1%}")
        m4.metric("Confidence", f"{max(prob, 1-prob):.1%}")
    else:
        m3.metric("Prob (TARGET=1)", "N/A")
        m4.metric("Model", model_name.split("(")[0].strip())

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("## 📊 H0–H5 Session Data *(observed + simulated)*")
    st.caption(
        "<span class='chip-observed'>● Observed</span>&nbsp;&nbsp;"
        "<span class='chip-simulated'>● Simulated</span>",
        unsafe_allow_html=True,
    )

    # Build display DataFrame
    rows = []
    for var in CLINICAL_VARS:
        row = {"Variable": f"{var} — {VAR_LABELS[var]}"}
        for h in ALL_HOURS:
            val = full_result[var].get(h)
            is_obs = h in observed_hours[var]
            if val is not None:
                tag = "●" if is_obs else "◌"
                row[h] = f"{tag} {val:.2f}"
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

    styled = df_display.style.map(style_cell)
    st.dataframe(styled, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTIVE CHARTS — grouped by clinical category
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("## 📈 Clinical Trajectories")

    VAR_GROUPS = {
        "Haemodynamics": ["SBP", "DBP", "HRA"],
        "Pressures": ["APR", "VPR", "TMP"],
        "Flow & Clearance": ["BFR", "KT", "TUF"],
        "Volume & Weight": ["IWG", "VOL"],
        "Bath Parameters": ["HBC"],
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

                    # Full line (dashed)
                    x_all = [i for i, v in enumerate(vals) if v is not None]
                    y_all = [v for v in vals if v is not None]
                    fig.add_trace(go.Scatter(
                        x=x_all, y=y_all,
                        mode="lines",
                        line=dict(color="#1e3a5f", width=1.5, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ))

                    # Observed points
                    if x_obs:
                        fig.add_trace(go.Scatter(
                            x=x_obs, y=y_obs,
                            mode="markers+lines",
                            name="Observed",
                            marker=dict(size=9, color=OBS_COLOR, symbol="circle",
                                        line=dict(color="#e3f2fd", width=1)),
                            line=dict(color=OBS_COLOR, width=2),
                        ))

                    # Simulated points
                    if x_sim:
                        fig.add_trace(go.Scatter(
                            x=x_sim, y=y_sim,
                            mode="markers+lines",
                            name="Simulated",
                            marker=dict(size=9, color=SIM_COLOR, symbol="diamond",
                                        line=dict(color="#f9fbe7", width=1)),
                            line=dict(color=SIM_COLOR, width=2, dash="dash"),
                        ))

                    chart_layout = {**PLOT_LAYOUT}
                    chart_layout["title"] = dict(text=f"{var}", font=dict(size=13, color="#81d4fa"))
                    chart_layout["xaxis"] = dict(
                        tickmode="array",
                        tickvals=list(range(6)),
                        ticktext=ALL_HOURS,
                        gridcolor="#1e2d45",
                    )
                    chart_layout["yaxis"] = dict(
                        title=VAR_LABELS[var].split("(")[-1].replace(")", ""),
                        gridcolor="#1e2d45",
                    )
                    chart_layout["showlegend"] = True
                    chart_layout["legend"] = dict(orientation="h", y=1.12, x=0)
                    fig.update_layout(**chart_layout)
                    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # PROBABILITY GAUGE (if available)
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
                    "borderwidth": 1,
                    "bordercolor": "#1e3a5f",
                    "steps": [
                        {"range": [0, 30], "color": "#1b3a2d"},
                        {"range": [30, 60], "color": "#2a3a00"},
                        {"range": [60, 80], "color": "#3e2000"},
                        {"range": [80, 100], "color": "#3e0000"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffeb3b", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
                title={"text": "P(Hypotension)", "font": {"color": "#81d4fa", "family": "IBM Plex Mono"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0a0e17",
                height=300,
                margin=dict(l=30, r=30, t=40, b=10),
                font=dict(color="#c8d8e8"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with text_col:
            st.markdown("### Interpretation")
            thresholds = [
                (0.8, "🔴 **CRITICAL** — Very high risk. Immediate monitoring required."),
                (0.6, "🟠 **HIGH** — Significant risk. Increased vigilance advised."),
                (0.4, "🟡 **MODERATE** — Borderline. Standard monitoring."),
                (0.0, "🟢 **LOW** — Low risk. Routine session expected."),
            ]
            for thr, msg in thresholds:
                if prob >= thr:
                    st.markdown(msg)
                    break

            st.markdown(f"""
| Metric | Value |
|--------|-------|
| P(TARGET=1) | `{prob:.4f}` |
| P(TARGET=0) | `{1-prob:.4f}` |
| Predicted class | `{prediction}` |
| Model | `{model_name.split("(")[0].strip()}` |
""")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT CSV
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("## 💾 Export Report")

    # Build export DataFrame
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
                "Prediction_TARGET": prediction,
                "Probability_TARGET1": round(prob, 4) if prob is not None else None,
                "Model": model_name,
                "K_neighbors": k_neighbors,
            })

    df_export = pd.DataFrame(export_rows)

    csv_buffer = io.StringIO()
    df_export.to_csv(csv_buffer, index=False, sep=";")
    csv_str = csv_buffer.getvalue()

    exp_col1, exp_col2 = st.columns([1, 3])
    exp_col1.download_button(
        label="⬇  Download CSV Report",
        data=csv_str,
        file_name=f"hd_session_SEX{sex_val}_AGE{age_val}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    exp_col2.caption(
        f"Report contains {len(df_export)} rows "
        f"({len([r for r in export_rows if r['Source']=='Observed'])} observed, "
        f"{len([r for r in export_rows if r['Source']=='Simulated'])} simulated) "
        f"across {len(CLINICAL_VARS)} clinical variables."
    )

    # Preview
    with st.expander("Preview export data"):
        st.dataframe(df_export, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "HD Risk Predictor · KNN Trajectory Imputation · "
    "Built with Streamlit & Plotly · "
    "All predictions are decision-support only and must be reviewed by clinical staff."
)