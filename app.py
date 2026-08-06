"""
app.py

Streamlit Dashboard for Mobile Gaming Player Churn & Retention Analytics.
Features Sidebar Navigation, Multi-Tab Analytics, A/B Testing, Churn Prediction, and Model Insights.
"""

import os
import time
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cookie Cats — Churn & Retention Analytics",
    page_icon="🕹️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1.1  Global CSS — single source of truth for all styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root & Typography ───────────────────────────── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
}
.material-icons, .material-symbols-rounded, i[class*="icon"] {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}

/* ── Hide Streamlit toolbar & header chrome ──────── */
#MainMenu { visibility: hidden; }
.stDeployButton { display: none !important; }
footer { visibility: hidden; }

/* ── Page hero header ────────────────────────────── */
/* ── Hero fade-slide-up animation ───────────────── */
@keyframes heroFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Hero header box ─────────────────────────────── */
.hero-wrapper {
    text-align: center;
    padding: 2.4rem 1.4rem 2.2rem;
    animation: heroFadeIn 0.8s ease-out both;
    width: 100%;
    background: linear-gradient(
        135deg,
        rgba(0,173,181,0.07) 0%,
        rgba(22,33,62,0.96)  40%,
        rgba(26,37,64,0.96) 100%
    );
    border: 1px solid rgba(0,173,181,0.28);
    border-radius: 14px;
    box-shadow: 0 6px 28px rgba(0,0,0,0.45),
                inset 0 1px 0 rgba(255,255,255,0.05);
    margin: 0.5rem 0 1.6rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}
.hero-wrapper:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 36px rgba(0,173,181,0.25),
                inset 0 1px 0 rgba(255,255,255,0.05);
    border-color: rgba(0,173,181,0.5);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00adb5 0%, #00fff5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin: 0 0 0.7rem;
    text-align: center;
}
.hero-sub {
    font-size: 1.1rem;
    font-weight: 400;
    color: #7b8fa1;
    letter-spacing: 0.3px;
    margin: 0;
    line-height: 1.65;
    text-align: center;
}
@media (max-width: 768px) {
    .hero-title { font-size: 2rem; letter-spacing: 0; }
    .hero-sub   { font-size: 0.92rem; }
    .hero-wrapper { padding: 1.8rem 1rem 1.6rem; }
}

/* Centre-align the Streamlit main block */
div.block-container {
    max-width: 1280px;
    margin: 0 auto;
    padding-top: 4.5rem !important;
}

/* ── Section header badges ───────────────────────── */
.section-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(0,173,181,0.12) 0%, rgba(0,173,181,0.04) 100%);
    border: 1px solid rgba(0,173,181,0.3);
    border-radius: 999px;
    padding: 5px 16px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #00adb5;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.section-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 0.3rem;
    line-height: 1.3;
}
.section-sub {
    font-size: 0.92rem;
    color: #7b8fa1;
    margin: 0 0 1.2rem;
    line-height: 1.5;
}

/* ── KPI Metric Cards ────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #16213e 0%, #1a2540 100%);
    border-radius: 14px;
    padding: 20px 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    border: 1px solid rgba(0,173,181,0.12);
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 28px rgba(0,173,181,0.18);
    border-color: rgba(0,173,181,0.45);
}
div[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    color: #7b8fa1 !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #00adb5 !important;
}

/* ── Info/methodology cards ──────────────────────── */
.info-card {
    background: linear-gradient(135deg, #16213e 0%, #1a2540 100%);
    border: 1px solid rgba(0,173,181,0.14);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    height: 100%;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.info-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.4);
    border-color: rgba(0,173,181,0.4);
}
.info-card h4 {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #00adb5;
    margin: 0 0 0.5rem;
}
.info-card p, .info-card li {
    font-size: 0.9rem;
    line-height: 1.65;
    color: #cbd5e1;
    margin: 0;
}
.info-card ul {
    padding-left: 1.1rem;
    margin: 0;
}
.info-card li { margin-bottom: 0.35rem; }

/* ── Feature importance rows ─────────────────────── */
.fi-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.88rem;
    border-radius: 8px;
    transition: background 0.2s ease, padding-left 0.2s ease;
}
.fi-row:hover {
    background: rgba(0,173,181,0.08);
    padding-left: 0.9rem;
}
.fi-row:last-child { border-bottom: none; }
.fi-label { color: #cbd5e1; font-weight: 500; }
.fi-bar-wrap {
    flex: 1;
    margin: 0 12px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}
.fi-bar {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #00adb5, #00fff5);
}
.fi-val { color: #00adb5; font-weight: 700; min-width: 44px; text-align: right; }

/* ── Sidebar ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid rgba(0,173,181,0.12);
}
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.4rem 0 0.8rem;
}
.sidebar-logo-icon {
    font-size: 2rem;
    line-height: 1;
}
.sidebar-logo-text h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1.2;
}
.sidebar-logo-text span {
    font-size: 0.72rem;
    color: #7b8fa1;
    letter-spacing: 0.3px;
}
.sidebar-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.38rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.84rem;
}
.sidebar-stat-row:last-child { border-bottom: none; }
.sidebar-stat-label { color: #7b8fa1; }
.sidebar-stat-val { color: #e2e8f0; font-weight: 600; }

/* ── Tabs ────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 10px 22px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}

/* ── Divider with label ──────────────────────────── */
.labeled-divider {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 1.8rem 0 1.4rem;
}
.labeled-divider hr {
    flex: 1;
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 0;
}
.labeled-divider span {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #4a5568;
    white-space: nowrap;
}

/* ── Churn gauge caption ─────────────────────────── */
.model-caption {
    background: rgba(0,173,181,0.06);
    border-left: 3px solid #00adb5;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #94a3b8;
    margin-top: 0.6rem;
}

/* ── Loader spinner ──────────────────────────────── */
.loader-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    padding: 3rem 0;
}
.loader-ring {
    width: 44px; height: 44px;
    border: 3px solid rgba(0,173,181,0.15);
    border-top-color: #00adb5;
    border-radius: 50%;
    animation: spin 0.85s linear infinite;
}
.loader-text {
    font-size: 0.82rem;
    font-weight: 500;
    color: #4a5568;
    letter-spacing: 0.5px;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ── Prediction history table ────────────────────── */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Cached Data & Model Loaders
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset(data_path="data/cookie_cats.csv"):
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path)
    if df.empty:
        return None
    df = df[df['sum_gamerounds'] < 5000].reset_index(drop=True)
    return df

@st.cache_resource
def load_trained_model(model_path="models/churn_model.pkl"):
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None

@st.cache_data
def load_metrics_json(metrics_path="models/metrics.json"):
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception:
        return None

# Shared Plotly layout defaults
def base_layout(**kwargs):
    defaults = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#94a3b8', size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
    )
    defaults.update(kwargs)
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── Hero Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrapper">
        <h1 class="hero-title">🎮 Cookie Cats — Player Churn &amp; Retention Analytics</h1>
        <p class="hero-sub">A/B Testing &nbsp;·&nbsp; Predictive Churn Modeling &nbsp;·&nbsp; Feature Explainability &nbsp;·&nbsp; 90,189 Real Players</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Loading spinner ───────────────────────────────────────────────────────
    _loader = st.empty()
    _loader.markdown("""
    <div class="loader-wrapper">
        <div class="loader-ring"></div>
        <p class="loader-text">Loading dataset & model…</p>
    </div>
    """, unsafe_allow_html=True)

    df     = load_dataset()
    model  = load_trained_model()
    metrics_info = load_metrics_json()

    time.sleep(0.3)
    _loader.empty()

    # Guard: no data
    if df is None or df.empty:
        st.error("⚠️ Dataset not found. Place `cookie_cats.csv` inside the `/data` folder and reload.")
        st.stop()

    # Warn: no model
    if model is None:
        st.warning("⚠️ Model file missing. Run `python churn_model.py` to train and save the model.")

    # ── Pre-compute shared values ─────────────────────────────────────────────
    total_players = len(df)
    d1_ret  = df['retention_1'].mean() * 100
    d7_ret  = df['retention_7'].mean() * 100
    avg_rnd = df['sum_gamerounds'].mean()
    gate30  = (df['version'] == 'gate_30').sum()
    gate40  = (df['version'] == 'gate_40').sum()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🕹️</div>
            <div class="sidebar-logo-text">
                <h3>Cookie Cats<br>Analytics</h3>
                <span>Player Engagement Dashboard</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("<p style='font-size:0.7rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#4a5568;margin-bottom:0.6rem;'>Dataset Stats</p>", unsafe_allow_html=True)
        rows = [
            ("Total Players",  f"{total_players:,}"),
            ("Control (gate_30)", f"{gate30:,}"),
            ("Test (gate_40)",  f"{gate40:,}"),
            ("Columns",        str(len(df.columns))),
            ("Outliers Removed", "1 (≥5,000 rounds)"),
        ]
        for label, val in rows:
            st.markdown(f"""
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">{label}</span>
                <span class="sidebar-stat-val">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("<p style='font-size:0.7rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#4a5568;margin-bottom:0.6rem;'>Links</p>", unsafe_allow_html=True)
        st.markdown("🐙 [GitHub Repository](https://github.com/Shlok2814/Mobile-Gaming-Churn-Analytics)")

        st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_overview, tab_abtest, tab_predictor, tab_insights, tab_about = st.tabs([
        "📊  Overview",
        "🧪  A/B Test Results",
        "🤖  Churn Predictor",
        "📈  Model Insights",
        "ℹ️  Project Info",
    ])


    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown("<br>", unsafe_allow_html=True)

        # KPI row
        k1, k2, k3, k4 = st.columns(4, gap="medium")
        with k1: st.metric("👥 Total Players",        f"{total_players:,}")
        with k2: st.metric("📅 D1 Retention",          f"{d1_ret:.2f}%")
        with k3: st.metric("📅 D7 Retention",          f"{d7_ret:.2f}%")
        with k4: st.metric("🎮 Avg Rounds / Player",   f"{avg_rnd:.1f}")

        st.markdown("<div class='labeled-divider'><hr><span>Project Context</span><hr></div>", unsafe_allow_html=True)

        ctx_col, prev_col = st.columns([1, 1], gap="large")

        with ctx_col:
            st.markdown("""
            <div class="info-card">
                <h4>🎯 What is this project?</h4>
                <p>
                    <strong style="color:#e2e8f0;">Cookie Cats</strong> is a wildly popular mobile puzzle game.
                    This dashboard analyses a real Kaggle A/B test (90,189 players) to answer two questions:
                </p>
                <ul style="margin-top:0.7rem;">
                    <li><strong style="color:#00adb5;">A/B Impact:</strong> Does moving the first gate from Level 30 → Level 40 harm or help 7-day retention?</li>
                    <li><strong style="color:#00adb5;">Churn Prediction:</strong> Can we flag churners early from Week-1 behaviour and trigger timely retention interventions?</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with prev_col:
            st.markdown("<p style='font-size:0.78rem;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;color:#4a5568;margin-bottom:0.5rem;'>Dataset Preview</p>", unsafe_allow_html=True)
            st.dataframe(df.head(8), use_container_width=True, hide_index=True)


    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — A/B TEST RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_abtest:
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div>
            <span class="section-badge">📊 Experiment Analysis</span>
            <p class="section-title">Level Gate A/B Test Impact</p>
            <p class="section-sub">Comparing D1 & D7 retention rates — <strong>gate_30</strong> (Control) vs <strong>gate_40</strong> (Treatment)</p>
        </div>
        """, unsafe_allow_html=True)

        ab_summary = df.groupby('version').agg(
            D1_Retention=('retention_1', lambda x: x.mean() * 100),
            D7_Retention=('retention_7', lambda x: x.mean() * 100),
            Player_Count=('userid', 'count')
        ).reset_index()

        plot_df = ab_summary.melt(
            id_vars=['version'],
            value_vars=['D1_Retention', 'D7_Retention'],
            var_name='Metric', value_name='Retention %'
        )
        plot_df['Metric'] = plot_df['Metric'].map({
            'D1_Retention': '1-Day Retention (D1)',
            'D7_Retention': '7-Day Retention (D7)'
        })

        bar_col, ins_col = st.columns([3, 2], gap="large")

        with bar_col:
            fig_bar = px.bar(
                plot_df, x='version', y='Retention %', color='Metric',
                barmode='group', text_auto='.2f',
                title="Retention Rate by Gate Version (%)",
                labels={'version': 'Gate Version'},
                color_discrete_sequence=['#00adb5', '#f59e0b']
            )
            fig_bar.update_traces(textfont_size=11, textposition='outside', marker_line_width=0)
            fig_bar.update_layout(**base_layout(
                yaxis_range=[0, 65], height=380,
                title_font=dict(size=15, color='#00adb5'),
                legend_title_text="Metric"
            ))
            st.plotly_chart(fig_bar, use_container_width=True)

        with ins_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card">
                <h4>💡 Key Finding</h4>
                <p>
                    The Chi-Square test returned <strong style="color:#ff2e63;">p = 0.0016</strong>, well below the α = 0.05 threshold,
                    confirming the gate position has a statistically significant effect on 7-day retention.
                </p>
                <br>
                <h4 style="margin-top:0.8rem;">📉 What happened?</h4>
                <ul>
                    <li>D7 Retention at <strong style="color:#00adb5;">gate_30: 19.02%</strong></li>
                    <li>D7 Retention at <strong style="color:#f59e0b;">gate_40: 18.20%</strong></li>
                    <li>Delta: <strong style="color:#ff2e63;">−0.82 pp</strong></li>
                </ul>
                <br>
                <h4 style="margin-top:0.8rem;">🏆 Recommendation</h4>
                <p>
                    Keep the gate at Level 30. The earlier friction creates a natural cooldown that reduces burnout
                    and improves long-term retention economics.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='labeled-divider'><hr><span>Engagement Distribution</span><hr></div>", unsafe_allow_html=True)

        df_dist = df[df['sum_gamerounds'] < 100]
        hist_col, box_col = st.columns(2, gap="large")

        with hist_col:
            fig_hist = px.histogram(
                df_dist, x="sum_gamerounds", color="version",
                barmode="overlay", nbins=50, opacity=0.7,
                title="Game Rounds Distribution (< 100)",
                labels={'sum_gamerounds': 'Rounds Played', 'version': 'Gate Group'},
                color_discrete_sequence=['#00adb5', '#ff2e63']
            )
            fig_hist.update_layout(**base_layout(height=340, title_font=dict(size=14, color='#00adb5')))
            st.plotly_chart(fig_hist, use_container_width=True)

        with box_col:
            fig_box = px.box(
                df_dist, x="version", y="sum_gamerounds", color="version",
                points=False,
                title="Rounds Played — Boxplot by Group",
                labels={'sum_gamerounds': 'Rounds Played', 'version': 'Gate Group'},
                color_discrete_sequence=['#00adb5', '#ff2e63']
            )
            fig_box.update_layout(**base_layout(height=340, title_font=dict(size=14, color='#00adb5'), showlegend=False))
            st.plotly_chart(fig_box, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — CHURN PREDICTOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab_predictor:
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div>
            <span class="section-badge">🤖 ML Inference</span>
            <p class="section-title">Real-Time Player Churn Predictor</p>
            <p class="section-sub">Configure a player profile below to get an instant 7-day churn probability score.</p>
        </div>
        """, unsafe_allow_html=True)

        if model is None:
            st.error("Model unavailable. Run `python churn_model.py` to train the model first.")
        else:
            # Input controls
            inp1, inp2, inp3 = st.columns(3, gap="large")

            with inp1:
                input_rounds = st.number_input(
                    "🎮 Game Rounds Played (Week 1)",
                    min_value=0, max_value=3000, value=15, step=1,
                    help="Total rounds completed in the first 7 days."
                )
            with inp2:
                input_version = st.selectbox(
                    "🚪 Gate Version",
                    options=["gate_30", "gate_40 / gate_45"],
                    help="Which gate group the player belongs to."
                )
            with inp3:
                input_ret1 = st.selectbox(
                    "📅 Returned on Day 1?",
                    options=["Yes", "No"],
                    help="Did the player open the app again the next day?"
                )

            gate_flag = 1 if "gate_40" in input_version or "gate_45" in input_version else 0
            ret1_flag = 1 if input_ret1 == "Yes" else 0

            feat_df = pd.DataFrame([{
                'sum_gamerounds': input_rounds,
                'gate_45_flag': gate_flag,
                'retention_1_flag': ret1_flag
            }])

            churn_prob = model.predict_proba(feat_df)[0][1] * 100

            # What-if counterfactual
            whatif_df = pd.DataFrame([{
                'sum_gamerounds': input_rounds,
                'gate_45_flag': gate_flag,
                'retention_1_flag': 1 - ret1_flag
            }])
            whatif_prob = model.predict_proba(whatif_df)[0][1] * 100
            delta = churn_prob - whatif_prob

            st.markdown("<div class='labeled-divider'><hr><span>Prediction Output</span><hr></div>", unsafe_allow_html=True)

            gauge_col, action_col = st.columns([1, 1], gap="large")

            with gauge_col:
                gauge_color = "#ff2e63" if churn_prob > 50 else "#00adb5"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "7-Day Churn Probability", 'font': {'size': 14, 'color': '#94a3b8', 'family': 'Inter'}},
                    number={'suffix': "%", 'font': {'size': 40, 'color': gauge_color, 'family': 'Inter'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4a5568", 'tickfont': {'size': 10}},
                        'bar': {'color': gauge_color, 'thickness': 0.22},
                        'bgcolor': "#1a2540",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0,  35], 'color': 'rgba(0,173,181,0.10)'},
                            {'range': [35, 65], 'color': 'rgba(245,158,11,0.10)'},
                            {'range': [65,100], 'color': 'rgba(255,46,99,0.10)'},
                        ],
                        'threshold': {
                            'line': {'color': '#ff2e63', 'width': 2},
                            'thickness': 0.65, 'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=260, paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=50, b=20),
                    font=dict(family='Inter')
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("""
                <div class="model-caption">
                    <strong>How the model decides:</strong> The Random Forest aggregates 100 decision trees.
                    Each tree votes based on <em>game rounds played</em>, <em>Day-1 return</em>,
                    and <em>gate version</em>. Players exceeding 50% churn probability are flagged High-Risk.
                </div>
                """, unsafe_allow_html=True)

            with action_col:
                risk_label = "High Churn Risk" if churn_prob > 50 else "Low Churn Risk"
                risk_icon  = "🚨" if churn_prob > 50 else "✅"
                border_col = "#ff2e63" if churn_prob > 50 else "#00adb5"
                action_text = (
                    "Trigger an in-game push notification immediately. "
                    "Offer a timed reward — e.g. 1-hour unlimited lives — to bring the player back before the 24-hour window closes."
                    if churn_prob > 50 else
                    "Player is well-engaged. Maintain standard experience and consider introducing monetisation touchpoints "
                    "or social features to deepen long-term stickiness."
                )

                st.markdown(f"""
                <div class="info-card" style="border-left:3px solid {border_col}; margin-bottom:1rem;">
                    <h4>{risk_icon} {risk_label}</h4>
                    <p style="font-size:2rem;font-weight:800;color:{border_col};margin:0.3rem 0 0.6rem;">{churn_prob:.1f}%</p>
                    <p style="font-size:0.88rem;color:#94a3b8;line-height:1.6;">{action_text}</p>
                </div>
                """, unsafe_allow_html=True)

                # What-if
                flip_label = "not returned" if input_ret1 == "Yes" else "returned"
                direction  = "⬆️" if delta > 0 else "⬇️"
                diff_color = "#ff2e63" if delta > 0 else "#00adb5"
                st.markdown(f"""
                <div class="info-card">
                    <h4>🔄 What-If: Day-1 Return Flipped</h4>
                    <p>
                        If this player had <strong style="color:#e2e8f0;">{flip_label}</strong> on Day 1,
                        their churn probability would be
                        <strong style="color:{diff_color};">{whatif_prob:.1f}%</strong>
                        &nbsp;{direction}&nbsp;
                        <span style="color:{diff_color};">{abs(delta):.1f} pp {'higher' if delta > 0 else 'lower'}</span>.
                    </p>
                    <p style="margin-top:0.6rem;color:#64748b;font-size:0.82rem;">
                        Day-1 return is the strongest single behavioural signal the model has access to in Week 1.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Prediction history
            st.markdown("<div class='labeled-divider'><hr><span>Prediction Log</span><hr></div>", unsafe_allow_html=True)

            if 'prediction_history' not in st.session_state:
                st.session_state['prediction_history'] = []

            log_col, dl_col, _ = st.columns([1, 1, 2])
            with log_col:
                if st.button("➕ Log This Prediction", use_container_width=True):
                    st.session_state['prediction_history'].append({
                        'Timestamp':     pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Rounds':        input_rounds,
                        'Gate Version':  input_version,
                        'Day-1 Return':  input_ret1,
                        'Churn Prob (%)': round(churn_prob, 2),
                        'Risk':          'High' if churn_prob > 50 else 'Low'
                    })
                    st.success("Logged ✓")

            if st.session_state['prediction_history']:
                hist_df  = pd.DataFrame(st.session_state['prediction_history'])
                csv_data = hist_df.to_csv(index=False).encode('utf-8')
                with dl_col:
                    st.download_button(
                        "📥 Download CSV", data=csv_data,
                        file_name="churn_prediction_log.csv",
                        mime="text/csv", use_container_width=True
                    )
                st.dataframe(hist_df, use_container_width=True, hide_index=True)


    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — MODEL INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_insights:
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div>
            <span class="section-badge">📈 Model Diagnostics</span>
            <p class="section-title">Random Forest Performance & Explainability</p>
            <p class="section-sub">Evaluated on a held-out 20% stratified test set (18,038 players).</p>
        </div>
        """, unsafe_allow_html=True)

        if model is None:
            st.error("Model file missing. Run `python churn_model.py` to train the model.")
        else:
            # Reconstruct test predictions
            df_eval = df.copy()
            df_eval['churn_d7']        = (~df_eval['retention_7']).astype(int)
            df_eval['gate_45_flag']    = (df_eval['version'] != 'gate_30').astype(int)
            df_eval['retention_1_flag'] = df_eval['retention_1'].astype(int)

            X = df_eval[['sum_gamerounds', 'gate_45_flag', 'retention_1_flag']]
            y = df_eval['churn_d7']
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
            y_pred       = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            fi_col, cm_col = st.columns(2, gap="large")

            with fi_col:
                imps = model.feature_importances_
                fi_df = pd.DataFrame({
                    'Feature':    ['Sum Game Rounds', 'Gate 40/45 Flag', 'Retained Day 1'],
                    'Importance': imps
                }).sort_values('Importance', ascending=True)

                fig_fi = px.bar(
                    fi_df, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance", text_auto='.3f',
                    color='Importance',
                    color_continuous_scale=[[0,'#0f3460'],[0.4,'#00adb5'],[1,'#00fff5']]
                )
                fig_fi.update_traces(textfont_size=11)
                fig_fi.update_layout(**base_layout(
                    height=300,
                    title_font=dict(size=14, color='#00adb5'),
                    coloraxis_showscale=False
                ))
                st.plotly_chart(fig_fi, use_container_width=True)

            with cm_col:
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=['Actual: Retained', 'Actual: Churned'],
                    columns=['Pred: Retained', 'Pred: Churned']
                )
                fig_cm = px.imshow(
                    cm_df, text_auto=True, aspect='auto',
                    title="Confusion Matrix (Test Set)",
                    color_continuous_scale=[[0,'#0f3460'],[0.5,'#16213e'],[1,'#00adb5']]
                )
                fig_cm.update_traces(textfont_size=14)
                fig_cm.update_layout(**base_layout(
                    height=300,
                    title_font=dict(size=14, color='#00adb5')
                ))
                st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("<div class='labeled-divider'><hr><span>ROC Curve</span><hr></div>", unsafe_allow_html=True)

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score   = roc_auc_score(y_test, y_pred_proba)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', fill='tozeroy',
                name=f'Random Forest  (AUC = {auc_score:.4f})',
                line=dict(color='#00adb5', width=2.5),
                fillcolor='rgba(0,173,181,0.08)'
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline',
                line=dict(color='#ff2e63', dash='dash', width=1.5)
            ))
            fig_roc.update_layout(**base_layout(
                title=f"ROC Curve — AUC = {auc_score:.4f}",
                title_font=dict(size=15, color='#00adb5'),
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=380
            ))
            st.plotly_chart(fig_roc, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — PROJECT INFO
    # ══════════════════════════════════════════════════════════════════════════
    with tab_about:
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div>
            <span class="section-badge">ℹ️ Project Reference</span>
            <p class="section-title">Architecture, Methodology & Model Metrics</p>
            <p class="section-sub">A complete reference sheet covering the analytical approach, model performance, and real dataset used.</p>
        </div>
        """, unsafe_allow_html=True)

        # Model KPI row (only if metrics.json exists)
        if metrics_info:
            st.markdown("<div class='labeled-divider'><hr><span>Model Performance</span><hr></div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4, gap="medium")
            with m1: st.metric("🎯 Test Accuracy",    f"{metrics_info['accuracy']*100:.2f}%")
            with m2: st.metric("📈 ROC-AUC",          f"{metrics_info['roc_auc']:.4f}")
            with m3: st.metric("🏋️ Training Samples", f"{metrics_info['train_samples']:,}")
            with m4: st.metric("🧪 Test Samples",     f"{metrics_info['test_samples']:,}")

        st.markdown("<div class='labeled-divider'><hr><span>Deep Dive</span><hr></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("""
            <div class="info-card">
                <h4>🎯 Business Problem & Retention Economics</h4>
                <p>
                    In free-to-play mobile games, 7-day retention is one of the clearest proxies for Lifetime Value (LTV).
                    Our Chi-Square test confirmed a statistically significant drop in D7 retention when the gate moved
                    from Level 30 (19.02%) to Level 40 (18.20%) — <strong style="color:#ff2e63;">p = 0.0016 &lt; 0.05</strong>.
                </p>
                <p style="margin-top:0.8rem;">
                    Counterintuitively, the earlier gate <em>helps</em> retention by enforcing a cooldown before
                    player momentum peaks — preventing burnout. At scale, an 0.82 pp shift across millions of installs
                    translates to thousands of additional engaged users per cohort.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="info-card">
                <h4>🔬 Methodology & Feature Engineering</h4>
                <ul>
                    <li>
                        <strong style="color:#00adb5;">A/B Testing</strong> — Chi-Square Test of Independence on a 2×2
                        contingency table (version × retention_7).
                    </li>
                    <li style="margin-top:0.5rem;">
                        <strong style="color:#00adb5;">ML Pipeline</strong> — RandomForestClassifier
                        (100 trees, max_depth=6, random_state=42) trained on an 80/20 stratified split.
                    </li>
                    <li style="margin-top:0.5rem;">
                        <strong style="color:#00adb5;">Features</strong> — <code>sum_gamerounds</code>,
                        <code>retention_1_flag</code>, <code>gate_45_flag</code>.
                    </li>
                    <li style="margin-top:0.5rem;">
                        <strong style="color:#00adb5;">Outlier Removal</strong> — Filtered 1 bot row with
                        sum_gamerounds ≥ 5,000 before training.
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        # ── Feature Importances — full width below ────────────────────────────
        st.markdown("<div class='labeled-divider'><hr><span>Feature Importances</span><hr></div>", unsafe_allow_html=True)

        if metrics_info:
            fi_html = ""
            for fname, fimp in metrics_info.get("feature_importances", {}).items():
                pct = fimp * 100
                fi_html += f"""
                    <div class="fi-row">
                        <span class="fi-label">{fname}</span>
                        <div class="fi-bar-wrap"><div class="fi-bar" style="width:{pct:.1f}%"></div></div>
                        <span class="fi-val">{pct:.1f}%</span>
                    </div>"""
            st.markdown(f"""
            <div class="info-card" style="max-width:720px; margin:0 auto;">
                <h4>📊 Feature Importances</h4>
                {fi_html}
                <div style="margin-top:1.2rem;padding-top:0.9rem;border-top:1px solid rgba(255,255,255,0.06);">
                    <h4>⚡ Dataset Verification</h4>
                    <p>
                        Official <strong style="color:#e2e8f0;">Kaggle Cookie Cats</strong> A/B Testing dataset —
                        <strong style="color:#00adb5;">90,189 real, anonymised players</strong>.
                        No synthetic data was generated at any stage.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run `python churn_model.py` to generate `models/metrics.json`.")


if __name__ == "__main__":
    main()
