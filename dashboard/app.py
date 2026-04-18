import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(
    page_title="AeroSpeed Analytics",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none; }
.stApp { background-color: #0e1117; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }
h1 { font-size: 2.4rem !important; font-weight: 700 !important; color: #ffffff !important; }
h2 { font-size: 1.4rem !important; font-weight: 600 !important; color: #ffffff !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #e0e0e0 !important; }
p  { color: #9ba3b2 !important; font-size: 0.95rem !important; }
.card {
    background: #161b27;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}
.section-title { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin: 2rem 0 0.3rem 0; }
.section-sub   { color: #6b7280; font-size: 0.88rem; margin-bottom: 1.5rem; }
[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280;
    border-radius: 0;
    padding: 0.6rem 1.4rem;
    font-size: 0.9rem;
    font-weight: 500;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #ffffff !important;
    border-bottom: 2px solid #e10600 !important;
}
.stSelectbox > div > div {
    background: #161b27 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: white !important;
}
.stButton > button {
    background: #7c3aed;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
}
.stButton > button:hover { background: #6d28d9; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1.5rem 0; }
.result-box {
    background: #161b27;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 2.5rem;
    text-align: center;
}
.result-value { font-size: 4rem; font-weight: 800; color: #ffffff; line-height: 1; }
.result-label { font-size: 0.85rem; color: #6b7280; margin-top: 0.5rem; }
.badge {
    display: inline-block;
    background: rgba(124,58,237,0.2);
    color: #a78bfa;
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font_color="#9ba3b2",
    font_size=12,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#9ba3b2"),
)

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6", "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",       "McLaren": "#FF8000",
    "Aston Martin": "#229971",   "Alpine": "#FF87BC",
    "Williams": "#64C4FF",       "AlphaTauri": "#6692FF",
    "Alfa Romeo": "#C92D4B",     "Haas F1 Team": "#B6BABD",
    "RB": "#6692FF",             "Kick Sauber": "#52E252",
}

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, "..", "data", "processed")
    laps = pd.read_csv(os.path.join(data, "f1_ml_ready.csv"))
    qual = pd.read_csv(os.path.join(data, "qual_ml_ready.csv"))
    cars = pd.read_csv(os.path.join(data, "cars_ml_ready.csv"))
    return laps, qual, cars

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    mdir = os.path.join(base, "..", "models")
    return (
        joblib.load(os.path.join(mdir, "mpg_predictor.pkl")),
        joblib.load(os.path.join(mdir, "laptime_predictor.pkl")),
        joblib.load(os.path.join(mdir, "win_probability.pkl")),
    )

laps, qual, cars = load_data()
mpg_model, lap_model, win_model = load_models()

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1rem 0 0.5rem 0;">
    <div style="display:flex; align-items:center; gap:0.75rem;">
        <span style="font-size:2.2rem;">🏎️</span>
        <h1 style="margin:0;">AeroSpeed Analytics</h1>
    </div>
    <p style="margin:0.4rem 0 0 0; color:#6b7280; font-size:0.95rem;">
        F1 Telemetry &amp; Road Car Aerodynamics — 2022 to 2024
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Laps",  f"{len(laps):,}")
c2.metric("Circuits",    f"{laps['GP'].nunique()}")
c3.metric("Teams",       f"{laps['Team'].nunique()}")
c4.metric("Road Cars",   f"{len(cars):,}")
c5.metric("Seasons",     "3  (22–24)")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Overview", "🏁  Team Analysis", "🗺️  Circuit Insights",
    "⛽  Predict MPG", "⏱️  Predict Lap Time", "🏆  Win Probability"
])

# ── TAB 1: OVERVIEW ───────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">Season overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Median lap times and team pace trends across all circuits.</p>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Median lap time by circuit**")
        circuit_med = laps.groupby("GP")["LapTime"].median().sort_values(ascending=True).reset_index()
        circuit_med["GP"] = circuit_med["GP"].str.replace(" Grand Prix", "")
        fig = px.bar(circuit_med, x="LapTime", y="GP", orientation="h",
                     color_discrete_sequence=["#e10600"])
        fig.update_layout(**CHART_LAYOUT, height=480, yaxis_title="", xaxis_title="Median lap (s)")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Team pace evolution by season**")
        dry = laps[laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
        team_yr = dry.groupby(["Team", "Year"])["LapTime"].median().reset_index()
        fig2 = px.line(team_yr, x="Year", y="LapTime", color="Team",
                       color_discrete_map=TEAM_COLORS, markers=True)
        fig2.update_layout(**CHART_LAYOUT, height=480)
        fig2.update_traces(line_width=2)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if "model" in cars.columns and "cd_value" in cars.columns:
        st.markdown('<p class="section-title">Drag coefficient comparison</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Road cars ranked by aerodynamic drag — lower Cd = more efficient.</p>', unsafe_allow_html=True)
        cd_sorted = cars[["model", "cd_value"]].dropna().sort_values("cd_value")
        fig3 = px.bar(cd_sorted, x="cd_value", y="model", orientation="h",
                      color="cd_value", color_continuous_scale=["#27F4D2", "#e10600"])
        fig3.update_layout(**CHART_LAYOUT, height=500, yaxis_title="",
                           xaxis_title="Drag coefficient (Cd)", coloraxis_showscale=False)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 2: TEAM ANALYSIS ──────────────────────────────────────
with tab2:
    st.markdown('<p class="section-title">Constructor performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Compare teams by lap time, speed, and tyre strategy.</p>', unsafe_allow_html=True)

    selected_year = st.selectbox("Season", [2022, 2023, 2024], index=2, key="team_year")
    dry = laps[laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"]) & (laps["Year"] == selected_year)]
    teams = sorted(dry["Team"].unique())
    cols  = st.columns(4)
    for i, team in enumerate(teams):
        t_data = dry[dry["Team"] == team]
        med    = t_data["LapTime"].median()
        speed  = t_data["SpeedST"].median()
        color  = TEAM_COLORS.get(team, "#ffffff")
        with cols[i % 4]:
            st.markdown(f"""
            <div class="card" style="border-top: 3px solid {color};">
                <div style="font-weight:700; color:{color}; font-size:0.95rem; margin-bottom:0.6rem;">{team}</div>
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <div style="color:#6b7280; font-size:0.72rem; text-transform:uppercase;">Median lap</div>
                        <div style="color:#fff; font-weight:700; font-size:1.1rem;">{med:.2f}s</div>
                    </div>
                    <div>
                        <div style="color:#6b7280; font-size:0.72rem; text-transform:uppercase;">Top speed</div>
                        <div style="color:#fff; font-weight:700; font-size:1.1rem;">{speed:.0f} km/h</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Tyre strategy distribution**")
    compound_dist = dry.groupby(["Team", "Compound"]).size().reset_index(name="count")
    fig = px.bar(compound_dist, x="Team", y="count", color="Compound",
                 color_discrete_map={"SOFT": "#e10600", "MEDIUM": "#ffd700", "HARD": "#e0e0e0"},
                 barmode="stack")
    fig.update_layout(**CHART_LAYOUT, xaxis_tickangle=-35, yaxis_title="Laps")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 3: CIRCUIT INSIGHTS ───────────────────────────────────
with tab3:
    st.markdown('<p class="section-title">Circuit deep dive</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Lap time distributions, speed traps, and tyre wear by circuit.</p>', unsafe_allow_html=True)

    gp_list     = sorted(laps["GP"].unique())
    selected_gp = st.selectbox("Select circuit", gp_list, key="circuit_gp")
    c_data = laps[laps["GP"] == selected_gp]
    dry_c  = c_data[c_data["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median lap",    f"{dry_c['LapTime'].median():.2f}s")
    m2.metric("Top speed",     f"{dry_c['SpeedST'].median():.0f} km/h")
    m3.metric("Avg tyre life", f"{dry_c['TyreLife'].median():.0f} laps")
    m4.metric("Laps recorded", f"{len(dry_c):,}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Lap time distribution by year**")
        fig = px.box(dry_c, x="Year", y="LapTime", color="Year",
                     color_discrete_sequence=["#e10600", "#ff8c00", "#ffd700"])
        fig.update_layout(**CHART_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Speed trap by team**")
        speed_team = dry_c.groupby("Team")["SpeedST"].median().sort_values(ascending=False).reset_index()
        fig2 = px.bar(speed_team, x="Team", y="SpeedST",
                      color="Team", color_discrete_map=TEAM_COLORS)
        fig2.update_layout(**CHART_LAYOUT, showlegend=False, xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 4: PREDICT MPG ────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-title">Highway MPG predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Predict fuel efficiency using engine and aerodynamic specs.</p>', unsafe_allow_html=True)

    col_in, col_out = st.columns(2, gap="large")
    with col_in:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Vehicle parameters**")
        displacement = st.slider("Engine displacement (L)", 1.0, 8.5, 2.0, 0.1)
        cylinders    = st.selectbox("Cylinders", [3, 4, 6, 8, 10, 12], index=1)
        fuel_type    = st.selectbox("Fuel type", ["regular", "premium", "diesel", "hybrid"])
        hwy_city     = st.slider("Highway / city ratio", 1.0, 2.0, 1.3, 0.05)
        year         = st.slider("Model year", 2010, 2026, 2023)
        st.markdown("</div>", unsafe_allow_html=True)

    fuel_map     = {"regular": 2, "premium": 1, "diesel": 0, "hybrid": 3}
    disp_per_cyl = displacement / cylinders
    pred_mpg     = mpg_model.predict([[displacement, cylinders, disp_per_cyl,
                                       fuel_map[fuel_type], hwy_city, year - 2010]])[0]
    with col_out:
        st.markdown(f"""
        <div class="result-box">
            <div style="color:#6b7280; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">Predicted highway MPG</div>
            <div class="result-value">{pred_mpg:.1f}</div>
            <div class="result-label">miles per gallon</div>
            <div class="badge">Random Forest · R²=0.865</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Similar cars in dataset**")
    similar = cars[
        (cars["engine_displacement_l"].between(displacement - 0.5, displacement + 0.5)) &
        (cars["cylinders"] == cylinders)
    ][["make", "model", "year", "highway_mpg", "engine_displacement_l"]].dropna()
    st.dataframe(similar.head(10), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 5: PREDICT LAP TIME ───────────────────────────────────
with tab5:
    st.markdown('<p class="section-title">Lap time delta predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Estimate how a lap compares to the circuit median.</p>', unsafe_allow_html=True)

    col_in, col_out = st.columns(2, gap="large")
    with col_in:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Lap parameters**")
        compound   = st.selectbox("Tyre compound", ["SOFT", "MEDIUM", "HARD"])
        tyre_life  = st.slider("Tyre age (laps)", 1, 50, 10)
        lap_number = st.slider("Lap number", 1, 70, 20)
        year_lt    = st.selectbox("Season", [2022, 2023, 2024], index=2, key="lt_year")
        speed_norm = st.slider("Relative speed (1.0 = avg)", 0.90, 1.10, 1.0, 0.01)
        s1_ratio   = st.slider("Sector 1 ratio", 0.20, 0.45, 0.30, 0.01)
        s2_ratio   = st.slider("Sector 2 ratio", 0.25, 0.50, 0.38, 0.01)
        team_lt    = st.selectbox("Team", sorted(laps["Team"].unique()), key="lt_team")
        st.markdown("</div>", unsafe_allow_html=True)

    s3_ratio     = max(0.01, 1 - s1_ratio - s2_ratio)
    compound_enc = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}[compound]
    team_enc     = sorted(laps["Team"].unique()).index(team_lt)
    pred_delta   = lap_model.predict([[compound_enc, tyre_life, speed_norm,
                                       s1_ratio, s2_ratio, s3_ratio,
                                       lap_number, year_lt - 2022, team_enc]])[0]
    accent = "#e10600" if pred_delta > 0 else "#27F4D2"
    label  = "slower than median" if pred_delta > 0 else "faster than median"

    with col_out:
        st.markdown(f"""
        <div class="result-box" style="border-top: 3px solid {accent};">
            <div style="color:#6b7280; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">Predicted lap delta</div>
            <div class="result-value" style="color:{accent};">{pred_delta:+.2f}s</div>
            <div class="result-label">{abs(pred_delta):.2f}s {label}</div>
            <div class="badge">XGBoost · R²=0.890</div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 6: WIN PROBABILITY ────────────────────────────────────
with tab6:
    st.markdown('<p class="section-title">Podium probability</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Estimate podium likelihood based on qualifying performance.</p>', unsafe_allow_html=True)

    col_in, col_out = st.columns(2, gap="large")
    with col_in:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Qualifying inputs**")
        best_qual   = st.number_input("Best qualifying lap (s)", 60.0, 130.0, 88.0, 0.1)
        gap_to_pole = st.number_input("Gap to pole (s)", 0.0, 5.0, 0.5, 0.05)
        team_avg    = st.number_input("Team avg qualifying (s)", 60.0, 130.0, 89.0, 0.1)
        recent_form = st.number_input("Recent avg gap to pole (s)", 0.0, 5.0, 0.8, 0.05)
        st.markdown("</div>", unsafe_allow_html=True)

    prob      = win_model.predict_proba([[best_qual, gap_to_pole, team_avg, recent_form]])[0][1]
    predicted = win_model.predict([[best_qual, gap_to_pole, team_avg, recent_form]])[0]
    p_color   = "#e10600" if prob > 0.5 else "#6b7280"
    p_label   = "Podium contender 🏆" if predicted else "Outside podium"

    with col_out:
        st.markdown(f"""
        <div class="result-box" style="border-top: 3px solid {p_color};">
            <div style="color:#6b7280; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">Podium probability</div>
            <div class="result-value" style="color:{p_color};">{prob*100:.1f}%</div>
            <div class="result-label">{p_label}</div>
            <div class="badge">Gradient Boosting · Acc=0.94</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Race podium outlook</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Podium probability for all drivers in a selected race.</p>', unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    sel_gp   = rc1.selectbox("Race", sorted(qual["GP"].unique()), key="win_gp")
    sel_yr   = rc2.selectbox("Year", [2022, 2023, 2024], index=2, key="win_yr")

    race_qual = qual[
        (qual["GP"] == sel_gp) & (qual["Year"] == sel_yr)
    ].dropna(subset=["BestQual", "GapToPole", "TeamAvgQual", "RecentForm"])

    if not race_qual.empty:
        race_qual = race_qual.copy()
        race_qual["PodiumProb"] = win_model.predict_proba(
            race_qual[["BestQual", "GapToPole", "TeamAvgQual", "RecentForm"]]
        )[:, 1] * 100
        race_qual = race_qual.sort_values("PodiumProb", ascending=False)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.bar(race_qual.head(10), x="Abbreviation", y="PodiumProb",
                     color="PodiumProb",
                     color_continuous_scale=["#1e293b", "#e10600"],
                     labels={"PodiumProb": "Podium prob (%)"})
        fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="Probability (%)")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No qualifying data available for this race/year.")
