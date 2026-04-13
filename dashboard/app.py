import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AeroSpeed Analytics",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e0e0e; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #e10600; }
    .metric-card {
        background: #1e1e1e;
        border-left: 4px solid #e10600;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stMetric label { color: #aaaaaa !important; }
    .stMetric value { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, '..', 'data', 'processed')

    laps  = pd.read_csv(os.path.join(data, 'f1_ml_ready.csv'))
    qual  = pd.read_csv(os.path.join(data, 'qual_ml_ready.csv'))
    cars  = pd.read_csv(os.path.join(data, 'cars_ml_ready.csv'))
    return laps, qual, cars

@st.cache_resource
def load_models():
    base   = os.path.dirname(os.path.abspath(__file__))
    models = os.path.join(base, '..', 'models')
    mpg_model  = joblib.load(os.path.join(models, 'mpg_predictor.pkl'))
    lap_model  = joblib.load(os.path.join(models, 'laptime_predictor.pkl'))
    win_model  = joblib.load(os.path.join(models, 'win_probability.pkl'))
    return mpg_model, lap_model, win_model

laps, qual, cars = load_data()
mpg_model, lap_model, win_model = load_models()

# ── Team colors ───────────────────────────────────────────────
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Ferrari':         '#E8002D',
    'Mercedes':        '#27F4D2',
    'McLaren':         '#FF8000',
    'Aston Martin':    '#229971',
    'Alpine':          '#FF87BC',
    'Williams':        '#64C4FF',
    'AlphaTauri':      '#6692FF',
    'Alfa Romeo':      '#C92D4B',
    'Haas F1 Team':    '#B6BABD',
    'RB':              '#6692FF',
    'Kick Sauber':     '#52E252',
}

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("AeroSpeed Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Team Analysis", "Circuit Insights",
     "Predict MPG", "Predict Lap Time", "Win Probability"]
)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("AeroSpeed Analytics")
    st.markdown("##### F1 Telemetry + Road Car Aerodynamics — 2022 to 2024")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Laps",    f"{len(laps):,}")
    col2.metric("Circuits",      f"{laps['GP'].nunique()}")
    col3.metric("Teams",         f"{laps['Team'].nunique()}")
    col4.metric("Road Cars",     f"{len(cars):,}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Lap time by circuit")
        circuit_med = (laps.groupby('GP')['LapTime']
                       .median().sort_values(ascending=False)
                       .reset_index())
        circuit_med['GP'] = circuit_med['GP'].str.replace(' Grand Prix','')
        fig = px.bar(circuit_med, x='LapTime', y='GP',
                     orientation='h', color_discrete_sequence=['#e10600'])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=500,
            yaxis_title='', xaxis_title='Median lap time (s)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Team pace evolution")
        dry = laps[laps['Compound'].isin(['SOFT','MEDIUM','HARD'])]
        team_yr = (dry.groupby(['Team','Year'])['LapTime']
                   .median().reset_index())
        fig2 = px.line(team_yr, x='Year', y='LapTime', color='Team',
                       color_discrete_map=TEAM_COLORS, markers=True)
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=500
        )
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "Team Analysis":
    st.title("Team Analysis")
    st.markdown("---")

    selected_year = st.selectbox("Season", [2022, 2023, 2024], index=2)
    dry = laps[
        laps['Compound'].isin(['SOFT','MEDIUM','HARD']) &
        (laps['Year'] == selected_year)
    ]

    teams = sorted(dry['Team'].unique())
    cols  = st.columns(3)

    for i, team in enumerate(teams):
        t_data = dry[dry['Team'] == team]
        med    = t_data['LapTime'].median()
        speed  = t_data['SpeedST'].median()
        color  = TEAM_COLORS.get(team, '#ffffff')

        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#1e1e1e; border-left: 5px solid {color};
                        padding:1rem; border-radius:8px; margin:0.5rem 0;">
                <h4 style="color:{color}; margin:0">{team}</h4>
                <p style="color:#aaa; margin:4px 0">Median lap: <b style="color:white">{med:.2f}s</b></p>
                <p style="color:#aaa; margin:4px 0">Top speed: <b style="color:white">{speed:.0f} km/h</b></p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Tyre strategy by team")
    compound_dist = (dry.groupby(['Team','Compound'])
                     .size().reset_index(name='count'))
    fig = px.bar(compound_dist, x='Team', y='count', color='Compound',
                 color_discrete_map={
                     'SOFT':'#e10600','MEDIUM':'#ffd700','HARD':'#f0f0f0'
                 }, barmode='stack')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — CIRCUIT INSIGHTS
# ═══════════════════════════════════════════════════════════════
elif page == "Circuit Insights":
    st.title("Circuit Insights")
    st.markdown("---")

    gp_list = sorted(laps['GP'].unique())
    selected_gp = st.selectbox("Select circuit", gp_list)

    c_data = laps[laps['GP'] == selected_gp]
    dry_c  = c_data[c_data['Compound'].isin(['SOFT','MEDIUM','HARD'])]

    col1, col2, col3 = st.columns(3)
    col1.metric("Median lap time", f"{dry_c['LapTime'].median():.2f}s")
    col2.metric("Median top speed", f"{dry_c['SpeedST'].median():.0f} km/h")
    col3.metric("Avg tyre life",    f"{dry_c['TyreLife'].median():.0f} laps")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Lap time distribution by year")
        fig = px.box(dry_c, x='Year', y='LapTime', color='Year',
                     color_discrete_sequence=['#e10600','#ff8c00','#ffd700'])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Speed trap by team")
        speed_team = (dry_c.groupby('Team')['SpeedST']
                      .median().sort_values(ascending=False)
                      .reset_index())
        fig2 = px.bar(speed_team, x='Team', y='SpeedST',
                      color='Team', color_discrete_map=TEAM_COLORS)
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', showlegend=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT MPG
# ═══════════════════════════════════════════════════════════════
elif page == "Predict MPG":
    st.title("Predict Highway MPG")
    st.markdown("Adjust the sliders to predict highway fuel efficiency.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        displacement = st.slider("Engine displacement (L)", 1.0, 8.5, 2.0, 0.1)
        cylinders    = st.selectbox("Cylinders", [3, 4, 6, 8, 10, 12], index=1)
        fuel_type    = st.selectbox("Fuel type", ['regular','premium','diesel','hybrid'])
    with col2:
        hwy_city     = st.slider("Highway/city ratio", 1.0, 2.0, 1.3, 0.05)
        year         = st.slider("Model year", 2010, 2026, 2023)

    fuel_map = {'regular': 2, 'premium': 1, 'diesel': 0, 'hybrid': 3}
    disp_per_cyl = displacement / cylinders
    year_rel     = year - 2010
    fuel_enc     = fuel_map[fuel_type]

    features = [[displacement, cylinders, disp_per_cyl,
                 fuel_enc, hwy_city, year_rel]]
    pred_mpg = mpg_model.predict(features)[0]

    st.markdown("---")
    st.markdown(f"""
    <div style="background:#1e1e1e; border-left:5px solid #e10600;
                padding:2rem; border-radius:8px; text-align:center;">
        <h2 style="color:#e10600">Predicted Highway MPG</h2>
        <h1 style="color:white; font-size:4rem">{pred_mpg:.1f}</h1>
        <p style="color:#aaa">Based on Random Forest model (R²=0.865)</p>
    </div>
    """, unsafe_allow_html=True)

    # Show similar cars
    st.markdown("---")
    st.subheader("Similar cars in dataset")
    similar = cars[
        (cars['engine_displacement_l'].between(displacement-0.5, displacement+0.5)) &
        (cars['cylinders'] == cylinders)
    ][['make','model','year','highway_mpg','engine_displacement_l']].dropna()
    st.dataframe(similar.head(10), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT LAP TIME
# ═══════════════════════════════════════════════════════════════
elif page == "Predict Lap Time":
    st.title("Predict Lap Time Delta")
    st.markdown("Predict how a driver's lap compares to circuit median.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        compound    = st.selectbox("Tyre compound", ['SOFT','MEDIUM','HARD'])
        tyre_life   = st.slider("Tyre age (laps)", 1, 50, 10)
        lap_number  = st.slider("Lap number", 1, 70, 20)
        year        = st.selectbox("Season", [2022, 2023, 2024], index=2)
    with col2:
        speed_norm  = st.slider("Relative speed (1.0 = average)", 0.90, 1.10, 1.0, 0.01)
        s1_ratio    = st.slider("Sector 1 ratio", 0.20, 0.45, 0.30, 0.01)
        s2_ratio    = st.slider("Sector 2 ratio", 0.25, 0.50, 0.38, 0.01)
        team        = st.selectbox("Team", sorted(laps['Team'].unique()))

    s3_ratio    = max(0.01, 1 - s1_ratio - s2_ratio)
    compound_enc = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}[compound]
    team_enc    = sorted(laps['Team'].unique()).index(team)
    year_rel    = year - 2022

    features = [[compound_enc, tyre_life, speed_norm,
                 s1_ratio, s2_ratio, s3_ratio,
                 lap_number, year_rel, team_enc]]
    pred_delta = lap_model.predict(features)[0]

    color = '#e10600' if pred_delta > 0 else '#27F4D2'
    label = 'slower than' if pred_delta > 0 else 'faster than'

    st.markdown("---")
    st.markdown(f"""
    <div style="background:#1e1e1e; border-left:5px solid {color};
                padding:2rem; border-radius:8px; text-align:center;">
        <h2 style="color:{color}">Predicted lap delta</h2>
        <h1 style="color:white; font-size:4rem">{pred_delta:+.2f}s</h1>
        <p style="color:#aaa">{abs(pred_delta):.2f}s {label} circuit median</p>
        <p style="color:#aaa">Based on XGBoost model (R²=0.890)</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 6 — WIN PROBABILITY
# ═══════════════════════════════════════════════════════════════
elif page == "Win Probability":
    st.title("Podium Probability")
    st.markdown("Estimate podium likelihood based on qualifying performance.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        best_qual    = st.number_input("Best qualifying lap (s)", 60.0, 130.0, 88.0, 0.1)
        gap_to_pole  = st.number_input("Gap to pole (s)", 0.0, 5.0, 0.5, 0.05)
    with col2:
        team_avg     = st.number_input("Team avg qualifying (s)", 60.0, 130.0, 89.0, 0.1)
        recent_form  = st.number_input("Recent avg gap to pole (s)", 0.0, 5.0, 0.8, 0.05)

    features  = [[best_qual, gap_to_pole, team_avg, recent_form]]
    prob      = win_model.predict_proba(features)[0][1]
    predicted = win_model.predict(features)[0]

    color = '#e10600' if prob > 0.5 else '#aaaaaa'
    label = 'Podium contender' if predicted else 'Outside podium'

    st.markdown("---")
    st.markdown(f"""
    <div style="background:#1e1e1e; border-left:5px solid {color};
                padding:2rem; border-radius:8px; text-align:center;">
        <h2 style="color:{color}">{label}</h2>
        <h1 style="color:white; font-size:4rem">{prob*100:.1f}%</h1>
        <p style="color:#aaa">Podium probability — Gradient Boosting (Acc=0.94)</p>
    </div>
    """, unsafe_allow_html=True)

    # Show all drivers for a selected race
    st.markdown("---")
    st.subheader("Race podium outlook")
    selected_gp   = st.selectbox("Select race", sorted(qual['GP'].unique()))
    selected_year = st.selectbox("Year", [2022, 2023, 2024], index=2)

    race_qual = qual[
        (qual['GP'] == selected_gp) &
        (qual['Year'] == selected_year)
    ].dropna(subset=['BestQual','GapToPole','TeamAvgQual','RecentForm'])

    if not race_qual.empty:
        race_qual = race_qual.copy()
        race_qual['PodiumProb'] = win_model.predict_proba(
            race_qual[['BestQual','GapToPole','TeamAvgQual','RecentForm']]
        )[:, 1] * 100
        race_qual = race_qual.sort_values('PodiumProb', ascending=False)

        fig = px.bar(race_qual.head(10),
                     x='Abbreviation', y='PodiumProb',
                     color='PodiumProb',
                     color_continuous_scale=['#333333','#e10600'],
                     labels={'PodiumProb': 'Podium probability (%)'})
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No qualifying data for this race/year combination.")
