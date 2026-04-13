from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="AeroSpeed Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models + data ─────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
MODELS  = os.path.join(BASE, '..', 'models')
DATA    = os.path.join(BASE, '..', 'data', 'processed')

mpg_model = joblib.load(os.path.join(MODELS, 'mpg_predictor.pkl'))
lap_model = joblib.load(os.path.join(MODELS, 'laptime_predictor.pkl'))
win_model = joblib.load(os.path.join(MODELS, 'win_probability.pkl'))

laps = pd.read_csv(os.path.join(DATA, 'f1_ml_ready.csv'))
qual = pd.read_csv(os.path.join(DATA, 'qual_ml_ready.csv'))
cars = pd.read_csv(os.path.join(DATA, 'cars_ml_ready.csv'))

# ── Request schemas ────────────────────────────────────────────
class MPGRequest(BaseModel):
    displacement: float
    cylinders: int
    fuel_type: str
    hwy_city_ratio: float
    year: int

class LapRequest(BaseModel):
    compound: str
    tyre_life: int
    lap_number: int
    year: int
    speed_norm: float
    s1_ratio: float
    s2_ratio: float
    team: str

class WinRequest(BaseModel):
    best_qual: float
    gap_to_pole: float
    team_avg: float
    recent_form: float

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AeroSpeed API running"}

@app.get("/data/overview")
def overview():
    return {
        "total_laps":    int(len(laps)),
        "circuits":      int(laps['GP'].nunique()),
        "teams":         int(laps['Team'].nunique()),
        "road_cars":     int(len(cars)),
        "years":         sorted(laps['Year'].unique().tolist()),
    }

@app.get("/data/teams")
def teams(year: int = 2024):
    dry = laps[
        laps['Compound'].isin(['SOFT','MEDIUM','HARD']) &
        (laps['Year'] == year)
    ]
    result = []
    for team in sorted(dry['Team'].unique()):
        t = dry[dry['Team'] == team]
        result.append({
            "team":       team,
            "median_lap": round(float(t['LapTime'].median()), 3),
            "top_speed":  round(float(t['SpeedST'].median()), 1),
            "avg_tyre_life": round(float(t['TyreLife'].median()), 1),
        })
    return sorted(result, key=lambda x: x['median_lap'])

@app.get("/data/circuits")
def circuits():
    dry = laps[laps['Compound'].isin(['SOFT','MEDIUM','HARD'])]
    result = []
    for gp in sorted(dry['GP'].unique()):
        c = dry[dry['GP'] == gp]
        result.append({
            "gp":         gp,
            "median_lap": round(float(c['LapTime'].median()), 3),
            "top_speed":  round(float(c['SpeedST'].median()), 1),
            "avg_tyre":   round(float(c['TyreLife'].median()), 1),
        })
    return sorted(result, key=lambda x: x['median_lap'])

@app.get("/data/team_evolution")
def team_evolution():
    dry = laps[laps['Compound'].isin(['SOFT','MEDIUM','HARD'])]
    result = (dry.groupby(['Team','Year'])['LapTime']
              .median().reset_index()
              .rename(columns={'LapTime':'median_lap'}))
    result['median_lap'] = result['median_lap'].round(3)
    return result.to_dict(orient='records')

@app.get("/data/qualifying/{gp}/{year}")
def qualifying(gp: str, year: int):
    q = qual[
        (qual['GP'] == gp) &
        (qual['Year'] == year)
    ].dropna(subset=['BestQual','GapToPole','TeamAvgQual','RecentForm'])

    if q.empty:
        return []

    q = q.copy()
    q['PodiumProb'] = (win_model.predict_proba(
        q[['BestQual','GapToPole','TeamAvgQual','RecentForm']]
    )[:, 1] * 100).round(1)

    return q[['Abbreviation','FullName','TeamName',
              'BestQual','GapToPole','QualPos','PodiumProb']].to_dict(orient='records')

@app.post("/predict/mpg")
def predict_mpg(req: MPGRequest):
    fuel_map = {'regular': 2, 'premium': 1, 'diesel': 0, 'hybrid': 3}
    features = [[
        req.displacement,
        req.cylinders,
        req.displacement / req.cylinders,
        fuel_map.get(req.fuel_type, 2),
        req.hwy_city_ratio,
        req.year - 2010
    ]]
    pred = mpg_model.predict(features)[0]
    return {"predicted_mpg": round(float(pred), 1)}

@app.post("/predict/laptime")
def predict_laptime(req: LapRequest):
    compound_enc = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}.get(req.compound, 1)
    teams        = sorted(laps['Team'].unique().tolist())
    team_enc     = teams.index(req.team) if req.team in teams else 0
    s3_ratio     = max(0.01, 1 - req.s1_ratio - req.s2_ratio)

    features = [[
        compound_enc, req.tyre_life, req.speed_norm,
        req.s1_ratio, req.s2_ratio, s3_ratio,
        req.lap_number, req.year - 2022, team_enc
    ]]
    pred = lap_model.predict(features)[0]
    return {
        "predicted_delta": round(float(pred), 3),
        "faster_or_slower": "faster" if pred < 0 else "slower"
    }

@app.post("/predict/winprob")
def predict_winprob(req: WinRequest):
    features = [[req.best_qual, req.gap_to_pole,
                 req.team_avg, req.recent_form]]
    prob      = win_model.predict_proba(features)[0][1]
    predicted = win_model.predict(features)[0]
    return {
        "podium_probability": round(float(prob) * 100, 1),
        "is_podium_contender": bool(predicted)
    }
