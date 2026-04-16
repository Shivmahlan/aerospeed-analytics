# AeroSpeed Analytics

> F1 telemetry meets road car aerodynamics — a full data science pipeline from scraping through ML modeling and dual dashboard deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-R²%3D0.89-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Overview

AeroSpeed Analytics is a dual-domain data science project that combines **Formula 1 race telemetry** (2022–2024) with **EPA road car fuel economy and drag coefficient data** to explore the relationship between aerodynamic efficiency and performance — across both motorsport and consumer vehicles.

The project covers the full pipeline: data scraping, cleaning, EDA, physics-informed feature engineering, ML modeling, and deployment via both a Streamlit dashboard and a React + FastAPI web app.

---

## Screenshots

### Streamlit Dashboard
![Streamlit Overview](docs/streamlit_overview.png)
*Overview page — 60,217 laps across 25 circuits and 3 seasons*

### React Frontend
![React Hero](docs/react_hero.png)
*Animated hero with speed lines and live team cards pulled from FastAPI*

---

## Key Results

| Model | Algorithm | Metric |
|---|---|---|
| Highway MPG predictor | Random Forest | R² = 0.865, MAE = 1.45 MPG |
| Lap time delta predictor | XGBoost | R² = 0.890, MAE = 0.51s |
| Podium probability | Gradient Boosting | Accuracy = 94%, Precision = 80% |

**Notable EDA findings:**
- Qualifying pace vs race pace correlation: **r = 0.92** across 3 seasons
- Engine displacement vs MPG: **r = -0.72** across 18,392 vehicles
- Soft tyres degrade measurably faster than mediums from lap 5 onward
- Haas and Williams run lower downforce setups — trading cornering for straight-line speed

---

## Data Sources

| Source | Method | Records |
|---|---|---|
| FastF1 + OpenF1 API | Python scraper | 60,217 F1 race laps |
| FastF1 qualifying | Python scraper | 1,029 qualifying rows |
| EPA fueleconomy.gov | ZIP download + clean | 18,392 road cars |
| Wikipedia Cd table | BeautifulSoup scrape | 21 drag coefficient values |

---

## Project Structure

```
aerospeed/
├── data/
│   ├── cache/               # FastF1 session cache (gitignored)
│   └── processed/           # Cleaned CSVs
│       ├── f1_race_laps.csv
│       ├── f1_qualifying.csv
│       ├── f1_ml_ready.csv
│       ├── cars_epa.csv
│       ├── cars_unified.csv
│       └── cars_ml_ready.csv
├── scripts/
│   ├── scraper_f1.py        # Tiered F1 data collector (--tier 1/2/3)
│   ├── scraper_cars.py      # EPA + Cd scraper and merger
│   └── validate_data.py     # Post-collection data quality checker
├── notebooks/
│   ├── 01_eda_f1.ipynb      # F1 EDA — 7 charts
│   ├── 02_eda_cars.ipynb    # Cars EDA — 4 charts
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── models/
│   ├── mpg_predictor.pkl
│   ├── laptime_predictor.pkl
│   └── win_probability.pkl
├── dashboard/
│   └── app.py               # Streamlit app (6 pages)
├── backend/
│   └── main.py              # FastAPI — 8 endpoints
└── frontend/                # React + Framer Motion + Three.js
    └── src/
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/aerospeed-analytics.git
cd aerospeed-analytics

conda create -n aerospeed python=3.11 -y
conda activate aerospeed
pip install -r requirements_data.txt
```

### 2. Collect data

```bash
# Tier 1 — race laps, all circuits (run overnight)
python scripts/scraper_f1.py --tier 1

# Tier 3 — qualifying data
python scripts/scraper_f1.py --tier 3

# Road car data
python scripts/scraper_cars.py

# Validate
python scripts/validate_data.py
```

### 3. Run notebooks

Open Jupyter and run notebooks 01 through 04 in order.

```bash
jupyter notebook
```

### 4. Launch Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Launch FastAPI + React (optional)

```bash
# Terminal 1
uvicorn backend.main:app --reload --port 8000

# Terminal 2
cd frontend
npm install
npm run dev
```

---

## Features

**Streamlit dashboard (6 pages)**
- Overview — dataset stats, circuit ranking, team pace evolution
- Team Analysis — per-team metrics with tyre strategy breakdown
- Circuit Insights — lap distribution, speed trap by team
- Predict MPG — interactive sliders, Random Forest inference
- Predict Lap Time — XGBoost delta prediction with compound + tyre inputs
- Win Probability — podium likelihood per driver per race

**FastAPI backend (8 endpoints)**
- `/data/overview`, `/data/teams`, `/data/circuits`, `/data/team_evolution`
- `/data/qualifying/{gp}/{year}`
- `POST /predict/mpg`, `POST /predict/laptime`, `POST /predict/winprob`

**React frontend**
- Animated hero with procedural speed lines
- Live team cards with team brand colors
- Framer Motion page transitions
- Three.js 3D components (in progress)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data collection | `fastf1`, `requests`, `beautifulsoup4`, `rapidfuzz` |
| Data processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| ML | `scikit-learn`, `xgboost` |
| Dashboard | `streamlit` |
| Backend | `fastapi`, `uvicorn`, `joblib` |
| Frontend | `react`, `framer-motion`, `three.js`, `plotly.js` |

---

## Dataset on Kaggle

The cleaned dataset (F1 race laps 2022–2024 + EPA fuel economy) is published on Kaggle:
👉 https://www.kaggle.com/datasets/shivmahlan/aerospeed-analytics

---

## License

MIT — free to use, just credit the repo.

---

## Author

**Shiv** — B.Tech CSE (Data Science), Chaudhary Devi Lal University, Sirsa  
· [LinkedIn](https://www.linkedin.com/in/shivmahlan/)
