"""
AeroSpeed Analytics — Road Car Data Collector
===============================================
Pulls two datasets:
  1. EPA fuel economy data (fueleconomy.gov)  — MPG, displacement, weight
  2. Drag coefficient (Cd) values             — Wikipedia wikitable scrape

Usage:
  python scraper_cars.py
"""

import os
import io
import time
import logging
import zipfile
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ── Logging ────────────────────────────────────────────────────
os.makedirs('./logs', exist_ok=True)
os.makedirs('./data/processed', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('./logs/scraper_cars.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}


# ── EPA Fuel Economy ───────────────────────────────────────────
def pull_epa():
    log.info("=" * 50)
    log.info("EPA Fuel Economy — fueleconomy.gov")
    log.info("=" * 50)

    url = 'https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip'

    log.info("Downloading vehicles.csv.zip ...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
    except Exception as e:
        log.error(f"Download failed: {e}")
        return

    log.info(f"Downloaded {len(r.content)/1e6:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        filename = [f for f in z.namelist() if f.endswith('.csv')][0]
        df_raw = pd.read_csv(z.open(filename), low_memory=False)

    log.info(f"Raw shape: {df_raw.shape}")

    # ── Column selection ──────────────────────────────────────
    wanted = {
        'id'         : 'vehicle_id',
        'make'       : 'make',
        'model'      : 'model',
        'year'       : 'year',
        'VClass'     : 'vehicle_class',
        'drive'      : 'drive_type',
        'trany'      : 'transmission',
        'displ'      : 'engine_displacement_l',
        'cylinders'  : 'cylinders',
        'fuelType'   : 'fuel_type',
        'fuelType1'  : 'fuel_type_detail',
        'city08'     : 'city_mpg',
        'highway08'  : 'highway_mpg',
        'comb08'     : 'combined_mpg',
        'co2'        : 'co2_grams_per_mile',
        'ghgScore'   : 'ghg_score',
        'hlv'        : 'hatchback_luggage_vol',
        'range'      : 'ev_range_miles',
        'charge240'  : 'ev_charge_time_hrs',
    }

    available = {k: v for k, v in wanted.items() if k in df_raw.columns}
    df = df_raw[list(available.keys())].rename(columns=available)

    # ── Cleaning ──────────────────────────────────────────────
    # Keep only ICE and hybrid vehicles (excludes pure EV MPGe rows)
    df = df[df['fuel_type'].isin([
        'Regular', 'Premium', 'Midgrade', 'Diesel',
        'Regular Gas or Electricity', 'Premium Gas or Electricity',
        'Regular Gas and Electricity', 'Premium Gas and Electricity'
    ])]

    df = df.dropna(subset=['city_mpg', 'highway_mpg', 'engine_displacement_l'])
    df = df[df['city_mpg'] > 0]
    df = df[df['highway_mpg'] > 0]
    df = df[df['engine_displacement_l'] > 0]

    # Filter to recent years (2010+) — older data has different standards
    df = df[df['year'] >= 2010]

    # Normalise text fields
    df['make']  = df['make'].str.strip().str.title()
    df['model'] = df['model'].str.strip()

    # Create a join key for merging with Cd data later
    df['make_model_key'] = (
        df['make'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True) +
        '_' +
        df['model'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    )

    # CO2 cleaning — EPA stores -1 for missing
    if 'co2_grams_per_mile' in df.columns:
        df['co2_grams_per_mile'] = df['co2_grams_per_mile'].replace(-1, None)

    out = './data/processed/cars_epa.csv'
    df.to_csv(out, index=False)

    log.info(f"EPA complete")
    log.info(f"  Rows       : {len(df):,}")
    log.info(f"  Makes      : {df['make'].nunique()}")
    log.info(f"  Year range : {int(df['year'].min())}–{int(df['year'].max())}")
    log.info(f"  Saved to   : {out}")
    return df


# ── Drag Coefficient (Cd) scrape ──────────────────────────────
def pull_cd():
    log.info("=" * 50)
    log.info("Drag Coefficient — Wikipedia scrape")
    log.info("=" * 50)

    url = 'https://en.wikipedia.org/wiki/Automobile_drag_coefficient'
    log.info(f"Fetching {url}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        log.error(f"Request failed: {e}")
        return

    soup = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})
    log.info(f"Found {len(tables)} wikitables")

    all_frames = []
    for i, table in enumerate(tables):
        try:
            df_t = pd.read_html(str(table))[0]
            # Flatten multi-level columns if present
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = [
                    ' '.join(str(c) for c in col).strip()
                    for col in df_t.columns
                ]
            df_t['source_table'] = i
            all_frames.append(df_t)
            log.info(f"  Table {i}: {df_t.shape[0]} rows, cols: {list(df_t.columns)[:6]}")
        except Exception as e:
            log.debug(f"  Table {i} parse failed: {e}")

    if not all_frames:
        log.error("No tables parsed.")
        return

    df = pd.concat(all_frames, ignore_index=True)

    # ── Normalise column names ────────────────────────────────
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'cd' in cl or 'drag' in cl or 'coefficient' in cl:
            col_map[col] = 'cd'
        elif 'make' in cl or 'manufacturer' in cl or 'brand' in cl:
            col_map[col] = 'make'
        elif 'model' in cl or 'car' in cl or 'vehicle' in cl:
            col_map[col] = 'model'
        elif 'year' in cl:
            col_map[col] = 'year'
        elif 'note' in cl or 'comment' in cl:
            col_map[col] = 'notes'

    df = df.rename(columns=col_map)

    # Keep only rows that have a Cd value
    if 'cd' not in df.columns:
        log.error("Could not find a Cd column — check Wikipedia table structure")
        return

    df = df[df['cd'].notna()]
    df['cd'] = pd.to_numeric(df['cd'], errors='coerce')
    df = df.dropna(subset=['cd'])
    df = df[(df['cd'] > 0.1) & (df['cd'] < 1.5)]  # sanity range

    # Normalise make/model if present
    for col in ['make', 'model']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Create join key
    if 'make' in df.columns and 'model' in df.columns:
        df['make_model_key'] = (
            df['make'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True) +
            '_' +
            df['model'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        )

    out = './data/processed/cars_cd.csv'
    df.to_csv(out, index=False)

    log.info(f"Cd scrape complete")
    log.info(f"  Rows     : {len(df)}")
    log.info(f"  Cd range : {df['cd'].min():.3f} – {df['cd'].max():.3f}")
    log.info(f"  Saved to : {out}")
    return df


# ── Merge EPA + Cd ─────────────────────────────────────────────
def merge_car_data():
    log.info("=" * 50)
    log.info("Merging EPA + Cd into unified cars dataset")
    log.info("=" * 50)

    epa_path = './data/processed/cars_epa.csv'
    cd_path  = './data/processed/cars_cd.csv'

    if not os.path.exists(epa_path):
        log.error("EPA file missing — run pull_epa() first")
        return
    if not os.path.exists(cd_path):
        log.error("Cd file missing — run pull_cd() first")
        return

    epa = pd.read_csv(epa_path)
    cd  = pd.read_csv(cd_path)

    if 'make_model_key' not in epa.columns or 'make_model_key' not in cd.columns:
        log.warning("make_model_key missing — skipping merge")
        return

    # Try exact key merge first
    cd_slim = cd[['make_model_key', 'cd']].drop_duplicates('make_model_key')
    merged  = epa.merge(cd_slim, on='make_model_key', how='left')

    match_rate = merged['cd'].notna().mean() * 100
    log.info(f"  Exact match rate: {match_rate:.1f}%")

    # Fuzzy fallback for unmatched rows using rapidfuzz if available
    unmatched = merged[merged['cd'].isna()]
    if len(unmatched) > 0:
        try:
            from rapidfuzz import process, fuzz
            cd_keys = cd_slim['make_model_key'].tolist()
            filled  = 0
            for idx, row in unmatched.iterrows():
                result = process.extractOne(
                    row['make_model_key'],
                    cd_keys,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=85
                )
                if result:
                    matched_cd = cd_slim.loc[
                        cd_slim['make_model_key'] == result[0], 'cd'
                    ].values
                    if len(matched_cd):
                        merged.at[idx, 'cd'] = matched_cd[0]
                        filled += 1
            log.info(f"  Fuzzy matched {filled} additional rows")
        except ImportError:
            log.info("  rapidfuzz not installed — skipping fuzzy match")
            log.info("  Install with: pip install rapidfuzz")

    out = './data/processed/cars_unified.csv'
    merged.to_csv(out, index=False)

    final_match = merged['cd'].notna().mean() * 100
    log.info(f"  Final Cd coverage: {final_match:.1f}% of rows")
    log.info(f"  Total rows: {len(merged):,}")
    log.info(f"  Saved to  : {out}")


# ── Entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    pull_epa()
    time.sleep(2)
    pull_cd()
    time.sleep(1)
    merge_car_data()
