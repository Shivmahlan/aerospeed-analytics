"""
AeroSpeed Analytics — Data Validator
======================================
Run this after collection to check data quality
before moving to EDA.

Usage:
  python validate_data.py
"""

import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

PROCESSED = './data/processed'

FILES = {
    'f1_race_laps'  : 'f1_race_laps.csv',
    'f1_telemetry'  : 'f1_telemetry.csv',
    'f1_qualifying' : 'f1_qualifying.csv',
    'cars_epa'      : 'cars_epa.csv',
    'cars_cd'       : 'cars_cd.csv',
    'cars_unified'  : 'cars_unified.csv',
}

EXPECTED_F1_CIRCUITS = 22  # minimum circuits per season
EXPECTED_F1_TEAMS    = 10
EXPECTED_YEARS       = [2022, 2023, 2024]


def check_file(name, filename):
    path = os.path.join(PROCESSED, filename)
    if not os.path.exists(path):
        log.error(f"[{name}] FILE MISSING: {path}")
        return None

    size_mb = os.path.getsize(path) / 1e6
    df = pd.read_csv(path, low_memory=False)
    log.info(f"\n[{name}]")
    log.info(f"  File size : {size_mb:.1f} MB")
    log.info(f"  Rows      : {len(df):,}")
    log.info(f"  Columns   : {list(df.columns)}")
    log.info(f"  Null rate : {df.isnull().mean().sort_values(ascending=False).head(5).to_dict()}")
    return df


def validate_f1_laps(df):
    if df is None:
        return
    log.info("\n  F1 Race Laps — Validation")
    years = sorted(df['Year'].unique()) if 'Year' in df.columns else []
    circuits = df['GP'].nunique() if 'GP' in df.columns else 0
    teams = df['Team'].nunique() if 'Team' in df.columns else 0

    log.info(f"  Years     : {years}")
    log.info(f"  Circuits  : {circuits}")
    log.info(f"  Teams     : {teams}")
    log.info(f"  Lap range : {df['LapTime'].min():.1f}s – {df['LapTime'].max():.1f}s")
    log.info(f"  Compounds : {df['Compound'].value_counts().to_dict() if 'Compound' in df.columns else 'N/A'}")

    if years != EXPECTED_YEARS:
        log.warning(f"  WARN: Expected years {EXPECTED_YEARS}, got {years}")
    if circuits < EXPECTED_F1_CIRCUITS:
        log.warning(f"  WARN: Only {circuits} circuits, expected {EXPECTED_F1_CIRCUITS}+")
    if teams < EXPECTED_F1_TEAMS:
        log.warning(f"  WARN: Only {teams} teams, expected {EXPECTED_F1_TEAMS}")


def validate_telemetry(df):
    if df is None:
        return
    log.info("\n  F1 Telemetry — Validation")
    log.info(f"  Circuits  : {df['GP'].unique() if 'GP' in df.columns else 'N/A'}")
    log.info(f"  Speed range: {df['Speed'].min():.0f} – {df['Speed'].max():.0f} km/h")
    has_xyz = all(c in df.columns for c in ['X', 'Y', 'Z'])
    xyz_coverage = df['X'].notna().mean() * 100 if 'X' in df.columns else 0
    log.info(f"  XYZ coords: {'YES' if has_xyz else 'NO'} ({xyz_coverage:.0f}% populated)")
    if df['Speed'].max() > 400:
        log.warning("  WARN: Speed values over 400 km/h — check for unit issues")


def validate_cars(df):
    if df is None:
        return
    log.info("\n  Cars EPA — Validation")
    log.info(f"  Year range : {int(df['year'].min())}–{int(df['year'].max())}")
    log.info(f"  Makes      : {df['make'].nunique()}")
    log.info(f"  MPG range  : city {df['city_mpg'].min():.0f}–{df['city_mpg'].max():.0f}, "
             f"hwy {df['highway_mpg'].min():.0f}–{df['highway_mpg'].max():.0f}")
    if 'cd' in df.columns:
        cd_cov = df['cd'].notna().mean() * 100
        log.info(f"  Cd coverage: {cd_cov:.1f}%")
        log.info(f"  Cd range   : {df['cd'].min():.3f}–{df['cd'].max():.3f}")


def print_summary(results):
    log.info("\n" + "=" * 50)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 50)
    for name, status in results.items():
        symbol = "OK" if status else "MISSING"
        log.info(f"  [{symbol}] {name}")
    log.info("")
    log.info("If all OK — you're ready for EDA.")
    log.info("If any MISSING — re-run the relevant scraper.")


if __name__ == '__main__':
    results = {}

    df_laps  = check_file('f1_race_laps',  FILES['f1_race_laps'])
    df_tel   = check_file('f1_telemetry',  FILES['f1_telemetry'])
    df_qual  = check_file('f1_qualifying', FILES['f1_qualifying'])
    df_epa   = check_file('cars_epa',      FILES['cars_epa'])
    df_cd    = check_file('cars_cd',       FILES['cars_cd'])
    df_uni   = check_file('cars_unified',  FILES['cars_unified'])

    validate_f1_laps(df_laps)
    validate_telemetry(df_tel)
    validate_cars(df_uni if df_uni is not None else df_epa)

    results = {
        'f1_race_laps'  : df_laps is not None,
        'f1_telemetry'  : df_tel is not None,
        'f1_qualifying' : df_qual is not None,
        'cars_epa'      : df_epa is not None,
        'cars_cd'       : df_cd is not None,
        'cars_unified'  : df_uni is not None,
    }

    print_summary(results)
