"""
AeroSpeed Analytics — F1 Data Collector
========================================
Pulls 3 years (2022-2024) of F1 data in three tiers:
  Tier 1 — Race laps, all tracks       (~500MB cache)
  Tier 2 — Full telemetry, 6 circuits  (~1GB cache)
  Tier 3 — Qualifying, all tracks      (~300MB cache)

Usage:
  python scraper_f1.py --tier 1
  python scraper_f1.py --tier 2
  python scraper_f1.py --tier 3
  python scraper_f1.py --tier all
"""

import os
import time
import argparse
import logging
import pandas as pd
import fastf1

# ── Logging setup ──────────────────────────────────────────────
os.makedirs('./logs', exist_ok=True)
os.makedirs('./data/cache', exist_ok=True)
os.makedirs('./data/processed', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('./logs/scraper_f1.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

fastf1.Cache.enable_cache('./data/cache')

# ── Config ─────────────────────────────────────────────────────
YEARS = [2022, 2023, 2024]

# 6 circuits that represent full aero spectrum
TELEMETRY_CIRCUITS = [
    'Italian Grand Prix',       # Monza   — lowest downforce, highest speed
    'Monaco Grand Prix',        # Monaco  — highest downforce, lowest speed
    'British Grand Prix',       # Silverstone — balanced
    'Belgian Grand Prix',       # Spa     — mixed, long straights + Eau Rouge
    'Singapore Grand Prix',     # Singapore — street circuit, high downforce
    'Bahrain Grand Prix',       # Bahrain — desert, mixed aero, night race
]

LAP_COLUMNS = [
    'Driver', 'DriverNumber', 'Team', 'LapTime', 'LapNumber',
    'Compound', 'TyreLife', 'FreshTyre',
    'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
    'Sector1Time', 'Sector2Time', 'Sector3Time',
    'PitInTime', 'PitOutTime', 'TrackStatus', 'IsAccurate'
]

QUAL_COLUMNS = [
    'DriverNumber', 'Abbreviation', 'FullName',
    'TeamName', 'TeamColor', 'GridPosition', 'Q1', 'Q2', 'Q3'
]

TELEMETRY_COLUMNS = [
    'Time', 'RPM', 'Speed', 'nGear', 'Throttle',
    'Brake', 'DRS', 'Distance'
]

POSITION_COLUMNS = ['X', 'Y', 'Z']


# ── Helpers ────────────────────────────────────────────────────
def safe_load(year, gp, session_type, telemetry=False):
    """Load a session with error handling. Returns None on failure."""
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(
            telemetry=telemetry,
            laps=True,
            weather=False,
            messages=False
        )
        return session
    except Exception as e:
        log.warning(f"  Could not load {year} {gp} {session_type}: {e}")
        return None


def timedelta_to_seconds(td_series):
    """Convert Timedelta series to float seconds, nulls stay null."""
    return pd.to_timedelta(td_series).dt.total_seconds()


def get_schedule(year):
    """Get race schedule excluding testing events."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        # Filter out pre-season testing if it snuck in
        schedule = schedule[schedule['EventFormat'] != 'testing']
        return schedule
    except Exception as e:
        log.error(f"Could not get {year} schedule: {e}")
        return pd.DataFrame()


# ── Tier 1: Race laps, all tracks ──────────────────────────────
def pull_race_laps():
    log.info("=" * 50)
    log.info("TIER 1 — Race laps, all tracks, 2022-2024")
    log.info("=" * 50)

    all_laps = []
    failed = []

    for year in YEARS:
        schedule = get_schedule(year)
        if schedule.empty:
            continue

        total = len(schedule)
        for idx, (_, event) in enumerate(schedule.iterrows(), 1):
            gp = event['EventName']
            location = event.get('Location', '')
            log.info(f"[{year}] ({idx}/{total}) {gp}")

            session = safe_load(year, gp, 'R', telemetry=False)
            if session is None:
                failed.append(f"{year} {gp} R")
                continue

            try:
                # Pick available lap columns only
                available = [c for c in LAP_COLUMNS if c in session.laps.columns]
                laps = session.laps[available].copy()

                # Convert timedeltas to seconds for portability
                for col in ['LapTime', 'Sector1Time', 'Sector2Time',
                            'Sector3Time', 'PitInTime', 'PitOutTime']:
                    if col in laps.columns:
                        laps[col] = timedelta_to_seconds(laps[col])

                laps['GP']       = gp
                laps['Location'] = location
                laps['Year']     = year
                laps['Round']    = event.get('RoundNumber', idx)

                # Derive useful flags
                laps['HasPitStop'] = laps['PitInTime'].notna()

                all_laps.append(laps)
                log.info(f"  OK — {len(laps)} laps")

            except Exception as e:
                log.warning(f"  Parse error: {e}")
                failed.append(f"{year} {gp} R")

            time.sleep(1.5)

    if not all_laps:
        log.error("No lap data collected.")
        return

    df = pd.concat(all_laps, ignore_index=True)

    # Basic cleaning
    df = df[df['LapTime'].notna()]
    df = df[df['LapTime'] > 60]   # remove sub-60s ghost laps
    df = df[df['LapTime'] < 300]  # remove laps over 5 min (safety car etc.)

    out = './data/processed/f1_race_laps.csv'
    df.to_csv(out, index=False)

    log.info(f"\nTier 1 complete")
    log.info(f"  Rows    : {len(df):,}")
    log.info(f"  Seasons : {sorted(df['Year'].unique())}")
    log.info(f"  Circuits: {df['GP'].nunique()}")
    log.info(f"  Saved to: {out}")

    if failed:
        log.warning(f"  Failed sessions ({len(failed)}): {failed}")


# ── Tier 2: Full telemetry, archetype circuits ──────────────────
def pull_telemetry():
    log.info("=" * 50)
    log.info("TIER 2 — Full telemetry, 6 archetype circuits")
    log.info("=" * 50)

    all_tel = []
    failed = []

    for year in YEARS:
        for gp_name in TELEMETRY_CIRCUITS:
            log.info(f"[{year}] {gp_name}")

            session = safe_load(year, gp_name, 'R', telemetry=True)
            if session is None:
                failed.append(f"{year} {gp_name}")
                continue

            drivers_pulled = 0
            for driver in session.drivers:
                try:
                    drv_laps = session.laps.pick_driver(driver)
                    if drv_laps.empty:
                        continue

                    fastest = drv_laps.pick_fastest()
                    if fastest is None or fastest.empty:
                        continue

                    # Car telemetry
                    tel = fastest.get_car_data().add_distance()
                    available_tel = [c for c in TELEMETRY_COLUMNS
                                     if c in tel.columns]
                    tel = tel[available_tel].copy()

                    # Position data (X, Y, Z)
                    try:
                        pos = fastest.get_pos_data()
                        pos_clean = pos[
                            [c for c in POSITION_COLUMNS if c in pos.columns]
                        ].copy()
                        # Align by index
                        tel = tel.join(pos_clean, how='left')
                    except Exception:
                        tel['X'] = None
                        tel['Y'] = None
                        tel['Z'] = None

                    # Metadata
                    tel['Driver']   = driver
                    tel['Team']     = fastest.get('Team', 'Unknown')
                    tel['GP']       = gp_name
                    tel['Year']     = year
                    tel['LapTime']  = timedelta_to_seconds(
                                          pd.Series([fastest['LapTime']])
                                      ).iloc[0]
                    tel['Compound'] = fastest.get('Compound', 'Unknown')

                    # Convert Time column
                    if 'Time' in tel.columns:
                        tel['Time'] = tel['Time'].dt.total_seconds()

                    all_tel.append(tel)
                    drivers_pulled += 1

                except Exception as e:
                    log.debug(f"  Driver {driver} failed: {e}")
                    continue

            log.info(f"  OK — {drivers_pulled} drivers")
            time.sleep(3)

    if not all_tel:
        log.error("No telemetry data collected.")
        return

    df = pd.concat(all_tel, ignore_index=True)
    df = df[df['Speed'].notna()]
    df = df[df['Speed'] > 0]

    out = './data/processed/f1_telemetry.csv'
    df.to_csv(out, index=False)

    log.info(f"\nTier 2 complete")
    log.info(f"  Rows    : {len(df):,}")
    log.info(f"  Circuits: {df['GP'].nunique()}")
    log.info(f"  Saved to: {out}")

    if failed:
        log.warning(f"  Failed sessions ({len(failed)}): {failed}")


# ── Tier 3: Qualifying, all tracks ─────────────────────────────
def pull_qualifying():
    log.info("=" * 50)
    log.info("TIER 3 — Qualifying, all tracks, 2022-2024")
    log.info("=" * 50)

    all_qual = []
    failed = []

    for year in YEARS:
        schedule = get_schedule(year)
        if schedule.empty:
            continue

        total = len(schedule)
        for idx, (_, event) in enumerate(schedule.iterrows(), 1):
            gp = event['EventName']
            location = event.get('Location', '')
            log.info(f"[{year}] ({idx}/{total}) {gp}")

            session = safe_load(year, gp, 'Q', telemetry=False)
            if session is None:
                failed.append(f"{year} {gp} Q")
                continue

            try:
                available = [c for c in QUAL_COLUMNS
                             if c in session.results.columns]
                results = session.results[available].copy()

                # Convert Q time timedeltas to seconds
                for col in ['Q1', 'Q2', 'Q3']:
                    if col in results.columns:
                        results[col] = timedelta_to_seconds(results[col])

                # Gap to pole (Q3 pole time = min Q3)
                if 'Q3' in results.columns:
                    pole_time = results['Q3'].min()
                    results['GapToPole_Q3'] = results['Q3'] - pole_time

                results['GP']       = gp
                results['Location'] = location
                results['Year']     = year
                results['Round']    = event.get('RoundNumber', idx)

                all_qual.append(results)
                log.info(f"  OK — {len(results)} drivers")

            except Exception as e:
                log.warning(f"  Parse error: {e}")
                failed.append(f"{year} {gp} Q")

            time.sleep(1.5)

    if not all_qual:
        log.error("No qualifying data collected.")
        return

    df = pd.concat(all_qual, ignore_index=True)

    out = './data/processed/f1_qualifying.csv'
    df.to_csv(out, index=False)

    log.info(f"\nTier 3 complete")
    log.info(f"  Rows    : {len(df):,}")
    log.info(f"  Seasons : {sorted(df['Year'].unique())}")
    log.info(f"  Saved to: {out}")

    if failed:
        log.warning(f"  Failed sessions ({len(failed)}): {failed}")


# ── Entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AeroSpeed F1 Data Collector')
    parser.add_argument(
        '--tier',
        choices=['1', '2', '3', 'all'],
        required=True,
        help='Which tier to pull: 1=race laps, 2=telemetry, 3=qualifying, all=everything'
    )
    args = parser.parse_args()

    start = time.time()

    if args.tier in ('1', 'all'):
        pull_race_laps()

    if args.tier in ('2', 'all'):
        pull_telemetry()

    if args.tier in ('3', 'all'):
        pull_qualifying()

    elapsed = time.time() - start
    log.info(f"\nTotal time: {elapsed/60:.1f} minutes")
