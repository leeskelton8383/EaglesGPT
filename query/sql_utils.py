# sql_utils.py

import os
import pandas as pd
import numpy as np
import sqlite3

# -----------------------------------
# Config (kept exactly as-is)
# -----------------------------------

START_YEAR = 2020
END_YEAR   = 2025
TEAM = 'PHI'
HOME_FIELD_ADV = 0.0

# -----------------------------------
# 1. TEAM / GAME / SEASON PROCESSING
# -----------------------------------

def process_season(year, team, home_field_adv=0.0):
    # -----------------------------
    # Load PBP
    # -----------------------------
    url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz"
    pbp = pd.read_csv(url, compression='gzip', low_memory=False)

    pbp = pbp[pbp['season_type'].isin(['REG', 'POST'])]

    valid_play = (pbp['play_type'] != 'no_play')
    rush_mask = valid_play & (pbp['rush_attempt'] == 1)
    pass_mask = valid_play & (pbp['pass_attempt'] == 1)

    # -----------------------------
    # Base game table (Eagles)
    # -----------------------------
    game_scores = pbp[['game_id', 'week', 'home_team', 'away_team',
                       'home_score', 'away_score']].drop_duplicates()

    eagles_games = game_scores[
        (game_scores['home_team'] == team) |
        (game_scores['away_team'] == team)
    ].copy()

    eagles_games['is_home'] = eagles_games['home_team'] == team
    eagles_games['opponent'] = np.where(
        eagles_games['is_home'],
        eagles_games['away_team'],
        eagles_games['home_team']
    )

    eagles_games['points_scored'] = np.where(
        eagles_games['is_home'],
        eagles_games['home_score'],
        eagles_games['away_score']
    )
    eagles_games['points_allowed'] = np.where(
        eagles_games['is_home'],
        eagles_games['away_score'],
        eagles_games['home_score']
    )

    eagles_games['margin_of_victory'] = (
        eagles_games['points_scored'] - eagles_games['points_allowed']
    )
    eagles_games['win'] = eagles_games['margin_of_victory'] > 0

    eagles_games = eagles_games.sort_values('week').reset_index(drop=True)
    eagles_games['cumulative_wins'] = eagles_games['win'].cumsum()
    eagles_games['cumulative_win_pct'] = (
        eagles_games['cumulative_wins'] / (eagles_games.index + 1)
    )

    # --- EVERYTHING BELOW REMAINS UNCHANGED ---
    # (Turnovers, EPA, yards, red zone, SRS, league aggregates, z-scores, ranks,
    # column ordering, etc.)

    # ⛔ SNIPPED FOR BREVITY IN EXPLANATION
    # ⛔ KEEP YOUR FULL FUNCTION BODY EXACTLY AS YOU PROVIDED
    # ⛔ RETURN eagles_results[final_cols] AT THE END

    return eagles_results[final_cols]


# -----------------------------------
# 2. PLAYER STATS DOWNLOAD HELPERS
# -----------------------------------

from paths import DATA_DIR, PLAYERS_CSV, PLAYER_WEEKS_CSV

YEARS = range(2020, 2026)

URL_TEMPLATE_SEASON = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_regpost_{year}.csv.gz"
)

URL_TEMPLATE_WEEK = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{year}.csv.gz"
)

def download_and_concat(url_template, years, label):
    frames = []

    for year in years:
        url = url_template.format(year=year)
        print(f"Loading {label} {year} from {url} ...")

        try:
            df = pd.read_csv(url, compression="gzip", low_memory=False)

            if "season" not in df.columns:
                df["season"] = year
            else:
                df["season"] = df["season"].fillna(year)

            frames.append(df)
            print(f"  -> Loaded {df.shape[0]:,} rows")

        except Exception as e:
            print(f"  !! Failed to load {label} {year}: {e}")

    if not frames:
        raise RuntimeError(f"No {label} data files were loaded.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Saved WEEK CSV → {PLAYER_WEEKS_CSV}")
          





