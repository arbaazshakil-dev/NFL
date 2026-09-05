"""
NFL Data Fetching Module
==========================
Pulls play-by-play and schedule data via nfl_data_py (a Python wrapper
around the nflverse/nflfastR data repositories).

Install:
    pip install nfl_data_py pandas

Data source is free and public: https://github.com/nflverse/nflverse-data
No API key required.
"""

import pandas as pd
import nfl_data_py as nfl


def fetch_pbp(years: list[int]) -> pd.DataFrame:
    """
    Fetch play-by-play data for given seasons.
    Contains EPA, success rate, and situational data per play.
    """
    pbp = nfl.import_pbp_data(years, downcast=True)
    return pbp


def fetch_schedules(years: list[int]) -> pd.DataFrame:
    """
    Fetch game schedules and final scores (includes home/away, week, result).
    """
    sched = nfl.import_schedules(years)
    return sched


def fetch_team_stats(years: list[int]) -> pd.DataFrame:
    """
    Fetch weekly team-level aggregated stats (offense/defense).
    """
    team_stats = nfl.import_weekly_data(years)
    return team_stats


def fetch_injuries(years: list[int]) -> pd.DataFrame:
    """
    Fetch injury reports. Useful for QB-out / key-starter-out flags.
    """
    injuries = nfl.import_injuries(years)
    return injuries


def build_team_game_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse play-by-play into one row per team per game with:
    - offensive EPA/play
    - defensive EPA/play allowed
    - success rate
    - explosive play rate (plays gaining 20+ yards)
    """
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    off = (
        plays.groupby(["game_id", "posteam", "season", "week"])
        .agg(
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
            off_plays=("epa", "count"),
            off_explosive_rate=("yards_gained", lambda x: (x >= 20).mean()),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    deff = (
        plays.groupby(["game_id", "defteam", "season", "week"])
        .agg(
            def_epa_per_play=("epa", "mean"),
            def_success_rate=("success", "mean"),
            def_plays=("epa", "count"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    team_game = off.merge(deff, on=["game_id", "team", "season", "week"], how="outer")
    return team_game


if __name__ == "__main__":
    YEARS = list(range(2018, 2026))  # adjust to taste; more years = more training data

    print("Fetching play-by-play...")
    pbp = fetch_pbp(YEARS)

    print("Fetching schedules...")
    schedules = fetch_schedules(YEARS)

    print("Building team-game EPA table...")
    team_game_epa = build_team_game_epa(pbp)

    print("Saving to disk...")
    team_game_epa.to_parquet("nfl_team_game_epa.parquet")
    schedules.to_parquet("nfl_schedules.parquet")

    print(f"Done. {len(team_game_epa)} team-game rows, {len(schedules)} scheduled games.")
