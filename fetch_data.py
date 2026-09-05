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

    Fetched one year at a time and any year that isn't available yet
    (e.g. the current season before its data file is published) is
    skipped rather than failing the whole run.
    """
    frames = []
    for year in years:
        try:
            frames.append(nfl.import_pbp_data([year], downcast=True))
        except Exception as e:
            print(f"  [skip] pbp {year}: not available yet ({e})")

    if not frames:
        raise RuntimeError("No play-by-play data could be fetched for any requested year.")

    return pd.concat(frames, ignore_index=True)


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
    from datetime import date
    CURRENT_YEAR = date.today().year
    YEARS = list(range(2018, CURRENT_YEAR + 1))  # always include the current season

    print("Fetching play-by-play...")
    pbp = fetch_pbp(YEARS)

    print("Fetching schedules...")
    schedules = fetch_schedules(YEARS)

    print("Building team-game EPA table...")
    team_game_epa = build_team_game_epa(pbp)

    # Injury reports are only fetched for the current season — that's the
    # only data relevant to "who's playing this week," and older seasons'
    # reports aren't needed for anything else in the pipeline.
    print("Fetching current-season injury reports...")
    current_year = YEARS[-1]
    try:
        injuries = fetch_injuries([current_year])
    except Exception as e:
        print(f"  [skip] injury data not available yet for {current_year}: {e}")
        injuries = pd.DataFrame()

    print("Saving to disk...")
    team_game_epa.to_parquet("nfl_team_game_epa.parquet")
    schedules.to_parquet("nfl_schedules.parquet")
    injuries.to_parquet("nfl_injuries.parquet")

    print(f"Done. {len(team_game_epa)} team-game rows, {len(schedules)} scheduled games, {len(injuries)} injury report rows.")
