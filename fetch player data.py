"""
NFL Player Data Fetching Module
==========================
Pulls player-level weekly stats via nfl_data_py — the same free source as
the team-level pipeline, but at player granularity. This is what player
props (passing yards, rushing yards, receiving yards, receptions) get
trained on.
"""

import pandas as pd
import nfl_data_py as nfl


PROP_STAT_COLUMNS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
}

# Which position group each prop type is relevant for — used later to avoid
# e.g. trying to predict a lineman's "receiving yards."
PROP_POSITIONS = {
    "passing_yards": ["QB"],
    "rushing_yards": ["RB", "QB"],
    "receiving_yards": ["WR", "TE", "RB"],
    "receptions": ["WR", "TE", "RB"],
}


def fetch_player_weekly_stats(years: list[int]) -> pd.DataFrame:
    """
    Fetch weekly player stats (one row per player per game) including
    passing/rushing/receiving totals, position, team, and opponent.

    Some recent seasons may not have this data file published yet on
    nflverse even if play-by-play data exists, so years are fetched one
    at a time and any that 404 are skipped rather than failing the whole run.
    """
    frames = []
    for year in years:
        try:
            frames.append(nfl.import_weekly_data([year]))
        except Exception as e:
            print(f"  [skip] {year}: not available yet ({e})")

    if not frames:
        raise RuntimeError("No player weekly data could be fetched for any requested year.")

    weekly = pd.concat(frames, ignore_index=True)

    keep_cols = [
        "player_id", "player_display_name", "position", "recent_team",
        "opponent_team", "season", "week",
        "passing_yards", "rushing_yards", "receiving_yards", "receptions",
        "attempts", "carries", "targets",
    ]
    available = [c for c in keep_cols if c in weekly.columns]
    return weekly[available].copy()


if __name__ == "__main__":
    YEARS = list(range(2018, 2026))

    print("Fetching player weekly stats...")
    player_stats = fetch_player_weekly_stats(YEARS)

    player_stats.to_parquet("nfl_player_weekly_stats.parquet")
    print(f"Done. {len(player_stats)} player-game rows.")
