"""
NFL Feature Engineering
==========================
Turns per-game team EPA stats into rolling-window features, then
joins home vs. away into a single row per game with differential
features (home - away), which is what the model trains on.
"""

import pandas as pd
import numpy as np


ROLLING_WINDOW = 8  # games of recent form to average over


def add_rolling_features(team_game_epa: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute a trailing rolling average of their EPA/success
    metrics BEFORE each game (so we never leak the current game's result
    into its own features).
    """
    df = team_game_epa.sort_values(["team", "season", "week"]).copy()

    roll_cols = [
        "off_epa_per_play",
        "off_success_rate",
        "off_explosive_rate",
        "def_epa_per_play",
        "def_success_rate",
    ]

    for col in roll_cols:
        df[f"{col}_roll"] = (
            df.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).mean())
        )

    return df


def build_game_level_dataset(rolled: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rolling team form into home/away rows, then pivot into one row
    per game with home_* and away_* columns plus the actual result
    (margin, total) as training targets.
    """
    home = rolled.merge(
        schedules[["game_id", "home_team", "away_team", "home_score", "away_score", "gameday"]],
        left_on=["game_id", "team"],
        right_on=["game_id", "home_team"],
    )
    away = rolled.merge(
        schedules[["game_id", "home_team", "away_team", "home_score", "away_score", "gameday"]],
        left_on=["game_id", "team"],
        right_on=["game_id", "away_team"],
    )

    feature_cols = [c for c in rolled.columns if c.endswith("_roll")]

    home_feats = home[["game_id"] + feature_cols].add_prefix("home_").rename(
        columns={"home_game_id": "game_id"}
    )
    away_feats = away[["game_id"] + feature_cols].add_prefix("away_").rename(
        columns={"away_game_id": "game_id"}
    )

    game_df = home_feats.merge(away_feats, on="game_id")
    game_df = game_df.merge(
        schedules[["game_id", "home_team", "away_team", "home_score", "away_score", "gameday"]],
        on="game_id",
    )

    # Differential features (home minus away) — these tend to matter most
    for col in feature_cols:
        game_df[f"diff_{col}"] = game_df[f"home_{col}"] - game_df[f"away_{col}"]

    # Targets
    game_df["margin"] = game_df["home_score"] - game_df["away_score"]  # positive = home won by X
    game_df["total"] = game_df["home_score"] + game_df["away_score"]

    # Drop rows without enough history to have rolling features (early season games)
    game_df = game_df.dropna(subset=[f"diff_{feature_cols[0]}"])

    return game_df


if __name__ == "__main__":
    team_game_epa = pd.read_parquet("nfl_team_game_epa.parquet")
    schedules = pd.read_parquet("nfl_schedules.parquet")

    rolled = add_rolling_features(team_game_epa)
    game_dataset = build_game_level_dataset(rolled, schedules)

    game_dataset.to_parquet("nfl_game_features.parquet")
    print(f"Built feature set: {len(game_dataset)} games, {game_dataset.shape[1]} columns")
