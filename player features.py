"""
NFL Player Prop Feature Engineering
==========================
Builds two kinds of rolling features per player-game:
  1. The player's own trailing average for each stat (their recent form)
  2. The opponent's trailing average allowed for that stat (matchup strength)

Both are computed as trailing windows so no game leaks into its own features.
"""

import pandas as pd

ROLLING_WINDOW = 5  # games of recent form


def add_player_rolling_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, compute their trailing rolling average of each prop
    stat before each game.
    """
    df = player_stats.sort_values(["player_id", "season", "week"]).copy()

    stat_cols = ["passing_yards", "rushing_yards", "receiving_yards", "receptions",
                 "attempts", "carries", "targets"]
    stat_cols = [c for c in stat_cols if c in df.columns]

    for col in stat_cols:
        df[f"{col}_roll"] = (
            df.groupby("player_id")[col]
            .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).mean())
        )

    return df


def add_opponent_defense_allowed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes, for each team, the trailing average of each stat category
    allowed to opponents (i.e. defensive strength against that stat).
    Merged back onto each player-game row keyed by the player's opponent
    that week.
    """
    stat_cols = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
    stat_cols = [c for c in stat_cols if c in df.columns]

    # Sum each stat allowed by each team, per week (across all opposing players)
    allowed = (
        df.groupby(["opponent_team", "season", "week"])[stat_cols]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "team"})
        .sort_values(["team", "season", "week"])
    )

    for col in stat_cols:
        allowed[f"{col}_allowed_roll"] = (
            allowed.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).mean())
        )

    allowed_cols = ["team", "season", "week"] + [f"{c}_allowed_roll" for c in stat_cols]
    df = df.merge(
        allowed[allowed_cols],
        left_on=["opponent_team", "season", "week"],
        right_on=["team", "season", "week"],
        how="left",
    )
    df = df.drop(columns=["team"])
    return df


if __name__ == "__main__":
    player_stats = pd.read_parquet("nfl_player_weekly_stats.parquet")

    rolled = add_player_rolling_stats(player_stats)
    with_matchup = add_opponent_defense_allowed(rolled)

    with_matchup.to_parquet("nfl_player_features.parquet")
    print(f"Built player feature set: {len(with_matchup)} rows, {with_matchup.shape[1]} columns")
