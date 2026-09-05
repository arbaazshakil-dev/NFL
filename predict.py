"""
NFL Live Prediction Script
==========================
Ties the whole pipeline together for upcoming games:
  1. Loads the trained model (margin + total regressors)
  2. Pulls current team rolling-form features (from the latest saved features file)
  3. Pulls live odds for upcoming games from The Odds API
  4. Runs win/cover/total probabilities, edge detection, upset watch,
     and scoring fade for every upcoming game
  5. Prints a report (later: push to Supabase / send alerts)

Run this from inside the nfl/ folder, with nfl_model.pkl and
nfl_game_features.parquet already present (produced by train.py / features.py),
and shared/ on your Python path.
"""

import os
import json
from datetime import datetime, timezone
import joblib
import pandas as pd

from odds_api import get_odds, american_to_implied_prob, remove_vig_two_way
from edge_detection import (
    evaluate_market,
    evaluate_upset,
    evaluate_scoring_fade,
    calculate_implied_team_total,
    find_best_odds_across_books,
)
from train import predict_probabilities


def load_model(path="nfl_model.pkl"):
    return joblib.load(path)


def get_team_recent_form(features_df: pd.DataFrame, team: str) -> dict | None:
    """
    Pulls the most recent rolling-feature row for a team — this represents
    their current form heading into their next game.
    """
    team_rows = features_df[
        (features_df["home_team"] == team) | (features_df["away_team"] == team)
    ].sort_values("gameday")

    if team_rows.empty:
        return None

    return team_rows.iloc[-1].to_dict()


def get_team_recent_scores(features_df: pd.DataFrame, team: str, n_games: int = 5) -> list[float]:
    """
    Pulls a team's actual points scored over their last n games, for the
    scoring-fade check.
    """
    home_games = features_df[features_df["home_team"] == team][["gameday", "home_score"]]
    home_games = home_games.rename(columns={"home_score": "points"})

    away_games = features_df[features_df["away_team"] == team][["gameday", "away_score"]]
    away_games = away_games.rename(columns={"away_score": "points"})

    all_games = pd.concat([home_games, away_games]).sort_values("gameday")
    return all_games["points"].tail(n_games).tolist()


def build_feature_row(features_df: pd.DataFrame, home_team: str, away_team: str, feature_cols: list[str]) -> pd.DataFrame | None:
    """
    Builds a single-row feature dataframe for an UPCOMING game using each
    team's most recent rolling form. This mimics the diff_ features the
    model was trained on.
    """
    home_form = get_team_recent_form(features_df, home_team)
    away_form = get_team_recent_form(features_df, away_team)

    if home_form is None or away_form is None:
        return None

    row = {}
    for col in feature_cols:
        base = col.replace("diff_", "")
        home_col = f"home_{base}"
        away_col = f"away_{base}"
        # Use whichever side of the historical row corresponds to this team's
        # own rolling stat (home_form/away_form each store their own home_*/away_* cols
        # from whichever side they last played on).
        home_val = home_form.get(home_col, home_form.get(away_col))
        away_val = away_form.get(away_col, away_form.get(home_col))
        if home_val is None or away_val is None:
            return None
        row[col] = home_val - away_val

    return pd.DataFrame([row])


def run_predictions(api_key: str):
    bundle = load_model()
    margin_model = bundle["margin_model"]
    total_model = bundle["total_model"]
    margin_std = bundle["margin_std"]
    total_std = bundle["total_std"]
    feature_cols = bundle["feature_columns"]

    features_df = pd.read_parquet("nfl_game_features.parquet")

    print("Fetching live NFL odds...")
    odds_data = get_odds("nfl", api_key)
    print(f"Found {len(odds_data)} upcoming games\n")

    dashboard_games = []

    for game in odds_data:
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_label = f"{away_team} @ {home_team}"

        feature_row = build_feature_row(features_df, home_team, away_team, feature_cols)
        if feature_row is None:
            print(f"[skip] {game_label} — not enough recent form data yet")
            continue

        predicted_margin = margin_model.predict(feature_row)[0]
        predicted_total = total_model.predict(feature_row)[0]

        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            print(f"[skip] {game_label} — no odds posted yet")
            continue

        # Use the first bookmaker's spread/total lines to anchor probability
        # calculations (best-odds shopping happens separately below).
        primary_book = bookmakers[0]
        spread_line, total_line = None, None
        home_ml, away_ml = None, None

        for market in primary_book["markets"]:
            if market["key"] == "spreads":
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        spread_line = outcome["point"]
            elif market["key"] == "totals":
                total_line = market["outcomes"][0]["point"]
            elif market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        home_ml = outcome["price"]
                    elif outcome["name"] == away_team:
                        away_ml = outcome["price"]

        probs = predict_probabilities(
            predicted_margin, predicted_total, margin_std, total_std,
            spread_line=spread_line or 0.0, total_line=total_line,
        )

        print(f"=== {game_label} ===")
        print(f"  Predicted margin: {probs['predicted_margin']} (home perspective)")
        print(f"  Predicted total:  {probs['predicted_total']}")
        print(f"  Home win prob: {probs['home_win_prob']}  |  Away win prob: {probs['away_win_prob']}")

        game_entry = {
            "matchup": game_label,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": game.get("commence_time"),
            "predicted_margin": probs["predicted_margin"],
            "predicted_total": probs["predicted_total"],
            "home_win_prob": probs["home_win_prob"],
            "away_win_prob": probs["away_win_prob"],
            "spread_line": spread_line,
            "total_line": total_line,
            "sportsbook": primary_book["title"],
            "value_bet": None,
            "high_confidence": None,
            "upset_watch": None,
            "scoring_fades": [],
        }

        if home_ml is not None and away_ml is not None:
            signal = evaluate_market(
                game_label, "moneyline", home_team, probs["home_win_prob"],
                home_ml, away_ml, is_home_side=True, sportsbook=primary_book["title"],
            )
            if signal.is_value_bet:
                print(f"  >>> VALUE BET: {home_team} moneyline, edge={signal.edge}")
                game_entry["value_bet"] = {"side": home_team, "edge": signal.edge, "odds": signal.best_odds}
            if signal.is_high_confidence:
                print(f"  >>> HIGH CONFIDENCE: {home_team} win prob={signal.model_prob}")
                game_entry["high_confidence"] = {"side": home_team, "prob": signal.model_prob}

            upset = evaluate_upset(game_label, home_team, away_team, probs["home_win_prob"], home_ml, away_ml, primary_book["title"])
            if upset:
                print(f"  >>> UPSET WATCH: {upset.underdog} (+{upset.underdog_odds}) model gives {upset.model_underdog_win_prob} vs market {upset.market_underdog_implied_prob}")
                game_entry["upset_watch"] = {
                    "underdog": upset.underdog,
                    "odds": upset.underdog_odds,
                    "model_prob": upset.model_underdog_win_prob,
                    "market_prob": upset.market_underdog_implied_prob,
                }

        if spread_line is not None:
            for team, is_fav in [(home_team, spread_line < 0), (away_team, spread_line > 0)]:
                implied_total = calculate_implied_team_total(total_line, spread_line, is_fav)
                recent_scores = get_team_recent_scores(features_df, team)
                if len(recent_scores) >= 2:
                    fade = evaluate_scoring_fade(team, game_label, implied_total, recent_scores)
                    if fade.is_fading:
                        print(f"  >>> SCORING FADE: {team} implied {fade.implied_team_total} pts but averaging {fade.recent_scoring_avg} over last {fade.recent_games_used}")
                        game_entry["scoring_fades"].append({
                            "team": team,
                            "implied_total": fade.implied_team_total,
                            "recent_avg": fade.recent_scoring_avg,
                            "games_used": fade.recent_games_used,
                        })

        dashboard_games.append(game_entry)
        print()

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "games": dashboard_games,
    }
    with open("predictions.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(dashboard_games)} games to predictions.json")


if __name__ == "__main__":
    API_KEY = os.environ.get("ODDS_API_KEY")
    if not API_KEY:
        raise SystemExit("ODDS_API_KEY not set. Export it or add it to .env")
    run_predictions(API_KEY)
