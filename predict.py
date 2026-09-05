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

from odds_api import get_odds, get_player_props, american_to_implied_prob, remove_vig_two_way
from edge_detection import (
    evaluate_market,
    evaluate_upset,
    evaluate_scoring_fade,
    evaluate_prop_bet,
    calculate_implied_team_total,
    find_best_odds_across_books,
)
from train import predict_probabilities
from train_props import PROP_FEATURES

PROP_MARKET_TO_STAT = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
}

# nflverse schedule data uses team abbreviations (e.g. "KC"); The Odds API
# uses full names (e.g. "Kansas City Chiefs"). This maps abbreviation to
# full name so the two sources can be matched and merged correctly.
NFL_TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def load_props_model(path="nfl_props_model.pkl"):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def get_player_latest_features(player_features_df: pd.DataFrame, player_name: str, feature_cols: list[str]) -> dict | None:
    """
    Pulls a player's most recent rolling-feature row by display name.
    """
    rows = player_features_df[player_features_df["player_display_name"] == player_name]
    rows = rows.sort_values(["season", "week"])
    if rows.empty:
        return None
    latest = rows.iloc[-1]
    if latest[feature_cols].isnull().any():
        return None
    return latest[feature_cols].to_dict()


def run_player_props(props_bundle, player_features_df, event_id, home_team, away_team, game_label, api_key):
    """
    Fetches player prop odds for one game and evaluates each posted prop
    against the trained per-stat models. Returns a list of prop signal dicts
    for the dashboard, printing value bets as it goes.
    """
    results = []
    if props_bundle is None:
        return results

    try:
        event_odds = get_player_props("nfl", event_id, api_key)
    except Exception as e:
        print(f"  [props] could not fetch props for {game_label}: {e}")
        return results

    bookmakers = event_odds.get("bookmakers", [])
    if not bookmakers:
        return results

    primary_book = bookmakers[0]

    for market in primary_book.get("markets", []):
        stat = PROP_MARKET_TO_STAT.get(market["key"])
        if stat is None or stat not in props_bundle:
            continue

        # Group outcomes by player name to pair up Over/Under
        by_player = {}
        for outcome in market["outcomes"]:
            player_name = outcome.get("description")
            if not player_name:
                continue
            by_player.setdefault(player_name, {})[outcome["name"]] = outcome

        model_info = props_bundle[stat]
        feature_cols = model_info["features"]

        for player_name, sides in by_player.items():
            if "Over" not in sides or "Under" not in sides:
                continue

            player_feats = get_player_latest_features(player_features_df, player_name, feature_cols)
            if player_feats is None:
                continue

            feature_row = pd.DataFrame([player_feats])
            predicted_value = model_info["model"].predict(feature_row)[0]

            line = sides["Over"]["point"]
            over_odds = sides["Over"]["price"]
            under_odds = sides["Under"]["price"]

            signal = evaluate_prop_bet(
                player_name, stat, predicted_value, model_info["std"],
                line, over_odds, under_odds, primary_book["title"],
            )

            if signal.is_value_bet:
                print(f"  >>> PROP VALUE: {player_name} {stat} {signal.side} {line} (model predicts {signal.predicted_value}), edge={signal.edge}")

            results.append({
                "player": player_name,
                "stat": stat,
                "line": line,
                "predicted_value": signal.predicted_value,
                "side": signal.side,
                "edge": signal.edge,
                "is_value_bet": signal.is_value_bet,
                "odds": signal.odds,
                "sportsbook": primary_book["title"],
            })

    return results


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


def load_injury_report(path="nfl_injuries.parquet") -> pd.DataFrame | None:
    try:
        injuries = pd.read_parquet(path)
    except FileNotFoundError:
        return None
    if injuries.empty:
        return None
    return injuries


def get_team_injury_report(injuries: pd.DataFrame, team: str) -> list[dict]:
    """
    Pulls the most recent week's injury designations for a team, limited to
    players actually in question (Out, Doubtful, Questionable) — not the
    full roster. Column names vary by nflverse release, so this checks for
    the common variants defensively.
    """
    if injuries is None:
        return []

    team_col = next((c for c in ["team", "recent_team", "club_code"] if c in injuries.columns), None)
    status_col = next((c for c in ["report_status", "game_status"] if c in injuries.columns), None)
    name_col = next((c for c in ["full_name", "player_name", "player_display_name"] if c in injuries.columns), None)
    pos_col = "position" if "position" in injuries.columns else None
    week_col = "week" if "week" in injuries.columns else None

    if not all([team_col, status_col, name_col]):
        return []

    team_rows = injuries[injuries[team_col] == team]
    if week_col and not team_rows.empty:
        latest_week = team_rows[week_col].max()
        team_rows = team_rows[team_rows[week_col] == latest_week]

    of_note = team_rows[team_rows[status_col].isin(["Out", "Doubtful", "Questionable"])]

    report = []
    for _, row in of_note.iterrows():
        report.append({
            "player": row[name_col],
            "position": row[pos_col] if pos_col else None,
            "status": row[status_col],
        })
    return report


def get_upcoming_schedule(days_ahead: int = 9) -> list[dict]:
    """
    Pulls games from the schedule file that haven't been played yet and
    kick off within the next `days_ahead` days. This is known well before
    sportsbooks post odds, so it lets the dashboard show "who's playing"
    even when predictions aren't available yet.
    """
    try:
        schedules = pd.read_parquet("nfl_schedules.parquet")
    except FileNotFoundError:
        return []

    schedules = schedules.copy()
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    cutoff = now + pd.Timedelta(days=days_ahead)

    unplayed = schedules[schedules["home_score"].isna()]
    upcoming = unplayed[(unplayed["gameday"] >= now.normalize()) & (unplayed["gameday"] <= cutoff)]

    games = []
    for _, row in upcoming.sort_values("gameday").iterrows():
        home_abbr = row["home_team"]
        away_abbr = row["away_team"]
        games.append({
            "home_team": NFL_TEAM_NAMES.get(home_abbr, home_abbr),
            "away_team": NFL_TEAM_NAMES.get(away_abbr, away_abbr),
            "gameday": row["gameday"].isoformat(),
            "week": int(row["week"]) if pd.notna(row["week"]) else None,
        })
    return games


def run_predictions(api_key: str):
    bundle = load_model()
    margin_model = bundle["margin_model"]
    total_model = bundle["total_model"]
    margin_std = bundle["margin_std"]
    total_std = bundle["total_std"]
    feature_cols = bundle["feature_columns"]

    features_df = pd.read_parquet("nfl_game_features.parquet")

    props_bundle = load_props_model()
    injury_report_df = load_injury_report()
    try:
        player_features_df = pd.read_parquet("nfl_player_features.parquet")
    except FileNotFoundError:
        player_features_df = None
    if props_bundle is None or player_features_df is None:
        print("[props] props model or player features not found — skipping player props this run\n")

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
            "injury_report": {
                "home": get_team_injury_report(injury_report_df, home_team),
                "away": get_team_injury_report(injury_report_df, away_team),
            },
        }

        for side_label, team in [("home", home_team), ("away", away_team)]:
            for injured in game_entry["injury_report"][side_label]:
                print(f"  [injury] {team}: {injured['player']} ({injured['position']}) — {injured['status']}")

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

        if props_bundle is not None and player_features_df is not None:
            prop_signals = run_player_props(
                props_bundle, player_features_df, game["id"], home_team, away_team, game_label, api_key,
            )
            game_entry["player_props"] = prop_signals
        else:
            game_entry["player_props"] = []

        print()

    # Add any scheduled games that don't have odds posted yet, so the
    # dashboard shows the full upcoming slate rather than only games
    # sportsbooks have already priced.
    odds_matchups = {(g["home_team"], g["away_team"]) for g in dashboard_games}
    upcoming_schedule = get_upcoming_schedule()
    scheduled_pending = 0

    for sched_game in upcoming_schedule:
        pair = (sched_game["home_team"], sched_game["away_team"])
        if pair in odds_matchups:
            continue
        dashboard_games.append({
            "matchup": f"{sched_game['away_team']} @ {sched_game['home_team']}",
            "home_team": sched_game["home_team"],
            "away_team": sched_game["away_team"],
            "commence_time": sched_game["gameday"],
            "week": sched_game["week"],
            "odds_pending": True,
        })
        scheduled_pending += 1

    if scheduled_pending:
        print(f"Added {scheduled_pending} upcoming game(s) awaiting posted odds\n")

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
