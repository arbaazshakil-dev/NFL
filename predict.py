"""
NFL Live Prediction Script
==========================
Ties the whole pipeline together for upcoming games:
  1. Loads the trained model (margin + total regressors)
  2. Pulls current team rolling-form features (from the latest saved features file)
  3. Pulls live odds for upcoming games from The Odds API
  4. Runs win/cover/total probabilities, edge detection, upset watch,
     and scoring fade for every upcoming game
  5. Writes predictions.json for the dashboard

Changes in this version:
  - Logs are line-buffered, so tracebacks appear in the right place in
    GitHub Actions instead of above the output.
  - Every game is processed inside its own try/except. One bad game (or one
    bad prop market) no longer kills the run; predictions.json always gets
    written, and failures are listed in its "errors" field.
  - Guards against missing total/spread lines before the scoring-fade math.
  - QB-out check: the model doesn't know about injuries, so any value bet,
    high-confidence, or upset signal backing a team with a QB listed Out is
    suppressed and flagged instead of alerted.

Run this from inside the nfl/ folder, with nfl_model.pkl and
nfl_game_features.parquet already present (produced by train.py / features.py),
and shared/ on your Python path.
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

# Make print() output appear immediately in GitHub Actions logs, in order
# with any tracebacks (stdout is otherwise block-buffered when piped).
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

from odds_api import get_odds, get_player_props
from edge_detection import (
    evaluate_market,
    evaluate_upset,
    evaluate_scoring_fade,
    evaluate_prop_bet,
    calculate_implied_team_total,
)
from train import predict_probabilities

# Set to False if you want QB-out signals alerted anyway (still flagged).
SUPPRESS_SIGNALS_WHEN_QB_OUT = True

# Odds API plan allows 10 requests/minute. One props request is made per
# game, so wait between them to stay under the cap (7s ≈ 8-9 requests/min).
PROPS_REQUEST_DELAY_SECONDS = 7


class NumpyEncoder(json.JSONEncoder):
    """Converts NumPy scalar/array types to native Python types for json."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def to_float(x, ndigits=3):
    """Cast NumPy floats to rounded Python floats (fixes 43.79999923706055)."""
    if x is None:
        return None
    return round(float(x), ndigits)


PROP_MARKET_TO_STAT = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
}

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
NFL_TEAM_ABBR = {v: k for k, v in NFL_TEAM_NAMES.items()}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_model(path="nfl_model.pkl"):
    return joblib.load(path)


def load_props_model(path="nfl_props_model.pkl"):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def load_parquet_optional(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        return None
    return None if df.empty else df


# ---------------------------------------------------------------------------
# Team features / scores
# ---------------------------------------------------------------------------

def get_team_recent_form(features_df: pd.DataFrame, team_abbr: str) -> dict | None:
    team_rows = features_df[
        (features_df["home_team"] == team_abbr) | (features_df["away_team"] == team_abbr)
    ].sort_values("gameday")
    if team_rows.empty:
        return None
    return team_rows.iloc[-1].to_dict()


def get_team_recent_scores(features_df: pd.DataFrame, team_abbr: str, n_games: int = 5) -> list[float]:
    home = features_df[features_df["home_team"] == team_abbr][["gameday", "home_score"]]
    home = home.rename(columns={"home_score": "points"})
    away = features_df[features_df["away_team"] == team_abbr][["gameday", "away_score"]]
    away = away.rename(columns={"away_score": "points"})
    all_games = pd.concat([home, away]).dropna(subset=["points"]).sort_values("gameday")
    return [float(p) for p in all_games["points"].tail(n_games)]


def build_feature_row(features_df, home_abbr, away_abbr, feature_cols) -> pd.DataFrame | None:
    home_form = get_team_recent_form(features_df, home_abbr)
    away_form = get_team_recent_form(features_df, away_abbr)
    if home_form is None or away_form is None:
        return None

    row = {}
    for col in feature_cols:
        base = col.replace("diff_", "")
        home_col, away_col = f"home_{base}", f"away_{base}"
        home_val = home_form.get(home_col, home_form.get(away_col))
        away_val = away_form.get(away_col, away_form.get(home_col))
        if home_val is None or away_val is None or pd.isna(home_val) or pd.isna(away_val):
            return None
        row[col] = home_val - away_val

    return pd.DataFrame([row], columns=feature_cols)


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------

def get_team_injury_report(injuries: pd.DataFrame | None, team_abbr: str) -> list[dict]:
    if injuries is None:
        return []

    team_col = next((c for c in ["team", "recent_team", "club_code"] if c in injuries.columns), None)
    status_col = next((c for c in ["report_status", "game_status"] if c in injuries.columns), None)
    name_col = next((c for c in ["full_name", "player_name", "player_display_name"] if c in injuries.columns), None)
    pos_col = "position" if "position" in injuries.columns else None
    week_col = "week" if "week" in injuries.columns else None

    if not all([team_col, status_col, name_col]):
        return []

    team_rows = injuries[injuries[team_col] == team_abbr]
    if week_col and not team_rows.empty:
        team_rows = team_rows[team_rows[week_col] == team_rows[week_col].max()]

    of_note = team_rows[team_rows[status_col].isin(["Out", "Doubtful", "Questionable"])]
    return [
        {
            "player": row[name_col],
            "position": row[pos_col] if pos_col else None,
            "status": row[status_col],
        }
        for _, row in of_note.iterrows()
    ]


def qb_out_players(injury_list: list[dict]) -> list[str]:
    """QBs listed Out or Doubtful. The model's features don't include this."""
    return [
        p["player"] for p in injury_list
        if p.get("position") == "QB" and p.get("status") in ("Out", "Doubtful")
    ]


# ---------------------------------------------------------------------------
# Player props
# ---------------------------------------------------------------------------

def get_player_latest_features(player_features_df, player_name, feature_cols) -> dict | None:
    if "player_display_name" not in player_features_df.columns:
        return None
    missing = [c for c in feature_cols if c not in player_features_df.columns]
    if missing:
        print(f"    [props] player features missing columns: {missing}")
        return None
    rows = player_features_df[player_features_df["player_display_name"] == player_name]
    if rows.empty:
        return None
    latest = rows.sort_values(["season", "week"]).iloc[-1]
    if latest[feature_cols].isnull().any():
        return None
    return latest[feature_cols].to_dict()


def get_player_recent_stat_history(weekly_df, player_name, stat_col, n_games=5) -> list[float]:
    if weekly_df is None or stat_col not in weekly_df.columns:
        return []
    rows = weekly_df[weekly_df["player_display_name"] == player_name].sort_values(["season", "week"])
    return [float(v) for v in rows[stat_col].dropna().tail(n_games)]


def run_player_props(props_bundle, player_features_df, player_weekly_df, event_id, game_label, api_key, errors):
    results = []
    if props_bundle is None or player_features_df is None:
        return results

    # Rate limit: pause before every props request (the main odds call has
    # already used one request this minute).
    time.sleep(PROPS_REQUEST_DELAY_SECONDS)

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
        stat = PROP_MARKET_TO_STAT.get(market.get("key"))
        if stat is None or stat not in props_bundle:
            continue

        by_player = {}
        for outcome in market.get("outcomes", []):
            name = outcome.get("description")
            if name:
                by_player.setdefault(name, {})[outcome.get("name")] = outcome

        model_info = props_bundle[stat]
        feature_cols = model_info["features"]

        for player_name, sides in by_player.items():
            # Each player is isolated: one bad row can't kill the game or run.
            try:
                if "Over" not in sides or "Under" not in sides:
                    continue
                feats = get_player_latest_features(player_features_df, player_name, feature_cols)
                if feats is None:
                    continue

                feature_row = pd.DataFrame([feats], columns=feature_cols)
                predicted_value = model_info["model"].predict(feature_row)[0]

                line = sides["Over"].get("point")
                if line is None:
                    continue

                signal = evaluate_prop_bet(
                    player_name, stat, predicted_value, model_info["std"],
                    line, sides["Over"]["price"], sides["Under"]["price"], primary_book["title"],
                )

                if signal.is_value_bet:
                    print(f"  >>> PROP VALUE: {player_name} {stat} {signal.side} {line} "
                          f"(model {to_float(signal.predicted_value, 1)}), edge={to_float(signal.edge)}")

                results.append({
                    "player": player_name,
                    "stat": stat,
                    "line": line,
                    "predicted_value": to_float(signal.predicted_value, 1),
                    "side": signal.side,
                    "edge": to_float(signal.edge),
                    "is_value_bet": bool(signal.is_value_bet),
                    "odds": signal.odds,
                    "sportsbook": primary_book["title"],
                    "recent_games": get_player_recent_stat_history(player_weekly_df, player_name, stat),
                })
            except Exception as e:
                msg = f"props {game_label} / {player_name} / {stat}: {e!r}"
                print(f"  [error] {msg}")
                traceback.print_exc()
                errors.append(msg)

    return results


# ---------------------------------------------------------------------------
# Schedule (games without odds yet)
# ---------------------------------------------------------------------------

def get_upcoming_schedule(days_ahead: int = 9) -> list[dict]:
    schedules = load_parquet_optional("nfl_schedules.parquet")
    if schedules is None:
        print("  [schedule] nfl_schedules.parquet not found")
        return []

    schedules = schedules.copy()
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])
    if schedules["gameday"].dt.tz is not None:
        schedules["gameday"] = schedules["gameday"].dt.tz_localize(None)

    now = pd.Timestamp.now().normalize()
    cutoff = now + pd.Timedelta(days=days_ahead)
    unplayed = schedules[schedules["home_score"].isna()]
    upcoming = unplayed[(unplayed["gameday"] >= now) & (unplayed["gameday"] <= cutoff)]

    print(f"  [schedule] {len(schedules)} total, {len(unplayed)} unplayed, "
          f"{len(upcoming)} within {days_ahead} days ({now.date()} to {cutoff.date()})")

    return [
        {
            "home_team": NFL_TEAM_NAMES.get(r["home_team"], r["home_team"]),
            "away_team": NFL_TEAM_NAMES.get(r["away_team"], r["away_team"]),
            "gameday": r["gameday"].isoformat(),
            "week": int(r["week"]) if pd.notna(r["week"]) else None,
        }
        for _, r in upcoming.sort_values("gameday").iterrows()
    ]


# ---------------------------------------------------------------------------
# Per-game processing
# ---------------------------------------------------------------------------

def parse_primary_lines(primary_book, home_team, away_team):
    spread_line = total_line = home_ml = away_ml = None
    for market in primary_book.get("markets", []):
        key = market.get("key")
        outcomes = market.get("outcomes", [])
        if key == "spreads":
            for o in outcomes:
                if o.get("name") == home_team:
                    spread_line = o.get("point")
        elif key == "totals" and outcomes:
            total_line = outcomes[0].get("point")
        elif key == "h2h":
            for o in outcomes:
                if o.get("name") == home_team:
                    home_ml = o.get("price")
                elif o.get("name") == away_team:
                    away_ml = o.get("price")
    return spread_line, total_line, home_ml, away_ml


def process_game(game, ctx, errors) -> dict | None:
    home_team, away_team = game["home_team"], game["away_team"]
    home_abbr = NFL_TEAM_ABBR.get(home_team, home_team)
    away_abbr = NFL_TEAM_ABBR.get(away_team, away_team)
    game_label = f"{away_team} @ {home_team}"
    features_df = ctx["features_df"]

    feature_row = build_feature_row(features_df, home_abbr, away_abbr, ctx["feature_cols"])
    if feature_row is None:
        print(f"[skip] {game_label} — not enough recent form data yet")
        return None

    bookmakers = game.get("bookmakers", [])
    if not bookmakers:
        print(f"[skip] {game_label} — no odds posted yet")
        return None
    primary_book = bookmakers[0]

    predicted_margin = ctx["margin_model"].predict(feature_row)[0]
    predicted_total = ctx["total_model"].predict(feature_row)[0]
    spread_line, total_line, home_ml, away_ml = parse_primary_lines(primary_book, home_team, away_team)

    probs = predict_probabilities(
        predicted_margin, predicted_total, ctx["margin_std"], ctx["total_std"],
        spread_line=spread_line or 0.0, total_line=total_line,
    )

    print(f"=== {game_label} ===")
    print(f"  Predicted margin: {to_float(probs['predicted_margin'], 1)} (home perspective)")
    print(f"  Predicted total:  {to_float(probs['predicted_total'], 1)}")
    print(f"  Home win prob: {to_float(probs['home_win_prob'])}  |  Away win prob: {to_float(probs['away_win_prob'])}")

    injuries = {
        "home": get_team_injury_report(ctx["injuries_df"], home_abbr),
        "away": get_team_injury_report(ctx["injuries_df"], away_abbr),
    }
    qb_out = {
        home_team: qb_out_players(injuries["home"]),
        away_team: qb_out_players(injuries["away"]),
    }

    entry = {
        "matchup": game_label,
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": game.get("commence_time"),
        "predicted_margin": to_float(probs["predicted_margin"], 1),
        "predicted_total": to_float(probs["predicted_total"], 1),
        "home_win_prob": to_float(probs["home_win_prob"]),
        "away_win_prob": to_float(probs["away_win_prob"]),
        "spread_line": spread_line,
        "total_line": total_line,
        "sportsbook": primary_book.get("title"),
        "value_bet": None,
        "high_confidence": None,
        "upset_watch": None,
        "scoring_fades": [],
        "injury_report": injuries,
        "qb_out": {team: names for team, names in qb_out.items() if names},
        "suppressed_signals": [],
        "player_props": [],
    }

    for side, team in [("home", home_team), ("away", away_team)]:
        for inj in injuries[side]:
            print(f"  [injury] {team}: {inj['player']} ({inj['position']}) — {inj['status']}")

    def backing_blocked(team: str, signal_name: str) -> bool:
        """True if this signal backs a team whose QB is out and we're suppressing."""
        if qb_out.get(team):
            note = f"{signal_name} on {team} (QB out: {', '.join(qb_out[team])})"
            print(f"  [qb-out] {'suppressed' if SUPPRESS_SIGNALS_WHEN_QB_OUT else 'flagged'}: {note}")
            entry["suppressed_signals"].append(note)
            return SUPPRESS_SIGNALS_WHEN_QB_OUT
        return False

    # --- Moneyline value / high confidence / upset ---
    if home_ml is not None and away_ml is not None:
        signal = evaluate_market(
            game_label, "moneyline", home_team, probs["home_win_prob"],
            home_ml, away_ml, is_home_side=True, sportsbook=primary_book.get("title"),
        )
        if signal.is_value_bet and not backing_blocked(home_team, "value bet"):
            print(f"  >>> VALUE BET: {home_team} moneyline, edge={to_float(signal.edge)}")
            entry["value_bet"] = {"side": home_team, "edge": to_float(signal.edge), "odds": signal.best_odds}
        if signal.is_high_confidence and not backing_blocked(home_team, "high confidence"):
            print(f"  >>> HIGH CONFIDENCE: {home_team} win prob={to_float(signal.model_prob)}")
            entry["high_confidence"] = {"side": home_team, "prob": to_float(signal.model_prob)}

        upset = evaluate_upset(game_label, home_team, away_team, probs["home_win_prob"],
                               home_ml, away_ml, primary_book.get("title"))
        if upset and not backing_blocked(upset.underdog, "upset watch"):
            print(f"  >>> UPSET WATCH: {upset.underdog} (+{upset.underdog_odds}) model gives "
                  f"{to_float(upset.model_underdog_win_prob)} vs market {to_float(upset.market_underdog_implied_prob)}")
            dog_home = upset.underdog == home_team
            entry["upset_watch"] = {
                "underdog": upset.underdog,
                "odds": upset.underdog_odds,
                "model_prob": to_float(upset.model_underdog_win_prob),
                "market_prob": to_float(upset.market_underdog_implied_prob),
                "favorite": away_team if dog_home else home_team,
                "underdog_recent_scores": get_team_recent_scores(features_df, home_abbr if dog_home else away_abbr),
                "favorite_recent_scores": get_team_recent_scores(features_df, away_abbr if dog_home else home_abbr),
            }

    # --- Scoring fade (needs both spread and total) ---
    if spread_line is not None and total_line is not None:
        for team, abbr, is_fav in [
            (home_team, home_abbr, spread_line < 0),
            (away_team, away_abbr, spread_line > 0),
        ]:
            try:
                implied_total = calculate_implied_team_total(total_line, spread_line, is_fav)
                recent = get_team_recent_scores(features_df, abbr)
                if len(recent) < 2:
                    continue
                fade = evaluate_scoring_fade(team, game_label, implied_total, recent)
                if fade.is_fading:
                    print(f"  >>> SCORING FADE: {team} implied {to_float(fade.implied_team_total, 1)} pts "
                          f"but averaging {to_float(fade.recent_scoring_avg, 1)} over last {fade.recent_games_used}")
                    entry["scoring_fades"].append({
                        "team": team,
                        "implied_total": to_float(fade.implied_team_total, 1),
                        "recent_avg": to_float(fade.recent_scoring_avg, 1),
                        "games_used": fade.recent_games_used,
                    })
            except Exception as e:
                msg = f"scoring fade {game_label} / {team}: {e!r}"
                print(f"  [error] {msg}")
                traceback.print_exc()
                errors.append(msg)

    # --- Player props ---
    entry["player_props"] = run_player_props(
        ctx["props_bundle"], ctx["player_features_df"], ctx["player_weekly_df"],
        game.get("id"), game_label, ctx["api_key"], errors,
    )

    print()
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_output(games, errors):
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "games": games,
        "errors": errors,
    }
    with open("predictions.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"Wrote {len(games)} games to predictions.json")


def run_predictions(api_key: str):
    bundle = load_model()
    ctx = {
        "api_key": api_key,
        "margin_model": bundle["margin_model"],
        "total_model": bundle["total_model"],
        "margin_std": bundle["margin_std"],
        "total_std": bundle["total_std"],
        "feature_cols": bundle["feature_columns"],
        "features_df": pd.read_parquet("nfl_game_features.parquet"),
        "props_bundle": load_props_model(),
        "injuries_df": load_parquet_optional("nfl_injuries.parquet"),
        "player_features_df": load_parquet_optional("nfl_player_features.parquet"),
        "player_weekly_df": load_parquet_optional("nfl_player_weekly_stats.parquet"),
    }
    if ctx["props_bundle"] is None or ctx["player_features_df"] is None:
        print("[props] props model or player features not found — skipping player props this run\n")

    print("Fetching live NFL odds...")
    odds_data = get_odds("nfl", api_key)
    print(f"Found {len(odds_data)} upcoming games\n")

    dashboard_games, errors = [], []

    for game in odds_data:
        label = f"{game.get('away_team')} @ {game.get('home_team')}"
        try:
            entry = process_game(game, ctx, errors)
            if entry is not None:
                dashboard_games.append(entry)
        except Exception as e:
            msg = f"game {label}: {e!r}"
            print(f"[error] {msg}")
            traceback.print_exc()
            errors.append(msg)

    # Scheduled games with no odds yet
    try:
        priced = {(g["home_team"], g["away_team"]) for g in dashboard_games}
        pending = 0
        for s in get_upcoming_schedule():
            if (s["home_team"], s["away_team"]) in priced:
                continue
            dashboard_games.append({
                "matchup": f"{s['away_team']} @ {s['home_team']}",
                "home_team": s["home_team"],
                "away_team": s["away_team"],
                "commence_time": s["gameday"],
                "week": s["week"],
                "odds_pending": True,
            })
            pending += 1
        if pending:
            print(f"Added {pending} upcoming game(s) awaiting posted odds\n")
    except Exception as e:
        msg = f"schedule: {e!r}"
        print(f"[error] {msg}")
        traceback.print_exc()
        errors.append(msg)

    write_output(dashboard_games, errors)

    if errors:
        print(f"\nFinished with {len(errors)} non-fatal error(s):")
        for msg in errors:
            print(f"  - {msg}")


if __name__ == "__main__":
    API_KEY = os.environ.get("ODDS_API_KEY")
    if not API_KEY:
        raise SystemExit("ODDS_API_KEY not set. Export it or add it to .env")
    run_predictions(API_KEY)
