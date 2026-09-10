"""
Odds API Client (shared across NFL and CFB)
==========================
Wraps the SportsGameOdds API (https://sportsgameodds.com) to fetch live
spreads, moneylines, and totals. Reshapes the response into the same
format the-odds-api.com used to return, so the rest of the pipeline
(predict.py) doesn't need any changes.

Get a free API key at: https://sportsgameodds.com/pricing
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory and loads it into environment variables

BASE_URL = "https://api.sportsgameodds.com/v2"

LEAGUE_IDS = {
    "nfl": "NFL",
    "cfb": "NCAAF",
}

# Game-level oddIDs for moneyline/spread/total.
# includeOpposingOdds=true auto-fills the away/under side.
GAME_ODD_IDS = "points-home-game-ml-home,points-home-game-sp-home,points-all-game-ou-over"


def _american_str_to_int(odds_str):
    """SportsGameOdds returns American odds as strings like '-112' or '+150'."""
    if odds_str is None:
        return None
    return int(str(odds_str).replace("+", ""))


def get_odds(sport: str, api_key: str, regions: str = "us", markets: str = "h2h,spreads,totals"):
    """
    Fetch current odds for all upcoming games in a sport, reshaped to match
    the-odds-api.com's old response format so predict.py doesn't change:

    [
      {
        "id": ..., "commence_time": ..., "home_team": ..., "away_team": ...,
        "bookmakers": [
          {"key": ..., "title": ..., "markets": [
              {"key": "h2h", "outcomes": [{"name": ..., "price": ...}, ...]},
              {"key": "spreads", "outcomes": [{"name": ..., "price": ..., "point": ...}, ...]},
              {"key": "totals", "outcomes": [{"name": "Over"/"Under", "price": ..., "point": ...}, ...]},
          ]}
        ]
      }, ...
    ]

    'regions' and 'markets' are accepted for backwards compatibility but
    aren't used the way SportsGameOdds structures requests.
    """
    league_id = LEAGUE_IDS[sport]
    url = f"{BASE_URL}/events"
    params = {
        "apiKey": api_key,
        "leagueID": league_id,
        "oddsAvailable": "true",
        "oddID": GAME_ODD_IDS,
        "includeOpposingOdds": "true",
        "limit": 50,
    }

    events = []
    next_cursor = None
    while True:
        if next_cursor:
            params["cursor"] = next_cursor
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success", False):
            raise RuntimeError(f"SportsGameOdds error: {payload.get('error')}")
        events.extend(payload.get("data", []))
        next_cursor = payload.get("nextCursor")
        if not next_cursor:
            break

    games = []
    for event in events:
        home_name = event.get("teams", {}).get("home", {}).get("names", {}).get("long", "Home")
        away_name = event.get("teams", {}).get("away", {}).get("names", {}).get("long", "Away")
        commence_time = event.get("status", {}).get("startsAt")

        odds = event.get("odds", {})
        ml_home = odds.get("points-home-game-ml-home", {})
        ml_away = odds.get("points-away-game-ml-away", {})
        sp_home = odds.get("points-home-game-sp-home", {})
        sp_away = odds.get("points-away-game-sp-away", {})
        ou_over = odds.get("points-all-game-ou-over", {})
        ou_under = odds.get("points-all-game-ou-under", {})

        # Collect every bookmaker that appears across any of these markets
        bookmaker_ids = set()
        for market in (ml_home, ml_away, sp_home, sp_away, ou_over, ou_under):
            bookmaker_ids.update(market.get("byBookmaker", {}).keys())

        bookmakers = []
        for bm_id in bookmaker_ids:
            bm_h2h, bm_spreads, bm_totals = [], [], []

            h = ml_home.get("byBookmaker", {}).get(bm_id)
            a = ml_away.get("byBookmaker", {}).get(bm_id)
            if h and a:
                bm_h2h = [
                    {"name": home_name, "price": _american_str_to_int(h.get("odds"))},
                    {"name": away_name, "price": _american_str_to_int(a.get("odds"))},
                ]

            sh = sp_home.get("byBookmaker", {}).get(bm_id)
            sa = sp_away.get("byBookmaker", {}).get(bm_id)
            if sh and sa:
                bm_spreads = [
                    {
                        "name": home_name,
                        "price": _american_str_to_int(sh.get("odds")),
                        "point": float(sh.get("spread", 0)),
                    },
                    {
                        "name": away_name,
                        "price": _american_str_to_int(sa.get("odds")),
                        "point": float(sa.get("spread", 0)),
                    },
                ]

            o = ou_over.get("byBookmaker", {}).get(bm_id)
            u = ou_under.get("byBookmaker", {}).get(bm_id)
            if o and u:
                bm_totals = [
                    {
                        "name": "Over",
                        "price": _american_str_to_int(o.get("odds")),
                        "point": float(o.get("overUnder", 0)),
                    },
                    {
                        "name": "Under",
                        "price": _american_str_to_int(u.get("odds")),
                        "point": float(u.get("overUnder", 0)),
                    },
                ]

            markets_list = []
            if bm_h2h:
                markets_list.append({"key": "h2h", "outcomes": bm_h2h})
            if bm_spreads:
                markets_list.append({"key": "spreads", "outcomes": bm_spreads})
            if bm_totals:
                markets_list.append({"key": "totals", "outcomes": bm_totals})

            if markets_list:
                bookmakers.append({"key": bm_id, "title": bm_id, "markets": markets_list})

        games.append(
            {
                "id": event.get("eventID"),
                "commence_time": commence_time,
                "home_team": home_name,
                "away_team": away_name,
                "bookmakers": bookmakers,
            }
        )

    return games


def get_player_props(sport: str, event_id: str, api_key: str, markets: str = "player_pass_yds,player_rush_yds,player_reception_yds,player_receptions"):
    """
    Placeholder for player props (SportsGameOdds version) — not yet implemented.
    Kept here so predict.py's import doesn't break. Will be built out when
    player props support is added (phase 2).
    """
    raise NotImplementedError("Player props via SportsGameOdds not yet implemented")


def american_to_implied_prob(american_odds: int) -> float:
    """
    Converts American odds to implied probability (still includes vig).
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)


def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Normalizes two implied probabilities (that sum to >1 due to vig)
    back down to a fair, no-vig probability split.
    """
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


if __name__ == "__main__":
    API_KEY = os.environ.get("ODDS_API_KEY", "YOUR_KEY_HERE")
    odds = get_odds("nfl", API_KEY)
    print(f"Fetched odds for {len(odds)} upcoming NFL games")
    if odds:
        print(odds[0])
