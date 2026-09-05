"""
Odds API Client (shared across NFL and CFB)
==========================
Wraps The Odds API (https://the-odds-api.com) to fetch live spreads,
moneylines, and totals. Free tier covers current/upcoming odds and
scores; historical odds require a paid plan.

Get a free API key at: https://the-odds-api.com/#get-access
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory and loads it into environment variables

BASE_URL = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "nfl": "americanfootball_nfl",
    "cfb": "americanfootball_ncaaf",
}


def get_odds(sport: str, api_key: str, regions: str = "us", markets: str = "h2h,spreads,totals"):
    """
    Fetch current odds for all upcoming games in a sport.

    sport: 'nfl' or 'cfb'
    markets: comma-separated, e.g. 'h2h,spreads,totals'
             (h2h = moneyline)
    """
    sport_key = SPORT_KEYS[sport]
    url = f"{BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def get_player_props(sport: str, event_id: str, api_key: str, markets: str = "player_pass_yds,player_rush_yds,player_reception_yds,player_receptions"):
    """
    Player props require the event-odds endpoint (per-game, not bulk).
    You need the event_id from get_odds() first.
    """
    sport_key = SPORT_KEYS[sport]
    url = f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


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
