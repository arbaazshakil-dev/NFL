"""
Edge Detection (shared across NFL and CFB)
==========================
Compares your model's fair probability against the market's no-vig
implied probability to find value bets, and flags high-confidence
picks independent of edge.
"""

from dataclasses import dataclass
from scipy.stats import norm
from odds_api import american_to_implied_prob, remove_vig_two_way


EDGE_THRESHOLD = 0.03          # 3 percentage points of edge required to flag a value bet
HIGH_CONFIDENCE_THRESHOLD = 0.75  # model probability itself must exceed this to flag as high-confidence


@dataclass
class BetSignal:
    game: str
    market: str          # 'moneyline', 'spread', 'total'
    side: str             # e.g. 'home', 'away', 'over', 'under'
    model_prob: float
    market_prob: float
    edge: float
    is_value_bet: bool
    is_high_confidence: bool
    best_odds: int
    sportsbook: str


def evaluate_market(
    game_label: str,
    market_name: str,
    side: str,
    model_prob: float,
    home_odds: int,
    away_odds: int,
    is_home_side: bool,
    sportsbook: str,
) -> BetSignal:
    """
    Given model probability for one side of a two-way market, compute
    the no-vig market probability and flag value / high-confidence bets.
    """
    implied_home = american_to_implied_prob(home_odds)
    implied_away = american_to_implied_prob(away_odds)
    fair_home, fair_away = remove_vig_two_way(implied_home, implied_away)

    market_prob = fair_home if is_home_side else fair_away
    edge = model_prob - market_prob
    best_odds = home_odds if is_home_side else away_odds

    return BetSignal(
        game=game_label,
        market=market_name,
        side=side,
        model_prob=round(model_prob, 3),
        market_prob=round(market_prob, 3),
        edge=round(edge, 3),
        is_value_bet=edge >= EDGE_THRESHOLD,
        is_high_confidence=model_prob >= HIGH_CONFIDENCE_THRESHOLD,
        best_odds=best_odds,
        sportsbook=sportsbook,
    )


UPSET_MIN_UNDERDOG_ODDS = 130  # only consider it an "underdog" if priced at +130 or worse


@dataclass
class UpsetSignal:
    game: str
    underdog: str
    favorite: str
    model_underdog_win_prob: float
    market_underdog_implied_prob: float
    underdog_odds: int
    upset_score: float   # how much the model disagrees with the market, weighted by how big a dog they are
    sportsbook: str


def evaluate_upset(
    game_label: str,
    home_team: str,
    away_team: str,
    model_home_win_prob: float,
    home_odds: int,
    away_odds: int,
    sportsbook: str,
) -> UpsetSignal | None:
    """
    Flags a potential upset: the market has one team as a clear underdog
    (positive American odds beyond the threshold), but the model still
    gives that underdog a real (>50%, or notably elevated) chance to win.

    upset_score combines two things:
      - how far the model's probability sits above the market's no-vig
        probability for the underdog (the disagreement)
      - how big an underdog the market thinks they are (bigger dog = bigger upset)
    Both matter: a small edge on a heavy underdog is a more interesting
    upset alert than the same edge on a near-even game.
    """
    implied_home = american_to_implied_prob(home_odds)
    implied_away = american_to_implied_prob(away_odds)
    fair_home, fair_away = remove_vig_two_way(implied_home, implied_away)

    if home_odds > away_odds:
        underdog, favorite = home_team, away_team
        underdog_odds = home_odds
        model_underdog_prob = model_home_win_prob
        market_underdog_prob = fair_home
    else:
        underdog, favorite = away_team, home_team
        underdog_odds = away_odds
        model_underdog_prob = 1 - model_home_win_prob
        market_underdog_prob = fair_away

    if underdog_odds < UPSET_MIN_UNDERDOG_ODDS:
        return None  # not enough of an underdog to call this an "upset" watch

    edge = model_underdog_prob - market_underdog_prob
    if edge <= 0:
        return None  # model agrees with or is more skeptical than the market — no upset signal

    # Weight the raw edge by how big a dog they are (using market implied prob as the weight,
    # inverted so bigger dogs score higher for the same edge)
    upset_score = edge * (1 - market_underdog_prob)

    return UpsetSignal(
        game=game_label,
        underdog=underdog,
        favorite=favorite,
        model_underdog_win_prob=round(model_underdog_prob, 3),
        market_underdog_implied_prob=round(market_underdog_prob, 3),
        underdog_odds=underdog_odds,
        upset_score=round(upset_score, 4),
        sportsbook=sportsbook,
    )


def rank_upset_watch(signals: list[UpsetSignal]) -> list[UpsetSignal]:
    """Sorts upset candidates from most to least notable."""
    return sorted(signals, key=lambda s: s.upset_score, reverse=True)


RECENT_FORM_GAMES = 5          # "lately" = last 4-5 games
FADE_THRESHOLD = 3.0            # implied total must exceed recent scoring avg by this many points to flag


@dataclass
class ScoringFadeSignal:
    team: str
    game: str
    implied_team_total: float
    recent_scoring_avg: float
    recent_games_used: int
    fade_gap: float       # how far the market's expectation exceeds recent reality
    is_fading: bool


def calculate_implied_team_total(total_line: float, spread_line: float, is_favorite: bool) -> float:
    """
    Splits the game total into each team's individual implied total using
    the standard formula:
      favorite_implied  = total/2 + |spread|/2
      underdog_implied  = total/2 - |spread|/2

    spread_line should be passed as a positive number representing the
    point spread (margin), regardless of which team is favored.
    """
    half_total = total_line / 2
    half_spread = abs(spread_line) / 2
    return half_total + half_spread if is_favorite else half_total - half_spread


def evaluate_scoring_fade(
    team: str,
    game_label: str,
    implied_team_total: float,
    recent_scores: list[float],
) -> ScoringFadeSignal:
    """
    Compares a team's market-implied total for this game against their
    actual points scored over their last RECENT_FORM_GAMES games.

    recent_scores should be ordered most-recent-last; only the trailing
    RECENT_FORM_GAMES entries are used. Flags when the market still
    expects a big scoring output but recent games haven't supported it —
    e.g. implied total of 30 but they've averaged 19 over the last 5.
    """
    trailing = recent_scores[-RECENT_FORM_GAMES:]
    recent_avg = sum(trailing) / len(trailing)
    fade_gap = implied_team_total - recent_avg

    return ScoringFadeSignal(
        team=team,
        game=game_label,
        implied_team_total=round(implied_team_total, 1),
        recent_scoring_avg=round(recent_avg, 1),
        recent_games_used=len(trailing),
        fade_gap=round(fade_gap, 1),
        is_fading=fade_gap >= FADE_THRESHOLD,
    )


PROP_EDGE_THRESHOLD = 0.04   # slightly higher bar than game markets — prop lines move fast and data is noisier


@dataclass
class PropSignal:
    player: str
    stat: str            # 'passing_yards', 'rushing_yards', 'receiving_yards', 'receptions'
    line: float
    predicted_value: float
    over_prob: float
    under_prob: float
    market_over_prob: float
    market_under_prob: float
    edge: float           # positive = model favors over, negative = model favors under
    is_value_bet: bool
    side: str              # 'over' or 'under' — whichever side has the edge
    odds: int
    sportsbook: str


def evaluate_prop_bet(
    player: str,
    stat: str,
    predicted_value: float,
    residual_std: float,
    line: float,
    over_odds: int,
    under_odds: int,
    sportsbook: str,
) -> PropSignal:
    """
    Compares the model's predicted stat value against the book's posted
    line, converting the model's prediction into an over/under probability
    via a normal distribution centered on the prediction.
    """
    model_over_prob = 1 - norm.cdf(line, loc=predicted_value, scale=residual_std)
    model_under_prob = 1 - model_over_prob

    implied_over = american_to_implied_prob(over_odds)
    implied_under = american_to_implied_prob(under_odds)
    market_over_prob, market_under_prob = remove_vig_two_way(implied_over, implied_under)

    edge_over = model_over_prob - market_over_prob
    edge_under = model_under_prob - market_under_prob

    if edge_over >= edge_under:
        side, edge, odds = "over", edge_over, over_odds
    else:
        side, edge, odds = "under", edge_under, under_odds

    return PropSignal(
        player=player,
        stat=stat,
        line=line,
        predicted_value=round(predicted_value, 1),
        over_prob=round(float(model_over_prob), 3),
        under_prob=round(float(model_under_prob), 3),
        market_over_prob=round(float(market_over_prob), 3),
        market_under_prob=round(float(market_under_prob), 3),
        edge=round(float(edge), 3),
        is_value_bet=edge >= PROP_EDGE_THRESHOLD,
        side=side,
        odds=odds,
        sportsbook=sportsbook,
    )


def find_best_odds_across_books(bookmakers: list[dict], market_key: str, outcome_name: str) -> tuple[int, str]:
    """
    Scans all sportsbooks in an Odds API response for a given game and
    returns the single best (most favorable) price for one outcome —
    this is 'line shopping' automated.
    """
    best_price = None
    best_book = None

    for book in bookmakers:
        for market in book.get("markets", []):
            if market["key"] != market_key:
                continue
            for outcome in market["outcomes"]:
                if outcome["name"] != outcome_name:
                    continue
                price = outcome["price"]
                if best_price is None or price > best_price:
                    best_price = price
                    best_book = book["title"]

    return best_price, best_book
