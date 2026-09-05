# Sports Odds Project — NFL Module

Free-data pipeline: NFL play-by-play → team form features → margin/total
prediction model → live odds comparison → value bet & high-confidence alerts.

## Setup

```bash
pip install nfl_data_py pandas numpy scipy scikit-learn xgboost joblib requests
```

Get a free Odds API key: https://the-odds-api.com/#get-access
(500 free requests/month on the free tier — plenty for daily NFL odds pulls)

```bash
export ODDS_API_KEY="your_key_here"
```

## Pipeline (run in order)

```bash
cd nfl
python fetch_data.py      # pulls historical PBP + schedules, saves .parquet files
python features.py        # builds rolling team-form features per game
python train.py           # trains margin + total models, saves nfl_model.pkl
```

Then, for live predictions:

```bash
cd ../shared
python odds_api.py        # test pulling live NFL odds
```

## What each file does

| File | Purpose |
|---|---|
| `nfl/fetch_data.py` | Pulls historical play-by-play, schedules, weekly stats, injuries via `nfl_data_py` (free, no key) |
| `nfl/features.py` | Builds trailing rolling-window EPA/success-rate features per team, merges into home-vs-away differential features per game |
| `nfl/train.py` | Trains XGBoost regressors for margin and total, derives win/cover/over probabilities from a normal distribution around the prediction |
| `shared/odds_api.py` | Pulls live NFL/CFB odds from The Odds API, converts American odds to implied probability |
| `shared/edge_detection.py` | Compares model probability to no-vig market probability, flags value bets and high-confidence picks, finds best price across sportsbooks |

## Not built yet (next steps)

- **Live prediction script** — glue code that loads `nfl_model.pkl`, pulls this week's upcoming games + current team form, and generates predictions for games that haven't happened yet
- **Injury adjustment** — QB-out flag should meaningfully shift predicted margin; not yet wired into features.py
- **Dashboard** — visual layer to browse predictions/edges
- **Push notifications** — alert delivery when a value bet or high-confidence pick is found
- **Player props models** — phase 2, per your earlier scope decision
- **Backtesting report** — how the model would have performed against closing lines historically, to sanity-check before betting real money
