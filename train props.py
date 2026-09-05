"""
NFL Player Props Model Training
==========================
Trains one XGBoost regressor per prop category (passing yards, rushing
yards, receiving yards, receptions), using each player's own recent form
plus the opponent's recent defensive strength against that stat as features.

Install: pip install xgboost scikit-learn pandas numpy joblib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib

from fetch_player_data import PROP_POSITIONS


# Feature sets per prop type: player's own volume/production roll + matchup
PROP_FEATURES = {
    "passing_yards": ["passing_yards_roll", "attempts_roll", "passing_yards_allowed_roll"],
    "rushing_yards": ["rushing_yards_roll", "carries_roll", "rushing_yards_allowed_roll"],
    "receiving_yards": ["receiving_yards_roll", "targets_roll", "receiving_yards_allowed_roll"],
    "receptions": ["receptions_roll", "targets_roll", "receptions_allowed_roll"],
}


def train_prop_model(df: pd.DataFrame, stat: str) -> tuple[xgb.XGBRegressor, float]:
    """
    Trains a regressor for one prop stat, restricted to the positions that
    stat is relevant for (e.g. only QBs for passing yards).
    """
    positions = PROP_POSITIONS[stat]
    features = PROP_FEATURES[stat]

    subset = df[df["position"].isin(positions)].dropna(subset=features + [stat])

    X = subset[features]
    y = subset[stat]

    tscv = TimeSeriesSplit(n_splits=5)
    residuals = []

    for train_idx, test_idx in tscv.split(X):
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        residuals.extend((y.iloc[test_idx].values - preds).tolist())
        mae = mean_absolute_error(y.iloc[test_idx], preds)
        print(f"  Fold MAE ({stat}): {mae:.1f}")

    residual_std = float(np.std(residuals))

    final_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    )
    final_model.fit(X, y)

    return final_model, residual_std


if __name__ == "__main__":
    df = pd.read_parquet("nfl_player_features.parquet")

    bundle = {}
    for stat in PROP_FEATURES:
        print(f"Training {stat} model...")
        model, std = train_prop_model(df, stat)
        bundle[stat] = {"model": model, "std": std, "features": PROP_FEATURES[stat]}

    joblib.dump(bundle, "nfl_props_model.pkl")
    print("\nSaved player props models.")
