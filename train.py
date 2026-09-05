"""
NFL Model Training
==========================
Trains two gradient boosting regressors:
  1. Predicted margin (home_score - away_score)
  2. Predicted total (home_score + away_score)

From these two numbers we derive win probability, spread-cover
probability, and over/under probability using a normal distribution
around the prediction (residual std from backtesting).

Install:
    pip install xgboost scikit-learn pandas numpy scipy
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib


FEATURE_PREFIX = "diff_"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def train_model(df: pd.DataFrame, target: str) -> tuple[xgb.XGBRegressor, float]:
    """
    Trains an XGBoost regressor for the given target ('margin' or 'total'),
    using time-series cross-validation (never train on future games to
    predict past ones). Returns the fitted model and the residual std
    (needed later to convert point predictions into probabilities).
    """
    features = get_feature_columns(df)
    X = df[features]
    y = df[target]

    tscv = TimeSeriesSplit(n_splits=5)
    residuals = []

    for train_idx, test_idx in tscv.split(X):
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        residuals.extend((y.iloc[test_idx].values - preds).tolist())
        mae = mean_absolute_error(y.iloc[test_idx], preds)
        print(f"  Fold MAE ({target}): {mae:.2f}")

    residual_std = float(np.std(residuals))

    # Final model trained on ALL data for production use
    final_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )
    final_model.fit(X, y)

    return final_model, residual_std


def predict_probabilities(
    predicted_margin: float,
    predicted_total: float,
    margin_std: float,
    total_std: float,
    spread_line: float = 0.0,
    total_line: float | None = None,
) -> dict:
    """
    Converts point predictions into betting-relevant probabilities.

    spread_line: the market spread, expressed as home team's line
                 (e.g. -3.5 means home favored by 3.5)
    total_line: the market over/under total
    """
    # Home win probability: P(margin > 0)
    home_win_prob = 1 - norm.cdf(0, loc=predicted_margin, scale=margin_std)

    # Home covers the spread: P(margin > -spread_line)
    # (if home is -3.5, they need to win by more than 3.5, i.e. margin > 3.5)
    cover_threshold = -spread_line
    home_cover_prob = 1 - norm.cdf(cover_threshold, loc=predicted_margin, scale=margin_std)

    result = {
        "predicted_margin": round(predicted_margin, 1),
        "predicted_total": round(predicted_total, 1),
        "home_win_prob": round(float(home_win_prob), 3),
        "away_win_prob": round(float(1 - home_win_prob), 3),
        "home_cover_prob": round(float(home_cover_prob), 3),
        "away_cover_prob": round(float(1 - home_cover_prob), 3),
    }

    if total_line is not None:
        over_prob = 1 - norm.cdf(total_line, loc=predicted_total, scale=total_std)
        result["over_prob"] = round(float(over_prob), 3)
        result["under_prob"] = round(float(1 - over_prob), 3)

    return result


if __name__ == "__main__":
    df = pd.read_parquet("nfl_game_features.parquet")

    print("Training margin model...")
    margin_model, margin_std = train_model(df, "margin")

    print("Training total model...")
    total_model, total_std = train_model(df, "total")

    joblib.dump(
        {
            "margin_model": margin_model,
            "total_model": total_model,
            "margin_std": margin_std,
            "total_std": total_std,
            "feature_columns": get_feature_columns(df),
        },
        "nfl_model.pkl",
    )

    print(f"\nSaved model. margin_std={margin_std:.2f}, total_std={total_std:.2f}")
    print("These std values calibrate confidence — lower is a tighter, more confident model.")
