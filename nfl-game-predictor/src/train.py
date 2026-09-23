"""Train and compare three models, then save the best one.

Usage:
    python src/train.py            # use cached data
    python src/train.py --refresh  # re-download latest results first
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_games
from features import FEATURES, build_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FIRST_SEASON = 2002   # skip first few seasons while Elo ratings warm up
TEST_SEASONS = 4      # hold out the most recent 4 finished seasons for testing


def get_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=25, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download latest game data")
    args = parser.parse_args()

    df = build_features(load_games(force_download=args.refresh))
    done = df[df["played"] & ~df["tie"] & (df["season"] >= FIRST_SEASON)]

    # Only use seasons that are fully finished for the train/test split
    last_full = done.groupby("season")["game_type"].apply(lambda s: (s == "SB").any())
    full_seasons = sorted(last_full[last_full].index)
    test_seasons = full_seasons[-TEST_SEASONS:]
    train = done[done["season"] < test_seasons[0]]
    test = done[done["season"].isin(test_seasons)]

    print(f"Training on {train.season.min()}-{train.season.max()} ({len(train)} games)")
    print(f"Testing on  {test_seasons[0]}-{test_seasons[-1]} ({len(test)} games)\n")

    baseline = accuracy_score(test["home_win"], [1] * len(test))
    print(f"Baseline (always pick home team): {baseline:.1%} accuracy\n")

    results = []
    for name, model in get_models().items():
        model.fit(train[FEATURES], train["home_win"])
        p = model.predict_proba(test[FEATURES])[:, 1]
        results.append({
            "model": name,
            "accuracy": accuracy_score(test["home_win"], p > 0.5),
            "log_loss": log_loss(test["home_win"], p),
            "brier": brier_score_loss(test["home_win"], p),
        })
    table = pd.DataFrame(results).sort_values("log_loss")
    print(table.to_string(index=False, formatters={
        "accuracy": "{:.1%}".format, "log_loss": "{:.3f}".format, "brier": "{:.3f}".format}))
    print("\n(Lower log loss / Brier = better-calibrated probabilities.)")

    # Retrain the winner on ALL finished games and save it
    best = table.iloc[0]["model"]
    final = get_models()[best]
    final.fit(done[FEATURES], done["home_win"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"name": best, "model": final}, MODEL_DIR / "model.joblib")

    # Also keep a logistic model for explaining predictions
    explainer = get_models()["Logistic Regression"]
    explainer.fit(done[FEATURES], done["home_win"])
    joblib.dump(explainer, MODEL_DIR / "explainer.joblib")
    print(f"\nSaved best model ({best}) to models/model.joblib")


if __name__ == "__main__":
    main()
