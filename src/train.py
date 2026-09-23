"""Test which features help, compare models against each other and Vegas, save the best.

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

from data import load_games, load_qb_stats
from features import FEATURE_GROUPS, build_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FIRST_SEASON = 2002   # skip first few seasons while Elo/QB ratings warm up
TEST_SEASONS = 4      # hold out the most recent 4 finished seasons for testing


def load_all(refresh: bool = False) -> pd.DataFrame:
    games = load_games(force_download=refresh)
    qb = load_qb_stats(games["season"].unique(), force_download=refresh)
    return build_features(games, qb)


def get_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=25, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42),
    }


def score(y, p):
    return {"accuracy": accuracy_score(y, p > 0.5), "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p)}


def fmt(table):
    return table.to_string(index=False, formatters={
        "accuracy": "{:.1%}".format, "log_loss": "{:.3f}".format, "brier": "{:.3f}".format})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download latest game data")
    args = parser.parse_args()

    df = load_all(args.refresh)
    done = df[df["played"] & ~df["tie"] & (df["season"] >= FIRST_SEASON)]

    # Only use seasons that are fully finished (have a Super Bowl) for training/testing
    has_sb = done.groupby("season")["game_type"].apply(lambda s: (s == "SB").any())
    full_seasons = sorted(has_sb[has_sb].index)
    test_seasons = full_seasons[-TEST_SEASONS:]
    train = done[done["season"] < test_seasons[0]]
    test = done[done["season"].isin(test_seasons)]
    print(f"\nTraining on {train.season.min()}-{train.season.max()} ({len(train)} games)")
    print(f"Testing on  {test_seasons[0]}-{test_seasons[-1]} ({len(test)} games)\n")

    # 1) Which feature groups actually help? (quick test with logistic regression)
    print("STEP 1: Which features help?")
    combos = {
        "base": ["base"],
        "base + QB": ["base", "qb"],
        "base + weather": ["base", "weather"],
        "base + QB + weather": ["base", "qb", "weather"],
    }
    rows = []
    for label, groups in combos.items():
        feats = [f for g in groups for f in FEATURE_GROUPS[g]]
        m = get_models()["Logistic Regression"].fit(train[feats], train["home_win"])
        rows.append({"features": label, **score(test["home_win"], m.predict_proba(test[feats])[:, 1]), "_feats": feats})
    ablation = pd.DataFrame(rows).sort_values("log_loss")
    print(fmt(ablation.drop(columns="_feats")))
    features = ablation.iloc[0]["_feats"]
    print(f"-> using: {ablation.iloc[0]['features']}\n")

    # 2) Compare model types on the best feature set
    print("STEP 2: Which model is best?")
    rows = []
    for name, model in get_models().items():
        model.fit(train[features], train["home_win"])
        rows.append({"model": name, **score(test["home_win"], model.predict_proba(test[features])[:, 1])})
    results = pd.DataFrame(rows).sort_values("log_loss")
    print(fmt(results))

    # 3) How do we stack up against Vegas and a dumb baseline?
    print("\nSTEP 3: Model vs Vegas (same test games)")
    best = results.iloc[0]["model"]
    has_line = test["vegas_prob"].notna()
    best_model = get_models()[best].fit(train[features], train["home_win"])
    p_model = best_model.predict_proba(test.loc[has_line, features])[:, 1]
    y = test.loc[has_line, "home_win"]
    vs = pd.DataFrame([
        {"source": f"Our model ({best})", **score(y, p_model)},
        {"source": "Vegas", **score(y, test.loc[has_line, "vegas_prob"])},
        {"source": "Always pick home team", **score(y, pd.Series(0.57, index=y.index))},
    ])
    print(fmt(vs))
    agree = ((p_model > 0.5) == (test.loc[has_line, "vegas_prob"] > 0.5)).mean()
    print(f"Our model agrees with the Vegas favorite in {agree:.0%} of games.")

    # 4) Retrain on all finished seasons and save.
    # (The current season is left out so its weekly record is an honest test.)
    final_data = done[done["season"].isin(full_seasons)]
    final = get_models()[best].fit(final_data[features], final_data["home_win"])
    explainer = get_models()["Logistic Regression"].fit(final_data[features], final_data["home_win"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"name": best, "model": final, "features": features, "explainer": explainer},
                MODEL_DIR / "model.joblib")
    print(f"\nSaved {best} ({len(features)} features) to models/model.joblib")


if __name__ == "__main__":
    main()
