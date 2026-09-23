"""Predict upcoming NFL games with win probabilities and the top reasons why.

Usage:
    python src/predict.py                 # next week's unplayed games
    python src/predict.py --week 5        # a specific week this season
    python src/predict.py --team PHI      # only games involving one team
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from data import load_games
from features import FEATURES, FEATURE_LABELS, build_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def explain(explainer, row: pd.Series, top_n: int = 3) -> str:
    """Use the logistic model's weights to show which factors pushed the pick most."""
    scaler = explainer.named_steps["standardscaler"]
    lr = explainer.named_steps["logisticregression"]
    # Compare against a "neutral" matchup: evenly matched teams, normal home game,
    # not a division game. So a factor only shows up if it actually tilts THIS game.
    baseline = pd.Series(0.0, index=FEATURES)
    baseline["home_field"] = 1.0
    x = (row[FEATURES].astype(float).values - baseline.values) / scaler.scale_
    contrib = pd.Series(lr.coef_[0] * x, index=FEATURES)
    contrib = contrib[contrib.abs() > 1e-9]
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top_n]
    parts = []
    for feat, val in contrib.items():
        team = row["home_team"] if val > 0 else row["away_team"]
        label = "Neutral site, no home edge" if feat == "home_field" else FEATURE_LABELS[feat]
        parts.append(f"{label} -> {team}")
    return "; ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, help="week number (default: next unplayed week)")
    parser.add_argument("--season", type=int, help="season (default: latest)")
    parser.add_argument("--team", type=str, help="team abbreviation, e.g. KC")
    parser.add_argument("--refresh", action="store_true", help="re-download latest game data")
    args = parser.parse_args()

    saved = joblib.load(MODEL_DIR / "model.joblib")
    model, explainer = saved["model"], joblib.load(MODEL_DIR / "explainer.joblib")

    df = build_features(load_games(force_download=args.refresh))
    season = args.season or int(df["season"].max())
    games = df[df["season"] == season]
    if args.week:
        games = games[games["week"] == args.week]
    else:
        upcoming = games[~games["played"]]
        if upcoming.empty:
            print(f"No unplayed games left in {season}. Try --week N.")
            return
        games = upcoming[upcoming["week"] == upcoming["week"].min()]
    if args.team:
        t = args.team.upper()
        games = games[(games["home_team"] == t) | (games["away_team"] == t)]

    probs = model.predict_proba(games[FEATURES])[:, 1]
    print(f"\n{season} Week {int(games['week'].iloc[0])} predictions ({saved['name']})\n")
    for (_, row), p in zip(games.iterrows(), probs):
        pick, conf = (row["home_team"], p) if p >= 0.5 else (row["away_team"], 1 - p)
        line = f"{row['away_team']:>3} @ {row['home_team']:<3}  ->  {pick} {conf:.0%}"
        if row["played"]:
            actual = row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]
            mark = "correct" if actual == pick else "wrong"
            line += f"   (final {int(row['away_score'])}-{int(row['home_score'])}, {mark})"
        print(line)
        print(f"      why: {explain(explainer, row)}")


if __name__ == "__main__":
    main()
