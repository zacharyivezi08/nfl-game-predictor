"""Predict NFL games with win probabilities, the top reasons why, and the Vegas line.

Usage:
    python src/predict.py                 # next week's unplayed games
    python src/predict.py --week 5        # a specific week this season
    python src/predict.py --team PHI      # only games involving one team
    python src/predict.py --refresh       # pull the newest results first
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_LABELS

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model():
    return joblib.load(MODEL_DIR / "model.joblib")


def reasons(saved, row: pd.Series, top_n: int = 3):
    """Use the logistic model's weights to find which factors pushed the pick most.

    Each factor is compared to a "neutral" matchup (evenly matched teams, normal
    home game, calm weather), so it only shows up if it actually tilts THIS game.
    """
    feats, explainer = saved["features"], saved["explainer"]
    scaler = explainer.named_steps["standardscaler"]
    lr = explainer.named_steps["logisticregression"]
    baseline = pd.Series(0.0, index=feats)
    baseline["home_field"] = 1.0
    x = (row[feats].astype(float).values - baseline.values) / scaler.scale_
    contrib = pd.Series(lr.coef_[0] * x, index=feats)
    contrib = contrib[contrib.abs() > 0.01]
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top_n]
    out = []
    for feat, val in contrib.items():
        label = "Neutral site, no home edge" if feat == "home_field" else FEATURE_LABELS[feat]
        out.append((label, row["home_team"] if val > 0 else row["away_team"]))
    return out


def predict_games(saved, games: pd.DataFrame) -> pd.DataFrame:
    """Add pick, confidence, Vegas comparison and reasons to each game."""
    games = games.copy()
    games["home_prob"] = saved["model"].predict_proba(games[saved["features"]])[:, 1]
    games["pick"] = games.apply(lambda r: r.home_team if r.home_prob >= 0.5 else r.away_team, axis=1)
    games["confidence"] = games["home_prob"].where(games["home_prob"] >= 0.5, 1 - games["home_prob"])
    games["vegas_pick"] = games.apply(
        lambda r: None if pd.isna(r.vegas_prob) else (r.home_team if r.vegas_prob >= 0.5 else r.away_team), axis=1)
    games["vegas_conf"] = games["vegas_prob"].where(games["vegas_prob"] >= 0.5, 1 - games["vegas_prob"])
    games["winner"] = games.apply(
        lambda r: None if not r.played or r.tie else (r.home_team if r.home_score > r.away_score else r.away_team), axis=1)
    games["correct"] = games.apply(lambda r: None if pd.isna(r.winner) else r.pick == r.winner, axis=1)
    games["vegas_correct"] = games.apply(
        lambda r: None if pd.isna(r.winner) or pd.isna(r.vegas_pick) else r.vegas_pick == r.winner, axis=1)
    games["reasons"] = [reasons(saved, r) for _, r in games.iterrows()]
    return games


def time_slot(weekday, gametime) -> str:
    """Group kickoffs into TV windows, e.g. 'Sunday 1 PM', 'Sunday Night' (times are Eastern)."""
    hour = int(str(gametime)[:2]) if isinstance(gametime, str) and gametime[:2].isdigit() else 13
    if weekday == "Sunday":
        if hour < 12:
            return "Sunday Morning"
        if hour < 15:
            return "Sunday 1 PM"
        if hour < 19:
            return "Sunday 4 PM"
        return "Sunday Night"
    if hour >= 19:
        return f"{weekday} Night"
    return f"{weekday} {hour % 12 or 12} {'PM' if hour >= 12 else 'AM'}"


def best_picks(preds: pd.DataFrame) -> pd.DataFrame:
    """The model's most confident pick in each time slot, in kickoff order."""
    p = preds.copy()
    p["slot"] = [time_slot(w, t) for w, t in zip(p["weekday"], p["gametime"])]
    p["games_in_slot"] = p.groupby("slot")["game_id"].transform("count")
    p = p.sort_values(["gameday", "gametime", "confidence"], ascending=[True, True, False])
    best = p.loc[p.groupby("slot", sort=False)["confidence"].idxmax()]
    return best.sort_values(["gameday", "gametime"])


def season_record(preds: pd.DataFrame):
    done = preds[preds["correct"].isin([True, False])]
    vegas = done[done["vegas_correct"].isin([True, False])]
    return int(done["correct"].sum()), len(done), int(vegas["vegas_correct"].sum()), len(vegas)


def pick_week(df, season, week=None):
    games = df[df["season"] == season]
    if week:
        return games[games["week"] == week]
    upcoming = games[~games["played"]]
    if upcoming.empty:
        return upcoming
    return upcoming[upcoming["week"] == upcoming["week"].min()]


def main():
    from train import load_all

    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, help="week number (default: next unplayed week)")
    parser.add_argument("--season", type=int, help="season (default: latest)")
    parser.add_argument("--team", type=str, help="team abbreviation, e.g. KC")
    parser.add_argument("--refresh", action="store_true", help="re-download latest game data")
    args = parser.parse_args()

    saved = load_model()
    df = load_all(args.refresh)
    season = args.season or int(df["season"].max())
    games = pick_week(df, season, args.week)
    if args.team:
        t = args.team.upper()
        games = games[(games["home_team"] == t) | (games["away_team"] == t)]
    if games.empty:
        print("No games found. Try --week N.")
        return

    preds = predict_games(saved, games)
    print(f"\n{season} Week {int(preds['week'].iloc[0])} predictions ({saved['name']})\n")
    for _, r in preds.iterrows():
        line = f"{r.away_team:>3} @ {r.home_team:<3}  ->  {r.pick} {r.confidence:.0%}"
        if isinstance(r.vegas_pick, str):
            line += f"   | Vegas: {r.vegas_pick} {r.vegas_conf:.0%}"
            if r.vegas_pick != r.pick:
                line += "  << DISAGREES"
        if isinstance(r.winner, str):
            line += f"   | final {int(r.away_score)}-{int(r.home_score)} {'correct' if r.correct else 'wrong'}"
        print(line)
        print("      why: " + "; ".join(f"{lbl} -> {team}" for lbl, team in r.reasons))

    if not args.team:
        print("\nMOST CONFIDENT PICK OF EACH TIME SLOT (not betting advice)")
        for _, r in best_picks(preds).iterrows():
            opp = r.home_team if r.pick == r.away_team else r.away_team
            n = f"(best of {r.games_in_slot})" if r.games_in_slot > 1 else "(only game)"
            print(f"  {r.slot:<15} {r.pick} over {opp} {r.confidence:.0%}  {n}")

    right, total, v_right, v_total = season_record(predict_games(saved, df[df["season"] == season]))
    if total:
        print(f"\n{season} season so far: model {right}-{total - right} ({right / total:.0%}), "
              f"Vegas favorites {v_right}-{v_total - v_right} ({v_right / v_total:.0%})")


if __name__ == "__main__":
    main()
