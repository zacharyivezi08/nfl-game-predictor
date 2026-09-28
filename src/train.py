"""Pick the features that help, compare models against each other and Vegas, save the best.

How the seasons are used (so the final test is honest):
    2002-2017  train while choosing features
    2018-2021  "validation": decides which feature groups to keep
    2022-2025  "test": never looked at until the very end

It also learns a "model + Vegas" blend from out-of-sample predictions (each season predicted by a
model trained only on earlier seasons), since combining two forecasts can beat either one.

Usage:
    python src/train.py            # use cached data
    python src/train.py --refresh  # re-download latest results first
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_games, load_injuries, load_qb_stats, load_snaps, load_team_epa
from features import FEATURE_GROUPS, build_features
from injuries import injury_loads
from scores import evaluate, fit_score_models

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FIRST_SEASON = 2002   # skip first few seasons while ratings warm up
TEST_SEASONS = 4      # most recent finished seasons held out for the final test
VAL_SEASONS = 4       # seasons before that used to choose features


def load_all(refresh: bool = False) -> pd.DataFrame:
    games = load_games(force_download=refresh)
    seasons = games["season"].unique()
    qb = load_qb_stats(seasons, force_download=refresh)
    epa = load_team_epa(seasons, force_download=refresh)
    inj = injury_loads(load_injuries(seasons, force_download=refresh), load_snaps(seasons, force_download=refresh))
    return build_features(games, qb, epa, inj)


def get_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=25, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42),
    }


def score(y, p):
    return {"accuracy": accuracy_score(y, p > 0.5), "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p)}


def fmt(table):
    return table.to_string(index=False, formatters={
        "accuracy": "{:.1%}".format, "log_loss": "{:.4f}".format, "brier": "{:.4f}".format})


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend_inputs(model_prob, vegas_prob):
    return np.c_[logit(model_prob), logit(vegas_prob)]


def walk_forward(done, seasons, features, model_name):
    """Out-of-sample predictions: each season is predicted by a model trained only on earlier seasons."""
    parts = []
    for s in seasons:
        tr, te = done[done["season"] < s], done[done["season"] == s].copy()
        te["p_model"] = get_models()[model_name].fit(tr[features], tr["home_win"]).predict_proba(te[features])[:, 1]
        parts.append(te)
    return pd.concat(parts)


def fit_blend(oos):
    """Learn how much to trust the model vs Vegas, from out-of-sample predictions."""
    o = oos[oos["vegas_prob"].notna()]
    return LogisticRegression(C=100, max_iter=2000).fit(blend_inputs(o["p_model"], o["vegas_prob"]), o["home_win"])


def blend_prob(blend, model_prob, vegas_prob):
    """Model + Vegas combined. Falls back to the model alone when there's no Vegas line yet."""
    model_prob = np.asarray(model_prob, dtype=float)
    vegas_prob = np.asarray(vegas_prob, dtype=float)
    out = model_prob.copy()
    ok = ~np.isnan(vegas_prob)
    if ok.any():
        out[ok] = blend.predict_proba(blend_inputs(model_prob[ok], vegas_prob[ok]))[:, 1]
    return out


BLEND_FIRST_SEASON = 2008  # out-of-sample predictions used to learn the blend start here


def split_seasons(df):
    done = df[df["played"] & ~df["tie"] & (df["season"] >= FIRST_SEASON)]
    has_sb = done.groupby("season")["game_type"].apply(lambda s: (s == "SB").any())
    full = sorted(has_sb[has_sb].index)
    test = full[-TEST_SEASONS:]
    val = full[-TEST_SEASONS - VAL_SEASONS:-TEST_SEASONS]
    return done, full, val, test


def feats_for(groups):
    return [f for g in groups for f in FEATURE_GROUPS[g]]


def choose_groups(done, val):
    """Greedy: start with base, keep adding whichever group lowers validation log loss most."""
    tr = done[done["season"] < val[0]]
    va = done[done["season"].isin(val)]

    def val_loss(groups):
        f = feats_for(groups)
        m = get_models()["Logistic Regression"].fit(tr[f], tr["home_win"])
        return log_loss(va["home_win"], m.predict_proba(va[f])[:, 1])

    chosen = ["base"]
    best = val_loss(chosen)
    print(f"  base only{'':<22} val log loss {best:.4f}")
    remaining = [g for g in FEATURE_GROUPS if g != "base"]
    while remaining:
        trials = {g: val_loss(chosen + [g]) for g in remaining}
        g, loss = min(trials.items(), key=lambda kv: kv[1])
        if loss >= best - 0.0003:  # must help by a meaningful amount
            for g2, l2 in sorted(trials.items(), key=lambda kv: kv[1]):
                print(f"  + {g2:<28} val log loss {l2:.4f}  (not enough gain, skipped)")
            break
        print(f"  + {g:<28} val log loss {loss:.4f}  (kept)")
        chosen.append(g)
        best = loss
        remaining.remove(g)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download latest game data")
    args = parser.parse_args()

    df = load_all(args.refresh)
    done, full, val, test_seasons = split_seasons(df)
    train = done[done["season"] < test_seasons[0]]
    test = done[done["season"].isin(test_seasons)]
    print(f"\nChoose features: train {FIRST_SEASON}-{val[0] - 1}, validate {val[0]}-{val[-1]}")
    print(f"Final test:      train {FIRST_SEASON}-{test_seasons[0] - 1}, test {test_seasons[0]}-{test_seasons[-1]} "
          f"({len(test)} games)\n")

    # 1) Which feature groups help?
    print("STEP 1: Which features help? (decided on validation seasons)")
    groups = choose_groups(done, val)
    features = feats_for(groups)
    print(f"-> using: {' + '.join(groups)} ({len(features)} features)\n")

    # 2) Choose the model type on the VALIDATION seasons (not the test!), then report test scores
    print("STEP 2: Which model is best? (chosen on validation, then scored on test)")
    tr_v, va = done[done["season"] < val[0]], done[done["season"].isin(val)]
    rows = []
    for name in get_models():
        mv = get_models()[name].fit(tr_v[features], tr_v["home_win"])
        mt = get_models()[name].fit(train[features], train["home_win"])
        rows.append({"model": name, "val_log_loss": log_loss(va["home_win"], mv.predict_proba(va[features])[:, 1]),
                     **score(test["home_win"], mt.predict_proba(test[features])[:, 1])})
    results = pd.DataFrame(rows).sort_values("val_log_loss")
    print(results.to_string(index=False, formatters={"val_log_loss": "{:.4f}".format, "accuracy": "{:.1%}".format,
                                                     "log_loss": "{:.4f}".format, "brier": "{:.4f}".format}))
    print("(val_log_loss picks the winner; accuracy/log_loss/brier are on the untouched test seasons)")

    # 3) Against Vegas
    print("\nSTEP 3: Model vs Vegas (same test games)")
    best = results.iloc[0]["model"]
    has_line = test["vegas_prob"].notna()
    base_only = get_models()["Logistic Regression"].fit(train[feats_for(["base"])], train["home_win"])
    best_model = get_models()[best].fit(train[features], train["home_win"])
    t = test[has_line]
    y = t["home_win"]
    vs = pd.DataFrame([
        {"source": f"Our model ({best})", **score(y, best_model.predict_proba(t[features])[:, 1])},
        {"source": "Basic model (Elo + form only)", **score(y, base_only.predict_proba(t[feats_for(['base'])])[:, 1])},
        {"source": "Vegas", **score(y, t["vegas_prob"])},
        {"source": "Always pick home team", **score(y, pd.Series(0.57, index=y.index))},
    ])
    print(fmt(vs))
    agree = ((best_model.predict_proba(t[features])[:, 1] > 0.5) == (t["vegas_prob"] > 0.5)).mean()
    print(f"Our model agrees with the Vegas favorite in {agree:.0%} of games.")

    # 4) Predicted scores: spread and total vs Vegas
    print("\nSTEP 4: Predicted spread and total vs Vegas (test seasons)")
    ev = evaluate(fit_score_models(train, features), test)
    print(f"  Spread: we miss the real margin by {ev['our_spread_err']:.1f} pts on average, Vegas by {ev['vegas_spread_err']:.1f}")
    print(f"  Total:  we miss the real total by {ev['our_total_err']:.1f} pts on average, Vegas by {ev['vegas_total_err']:.1f}")
    print(f"  Picking against the spread: {ev['ats_win_pct']:.1%}  |  over/under: {ev['ou_win_pct']:.1%}  "
          f"(need 52.4% to beat the bookmaker's cut)")

    # 5) Model + Vegas blend: learned on out-of-sample seasons BEFORE the test, scored on the test
    print("\nSTEP 5: Blend the model with Vegas (weights learned before the test seasons)")
    oos = walk_forward(done, [s for s in full if s >= BLEND_FIRST_SEASON], features, best)
    pre, post = oos[oos["season"] < test_seasons[0]], oos[oos["season"].isin(test_seasons) & oos["vegas_prob"].notna()]
    blend = fit_blend(pre)
    pb = blend_prob(blend, post["p_model"], post["vegas_prob"])
    y = post["home_win"]
    print(fmt(pd.DataFrame([
        {"source": "Our model (each season trained on earlier ones)", **score(y, post["p_model"])},
        {"source": "Vegas", **score(y, post["vegas_prob"])},
        {"source": "Model + Vegas blend", **score(y, pd.Series(pb, index=y.index))},
    ])))
    w_model, w_vegas = blend.coef_[0]
    print(f"Blend weights: model {w_model:.2f}, Vegas {w_vegas:.2f} (Vegas does most of the work)")
    late = (post["week"] >= 17) & (post["game_type"] == "REG")
    print(f"Last 2 weeks of the regular season: model {accuracy_score(y[late], post['p_model'][late] > .5):.1%}, "
          f"Vegas {accuracy_score(y[late], post['vegas_prob'][late] > .5):.1%} ({late.sum()} games)")

    # 6) Retrain on every finished season and save
    final_data = done[done["season"].isin(full)]
    final = get_models()[best].fit(final_data[features], final_data["home_win"])
    explainer = get_models()["Logistic Regression"].fit(final_data[features], final_data["home_win"])
    MODEL_DIR.mkdir(exist_ok=True)
    scores = fit_score_models(final_data, features)
    joblib.dump({"name": best, "model": final, "features": features, "groups": groups, "explainer": explainer,
                 "scores": scores, "blend": fit_blend(oos)},
                MODEL_DIR / "model.joblib")
    print(f"\nSaved {best} ({len(features)} features) to models/model.joblib")


if __name__ == "__main__":
    main()
