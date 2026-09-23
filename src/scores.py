"""Predict the actual score: point spread (margin) and total points.

Two regression models:
  - margin = home score - away score, from the same matchup features as the win model
  - total  = home score + away score, from "sum" features: both offenses, both defenses,
             weather, dome, and how high-scoring the league has been lately
Predicted scores: home = (total + margin) / 2, away = (total - margin) / 2.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TOTAL_FEATURES = ["off_epa_sum", "def_epa_sum", "off_form_sum", "def_form_sum",
                  "league_total_avg", "wind", "cold", "indoor"]


def fit_score_models(train, win_features):
    margin = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(
        train[win_features], train["home_score"] - train["away_score"])
    total = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(
        train[TOTAL_FEATURES], train["home_score"] + train["away_score"])
    return {"margin": margin, "total": total, "win_features": win_features}


def predict_scores(models, games):
    m = models["margin"].predict(games[models["win_features"]])
    t = models["total"].predict(games[TOTAL_FEATURES])
    return m, t, (t + m) / 2, (t - m) / 2


def evaluate(models, test):
    """Compare our spread/total to Vegas on the same games."""
    t = test[test["spread_line"].notna() & test["total_line"].notna()]
    m, tot, _, _ = predict_scores(models, t)
    actual_m = (t["home_score"] - t["away_score"]).values
    actual_t = (t["home_score"] + t["away_score"]).values
    res = {
        "games": len(t),
        "our_spread_err": np.mean(np.abs(m - actual_m)),
        "vegas_spread_err": np.mean(np.abs(t["spread_line"].values - actual_m)),
        "our_total_err": np.mean(np.abs(tot - actual_t)),
        "vegas_total_err": np.mean(np.abs(t["total_line"].values - actual_t)),
    }
    # "Against the spread": when we disagree with Vegas' line, which side covers?
    diff = m - t["spread_line"].values
    cover = actual_m - t["spread_line"].values
    pick = np.sign(diff)
    mask = (cover != 0) & (pick != 0)
    res["ats_win_pct"] = np.mean(np.sign(cover[mask]) == pick[mask])
    over = np.sign(tot - t["total_line"].values)
    real = np.sign(actual_t - t["total_line"].values)
    mask = (real != 0) & (over != 0)
    res["ou_win_pct"] = np.mean(over[mask] == real[mask])
    return res
