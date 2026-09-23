"""Would betting the model's picks have made money? A walk-forward backtest.

For each season from 2016 on, the model is trained ONLY on earlier seasons, then
"bets" $1 on games using the real closing moneyline odds.

Usage:
    python src/backtest.py
"""
import numpy as np
import pandas as pd

from features import FEATURE_GROUPS
from train import get_models, load_all

FIRST_TEST_SEASON = 2016


def profit_per_dollar(moneyline):
    return np.where(moneyline > 0, moneyline / 100, 100 / -moneyline)


def main():
    feats = FEATURE_GROUPS["base"] + FEATURE_GROUPS["qb"] + FEATURE_GROUPS["weather"]
    df = load_all()
    done = df[df["played"] & ~df["tie"] & (df["season"] >= 2002)]
    last = int(done["season"].max())

    parts = []
    for season in range(FIRST_TEST_SEASON, last + 1):
        train = done[done["season"] < season]
        test = done[(done["season"] == season) & done["home_moneyline"].notna() & done["away_moneyline"].notna()].copy()
        if test.empty:
            continue
        model = get_models()["Logistic Regression"].fit(train[feats], train["home_win"])
        test["p"] = model.predict_proba(test[feats])[:, 1]
        parts.append(test)
    t = pd.concat(parts)

    bets = []
    for side in ("home", "away"):
        p = t["p"] if side == "home" else 1 - t["p"]
        vegas = t["vegas_prob"] if side == "home" else 1 - t["vegas_prob"]
        won = (t["home_win"] == 1) if side == "home" else (t["home_win"] == 0)
        bets.append(pd.DataFrame({
            "season": t["season"], "p": p, "edge": p - vegas, "won": won,
            "profit": np.where(won, profit_per_dollar(t[f"{side}_moneyline"]), -1.0),
        }))
    b = pd.concat(bets)

    print(f"\nBacktest {FIRST_TEST_SEASON}-{last}: $1 bets at real moneyline odds\n")
    strategies = {
        "Bet every model pick": b[b["p"] > 0.5],
        "Only picks the model is >75% sure of": b[b["p"] > 0.75],
        "Model likes team more than Vegas (any edge)": b[b["edge"] > 0],
        "Model edge over Vegas > 5%": b[b["edge"] > 0.05],
        "Model edge over Vegas > 10%": b[b["edge"] > 0.10],
    }
    rows = [{"strategy": k, "bets": len(v), "win %": f"{v['won'].mean():.1%}",
             "return": f"{v['profit'].mean():+.1%}", "$ on $100/bet": f"{v['profit'].sum() * 100:+,.0f}"}
            for k, v in strategies.items()]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n'return' = profit per dollar bet. Negative = the strategy lost money.")


if __name__ == "__main__":
    main()
