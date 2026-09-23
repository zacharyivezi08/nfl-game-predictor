"""Simulate the rest of the season 10,000 times to get playoff and Super Bowl odds.

Each simulation:
  1. Plays every remaining regular-season game, using the model's win probability.
  2. Adds a little random "true strength" wobble per team, since the ratings aren't perfect
     (a team rated 60% might really be 55% or 65% good).
  3. Seeds each conference: 4 division winners (seeds 1-4) + 3 wild cards (5-7).
  4. Plays out the playoffs: higher seed hosts; the #1 seed gets a bye; Super Bowl is neutral.

Simplification: real NFL tiebreakers (head-to-head, division record, ...) are complicated,
so teams tied on wins are ordered randomly here.

Usage:
    python src/simulate.py
"""
import numpy as np
import pandas as pd

from predict import load_model, predict_games
from ratings import DIVISIONS, TEAM_DIV, TEAMS, prob_matrix

N_SIMS = 10_000
STRENGTH_WOBBLE = 0.25   # std dev of per-team strength uncertainty (in log-odds)
ROUNDS = ["WC", "DIV", "CON", "SB"]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def simulate_season(df: pd.DataFrame, saved, season: int, n: int = N_SIMS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(TEAMS)}
    season_games = df[df["season"] == season]
    reg = season_games[season_games["game_type"] == "REG"]

    # Current records from games already played
    wins = np.zeros(len(TEAMS))
    for r in reg[reg["played"]].itertuples():
        if r.home_score > r.away_score:
            wins[idx[r.home_team]] += 1
        elif r.home_score < r.away_score:
            wins[idx[r.away_team]] += 1
        else:
            wins[idx[r.home_team]] += 0.5
            wins[idx[r.away_team]] += 0.5

    wobble = rng.normal(0, STRENGTH_WOBBLE, size=(n, len(TEAMS)))
    W = np.tile(wins, (n, 1))

    # 1) Remaining regular-season games
    left = reg[~reg["played"]]
    if len(left):
        preds = predict_games(saved, left)
        hi = np.array([idx[t] for t in preds["home_team"]])
        ai = np.array([idx[t] for t in preds["away_team"]])
        logit = _logit(preds["home_prob"].values)[None, :] + wobble[:, hi] - wobble[:, ai]
        home_won = rng.random(logit.shape) < 1 / (1 + np.exp(-logit))
        np.add.at(W.T, hi, home_won.T)
        np.add.at(W.T, ai, ~home_won.T)

    # 2) Seeding (random order among teams tied on wins)
    sort_key = W + rng.random(W.shape) * 0.01
    div_winner = np.zeros(W.shape, dtype=bool)
    for teams in DIVISIONS.values():
        cols = [idx[t] for t in teams]
        best = np.argmax(sort_key[:, cols], axis=1)
        div_winner[np.arange(n), np.array(cols)[best]] = True

    seeds = {}  # conf -> (n, 7) team indices in seed order
    for conf in ("AFC", "NFC"):
        cols = np.array([idx[t] for t in TEAMS if TEAM_DIV[t].startswith(conf)])
        k = sort_key[:, cols]
        dw = div_winner[:, cols]
        order_dw = np.argsort(-(k + dw * 1000), axis=1)[:, :4]       # division winners first
        order_wc = np.argsort(-(k - dw * 1000), axis=1)[:, :3]       # best non-winners
        seeds[conf] = np.concatenate([cols[order_dw], cols[order_wc]], axis=1)

    # 3) Playoffs (real results are used for any playoff games already played)
    played_po = {}
    for r in season_games[season_games["played"] & (season_games["game_type"] != "REG")].itertuples():
        winner = r.home_team if r.home_score > r.away_score else r.away_team
        played_po[frozenset((r.home_team, r.away_team))] = winner

    P_home = prob_matrix(saved, _state(df, season), neutral=False).values
    P_neu = prob_matrix(saved, _state(df, season), neutral=True).values
    inv = {i: t for t, i in idx.items()}

    def play(home, away, neutral=False):
        p = (P_neu if neutral else P_home)[home, away]
        logit = _logit(p) + wobble[np.arange(n), home] - wobble[np.arange(n), away]
        won = rng.random(n) < 1 / (1 + np.exp(-logit))
        for j in range(n) if played_po else []:
            key = frozenset((inv[home[j]], inv[away[j]]))
            if key in played_po:
                won[j] = played_po[key] == inv[home[j]]
        return np.where(won, home, away)

    reached = {r: np.zeros((n, len(TEAMS)), dtype=bool) for r in ["DIV", "CON", "SB", "CHAMP"]}
    conf_champs = {}
    for conf, s in seeds.items():
        # Wild card: 2v7, 3v6, 4v5 (#1 seed has a bye)
        wc = [play(s[:, a], s[:, b]) for a, b in ((1, 6), (2, 5), (3, 4))]
        alive = np.stack([s[:, 0]] + wc, axis=1)
        # Re-seed: order survivors by original seed
        pos = np.array([[list(s[j]).index(t) for t in alive[j]] for j in range(n)])
        alive = np.take_along_axis(alive, np.argsort(pos, axis=1), axis=1)
        reached["DIV"][np.arange(n)[:, None], alive] = True
        d1 = play(alive[:, 0], alive[:, 3])
        d2 = play(alive[:, 1], alive[:, 2])
        final = np.stack([d1, d2], axis=1)
        pos = np.array([[list(s[j]).index(t) for t in final[j]] for j in range(n)])
        final = np.take_along_axis(final, np.argsort(pos, axis=1), axis=1)
        reached["CON"][np.arange(n)[:, None], final] = True
        champ = play(final[:, 0], final[:, 1])
        reached["SB"][np.arange(n), champ] = True
        conf_champs[conf] = champ
    sb = play(conf_champs["AFC"], conf_champs["NFC"], neutral=True)
    reached["CHAMP"][np.arange(n), sb] = True

    made = np.zeros((n, len(TEAMS)), dtype=bool)
    for s in seeds.values():
        made[np.arange(n)[:, None], s] = True
    top_seed = np.zeros((n, len(TEAMS)), dtype=bool)
    for s in seeds.values():
        top_seed[np.arange(n), s[:, 0]] = True

    out = pd.DataFrame({
        "team": TEAMS,
        "division": [TEAM_DIV[t] for t in TEAMS],
        "wins_now": wins,
        "proj_wins": W.mean(axis=0),
        "playoffs": made.mean(axis=0),
        "division_title": div_winner.mean(axis=0),
        "top_seed": top_seed.mean(axis=0),
        "conf_title": reached["SB"].mean(axis=0),
        "super_bowl": reached["CHAMP"].mean(axis=0),
    })
    return out.sort_values(["super_bowl", "playoffs"], ascending=False).reset_index(drop=True)


def _state(df, season):
    from features import SNAPSHOTS
    return SNAPSHOTS["latest"]


def main():
    from train import load_all
    saved = load_model()
    df = load_all()
    season = int(df["season"].max())
    res = simulate_season(df, saved, season)
    pct = lambda x: f"{x:.0%}" if x >= 0.01 else ("<1%" if x > 0 else "-")
    show = res.copy()
    for c in ["playoffs", "division_title", "top_seed", "conf_title", "super_bowl"]:
        show[c] = show[c].map(pct)
    show["proj_wins"] = show["proj_wins"].map("{:.1f}".format)
    show["wins_now"] = show["wins_now"].map("{:g}".format)
    print(f"\n{season} playoff odds ({N_SIMS:,} simulations)\n")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
