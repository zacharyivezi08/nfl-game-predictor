"""Player yardage projections: passing yards (QBs), rushing yards (RBs), receiving yards (WRs, TEs, RBs).

For every player-game, using only games played BEFORE it:
  - the player's recent production and usage (weighted average, recent games count most, across seasons)
  - how many yards the opponent's defense has been allowing to that kind of player lately
  - how many points Vegas expects the player's team to score (more points = more yards), and the spread
    (favorites run more late in games, underdogs throw more)
  - the starting QB: his recent yards per attempt, and whether he's new (passing and rushing)
  - running backs only: carries left behind by teammates who are out (their carries go to the backs who play)
Tested and left out (no real gain on 2018-2021 validation / 2022-2025 test): target share, air yards, snap share,
and teammate absences for receivers.
A ridge regression per stat turns those into a projection. Projections assume the player plays.

Usage:
    python src/players.py     # honest test (2022-2025) vs simple baselines, then this week's projections
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# How fast recent games take over, per stat (a game HALFLIFE games ago counts half as much as the latest one).
# Chosen on the 2018-2021 validation seasons (tested 3, 4, 6, 8): QB passing changes fastest, receiving slowest.
HALFLIFE = {"pass": 3, "rush": 4, "rec": 6}
OPP_WINDOW = 8      # opponent defense: yards allowed over its last 8 games
MIN_GAMES = 3       # need at least 3 prior games to project a player

STATS = {
    # key: (label, yards column, usage column, positions, defense group)
    "pass": ("Passing yards", "passing_yards", "attempts", ["QB"], "QB"),
    "rush": ("Rushing yards", "rushing_yards", "carries", ["RB"], "RB"),
    "rec": ("Receiving yards", "receiving_yards", "targets", ["WR", "TE", "RB"], None),  # grouped by position
}


def _implied_points(games):
    """Vegas implied points for each team in each game (falls back to 22 when there's no line)."""
    g = games[["game_id", "home_team", "away_team", "spread_line", "total_line"]].copy()
    tot = g["total_line"].fillna(44.0)
    spr = g["spread_line"].fillna(0.0)
    home = pd.DataFrame({"game_id": g["game_id"], "team": g["home_team"], "implied": tot / 2 + spr / 2, "spread": spr})
    away = pd.DataFrame({"game_id": g["game_id"], "team": g["away_team"], "implied": tot / 2 - spr / 2, "spread": -spr})
    return pd.concat([home, away], ignore_index=True)


def _order(games):
    o = games[["game_id", "season", "week", "gameday"]].copy()
    o["order"] = o["gameday"].rank(method="dense")
    return o[["game_id", "order"]]


def player_features(stats, games):
    """One row per player-game with features built only from earlier games (plus 'next game' state per player)."""
    s = stats.merge(_order(games), on="game_id", how="inner").sort_values(["player_id", "order"])
    s = s.merge(_implied_points(games), on=["game_id", "team"], how="left")
    out = []
    for key, (label, ycol, ucol, positions, _) in STATS.items():
        d = s[s["position"].isin(positions)].copy()
        if key == "rush":
            d = d[d[ucol] > 0]
        if key == "rec":
            d = d[(d[ucol] > 0) | (d["position"] != "RB")]
        g = d.groupby("player_id", sort=False)
        hl = HALFLIFE[key]
        d["y_ewm"] = g[ycol].transform(lambda x: x.shift().ewm(halflife=hl).mean())
        d["u_ewm"] = g[ucol].transform(lambda x: x.shift().ewm(halflife=hl).mean())
        d["y_next"] = g[ycol].transform(lambda x: x.ewm(halflife=hl).mean())    # state after this game
        d["u_next"] = g[ucol].transform(lambda x: x.ewm(halflife=hl).mean())
        d["n_prior"] = g.cumcount()
        d["stat"] = key
        d["y"] = d[ycol]
        d["usage"] = d[ucol]
        out.append(d)
    f = pd.concat(out, ignore_index=True)

    # Defense: yards allowed to each group per game, rolling average over its previous OPP_WINDOW games
    f["dgroup"] = np.where(f["stat"] == "rec", f["position"], f["stat"])
    allowed = f.groupby(["opponent_team", "game_id", "dgroup", "order"], as_index=False)["y"].sum()
    allowed = allowed.sort_values(["opponent_team", "dgroup", "order"])
    ga = allowed.groupby(["opponent_team", "dgroup"], sort=False)["y"]
    allowed["opp_allowed"] = ga.transform(lambda x: x.shift().rolling(OPP_WINDOW, min_periods=2).mean())
    allowed["opp_next"] = ga.transform(lambda x: x.rolling(OPP_WINDOW, min_periods=2).mean())
    f = f.merge(allowed[["opponent_team", "game_id", "dgroup", "opp_allowed"]],
                on=["opponent_team", "game_id", "dgroup"], how="left")
    league = allowed.groupby("dgroup")["y"].mean()
    f["opp_allowed"] = f["opp_allowed"].fillna(f["dgroup"].map(league))
    f["is_te"] = (f["position"] == "TE").astype(int)
    f["is_rb"] = (f["position"] == "RB").astype(int)
    return _add_context(f, games), allowed, league


def _starters(games):
    """Each team's starting QB per game, and whether he's different from the usual starter (last 4 games)."""
    g = games[["game_id", "gameday", "home_team", "away_team", "home_qb_id", "away_qb_id"]]
    t = pd.concat([g.rename(columns={"home_team": "team", "home_qb_id": "qb"})[["game_id", "gameday", "team", "qb"]],
                   g.rename(columns={"away_team": "team", "away_qb_id": "qb"})[["game_id", "gameday", "team", "qb"]]])
    t = t.sort_values("gameday")

    def usual(col):
        out, hist = [], []
        for q in col:
            recent = hist[-4:]
            out.append(max(set(recent), key=recent.count) if recent else None)
            if isinstance(q, str):
                hist.append(q)
        return pd.Series(out, index=col.index)
    t["usual"] = t.groupby("team")["qb"].transform(usual)
    t["qb_new"] = ((t["qb"] != t["usual"]) & t["usual"].notna()).astype(int)
    return t


def _shrunk_ypa(y, u):
    w = (u.fillna(0) / 30).clip(0, 1)
    return w * (y / u.clip(lower=1)).fillna(6.0) + (1 - w) * 6.0


def _add_context(f, games):
    """Starting-QB features (all stats) + carries vacated by absent teammates (running backs)."""
    st = _starters(games)
    f = f.merge(st[["game_id", "team", "qb", "qb_new"]], on=["game_id", "team"], how="left")
    qp = f[f["stat"] == "pass"][["player_id", "game_id", "y_ewm", "u_ewm"]].rename(
        columns={"player_id": "qb", "y_ewm": "qy", "u_ewm": "qu"})
    f = f.merge(qp, on=["qb", "game_id"], how="left")
    f["qb_ypa"] = _shrunk_ypa(f["qy"], f["qu"])
    f["qb_new"] = f["qb_new"].fillna(0)
    f = f.drop(columns=["qy", "qu"])

    # RBs who played the team's previous game but not this one leave their carries behind
    f["vacated"] = 0.0
    r = f[f["stat"] == "rush"].sort_values("order")
    vac = {}
    for team, d in r.groupby("team"):
        last_seen = {}  # player -> (carries per game, team game they last played)
        prev = None
        for o, rows in d.groupby("order"):
            present = set(rows["player_id"])
            if prev is not None:
                vac[(team, o)] = sum(u for pid, (u, oo) in last_seen.items() if oo == prev and pid not in present and u >= 5)
            for x in rows.itertuples():
                last_seen[x.player_id] = (x.u_next, o)
            prev = o
    idx = f["stat"] == "rush"
    f.loc[idx, "vacated"] = [vac.get((t, o), 0.0) for t, o in zip(f.loc[idx, "team"], f.loc[idx, "order"])]
    tot = f[idx].groupby(["team", "order"])["u_ewm"].transform("sum")
    f["vac_share"] = 0.0
    f.loc[idx, "vac_share"] = (f.loc[idx, "vacated"] * f.loc[idx, "u_ewm"] / tot.clip(lower=1)).fillna(0)
    return f


FEATURES = ["y_ewm", "u_ewm", "opp_allowed", "implied", "spread", "is_te", "is_rb"]
FEATURES_BY_STAT = {  # chosen on 2018-2021 validation, confirmed on the 2022-2025 test
    "pass": FEATURES + ["qb_ypa", "qb_new"],
    "rush": FEATURES + ["vacated", "vac_share", "qb_ypa", "qb_new"],
    "rec": FEATURES,
}


def fit_models(f, before_season):
    models = {}
    for key in STATS:
        feats = FEATURES_BY_STAT[key]
        d = f[(f["stat"] == key) & (f["season"] < before_season) & (f["n_prior"] >= MIN_GAMES)].dropna(subset=FEATURES)
        fill = d[feats].median()
        m = make_pipeline(StandardScaler(), Ridge(alpha=5)).fit(d[feats].fillna(fill), d["y"])
        models[key] = (m, feats, fill)
    return models


def _predict(models, key, X):
    m, feats, fill = models[key]
    return m.predict(X[feats].astype(float).fillna(fill))


def evaluate(f, test_seasons):
    """Honest test: models trained on seasons before the test, compared with simple baselines."""
    rows = []
    models = fit_models(f, test_seasons[0])
    for key, (label, *_rest) in STATS.items():
        d = f[(f["stat"] == key) & f["season"].isin(test_seasons) & (f["n_prior"] >= MIN_GAMES)
              & (f["season_type"] == "REG")].dropna(subset=FEATURES)
        d = d[d["u_ewm"] >= (15 if key == "pass" else 3)]  # regulars, not spot players
        pred = _predict(models, key, d)
        rows.append({"stat": label, "players": len(d),
                     "our_miss": np.mean(np.abs(pred - d["y"])),
                     "recent_avg_miss": np.mean(np.abs(d["y_ewm"] - d["y"])),
                     "season_avg_miss": np.mean(np.abs(d.groupby(["player_id", "season"])["y"].transform(
                         lambda x: x.shift().expanding().mean()).fillna(d["y_ewm"]) - d["y"]))})
    return pd.DataFrame(rows)


def project_week(f, allowed, league, models, week_games, injuries=None, all_games=None):
    """Projections for the players likely to play in these (unplayed) games."""
    last = f.sort_values("order").groupby(["player_id", "stat"]).tail(1)
    teams_recent = f.groupby("team")["order"].apply(lambda x: sorted(set(x))[-3:] if len(x) else [])
    opp_state = allowed.sort_values("order").groupby(["opponent_team", "dgroup"]).tail(1).set_index(
        ["opponent_team", "dgroup"])["opp_next"]
    out = []
    lines = _implied_points(week_games).set_index(["game_id", "team"])
    pass_state = last[last["stat"] == "pass"].drop_duplicates("player_id").set_index("player_id")
    team_last_order = f.groupby("team")["order"].max()
    new_qb = _starters(all_games).set_index(["game_id", "team"])["qb_new"].to_dict() if all_games is not None else {}
    for g in week_games.itertuples():
        for team, opp, qb_id in ((g.home_team, g.away_team, g.home_qb_id), (g.away_team, g.home_team, g.away_qb_id)):
            recent = set(teams_recent.get(team, []))
            cand = last[(last["team"] == team) & last["order"].isin(recent) & (last["n_prior"] >= MIN_GAMES - 1)]
            out_names = set()
            if injuries is not None and len(injuries):
                inj = injuries[(injuries["team"] == team) & injuries["report_status"].isin(["Out", "Doubtful"])]
                out_names = set(inj["full_name"].str.lower())
            picks = []
            qb = cand[(cand["stat"] == "pass")]
            qb = qb[qb["player_id"] == qb_id] if isinstance(qb_id, str) and (qb["player_id"] == qb_id).any() else \
                qb.sort_values("u_next", ascending=False).head(1)
            picks.append(qb)
            picks.append(cand[cand["stat"] == "rush"].sort_values("u_next", ascending=False).head(2))
            picks.append(cand[cand["stat"] == "rec"].sort_values("u_next", ascending=False).head(4))
            sel = pd.concat(picks)
            sel = sel[~sel["player_display_name"].str.lower().isin(out_names)]
            if sel.empty:
                continue
            # A backup QB listed as this week's starter: his history is mostly short relief appearances, so project a
            # full game (starter-level attempts) at his own yards per attempt, pulled toward a typical backup's 6.0.
            sel = sel.copy()
            qb_low = (sel["stat"] == "pass") & (sel["u_next"] < 20)
            if qb_low.any():
                ypa = sel.loc[qb_low, "y_next"] / sel.loc[qb_low, "u_next"].clip(lower=1)
                w = (sel.loc[qb_low, "u_next"] / 30).clip(0, 1)
                sel.loc[qb_low, "u_next"] = 31.0
                sel.loc[qb_low, "y_next"] = 31.0 * (w * ypa + (1 - w) * 6.0)
            imp, spr = lines.loc[(g.game_id, team)] if (g.game_id, team) in lines.index else (22.0, 0.0)
            # starting QB context
            starter = qb_id if isinstance(qb_id, str) else (qb["player_id"].iloc[0] if len(qb) else None)
            if starter in pass_state.index:
                qs = pass_state.loc[starter]
                qb_ypa = float(_shrunk_ypa(pd.Series([qs["y_next"]]), pd.Series([qs["u_next"]])).iloc[0])
            else:
                qb_ypa = 6.0
            qb_new = new_qb.get((g.game_id, team), 0)
            # carries left behind by running backs who played the team's last game but are out this week
            rbs_last = last[(last["team"] == team) & (last["stat"] == "rush") & (last["order"] == team_last_order.get(team))]
            is_out = rbs_last["player_display_name"].str.lower().isin(out_names)
            vacated = float(rbs_last.loc[is_out & (rbs_last["u_next"] >= 5), "u_next"].sum())
            active_carries = float(rbs_last.loc[~is_out, "u_next"].sum()) or 1.0
            X = pd.DataFrame({
                "y_ewm": sel["y_next"].values, "u_ewm": sel["u_next"].values,
                "opp_allowed": [opp_state.get((opp, dg), league.get(dg, np.nan)) for dg in sel["dgroup"]],
                "implied": imp, "spread": spr, "is_te": sel["is_te"].values, "is_rb": sel["is_rb"].values,
                "qb_ypa": qb_ypa, "qb_new": qb_new,
                "vacated": np.where(sel["stat"] == "rush", vacated, 0.0),
                "vac_share": np.where(sel["stat"] == "rush", vacated * sel["u_next"].values / active_carries, 0.0)})
            for (_, r), (_, x) in zip(sel.iterrows(), X.iterrows()):
                proj = float(_predict(models, r["stat"], x.to_frame().T)[0])
                out.append({"game_id": g.game_id, "team": team, "opp": opp, "player": r["player_display_name"],
                            "position": r["position"], "stat": r["stat"], "label": STATS[r["stat"]][0],
                            "proj": max(proj, 0.0), "recent": float(r["y_next"])})
    return pd.DataFrame(out)


def main():
    from data import load_games, load_player_stats
    games = load_games()
    stats = load_player_stats(games["season"].unique())
    f, allowed, league = player_features(stats, games)
    full = sorted(s for s in games["season"].unique() if (games[(games["season"] == s)]["game_type"] == "SB").any()
                  and games[(games["season"] == s) & (games["game_type"] == "SB")]["home_score"].notna().any())
    test = full[-4:]
    ev = evaluate(f, test)
    print(f"\nHonest test {test[0]}-{test[-1]} (models trained only on earlier seasons). Average miss, in yards:\n")
    print(ev.to_string(index=False, formatters={c: "{:.1f}".format for c in ["our_miss", "recent_avg_miss", "season_avg_miss"]}))


if __name__ == "__main__":
    main()


def weekly_projections(season, week):
    """Projections for every game in (season, week) that hasn't been played yet, plus the honest-test summary."""
    from data import load_games, load_injuries, load_player_stats
    games = load_games()
    stats = load_player_stats(games["season"].unique())
    f, allowed, league = player_features(stats, games)
    models = fit_models(f, season + 1)  # features only use earlier games, so this season's finished games can train too
    wk = games[(games["season"] == season) & (games["week"] == week) & games["home_score"].isna()]
    inj = load_injuries([season])
    inj = inj[inj["week"] == week] if len(inj) else inj
    proj = project_week(f, allowed, league, models, wk, inj, games) if len(wk) else pd.DataFrame()
    return proj
