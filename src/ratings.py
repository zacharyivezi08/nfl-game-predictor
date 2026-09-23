"""Predict ANY matchup (even ones not on the schedule) from each team's current ratings.

Used by the playoff simulator (which needs made-up playoff games) and power rankings.
"""
import numpy as np
import pandas as pd

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}
TEAMS = sorted(TEAM_DIV)


def matchup_row(state: dict, home: str, away: str, neutral: bool = False) -> dict:
    """Build the same features the model uses, for a hypothetical game."""
    h, a = state[home], state[away]
    return {
        "elo_diff": h["elo"] - a["elo"],
        "point_diff_form": h["pd"] - a["pd"],
        "offense_form": h["off"] - a["off"],
        "defense_form": h["def"] - a["def"],
        "win_pct_form": h["win_pct"] - a["win_pct"],
        "rest_diff": 0, "home_field": 0 if neutral else 1,
        "div_game": int(TEAM_DIV[home] == TEAM_DIV[away]),
        "qb_diff": h["qb"] - a["qb"],
        "wind": 0, "cold": 0, "dome_team_in_cold": 0,
        "off_epa_diff": h["off_epa"] - a["off_epa"],
        "def_epa_diff": h["def_epa"] - a["def_epa"],
        "net_sr_diff": (h["off_sr"] - h["def_sr"]) - (a["off_sr"] - a["def_sr"]),
        "inj_diff": 0,
        "tz_travel": 0, "early_body_clock": 0, "bye_diff": 0, "short_week_diff": 0,
    }


def prob_matrix(saved, state: dict, neutral: bool = False) -> pd.DataFrame:
    """P[home][away] = chance the home team wins, for every pair of teams."""
    pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
    X = pd.DataFrame([matchup_row(state, h, a, neutral) for h, a in pairs])[saved["features"]]
    p = saved["model"].predict_proba(X)[:, 1]
    pos = {t: i for i, t in enumerate(TEAMS)}
    M = np.full((len(TEAMS), len(TEAMS)), 0.5)
    for (h, a), v in zip(pairs, p):
        M[pos[h], pos[a]] = v
    if neutral:  # a neutral field should be symmetric: average both "home" orientations
        M = (M + (1 - M.T)) / 2
        np.fill_diagonal(M, 0.5)
    return pd.DataFrame(M, index=TEAMS, columns=TEAMS)


def power_ratings(saved, state: dict) -> pd.Series:
    """Chance each team beats an average NFL team on a neutral field (0-1)."""
    P = prob_matrix(saved, state, neutral=True)
    M = P.values.copy()
    np.fill_diagonal(M, np.nan)
    return pd.Series(np.nanmean(M, axis=1), index=P.index).sort_values(ascending=False)


def team_records(df: pd.DataFrame, season: int) -> dict:
    """{team: "W-L" or "W-L-T"} from played regular-season games."""
    reg = df[(df["season"] == season) & (df["game_type"] == "REG") & df["played"]]
    rec = {t: [0, 0, 0] for t in TEAMS}
    for r in reg.itertuples():
        if r.home_score > r.away_score:
            rec[r.home_team][0] += 1; rec[r.away_team][1] += 1
        elif r.home_score < r.away_score:
            rec[r.away_team][0] += 1; rec[r.home_team][1] += 1
        else:
            rec[r.home_team][2] += 1; rec[r.away_team][2] += 1
    return {t: f"{w}-{l}" + (f"-{ti}" if ti else "") for t, (w, l, ti) in rec.items()}


def power_rankings(saved, df: pd.DataFrame, snapshots: dict, season: int) -> pd.DataFrame:
    """Rank all 32 teams by the model's rating now vs. one week ago.

    rating = chance of beating an average NFL team on a neutral field.
    """
    weeks = sorted(w for (s, w) in (k for k in snapshots if k != "latest") if s == season)
    now_state = snapshots["latest"]
    played_weeks = sorted(df[(df["season"] == season) & df["played"]]["week"].unique())
    # ratings going into the most recent played week = "last week's" rankings
    prev_key = (season, played_weeks[-1]) if played_weeks else None
    now = power_ratings(saved, now_state)
    out = pd.DataFrame({"team": now.index, "rating": now.values})
    out["rank"] = np.arange(1, len(out) + 1)
    if prev_key in snapshots:
        prev = power_ratings(saved, snapshots[prev_key])
        prev_rank = {t: i + 1 for i, t in enumerate(prev.index)}
        out["prev_rank"] = out["team"].map(prev_rank)
        out["change"] = out["prev_rank"] - out["rank"]
    else:
        out["prev_rank"] = np.nan
        out["change"] = 0
    off = pd.Series({t: now_state[t]["off_epa"] for t in TEAMS}).rank(ascending=False).astype(int)
    dfn = pd.Series({t: now_state[t]["def_epa"] for t in TEAMS}).rank(ascending=True).astype(int)
    out["off_rank"] = out["team"].map(off)
    out["def_rank"] = out["team"].map(dfn)
    out["record"] = out["team"].map(team_records(df, season))
    return out[["rank", "team", "record", "rating", "change", "prev_rank", "off_rank", "def_rank"]]


def main():
    from features import SNAPSHOTS
    from predict import load_model
    from train import load_all
    df = load_all()
    season = int(df["season"].max())
    pr = power_rankings(load_model(), df, SNAPSHOTS, season)
    arrow = lambda c: f"▲{int(c)}" if c > 0 else (f"▼{-int(c)}" if c < 0 else "–")
    pr["move"] = pr["change"].map(arrow)
    pr["rating"] = pr["rating"].map("{:.0%}".format)
    print(f"\n{season} power rankings (rating = chance to beat an average team on a neutral field)\n")
    print(pr[["rank", "move", "team", "record", "rating", "off_rank", "def_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
