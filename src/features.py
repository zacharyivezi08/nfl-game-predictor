"""Turn raw game results into model features.

Every feature for a game is built ONLY from games played before it,
so the model never "peeks" at the result it is trying to predict.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

ELO_START = 1500
ELO_K = 20
ELO_HOME_ADV = 55          # home field is worth ~2 points in Elo terms
ELO_SEASON_REVERT = 1 / 3  # pull ratings 1/3 back toward average each offseason
FORM_WINDOW = 5            # "recent form" = last 5 games

QB_WINDOW = 16             # QB rating uses his last 16 starts (across seasons)
QB_PRIOR_EPA = -0.05       # unknown/new QBs start slightly below average...
QB_PRIOR_DROPBACKS = 150   # ...until they have ~150 dropbacks of their own

INDOOR = {"dome", "closed"}

# Feature groups (train.py tests which groups actually help)
BASE_FEATURES = [
    "elo_diff",          # overall team strength (home minus away)
    "point_diff_form",   # avg margin over last 5 games
    "offense_form",      # avg points scored over last 5
    "defense_form",      # avg points allowed over last 5 (lower is better)
    "win_pct_form",      # win % over last 5
    "rest_diff",         # extra days of rest for home team
    "home_field",        # 1 = true home game, 0 = neutral site
    "div_game",          # 1 = divisional rivalry
]
QB_FEATURES = ["qb_diff"]                       # starting QB EPA per dropback (home minus away)
WEATHER_FEATURES = ["wind", "cold", "dome_team_in_cold"]
FEATURE_GROUPS = {"base": BASE_FEATURES, "qb": QB_FEATURES, "weather": WEATHER_FEATURES}

# Short, readable names used when explaining predictions
FEATURE_LABELS = {
    "elo_diff": "Team strength (Elo)",
    "point_diff_form": "Recent point differential",
    "offense_form": "Recent offense",
    "defense_form": "Recent defense",
    "win_pct_form": "Recent win %",
    "rest_diff": "Rest advantage",
    "home_field": "Home field",
    "div_game": "Division game",
    "qb_diff": "Starting QB",
    "wind": "Wind",
    "cold": "Cold weather",
    "dome_team_in_cold": "Dome team in the cold",
}


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def _add_weather(games: pd.DataFrame) -> pd.DataFrame:
    """Wind, cold, and whether a dome team is playing outdoors in the cold.

    Future games don't have weather yet, so missing temps are filled with that
    stadium's typical temperature for the month (its "climate").
    """
    g = games.copy()
    indoor = g["roof"].isin(INDOOR)
    g["month"] = g["gameday"].dt.month
    climate = g[~indoor].groupby(["home_team", "month"])["temp"].transform("mean")
    temp = g["temp"].fillna(climate).fillna(g.loc[~indoor, "temp"].mean())
    wind = g["wind"].fillna(g.loc[~indoor, "wind"].median())
    g["wind"] = np.where(indoor, 0, wind)
    g["cold"] = np.where(indoor, 0, np.clip(45 - temp, 0, None))  # degrees below 45F

    # A "dome team" plays most home games indoors that season
    home_roofs = g.groupby(["season", "home_team"])["roof"].agg(lambda r: r.isin(INDOOR).mean() > 0.5)
    home_dome = g.set_index(["season", "home_team"]).index.map(home_roofs).fillna(False).astype(int)
    away_dome = g.set_index(["season", "away_team"]).index.map(home_roofs).fillna(False).astype(int)
    is_cold = (g["cold"] > 0).astype(int)
    # +1 if the AWAY team is a dome team stuck in the cold (helps home), -1 if the home team is
    g["dome_team_in_cold"] = is_cold * (np.asarray(away_dome) - np.asarray(home_dome))
    return g


def build_features(games: pd.DataFrame, qb_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per game with pre-game features and (if played) the result."""
    games = _add_weather(games)

    # Index QB stats by game so we can update each QB's history after every game
    qb_by_game = defaultdict(list)
    if qb_stats is not None and len(qb_stats):
        for r in qb_stats.itertuples(index=False):
            if pd.notna(r.passing_epa):
                qb_by_game[r.game_id].append((r.player_id, r.team, r.dropbacks, r.passing_epa))

    elo = defaultdict(lambda: ELO_START)
    scored = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    allowed = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    qb_hist = defaultdict(lambda: deque(maxlen=QB_WINDOW))  # player_id -> [(dropbacks, epa)]
    last_qb = {}                                            # team -> most recent starter
    current_season = None
    rows = []

    def avg(d):
        return float(np.mean(d)) if len(d) else 0.0

    def qb_rating(qb_id, team):
        qb_id = qb_id if isinstance(qb_id, str) else last_qb.get(team)
        hist = qb_hist[qb_id] if qb_id else []
        db = sum(h[0] for h in hist)
        epa = sum(h[1] for h in hist)
        return (epa + QB_PRIOR_EPA * QB_PRIOR_DROPBACKS) / (db + QB_PRIOR_DROPBACKS)

    for g in games.itertuples(index=False):
        # New season: regress everyone's Elo toward the mean
        if g.season != current_season:
            for team in list(elo):
                elo[team] = elo[team] + ELO_SEASON_REVERT * (ELO_START - elo[team])
            current_season = g.season

        h, a = g.home_team, g.away_team
        neutral = 1 if g.location == "Neutral" else 0
        home_adv = 0 if neutral else ELO_HOME_ADV

        h_pd = [s - al for s, al in zip(scored[h], allowed[h])]
        a_pd = [s - al for s, al in zip(scored[a], allowed[a])]
        h_wins = [1.0 if x > 0 else 0.5 if x == 0 else 0.0 for x in h_pd]
        a_wins = [1.0 if x > 0 else 0.5 if x == 0 else 0.0 for x in a_pd]
        h_qb, a_qb = qb_rating(g.home_qb_id, h), qb_rating(g.away_qb_id, a)

        rows.append({
            "game_id": g.game_id,
            "season": g.season,
            "week": g.week,
            "game_type": g.game_type,
            "gameday": g.gameday,
            "home_team": h,
            "away_team": a,
            "home_qb": g.home_qb_name,
            "away_qb": g.away_qb_name,
            "home_score": g.home_score,
            "away_score": g.away_score,
            "spread_line": g.spread_line,
            "home_moneyline": g.home_moneyline,
            "away_moneyline": g.away_moneyline,
            "elo_diff": elo[h] - elo[a],
            "point_diff_form": avg(h_pd) - avg(a_pd),
            "offense_form": avg(scored[h]) - avg(scored[a]),
            "defense_form": avg(allowed[h]) - avg(allowed[a]),
            "win_pct_form": avg(h_wins) - avg(a_wins),
            "rest_diff": (g.home_rest - g.away_rest) if pd.notna(g.home_rest) and pd.notna(g.away_rest) else 0,
            "home_field": 1 - neutral,
            "div_game": int(g.div_game) if pd.notna(g.div_game) else 0,
            "qb_diff": h_qb - a_qb,
            "wind": g.wind,
            "cold": g.cold,
            "dome_team_in_cold": g.dome_team_in_cold,
        })

        # Game not played yet -> nothing to update
        if pd.isna(g.home_score) or pd.isna(g.away_score):
            continue

        # Update Elo (bigger wins move ratings a bit more)
        margin = g.home_score - g.away_score
        outcome = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        expected = _elo_expected(elo[h] + home_adv, elo[a])
        shift = ELO_K * np.log(abs(margin) + 1) * (outcome - expected)
        elo[h] += shift
        elo[a] -= shift

        scored[h].append(g.home_score); allowed[h].append(g.away_score)
        scored[a].append(g.away_score); allowed[a].append(g.home_score)

        # Update QB histories (the main passer for each team)
        for team, qb_id in ((h, g.home_qb_id), (a, g.away_qb_id)):
            if isinstance(qb_id, str):
                last_qb[team] = qb_id
        for pid, team, db, epa in qb_by_game.get(g.game_id, []):
            if db > 0:
                qb_hist[pid].append((db, epa))

    df = pd.DataFrame(rows)
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["home_win"] = np.where(df["played"], (df["home_score"] > df["away_score"]).astype(float), np.nan)
    df["tie"] = df["played"] & (df["home_score"] == df["away_score"])
    df["vegas_prob"] = vegas_home_prob(df)
    return df


def vegas_home_prob(df: pd.DataFrame) -> pd.Series:
    """Vegas' implied home win probability.

    Uses moneylines (with the bookmaker's cut removed) when available,
    otherwise converts the point spread (positive = home favored).
    """
    def ml_to_prob(ml):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))

    h, a = ml_to_prob(df["home_moneyline"]), ml_to_prob(df["away_moneyline"])
    from_ml = pd.Series(h / (h + a), index=df.index)
    from_spread = 1 / (1 + np.exp(-df["spread_line"] / 6.5))
    return from_ml.where(df["home_moneyline"].notna() & df["away_moneyline"].notna(), from_spread)
