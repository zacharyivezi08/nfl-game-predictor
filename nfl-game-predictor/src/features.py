"""Turn raw game results into model features.

Every feature for a game is built ONLY from games played before it,
so the model never "peeks" at the result it is trying to predict.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

ELO_START = 1500
ELO_K = 20
ELO_HOME_ADV = 55        # home field is worth ~2 points in Elo terms
ELO_SEASON_REVERT = 1 / 3  # pull ratings 1/3 back toward average each offseason
FORM_WINDOW = 5          # "recent form" = last 5 games

FEATURES = [
    "elo_diff",          # overall team strength (home minus away)
    "point_diff_form",   # avg margin over last 5 games
    "offense_form",      # avg points scored over last 5
    "defense_form",      # avg points allowed over last 5 (lower is better)
    "win_pct_form",      # win % over last 5
    "rest_diff",         # extra days of rest for home team
    "home_field",        # 1 = true home game, 0 = neutral site
    "div_game",          # 1 = divisional rivalry
]

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
}


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Return one row per game with pre-game features and (if played) the result."""
    elo = defaultdict(lambda: ELO_START)
    scored = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    allowed = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    current_season = None
    rows = []

    def avg(d):
        return float(np.mean(d)) if len(d) else 0.0

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

        row = {
            "game_id": g.game_id,
            "season": g.season,
            "week": g.week,
            "game_type": g.game_type,
            "gameday": g.gameday,
            "home_team": h,
            "away_team": a,
            "home_score": g.home_score,
            "away_score": g.away_score,
            "elo_diff": elo[h] - elo[a],
            "point_diff_form": avg(h_pd) - avg(a_pd),
            "offense_form": avg(scored[h]) - avg(scored[a]),
            "defense_form": avg(allowed[h]) - avg(allowed[a]),
            "win_pct_form": avg(h_wins) - avg(a_wins),
            "rest_diff": (g.home_rest - g.away_rest) if pd.notna(g.home_rest) and pd.notna(g.away_rest) else 0,
            "home_field": 1 - neutral,
            "div_game": int(g.div_game) if pd.notna(g.div_game) else 0,
        }
        rows.append(row)

        # Game not played yet -> nothing to update
        if pd.isna(g.home_score) or pd.isna(g.away_score):
            continue

        # Update Elo (bigger wins move ratings a bit more)
        margin = g.home_score - g.away_score
        outcome = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        expected = _elo_expected(elo[h] + home_adv, elo[a])
        mov_mult = np.log(abs(margin) + 1)
        shift = ELO_K * mov_mult * (outcome - expected)
        elo[h] += shift
        elo[a] -= shift

        scored[h].append(g.home_score); allowed[h].append(g.away_score)
        scored[a].append(g.away_score); allowed[a].append(g.home_score)

    df = pd.DataFrame(rows)
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["home_win"] = np.where(df["played"], (df["home_score"] > df["away_score"]).astype(float), np.nan)
    df["tie"] = df["played"] & (df["home_score"] == df["away_score"])
    return df
