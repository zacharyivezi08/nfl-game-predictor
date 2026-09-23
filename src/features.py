"""Turn raw game results into model features.

Every feature for a game is built ONLY from games played before it,
so the model never "peeks" at the result it is trying to predict.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

ELO_START = 1500
ELO_K = 20                 # how fast ratings move
ELO_HOME_ADV = 55          # home field edge in Elo points (~2 points on the scoreboard)
ELO_SEASON_REVERT = 1 / 3  # pull ratings 1/3 back toward average each offseason
FORM_WINDOW = 5            # "recent form" = last 5 games
# These settings were tuned in step 3 (see README): the "best" tuned values only improved
# the validation score by ~0.001 and made the untouched test seasons WORSE, so the simpler
# standard values are kept.

QB_WINDOW = 16             # QB rating uses his last 16 games played (across seasons)
QB_PRIOR_EPA = -0.05       # unknown/new QBs start slightly below average...
QB_PRIOR_DROPBACKS = 150   # ...until they have ~150 dropbacks of their own

EPA_ALPHA = 0.12           # each new game counts 12% in the running efficiency average
EPA_SEASON_REVERT = 0.3    # pull efficiency 30% back toward average each offseason
SR_MEAN = 0.44             # typical success rate (share of plays that "succeed")

INDOOR = {"dome", "closed"}

# Home stadium time zone (hours from UTC, standard time). Arizona doesn't do daylight saving,
# so during the season it's usually on Pacific time; -7.5 splits the difference.
TEAM_TZ = {
    "BUF": -5, "MIA": -5, "NE": -5, "NYJ": -5, "BAL": -5, "CIN": -5, "CLE": -5, "PIT": -5,
    "IND": -5, "JAX": -5, "NYG": -5, "PHI": -5, "WAS": -5, "DET": -5, "ATL": -5, "CAR": -5, "TB": -5,
    "HOU": -6, "TEN": -6, "KC": -6, "DAL": -6, "CHI": -6, "GB": -6, "MIN": -6, "NO": -6,
    "DEN": -7, "ARI": -7.5, "LV": -8, "LAC": -8, "SF": -8, "SEA": -8, "LA": -8,
}


def team_tz(team, season):
    if team == "LA" and season < 2016:  # the Rams were in St. Louis (Central) until 2016
        return -6
    return TEAM_TZ.get(team, -5)

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
EPA_FEATURES = [
    "off_epa_diff",      # offense EPA per play (home minus away)
    "def_epa_diff",      # defense EPA allowed per play (home minus away, lower is better)
    "net_sr_diff",       # (offense success rate - defense success rate allowed), home minus away
]
TRAVEL_FEATURES = [
    "tz_travel",         # time zones the away team crossed to get here (0-3)
    "early_body_clock",  # hours before noon it feels like to the away team (West Coast team at 1 PM ET = 3)
    "bye_diff",          # coming off a bye week (home minus away)
    "short_week_diff",   # playing on short rest, e.g. Thursday after Sunday (home minus away)
]
INJURY_FEATURES = ["inj_diff"]  # starters' snaps missing to injury, offense + defense (home minus away)
FEATURE_GROUPS = {"base": BASE_FEATURES, "qb": QB_FEATURES, "weather": WEATHER_FEATURES,
                  "epa": EPA_FEATURES, "injuries": INJURY_FEATURES, "travel": TRAVEL_FEATURES}

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
    "off_epa_diff": "Offense efficiency (EPA)",
    "def_epa_diff": "Defense efficiency (EPA)",
    "net_sr_diff": "Success rate",
    "inj_diff": "Injuries",
    "tz_travel": "Travel / time zones",
    "early_body_clock": "Early body-clock kickoff",
    "bye_diff": "Coming off a bye",
    "short_week_diff": "Short week",
}


# Every team's ratings going into each (season, week), plus "latest". Filled by build_features.
# (Kept out of the DataFrame on purpose: pandas would copy it on every operation.)
SNAPSHOTS: dict = {}


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


def _add_travel(g: pd.DataFrame) -> pd.DataFrame:
    """Time zones crossed, body-clock kickoffs, byes and short weeks."""
    neutral = g["location"].eq("Neutral")
    home_tz = np.array([team_tz(t, s) for t, s in zip(g["home_team"], g["season"])])
    away_tz = np.array([team_tz(t, s) for t, s in zip(g["away_team"], g["season"])])
    g["tz_travel"] = np.where(neutral, 0, np.abs(home_tz - away_tz))
    et_hour = pd.to_numeric(g["gametime"].astype(str).str[:2], errors="coerce").fillna(13)
    et_min = pd.to_numeric(g["gametime"].astype(str).str[3:5], errors="coerce").fillna(0)
    away_local = et_hour + et_min / 60 + (away_tz - (-5))
    g["early_body_clock"] = np.where(neutral, 0, np.clip(12 - away_local, 0, None))
    hr, ar = g["home_rest"].fillna(7), g["away_rest"].fillna(7)
    g["bye_diff"] = (hr >= 13).astype(int) - (ar >= 13).astype(int)
    g["short_week_diff"] = (hr <= 5).astype(int) - (ar <= 5).astype(int)
    return g


def build_features(games: pd.DataFrame, qb_stats: pd.DataFrame | None = None,
                   team_epa: pd.DataFrame | None = None,
                   injury_loads: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per game with pre-game features and (if played) the result."""
    games = _add_travel(_add_weather(games))

    # Team efficiency per game: {game_id: {team: (off_epa/play, def_epa/play, off_sr, def_sr)}}
    epa_by_game = defaultdict(dict)
    if team_epa is not None and len(team_epa):
        e = team_epa.dropna(subset=["off_plays", "def_plays"])
        for r in e.itertuples(index=False):
            if r.off_plays > 0 and r.def_plays > 0:
                epa_by_game[r.game_id][r.team] = (r.off_epa / r.off_plays, r.def_epa / r.def_plays,
                                                  r.off_success / r.off_plays, r.def_success / r.def_plays)
    eff = defaultdict(lambda: [0.0, 0.0, SR_MEAN, SR_MEAN])  # team -> running [off_epa, def_epa, off_sr, def_sr]

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
    current_week = None
    rows = []
    snapshots = {}  # (season, week) -> every team's ratings going INTO that week
    recent_totals = deque(maxlen=256)  # last ~1 season of game totals, to track scoring trends

    def avg(d):
        return float(np.mean(d)) if len(d) else 0.0

    def team_state(t):
        pdl = [x - y for x, y in zip(scored[t], allowed[t])]
        return {"elo": elo[t], "pd": avg(pdl), "off": avg(scored[t]), "def": avg(allowed[t]),
                "win_pct": avg([1.0 if x > 0 else 0.5 if x == 0 else 0.0 for x in pdl]),
                "qb": qb_rating(None, t), "qb_id": last_qb.get(t),
                "off_epa": eff[t][0], "def_epa": eff[t][1], "off_sr": eff[t][2], "def_sr": eff[t][3]}

    def qb_rating(qb_id, team):
        qb_id = qb_id if isinstance(qb_id, str) else last_qb.get(team)
        hist = qb_hist[qb_id] if qb_id else []
        db = sum(h[0] for h in hist)
        epa = sum(h[1] for h in hist)
        return (epa + QB_PRIOR_EPA * QB_PRIOR_DROPBACKS) / (db + QB_PRIOR_DROPBACKS)

    for g in games.itertuples(index=False):
        if (g.season, g.week) != (current_season, current_week) and g.season == current_season:
            snapshots[(g.season, g.week)] = {t: team_state(t) for t in list(elo)}
        # New season: regress everyone's Elo toward the mean
        if g.season != current_season:
            for team in list(elo):
                elo[team] = elo[team] + ELO_SEASON_REVERT * (ELO_START - elo[team])
            for team, v in eff.items():
                for i, mean in enumerate((0.0, 0.0, SR_MEAN, SR_MEAN)):
                    v[i] += EPA_SEASON_REVERT * (mean - v[i])
            current_season = g.season
            snapshots[(g.season, g.week)] = {t: team_state(t) for t in list(elo)}
        current_week = g.week

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
            "weekday": g.weekday,
            "gametime": g.gametime,
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
            "tz_travel": g.tz_travel,
            "early_body_clock": g.early_body_clock,
            "bye_diff": g.bye_diff,
            "short_week_diff": g.short_week_diff,
            "off_epa_diff": eff[h][0] - eff[a][0],
            "def_epa_diff": eff[h][1] - eff[a][1],
            "net_sr_diff": (eff[h][2] - eff[h][3]) - (eff[a][2] - eff[a][3]),
            "off_epa_sum": eff[h][0] + eff[a][0],
            "def_epa_sum": eff[h][1] + eff[a][1],
            "off_form_sum": avg(scored[h]) + avg(scored[a]),
            "def_form_sum": avg(allowed[h]) + avg(allowed[a]),
            "league_total_avg": avg(recent_totals) if recent_totals else 42.0,
            "indoor": int(g.roof in INDOOR),
            "total_line": g.total_line,
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

        for team, stats in epa_by_game.get(g.game_id, {}).items():
            v = eff[team]
            for i in range(4):
                v[i] += EPA_ALPHA * (stats[i] - v[i])

        recent_totals.append(g.home_score + g.away_score)
        scored[h].append(g.home_score); allowed[h].append(g.away_score)
        scored[a].append(g.away_score); allowed[a].append(g.home_score)

        # Update QB histories (the main passer for each team)
        for team, qb_id in ((h, g.home_qb_id), (a, g.away_qb_id)):
            if isinstance(qb_id, str):
                last_qb[team] = qb_id
        for pid, team, db, epa in qb_by_game.get(g.game_id, []):
            if db > 0:
                qb_hist[pid].append((db, epa))

    snapshots["latest"] = {t: team_state(t) for t in list(elo)}

    df = pd.DataFrame(rows)
    SNAPSHOTS.clear()
    SNAPSHOTS.update(snapshots)
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["home_win"] = np.where(df["played"], (df["home_score"] > df["away_score"]).astype(float), np.nan)
    df["tie"] = df["played"] & (df["home_score"] == df["away_score"])
    df["vegas_prob"] = vegas_home_prob(df)

    # Injuries: snap share each team is missing this week (0 when no report data, e.g. before 2013)
    for side in ("home", "away"):
        for c in ("inj_off", "inj_def"):
            df[f"{side}_{c}"] = 0.0
    if injury_loads is not None and len(injury_loads):
        key = injury_loads.set_index(["season", "week", "team"])
        for side in ("home", "away"):
            idx = pd.MultiIndex.from_arrays([df["season"], df["week"], df[f"{side}_team"]])
            for c in ("inj_off", "inj_def"):
                df[f"{side}_{c}"] = key[c].reindex(idx).fillna(0.0).values
    df["inj_off_diff"] = df["home_inj_off"] - df["away_inj_off"]
    df["inj_def_diff"] = df["home_inj_def"] - df["away_inj_def"]
    df["inj_diff"] = df["inj_off_diff"] + df["inj_def_diff"]
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
