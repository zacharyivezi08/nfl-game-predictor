"""How much talent is each team missing this week because of injuries?

For every player listed on the official injury report, we look up how big a role
he's had lately: his average share of snaps over the team's last 4 games. Missing
an every-down left tackle (share ~1.0) matters more than a backup (share ~0.1).
A player who's been out for weeks has a low recent share, because the team has
already adjusted and that's already reflected in its recent stats.

QBs are skipped here since the starting-QB feature already handles them.
"""
import re

import pandas as pd

STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.8, "Questionable": 0.25}  # chance-ish the player sits
WINDOW = 4


def _name_key(name) -> str:
    name = str(name).lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", name)
    return re.sub(r"[^a-z]", "", name)


def injury_loads(injuries: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team): offense and defense snap-share missing to injury."""
    if injuries.empty or snaps.empty:
        return pd.DataFrame(columns=["season", "week", "team", "inj_off", "inj_def"])

    snaps = snaps[snaps["position"] != "QB"].copy()
    snaps["key"] = snaps["player"].map(_name_key)
    snaps["order"] = snaps["season"] * 100 + snaps["week"]

    # Each player's recent role: rolling average share over the team's last 4 games (0 if he sat)
    importance = []
    for team, t in snaps.groupby("team"):
        wide = t.pivot_table(index="order", columns="key", values="share", aggfunc="max").fillna(0.0)
        recent = wide.rolling(WINDOW, min_periods=1).mean()   # role AFTER each game
        side = t.groupby("key")["side"].agg(lambda s: s.mode().iat[0])
        long = recent.stack().rename("importance").reset_index()
        long["team"] = team
        long["side"] = long["key"].map(side)
        importance.append(long)
    imp = pd.concat(importance, ignore_index=True)

    inj = injuries[injuries["position"] != "QB"].copy()
    inj["key"] = inj["full_name"].map(_name_key)
    inj["order"] = inj["season"] * 100 + inj["week"]
    inj["weight"] = inj["report_status"].map(STATUS_WEIGHT).fillna(0)

    # Use each player's role as of the team's most recent game BEFORE this week
    inj = inj.sort_values("order")
    imp = imp.sort_values("order")
    merged = pd.merge_asof(inj, imp.rename(columns={"order": "imp_order"}),
                           left_on="order", right_on="imp_order", by=["team", "key"],
                           allow_exact_matches=False)
    merged = merged.dropna(subset=["importance"])
    merged["lost"] = merged["weight"] * merged["importance"]
    out = merged.pivot_table(index=["season", "week", "team"], columns="side", values="lost",
                             aggfunc="sum", fill_value=0.0).reset_index()
    out = out.rename(columns={"off": "inj_off", "def": "inj_def"})
    for c in ("inj_off", "inj_def"):
        if c not in out:
            out[c] = 0.0
    return out[["season", "week", "team", "inj_off", "inj_def"]]
