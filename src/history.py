"""Pick tracker: saves every pick the site made, so the record can't be rewritten later.

docs/data/picks.json keeps one entry per game. Before kickoff, the entry is updated every run
(new injury reports, new Vegas line, ...). Once the game kicks off it's frozen, and the final
score is filled in when it's available. Games from before the tracker existed are "backfilled"
with what the model would have said (it was trained only on earlier seasons, so it's still fair),
and are marked that way.

docs/data/odds_history.json keeps each team's playoff odds every week, for the team pages.

Usage:
    python src/history.py        # print the season record + biggest upsets called
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"
PICKS_PATH = DATA / "picks.json"
ODDS_PATH = DATA / "odds_history.json"
ET = ZoneInfo("America/New_York")


def _load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(path, obj):
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))


def kickoff(r):
    t = r.gametime if isinstance(r.gametime, str) and ":" in r.gametime else "13:00"
    return datetime.strptime(f"{r.gameday:%Y-%m-%d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def _r(x, n=3):
    return None if pd.isna(x) else round(float(x), n)


def update_picks(preds: pd.DataFrame, now: datetime | None = None) -> dict:
    """Merge this run's predictions into the saved pick history and return it."""
    now = now or datetime.now(ET)
    hist = _load(PICKS_PATH)
    for r in preds.itertuples():
        old = hist.get(r.game_id)
        started = now >= kickoff(r) or bool(r.played)
        if old is None or not started:
            hist[r.game_id] = {
                "season": int(r.season), "week": int(r.week), "type": r.game_type,
                "date": f"{r.gameday:%Y-%m-%d}", "home": r.home_team, "away": r.away_team,
                "home_prob": _r(r.home_prob), "pick": r.pick,
                "vegas_prob": _r(r.vegas_prob), "vegas_pick": r.vegas_pick if isinstance(r.vegas_pick, str) else None,
                "blend_prob": _r(r.blend_prob), "blend_pick": r.blend_pick,
                "saved": now.strftime("%Y-%m-%d %H:%M ET"),
                "source": "backfill" if old is None and started else "live",
            }
        e = hist[r.game_id]
        if r.played:
            e["final"] = [int(r.away_score), int(r.home_score)]
            winner = None if r.tie else (r.home_team if r.home_score > r.away_score else r.away_team)
            e["winner"] = winner
            for k in ("pick", "vegas_pick", "blend_pick"):
                e[k.replace("pick", "correct")] = None if winner is None or e.get(k) is None else e[k] == winner
    _save(PICKS_PATH, hist)
    return hist


def picks_frame(hist: dict, season: int) -> pd.DataFrame:
    rows = [{"game_id": g, **v} for g, v in hist.items() if v["season"] == season]
    return pd.DataFrame(rows).sort_values(["week", "date", "game_id"]) if rows else pd.DataFrame()


def weekly_record(p: pd.DataFrame) -> pd.DataFrame:
    """Right/wrong per week for the model, Vegas favorite and blend."""
    done = p[p.get("correct", pd.Series(dtype=object)).isin([True, False])] if len(p) else p
    if done.empty:
        return pd.DataFrame()
    out = []
    for w, g in done.groupby("week"):
        row = {"week": int(w), "games": len(g)}
        for k in ("correct", "vegas_correct", "blend_correct"):
            row[k] = int((g[k] == True).sum())  # noqa: E712
            row[k + "_n"] = int(g[k].isin([True, False]).sum())
        out.append(row)
    return pd.DataFrame(out)


def upsets_called(p: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Games the model got right while picking AGAINST the Vegas favorite, biggest underdogs first."""
    if p.empty or "correct" not in p:
        return p
    u = p[(p["correct"] == True) & p["vegas_pick"].notna() & (p["pick"] != p["vegas_pick"])].copy()  # noqa: E712
    u["vegas_fav_conf"] = [vp if vk == h else 1 - vp for vp, vk, h in zip(u["vegas_prob"], u["vegas_pick"], u["home"])]
    return u.sort_values("vegas_fav_conf", ascending=False).head(n)


def worst_misses(p: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if p.empty or "correct" not in p:
        return p
    m = p[p["correct"] == False].copy()  # noqa: E712
    m["conf"] = [hp if pk == h else 1 - hp for hp, pk, h in zip(m["home_prob"], m["pick"], m["home"])]
    return m.sort_values("conf", ascending=False).head(n)


def update_odds(odds: pd.DataFrame, season: int, week: int) -> dict:
    """Save this week's playoff + Super Bowl odds for every team (overwrites the same week)."""
    hist = _load(ODDS_PATH)
    hist.setdefault(str(season), {})[str(week)] = {
        r.team: {"playoffs": round(float(r.playoffs), 4), "super_bowl": round(float(r.super_bowl), 4),
                 "proj_wins": round(float(r.proj_wins), 2)} for r in odds.itertuples()}
    _save(ODDS_PATH, hist)
    return hist


def main():
    hist = _load(PICKS_PATH)
    if not hist:
        print("No saved picks yet. Run python src/build_site.py first.")
        return
    season = max(v["season"] for v in hist.values())
    p = picks_frame(hist, season)
    wk = weekly_record(p)
    print(f"\n{season} pick record by week\n")
    for r in wk.itertuples():
        print(f"  Week {r.week:>2}: model {r.correct}-{r.correct_n - r.correct}   "
              f"Vegas {r.vegas_correct}-{r.vegas_correct_n - r.vegas_correct}   "
              f"blend {r.blend_correct}-{r.blend_correct_n - r.blend_correct}")
    print("\nBiggest upsets called (model beat the Vegas favorite)")
    for r in upsets_called(p).itertuples():
        print(f"  Wk {r.week} {r.away} @ {r.home}: picked {r.pick}, Vegas had {r.vegas_pick} {r.vegas_fav_conf:.0%}"
              f" · final {r.final[0]}-{r.final[1]}")


if __name__ == "__main__":
    main()
