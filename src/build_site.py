"""Build a weekly picks web page at docs/index.html (hosted free with GitHub Pages).

Usage:
    python src/build_site.py            # build from cached data
    python src/build_site.py --refresh  # pull newest results first
"""
import argparse
import html
from datetime import datetime
from pathlib import Path

from predict import best_picks, load_model, pick_week, predict_games, season_record, time_slot
from train import load_all

DOCS = Path(__file__).resolve().parent.parent / "docs"

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14171c;--muted:#5d6570;--line:#e3e6ea;
--home:#1f6feb;--away:#d9480f;--good:#1a7f37;--bad:#c2255c;--warn:#9a6700;--warnbg:#fff4d6}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;--muted:#8b949e;
--line:#30363d;--home:#4c8dff;--away:#ff7b45;--good:#3fb950;--bad:#f778ba;--warn:#e3b341;--warnbg:#2d2410}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px}
.sub{color:var(--muted);margin:0 0 20px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.stat b{display:block;font-size:24px}.stat span{color:var(--muted);font-size:13px}
.game{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:10px 0}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
.teams{font-weight:700;font-size:17px}.teams small{color:var(--muted);font-weight:400}
.pick{font-weight:700}.bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin:10px 0 8px;background:var(--line)}
.bar i{display:block}.a{background:var(--away)}.h{background:var(--home)}
.meta{color:var(--muted);font-size:13px}.why{font-size:13px;margin-top:6px}
.tag{font-size:12px;font-weight:700;padding:2px 8px;border-radius:999px;background:var(--warnbg);color:var(--warn)}
.best{border:2px solid var(--home)}.star{font-size:12px;font-weight:700;padding:2px 8px;border-radius:999px;background:var(--home);color:#fff}
.slots{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.slot{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.slot span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.slot b{display:block;font-size:20px;margin:2px 0}.slot small{color:var(--muted)}
h3{font-size:14px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin:22px 0 4px}
.ok{color:var(--good);font-weight:700}.no{color:var(--bad);font-weight:700}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
td,th{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;font-size:14px}
th{color:var(--muted);font-weight:600}footer{color:var(--muted);font-size:13px;margin-top:40px}
footer a{color:var(--home)}
"""


def slot_card(r):
    opp = r.home_team if r.pick == r.away_team else r.away_team
    n = f"best of {r.games_in_slot} games" if r.games_in_slot > 1 else "only game"
    return (f'<div class="slot"><span>{html.escape(r.slot)}</span><b>{r.pick} {r.confidence:.0%}</b>'
            f'<small>over {opp} · {n}</small></div>')


def game_card(r, is_best=False):
    e = html.escape
    away_pct = round((1 - r.home_prob) * 100)
    disagree = isinstance(r.vegas_pick, str) and r.vegas_pick != r.pick
    vegas = f"Vegas: {e(r.vegas_pick)} {r.vegas_conf:.0%}" if isinstance(r.vegas_pick, str) else "Vegas: no line yet"
    qbs = f"{e(str(r.away_qb))} vs {e(str(r.home_qb))}" if isinstance(r.home_qb, str) else ""
    why = " · ".join(f"{e(lbl)} → {e(team)}" for lbl, team in r.reasons)
    return f"""
<div class="game{' best' if is_best else ''}">
  <div class="row"><div class="teams">{e(r.away_team)} <small>@</small> {e(r.home_team)}</div>
  <div>{'<span class="star">★ Top pick of slot</span> ' if is_best and r.games_in_slot > 1 else ''}{'<span class="tag">Disagrees with Vegas</span> ' if disagree else ''}<span class="pick">{e(r.pick)} {r.confidence:.0%}</span></div></div>
  <div class="bar" title="{e(r.away_team)} {away_pct}% / {e(r.home_team)} {100 - away_pct}%">
    <i class="a" style="width:{away_pct}%"></i><i class="h" style="width:{100 - away_pct}%"></i></div>
  <div class="row meta"><span>{e(r.away_team)} {away_pct}% · {e(r.home_team)} {100 - away_pct}%</span><span>{vegas}</span></div>
  <div class="meta">{qbs} · {r.gameday:%a %b %-d}</div>
  <div class="why"><b>Why:</b> {why}</div>
</div>"""


def results_table(preds):
    rows = []
    for _, r in preds.iterrows():
        if not isinstance(r.winner, str):
            continue
        mark = '<span class="ok">✓</span>' if r.correct else '<span class="no">✗</span>'
        vmark = "" if r.vegas_correct is None else ('<span class="ok">✓</span>' if r.vegas_correct else '<span class="no">✗</span>')
        rows.append(f"<tr><td>{r.away_team} @ {r.home_team}</td><td>{int(r.away_score)}-{int(r.home_score)}</td>"
                    f"<td>{r.pick} {r.confidence:.0%} {mark}</td><td>{r.vegas_pick or '-'} {vmark}</td></tr>")
    return ("<table><tr><th>Game</th><th>Final</th><th>Model</th><th>Vegas</th></tr>" + "".join(rows) + "</table>") if rows else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download latest game data")
    args = parser.parse_args()

    saved = load_model()
    df = load_all(args.refresh)
    season = int(df["season"].max())
    all_preds = predict_games(saved, df[df["season"] == season])
    right, total, v_right, v_total = season_record(all_preds)

    upcoming = pick_week(df, season)
    week = int(upcoming["week"].min()) if len(upcoming) else int(all_preds["week"].max())
    this_week = all_preds[all_preds["week"] == week].sort_values(["gameday", "gametime", "game_id"])
    this_week["slot"] = [time_slot(w, t) for w, t in zip(this_week["weekday"], this_week["gametime"])]
    this_week["games_in_slot"] = this_week.groupby("slot")["game_id"].transform("count")
    best = best_picks(this_week)
    best_ids = set(best["game_id"])
    cards = []
    for slot in best["slot"]:
        cards.append(f"<h3>{html.escape(slot)}</h3>")
        cards += [game_card(r, r.game_id in best_ids) for _, r in this_week[this_week["slot"] == slot].iterrows()]
    last_week = all_preds[all_preds["week"] == week - 1]
    lw_right = int(last_week["correct"].isin([True]).sum())
    lw_total = int(last_week["correct"].isin([True, False]).sum())
    n_disagree = int((this_week["vegas_pick"].notna() & (this_week["vegas_pick"] != this_week["pick"])).sum())

    stats = f"""<div class="stats">
<div class="stat"><b>{right}-{total - right}</b><span>Model record in {season} ({right / max(total, 1):.0%})</span></div>
<div class="stat"><b>{v_right}-{v_total - v_right}</b><span>Vegas favorites in {season} ({v_right / max(v_total, 1):.0%})</span></div>
<div class="stat"><b>{lw_right}-{lw_total - lw_right}</b><span>Model in week {week - 1}</span></div>
<div class="stat"><b>{n_disagree}</b><span>Week {week} picks that disagree with Vegas</span></div></div>"""

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NFL Game Predictor</title>
<style>{CSS}</style></head><body><main>
<h1>NFL Game Predictor</h1>
<p class="sub">Week {week}, {season} · machine learning win probabilities · updated {datetime.now():%b %-d, %Y}</p>
{stats}
<h2>Most confident pick of each time slot</h2>
<div class="slots">{''.join(slot_card(r) for _, r in best.iterrows())}</div>
<h2>Week {week} picks</h2>
{''.join(cards)}
{f'<h2>Week {week - 1} results</h2>' + results_table(last_week) if lw_total else ''}
<footer>Model: {saved['name']} using team strength (Elo), recent form, starting QB efficiency (EPA per dropback),
rest, home field and weather. Blue bar = home team, orange = away. Data from
<a href="https://github.com/nflverse">nflverse</a>.<br><br><b>Not betting advice.</b> "Most confident" means most likely
to win, not a good bet: favorites pay less. In a 10-season backtest against real moneyline odds, betting the model's
picks lost about 2–6% of the money wagered, because Vegas is more accurate than this model.</footer>
</main></body></html>"""

    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(page)
    print(f"Built docs/index.html for {season} week {week}")


if __name__ == "__main__":
    main()
