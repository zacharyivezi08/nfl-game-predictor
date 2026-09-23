"""Build the website at docs/index.html (hosted free with GitHub Pages).

Sections: this week's picks (with predicted scores), most confident pick per time slot,
season accuracy chart vs Vegas, power rankings, playoff odds, last week's results.

Usage:
    python src/build_site.py            # build from cached data
    python src/build_site.py --refresh  # pull newest results first
"""
import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from features import SNAPSHOTS
from predict import best_picks, load_model, pick_week, predict_games, season_record, spread_text, time_slot
from ratings import power_rankings, team_records
from simulate import simulate_season
from train import load_all

DOCS = Path(__file__).resolve().parent.parent / "docs"
e = html.escape

# Chart/bar colors: validated colorblind-safe pair (blue = home / our model, orange = away / Vegas)
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14171c;--muted:#5d6570;--line:#e3e6ea;--grid:#eceef1;
--s1:#2a78d6;--s2:#eb6834;--good:#1a7f37;--bad:#c2255c;--warn:#9a6700;--warnbg:#fff4d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d1117;--card:#161b22;--ink:#e6edf3;
--muted:#8b949e;--line:#30363d;--grid:#21262d;--s1:#3987e5;--s2:#d95926;--good:#3fb950;--bad:#f778ba;
--warn:#e3b341;--warnbg:#2d2410}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:900px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 6px;scroll-margin-top:12px}
.sub{color:var(--muted);margin:0 0 14px}.note{color:var(--muted);font-size:13px;margin:0 0 12px}
nav{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 18px}
nav a{font-size:13px;font-weight:600;color:var(--ink);text-decoration:none;padding:5px 11px;border:1px solid var(--line);
border-radius:999px;background:var(--card)}nav a:hover{border-color:var(--s1)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.stat b{display:block;font-size:24px}.stat span{color:var(--muted);font-size:13px}
.game{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:10px 0}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
.teams{font-weight:700;font-size:17px}.teams small{color:var(--muted);font-weight:400}
.pick{font-weight:700}.bar{display:flex;gap:2px;height:8px;margin:10px 0 8px}
.bar i{display:block;border-radius:4px}.a{background:var(--s2)}.h{background:var(--s1)}
.meta{color:var(--muted);font-size:13px}.why{font-size:13px;margin-top:6px}
.score{font-size:13px;margin-top:6px;padding-top:6px;border-top:1px dashed var(--line)}
.tag{font-size:12px;font-weight:700;padding:2px 8px;border-radius:999px;background:var(--warnbg);color:var(--warn)}
.best{border:2px solid var(--s1)}.star{font-size:12px;font-weight:700;padding:2px 8px;border-radius:999px;background:var(--s1);color:#fff}
.slots{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.slot{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.slot span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.slot b{display:block;font-size:20px;margin:2px 0}.slot small{color:var(--muted)}
h3{font-size:14px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin:22px 0 4px}
.ok{color:var(--good);font-weight:700}.no{color:var(--bad);font-weight:700}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card)}
table{width:100%;border-collapse:collapse}
td,th{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;font-size:14px;white-space:nowrap}
tr:last-child td{border-bottom:0}th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.meter{display:inline-block;width:60px;height:6px;border-radius:3px;background:var(--grid);vertical-align:middle;margin-left:6px}
.meter i{display:block;height:6px;border-radius:3px;background:var(--s1)}
.up{color:var(--good)}.down{color:var(--bad)}.same{color:var(--muted)}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}.grid2>div{min-width:0}@media (max-width:700px){.grid2{grid-template-columns:minmax(0,1fr)}}
.chart{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 12px 6px;position:relative}
.legend{display:flex;gap:16px;font-size:13px;color:var(--muted);margin:0 0 4px 4px}
.legend span::before{content:"";display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle;background:var(--c)}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12px;padding:6px 8px;border-radius:6px;
display:none;white-space:nowrap;transform:translate(-50%,-110%)}
details{margin-top:8px;font-size:13px}summary{cursor:pointer;color:var(--muted)}
footer{color:var(--muted);font-size:13px;margin-top:44px}footer a{color:var(--s1)}
"""


# ---------- this week ----------

def slot_card(r):
    opp = r.home_team if r.pick == r.away_team else r.away_team
    n = f"best of {r.games_in_slot} games" if r.games_in_slot > 1 else "only game"
    return (f'<div class="slot"><span>{e(r.slot)}</span><b>{r.pick} {r.confidence:.0%}</b>'
            f'<small>over {opp} · {n}</small></div>')


def game_card(r, is_best=False):
    away_pct = round((1 - r.home_prob) * 100)
    disagree = isinstance(r.vegas_pick, str) and r.vegas_pick != r.pick
    vegas = f"Vegas: {e(r.vegas_pick)} {r.vegas_conf:.0%}" if isinstance(r.vegas_pick, str) else "Vegas: no line yet"
    qbs = f"{e(str(r.away_qb))} vs {e(str(r.home_qb))} · " if isinstance(r.home_qb, str) else ""
    why = " · ".join(f"{e(lbl)} → {e(team)}" for lbl, team in r.reasons)
    score = ""
    if "pred_margin" in r:
        v_total = f"{r.total_line:g}" if pd.notna(r.total_line) else "-"
        score = (f'<div class="score"><b>Predicted:</b> {e(r.away_team)} {r.pred_away_pts}, {e(r.home_team)} {r.pred_home_pts}'
                 f' · Spread {e(spread_text(r.home_team, r.away_team, r.pred_margin))}'
                 f' <span class="meta">(Vegas {e(spread_text(r.home_team, r.away_team, r.spread_line))})</span>'
                 f' · Total {r.pred_total:.1f} <span class="meta">(Vegas {v_total})</span></div>')
    badges = ('<span class="star">★ Top pick of slot</span> ' if is_best and r.games_in_slot > 1 else '') + \
             ('<span class="tag">Disagrees with Vegas</span> ' if disagree else '')
    return f"""
<div class="game{' best' if is_best else ''}">
  <div class="row"><div class="teams">{e(r.away_team)} <small>@</small> {e(r.home_team)}</div>
  <div>{badges}<span class="pick">{e(r.pick)} {r.confidence:.0%}</span></div></div>
  <div class="bar" role="img" aria-label="{e(r.away_team)} {away_pct}%, {e(r.home_team)} {100 - away_pct}%">
    <i class="a" style="width:{away_pct}%"></i><i class="h" style="width:{100 - away_pct}%"></i></div>
  <div class="row meta"><span>{e(r.away_team)} {away_pct}% · {e(r.home_team)} {100 - away_pct}%</span><span>{vegas}</span></div>
  <div class="meta">{qbs}{r.gameday:%a %b %-d}</div>
  {score}
  <div class="why"><b>Why:</b> {why}</div>
</div>"""


def results_table(preds):
    rows = []
    for _, r in preds.iterrows():
        if not isinstance(r.winner, str):
            continue
        mark = '<span class="ok">✓</span>' if r.correct else '<span class="no">✗</span>'
        vmark = "" if r.vegas_correct is None else ('<span class="ok">✓</span>' if r.vegas_correct else '<span class="no">✗</span>')
        pred = f"{r.pred_away_pts}-{r.pred_home_pts}" if "pred_margin" in r else "-"
        rows.append(f"<tr><td>{r.away_team} @ {r.home_team}</td><td class='n'>{int(r.away_score)}-{int(r.home_score)}</td>"
                    f"<td class='n'>{pred}</td><td>{r.pick} {r.confidence:.0%} {mark}</td><td>{r.vegas_pick or '-'} {vmark}</td></tr>")
    if not rows:
        return ""
    return ('<div class="wrap"><table><tr><th>Game</th><th class="n">Final</th><th class="n">Predicted</th>'
            '<th>Model pick</th><th>Vegas pick</th></tr>' + "".join(rows) + "</table></div>")


# ---------- accuracy chart ----------

def accuracy_chart(all_preds, season):
    done = all_preds[all_preds["correct"].isin([True, False])]
    if done.empty:
        return '<p class="note">The chart appears after the first week of games.</p>'
    wk = done.groupby("week").agg(n=("correct", "size"), m=("correct", "sum"),
                                  vn=("vegas_correct", lambda s: s.isin([True, False]).sum()),
                                  v=("vegas_correct", lambda s: (s == True).sum())).reset_index()  # noqa: E712
    wk["m_cum"] = wk["m"].cumsum() / wk["n"].cumsum()
    wk["v_cum"] = wk["v"].cumsum() / wk["vn"].cumsum().clip(lower=1)
    W, H, L, R, T, B = 640, 240, 44, 86, 14, 30
    weeks = wk["week"].tolist()
    lo = min(0.4, wk[["m_cum", "v_cum"]].min().min() - 0.05)
    hi = max(0.85, wk[["m_cum", "v_cum"]].max().max() + 0.05)
    x = lambda w: L + (0 if len(weeks) == 1 else (weeks.index(w)) / (len(weeks) - 1) * (W - L - R)) + (
        (W - L - R) / 2 if len(weeks) == 1 else 0)
    y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    grid = ""
    for g in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if lo <= g <= hi:
            grid += (f'<line x1="{L}" x2="{W - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="var(--grid)"/>'
                     f'<text x="{L - 8}" y="{y(g) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{g:.0%}</text>')
    xlab = "".join(f'<text x="{x(w):.1f}" y="{H - 10}" text-anchor="middle" font-size="11" fill="var(--muted)">Wk {w}</text>'
                   for w in weeks)
    lines = ""
    for col, var, name in (("m_cum", "--s1", "Model"), ("v_cum", "--s2", "Vegas")):
        pts = " ".join(f"{x(w):.1f},{y(v):.1f}" for w, v in zip(weeks, wk[col]))
        lines += f'<polyline points="{pts}" fill="none" stroke="var({var})" stroke-width="2" stroke-linejoin="round"/>'
        lines += "".join(f'<circle cx="{x(w):.1f}" cy="{y(v):.1f}" r="4.5" fill="var({var})" stroke="var(--card)" stroke-width="2"/>'
                         for w, v in zip(weeks, wk[col]))
        last = wk[col].iloc[-1]
        lines += (f'<text x="{x(weeks[-1]) + 10:.1f}" y="{y(last) + 4:.1f}" font-size="12" font-weight="600" '
                  f'fill="var(--ink)">{name} {last:.0%}</text>')
    hits = "".join(f'<rect x="{x(w) - 20:.1f}" y="{T}" width="40" height="{H - T - B}" fill="transparent" data-i="{i}"/>'
                   for i, w in enumerate(weeks))
    data = [{"w": int(r.week), "m": f"{r.m_cum:.1%}", "v": f"{r.v_cum:.1%}", "mw": f"{int(r.m)}-{int(r.n - r.m)}",
             "vw": f"{int(r.v)}-{int(r.vn - r.v)}"} for r in wk.itertuples()]
    table = "".join(f"<tr><td>Week {d['w']}</td><td class='n'>{d['mw']}</td><td class='n'>{d['m']}</td>"
                    f"<td class='n'>{d['vw']}</td><td class='n'>{d['v']}</td></tr>" for d in data)
    return f"""
<div class="chart" id="acc">
  <div class="legend"><span style="--c:var(--s1)">Our model</span><span style="--c:var(--s2)">Vegas favorite</span></div>
  <svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Season-to-date pick accuracy by week, model vs Vegas">
    {grid}{xlab}{lines}<line id="xh" y1="{T}" y2="{H - B}" stroke="var(--muted)" stroke-dasharray="3 3" visibility="hidden"/>{hits}
  </svg>
  <div class="tip" id="tip"></div>
</div>
<details><summary>Show as a table</summary><div class="wrap" style="margin-top:8px"><table>
<tr><th>Week</th><th class="n">Model that week</th><th class="n">Model season %</th><th class="n">Vegas that week</th>
<th class="n">Vegas season %</th></tr>{table}</table></div></details>
<script>
(function(){{const d={json.dumps(data)},c=document.getElementById('acc'),svg=c.querySelector('svg'),
tip=document.getElementById('tip'),xh=document.getElementById('xh');
svg.querySelectorAll('rect[data-i]').forEach(r=>{{
 r.addEventListener('mouseenter',()=>{{const i=+r.dataset.i,p=d[i],x=+r.getAttribute('x')+20;
  xh.setAttribute('x1',x);xh.setAttribute('x2',x);xh.setAttribute('visibility','visible');
  tip.innerHTML='<b>Week '+p.w+'</b><br>Model: '+p.m+' season ('+p.mw+' this week)<br>Vegas: '+p.v+' season ('+p.vw+' this week)';
  const b=svg.getBoundingClientRect(),cb=c.getBoundingClientRect();
  tip.style.left=(b.left-cb.left+x*b.width/{W})+'px';tip.style.top=(b.top-cb.top+30)+'px';tip.style.display='block';}});
 r.addEventListener('mouseleave',()=>{{tip.style.display='none';xh.setAttribute('visibility','hidden');}});
}});}})();
</script>"""


# ---------- rankings + playoff odds ----------

def pct(x):
    return "-" if x == 0 else ("<1%" if x < 0.01 else (">99%" if x > 0.99 else f"{x:.0%}"))


def rankings_table(pr):
    rows = []
    for r in pr.itertuples():
        if r.change > 0:
            mv = f'<span class="up">▲ {int(r.change)}</span>'
        elif r.change < 0:
            mv = f'<span class="down">▼ {-int(r.change)}</span>'
        else:
            mv = '<span class="same">–</span>'
        rows.append(f"<tr><td class='n'>{r.rank}</td><td>{mv}</td><td><b>{r.team}</b></td><td class='n'>{r.record}</td>"
                    f"<td class='n'>{r.rating:.0%}<span class='meter'><i style='width:{r.rating * 100:.0f}%'></i></span></td>"
                    f"<td class='n'>{r.off_rank}</td><td class='n'>{r.def_rank}</td></tr>")
    return ('<div class="wrap"><table><tr><th class="n">#</th><th>Move</th><th>Team</th><th class="n">Record</th>'
            '<th class="n">Rating</th><th class="n">Offense</th><th class="n">Defense</th></tr>' + "".join(rows) + "</table></div>")


def playoff_tables(odds, records):
    out = []
    for conf in ("AFC", "NFC"):
        c = odds[odds["division"].str.startswith(conf)].sort_values(["playoffs", "super_bowl"], ascending=False)
        rows = "".join(
            f"<tr><td><b>{r.team}</b></td><td class='n'>{records[r.team]}</td><td class='n'>{r.proj_wins:.1f}</td>"
            f"<td class='n'>{pct(r.playoffs)}</td><td class='n'>{pct(r.division_title)}</td>"
            f"<td class='n'>{pct(r.top_seed)}</td><td class='n'>{pct(r.conf_title)}</td><td class='n'><b>{pct(r.super_bowl)}</b></td></tr>"
            for r in c.itertuples())
        out.append(f'<div><h3>{conf}</h3><div class="wrap"><table><tr><th>Team</th><th class="n">Now</th>'
                   f'<th class="n">Proj W</th><th class="n">Playoffs</th><th class="n">Division</th><th class="n">#1 seed</th>'
                   f'<th class="n">Make SB</th><th class="n">Win SB</th></tr>{rows}</table></div></div>')
    return "".join(out)


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
        cards.append(f"<h3>{e(slot)}</h3>")
        cards += [game_card(r, r.game_id in best_ids) for _, r in this_week[this_week["slot"] == slot].iterrows()]
    last_week = all_preds[all_preds["week"] == week - 1]
    lw_right = int(last_week["correct"].isin([True]).sum())
    lw_total = int(last_week["correct"].isin([True, False]).sum())

    print("Simulating the season 10,000 times ...")
    odds = simulate_season(df, saved, season)
    records = team_records(df, season)
    ranks = power_rankings(saved, df, SNAPSHOTS, season)

    stats = f"""<div class="stats">
<div class="stat"><b>{right}-{total - right}</b><span>Model record in {season} ({right / max(total, 1):.0%})</span></div>
<div class="stat"><b>{v_right}-{v_total - v_right}</b><span>Vegas favorites in {season} ({v_right / max(v_total, 1):.0%})</span></div>
<div class="stat"><b>{lw_right}-{lw_total - lw_right}</b><span>Model in week {week - 1}</span></div>
<div class="stat"><b>{ranks.iloc[0]['team']}</b><span>#1 in power rankings</span></div></div>"""

    features = {"base": "Elo team strength, recent form, rest, home field",
                "qb": "starting QB efficiency", "epa": "offense & defense efficiency (EPA per play, success rate)",
                "injuries": "injured starters", "weather": "weather", "travel": "travel & schedule"}
    used = "; ".join(features[g] for g in saved.get("groups", []))

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NFL Game Predictor</title>
<style>{CSS}</style></head><body><main>
<h1>NFL Game Predictor</h1>
<p class="sub">Week {week}, {season} · machine learning picks, scores and playoff odds · updated {datetime.now():%b %-d, %Y}</p>
<nav><a href="#picks">Picks</a><a href="#accuracy">Accuracy</a><a href="#rankings">Power rankings</a>
<a href="#playoffs">Playoff odds</a>{'<a href="#results">Last week</a>' if lw_total else ''}</nav>
{stats}
<h2>Most confident pick of each time slot</h2>
<div class="slots">{''.join(slot_card(r) for _, r in best.iterrows())}</div>
<h2 id="picks">Week {week} picks</h2>
<p class="note">Blue = home team, orange = away team. Predicted scores come from separate spread and total models.</p>
{''.join(cards)}
<h2 id="accuracy">{season} accuracy: model vs Vegas</h2>
<p class="note">Share of games picked correctly so far this season. (The model was trained only on earlier seasons,
so these are real predictions.)</p>
{accuracy_chart(all_preds, season)}
<h2 id="rankings">Power rankings</h2>
<p class="note">Rating = chance to beat an average NFL team on a neutral field. Offense/defense = efficiency rank (EPA per play).
Arrows = movement since last week.</p>
{rankings_table(ranks)}
<h2 id="playoffs">Playoff odds</h2>
<p class="note">From 10,000 simulations of the rest of the season and the playoffs. Simplified tiebreakers:
teams tied on wins are ordered randomly.</p>
{playoff_tables(odds, records)}
{f'<h2 id="results">Week {week - 1} results</h2>' + results_table(last_week) if lw_total else ''}
<footer>Model: {saved['name']} using {used}. It picked these by testing each group of stats and keeping only the
ones that improved predictions. Data from <a href="https://github.com/nflverse">nflverse</a>.
Code: <a href="https://github.com/zacharyivezi08/nfl-game-predictor">github.com/zacharyivezi08/nfl-game-predictor</a>.
<br><br><b>Not betting advice.</b> "Most confident" means most likely to win, not a good bet. In a 10-season backtest
against real odds, betting the model's picks lost about 2–6% of the money wagered, and its spread and over/under
picks hit about 49–50%, below the 52.4% needed to break even. Vegas is more accurate than this model.</footer>
</main></body></html>"""

    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(page)
    print(f"Built docs/index.html for {season} week {week}")


if __name__ == "__main__":
    main()
