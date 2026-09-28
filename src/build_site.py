"""Build the website at docs/index.html (hosted free with GitHub Pages).

Sections: this week's picks (with predicted scores and the model + Vegas blend), most confident
pick per time slot, season accuracy chart vs Vegas, pick tracker (record by week, upsets called),
power rankings, playoff odds, last week's results.
Also builds docs/history.html (every saved pick) and docs/teams/<TEAM>.html (one page per team).

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
from history import (picks_frame, update_odds, update_picks, upsets_called, weekly_record,
                     worst_misses)
from ratings import TEAMS, TEAM_DIV, power_ratings, power_rankings, team_records
from simulate import simulate_season
from train import load_all

DOCS = Path(__file__).resolve().parent.parent / "docs"
e = html.escape

# Chart/bar colors: validated colorblind-safe pair (blue = home / our model, orange = away / Vegas)
CSS = """
:root{--bg:#f3f5f9;--card:#fff;--card2:#f7f8fb;--ink:#0f141c;--muted:#5b6472;--line:#e2e6ec;--grid:#e9ecf1;
--s1:#2a78d6;--s2:#eb6834;--good:#1a7f37;--bad:#c2255c;--warn:#9a6700;--warnbg:#fff4d6;
--hero1:#0b1a33;--hero2:#123a73;--glow:rgba(42,120,214,.18);--shadow:0 1px 2px rgba(15,20,28,.06),0 4px 16px rgba(15,20,28,.05)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#070b12;--card:#0f1520;--card2:#131b28;--ink:#e8eef6;
--muted:#8a96a8;--line:#1f2937;--grid:#1a2230;--s1:#3d8bf0;--s2:#f07033;--good:#3fb950;--bad:#f778ba;
--warn:#e3b341;--warnbg:#2d2410;--hero1:#060d1a;--hero2:#0f2c5c;--glow:rgba(61,139,240,.22);--shadow:0 1px 2px rgba(0,0,0,.4)}}
*{box-sizing:border-box}html{scroll-behavior:smooth;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
font-variant-numeric:tabular-nums}
main{max-width:960px;margin:0 auto;padding:0 16px 60px}
.hero{background:radial-gradient(1200px 400px at 85% -20%,rgba(61,139,240,.45),transparent 60%),
linear-gradient(135deg,var(--hero1),var(--hero2));color:#fff;padding:30px 0 26px;margin-bottom:0}
.hero .in{max-width:960px;margin:0 auto;padding:0 16px}
.kicker{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#9cc3ff}
h1{font-size:clamp(28px,5vw,40px);line-height:1.1;margin:6px 0 6px;letter-spacing:-.02em;font-weight:800}
.hero .sub{color:#c4d3ea;margin:0 0 20px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);border-radius:14px;padding:12px 14px;
backdrop-filter:blur(6px)}
.stat b{display:block;font-size:26px;font-weight:800;letter-spacing:-.01em}.stat span{color:#b5c5de;font-size:12.5px}
@media (max-width:520px){.stats .stat:last-child:nth-child(odd){grid-column:1/-1}}
main .stat{background:var(--card);border-color:var(--line);box-shadow:var(--shadow)}main .stat span{color:var(--muted)}
nav{position:sticky;top:0;z-index:5;display:flex;gap:6px;overflow-x:auto;scrollbar-width:none;padding:10px 16px;
margin:0 -16px 8px;background:color-mix(in srgb,var(--bg) 82%,transparent);backdrop-filter:saturate(1.6) blur(10px);
border-bottom:1px solid var(--line)}nav::-webkit-scrollbar{display:none}
nav a{flex:none;font-size:13px;font-weight:650;color:var(--ink);text-decoration:none;padding:6px 12px;border:1px solid var(--line);
border-radius:999px;background:var(--card);transition:border-color .15s,color .15s}nav a:hover{border-color:var(--s1);color:var(--s1)}
h2{font-size:21px;font-weight:800;letter-spacing:-.01em;margin:40px 0 6px;scroll-margin-top:64px;display:flex;align-items:center;gap:10px}
h2::before{content:"";width:4px;height:20px;border-radius:2px;background:linear-gradient(var(--s1),var(--s2))}
h3{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin:24px 0 6px}
.sub{color:var(--muted);margin:0 0 14px}.note{color:var(--muted);font-size:13px;margin:0 0 12px;max-width:760px}
.card,.game,.slot,.chart,.wrap{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}
.game{padding:14px 16px;margin:10px 0;transition:transform .15s,box-shadow .15s,border-color .15s}
.game:hover,.slot:hover{transform:translateY(-2px);border-color:color-mix(in srgb,var(--s1) 45%,var(--line));box-shadow:0 8px 28px var(--glow)}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
.teams{font-weight:800;font-size:17px;display:flex;align-items:center;gap:8px}.teams small{color:var(--muted);font-weight:500}
.pick{font-weight:800;font-size:17px}.bar{display:flex;gap:3px;height:10px;margin:12px 0 8px}
.bar i{display:block;border-radius:5px;box-shadow:inset 0 0 0 1px rgba(128,128,128,.35)}
.meta{color:var(--muted);font-size:13px}.why{font-size:13px;margin-top:8px;color:var(--muted)}.why b{color:var(--ink)}
.score{font-size:13px;margin-top:8px;padding-top:8px;border-top:1px dashed var(--line)}
.blend{font-size:13px;margin-top:6px;display:inline-block;padding:3px 10px;border-radius:8px;
background:color-mix(in srgb,var(--s1) 12%,transparent)}.blend b{color:var(--s1)}
.tag{font-size:11.5px;font-weight:700;padding:3px 9px;border-radius:999px;background:var(--warnbg);color:var(--warn)}
.best{border:1.5px solid var(--s1);box-shadow:0 0 0 3px var(--glow)}
.star{font-size:11.5px;font-weight:700;padding:3px 9px;border-radius:999px;background:var(--s1);color:#fff}
.slots{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(160px,1fr);gap:10px;overflow-x:auto;
scroll-snap-type:x mandatory;padding:2px 2px 8px;scrollbar-width:thin}
.slot{padding:12px 14px;scroll-snap-align:start;transition:transform .15s,border-color .15s,box-shadow .15s}
.slot span{display:block;color:var(--muted);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em}
.slot b{display:block;font-size:24px;font-weight:800;margin:4px 0 2px}.slot small{color:var(--muted)}
.ok{color:var(--good);font-weight:800}.no{color:var(--bad);font-weight:800}
.wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse}
td,th{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;font-size:14px;white-space:nowrap}
tr:last-child td{border-bottom:0}tbody tr:hover td,table tr:hover td{background:var(--card2)}
th{color:var(--muted);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em;background:var(--card2)}
td.n,th.n{text-align:right}
.meter{display:inline-block;width:64px;height:6px;border-radius:3px;background:var(--grid);vertical-align:middle;margin-left:8px}
.meter i{display:block;height:6px;border-radius:3px;background:linear-gradient(90deg,var(--s1),#6fb0ff)}
.up{color:var(--good);font-weight:700}.down{color:var(--bad);font-weight:700}.same{color:var(--muted)}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}.grid2>div{min-width:0}
@media (max-width:700px){.grid2{grid-template-columns:minmax(0,1fr)}}
.chart{padding:12px 12px 6px;position:relative}
.legend{display:flex;gap:16px;font-size:13px;color:var(--muted);margin:0 0 4px 4px}
.legend span::before{content:"";display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle;background:var(--c)}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12px;padding:6px 8px;border-radius:6px;
display:none;white-space:nowrap;transform:translate(-50%,-110%)}
details{margin-top:8px;font-size:13px}summary{cursor:pointer;color:var(--muted)}
footer{color:var(--muted);font-size:13px;margin-top:48px;padding-top:20px;border-top:1px solid var(--line)}footer a{color:var(--s1)}
.tb{display:inline-flex;align-items:center;gap:6px;font-weight:800;color:var(--ink);text-decoration:none}
.tb::before{content:"";width:10px;height:10px;border-radius:3px;background:var(--tc);box-shadow:0 0 0 1.5px color-mix(in srgb,var(--tc) 45%,var(--muted))}
a.tb:hover{color:var(--s1)}
.tb{vertical-align:middle}.tb:has(img)::before{display:none}td .tb .lg{width:20px;height:20px}
.slot b{display:flex;align-items:center;gap:6px;white-space:nowrap}.slot b .tb{gap:5px}.tb .lg{width:22px;height:22px;object-fit:contain;flex:none}
.teams .tb .lg{width:28px;height:28px}.slot .tb .lg{width:24px;height:24px}.cmp .ct .tb .lg{width:40px;height:40px}
.cmp .ct .tb{flex-direction:column;gap:2px}
.hlogo{width:88px;height:88px;object-fit:contain;float:right;margin:0 0 8px 12px;filter:drop-shadow(0 4px 14px rgba(0,0,0,.35))}
.back{display:inline-block;font-size:13px;color:#9cc3ff;text-decoration:none;margin-bottom:6px}
.teamnav{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 0}
.teamnav a{font-size:12px;padding:4px 8px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
.teamnav a:hover{border-color:var(--s1)}
.thero{background:radial-gradient(900px 300px at 90% -30%,color-mix(in srgb,var(--tc) 70%,transparent),transparent 65%),
linear-gradient(135deg,var(--hero1),var(--hero2))}
details.game>summary{list-style:none;cursor:pointer}details.game>summary::-webkit-details-marker{display:none}
.more{margin-top:10px;font-size:12.5px;font-weight:700;color:var(--s1);display:flex;align-items:center;gap:6px}
.more::after{content:"▾";transition:transform .2s}details[open] .more::after{transform:rotate(180deg)}
details[open] .more{color:var(--muted)}details[open]{transform:none!important}
.cmp{margin-top:12px;padding-top:12px;border-top:1px solid var(--line);animation:fade .2s ease-out}
.cmp table{table-layout:fixed}.cmp td,.cmp th{white-space:normal;padding:7px 6px;text-align:center;border-bottom:1px solid var(--grid)}
.cmp th{background:none;font-size:11.5px;letter-spacing:.03em;width:46%}.cmp td{font-weight:600;font-size:14.5px}
.cmp td.ct{font-size:16px;border-bottom:1px solid var(--line)}.cmp tr:hover td{background:none}
.cmp td.win{font-weight:800;background:color-mix(in srgb,var(--tc) 18%,transparent)!important;border-radius:8px}
.cmp .note{margin:10px 0 0;font-size:12px}
.flag{font-size:13px;margin-top:8px;padding:8px 10px;border-radius:10px;background:var(--warnbg);border-left:3px solid var(--warn)}
.flag b{color:var(--warn)}.flag .meta{display:block;margin-top:2px}
.glance{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
.glance>div{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:12px 16px;box-shadow:var(--shadow)}
.glance h4,.cmp h4{margin:0 0 8px;font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.glance ul,.edges{list-style:none;margin:0;padding:0}.glance li{padding:6px 0;border-bottom:1px solid var(--grid);display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.glance li:last-child{border-bottom:0}
.edges li{padding:7px 10px;margin:0 0 6px;border-radius:10px;font-size:13.5px;background:var(--card2)}
.edges li.big{background:color-mix(in srgb,var(--good) 14%,transparent)}.cmp h4{margin-top:4px}
.mypick{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:10px}
.mypick button{display:inline-flex;align-items:center;gap:6px;font:inherit;font-size:13px;font-weight:700;color:var(--ink);
background:var(--card2);border:1.5px solid var(--line);border-radius:999px;padding:4px 12px 4px 6px;cursor:pointer}
.mypick button .lg{width:20px;height:20px}.mypick button:hover:not(:disabled){border-color:var(--s1)}
.mypick button.on{border-color:var(--s1);background:color-mix(in srgb,var(--s1) 16%,transparent)}
.mypick button:disabled{opacity:.5;cursor:default}.mypick button.on:disabled{opacity:1}
#myrec:empty{display:none}
.upsets{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.upset{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--tc);border-radius:14px;padding:12px 14px;box-shadow:var(--shadow)}
.uprow{display:flex;align-items:center;gap:10px;margin:6px 0 4px}.uprow b{font-size:26px;font-weight:800;min-width:58px}
.umeter{flex:1;height:8px;border-radius:4px;background:var(--grid);overflow:hidden}.umeter i{display:block;height:8px;max-width:100%;background:var(--tc)}
@keyframes fade{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important}}
"""

# Team primary colors (for the small color chips next to team names)
TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D", "CAR": "#0085CA", "CHI": "#C83803",
    "CIN": "#FB4F14", "CLE": "#FF3C00", "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#A71930", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837", "LA": "#003594", "LAC": "#0080C6",
    "LV": "#A5ACAF", "MIA": "#008E97", "MIN": "#4F2683", "NE": "#C60C30", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#69BE28", "SF": "#AA0000", "TB": "#D50A0A",
    "TEN": "#4B92DB", "WAS": "#5A1414",
}
# Second color, used for the away team's bar when both teams' main colors look too alike
TEAM_ALT_COLORS = {
    "ARI": "#FFB612", "ATL": "#A5ACAF", "BAL": "#9E7C0C", "BUF": "#C60C30", "CAR": "#BFC0BF", "CHI": "#0B162A",
    "CIN": "#1A1A1A", "CLE": "#311D00", "DAL": "#869397", "DEN": "#002244", "DET": "#B0B7BC", "GB": "#FFB612",
    "HOU": "#03202F", "IND": "#A2AAAD", "JAX": "#D7A22A", "KC": "#FFB81C", "LA": "#FFA300", "LAC": "#FFC20E",
    "LV": "#1A1A1A", "MIA": "#FC4C02", "MIN": "#FFC62F", "NE": "#002244", "NO": "#101820", "NYG": "#A71930",
    "NYJ": "#1A1A1A", "PHI": "#A5ACAF", "PIT": "#101820", "SEA": "#002244", "SF": "#B3995D", "TB": "#34302B",
    "TEN": "#0C2340", "WAS": "#FFB612",
}


def _rgb(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def _too_close(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(_rgb(c1), _rgb(c2))) ** 0.5 < 100


def bar_colors(away, home):
    """Each team's own color; if they'd look the same, the away team switches to its second color."""
    h = TEAM_COLORS.get(home, "#2a78d6")
    a = TEAM_COLORS.get(away, "#eb6834")
    if _too_close(a, h):
        a = TEAM_ALT_COLORS.get(away, a)
        if _too_close(a, h):
            a = "#8a96a8"  # neutral gray as a last resort
    return a, h


ESPN_ABBR = {"LA": "lar", "WAS": "wsh"}  # the logo links are the same ones nflverse's team list uses


def logo_url(team, size=64):
    """Small team logo (resized by ESPN's image server, so each is only a few KB)."""
    code = ESPN_ABBR.get(team, team.lower())
    return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/{code}.png&h={size}&w={size}"


def logo(team, size=64, cls="lg"):
    """Logo <img>. If the small version fails it tries the full-size one; if that fails too, it removes
    itself and the team's color chip shows instead."""
    full = f"https://a.espncdn.com/i/teamlogos/nfl/500/{ESPN_ABBR.get(team, team.lower())}.png"
    return (f'<img class="{cls}" src="{logo_url(team, size)}" alt="" width="{size // 2}" height="{size // 2}" '
            f'loading="lazy" decoding="async" onerror="this.onerror=()=>this.remove();this.src=\'{full}\'">')


def tl(team, prefix=""):
    """Team logo + abbreviation, linked to its team page."""
    return (f'<a class="tb" style="--tc:{TEAM_COLORS.get(team, "#888")}" '
            f'href="{prefix}teams/{e(team)}.html">{logo(team)}{e(team)}</a>')


# ---------- this week ----------

def slot_card(r):
    opp = r.home_team if r.pick == r.away_team else r.away_team
    n = f"best of {r.games_in_slot} games" if r.games_in_slot > 1 else "only game"
    return (f'<div class="slot"><span>{e(r.slot)}</span><b>{tl(r.pick)} {r.confidence:.0%}</b>'
            f'<small>over {opp} · {n}</small></div>')


# ---------- upset watch + calibration ----------

UPSET_MIN = 0.35  # underdogs with at least this chance make the upset watch


def upset_history(cal, p):
    for b in cal.get("upsets", []):
        if b["lo"] <= p < b["hi"]:
            return b
    return None


def upset_watch(week_preds, cal, prefix=""):
    """This week's underdogs with a real chance, by the blend (the most accurate forecast)."""
    rows = []
    for r in week_preds.itertuples():
        fav_home = r.blend_prob >= 0.5
        dog, fav = (r.away_team, r.home_team) if fav_home else (r.home_team, r.away_team)
        p = 1 - r.blend_conf
        if p < UPSET_MIN:
            continue
        rows.append((p, r, dog, fav))
    if not rows:
        return '<p class="note">No underdog has a 35%+ chance this week. Every favorite is a solid favorite.</p>'
    cards = []
    for p, r, dog, fav in sorted(rows, key=lambda x: -x[0]):
        h = upset_history(cal, p) if cal else None
        hist = (f"Underdogs given {h['lo']:.0%}–{h['hi']:.0%} have won <b>{h['won']:.0%}</b> of the time "
                f"({h['n']:,} games since 2012)") if h else ""
        model_dog = r.pick == dog
        badge = '<span class="tag">Model picks the upset</span>' if model_dog else ""
        result = ""
        if isinstance(r.winner, str):
            result = ('<span class="ok">Upset! ✓</span>' if r.winner == dog else '<span class="meta">Favorite won</span>')
        cards.append(f"""<div class="upset" style="--tc:{TEAM_COLORS.get(dog, '#888')}">
<div class="row"><div class="teams">{tl(dog, prefix)} <small>over</small> {tl(fav, prefix)}</div><div>{badge} {result}</div></div>
<div class="uprow"><b>{p:.0%}</b><div class="umeter"><i style="width:{p * 200:.0f}%"></i></div></div>
<div class="meta">{hist}</div></div>""")
    return '<div class="upsets">' + "".join(cards) + "</div>"


def calibration_chart(cal):
    """Predicted chance vs how often it actually happened, for the model and the blend."""
    if not cal or not cal.get("blend"):
        return ""
    W, H, L, R, T, B = 640, 300, 50, 20, 14, 40
    lo, hi = 0.45, 1.0
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    out = ""
    for g in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        out += (f'<line x1="{L}" x2="{W - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="var(--grid)"/>'
                f'<text x="{L - 8}" y="{y(g) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{g:.0%}</text>'
                f'<text x="{x(g):.1f}" y="{H - 22}" text-anchor="middle" font-size="11" fill="var(--muted)">{g:.0%}</text>')
    out += (f'<line x1="{x(.5):.1f}" y1="{y(.5):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}" stroke="var(--muted)" '
            f'stroke-dasharray="4 4"/><text x="{x(.93):.1f}" y="{y(.97):.1f}" font-size="11" fill="var(--muted)" '
            f'text-anchor="end">perfect</text>')
    out += (f'<text x="{(L + W - R) / 2:.0f}" y="{H - 4}" text-anchor="middle" font-size="11" fill="var(--muted)">'
            f'Predicted chance for the favorite</text>')
    for key, var, name in (("model", "--s2", "Model"), ("blend", "--s1", "Model + Vegas")):
        pts = cal[key]
        line = " ".join(f"{x(b['pred']):.1f},{y(b['won']):.1f}" for b in pts)
        out += (f'<polyline points="{line}" fill="none" '
                f'stroke="var({var})" stroke-width="2"/>')
        out += "".join(f'<circle cx="{x(b["pred"]):.1f}" cy="{y(b["won"]):.1f}" r="{3 + min(b["n"], 700) / 140:.1f}" '
                       f'fill="var({var})" stroke="var(--card)" stroke-width="1.5"><title>{name}: said {b["pred"]:.0%}, '
                       f'won {b["won"]:.0%} ({b["n"]} games)</title></circle>' for b in pts)
    rows = "".join(f"<tr><td>{b['lo']:.0%}–{b['hi']:.0%}</td><td class='n'>{b['pred']:.1%}</td><td class='n'>{b['won']:.1%}</td>"
                   f"<td class='n'>{b['n']:,}</td></tr>" for b in cal["blend"])
    return f"""<div class="chart"><div class="legend"><span style="--c:var(--s1)">Model + Vegas</span>
<span style="--c:var(--s2)">Model</span></div>
<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Calibration: predicted vs actual win rate">{out}</svg></div>
<details><summary>Show as a table (Model + Vegas)</summary><div class="wrap" style="margin-top:8px"><table>
<tr><th>Favorite's chance</th><th class="n">Said</th><th class="n">Actually won</th><th class="n">Games</th></tr>{rows}</table></div></details>"""


# ---------- this week at a glance ----------

def glance_card(week_preds, prefix=""):
    """One box: safest picks, best upset chances, and spread/total flags for the games still to be played."""
    g = week_preds[~week_preds["played"]]
    if g.empty:
        g = week_preds
    safest = g.sort_values("blend_conf", ascending=False).head(3)
    safe = "".join(f"<li>{tl(r.blend_pick, prefix)} <b>{r.blend_conf:.0%}</b> <span class='meta'>vs "
                   f"{e(r.home_team if r.blend_pick == r.away_team else r.away_team)}</span></li>" for r in safest.itertuples())
    dogs = g.assign(dog_p=1 - g["blend_conf"]).sort_values("dog_p", ascending=False).head(3)
    ups = "".join(f"<li>{tl(r.home_team if r.blend_pick == r.away_team else r.away_team, prefix)} <b>{r.dog_p:.0%}</b> "
                  f"<span class='meta'>to beat {e(r.blend_pick)}</span></li>" for r in dogs.itertuples())
    flags = []
    for r in g.itertuples():
        if "pred_margin" not in g or pd.isna(r.spread_line) or pd.isna(r.total_line):
            continue
        sg, tg = r.pred_margin - r.spread_line, r.pred_total - r.total_line
        if abs(sg) >= 4:
            flags.append(f"<li>{tl(r.home_team if sg > 0 else r.away_team, prefix)} to cover "
                         f"<span class='meta'>({e(spread_text(r.home_team, r.away_team, r.spread_line))})</span></li>")
        if abs(tg) >= 4:
            flags.append(f"<li><b>{'Over' if tg > 0 else 'Under'} {r.total_line:g}</b> <span class='meta'>"
                         f"{e(r.away_team)} @ {e(r.home_team)}</span></li>")
    fl = "".join(flags[:5]) or "<li class='meta'>None this week</li>"
    return f"""<div class="glance">
<div><h4>Safest picks</h4><ul>{safe}</ul></div>
<div><h4>Best upset chances</h4><ul>{ups}</ul></div>
<div><h4>Spread / total flags</h4><ul>{fl}</ul></div></div>"""


# ---------- make your own picks (saved in the viewer's browser) ----------

def pick_buttons(r):
    from history import kickoff
    return (f'<div class="mypick" data-g="{e(r.game_id)}" data-k="{kickoff(r).isoformat()}"><span class="meta">Your pick:</span>'
            f'<button type="button" data-t="{e(r.away_team)}">{logo(r.away_team)}{e(r.away_team)}</button>'
            f'<button type="button" data-t="{e(r.home_team)}">{logo(r.home_team)}{e(r.home_team)}</button>'
            f'<span class="res"></span></div>')


def my_picks_script(all_preds):
    """Results for every finished game this season, so the viewer's record can be scored in the browser."""
    res = {}
    for r in all_preds.itertuples():
        if r.played:
            res[r.game_id] = {"w": r.winner if isinstance(r.winner, str) else "TIE", "m": r.pick,
                              "v": r.vegas_pick if isinstance(r.vegas_pick, str) else None}
    return """<script>
(function(){const R=%s;let P={};
try{P=JSON.parse(localStorage.getItem('nflgp-picks')||'{}')}catch(e){}
function save(){try{localStorage.setItem('nflgp-picks',JSON.stringify(P))}catch(e){}}
function rec(a,b){return a+'-'+(b-a)+(b?' ('+Math.round(100*a/b)+'%%)':'')}
function render(){
 document.querySelectorAll('.mypick').forEach(el=>{const g=el.dataset.g,done=R[g],locked=done||Date.now()>=Date.parse(el.dataset.k);
  el.querySelectorAll('button').forEach(b=>{b.classList.toggle('on',P[g]===b.dataset.t);b.disabled=!!locked;});
  const s=el.querySelector('.res');
  if(done&&P[g])s.innerHTML=done.w===P[g]?'<span class="ok">✓ Right</span>':(done.w==='TIE'?'Tie':'<span class="no">✗ Wrong</span>');
  else s.textContent=locked&&!P[g]?'Locked (kicked off)':'';});
 let n=0,y=0,m=0,v=0,vn=0,open=0;
 for(const g in P){const r=R[g];if(!r){open++;continue}if(r.w==='TIE')continue;n++;if(P[g]===r.w)y++;if(r.m===r.w)m++;
  if(r.v){vn++;if(r.v===r.w)v++}}
 const box=document.getElementById('myrec');if(!box)return;
 box.innerHTML=n?'<div class="stat"><b>'+rec(y,n)+'</b><span>Your picks</span></div><div class="stat"><b>'+rec(m,n)+
  '</b><span>Model on the same games</span></div><div class="stat"><b>'+rec(v,vn)+'</b><span>Vegas on the same games</span></div>'+
  (open?'<div class="stat"><b>'+open+'</b><span>Picks waiting on results</span></div>':'')
  :'<p class="note">'+(open?'You have '+open+' pick(s) waiting on results. ':'')+'Tap a team under any game below to make your pick. '+
  'Picks lock at kickoff, and your record vs the model shows up here once games finish.</p>';}
document.addEventListener('click',ev=>{const b=ev.target.closest('.mypick button');if(!b)return;ev.preventDefault();ev.stopPropagation();
 if(b.disabled)return;const g=b.closest('.mypick').dataset.g;if(P[g]===b.dataset.t)delete P[g];else P[g]=b.dataset.t;save();render();});
render();})();
</script>""" % json.dumps(res)


# ---------- plain-English matchup edges ----------

def edge_lines(r, stats):
    """Pass/run offense vs the other team's pass/run defense, in plain English."""
    lines = []
    for off, dfn in ((r.away_team, r.home_team), (r.home_team, r.away_team)):
        for kind, word in (("pass", "passing"), ("run", "running")):
            o, d = stats[off].get(f"{kind}_off_rank"), stats[dfn].get(f"{kind}_def_rank")
            if o is None or d is None:
                continue
            gap = d - o  # positive = the offense ranks better than the defense it faces
            if gap >= 15:
                verdict, who, cls = "Big edge", off, "big"
            elif gap >= 8:
                verdict, who, cls = "Edge", off, "edge"
            elif gap <= -15:
                verdict, who, cls = "Big edge", dfn, "big"
            elif gap <= -8:
                verdict, who, cls = "Edge", dfn, "edge"
            else:
                continue
            lines.append((abs(gap), f'<li class="{cls}">{tl(off)} {word} offense <b>#{o}</b> vs {tl(dfn)} {kind} defense '
                                    f'<b>#{d}</b>: <b>{verdict} {e(who)}</b></li>'))
    if not lines:
        return '<p class="note">No big pass or run mismatches in this game: the units are evenly matched.</p>'
    return "<ul class='edges'>" + "".join(x for _, x in sorted(lines, key=lambda t: -t[0])) + "</ul>"


# ---------- click-to-compare team stats ----------

# (label, key, which is better: "hi" / "lo" / None, format)
COMPARE_STATS = [
    ("Record", "record", None, "{}"),
    ("Power ranking", "rank", "lo", "#{:.0f}"),
    ("Power rating", "rating", "hi", "{:.0%}"),
    ("Points per game", "ppg", "hi", "{:.1f}"),
    ("Points allowed per game", "papg", "lo", "{:.1f}"),
    ("Point differential per game", "pdpg", "hi", "{:+.1f}"),
    ("Last 5 games*: avg margin", "form_pd", "hi", "{:+.1f}"),
    ("Last 5 games*: win %", "form_win", "hi", "{:.0%}"),
    ("Offense EPA per play", "off_epa", "hi", "{:+.3f}"),
    ("Defense EPA per play allowed", "def_epa", "lo", "{:+.3f}"),
    ("Offense success rate", "off_sr", "hi", "{:.1%}"),
    ("Defense success rate allowed", "def_sr", "lo", "{:.1%}"),
    ("Offense rank (EPA)", "off_rank", "lo", "#{:.0f}"),
    ("Defense rank (EPA)", "def_rank", "lo", "#{:.0f}"),
    ("Pass offense rank", "pass_off_rank", "lo", "#{:.0f}"),
    ("Run offense rank", "run_off_rank", "lo", "#{:.0f}"),
    ("Pass defense rank", "pass_def_rank", "lo", "#{:.0f}"),
    ("Run defense rank", "run_def_rank", "lo", "#{:.0f}"),
    ("Starting QB: EPA per dropback", "qb", "hi", "{:+.3f}"),
    ("Elo rating", "elo", "hi", "{:.0f}"),
    ("Starters' snaps lost to injury", "inj", "lo", "{:.1f}"),
    ("Playoff odds", "playoffs", "hi", "{:.0%}"),
]


def team_stats(df, season, ranks, odds):
    """Every team's stats so far: season averages (all games played) + the model's current ratings."""
    from data import load_team_epa
    before = df[(df["season"] == season) & df["played"]]
    epa = load_team_epa([season])
    if len(epa):
        epa = epa[epa["game_id"].isin(before["game_id"])]
    state = SNAPSHOTS["latest"]
    rk = ranks.set_index("team")
    od = odds.set_index("team")
    out = {}
    unit_rank = {}
    for key, best_high in (("off_pass", True), ("off_run", True), ("def_pass", False), ("def_run", False)):
        vals = pd.Series({t: state[t][key] for t in TEAMS})
        unit_rank[key] = vals.rank(ascending=not best_high, method="min").astype(int).to_dict()
    for t in TEAMS:
        home, away = before[before["home_team"] == t], before[before["away_team"] == t]
        pts = list(home["home_score"]) + list(away["away_score"])
        opp = list(home["away_score"]) + list(away["home_score"])
        n = len(pts)
        s = {"record": rk.loc[t, "record"], "rank": rk.loc[t, "rank"], "rating": rk.loc[t, "rating"],
             "off_rank": rk.loc[t, "off_rank"], "def_rank": rk.loc[t, "def_rank"],
             "ppg": sum(pts) / n if n else None, "papg": sum(opp) / n if n else None,
             "pdpg": (sum(pts) - sum(opp)) / n if n else None,
             "form_pd": state[t]["pd"] if n else None, "form_win": state[t]["win_pct"] if n else None,
             "qb": state[t]["qb"], "elo": state[t]["elo"], "playoffs": od.loc[t, "playoffs"],
             "off_epa": None, "def_epa": None, "off_sr": None, "def_sr": None,
             "pass_off_rank": unit_rank["off_pass"][t], "run_off_rank": unit_rank["off_run"][t],
             "pass_def_rank": unit_rank["def_pass"][t], "run_def_rank": unit_rank["def_run"][t]}
        te = epa[epa["team"] == t] if len(epa) else epa
        if len(te) and te["off_plays"].sum() and te["def_plays"].sum():
            s["off_epa"] = te["off_epa"].sum() / te["off_plays"].sum()
            s["def_epa"] = te["def_epa"].sum() / te["def_plays"].sum()
            s["off_sr"] = te["off_success"].sum() / te["off_plays"].sum()
            s["def_sr"] = te["def_success"].sum() / te["def_plays"].sum()
        out[t] = s
    return out


def compare_panel(r, stats):
    a, h = stats[r.away_team], stats[r.home_team]
    a = {**a, "inj": r.away_inj_off + r.away_inj_def}
    h = {**h, "inj": r.home_inj_off + r.home_inj_def}
    ca, ch = bar_colors(r.away_team, r.home_team)
    rows = []
    for label, key, better, fmt in COMPARE_STATS:
        va, vh = a.get(key), h.get(key)
        if key == "qb" and isinstance(r.home_qb, str):
            label = f"QB EPA per dropback ({e(str(r.away_qb).split()[-1])} / {e(str(r.home_qb).split()[-1])})"
        show = lambda v: "—" if v is None or v != v else fmt.format(v)
        wa = wh = ""
        if better and va is not None and vh is not None and va == va and vh == vh and show(va) != show(vh):
            a_wins = (va > vh) if better == "hi" else (va < vh)
            wa, wh = (" class='win'", "") if a_wins else ("", " class='win'")
        rows.append(f"<tr><td{wa} style='--tc:{ca}'>{show(va)}</td><th>{label}</th><td{wh} style='--tc:{ch}'>{show(vh)}</td></tr>")
    return (f'<div class="cmp"><h4>Matchup edges</h4>{edge_lines(r, stats)}<table><tr><td class="ct">{tl(r.away_team)}</td><th></th><td class="ct">{tl(r.home_team)}</td></tr>'
            + "".join(rows) + '</table><p class="note">Pass/run ranks use the model\'s running ratings (recent games count most). '
            'Season averages so far (— = no games yet). '
            'Highlighted = better. * Can include games from last season. EPA = expected points added per play, the best single measure of efficiency.</p></div>')


def line_flags(r, gaps):
    """Highlight spreads/totals where the model's predicted score is far from the Vegas line."""
    if not gaps or "pred_margin" not in r or pd.isna(r.spread_line) or pd.isna(r.total_line):
        return ""
    pts, out = gaps["points"], []
    sg = r.pred_margin - r.spread_line
    if abs(sg) >= pts:
        team = r.home_team if sg > 0 else r.away_team
        line = spread_text(r.home_team, r.away_team, r.spread_line)
        out.append(f'<div class="flag"><b>Spread: model likes {e(team)} to cover</b> ({e(line)}). Its predicted margin is '
                   f'{abs(sg):.1f} pts better for {e(team)}. <span class="meta">{pts}+ pt gaps have been right '
                   f'{gaps["spread"]["hit"]:.0%} ({gaps["spread"]["n"]} games, {gaps["seasons"]})</span></div>')
    tg = r.pred_total - r.total_line
    if abs(tg) >= pts:
        side = "OVER" if tg > 0 else "UNDER"
        out.append(f'<div class="flag"><b>Total: model says {side} {r.total_line:g}</b>. It predicts {r.pred_total:.1f}, '
                   f'{abs(tg):.1f} pts {"higher" if tg > 0 else "lower"}. <span class="meta">{pts}+ pt gaps have been right '
                   f'{gaps["total"]["hit"]:.0%} ({gaps["total"]["n"]} games, {gaps["seasons"]})</span></div>')
    return "".join(out)


def game_card(r, is_best=False, stats=None, gaps=None):
    away_pct = round((1 - r.home_prob) * 100)
    ca, ch = bar_colors(r.away_team, r.home_team)
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
    blend = ""
    if isinstance(r.vegas_pick, str):
        blend = f'<div class="blend"><b>Model + Vegas blend: {e(r.blend_pick)} {r.blend_conf:.0%}</b></div>'
    badges = ('<span class="star">★ Top pick of slot</span> ' if is_best and r.games_in_slot > 1 else '') + \
             ('<span class="tag">Disagrees with Vegas</span> ' if disagree else '')
    return f"""
<details class="game{' best' if is_best else ''}"><summary>
  <div class="row"><div class="teams">{tl(r.away_team)} <small>@</small> {tl(r.home_team)}</div>
  <div>{badges}<span class="pick">{e(r.pick)} {r.confidence:.0%}</span></div></div>
  <div class="bar" role="img" aria-label="{e(r.away_team)} {away_pct}%, {e(r.home_team)} {100 - away_pct}%">
    <i style="width:{away_pct}%;background:{ca}"></i><i style="width:{100 - away_pct}%;background:{ch}"></i></div>
  <div class="row meta"><span>{e(r.away_team)} {away_pct}% · {e(r.home_team)} {100 - away_pct}%</span><span>{vegas}</span></div>
  <div class="meta">{qbs}{r.gameday:%a %b %-d}</div>
  {pick_buttons(r)}
  {score}
  {line_flags(r, gaps)}
  {blend}
  <div class="why"><b>Why:</b> {why}</div>
  <div class="more">Compare team stats</div>
</summary>{compare_panel(r, stats) if stats else ''}
</details>"""


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
        rows.append(f"<tr><td class='n'>{r.rank}</td><td>{mv}</td><td><b>{tl(r.team)}</b></td><td class='n'>{r.record}</td>"
                    f"<td class='n'>{r.rating:.0%}<span class='meter'><i style='width:{r.rating * 100:.0f}%'></i></span></td>"
                    f"<td class='n'>{r.off_rank}</td><td class='n'>{r.def_rank}</td></tr>")
    return ('<div class="wrap"><table><tr><th class="n">#</th><th>Move</th><th>Team</th><th class="n">Record</th>'
            '<th class="n">Rating</th><th class="n">Offense</th><th class="n">Defense</th></tr>' + "".join(rows) + "</table></div>")


def playoff_tables(odds, records):
    out = []
    for conf in ("AFC", "NFC"):
        c = odds[odds["division"].str.startswith(conf)].sort_values(["playoffs", "super_bowl"], ascending=False)
        rows = "".join(
            f"<tr><td><b>{tl(r.team)}</b></td><td class='n'>{records[r.team]}</td><td class='n'>{r.proj_wins:.1f}</td>"
            f"<td class='n'>{pct(r.playoffs)}</td><td class='n'>{pct(r.division_title)}</td>"
            f"<td class='n'>{pct(r.top_seed)}</td><td class='n'>{pct(r.conf_title)}</td><td class='n'><b>{pct(r.super_bowl)}</b></td></tr>"
            for r in c.itertuples())
        out.append(f'<div><h3>{conf}</h3><div class="wrap"><table><tr><th>Team</th><th class="n">Now</th>'
                   f'<th class="n">Proj W</th><th class="n">Playoffs</th><th class="n">Division</th><th class="n">#1 seed</th>'
                   f'<th class="n">Make SB</th><th class="n">Win SB</th></tr>{rows}</table></div></div>')
    return "".join(out)


# ---------- simple line chart (team pages) ----------

def line_chart(labels, series, label, lo=0.0, hi=1.0, ticks=(0, .25, .5, .75, 1)):
    """Small SVG line chart. series = [(name, css color var, [values or None])]; values are 0-1."""
    if not labels:
        return '<p class="note">Not enough games yet.</p>'
    W, H, L, R, T, B = 640, 200, 44, 70, 12, 28
    n = len(labels)
    x = lambda i: L + ((W - L - R) / 2 if n == 1 else i / (n - 1) * (W - L - R))
    y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    out = ""
    for g in ticks:
        out += (f'<line x1="{L}" x2="{W - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="var(--grid)"/>'
                f'<text x="{L - 8}" y="{y(g) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{g:.0%}</text>')
    step = max(1, n // 9)
    out += "".join(f'<text x="{x(i):.1f}" y="{H - 8}" text-anchor="middle" font-size="11" fill="var(--muted)">{e(lb)}</text>'
                   for i, lb in enumerate(labels) if i % step == 0 or i == n - 1)
    legend = ""
    for name, var, vals in series:
        pts = [(i, v) for i, v in enumerate(vals) if v is not None]
        if not pts:
            continue
        legend += f'<span style="--c:var({var})">{e(name)}</span>'
        out += (f'<polyline points="{" ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in pts)}" fill="none" '
                f'stroke="var({var})" stroke-width="2" stroke-linejoin="round"/>')
        out += "".join(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" fill="var({var})" stroke="var(--card)" '
                       f'stroke-width="2"><title>{e(name)} {e(labels[i])}: {v:.0%}</title></circle>' for i, v in pts)
        i, v = pts[-1]
        out += (f'<text x="{x(i) + 9:.1f}" y="{y(v) + 4:.1f}" font-size="12" font-weight="600" '
                f'fill="var(--ink)">{v:.0%}</text>')
    return (f'<div class="chart"><div class="legend">{legend}</div><svg viewBox="0 0 {W} {H}" width="100%" role="img" '
            f'aria-label="{e(label)}">{out}</svg></div>')


# ---------- pick tracker ----------

def mark(v):
    return "" if v is None or v != v else ('<span class="ok">✓</span>' if v else '<span class="no">✗</span>')


def rec(w, n):
    return f"{w}-{n - w}" + (f" ({w / n:.0%})" if n else "")


def tracker_section(p, prefix=""):
    wk = weekly_record(p)
    if wk.empty:
        return '<p class="note">The tracker fills in after the first week of games.</p>'
    rows = "".join(f"<tr><td>Week {r.week}</td><td class='n'>{rec(r.correct, r.correct_n)}</td>"
                   f"<td class='n'>{rec(r.vegas_correct, r.vegas_correct_n)}</td>"
                   f"<td class='n'>{rec(r.blend_correct, r.blend_correct_n)}</td></tr>" for r in wk.itertuples())
    tot = wk.sum(numeric_only=True)
    rows += (f"<tr><td><b>Season</b></td><td class='n'><b>{rec(int(tot.correct), int(tot.correct_n))}</b></td>"
             f"<td class='n'><b>{rec(int(tot.vegas_correct), int(tot.vegas_correct_n))}</b></td>"
             f"<td class='n'><b>{rec(int(tot.blend_correct), int(tot.blend_correct_n))}</b></td></tr>")
    table = ('<div class="wrap"><table><tr><th>Week</th><th class="n">Model</th><th class="n">Vegas favorite</th>'
             '<th class="n">Model + Vegas</th></tr>' + rows + "</table></div>")

    def game_list(df, text):
        if df.empty:
            return '<p class="note">None yet.</p>'
        items = "".join(f"<tr><td>Wk {r.week}</td><td>{tl(r.away, prefix)} @ {tl(r.home, prefix)}</td>"
                        f"<td class='n'>{r.final[0]}-{r.final[1]}</td><td>{text(r)}</td></tr>" for r in df.itertuples())
        return f'<div class="wrap"><table>{items}</table></div>'
    ups = game_list(upsets_called(p), lambda r: f"Picked <b>{e(r.pick)}</b>; Vegas had {e(r.vegas_pick)} {r.vegas_fav_conf:.0%}")
    miss = game_list(worst_misses(p), lambda r: f"Picked {e(r.pick)} {r.conf:.0%}, {e(r.winner)} won")
    return (f'{table}<div class="grid2"><div><h3>Biggest upsets called</h3>'
            f'<p class="note">Model picked against the Vegas favorite and was right.</p>{ups}</div>'
            f'<div><h3>Worst misses</h3><p class="note">Most confident picks that lost.</p>{miss}</div></div>'
            f'<p class="note" style="margin-top:10px"><a href="{prefix}history.html">See every pick →</a></p>')


def page(title, hero, body, prefix="", color=None):
    """Sub-page shell: colored header + content + team links."""
    style = f' style="--tc:{color}"' if color else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title>
<meta name="theme-color" content="#0b1a33"><style>{CSS}</style></head><body>
<header class="hero{' thero' if color else ''}"{style}><div class="in"><a class="back" href="{prefix}index.html">← All picks</a>{hero}</div></header>
<main>{body}
<footer><b>Teams</b><div class="teamnav">{''.join(tl(t, prefix) for t in TEAMS)}</div><br><b>Not betting advice.</b>
Data from <a href="https://github.com/nflverse">nflverse</a>.</footer></main></body></html>"""


def history_page(p, season):
    if p.empty:
        body = "<p>No saved picks yet.</p>"
    else:
        rows = []
        for r in p.sort_values(["week", "date", "game_id"], ascending=[False, True, True]).itertuples():
            fin = f"{r.final[0]}-{r.final[1]}" if isinstance(r.final, list) else "-"
            conf = r.home_prob if r.pick == r.home else 1 - r.home_prob
            bconf = r.blend_prob if r.blend_pick == r.home else 1 - r.blend_prob
            vg = "-" if r.vegas_pick is None or r.vegas_pick != r.vegas_pick else e(r.vegas_pick)
            src = "" if r.source == "live" else " <span class='meta'>*</span>"
            rows.append(f"<tr><td>Wk {r.week}</td><td>{tl(r.away, '')} @ {tl(r.home, '')}</td><td class='n'>{fin}</td>"
                        f"<td>{e(r.pick)} {conf:.0%} {mark(getattr(r, 'correct', None))}{src}</td>"
                        f"<td>{vg} {mark(getattr(r, 'vegas_correct', None))}</td>"
                        f"<td>{e(r.blend_pick)} {bconf:.0%} {mark(getattr(r, 'blend_correct', None))}</td></tr>")
        body = ('<div class="wrap"><table><tr><th>Week</th><th>Game</th><th class="n">Final</th><th>Model</th>'
                '<th>Vegas</th><th>Model + Vegas</th></tr>' + "".join(rows) + "</table></div>")
    hero = f"""<div class="kicker">Pick tracker</div><h1>Every {season} pick</h1>
<p class="sub">Each pick is saved before kickoff and frozen once the game starts, so this record can't be changed after
the fact.</p>"""
    return page(f"Every pick · {season}", hero, f"""<h2>Record</h2>{tracker_section(p, '') if not p.empty else ''}
<h2>All picks</h2><p class="note">* = from before the tracker started: what the model would have said (it was trained
only on earlier seasons).</p>{body}""")


def team_page(team, season, all_preds, ranks, odds, odds_hist, rating_hist, records):
    rk = ranks.set_index("team").loc[team]
    od = odds.set_index("team").loc[team]
    g = all_preds[(all_preds["home_team"] == team) | (all_preds["away_team"] == team)].sort_values(["week", "gameday"])
    rows = []
    for r in g.itertuples():
        home = r.home_team == team
        opp = r.away_team if home else r.home_team
        prob = r.home_prob if home else 1 - r.home_prob
        bprob = r.blend_prob if home else 1 - r.blend_prob
        if isinstance(r.winner, str) or r.played:
            us, them = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
            res = f"{'W' if us > them else 'L' if us < them else 'T'} {int(us)}-{int(them)}"
        else:
            res = f"{r.gameday:%a %b %-d}"
        kind = "" if r.game_type == "REG" else f" ({e(r.game_type)})"
        rows.append(f"<tr><td>Wk {r.week}{kind}</td><td>{'vs' if home else '@'} {tl(opp, '../')}</td><td>{res}</td>"
                    f"<td class='n'>{prob:.0%} {mark(r.correct)}</td><td class='n'>{bprob:.0%} {mark(r.blend_correct)}</td></tr>")
    sched = ('<div class="wrap"><table><tr><th>Week</th><th>Opponent</th><th>Result / date</th>'
             f'<th class="n">Model: {e(team)} wins</th><th class="n">Model + Vegas</th></tr>' + "".join(rows) + "</table></div>")

    wk_labels = [f"Wk {w}" for w in sorted(rating_hist)]
    rating_chart = line_chart(wk_labels, [("Rating", "--s1", [rating_hist[w].get(team) for w in sorted(rating_hist)])],
                              f"{team} power rating by week", 0, 1)
    oh = odds_hist.get(str(season), {})
    weeks = sorted(oh, key=int)
    odds_chart = line_chart([f"Wk {w}" for w in weeks], [
        ("Playoffs", "--s1", [oh[w].get(team, {}).get("playoffs") for w in weeks]),
        ("Win Super Bowl", "--s2", [oh[w].get(team, {}).get("super_bowl") for w in weeks])],
        f"{team} playoff odds by week", 0, 1)
    mv = int(rk["change"]) if rk["change"] == rk["change"] else 0
    move = f"▲ {mv}" if mv > 0 else (f"▼ {-mv}" if mv < 0 else "no change")
    hero = f"""{logo(team, 176, "hlogo")}<div class="kicker">{e(TEAM_DIV[team])} · {season}</div><h1>{e(team)} <span style="opacity:.6">{records[team]}</span></h1>
<p class="sub">Model ratings, schedule picks and playoff odds.</p>
<div class="stats">
<div class="stat"><b>#{int(rk['rank'])}</b><span>Power ranking ({move} this week)</span></div>
<div class="stat"><b>{rk['rating']:.0%}</b><span>Chance to beat an average team (neutral field)</span></div>
<div class="stat"><b>{pct(od['playoffs'])}</b><span>Playoff odds · {od['proj_wins']:.1f} projected wins</span></div>
<div class="stat"><b>{pct(od['super_bowl'])}</b><span>Super Bowl odds</span></div></div>"""
    body = f"""<p class="note" style="margin-top:16px">Offense rank {int(rk['off_rank'])} · Defense rank {int(rk['def_rank'])} (EPA per play)</p>
<h2>Schedule and picks</h2>{sched}
<h2>Power rating by week</h2><p class="note">Chance to beat an average NFL team on a neutral field, going into each week.</p>
{rating_chart}
<h2>Playoff odds by week</h2><p class="note">Saved each week from 10,000 season simulations (starts the week the tracker went live).</p>
{odds_chart}"""
    return page(f"{team} · NFL Game Predictor", hero, body, "../", TEAM_COLORS.get(team))


def rating_history(saved, season, upto):
    """Every team's power rating going into each week of the season (up to this week)."""
    out = {}
    weeks = sorted(w for k in SNAPSHOTS if k != "latest" for (s, w) in [k] if s == season and w <= upto)
    for w in weeks:
        out[w] = power_ratings(saved, SNAPSHOTS[(season, w)]).to_dict()
    return out


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
    last_week = all_preds[all_preds["week"] == week - 1]
    lw_right = int(last_week["correct"].isin([True]).sum())
    lw_total = int(last_week["correct"].isin([True, False]).sum())

    print("Simulating the season 10,000 times ...")
    odds = simulate_season(df, saved, season)
    records = team_records(df, season)
    ranks = power_rankings(saved, df, SNAPSHOTS, season)
    stats = team_stats(df, season, ranks, odds)
    cards = []
    for slot in best["slot"]:
        cards.append(f"<h3>{e(slot)}</h3>")
        cards += [game_card(r, r.game_id in best_ids, stats, saved.get('line_gaps')) for _, r in this_week[this_week["slot"] == slot].iterrows()]

    # Pick tracker + odds history (saved in docs/data/ so they build up over the season)
    picks = picks_frame(update_picks(all_preds), season)
    odds_hist = update_odds(odds, season, week)
    b_right = int((picks["blend_correct"] == True).sum()) if "blend_correct" in picks else 0  # noqa: E712
    b_total = int(picks["blend_correct"].isin([True, False]).sum()) if "blend_correct" in picks else 0

    stats = f"""<div class="stats">
<div class="stat"><b>{right}-{total - right}</b><span>Model record in {season} ({right / max(total, 1):.0%})</span></div>
<div class="stat"><b>{v_right}-{v_total - v_right}</b><span>Vegas favorites in {season} ({v_right / max(v_total, 1):.0%})</span></div>
<div class="stat"><b>{lw_right}-{lw_total - lw_right}</b><span>Model in week {week - 1}</span></div>
<div class="stat"><b>{b_right}-{b_total - b_right}</b><span>Model + Vegas blend in {season} ({b_right / max(b_total, 1):.0%})</span></div>
<div class="stat"><b>{ranks.iloc[0]['team']}</b><span>#1 in power rankings</span></div></div>"""

    features = {"base": "Elo team strength, recent form, rest, home field",
                "qb": "starting QB efficiency", "epa": "offense & defense efficiency (EPA per play, success rate)",
                "injuries": "injured starters", "weather": "weather", "travel": "travel & schedule",
                "stakes": "late-season stakes (seed locked or eliminated)"}
    used = "; ".join(features[g] for g in saved.get("groups", []))

    html_page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NFL Game Predictor</title>
<meta name="description" content="Machine learning picks, predicted scores, power rankings and playoff odds for every NFL game.">
<meta name="theme-color" content="#0b1a33"><style>{CSS}</style></head><body>
<header class="hero"><div class="in">
<div class="kicker">Week {week} · {season} season · updated {datetime.now():%b %-d, %Y}</div>
<h1>NFL Game Predictor</h1>
<p class="sub">Machine learning picks, predicted scores, power rankings and playoff odds for every game.</p>
{stats}</div></header>
<main>
<nav><a href="#glance">At a glance</a><a href="#mine">Your picks</a><a href="#slots">Top picks</a><a href="#upsets">Upset watch</a><a href="#picks">All picks</a><a href="#accuracy">Accuracy</a><a href="#tracker">Pick tracker</a>
<a href="#rankings">Power rankings</a><a href="#playoffs">Playoff odds</a>{'<a href="#results">Last week</a>' if lw_total else ''}
<a href="history.html">Every pick</a></nav>
<h2 id="glance">Week {week} at a glance</h2>
{glance_card(this_week)}
<h2 id="mine">Your picks</h2>
<div class="stats" id="myrec"></div>
<p class="note">Saved in this browser only (not shared, not sent anywhere). Picks lock at kickoff.</p>
<h2 id="slots">Most confident pick of each time slot</h2>
<div class="slots">{''.join(slot_card(r) for _, r in best.iterrows())}</div>
<h2 id="upsets">Upset watch</h2>
<p class="note">Underdogs with at least a 35% chance, from the Model + Vegas blend. The bar fills up at 50% (a coin flip).
Upsets are mostly luck, but these percentages are honest: in past seasons, underdogs won about as often as predicted.</p>
{upset_watch(this_week, saved.get("calibration"))}
<h2 id="picks">Week {week} picks</h2>
<p class="note">Bars are in team colors: away team on the left, home team on the right. Predicted scores come from separate spread and total models.
"Model + Vegas blend" combines the model with the betting line. It's the most accurate forecast on this page (in 2022–2025
testing it matched Vegas, 67.7% vs 67.6%). When there's no line yet, it's just the model.
Yellow boxes = the predicted score is {saved.get("line_gaps", {}).get("points", 4)}+ points away from the Vegas spread or total.
Break-even for betting is 52.4%, so these are interesting, not proven: see the hit rates.</p>
{''.join(cards)}
<section><h2 id="accuracy">{season} accuracy: model vs Vegas</h2>
<p class="note">Share of games picked correctly so far this season. (The model was trained only on earlier seasons,
so these are real predictions.)</p>
{accuracy_chart(all_preds, season)}
<h3>Are the percentages honest?</h3>
<p class="note">Every game from {saved.get("calibration", {}).get("seasons", "past seasons")}, each predicted using only earlier
seasons. Dots on the dashed line = when it says 70%, the favorite really wins about 70% of the time. Bigger dots = more games.</p>
{calibration_chart(saved.get("calibration"))}</section>
<section><h2 id="tracker">Pick tracker</h2>
<p class="note">Every pick is saved before kickoff and frozen once the game starts, so the record can't be rewritten.</p>
{tracker_section(picks)}</section>
<section><h2 id="rankings">Power rankings</h2>
<p class="note">Rating = chance to beat an average NFL team on a neutral field. Offense/defense = efficiency rank (EPA per play).
Arrows = movement since last week. Tap a team for its page.</p>
{rankings_table(ranks)}</section>
<section><h2 id="playoffs">Playoff odds</h2>
<p class="note">From 10,000 simulations of the rest of the season and the playoffs. Simplified tiebreakers:
teams tied on wins are ordered randomly.</p>
{playoff_tables(odds, records)}</section>
{f'<section><h2 id="results">Week {week - 1} results</h2>' + results_table(last_week) + '</section>' if lw_total else ''}
<footer><b>Teams</b><div class="teamnav">{''.join(tl(t) for t in TEAMS)}</div><br>
Model: {saved['name']} using {used}. It picked these by testing each group of stats and keeping only the
ones that improved predictions. Data from <a href="https://github.com/nflverse">nflverse</a>.
Code: <a href="https://github.com/zacharyivezi08/nfl-game-predictor">github.com/zacharyivezi08/nfl-game-predictor</a>.
<br><br><b>Not betting advice.</b> "Most confident" means most likely to win, not a good bet. In a 10-season backtest
against real odds, betting the model's picks lost about 2–6% of the money wagered, and its spread and over/under
picks hit about 49–50%, below the 52.4% needed to break even. Vegas is more accurate than this model.</footer>
</main>{my_picks_script(all_preds)}</body></html>"""

    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(html_page)
    (DOCS / "history.html").write_text(history_page(picks, season))
    (DOCS / "teams").mkdir(exist_ok=True)
    rating_hist = rating_history(saved, season, week)
    for t in TEAMS:
        (DOCS / "teams" / f"{t}.html").write_text(
            team_page(t, season, all_preds, ranks, odds, odds_hist, rating_hist, records))
    print(f"Built docs/index.html, history.html and {len(TEAMS)} team pages for {season} week {week}")


if __name__ == "__main__":
    main()
