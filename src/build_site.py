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

import numpy as np
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
.bar i{display:block;border-radius:5px;box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.55)}
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
.namebox{padding:12px 16px;border-radius:16px;margin-bottom:10px}.namerow{display:flex;gap:8px;margin:8px 0 4px}
.namerow input{flex:1;min-width:0;font:inherit;padding:8px 12px;border-radius:10px;border:1px solid rgba(255,255,255,.4);
background:#fff;color:#14171c}.namerow button{font:inherit;font-weight:800;padding:8px 16px;border-radius:10px;border:0;
background:#fff;color:#F26A00;cursor:pointer}
tr.me td{background:rgba(255,255,255,.2)}tr.bench td{font-style:italic;opacity:.85}
.sb{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:8px;margin:12px 0 4px;padding:14px 8px;
border-radius:14px;background:var(--card2);border:1px solid var(--grid)}
.sbt{display:flex;flex-direction:column;align-items:center;gap:2px;text-align:center}
.sbt .sblg{width:52px;height:52px;object-fit:contain}.sbt span{font-size:13px;font-weight:700;color:var(--muted)}
.sbt b{font-size:clamp(34px,8vw,46px);line-height:1;font-weight:900;letter-spacing:-.02em;color:var(--muted);
font-family:"Arial Narrow","Roboto Condensed","Helvetica Neue",sans-serif;font-stretch:condensed}
.sbt b.w{color:var(--good)}.sbt i.hid{visibility:hidden}.sbt i{font-style:normal;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:var(--good)}
.sbm{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);text-align:center;line-height:1.3}
.sbf{text-align:center;font-size:13px;font-weight:700;margin:2px 0}
.ppgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media (max-width:520px){.ppgrid{grid-template-columns:1fr}}
.ppgrid h5{margin:0 0 6px;font-size:15px}td.wrapcell{white-space:normal}table.acc td,table.acc th{padding:8px 8px;white-space:normal}td.wrapcell .meta{display:block}.pgame h4{margin:12px 0 6px}.pgame .teams{margin-bottom:4px}.pp{list-style:none;margin:0;padding:0}
.pp li{display:grid;grid-template-columns:1fr auto auto;gap:6px;align-items:baseline;padding:6px 0;border-bottom:1px solid var(--line)}
.pp li:last-child{border-bottom:0}.pp b{font-size:18px}.pp small{font-size:13.5px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 18px}
.tabs a{color:#fff;text-decoration:none;font-weight:800;font-size:15px;padding:8px 16px;border-radius:999px;
border:2px solid rgba(255,255,255,.55)}.tabs a:hover{background:rgba(255,255,255,.15)}
.tabs a.on{background:#fff;color:#a84400;border-color:#fff}
.upsets{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.upset{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--tc);border-radius:14px;padding:12px 14px;box-shadow:var(--shadow)}
.uprow{display:flex;align-items:center;gap:10px;margin:6px 0 4px}.uprow b{font-size:26px;font-weight:800;min-width:58px}
.umeter{flex:1;height:8px;border-radius:4px;background:var(--grid);overflow:hidden}.umeter i{display:block;height:8px;max-width:100%;background:var(--tc)}
@keyframes fade{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
/* White page, orange header + cards. Card orange (#C85000) is the brightest orange where white text meets the
   4.5:1 readability standard; charts and tables sit on white cards so every line, label and number is easy to read. */
:root,:root:not([data-theme="light"]){--bg:#ffffff;--ink:#14171c;--muted:#5b6472;--line:#e7e9ee;--grid:#eef0f3;--card2:#f6f7f9;
--s1:#1f6fd1;--s2:#e0561b;--good:#1a7f37;--bad:#c2255c;--card:#fff;--orange:#C85000}
body{background:#fff;color:var(--ink)}
.hero{background:radial-gradient(1200px 400px at 85% -20%,rgba(255,190,120,.35),transparent 60%),
linear-gradient(135deg,#b84a00,#ec6a0c)}
.hero .sub,.hero .stat span{color:#fff}.kicker,.back{color:#fff}
nav{background:rgba(255,255,255,.94);border-bottom-color:var(--line)}
nav a{background:#fff;color:#a84400;border-color:#f3c9a3}nav a:hover{border-color:var(--orange);color:#a84400}
h2::before{background:var(--orange)}
.game,.slot,.glance>div,.upset,main .stat,.teamnav a,.card{
--card:#C85000;--card2:rgba(0,0,0,.14);--ink:#fff;--muted:#fff;--line:rgba(255,255,255,.35);--grid:rgba(255,255,255,.25);
--s1:#fff;--good:#fff;--bad:#fff;--warn:#fff;--warnbg:rgba(0,0,0,.22);
background:#C85000;color:#fff;border-color:#a84400;box-shadow:0 2px 10px rgba(200,80,0,.25)}
.game:hover,.slot:hover{box-shadow:0 10px 28px rgba(200,80,0,.35)}.best{box-shadow:0 0 0 3px rgba(200,80,0,.35)}
th{background:rgba(0,0,0,.14)}
/* tables: white card, orange header row */
.wrap{background:#fff;color:#14171c;border-color:#ecd9c9;--ink:#14171c;--muted:#4b5563;--line:#eceef1;--grid:#eef0f3;
--card2:#fbf4ee;--good:#1a7f37;--bad:#c2255c;--s1:#a84400}
.wrap th{background:#C85000;color:#fff}.wrap tr:hover td{background:#fbf4ee}.wrap .meter{background:#f1e2d6}.wrap .meter i{background:#C85000}
.wrap .ok,.wrap .no{text-decoration:none}.wrap tr.me td{background:#fbe6d6}
.cmp .wrap,.game .wrap{--ink:#14171c}
.blend{background:rgba(0,0,0,.18)}.star{background:#fff;color:#a84400}.tag{background:rgba(0,0,0,.25);color:#fff}
.ok{text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:3px}
.no{text-decoration:line-through;text-decoration-thickness:2px}
.chart .ok,.chart .no{text-decoration:none}
.mypick button{color:#fff}.mypick button.on{background:#fff;color:#a84400;border-color:#fff}
.mypick button:disabled{opacity:1;border-style:dashed;cursor:default}
.cmp td.win{background:rgba(0,0,0,.2)!important}
.meter{background:rgba(255,255,255,.3)}.meter i{background:#fff}
.sb{background:rgba(0,0,0,.16);border-color:rgba(0,0,0,.1)}.sbt b{color:rgba(255,255,255,.85)}.sbt b.w{color:#fff}
.sbt i{background:#fff;color:#a84400;padding:2px 8px;border-radius:999px}
.flag{background:rgba(0,0,0,.2);border-left-color:#fff}.flag b{color:#fff}
.edges li.big{background:rgba(0,0,0,.22)}
tr.me td{background:rgba(0,0,0,.18)}
/* charts: white card, normal colors */
.chart{background:#fff;color:#14171c;border-color:var(--line);box-shadow:0 2px 10px rgba(20,23,28,.08)}
.tip{background:#14171c!important;color:#fff!important}
footer{color:var(--muted)}footer a{color:#a84400;font-weight:700}
.teamnav a{color:#fff}
.namerow button{color:#a84400}
/* Important info = light-green highlight with dark green text (readable on the orange cards) */
:root{--hl:#d6f5cd;--hlink:#0b3d17;--hlmid:#1d5c2b}
.flag{background:var(--hl);color:var(--hlink);border-left:4px solid #1f8a3b}.flag b{color:var(--hlink)}.flag .meta{color:var(--hlmid)}
.blend{background:var(--hl);color:var(--hlink)}.blend b{color:var(--hlink)}
.pick{background:var(--hl);color:var(--hlink);padding:2px 10px;border-radius:999px}
.tag{background:var(--hl);color:var(--hlink)}
.sbt i{background:var(--hl);color:var(--hlink)}
.edges li.big{background:var(--hl);color:var(--hlink)}.edges li.big .tb,.edges li.big b{color:var(--hlink)}
.game .ok,.upset .ok,.cmp .ok{background:var(--hl);color:var(--hlink);padding:1px 8px;border-radius:999px;text-decoration:none}
.game .no,.upset .no{text-decoration:none;background:rgba(0,0,0,.25);color:#fff;padding:1px 8px;border-radius:999px}
.uprow b{background:var(--hl);color:var(--hlink);padding:0 10px;border-radius:10px}
/* Easier reading: bigger base text, no tiny labels, a bit more line spacing */
body{font-size:16px;line-height:1.55}
.note,.meta,.why,.score,.blend,.flag,.glance li,.upset .meta,.slot small,footer{font-size:14.5px}
h3,.slot span,.sbm,th,.glance h4,.cmp h4,.kicker,.star,.tag,.sbt i,.more,.legend{font-size:12.5px;letter-spacing:.04em}
.stat span{font-size:14px}td,th{font-size:15px}.cmp th{font-size:12.5px}.cmp td{font-size:15.5px}
.mypick button{font-size:14.5px}.edges li{font-size:14.5px}.sbt span{font-size:14.5px}
/* chart text: sized in chart units, so it's bigger on phones where the chart shrinks */
.chart svg text{font-size:14px}
@media (max-width:600px){.chart svg text{font-size:22px}}
.odds,.range{font-size:14.5px;margin-top:6px}.agree{margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.tag.split{background:rgba(0,0,0,.25);color:#fff}
/* compact game rows: tap to open everything */
details.game{padding:0;margin:8px 0;overflow:hidden}
summary.crow{display:grid;grid-template-columns:1fr auto 1fr auto 18px;align-items:center;gap:8px;padding:12px 14px;cursor:pointer;list-style:none}
summary.crow::-webkit-details-marker{display:none}
.cteam{display:flex;align-items:center;gap:6px;min-width:0}.cteam .clg{width:28px;height:28px;object-fit:contain;flex:none}
.cab{font-weight:800;font-size:16px}.cteam b{font-size:22px;font-weight:900;margin-left:auto;opacity:.8}.cteam b.cw{opacity:1}
.cat{opacity:.85;font-size:13px}
.cside{display:flex;flex-direction:column;align-items:flex-end;gap:3px;min-width:92px}.cside .pick{font-size:15px}
.ckick{font-size:12.5px;white-space:nowrap}.cicons{display:flex;gap:4px}
.ic{font-style:normal;font-size:11.5px;font-weight:800;padding:1px 7px;border-radius:999px;background:var(--hl);color:var(--hlink)}
.ic.split{background:rgba(0,0,0,.28);color:#fff}.ic.star{background:#fff;color:#a84400}
.cchev{font-size:14px;transition:transform .2s;text-align:center}details[open] .cchev{transform:rotate(180deg)}
details.game .details{padding:0 14px 14px;border-top:1px solid rgba(255,255,255,.3);animation:fade .2s ease-out}
details.game .details .sb{margin-top:12px}
@media (max-width:430px){summary.crow{grid-template-columns:1fr 1fr auto 14px;row-gap:4px}.cat{display:none}
 .cteam b{font-size:20px}.cside{grid-column:3;grid-row:1/3}.cchev{grid-column:4;grid-row:1/3}
 .cteam:nth-child(3){grid-column:2;grid-row:1}}
/* game list rows (link to game page) */
a.glink{display:grid;grid-template-columns:1fr auto 16px;gap:10px;align-items:center;text-decoration:none;color:#fff;padding:12px 14px;margin:8px 0}
.gteams{display:flex;flex-direction:column;gap:6px}.gt{display:flex;align-items:center;gap:8px}
.gt .clg{width:30px;height:30px;object-fit:contain}.gt span{font-weight:800;font-size:17px}.gt b{margin-left:auto;font-size:22px;font-weight:900;opacity:.75}
.gt b.cw{opacity:1}.gmeta{display:flex;flex-direction:column;align-items:flex-end;gap:4px;font-size:13px;text-align:right}
.gmeta .venue{max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
a.glink .cchev{font-size:26px;font-weight:800}
/* game page */
.ghead{display:flex;align-items:center;justify-content:center;gap:18px;margin:6px 0}.ghead div{display:flex;flex-direction:column;align-items:center;gap:4px}
.ghead .glg{width:72px;height:72px;object-fit:contain}.ghead b{font-size:22px}.ghead span{font-size:20px;opacity:.85}
.gtabs{display:flex;gap:6px;justify-content:center;flex-wrap:wrap;margin-top:12px}
.gtabs button,.gtabs a{font:inherit;font-weight:800;font-size:15px;padding:8px 16px;border-radius:999px;border:2px solid rgba(255,255,255,.6);
background:transparent;color:#fff;cursor:pointer;text-decoration:none}.gtabs button.on{background:#fff;color:#a84400;border-color:#fff}
.ch{margin:0 0 4px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#fff}.ch2{margin:0 0 8px;font-size:14px;letter-spacing:.04em;text-transform:uppercase}
.simcard{padding:16px}.dcard{text-align:center}.dlegend{display:flex;justify-content:center;gap:16px;font-weight:700;font-size:15px;margin:6px 0}
.dlegend i{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.fair{background:var(--hl);color:var(--hlink);border-radius:16px;padding:14px 16px;border:1px solid #b9e6ad}.fair .meta{color:var(--hlmid)}
.fodds{display:flex;justify-content:space-around;margin:6px 0}.fodds div{display:flex;flex-direction:column;align-items:center}
.fodds span{font-weight:700}.fodds b{font-size:34px;font-weight:900}
.pstat{margin:4px 4px 14px}.pstat h5{margin:0 0 6px;font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:#4b5563}
.pbar{margin:4px 0}.prow{display:flex;justify-content:space-between;font-size:15px}.ptrack{height:14px;border-radius:7px;background:#eef0f3;overflow:hidden}
.ptrack i{display:block;height:14px;border-radius:7px}
details.sect{margin:10px 0}details.sect>summary{cursor:pointer;font-weight:800;font-size:18px;padding:8px 2px;list-style:none;color:var(--ink)}
details.sect>summary::after{content:" ▾"}details.sect:not([open])>summary::after{content:" ▸"}
table.cmpt th{text-transform:none;letter-spacing:0;font-size:15px;font-weight:600;background:none;color:#14171c;white-space:normal}
table.cmpt td{text-align:center;font-weight:700;font-size:15.5px}table.cmpt td.vs{color:#6b7280;font-weight:500;width:28px}
table.cmpt td.good{background:#d6f5cd!important;color:#0b3d17;border-radius:8px}table.cmpt td.bad{background:#fde2e2!important;color:#7f1d1d;border-radius:8px}
table.cmpt td.ct .tb{color:#14171c}
.glist{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}.gl{padding:12px 14px;border-radius:14px}
.gl p{margin:6px 0 0;font-size:14.5px}.chip{font-size:12px;font-weight:800;padding:2px 10px;border-radius:999px;background:#fff;color:#a84400}
/* tab icons; on phones the tabs become an app-style bar at the bottom of the screen */
.tabs a{display:inline-flex;align-items:center;gap:6px}.tabs svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:2.2;
stroke-linecap:round;stroke-linejoin:round}
@media (max-width:700px){
 .tabs{position:fixed;left:0;right:0;bottom:0;z-index:60;margin:0;display:grid;grid-template-columns:repeat(5,1fr);gap:0;
  background:#fff;border-top:1px solid #e7e9ee;padding:6px 4px calc(6px + env(safe-area-inset-bottom));box-shadow:0 -4px 16px rgba(0,0,0,.08)}
 .tabs a{flex-direction:column;gap:2px;border:0;border-radius:12px;padding:5px 2px;color:#4b5563;font-size:11.5px;font-weight:800;justify-content:center}
 .tabs a svg{width:22px;height:22px}.tabs a.on{background:#fbe6d6;color:#a84400}.tabs a:hover{background:#f6f7f9}
 body{padding-bottom:84px}}
.hidden{display:none!important}
.favbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px 10px;margin:14px 0 4px}.favbar label{font-weight:800}
.favbar select{font:inherit;font-weight:700;padding:7px 10px;border-radius:10px;border:1.5px solid #f3c9a3;background:#fff;color:#14171c}
.fav{outline:3px solid #1f8a3b;outline-offset:2px}tr.fav td{background:#d6f5cd!important;color:#0b3d17}
tr.fav td .tb{color:#0b3d17}
.weeks{display:flex;align-items:center;gap:6px;overflow-x:auto;padding:12px 0 6px;scrollbar-width:none}.weeks::-webkit-scrollbar{display:none}
.weeks span{font-weight:800;margin-right:4px}.weeks a.wk{flex:none;min-width:38px;text-align:center;padding:7px 0;border-radius:10px;
border:1.5px solid #f3c9a3;color:#a84400;font-weight:800;text-decoration:none;background:#fff}.weeks a.wk.on{background:#C85000;color:#fff;border-color:#C85000}
.recap{padding:14px 16px;border-radius:16px;margin:10px 0}.rgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:6px 0}
.rgrid div{background:rgba(0,0,0,.16);border-radius:12px;padding:8px;text-align:center}.rgrid b{display:block;font-size:24px;font-weight:900}
.recap ul{margin:8px 0 0;padding-left:18px}.recap li{margin:4px 0}
.glink.nolink{cursor:default}
.tag.split{background:#374151!important;color:#fff!important}
/* quick picks */
.qpbox{margin:6px 0 10px}.qphead{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}
#qpfill{font:inherit;font-weight:800;font-size:14px;padding:8px 14px;border-radius:999px;border:2px solid #C85000;background:#fff;color:#a84400;cursor:pointer}
.qplist{display:flex;flex-direction:column;gap:8px}
.qp{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:6px;margin:0;padding:8px;border-radius:16px;
background:#fff;border:1px solid #ecd9c9;box-shadow:0 1px 4px rgba(20,23,28,.06)}
.qp button{display:flex;align-items:center;gap:8px;justify-content:center;font:inherit;padding:10px 8px;border-radius:12px;
border:2px solid #ecd9c9;background:#fbf4ee;color:#14171c;cursor:pointer;min-height:52px}
.qp button b{font-size:17px}.qp button small{font-size:13px;color:#4b5563}.qp .qlg{width:28px;height:28px;object-fit:contain}
.qp button.on{background:#C85000;border-color:#C85000;color:#fff}.qp button.on small{color:#fff}
.qp button:disabled{opacity:1;border-style:dashed;cursor:default}
.qp .qat{color:#6b7280;font-weight:700}.qp .qk,.qp .res{grid-column:1/-1;text-align:center;font-size:13px;color:#4b5563;margin-top:-2px}
.qp .res:empty{display:none}.qp .res .ok{background:#d6f5cd;color:#0b3d17}.qp .res .no{background:#fde2e2;color:#7f1d1d}
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


CARD_ORANGE = "#C85000"


def on_card(team):
    """Team color that stays visible on the orange cards (second color, or white, for orange teams)."""
    c = TEAM_COLORS.get(team, "#ffffff")
    if _too_close(c, CARD_ORANGE):
        c = TEAM_ALT_COLORS.get(team, "#ffffff")
        if _too_close(c, CARD_ORANGE):
            c = "#ffffff"
    return c


def bar_colors(away, home):
    """Each team's own color; if they'd look the same, the away team switches to its second color."""
    h = on_card(home)
    a = on_card(away)
    if _too_close(a, h):
        a = TEAM_ALT_COLORS.get(away, a)
        if _too_close(a, h):
            a = "#ffffff"  # white as a last resort
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
        cards.append(f"""<div class="upset" style="--tc:{on_card(dog)}">
<div class="row"><div class="teams">{tl(dog, prefix)} <small>over</small> {tl(fav, prefix)}</div><div>{badge} {result}</div></div>
<div class="uprow"><b>{p:.0%}</b><div class="umeter"><i style="width:{p * 200:.0f}%"></i></div></div>
<div class="meta">{hist}</div></div>""")
    return '<div class="upsets">' + "".join(cards) + "</div>"


def calibration_chart(cal):
    """Predicted chance vs how often it actually happened, for the model and the blend."""
    if not cal or not cal.get("blend"):
        return ""
    W, H, L, R, T, B = 640, 320, 64, 48, 24, 60
    lo, hi = 0.45, 1.0
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    out = ""
    for g in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        out += (f'<line x1="{L}" x2="{W - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="var(--grid)"/>'
                f'<text x="{L - 8}" y="{y(g) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{g:.0%}</text>'
                f'<text x="{x(g):.1f}" y="{H - 32}" text-anchor="middle" font-size="11" fill="var(--muted)">{g:.0%}</text>')
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


def quick_picks(week_preds, picks_url):
    """Every game this week in one list with two big team buttons: pick the whole week without opening each game."""
    from history import kickoff
    if week_preds is None or week_preds.empty:
        return ""
    rows = []
    for r in week_preds.sort_values(["gameday", "gametime", "game_id"]).itertuples():
        pa, ph = 1 - r.blend_prob, r.blend_prob
        rows.append(
            f'<div class="qp mypick" data-g="{e(r.game_id)}" data-k="{kickoff(r).isoformat()}" data-m="{e(r.blend_pick)}">'
            f'<button type="button" data-t="{e(r.away_team)}">{logo(r.away_team, 64, "qlg")}<b>{e(r.away_team)}</b>'
            f'<small>{pa:.0%}</small></button><span class="qat">@</span>'
            f'<button type="button" data-t="{e(r.home_team)}">{logo(r.home_team, 64, "qlg")}<b>{e(r.home_team)}</b>'
            f'<small>{ph:.0%}</small></button>'
            f'<span class="qk">{e(kick_text(r))}</span><span class="res"></span></div>')
    name_note = ('<p class="note hidden" id="qpname"><b>Want to be on the leaderboard?</b> '
                 '<a href="leaderboard.html">Add your name</a> first, then your picks count.</p>') if picks_url else ""
    return f"""<div class="qpbox">
<div class="qphead"><span id="qpcount"></span><button type="button" id="qpfill">Fill the rest with the model's picks</button></div>
{name_note}<div class="qplist">{''.join(rows)}</div>
<p class="note">Tap a team to pick it, tap again to undo. % = each team's chance (best estimate). Picks lock at kickoff.</p></div>"""


def load_site_config():
    """site_config.json at the repo root: {"picks_url": "<Google Apps Script web app URL>"} ("" = picks stay local)."""
    try:
        return json.loads((DOCS.parent / "site_config.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_schedule(all_preds):
    """docs/data/schedule.json: kickoff time + teams for every game, so the picks sheet can lock games at kickoff."""
    from history import kickoff
    sched = {r.game_id: {"kick": kickoff(r).isoformat(), "home": r.home_team, "away": r.away_team}
             for r in all_preds.itertuples()}
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "data" / "schedule.json").write_text(json.dumps(sched))


def my_picks_section(url, standalone=False):
    if not url:
        return ('<div class="stats" id="myrec"></div>'
                '<p class="note">Saved in this browser only (not shared, not sent anywhere). Picks lock at kickoff.</p>')
    return """<div class="namebox card"><label for="myname"><b>Your name for the leaderboard</b></label>
<div class="namerow"><input id="myname" maxlength="24" placeholder="Name or nickname" autocomplete="nickname">
<button type="button" id="savename">Save</button></div><span class="meta" id="namestatus"></span></div>
<div class="stats" id="myrec"></div>
<p class="note">Picks lock at kickoff. <b>Your name and picks are saved to the site owner's Google Sheet and shown on the
leaderboard</b> once each game kicks off. Leave the name blank to keep your picks private in this browser only.</p>
<h2 id="leaderboard">Standings</h2>
<div id="lb"><p class="note">Loading…</p></div>"""


def my_picks_script(all_preds, url="", crown_week=0):
    """Results for every finished game this season, so records can be scored in the browser. If a picks URL is set,
    picks are also sent to the Google Sheet and everyone's locked picks come back for the leaderboard."""
    res = {}
    for r in all_preds.itertuples():
        if r.played:
            res[r.game_id] = {"w": r.winner if isinstance(r.winner, str) else "TIE", "m": r.pick, "wk": int(r.week),
                              "v": r.vegas_pick if isinstance(r.vegas_pick, str) else None}
    return """<script>
(function(){const R=%s,URL=%s,CROWNWK=%d;let P={},NAME='';
try{P=JSON.parse(localStorage.getItem('nflgp-picks')||'{}');NAME=localStorage.getItem('nflgp-name')||''}catch(e){}
function save(){try{localStorage.setItem('nflgp-picks',JSON.stringify(P));localStorage.setItem('nflgp-name',NAME)}catch(e){}}
function rec(a,b){return a+'-'+(b-a)+(b?' ('+Math.round(100*a/b)+'%%)':'')}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function isLocked(el){return R[el.dataset.g]||Date.now()>=Date.parse(el.dataset.k)}
function send(g,team,el){
 if(!URL||!NAME)return;const s=el&&el.querySelector('.res');if(s)s.textContent='Saving…';
 fetch(URL,{method:'POST',body:JSON.stringify({name:NAME,game:g,team:team||''})}).then(r=>r.json()).then(j=>{
  if(s)s.innerHTML=j.ok?'<span class="ok">Saved ✓</span>':'<span class="no">'+esc(j.error||'Not saved')+'</span>';
 }).catch(()=>{if(s)s.innerHTML='<span class="no">Not saved (offline?)</span>'});}
function render(){
 document.querySelectorAll('.mypick').forEach(el=>{const g=el.dataset.g,done=R[g],locked=isLocked(el);
  el.querySelectorAll('button').forEach(b=>{b.classList.toggle('on',P[g]===b.dataset.t);b.disabled=!!locked;});
  const s=el.querySelector('.res');
  if(done&&P[g])s.innerHTML=done.w===P[g]?'<span class="ok">✓ Right</span>':(done.w==='TIE'?'Tie':'<span class="no">✗ Wrong</span>');
  else if(locked&&!P[g])s.textContent='Locked (kicked off)';});
 let n=0,y=0,m=0,v=0,vn=0,open=0;
 for(const g in P){const r=R[g];if(!r){open++;continue}if(r.w==='TIE')continue;n++;if(P[g]===r.w)y++;if(r.m===r.w)m++;
  if(r.v){vn++;if(r.v===r.w)v++}}
 const box=document.getElementById('myrec');if(!box)return;
 box.innerHTML=n?'<div class="stat"><b>'+rec(y,n)+'</b><span>Your picks</span></div><div class="stat"><b>'+rec(m,n)+
  '</b><span>Model on the same games</span></div><div class="stat"><b>'+rec(v,vn)+'</b><span>Vegas on the same games</span></div>'+
  (open?'<div class="stat"><b>'+open+'</b><span>Picks waiting on results</span></div>':'')
  :'<p class="note">'+(open?'You have '+open+' pick(s) waiting on results. ':'')+'Tap a team under any game below to make your pick. '+
  'Picks lock at kickoff, and your record vs the model shows up here once games finish.</p>';}
function qpCount(){const c=document.getElementById('qpcount');if(!c)return;const els=[...document.querySelectorAll('.qp')];
 const open=els.filter(el=>!isLocked(el)),done=open.filter(el=>P[el.dataset.g]).length;
 c.innerHTML=open.length?'<b>'+done+' of '+open.length+'</b> open games picked':'All games this week have kicked off';
 const f=document.getElementById('qpfill');if(f)f.hidden=!open.length||done===open.length;
 const n=document.getElementById('qpname');if(n)n.classList.toggle('hidden',!!NAME||!URL);}
document.addEventListener('click',ev=>{if(ev.target.id!=='qpfill')return;
 document.querySelectorAll('.qp').forEach(el=>{const g=el.dataset.g;if(isLocked(el)||P[g])return;P[g]=el.dataset.m;send(g,P[g],el);});
 save();render();qpCount();});
document.addEventListener('click',ev=>{const b=ev.target.closest('.mypick button');if(!b)return;ev.preventDefault();ev.stopPropagation();
 if(b.disabled)return;const el=b.closest('.mypick'),g=el.dataset.g;if(P[g]===b.dataset.t)delete P[g];else P[g]=b.dataset.t;
 save();render();send(g,P[g],el);qpCount();});
// name box: saving a name also sends any picks already made for games that haven't kicked off
const ni=document.getElementById('myname'),sb=document.getElementById('savename'),ns=document.getElementById('namestatus');
if(ni){ni.value=NAME;if(NAME)ns.textContent='Picks are going to the leaderboard as '+NAME+'.';
 sb.addEventListener('click',()=>{NAME=ni.value.replace(/\s+/g,' ').trim().slice(0,24);save();
  ns.textContent=NAME?'Saved. Picks are going to the leaderboard as '+NAME+'.':'No name: picks stay in this browser only.';
  document.querySelectorAll('.mypick').forEach(el=>{const g=el.dataset.g;if(P[g]&&!isLocked(el))send(g,P[g],el)});});}
// leaderboard
function leaderboard(){const lb=document.getElementById('lb');if(!lb||!URL)return;
 fetch(URL).then(r=>r.json()).then(d=>{const ppl={};
  (d.picks||[]).forEach(p=>{const k=String(p.n).toLowerCase();const o=ppl[k]||(ppl[k]={name:p.n,w:0,n:0,open:0});
   const r=R[p.g];if(!r){o.open++;return}if(r.w==='TIE')return;o.n++;if(p.t===r.w)o.w++;});
  const crown=weekWinner(d.picks||[]);
  let mw=0,mn=0,vw=0,vn=0;for(const g in R){const r=R[g];if(r.w==='TIE')continue;mn++;if(r.m===r.w)mw++;if(r.v){vn++;if(r.v===r.w)vw++}}
  const rows=Object.values(ppl).sort((a,b)=>b.w-a.w||(b.w/Math.max(b.n,1))-(a.w/Math.max(a.n,1)));
  if(!rows.length){lb.innerHTML='<p class="note">No locked picks yet. Names show up here once their picked games kick off.</p>';return}
  lb.innerHTML='<div class="wrap"><table><tr><th class="n">#</th><th>Name</th><th class="n">Record</th><th class="n">Pending</th></tr>'+
   rows.map((o,i)=>'<tr'+(o.name.toLowerCase()===NAME.toLowerCase()?' class="me"':'')+'><td class="n">'+(i+1)+'</td><td><b>'+esc(o.name)+
   '</b>'+(crown&&crown.names.includes(o.name.toLowerCase())?' <span title="Week '+crown.wk+' winner">👑</span>':'')+'</td><td class="n">'+rec(o.w,o.n)+'</td><td class="n">'+o.open+'</td></tr>').join('')+
   '<tr class="bench"><td></td><td>Model (every game)</td><td class="n">'+rec(mw,mn)+'</td><td></td></tr>'+
   '<tr class="bench"><td></td><td>Vegas favorite (every game)</td><td class="n">'+rec(vw,vn)+'</td><td></td></tr></table></div>'+
   '<p class="note">Ranked by wins. 👑 = best record last week'+(crown?' (Week '+crown.wk+')':'')+'. Only picks made on this site after the leaderboard started count.</p>';
 }).catch(()=>{lb.innerHTML='<p class="note">Couldn\\'t load the leaderboard right now.</p>'});}
// weekly winner: best record in the most recent week with results (at least 3 picks), ties share the crown
function weekWinner(picks){const wk=CROWNWK;if(!wk)return null;
 const by={};picks.forEach(p=>{const r=R[p.g];if(!r||r.wk!==wk||r.w==='TIE')return;const k=String(p.n).toLowerCase();
  const o=by[k]||(by[k]={name:p.n,w:0,n:0});o.n++;if(p.t===r.w)o.w++;});
 const c=Object.values(by).filter(o=>o.n>=3);if(!c.length)return null;const best=Math.max(...c.map(o=>o.w));
 const win=c.filter(o=>o.w===best);return {wk:wk,names:win.map(o=>o.name.toLowerCase()),label:win.map(o=>esc(o.name)).join(', '),rec:best+'-'+(win[0].n-best)};}
function recapWinner(){const li=document.getElementById('lbwinner');if(!li||!URL)return;
 fetch(URL).then(r=>r.json()).then(d=>{const c=weekWinner(d.picks||[]);if(!c)return;
  li.innerHTML='<b>👑 Leaderboard winner:</b> '+c.label+' ('+c.rec+' in Week '+c.wk+')';li.classList.remove('hidden');}).catch(()=>{});}
render();qpCount();leaderboard();recapWinner();})();
</script>""" % (json.dumps(res), json.dumps(url or ""), int(crown_week))


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


# ---------- player yardage projections ----------

STAT_SHORT = {"pass": "pass yds", "rush": "rush yds", "rec": "rec yds"}


def player_block(proj, r):
    """Projected yards for both teams, shown inside a game's expanded card."""
    if proj is None or proj.empty:
        return ""
    g = proj[proj["game_id"] == r.game_id]
    if g.empty:
        return ""
    cols = []
    for team in (r.away_team, r.home_team):
        t = g[g["team"] == team]
        items = "".join(f"<li><span>{e(x.player)} <small>{e(x.position)}</small></span><b>{x.proj:.0f}</b>"
                        f"<small>{STAT_SHORT[x.stat]}</small></li>" for x in t.itertuples())
        cols.append(f"<div><h5>{tl(team)}</h5><ul class='pp'>{items}</ul></div>")
    return (f'<h4>Projected yards</h4><div class="ppgrid">{"".join(cols)}</div>'
            '')


def players_page(proj, ev, season, week, week_games=None):
    if proj is None or proj.empty:
        body = "<p class='note'>Player projections appear once this week's games are on the schedule.</p>"
    else:
        boards = []
        for key, title in (("pass", "Passing yards"), ("rush", "Rushing yards"), ("rec", "Receiving yards")):
            t = proj[proj["stat"] == key].sort_values("proj", ascending=False)
            rows = "".join(f"<tr><td class='n'>{i + 1}</td><td class='wrapcell'><b>{e(x.player)}</b> "
                           f"<span class='meta'>{e(x.position)} · {e(x.team)} vs {e(x.opp)}</span></td>"
                           f"<td class='n'><b>{x.proj:.0f}</b></td></tr>" for i, x in enumerate(t.itertuples()))
            boards.append(f"<h2 id='{key}'>{title} leaders</h2><div class='wrap'><table><tr><th class='n'>#</th><th>Player</th>"
                          f"<th class='n'>Yds</th></tr>{rows}</table></div>")
        by_game = ""
        if week_games is not None:
            for r in week_games.itertuples():
                blk = player_block(proj, r)
                if blk:
                    by_game += f'<div class="game pgame"><div class="teams">{tl(r.away_team)} <small>@</small> {tl(r.home_team)}' \
                               f' <span class="meta">{r.gameday:%a %b %-d}</span></div>{blk}</div>'
        nav = ('<nav><a href="#g">By game</a><a href="#pass">Passing</a><a href="#rush">Rushing</a><a href="#rec">Receiving</a>'
               '<a href="#acc">Accuracy</a></nav>')
        body = nav + (f'<h2 id="g">By game</h2>{by_game}' if by_game else "") + "".join(boards)
    test = ""
    if ev is not None and len(ev):
        rows = "".join(f"<tr><td class='wrapcell'>{e(x.stat)}</td><td class='n'><b>{x.our_miss:.1f}</b></td><td class='n'>{x.recent_avg_miss:.1f}</td>"
                       f"<td class='n'>{x.season_avg_miss:.1f}</td></tr>" for x in ev.itertuples())
        test = ("<h2 id='acc'>How accurate are these?</h2><p class='note'>Tested on 2022–2025 with models trained only on earlier seasons. "
                "Average miss in yards (lower is better). Single-game yardage is very noisy, so even the best projections miss by a lot; "
                "these beat simple averages but not by much.</p><div class='wrap'><table class='acc'><tr><th>Stat</th><th class='n'>Model</th>"
                "<th class='n'>Recent avg</th><th class='n'>Season avg</th></tr>" + rows + "</table></div>")
    hero = (f"<div class='kicker'>Week {week} · {season}</div><h1>Player projections</h1>"
            "<p class='sub'>Projected passing, rushing and receiving yards for this week's likely starters. "
            "Players listed Out or Doubtful are left off. Projections assume the player plays.</p>")
    return page(f"Player projections · Week {week}", hero, body + test, active="players")


# ---------- fair odds, score ranges, model agreement, luck, ATS/OU records ----------

def american(p):
    """Win probability -> fair American odds (no sportsbook cut), e.g. 0.64 -> -178, 0.36 -> +178."""
    p = min(max(float(p), 0.01), 0.99)
    return f"−{round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def book_odds(ml):
    return "-" if pd.isna(ml) else (f"+{int(ml)}" if ml > 0 else f"−{abs(int(ml))}")


def odds_line(r):
    """Fair odds from the Model + Vegas blend next to the sportsbook's moneyline."""
    fair = f"{e(r.away_team)} {american(1 - r.blend_prob)} · {e(r.home_team)} {american(r.blend_prob)}"
    book = ""
    if pd.notna(r.home_moneyline) and pd.notna(r.away_moneyline):
        book = (f' <span class="meta">· Sportsbook: {e(r.away_team)} {book_odds(r.away_moneyline)} · '
                f'{e(r.home_team)} {book_odds(r.home_moneyline)}</span>')
    return f'<div class="odds"><b>Fair odds:</b> {fair}{book}</div>'


def range_line(r, scores):
    """Where 8 in 10 games like this land (±1.28 typical misses around the predicted margin and total)."""
    if not scores or "margin_sd" not in scores or "pred_margin" not in r:
        return ""
    k = 1.28
    lo, hi = r.pred_margin - k * scores["margin_sd"], r.pred_margin + k * scores["margin_sd"]

    def side(m):
        if abs(m) < 0.5:
            return "a tie"
        return f"{e(r.home_team if m > 0 else r.away_team)} by {abs(m):.0f}"
    tlo, thi = r.pred_total - k * scores["total_sd"], r.pred_total + k * scores["total_sd"]
    return (f'<div class="range"><b>Likely range</b> <span class="meta">(8 in 10 games like this)</span>: '
            f'{side(lo)} to {side(hi)} · total {tlo:.0f}–{thi:.0f} points</div>')


def agreement_line(r, cal):
    """Do the model and Vegas pick the same winner? And how have those picks done?"""
    a = (cal or {}).get("agreement")
    if not a or not isinstance(r.vegas_pick, str):
        return ""
    if r.vegas_pick == r.pick:
        return (f'<div class="agree"><span class="tag">✓ Model &amp; Vegas agree</span> '
                f'<span class="meta">Agreed picks have won {a["agree_won"]:.0%} ({a["agree_n"]:,} games since 2012)</span></div>')
    return (f'<div class="agree"><span class="tag split">Split: model {e(r.pick)}, Vegas {e(r.vegas_pick)}</span> '
            f'<span class="meta">In splits, Vegas\'s side has won {a["split_vegas_won"]:.0%} '
            f'({a["split_n"]:,} games since 2012)</span></div>')


def season_extras(df, season):
    """Per team: Pythagorean luck (actual wins minus wins expected from points), and ATS / over-under records."""
    g = df[(df["season"] == season) & (df["game_type"] == "REG") & df["played"]]
    out = {}
    for t in TEAMS:
        home, away = g[g["home_team"] == t], g[g["away_team"] == t]
        pf = home["home_score"].sum() + away["away_score"].sum()
        pa = home["away_score"].sum() + away["home_score"].sum()
        n = len(home) + len(away)
        wins = ((home["home_score"] > home["away_score"]).sum() + (away["away_score"] > away["home_score"]).sum()
                + 0.5 * ((home["home_score"] == home["away_score"]).sum() + (away["away_score"] == away["home_score"]).sum()))
        exp = n * pf ** 2.37 / (pf ** 2.37 + pa ** 2.37) if pf + pa > 0 else 0
        ats, ou = [0, 0, 0], [0, 0, 0]  # wins/covers, losses, pushes
        for side, d in (("home", home), ("away", away)):
            for x in d.itertuples():
                if pd.notna(x.spread_line):
                    m = (x.home_score - x.away_score) - x.spread_line
                    m = m if side == "home" else -m
                    ats[0 if m > 0 else 1 if m < 0 else 2] += 1
                if pd.notna(x.total_line):
                    o = x.home_score + x.away_score - x.total_line
                    ou[0 if o > 0 else 1 if o < 0 else 2] += 1
        fmt = lambda a: f"{a[0]}-{a[1]}" + (f"-{a[2]}" if a[2] else "")
        out[t] = {"luck": wins - exp, "exp_wins": exp, "ats": fmt(ats), "ou": fmt(ou)}
    return out


def points_rating(p):
    """Chance to beat an average team (neutral field) -> points better/worse than average."""
    p = min(max(float(p), 0.01), 0.99)
    return 6.5 * np.log(p / (1 - p))


# ---------- app features: install, favorite team, week picker, link previews, recap ----------

SITE_URL = "https://zacharyivezi08.github.io/nfl-game-predictor/"

TAB_ICONS = {  # simple outline icons (24x24)
    "picks": '<path d="M4 6h16M4 12h16M4 18h10"/>',
    "rankings": '<path d="M6 20V10M12 20V4M18 20v-7"/>',
    "players": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/>',
    "results": '<path d="M4 13l5 5L20 7"/>',
    "leaderboard": '<path d="M7 4h10v5a5 5 0 0 1-10 0V4zM5 4h2M17 4h2M12 14v4M8 21h8"/>',
}


def head_extra(prefix="", og=None):
    """Install-as-app tags (home screen icon, full screen) + link preview tags."""
    tags = [f'<link rel="manifest" href="{prefix}manifest.json">',
            f'<link rel="icon" href="{prefix}icons/icon-192.png">',
            f'<link rel="apple-touch-icon" href="{prefix}icons/icon-192.png">',
            '<meta name="apple-mobile-web-app-capable" content="yes">',
            '<meta name="mobile-web-app-capable" content="yes">',
            '<meta name="apple-mobile-web-app-title" content="NFL Picks">',
            '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">']
    og = og or {"title": "NFL Game Predictor", "desc": "Free machine learning picks, predicted scores and playoff odds for every NFL game.",
                "image": SITE_URL + "icons/og-default.png", "url": SITE_URL}
    tags += [f'<meta property="og:title" content="{e(og["title"])}">',
             f'<meta property="og:description" content="{e(og["desc"])}">',
             f'<meta property="og:image" content="{e(og["image"])}">',
             f'<meta property="og:url" content="{e(og["url"])}">',
             '<meta property="og:type" content="website">',
             '<meta name="twitter:card" content="summary_large_image">']
    tags.append(f"<script>if('serviceWorker' in navigator)navigator.serviceWorker.register('{prefix}sw.js').catch(()=>{{}})</script>")
    return "\n".join(tags)


FAV_JS = """<script>
(function(){let F='';try{F=localStorage.getItem('nflgp-fav')||''}catch(e){}
const sel=document.getElementById('favsel');
function apply(){
 document.querySelectorAll('[data-teams]').forEach(el=>el.classList.toggle('fav',!!F&&el.dataset.teams.split(' ').includes(F)));
 const slot=document.getElementById('favslot');
 if(slot){slot.innerHTML='';const g=F&&document.querySelector('#games [data-teams~="'+F+'"]');
  if(g){const c=g.cloneNode(true);slot.innerHTML='<h2>Your team</h2>';slot.appendChild(c);}
  else if(F){slot.innerHTML='<p class="note">'+F+' doesn\\'t play this week.</p>';}}
}
if(sel){sel.value=F;sel.addEventListener('change',()=>{F=sel.value;try{localStorage.setItem('nflgp-fav',F)}catch(e){};apply();});}
apply();})();
</script>"""


def fav_picker():
    opts = "".join(f'<option value="{t}">{t}</option>' for t in TEAMS)
    return (f'<div class="favbar"><label for="favsel">Your team</label><select id="favsel"><option value="">— pick a team —</option>'
            f'{opts}</select><span class="meta">Saved on this device. Its game shows first.</span></div><div id="favslot"></div>')


def week_picker(weeks, current, prefix=""):
    chips = []
    for w in weeks:
        href = f"{prefix}index.html" if w == current else f"{prefix}weeks/week-{w}.html"
        chips.append(f'<a href="{href}" class="wk{" on" if w == current else ""}">{w}</a>')
    return f'<div class="weeks"><span>Week</span>{"".join(chips)}</div>'


def og_image(r, path):
    """1200x630 link preview card for a game (team colors, predicted score, the pick)."""
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                pass
        return ImageFont.load_default(size=size)
    away, home = r.away_team, r.home_team
    hpts, apts = (r.pred_total + r.pred_margin) / 2, (r.pred_total - r.pred_margin) / 2
    img = Image.new("RGB", (1200, 630), "#C85000")
    d = ImageDraw.Draw(img)
    for x in range(1200):  # left-to-right fade between the two teams' colors, darkened so white text reads
        t = x / 1199
        ca, ch = _rgb(TEAM_COLORS.get(away, "#C85000")), _rgb(TEAM_COLORS.get(home, "#C85000"))
        c = tuple(int((ca[i] * (1 - t) + ch[i] * t) * 0.55) for i in range(3))
        d.line([(x, 0), (x, 630)], fill=c)
    d.rectangle([0, 520, 1200, 630], fill="#C85000")
    W = lambda txt, f: d.textlength(txt, font=f)
    big, mid, small = font(150), font(64), font(40)
    for txt, cx, pts in ((away, 300, apts), (home, 900, hpts)):
        d.text((cx - W(txt, mid) / 2, 70), txt, font=mid, fill="white")
        s = f"{pts:.0f}"
        d.text((cx - W(s, big) / 2, 160), s, font=big, fill="white")
    d.text((600 - W("@", mid) / 2, 200), "@", font=mid, fill="white")
    line = f"Pick: {r.blend_pick} {r.blend_conf:.0%}  ·  Week {r.week}"
    d.text((600 - W(line, small) / 2, 400), line, font=small, fill="white")
    foot = "NFL Game Predictor · free ML picks"
    d.text((600 - W(foot, small) / 2, 552), foot, font=small, fill="white")
    img.save(path, optimize=True)


def recap_box(last_week, picks, week):
    """Last week in one box: model vs Vegas vs blend, the biggest upset, and the model's best call."""
    done = last_week[last_week["correct"].isin([True, False])] if len(last_week) else last_week
    if done.empty:
        return ""
    rec = lambda col: f"{int((done[col] == True).sum())}-{int(done[col].isin([True, False]).sum() - (done[col] == True).sum())}"  # noqa: E712
    ups = done[done["winner"] != done["blend_pick"]].copy()
    big = ""
    if len(ups):
        ups["dog_p"] = 1 - ups["blend_conf"]
        u = ups.sort_values("dog_p").iloc[0]
        loser = u.home_team if u.winner == u.away_team else u.away_team
        big = (f"<li><b>Biggest upset:</b> {tl(u.winner)} beat {tl(loser)} "
               f"<span class='meta'>(had a {u.dog_p:.0%} chance)</span></li>")
    calls = done[(done["correct"] == True) & done["vegas_pick"].notna() & (done["pick"] != done["vegas_pick"])]  # noqa: E712
    call = ""
    if len(calls):
        c = calls.iloc[0]
        call = f"<li><b>Model's best call:</b> picked {tl(c.pick)} against the Vegas favorite, and was right</li>"
    return f"""<div class="recap card"><h3 class="ch">Week {week - 1} recap</h3><div class="rgrid">
<div><b>{rec('correct')}</b><span>Model</span></div><div><b>{rec('vegas_correct')}</b><span>Vegas</span></div>
<div><b>{rec('blend_correct')}</b><span>Best estimate</span></div></div><ul>{big}{call}
<li id="lbwinner" class="hidden"></li></ul></div>"""


# ---------- season box-score stats (for game pages) ----------

def team_box(df, season):
    """Season-to-date team stats: yards, third downs, turnovers, explosive plays, SRS, home/away margins, betting."""
    from data import load_team_extra
    played = df[(df["season"] == season) & df["played"] & (df["game_type"] == "REG")]
    try:
        ex = load_team_extra([season])
        ex = ex[ex["game_id"].isin(played["game_id"])]
    except Exception as err:  # noqa: BLE001
        print(f"(box-score stats skipped: {err})")
        ex = pd.DataFrame()
    games = []
    for r in played.itertuples():
        games.append((r.home_team, r.away_team, r.home_score - r.away_score, True, r))
        games.append((r.away_team, r.home_team, r.away_score - r.home_score, False, r))
    # SRS (simple rating system): average margin + average opponent rating, solved by repeating until it settles
    srs = dict.fromkeys(TEAMS, 0.0)
    for _ in range(60):
        new = {}
        for t in TEAMS:
            g = [(opp, m) for team, opp, m, _, _ in games if team == t]
            new[t] = (np.mean([m for _, m in g]) + np.mean([srs.get(o, 0) for o, _ in g])) if g else 0.0
        mean = np.mean(list(new.values()))
        srs = {t: v - mean for t, v in new.items()}
    out = {}
    for t in TEAMS:
        g = [x for x in games if x[0] == t]
        n = len(g)
        d = {"srs": srs[t], "games": n}
        home_m = [m for team, _, m, home, _ in g if home]
        away_m = [m for team, _, m, home, _ in g if not home]
        d["home_margin"] = np.mean(home_m) if home_m else None
        d["away_margin"] = np.mean(away_m) if away_m else None
        ats_m, overs = [], []
        for team, _, m, home, r in g:
            if pd.notna(r.spread_line):
                ats_m.append(m - r.spread_line if home else m + r.spread_line)
            if pd.notna(r.total_line):
                tot = r.home_score + r.away_score - r.total_line
                if tot != 0:
                    overs.append(tot > 0)
        covers = [x for x in ats_m if x != 0]
        d["ats_pct"] = np.mean([x > 0 for x in covers]) if covers else None
        d["ats_margin"] = np.mean(ats_m) if ats_m else None
        d["over_pct"] = np.mean(overs) if overs else None
        if len(ex):
            e_ = ex[ex["team"] == t]
            k = max(len(e_), 1)
            s_ = e_.sum(numeric_only=True)
            d.update({
                "ypg": s_["off_yards"] / k, "pass_ypg": s_["off_pass_yds"] / k, "rush_ypg": s_["off_rush_yds"] / k,
                "plays_pg": s_["off_plays"] / k, "ypp": s_["off_yards"] / max(s_["off_plays"], 1),
                "fd_pg": s_["off_first_downs"] / k,
                "third_pct": s_["off_third_conv"] / max(s_["off_third_conv"] + s_["off_third_fail"], 1),
                "to_pg": s_["off_turnovers"] / k, "expl_pg": s_["off_explosive"] / k,
                "ypg_allowed": s_["def_yards"] / k, "pass_allowed": s_["def_pass_yds"] / k,
                "rush_allowed": s_["def_rush_yds"] / k, "ypp_allowed": s_["def_yards"] / max(s_["def_plays"], 1),
                "third_pct_allowed": s_["def_third_conv"] / max(s_["def_third_conv"] + s_["def_third_fail"], 1),
                "takeaways_pg": s_["def_turnovers"] / k, "expl_allowed": s_["def_explosive"] / k,
                "to_margin": (s_["def_turnovers"] - s_["off_turnovers"]) / k,
            })
        out[t] = d
    return out


# (label, key, which is better, format, glossary category)
GAME_SECTIONS = [
    ("Overall performance", [
        ("Record", "record", None, "{}"), ("Power ranking", "rank", "lo", "#{:.0f}"),
        ("Rating (points vs avg)", "rating_pts", "hi", "{:+.1f}"), ("Playoff odds", "playoffs", "hi", "{:.0%}"),
        ("Points per game", "ppg", "hi", "{:.1f}"), ("Points allowed per game", "papg", "lo", "{:.1f}"),
        ("Total yards per game", "ypg", "hi", "{:.1f}"), ("Pass yards per game", "pass_ypg", "hi", "{:.1f}"),
        ("Rush yards per game", "rush_ypg", "hi", "{:.1f}"), ("Plays per game", "plays_pg", None, "{:.1f}"),
        ("Yards per play", "ypp", "hi", "{:.2f}"), ("First downs per game", "fd_pg", "hi", "{:.1f}")]),
    ("Defense", [
        ("Yards allowed per game", "ypg_allowed", "lo", "{:.1f}"), ("Pass yards allowed", "pass_allowed", "lo", "{:.1f}"),
        ("Rush yards allowed", "rush_allowed", "lo", "{:.1f}"), ("Yards per play allowed", "ypp_allowed", "lo", "{:.2f}"),
        ("Opponent 3rd down %", "third_pct_allowed", "lo", "{:.0%}"), ("Big plays allowed per game", "expl_allowed", "lo", "{:.1f}"),
        ("Takeaways per game", "takeaways_pg", "hi", "{:.1f}")]),
    ("Advanced analytics", [
        ("SRS rating", "srs", "hi", "{:+.1f}"), ("Elo rating", "elo", "hi", "{:.0f}"),
        ("Offense EPA per play", "off_epa", "hi", "{:+.3f}"), ("Defense EPA per play allowed", "def_epa", "lo", "{:+.3f}"),
        ("Offense success rate", "off_sr", "hi", "{:.1%}"), ("Defense success rate allowed", "def_sr", "lo", "{:.1%}"),
        ("3rd down conversion %", "third_pct", "hi", "{:.0%}"), ("Big plays per game", "expl_pg", "hi", "{:.1f}"),
        ("Turnovers per game", "to_pg", "lo", "{:.1f}"), ("Turnover margin per game", "to_margin", "hi", "{:+.1f}"),
        ("Pass offense rank", "pass_off_rank", "lo", "#{:.0f}"), ("Run offense rank", "run_off_rank", "lo", "#{:.0f}"),
        ("Pass defense rank", "pass_def_rank", "lo", "#{:.0f}"), ("Run defense rank", "run_def_rank", "lo", "#{:.0f}"),
        ("QB EPA per dropback", "qb", "hi", "{:+.3f}"), ("Luck (wins vs expected)", "luck", None, "{:+.1f}"),
        ("Starters' snaps lost to injury", "inj", "lo", "{:.1f}")]),
    ("Betting", [
        ("Against the spread", "ats", None, "{}"), ("Cover %", "ats_pct", "hi", "{:.0%}"),
        ("Avg margin vs the spread", "ats_margin", "hi", "{:+.1f}"), ("Over / under", "ou", None, "{}"),
        ("Over %", "over_pct", None, "{:.0%}"), ("Home margin", "home_margin", "hi", "{:+.1f}"),
        ("Away margin", "away_margin", "hi", "{:+.1f}")]),
]

GLOSSARY = [
    ("Predicted final score", "Model", "From two models: one predicts the margin, one the total points. Home = (total + margin) / 2."),
    ("Win probability", "Model", "The model's chance each team wins, from team strength, QBs, efficiency, injuries and late-season stakes."),
    ("Best estimate (Model + Vegas)", "Model", "The model blended with the betting line. In testing it matched Vegas's accuracy, far better than the model alone."),
    ("Fair moneyline odds", "Betting", "The win probability written as betting odds with no sportsbook cut. 64% = −178, 36% = +178."),
    ("Likely range", "Model", "Where 8 in 10 games like this one end up (the prediction ± 1.28 typical misses)."),
    ("Spread / total flag", "Betting", "Shown when the predicted score is 4+ points from the Vegas spread or total. Those have been right about 54% (2012–2025)."),
    ("Model & Vegas agree / split", "Model", "Whether both pick the same winner. Agreed picks have won about 68%; in splits, Vegas's side has won about 56%."),
    ("Record", "Overall", "Wins and losses this regular season."),
    ("Power ranking / rating", "Overall", "Teams ranked by the model's chance to beat an average team on a neutral field."),
    ("Rating (points vs avg)", "Overall", "The power rating in points: +5 means about 5 points better than an average team."),
    ("Playoff odds", "Overall", "Share of 10,000 simulated seasons in which the team makes the playoffs."),
    ("Points per game / allowed", "Offense", "Average points scored and given up per game."),
    ("Total / pass / rush yards per game", "Offense", "Average offensive yards per game (net of sacks for total yards)."),
    ("Plays per game", "Offense", "Average runs and passes per game."),
    ("Yards per play", "Efficiency", "Total yards ÷ plays. A simple efficiency measure."),
    ("First downs per game", "Offense", "Average first downs gained per game."),
    ("Yards allowed (total / pass / rush)", "Defense", "Average yards given up per game."),
    ("Opponent 3rd down %", "Defense", "How often opponents convert third downs. Lower is better."),
    ("Big plays", "Efficiency", "Explosive plays: runs of 10+ yards and passes of 20+ yards, per game (and allowed)."),
    ("Takeaways / turnovers / margin", "Efficiency", "Interceptions + lost fumbles forced, committed, and the difference per game."),
    ("3rd down conversion %", "Efficiency", "Third downs converted into a first down or touchdown ÷ third-down attempts."),
    ("SRS rating", "Advanced", "Simple Rating System: average margin of victory adjusted for strength of schedule, in points."),
    ("Elo rating", "Advanced", "A running team rating (start 1500) that moves after every game based on result and margin."),
    ("EPA per play", "Advanced", "Expected Points Added: how much each play changes the expected points of the drive. Higher is better for offense, lower for defense."),
    ("Success rate", "Advanced", "Share of plays that gain at least 40% of needed yards on 1st down, 60% on 2nd, or 100% on 3rd/4th."),
    ("Pass / run offense and defense rank", "Advanced", "Rank (1–32) in EPA per pass play and per run play, recent games weighted most."),
    ("QB EPA per dropback", "Advanced", "The starting quarterback's EPA per pass attempt or sack over his last 16 games."),
    ("Luck (wins vs expected)", "Advanced", "Actual wins minus the wins a team's points scored and allowed would normally produce (Pythagorean expectation)."),
    ("Starters' snaps lost to injury", "Advanced", "Injured players listed Out/Doubtful/Questionable, weighted by how much they've been playing."),
    ("Against the spread / cover %", "Betting", "Record and win rate versus the point spread (pushes excluded)."),
    ("Avg margin vs the spread", "Betting", "How many points per game a team has beaten (or missed) the spread by."),
    ("Over / under, over %", "Betting", "Record of games going over or under the total, and the share that went over."),
    ("Home / away margin", "Betting", "Average point differential in home games and in away games."),
]


def donut(p_away, away, home):
    """Win probability donut chart (inline SVG)."""
    import math
    ca, ch = TEAM_COLORS.get(away, "#888"), TEAM_COLORS.get(home, "#2a78d6")
    if _too_close(ca, ch):
        ca = TEAM_ALT_COLORS.get(away, "#888")
    r, cx, cy, sw = 70, 90, 90, 30
    circ = 2 * math.pi * r
    a_len = circ * p_away
    return f"""<svg viewBox="0 0 180 180" width="180" height="180" role="img" aria-label="{e(away)} {p_away:.0%}, {e(home)} {1 - p_away:.0%}">
<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{ch}" stroke-width="{sw}"/>
<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{ca}" stroke-width="{sw}" stroke-dasharray="{a_len:.1f} {circ:.1f}"
 transform="rotate(-90 {cx} {cy})"/>
<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="15" font-weight="800" fill="#14171c">Win</text>
<text x="{cx}" y="{cy + 16}" text-anchor="middle" font-size="15" font-weight="800" fill="#14171c">chance</text></svg>
<div class="dlegend"><span><i style="background:{ca}"></i>{e(away)} {p_away:.1%}</span><span><i style="background:{ch}"></i>{e(home)} {1 - p_away:.1%}</span></div>"""


def bars(label, a_val, h_val, away, home, fmt="{:.0f}"):
    if a_val is None or h_val is None:
        return ""
    top = max(a_val, h_val, 1)
    ca, ch = TEAM_COLORS.get(away, "#888"), TEAM_COLORS.get(home, "#2a78d6")
    if _too_close(ca, ch):
        ca = TEAM_ALT_COLORS.get(away, "#888")
    row = lambda t, v, c: (f'<div class="pbar"><div class="prow"><b>{e(t)}</b><b>{fmt.format(v)}</b></div>'
                          f'<div class="ptrack"><i style="width:{100 * v / top:.0f}%;background:{c}"></i></div></div>')
    return f'<div class="pstat"><h5>{e(label)}</h5>{row(away, a_val, ca)}{row(home, h_val, ch)}</div>'


def compare_tables(r, stats, box):
    a = {**stats[r.away_team], **box.get(r.away_team, {}), "inj": r.away_inj_off + r.away_inj_def}
    h = {**stats[r.home_team], **box.get(r.home_team, {}), "inj": r.home_inj_off + r.home_inj_def}
    for d in (a, h):
        d["rating_pts"] = points_rating(d["rating"])
    out = []
    for title, rows in GAME_SECTIONS:
        trs = []
        for label, key, better, fmt in rows:
            va, vh = a.get(key), h.get(key)
            show = lambda v: "—" if v is None or (isinstance(v, float) and v != v) else fmt.format(v)
            ca = chh = ""
            if better and show(va) != "—" and show(vh) != "—" and show(va) != show(vh):
                a_better = (va > vh) if better == "hi" else (va < vh)
                ca, chh = ("good", "bad") if a_better else ("bad", "good")
            trs.append(f"<tr><th>{e(label)}</th><td class='{ca}'>{show(va)}</td><td class='vs'>vs</td><td class='{chh}'>{show(vh)}</td></tr>")
        out.append(f"""<details class="sect" open><summary>{e(title)}</summary><div class="wrap"><table class="cmpt">
<tr><td></td><td class='ct'>{tl(r.away_team, '../')}</td><td></td><td class='ct'>{tl(r.home_team, '../')}</td></tr>{''.join(trs)}
</table></div></details>""")
    return "".join(out)


def game_page(r, stats, box, saved, proj, gaps, all_preds, picks_url):
    """One page per game: Simulation tab + Team Comparison tab (like the big pick sites, but free and honest)."""
    away, home = r.away_team, r.home_team
    hpts, apts = (r.pred_total + r.pred_margin) / 2, (r.pred_total - r.pred_margin) / 2
    hw = hpts >= apts
    kick = kick_text(r)
    venue = e(str(r.stadium)) if hasattr(r, "stadium") and isinstance(r.stadium, str) else ""
    sim_proj = ""
    if proj is not None and not proj.empty:
        g = proj[proj["game_id"] == r.game_id]
        if len(g):
            pa = g[(g["team"] == away) & (g["stat"] == "pass")]["proj"].sum()
            ph = g[(g["team"] == home) & (g["stat"] == "pass")]["proj"].sum()
            ra = g[(g["team"] == away) & (g["stat"] == "rush")]["proj"].sum()
            rh = g[(g["team"] == home) & (g["stat"] == "rush")]["proj"].sum()
            sim_proj = (bars("Passing yards (starting QB)", pa, ph, away, home)
                        + bars("Rushing yards (top 2 running backs)", ra, rh, away, home))
    ba, bh = box.get(away, {}), box.get(home, {})
    season_bars = (bars("Points per game (season)", stats[away].get("ppg"), stats[home].get("ppg"), away, home, "{:.1f}")
                   + bars("Turnovers per game (season)", ba.get("to_pg"), bh.get("to_pg"), away, home, "{:.1f}"))
    fair_a, fair_h = american(1 - r.blend_prob), american(r.blend_prob)
    book = (f"<p class='meta'>Sportsbook: {e(away)} {book_odds(r.away_moneyline)} · {e(home)} {book_odds(r.home_moneyline)}</p>"
            if pd.notna(r.home_moneyline) else "")
    fav = home if r.blend_prob >= 0.5 else away
    hero = f"""<a class="back" href="../index.html">← All games</a>
<div class="ghead"><div>{logo(away, 128, "glg")}<b>{e(away)}</b></div><span>@</span><div>{logo(home, 128, "glg")}<b>{e(home)}</b></div></div>
<p class="sub" style="text-align:center">{e(kick)}{' · ' + venue if venue else ''} · Week {r.week}, {r.season}</p>
<div class="gtabs" role="tablist"><button class="on" data-tab="sim" role="tab">Simulation</button>
<button data-tab="cmp" role="tab">Team comparison</button><a href="../glossary.html">Stats glossary</a></div>"""
    sim = f"""<section id="sim" class="gtab">
<div class="game card simcard"><h3 class="ch">Final score prediction</h3>
<div class="sb"><div class="sbt">{logo(away, 128, "sblg")}<span>{e(away)}</span><b class="{'w' if not hw else ''}">{apts:.1f}</b>
<i{'' if not hw else ' class="hid"'}>{'Away win' if not hw else ''}</i></div>
<div class="sbm">Predicted<br>final score</div>
<div class="sbt">{logo(home, 128, "sblg")}<span>{e(home)}</span><b class="{'w' if hw else ''}">{hpts:.1f}</b>
<i{'' if hw else ' class="hid"'}>{'Home win' if hw else ''}</i></div></div>
{'<div class="sbf">Final: ' + e(away) + ' ' + str(int(r.away_score)) + ', ' + e(home) + ' ' + str(int(r.home_score)) + '</div>' if r.played else ''}
<div class="score">Spread {e(spread_text(home, away, r.pred_margin))} <span class="meta">(Vegas {e(spread_text(home, away, r.spread_line))})</span>
· Total {r.pred_total:.1f} <span class="meta">(Vegas {r.total_line if pd.notna(r.total_line) else '-'})</span></div>
{range_line(r, saved.get("scores"))}
{line_flags(r, gaps)}
{pick_buttons(r)}
</div>
<h2>Betting analysis</h2>
<div class="grid2">
<div class="chart dcard"><h3 class="ch2">Win probability</h3>{donut(1 - r.home_prob, away, home)}
<p class="meta">Model alone. <b>Best estimate</b> (Model + Vegas): {e(r.blend_pick)} {r.blend_conf:.0%}</p></div>
<div class="fair"><h3 class="ch2">Fair moneyline odds</h3><div class="fodds"><div><span>{e(away)}</span><b>{fair_a}</b></div>
<div><span>{e(home)}</span><b>{fair_h}</b></div></div>{book}
<p class="meta">From the best estimate, with no sportsbook cut. {e(fav)} is the favorite.</p></div></div>
{agreement_line(r, saved.get("calibration"))}
<div class="why"><b>Why the model leans this way:</b> {' · '.join(f"{e(l)} → {e(t)}" for l, t in r.reasons)}</div>
<h2>Projected team stats</h2>
<div class="chart">{sim_proj}{season_bars or ''}</div>
</section>"""
    cmp = f"""<section id="cmp" class="gtab" hidden>
<p class="note">Season averages so far. Green = better, red = worse. <a href="../glossary.html">What do these mean?</a></p>
{compare_tables(r, stats, box)}</section>"""
    script = """<script>document.querySelectorAll('.gtabs button').forEach(b=>b.addEventListener('click',()=>{
document.querySelectorAll('.gtabs button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('.gtab').forEach(s=>s.hidden=s.id!==b.dataset.tab);}));</script>"""
    ca, ch = TEAM_COLORS.get(away, "#C85000"), TEAM_COLORS.get(home, "#C85000")
    bg = f"linear-gradient(rgba(0,0,0,.5),rgba(0,0,0,.5)),linear-gradient(120deg,{ca},{ch})"
    og = {"title": f"{away} @ {home}: predicted {away} {apts:.0f}, {home} {hpts:.0f}",
          "desc": f"Pick: {r.blend_pick} {r.blend_conf:.0%} · Week {r.week} · free ML picks, simulation and team comparison",
          "image": f"{SITE_URL}og/{r.game_id}.png", "url": f"{SITE_URL}games/{r.game_id}.html"}
    return page(f"{away} @ {home} · Week {r.week}", hero, sim + cmp, "../", active="picks", bg=bg, og=og,
                extra=script + my_picks_script(all_preds, picks_url))


def glossary_page():
    cats = {"Model": "mdl", "Overall": "ovr", "Offense": "off", "Defense": "def", "Efficiency": "eff",
            "Advanced": "adv", "Betting": "bet"}
    items = "".join(f'<div class="gl card"><div class="row"><b>{e(n)}</b><span class="chip {cats[c]}">{c}</span></div>'
                    f'<p>{e(d)}</p></div>' for n, c, d in GLOSSARY)
    hero = "<div class='kicker'>Reference</div><h1>Stats glossary</h1><p class='sub'>What every number on this site means.</p>"
    return page("Stats glossary", hero, f'<div class="glist">{items}</div>', active=None)


def game_row(r, prefix="", link=True):
    """Compact game card on the Picks page that opens the game's own page."""
    hpts, apts = (r.pred_total + r.pred_margin) / 2, (r.pred_total - r.pred_margin) / 2
    hw = hpts >= apts
    venue = e(str(r.stadium)) if isinstance(getattr(r, "stadium", None), str) else ""
    tag_open = (f'<a class="glink game" data-teams="{e(r.away_team)} {e(r.home_team)}" href="{prefix}games/{e(r.game_id)}.html">'
                if link else f'<div class="glink game nolink" data-teams="{e(r.away_team)} {e(r.home_team)}">')
    return f"""{tag_open}
<div class="gteams"><div class="gt">{logo(r.away_team, 64, "clg")}<span>{e(r.away_team)}</span><b class="{'cw' if not hw else ''}">{apts:.0f}</b></div>
<div class="gt">{logo(r.home_team, 64, "clg")}<span>{e(r.home_team)}</span><b class="{'cw' if hw else ''}">{hpts:.0f}</b></div></div>
<div class="gmeta"><span class="pick">{e(r.blend_pick)} {r.blend_conf:.0%}</span><span>{e(kick_text(r))}</span>
{'<span class="venue">' + venue + '</span>' if venue else ''}</div><div class="cchev">{'›' if link else ''}</div>{'</a>' if link else '</div>'}"""


# ---------- click-to-compare team stats ----------

# (label, key, which is better: "hi" / "lo" / None, format)
COMPARE_STATS = [
    ("Record", "record", None, "{}"),
    ("Against the spread", "ats", None, "{}"),
    ("Over / under", "ou", None, "{}"),
    ("Luck (wins vs expected)", "luck", None, "{:+.1f}"),
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


def kick_text(r):
    """'Sun 1:00 PM' style kickoff (Eastern), or the final score once played."""
    if r.played:
        return f"Final {int(r.away_score)}-{int(r.home_score)}"
    t = r.gametime if isinstance(r.gametime, str) and ":" in r.gametime else "13:00"
    h, m = int(t[:2]), t[3:5]
    return f"{r.gameday:%a} {h % 12 or 12}:{m} {'PM' if h >= 12 else 'AM'} ET"


def game_card(r, is_best=False, stats=None, gaps=None, proj=None, saved=None):
    away_pct = round((1 - r.home_prob) * 100)
    ca, ch = bar_colors(r.away_team, r.home_team)
    vegas = f"Vegas: {e(r.vegas_pick)} {r.vegas_conf:.0%}" if isinstance(r.vegas_pick, str) else "Vegas: no line yet"
    qbs = f"{e(str(r.away_qb))} vs {e(str(r.home_qb))} · " if isinstance(r.home_qb, str) else ""
    why = " · ".join(f"{e(lbl)} → {e(team)}" for lbl, team in r.reasons)
    score = score_line = ""
    if "pred_margin" in r:
        v_total = f"{r.total_line:g}" if pd.notna(r.total_line) else "-"
        home_pts = (r.pred_total + r.pred_margin) / 2
        away_pts = (r.pred_total - r.pred_margin) / 2
        hw = home_pts >= away_pts
        final = ""
        if r.played:
            final = (f'<div class="sbf">Final: {e(r.away_team)} {int(r.away_score)}, {e(r.home_team)} {int(r.home_score)}</div>')
        score = f"""<div class="sb">
  <div class="sbt">{logo(r.away_team, 128, "sblg")}<span>{e(r.away_team)}</span><b class="{'w' if not hw else ''}">{away_pts:.1f}</b>
    <i{'' if not hw else ' class="hid"'}>Winner</i></div>
  <div class="sbm">Predicted<br>final score</div>
  <div class="sbt">{logo(r.home_team, 128, "sblg")}<span>{e(r.home_team)}</span><b class="{'w' if hw else ''}">{home_pts:.1f}</b>
    <i{'' if hw else ' class="hid"'}>Winner</i></div>
</div>{final}"""
        score_line = f"""<div class="score">Spread {e(spread_text(r.home_team, r.away_team, r.pred_margin))}
 <span class="meta">(Vegas {e(spread_text(r.home_team, r.away_team, r.spread_line))})</span>
 · Total {r.pred_total:.1f} <span class="meta">(Vegas {v_total})</span></div>"""
    blend = ""
    if isinstance(r.vegas_pick, str):
        blend = f'<div class="blend"><b>Best estimate: {e(r.blend_pick)} {r.blend_conf:.0%}</b> <span>(Model + Vegas)</span></div>'
    badges = '<span class="star">★ Top pick of slot</span> ' if is_best and r.games_in_slot > 1 else ''
    saved = saved or {}
    if "pred_margin" in r:
        hpts, apts = (r.pred_total + r.pred_margin) / 2, (r.pred_total - r.pred_margin) / 2
    else:
        hpts = apts = None
    hw = r.home_prob >= 0.5 if hpts is None else hpts >= apts
    home_txt = f"{hpts:.0f}" if hpts is not None else f"{r.home_prob:.0%}"
    away_txt = f"{apts:.0f}" if apts is not None else f"{1 - r.home_prob:.0%}"
    icons = []
    if is_best and r.games_in_slot > 1:
        icons.append('<i class="ic star">★ Top</i>')
    if line_flags(r, gaps):
        icons.append('<i class="ic">Flag</i>')
    if isinstance(r.vegas_pick, str) and r.vegas_pick != r.pick:
        icons.append('<i class="ic split">Split</i>')
    mini = f'<span class="cicons">{"".join(icons)}</span>' if icons else ""
    return f"""
<details class="game{' best' if is_best else ''}"><summary class="crow">
  <div class="cteam">{logo(r.away_team, 64, "clg")}<span class="cab">{e(r.away_team)}</span>
    <b class="{'cw' if not hw else ''}">{away_txt}</b></div>
  <div class="cat">@</div>
  <div class="cteam">{logo(r.home_team, 64, "clg")}<span class="cab">{e(r.home_team)}</span>
    <b class="{'cw' if hw else ''}">{home_txt}</b></div>
  <div class="cside"><span class="pick">{e(r.blend_pick)} {r.blend_conf:.0%}</span>
    <span class="ckick">{e(kick_text(r))}</span>{mini}</div>
  <div class="cchev" aria-hidden="true">▾</div>
</summary>
<div class="details">
  {score}
  <div class="row picks"><div>{badges}<span class="pick">Model: {e(r.pick)} {r.confidence:.0%}</span></div>{blend}</div>
  {line_flags(r, gaps)}
  {pick_buttons(r)}
  <div class="bar" role="img" aria-label="{e(r.away_team)} {away_pct}%, {e(r.home_team)} {100 - away_pct}%">
    <i style="width:{away_pct}%;background:{ca}"></i><i style="width:{100 - away_pct}%;background:{ch}"></i></div>
  <div class="row meta"><span>{e(r.away_team)} {away_pct}% · {e(r.home_team)} {100 - away_pct}%</span><span>{vegas}</span></div>
  <div class="meta">{qbs}{r.gameday:%a %b %-d}</div>
  {score_line}
  {range_line(r, saved.get("scores"))}
  {odds_line(r)}
  {agreement_line(r, saved.get("calibration"))}
  <div class="why"><b>Why:</b> {why}</div>
  {compare_panel(r, stats) if stats else ''}
</div>
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
    W, H, L, R, T, B = 640, 280, 64, 130, 16, 38
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
    step = max(1, -(-len(weeks) // 7))  # at most ~7 week labels so they never overlap
    xlab = "".join(f'<text x="{x(w):.1f}" y="{H - 8}" text-anchor="middle" font-size="11" fill="var(--muted)">Wk {w}</text>'
                   for i, w in enumerate(weeks) if i % step == 0 or i == len(weeks) - 1)
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


def rankings_table(pr, extras=None):
    extras = extras or {}
    rows = []
    for r in pr.itertuples():
        if r.change > 0:
            mv = f'<span class="up">▲ {int(r.change)}</span>'
        elif r.change < 0:
            mv = f'<span class="down">▼ {-int(r.change)}</span>'
        else:
            mv = '<span class="same">–</span>'
        pts = points_rating(r.rating)
        luck = extras.get(r.team, {}).get("luck", 0.0)
        luck_txt = f"{luck:+.1f}" if abs(luck) >= 0.05 else "0.0"
        rows.append(f"<tr data-teams='{r.team}'><td class='n'>{r.rank}</td><td>{mv}</td><td><b>{tl(r.team)}</b></td><td class='n'>{r.record}</td>"
                    f"<td class='n'><b>{pts:+.1f}</b></td><td class='n'>{r.rating:.0%}</td>"
                    f"<td class='n'>{r.off_rank}</td><td class='n'>{r.def_rank}</td><td class='n'>{luck_txt}</td></tr>")
    return ('<div class="wrap"><table><tr><th class="n">#</th><th>Move</th><th>Team</th><th class="n">Record</th>'
            '<th class="n">Rating (pts)</th><th class="n">Win vs avg</th><th class="n">Offense</th><th class="n">Defense</th>'
            '<th class="n">Luck</th></tr>' + "".join(rows) + "</table></div>")


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
    W, H, L, R, T, B = 640, 230, 64, 90, 14, 36
    n = len(labels)
    x = lambda i: L + ((W - L - R) / 2 if n == 1 else i / (n - 1) * (W - L - R))
    y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    out = ""
    for g in ticks:
        out += (f'<line x1="{L}" x2="{W - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="var(--grid)"/>'
                f'<text x="{L - 8}" y="{y(g) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{g:.0%}</text>')
    step = max(1, -(-n // 7))
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


def tabs(active, prefix=""):
    """Top tab bar shared by every page."""
    items = (("picks", "index.html", "Picks"), ("rankings", "rankings.html", "Rankings"),
             ("players", "players.html", "Players"), ("results", "results.html", "Results"),
             ("leaderboard", "leaderboard.html", "Leaderboard"))
    return '<div class="tabs">' + "".join(
        f'<a href="{prefix}{href}"{" class=on aria-current=page" if key == active else ""}>'
        f'<svg viewBox="0 0 24 24" aria-hidden="true">{TAB_ICONS[key]}</svg><span>{label}</span></a>'
        for key, href, label in items) + "</div>"


def page(title, hero, body, prefix="", color=None, active=None, extra="", bg=None, og=None):
    """Sub-page shell: colored header + content + team links."""
    style = f' style="--tc:{color}"' if color else ""
    if bg:
        style = f' style="background:{bg}"'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title>
<meta name="theme-color" content="#C85000">{head_extra(prefix, og)}<style>{CSS}</style></head><body>
<header class="hero{' thero' if color and not bg else ''}"{style}><div class="in">{tabs(active, prefix)}{hero}</div></header>
<main>{body}
<footer><b>Teams</b><div class="teamnav">{''.join(tl(t, prefix) for t in TEAMS)}</div><br><b>Not betting advice.</b>
Data from <a href="https://github.com/nflverse">nflverse</a>.<br><br>© 2026 Zachary Ivezi. All rights reserved.</footer></main>{extra}</body></html>"""


def all_picks_table(p):
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
    return body + ('<p class="note">* = from before the tracker started: what the model would have said (it was trained only '
                   'on earlier seasons).</p>' if not p.empty else '')


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
    return page(f"Every pick · {season}", hero, active="history", body= f"""<h2>Record</h2>{tracker_section(p, '') if not p.empty else ''}
<h2>All picks</h2><p class="note">* = from before the tracker started: what the model would have said (it was trained
only on earlier seasons).</p>{body}""")


def team_page(team, season, all_preds, ranks, odds, odds_hist, rating_hist, records, extras=None):
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
    x = (extras or {}).get(team, {})
    luck = x.get("luck", 0.0)
    body = f"""<div class="stats" style="margin-top:16px">
<div class="stat"><b>{points_rating(rk['rating']):+.1f}</b><span>Rating: points vs an average team</span></div>
<div class="stat"><b>{x.get('ats', '-')}</b><span>Against the spread</span></div>
<div class="stat"><b>{x.get('ou', '-')}</b><span>Over / under (overs first)</span></div>
<div class="stat"><b>{luck:+.1f}</b><span>Luck: wins vs expected ({x.get('exp_wins', 0):.1f} expected)</span></div></div>
<p class="note" style="margin-top:12px">Offense rank {int(rk['off_rank'])} · Defense rank {int(rk['def_rank'])} (EPA per play)</p>
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
    try:
        from data import load_games as _lg
        all_preds["stadium"] = all_preds["game_id"].map(_lg().set_index("game_id")["stadium"])
    except Exception:  # noqa: BLE001
        all_preds["stadium"] = None
    right, total, v_right, v_total = season_record(all_preds)

    upcoming = pick_week(df, season)
    week = int(upcoming["week"].min()) if len(upcoming) else int(all_preds["week"].max())
    this_week = all_preds[all_preds["week"] == week].sort_values(["gameday", "gametime", "game_id"])
    this_week["slot"] = [time_slot(w, t) for w, t in zip(this_week["weekday"], this_week["gametime"])]
    this_week["games_in_slot"] = this_week.groupby("slot")["game_id"].transform("count")
    best = best_picks(this_week)
    last_week = all_preds[all_preds["week"] == week - 1]
    lw_right = int(last_week["correct"].isin([True]).sum())
    lw_total = int(last_week["correct"].isin([True, False]).sum())

    print("Simulating the season 10,000 times ...")
    odds = simulate_season(df, saved, season)
    records = team_records(df, season)
    ranks = power_rankings(saved, df, SNAPSHOTS, season)
    stats = team_stats(df, season, ranks, odds)
    extras = season_extras(df, season)
    box = team_box(df, season)
    all_weeks = sorted(int(w) for w in all_preds[all_preds["game_type"] == "REG"]["week"].unique())
    for t in TEAMS:
        stats[t]["ats"] = extras[t]["ats"]
        stats[t]["ou"] = extras[t]["ou"]
        stats[t]["luck"] = extras[t]["luck"]
    try:  # player projections are extra: never let them break the site
        from players import evaluate as eval_players, player_features, weekly_projections
        proj = weekly_projections(season, week)
    except Exception as err:  # noqa: BLE001
        print(f"(player projections skipped: {err})")
        proj = None
    cards = []
    for slot in best["slot"]:
        cards.append(f"<h3>{e(slot)}</h3>")
        cards += [game_row(r) for r in this_week[this_week["slot"] == slot].itertuples()]

    # Pick tracker + odds history (saved in docs/data/ so they build up over the season)
    picks = picks_frame(update_picks(all_preds), season)
    odds_hist = update_odds(odds, season, week)
    b_right = int((picks["blend_correct"] == True).sum()) if "blend_correct" in picks else 0  # noqa: E712
    b_total = int(picks["blend_correct"].isin([True, False]).sum()) if "blend_correct" in picks else 0

    hero_stats = f"""<div class="stats">
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

    picks_url = load_site_config().get("picks_url", "")
    write_schedule(all_preds)
    footer = f"""<footer><b>Teams</b><div class="teamnav">{''.join(tl(t) for t in TEAMS)}</div><br>
Model: {saved['name']} using {used}. Data from <a href="https://github.com/nflverse">nflverse</a>.
Code: <a href="https://github.com/zacharyivezi08/nfl-game-predictor">github.com/zacharyivezi08/nfl-game-predictor</a>.
<br><br><b>Not betting advice.</b> In a 10-season backtest against real odds, betting the model's picks lost about 2–6% of the
money wagered. Vegas is more accurate than this model.<br><br>© 2026 Zachary Ivezi. All rights reserved.</footer>"""

    # ---------- Picks (home page): just this week ----------
    html_page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NFL Game Predictor</title>
<meta name="description" content="Machine learning picks, predicted scores, power rankings and playoff odds for every NFL game.">
<meta name="theme-color" content="#C85000">{head_extra()}<style>{CSS}</style></head><body>
<header class="hero"><div class="in">{tabs("picks")}
<div class="kicker">Week {week} · {season} season · updated {datetime.now():%b %-d, %Y}</div>
<h1>Week {week} picks</h1>
<p class="sub">Predicted scores and picks for every game. Tap a game for more details.</p>
{hero_stats}</div></header>
<main>
<nav><a href="#make">Make picks</a><a href="#glance">At a glance</a><a href="#slots">Top picks</a><a href="#upsets">Upset watch</a><a href="#games">All games</a></nav>
{week_picker(all_weeks, week)}
{recap_box(last_week, picks, week)}
<h2 id="make">Make your picks</h2>
{quick_picks(this_week, picks_url)}
{fav_picker()}
<h2 id="glance">At a glance</h2>
{glance_card(this_week)}
<h2 id="slots">Most confident pick of each time slot</h2>
<div class="slots">{''.join(slot_card(r) for _, r in best.iterrows())}</div>
<h2 id="upsets">Upset watch</h2>
<p class="note">Underdogs with at least a 35% chance. The bar is full at 50% (a coin flip).</p>
{upset_watch(this_week, saved.get("calibration"))}
<h2>All games</h2>
<p class="note">Tap a game for the full simulation and team comparison. The % is the <b>best estimate</b>
(the model blended with the Vegas line, the most accurate forecast here).</p>
<div id="games">{''.join(cards)}</div>
{footer}
</main>{my_picks_script(all_preds, picks_url, week - 1)}{FAV_JS}</body></html>"""

    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(html_page)
    (DOCS / "games").mkdir(exist_ok=True)
    (DOCS / "og").mkdir(exist_ok=True)
    for r in all_preds[all_preds["week"] <= week].itertuples():
        path = DOCS / "games" / f"{r.game_id}.html"
        if r.week == week or not path.exists():
            path.write_text(game_page(r, stats, box, saved, proj if r.week == week else None,
                                      saved.get("line_gaps"), all_preds, picks_url))
            try:
                og_image(r, DOCS / "og" / f"{r.game_id}.png")
            except Exception as err:  # noqa: BLE001
                print(f"(preview image skipped: {err})")
    (DOCS / "weeks").mkdir(exist_ok=True)
    for w in all_weeks:
        wg = all_preds[all_preds["week"] == w].sort_values(["gameday", "gametime", "game_id"])
        rows = "".join(game_row(r, "../", link=w <= week) for r in wg.itertuples())
        wrec = wg[wg["correct"].isin([True, False])]
        sub = (f"Model {int((wrec['correct'] == True).sum())}-{int((wrec['correct'] == False).sum())} this week · "  # noqa: E712
               if len(wrec) else "") + ("Tap a game for its page." if w <= week else "Predictions so far; game pages open the week of the game.")
        whero = f"<div class='kicker'>{season} season</div><h1>Week {w}</h1><p class='sub'>{sub}</p>"
        wbody = week_picker(all_weeks, week, "../") + f'<div id="games">{rows}</div>'
        (DOCS / "weeks" / f"week-{w}.html").write_text(page(f"Week {w} · {season}", whero, wbody, "../", active="picks",
                                                             extra=FAV_JS))
    (DOCS / "glossary.html").write_text(glossary_page())

    # ---------- Rankings: power rankings + playoff odds ----------
    rank_body = f"""<nav><a href="#rankings">Power rankings</a><a href="#playoffs">Playoff odds</a></nav>
<h2 id="rankings">Power rankings</h2>
<p class="note"><b>Rating (pts)</b> = points better or worse than an average team on a neutral field. <b>Win vs avg</b> = chance
to beat an average team. Offense/defense = efficiency rank (EPA per play). <b>Luck</b> = actual wins minus the wins a team's
points scored and allowed would normally produce (+ = winning more than its play suggests). Tap a team for its page.</p>
{rankings_table(ranks, extras)}
<h2 id="playoffs">Playoff odds</h2>
<p class="note">From 10,000 simulations of the rest of the season and the playoffs (teams tied on wins are ordered randomly).</p>
{playoff_tables(odds, records)}"""
    rank_hero = (f"<div class='kicker'>Week {week} · {season}</div><h1>Rankings &amp; playoff odds</h1>"
                 "<p class='sub'>Every team ranked by the model, plus each team's chances to make the playoffs and win it all.</p>")
    (DOCS / "rankings.html").write_text(page(f"Rankings · Week {week}", rank_hero, rank_body, active="rankings", extra=FAV_JS))

    # ---------- Results: accuracy, honesty check, tracker, last week, every pick ----------
    res_body = f"""<nav><a href="#accuracy">Accuracy</a><a href="#honest">Honest %</a><a href="#tracker">Tracker</a>
{'<a href="#lastweek">Last week</a>' if lw_total else ''}<a href="#all">Every pick</a></nav>
<h2 id="accuracy">{season} accuracy: model vs Vegas</h2>
<p class="note">Share of games picked correctly so far this season. The model was trained only on earlier seasons, so these are
real predictions.</p>
{accuracy_chart(all_preds, season)}
<h2 id="honest">Are the percentages honest?</h2>
<p class="note">Every game from {saved.get("calibration", {}).get("seasons", "past seasons")}, each predicted using only earlier
seasons. Dots on the dashed line = when it says 70%, the favorite really wins about 70% of the time.</p>
{calibration_chart(saved.get("calibration"))}
<h2 id="tracker">Pick tracker</h2>
<p class="note">Every pick is saved before kickoff and frozen once the game starts, so the record can't be rewritten.</p>
{tracker_section(picks)}
{f'<h2 id="lastweek">Week {week - 1} results</h2>' + results_table(last_week) if lw_total else ''}
<h2 id="all">Every pick</h2>
{all_picks_table(picks)}"""
    res_hero = (f"<div class='kicker'>{season} season</div><h1>Results</h1>"
                "<p class='sub'>How the picks have done, week by week, compared with Vegas.</p>")
    res_html = page(f"Results · {season}", res_hero, res_body, active="results")
    (DOCS / "results.html").write_text(res_html)
    (DOCS / "history.html").write_text(res_html)  # old link still works

    # ---------- Leaderboard: your name, your record, everyone's record ----------
    lb_hero = ("<div class='kicker'>Pick'em</div><h1>Leaderboard</h1>"
               "<p class='sub'>Make picks on the Picks page. See how you stack up against friends, the model and Vegas.</p>")
    (DOCS / "leaderboard.html").write_text(page("Leaderboard", lb_hero, my_picks_section(picks_url, standalone=True),
                                                active="leaderboard", extra=my_picks_script(all_preds, picks_url, week - 1)))

    ev = None
    try:
        from data import load_games, load_player_stats
        g_all = load_games()
        f_all, _, _ = player_features(load_player_stats(g_all["season"].unique()), g_all)
        done_seasons = sorted(df[df["played"] & (df["game_type"] == "SB")]["season"].unique())
        ev = eval_players(f_all, done_seasons[-4:])
    except Exception as err:  # noqa: BLE001
        print(f"(player accuracy table skipped: {err})")
    unplayed = this_week[~this_week["played"]] if len(this_week) else this_week
    (DOCS / "players.html").write_text(players_page(proj, ev, season, week, unplayed))
    (DOCS / "teams").mkdir(exist_ok=True)
    rating_hist = rating_history(saved, season, week)
    for t in TEAMS:
        (DOCS / "teams" / f"{t}.html").write_text(
            team_page(t, season, all_preds, ranks, odds, odds_hist, rating_hist, records, extras))
    print(f"Built docs/index.html, history.html and {len(TEAMS)} team pages for {season} week {week}")


if __name__ == "__main__":
    main()
