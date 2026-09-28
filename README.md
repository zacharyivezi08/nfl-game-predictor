# NFL Game Predictor

A machine learning model that predicts every NFL game: win probability, predicted score, spread and total, plus weekly power rankings and playoff odds from 10,000 season simulations. It explains each pick, compares itself honestly against Vegas, blends itself with Vegas for the most accurate forecast, keeps a permanent record of every pick, and updates its own website automatically.

Built with Python, pandas and scikit-learn on free data from [nflverse](https://github.com/nflverse).

**Live site:** https://zacharyivezi08.github.io/nfl-game-predictor/

## What's on the site

- **Weekly picks**: win probability, predicted score, spread and total for every game, compared with Vegas, plus the top 3 reasons for each pick
- **Model + Vegas blend**: the most accurate forecast on the site (see below)
- **Week at a glance**: the safest picks, best upset chances and spread/total flags in one box
- **Make your own picks**: tap a team on any game before kickoff; your record vs the model and Vegas is kept in your browser
- **Matchup edges**: in each game's stat comparison, plain-English pass/run mismatches (e.g. "BUF passing offense #2 vs MIA pass defense #28: Big edge BUF")
- **Upset watch**: every underdog with a 35%+ chance, with how often underdogs at that level have actually won
- **Spread and total flags**: highlighted when the predicted score is 4+ points away from the Vegas spread or total, with the historical hit rate
- **Most confident pick of each time slot** (Thursday night, Sunday 1 PM, 4 PM, Sunday night, Monday night)
- **Season accuracy chart**: the model vs the Vegas favorite, week by week
- **Power rankings**: all 32 teams ranked by the model, with weekly movement and offense/defense ranks
- **Playoff odds**: playoffs, division, #1 seed and Super Bowl chances from 10,000 simulations
- **Last week's results**: every pick marked right or wrong
- **Pick tracker**: record by week for the model, Vegas and the blend, biggest upsets called, worst misses, and a page with [every pick](https://zacharyivezi08.github.io/nfl-game-predictor/history.html). Picks are saved before kickoff and frozen once the game starts, so the record can't be rewritten.
- **Team pages**: one per team (e.g. [KC](https://zacharyivezi08.github.io/nfl-game-predictor/teams/KC.html)) with the schedule and picks, power rating by week and playoff odds by week

It updates itself every 30 minutes (GitHub Actions) and only commits when something changed.

## How it works

**Data** (all free from nflverse): every game since 1999 with scores, rest, weather, Vegas lines and starting QBs; every play since 1999 (about 1.2 million runs and passes); QB stats; official injury reports (2009+); and snap counts (2012+).

**Features**: for each game, the model only sees information from *before* kickoff.

| Group | What it measures | Kept? |
|---|---|---|
| Base | Elo team rating, last-5-games form, rest, home field, division game | ✅ |
| QB | Starting QB's EPA per dropback, plus a **backup-QB drop-off**: how much worse this week's QB is than the team's usual starter (catches injuries and resting starters) | ✅ |
| Efficiency (EPA) | Offense and defense EPA per play and success rate, from play-by-play (garbage time removed) | ✅ |
| Injuries | Snap share of injured starters (Out/Doubtful/Questionable), weighted by their recent role | ✅ |
| Stakes | Late season only: has a team **locked its playoff seed** (may rest starters) or been **eliminated**? | ✅ |
| Weather | Wind, cold, dome team playing outside in the cold | ❌ didn't help |
| Travel | Time zones crossed, body-clock kickoffs, byes, short weeks | ❌ didn't help |

**Choosing features and models honestly.** Seasons are split three ways so the final score can't be gamed:

- **2002–2017**: training
- **2018–2021**: validation, used to decide which feature groups to keep (greedy forward selection) and which model type wins
- **2022–2025**: the test, looked at only once at the very end

## Results (test seasons 2022–2025, 1,136 games)

**Which features help?** Each group was added only if it improved the validation score:

| Features | Validation log loss (lower = better) |
|---|---|
| Base only | 0.6348 |
| + QB | 0.6297 |
| + Efficiency (EPA) | 0.6278 |
| + Stakes | 0.6272 |
| + Injuries | 0.6269 |
| + Weather | 0.6281 (worse, skipped) |
| + Travel | 0.6322 (worse, skipped) |

**Model vs Vegas**

| | Accuracy | Log loss | Brier |
|---|---|---|---|
| **Our model** (logistic regression) | **64.3%** | 0.622 | 0.217 |
| Basic model (Elo + form only) | 63.2% | 0.635 | 0.223 |
| Vegas | 67.6% | 0.607 | 0.210 |
| **Model + Vegas blend** | **67.7%** | 0.607 | 0.210 |
| Always pick home team | 55.4% | 0.688 | 0.247 |

Vegas is still better. It has real-time injury news, depth charts and millions of dollars of bets moving the line. The model agrees with the Vegas favorite in 86% of games.

### The Model + Vegas blend (the most accurate forecast)

Two forecasts combined can beat either one alone, so `train.py` learns how much to trust each. It uses out-of-sample
predictions from 2008–2021 (each season predicted by a model trained only on earlier seasons) and is then scored on the
untouched 2022–2025 test. The blend gives Vegas about 8 times the weight of the model and ties Vegas (67.7% vs 67.6%,
same log loss). Ten other ways of combining them were tried (Vegas + individual features, all features + Vegas, recent seasons only),
and none beat Vegas by more than noise. The closing line already contains nearly everything this model knows. The blend is
still the best number on the site, and when there's no Vegas line yet (games more than a week out) it falls back to the model.

### Are the percentages honest? (calibration)

Predicting every game from 2012 on (each season using only earlier seasons), the Model + Vegas blend's percentages match
reality: favorites it gave 55–60% won 57.6%, 75–80% won 77.9%, and underdogs it gave 35–40% won 40.0%. The site shows
this as a chart. That's what makes the **upset watch** meaningful: a 40% underdog really does win about 4 times in 10.

Does the model picking the underdog outright add anything? Not really. Underdogs the model picked won 43.6% of the time,
almost exactly the 44.1% the blend predicted. So the upset watch ranks games by the blend's chance and only notes when the
model also picks the upset.

### Spread and total flags

Overall the model's spread and over/under picks are coin flips (about 50%). But the bigger its disagreement with Vegas,
the better it has done (walk-forward, 2012–2025):

| Model's number vs the Vegas line | Spread side right | Over/under right |
|---|---|---|
| Any gap | 50.5% | 50.1% |
| 2+ points | 51.2% | 51.9% |
| 4+ points (flagged on the site) | 53.9% (562 games) | 54.3% (599 games) |
| 6+ points | 56.7% (141 games) | 57.4% (129 games) |

Break-even after the sportsbook's cut is 52.4%. The 4+ point results are above it, but with ~600 games that's still within
the range luck could produce, so the flags are "worth a look," not proven. **Not betting advice.**

**Predicted scores**

| | Our model | Vegas |
|---|---|---|
| Average miss on the margin | 9.9 pts | 9.5 pts |
| Average miss on total points | 10.4 pts | 10.2 pts |

### Error analysis: where the model loses to Vegas

Predicting 2016–2025 season by season (each season using only earlier data), the model and Vegas pick the same
winner in 87% of games, and in those games they are **exactly equally accurate (68.1%)**. The whole gap comes from the
games where the model disagrees with Vegas. The biggest cause was **backup quarterbacks and resting starters**: games
with a QB change made up 46% of the model's biggest misses. For example, it gave the 2020 Chiefs an 80% chance in
Week 17, not knowing Mahomes was resting (they lost 38-21).

The **backup-QB drop-off** feature was built to fix this. In 317 games where a clearly worse QB started (2010–2025):

| | Accuracy | Log loss |
|---|---|---|
| Before | 64.7% | 0.643 |
| After | 64.7% | 0.632 |
| Vegas | 65.9% | 0.617 |

In those games the model's probabilities got much closer to Vegas (the log-loss gap shrank by about 40%), though it
picks the same winners. Overall, headline accuracy moved slightly *down*, from 64.9% to 63.9% on the test seasons.
That's about 11 games out of 1,136, which is within luck. Log loss, the more reliable measure, improved from 0.625 to
0.624, so the feature was kept.

The **stakes** feature was built for the other half of that problem: the last week of the season, when teams that have
locked their seed rest starters (the 2009 Colts, 2020 Chiefs, 2023 Ravens and 2024 Chiefs are all flagged). Uses only
wins, not full tiebreakers, so it's cautious. Predicting 2016–2025 season by season:

| Final regular-season week (160 games) | Accuracy | Log loss |
|---|---|---|
| Before | 70.0% | 0.599 |
| After | 71.9% | 0.579 |
| Vegas | 71.9% | |

It now ties Vegas in the final week. The 7 games with a "locked" team went from 4-3 to 7-0, but that's a tiny sample.

### Things that didn't work (and why that matters)

- **Tuning the settings overfit.** Grid-searching Elo, EPA and QB settings (about 200 combinations) improved the validation score but made the untouched test *worse*. Re-tuning across 12 rolling seasons gained only about 0.001, which is noise, so the standard settings were kept.
- **Travel hurt because home-field advantage has shrunk.** Home teams won 57.6% of games in 1999–2009, 50.4% in 2020 (no fans), and 54.5% in 2021–2026. Travel features mostly acted like extra home-field advantage, which older seasons overstated.
- **Early-season fixes didn't help.** The model trails Vegas most in weeks 3–6. Tested: a new-head-coach flag, letting the
  model trust ratings less before teams have played 6 games, and last season's point differential. None improved both
  validation and test, so none were kept. Early in the season the blend is the fix, since Vegas knows about offseason moves.
- **Pass/run matchups passed validation but failed the test.** Splitting efficiency into passing and running (each offense
  vs the other team's matching defense) improved validation slightly (0.6265 vs 0.6269) but made the untouched test worse
  (0.6223 vs 0.6218), the same overfitting pattern as tuning, so the model doesn't use them. The site still shows them as matchup edges.
- **Actual kickoff weather would help totals a little.** For past games, knowing the real wind and temperature instead of the
  stadium's typical weather cut the average total miss from 10.39 to 10.31 points (10.54 to 10.18 in games with 15+ mph wind).
  Vegas is still better (9.31 in windy games). Not added yet: it would need a live forecast for upcoming games.
- **Line movement couldn't be tested**: the free data has one betting line per game, not the opening line.
- **QB accuracy (CPOE) added nothing.** Completion % over expected, from nflfastR's completion probability model, was tested as its own feature group. Validation got slightly worse (0.6271 vs 0.6269) and the test was unchanged (64.3%), because it overlaps heavily with the QB EPA rating the model already uses (correlation 0.68): accurate QBs already show up as efficient ones. It was left out so the site updates stay fast.
- **Splitting injuries into offense and defense** added noise. A single "total missing talent" number worked.
- **Choosing the model type on the test seasons** was accidental peeking. It's now chosen on validation. Gradient boosting and random forest looked slightly better on the test, but logistic regression won on validation, so that's what's used.

## Could you bet on it?

`python src/backtest.py` checks this honestly: for each season since 2016, the model is trained only on earlier seasons, then bets $1 per game at the real closing moneyline odds.

| Strategy | Bets | Win % | Return per $1 |
|---|---|---|---|
| Bet every model pick | 2,782 | 64.4% | -3.6% |
| Only picks the model is >75% sure of | 555 | 79.1% | -3.0% |
| Model edge over Vegas > 5% | 1,377 | 43.5% | -6.1% |
| Model edge over Vegas > 10% | 557 | 43.3% | -0.4% |

Every strategy lost money. Against the spread the model hits 48.6%, and on over/unders 50.3%. You need 52.4% just to break even after the sportsbook's cut. Beating the market takes information it doesn't have yet, which a model built on public data can't provide. **Not betting advice.**

## Run it yourself

```bash
pip install -r requirements.txt
python src/train.py              # downloads data (first run ~2 min), picks features, trains, compares to Vegas
python src/predict.py            # this week's picks, scores, Vegas lines, most confident per time slot
python src/predict.py --week 2   # any week (shows if picks were right)
python src/predict.py --team PHI # one team only
python src/ratings.py            # power rankings
python src/simulate.py           # playoff odds (10,000 simulations)
python src/backtest.py           # would betting the picks have made money?
python src/build_site.py         # builds the website: docs/index.html, history.html, teams/*.html
python src/history.py            # pick tracker: record by week + biggest upsets called
```

## Project structure

```
src/data.py        download + cache nflverse data (games, play-by-play, QB stats, injuries, snaps)
src/features.py    pre-game features: Elo, form, QB, EPA, injuries, weather, travel
src/injuries.py    injured players' snap share -> "missing talent" per team per week
src/train.py       choose features + model on validation, test vs Vegas, save the model
src/scores.py      spread and total-points models
src/predict.py     picks, explanations, predicted scores, most confident pick per time slot
src/ratings.py     any-matchup predictions + power rankings
src/simulate.py    10,000-season playoff simulator
src/backtest.py    betting backtest against real odds
src/history.py     pick tracker: saves every pick before kickoff (docs/data/picks.json) + weekly playoff odds
src/build_site.py  builds the website (docs/index.html, history.html, teams/*.html)
.github/workflows  auto-updates the site every 30 minutes
```

Project idea from @ethandojo's NFL Project Ideas list.

---

© 2026 Zachary Ivezi. All rights reserved.
