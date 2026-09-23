# NFL Game Predictor

A machine learning model that predicts every NFL game: win probability, predicted score, spread and total, plus weekly power rankings and playoff odds from 10,000 season simulations. It explains each pick, compares itself honestly against Vegas, and updates its own website automatically.

Built with Python, pandas and scikit-learn on free data from [nflverse](https://github.com/nflverse).

**Live site:** https://zacharyivezi08.github.io/nfl-game-predictor/

## What's on the site

- **Weekly picks**: win probability, predicted score, spread and total for every game, compared with Vegas, plus the top 3 reasons for each pick
- **Most confident pick of each time slot** (Thursday night, Sunday 1 PM, 4 PM, Sunday night, Monday night)
- **Season accuracy chart**: the model vs the Vegas favorite, week by week
- **Power rankings**: all 32 teams ranked by the model, with weekly movement and offense/defense ranks
- **Playoff odds**: playoffs, division, #1 seed and Super Bowl chances from 10,000 simulations
- **Last week's results**: every pick marked right or wrong

It updates itself every Wednesday at 7 PM and Sunday at 12:30, 3:30 and 7:30 PM Eastern (GitHub Actions).

## How it works

**Data** (all free from nflverse): every game since 1999 with scores, rest, weather, Vegas lines and starting QBs; every play since 1999 (about 1.2 million runs and passes); QB stats; official injury reports (2009+); and snap counts (2012+).

**Features**: for each game, the model only sees information from *before* kickoff.

| Group | What it measures | Kept? |
|---|---|---|
| Base | Elo team rating, last-5-games form, rest, home field, division game | ✅ |
| QB | Starting QB's EPA per dropback over his recent games | ✅ |
| Efficiency (EPA) | Offense and defense EPA per play and success rate, from play-by-play (garbage time removed) | ✅ |
| Injuries | Snap share of injured starters (Out/Doubtful/Questionable), weighted by their recent role | ✅ |
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
| + QB | 0.6296 |
| + Efficiency (EPA) | 0.6278 |
| + Injuries | 0.6273 |
| + Weather | 0.6286 (worse, skipped) |
| + Travel | 0.6332 (worse, skipped) |

**Model vs Vegas**

| | Accuracy | Log loss | Brier |
|---|---|---|---|
| **Our model** (logistic regression) | **64.9%** | 0.625 | 0.218 |
| Basic model (Elo + form only) | 63.2% | 0.635 | 0.223 |
| Vegas | 67.6% | 0.607 | 0.210 |
| Always pick home team | 55.4% | 0.688 | 0.247 |

Vegas is still better. It has real-time injury news, depth charts and millions of dollars of bets moving the line. The model agrees with the Vegas favorite in 86% of games.

**Predicted scores**

| | Our model | Vegas |
|---|---|---|
| Average miss on the margin | 9.9 pts | 9.5 pts |
| Average miss on total points | 10.4 pts | 10.2 pts |

### Things that didn't work (and why that matters)

- **Tuning the settings overfit.** Grid-searching Elo, EPA and QB settings (about 200 combinations) improved the validation score but made the untouched test *worse*. Re-tuning across 12 rolling seasons gained only about 0.001, which is noise, so the standard settings were kept.
- **Travel hurt because home-field advantage has shrunk.** Home teams won 57.6% of games in 1999–2009, 50.4% in 2020 (no fans), and 54.5% in 2021–2026. Travel features mostly acted like extra home-field advantage, which older seasons overstated.
- **Splitting injuries into offense and defense** added noise. A single "total missing talent" number worked.
- **Choosing the model type on the test seasons** was accidental peeking. It's now chosen on validation. Gradient boosting and random forest looked slightly better on the test, but logistic regression won on validation, so that's what's used.

## Could you bet on it?

`python src/backtest.py` checks this honestly: for each season since 2016, the model is trained only on earlier seasons, then bets $1 per game at the real closing moneyline odds.

| Strategy | Bets | Win % | Return per $1 |
|---|---|---|---|
| Bet every model pick | 2,782 | 64.7% | -3.0% |
| Only picks the model is >75% sure of | 533 | 79.2% | -2.7% |
| Model edge over Vegas > 5% | 1,398 | 43.3% | -3.4% |
| Model edge over Vegas > 10% | 560 | 41.1% | -3.9% |

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
python src/build_site.py         # builds the website in docs/
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
src/build_site.py  builds the website (docs/index.html)
.github/workflows  auto-updates the site Wed 7 PM + Sun 12:30/3:30/7:30 PM ET
```

Project idea from @ethandojo's NFL Project Ideas list.
