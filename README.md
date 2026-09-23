# NFL Game Predictor

A machine learning model that predicts the probability of each team winning an NFL game, explains which factors drove each pick, and compares itself against Vegas. It also builds a weekly picks web page.

Built with Python, pandas and scikit-learn on free data from [nflverse](https://github.com/nflverse).

**Live picks page:** https://zacharyivezi08.github.io/nfl-game-predictor/

## Example output

```
2026 Week 3 predictions (Logistic Regression)

ATL @ GB   ->  GB 70%   | Vegas: GB 70%
      why: Team strength (Elo) -> GB; Starting QB -> GB; Recent defense -> ATL
CAR @ CLE  ->  CLE 51%   | Vegas: CAR 57%  << DISAGREES
      why: Starting QB -> CAR; Team strength (Elo) -> CAR; Recent defense -> CLE
NYJ @ DET  ->  DET 82%   | Vegas: DET 72%
      why: Team strength (Elo) -> DET; Starting QB -> DET; Rest advantage -> DET
```

## How it works

1. **Data**: every NFL game since 1999 (scores, rest, weather, Vegas lines, starting QBs) plus every QB's passing stats, all from nflverse.
2. **Features**: for each game, the model only sees information from *before* kickoff:
   - **Elo rating difference**: a running team strength score that goes up after wins and down after losses
   - **Recent form** over the last 5 games: point differential, points scored, points allowed, win %
   - **Starting QB**: each starter's EPA per dropback over his last 16 starts. New or backup QBs start slightly below average until they build up a sample, so a QB injury changes the pick.
   - **Weather**: wind, cold (degrees below 45°F), and whether a dome team is playing outside in the cold. Future games use the stadium's typical temperature for that month.
   - **Rest advantage**, **home field** (0 for neutral sites like London) and **division game**
3. **Feature testing**: `train.py` checks which feature groups actually improve predictions on games the model never saw.
4. **Models**: logistic regression, random forest and gradient boosting are trained on 2002–2021 and tested on 2022–2025.
5. **Vegas comparison**: moneylines are converted to win probabilities (with the bookmaker's cut removed) and scored the same way as the model.

## Results (test seasons 2022–2025, 1,136 games)

**Which features help?**

| Features | Accuracy | Log loss |
|---|---|---|
| Base (Elo, form, rest, home) | 63.5% | 0.635 |
| Base + weather | 63.7% | 0.635 |
| Base + QB | 64.3% | 0.628 |
| Base + QB + weather | 64.1% | 0.628 |

The QB feature gives the biggest boost. Weather barely matters for picking winners (it matters more for how many points get scored).

**Model vs Vegas**

| | Accuracy | Log loss | Brier |
|---|---|---|---|
| Our model | 64.1% | 0.628 | 0.219 |
| Vegas | 67.6% | 0.607 | 0.210 |
| Always pick home team | 55.4% | 0.688 | 0.247 |

Vegas is still better. It uses injury reports, depth charts, and millions of dollars of bets. The model agrees with the Vegas favorite in about 85% of games, and those disagreements are the interesting ones to watch.

Log loss and Brier score measure how good the *probabilities* are, not just the picks (lower is better).

## Run it yourself

```bash
pip install -r requirements.txt
python src/train.py              # downloads data, tests features, trains, compares to Vegas
python src/predict.py            # next week's picks + Vegas line + season record
python src/predict.py --week 2   # any week (shows if picks were right)
python src/predict.py --team PHI # one team only
python src/build_site.py         # builds the picks web page in docs/
```

Each week, run with `--refresh` to pull the newest results:

```bash
python src/train.py --refresh && python src/build_site.py
git add docs && git commit -m "Week N picks" && git push
```

## Project structure

```
src/data.py      download + load games and QB stats
src/features.py  build pre-game features (Elo, form, QB, weather) + Vegas probabilities
src/train.py     test features, compare models and Vegas, save the best
src/predict.py   predict games and explain the picks
src/build_site.py  build the weekly picks page (docs/index.html)
```

Project idea from @ethandojo's NFL Project Ideas list. For fun, not betting advice.
