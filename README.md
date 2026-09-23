NFL Game Predictor
A machine learning model that predicts the probability of each team winning an NFL game, and explains which factors drove each pick.

Built with Python, pandas and scikit-learn on free data from nflverse.

Example output
2026 Week 3 predictions (Gradient Boosting)

LAC @ BUF  ->  BUF 87%
      why: Team strength (Elo) -> BUF; Recent offense -> BUF; Recent point differential -> BUF
 KC @ MIA  ->  KC 61%
      why: Team strength (Elo) -> KC; Recent point differential -> KC; Recent offense -> KC
BAL @ DAL  ->  BAL 63%
      why: Neutral site, no home edge -> BAL; Team strength (Elo) -> BAL; Recent offense -> BAL
How it works
Data: every NFL game since 1999 (scores, dates, rest days, location) from nflverse.
Features: for each game, the model only sees information from before kickoff:
Elo rating difference: a running team strength score that goes up after wins and down after losses (bigger wins move it more; ratings regress toward average each offseason)
Recent form over the last 5 games: point differential, points scored, points allowed, win %
Rest advantage: days of rest for home team minus away team
Home field (0 for neutral-site games like London) and division game
Models: logistic regression, random forest and gradient boosting are trained on 2002–2021 and tested on 2022–2025 (games the model never saw).
Explanations: the logistic regression weights show which factors pushed each prediction toward one team.
Results (test seasons 2022–2025, 1,136 games)
Model	Accuracy	Log loss	Brier
Gradient Boosting	62.8%	0.634	0.222
Logistic Regression	63.5%	0.635	0.223
Random Forest	62.1%	0.637	0.224
Always pick home team	55.4%		
Log loss and Brier score measure how good the probabilities are, not just the picks, which is why they're used to choose the best model. For reference, Vegas favorites typically win around 65–67% of games.

Run it yourself
pip install -r requirements.txt
python src/train.py              # downloads data, trains, prints results
python src/predict.py            # predicts the next week's games
python src/predict.py --week 2   # any week (shows if picks were right)
python src/predict.py --team PHI # one team only
python src/train.py --refresh    # pull the newest results each week
Ideas to improve it
Add quarterback-level stats (EPA per play) from nflverse play-by-play data
Add weather (temperature, wind) for outdoor games
Compare predictions against Vegas spreads/moneylines
Build a simple web page to show weekly picks
Project structure
src/data.py      download + load game data
src/features.py  build pre-game features (Elo, recent form, etc.)
src/train.py     train/compare models, save the best
src/predict.py   predict games and explain the picks
Project idea from @ethandojo's NFL Project Ideas list.
