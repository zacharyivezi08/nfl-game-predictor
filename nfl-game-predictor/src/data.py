"""Download and load NFL game results from nflverse (free, updated weekly)."""
from pathlib import Path
import urllib.request

import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GAMES_PATH = DATA_DIR / "games.csv"


def download_games(force: bool = False) -> Path:
    """Save the latest games.csv into data/. Re-download with force=True for new weeks."""
    DATA_DIR.mkdir(exist_ok=True)
    if force or not GAMES_PATH.exists():
        print(f"Downloading {GAMES_URL} ...")
        urllib.request.urlretrieve(GAMES_URL, GAMES_PATH)
    return GAMES_PATH


def load_games(force_download: bool = False) -> pd.DataFrame:
    path = download_games(force=force_download)
    games = pd.read_csv(path, parse_dates=["gameday"])
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return games
