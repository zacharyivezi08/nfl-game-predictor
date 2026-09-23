"""Download and load free NFL data from nflverse (updated weekly during the season)."""
from pathlib import Path
import urllib.request

import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
QB_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GAMES_PATH = DATA_DIR / "games.csv"
QB_DIR = DATA_DIR / "qb"


def _download(url: str, path: Path):
    DATA_DIR.mkdir(exist_ok=True)
    path.parent.mkdir(exist_ok=True)
    print(f"Downloading {url.split('/')[-1]} ...")
    urllib.request.urlretrieve(url, path)


def load_games(force_download: bool = False) -> pd.DataFrame:
    """Every NFL game since 1999: teams, scores, rest, weather, Vegas lines, starting QBs."""
    if force_download or not GAMES_PATH.exists():
        _download(GAMES_URL, GAMES_PATH)
    games = pd.read_csv(GAMES_PATH, parse_dates=["gameday"])
    return games.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def load_qb_stats(seasons, force_download: bool = False) -> pd.DataFrame:
    """Passing EPA and dropbacks for every QB in every game.

    Each season is downloaded once and trimmed to just the columns we need.
    With force_download, only the latest season is re-downloaded (older ones don't change).
    """
    seasons = sorted(set(int(s) for s in seasons))
    frames = []
    for season in seasons:
        path = QB_DIR / f"qb_{season}.csv"
        refresh = force_download and season == seasons[-1]
        if refresh or not path.exists():
            raw_path = QB_DIR / f"raw_{season}.csv"
            try:
                _download(QB_URL.format(season=season), raw_path)
            except Exception as e:  # season not published yet, etc.
                print(f"  (no player stats for {season}: {e})")
                continue
            raw = pd.read_csv(raw_path, low_memory=False)
            cols = ["player_id", "player_display_name", "game_id", "team", "attempts", "sacks_suffered", "passing_epa"]
            qbs = raw.loc[raw["attempts"].fillna(0) > 0, cols].copy()
            qbs["dropbacks"] = qbs["attempts"] + qbs["sacks_suffered"].fillna(0)
            qbs[["player_id", "player_display_name", "game_id", "team", "dropbacks", "passing_epa"]].to_csv(path, index=False)
            raw_path.unlink()
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
