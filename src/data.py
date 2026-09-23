"""Download and load free NFL data from nflverse (updated weekly during the season).

Big files (play-by-play, player stats) are downloaded once per season and shrunk
down to just what the model needs, then cached in data/. With --refresh, only the
current season is re-downloaded, since older seasons never change.
"""
from pathlib import Path
import urllib.request

import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
QB_URL = RELEASES + "/stats_player/stats_player_week_{season}.csv"
PBP_URL = RELEASES + "/pbp/play_by_play_{season}.csv.gz"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GAMES_PATH = DATA_DIR / "games.csv"


def _download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url.split('/')[-1]} ...")
    urllib.request.urlretrieve(url, path)


def _per_season(kind, seasons, url, shrink, force_download=False, first_season=1999):
    """Download + shrink one file per season, caching the small version in data/<kind>/."""
    seasons = sorted(set(int(s) for s in seasons if int(s) >= first_season))
    folder = DATA_DIR / kind
    frames = []
    for season in seasons:
        path = folder / f"{kind}_{season}.csv"
        if (force_download and season == seasons[-1]) or not path.exists():
            raw_path = folder / f"raw_{season}{Path(url).suffix}"
            try:
                _download(url.format(season=season), raw_path)
                shrink(raw_path).to_csv(path, index=False)
            except Exception as e:  # season not published yet, etc.
                print(f"  (no {kind} data for {season}: {e})")
                continue
            finally:
                raw_path.unlink(missing_ok=True)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_games(force_download: bool = False) -> pd.DataFrame:
    """Every NFL game since 1999: teams, scores, rest, weather, Vegas lines, starting QBs."""
    if force_download or not GAMES_PATH.exists():
        _download(GAMES_URL, GAMES_PATH)
    games = pd.read_csv(GAMES_PATH, parse_dates=["gameday"])
    return games.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _shrink_qb(raw_path):
    raw = pd.read_csv(raw_path, low_memory=False)
    cols = ["player_id", "player_display_name", "game_id", "team", "attempts", "sacks_suffered", "passing_epa"]
    qbs = raw.loc[raw["attempts"].fillna(0) > 0, cols].copy()
    qbs["dropbacks"] = qbs["attempts"] + qbs["sacks_suffered"].fillna(0)
    return qbs


def load_qb_stats(seasons, force_download: bool = False) -> pd.DataFrame:
    """Passing EPA and dropbacks for every QB in every game."""
    return _per_season("qb", seasons, QB_URL, _shrink_qb, force_download)


def _shrink_pbp(raw_path):
    """Turn ~45,000 plays per season into offense + defense efficiency per team per game.

    Only real runs and passes count. Kneel-downs, spikes and "garbage time"
    (when one team's win chance is below 5% or above 95%) are left out, since
    those plays say little about how good a team is.
    """
    cols = ["game_id", "posteam", "defteam", "play_type", "epa", "success", "wp", "qb_kneel", "qb_spike"]
    p = pd.read_csv(raw_path, usecols=cols, low_memory=False)
    p = p[p["play_type"].isin(["pass", "run"]) & p["epa"].notna() & p["posteam"].notna()]
    p = p[(p["qb_kneel"].fillna(0) == 0) & (p["qb_spike"].fillna(0) == 0)]
    p = p[p["wp"].between(0.05, 0.95)]
    off = p.groupby(["game_id", "posteam"]).agg(off_plays=("epa", "size"), off_epa=("epa", "sum"),
                                                off_success=("success", "sum"))
    off.index = off.index.set_names(["game_id", "team"])
    dfn = p.groupby(["game_id", "defteam"]).agg(def_plays=("epa", "size"), def_epa=("epa", "sum"),
                                                def_success=("success", "sum"))
    dfn.index = dfn.index.set_names(["game_id", "team"])
    return off.join(dfn, how="outer").reset_index()


def load_team_epa(seasons, force_download: bool = False) -> pd.DataFrame:
    """Offense and defense EPA + success rate totals for every team in every game."""
    return _per_season("epa", seasons, PBP_URL, _shrink_pbp, force_download)


INJ_URL = RELEASES + "/injuries/injuries_{season}.csv"
SNAP_URL = RELEASES + "/snap_counts/snap_counts_{season}.csv"


def _shrink_injuries(raw_path):
    i = pd.read_csv(raw_path, low_memory=False)
    i = i[i["report_status"].isin(["Out", "Doubtful", "Questionable"])]
    return i[["season", "week", "team", "full_name", "position", "report_status"]]


def load_injuries(seasons, force_download: bool = False) -> pd.DataFrame:
    """Official injury reports: who was listed Out / Doubtful / Questionable each week (2009+)."""
    return _per_season("injuries", seasons, INJ_URL, _shrink_injuries, force_download, first_season=2009)


def _shrink_snaps(raw_path):
    s = pd.read_csv(raw_path, low_memory=False)
    s["side"] = (s["defense_pct"].fillna(0) > s["offense_pct"].fillna(0)).map({True: "def", False: "off"})
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    return s[["game_id", "season", "week", "team", "player", "position", "side", "share"]]


def load_snaps(seasons, force_download: bool = False) -> pd.DataFrame:
    """Share of offensive/defensive snaps each player played in each game (2012+)."""
    return _per_season("snaps", seasons, SNAP_URL, _shrink_snaps, force_download, first_season=2012)
