import os
import sys
import traceback
import pandas as pd
import pybaseball
pybaseball.cache.enable()
import statsapi
from pybaseball import statcast

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")

def fetch_statcast_for_date(date: str) -> pd.DataFrame:
    out = os.path.join(RAW_DIR, f"statcast_{date}.parquet")
    if os.path.exists(out):
        print(f"Loading cached Statcast data for {date}")
        return pd.read_parquet(out)
    print(f"Fetching Statcast data for {date} from March 1st of the season...")
    year = date.split("-")[0]
    start_dt = f"{year}-03-01"
    df = statcast(start_dt=start_dt, end_dt=date)
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved Statcast data to {out}")
    return df

def fetch_lineups_for_date(date: str) -> pd.DataFrame:
    out = os.path.join(RAW_DIR, f"lineups_{date}.parquet")
    if os.path.exists(out):
        print(f"Loading cached lineup data for {date}")
        return pd.read_parquet(out)
    print(f"Fetching lineup data for {date}...")
    records = []
    sched = statsapi.get(
        "schedule",
        {"sportId": 1, "startDate": date, "endDate": date, "hydrate": "probablePitcher"}
    )
    for day in sched.get("dates", []):
        for game in day.get("games", []):
            if not isinstance(game, dict):
                continue
            gid = game.get("gamePk")
            away = game.get("teams", {}).get("away", {}).get("team", {}).get("name")
            home = game.get("teams", {}).get("home", {}).get("team", {}).get("name")
            pp = game.get("probablePitcher", {}) or {}
            away_prob = pp.get("awayProbablePitcher") or {}
            home_prob = pp.get("homeProbablePitcher") or {}
            try:
                box = statsapi.boxscore_data(gid)
            except Exception as e:
                print(f"ERROR: Failed to fetch boxscore for game {gid} on {date}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                box = {}
            for side, team_name, prob in (("away", away, away_prob), ("home", home, home_prob)):
                prob_name = prob.get("fullName")
                prob_id = prob.get("id")
                side_data = box.get(side, {}) or {}
                bat_ids = side_data.get("battingOrder", []) or []
                players = side_data.get("players", {}) or {}
                pitchers = side_data.get("pitchers", []) or []
                starter = pitchers[0] if pitchers else None
                starter_name = players.get(f"ID{starter}", {}).get("person", {}).get("fullName") if starter else None
                is_confirmed = starter == prob_id if starter and prob_id else False
                if bat_ids:
                    for idx, bid in enumerate(bat_ids, 1):
                        b = players.get(f"ID{bid}", {}) or {}
                        records.append({
                            "game_id": gid,
                            "team": team_name,
                            "side": side,
                            "batting_order": idx,
                            "batter_id": bid,
                            "batter_name": b.get("person", {}).get("fullName"),
                            "position": b.get("position", {}).get("abbreviation"),
                            "pitcher_id": starter,
                            "pitcher_name": starter_name,
                            "probable_pitcher": prob_name,
                            "probable_pitcher_id": prob_id,
                            "is_confirmed": is_confirmed
                        })
                else:
                    records.append({
                        "game_id": gid,
                        "team": team_name,
                        "side": side,
                        "batting_order": None,
                        "batter_id": None,
                        "batter_name": None,
                        "position": None,
                        "pitcher_id": starter,
                        "pitcher_name": starter_name,
                        "probable_pitcher": prob_name,
                        "probable_pitcher_id": prob_id,
                        "is_confirmed": is_confirmed
                    })
    df = pd.DataFrame(records)
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved lineup data to {out}")
    return df

