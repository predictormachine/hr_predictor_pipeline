# src/data_fetcher.py
import os
import pandas as pd
import statsapi
from pybaseball import statcast

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")

def fetch_statcast_for_date(date: str) -> pd.DataFrame:
    df = statcast(start_dt=date, end_dt=date)
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_parquet(os.path.join(RAW_DIR, f"statcast_{date}.parquet"), index=False)
    return df

def fetch_lineups_for_date(date: str) -> pd.DataFrame:
    records = []
    sched = statsapi.get(
        "schedule",
        {"sportId": 1, "startDate": date, "endDate": date, "hydrate": "probablePitcher"}
    )
    for day in sched.get("dates", []):
        for g in day.get("games", []):
            gid = g.get("gamePk")
            away_name = g["teams"]["away"]["team"]["name"]
            home_name = g["teams"]["home"]["team"]["name"]
            pp = g.get("probablePitcher", {})
            away_prob = pp.get("awayProbablePitcher", {})
            home_prob = pp.get("homeProbablePitcher", {})
            box = statsapi.boxscore_data(gid)
            for side in ("away", "home"):
                team = away_name if side == "away" else home_name
                prob = away_prob if side == "away" else home_prob
                bat_ids = box[side].get("battingOrder", [])
                players = box[side].get("players", {})
                pits = box[side].get("pitchers", [])
                confirmed_id = pits[0] if pits else None
                confirmed_name = players.get(f"ID{confirmed_id}", {}).get("person", {}).get("fullName")
                for idx, bid in enumerate(bat_ids, 1):
                    b = players.get(f"ID{bid}", {})
                    records.append({
                        "game_id":            gid,
                        "team":               team,
                        "side":               side,
                        "batting_order":      idx,
                        "batter":             b.get("person", {}).get("fullName"),
                        "position":           b.get("position", {}).get("abbreviation"),
                        "pitcher":            confirmed_name,
                        "confirmed_pitcher_id": confirmed_id,
                        "probable_pitcher":   prob.get("fullName"),
                        "probable_pitcher_id": prob.get("id"),
                        "is_confirmed":       confirmed_id == prob.get("id")
                    })
    df = pd.DataFrame(records)
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_parquet(os.path.join(RAW_DIR, f"lineups_{date}.parquet"), index=False)
    return df

