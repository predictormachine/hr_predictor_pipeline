import os
import pandas as pd
import statsapi
from pybaseball import statcast

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")

def fetch_statcast_for_date(date: str) -> pd.DataFrame:
    year = date.split("-")[0]
    start_dt = f"{year}-03-01"
    df = statcast(start_dt=start_dt, end_dt=date)
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_parquet(os.path.join(RAW_DIR, f"statcast_{date}.parquet"), index=False)
    return df

def fetch_lineups_for_date(date: str) -> pd.DataFrame:
    records = []
    sched = statsapi.get("schedule", {
        "sportId": 1, "startDate": date, "endDate": date,
        "hydrate": "probablePitcher"
    })
    for day in sched.get("dates", []):
        for g in day.get("games", []):
            if not isinstance(g, dict): continue
            if g.get("status", {}).get("abstractGameState") != "Preview": continue
            gid = g["gamePk"]
            away = g["teams"]["away"]["team"]["name"]
            home = g["teams"]["home"]["team"]["name"]
            pp = g.get("probablePitcher", {})
            away_prob = pp.get("awayProbablePitcher", {})
            home_prob = pp.get("homeProbablePitcher", {})
            box = statsapi.boxscore_data(gid)
            for side in ("away","home"):
                team     = away if side=="away" else home
                prob     = away_prob if side=="away" else home_prob
                bat_ids  = box[side].get("battingOrder", [])
                players  = box[side].get("players", {})
                pits     = box[side].get("pitchers", [])
                starter  = pits[0] if pits else None
                starter_name = players.get(f"ID{starter}",{}).get("person",{}).get("fullName")
                for idx, bid in enumerate(bat_ids,1):
                    b = players.get(f"ID{bid}",{})
                    records.append({
                        "game_id":             gid,
                        "team":                team,
                        "side":                side,
                        "batting_order":       idx,
                        "batter_id":           bid,
                        "batter_name":         b.get("person",{}).get("fullName"),
                        "position":            b.get("position",{}).get("abbreviation"),
                        "pitcher_id":          starter,
                        "pitcher_name":        starter_name,
                        "probable_pitcher":    prob.get("fullName"),
                        "probable_pitcher_id": prob.get("id"),
                        "is_confirmed":        (starter == prob.get("id"))
                    })
    df = pd.DataFrame(records)
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_parquet(os.path.join(RAW_DIR, f"lineups_{date}.parquet"), index=False)
    return df
