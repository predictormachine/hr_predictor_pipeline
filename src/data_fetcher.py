import os
import pandas as pd
try:
    import statsapi
except ImportError:
    statsapi = None
try:
    from pybaseball import statcast as _statcast_func
except ImportError:
    _statcast_func = None

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")

def fetch_statcast_for_date(date: str) -> pd.DataFrame:
    year = date.split("-")[0]
    start_dt = f"{year}-03-01"
    if _statcast_func is None:
        df = pd.DataFrame()
    else:
        try:
            df = _statcast_func(start_dt=start_dt, end_dt=date)
        except Exception:
            df = pd.DataFrame()
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_parquet(os.path.join(RAW_DIR, f"statcast_{date}.parquet"), index=False)
    return df

def fetch_lineups_for_date(date: str) -> pd.DataFrame:
    if statsapi is None:
        return pd.DataFrame()
    records = []
    try:
        sched = statsapi.get(
            "schedule",
            {"sportId": 1, "startDate": date, "endDate": date, "hydrate": "probablePitcher"}
        )
    except Exception:
        sched = {}
    for day in sched.get("dates", []):
        for game in day.get("games", []):
            if not isinstance(game, dict):
                continue
            gid = game.get("gamePk")
            away_team = game.get("teams", {}).get("away", {}).get("team", {}).get("name")
            home_team = game.get("teams", {}).get("home", {}).get("team", {}).get("name")
            probable = game.get("probablePitcher", {}) or {}
            away_prob = probable.get("awayProbablePitcher", {}) or {}
            home_prob = probable.get("homeProbablePitcher", {}) or {}
            try:
                box = statsapi.boxscore_data(gid) if gid else None
            except Exception:
                box = None
            for side, team_name, prob in [
                ("away", away_team, away_prob),
                ("home", home_team, home_prob)
            ]:
                prob_name = prob.get("fullName")
                prob_id = prob.get("id")
                if box and side in box:
                    side_data = box.get(side, {}) or {}
                    bat_ids = side_data.get("battingOrder", []) or []
                    players = side_data.get("players", {}) or {}
                    pitchers_list = side_data.get("pitchers", []) or []
                    confirmed_id = pitchers_list[0] if pitchers_list else None
                    confirmed_name = None
                    if confirmed_id is not None:
                        player_key = f"ID{confirmed_id}"
                        confirmed_name = players.get(player_key, {}).get("person", {}).get("fullName")
                    is_confirmed = (
                        confirmed_id == prob_id
                        if confirmed_id is not None and prob_id is not None
                        else False
                    )
                    for idx, bid in enumerate(bat_ids, start=1):
                        batter_info = players.get(f"ID{bid}", {}) or {}
                        person_data = batter_info.get("person", {}) or {}
                        position_data = batter_info.get("position", {}) or {}
                        batter_name = person_data.get("fullName")
                        position_abbrev = position_data.get("abbreviation")
                        records.append({
                            "game_id": gid,
                            "team": team_name,
                            "side": side,
                            "batting_order": idx,
                            "batter_id": bid,
                            "batter_name": batter_name,
                            "position": position_abbrev,
                            "pitcher_id": confirmed_id,
                            "pitcher_name": confirmed_name,
                            "probable_pitcher": prob_name,
                            "probable_pitcher_id": prob_id,
                            "is_confirmed": is_confirmed,
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
                        "pitcher_id": None,
                        "pitcher_name": None,
                        "probable_pitcher": prob_name,
                        "probable_pitcher_id": prob_id,
                        "is_confirmed": False,
                    })
    df = pd.DataFrame(records)
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_parquet(os.path.join(RAW_DIR, f"lineups_{date}.parquet"), index=False)
    return df

