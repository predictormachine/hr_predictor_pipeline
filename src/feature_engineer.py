import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")

def load_statcast_for_date(date: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"statcast_{date}.parquet")
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def compute_batter_features(sc: pd.DataFrame) -> pd.DataFrame:
    if sc.empty or "player_name" not in sc.columns:
        return pd.DataFrame(columns=["batter", "recent_hr_rate", "avg_exit_velocity", "avg_launch_angle", "barrel_rate"])
    df = sc.copy()
    if "barrel" not in df.columns:
        if "launch_speed_angle" in df.columns:
            df["barrel"] = (df["launch_speed_angle"] == 6).fillna(False).astype(int)
        elif "launch_speed" in df.columns and "launch_angle" in df.columns:
            df["barrel"] = ((df["launch_speed"] >= 98) & df["launch_angle"].between(18, 32)).astype(int)
        else:
            df["barrel"] = 0
    df["batter"] = df["player_name"].astype(str)
    agg_dict = {}
    if "events" in df.columns:
        agg_dict["recent_hr_rate"] = ("events", lambda x: (x == "home_run").mean())
    else:
        df["_dummy"] = 0
        agg_dict["recent_hr_rate"] = ("_dummy", lambda x: 0.0)
    if "launch_speed" in df.columns:
        agg_dict["avg_exit_velocity"] = ("launch_speed", "mean")
    else:
        df["_dummy2"] = 0
        agg_dict["avg_exit_velocity"] = ("_dummy2", "mean")
    if "launch_angle" in df.columns:
        agg_dict["avg_launch_angle"] = ("launch_angle", "mean")
    else:
        df["_dummy3"] = 0
        agg_dict["avg_launch_angle"] = ("_dummy3", "mean")
    agg_dict["barrel_rate"] = ("barrel", "mean")
    result = df.groupby("batter").agg(**agg_dict).reset_index()
    return result

def compute_pitcher_features(sc: pd.DataFrame) -> pd.DataFrame:
    if sc.empty or "player_name" not in sc.columns:
        return pd.DataFrame(columns=["pitcher", "hr_rate_allowed", "avg_exit_velocity_allowed", "barrel_rate_allowed"])
    df = sc.copy()
    df["pitcher"] = df["player_name"].astype(str)
    if "barrel" not in df.columns:
        if "launch_speed_angle" in df.columns:
            df["barrel"] = (df["launch_speed_angle"] == 6).fillna(False).astype(int)
        elif "launch_speed" in df.columns and "launch_angle" in df.columns:
            df["barrel"] = ((df["launch_speed"] >= 98) & df["launch_angle"].between(18, 32)).astype(int)
        else:
            df["barrel"] = 0
    agg_dict = {}
    if "events" in df.columns:
        agg_dict["hr_rate_allowed"] = ("events", lambda x: (x == "home_run").mean())
    else:
        df["_dummy"] = 0
        agg_dict["hr_rate_allowed"] = ("_dummy", lambda x: 0.0)
    if "launch_speed" in df.columns:
        agg_dict["avg_exit_velocity_allowed"] = ("launch_speed", "mean")
    else:
        df["_dummy2"] = 0
        agg_dict["avg_exit_velocity_allowed"] = ("_dummy2", "mean")
    agg_dict["barrel_rate_allowed"] = ("barrel", "mean")
    result = df.groupby("pitcher").agg(**agg_dict).reset_index()
    return result

def load_lineups_for_date(date: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"lineups_{date}.parquet")
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def build_matchup_features(date: str) -> pd.DataFrame:
    sc = load_statcast_for_date(date)
    bats = compute_batter_features(sc)
    pits = compute_pitcher_features(sc)
    lu = load_lineups_for_date(date)
    if "batter_id" in lu.columns:
        lu["batter_id"] = lu["batter_id"].astype(str)
    if "pitcher_id" in lu.columns:
        lu["pitcher_id"] = lu["pitcher_id"].astype(str)
    if "batter" in bats.columns:
        bats["batter"] = bats["batter"].astype(str)
    if "pitcher" in pits.columns:
        pits["pitcher"] = pits["pitcher"].astype(str)
    df = lu.copy()
    if not bats.empty:
        df = df.merge(bats, how="left", left_on="batter_id", right_on="batter", suffixes=("", "_bats"))
    if not pits.empty:
        df = df.merge(pits, how="left", left_on="pitcher_id", right_on="pitcher", suffixes=("", "_pits"))
    if "batter_name" in df.columns:
        df.rename(columns={"batter_name": "batter"}, inplace=True)
    if "pitcher_name" in df.columns:
        df.rename(columns={"pitcher_name": "pitcher"}, inplace=True)
    df.drop(columns=["batter", "pitcher"], errors="ignore", inplace=True)
    if "batter" not in df.columns and "batter_name" in lu.columns:
        df["batter"] = lu["batter_name"]
    if "pitcher" not in df.columns and "pitcher_name" in lu.columns:
        df["pitcher"] = lu["pitcher_name"]
    for col in [
        "recent_hr_rate",
        "barrel_rate",
        "hr_rate_allowed",
        "avg_exit_velocity",
        "avg_launch_angle",
        "avg_exit_velocity_allowed",
        "barrel_rate_allowed",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["composite_score"] = (
        0.45 * df.get("recent_hr_rate", 0)
        + 0.45 * (1 - df.get("hr_rate_allowed", 0))
        + 0.10 * df.get("barrel_rate", 0)
    ) * 100
    return df.sort_values("composite_score", ascending=False)
