import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")

def load_statcast_for_date(date: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"statcast_{date}.parquet")
    try:
        return pd.read_parquet(path)
    except:
        return pd.DataFrame()

def compute_batter_features(sc: pd.DataFrame) -> pd.DataFrame:
    if sc.empty or "batter" not in sc.columns:
        return pd.DataFrame(columns=[
            "batter_id","batter_name",
            "recent_hr_rate","avg_exit_velocity",
            "avg_launch_angle","barrel_rate","hard_hit_rate"
        ])
    df = sc.copy()
    df["batter_id"] = df["batter"].astype(str)
    if "player_name" not in df.columns:
        df["player_name"] = ""
    if "launch_speed_angle" not in df.columns:
        df["launch_speed_angle"] = float("nan")
    if "launch_speed" not in df.columns:
        df["launch_speed"] = float("nan")
    df["barrel"]   = (df["launch_speed_angle"] == 6).fillna(False).astype(int)
    df["hard_hit"] = (df["launch_speed"] >= 95).fillna(False).astype(int)
    agg = df.groupby("batter_id").agg(
        batter_name         = ("player_name", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
        recent_hr_rate      = ("events",       lambda x: (x == "home_run").mean() * 100),
        avg_exit_velocity   = ("launch_speed", "mean"),
        avg_launch_angle    = ("launch_angle", "mean"),
        barrel_rate         = ("barrel",       lambda x: x.mean() * 100),
        hard_hit_rate       = ("hard_hit",     lambda x: x.mean() * 100)
    ).reset_index()
    agg["batter_name"] = agg["batter_name"].fillna("").astype(str)
    return agg

def compute_pitcher_features(sc: pd.DataFrame) -> pd.DataFrame:
    if sc.empty or "pitcher" not in sc.columns:
        return pd.DataFrame(columns=[
            "pitcher_id",
            "hr_rate_allowed","avg_exit_velocity_allowed","barrel_rate_allowed"
        ])
    df = sc.copy()
    df["pitcher_id"] = df["pitcher"].astype(str)
    if "launch_speed_angle" not in df.columns:
        df["launch_speed_angle"] = float("nan")
    if "launch_speed" not in df.columns:
        df["launch_speed"] = float("nan")
    df["barrel"] = (df["launch_speed_angle"] == 6).fillna(False).astype(int)
    agg = df.groupby("pitcher_id").agg(
        hr_rate_allowed           = ("events",       lambda x: (x == "home_run").mean() * 100),
        avg_exit_velocity_allowed = ("launch_speed", "mean"),
        barrel_rate_allowed       = ("barrel",       lambda x: x.mean() * 100)
    ).reset_index()
    return agg

def load_lineups_for_date(date: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"lineups_{date}.parquet")
    try:
        return pd.read_parquet(path)
    except:
        return pd.DataFrame()

def build_matchup_features(date: str) -> pd.DataFrame:
    sc   = load_statcast_for_date(date)
    bats = compute_batter_features(sc)
    pits = compute_pitcher_features(sc)
    lu   = load_lineups_for_date(date)
    if lu.empty:
        return pd.DataFrame()
    if "batter_id" in lu.columns:
        lu["batter_id"] = lu["batter_id"].astype(str)
    if "pitcher_id" in lu.columns:
        lu["pitcher_id"] = lu["pitcher_id"].astype(str)

    df = lu.merge(bats, on="batter_id", how="left")
    df = df.merge(pits, on="pitcher_id", how="left")

    if "batter_name_x" in df.columns:
        df.rename(columns={"batter_name_x": "batter"}, inplace=True)
        if "batter_name_y" in df.columns:
            df.drop(columns=["batter_name_y"], inplace=True, errors="ignore")
    elif "batter_name" in df.columns:
        df.rename(columns={"batter_name": "batter"}, inplace=True)

    if "pitcher_name_x" in df.columns:
        df.rename(columns={"pitcher_name_x": "pitcher"}, inplace=True)
        if "pitcher_name_y" in df.columns:
            df.drop(columns=["pitcher_name_y"], inplace=True, errors="ignore")
    elif "pitcher_name" in df.columns:
        df.rename(columns={"pitcher_name": "pitcher"}, inplace=True)

    df = df.fillna({
        "recent_hr_rate":            0,
        "avg_exit_velocity":         0,
        "avg_launch_angle":          0,
        "barrel_rate":               0,
        "hard_hit_rate":             0,
        "hr_rate_allowed":           0,
        "avg_exit_velocity_allowed": 0,
        "barrel_rate_allowed":       0
    })

    df["composite_score"] = (
        0.45 * df["recent_hr_rate"]
      + 0.45 * (100 - df["hr_rate_allowed"])
      + 0.10 * df["barrel_rate"]
    )

    return df.sort_values("composite_score", ascending=False)
