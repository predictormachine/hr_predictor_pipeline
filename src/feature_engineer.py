import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")

def load_statcast_for_date(date: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(RAW_DIR, f"statcast_{date}.parquet"))

def compute_batter_features(sc: pd.DataFrame) -> pd.DataFrame:
    df = sc.copy()
    if "barrel" not in df.columns:
        if "launch_speed_angle" in df.columns:
            df["barrel"] = (df["launch_speed_angle"] == 6).fillna(False).astype(int)
        else:
            df["barrel"] = ((df["launch_speed"] >= 98) & df["launch_angle"].between(18, 32)).astype(int)
    df["batter"] = df["player_name"]
    return (
        df.groupby("batter")
          .agg(
            recent_hr_rate      = ("events",       lambda x: (x=="home_run").mean()),
            avg_exit_velocity   = ("launch_speed", "mean"),
            avg_launch_angle    = ("launch_angle", "mean"),
            barrel_rate         = ("barrel",       "mean")
          )
          .reset_index()
    )

def compute_pitcher_features(sc: pd.DataFrame) -> pd.DataFrame:
    df = sc.copy()
    df["pitcher"] = df["player_name"]
    if "barrel" not in df.columns:
        if "launch_speed_angle" in df.columns:
            df["barrel"] = (df["launch_speed_angle"] == 6).fillna(False).astype(int)
        else:
            df["barrel"] = ((df["launch_speed"] >= 98) & df["launch_angle"].between(18, 32)).astype(int)
    return (
        df.groupby("pitcher")
          .agg(
            hr_rate_allowed           = ("events",       lambda x: (x=="home_run").mean()),
            avg_exit_velocity_allowed = ("launch_speed", "mean"),
            barrel_rate_allowed       = ("barrel",       "mean")
          )
          .reset_index()
    )

def load_lineups_for_date(date: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(RAW_DIR, f"lineups_{date}.parquet"))

def build_matchup_features(date: str) -> pd.DataFrame:
    sc   = load_statcast_for_date(date)
    bats = compute_batter_features(sc)
    pits = compute_pitcher_features(sc)
    lu   = load_lineups_for_date(date)

    lu["batter_id"]  = lu["batter_id"].astype(str)
    lu["pitcher_id"] = lu["pitcher_id"].astype(str)
    bats["batter"]   = bats["batter"].astype(str)
    pits["pitcher"]  = pits["pitcher"].astype(str)

    df = lu.merge(bats, left_on="batter_id", right_on="batter", how="left")
    df = df.merge(pits, left_on="pitcher_id", right_on="pitcher", how="left")

    df.rename(columns={"batter_name": "batter", "pitcher_name": "pitcher"}, inplace=True)
    df.drop(columns=["batter_id", "pitcher_id", "batter", "pitcher"], errors="ignore", inplace=True)
    df["batter"]  = lu["batter_name"]
    df["pitcher"] = lu["pitcher_name"]

    for col in [
        "recent_hr_rate", "barrel_rate", "hr_rate_allowed",
        "avg_exit_velocity", "avg_launch_angle",
        "avg_exit_velocity_allowed", "barrel_rate_allowed"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["composite_score"] = (
        0.45 * df["recent_hr_rate"]
      + 0.45 * (1 - df["hr_rate_allowed"])
      + 0.10 * df["barrel_rate"]
    ) * 100

    return df.sort_values("composite_score", ascending=False)
