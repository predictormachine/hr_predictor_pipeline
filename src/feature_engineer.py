import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")

def load_statcast_for_date(date: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(RAW_DIR, f"statcast_{date}.parquet"))

def compute_batter_features(sc: pd.DataFrame) -> pd.DataFrame:
    df = sc.copy()
    if "barrel" not in df.columns:
        df["barrel"] = (
            (df.get("launch_speed_angle")==6).fillna(False).astype(int)
            if "launch_speed_angle" in df.columns
            else ((df["launch_speed"]>=98)&df["launch_angle"].between(18,32)).astype(int)
        )
    result = (
        df.groupby("batter")
          .agg(
            recent_hr_rate   = ("events",      lambda x: (x=="home_run").mean()),
            avg_exit_velocity= ("launch_speed","mean"),
            avg_launch_angle = ("launch_angle","mean"),
            barrel_rate      = ("barrel",      "mean")
          )
          .reset_index()
    )
    result["batter"] = result["batter"].astype(str)
    return result

def compute_pitcher_features(sc: pd.DataFrame) -> pd.DataFrame:
    df = sc.copy()
    if "player_name" in df.columns:
        df["pitcher"] = df["player_name"]
    if "barrel" not in df.columns:
        df["barrel"] = (
            (df.get("launch_speed_angle")==6).fillna(False).astype(int)
            if "launch_speed_angle" in df.columns
            else ((df["launch_speed"]>=98)&df["launch_angle"].between(18,32)).astype(int)
        )
    result = (
        df.groupby("pitcher")
          .agg(
            hr_rate_allowed        = ("events",      lambda x: (x=="home_run").mean()),
            avg_exit_velocity_allowed = ("launch_speed","mean"),
            barrel_rate_allowed    = ("barrel",      "mean")
          )
          .reset_index()
    )
    result["pitcher"] = result["pitcher"].astype(str)
    return result

def load_lineups_for_date(date: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(RAW_DIR, f"lineups_{date}.parquet"))

def build_matchup_features(date: str) -> pd.DataFrame:
    sc   = load_statcast_for_date(date)
    bats = compute_batter_features(sc)
    pits = compute_pitcher_features(sc)
    lu   = load_lineups_for_date(date)
    lu["batter"]  = lu["batter"].astype(str)
    lu["pitcher"] = lu["pitcher"].astype(str)
    df = lu.merge(bats, on="batter", how="left")
    df = df[df["recent_hr_rate"].notna()]
    df = df.merge(pits, on="pitcher", how="left")
    df["composite_score"] = (
        0.45 * df["recent_hr_rate"]
      + 0.45 * (1 - df["hr_rate_allowed"].fillna(df["hr_rate_allowed"].mean()))
      + 0.10 * df["barrel_rate"]
    ) * 100
    return df.sort_values("composite_score", ascending=False)

