import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")

def load_statcast_for_date(date: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(RAW_DIR, f"statcast_{date}.parquet"))

def compute_batter_features(sc: pd.DataFrame) -> pd.DataFrame:
    df = sc.copy()
    if "barrel" not in df.columns:
        if "launch_speed_angle" in df.columns:
            mask = df["launch_speed_angle"] == 6
            df["barrel"] = mask.fillna(False).astype(int)
        else:
            df["barrel"] = ((df["launch_speed"] >= 98) & df["launch_angle"].between(18, 32)).astype(int)
    result = (
        df.groupby("batter")
          .agg(
            recent_hr_rate=("events", lambda x: (x=="home_run").mean()),
            avg_exit_velocity=("launch_speed","mean"),
            avg_launch_angle=("launch_angle","mean"),
            barrel_rate=("barrel","mean")
          )
          .reset_index()
          .rename(columns={"batter":"batter_id"})
    )
    return result

def compute_pitcher_features(sc: pd.DataFrame) -> pd.DataFrame:
    df = sc.copy()
    if "barrel" not in df.columns:
        if "launch_speed_angle" in df.columns:
            mask = df["launch_speed_angle"] == 6
            df["barrel"] = mask.fillna(False).astype(int)
        else:
            df["barrel"] = ((df["launch_speed"] >= 98) & df["launch_angle"].between(18, 32)).astype(int)
    result = (
        df.groupby("pitcher")
          .agg(
            hr_rate_allowed=("events", lambda x: (x=="home_run").mean()),
            avg_exit_velocity_allowed=("launch_speed","mean"),
            barrel_rate_allowed=("barrel","mean")
          )
          .reset_index()
          .rename(columns={"pitcher":"pitcher_id"})
    )
    return result

def load_lineups_for_date(date: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(RAW_DIR, f"lineups_{date}.parquet"))

def build_matchup_features(date: str) -> pd.DataFrame:
    sc   = load_statcast_for_date(date)
    bats = compute_batter_features(sc)
    pits = compute_pitcher_features(sc)
    lu   = load_lineups_for_date(date)
    df = (
        lu
        .merge(bats, on="batter_id", how="left")
        .merge(pits, on="pitcher_id", how="left")
    )
    df["composite_score"] = (
        0.45 * df["recent_hr_rate"].fillna(0)
      + 0.45 * (1 - df["hr_rate_allowed"].fillna(df["hr_rate_allowed"].mean()))
      + 0.10 * df["barrel_rate"].fillna(0)
    ) * 100
    return df.sort_values("composite_score", ascending=False)

