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

    if "batter_id" in lu.columns:
        lu["batter_id"] = lu["batter_id"].astype(str)
    else:
        raise KeyError("Expected 'batter_id' column not found in lineups DataFrame.")

    if "batter" in bats.columns:
        bats["batter"] = bats["batter"].astype(str)
    else:
        raise KeyError("Expected 'batter' column not found in batter features DataFrame.")

    df = lu.merge(bats, left_on="batter_id", right_on="batter", how="left")

    if "pitcher_id" in df.columns:
        df["pitcher_id"] = df["pitcher_id"].astype(str)
    else:
        raise KeyError("Expected 'pitcher_id' column not found in merged DataFrame.")

    if "pitcher" in pits.columns:
        pits["pitcher"] = pits["pitcher"].astype(str)
    else:
        raise KeyError("Expected 'pitcher' column not found in pitcher features DataFrame.")

    df = df.merge(pits, left_on="pitcher_id", right_on="pitcher", how="left")

    df.drop(columns=["batter","pitcher"], errors="ignore", inplace=True)
    df.fillna(0, inplace=True)

    df["composite_score"] = (
        0.45 * df["recent_hr_rate"]
      + 0.45 * (1 - df["hr_rate_allowed"])
      + 0.10 * df["barrel_rate"]
    ) * 100

    return df.sort_values("composite_score", ascending=False)
