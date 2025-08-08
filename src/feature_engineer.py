import pandas as pd
from src.data_fetcher import fetch_statcast_for_date, fetch_lineups_for_date

def compute_batter_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("batter").agg(
        batter_name=("player_name", lambda x: x.mode().iloc[0] if not x.empty else None),
        recent_hr_rate=("events", lambda x: (x == "home_run").mean() * 100),
        barrel_rate=("barrel", "mean"),
        hard_hit_rate=("hard_hit", "mean")
    ).reset_index()
    agg.rename(columns={"batter": "batter_id"}, inplace=True)
    agg["batter_name"] = agg["batter_name"].fillna("").astype(str)
    return agg

def compute_pitcher_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("pitcher").agg(
        pitcher_name=("player_name", lambda x: x.mode().iloc[0] if not x.empty else None),
        hr_rate_allowed=("events", lambda x: (x == "home_run").mean() * 100)
    ).reset_index()
    agg.rename(columns={"pitcher": "pitcher_id"}, inplace=True)
    agg["pitcher_name"] = agg["pitcher_name"].fillna("").astype(str)
    return agg

def build_matchup_features(date: str) -> pd.DataFrame:
    sc = fetch_statcast_for_date(date)
    lu = fetch_lineups_for_date(date)
    bats = compute_batter_features(sc)
    pits = compute_pitcher_features(sc)
    df = lu.merge(bats, on="batter_id", how="left")
    df = df.merge(pits, on="pitcher_id", how="left")
    df.rename(columns={"batter_name_x": "batter"}, inplace=True)
    df.drop(columns=["batter_name_y"], inplace=True)
    df.rename(columns={"pitcher_name_x": "pitcher"}, inplace=True)
    df.drop(columns=["pitcher_name_y"], inplace=True)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df

