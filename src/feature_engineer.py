import pandas as pd
from src.data_fetcher import load_statcast_for_date, load_lineups_for_date

def compute_batter_features(sc: pd.DataFrame) -> pd.DataFrame:
    df = sc.copy()
    agg = df.groupby("batter_id").agg(
        batter_name=("player_name", lambda x: x.mode().iloc[0] if not x.empty else None),
        recent_hr_rate=("events", lambda x: (x == "home_run").mean() * 100),
        barrel_rate=("barrel", "mean"),
        hard_hit_rate=("launch_speed", lambda x: (x >= 95).mean() * 100)
    ).reset_index()
    agg['batter_name'] = agg['batter_name'].fillna('').astype(str)
    return agg

def compute_pitcher_features(sc: pd.DataFrame) -> pd.DataFrame:
    df = sc.copy()
    agg = df.groupby("pitcher_id").agg(
        pitcher_name=("player_name", lambda x: x.mode().iloc[0] if not x.empty else None),
        hr_rate_allowed=("events", lambda x: (x == "home_run").mean() * 100)
    ).reset_index()
    agg['pitcher_name'] = agg['pitcher_name'].fillna('').astype(str)
    return agg

def build_matchup_features(date: str) -> pd.DataFrame:
    sc = load_statcast_for_date(date)
    bats = compute_batter_features(sc)
    pits = compute_pitcher_features(sc)
    lu = load_lineups_for_date(date)
    print("\n--- DEBUG: Lineups DataFrame ---")
    print(lu.head(20).to_string())
    print(f"Columns in lineups: {lu.columns.tolist()}")
    if "probable_pitcher" in lu.columns:
        print(f"Number of NaNs in probable_pitcher before merge: {lu['probable_pitcher'].isnull().sum()}")
    else:
        print("No probable_pitcher column in lineups!")
    if not lu.empty:
        lu["batter_id"] = lu["batter_id"].astype(str)
        lu["pitcher_id"] = lu["pitcher_id"].astype(str)
    df = lu.merge(bats, on="batter_id", how="left")
    df = df.merge(pits, on="pitcher_id", how="left", suffixes=("", "_pitcher"))
    print("\n--- DEBUG: After merges ---")
    print(df.head(20).to_string())
    print(f"Columns in df: {df.columns.tolist()}")
    if "probable_pitcher" in df.columns:
        print(f"Number of NaNs in probable_pitcher after merge: {df['probable_pitcher'].isnull().sum()}")
    else:
        print("No probable_pitcher column after merge!")
    if "batter_name" in df.columns:
        df.rename(columns={"batter_name": "batter"}, inplace=True)
    if "pitcher_name" in df.columns:
        df.rename(columns={"pitcher_name": "pitcher"}, inplace=True)
    if "batter_name_pitcher" in df.columns:
        df.drop(columns=["batter_name_pitcher"], inplace=True)
    if "pitcher_name_pitcher" in df.columns:
        df.drop(columns=["pitcher_name_pitcher"], inplace=True)
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(0)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("N/A")
    return df

