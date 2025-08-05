import sys
import pandas as pd
from src.data_fetcher import fetch_statcast_for_date, fetch_lineups_for_date
from src.feature_engineer import build_matchup_features

def predict_df(date: str, top_n: int = 10) -> pd.DataFrame:
    try:
        fetch_statcast_for_date(date)
    except Exception as exc:
        print(f"Warning: failed to fetch Statcast data for {date}: {exc}")
    try:
        fetch_lineups_for_date(date)
    except Exception as exc:
        print(f"Warning: failed to fetch lineups for {date}: {exc}")
    df = build_matchup_features(date)
    if df.empty:
        return df
    expected_columns = [
        "team",
        "side",
        "batter",
        "pitcher",
        "probable_pitcher_id",
        "is_confirmed",
        "recent_hr_rate",
        "barrel_rate",
        "hr_rate_allowed",
        "composite_score",
    ]
    existing_columns = [col for col in expected_columns if col in df.columns]
    if "batter" not in existing_columns:
        print("Error: 'batter' column is missing after merging. Check feature_engineer.py renaming.")
    if "composite_score" not in existing_columns:
        print("Error: 'composite_score' column is missing after calculations.")
    return df[existing_columns].head(top_n)

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.predict YYYY-MM-DD [top_n]")
    date = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    result = predict_df(date, n)
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
