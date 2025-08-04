import sys
import pandas as pd
from src.data_fetcher import fetch_statcast_for_date, fetch_lineups_for_date
from src.feature_engineer import build_matchup_features

def predict_df(date: str, top_n: int = 10) -> pd.DataFrame:
    fetch_statcast_for_date(date)
    fetch_lineups_for_date(date)
    df = build_matchup_features(date)

    expected_columns = [
        "team", "side", "batter", "pitcher",
        "probable_pitcher", "is_confirmed",
        "recent_hr_rate", "barrel_rate", "hr_rate_allowed", "composite_score"
    ]
    existing = [c for c in expected_columns if c in df.columns]
    missing  = set(expected_columns) - set(existing)
    if missing:
        print(f"Warning: these expected columns are missing: {sorted(missing)}")
        print("Actual columns:", df.columns.tolist())

    return df[existing].head(top_n)

def main():
    date = sys.argv[1]
    n    = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    result = predict_df(date, n)
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
