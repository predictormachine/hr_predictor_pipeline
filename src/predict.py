import sys
import pandas as pd
from src.data_fetcher import fetch_statcast_for_date, fetch_lineups_for_date
from src.feature_engineer import build_matchup_features

def predict_df(date: str, top_n: int = 10) -> pd.DataFrame:
    fetch_statcast_for_date(date)
    fetch_lineups_for_date(date)
    df = build_matchup_features(date)
    cols = [
        "team", "side", "batter", "pitcher",
        "probable_pitcher", "is_confirmed",
        "recent_hr_rate", "barrel_rate", "hard_hit_rate",
        "hr_rate_allowed", "composite_score"
    ]
    existing = [c for c in cols if c in df.columns]
    selected = df[existing].copy()
    for col in selected.select_dtypes(include=["float64"]).columns:
        selected[col] = selected[col].fillna(0)
    if "batting_order" in selected.columns:
        selected["batting_order"] = pd.to_numeric(selected["batting_order"], errors="coerce").fillna(0).astype("Int64")
    return selected.head(top_n)

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m src.predict YYYY-MM-DD [top_n]")
    date = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    df = predict_df(date, n)
    if df.empty:
        print("No predictions available for this date.", file=sys.stderr)
        sys.exit(1)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

