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
    for col in selected.columns:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")
    selected = selected.fillna(0)
    integer_cols_to_convert = ["batting_order"]
    for col in integer_cols_to_convert:
        if col in selected.columns:
            selected[col] = selected[col].astype("Int64")
    return selected.head(top_n)

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m src.predict YYYY-MM-DD [top_n]")
    date = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(predict_df(date, n).to_string(index=False))

if __name__ == "__main__":
    main()

