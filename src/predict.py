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

    print("\n--- DEBUG: Before NA Handling ---")
    print(f"Selected Columns: {selected.columns.tolist()}")
    print("Selected Dtypes:\n", selected.dtypes.to_string())
    print("NaN Counts:\n", selected.isnull().sum().to_string())
    print("DataFrame Head (with NaNs):\n", selected.head(10).to_string())

    for col in selected.select_dtypes(include=['float64', 'int64', 'Int64']).columns:
        if selected[col].isnull().any():
            print(f"DEBUG: Filling NaNs in column {col} (dtype: {selected[col].dtype})")
            selected[col] = selected[col].fillna(0)

    integer_cols_to_convert = ["batting_order"]

    for col in integer_cols_to_convert:
        if col in selected.columns and selected[col].dtype != 'Int64':
            print(f"DEBUG: Converting column to Int64: {col} (current dtype: {selected[col].dtype})")
            try:
                selected[col] = pd.to_numeric(selected[col], errors='coerce').fillna(0).astype('Int64')
                print(f"DEBUG: Successfully converted {col} to Int64")
            except Exception as e:
                print(f"ERROR: Failed to convert {col} to Int64: {e}")
                selected[col] = pd.to_numeric(selected[col], errors='coerce').fillna(0)

    print("\n--- DEBUG: After NA Handling and Conversions ---")
    print("Selected Dtypes:\n", selected.dtypes.to_string())
    print("NaN Counts:\n", selected.isnull().sum().to_string())
    print("DataFrame Head (after handling):\n", selected.head(10).to_string())

    return selected.head(top_n)

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m src.predict YYYY-MM-DD [top_n]")
    date = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(predict_df(date, n).to_string(index=False))

if __name__ == "__main__":
    main()

