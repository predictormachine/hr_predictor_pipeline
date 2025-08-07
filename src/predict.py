import sys
import pandas as pd
from src.data_fetcher import fetch_statcast_for_date, fetch_lineups_for_date
from src.feature_engineer import build_matchup_features

def predict_df(date: str, top_n: int = 10) -> pd.DataFrame:
    print(f"DEBUG: Starting predict_df for date: {date}")
    fetch_statcast_for_date(date)
    print(f"DEBUG: Statcast data fetching attempted for {date}")
    fetch_lineups_for_date(date)
    print(f"DEBUG: Lineups data fetching attempted for {date}")

    df = build_matchup_features(date)
    print(f"DEBUG: build_matchup_features returned DataFrame. Is empty: {df.empty}, Shape: {df.shape}")

    cols = [
        "team", "side", "batter", "pitcher",
        "probable_pitcher", "is_confirmed",
        "recent_hr_rate", "barrel_rate", "hard_hit_rate",
        "hr_rate_allowed", "composite_score"
    ]
    existing = [c for c in cols if c in df.columns]
    selected = df[existing].copy()

    print(f"DEBUG: Selected columns: {selected.columns.tolist()}")
    print(f"DEBUG: NaNs in selected DF after initial select but before fillna/convert:\n{selected.isnull().sum().to_string()}")

    if 'is_confirmed' in selected.columns:
        selected['is_confirmed'] = selected['is_confirmed'].fillna(False).astype(bool)

    for col in selected.select_dtypes(include=['float64', 'Int64']).columns:
        if selected[col].isnull().any():
            print(f"DEBUG: Filling NaNs in numeric column: {col} (dtype: {selected[col].dtype}) with 0")
            selected[col] = selected[col].fillna(0)

    integer_cols_to_convert = ["batting_order"]
    for col in integer_cols_to_convert:
        if col in selected.columns and selected[col].dtype != 'Int64':
            print(f"DEBUG: Converting column to 'Int64': {col} (current dtype: {selected[col].dtype})")
            try:
                selected[col] = pd.to_numeric(selected[col], errors='coerce').fillna(0).astype('Int64')
                print(f"DEBUG: Successfully converted {col} to Int64")
            except Exception as e:
                print(f"ERROR: Failed to convert {col} to Int64: {e}")
                selected[col] = pd.to_numeric(selected[col], errors='coerce').fillna(0)

    for col in selected.select_dtypes(include=['object']).columns:
        if selected[col].isnull().any():
            print(f"DEBUG: Filling NaNs/None in string column: {col} (dtype: {selected[col].dtype}) with 'N/A'")
            selected[col] = selected[col].fillna("N/A")

    print(f"DEBUG: NaNs in DF after final fillna/convert:\n{selected.isnull().sum().to_string()}")
    print(f"DEBUG: Final DataFrame shape: {selected.shape}, Is empty: {selected.empty}")
    print(f"DEBUG: Returning head({top_n}):\n{selected.head(top_n).to_string()}")

    return selected.head(top_n)

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m src.predict YYYY-MM-DD [top_n]")
    date = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    df = predict_df(date, n)
    if df.empty:
        print("No predictions available for this date.")
        sys.exit(0)

    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

