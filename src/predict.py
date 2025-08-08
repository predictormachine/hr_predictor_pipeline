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

    if "is_confirmed" in selected.columns:
        selected["is_confirmed"] = selected["is_confirmed"].fillna(False).astype(bool)

    for col in selected.select_dtypes(include=["float64", "Int64"]).columns:
        selected[col] = selected[col].fillna(0)

    integer_cols_to_convert = ["batting_order"]
    for col in integer_cols_to_convert:
        if col in selected.columns and selected[col].dtype != "Int64":
            selected[col] = pd.to_numeric(selected[col], errors="coerce").fillna(0).astype("Int64")

    for col in selected.select_dtypes(include=["object"]).columns:
        selected[col] = selected[col].fillna("N/A")

    return selected.head(top_n)

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m src.predict YYYY-MM-DD [top_n]")
    date = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    result = predict_df(date, n)
    if result.empty:
        sys.exit("Prediction dataframe is empty! No HR picks generated for this date.")
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()

