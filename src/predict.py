import sys
from src.data_fetcher import fetch_statcast_for_date, fetch_lineups_for_date
from src.feature_engineer import build_matchup_features
import pandas as pd

def predict_df(date: str, top_n: int = 10) -> pd.DataFrame:
    fetch_statcast_for_date(date)
    fetch_lineups_for_date(date)
    df = build_matchup_features(date)
    return df[[
        "team","side","batter","pitcher",
        "probable_pitcher","is_confirmed",
        "recent_hr_rate","barrel_rate","hr_rate_allowed","composite_score"
    ]].head(top_n)

def main():
    date = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv)>2 else 10
    df = predict_df(date, n)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
