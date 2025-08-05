import streamlit as st
import pandas as pd
import datetime
from src.predict import predict_df

st.set_page_config(page_title="MLB HR Predictor", layout="wide")
st.title("🏟️ Statcast-Powered Home Run Predictor")

date_input = st.date_input("Select Date", value=datetime.date.today())
n = st.slider("Number of hitters", 1, 30, 10)

@st.cache_data
def get_data(date_str: str, top_n: int) -> pd.DataFrame:
    return predict_df(date_str, top_n)

date_str = date_input.strftime("%Y-%m-%d")
df_all = get_data(date_str, n)

available_teams = sorted(df_all["team"].dropna().unique().tolist()) if not df_all.empty else []

if "selected_teams" not in st.session_state:
    st.session_state["selected_teams"] = available_teams

selected_teams = st.multiselect(
    "Select Teams",
    options=available_teams,
    default=st.session_state.get("selected_teams", available_teams),
    key="team_selector_widget"
)
st.session_state["selected_teams"] = selected_teams

status = st.radio(
    "Starter Status",
    ["All", "Confirmed", "Probable"],
    index=0,
    key="status_selector"
)

if st.button("Generate Picks"):
    df = df_all.copy()
    if selected_teams:
        df = df[df["team"].isin(selected_teams)]
    if status == "Confirmed":
        df = df[df["is_confirmed"] == True]
    elif status == "Probable":
        df = df[df["is_confirmed"] == False]
    df = df.reset_index(drop=True)
    st.subheader("Predicted Home Run Hitters")
    st.dataframe(df)
    if not df.empty and {"batter", "composite_score"}.issubset(df.columns):
        st.bar_chart(df.set_index("batter")["composite_score"])

