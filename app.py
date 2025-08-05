import streamlit as st
import pandas as pd
import datetime
from src.predict import predict_df

st.title("HR Predictor Tool")

date = st.date_input("Select Date", datetime.date.today())
n    = st.number_input("Number of Players", min_value=1, value=10)

@st.cache_data
def get_data(d, count):
    return predict_df(d, count)

df_all = get_data(date.strftime("%Y-%m-%d"), n)

teams = sorted(df_all.get("team", []))
if "selected_teams" not in st.session_state:
    st.session_state["selected_teams"] = teams

selected = st.multiselect(
    "Select Teams",
    options=teams,
    default=st.session_state["selected_teams"],
    key="team_selector_widget"
)
st.session_state["selected_teams"] = selected

status = st.radio(
    "Starter Status",
    ["All", "Confirmed", "Probable"],
    index=0,
    key="status_selector"
)

if st.button("Generate Picks"):
    df = df_all.copy()
    if selected:
        df = df[df["team"].isin(selected)]
    if status == "Confirmed":
        df = df[df.get("is_confirmed", False)]
    elif status == "Probable":
        df = df[~df.get("is_confirmed", False)]
    df = df.reset_index(drop=True)
    st.dataframe(df)

    if "batter" in df.columns and "composite_score" in df.columns:
        st.bar_chart(df.set_index("batter")["composite_score"])
    else:
        st.warning("Cannot generate chart: 'batter' or 'composite_score' missing.")
