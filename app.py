import streamlit as st
import pandas as pd
from src.predict import predict_df

st.set_page_config(page_title="MLB HR Predictor", layout="wide")
st.title("🏟️ Statcast-Powered Home Run Predictor")

date = st.date_input("Select date", value=pd.to_datetime("2025-07-30"))
n    = st.slider("Number of hitters", 1, 30, 10)
teams = st.multiselect("Teams", options=[], key="teams")
status = st.radio("Starter status", ["All","Confirmed","Probable"], index=0)

@st.cache_data
def get_data(d, top_n):
    df = predict_df(d, top_n)
    st.session_state["teams"] = sorted(df["team"].unique())
    return df

if st.button("Generate Picks"):
    df = get_data(date.strftime("%Y-%m-%d"), n)
    if teams:
        df = df[df["team"].isin(teams)]
    if status=="Confirmed":
        df = df[df["is_confirmed"]]
    elif status=="Probable":
        df = df[~df["is_confirmed"]]
    df = df.reset_index(drop=True)
    st.dataframe(df.style.format({
        "recent_hr_rate":"{:.1%}",
        "barrel_rate":"{:.1%}",
        "hr_rate_allowed":"{:.1%}",
        "composite_score":"{:.1f}"
    }).background_gradient("Blues", subset=["composite_score"]))
    st.bar_chart(df.set_index("batter")["composite_score"])
