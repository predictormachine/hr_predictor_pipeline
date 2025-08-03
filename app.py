import streamlit as st
import pandas as pd
import datetime
from src.predict import predict_df

st.set_page_config(page_title="MLB HR Predictor", layout="wide")
st.title("🏟️ Statcast-Powered Home Run Predictor")

date = st.date_input("Select Date", value=datetime.date.today())
n    = st.number_input("Number of hitters", min_value=1, value=10)

@st.cache_data
def load_data(d, top_n):
    return predict_df(d, top_n)

date_str   = date.strftime("%Y-%m-%d")
df_all     = load_data(date_str, n)
teams_list = sorted(df_all["team"].dropna().unique())

if "selected_teams" not in st.session_state:
    st.session_state["selected_teams"] = teams_list

selected = st.multiselect(
    "Select Teams",
    options=teams_list,
    default=st.session_state["selected_teams"],
    key="team_selector_widget"
)
st.session_state["selected_teams"] = selected

status = st.radio(
    "Starter Status",
    ["All","Confirmed","Probable"],
    index=0,
    key="status_selector"
)

if st.button("Generate Picks"):
    df = df_all.copy()
    if selected:
        df = df[df["team"].isin(selected)]
    if status=="Confirmed":
        df = df[df["is_confirmed"]]
    elif status=="Probable":
        df = df[~df["is_confirmed"]]
    df = df.reset_index(drop=True)
    st.dataframe(
        df.style.format({
            "recent_hr_rate":"{:.1%}",
            "barrel_rate":"{:.1%}",
            "hr_rate_allowed":"{:.1%}",
            "composite_score":"{:.1f}"
        }).background_gradient("Blues", subset=["composite_score"]),
        use_container_width=True
    )
    st.bar_chart(df.set_index("batter")["composite_score"])
