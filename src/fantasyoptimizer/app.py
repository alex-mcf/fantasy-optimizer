# app.py
import streamlit as st
from fantasyoptimizer.scoring.scoring_engine import compute_scores

st.title("Fantasy Football Optimizer")
st.write("Welcome to the Fantasy Football Optimizer App!")

years = st.multiselect(
    "Select Years to Analyze",
    options=[2022, 2023, 2024],
    default=[2022, 2023, 2024]
)

if years:
    scores = compute_scores(years)

    scores_display = scores[[
        "player", "pos", "team_adp", "adp_avg", "projection",
        "value_over_cost", "vorp", "score", "round"
    ]]

    scores_display = scores_display.rename(columns={
        "player": "Player",
        "pos": "Position",
        "team_adp": "Team",
        "adp_avg": "ADP",
        "projection": "Projected Points",
        "value_over_cost": "Value Over Cost",
        "vorp": "VORP",
        "score": "Score",
        "round": "Draft Round"
    })
    
    pos_filter = st.selectbox(
        "Filter by Position",
        ["ALL", "QB", "RB", "WR", "TE"]
    )

    if pos_filter != "ALL":
        scores_display = scores_display[scores_display["Position"] == pos_filter]

    st.dataframe(scores_display)
else:
    st.write("Please select at least one year to display scores.")