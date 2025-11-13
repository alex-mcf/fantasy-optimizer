# app.py
import streamlit as st
from fantasyoptimizer.scoring.scoring_engine import compute_scores

st.title("Fantasy Football Optimizer")
st.write("Welcome to the Fantasy Football Optimizer App!")

years = st.multiselect("Select Years to Analyze", options=[2022, 2023, 2024], default=[2022, 2023, 2024])

if years:
    scores = compute_scores(years)
    st.dataframe(scores)
else:
    st.write("Please select at least one year to display scores.")