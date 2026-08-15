import streamlit as st
from fantasyoptimizer.scoring.scoring_engine import compute_scores
from fantasyoptimizer.utils.data_loader import available_years

st.title("Fantasy Football Optimizer")
st.caption("Historical draft-value analysis using preseason ADP and season results")

year_options = available_years()
if not year_options:
    st.error("No complete seasons were found. Follow the data setup steps in README.md.")
    st.stop()

years = st.multiselect("Select seasons", options=year_options, default=year_options)

if years:
    with st.spinner("Calculating historical draft value..."):
        scores = compute_scores(years)

    scores_display = scores[[
        "player", "pos", "team_adp", "adp_avg", "projection",
        "value_over_cost", "vorp", "risk_penalty", "score", "round"
    ]]

    scores_display = scores_display.rename(columns={
        "player": "Player",
        "pos": "Position",
        "team_adp": "Team",
        "adp_avg": "ADP",
        "projection": "Fantasy Points",
        "value_over_cost": "Value Over Cost",
        "vorp": "VORP",
        "risk_penalty": "Risk Penalty",
        "score": "Score",
        "round": "Draft Round"
    })
    
    pos_filter = st.selectbox(
        "Filter by Position",
        ["ALL", "QB", "RB", "WR", "TE"]
    )

    if pos_filter != "ALL":
        scores_display = scores_display[scores_display["Position"] == pos_filter]

    scores_display["Player"] = scores_display["Player"].str.title()
    scores_display["Team"] = scores_display["Team"].fillna("—")
    scores_display["Draft Round"] = scores_display["Draft Round"].round().astype(int)
    st.dataframe(
        scores_display,
        hide_index=True,
        width="stretch",
        column_config={
            "ADP": st.column_config.NumberColumn(format="%.1f"),
            "Fantasy Points": st.column_config.NumberColumn(format="%.1f"),
            "Value Over Cost": st.column_config.NumberColumn(format="%.1f"),
            "VORP": st.column_config.NumberColumn(format="%.1f"),
            "Risk Penalty": st.column_config.NumberColumn(format="%.1f"),
            "Score": st.column_config.NumberColumn(format="%.1f"),
        },
    )
else:
    st.info("Select at least one season to display scores.")
