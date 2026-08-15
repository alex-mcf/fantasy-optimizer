import streamlit as st

from fantasyoptimizer.config.league_config import LeagueConfig
from fantasyoptimizer.forecasting.forecaster import (
    backtest_draft_value,
    backtest_forecaster,
    build_training_examples,
    build_team_position_outlook,
    forecast_season,
)
from fantasyoptimizer.scoring.scoring_engine import (
    SUPPORTED_POSITIONS,
    compute_player_season_scores,
    compute_scores,
)
from fantasyoptimizer.utils.data_loader import (
    available_adp_years,
    available_years,
    data_quality_report,
)

st.set_page_config(page_title="Fantasy Draft Value Forecaster", layout="wide")
st.title("Fantasy Draft Value Forecaster")
st.caption(
    "Half-PPR historical analysis and independent model value versus mock-draft ADP"
)
st.info(
    "The forecast uses prior NFL results only. Current mock ADP is kept out of the "
    "prediction and is joined afterward so differences remain meaningful.",
    icon="ℹ️",
)


@st.cache_data
def load_analysis(years: tuple[int, ...], config: LeagueConfig):
    return (
        compute_scores(list(years), config),
        compute_player_season_scores(list(years), config),
        data_quality_report(list(years)),
    )


@st.cache_data
def load_forecast(target_year: int, config: LeagueConfig):
    training = build_training_examples(target_year - 1)
    return (
        forecast_season(target_year, config, training_examples=training),
        backtest_forecaster(target_year - 1, training_examples=training),
        backtest_draft_value(
            target_year - 1, config, training_examples=training
        ),
    )


year_options = available_years()
if not year_options:
    st.error("No complete seasons were found. Follow the data setup steps in README.md.")
    st.stop()

with st.sidebar:
    st.header("Analysis settings")
    years = st.multiselect("Seasons", options=year_options, default=year_options)
    st.subheader("League roster")
    league_size = st.number_input("Teams", min_value=4, max_value=20, value=12)
    qb = st.number_input("Starting QB", min_value=0, max_value=3, value=1)
    rb = st.number_input("Starting RB", min_value=0, max_value=5, value=2)
    wr = st.number_input("Starting WR", min_value=0, max_value=5, value=2)
    te = st.number_input("Starting TE", min_value=0, max_value=3, value=1)
    flex = st.number_input("FLEX", min_value=0, max_value=4, value=1)
    superflex = st.number_input("Superflex", min_value=0, max_value=2, value=0)

if not years:
    st.warning("Select at least one complete season.")
    st.stop()

league_config = LeagueConfig(
    league_size=int(league_size),
    qb=int(qb),
    rb=int(rb),
    wr=int(wr),
    te=int(te),
    flex=int(flex),
    superflex=int(superflex),
)
with st.sidebar:
    replacements = league_config.replacement_ranks()
    st.caption(
        "Estimated replacement ranks: "
        + ", ".join(f"{position}{rank}" for position, rank in replacements.items())
    )

with st.spinner("Calculating historical draft value..."):
    summary, player_seasons, quality = load_analysis(tuple(years), league_config)

forecast_years = sorted(set(available_adp_years()) - set(year_options))
tab_names = ["Player summary", "Season details", "Data quality & method"]
if forecast_years:
    tab_names.insert(0, f"{max(forecast_years)} Forecast")
tabs = st.tabs(tab_names)
if forecast_years:
    forecast_tab, summary_tab, season_tab, quality_tab = tabs
else:
    summary_tab, season_tab, quality_tab = tabs

if forecast_years:
    with forecast_tab:
        target_year = max(forecast_years)
        with st.spinner(f"Training chronological model and forecasting {target_year}..."):
            forecast, backtest, value_backtest_result = load_forecast(
                target_year, league_config
            )
            value_backtest, value_backtest_players = value_backtest_result

        st.subheader(f"{target_year} model value versus mock ADP")
        st.caption(
            "Positive value gap means the model ranks a player earlier than human "
            "half-PPR mock drafts. Forecast points do not use ADP as an input."
        )
        fc_search_col, fc_position_col, fc_confidence_col = st.columns([3, 1, 1])
        fc_search = fc_search_col.text_input(
            "Forecast player search", placeholder="Player name"
        )
        fc_position = fc_position_col.selectbox(
            "Forecast position", ["ALL", *SUPPORTED_POSITIONS]
        )
        fc_confidence = fc_confidence_col.selectbox(
            "Forecast confidence",
            ["ALL", "High", "Medium", "Low", "Rookie / no NFL history"],
        )
        forecast_filtered = forecast.copy()
        if fc_search:
            forecast_filtered = forecast_filtered[
                forecast_filtered["player"].str.contains(fc_search, case=False, na=False)
            ]
        if fc_position != "ALL":
            forecast_filtered = forecast_filtered[forecast_filtered["pos"] == fc_position]
        if fc_confidence != "ALL":
            forecast_filtered = forecast_filtered[
                forecast_filtered["confidence"] == fc_confidence
            ]

        value_count = int((forecast_filtered["value_gap"] >= 12).sum())
        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Players modeled", len(forecast_filtered))
        metric_2.metric("Model ≥1 round earlier", value_count)
        metric_3.metric("Training result seasons", len(year_options))

        chart = forecast_filtered[
            ["player", "pos", "adp_avg", "model_rank"]
        ].rename(columns={"adp_avg": "Mock ADP", "model_rank": "Model Rank"})
        if not chart.empty:
            st.scatter_chart(
                chart,
                x="Mock ADP",
                y="Model Rank",
                color="pos",
                size=40,
                height=350,
            )

        forecast_display = forecast_filtered[
            [
                "model_rank",
                "player",
                "pos",
                "team",
                "forecast_points",
                "player_only_points",
                "context_adjustment",
                "position_rank",
                "model_value",
                "team_context_rank",
                "weighted_ppo",
                "weighted_opportunity_share",
                "team_pos_weighted_opportunities",
                "team_pos_other_weighted_points",
                "same_team_last_year",
                "changed_team",
                "adp_avg",
                "value_gap",
                "timesdrafted",
                "confidence",
            ]
        ].rename(
            columns={
                "model_rank": "Model Rank",
                "player": "Player",
                "pos": "Position",
                "team": "Team",
                "forecast_points": "Forecast Points",
                "player_only_points": "Player-Only Points",
                "context_adjustment": "Team Context Adjustment",
                "position_rank": "Position Rank",
                "model_value": "Forecast VORP",
                "team_context_rank": "Team Position Context Rank",
                "weighted_ppo": "Prior Points / Opportunity",
                "weighted_opportunity_share": "Prior Opportunity Share %",
                "team_pos_weighted_opportunities": "3-Year Team Position Opportunities",
                "team_pos_other_weighted_points": "3-Year Teammate Points",
                "same_team_last_year": "Same Team",
                "changed_team": "Changed Team",
                "adp_avg": "Mock ADP",
                "value_gap": "Value Gap",
                "timesdrafted": "Mock Samples",
                "confidence": "History Confidence",
            }
        )
        forecast_display["Player"] = forecast_display["Player"].str.title()
        forecast_display["Prior Opportunity Share %"] = (
            100 * forecast_display["Prior Opportunity Share %"]
        ).round(0)
        st.dataframe(forecast_display, hide_index=True, width="stretch")
        st.download_button(
            f"Download {target_year} forecast",
            forecast_display.to_csv(index=False),
            file_name=f"fantasy_forecast_{target_year}.csv",
            mime="text/csv",
        )

        st.subheader("Team-position outlook")
        st.caption(
            "This view measures the destination team's recent production at each "
            "position. For each candidate, his own past production is removed from "
            "the teammate-environment signal—even when he stayed on the same team."
        )
        outlook = build_team_position_outlook(forecast)
        outlook_position = st.selectbox(
            "Outlook position", ["ALL", *SUPPORTED_POSITIONS]
        )
        if outlook_position != "ALL":
            outlook = outlook[outlook["pos"] == outlook_position]
        outlook_display = outlook[
            [
                "team",
                "pos",
                "context_rank",
                "weighted_room_opportunities",
                "last_year_room_points",
                "last_year_room_opportunities",
                "last_year_other_points",
                "last_year_other_opportunities",
                "last_year_top_two",
                "last_year_rank_percentile",
                "model_favorite",
                "favorite_model_rank",
                "favorite_forecast_points",
                "favorite_context_adjustment",
                "favorite_role_share",
                "candidates",
            ]
        ].rename(
            columns={
                "team": "Team",
                "pos": "Position",
                "context_rank": "Context Rank",
                "weighted_room_opportunities": "3-Year Weighted Room Opportunities",
                "last_year_room_points": "Last-Year Room Points",
                "last_year_room_opportunities": "Last-Year Room Opportunities",
                "last_year_other_points": "Last-Year Teammate Points",
                "last_year_other_opportunities": "Last-Year Teammate Opportunities",
                "last_year_top_two": "Last-Year Top-Two Points",
                "last_year_rank_percentile": "Last-Year Position Percentile",
                "model_favorite": "Model Favorite",
                "favorite_model_rank": "Favorite Model Rank",
                "favorite_forecast_points": "Favorite Forecast Points",
                "favorite_context_adjustment": "Favorite Context Adjustment",
                "favorite_role_share": "Favorite Prior Opportunity Share",
                "candidates": "Current Candidates",
            }
        )
        outlook_display["Model Favorite"] = outlook_display["Model Favorite"].str.title()
        outlook_display["Current Candidates"] = outlook_display[
            "Current Candidates"
        ].str.title()
        outlook_display["Last-Year Position Percentile"] = (
            100 * outlook_display["Last-Year Position Percentile"]
        ).round(0)
        outlook_display["Favorite Prior Opportunity Share"] = (
            100 * outlook_display["Favorite Prior Opportunity Share"]
        ).round(0)
        for column in [
            "3-Year Weighted Room Opportunities",
            "Last-Year Room Points",
            "Last-Year Room Opportunities",
            "Last-Year Teammate Points",
            "Last-Year Teammate Opportunities",
            "Last-Year Top-Two Points",
            "Favorite Forecast Points",
            "Favorite Context Adjustment",
        ]:
            outlook_display[column] = outlook_display[column].round(1)
        st.dataframe(outlook_display, hide_index=True, width="stretch")

        st.subheader("Chronological backtest")
        if backtest.empty:
            st.warning("Not enough earlier seasons are available for a backtest.")
        else:
            backtest_display = backtest.rename(
                columns={
                    "year": "Season",
                    "players": "Players",
                    "mae": "Point MAE",
                    "player_only_mae": "Player-Only MAE",
                    "context_mae_lift": "Context MAE Improvement",
                    "rank_correlation": "Rank Correlation",
                }
            )
            for column in [
                "Point MAE",
                "Player-Only MAE",
                "Context MAE Improvement",
            ]:
                backtest_display[column] = backtest_display[column].round(1)
            backtest_display["Rank Correlation"] = backtest_display[
                "Rank Correlation"
            ].round(3)
            st.dataframe(backtest_display, hide_index=True, width="stretch")

        st.subheader("Did model bargains actually beat ADP?")
        st.caption(
            "For each season, the model was trained only on earlier seasons. A "
            "bargain means its league-adjusted model rank was at least one round "
            "ahead of market rank; a hit means its realized league-adjusted rank "
            "finished ahead of that market rank."
        )
        if value_backtest.empty:
            st.warning("Not enough earlier seasons are available for a value backtest.")
        else:
            overall = value_backtest[value_backtest["year"] == "Overall"].iloc[0]
            value_metric_1, value_metric_2, value_metric_3, value_metric_4 = st.columns(4)
            value_metric_1.metric("Historical bargains", int(overall["bargains"]))
            value_metric_2.metric(
                "Bargain hit rate", f'{100 * overall["bargain_hit_rate"]:.1f}%'
            )
            value_metric_3.metric(
                "ADP-round-adjusted lift",
                f'{100 * overall["round_adjusted_hit_lift"]:+.1f} pts',
            )
            value_metric_4.metric(
                "Model override win rate",
                f'{100 * overall["bargain_override_win_rate"]:.1f}%'
            )
            value_display = value_backtest.rename(
                columns={
                    "year": "Season",
                    "players": "Players",
                    "bargains": "Bargains",
                    "bargain_hit_rate": "Bargain Hit Rate",
                    "non_bargain_hit_rate": "Other-Player Hit Rate",
                    "bargain_hit_lift": "Hit-Rate Lift",
                    "round_adjusted_expected_hit_rate": "Same-Round Baseline Hit Rate",
                    "round_adjusted_hit_lift": "Same-Round Hit-Rate Lift",
                    "bargain_override_win_rate": "Model Override Win Rate",
                    "bargain_avg_actual_surplus": "Bargain Avg Actual Rank Surplus",
                    "non_bargain_avg_actual_surplus": "Other Avg Actual Rank Surplus",
                    "fades": "Fades",
                    "fade_hit_rate": "Fade Hit Rate",
                    "market_rank_mae": "Market Rank MAE",
                    "model_rank_mae": "Model Rank MAE",
                    "rank_mae_improvement": "Rank MAE Improvement",
                    "market_correlation": "Market Correlation",
                    "signal_correlation": "Signal Correlation",
                }
            )
            value_display["Season"] = value_display["Season"].astype(str)
            for column in [
                "Bargain Hit Rate",
                "Other-Player Hit Rate",
                "Hit-Rate Lift",
                "Same-Round Baseline Hit Rate",
                "Same-Round Hit-Rate Lift",
                "Model Override Win Rate",
                "Fade Hit Rate",
            ]:
                value_display[column] = (100 * value_display[column]).round(1)
            for column in [
                "Bargain Avg Actual Rank Surplus",
                "Other Avg Actual Rank Surplus",
                "Market Rank MAE",
                "Model Rank MAE",
                "Rank MAE Improvement",
                "Market Correlation",
                "Signal Correlation",
            ]:
                value_display[column] = value_display[column].round(2)
            st.dataframe(value_display, hide_index=True, width="stretch")
            st.download_button(
                "Download historical value-backtest players",
                value_backtest_players.to_csv(index=False),
                file_name="historical_model_value_backtest.csv",
                mime="text/csv",
            )

with summary_tab:
    filter_col, position_col, history_col, active_col = st.columns([3, 1, 1, 1])
    search = filter_col.text_input("Search players", placeholder="Player name")
    position = position_col.selectbox("Position", ["ALL", *SUPPORTED_POSITIONS])
    min_seasons = history_col.number_input(
        "Minimum seasons", min_value=1, max_value=len(years), value=1
    )
    latest_only = active_col.checkbox("Latest season only", value=True)

    filtered = summary.copy()
    if search:
        filtered = filtered[filtered["player"].str.contains(search, case=False, na=False)]
    if position != "ALL":
        filtered = filtered[filtered["pos"] == position]
    filtered = filtered[filtered["seasons_played"] >= min_seasons]
    if latest_only:
        filtered = filtered[filtered["latest_year"] == max(years)]

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Players shown", len(filtered))
    metric_2.metric("Complete seasons", len(years))
    metric_3.metric("Latest season", max(years))

    chart_data = filtered[["player", "pos", "adp_avg", "score"]].rename(
        columns={"adp_avg": "ADP", "score": "Historical score"}
    )
    if not chart_data.empty:
        st.scatter_chart(
            chart_data,
            x="ADP",
            y="Historical score",
            color="pos",
            size=40,
            height=350,
        )

    summary_display = filtered[
        [
            "player",
            "pos",
            "team_adp",
            "latest_year",
            "adp_avg",
            "projection",
            "value_over_cost",
            "vorp",
            "risk_penalty",
            "seasons_played",
            "confidence",
            "score",
        ]
    ].rename(
        columns={
            "player": "Player",
            "pos": "Position",
            "team_adp": "Team",
            "latest_year": "Latest Season",
            "adp_avg": "Latest ADP",
            "projection": "Weighted Points",
            "value_over_cost": "Value Over Cost",
            "vorp": "VORP",
            "risk_penalty": "Risk Penalty",
            "seasons_played": "Seasons",
            "confidence": "History Confidence",
            "score": "Historical Score",
        }
    )
    summary_display["Player"] = summary_display["Player"].str.title()
    summary_display["Team"] = summary_display["Team"].fillna("—")
    st.dataframe(summary_display, hide_index=True, width="stretch")
    st.download_button(
        "Download filtered summary",
        summary_display.to_csv(index=False),
        file_name="fantasy_draft_value_summary.csv",
        mime="text/csv",
    )

with season_tab:
    selected_year = st.selectbox("Season", sorted(years, reverse=True))
    season_position = st.selectbox(
        "Season position", ["ALL", *SUPPORTED_POSITIONS], key="season_position"
    )
    season_search = st.text_input(
        "Season player search", placeholder="Player name", key="season_search"
    )
    season_rows = player_seasons[player_seasons["year"] == selected_year].copy()
    if season_position != "ALL":
        season_rows = season_rows[season_rows["pos"] == season_position]
    if season_search:
        season_rows = season_rows[
            season_rows["player"].str.contains(season_search, case=False, na=False)
        ]

    if not season_rows.empty:
        st.scatter_chart(
            season_rows,
            x="adp_avg",
            y="value_over_cost",
            color="pos",
            size=40,
            height=350,
        )
    season_display = season_rows[
        [
            "player",
            "pos",
            "team_adp",
            "adp_avg",
            "round",
            "projection",
            "expected_points_at_cost",
            "value_over_cost",
            "vorp",
            "risk",
            "season_score",
        ]
    ].rename(
        columns={
            "player": "Player",
            "pos": "Position",
            "team_adp": "Team",
            "adp_avg": "ADP",
            "round": "Round",
            "projection": "Fantasy Points",
            "expected_points_at_cost": "Expected at Cost",
            "value_over_cost": "Value Over Cost",
            "vorp": "VORP",
            "risk": "Multi-Year PPG Risk",
            "season_score": "Season Score",
        }
    )
    season_display["Player"] = season_display["Player"].str.title()
    st.dataframe(season_display, hide_index=True, width="stretch")
    st.download_button(
        "Download season rows",
        season_display.to_csv(index=False),
        file_name=f"fantasy_draft_value_{selected_year}.csv",
        mime="text/csv",
    )

with quality_tab:
    st.subheader("ADP-to-results matching")
    quality_display = quality.copy()
    quality_display["adp_match_rate"] = quality_display["adp_match_rate"].map(
        lambda value: f"{value:.1%}"
    )
    st.dataframe(quality_display, hide_index=True, width="stretch")
    st.caption(
        "Only matched player-position records can be scored. The examples above make "
        "name differences, defenses, retirements, and players without results visible."
    )
    st.subheader("Method")
    st.markdown(
        """
        - **Value over cost** compares realized points with the result expected at the
          same season and positional ADP slot.
        - **VORP** uses replacement ranks derived from the league roster settings.
        - **Risk** measures multi-season points-per-game variation. One-season players
          are marked low-confidence instead of being described as risk-free.
        - Each older season receives half the weight of the following season.

        The forecast is a separate ridge model trained on prior results. It does not
        use current ADP as an input. Mock ADP comes from Fantasy Football Calculator's
        human half-PPR drafts; results come from nflverse game data. Separate position
        models combine player history with the destination team's prior production at
        that position.

        Data provenance and import commands are documented in `data/SOURCES.md`.
        """
    )
