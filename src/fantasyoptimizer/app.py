import streamlit as st

from fantasyoptimizer.config.league_config import LeagueConfig
from fantasyoptimizer.forecasting.forecaster import (
    backtest_draft_value,
    backtest_forecaster,
    build_training_examples,
    build_team_position_outlook,
    calibrate_edge_probabilities,
    forecast_season,
    segment_draft_value,
)
from fantasyoptimizer.optimizer import (
    build_draft_recommendations,
    simulate_historical_draft_strategies,
)
from fantasyoptimizer.scoring.scoring_engine import (
    SUPPORTED_POSITIONS,
    compute_player_season_scores,
    compute_scores,
)
from fantasyoptimizer.utils.data_loader import (
    available_adp_snapshots,
    available_adp_years,
    available_years,
    build_adp_movement,
    data_quality_report,
    load_adp_metadata,
)

st.set_page_config(page_title="Fantasy Draft Value Forecaster", layout="wide")
st.title("Fantasy Draft Value Forecaster")
st.caption(
    "Half-PPR market-error forecasting and live draft decision support"
)
st.info(
    "ADP is the market baseline. Position-specific models use prior player, role, "
    "and team evidence to predict where that baseline is wrong.",
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
    forecast = forecast_season(target_year, config, training_examples=training)
    backtest = backtest_forecaster(target_year - 1, training_examples=training)
    value_backtest, value_players = backtest_draft_value(
        target_year - 1, config, training_examples=training
    )
    forecast = calibrate_edge_probabilities(forecast, value_players, config)
    segments = segment_draft_value(value_players, config)
    draft_simulation = simulate_historical_draft_strategies(value_players, config)
    return (
        forecast,
        backtest,
        value_backtest,
        value_players,
        segments,
        draft_simulation,
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
        adp_metadata = load_adp_metadata(target_year)
        adp_fetched_at = str(adp_metadata.get("fetched_at_utc", "Unknown"))
        snapshot_count = len(available_adp_snapshots(target_year))
        adp_movement = build_adp_movement(target_year)
        with st.spinner(f"Training chronological model and forecasting {target_year}..."):
            (
                forecast,
                backtest,
                value_backtest,
                value_backtest_players,
                value_segments,
                draft_simulation,
            ) = load_forecast(
                target_year, league_config
            )

        st.subheader(f"{target_year} model value versus mock ADP")
        st.caption(
            f"Market snapshot fetched: {adp_fetched_at} · Preserved snapshots: "
            f"{snapshot_count}"
        )
        st.caption(
            "Positive value gap means the model's fair ADP is earlier than the "
            "normalized mock market. The model starts from ADP, then adjusts it "
            "with player ability, role, destination context, and availability. "
            "Room fields are derived from the ADP pool; they are not an official "
            "NFL depth chart."
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

        value_count = int((forecast_filtered["value_gap"] >= league_size).sum())
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
                "age",
                "experience",
                "rookie",
                "draft_round",
                "draft_pick",
                "forecast_points",
                "market_points",
                "market_adjustment",
                "player_only_points",
                "context_adjustment",
                "forecast_ppg",
                "forecast_games",
                "position_rank",
                "model_value",
                "team_context_rank",
                "weighted_ppo",
                "weighted_opportunity_share",
                "team_pos_weighted_opportunities",
                "team_pos_other_weighted_points",
                "same_team_last_year",
                "changed_team",
                "market_room_rank",
                "market_room_size",
                "market_room_leader",
                "market_room_adp_gap",
                "adp_avg",
                "market_rank",
                "fair_adp",
                "actionable_adp",
                "value_gap",
                "edge_probability",
                "edge_confidence",
                "timesdrafted",
                "confidence",
            ]
        ].rename(
            columns={
                "model_rank": "Model Rank",
                "player": "Player",
                "pos": "Position",
                "team": "Team",
                "age": "Age",
                "experience": "NFL Experience",
                "rookie": "Rookie",
                "draft_round": "NFL Draft Round",
                "draft_pick": "NFL Draft Pick",
                "forecast_points": "Forecast Points",
                "market_points": "Market-Implied Points",
                "market_adjustment": "Predicted Market Error",
                "player_only_points": "Player-Only Points",
                "context_adjustment": "Team Context Adjustment",
                "forecast_ppg": "Forecast PPG",
                "forecast_games": "Forecast Games",
                "position_rank": "Position Rank",
                "model_value": "Forecast VORP",
                "team_context_rank": "Team Position Context Rank",
                "weighted_ppo": "Prior Points / Opportunity",
                "weighted_opportunity_share": "Prior Opportunity Share %",
                "team_pos_weighted_opportunities": "3-Year Team Position Opportunities",
                "team_pos_other_weighted_points": "3-Year Teammate Points",
                "same_team_last_year": "Same Team",
                "changed_team": "Changed Team",
                "market_room_rank": "Same-Team Position ADP Rank",
                "market_room_size": "Same-Team Position ADP Candidates",
                "market_room_leader": "Room ADP Leader",
                "market_room_adp_gap": "ADP Picks Behind Room Leader",
                "adp_avg": "Mock ADP",
                "market_rank": "Normalized Market Rank",
                "fair_adp": "Fair ADP",
                "actionable_adp": "Actionable ADP",
                "value_gap": "Expected Pick Value",
                "edge_probability": "Probability Beat ADP %",
                "edge_confidence": "Edge Confidence",
                "timesdrafted": "Mock Samples",
                "confidence": "History Confidence",
            }
        )
        forecast_display["Player"] = forecast_display["Player"].str.title()
        forecast_display["Prior Opportunity Share %"] = (
            100 * forecast_display["Prior Opportunity Share %"]
        ).round(0)
        forecast_display["Probability Beat ADP %"] = (
            100 * forecast_display["Probability Beat ADP %"]
        ).round(0)
        st.dataframe(forecast_display, hide_index=True, width="stretch")
        st.download_button(
            f"Download {target_year} forecast",
            forecast_display.to_csv(index=False),
            file_name=f"fantasy_forecast_{target_year}.csv",
            mime="text/csv",
        )

        with st.expander("ADP movement"):
            if adp_movement.empty:
                st.info(
                    "One ADP snapshot is preserved. Refresh the ADP importer on "
                    "another day to begin measuring risers, fallers, and future "
                    "draft-day ADP."
                )
            else:
                movement_display = adp_movement[
                    [
                        "player",
                        "pos",
                        "team",
                        "first_adp",
                        "latest_adp",
                        "adp_movement",
                        "snapshots",
                    ]
                ].rename(
                    columns={
                        "player": "Player",
                        "pos": "Position",
                        "team": "Team",
                        "first_adp": "First ADP",
                        "latest_adp": "Latest ADP",
                        "adp_movement": "Picks Risen",
                        "snapshots": "Snapshots",
                    }
                )
                movement_display["Player"] = movement_display["Player"].str.title()
                st.dataframe(movement_display, hide_index=True, width="stretch")

        st.subheader("Live draft decision board")
        st.caption(
            "Enter the current and next selection, then mark drafted players. The "
            "board distinguishes players to take now from values likely to survive "
            "until your next turn. Availability uses mock-pick variability and is "
            "an estimate, not a guarantee."
        )
        pick_col, next_col = st.columns(2)
        current_pick = int(
            pick_col.number_input("Current overall pick", min_value=1, value=1)
        )
        next_pick = int(
            next_col.number_input(
                "Your next overall pick",
                min_value=current_pick + 1,
                value=max(current_pick + 1, 2 * league_size),
            )
        )
        drafted_players = st.multiselect(
            "Players already drafted",
            options=forecast["player"].tolist(),
            format_func=lambda player: str(player).title(),
        )
        st.caption("Your current roster counts")
        roster_cols = st.columns(4)
        roster_counts = {
            position: int(
                column.number_input(
                    position,
                    min_value=0,
                    max_value=10,
                    value=0,
                    key=f"roster_{position}",
                )
            )
            for position, column in zip(SUPPORTED_POSITIONS, roster_cols)
        }
        draft_board = build_draft_recommendations(
            forecast,
            current_pick,
            next_pick,
            roster_counts,
            set(drafted_players),
            league_config,
        )
        draft_display = draft_board.head(25)[
            [
                "recommendation",
                "player",
                "pos",
                "team",
                "market_room_rank",
                "market_room_size",
                "market_room_adp_gap",
                "fair_adp",
                "actionable_adp",
                "adp_avg",
                "value_gap",
                "edge_probability",
                "available_next_pick_probability",
                "forecast_points",
                "remaining_starter_need",
            ]
        ].rename(
            columns={
                "recommendation": "Recommendation",
                "player": "Player",
                "pos": "Position",
                "team": "Team",
                "market_room_rank": "Room ADP Rank",
                "market_room_size": "ADP Room Candidates",
                "market_room_adp_gap": "Picks Behind Room Leader",
                "fair_adp": "Fair ADP",
                "actionable_adp": "Actionable ADP",
                "adp_avg": "Mock ADP",
                "value_gap": "Expected Pick Value",
                "edge_probability": "Probability Beat ADP %",
                "available_next_pick_probability": "Available Next Pick %",
                "forecast_points": "Forecast Points",
                "remaining_starter_need": "Remaining Starter Need",
            }
        )
        draft_display["Player"] = draft_display["Player"].str.title()
        for column in ["Probability Beat ADP %", "Available Next Pick %"]:
            draft_display[column] = (100 * draft_display[column]).round(0)
        st.dataframe(draft_display, hide_index=True, width="stretch")

        st.subheader("Team-position outlook")
        st.caption(
            "This view measures the destination team's recent production at each "
            "position. For each candidate, his own past production is removed from "
            "the teammate-environment signal—even when he stayed on the same team. "
            "The player and draft-board tables separately show how mock drafters "
            "rank each player against current same-team, same-position candidates."
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
                    "market_mae": "Market Point MAE",
                    "market_mae_lift": "Point MAE Improvement vs Market",
                    "player_only_mae": "Player-Only MAE",
                    "context_mae_lift": "Context MAE Improvement",
                    "rank_correlation": "Rank Correlation",
                }
            )
            for column in [
                "Point MAE",
                "Market Point MAE",
                "Point MAE Improvement vs Market",
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
            "finished ahead of that market rank. These seasons are the model's "
            "development evidence; 2026 is the first untouched prospective test."
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
            st.caption(
                "Segment results reveal whether pooled performance is concentrated "
                "in a particular position or part of the draft."
            )
            segment_display = value_segments.rename(
                columns={
                    "pos": "Position",
                    "market_tier": "Market Tier",
                    "bargains": "Bargains",
                    "hit_rate": "Hit Rate %",
                    "override_win_rate": "Override Win Rate %",
                    "average_actual_surplus": "Average Actual Rank Surplus",
                }
            )
            for column in ["Hit Rate %", "Override Win Rate %"]:
                segment_display[column] = (100 * segment_display[column]).round(1)
            segment_display["Market Tier"] = segment_display["Market Tier"].astype(str)
            segment_display["Average Actual Rank Surplus"] = segment_display[
                "Average Actual Rank Surplus"
            ].round(1)
            st.dataframe(segment_display, hide_index=True, width="stretch")

            st.subheader("Historical full-draft simulation")
            st.caption(
                "Paired snake-draft simulations use the same draft slot and sampled "
                "opponent behavior for each policy, then score the best realized "
                "starting lineup. The actionable policy keeps 75% of ADP and applies "
                "25% of the model adjustment. Pure model ranking is shown because it "
                "performed poorly and should not be used as a complete draft board."
            )
            simulation_overall = draft_simulation[
                draft_simulation["year"] == "Overall"
            ].iloc[0]
            simulation_metric_1, simulation_metric_2, simulation_metric_3 = st.columns(3)
            simulation_metric_1.metric(
                "Actionable-policy lineup lift",
                f'{simulation_overall["edge_policy_lift"]:+.1f} points',
            )
            simulation_metric_2.metric(
                "Actionable-policy win rate",
                f'{100 * simulation_overall["edge_policy_win_rate"]:.1f}%',
            )
            simulation_metric_3.metric(
                "Pure-model lineup lift",
                f'{simulation_overall["pure_model_lift"]:+.1f} points',
            )
            simulation_display = draft_simulation.rename(
                columns={
                    "year": "Season",
                    "simulations": "Simulations",
                    "rounds": "Rounds",
                    "model_weight": "Model Weight",
                    "edge_policy_lineup_points": "Actionable Lineup Points",
                    "pure_model_lineup_points": "Pure Model Lineup Points",
                    "market_lineup_points": "ADP Lineup Points",
                    "edge_policy_lift": "Actionable Lift",
                    "pure_model_lift": "Pure Model Lift",
                    "edge_policy_win_rate": "Actionable Win Rate %",
                    "pure_model_win_rate": "Pure Model Win Rate %",
                    "tie_rate": "Tie Rate %",
                }
            )
            simulation_display["Season"] = simulation_display["Season"].astype(str)
            for column in [
                "Actionable Win Rate %",
                "Pure Model Win Rate %",
                "Tie Rate %",
            ]:
                simulation_display[column] = (100 * simulation_display[column]).round(1)
            for column in [
                "Actionable Lineup Points",
                "Pure Model Lineup Points",
                "ADP Lineup Points",
                "Actionable Lift",
                "Pure Model Lift",
            ]:
                simulation_display[column] = simulation_display[column].round(1)
            st.dataframe(simulation_display, hide_index=True, width="stretch")
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

        The forecast uses ADP—including each player's hierarchy among same-team,
        same-position candidates—as a market baseline. Heavily regularized,
        position-specific models estimate its error from prior player ability, role,
        availability, and destination-team evidence. Mock ADP comes from Fantasy
        Football Calculator's human half-PPR drafts; results come from nflverse game
        data. Beat-ADP probabilities are calibrated only from chronological historical
        predictions.

        Data provenance and import commands are documented in `data/SOURCES.md`.
        """
    )
