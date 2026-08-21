"""Draft-day board: one filterable player list, plus the tools you use at a pick.

The forecast is a second opinion on a mock-draft ADP, so the app is arranged the
way that opinion gets used: scan a list, filter it to the players you can
actually take, and check the evidence when a call looks surprising. Everything
that justifies the board lives behind the Evidence tab rather than in front of
it.
"""

import numpy as np
import pandas as pd
import streamlit as st

from fantasyoptimizer.config.league_config import LeagueConfig
from fantasyoptimizer.forecasting.forecaster import (
    MODEL_PROMOTION_CAP_ROUNDS,
    backtest_draft_value,
    backtest_forecaster,
    build_training_examples,
    build_team_position_outlook,
    calibrate_edge_probabilities,
    forecast_season,
    segment_draft_value,
)
from fantasyoptimizer.market import (
    PLATFORM_OPTIONS,
    apply_platform_adp,
    parse_platform_adp,
    unmatched_platform_players,
)
from fantasyoptimizer.optimizer import (
    build_draft_recommendations,
    policy_blend_weights,
    simulate_historical_draft_strategies,
    snake_pick_numbers,
)
from fantasyoptimizer.scoring.scoring_engine import SUPPORTED_POSITIONS
from fantasyoptimizer.utils.data_loader import (
    available_adp_snapshots,
    available_adp_years,
    available_years,
    build_adp_movement,
    data_quality_report,
    load_adp_metadata,
)

st.set_page_config(page_title="Draft Board", layout="wide")

CONFIDENCE_ORDER = ["High", "Medium", "Low", "Rookie / no NFL history"]

# Column groups, in the order a draft-day scan wants them. "Decision" is what the
# board shows by default; the rest are opt-in, because a 48-column table is not a
# thing anyone reads under a pick clock.
DECISION_COLUMNS = {
    "player": "Player",
    "pos": "Pos",
    "team": "Team",
    "draft_adp": "ADP",
    "platform_actionable_adp": "Take At",
    "rounds_moved": "Rounds Moved",
    "edge_probability": "Beat ADP %",
    "confidence": "Confidence",
}
PROJECTION_COLUMNS = {
    "forecast_points": "Proj Points",
    "forecast_ppg": "Proj PPG",
    "forecast_games": "Proj Games",
    "model_value": "VORP",
    "position_rank": "Pos Rank",
}
EVIDENCE_COLUMNS = {
    "weighted_ppg": "Prior PPG",
    "weighted_ppo": "Prior Pts/Opp",
    "weighted_opportunity_share": "Prior Opp Share %",
    "official_depth_rank": "Depth Rank",
    "official_starter": "Listed Starter",
    "market_room_rank": "Room ADP Rank",
    "market_room_size": "Room Size",
    "age": "Age",
    "experience": "Exp",
}
EXTRA_COLUMNS = {
    "model_rank": "Model Rank",
    "uncapped_model_rank": "Model Rank (Pre-Guard)",
    "fair_adp": "Fair ADP",
    "adp_avg": "FFC ADP",
    "market_rank": "Market Rank",
    "platform_value_gap": "Value Gap",
    "market_points": "Market-Implied Points",
    "market_adjustment": "Predicted Market Error",
    "team_context_rank": "Team Context Rank",
    "role_agreement": "Market / Depth Agreement",
    "official_roster_status": "Roster Status",
    "changed_team": "Changed Team",
    "draft_round": "NFL Draft Round",
    "draft_pick": "NFL Draft Pick",
    "timesdrafted": "Mock Samples",
    "edge_confidence": "Edge Confidence",
    "platform_source": "ADP Source",
}
PERCENT_COLUMNS = {"Beat ADP %", "Prior Opp Share %", "Available Next Pick %"}


@st.cache_data
def load_training(target_year: int):
    """Cache the league-independent work so roster changes stay cheap.

    Building training examples and the point backtest is most of the pipeline's
    runtime and none of it depends on the league configuration, so it is keyed on
    the target year alone.
    """
    training = build_training_examples(target_year - 1)
    backtest = backtest_forecaster(target_year - 1, training_examples=training)
    return training, backtest


@st.cache_data
def load_forecast(target_year: int, config: LeagueConfig):
    training, backtest = load_training(target_year)
    value_backtest, value_players = backtest_draft_value(
        target_year - 1, config, training_examples=training
    )
    model_weight, season_weights = policy_blend_weights(value_players, config)
    forecast = forecast_season(
        target_year, config, training_examples=training, model_weight=model_weight
    )
    forecast = calibrate_edge_probabilities(forecast, value_players, config)
    segments = segment_draft_value(value_players, config)
    draft_simulation = simulate_historical_draft_strategies(
        value_players, config, season_weights=season_weights
    )
    return (
        forecast,
        backtest,
        value_backtest,
        value_players,
        segments,
        draft_simulation,
    )


def display_table(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Rename and tidy a slice of the board for display."""
    present = {key: label for key, label in columns.items() if key in frame}
    table = frame[list(present)].rename(columns=present)
    if "Player" in table:
        table["Player"] = table["Player"].astype(str).str.title()
    for column in PERCENT_COLUMNS & set(table.columns):
        table[column] = (100 * table[column]).round(0)
    return table


year_options = available_years()
forecast_years = sorted(set(available_adp_years()) - set(year_options))
if not year_options or not forecast_years:
    st.error(
        "No complete seasons or no upcoming-season ADP was found. Follow the data "
        "setup steps in README.md."
    )
    st.stop()
target_year = max(forecast_years)

with st.sidebar:
    st.subheader("League")
    platform = st.selectbox("Draft platform", PLATFORM_OPTIONS)
    league_size = int(st.number_input("Teams", min_value=4, max_value=20, value=12))
    draft_slot = int(
        st.number_input("Your draft slot", min_value=1, max_value=league_size, value=1)
    )
    draft_rounds = int(
        st.number_input("Draft rounds", min_value=5, max_value=30, value=15)
    )

    with st.expander("Starting lineup", expanded=False):
        qb = int(st.number_input("QB", min_value=0, max_value=3, value=1))
        rb = int(st.number_input("RB", min_value=0, max_value=5, value=2))
        wr = int(st.number_input("WR", min_value=0, max_value=5, value=2))
        te = int(st.number_input("TE", min_value=0, max_value=3, value=1))
        flex = int(st.number_input("FLEX", min_value=0, max_value=4, value=1))
        superflex = int(st.number_input("Superflex", min_value=0, max_value=2, value=0))
        st.caption("Kickers and defenses are not modeled. Half-PPR only.")

    platform_adp = None
    if platform != PLATFORM_OPTIONS[0]:
        with st.expander(f"{platform} rankings", expanded=True):
            st.caption(
                "Upload your platform's export so prices and availability match "
                "your room. Unmatched players fall back to the FFC market."
            )
            platform_file = st.file_uploader(
                "Rankings CSV", type=["csv"], key="platform_adp"
            )
            st.download_button(
                "Template",
                "Player,Position,ADP\nExample Player,RB,24.5\n",
                file_name="platform_adp_template.csv",
                mime="text/csv",
            )
            if platform_file is not None:
                try:
                    platform_adp = parse_platform_adp(pd.read_csv(platform_file))
                    st.success(f"Loaded {len(platform_adp)} rankings.")
                except (ValueError, pd.errors.ParserError) as error:
                    st.error(str(error))

league_config = LeagueConfig(
    league_size=league_size,
    qb=qb,
    rb=rb,
    wr=wr,
    te=te,
    flex=flex,
    superflex=superflex,
)
with st.sidebar:
    replacements = league_config.replacement_ranks()
    st.caption(
        "Replacement level: "
        + ", ".join(f"{position}{rank}" for position, rank in replacements.items())
    )
    if league_size != 12:
        st.caption(
            "Replacement value follows this league size, but the historical FFC "
            "market baseline is from 12-team drafts."
        )
    if superflex > 0 and platform == PLATFORM_OPTIONS[0]:
        st.caption(
            "FFC baseline ADP is not superflex ADP. Upload your platform's "
            "rankings for useful superflex prices."
        )

with st.spinner(f"Training on prior seasons and forecasting {target_year}..."):
    (
        forecast,
        backtest,
        value_backtest,
        value_players,
        value_segments,
        draft_simulation,
    ) = load_forecast(target_year, league_config)

unmatched_uploads = unmatched_platform_players(forecast, platform_adp)
forecast = apply_platform_adp(forecast, platform_adp, platform)
forecast["rounds_moved"] = (
    (forecast["draft_market_rank"] - forecast["model_rank"]) / league_size
).round(1)
promotion_limit = MODEL_PROMOTION_CAP_ROUNDS * league_size
forecast["guarded"] = (
    forecast["market_rank"] - forecast["uncapped_model_rank"] > promotion_limit
)
blend_weight = float(forecast["blend_model_weight"].iloc[0])
adp_metadata = load_adp_metadata(target_year)

st.title(f"{target_year} Draft Board")
st.caption(
    f"Half-PPR · {league_size}-team · pick {draft_slot} · {platform} prices · "
    f"ADP fetched {str(adp_metadata.get('fetched_at_utc', 'unknown'))[:10]}"
)

board_tab, draft_tab, evidence_tab = st.tabs(["Board", "Draft room", "Evidence"])

with board_tab:
    drafted = set(st.session_state.get("drafted_players", []))
    filter_row = st.columns([2, 3])
    search = filter_row[0].text_input(
        "Search", placeholder="Search players", label_visibility="collapsed"
    )
    positions = filter_row[1].pills(
        "Positions",
        SUPPORTED_POSITIONS,
        selection_mode="multi",
        default=list(SUPPORTED_POSITIONS),
        label_visibility="collapsed",
    )
    call = st.pills(
        "Call",
        [
            "Every player",
            "Values 2+ rounds",
            "Values 1+ round",
            "Fades",
            "Guard held back",
        ],
        default="Every player",
        label_visibility="collapsed",
    )

    with st.expander("More filters"):
        more = st.columns([2, 2, 2])
        confidence = more[0].multiselect(
            "Confidence", CONFIDENCE_ORDER, default=CONFIDENCE_ORDER
        )
        teams = more[1].multiselect(
            "Team", sorted(forecast["team"].dropna().astype(str).unique())
        )
        last_round = int(np.ceil(forecast["draft_adp"].max() / league_size))
        rounds = more[2].slider(
            "Going in rounds", 1, max(last_round, 2), (1, max(last_round, 2))
        )
        hide_drafted = st.checkbox(
            "Hide players marked drafted in the draft room", value=True
        )

    board = forecast.copy()
    if search:
        board = board[board["player"].str.contains(search, case=False, na=False)]
    if positions:
        board = board[board["pos"].isin(positions)]
    if confidence:
        board = board[board["confidence"].isin(confidence)]
    if teams:
        board = board[board["team"].astype(str).isin(teams)]
    board = board[
        board["draft_adp"].between(
            (rounds[0] - 1) * league_size + 1, rounds[1] * league_size
        )
    ]
    if hide_drafted and drafted:
        board = board[~board["player"].isin(drafted)]
    if call == "Values 2+ rounds":
        board = board[board["platform_value_gap"] >= 2 * league_size]
    elif call == "Values 1+ round":
        board = board[board["platform_value_gap"] >= league_size]
    elif call == "Fades":
        board = board[board["platform_value_gap"] <= -league_size]
    elif call == "Guard held back":
        board = board[board["guarded"]]
    board = board.sort_values("platform_actionable_adp", ignore_index=True)

    summary = st.columns(4)
    summary[0].metric("Players shown", len(board))
    summary[1].metric(
        "Values ≥2 rounds",
        int((board["platform_value_gap"] >= 2 * league_size).sum()),
    )
    summary[2].metric("Model weight", f"{blend_weight:.0%}")
    summary[3].metric(
        "Platform coverage", f"{int(forecast['platform_match'].sum())}/{len(forecast)}"
    )

    detail = st.segmented_control(
        "Detail",
        ["Decision", "+ Projection", "+ Evidence", "Everything"],
        default="Decision",
        label_visibility="collapsed",
    )
    columns = dict(DECISION_COLUMNS)
    if detail in {"+ Projection", "Everything"}:
        columns.update(PROJECTION_COLUMNS)
    if detail in {"+ Evidence", "Everything"}:
        columns.update(EVIDENCE_COLUMNS)
    if detail == "Everything":
        columns.update(EXTRA_COLUMNS)

    if board.empty:
        st.info("No players match these filters.")
    else:
        selection = st.dataframe(
            display_table(board, columns),
            hide_index=True,
            width="stretch",
            height=560,
            on_select="rerun",
            selection_mode="single-row",
        )
        st.caption(
            "**Take At** blends market rank with the model at the fitted weight — "
            "it is where this player is worth drafting, not where he will go. "
            "**Rounds Moved** is how far the model disagrees with ADP. Two rounds "
            "is the threshold worth acting on: one-round calls beat their ADP 49% "
            "of the time against a 42% base rate, two-round calls 63%, and "
            "four-plus 90%. Select a row for the reasoning behind a call."
        )
        chosen = selection.selection.rows if selection and selection.selection else []
        if chosen:
            player = board.iloc[chosen[0]]
            st.divider()
            headline = st.columns([3, 1, 1, 1, 1])
            headline[0].markdown(
                f"### {str(player['player']).title()}  \n"
                f"{player['pos']} · {player['team']} · {player['confidence']}"
            )
            headline[1].metric("ADP", f"{player['draft_adp']:.0f}")
            headline[2].metric("Take at", int(player["platform_actionable_adp"]))
            headline[3].metric(
                "Rounds moved", f"{player['rounds_moved']:+.1f}"
            )
            headline[4].metric(
                "Beat ADP", f"{100 * player['edge_probability']:.0f}%"
            )
            why = st.columns(2)
            why[0].markdown(
                f"**Projection** {player['forecast_points']:.0f} points "
                f"({player['forecast_ppg']:.1f} per game over "
                f"{player['forecast_games']:.1f} games), VORP "
                f"{player['model_value']:.0f}, {player['pos']}"
                f"{int(player['position_rank'])} on this board.  \n"
                f"**Market says** {player['market_points']:.0f} points at this ADP; "
                f"the model adds {player['market_adjustment']:+.0f}."
            )
            prior = (
                f"{player['weighted_ppg']:.1f} PPG and "
                f"{player['weighted_ppo']:.2f} points per opportunity across "
                f"{int(player['history_seasons'])} prior season(s)"
                if player["history_seasons"]
                else "no NFL history — the market baseline is doing the work"
            )
            role = (
                f"listed {int(player['official_depth_rank'])} on the depth chart"
                if player.get("official_role_known")
                and pd.notna(player.get("official_depth_rank"))
                else "no official depth-chart data"
            )
            why[1].markdown(
                f"**Prior form** {prior}.  \n"
                f"**Role** {role}; mock drafters rank him "
                f"{int(player['market_room_rank'])} of "
                f"{int(player['market_room_size'])} at his position on this team."
            )
            if player["guarded"]:
                st.warning(
                    f"The model had him at {int(player['uncapped_model_rank'])} "
                    f"overall — more than {MODEL_PROMOTION_CAP_ROUNDS} rounds ahead "
                    f"of the market — and the guard pulled him back to "
                    f"{int(player['model_rank'])}.",
                    icon="🛑",
                )

    st.download_button(
        f"Download {target_year} board",
        display_table(
            board, {**DECISION_COLUMNS, **PROJECTION_COLUMNS, **EVIDENCE_COLUMNS, **EXTRA_COLUMNS}
        ).to_csv(index=False),
        file_name=f"{target_year}_draft_board.csv",
        mime="text/csv",
    )

    if not unmatched_uploads.empty:
        with st.expander(
            f"⚠️ {len(unmatched_uploads)} uploaded {platform} rankings matched no "
            "modeled player"
        ):
            st.dataframe(unmatched_uploads, hide_index=True, width="stretch")

with draft_tab:
    st.caption(
        "Set where you are in the draft, mark what is gone, and the board becomes "
        "a shortlist. Availability is an estimate from mock-draft variability, "
        "not a guarantee."
    )
    pick_row = st.columns([1, 1, 2])
    manual = pick_row[0].toggle("Enter pick numbers", value=False)
    if manual:
        current_pick = int(
            pick_row[1].number_input("Current overall pick", min_value=1, value=1)
        )
        next_pick = int(
            pick_row[2].number_input(
                "Your next overall pick",
                min_value=current_pick + 1,
                value=max(current_pick + 1, 2 * league_size),
            )
        )
    else:
        current_round = int(
            pick_row[1].number_input(
                "Your round", min_value=1, max_value=draft_rounds, value=1
            )
        )
        snake_picks = snake_pick_numbers(league_size, draft_slot, draft_rounds)
        current_pick = snake_picks[current_round - 1]
        next_pick = (
            snake_picks[current_round]
            if current_round < len(snake_picks)
            else current_pick + league_size
        )
        pick_row[2].markdown(
            f"### Pick {current_pick}  \nNext pick {next_pick}"
        )

    st.multiselect(
        "Players already drafted",
        options=forecast["player"].tolist(),
        format_func=lambda player: str(player).title(),
        key="drafted_players",
    )
    st.caption("Your roster so far")
    roster_cols = st.columns(len(SUPPORTED_POSITIONS))
    roster_counts = {
        position: int(
            column.number_input(
                position, min_value=0, max_value=10, value=0, key=f"roster_{position}"
            )
        )
        for position, column in zip(SUPPORTED_POSITIONS, roster_cols)
    }

    recommendations = build_draft_recommendations(
        forecast,
        current_pick,
        next_pick,
        roster_counts,
        set(st.session_state.get("drafted_players", [])),
        league_config,
    )
    if recommendations.empty:
        st.info("Every modeled player has been marked drafted.")
    else:
        draft_columns = {
            "recommendation": "Call",
            "player": "Player",
            "pos": "Pos",
            "team": "Team",
            "draft_adp": "ADP",
            "platform_actionable_adp": "Take At",
            "available_next_pick_probability": "Available Next Pick %",
            "edge_probability": "Beat ADP %",
            "remaining_starter_need": "Starters Needed",
            "forecast_points": "Proj Points",
        }
        st.dataframe(
            display_table(recommendations.head(25), draft_columns),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "**Draft now** means fair value that probably will not survive to your "
            "next pick. **Target — may wait** is a value the model expects to still "
            "be there. This is an auditable heuristic, not a solved draft."
        )

with evidence_tab:
    st.caption(
        "Every season below was forecast by a model trained only on earlier "
        f"seasons. {target_year} is the first untouched prospective test."
    )
    if not draft_simulation.empty:
        overall = draft_simulation.iloc[-1]
        headline = st.columns(4)
        headline[0].metric(
            "Policy lift vs ADP", f"{overall['edge_policy_lift']:+.0f} pts"
        )
        headline[1].metric(
            "Seasons it won", f"{100 * overall['edge_policy_win_rate']:.0f}%"
        )
        headline[2].metric(
            "Model board alone", f"{overall['pure_model_lift']:+.0f} pts"
        )
        if not value_backtest.empty:
            headline[3].metric(
                "Override win rate",
                f"{100 * value_backtest.iloc[-1]['bargain_override_win_rate']:.0f}%",
            )

    st.markdown(
        """
**How to read a disagreement.** Size, position, and draft stage all matter:

| Model moves him | Beat their ADP | Mean rank surplus |
|---|---|---|
| Within a round (base rate) | 42% | −8 |
| 1 round | 49% | −4 |
| 2 rounds | 63% | +14 |
| 3 rounds | 69% | +28 |
| 4+ rounds | 90% | +51 |

Tight-end values after round 7 beat their ADP in 89–95% of cases. Quarterbacks
promoted into the first three rounds are the model's worst category. Run
`python scripts/evaluate_adp_comparison.py` for the full breakdown, including
whether a call survives being re-priced on a different mock market.
        """
    )

    with st.expander("Draft simulation by season"):
        simulation_columns = {
            "year": "Season",
            "model_weight": "Model Weight",
            "edge_policy_lift": "Policy Lift",
            "edge_policy_win_rate": "Policy Win Rate",
            "pure_model_lift": "Model-Only Lift",
            "market_lineup_points": "ADP Lineup Points",
            "simulations": "Drafts",
        }
        simulation_display = draft_simulation.rename(columns=simulation_columns)[
            list(simulation_columns.values())
        ]
        simulation_display["Season"] = simulation_display["Season"].astype(str)
        simulation_display["Policy Win Rate"] = (
            100 * simulation_display["Policy Win Rate"]
        ).round(0)
        st.dataframe(
            simulation_display.round(1), hide_index=True, width="stretch"
        )
        st.caption(
            "Paired snake drafts: every policy gets the same slot and the same "
            "sampled opponent behavior, then the best realized starting lineup is "
            "scored. Each season's model weight was fitted only on earlier seasons."
        )

    with st.expander("Did the model's values beat ADP?"):
        value_columns = {
            "year": "Season",
            "bargains": "Values Flagged",
            "bargain_hit_rate": "Beat ADP %",
            "non_bargain_hit_rate": "Everyone Else %",
            "round_adjusted_hit_lift": "Same-Round Lift",
            "bargain_override_win_rate": "Override Win Rate %",
            "market_rank_mae": "ADP Rank Error",
            "model_rank_mae": "Model Rank Error",
            "top_board_market_drift": "Top-60 Drift",
            "context_rank_mae_lift": "Context Would Add",
        }
        value_display = value_backtest.rename(columns=value_columns)[
            list(value_columns.values())
        ]
        value_display["Season"] = value_display["Season"].astype(str)
        for column in ["Beat ADP %", "Everyone Else %", "Override Win Rate %"]:
            value_display[column] = (100 * value_display[column]).round(0)
        st.dataframe(value_display.round(2), hide_index=True, width="stretch")
        st.caption(
            "A value is a player the model ranks at least one round ahead of "
            "normalized market ADP. **Top-60 Drift** is how far the board moves the "
            "early rounds off the market — the metric that would have caught this "
            "model's one real failure. **Context Would Add** scores a model that "
            "also uses team and role features; it stays near zero, which is why "
            "those features are measured but not trained on."
        )
        st.dataframe(
            value_segments.rename(
                columns={
                    "pos": "Pos",
                    "market_tier": "Market Tier",
                    "bargains": "Values",
                    "hit_rate": "Beat ADP %",
                    "override_win_rate": "Override Win Rate %",
                    "average_actual_surplus": "Mean Rank Surplus",
                }
            ).round(2),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download historical player-level results",
            value_players.to_csv(index=False),
            file_name="historical_value_backtest.csv",
            mime="text/csv",
        )

    with st.expander("Point-forecast accuracy"):
        point_columns = {
            "year": "Season",
            "players": "Players",
            "mae": "Model Error",
            "market_mae": "ADP Error",
            "market_mae_lift": "Model Improvement",
            "context_model_mae": "Context-Model Error",
            "rank_correlation": "Rank Correlation",
        }
        st.dataframe(
            backtest.rename(columns=point_columns)[list(point_columns.values())].round(2),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Point accuracy is not where this model earns its keep — it is close to "
            "flat against ADP. The edge is in rank ordering and in which players it "
            "flags, which the tables above measure."
        )

    with st.expander("Team and role context (not a model input)"):
        outlook = build_team_position_outlook(forecast)
        outlook_position = st.selectbox("Position", list(SUPPORTED_POSITIONS))
        outlook_columns = {
            "team": "Team",
            "pos": "Pos",
            "context_rank": "Context Rank",
            "weighted_room_opportunities": "3-Yr Room Opportunities",
            "last_year_room_points": "Last-Year Room Points",
            "last_year_other_points": "Last-Year Teammate Points",
            "model_favorite": "Model Favorite",
            "favorite_model_rank": "Favorite Model Rank",
            "candidates": "Candidates",
        }
        outlook_display = display_table(
            outlook[outlook["pos"] == outlook_position].sort_values("context_rank"),
            outlook_columns,
        )
        for column in ("Model Favorite", "Candidates"):
            outlook_display[column] = outlook_display[column].astype(str).str.title()
        st.dataframe(outlook_display.round(1), hide_index=True, width="stretch")
        st.caption(
            "The destination team's recent production at each position, with the "
            "candidate's own past output removed from the teammate signal. Shown "
            "for your judgement: a context-augmented model is scored every backtest "
            "window and does not rank better, so the forecast does not train on it."
        )

    with st.expander("ADP movement and data quality"):
        movement = build_adp_movement(target_year)
        snapshots = len(available_adp_snapshots(target_year))
        if movement.empty:
            st.info(
                f"{snapshots} ADP snapshot preserved. Re-run the ADP importer on "
                "another day to start measuring risers and fallers."
            )
        else:
            st.dataframe(
                display_table(
                    movement,
                    {
                        "player": "Player",
                        "pos": "Pos",
                        "team": "Team",
                        "first_adp": "First ADP",
                        "latest_adp": "Latest ADP",
                        "adp_movement": "Picks Risen",
                        "snapshots": "Snapshots",
                    },
                ),
                hide_index=True,
                width="stretch",
            )
        quality = data_quality_report(year_options)
        quality["adp_match_rate"] = quality["adp_match_rate"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(quality, hide_index=True, width="stretch")
        st.caption(
            "Only matched player-position records can be scored. Sources, scoring, "
            "and import commands are documented in `data/SOURCES.md`."
        )

    st.markdown(
        f"""
**Method in one paragraph.** ADP is the baseline, not the enemy: position-specific
ridge models estimate the points implied by a player's mock-draft price, then a
deliberately narrow model — eleven features of the player's own recent form —
predicts where that price is wrong. Its penalty is fitted per position by holding
out whole seasons inside the training window. League replacement levels turn the
adjusted forecast into VORP and a fair rank, no player may be ranked more than
{MODEL_PROMOTION_CAP_ROUNDS} rounds ahead of the market, and the board you draft
from blends market rank with model rank at a weight fitted on simulated draft
outcomes ({blend_weight:.0%} model today). Beat-ADP probabilities are calibrated
only from chronological out-of-sample predictions.
        """
    )
