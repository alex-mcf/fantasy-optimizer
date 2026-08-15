"""Point-in-draft recommendations built from forecast value and ADP uncertainty."""

from __future__ import annotations

from math import ceil
from statistics import NormalDist

import numpy as np
import pandas as pd

from fantasyoptimizer.config.league_config import DEFAULT_LEAGUE_CONFIG, LeagueConfig


def _starter_targets(config: LeagueConfig) -> dict[str, int]:
    flex = config.flex
    return {
        "QB": config.qb + config.superflex,
        "RB": config.rb + ceil(flex * 0.4),
        "WR": config.wr + ceil(flex * 0.5),
        "TE": config.te + ceil(flex * 0.1),
    }


def _available_next_pick(adp: float, deviation: float, next_pick: int) -> float:
    deviation = max(float(deviation), 4.0)
    return float(1 - NormalDist(mu=float(adp), sigma=deviation).cdf(next_pick))


def build_draft_recommendations(
    forecast: pd.DataFrame,
    current_pick: int,
    next_pick: int,
    roster_counts: dict[str, int] | None = None,
    drafted_players: set[str] | None = None,
    config: LeagueConfig = DEFAULT_LEAGUE_CONFIG,
) -> pd.DataFrame:
    """Rank available choices while accounting for whether each can be deferred.

    This is an auditable first-pass decision rule, not a claim of a solved draft.
    It combines fair value, historical edge probability, roster need, and the
    chance a player survives until the user's next selection.
    """
    if current_pick <= 0 or next_pick <= current_pick:
        raise ValueError("Next pick must be greater than the current pick.")
    roster_counts = roster_counts or {}
    drafted_players = drafted_players or set()
    board = forecast[~forecast["player"].isin(drafted_players)].copy()
    if board.empty:
        return board

    deviation = (
        board["stddev"]
        if "stddev" in board
        else pd.Series(12.0, index=board.index)
    ).fillna(12.0)
    board["available_next_pick_probability"] = [
        _available_next_pick(adp, spread, next_pick)
        for adp, spread in zip(board["adp_avg"], deviation)
    ]
    board["drafted_before_next_probability"] = (
        1 - board["available_next_pick_probability"]
    )

    targets = _starter_targets(config)
    board["remaining_starter_need"] = board["pos"].map(
        lambda position: max(
            targets.get(position, 0) - int(roster_counts.get(position, 0)), 0
        )
    )
    edge_probability = (
        board["edge_probability"]
        if "edge_probability" in board
        else pd.Series(0.5, index=board.index)
    ).fillna(0.5)
    board["decision_score"] = (
        board["model_value"]
        + 0.35 * board["value_gap"]
        + 12 * (edge_probability - 0.5)
        + 6 * board["remaining_starter_need"]
        + 8 * board["drafted_before_next_probability"]
    )

    urgent = board["available_next_pick_probability"] < 0.35
    strong_edge = board["value_gap"] >= config.league_size
    fair_now = board["fair_adp"] <= current_pick
    fair_before_next = board["fair_adp"] < next_pick
    board["recommendation"] = np.select(
        [
            (fair_now | strong_edge) & urgent,
            strong_edge & ~urgent,
            fair_before_next & urgent,
            board["fair_adp"] >= next_pick,
        ],
        ["Draft now", "Target — may wait", "Consider now", "Wait"],
        default="Pass at this price",
    )
    order = {
        "Draft now": 0,
        "Consider now": 1,
        "Target — may wait": 2,
        "Wait": 3,
        "Pass at this price": 4,
    }
    board["recommendation_order"] = board["recommendation"].map(order)
    return board.sort_values(
        ["recommendation_order", "decision_score", "model_rank"],
        ascending=[True, False, True],
        ignore_index=True,
    )
