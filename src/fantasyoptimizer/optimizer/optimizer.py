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
    decision_rank = (
        board["actionable_adp"] if "actionable_adp" in board else board["fair_adp"]
    )
    fair_now = decision_rank <= current_pick
    fair_before_next = decision_rank < next_pick
    board["recommendation"] = np.select(
        [
            (fair_now | strong_edge) & urgent,
            strong_edge & ~urgent,
            fair_before_next & urgent,
            decision_rank >= next_pick,
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


def _position_caps(config: LeagueConfig) -> dict[str, int]:
    return {
        "QB": max(config.qb + config.superflex + 1, 2),
        "RB": config.rb + config.flex + 3,
        "WR": config.wr + config.flex + 3,
        "TE": max(config.te + 1, 2),
    }


def _team_for_pick(overall_pick: int, league_size: int) -> int:
    round_index, within_round = divmod(overall_pick, league_size)
    return within_round if round_index % 2 == 0 else league_size - within_round - 1


def _select_candidate(
    available: set[int],
    positions: list[str],
    order: np.ndarray,
    roster_counts: dict[str, int],
    caps: dict[str, int],
) -> int | None:
    for index in order:
        candidate = int(index)
        if (
            candidate in available
            and roster_counts.get(positions[candidate], 0) < caps[positions[candidate]]
        ):
            return candidate
    return None


def _realized_lineup_points(
    roster: list[int],
    positions: list[str],
    actual_points: np.ndarray,
    config: LeagueConfig,
) -> float:
    remaining = set(roster)

    def take(position: str, count: int) -> list[int]:
        candidates = sorted(
            (index for index in remaining if positions[index] == position),
            key=lambda index: actual_points[index],
            reverse=True,
        )[:count]
        remaining.difference_update(candidates)
        return candidates

    starters: list[int] = []
    starters.extend(take("QB", config.qb))
    starters.extend(take("RB", config.rb))
    starters.extend(take("WR", config.wr))
    starters.extend(take("TE", config.te))
    flex_candidates = sorted(
        (
            index
            for index in remaining
            if positions[index] in {"RB", "WR", "TE"}
        ),
        key=lambda index: actual_points[index],
        reverse=True,
    )[: config.flex]
    starters.extend(flex_candidates)
    remaining.difference_update(flex_candidates)
    superflex_candidates = sorted(
        remaining,
        key=lambda index: actual_points[index],
        reverse=True,
    )[: config.superflex]
    starters.extend(superflex_candidates)
    return float(actual_points[starters].sum()) if starters else 0.0


def _simulate_one_draft(
    frame: pd.DataFrame,
    strategy_rank: str,
    user_team: int,
    rounds: int,
    market_draw: np.ndarray,
    config: LeagueConfig,
) -> float:
    positions = frame["pos"].tolist()
    actual_points = frame["actual_points"].to_numpy(dtype=float)
    strategy_order = np.argsort(frame[strategy_rank].to_numpy(dtype=float))
    market_order = np.argsort(market_draw)
    available = set(range(len(frame)))
    caps = _position_caps(config)
    rosters: list[list[int]] = [[] for _ in range(config.league_size)]
    counts: list[dict[str, int]] = [dict() for _ in range(config.league_size)]
    total_picks = min(rounds * config.league_size, len(frame))
    for overall_pick in range(total_picks):
        team = _team_for_pick(overall_pick, config.league_size)
        order = strategy_order if team == user_team else market_order
        selected = _select_candidate(
            available, positions, order, counts[team], caps
        )
        if selected is None:
            continue
        available.remove(selected)
        rosters[team].append(selected)
        position = positions[selected]
        counts[team][position] = counts[team].get(position, 0) + 1
    return _realized_lineup_points(
        rosters[user_team], positions, actual_points, config
    )


def simulate_historical_draft_strategies(
    historical_players: pd.DataFrame,
    config: LeagueConfig = DEFAULT_LEAGUE_CONFIG,
    simulations_per_year: int = 250,
    max_rounds: int = 10,
    model_weight: float = 0.25,
    seed: int = 20260815,
) -> pd.DataFrame:
    """Compare conservative-edge, pure-model, and ADP draft policies.

    Opponent selections are sampled around historical ADP. Each paired simulation
    gives every policy the same draft slot and market draw, then scores the best
    realized starting lineup. The edge policy gives ADP 75% weight by default,
    preventing useful disagreement signals from replacing the market wholesale.
    This omits waivers and weekly start/sit decisions.
    """
    required = {
        "target_year",
        "pos",
        "market_rank",
        "predicted_rank",
        "actual_points",
    }
    if historical_players.empty or not required.issubset(historical_players.columns):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    all_edge: list[float] = []
    all_pure: list[float] = []
    all_market: list[float] = []
    for year, season in historical_players.groupby("target_year", sort=True):
        frame = season.reset_index(drop=True)
        frame["edge_policy_rank"] = (
            (1 - model_weight) * frame["market_rank"]
            + model_weight * frame["predicted_rank"]
        )
        rounds = min(max_rounds, len(frame) // config.league_size)
        if rounds <= 0:
            continue
        market_rank = frame["market_rank"].to_numpy(dtype=float)
        spread = np.maximum(4.0, 0.12 * market_rank)
        edge_scores: list[float] = []
        pure_scores: list[float] = []
        market_scores: list[float] = []
        for _ in range(simulations_per_year):
            user_team = int(rng.integers(0, config.league_size))
            market_draw = market_rank + rng.normal(0, spread)
            edge_scores.append(
                _simulate_one_draft(
                    frame,
                    "edge_policy_rank",
                    user_team,
                    rounds,
                    market_draw,
                    config,
                )
            )
            pure_scores.append(
                _simulate_one_draft(
                    frame,
                    "predicted_rank",
                    user_team,
                    rounds,
                    market_draw,
                    config,
                )
            )
            market_scores.append(
                _simulate_one_draft(
                    frame,
                    "market_rank",
                    user_team,
                    rounds,
                    market_draw,
                    config,
                )
            )
        edge_values = np.asarray(edge_scores)
        pure_values = np.asarray(pure_scores)
        market_values = np.asarray(market_scores)
        all_edge.extend(edge_scores)
        all_pure.extend(pure_scores)
        all_market.extend(market_scores)
        rows.append(
            {
                "year": int(year),
                "simulations": simulations_per_year,
                "rounds": rounds,
                "model_weight": model_weight,
                "edge_policy_lineup_points": float(edge_values.mean()),
                "pure_model_lineup_points": float(pure_values.mean()),
                "market_lineup_points": float(market_values.mean()),
                "edge_policy_lift": float((edge_values - market_values).mean()),
                "pure_model_lift": float((pure_values - market_values).mean()),
                "edge_policy_win_rate": float((edge_values > market_values).mean()),
                "pure_model_win_rate": float((pure_values > market_values).mean()),
                "tie_rate": float((edge_values == market_values).mean()),
            }
        )
    if rows:
        edge_values = np.asarray(all_edge)
        pure_values = np.asarray(all_pure)
        market_values = np.asarray(all_market)
        rows.append(
            {
                "year": "Overall",
                "simulations": len(edge_values),
                "rounds": np.nan,
                "model_weight": model_weight,
                "edge_policy_lineup_points": float(edge_values.mean()),
                "pure_model_lineup_points": float(pure_values.mean()),
                "market_lineup_points": float(market_values.mean()),
                "edge_policy_lift": float((edge_values - market_values).mean()),
                "pure_model_lift": float((pure_values - market_values).mean()),
                "edge_policy_win_rate": float((edge_values > market_values).mean()),
                "pure_model_win_rate": float((pure_values > market_values).mean()),
                "tie_rate": float((edge_values == market_values).mean()),
            }
        )
    return pd.DataFrame(rows)
