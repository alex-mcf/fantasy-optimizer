"""Point-in-draft recommendations built from forecast value and ADP uncertainty."""

from __future__ import annotations

from math import ceil
from statistics import NormalDist

import numpy as np
import pandas as pd

from fantasyoptimizer.config.league_config import DEFAULT_LEAGUE_CONFIG, LeagueConfig
from fantasyoptimizer.forecasting.forecaster import DEFAULT_MODEL_BLEND_WEIGHT

# Candidate model shares of the market/model blend. The weight is fitted on
# realized draft outcomes rather than on rank error: rank error keeps improving
# as the model takes over the board, while the drafts those boards produce get
# sharply worse, so rank error selects a policy that loses.
POLICY_BLEND_WEIGHT_GRID = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)
POLICY_REQUIRED_COLUMNS = frozenset(
    {"target_year", "pos", "market_rank", "predicted_rank", "actual_points"}
)


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

    adp_column = "draft_adp" if "draft_adp" in board else "adp_avg"
    deviation_column = (
        "draft_adp_stddev" if "draft_adp_stddev" in board else "stddev"
    )
    deviation = board.get(
        deviation_column, pd.Series(12.0, index=board.index)
    ).fillna(12.0)
    board["available_next_pick_probability"] = [
        _available_next_pick(adp, spread, next_pick)
        for adp, spread in zip(board[adp_column], deviation)
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
    value_gap = board.get("platform_value_gap", board["value_gap"])
    board["decision_score"] = (
        board["model_value"]
        + 0.35 * value_gap
        + 12 * (edge_probability - 0.5)
        + 6 * board["remaining_starter_need"]
        + 8 * board["drafted_before_next_probability"]
    )

    urgent = board["available_next_pick_probability"] < 0.35
    strong_edge = value_gap >= config.league_size
    decision_rank = (
        board["platform_actionable_adp"]
        if "platform_actionable_adp" in board
        else board.get("actionable_adp", board["fair_adp"])
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


def snake_pick_numbers(
    league_size: int, draft_slot: int, rounds: int
) -> list[int]:
    """Return a team's one-indexed overall selections in a snake draft."""
    if league_size <= 0 or not 1 <= draft_slot <= league_size or rounds <= 0:
        raise ValueError("League size, draft slot, and rounds must be positive.")
    return [
        (round_number - 1) * league_size + draft_slot
        if round_number % 2 == 1
        else round_number * league_size - draft_slot + 1
        for round_number in range(1, rounds + 1)
    ]


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


def _policy_weight_points_by_season(
    historical_players: pd.DataFrame,
    config: LeagueConfig,
    grid: tuple[float, ...],
    simulations_per_year: int,
    max_rounds: int,
    seed: int,
) -> dict[int, dict[float, np.ndarray]]:
    """Score every candidate weight in every season on shared market draws.

    Each season's candidates face the same slots and the same opponent behavior,
    so the comparison between weights is paired and the market baseline cancels.
    Per-simulation scores are kept rather than averaged, because the selection
    rule needs the spread of the paired differences, not just their mean.
    """
    rng = np.random.default_rng(seed)
    scored: dict[int, dict[float, np.ndarray]] = {}
    for year, season in historical_players.groupby("target_year", sort=True):
        frame = season.reset_index(drop=True)
        rounds = min(max_rounds, len(frame) // config.league_size)
        if rounds <= 0:
            continue
        columns = {}
        for weight in grid:
            column = f"_policy_rank_{weight}"
            frame[column] = (1 - weight) * frame["market_rank"].astype(float) + (
                weight * frame["predicted_rank"].astype(float)
            )
            columns[weight] = column
        market_rank = frame["market_rank"].to_numpy(dtype=float)
        spread = np.maximum(4.0, 0.12 * market_rank)
        points: dict[float, list[float]] = {weight: [] for weight in grid}
        for _ in range(simulations_per_year):
            user_team = int(rng.integers(0, config.league_size))
            market_draw = market_rank + rng.normal(0, spread)
            for weight, column in columns.items():
                points[weight].append(
                    _simulate_one_draft(
                        frame, column, user_team, rounds, market_draw, config
                    )
                )
        scored[int(year)] = {
            weight: np.asarray(values, dtype=float)
            for weight, values in points.items()
        }
    return scored


def _best_weight(
    seasons: list[dict[float, np.ndarray]],
    grid: tuple[float, ...],
    default: float,
) -> float:
    """Pick the smallest tilt that is not measurably worse than the best one.

    Taking the raw argmax overfits the selection seasons: draft outcomes are
    noisy and a larger tilt carries more variance, so a lucky season can buy a
    weight that then loses badly. Candidates within one standard error of the
    best — measured on the paired, same-draw differences that produced them —
    are treated as tied, and the most conservative of those wins.
    """
    if not seasons:
        return default
    means = {
        weight: float(np.mean([season[weight].mean() for season in seasons]))
        for weight in grid
    }
    best = max(grid, key=lambda weight: means[weight])
    for weight in grid:
        if weight == best:
            return float(weight)
        # Cluster the error by season. Simulations inside one season share a
        # player pool and one set of realized outcomes, so treating them as
        # independent makes the objective look far more precise than it is and
        # the selected weight then moves with the random seed.
        gaps = np.array(
            [season[best].mean() - season[weight].mean() for season in seasons]
        )
        error = (
            float(gaps.std(ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0
        )
        if float(gaps.mean()) <= error:
            return float(weight)
    return float(best)


def policy_blend_weights(
    historical_players: pd.DataFrame,
    config: LeagueConfig = DEFAULT_LEAGUE_CONFIG,
    grid: tuple[float, ...] = POLICY_BLEND_WEIGHT_GRID,
    simulations_per_year: int = 100,
    max_rounds: int = 10,
    seed: int = 20260815,
    default: float = DEFAULT_MODEL_BLEND_WEIGHT,
) -> tuple[float, dict[int, float]]:
    """Fit the blend weight on the outcome it exists to improve: drafted lineups.

    Rank error is a poor selection criterion here — it improves monotonically as
    the model takes over the board while simulated drafts off that same board get
    much worse — so candidates are scored on realized starting-lineup points.

    Returns the weight for a prospective board, fitted on every available season,
    and the nested weight for each backtested season, fitted only on seasons
    before it. Both come from one scoring pass because the caller needs both.

    The objective is deliberately sampled hard enough to be reproducible. It is
    flat between roughly 0.2 and 0.5 — the model improved until the optimum
    flattened — so at a small simulation count the winner is whichever candidate
    got lucky, and the weight shown on a live board would move between runs.
    Do not read the fitted weight as a precise optimum; read it as the
    conservative end of a range that all performs about the same.
    """
    if historical_players.empty or not POLICY_REQUIRED_COLUMNS.issubset(
        historical_players.columns
    ):
        return default, {}
    scored = _policy_weight_points_by_season(
        historical_players, config, grid, simulations_per_year, max_rounds, seed
    )
    years = sorted(scored)
    nested = {
        year: _best_weight(
            [scored[earlier] for earlier in years if earlier < year], grid, default
        )
        for year in years
    }
    return _best_weight(list(scored.values()), grid, default), nested


def fit_policy_blend_weight(
    historical_players: pd.DataFrame,
    config: LeagueConfig = DEFAULT_LEAGUE_CONFIG,
    **kwargs,
) -> float:
    """Fit one blend weight for a prospective board on every available season."""
    return policy_blend_weights(historical_players, config, **kwargs)[0]


def nested_policy_blend_weights(
    historical_players: pd.DataFrame,
    config: LeagueConfig = DEFAULT_LEAGUE_CONFIG,
    **kwargs,
) -> dict[int, float]:
    """Fit each season's weight using only strictly earlier seasons.

    Without this the weight is chosen on the same seasons the policy is scored
    on, which flatters every reported draft-simulation result.
    """
    return policy_blend_weights(historical_players, config, **kwargs)[1]


def simulate_historical_draft_strategies(
    historical_players: pd.DataFrame,
    config: LeagueConfig = DEFAULT_LEAGUE_CONFIG,
    simulations_per_year: int = 250,
    max_rounds: int = 10,
    model_weight: float | None = None,
    season_weights: dict[int, float] | None = None,
    seed: int = 20260815,
) -> pd.DataFrame:
    """Compare conservative-edge, pure-model, and ADP draft policies.

    Opponent selections are sampled around historical ADP. Each paired simulation
    gives every policy the same draft slot and market draw, then scores the best
    realized starting lineup. The edge policy keeps the market in charge and only
    tilts toward the model, preventing useful disagreement signals from replacing
    the market wholesale. Its weight is fitted per season on strictly earlier
    seasons unless a fixed `model_weight` or already fitted `season_weights` are
    passed, so the simulated lift is not the product of a weight chosen on the
    same seasons it is scored on. This omits waivers and weekly start/sit
    decisions.
    """
    if historical_players.empty or not POLICY_REQUIRED_COLUMNS.issubset(
        historical_players.columns
    ):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    if model_weight is not None:
        nested_weights: dict[int, float] = {}
    elif season_weights is not None:
        nested_weights = season_weights
    else:
        nested_weights = nested_policy_blend_weights(
            historical_players, config, max_rounds=max_rounds, seed=seed
        )
    rows: list[dict[str, object]] = []
    all_edge: list[float] = []
    all_pure: list[float] = []
    all_market: list[float] = []
    year_weights: list[float] = []
    for year, season in historical_players.groupby("target_year", sort=True):
        frame = season.reset_index(drop=True)
        season_weight = (
            float(model_weight)
            if model_weight is not None
            else float(nested_weights.get(int(year), DEFAULT_MODEL_BLEND_WEIGHT))
        )
        frame["edge_policy_rank"] = (
            (1 - season_weight) * frame["market_rank"]
            + season_weight * frame["predicted_rank"]
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
        year_weights.append(season_weight)
        rows.append(
            {
                "year": int(year),
                "simulations": simulations_per_year,
                "rounds": rounds,
                "model_weight": season_weight,
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
                "model_weight": float(np.mean(year_weights)),
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
