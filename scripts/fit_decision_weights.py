#!/usr/bin/env python3
"""Fit the draft-room decision score on realized drafts, or retire its terms.

The residual penalty, the blend weight, and the promotion guard are all fitted
against outcomes now. The decision score that orders the draft-room shortlist is
not: it carries four hand-chosen constants. This scores candidate weightings the
same way — simulated snake drafts on the backtested seasons, each season fitted
only on earlier ones — and compares them against simply taking the best player
available on the board, which is the honest null hypothesis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasyoptimizer.config.league_config import (  # noqa: E402
    DEFAULT_LEAGUE_CONFIG,
    LeagueConfig,
)
from fantasyoptimizer.forecasting.forecaster import (  # noqa: E402
    backtest_draft_value,
    build_training_examples,
)
from fantasyoptimizer.optimizer import optimizer as opt  # noqa: E402

# The weights as they ship, in the order (value_gap, need, availability). Model
# value is the anchor term and always carries weight 1.
SHIPPED_WEIGHTS = (0.35, 6.0, 8.0)
CANDIDATES = {
    "value_gap": (0.0, 0.15, 0.35, 0.6, 1.0),
    "need": (0.0, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0),
    "availability": (0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
}


def _draft_with_score(
    frame: pd.DataFrame,
    weights: tuple[float, float, float],
    user_team: int,
    rounds: int,
    market_draw: np.ndarray,
    config: LeagueConfig,
) -> float:
    """Draft by recomputing the decision score at every one of your picks."""
    positions = frame["pos"].tolist()
    actual_points = frame["actual_points"].to_numpy(dtype=float)
    value = frame["predicted_value"].to_numpy(dtype=float)
    gap = (frame["market_rank"] - frame["predicted_rank"]).to_numpy(dtype=float)
    market_rank = frame["market_rank"].to_numpy(dtype=float)
    spread = np.maximum(4.0, 0.12 * market_rank)
    market_order = np.argsort(market_draw)
    targets = opt._starter_targets(config)
    caps = opt._position_caps(config)
    available = set(range(len(frame)))
    rosters: list[list[int]] = [[] for _ in range(config.league_size)]
    counts: list[dict[str, int]] = [dict() for _ in range(config.league_size)]
    gap_weight, need_weight, availability_weight = weights
    total_picks = min(rounds * config.league_size, len(frame))
    user_picks = [
        pick
        for pick in range(total_picks)
        if opt._team_for_pick(pick, config.league_size) == user_team
    ]
    for overall_pick in range(total_picks):
        team = opt._team_for_pick(overall_pick, config.league_size)
        if team != user_team:
            selected = opt._select_candidate(
                available, positions, market_order, counts[team], caps
            )
        else:
            later = [pick for pick in user_picks if pick > overall_pick]
            next_pick = later[0] + 1 if later else overall_pick + 1 + config.league_size
            eligible = [
                index
                for index in available
                if counts[team].get(positions[index], 0) < caps[positions[index]]
            ]
            if not eligible:
                continue
            eligible_array = np.array(eligible)
            need = np.array(
                [
                    max(
                        targets.get(positions[index], 0.0)
                        - counts[team].get(positions[index], 0),
                        0.0,
                    )
                    for index in eligible
                ]
            )
            gone = 1 - np.array(
                [
                    opt._available_next_pick(
                        market_rank[index], spread[index], next_pick
                    )
                    for index in eligible
                ]
            )
            score = (
                value[eligible_array]
                + gap_weight * gap[eligible_array]
                + need_weight * need
                + availability_weight * gone
            )
            selected = int(eligible_array[int(np.argmax(score))])
        if selected is None:
            continue
        available.remove(selected)
        rosters[team].append(selected)
        counts[team][positions[selected]] = (
            counts[team].get(positions[selected], 0) + 1
        )
    return opt._realized_lineup_points(
        rosters[user_team], positions, actual_points, config
    )


def score_weights(
    seasons: dict[int, pd.DataFrame],
    weights: tuple[float, float, float] | None,
    config: LeagueConfig,
    simulations: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Lineup points per season. `weights=None` drafts straight off the board."""
    scored: dict[int, np.ndarray] = {}
    for year, frame in seasons.items():
        rng = np.random.default_rng(seed)
        market_rank = frame["market_rank"].to_numpy(dtype=float)
        spread = np.maximum(4.0, 0.12 * market_rank)
        rounds = min(10, len(frame) // config.league_size)
        results = []
        for _ in range(simulations):
            user_team = int(rng.integers(0, config.league_size))
            draw = market_rank + rng.normal(0, spread)
            if weights is None:
                results.append(
                    opt._simulate_one_draft(
                        frame, "board_rank", user_team, rounds, draw, config
                    )
                )
            else:
                results.append(
                    _draft_with_score(
                        frame, weights, user_team, rounds, draw, config
                    )
                )
        scored[year] = np.asarray(results, dtype=float)
    return scored


def fit(
    seasons: dict[int, pd.DataFrame],
    config: LeagueConfig,
    simulations: int,
    seed: int,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """Coordinate search over the weights, scored against the board baseline."""
    baseline = score_weights(seasons, None, config, simulations, seed)

    def lift(weights):
        scored = score_weights(seasons, weights, config, simulations, seed)
        return float(
            np.mean([scored[year].mean() - baseline[year].mean() for year in scored])
        )

    best = list(SHIPPED_WEIGHTS)
    names = list(CANDIDATES)
    for sweep in range(2):
        for index, name in enumerate(names):
            results = []
            for candidate in CANDIDATES[name]:
                trial = list(best)
                trial[index] = candidate
                results.append((candidate, lift(tuple(trial))))
            results.sort(key=lambda row: row[1], reverse=True)
            best[index] = results[0][0]
            if verbose:
                print(
                    f"pass {sweep + 1} · {name}: "
                    + ", ".join(f"{value:g}→{value_lift:+.1f}" for value, value_lift in results)
                    + f"  [keeping {best[index]:g}]"
                )
    return tuple(best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    config = DEFAULT_LEAGUE_CONFIG
    pd.set_option("display.width", 200)

    training = build_training_examples(2025)
    _, players = backtest_draft_value(2025, config, training_examples=training)
    weight = 0.25
    seasons = {}
    for year, season in players.groupby("target_year"):
        frame = season.reset_index(drop=True)
        frame["board_rank"] = (
            (1 - weight) * frame["market_rank"] + weight * frame["predicted_rank"]
        )
        seasons[int(year)] = frame

    def mean_lift(scored, baseline):
        return float(
            np.mean(
                [scored[year].mean() - baseline[year].mean() for year in scored]
            )
        )

    baseline = score_weights(seasons, None, config, args.simulations, args.seed)
    print("Baseline: take the best player available on the blended board.")
    shipped = score_weights(seasons, SHIPPED_WEIGHTS, config, args.simulations, args.seed)
    print(
        f"Shipped decision score {SHIPPED_WEIGHTS}: "
        f"{mean_lift(shipped, baseline):+.1f} points vs the board\n"
    )

    best = fit(seasons, config, args.simulations, args.seed, verbose=True)
    print(f"\nFitted weights (value_gap, need, availability): {best}")
    fitted = score_weights(seasons, best, config, args.simulations, args.seed)
    print(f"Fitted score vs board: {mean_lift(fitted, baseline):+.1f} points")
    per_season = pd.DataFrame(
        {
            "year": list(seasons),
            "board": [baseline[year].mean() for year in seasons],
            "shipped score": [shipped[year].mean() for year in seasons],
            "fitted score": [fitted[year].mean() for year in seasons],
        }
    )
    print(per_season.round(1).to_string(index=False))

    print("\n=== nested check: fit on earlier seasons, score on the next one ===")
    years = sorted(seasons)
    rows = []
    for held_out in years[2:]:
        prior = {year: seasons[year] for year in years if year < held_out}
        weights = fit(prior, config, max(args.simulations // 2, 20), args.seed)
        one = {held_out: seasons[held_out]}
        scored = score_weights(one, weights, config, args.simulations, args.seed)
        board_only = score_weights(one, None, config, args.simulations, args.seed)
        shipped_one = score_weights(
            one, SHIPPED_WEIGHTS, config, args.simulations, args.seed
        )
        rows.append(
            {
                "season": held_out,
                "weights fitted on earlier seasons": str(weights),
                "fitted vs board": scored[held_out].mean() - board_only[held_out].mean(),
                "shipped vs board": shipped_one[held_out].mean()
                - board_only[held_out].mean(),
            }
        )
    nested = pd.DataFrame(rows)
    print(nested.round(1).to_string(index=False))
    print(
        f"  mean out-of-sample: fitted {nested['fitted vs board'].mean():+.1f}, "
        f"shipped {nested['shipped vs board'].mean():+.1f}"
    )


if __name__ == "__main__":
    main()
