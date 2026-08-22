#!/usr/bin/env python3
"""Snapshot the board you are about to draft from, so it can be graded later.

Everything in this project is refit as seasons arrive: the residual penalty, the
blend weight, the decision weights. That is the right behavior for a forecaster
and the wrong behavior for an evaluation, because in January the board will no
longer be the board you actually used. This writes the current one to a dated
file with the settings that produced it, so the first genuinely untouched test of
the model is graded against what it really said.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasyoptimizer.config.league_config import (  # noqa: E402
    DEFAULT_LEAGUE_CONFIG,
    LeagueConfig,
)
from fantasyoptimizer.forecasting import forecaster as fc  # noqa: E402
from fantasyoptimizer.optimizer import optimizer as opt  # noqa: E402
from fantasyoptimizer.optimizer import policy_blend_weights  # noqa: E402
from fantasyoptimizer.utils.atomic import write_csv, write_text  # noqa: E402
from fantasyoptimizer.utils.data_loader import (  # noqa: E402
    available_adp_years,
    available_years,
    load_adp_metadata,
)

SNAPSHOT_COLUMNS = [
    "player",
    "player_key",
    "pos",
    "team",
    "adp_avg",
    "market_rank",
    "uncapped_model_rank",
    "model_rank",
    "fair_adp",
    "actionable_adp",
    "value_gap",
    "forecast_points",
    "forecast_ppg",
    "forecast_games",
    "model_value",
    "position_rank",
    "edge_probability",
    "calibration_sample",
    "confidence",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", type=int, default=DEFAULT_LEAGUE_CONFIG.league_size)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "boards",
        help="Directory for the dated snapshot.",
    )
    args = parser.parse_args()
    config = LeagueConfig(
        league_size=args.teams,
        qb=DEFAULT_LEAGUE_CONFIG.qb,
        rb=DEFAULT_LEAGUE_CONFIG.rb,
        wr=DEFAULT_LEAGUE_CONFIG.wr,
        te=DEFAULT_LEAGUE_CONFIG.te,
        flex=DEFAULT_LEAGUE_CONFIG.flex,
        superflex=DEFAULT_LEAGUE_CONFIG.superflex,
    )

    target_year = max(available_adp_years())
    through_year = max(available_years())
    training = fc.build_training_examples(through_year)
    _, value_players = fc.backtest_draft_value(
        through_year, config, training_examples=training
    )
    model_weight, season_weights = policy_blend_weights(value_players, config)
    forecast = fc.forecast_season(
        target_year, config, training_examples=training, model_weight=model_weight
    )
    forecast = fc.calibrate_edge_probabilities(forecast, value_players, config)

    stamp = datetime.now(timezone.utc)
    destination = args.out / f"{target_year}_board_{stamp:%Y%m%d}.csv"
    board = forecast[SNAPSHOT_COLUMNS].sort_values("model_rank", ignore_index=True)
    write_csv(board, destination, index=False)

    allocation = (
        board.head(config.league_size * 5)["pos"].value_counts().to_dict()
    )
    market_allocation = (
        forecast.nsmallest(config.league_size * 5, "market_rank")["pos"]
        .value_counts()
        .to_dict()
    )
    metadata = {
        "target_season": target_year,
        "frozen_at_utc": stamp.isoformat(),
        "trained_through_season": through_year,
        "league": {
            "teams": config.league_size,
            "qb": config.qb,
            "rb": config.rb,
            "wr": config.wr,
            "te": config.te,
            "flex": config.flex,
            "superflex": config.superflex,
        },
        "adp_source_metadata": load_adp_metadata(target_year),
        "fitted": {
            "model_blend_weight": model_weight,
            "nested_season_weights": {
                str(year): weight for year, weight in season_weights.items()
            },
            "residual_alpha_grid": list(fc.RESIDUAL_ALPHA_GRID),
            "residual_features": list(fc.RESIDUAL_FEATURE_COLUMNS),
            "promotion_cap_share": fc.MODEL_PROMOTION_CAP_SHARE,
            "promotion_floor_rounds": fc.MODEL_PROMOTION_FLOOR_ROUNDS,
            "decision_weights": {
                "value_gap": opt.VALUE_GAP_WEIGHT,
                "edge_probability": opt.EDGE_PROBABILITY_WEIGHT,
                "starter_need": opt.STARTER_NEED_WEIGHT,
                "scarcity": opt.SCARCITY_WEIGHT,
            },
            "actionable_gap_rounds": opt.ACTIONABLE_GAP_ROUNDS,
        },
        "positional_allocation_first_five_rounds": {
            "model": allocation,
            "market": market_allocation,
        },
        "players": len(board),
    }
    write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        destination.with_suffix(".meta.json"),
    )

    print(f"Froze {len(board)} players to {destination}")
    print(f"Model blend weight: {model_weight:.2f}")
    print(f"First five rounds — model: {allocation}")
    print(f"First five rounds — market: {market_allocation}")


if __name__ == "__main__":
    main()
