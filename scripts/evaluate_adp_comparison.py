#!/usr/bin/env python3
"""Test the model the way it is used: as a second opinion on a mock-draft ADP.

The dashboard's draft simulation answers "should I draft off this board?". This
answers the narrower question a user actually asks in front of a mock ADP list:
when the model disagrees with the market about a specific player, how often has
that disagreement been right, does it depend on where in the draft the player
sits, and does the answer survive the market moving under it — because the ADP
you compare against is rarely the ADP you draft against.
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
    MODEL_PROMOTION_CAP_ROUNDS,
    backtest_draft_value,
    build_training_examples,
    forecast_season,
)
from fantasyoptimizer.optimizer import policy_blend_weights  # noqa: E402
from fantasyoptimizer.utils.data_loader import (  # noqa: E402
    available_adp_years,
    available_result_years,
)

# How far a player's price is expected to move between one mock-draft market and
# another. Matched to the spread the draft simulation uses for opponent
# behavior: a few picks near the top of the board, proportionally more later.
MARKET_DRIFT_FLOOR = 4.0
MARKET_DRIFT_RATE = 0.12


def disagreement_buckets(players: pd.DataFrame, league_size: int) -> pd.DataFrame:
    """Group calls by how far the model moves a player, and score each group."""
    frame = players.copy()
    # Floor, not round: a "1 round bargain" must mean the model moved the player
    # at least a full round, which is the threshold the dashboard flags on.
    frame["rounds_moved"] = np.floor(
        frame["predicted_rank_surplus"] / league_size
    ).astype(int)
    frame["bucket"] = pd.cut(
        frame["rounds_moved"],
        bins=[-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf],
        labels=[
            "fade 3+ rounds",
            "fade 2 rounds",
            "fade 1 round",
            "within a round",
            "bargain 1 round",
            "bargain 2 rounds",
            "bargain 3 rounds",
            "bargain 4+ rounds",
        ],
        right=False,
    )
    rows = []
    for bucket, group in frame.groupby("bucket", observed=True):
        beat = group["actual_rank_surplus"] > 0
        rows.append(
            {
                "call": bucket,
                "players": len(group),
                "beat their ADP %": 100 * float(beat.mean()),
                "mean realized rank surplus": float(group["actual_rank_surplus"].mean()),
                "model closer than ADP %": 100 * float(
                    (
                        (group["actual_rank"] - group["predicted_rank"]).abs()
                        < (group["actual_rank"] - group["market_rank"]).abs()
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def by_position_and_round(players: pd.DataFrame, league_size: int) -> pd.DataFrame:
    """Where on the board, and at which position, the disagreements pay."""
    frame = players[players["bargain_flag"]].copy()
    frame["market round"] = np.ceil(frame["market_rank"] / league_size).astype(int)
    frame["draft stage"] = pd.cut(
        frame["market round"],
        bins=[0, 3, 6, 10, np.inf],
        labels=["rounds 1-3", "rounds 4-6", "rounds 7-10", "round 11+"],
    )
    rows = []
    for (position, stage), group in frame.groupby(
        ["pos", "draft stage"], observed=True
    ):
        if len(group) < 5:
            continue
        rows.append(
            {
                "pos": position,
                "stage": stage,
                "bargains": len(group),
                "beat their ADP %": 100 * float((group["actual_rank_surplus"] > 0).mean()),
                "mean realized rank surplus": float(group["actual_rank_surplus"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["pos", "stage"], ignore_index=True)


def market_drift_stability(
    players: pd.DataFrame, league_size: int, trials: int = 200, seed: int = 20260821
) -> pd.DataFrame:
    """Re-price the board on a different plausible market and re-flag bargains.

    The model is trained and evaluated on one consistent mock-draft market. A
    real draft happens on a different one, so a call that only exists because of
    a few picks of ADP noise is not actionable.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for year, season in players.groupby("target_year"):
        original = season["market_rank"].to_numpy(dtype=float)
        model = season["predicted_rank"].to_numpy(dtype=float)
        actual_rank = season["actual_rank"].to_numpy(dtype=float)
        flagged = original - model >= league_size
        survived, still_paid, drifted_in = [], [], []
        for _ in range(trials):
            spread = np.maximum(MARKET_DRIFT_FLOOR, MARKET_DRIFT_RATE * original)
            moved = original + rng.normal(0, spread)
            moved_rank = pd.Series(moved).rank(method="first").to_numpy()
            now_flagged = moved_rank - model >= league_size
            if flagged.any():
                survived.append(float((now_flagged & flagged).sum() / flagged.sum()))
            if now_flagged.any():
                still_paid.append(
                    float((actual_rank[now_flagged] < moved_rank[now_flagged]).mean())
                )
                drifted_in.append(float((now_flagged & ~flagged).sum()))
        rows.append(
            {
                "year": int(year),
                "bargains on the real market": int(flagged.sum()),
                "still flagged after drift %": 100 * float(np.mean(survived)),
                "new names per drifted market": float(np.mean(drifted_in)),
                "beat their ADP after drift %": 100 * float(np.mean(still_paid)),
            }
        )
    return pd.DataFrame(rows)


def live_disagreements(
    target_year: int, config: LeagueConfig, training: pd.DataFrame, weight: float
) -> pd.DataFrame:
    """The current board's biggest arguments with the market, for eyeballing."""
    forecast = forecast_season(
        target_year, config, training_examples=training, model_weight=weight
    )
    forecast["rounds moved"] = (
        (forecast["market_rank"] - forecast["model_rank"]) / config.league_size
    ).round(1)
    forecast["guard held it back"] = (
        forecast["market_rank"] - forecast["uncapped_model_rank"]
        > MODEL_PROMOTION_CAP_ROUNDS * config.league_size
    )
    columns = [
        "player",
        "pos",
        "market_rank",
        "model_rank",
        "actionable_adp",
        "rounds moved",
        "confidence",
        "guard held it back",
    ]
    top = forecast.nlargest(12, "value_gap")[columns]
    bottom = forecast.nsmallest(8, "value_gap")[columns]
    return pd.concat([top, bottom], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", type=int, default=DEFAULT_LEAGUE_CONFIG.league_size)
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    config = LeagueConfig(
        league_size=args.teams,
        qb=DEFAULT_LEAGUE_CONFIG.qb,
        rb=DEFAULT_LEAGUE_CONFIG.rb,
        wr=DEFAULT_LEAGUE_CONFIG.wr,
        te=DEFAULT_LEAGUE_CONFIG.te,
        flex=DEFAULT_LEAGUE_CONFIG.flex,
    )
    pd.set_option("display.width", 200)

    complete = sorted(set(available_adp_years()) & set(available_result_years()))
    target_year = max(available_adp_years())
    training = build_training_examples(max(complete))
    _, players = backtest_draft_value(
        max(complete), config, training_examples=training
    )
    weight, _ = policy_blend_weights(players, config)

    print(f"Backtested seasons: {sorted(players['target_year'].unique())}")
    print(f"Fitted model weight for the live board: {weight:.2f}\n")

    print("=== Does a disagreement with ADP pay? ===")
    print(disagreement_buckets(players, config.league_size).round(1).to_string(index=False))

    print("\n=== Where the one-round-plus bargains pay ===")
    print(by_position_and_round(players, config.league_size).round(1).to_string(index=False))

    print("\n=== Does the call survive a different mock market? ===")
    print(
        market_drift_stability(players, config.league_size, args.trials)
        .round(1)
        .to_string(index=False)
    )

    print(f"\n=== {target_year} board versus the current mock ADP ===")
    print(live_disagreements(target_year, config, training, weight).to_string(index=False))


if __name__ == "__main__":
    main()
