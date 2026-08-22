import unittest

import numpy as np
import pandas as pd

from fantasyoptimizer.config.league_config import LeagueConfig
from fantasyoptimizer.optimizer import (
    build_draft_recommendations,
    fit_policy_blend_weight,
    nested_policy_blend_weights,
    simulate_historical_draft_strategies,
    snake_pick_numbers,
    value_over_next_available,
)
from fantasyoptimizer.forecasting.forecaster import DEFAULT_MODEL_BLEND_WEIGHT


def _seasons_where_the_model_is_right() -> pd.DataFrame:
    """Three seasons in which model rank matches results and ADP is backwards."""
    positions = ["QB", "RB", "WR", "TE"] * 4
    rows = []
    for year in (2022, 2023, 2024):
        for index, position in enumerate(positions):
            rows.append(
                {
                    "target_year": year,
                    "player": f"{year} player {index}",
                    "pos": position,
                    "market_rank": len(positions) - index,
                    "predicted_rank": index + 1,
                    "actual_rank": index + 1,
                    "actual_points": float(200 - 5 * index),
                }
            )
    return pd.DataFrame(rows)


def _seasons_where_the_market_is_right() -> pd.DataFrame:
    """Three seasons in which ADP matches results and model rank is noise."""
    rng = np.random.default_rng(11)
    positions = ["QB", "RB", "WR", "TE"] * 4
    rows = []
    for year in (2022, 2023, 2024):
        scrambled = rng.permutation(len(positions)) + 1
        for index, position in enumerate(positions):
            rows.append(
                {
                    "target_year": year,
                    "player": f"{year} player {index}",
                    "pos": position,
                    "market_rank": index + 1,
                    "predicted_rank": int(scrambled[index]),
                    "actual_rank": index + 1,
                    "actual_points": float(200 - 5 * index),
                }
            )
    return pd.DataFrame(rows)


class DraftRecommendationTests(unittest.TestCase):
    def test_snake_pick_numbers_follow_slot_each_round(self):
        self.assertEqual(snake_pick_numbers(12, 3, 4), [3, 22, 27, 46])

    def test_board_uses_selected_platform_adp_for_availability(self):
        forecast = pd.DataFrame(
            [
                {
                    "player": "platform bargain",
                    "pos": "RB",
                    "team": "A",
                    "adp_avg": 20.0,
                    "draft_adp": 80.0,
                    "draft_adp_stddev": 4.0,
                    "fair_adp": 30,
                    "platform_actionable_adp": 42,
                    "model_rank": 30,
                    "value_gap": 0,
                    "platform_value_gap": 50,
                    "model_value": 50.0,
                    "forecast_points": 200.0,
                }
            ]
        )
        board = build_draft_recommendations(forecast, 24, 48)
        self.assertGreater(board.loc[0, "available_next_pick_probability"], 0.99)
        self.assertEqual(board.loc[0, "recommendation"], "Target — may wait")

    def test_historical_simulation_compares_all_three_policies(self):
        rows = []
        positions = ["QB", "RB", "WR", "TE"] * 4
        for index, position in enumerate(positions):
            rows.append(
                {
                    "target_year": 2025,
                    "player": f"player {index}",
                    "pos": position,
                    "market_rank": index + 1,
                    "predicted_rank": 16 - index,
                    "actual_points": float(200 - index),
                }
            )
        result = simulate_historical_draft_strategies(
            pd.DataFrame(rows),
            LeagueConfig(league_size=4, qb=1, rb=1, wr=1, te=1, flex=0),
            simulations_per_year=5,
            max_rounds=4,
            seed=1,
        )
        self.assertEqual(result.iloc[-1]["year"], "Overall")
        self.assertIn("edge_policy_lift", result)
        self.assertIn("pure_model_lift", result)

    def test_blend_weight_is_fitted_on_draft_outcomes(self):
        config = LeagueConfig(league_size=4, qb=1, rb=1, wr=1, te=1, flex=0)
        weight = fit_policy_blend_weight(
            _seasons_where_the_model_is_right(), config, simulations_per_year=10
        )
        self.assertGreaterEqual(weight, 0.5)

    def test_blend_weight_stays_small_when_the_model_rank_is_noise(self):
        config = LeagueConfig(league_size=4, qb=1, rb=1, wr=1, te=1, flex=0)
        weight = fit_policy_blend_weight(
            _seasons_where_the_market_is_right(), config, simulations_per_year=25
        )
        # Nothing to gain by tilting, so any apparent gain is luck, and the
        # one-standard-error rule should refuse to pay for luck.
        self.assertLessEqual(weight, DEFAULT_MODEL_BLEND_WEIGHT)

    def test_nested_blend_weight_never_uses_its_own_season(self):
        config = LeagueConfig(league_size=4, qb=1, rb=1, wr=1, te=1, flex=0)
        weights = nested_policy_blend_weights(
            _seasons_where_the_model_is_right(), config, simulations_per_year=10
        )
        self.assertEqual(sorted(weights), [2022, 2023, 2024])
        # The first season has no earlier evidence, so it cannot be tuned at all.
        self.assertEqual(weights[2022], DEFAULT_MODEL_BLEND_WEIGHT)
        self.assertGreaterEqual(weights[2024], 0.5)
    def test_board_distinguishes_urgent_value_from_player_who_can_wait(self):
        forecast = pd.DataFrame(
            [
                {
                    "player": "urgent",
                    "pos": "RB",
                    "team": "A",
                    "adp_avg": 20.0,
                    "stddev": 4.0,
                    "fair_adp": 15,
                    "model_rank": 15,
                    "value_gap": 14,
                    "model_value": 100.0,
                    "edge_probability": 0.7,
                    "forecast_points": 250.0,
                },
                {
                    "player": "wait",
                    "pos": "WR",
                    "team": "B",
                    "adp_avg": 80.0,
                    "stddev": 10.0,
                    "fair_adp": 45,
                    "model_rank": 45,
                    "value_gap": 26,
                    "model_value": 60.0,
                    "edge_probability": 0.7,
                    "forecast_points": 190.0,
                },
                {
                    "player": "one round only",
                    "pos": "WR",
                    "team": "C",
                    "adp_avg": 80.0,
                    "stddev": 10.0,
                    "fair_adp": 68,
                    "model_rank": 68,
                    "value_gap": 12,
                    "model_value": 40.0,
                    "edge_probability": 0.7,
                    "forecast_points": 180.0,
                },
            ]
        )
        board = build_draft_recommendations(forecast, 24, 48)
        recommendations = board.set_index("player")["recommendation"]
        self.assertEqual(recommendations["urgent"], "Draft now")
        self.assertEqual(recommendations["wait"], "Target — may wait")
        # A one-round gap is not a value. Historically those beat their ADP 49%
        # of the time against a 42% base rate, so the board does not chase them.
        self.assertNotEqual(recommendations["one round only"], "Target — may wait")
        self.assertLess(
            board.set_index("player").loc["urgent", "available_next_pick_probability"],
            board.set_index("player").loc["wait", "available_next_pick_probability"],
        )

    def test_cost_of_waiting_sees_the_cliff_and_the_flat_position(self):
        # RB: one good player, and the next one is unlikely to last.
        # TE: two interchangeable players, one of whom will certainly reach you.
        forecast = pd.DataFrame(
            [
                {"player": "rb1", "pos": "RB", "adp_avg": 5.0, "stddev": 3.0,
                 "fair_adp": 5, "forecast_points": 260.0},
                {"player": "rb2", "pos": "RB", "adp_avg": 12.0, "stddev": 3.0,
                 "fair_adp": 12, "forecast_points": 150.0},
                {"player": "te1", "pos": "TE", "adp_avg": 6.0, "stddev": 3.0,
                 "fair_adp": 6, "forecast_points": 180.0},
                {"player": "te2", "pos": "TE", "adp_avg": 200.0, "stddev": 3.0,
                 "fair_adp": 200, "forecast_points": 178.0},
            ]
        )
        cost = value_over_next_available(forecast, next_pick=30).round(0)
        by_player = dict(zip(forecast["player"], cost))
        # Passing on rb1 costs most of his value; passing on te1 costs almost
        # nothing, because te2 is equivalent and certain to be there.
        self.assertGreater(by_player["rb1"], 100)
        self.assertLess(by_player["te1"], 10)

    def test_board_removes_drafted_players(self):
        forecast = pd.DataFrame(
            [
                {
                    "player": "gone",
                    "pos": "RB",
                    "team": "A",
                    "adp_avg": 20.0,
                    "fair_adp": 20,
                    "model_rank": 20,
                    "value_gap": 0,
                    "model_value": 50.0,
                    "forecast_points": 200.0,
                }
            ]
        )
        board = build_draft_recommendations(
            forecast, 1, 24, drafted_players={"gone"}
        )
        self.assertTrue(board.empty)


if __name__ == "__main__":
    unittest.main()
