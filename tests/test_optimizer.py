import unittest

import pandas as pd

from fantasyoptimizer.config.league_config import LeagueConfig
from fantasyoptimizer.optimizer import (
    build_draft_recommendations,
    simulate_historical_draft_strategies,
)


class DraftRecommendationTests(unittest.TestCase):
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
                    "value_gap": 20,
                    "model_value": 60.0,
                    "edge_probability": 0.7,
                    "forecast_points": 190.0,
                },
            ]
        )
        board = build_draft_recommendations(forecast, 24, 48)
        recommendations = board.set_index("player")["recommendation"]
        self.assertEqual(recommendations["urgent"], "Draft now")
        self.assertEqual(recommendations["wait"], "Target — may wait")
        self.assertLess(
            board.set_index("player").loc["urgent", "available_next_pick_probability"],
            board.set_index("player").loc["wait", "available_next_pick_probability"],
        )

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
