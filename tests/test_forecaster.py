import unittest

import pandas as pd

from fantasyoptimizer.forecasting.forecaster import (
    FEATURE_COLUMNS,
    RidgeModel,
    build_features,
)


class ForecasterTests(unittest.TestCase):
    def test_features_never_use_target_or_future_results(self):
        candidate = pd.DataFrame(
            [{"player": "example", "player_key": "example", "pos": "RB"}]
        )
        history = pd.DataFrame(
            [
                {
                    "player_key": "example",
                    "pos": "RB",
                    "year": 2024,
                    "pts_avg": 10,
                    "pts_ttl": 160,
                    "gp": 16,
                    "opp": 200,
                    "ppo": 0.8,
                },
                {
                    "player_key": "example",
                    "pos": "RB",
                    "year": 2025,
                    "pts_avg": 99,
                    "pts_ttl": 999,
                    "gp": 17,
                    "opp": 999,
                    "ppo": 1.0,
                },
            ]
        )
        features = build_features(candidate, history, target_year=2025)
        self.assertEqual(features.loc[0, "lag1_ppg"], 10)
        self.assertNotEqual(features.loc[0, "lag1_ppg"], 99)

    def test_ridge_model_returns_finite_predictions(self):
        rows = []
        for index in range(20):
            row = {column: 0.0 for column in FEATURE_COLUMNS}
            row.update(
                {
                    "lag1_ppg": float(index),
                    "weighted_ppg": float(index),
                    "lag1_games": 16.0,
                    "weighted_games": 16.0,
                    "lag1_total": float(index * 16),
                    "history_seasons": 1.0,
                    "pos_RB": 1.0,
                }
            )
            rows.append(row)
        frame = pd.DataFrame(rows)
        target = pd.Series([index * 15.0 for index in range(20)])
        predictions = RidgeModel().fit(frame, target).predict(frame)
        self.assertTrue(pd.Series(predictions).notna().all())

    def test_destination_team_position_context_follows_a_trade(self):
        candidate = pd.DataFrame(
            [
                {
                    "player": "example",
                    "player_key": "example",
                    "pos": "WR",
                    "team": "NEW",
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "player_key": "example",
                    "pos": "WR",
                    "team": "OLD",
                    "year": 2024,
                    "pts_avg": 10,
                    "pts_ttl": 100,
                    "gp": 10,
                    "opp": 20,
                    "ppo": 5.0,
                }
            ]
        )
        context = pd.DataFrame(
            [
                {
                    "year": 2024,
                    "team": "OLD",
                    "pos": "WR",
                    "total_points": 200,
                    "leader_points": 100,
                    "top_two_points": 180,
                    "total_opportunities": 50,
                    "leader_opportunities": 20,
                    "top_two_opportunities": 40,
                    "contributors": 4,
                    "rank_percentile": 0.4,
                    "opportunity_rank_percentile": 0.5,
                },
                {
                    "year": 2024,
                    "team": "NEW",
                    "pos": "WR",
                    "total_points": 300,
                    "leader_points": 170,
                    "top_two_points": 260,
                    "total_opportunities": 70,
                    "leader_opportunities": 30,
                    "top_two_opportunities": 55,
                    "contributors": 5,
                    "rank_percentile": 0.9,
                    "opportunity_rank_percentile": 0.8,
                },
            ]
        )
        player_context = pd.DataFrame(
            [
                {
                    "year": 2024,
                    "team": "OLD",
                    "pos": "WR",
                    "player_key": "example",
                    "player_points": 100,
                    "player_opportunities": 20,
                }
            ]
        )
        features = build_features(
            candidate, history, 2025, context, player_context
        )
        self.assertEqual(features.loc[0, "team_pos_lag1_total"], 300)
        self.assertEqual(features.loc[0, "destination_other_points"], 300)
        self.assertEqual(features.loc[0, "changed_team"], 1)
        self.assertEqual(features.loc[0, "same_team_last_year"], 0)
        self.assertEqual(features.loc[0, "lag1_role_share"], 0.4)

    def test_incumbent_is_removed_from_environment_signal(self):
        candidate = pd.DataFrame(
            [{"player": "example", "player_key": "example", "pos": "WR", "team": "OLD"}]
        )
        history = pd.DataFrame(
            [
                {
                    "player_key": "example",
                    "pos": "WR",
                    "team": "OLD",
                    "year": 2024,
                    "pts_avg": 10,
                    "pts_ttl": 100,
                    "gp": 10,
                    "opp": 20,
                    "ppo": 5.0,
                }
            ]
        )
        context = pd.DataFrame(
            [
                {
                    "year": 2024,
                    "team": "OLD",
                    "pos": "WR",
                    "total_points": 200,
                    "leader_points": 100,
                    "top_two_points": 180,
                    "total_opportunities": 50,
                    "leader_opportunities": 20,
                    "top_two_opportunities": 40,
                    "contributors": 4,
                    "rank_percentile": 0.4,
                    "opportunity_rank_percentile": 0.5,
                }
            ]
        )
        player_context = pd.DataFrame(
            [
                {
                    "year": 2024,
                    "team": "OLD",
                    "pos": "WR",
                    "player_key": "example",
                    "player_points": 100,
                    "player_opportunities": 20,
                }
            ]
        )
        features = build_features(
            candidate, history, 2025, context, player_context
        )
        self.assertEqual(features.loc[0, "team_pos_lag1_opportunities"], 50)
        self.assertEqual(features.loc[0, "team_pos_other_lag1_points"], 100)
        self.assertEqual(features.loc[0, "team_pos_other_lag1_opportunities"], 30)
        self.assertEqual(features.loc[0, "same_team_last_year"], 1)


if __name__ == "__main__":
    unittest.main()
