import unittest

import pandas as pd

from fantasyoptimizer.forecasting.forecaster import (
    FEATURE_COLUMNS,
    MarketResidualModel,
    RidgeModel,
    add_market_features,
    add_official_role_context,
    build_features,
    calibrate_edge_probabilities,
)


class ForecasterTests(unittest.TestCase):
    def test_official_role_context_is_explanatory_and_matches_by_id(self):
        forecast = pd.DataFrame(
            [
                {
                    "player": "example runner",
                    "player_key": "examplerunner",
                    "gsis_id": "p1",
                    "team": "EX",
                    "pos": "RB",
                    "market_room_rank": 2,
                }
            ]
        )
        roles = pd.DataFrame(
            [
                {
                    "player_key": "differentname",
                    "gsis_id": "p1",
                    "team": "EX",
                    "pos": "RB",
                    "depth_rank": 1,
                    "official_starter": 1,
                    "roster_status": "ACT",
                    "roster_status_description": "Active",
                    "depth_position": "Running Back",
                    "snapshot_at": "2026-08-14T08:00:00Z",
                    "timing_quality": "adp_aligned",
                }
            ]
        )
        enriched = add_official_role_context(forecast, roles)
        self.assertEqual(enriched.loc[0, "official_depth_rank"], 1)
        self.assertEqual(enriched.loc[0, "official_role_known"], 1)
        self.assertEqual(
            enriched.loc[0, "role_agreement"], "Depth chart ahead of market"
        )

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

    def test_player_metadata_is_derived_at_the_target_year(self):
        candidate = pd.DataFrame(
            [{"player": "example", "player_key": "example", "pos": "RB"}]
        )
        metadata = pd.DataFrame(
            [
                {
                    "player_key": "example",
                    "pos": "RB",
                    "gsis_id": "p1",
                    "birth_date": pd.Timestamp("2000-01-01"),
                    "height": 72,
                    "weight": 210,
                    "rookie_season": 2023,
                    "draft_round": 2,
                    "draft_pick": 50,
                    "college_name": "Example U",
                }
            ]
        )
        features = build_features(
            candidate,
            pd.DataFrame(),
            target_year=2025,
            player_metadata=metadata,
        )
        self.assertEqual(features.loc[0, "experience"], 2)
        self.assertEqual(features.loc[0, "rookie"], 0)
        self.assertEqual(features.loc[0, "draft_pick"], 50)
        self.assertAlmostEqual(features.loc[0, "age"], 25.67, places=1)

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

    def test_market_residual_model_starts_from_adp_and_adjusts_by_position(self):
        rows = []
        for index in range(24):
            row = {column: 0.0 for column in FEATURE_COLUMNS}
            row.update(
                {
                    "pos": "RB",
                    "target_year": 2023 + index // 12,
                    "adp_avg": float(index + 1),
                    "timesdrafted": 100,
                    "high": max(1, index - 2),
                    "low": index + 4,
                    "stddev": 3.0,
                    "lag1_ppg": float(index % 6),
                    "pos_RB": 1.0,
                }
            )
            rows.append(row)
        frame = pd.DataFrame(rows)
        target = pd.Series(260 - 5 * frame["adp_avg"] + 2 * frame["lag1_ppg"])
        model = MarketResidualModel().fit(frame, target)
        components = model.predict_components(frame)
        self.assertTrue(components.notna().all().all())
        self.assertTrue((components["market_prediction"] != 0).all())
        self.assertGreater(components["market_adjustment"].abs().max(), 0)
        self.assertIn("market_position_rank", add_market_features(frame))

    def test_market_features_measure_same_team_position_competition(self):
        frame = pd.DataFrame(
            [
                {"player": "leader", "team": "DEN", "pos": "RB", "adp_avg": 30},
                {"player": "backup", "team": "DEN", "pos": "RB", "adp_avg": 75},
                {"player": "other", "team": "SEA", "pos": "RB", "adp_avg": 40},
                {"player": "unknown", "team": "", "pos": "RB", "adp_avg": 90},
            ]
        )
        featured = add_market_features(frame).set_index("player")
        self.assertEqual(featured.loc["leader", "market_room_rank"], 1)
        self.assertEqual(featured.loc["leader", "market_room_size"], 2)
        self.assertEqual(featured.loc["leader", "market_room_leader"], 1)
        self.assertEqual(featured.loc["backup", "market_room_rank"], 2)
        self.assertEqual(featured.loc["backup", "market_room_adp_gap"], 45)
        self.assertEqual(featured.loc["other", "market_room_size"], 1)
        self.assertEqual(featured.loc["unknown", "market_room_known"], 0)
        self.assertEqual(featured.loc["unknown", "market_room_leader"], 0)

    def test_edge_probability_is_historical_and_bounded(self):
        forecast = pd.DataFrame(
            [
                {
                    "player": "example",
                    "pos": "RB",
                    "value_gap": 15,
                    "market_rank": 60,
                    "history_seasons": 2,
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "pos": "RB",
                    "predicted_rank_surplus": 15,
                    "market_rank": 60,
                    "beat_market": True,
                },
                {
                    "pos": "RB",
                    "predicted_rank_surplus": 16,
                    "market_rank": 61,
                    "beat_market": False,
                },
            ]
        )
        result = calibrate_edge_probabilities(forecast, history)
        self.assertGreaterEqual(result.loc[0, "edge_probability"], 0)
        self.assertLessEqual(result.loc[0, "edge_probability"], 1)
        self.assertEqual(result.loc[0, "calibration_sample"], 2)

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
