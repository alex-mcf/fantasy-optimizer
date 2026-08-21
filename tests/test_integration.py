import unittest

from fantasyoptimizer.config.league_config import DEFAULT_LEAGUE_CONFIG
from fantasyoptimizer.forecasting.forecaster import (
    MODEL_PROMOTION_CAP_ROUNDS,
    backtest_draft_value,
    backtest_forecaster,
    build_training_examples,
    calibrate_edge_probabilities,
    forecast_season,
)
from fantasyoptimizer.scoring.scoring_engine import (
    compute_player_season_scores,
    compute_scores,
)
from fantasyoptimizer.utils.data_loader import available_years


class HistoricalPipelineTests(unittest.TestCase):
    def test_restored_historical_data_runs_end_to_end(self):
        years = available_years()
        self.assertEqual(years, list(range(2018, 2026)))
        result = compute_scores(years)
        season_rows = compute_player_season_scores(years)
        self.assertFalse(result.empty)
        self.assertFalse(result[["player", "pos", "score"]].isna().any().any())
        self.assertTrue(result["score"].is_monotonic_decreasing)
        self.assertEqual(season_rows["year"].max(), 2025)
        self.assertIn("confidence", result.columns)
        self.assertIn("season_score", season_rows.columns)

    def test_2026_forecast_and_backtest_run_end_to_end(self):
        training = build_training_examples(2025)
        forecast = forecast_season(2026, training_examples=training)
        backtest = backtest_forecaster(2025, training_examples=training)
        value_backtest, value_players = backtest_draft_value(
            2025, training_examples=training
        )
        forecast = calibrate_edge_probabilities(forecast, value_players)
        self.assertGreater(len(forecast), 100)
        self.assertFalse(
            forecast[
                [
                    "forecast_points",
                    "market_points",
                    "market_adjustment",
                    "age",
                    "experience",
                    "draft_pick",
                    "forecast_ppg",
                    "forecast_games",
                    "model_rank",
                    "adp_avg",
                    "value_gap",
                    "actionable_adp",
                    "market_room_rank",
                    "market_room_size",
                    "market_room_adp_gap",
                    "edge_probability",
                ]
            ]
            .isna()
            .any()
            .any()
        )
        self.assertTrue(forecast["model_rank"].is_monotonic_increasing)
        # No player may be ranked more than three rounds ahead of the market.
        self.assertLessEqual(
            int((forecast["market_rank"] - forecast["model_rank"]).max()),
            MODEL_PROMOTION_CAP_ROUNDS * DEFAULT_LEAGUE_CONFIG.league_size,
        )
        self.assertIn("uncapped_model_rank", forecast)
        self.assertIn("official_depth_rank", forecast)
        self.assertIn("role_agreement", forecast)
        self.assertFalse(backtest.empty)
        self.assertLess(int(backtest["year"].max()), 2026)
        self.assertFalse(value_backtest.empty)
        self.assertEqual(value_backtest.iloc[-1]["year"], "Overall")
        self.assertGreater(int(value_backtest.iloc[-1]["bargains"]), 0)
        self.assertFalse(value_players.empty)
        self.assertLess(int(value_players["target_year"].max()), 2026)
        self.assertTrue(
            value_players["bargain_flag"].equals(
                value_players["predicted_rank_surplus"] >= 12
            )
        )


if __name__ == "__main__":
    unittest.main()
