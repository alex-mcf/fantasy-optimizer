import unittest

from fantasyoptimizer.scoring.scoring_engine import compute_scores
from fantasyoptimizer.utils.data_loader import available_years


class HistoricalPipelineTests(unittest.TestCase):
    def test_restored_historical_data_runs_end_to_end(self):
        years = available_years()
        self.assertEqual(years, [2022, 2023, 2024])
        result = compute_scores(years)
        self.assertFalse(result.empty)
        self.assertFalse(result[["player", "pos", "score"]].isna().any().any())
        self.assertTrue(result["score"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()

