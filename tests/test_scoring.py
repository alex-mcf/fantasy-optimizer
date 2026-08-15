import unittest

import pandas as pd

from fantasyoptimizer.scoring.cost import compute_cost, expected_points_from_adp
from fantasyoptimizer.scoring.projection import compute_projection
from fantasyoptimizer.scoring.risk import compute_risk
from fantasyoptimizer.scoring.vorp import compute_vorp


class CostTests(unittest.TestCase):
    def test_cost_curve_matches_documented_anchors(self):
        self.assertEqual(expected_points_from_adp(1), 260)
        self.assertEqual(expected_points_from_adp(100), 100)
        self.assertEqual(expected_points_from_adp(150), 70)
        self.assertGreater(expected_points_from_adp(50), expected_points_from_adp(100))

    def test_round_boundaries(self):
        frame = pd.DataFrame(
            {"adp_avg": [1.0, 12.0, 12.1, 24.0], "projection": [0] * 4}
        )
        result = compute_cost(frame)
        self.assertEqual(result["round"].tolist(), [1, 1, 2, 2])

    def test_cost_is_calibrated_within_position(self):
        frame = pd.DataFrame(
            {
                "year": [2024] * 4,
                "pos": ["QB", "QB", "RB", "RB"],
                "adp_avg": [10, 100, 1, 20],
                "projection": [200, 300, 100, 250],
            }
        )
        result = compute_cost(frame)
        # The late QB and RB each beat the best result expected at their cost.
        self.assertEqual(result["expected_points_at_cost"].tolist(), [300, 200, 250, 100])
        self.assertEqual(result["value_over_cost"].tolist(), [-100, 100, -150, 150])


class ProjectionAndRiskTests(unittest.TestCase):
    def test_projection_uses_season_total(self):
        frame = pd.DataFrame({"pts_avg": [20.0], "pts_ttl": [340.0]})
        self.assertEqual(compute_projection(frame).loc[0, "projection"], 340.0)

    def test_risk_is_calculated_per_player(self):
        frame = pd.DataFrame(
            {
                "player": ["steady", "volatile", "steady", "volatile"],
                "pts_avg": [10.0, 5.0, 10.0, 15.0],
            }
        )
        result = compute_risk(frame)
        self.assertEqual(
            result.loc[result.player == "steady", "risk"].tolist(), [0.0, 0.0]
        )
        self.assertEqual(
            result.loc[result.player == "volatile", "risk"].tolist(), [5.0, 5.0]
        )


class VorpTests(unittest.TestCase):
    def test_replacement_value_is_isolated_by_year(self):
        rows = []
        for year, base in [(2023, 100), (2024, 300)]:
            for rank in range(1, 14):
                rows.append(
                    {
                        "year": year,
                        "pos": "QB",
                        "player": f"{year}-{rank}",
                        "projection": base - rank,
                    }
                )
        result = compute_vorp(pd.DataFrame(rows))
        replacement_rows = result[result.player.isin(["2023-12", "2024-12"])]
        self.assertEqual(replacement_rows["vorp"].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
