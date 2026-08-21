import unittest

import pandas as pd

from fantasyoptimizer.market import (
    apply_platform_adp,
    parse_platform_adp,
    unmatched_platform_players,
)


class PlatformAdpTests(unittest.TestCase):
    def test_common_platform_columns_are_normalized(self):
        parsed = parse_platform_adp(
            pd.DataFrame(
                [
                    {
                        "Player Name": "Example Runner Jr.",
                        "Position": "RB12",
                        "Overall Rank": "24.5",
                    }
                ]
            )
        )
        self.assertEqual(parsed.loc[0, "player_key"], "examplerunner")
        self.assertEqual(parsed.loc[0, "pos"], "RB")
        self.assertEqual(parsed.loc[0, "draft_adp"], 24.5)

    def test_platform_cost_changes_without_replacing_forecast_baseline(self):
        forecast = pd.DataFrame(
            [
                {
                    "player": "example runner",
                    "player_key": "examplerunner",
                    "pos": "RB",
                    "adp_avg": 30.0,
                    "stddev": 5.0,
                    "fair_adp": 10,
                },
                {
                    "player": "other receiver",
                    "player_key": "otherreceiver",
                    "pos": "WR",
                    "adp_avg": 40.0,
                    "stddev": 6.0,
                    "fair_adp": 20,
                },
            ]
        )
        platform = parse_platform_adp(
            pd.DataFrame([{"Player": "Example Runner", "POS": "RB", "ADP": 80}])
        )
        result = apply_platform_adp(forecast, platform, "Sleeper").set_index("player")
        self.assertEqual(result.loc["example runner", "adp_avg"], 30)
        self.assertEqual(result.loc["example runner", "draft_adp"], 80)
        self.assertTrue(result.loc["example runner", "platform_match"])
        self.assertEqual(result.loc["other receiver", "platform_source"], "FFC fallback")

    def test_uploaded_players_off_the_board_are_reported_not_dropped(self):
        forecast = pd.DataFrame(
            [
                {
                    "player": "example runner",
                    "player_key": "examplerunner",
                    "pos": "RB",
                    "adp_avg": 30.0,
                    "stddev": 5.0,
                    "fair_adp": 10,
                }
            ]
        )
        platform = parse_platform_adp(
            pd.DataFrame(
                [
                    {"Player": "Example Runner", "POS": "RB", "ADP": 80},
                    {"Player": "Unknown Kicker", "POS": "K", "ADP": 140},
                ]
            )
        )
        unmatched = unmatched_platform_players(forecast, platform)
        self.assertEqual(list(unmatched["platform_player"]), ["Unknown Kicker"])

    def test_board_blend_follows_the_weight_the_forecast_was_fitted_with(self):
        forecast = pd.DataFrame(
            [
                {
                    "player": "example runner",
                    "player_key": "examplerunner",
                    "pos": "RB",
                    "adp_avg": 30.0,
                    "stddev": 5.0,
                    "fair_adp": 10,
                    "blend_model_weight": 0.5,
                },
                {
                    "player": "other receiver",
                    "player_key": "otherreceiver",
                    "pos": "WR",
                    "adp_avg": 40.0,
                    "stddev": 6.0,
                    "fair_adp": 20,
                    "blend_model_weight": 0.5,
                },
            ]
        )
        result = apply_platform_adp(forecast, None, "Fantasy Football Calculator")
        # Market ranks are 1 and 2, so a half-weight blend lands halfway to fair ADP.
        self.assertEqual(
            list(result["platform_actionable_adp"]),
            [round(0.5 * 1 + 0.5 * 10), round(0.5 * 2 + 0.5 * 20)],
        )


if __name__ == "__main__":
    unittest.main()
