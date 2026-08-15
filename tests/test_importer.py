import unittest

import pandas as pd

from scripts.import_nflverse_results import (
    build_half_ppr_results,
    build_player_team_context,
    build_team_position_context,
)
from scripts.import_ffc_adp import build_adp_export


class NflverseImporterTests(unittest.TestCase):
    def test_half_ppr_matches_fantasypros_interception_scoring(self):
        stats = pd.DataFrame(
            {
                "player_id": ["player-1"],
                "player_display_name": ["Example Quarterback"],
                "position": ["QB"],
                "team": ["EX"],
                "season": [2025],
                "week": [1],
                "season_type": ["REG"],
                # nflverse total already includes a -2 interception penalty.
                "fantasy_points": [18.0],
                "passing_interceptions": [1],
                "receptions": [2],
                "attempts": [30],
                "carries": [1],
                "targets": [0],
            }
        )
        result = build_half_ppr_results(stats, 2025)
        # Add one for FantasyPros' -1 INT convention and one for two half catches.
        self.assertEqual(result.loc[0, "TTL"], 20.0)
        self.assertEqual(result.loc[0, "AVG"], 20.0)
        self.assertEqual(result.loc[0, "OPP"], 31.0)

    def test_snap_counts_include_active_zero_point_games(self):
        stats = pd.DataFrame(
            {
                "player_id": ["player-1"],
                "player_display_name": ["Example Runner"],
                "position": ["RB"],
                "team": ["EX"],
                "season": [2025],
                "week": [1],
                "season_type": ["REG"],
                "fantasy_points": [10.0],
                "passing_interceptions": [0],
                "receptions": [0],
                "attempts": [0],
                "carries": [8],
                "targets": [2],
            }
        )
        snaps = pd.DataFrame(
            {
                "game_id": ["game-1", "game-2"],
                "season": [2025, 2025],
                "game_type": ["REG", "REG"],
                "week": [1, 2],
                "player": ["Example Runner", "Example Runner"],
                "position": ["RB", "RB"],
                "offense_snaps": [10, 5],
                "st_snaps": [0, 0],
            }
        )
        result = build_half_ppr_results(stats, 2025, snaps)
        self.assertEqual(result.loc[0, "GP"], 2)
        self.assertEqual(result.loc[0, "2"], 0)
        self.assertEqual(result.loc[0, "AVG"], 5.0)

    def test_ffc_import_filters_unsupported_and_deleted_players(self):
        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name": "Valid Runner",
                    "position": "RB",
                    "team": "EX",
                    "adp": 5.2,
                    "times_drafted": 100,
                    "high": 1,
                    "low": 10,
                    "stdev": 2.0,
                },
                {
                    "player_id": 2,
                    "name": "Deleted Deleted",
                    "position": "WR",
                    "team": "",
                    "adp": 10.0,
                    "times_drafted": 1,
                    "high": 10,
                    "low": 10,
                    "stdev": 0.0,
                },
                {
                    "player_id": 3,
                    "name": "Example Defense",
                    "position": "DEF",
                    "team": "EX",
                    "adp": 150.0,
                    "times_drafted": 20,
                    "high": 120,
                    "low": 180,
                    "stdev": 10.0,
                },
            ]
        )
        result = build_adp_export(players)
        self.assertEqual(result["Player"].tolist(), ["Valid Runner"])

    def test_team_position_context_uses_weekly_team_assignment(self):
        stats = pd.DataFrame(
            {
                "player_id": ["p1", "p1", "p2"],
                "player_display_name": ["Runner One", "Runner One", "Runner Two"],
                "position": ["RB", "RB", "RB"],
                "team": ["OLD", "NEW", "NEW"],
                "season": [2025, 2025, 2025],
                "week": [1, 2, 2],
                "season_type": ["REG", "REG", "REG"],
                "fantasy_points": [10.0, 8.0, 6.0],
                "passing_interceptions": [0, 0, 0],
                "receptions": [0, 0, 0],
                "attempts": [0, 0, 0],
                "carries": [3, 4, 2],
                "targets": [0, 0, 0],
            }
        )
        context = build_team_position_context(stats, 2025)
        old = context[context["team"] == "OLD"].iloc[0]
        new = context[context["team"] == "NEW"].iloc[0]
        self.assertEqual(old["total_points"], 10.0)
        self.assertEqual(new["total_points"], 14.0)
        self.assertEqual(new["contributors"], 2)
        self.assertEqual(new["total_opportunities"], 6)
        player_team = build_player_team_context(stats, 2025)
        moved = player_team[
            (player_team["player_id"] == "p1") & (player_team["team"] == "NEW")
        ].iloc[0]
        self.assertEqual(moved["player_points"], 8.0)
        self.assertEqual(moved["player_opportunities"], 4)


if __name__ == "__main__":
    unittest.main()
