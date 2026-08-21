import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.import_nflverse_results import (
    build_half_ppr_results,
    build_player_team_context,
    build_team_position_context,
)
from scripts.import_ffc_adp import build_adp_export, write_adp_export
from scripts.import_nflverse_players import build_player_export
from scripts.import_nflverse_roles import (
    adp_cutoff,
    build_depth_export,
    build_role_export,
    write_role_export,
)
from fantasyoptimizer.utils.atomic import write_csv


class NflverseImporterTests(unittest.TestCase):
    def test_dated_depth_chart_uses_last_snapshot_before_adp_cutoff(self):
        depth = pd.DataFrame(
            [
                {
                    "dt": "2026-08-14T08:00:00Z",
                    "team": "EX",
                    "player_name": "Example Receiver",
                    "gsis_id": "p1",
                    "pos_abb": "WR",
                    "pos_slot": 1,
                    "pos_rank": 3,
                    "pos_name": "Wide Receiver",
                },
                {
                    "dt": "2026-08-15T08:00:00Z",
                    "team": "EX",
                    "player_name": "Example Receiver",
                    "gsis_id": "p1",
                    "pos_abb": "WR",
                    "pos_slot": 1,
                    "pos_rank": 4,
                    "pos_name": "Wide Receiver",
                },
            ]
        )
        cutoff = adp_cutoff({"source_meta": {"end_date": "2026-08-14"}})
        result, metadata = build_depth_export(depth, 2026, cutoff)
        self.assertEqual(result.loc[0, "depth_rank"], 3)
        self.assertEqual(result.loc[0, "official_starter"], 1)
        self.assertEqual(metadata["timing_quality"], "adp_aligned")
        self.assertTrue(metadata["point_in_time_eligible"])

    def test_role_export_joins_week_one_roster_status(self):
        depth = pd.DataFrame(
            [
                {
                    "dt": "2026-08-14T08:00:00Z",
                    "team": "EX",
                    "player_name": "Example Runner",
                    "gsis_id": "p1",
                    "pos_abb": "RB",
                    "pos_slot": 11,
                    "pos_rank": 1,
                    "pos_name": "Running Back",
                }
            ]
        )
        roster = pd.DataFrame(
            [
                {
                    "season": 2026,
                    "team": "EX",
                    "position": "RB",
                    "full_name": "Example Runner",
                    "gsis_id": "p1",
                    "week": 1,
                    "game_type": "REG",
                    "status": "ACT",
                    "status_description_abbr": "Active",
                }
            ]
        )
        result, _ = build_role_export(depth, roster, 2026)
        self.assertEqual(result.loc[0, "roster_status"], "ACT")

    def test_role_import_preserves_timestamped_snapshot(self):
        role = pd.DataFrame([{"player": "Example Runner", "pos": "RB"}])
        metadata = {
            "fetched_at_utc": datetime(
                2026, 8, 14, tzinfo=timezone.utc
            ).isoformat()
        }
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "2026" / "Preseason_Role_2026.csv"
            snapshot = write_role_export(role, destination, metadata)
            self.assertTrue(destination.exists())
            self.assertTrue(snapshot.exists())
            self.assertTrue(snapshot.with_suffix(".meta.json").exists())

    def test_player_metadata_keeps_supported_position_and_draft_fields(self):
        players = pd.DataFrame(
            [
                {
                    "gsis_id": "p1",
                    "display_name": "Example Runner",
                    "birth_date": "2000-01-01",
                    "position_group": "RB",
                    "position": "RB",
                    "height": 72,
                    "weight": 210,
                    "college_name": "Example U",
                    "rookie_season": 2023,
                    "last_season": 2026,
                    "latest_team": "EX",
                    "status": "ACT",
                    "years_of_experience": 3,
                    "draft_year": 2023,
                    "draft_round": 2,
                    "draft_pick": 50,
                    "draft_team": "EX",
                },
                {
                    "gsis_id": "p2",
                    "display_name": "Example Defender",
                    "birth_date": "2000-01-01",
                    "position_group": "DB",
                    "position": "CB",
                    "height": 72,
                    "weight": 200,
                    "college_name": "Example U",
                    "rookie_season": 2023,
                    "last_season": 2026,
                    "latest_team": "EX",
                    "status": "ACT",
                    "years_of_experience": 3,
                    "draft_year": 2023,
                    "draft_round": 2,
                    "draft_pick": 51,
                    "draft_team": "EX",
                },
            ]
        )
        result = build_player_export(players)
        self.assertEqual(result["display_name"].tolist(), ["Example Runner"])
        self.assertEqual(result.loc[0, "draft_pick"], 50)

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

    def test_adp_import_preserves_timestamped_snapshot(self):
        adp = pd.DataFrame(
            [{"Rank": 1, "Player": "Example", "POS": "RB", "AVG": 1.0}]
        )
        metadata = {
            "fetched_at_utc": datetime(2026, 8, 14, tzinfo=timezone.utc).isoformat()
        }
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "2026" / "Pre_2026_ADP(HPPR).csv"
            snapshot = write_adp_export(adp, destination, metadata)
            self.assertTrue(destination.exists())
            self.assertIsNotNone(snapshot)
            self.assertTrue(snapshot.exists())
            self.assertTrue(snapshot.with_suffix(".meta.json").exists())

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


class AtomicWriteTests(unittest.TestCase):
    def test_failed_write_leaves_the_existing_archive_intact(self):
        class ExplodingFrame(pd.DataFrame):
            def to_csv(self, *args, **kwargs):
                raise OSError("connection dropped mid-write")

        with TemporaryDirectory() as directory:
            destination = Path(directory) / "2026" / "Post_2026_Results(HPPR).csv"
            write_csv(pd.DataFrame([{"player": "keep me"}]), destination, index=False)
            with self.assertRaises(OSError):
                write_csv(ExplodingFrame(), destination, index=False)
            self.assertIn("keep me", destination.read_text(encoding="utf-8"))
            leftovers = [path.name for path in destination.parent.iterdir()]
            self.assertEqual(leftovers, [destination.name])


if __name__ == "__main__":
    unittest.main()
