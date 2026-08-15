import tempfile
import unittest
from pathlib import Path

from fantasyoptimizer.utils.data_loader import (
    available_years,
    build_adp_movement,
    data_quality_report,
    join_year,
)


class DataLoaderTests(unittest.TestCase):
    def test_adp_movement_uses_preserved_snapshots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_dir = Path(temp_dir) / "2026" / "snapshots"
            snapshot_dir.mkdir(parents=True)
            for stamp, adp in [("20260801T000000Z", 50.0), ("20260808T000000Z", 40.0)]:
                path = snapshot_dir / f"ADP_{stamp}.csv"
                path.write_text(
                    f"Rank,Player,Team,POS,AVG\n1,Example Player,NE,RB,{adp}\n",
                    encoding="utf-8",
                )
                path.with_suffix(".meta.json").write_text(
                    '{"fetched_at_utc": "'
                    + ("2026-08-01T00:00:00+00:00" if adp == 50 else "2026-08-08T00:00:00+00:00")
                    + '"}\n',
                    encoding="utf-8",
                )
            movement = build_adp_movement(2026, temp_dir)
            self.assertEqual(movement.loc[0, "adp_movement"], 10.0)
            self.assertEqual(movement.loc[0, "snapshots"], 2)

    def test_complete_year_and_position_normalization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            year_dir = Path(temp_dir) / "2024"
            year_dir.mkdir()
            (year_dir / "Pre_2024_ADP(HPPR).csv").write_text(
                "Rank,Player,Team,POS,AVG\n1,Example Player,NA,RB12,12.0\n",
                encoding="utf-8",
            )
            (year_dir / "Post_2024_Results(HPPR).csv").write_text(
                "Rank,Player,Team,Pos,AVG,TTL\n3,Example Player,NA,RB,10.0,170.0\n",
                encoding="utf-8",
            )

            self.assertEqual(available_years(temp_dir), [2024])
            result = join_year(2024, temp_dir)
            self.assertEqual(result.loc[0, "pos"], "RB")
            self.assertEqual(result.loc[0, "team_adp"], "NA")
            self.assertEqual(result.loc[0, "year"], 2024)

    def test_suffix_and_punctuation_name_variants_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            year_dir = Path(temp_dir) / "2024"
            year_dir.mkdir()
            (year_dir / "Pre_2024_ADP(HPPR).csv").write_text(
                "Rank,Player,Team,POS,AVG\n1,Example Player Jr.,NE,WR1,10.0\n",
                encoding="utf-8",
            )
            (year_dir / "Post_2024_Results(HPPR).csv").write_text(
                "Rank,Player,Team,Pos,AVG,TTL\n1,Example Player,NE,WR,15.0,255.0\n",
                encoding="utf-8",
            )
            result = join_year(2024, temp_dir)
            self.assertEqual(len(result), 1)

    def test_quality_report_exposes_unmatched_players(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            year_dir = Path(temp_dir) / "2024"
            year_dir.mkdir()
            (year_dir / "Pre_2024_ADP(HPPR).csv").write_text(
                "Rank,Player,Team,POS,AVG\n1,Matched Player,NE,RB1,1.0\n"
                "2,Missing Player,NE,WR1,2.0\n",
                encoding="utf-8",
            )
            (year_dir / "Post_2024_Results(HPPR).csv").write_text(
                "Rank,Player,Team,Pos,AVG,TTL\n1,Matched Player,NE,RB,10,170\n",
                encoding="utf-8",
            )
            report = data_quality_report([2024], temp_dir).iloc[0]
            self.assertEqual(report["matched_players"], 1)
            self.assertEqual(report["adp_match_rate"], 0.5)
            self.assertIn("missing player", report["adp_only"])


if __name__ == "__main__":
    unittest.main()
