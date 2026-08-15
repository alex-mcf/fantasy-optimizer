import tempfile
import unittest
from pathlib import Path

from fantasyoptimizer.utils.data_loader import available_years, join_year


class DataLoaderTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

