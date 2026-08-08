import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from previous_season_pitcher import (  # noqa: E402
    attach_previous_season_pitcher_prior,
    build_previous_season_pitcher_prior,
)


class PreviousSeasonPitcherTest(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame(
            {
                "season": [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
                "game_type": ["R", "R", "F", "R", "R", "R", "R", "F"],
                "pitcher_id": [1, 1, 1, 2, 1, 2, 2, 2],
                "control_success": [1, 1, 0, 0, 0, 1, 1, 0],
            }
        )

    def test_uses_exactly_previous_regular_season(self):
        prior = build_previous_season_pitcher_prior(self.history, k=2)
        p1_2024 = prior[(prior.pitcher_id == 1) & (prior.season == 2024)].iloc[0]

        # 2023 R mean is 2/3. Pitcher 1 has 2/2; the 2023 F failure is excluded.
        expected = (2 + 2 * (2 / 3)) / (2 + 2)
        self.assertAlmostEqual(p1_2024.pitcher_prev_r_rate_eb, expected)
        self.assertEqual(p1_2024.source_season, 2023)

        p1_2025 = prior[(prior.pitcher_id == 1) & (prior.season == 2025)].iloc[0]
        # The 2025 feature sees the 2024 R failure, not the pitcher's 2023 labels.
        expected_2025 = (0 + 2 * (2 / 3)) / (1 + 2)
        self.assertAlmostEqual(p1_2025.pitcher_prev_r_rate_eb, expected_2025)
        self.assertEqual(p1_2025.source_season, 2024)

    def test_attach_preserves_order_and_marks_unknown_pitcher(self):
        prior = build_previous_season_pitcher_prior(self.history, k=2)
        rows = pd.DataFrame(
            {
                "row_id": ["b", "a"],
                "season": [2025, 2025],
                "pitcher_id": [99, 1],
            }
        )
        result = attach_previous_season_pitcher_prior(rows, prior)
        self.assertEqual(result.row_id.tolist(), ["b", "a"])
        self.assertEqual(result.pitcher_prev_r_missing.tolist(), [1, 0])
        self.assertEqual(result.pitcher_prev_r_n.tolist(), [0, 1])
        self.assertEqual(result.loc[0, "pitcher_prev_r_dev_eb"], 0.0)
        self.assertTrue(np.isnan(result.loc[0, "pitcher_prev_r_rate_eb"]))

    def test_rejects_duplicate_prior_keys(self):
        prior = build_previous_season_pitcher_prior(self.history, k=2)
        duplicate = pd.concat([prior, prior.iloc[[0]]], ignore_index=True)
        rows = pd.DataFrame({"season": [2024], "pitcher_id": [1]})
        with self.assertRaisesRegex(ValueError, "unique"):
            attach_previous_season_pitcher_prior(rows, duplicate)

    def test_rejects_invalid_shrinkage(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            build_previous_season_pitcher_prior(self.history, k=0)


if __name__ == "__main__":
    unittest.main()
