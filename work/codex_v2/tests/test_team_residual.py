import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from team_residual import apply_group_residual, fit_group_residual  # noqa: E402


class TeamResidualTest(unittest.TestCase):
    def test_fit_uses_oof_residual_and_shrinks(self):
        oof = pd.DataFrame(
            {
                "team": [1, 1, 2, 2],
                "y": [1.0, 1.0, 0.0, 0.0],
                "p": [0.5, 0.5, 0.5, 0.5],
            }
        )
        table = fit_group_residual(
            oof,
            group_col="team",
            target_col="y",
            prediction_col="p",
            shrinkage=2,
            max_abs_correction=1,
        ).set_index("team")
        self.assertAlmostEqual(table.loc[1, "residual_correction"], 0.25)
        self.assertAlmostEqual(table.loc[2, "residual_correction"], -0.25)
        self.assertAlmostEqual(table.loc[1, "residual_reliability"], 0.5)

    def test_apply_does_not_aggregate_destination_rows(self):
        table = pd.DataFrame(
            {
                "team": [1, 2],
                "residual_correction": [0.1, -0.2],
            }
        )
        rows = pd.DataFrame({"team": [2, 99, 1], "p": [0.5, 0.5, 0.5]})
        result = apply_group_residual(
            rows,
            table,
            group_col="team",
            prediction_col="p",
            weight=0.5,
        )
        np.testing.assert_allclose(result, [0.4, 0.5, 0.55])

    def test_rejects_duplicate_mapping(self):
        table = pd.DataFrame(
            {"team": [1, 1], "residual_correction": [0.1, 0.2]}
        )
        rows = pd.DataFrame({"team": [1], "p": [0.5]})
        with self.assertRaisesRegex(ValueError, "unique"):
            apply_group_residual(
                rows,
                table,
                group_col="team",
                prediction_col="p",
            )


if __name__ == "__main__":
    unittest.main()
