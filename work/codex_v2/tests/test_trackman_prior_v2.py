from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from trackman_prior_v2 import (  # noqa: E402
    NUMERIC_FEATURES,
    ValidationError,
    build_prior,
    build_season_summary,
    compare_priors,
    main,
    validate_many_to_one_attachment,
    validate_raw_trackman,
)


def row(pitcher, season, group, speed, spin=2200.0, **overrides):
    values = {
        "pitcher_trackman_id": pitcher,
        "season": season,
        "pitch_type_group": group,
        "rel_speed": speed,
        "spin_rate": spin,
        "induced_vert_break": 10.0,
        "horz_break": -5.0,
        "extension": 1.8,
        "rel_height": 1.7,
        "rel_side": -0.4,
        "zone_speed": 130.0,
    }
    values.update(overrides)
    return values


class TrackmanPriorTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame(
            [
                # Fastball has two rows but only one non-null speed.  Its speed
                # must therefore receive weight 1, not the shared group size 2.
                row("A", 2019, "fastball", 150.0, spin=2400.0),
                row("A", 2019, "fastball", np.nan, spin=2500.0),
                row("A", 2019, "offspeed", 120.0, spin=1800.0),
                row("A", 2020, "breaking", 90.0, spin=1900.0),
                row("B", 2020, "fastball", 145.0, spin=2300.0),
                # Unsupported groups remain accounted for but are not aggregated.
                row("A", 2020, "unknown", 199.0, spin=3000.0),
            ]
        )

    def test_feature_specific_non_null_denominator(self):
        summary = build_season_summary(self.fixture())
        a2019 = summary.values.query("pitcher_trackman_id == 'A' and season == 2019").iloc[0]
        support = summary.supports.query(
            "pitcher_trackman_id == 'A' and season == 2019"
        ).iloc[0]
        self.assertAlmostEqual(a2019["m_rel_speed"], 135.0)
        self.assertEqual(support["m_rel_speed"], 2.0)
        self.assertEqual(a2019["tm_n"], 3)
        self.assertEqual(summary.eligible_rows, 5)
        self.assertEqual(summary.ignored_rows, 1)

    def test_cumulative_prior_uses_each_features_support(self):
        summary = build_season_summary(self.fixture())
        prior = build_prior(summary, [2020, 2021])
        a2021 = prior.query("pitcher_trackman_id == 'A' and season == 2021").iloc[0]
        # 2019 contributes two valid speeds (150, 120), 2020 contributes one (90).
        self.assertAlmostEqual(a2021["tm_m_rel_speed"], 120.0)
        # No 2020 fastball/offspeed is present, so those values are not diluted.
        self.assertAlmostEqual(a2021["tm_v_fastball"], 150.0)
        self.assertAlmostEqual(a2021["tm_v_offspeed"], 120.0)
        self.assertAlmostEqual(a2021["tm_velo_sep"], 30.0)
        self.assertEqual(a2021["tm_prior_n"], 4)
        self.assertEqual(len(prior), 3)  # A@2020, A@2021, B@2021
        self.assertFalse(prior.duplicated(["pitcher_trackman_id", "season"]).any())

    def test_physical_guardrail_fails_fast(self):
        frame = self.fixture()
        frame.loc[0, "rel_speed"] = 999.0
        with self.assertRaisesRegex(ValidationError, "rel_speed"):
            validate_raw_trackman(frame)

    def test_many_to_one_attachment_preserves_rows(self):
        prior = build_prior(build_season_summary(self.fixture()), [2021])
        main_data = pd.DataFrame(
            {"pitcher_id": [1, 1, 2, 3], "season": [2021, 2021, 2021, 2021]}
        )
        id_map = pd.DataFrame(
            {
                "pitcher_id": [1, 2],
                "tm_pitcher_id": ["A", "B"],
                "share": [0.9, 0.8],
                "votes": [20, 10],
            }
        )
        report = validate_many_to_one_attachment(main_data, prior, id_map)
        self.assertEqual(report["main_rows_before"], report["main_rows_after"])
        self.assertEqual(report["prior_coverage_rows"], 3)

        duplicated = pd.concat([prior, prior.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValidationError, "duplicated"):
            validate_many_to_one_attachment(main_data, duplicated, id_map)

    def test_old_new_comparison_reports_missing_and_changed_values(self):
        candidate = build_prior(build_season_summary(self.fixture()), [2021])
        reference = candidate.copy()
        reference.loc[0, "tm_v_fastball"] = 30.0
        reference.loc[0, "tm_v_offspeed"] = np.nan
        report = compare_priors(reference, candidate)
        self.assertEqual(report["reference_only_keys"], 0)
        self.assertEqual(report["features"]["tm_v_fastball"]["changed"], 1)
        self.assertEqual(
            report["features"]["tm_v_offspeed"]["candidate_only_valid"], 1
        )

    def test_cli_writes_separate_candidate_and_manifest_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trackman.csv"
            output = root / "candidate.csv"
            manifest = root / "candidate.manifest.json"
            self.fixture().to_csv(source, index=False)
            code = main(
                [
                    "--trackman",
                    str(source),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--target-seasons",
                    "2020,2021",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(output.exists())
            self.assertTrue(manifest.exists())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_kind"], "candidate_trackman_prior_v2")
            self.assertIn("candidate_only", payload["adoption_policy"])
            self.assertEqual(main([
                "--trackman", str(source), "--output", str(output),
                "--manifest", str(manifest), "--target-seasons", "2020,2021"
            ]), 2)

    def test_every_numeric_feature_is_required(self):
        frame = self.fixture().drop(columns=[NUMERIC_FEATURES[-1]])
        with self.assertRaisesRegex(ValidationError, NUMERIC_FEATURES[-1]):
            build_season_summary(frame)


if __name__ == "__main__":
    unittest.main()
