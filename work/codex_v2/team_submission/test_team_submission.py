import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_candidate import build_candidate, required_model_files  # noqa: E402
from make_artifact import write_artifact  # noqa: E402
from runtime import _apply_team_residual_probability  # noqa: E402


class RuntimeTest(unittest.TestCase):
    def _artifact(self, root: Path, base_meta: Path) -> Path:
        oof = root / "oof.csv"
        pd.DataFrame(
            {
                "batter_team_id": [1, 1, 2, 2],
                "control_success": [1, 1, 0, 0],
                "prediction": [0.5, 0.5, 0.5, 0.5],
            }
        ).to_csv(oof, index=False)
        artifact = root / "artifact"
        write_artifact(
            oof_path=oof,
            output_dir=artifact,
            target_col="control_success",
            prediction_col="prediction",
            shrinkage=2.0,
            max_abs_correction=0.5 if False else 0.10,
            base_meta_path=base_meta,
            fit_description="unit-test OOF",
        )
        return artifact

    @staticmethod
    def _meta(weight: float) -> dict:
        return {
            "team_residual": {
                "enabled": True,
                "group_col": "batter_team_id",
                "applies_to_game_type": "R",
                "weight": weight,
                "correction_space": "probability",
                "application_order": "after_base_calibration",
                "unknown_correction": 0.0,
                "table_file": "team_residual.csv",
                "manifest_file": "team_residual_manifest.json",
            }
        }

    def test_probability_space_and_unknown_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_meta = root / "meta.json"
            base_meta.write_text("{}", encoding="utf-8")
            artifact = self._artifact(root, base_meta)
            rows = pd.DataFrame(
                {"batter_team_id": [1, 99, 2], "game_type": ["R", "R", "F"]}
            )
            p = np.array([0.20, 0.40, 0.80])
            got = _apply_team_residual_probability(rows, p, self._meta(0.5), artifact)
            # Corrections are capped at +/-0.10 by this fixture and are added
            # after calibration directly in probability space.
            np.testing.assert_array_equal(got, [0.25, 0.40, 0.80])

    def test_destination_rows_do_not_affect_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_meta = root / "meta.json"
            base_meta.write_text("{}", encoding="utf-8")
            artifact = self._artifact(root, base_meta)
            one = _apply_team_residual_probability(
                pd.DataFrame({"batter_team_id": [1], "game_type": ["R"]}),
                np.array([0.3]),
                self._meta(0.5),
                artifact,
            )
            many = _apply_team_residual_probability(
                pd.DataFrame(
                    {
                        "batter_team_id": [1, 2, 2, 99, 1],
                        "game_type": ["R", "R", "R", "R", "R"],
                    }
                ),
                np.array([0.3, 0.9, 0.1, 0.4, 0.8]),
                self._meta(0.5),
                artifact,
            )
            self.assertEqual(one[0], many[0])

    def test_weight_zero_is_bit_identical_and_does_not_open_artifact(self):
        p = np.array([0.123456789, 0.987654321])
        rows = pd.DataFrame({"batter_team_id": [1, 2], "game_type": ["R", "F"]})
        got = _apply_team_residual_probability(
            rows,
            p,
            self._meta(0.0),
            Path("definitely-does-not-exist"),
        )
        self.assertIs(got, p)
        self.assertEqual(got.tobytes(), p.tobytes())

    def test_manifest_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_meta = root / "meta.json"
            base_meta.write_text("{}", encoding="utf-8")
            artifact = self._artifact(root, base_meta)
            with (artifact / "team_residual.csv").open("a", encoding="utf-8") as fp:
                fp.write("3,0.01\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _apply_team_residual_probability(
                    pd.DataFrame({"batter_team_id": [1], "game_type": ["R"]}),
                    np.array([0.5]),
                    self._meta(1.0),
                    artifact,
                )


class BuilderTest(unittest.TestCase):
    def test_model_allowlist_tracks_active_flags(self):
        meta = {
            "n_seeds": 2,
            "use_trackman": True,
            "use_mlp": True,
            "use_mlp2": True,
            "use_mlp3": False,
            "use_prior": False,
        }
        self.assertEqual(
            required_model_files(meta),
            [
                "meta.json",
                "linear.npz",
                "lgb_0.txt",
                "lgb_1.txt",
                "tm_prior.csv",
                "mlp.npz",
                "mlp2.npz",
            ],
        )

    def test_build_is_isolated_and_ignores_stale_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            model = base / "model"
            model.mkdir(parents=True)
            meta = {
                "features": ["batter_team_id"],
                "n_seeds": 1,
                "use_trackman": False,
                "use_mlp": False,
                "use_mlp2": False,
                "use_mlp3": False,
                "use_prior": False,
            }
            (model / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (model / "linear.npz").write_bytes(b"linear")
            (model / "lgb_0.txt").write_text("tree", encoding="utf-8")
            (model / "stale_mlp3.npz").write_bytes(b"stale")
            (base / "script.py").write_text(
                "import numpy as np\nimport pandas as pd\n"
                "def _sigmoid(x): return x\n\n"
                "def main():\n"
                "    test = pd.DataFrame()\n    meta = {}\n    MODEL_DIR = '.'\n"
                "    z = np.array([0.5])\n"
                + "    p = np.clip(_sigmoid(z), 1e-6, 1 - 1e-6)\n",
                encoding="utf-8",
            )
            (base / "script_template.py").write_text(
                "__FEATURES_SOURCE__\n    p = np.clip(_sigmoid(z), 1e-6, 1 - 1e-6)\n",
                encoding="utf-8",
            )
            (base / "bundle.py").write_text(
                'REQUIREMENTS = "lightgbm==4.7.0\\n"\n', encoding="utf-8"
            )
            oof = root / "oof.csv"
            pd.DataFrame(
                {
                    "batter_team_id": [1, 1, 2, 2],
                    "control_success": [1, 1, 0, 0],
                    "prediction": [0.5, 0.5, 0.5, 0.5],
                }
            ).to_csv(oof, index=False)
            artifact = root / "artifact"
            write_artifact(
                oof_path=oof,
                output_dir=artifact,
                target_col="control_success",
                prediction_col="prediction",
                shrinkage=2.0,
                max_abs_correction=0.10,
                base_meta_path=model / "meta.json",
                fit_description="builder test",
            )
            output = root / "candidate"
            build_candidate(
                base_work=base,
                artifact_dir=artifact,
                output_dir=output,
                weight=0.0,
            )
            self.assertFalse((output / "model" / "stale_mlp3.npz").exists())
            self.assertEqual(
                sorted(p.name for p in (output / "model").iterdir()),
                [
                    "lgb_0.txt",
                    "linear.npz",
                    "meta.json",
                    "team_residual.csv",
                    "team_residual_manifest.json",
                ],
            )
            candidate_meta = json.loads((output / "model" / "meta.json").read_text())
            self.assertEqual(candidate_meta["team_residual"]["weight"], 0.0)
            self.assertIn("_apply_team_residual_probability", (output / "script.py").read_text())


if __name__ == "__main__":
    unittest.main()
