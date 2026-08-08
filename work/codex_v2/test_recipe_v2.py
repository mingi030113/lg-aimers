from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from recipe import (
    RecipeError,
    audit_model_meta,
    audit_train_script,
    blend_logits,
    component_weights,
    fit_group_shifts,
    fold_train_bounds,
    load_recipe,
    make_fold_masks,
    sigmoid,
    solve_shift,
)


HERE = Path(__file__).resolve().parent


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recipe = load_recipe(HERE / "recipe_current.json")

    def test_effective_weights_match_current_final_model(self) -> None:
        weights = component_weights(self.recipe)
        self.assertEqual(set(weights), {"lgbm", "logistic", "mlp1", "mlp2"})
        self.assertAlmostEqual(weights["lgbm"], 0.39)
        self.assertAlmostEqual(weights["logistic"], 0.21)
        self.assertAlmostEqual(weights["mlp1"], 0.20)
        self.assertAlmostEqual(weights["mlp2"], 0.20)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_rolling_window_reproduces_final_window(self) -> None:
        self.assertEqual(fold_train_bounds(self.recipe, 2023), (2021, 2022))
        self.assertEqual(fold_train_bounds(self.recipe, 2024), (2022, 2023))
        self.assertEqual(fold_train_bounds(self.recipe, 2025), (2023, 2024))

    def test_old_f_is_excluded_from_training_only(self) -> None:
        frame = pd.DataFrame(
            {
                "season": [2021, 2021, 2022, 2022, 2023, 2023, 2024],
                "game_type": ["R", "F", "R", "F", "R", "F", "R"],
            }
        )
        masks23 = make_fold_masks(frame, self.recipe, 2023)
        self.assertEqual(np.flatnonzero(masks23["train"]).tolist(), [0, 2])
        self.assertEqual(np.flatnonzero(masks23["evaluation"]).tolist(), [4, 5])
        masks24 = make_fold_masks(frame, self.recipe, 2024)
        self.assertEqual(np.flatnonzero(masks24["train"]).tolist(), [2, 4, 5])

    def test_shift_solver_hits_requested_mean(self) -> None:
        z = np.array([-2.0, -0.5, 0.2, 1.5])
        target = 0.43
        delta = solve_shift(z, target)
        self.assertAlmostEqual(float(sigmoid(z + delta).mean()), target, places=12)

    def test_blend_uses_logit_space_and_exact_weights(self) -> None:
        ones = np.ones(3)
        result = blend_logits(
            self.recipe,
            1 * ones,
            2 * ones,
            3 * ones,
            4 * ones,
        )
        expected = 0.39 * 1 + 0.21 * 2 + 0.20 * 3 + 0.20 * 4
        np.testing.assert_allclose(result, expected)

    def test_group_shift_uses_drift_minus_d_auto(self) -> None:
        n = 1200
        frame = pd.DataFrame(
            {
                "season": np.full(2 * n, 2023),
                "is_F": np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)],
                "control_success": np.r_[
                    np.tile([0, 1], n // 2), np.tile([0, 1], n // 2)
                ],
            }
        )
        z = np.zeros(2 * n)
        shifts = fit_group_shifts(frame, z, self.recipe, 2023)
        expected_r = 0.5 - 0.011 - (-0.0029)
        expected_f = 0.5 - 0.011 - (-0.0065)
        self.assertAlmostEqual(shifts["R"]["target_mean"], expected_r)
        self.assertAlmostEqual(shifts["F"]["target_mean"], expected_f)
        self.assertAlmostEqual(float(sigmoid(shifts["R"]["shift"])), expected_r)
        self.assertAlmostEqual(float(sigmoid(shifts["F"]["shift"])), expected_f)

    def test_2022_f_shift_is_intentionally_unavailable(self) -> None:
        n = 1200
        frame = pd.DataFrame(
            {
                "season": np.full(2 * n, 2022),
                "is_F": np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)],
                "control_success": np.tile([0, 1], n),
            }
        )
        shifts = fit_group_shifts(frame, np.zeros(2 * n), self.recipe, 2022)
        self.assertIsNotNone(shifts["R"])
        self.assertIsNone(shifts["F"])

    def test_model_meta_audit_detects_mlp3_mismatch(self) -> None:
        models = self.recipe["models"]
        meta = {
            "n_seeds": models["lgbm"]["n_seeds"],
            "blend_w": models["logistic"]["weight_within_base"],
            "logreg_C": models["logistic"]["C"],
            "use_trackman": True,
            "train_from": 2023,
            "decay_halflife": None,
            "use_mlp": True,
            "mlp_w": 0.2,
            "use_mlp2": True,
            "mlp2_w": 0.2,
            "use_mlp3": False,
            "mlp3_w": 0.15,
            "use_prior": False,
            "prior_decay": 0.9,
            "count_cal": {},
            "drift": -0.011,
            "d_auto": {"R": -0.0029, "F": -0.0065},
        }
        self.assertEqual(audit_model_meta(self.recipe, meta), [])
        meta["use_mlp3"] = True
        self.assertTrue(any("use_mlp3" in item for item in audit_model_meta(self.recipe, meta)))

    def test_train_script_is_parsed_without_importing_it(self) -> None:
        models = self.recipe["models"]
        constants = {
            "USE_TRACKMAN": True,
            "TRAIN_FROM": 2023,
            "DECAY_HALFLIFE": None,
            "USE_PRIOR": False,
            "PRIOR_DECAY": 0.9,
            "COUNT_CAL": {},
            "USE_MLP": True,
            "MLP_HIDDEN": tuple(models["mlp1"]["hidden"]),
            "MLP_ALPHA": 1.0,
            "MLP_SEEDS": 3,
            "MLP_W": 0.2,
            "USE_MLP2": True,
            "MLP2_W": 0.2,
            "USE_MLP3": False,
            "MLP3_W": 0.15,
            "ROUNDS": 100,
            "N_SEEDS": 5,
            "LOGREG_C": 0.01,
            "BLEND_W": 0.35,
            "DRIFT": -0.011,
        }
        lines = [f"{name} = {value!r}" for name, value in constants.items()]
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "train.py"
            script.write_text("\n".join(lines), encoding="utf-8")
            self.assertEqual(audit_train_script(self.recipe, script), [])

    def test_invalid_weight_is_rejected(self) -> None:
        recipe = json.loads(json.dumps(self.recipe))
        recipe["models"]["mlp2"]["weight"] = 0.9
        with self.assertRaises(RecipeError):
            component_weights(recipe)


if __name__ == "__main__":
    unittest.main()
