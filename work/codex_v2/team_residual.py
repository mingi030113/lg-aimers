"""Small, cross-fitted group residual corrections.

The table must be fitted from out-of-fold predictions, never in-sample model
predictions.  Evaluation/test rows are only mapped to a precomputed correction;
they are not counted or aggregated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_group_residual(
    oof: pd.DataFrame,
    *,
    group_col: str,
    target_col: str,
    prediction_col: str,
    shrinkage: float = 1000.0,
    max_abs_correction: float = 0.04,
) -> pd.DataFrame:
    required = {group_col, target_col, prediction_col}
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if not np.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("shrinkage must be positive and finite")
    if not np.isfinite(max_abs_correction) or max_abs_correction <= 0:
        raise ValueError("max_abs_correction must be positive and finite")
    if oof[group_col].isna().any():
        raise ValueError(f"{group_col} contains missing values")

    y = oof[target_col].to_numpy(np.float64)
    p = oof[prediction_col].to_numpy(np.float64)
    if not np.isfinite(p).all() or not ((p > 0) & (p < 1)).all():
        raise ValueError("OOF predictions must be finite probabilities in (0, 1)")
    if not np.isfinite(y).all() or not ((y >= 0) & (y <= 1)).all():
        raise ValueError("targets must be finite values in [0, 1]")

    residual = y - p
    centered = residual - residual.mean()
    grouped = (
        pd.DataFrame({group_col: oof[group_col].to_numpy(), "residual": centered})
        .groupby(group_col, sort=True, observed=True)["residual"]
        .agg(["sum", "size"])
        .rename(columns={"size": "residual_n"})
        .reset_index()
    )
    grouped["residual_correction"] = (
        grouped["sum"] / (grouped["residual_n"] + shrinkage)
    ).clip(-max_abs_correction, max_abs_correction)
    grouped["residual_reliability"] = grouped["residual_n"] / (
        grouped["residual_n"] + shrinkage
    )
    grouped["fit_global_residual"] = float(residual.mean())
    return grouped[
        [
            group_col,
            "residual_n",
            "residual_correction",
            "residual_reliability",
            "fit_global_residual",
        ]
    ]


def apply_group_residual(
    rows: pd.DataFrame,
    table: pd.DataFrame,
    *,
    group_col: str,
    prediction_col: str,
    weight: float = 1.0,
) -> np.ndarray:
    if group_col not in rows or prediction_col not in rows:
        raise ValueError(f"rows must contain {group_col!r} and {prediction_col!r}")
    required = {group_col, "residual_correction"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"table is missing columns: {missing}")
    if table.duplicated(group_col).any():
        raise ValueError(f"table must be unique by {group_col}")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be finite and in [0, 1]")

    mapping = table.set_index(group_col)["residual_correction"]
    correction = rows[group_col].map(mapping).fillna(0.0).to_numpy(np.float64)
    p = rows[prediction_col].to_numpy(np.float64)
    if not np.isfinite(p).all() or not ((p > 0) & (p < 1)).all():
        raise ValueError("predictions must be finite probabilities in (0, 1)")
    return np.clip(p + weight * correction, 1e-7, 1 - 1e-7)
