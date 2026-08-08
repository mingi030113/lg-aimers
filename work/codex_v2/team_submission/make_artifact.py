"""Fit and export an allowlisted batter-team correction artifact from OOF rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TABLE_FILE = "team_residual.csv"
MANIFEST_FILE = "team_residual_manifest.json"
TABLE_COLUMNS = ["batter_team_id", "residual_correction"]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_team_ids(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("batter_team_id must contain finite integer IDs")
    return numeric.astype(np.int64)


def fit_table(
    oof: pd.DataFrame,
    *,
    target_col: str,
    prediction_col: str,
    shrinkage: float,
    max_abs_correction: float,
) -> tuple[pd.DataFrame, dict]:
    required = {"batter_team_id", target_col, prediction_col}
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"OOF file is missing required columns: {missing}")
    if not np.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("shrinkage must be positive and finite")
    if not np.isfinite(max_abs_correction) or not 0 < max_abs_correction <= 0.10:
        raise ValueError("max_abs_correction must be in (0, 0.10]")

    team_ids = _integer_team_ids(oof["batter_team_id"])
    y = pd.to_numeric(oof[target_col], errors="coerce").to_numpy(np.float64)
    p = pd.to_numeric(oof[prediction_col], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(y).all() or np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("OOF targets must be finite values in [0, 1]")
    if not np.isfinite(p).all() or np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("OOF predictions must be finite probabilities in (0, 1)")

    residual = y - p
    centered = residual - residual.mean()
    grouped = (
        pd.DataFrame({"batter_team_id": team_ids, "residual": centered})
        .groupby("batter_team_id", sort=True, observed=True)["residual"]
        .agg(["sum", "size"])
        .reset_index()
    )
    grouped["residual_correction"] = (
        grouped["sum"] / (grouped["size"] + float(shrinkage))
    ).clip(-max_abs_correction, max_abs_correction)
    table = grouped[TABLE_COLUMNS].copy()
    table["batter_team_id"] = table["batter_team_id"].astype(np.int64)
    stats = {
        "oof_rows": int(len(oof)),
        "global_residual": float(residual.mean()),
        "shrinkage": float(shrinkage),
        "group_counts": {
            str(int(team)): int(size)
            for team, size in zip(grouped["batter_team_id"], grouped["size"])
        },
    }
    return table, stats


def write_artifact(
    *,
    oof_path: Path,
    output_dir: Path,
    target_col: str,
    prediction_col: str,
    shrinkage: float,
    max_abs_correction: float,
    base_meta_path: Path,
    fit_description: str,
) -> tuple[Path, Path]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite artifact directory: {output_dir}")
    if not fit_description.strip():
        raise ValueError("fit_description is required for provenance")
    oof = pd.read_csv(
        oof_path,
        usecols=["batter_team_id", target_col, prediction_col],
    )
    table, stats = fit_table(
        oof,
        target_col=target_col,
        prediction_col=prediction_col,
        shrinkage=shrinkage,
        max_abs_correction=max_abs_correction,
    )
    output_dir.mkdir(parents=True)
    table_path = output_dir / TABLE_FILE
    table.to_csv(table_path, index=False, lineterminator="\n", float_format="%.17g")
    manifest = {
        "schema_version": 1,
        "artifact_type": "batter_team_probability_residual",
        "group_col": "batter_team_id",
        "applies_to_game_type": "R",
        "correction_space": "probability",
        "application_order": "after_base_calibration",
        "unknown_correction": 0.0,
        "table_file": TABLE_FILE,
        "table_sha256": sha256_path(table_path),
        "columns": TABLE_COLUMNS,
        "n_rows": int(len(table)),
        "allowed_group_values": [int(v) for v in table["batter_team_id"]],
        "max_abs_correction": float(max_abs_correction),
        "fit": {
            **stats,
            "description": fit_description.strip(),
            "oof_sha256": sha256_path(oof_path),
            "base_meta_sha256": sha256_path(base_meta_path),
            "target_col": target_col,
            "prediction_col": prediction_col,
        },
    }
    manifest_path = output_dir / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return table_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-meta", type=Path, required=True)
    parser.add_argument("--target-col", default="control_success")
    parser.add_argument("--prediction-col", default="prediction")
    parser.add_argument("--shrinkage", type=float, default=1000.0)
    parser.add_argument("--max-abs-correction", type=float, default=0.04)
    parser.add_argument("--fit-description", required=True)
    args = parser.parse_args()
    table, manifest = write_artifact(
        oof_path=args.oof.resolve(),
        output_dir=args.output_dir.resolve(),
        target_col=args.target_col,
        prediction_col=args.prediction_col,
        shrinkage=args.shrinkage,
        max_abs_correction=args.max_abs_correction,
        base_meta_path=args.base_meta.resolve(),
        fit_description=args.fit_description,
    )
    print(f"wrote {table}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
