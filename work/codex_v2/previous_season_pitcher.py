"""Leakage-safe previous-season pitcher empirical-Bayes features.

This module never edits the supplied history.  A target-season row receives
statistics from exactly the previous regular season, so a 2025 row can only use
2024 labels.  The resulting table is intended to be evaluated as an isolated
challenger before it is added to the production feature set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "season",
    "game_type",
    "pitcher_id",
    "control_success",
}

FEATURE_COLUMNS = [
    "pitcher_prev_r_n",
    "pitcher_prev_r_rate_eb",
    "pitcher_prev_r_dev_eb",
    "pitcher_prev_r_reliability",
    "pitcher_prev_r_league_mean",
]


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def build_previous_season_pitcher_prior(
    history: pd.DataFrame,
    *,
    k: float = 500.0,
    target_seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Build one row per ``(pitcher_id, target season)``.

    Only regular-season rows from ``target season - 1`` contribute.  The
    empirical-Bayes rate is shrunk to that source season's regular-season mean.
    ``pitcher_prev_r_dev_eb`` is the most useful deployment feature: unknown
    pitchers can safely receive zero deviation and zero reliability.
    """

    _require_columns(history, REQUIRED_COLUMNS)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be a positive finite number")

    regular = history.loc[
        history["game_type"].eq("R") & history["control_success"].notna(),
        ["season", "pitcher_id", "control_success"],
    ].copy()
    if regular.empty:
        raise ValueError("history contains no labelled regular-season rows")
    if not regular["control_success"].between(0, 1).all():
        raise ValueError("control_success must be in [0, 1]")

    league = (
        regular.groupby("season", sort=True, observed=True)["control_success"]
        .mean()
        .rename("pitcher_prev_r_league_mean")
    )
    grouped = (
        regular.groupby(["season", "pitcher_id"], sort=True, observed=True)[
            "control_success"
        ]
        .agg(["sum", "size"])
        .rename(columns={"sum": "successes", "size": "pitcher_prev_r_n"})
        .reset_index()
        .merge(league.reset_index(), on="season", how="left", validate="many_to_one")
    )

    n = grouped["pitcher_prev_r_n"].to_numpy(np.float64)
    successes = grouped["successes"].to_numpy(np.float64)
    mean = grouped["pitcher_prev_r_league_mean"].to_numpy(np.float64)
    grouped["pitcher_prev_r_rate_eb"] = (successes + k * mean) / (n + k)
    grouped["pitcher_prev_r_dev_eb"] = grouped["pitcher_prev_r_rate_eb"] - mean
    grouped["pitcher_prev_r_reliability"] = n / (n + k)
    grouped["source_season"] = grouped["season"].astype(np.int64)
    grouped["season"] = grouped["source_season"] + 1

    if target_seasons is not None:
        wanted = {int(s) for s in target_seasons}
        grouped = grouped[grouped["season"].isin(wanted)].copy()

    out = grouped[
        ["pitcher_id", "season", "source_season", *FEATURE_COLUMNS]
    ].sort_values(["season", "pitcher_id"], kind="stable", ignore_index=True)
    if out.duplicated(["pitcher_id", "season"]).any():
        raise AssertionError("prior is not unique by (pitcher_id, season)")
    if not (out["source_season"] == out["season"] - 1).all():
        raise AssertionError("previous-season boundary was violated")
    return out


def attach_previous_season_pitcher_prior(
    rows: pd.DataFrame,
    prior: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the prior without changing row count or order."""

    _require_columns(rows, {"pitcher_id", "season"})
    _require_columns(prior, {"pitcher_id", "season", *FEATURE_COLUMNS})
    if prior.duplicated(["pitcher_id", "season"]).any():
        raise ValueError("prior must be unique by (pitcher_id, season)")

    marker = "__codex_row_order"
    if marker in rows.columns:
        raise ValueError(f"reserved column already exists: {marker}")
    left = rows.copy()
    left[marker] = np.arange(len(left), dtype=np.int64)
    result = left.merge(
        prior[["pitcher_id", "season", *FEATURE_COLUMNS]],
        on=["pitcher_id", "season"],
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values(marker, kind="stable")
    if len(result) != len(rows):
        raise AssertionError("prior merge changed row count")

    missing = result["pitcher_prev_r_n"].isna()
    result["pitcher_prev_r_missing"] = missing.astype(np.int8)
    result["pitcher_prev_r_n"] = result["pitcher_prev_r_n"].fillna(0).astype(np.int64)
    result["pitcher_prev_r_dev_eb"] = result["pitcher_prev_r_dev_eb"].fillna(0.0)
    result["pitcher_prev_r_reliability"] = result[
        "pitcher_prev_r_reliability"
    ].fillna(0.0)
    return result.drop(columns=[marker]).reset_index(drop=True)


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported input format: {path.suffix}")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"unsupported output format: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=float, default=500.0)
    parser.add_argument("--target-season", type=int, action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    history_path = args.history.resolve()
    output_path = args.output.resolve()
    if history_path == output_path:
        raise ValueError("output must differ from history")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = _read_frame(history_path)
    prior = build_previous_season_pitcher_prior(
        history,
        k=args.k,
        target_seasons=args.target_season,
    )
    _write_frame(prior, output_path)
    print(f"wrote {len(prior):,} rows to {output_path}")


if __name__ == "__main__":
    main()
