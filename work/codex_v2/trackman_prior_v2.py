"""Build a leakage-safe Trackman prior with feature-specific denominators.

This module is deliberately separate from ``work/tm03_features.py``.  It writes
only to an explicit output path and refuses to overwrite an existing artifact
unless ``--force`` is supplied.

The important invariant is that every feature is averaged with the number of
non-null observations for *that feature*.  Missing pitch-type/season cells
therefore do not silently contribute zero through a shared ``tm_n`` divisor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


KEYS = ["pitcher_trackman_id", "season"]
GROUP_KEY = "pitch_type_group"
PITCH_GROUPS = ("fastball", "breaking", "offspeed")
NUMERIC_FEATURES = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
VARIATION_FEATURES = ("rel_height", "rel_side", "extension")

# These are intentionally broad guardrails, not model-training clipping bounds.
# They catch unit/schema corruption while allowing realistic KBO Trackman tails.
RAW_PHYSICAL_BOUNDS: Mapping[str, tuple[float, float]] = {
    "rel_speed": (40.0, 200.0),
    "zone_speed": (30.0, 200.0),
    "spin_rate": (0.0, 6000.0),
    # Raw history contains a handful of valid, finite values just beyond
    # +/-100 cm (IVB max 153.326; HB max 103.699).  +/-200 still catches an
    # obvious metres-vs-centimetres or column-shift corruption without rejecting
    # those observed tails.
    "induced_vert_break": (-200.0, 200.0),
    "horz_break": (-200.0, 200.0),
    "extension": (-1.0, 5.0),
    "rel_height": (-2.0, 4.0),
    "rel_side": (-4.0, 4.0),
}


class ValidationError(ValueError):
    """Raised when an input or generated artifact violates an invariant."""


@dataclass(frozen=True)
class SeasonSummary:
    """Per-pitcher-season values and matching feature support counts."""

    values: pd.DataFrame
    supports: pd.DataFrame
    input_rows: int
    eligible_rows: int
    ignored_rows: int


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValidationError(f"{label} is missing required columns: {missing}")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan, 0.0
    denominator = float(w[ok].sum())
    return float(np.dot(v[ok], w[ok]) / denominator), denominator


def validate_raw_trackman(
    trackman: pd.DataFrame,
    physical_bounds: Mapping[str, tuple[float, float]] = RAW_PHYSICAL_BOUNDS,
) -> dict:
    """Validate raw schema, key types, finiteness, and broad physical ranges."""
    _require_columns(trackman, [*KEYS, GROUP_KEY, *NUMERIC_FEATURES], "trackman")
    if trackman.empty:
        raise ValidationError("trackman input has zero rows")
    if trackman["pitcher_trackman_id"].isna().any():
        raise ValidationError("pitcher_trackman_id contains null values")

    seasons = pd.to_numeric(trackman["season"], errors="coerce")
    bad_season = seasons.isna() | ~np.isfinite(seasons) | (seasons != np.floor(seasons))
    if bad_season.any():
        sample = trackman.loc[bad_season, "season"].head(5).tolist()
        raise ValidationError(f"season must contain finite integers; examples={sample}")

    range_report: dict[str, dict] = {}
    for column, (lower, upper) in physical_bounds.items():
        raw = trackman[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        malformed = raw.notna() & numeric.isna()
        if malformed.any():
            sample = raw[malformed].head(5).tolist()
            raise ValidationError(f"{column} contains non-numeric values: {sample}")
        finite = numeric.dropna().to_numpy(dtype=np.float64)
        if finite.size and not np.isfinite(finite).all():
            raise ValidationError(f"{column} contains infinite values")
        bad = numeric.notna() & ((numeric < lower) | (numeric > upper))
        if bad.any():
            examples = numeric[bad].head(5).tolist()
            observed = (float(numeric.min()), float(numeric.max()))
            raise ValidationError(
                f"{column} is outside physical guardrail [{lower}, {upper}]; "
                f"observed={observed}, examples={examples}"
            )
        range_report[column] = {
            "non_null": int(numeric.notna().sum()),
            "minimum": None if not finite.size else float(finite.min()),
            "maximum": None if not finite.size else float(finite.max()),
            "guardrail": [lower, upper],
        }

    groups = trackman[GROUP_KEY].dropna().astype(str)
    group_counts = groups.value_counts().sort_index()
    eligible = trackman[GROUP_KEY].isin(PITCH_GROUPS)
    if not eligible.any():
        raise ValidationError(f"no rows belong to supported groups {PITCH_GROUPS}")
    return {
        "rows": int(len(trackman)),
        "eligible_rows": int(eligible.sum()),
        "ignored_rows": int((~eligible).sum()),
        "seasons": sorted(int(x) for x in seasons.unique()),
        "pitch_group_counts": {str(k): int(v) for k, v in group_counts.items()},
        "physical_ranges": range_report,
    }


def build_season_summary(trackman: pd.DataFrame) -> SeasonSummary:
    """Create per-pitcher-season features and per-feature support counts.

    Counts are computed independently for every raw feature.  For example, a
    fastball row with missing spin contributes to ``tm_n`` and velocity support,
    but not to the spin denominator.
    """
    validation = validate_raw_trackman(trackman)
    eligible = trackman.loc[trackman[GROUP_KEY].isin(PITCH_GROUPS)].copy()
    eligible["season"] = pd.to_numeric(eligible["season"]).astype(np.int64)
    for column in NUMERIC_FEATURES:
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")

    group_keys = [*KEYS, GROUP_KEY]
    grouped = eligible.groupby(group_keys, sort=True, observed=True)
    means = grouped[list(NUMERIC_FEATURES)].mean().add_suffix("__mean")
    counts = grouped[list(NUMERIC_FEATURES)].count().add_suffix("__count")
    stds = grouped[list(VARIATION_FEATURES)].std(ddof=1).add_suffix("__std")
    sizes = grouped.size().rename("__rows")
    aggregate = pd.concat([means, counts, stds, sizes], axis=1).reset_index()

    # Row-count conservation catches accidental filtering or groupby loss.
    if int(aggregate["__rows"].sum()) != len(eligible):
        raise ValidationError("group aggregation did not conserve eligible row count")

    value_rows: list[dict] = []
    support_rows: list[dict] = []
    for (pitcher_id, season), sub in aggregate.groupby(KEYS, sort=True, observed=True):
        values: dict[str, object] = {
            "pitcher_trackman_id": pitcher_id,
            "season": int(season),
            "tm_n": int(sub["__rows"].sum()),
        }
        supports: dict[str, object] = {
            "pitcher_trackman_id": pitcher_id,
            "season": int(season),
            "tm_n": int(sub["__rows"].sum()),
        }

        for column in VARIATION_FEATURES:
            feature = f"rel_var_{column}"
            values[feature], supports[feature] = _weighted_mean(
                sub[f"{column}__std"], sub[f"{column}__count"]
            )

        for column in NUMERIC_FEATURES:
            feature = f"m_{column}"
            values[feature], supports[feature] = _weighted_mean(
                sub[f"{column}__mean"], sub[f"{column}__count"]
            )

        for pitch_group in PITCH_GROUPS:
            part = sub.loc[sub[GROUP_KEY] == pitch_group]
            for prefix, column in (("v", "rel_speed"), ("sp", "spin_rate")):
                feature = f"{prefix}_{pitch_group}"
                values[feature], supports[feature] = _weighted_mean(
                    part[f"{column}__mean"], part[f"{column}__count"]
                )

        total = float(sub["__rows"].sum())
        values["n_types"] = int(((sub["__rows"] / total) >= 0.05).sum())
        supports["n_types"] = total

        values["velo_sep"] = values["v_fastball"] - values["v_offspeed"]
        supports["velo_sep"] = min(
            float(supports["v_fastball"]), float(supports["v_offspeed"])
        )
        value_rows.append(values)
        support_rows.append(supports)

    values_frame = pd.DataFrame(value_rows)
    supports_frame = pd.DataFrame(support_rows)
    if values_frame.duplicated(KEYS).any() or supports_frame.duplicated(KEYS).any():
        raise ValidationError("season summary keys are not unique")
    if int(values_frame["tm_n"].sum()) != len(eligible):
        raise ValidationError("season summary did not conserve eligible row count")

    return SeasonSummary(
        values=values_frame,
        supports=supports_frame,
        input_rows=int(validation["rows"]),
        eligible_rows=int(validation["eligible_rows"]),
        ignored_rows=int(validation["ignored_rows"]),
    )


def _feature_columns(summary: SeasonSummary) -> list[str]:
    excluded = {*KEYS, "tm_n"}
    values = [column for column in summary.values.columns if column not in excluded]
    support_values = [column for column in summary.supports.columns if column not in excluded]
    if values != support_values:
        raise ValidationError("season value/support feature columns differ")
    return values


def build_prior(summary: SeasonSummary, target_seasons: Sequence[int]) -> pd.DataFrame:
    """Accumulate only seasons earlier than each target season."""
    features = _feature_columns(summary)
    rows: list[pd.DataFrame] = []
    requested = sorted(set(int(x) for x in target_seasons))
    if not requested:
        raise ValidationError("at least one target season is required")

    for target_season in requested:
        value_history = summary.values.loc[summary.values["season"] < target_season].copy()
        support_history = summary.supports.loc[summary.supports["season"] < target_season].copy()
        if value_history.empty:
            raise ValidationError(f"target season {target_season} has no historical rows")
        if not value_history[KEYS].equals(support_history[KEYS]):
            raise ValidationError("season values/supports lost row alignment")

        prior = (
            value_history.groupby("pitcher_trackman_id", sort=True, observed=True)["tm_n"]
            .sum()
            .rename("prior_n")
            .to_frame()
        )
        for feature in features:
            # velo_sep is recomputed from independently accumulated components.
            if feature == "velo_sep":
                continue
            value = pd.to_numeric(value_history[feature], errors="coerce")
            weight = pd.to_numeric(support_history[feature], errors="coerce")
            valid = value.notna() & weight.notna() & (weight > 0)
            numerator = (value.where(valid) * weight.where(valid)).groupby(
                value_history["pitcher_trackman_id"], observed=True
            ).sum(min_count=1)
            denominator = weight.where(valid).groupby(
                value_history["pitcher_trackman_id"], observed=True
            ).sum(min_count=1)
            prior[feature] = numerator / denominator.replace(0.0, np.nan)

        prior["velo_sep"] = prior["v_fastball"] - prior["v_offspeed"]
        prior = prior.reset_index()
        prior["season"] = target_season
        rows.append(prior)

    output = pd.concat(rows, ignore_index=True)
    output = output.rename(
        columns={
            column: f"tm_{column}"
            for column in output.columns
            if column not in KEYS
        }
    )
    ordered_features = [f"tm_{column}" for column in features]
    output = output[[*KEYS, "tm_prior_n", *ordered_features]]
    validate_prior(output, summary.values, requested)
    return output


def _prior_bounds() -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {
        "tm_prior_n": (1.0, float("inf")),
        "tm_n_types": (0.0, float(len(PITCH_GROUPS))),
        "tm_velo_sep": (-100.0, 100.0),
    }
    for column, limit in RAW_PHYSICAL_BOUNDS.items():
        bounds[f"tm_m_{column}"] = limit
    for group in PITCH_GROUPS:
        bounds[f"tm_v_{group}"] = RAW_PHYSICAL_BOUNDS["rel_speed"]
        bounds[f"tm_sp_{group}"] = RAW_PHYSICAL_BOUNDS["spin_rate"]
    for column in VARIATION_FEATURES:
        bounds[f"tm_rel_var_{column}"] = (0.0, 10.0)
    return bounds


def validate_prior(
    prior: pd.DataFrame,
    season_values: pd.DataFrame,
    target_seasons: Sequence[int],
) -> dict:
    """Fail fast on key cardinality, row counts, leakage, and physical ranges."""
    _require_columns(prior, [*KEYS, "tm_prior_n"], "prior")
    if prior.empty:
        raise ValidationError("generated prior has zero rows")
    if prior.duplicated(KEYS).any():
        examples = prior.loc[prior.duplicated(KEYS, keep=False), KEYS].head(5).to_dict("records")
        raise ValidationError(f"prior is not many-to-one on {KEYS}; examples={examples}")
    numeric = prior.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=np.float64)).any():
        raise ValidationError("generated prior contains infinite values")

    expected_rows = 0
    expected_by_season: dict[str, int] = {}
    for season in sorted(set(int(x) for x in target_seasons)):
        expected_ids = set(
            season_values.loc[season_values["season"] < season, "pitcher_trackman_id"]
        )
        actual = prior.loc[prior["season"] == season]
        actual_ids = set(actual["pitcher_trackman_id"])
        if actual_ids != expected_ids:
            raise ValidationError(
                f"target {season} key mismatch: missing={len(expected_ids-actual_ids)}, "
                f"extra={len(actual_ids-expected_ids)}"
            )
        expected_by_season[str(season)] = len(expected_ids)
        expected_rows += len(expected_ids)
    if len(prior) != expected_rows:
        raise ValidationError(f"prior row count {len(prior)} != expected {expected_rows}")
    if set(pd.to_numeric(prior["season"]).astype(int)) != set(target_seasons):
        raise ValidationError("prior contains an unexpected target season")

    range_report: dict[str, dict] = {}
    for column, (lower, upper) in _prior_bounds().items():
        if column not in prior:
            continue
        values = pd.to_numeric(prior[column], errors="coerce")
        bad = values.notna() & ((values < lower) | (values > upper))
        if bad.any():
            examples = values[bad].head(5).tolist()
            raise ValidationError(
                f"generated {column} is outside [{lower}, {upper}]; examples={examples}"
            )
        valid = values.dropna()
        range_report[column] = {
            "non_null": int(valid.size),
            "minimum": None if valid.empty else float(valid.min()),
            "maximum": None if valid.empty else float(valid.max()),
        }
    return {
        "rows": int(len(prior)),
        "unique_keys": int(prior[KEYS].drop_duplicates().shape[0]),
        "rows_by_target_season": expected_by_season,
        "ranges": range_report,
    }


def validate_many_to_one_attachment(
    main_data: pd.DataFrame,
    prior: pd.DataFrame,
    id_map: pd.DataFrame,
    minimum_share: float = 0.5,
) -> dict:
    """Validate that mapping and prior attachment preserve every main row."""
    _require_columns(main_data, ["pitcher_id", "season"], "main data")
    _require_columns(
        id_map,
        ["pitcher_id", "tm_pitcher_id", "share", "votes"],
        "pitcher id map",
    )
    _require_columns(prior, KEYS, "prior")
    if prior.duplicated(KEYS).any():
        raise ValidationError("prior keys are duplicated; many-to-one merge is unsafe")

    selected = id_map.loc[id_map["share"] > minimum_share].copy()
    selected = selected.sort_values(
        ["votes", "share"], ascending=[False, False], kind="mergesort"
    ).drop_duplicates("tm_pitcher_id", keep="first")
    if selected["pitcher_id"].duplicated().any():
        examples = selected.loc[
            selected["pitcher_id"].duplicated(keep=False),
            ["pitcher_id", "tm_pitcher_id"],
        ].head(10).to_dict("records")
        raise ValidationError(f"selected id map is not one-to-one; examples={examples}")

    mapping = selected[["pitcher_id", "tm_pitcher_id"]].rename(
        columns={"tm_pitcher_id": "pitcher_trackman_id"}
    )
    before = len(main_data)
    attached = main_data[["pitcher_id", "season"]].merge(
        mapping, on="pitcher_id", how="left", validate="many_to_one"
    )
    attached = attached.merge(prior, on=KEYS, how="left", validate="many_to_one")
    if len(attached) != before:
        raise ValidationError("Trackman attachment changed main-data row count")
    return {
        "main_rows_before": int(before),
        "main_rows_after": int(len(attached)),
        "selected_mapping_rows": int(len(mapping)),
        "mapped_main_rows": int(attached["pitcher_trackman_id"].notna().sum()),
        "prior_coverage_rows": int(attached["tm_prior_n"].notna().sum()),
        "prior_coverage_rate": float(attached["tm_prior_n"].notna().mean()),
    }


def compare_priors(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """Return a key/schema/value comparison suitable for a JSON report."""
    _require_columns(reference, KEYS, "reference prior")
    _require_columns(candidate, KEYS, "candidate prior")
    for label, frame in (("reference", reference), ("candidate", candidate)):
        if frame.duplicated(KEYS).any():
            raise ValidationError(f"{label} prior has duplicate keys")

    common_features = sorted(
        (set(reference.columns) & set(candidate.columns)) - set(KEYS)
    )
    merged = reference.merge(
        candidate, on=KEYS, how="outer", suffixes=("__reference", "__candidate"), indicator=True
    )
    feature_report: dict[str, dict] = {}
    for feature in common_features:
        old = pd.to_numeric(merged[f"{feature}__reference"], errors="coerce")
        new = pd.to_numeric(merged[f"{feature}__candidate"], errors="coerce")
        both = old.notna() & new.notna()
        delta = (new - old).abs()[both]
        denominator = new.abs().replace(0.0, np.nan)
        relative = (delta / denominator[both]).dropna()
        changed = both & ~np.isclose(old, new, rtol=1e-12, atol=1e-12, equal_nan=True)
        changed_relative = relative.loc[relative.index.intersection(changed[changed].index)]
        feature_report[feature] = {
            "both_valid": int(both.sum()),
            "reference_only_valid": int((old.notna() & new.isna()).sum()),
            "candidate_only_valid": int((old.isna() & new.notna()).sum()),
            "changed": int(changed.sum()),
            "median_absolute_change": None if delta.empty else float(delta.median()),
            "over_1pct": int((relative > 0.01).sum()),
            "over_10pct": int((relative > 0.10).sum()),
            "median_relative_change_among_changed": (
                None if changed_relative.empty else float(changed_relative.median())
            ),
            "p95_relative_change_among_changed": (
                None if changed_relative.empty else float(changed_relative.quantile(0.95))
            ),
        }
    counts = merged["_merge"].value_counts()
    return {
        "reference_rows": int(len(reference)),
        "candidate_rows": int(len(candidate)),
        "keys_in_both": int(counts.get("both", 0)),
        "reference_only_keys": int(counts.get("left_only", 0)),
        "candidate_only_keys": int(counts.get("right_only", 0)),
        "reference_only_columns": sorted(set(reference.columns) - set(candidate.columns)),
        "candidate_only_columns": sorted(set(candidate.columns) - set(reference.columns)),
        "features": feature_report,
    }


def _read_frame(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path, columns=columns)
        except ImportError as error:
            raise RuntimeError(
                "Parquet support requires pyarrow or fastparquet in the active Python environment"
            ) from error
    raise ValidationError(f"unsupported table format for {path}; use .csv or .parquet")


def _atomic_write_frame(frame: pd.DataFrame, destination: Path, force: bool) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}"
    )
    try:
        if destination.suffix.lower() == ".csv":
            frame.to_csv(temporary, index=False)
        elif destination.suffix.lower() in {".parquet", ".pq"}:
            try:
                frame.to_parquet(temporary, index=False)
            except ImportError as error:
                raise RuntimeError(
                    "Parquet support requires pyarrow or fastparquet in the active Python environment"
                ) from error
        else:
            raise ValidationError(
                f"unsupported output format for {destination}; use .csv or .parquet"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(payload: dict, destination: Path, force: bool) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing JSON artifact: {destination}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_target_seasons(raw: str) -> list[int]:
    try:
        values = sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))
    except ValueError as error:
        raise argparse.ArgumentTypeError("target seasons must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("target seasons cannot be empty")
    return values


def _assert_distinct_paths(named_paths: Mapping[str, Path | None]) -> None:
    seen: dict[Path, str] = {}
    for label, path in named_paths.items():
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in seen:
            raise ValidationError(f"{label} path collides with {seen[resolved]}: {resolved}")
        seen[resolved] = label


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trackman", type=Path, required=True, help="input .parquet or .csv")
    parser.add_argument("--output", type=Path, required=True, help="new prior .parquet or .csv")
    parser.add_argument(
        "--target-seasons",
        type=_parse_target_seasons,
        default=_parse_target_seasons("2020,2021,2022,2023,2024,2025"),
    )
    parser.add_argument("--manifest", type=Path, help="JSON manifest path")
    parser.add_argument("--reference-prior", type=Path, help="optional old prior for comparison")
    parser.add_argument("--comparison-report", type=Path, help="old/new comparison JSON path")
    parser.add_argument("--main-data", type=Path, help="optional train/main data for merge audit")
    parser.add_argument("--id-map", type=Path, help="pitcher ID map; required with --main-data")
    parser.add_argument("--minimum-share", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="replace explicit output artifacts")
    return parser


def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else output.with_name(f"{output.stem}.manifest.json")
    )
    comparison_path = None
    if args.reference_prior:
        comparison_path = (
            args.comparison_report.resolve()
            if args.comparison_report
            else output.with_name(f"{output.stem}.comparison.json")
        )
    elif args.comparison_report:
        raise ValidationError("--comparison-report requires --reference-prior")
    if bool(args.main_data) != bool(args.id_map):
        raise ValidationError("--main-data and --id-map must be supplied together")

    _assert_distinct_paths(
        {
            "trackman": args.trackman,
            "output": output,
            "manifest": manifest_path,
            "reference prior": args.reference_prior,
            "comparison report": comparison_path,
            "main data": args.main_data,
            "id map": args.id_map,
        }
    )
    if not args.trackman.exists():
        raise FileNotFoundError(args.trackman)
    # Preflight every destination before doing expensive aggregation.  This also
    # prevents a stale manifest/report from causing a partially written run.
    if not args.force:
        for label, destination in (
            ("output", output),
            ("manifest", manifest_path),
            ("comparison report", comparison_path),
        ):
            if destination is not None and destination.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing {label}: {destination}"
                )

    trackman = _read_frame(args.trackman, [*KEYS, GROUP_KEY, *NUMERIC_FEATURES])
    raw_validation = validate_raw_trackman(trackman)
    summary = build_season_summary(trackman)
    prior = build_prior(summary, args.target_seasons)
    prior_validation = validate_prior(prior, summary.values, args.target_seasons)

    attachment = None
    if args.main_data:
        attachment = validate_many_to_one_attachment(
            _read_frame(args.main_data, ["pitcher_id", "season"]),
            prior,
            _read_frame(
                args.id_map, ["pitcher_id", "tm_pitcher_id", "share", "votes"]
            ),
            minimum_share=args.minimum_share,
        )

    comparison = None
    if args.reference_prior:
        comparison = compare_priors(_read_frame(args.reference_prior), prior)

    # Write the candidate first, then hash it for the manifest.  Every write is
    # atomic, and existing artifacts require an explicit --force.
    _atomic_write_frame(prior, output, force=args.force)
    if comparison_path is not None:
        _atomic_write_json(comparison, comparison_path, force=args.force)

    support_report: dict[str, dict] = {}
    for feature in _feature_columns(summary):
        support = pd.to_numeric(summary.supports[feature], errors="coerce")
        positive = support[support > 0]
        support_report[feature] = {
            "positive_season_cells": int((support > 0).sum()),
            "zero_season_cells": int((support <= 0).sum()),
            "minimum_positive": None if positive.empty else float(positive.min()),
            "median_positive": None if positive.empty else float(positive.median()),
            "total_support": float(positive.sum()),
        }

    manifest = {
        "artifact_kind": "candidate_trackman_prior_v2",
        "adoption_policy": (
            "candidate_only; do not replace the champion prior without exact-pipeline "
            "time-forward OOF evidence"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": str(Path(__file__).resolve()),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "config": {
            "pitch_groups": list(PITCH_GROUPS),
            "numeric_features": list(NUMERIC_FEATURES),
            "target_seasons": list(args.target_seasons),
            "minimum_share": args.minimum_share,
            "weighting": "feature-specific non-null observation count",
            "temporal_rule": "source season < target season",
        },
        "input": {
            "path": str(args.trackman.resolve()),
            "bytes": args.trackman.stat().st_size,
            "sha256": _sha256(args.trackman),
            "validation": raw_validation,
        },
        "season_summary": {
            "rows": int(len(summary.values)),
            "input_rows": summary.input_rows,
            "eligible_rows": summary.eligible_rows,
            "ignored_rows": summary.ignored_rows,
            "feature_support": support_report,
        },
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "validation": prior_validation,
        },
        "attachment_validation": attachment,
        "reference": (
            None
            if args.reference_prior is None
            else {
                "path": str(args.reference_prior.resolve()),
                "bytes": args.reference_prior.stat().st_size,
                "sha256": _sha256(args.reference_prior),
            }
        ),
        "reference_comparison": comparison,
        "comparison_report": None if comparison_path is None else str(comparison_path),
    }
    _atomic_write_json(manifest, manifest_path, force=args.force)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        manifest = run(args)
    except (ValidationError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"built {manifest['output']['validation']['rows']:,} prior rows -> "
        f"{manifest['output']['path']}"
    )
    print(f"manifest -> {args.manifest or Path(manifest['output']['path']).with_name(Path(manifest['output']['path']).stem + '.manifest.json')}")
    if manifest["comparison_report"]:
        print(f"comparison -> {manifest['comparison_report']}")
    print("status: candidate only; champion model artifacts were not changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
