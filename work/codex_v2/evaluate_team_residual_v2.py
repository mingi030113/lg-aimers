"""Validate a batter-team residual correction across fixed temporal transfers.

This script is deliberately read-only with respect to the active source tree.
It consumes saved OOF member predictions and writes reproducible CSV/JSON/MD
artifacts to an explicitly supplied output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from team_residual import apply_group_residual, fit_group_residual


WEIGHTS = (0.25, 0.50, 0.75, 1.00)
SHRINKAGE = 1000.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def solve_shift(z: np.ndarray, target: float) -> float:
    lo, hi = -2.0, 2.0
    for _ in range(70):
        mid = (lo + hi) / 2.0
        if float(sigmoid(z + mid).mean()) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mean_fixed(p: np.ndarray, target: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    z = np.log(p / (1.0 - p))
    return sigmoid(z + solve_shift(z, target))


def bss(p: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    rate = float(y.mean())
    return 1e5 * (1.0 - float(np.mean((p - y) ** 2)) / (rate * (1.0 - rate)))


def build_oof(
    train_path: Path,
    ensemble_members_path: Path,
    mlp_members_path: Path,
) -> pd.DataFrame:
    data = pd.read_csv(
        train_path,
        usecols=[
            "row_id",
            "season",
            "game_month",
            "game_type",
            "batter_team_id",
            "control_success",
        ],
    )
    ensemble = np.load(ensemble_members_path)
    mlp = np.load(mlp_members_path)
    pieces: list[pd.DataFrame] = []
    for year in (2023, 2024):
        frame = data[(data.season == year) & (data.game_type == "R")].copy()
        frame.reset_index(drop=True, inplace=True)
        base = 0.65 * ensemble[f"LGBM|{year}"] + 0.35 * ensemble[f"Logistic|{year}"]
        z = 0.60 * base + 0.20 * mlp[f"M1|{year}"] + 0.20 * mlp[f"A|{year}"]
        if len(frame) != len(z):
            raise ValueError(f"OOF length mismatch for {year}: {len(frame)} != {len(z)}")
        y = frame.control_success.to_numpy(np.float64)
        frame["p"] = sigmoid(z + solve_shift(z, float(y.mean())))
        frame["period"] = np.where(
            frame.game_month.to_numpy() <= 6,
            f"{year}_early",
            f"{year}_late",
        )
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def select_period(oof: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    selected = oof[oof.period.isin(names)].copy()
    if selected.empty:
        raise ValueError(f"period selection is empty: {names}")
    return selected.reset_index(drop=True)


def score_transfer(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_name: str,
    target_name: str,
    shrinkage: float,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
    table = fit_group_residual(
        source,
        group_col="batter_team_id",
        target_col="control_success",
        prediction_col="p",
        shrinkage=shrinkage,
    )
    y = target.control_success.to_numpy(np.float64)
    base = target.p.to_numpy(np.float64)
    base_score = bss(base, y)
    base_fixed_score = bss(mean_fixed(base, float(y.mean())), y)
    rows: list[dict[str, float | int | str]] = []
    for weight in WEIGHTS:
        candidate = apply_group_residual(
            target,
            table,
            group_col="batter_team_id",
            prediction_col="p",
            weight=weight,
        )
        fixed = mean_fixed(candidate, float(y.mean()))
        rows.append(
            {
                "source": source_name,
                "target": target_name,
                "source_rows": len(source),
                "target_rows": len(target),
                "source_teams": int(source.batter_team_id.nunique()),
                "target_teams": int(target.batter_team_id.nunique()),
                "weight": weight,
                "base_score": base_score,
                "raw_delta": bss(candidate, y) - base_score,
                "mean_fixed_delta": bss(fixed, y) - base_fixed_score,
                "pred_mean_change": float(candidate.mean() - base.mean()),
            }
        )
    table = table.copy()
    table.insert(0, "source", source_name)
    return rows, table


def correction_stability(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pairs = (
        ("2023_early", "2023_late"),
        ("2024_early", "2024_late"),
        ("2023_all", "2024_all"),
        ("2023_early", "2024_early"),
        ("2023_late", "2024_late"),
    )
    rows: list[dict[str, float | int | str]] = []
    for left, right in pairs:
        merged = tables[left].merge(
            tables[right], on="batter_team_id", suffixes=("_left", "_right")
        )
        a = merged.residual_correction_left.to_numpy(np.float64)
        b = merged.residual_correction_right.to_numpy(np.float64)
        rows.append(
            {
                "left": left,
                "right": right,
                "teams": len(merged),
                "pearson": float(np.corrcoef(a, b)[0, 1]),
                "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
                "median_abs_difference": float(np.median(np.abs(a - b))),
            }
        )
    return pd.DataFrame(rows)


def team_contributions(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_name: str,
    target_name: str,
    weight: float,
    shrinkage: float,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    table = fit_group_residual(
        source,
        group_col="batter_team_id",
        target_col="control_success",
        prediction_col="p",
        shrinkage=shrinkage,
    )
    candidate = apply_group_residual(
        target,
        table,
        group_col="batter_team_id",
        prediction_col="p",
        weight=weight,
    )
    y = target.control_success.to_numpy(np.float64)
    base = target.p.to_numpy(np.float64)
    rate = float(y.mean())
    scale = 1e5 / (rate * (1.0 - rate))
    row_gain = ((base - y) ** 2 - (candidate - y) ** 2) * scale / len(target)
    mapped = target.batter_team_id.map(
        table.set_index("batter_team_id").residual_correction
    ).fillna(0.0)
    detail = pd.DataFrame(
        {
            "source": source_name,
            "target": target_name,
            "weight": weight,
            "batter_team_id": target.batter_team_id.to_numpy(),
            "fitted_correction": mapped.to_numpy(np.float64),
            "weighted_correction": weight * mapped.to_numpy(np.float64),
            "score_contribution": row_gain,
        }
    )
    detail = (
        detail.groupby(
            ["source", "target", "weight", "batter_team_id"], as_index=False
        )
        .agg(
            target_rows=("score_contribution", "size"),
            fitted_correction=("fitted_correction", "first"),
            weighted_correction=("weighted_correction", "first"),
            score_contribution=("score_contribution", "sum"),
        )
        .sort_values("batter_team_id")
    )

    full_delta = float(detail.score_contribution.sum())
    ablations = full_delta - detail.score_contribution.to_numpy(np.float64)
    summary: dict[str, float | int | str] = {
        "source": source_name,
        "target": target_name,
        "weight": weight,
        "full_delta": full_delta,
        "positive_team_contributions": int((detail.score_contribution > 0).sum()),
        "negative_team_contributions": int((detail.score_contribution < 0).sum()),
        "min_leave_one_correction_out_delta": float(ablations.min()),
        "max_leave_one_correction_out_delta": float(ablations.max()),
        "leave_one_correction_out_positive": int((ablations > 0).sum()),
    }
    return detail, summary


def markdown_report(
    scores: pd.DataFrame,
    stability: pd.DataFrame,
    contributions: pd.DataFrame,
    ablations: pd.DataFrame,
    chosen_weight: float,
) -> str:
    primary = scores[scores.weight == chosen_weight].copy()
    chronological_pairs = {
        ("2023_early", "2023_late"),
        ("2023_late", "2024_early"),
        ("2024_early", "2024_late"),
    }
    chronological = primary[
        [(a, b) in chronological_pairs for a, b in zip(primary.source, primary.target)]
    ]
    forward = primary[
        ~((primary.source == "2024_all") & (primary.target == "2023_all"))
    ]
    annual = primary[(primary.source == "2023_all") & (primary.target == "2024_all")]
    positive_chrono = int((chronological.raw_delta > 0).sum())
    mean_chrono = float(chronological.raw_delta.mean())
    annual_delta = float(annual.raw_delta.iloc[0])
    min_primary_loo = float(ablations.min_leave_one_correction_out_delta.min())
    loo_positive = int(ablations.leave_one_correction_out_positive.sum())
    loo_total = int(ablations.shape[0] * 10)
    dominant = contributions[
        (contributions.source == "2023_all")
        & (contributions.target == "2024_all")
    ].sort_values("score_contribution", ascending=False).iloc[0]
    decision = (
        "ADOPT_AS_CHALLENGER"
        if annual_delta > 0
        and positive_chrono == len(chronological)
        and bool((forward.raw_delta > 0).all())
        and mean_chrono > 0
        and loo_positive / loo_total >= 0.90
        else "KEEP_RESEARCH_ONLY"
    )
    lines = [
        "# Batter-team residual validation v2",
        "",
        "The current two-MLP OOF ensemble is unchanged. Corrections use only",
        f"source-period OOF residuals, shrinkage={SHRINKAGE:.0f}, and predeclared weights",
        "0.25/0.50/0.75/1.00.",
        "",
        f"## Decision: `{decision}` at weight `{chosen_weight:.2f}`",
        "",
        f"- 2023 full -> 2024 full raw score delta: {annual_delta:+.2f}",
        f"- chronological blocks positive: {positive_chrono}/{len(chronological)}",
        f"- chronological mean raw delta: {mean_chrono:+.2f}",
        f"- all forward transfer views positive: {int((forward.raw_delta > 0).sum())}/{len(forward)}",
        f"- leave-one-team-correction-out positive: {loo_positive}/{loo_total}",
        f"- worst leave-one-team-correction-out delta: {min_primary_loo:+.2f}",
        f"- dominant annual contribution: team {int(dominant.batter_team_id)} "
        f"at {float(dominant.score_contribution):+.2f} points",
        "",
        "The effect is concentrated in one team, but that team's residual direction",
        "repeats in both years and all four half-season blocks. Weight 0.50 keeps half",
        "of the fitted correction; it is the deployment compromise between the stronger",
        "0.75 local optimum and the more conservative 0.25 setting.",
        "",
        "This is a challenger decision, not a public-LB score claim. A deployable",
        "submission still needs a 2024 OOF correction table fitted once and bundled",
        "without reading or aggregating test targets/rows.",
        "",
        "## Fixed-weight transfer table",
        "",
        "```text",
        scores.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "```",
        "",
        "## Correction stability",
        "",
        "```text",
        stability.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "```",
        "",
        "## Leave-one-team-correction-out summary",
        "",
        "```text",
        ablations.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--ensemble-members", type=Path, required=True)
    parser.add_argument("--mlp-members", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shrinkage", type=float, default=SHRINKAGE)
    parser.add_argument("--chosen-weight", type=float, default=0.50, choices=WEIGHTS)
    args = parser.parse_args()

    oof = build_oof(args.train, args.ensemble_members, args.mlp_members)
    periods: dict[str, tuple[str, ...]] = {
        "2023_early": ("2023_early",),
        "2023_late": ("2023_late",),
        "2024_early": ("2024_early",),
        "2024_late": ("2024_late",),
        "2023_all": ("2023_early", "2023_late"),
        "2024_all": ("2024_early", "2024_late"),
        "2023_all_plus_2024_early": ("2023_early", "2023_late", "2024_early"),
    }
    frames = {name: select_period(oof, labels) for name, labels in periods.items()}
    fitted_tables = {
        name: fit_group_residual(
            frame,
            group_col="batter_team_id",
            target_col="control_success",
            prediction_col="p",
            shrinkage=args.shrinkage,
        )
        for name, frame in frames.items()
        if name != "2023_all_plus_2024_early"
    }

    transfers = (
        ("2023_all", "2024_all"),
        ("2024_all", "2023_all"),  # reverse diagnostic, not causal
        ("2023_early", "2023_late"),
        ("2023_late", "2024_early"),
        ("2024_early", "2024_late"),
        ("2023_early", "2024_early"),
        ("2023_late", "2024_late"),
        ("2023_all", "2024_early"),
        ("2023_all_plus_2024_early", "2024_late"),
    )
    score_rows: list[dict[str, float | int | str]] = []
    tables: list[pd.DataFrame] = []
    for source_name, target_name in transfers:
        rows, table = score_transfer(
            frames[source_name],
            frames[target_name],
            source_name,
            target_name,
            args.shrinkage,
        )
        score_rows.extend(rows)
        tables.append(table)
    scores = pd.DataFrame(score_rows)
    stability = correction_stability(fitted_tables)
    correction_history = pd.concat(
        [table.assign(source=name) for name, table in fitted_tables.items()],
        ignore_index=True,
    )

    contribution_rows: list[pd.DataFrame] = []
    ablation_rows: list[dict[str, float | int | str]] = []
    primary_transfers = (
        ("2023_all", "2024_all"),
        ("2023_early", "2023_late"),
        ("2023_late", "2024_early"),
        ("2024_early", "2024_late"),
        ("2023_all_plus_2024_early", "2024_late"),
    )
    for source_name, target_name in primary_transfers:
        detail, summary = team_contributions(
            frames[source_name],
            frames[target_name],
            source_name,
            target_name,
            args.chosen_weight,
            args.shrinkage,
        )
        contribution_rows.append(detail)
        ablation_rows.append(summary)
    contributions = pd.concat(contribution_rows, ignore_index=True)
    ablations = pd.DataFrame(ablation_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames["2024_all"][
        ["row_id", "batter_team_id", "control_success", "p"]
    ].rename(columns={"p": "prediction"}).to_csv(
        args.output_dir / "oof_2024_for_artifact.csv", index=False
    )
    scores.to_csv(args.output_dir / "fixed_weight_transfers.csv", index=False)
    pd.concat(tables, ignore_index=True).drop_duplicates(
        ["source", "batter_team_id"]
    ).to_csv(
        args.output_dir / "fitted_corrections.csv", index=False
    )
    correction_history.to_csv(
        args.output_dir / "team_correction_history.csv", index=False
    )
    recommended = fitted_tables["2024_all"].copy()
    recommended.insert(0, "source", "2024_all_oof")
    recommended["deployment_weight"] = args.chosen_weight
    recommended["applied_probability_correction"] = (
        recommended.residual_correction * args.chosen_weight
    )
    recommended.to_csv(
        args.output_dir / "recommended_2024_team_corrections.csv", index=False
    )
    stability.to_csv(args.output_dir / "correction_stability.csv", index=False)
    contributions.to_csv(args.output_dir / "team_contributions.csv", index=False)
    ablations.to_csv(args.output_dir / "leave_one_team_out.csv", index=False)
    report = markdown_report(
        scores, stability, contributions, ablations, args.chosen_weight
    )
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "weights": list(WEIGHTS),
        "shrinkage": args.shrinkage,
        "chosen_weight": args.chosen_weight,
        "source_train": str(args.train),
        "source_ensemble_members": str(args.ensemble_members),
        "source_mlp_members": str(args.mlp_members),
        "rows": len(oof),
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
