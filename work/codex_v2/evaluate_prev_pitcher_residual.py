"""Evaluate the previous-season pitcher prior against saved forward OOF members.

The correction coefficient is learned on one year and transferred unchanged to
the other year.  This is deliberately stricter than choosing a fresh coefficient
on every evaluation fold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from previous_season_pitcher import (
    attach_previous_season_pitcher_prior,
    build_previous_season_pitcher_prior,
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(q / (1 - q))


def solve_shift(z: np.ndarray, target: float) -> float:
    lo, hi = -2.0, 2.0
    for _ in range(70):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mean_fixed(p: np.ndarray, target: float) -> np.ndarray:
    z = logit(p)
    return sigmoid(z + solve_shift(z, target))


def bss(p: np.ndarray, y: np.ndarray) -> float:
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def fit_correction(
    p: np.ndarray, y: np.ndarray, x: np.ndarray
) -> tuple[float, float, float]:
    """Fit ``intercept + beta*x`` to the base residual; constrain beta to [0, 1]."""

    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    raw_beta = 0.0 if denom == 0 else float(np.dot(xc, y - p) / denom)
    beta = float(np.clip(raw_beta, 0.0, 1.0))
    intercept = float(np.mean(y - p - beta * x))
    return intercept, beta, raw_beta


def cluster_interval(
    gain: np.ndarray,
    pitcher_id: np.ndarray,
    *,
    scale: float,
    seed: int = 20260808,
    repetitions: int = 1000,
) -> tuple[float, float]:
    grouped = (
        pd.DataFrame({"pitcher_id": pitcher_id, "gain": gain})
        .groupby("pitcher_id", observed=True)["gain"]
        .agg(["sum", "size"])
    )
    sums = grouped["sum"].to_numpy()
    sizes = grouped["size"].to_numpy()
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions)
    for i in range(repetitions):
        take = rng.integers(0, len(grouped), len(grouped))
        values[i] = scale * sums[take].sum() / sizes[take].sum()
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--ensemble-members", type=Path, required=True)
    parser.add_argument("--mlp-members", type=Path, required=True)
    parser.add_argument("--k", type=float, default=500.0)
    args = parser.parse_args()

    train = pd.read_csv(
        args.train,
        usecols=["season", "game_type", "pitcher_id", "control_success"],
    )
    prior = build_previous_season_pitcher_prior(
        train,
        k=args.k,
        target_seasons=[2023, 2024],
    )
    ens = np.load(args.ensemble_members)
    mlp = np.load(args.mlp_members)

    folds: dict[int, dict[str, np.ndarray]] = {}
    for year in (2023, 2024):
        rows = train[(train.season == year) & (train.game_type == "R")].reset_index(drop=True)
        rows = attach_previous_season_pitcher_prior(rows, prior)
        y = rows.control_success.to_numpy(np.float64)
        base = 0.65 * ens[f"LGBM|{year}"] + 0.35 * ens[f"Logistic|{year}"]
        z = 0.60 * base + 0.20 * mlp[f"M1|{year}"] + 0.20 * mlp[f"A|{year}"]
        if len(z) != len(rows):
            raise ValueError(f"OOF member length mismatch for {year}: {len(z)} != {len(rows)}")
        p = sigmoid(z + solve_shift(z, float(y.mean())))
        folds[year] = {
            "p": p,
            "y": y,
            "x": rows.pitcher_prev_r_dev_eb.to_numpy(np.float64),
            "pitcher_id": rows.pitcher_id.to_numpy(),
        }

    print("source -> target   intercept   beta(raw)    raw_delta  mean_fixed_delta  pitcher95")
    for source, target in ((2023, 2024), (2024, 2023)):
        src, dst = folds[source], folds[target]
        intercept, beta, raw_beta = fit_correction(src["p"], src["y"], src["x"])
        candidate = np.clip(dst["p"] + intercept + beta * dst["x"], 1e-7, 1 - 1e-7)
        raw_delta = bss(candidate, dst["y"]) - bss(dst["p"], dst["y"])
        candidate_fixed = mean_fixed(candidate, float(dst["y"].mean()))
        fixed_delta = bss(candidate_fixed, dst["y"]) - bss(dst["p"], dst["y"])
        r = float(dst["y"].mean())
        scale = 1e5 / (r * (1 - r))
        gain = (dst["p"] - dst["y"]) ** 2 - (candidate_fixed - dst["y"]) ** 2
        lo, hi = cluster_interval(gain, dst["pitcher_id"], scale=scale)
        print(
            f"{source} -> {target}       {intercept:+.5f}  {beta:.3f}({raw_beta:+.3f})"
            f"   {raw_delta:+9.2f}       {fixed_delta:+9.2f}"
            f"   [{lo:+.2f}, {hi:+.2f}]"
        )


if __name__ == "__main__":
    main()
