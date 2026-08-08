"""Transfer-test a batter-team correction on the current two-MLP OOF ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from team_residual import apply_group_residual, fit_group_residual


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def solve_shift(z, target):
    lo, hi = -2.0, 2.0
    for _ in range(70):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mean_fixed(p, target):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    z = np.log(p / (1 - p))
    return sigmoid(z + solve_shift(z, target))


def bss(p, y):
    r = y.mean()
    return 1e5 * (1 - np.mean((p-y)**2)/(r*(1-r)))


def team_bootstrap(gain, team, scale, repetitions=5000):
    grouped = pd.DataFrame({"team":team,"gain":gain}).groupby("team").gain.agg(["sum","size"])
    sums, sizes = grouped["sum"].to_numpy(), grouped["size"].to_numpy()
    rng = np.random.default_rng(20260808)
    values = np.empty(repetitions)
    for i in range(repetitions):
        take = rng.integers(0, len(grouped), len(grouped))
        values[i] = scale*sums[take].sum()/sizes[take].sum()
    return np.percentile(values,[2.5,97.5]), len(grouped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--ensemble-members", type=Path, required=True)
    parser.add_argument("--mlp-members", type=Path, required=True)
    parser.add_argument("--shrinkage", type=float, default=1000.0)
    args = parser.parse_args()

    d = pd.read_csv(
        args.train,
        usecols=["season","game_type","batter_team_id","control_success"],
    )
    ens, mlp = np.load(args.ensemble_members), np.load(args.mlp_members)
    folds = {}
    for year in (2023,2024):
        q = d[(d.season==year)&(d.game_type=="R")].copy().reset_index(drop=True)
        base=.65*ens[f"LGBM|{year}"]+.35*ens[f"Logistic|{year}"]
        z=.60*base+.20*mlp[f"M1|{year}"]+.20*mlp[f"A|{year}"]
        if len(z)!=len(q):
            raise ValueError(f"OOF length mismatch for {year}")
        y=q.control_success.to_numpy(np.float64)
        q["p"]=sigmoid(z+solve_shift(z,float(y.mean())))
        folds[year]=q

    print("source -> target  teams  raw_delta  mean_fixed_delta  team95       leave-one-team range")
    for source,target in ((2023,2024),(2024,2023)):
        src,dst=folds[source],folds[target]
        table=fit_group_residual(
            src,group_col="batter_team_id",target_col="control_success",
            prediction_col="p",shrinkage=args.shrinkage,
        )
        candidate=apply_group_residual(
            dst,table,group_col="batter_team_id",prediction_col="p",
        )
        y=dst.control_success.to_numpy(np.float64)
        raw_delta=bss(candidate,y)-bss(dst.p.to_numpy(),y)
        fixed=mean_fixed(candidate,float(y.mean()))
        fixed_delta=bss(fixed,y)-bss(dst.p.to_numpy(),y)
        gain=(dst.p.to_numpy()-y)**2-(fixed-y)**2
        r=float(y.mean()); scale=1e5/(r*(1-r))
        ci,nteam=team_bootstrap(gain,dst.batter_team_id.to_numpy(),scale)
        loto=[]
        for held in sorted(dst.batter_team_id.unique()):
            keep=dst.batter_team_id.to_numpy()!=held
            loto.append(scale*gain[keep].mean())
        print(
            f"{source} -> {target}      {nteam:2d}   {raw_delta:+9.2f}"
            f"       {fixed_delta:+9.2f}   [{ci[0]:+.2f},{ci[1]:+.2f}]"
            f"   [{min(loto):+.2f},{max(loto):+.2f}]"
        )


if __name__ == "__main__":
    main()
