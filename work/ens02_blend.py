"""저장된 멤버 예측으로 블렌드 가중치 탐색 (재학습 없음).

정직한 검증: 한 폴드에서 가중치를 찾아 다른 폴드에 적용한다.
in-sample 최적은 참고용으로만 병기.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from features import build
from train_final import sigmoid, solve_shift, TARGET

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
del raw

Z = np.load("ens_members.npz")
MEM = ["LGBM", "Logistic", "XGBoost", "CatBoost", "CatBoost+ID", "MLP"]
Y_ = {}
for Y in (2023, 2024):
    ev = ((d.season == Y) & (d.is_F == 0)).values
    Y_[Y] = d.loc[ev, TARGET].values
P = {Y: np.column_stack([Z[f"{m}|{Y}"] for m in MEM]) for Y in (2023, 2024)}


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


def sc(Y, w):
    return sfix(Y_[Y], P[Y] @ np.asarray(w, dtype=np.float64))


CUR = np.array([0.65, 0.35, 0, 0, 0, 0])
print("=" * 92)
print("### 기준: 현재 블렌드 (LGBM 0.65 + Logistic 0.35)")
print("=" * 92)
b23, b24 = sc(2023, CUR), sc(2024, CUR)
print(f"  2023R {b23:7.1f}   2024R {b24:7.1f}   평균 {np.mean([b23,b24]):7.1f}")

print("\n" + "=" * 92)
print("### 후보를 하나씩 추가 (LGBM/Logistic 비를 0.65:0.35 로 유지하며 자리 내주기)")
print("=" * 92)
print(f"  {'추가 멤버':14s} {'비중':>5} {'2023R':>9} {'2024R':>9} {'평균':>9} {'Δ평균':>8}")
for j, m in enumerate(MEM):
    if m in ("LGBM", "Logistic"):
        continue
    best = None
    for wn in [0.05, 0.10, 0.15, 0.20, 0.30]:
        w = np.zeros(6)
        w[0], w[1] = 0.65 * (1 - wn), 0.35 * (1 - wn)
        w[j] = wn
        s = [sc(2023, w), sc(2024, w)]
        mm = np.mean(s)
        flag = "  ← 양쪽 개선" if s[0] > b23 and s[1] > b24 else ""
        print(f"  {m:14s} {wn:5.2f} {s[0]:9.1f} {s[1]:9.1f} {mm:9.1f} "
              f"{mm-np.mean([b23,b24]):+8.1f}{flag}")
    print()

print("=" * 92)
print("### 전체 가중치 최적화 (한 폴드에서 학습 -> 다른 폴드에 적용)")
print("=" * 92)


def fit_w(Y):
    def neg(u):
        w = np.abs(u); w = w / w.sum()
        return -sc(Y, w)
    best, bx = None, None
    for seed in range(6):
        rng = np.random.default_rng(seed)
        x0 = rng.random(6) + 0.1
        r_ = minimize(neg, x0, method="Nelder-Mead",
                      options=dict(maxiter=3000, fatol=1e-4, xatol=1e-4))
        if best is None or r_.fun < best:
            best, bx = r_.fun, r_.x
    w = np.abs(bx); return w / w.sum()


for src, dst in [(2023, 2024), (2024, 2023)]:
    w = fit_w(src)
    print(f"\n  {src} 에서 최적화한 가중치:")
    print("    " + "  ".join(f"{m}={v:.3f}" for m, v in zip(MEM, w) if v > 0.005))
    print(f"    {src} (in-sample) {sc(src, w):7.1f}   ->   {dst} (out-of-sample) {sc(dst, w):7.1f}"
          f"   [현재 블렌드는 {b24 if dst==2024 else b23:7.1f}]")

print("\n" + "=" * 92)
print("### LGBM+Logistic 2멤버 비중 재확인")
print("=" * 92)
for wl in [0.25, 0.30, 0.35, 0.40, 0.45]:
    w = np.zeros(6); w[0], w[1] = 1 - wl, wl
    s = [sc(2023, w), sc(2024, w)]
    print(f"  Logistic {wl:.2f}   2023R {s[0]:7.1f}   2024R {s[1]:7.1f}   평균 {np.mean(s):7.1f}")
