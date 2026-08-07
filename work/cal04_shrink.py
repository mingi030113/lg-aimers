"""볼카운트 보정의 수축계수 λ 탐색 (양방향 정직 검증).

cal03 결과: 방향은 안정(상관 0.783, 부호일치 9/12)이지만 크기가 연도마다 다르다.
  2023 -> 2024 적용: +11.1 (2023 보정이 작아서 과소보정 -> 그래도 이득)
  2024 -> 2023 적용:  -0.7 (2024 보정이 커서 과대보정 -> 손해)
=> λ 로 수축하면 과대보정 위험을 줄이면서 평균 이득을 남길 수 있는지 확인한다.
"""
import numpy as np
import pandas as pd

from features import build
from train_final import sigmoid, solve_shift, TARGET

Z = np.load("cal_z.npz")
raw = pd.read_parquet("train.parquet")
d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
del raw


def fold(Y):
    ev = ((d.season == Y) & (d.is_F == 0)).values
    sub = d.loc[ev].reset_index(drop=True)
    z = Z[str(Y)][ev]
    y = sub[TARGET].values
    z = z + solve_shift(z, y.mean())
    g = (sub["balls_before"].astype(str) + "-" + sub["strikes_before"].astype(str)).values
    return z, y, g


def score(y, p):
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def learn(z, y, g):
    return {k: solve_shift(z[g == k], y[g == k].mean())
            for k in np.unique(g) if (g == k).sum() >= 300}


F = {Y: fold(Y) for Y in (2023, 2024)}
C = {Y: learn(*F[Y]) for Y in (2023, 2024)}

print("=" * 92)
print("### 수축계수 λ 별 정직한 양방향 out-of-sample 이득")
print("=" * 92)
print(f"  {'λ':>5} {'2023->2024':>12} {'2024->2023':>12} {'평균':>10} {'최악':>10}")
best = None
for lam in [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]:
    deltas = []
    for src, dst in [(2023, 2024), (2024, 2023)]:
        z, y, g = F[dst]
        base = score(y, sigmoid(z))
        dz = np.array([lam * C[src].get(k, 0.0) for k in g])
        z2 = z + dz
        deltas.append(score(y, sigmoid(z2 + solve_shift(z2, y.mean()))) - base)
    m, w = np.mean(deltas), np.min(deltas)
    print(f"  {lam:5.2f} {deltas[0]:+12.1f} {deltas[1]:+12.1f} {m:+10.1f} {w:+10.1f}")
    if best is None or m > best[1]:
        best = (lam, m, w)
print(f"\n  평균 기준 최적 λ = {best[0]:.2f} (평균 {best[1]:+.1f}, 최악 {best[2]:+.1f})")

print("\n" + "=" * 92)
print("### 카운트를 더 굵게 묶으면 안정적인가 (그룹 수를 줄여 잡음 감소)")
print("=" * 92)


def coarse(sub_g, mode):
    b = np.array([int(x[0]) for x in sub_g])
    s = np.array([int(x[2]) for x in sub_g])
    if mode == "strikes":
        return s.astype(str)
    if mode == "balls":
        return b.astype(str)
    if mode == "b-s":
        return (b - s).astype(str)
    if mode == "3영역":
        r = np.where(s > b, "투수우위", np.where(b > s, "타자우위", "동등"))
        return r
    return sub_g


for mode in ["12칸(원본)", "strikes", "balls", "b-s", "3영역"]:
    deltas = []
    for src, dst in [(2023, 2024), (2024, 2023)]:
        zs, ys, gs = F[src]
        zd, yd, gd = F[dst]
        gs2 = gs if mode == "12칸(원본)" else coarse(gs, mode)
        gd2 = gd if mode == "12칸(원본)" else coarse(gd, mode)
        c = learn(zs, ys, gs2)
        base = score(yd, sigmoid(zd))
        z2 = zd + np.array([c.get(k, 0.0) for k in gd2])
        deltas.append(score(yd, sigmoid(z2 + solve_shift(z2, yd.mean()))) - base)
    print(f"  {mode:12s} 그룹수={len(set(gs2)):2d}  2023->2024 {deltas[0]:+7.1f}   "
          f"2024->2023 {deltas[1]:+7.1f}   평균 {np.mean(deltas):+7.1f}")
