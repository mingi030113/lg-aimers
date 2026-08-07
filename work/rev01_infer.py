"""LB 점수 역추론: 폴드 대비 +89점 격차의 출처를 분해한다.

가설
  H1 학습 시즌 수    최종 모델은 6시즌(2019~24), 폴드 모델은 5시즌(2019~23)으로 학습
  H2 asof 이력 성숙  2025 투수는 6년치 누적 -> 실력 추정이 정확 -> 해상도 상승
  H3 2025 자체가 더 변별 가능  (오프라인 검증 불가)
  H4 test 의 F 비중  F 판별력은 R 의 절반 (2024 폴드 368 vs 741)
  H5 r 이 0.5 에서 멀어져 분모 r(1-r) 축소  (효과 미미할 것)

Part A: 캐시된 예측으로 즉시 확인 (H2, H4, H5, 기울기 역산)
Part B: 학습 시즌 수를 바꿔 재학습 (H1)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from features import build, feature_list, attach_trackman
from train_final import (PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W,
                         sigmoid, solve_shift, TARGET)

pd.set_option("display.width", 250)

LB = {"base": 815.87199, "trackman": 828.1859212357}
FOLD24 = {"base": 725.0, "trackman": 741.2}

raw = pd.read_parquet("train.parquet")
idmap = pd.read_parquet("idmap_pitcher_v2.parquet")
sel = (idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
       .drop_duplicates("tm_pitcher_id", keep="first"))
t2p = {v: k for k, v in zip(sel.pitcher_id, sel.tm_pitcher_id)}
TMP = pd.read_parquet("tm_prior.parquet")
TMP["pitcher_id"] = TMP["pitcher_trackman_id"].map(t2p)
TMP = TMP.dropna(subset=["pitcher_id"]).drop(columns=["pitcher_trackman_id"])
TMP["pitcher_id"] = TMP["pitcher_id"].astype(int)
raw2 = attach_trackman(raw, TMP)
d = build(raw2)
d["season"] = raw2["season"].values
d[TARGET] = raw2[TARGET].values
FEATS = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
del raw2

Z = np.load("cal_z.npz")


def sfix(y, z):
    """전역 레벨을 정답에 맞춘 뒤 점수 = 순수 판별력."""
    if len(y) < 200:
        return np.nan
    r = y.mean()
    if r <= 0 or r >= 1:
        return np.nan
    z2 = z + solve_shift(z, r)
    return 1e5 * (1 - np.mean((sigmoid(z2) - y) ** 2) / (r * (1 - r)))


print("=" * 100)
print("### 기울기 역산: LB 점수와 양립하는 β = Cov/Var")
print("=" * 100)
print("  (Var(p)=0.001653 은 2024 폴드 실측값. r 은 0.478 가정, BASE=0.2495)")
VAR, BASE = 0.001653, 0.2495
for name, s in LB.items():
    print(f"  [{name}]")
    for e in [0.0, 0.005, 0.010, 0.015]:
        beta = 0.5 + (s * BASE / 1e5 + e * e) / (2 * VAR)
        print(f"     레벨오차 e={e:.3f} 가정 -> β={beta:.3f}")
print("  2024 폴드 실측 β = 1.088 / 2023 폴드 = 0.951")
print("  -> 2025 의 β 가 1.12 이상이어야 828 이 나온다. Var(p) 가 더 컸거나 β 가 더 컸거나.")

print("\n" + "=" * 100)
print("### H5: 분모 효과 — r 이 달라지면 점수가 얼마나 부풀려지나")
print("=" * 100)
for r in [0.460, 0.470, 0.478, 0.4861, 0.500]:
    print(f"  r={r:.4f}  r(1-r)={r*(1-r):.5f}  2024(r=.4861) 대비 점수배율 {0.24981/(r*(1-r)):.4f} "
          f"-> 828점 기준 {828.19*(1-0.24981/(r*(1-r))):+.1f}점 설명")

print("\n" + "=" * 100)
print("### H2: 투수 이력 길이(asof_pitcher_n)별 판별력 — 이력이 길수록 잘 맞히나")
print("=" * 100)
for Y in (2023, 2024):
    ev = ((d.season == Y) & (d.is_F == 0)).values
    sub = d.loc[ev].reset_index(drop=True)
    z = Z[str(Y)][ev]
    y = sub[TARGET].values
    z = z + solve_shift(z, y.mean())
    n = sub["asof_pitcher_n"].values
    print(f"\n  --- {Y}R (전체 {sfix(y, z):.1f}) ---")
    for lo, hi in [(0, 200), (200, 1000), (1000, 3000), (3000, 6000), (6000, 10**9)]:
        m = (n >= lo) & (n < hi)
        if m.sum() < 500:
            continue
        print(f"    asof_n [{lo:>5},{hi if hi < 10**8 else '∞':>5}) n={m.sum():7d} "
              f"({m.mean()*100:4.1f}%)  판별력={sfix(y[m], z[m]):7.1f}")

print("\n" + "=" * 100)
print("### H2 보강: 시즌별 asof_pitcher_n 분포 (2025 는 어디쯤일까)")
print("=" * 100)
g = d[d.is_F == 0].groupby("season")["asof_pitcher_n"].agg(
    평균="mean", 중앙값="median", n3000이상=lambda x: (x >= 3000).mean())
g["n3000이상"] = (g["n3000이상"] * 100).round(1)
print(g.round(1).to_string())
inc = g["평균"].diff().dropna()
print(f"\n  연간 증가 평균 {inc.mean():.0f}  -> 2025 예상 평균 asof_n ≈ {g['평균'].iloc[-1]+inc.mean():.0f}")

print("\n" + "=" * 100)
print("### H4: F 비중이 LB 에 주는 영향 (2024 폴드 실측: R 741.2 / F 368.5)")
print("=" * 100)
for share in [0.0, 0.06, 0.118, 0.20]:
    mixed = (1 - share) * 741.2 + share * 368.5
    print(f"  test 의 F 비중 {share*100:5.1f}% -> 폴드 등가 혼합점수 {mixed:.1f}  "
          f"-> LB828 이려면 2025 R 판별력 ≈ {828.19/mixed*741.2:.0f}")

print("\n" + "=" * 100)
print("### H1: 학습 시즌 수 효과 — 2024 를 고정 평가, 학습 구간만 늘려가며")
print("=" * 100)


def run(from_s, upto, Y=2024):
    msk = ((d.season >= from_s) & (d.season <= upto)
           & ~((d.is_F == 1) & (d.season <= 2022))).values
    X, y = d.loc[msk, FEATS], d.loc[msk, TARGET].values
    ev = ((d.season == Y) & (d.is_F == 0)).values
    z = np.zeros(ev.sum())
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        z += lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS).predict(
            d.loc[ev, FEATS], raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit((A - mu) / sd, y)
    B = d.loc[ev, FEATS].fillna(med).to_numpy(np.float64)
    zb = (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])
    return sfix(d.loc[ev, TARGET].values, zb), int(msk.sum())


for from_s in [2023, 2022, 2021, 2020, 2019]:
    sc, nrow = run(from_s, 2023)
    print(f"  학습 {from_s}~2023 ({2024-from_s}시즌, {nrow:>9,}행) -> 2024R 판별력 = {sc:7.1f}")
