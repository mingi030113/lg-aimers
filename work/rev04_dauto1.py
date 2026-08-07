"""1시즌 학습 구성(TRAIN_FROM=2024)의 Δ_auto 측정 + 최신성 축 탐색.

937 의 정체가 '레짐 단절 회피'가 아니라 '최신성'이라면, 더 좁히면 더 오를 수도 있다.
동시에 데이터는 절반이 된다. 어느 쪽이 이길지는 LB 만 안다 — 다만 Δ_auto 는 미리 재둔다.

측정 구조는 실제 제출과 동일: train {Y-1} 후 (Y-1 행 예측평균) vs (Y 행 예측평균).
  Y=2024 -> train {2023}
  Y=2023 -> train {2022}   (F 는 2022 가 구 레짐이라 R 만)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from features import build, feature_list, attach_trackman
from train_final import (PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W,
                         sigmoid, solve_shift, TARGET)

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
idmap = pd.read_parquet("idmap_pitcher_v2.parquet")
sel = (idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
       .drop_duplicates("tm_pitcher_id", keep="first"))
t2p = {v: k for k, v in zip(sel.pitcher_id, sel.tm_pitcher_id)}
TMP = pd.read_parquet("tm_prior.parquet")
TMP["pitcher_id"] = TMP["pitcher_trackman_id"].map(t2p)
TMP = TMP.dropna(subset=["pitcher_id"]).drop(columns=["pitcher_trackman_id"])
TMP["pitcher_id"] = TMP["pitcher_id"].astype(int)
raw = attach_trackman(raw, TMP)
d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
FEATS = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
NO_TM = [c for c in FEATS if not c.startswith("tm_")]
del raw


def blend_all(train_mask, feats=FEATS, weight=None):
    X, y = d.loc[train_mask, feats], d.loc[train_mask, TARGET].values
    w = None if weight is None else weight[train_mask]
    z = np.zeros(len(d))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        z += lgb.train(p, lgb.Dataset(X, y, weight=w), num_boost_round=ROUNDS).predict(
            d[feats], raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit(
        (A - mu) / sd, y, sample_weight=w)
    B = d[feats].fillna(med).to_numpy(np.float64)
    return (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


print("=" * 104)
print("### 1시즌 학습 구성의 Δ_auto (TRAIN_FROM=2024 용)")
print("=" * 104)
out = {0: [], 1: []}
for Y in (2023, 2024):
    tr = ((d.season == Y - 1) & ~((d.is_F == 1) & (d.season <= 2022))).values
    z = blend_all(tr)
    print(f"\n  --- train {{{Y-1}}} ({tr.sum():,}행) -> {Y}")
    for g, nm in [(0, "R"), (1, "F")]:
        if g == 1 and Y == 2023:
            print("      F: 학습셋에 신 레짐 F 없음 -> 측정 불가")
            continue
        a_m = ((d.season == Y - 1) & (d.is_F == g)).values
        b_m = ((d.season == Y) & (d.is_F == g)).values
        a, b = sigmoid(z[a_m]).mean(), sigmoid(z[b_m]).mean()
        out[g].append(b - a)
        print(f"      {nm}: 예측 {a:.4f}->{b:.4f}  Δ_auto={b-a:+.4f}")
    ev = ((d.season == Y) & (d.is_F == 0)).values
    print(f"      판별력 {Y}R = {sfix(d.loc[ev, TARGET].values, z[ev]):.1f}")

print("\n  === Δ_auto 결과 ===")
print(f"  R: {np.mean(out[0]):+.5f}   (2시즌 -0.00257 / 전체시즌 -0.00413)")
if out[1]:
    print(f"  F: {np.mean(out[1]):+.5f}   (2시즌 -0.01129 / 전체시즌 -0.01695)")
    print(f"  F 헤지값(R과 중간) = {(np.mean(out[0])+np.mean(out[1]))/2:+.5f}")

print("\n" + "=" * 104)
print("### 참고: 2024 를 고정 평가할 때 학습 구간/가중별 판별력")
print("=" * 104)
ev = ((d.season == 2024) & (d.is_F == 0)).values
yv = d.loc[ev, TARGET].values
cfg = []
for tag, m in [("2023만", (d.season == 2023).values),
               ("2022~2023", ((d.season >= 2022) & (d.season <= 2023)).values),
               ("2019~2023", (d.season <= 2023).values)]:
    m = m & ~((d.is_F == 1) & (d.season <= 2022)).values
    z = blend_all(m)
    cfg.append((tag, sfix(yv, z[ev]), int(m.sum())))
    print(f"  {tag:12s} {m.sum():>9,}행 -> {cfg[-1][1]:7.1f}")
# 지수 가중 (최근일수록 무겁게)
for half in (0.5, 1.0):
    w = (0.5 ** ((2023 - d["season"].values) / half)).astype(np.float64)
    m = ((d.season <= 2023) & ~((d.is_F == 1) & (d.season <= 2022))).values
    z = blend_all(m, weight=w)
    print(f"  전체+반감기{half}시즌 가중 -> {sfix(yv, z[ev]):7.1f}")

print("\n" + "=" * 104)
print("### Trackman 이 최신성 제한 하에서도 여전히 도움이 되나 (train 2022~2023 -> 2024)")
print("=" * 104)
m = ((d.season >= 2022) & (d.season <= 2023) & ~((d.is_F == 1) & (d.season <= 2022))).values
for tag, fs in [("+Trackman (100)", FEATS), ("Trackman 제외 (78)", NO_TM)]:
    z = blend_all(m, feats=fs)
    print(f"  {tag:22s} -> {sfix(yv, z[ev]):7.1f}")
