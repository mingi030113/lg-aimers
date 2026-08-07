"""지수 감쇠 가중 반감기 스윕 + 최적값의 Δ_auto 측정.

rev04 에서 '전체 유지 + 반감기 1.0시즌 가중'(763.1)이 어떤 하드 컷오프보다 높았다.
반감기를 더 넓게 훑고, 양쪽 폴드에서 확인한 뒤, 채택 후보의 Δ_auto 를 재둔다.

주의: 폴드는 학습데이터 축에서 부호까지 틀린 전력이 있다(2시즌 -21.3 예측 -> LB +108.9).
      다만 가중은 하드 컷오프와 달리 '연속' 축이라 폴드가 순위는 맞힐 가능성이 있다.
      최종 판단은 LB.
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
del raw


def blend_all(train_mask, weight=None):
    X, y = d.loc[train_mask, FEATS], d.loc[train_mask, TARGET].values
    w = None if weight is None else weight[train_mask]
    z = np.zeros(len(d))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        z += lgb.train(p, lgb.Dataset(X, y, weight=w), num_boost_round=ROUNDS).predict(
            d[FEATS], raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit(
        (A - mu) / sd, y, sample_weight=w)
    B = d[FEATS].fillna(med).to_numpy(np.float64)
    return (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


HALFS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
print("=" * 100)
print("### 반감기 스윕 (양쪽 폴드, R채점 판별력)")
print("=" * 100)
res = {}
for h in HALFS:
    line = []
    for upto, Y in [(2022, 2023), (2023, 2024)]:
        m = ((d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))).values
        w = (0.5 ** ((upto - d["season"].values) / h)).astype(np.float64)
        z = blend_all(m, weight=w)
        ev = ((d.season == Y) & (d.is_F == 0)).values
        line.append(sfix(d.loc[ev, TARGET].values, z[ev]))
    res[h] = line
    print(f"  반감기 {h:4.2f}시즌   2023R {line[0]:7.1f}   2024R {line[1]:7.1f}   "
          f"평균 {np.mean(line):7.1f}")

print(f"\n  참고 — 균등 가중 전체: 2023R 557.2 / 2024R 741.2 (평균 649.2)")
print(f"         2023~2024 하드컷(LB 937 구성): 2023R 558.6 / 2024R 719.9")
best_h = max(res, key=lambda k: np.mean(res[k]))
print(f"\n  평균 최고 반감기 = {best_h}")

print("\n" + "=" * 100)
print(f"### 반감기 {best_h} 구성의 Δ_auto")
print("=" * 100)
out = {0: [], 1: []}
for Y in (2023, 2024):
    upto = Y - 1
    m = ((d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))).values
    w = (0.5 ** ((upto - d["season"].values) / best_h)).astype(np.float64)
    z = blend_all(m, weight=w)
    for g, nm in [(0, "R"), (1, "F")]:
        if g == 1 and Y == 2023:
            continue
        a = sigmoid(z[((d.season == upto) & (d.is_F == g)).values]).mean()
        b = sigmoid(z[((d.season == Y) & (d.is_F == g)).values]).mean()
        out[g].append(b - a)
        print(f"  {upto}->{Y} {nm}: {a:.4f} -> {b:.4f}   Δ_auto={b-a:+.4f}")
print(f"\n  R: {np.mean(out[0]):+.5f}")
if out[1]:
    print(f"  F: {np.mean(out[1]):+.5f}   헤지값 = {(np.mean(out[0])+np.mean(out[1]))/2:+.5f}")
