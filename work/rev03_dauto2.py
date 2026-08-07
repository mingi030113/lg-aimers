"""최근 2시즌 학습 구성에서의 Δ_auto 재측정.

실제 제출: train {2023, 2024} -> test 2025. 시프트는 2024 행 기준으로 잡는다.
따라서 측정도 동일 구조로: train {Y-2, Y-1} 후 (Y-1 행 예측평균) vs (Y 행 예측평균).
  Y=2024 -> train {2022, 2023}
  Y=2023 -> train {2021, 2022}   (F 는 2022 이전이 구 레짐이라 측정 불가)
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


def blend_all(train_mask):
    X, y = d.loc[train_mask, FEATS], d.loc[train_mask, TARGET].values
    z = np.zeros(len(d))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        z += lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS).predict(d[FEATS], raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit((A - mu) / sd, y)
    B = d[FEATS].fillna(med).to_numpy(np.float64)
    return (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


print("=" * 104)
print("### 최근 2시즌 학습 구성의 Δ_auto")
print("=" * 104)
out = {0: [], 1: []}
for Y in (2023, 2024):
    tr = ((d.season >= Y - 2) & (d.season <= Y - 1)
          & ~((d.is_F == 1) & (d.season <= 2022))).values
    z = blend_all(tr)
    print(f"\n  --- train {{{Y-2}, {Y-1}}} ({tr.sum():,}행) -> {Y}")
    for g, nm in [(0, "R"), (1, "F")]:
        a_m = ((d.season == Y - 1) & (d.is_F == g)).values
        b_m = ((d.season == Y) & (d.is_F == g)).values
        if g == 1 and Y == 2023:
            print(f"      F: 학습셋에 신 레짐 F 없음 -> 측정 불가")
            continue
        a, b = sigmoid(z[a_m]).mean(), sigmoid(z[b_m]).mean()
        ta, tb = d.loc[a_m, TARGET].mean(), d.loc[b_m, TARGET].mean()
        out[g].append(b - a)
        print(f"      {nm}: 예측 {a:.4f}->{b:.4f}  Δ_auto={b-a:+.4f}   "
              f"(실제 {ta:.4f}->{tb:.4f}  Δ_true={tb-ta:+.4f})")
    ev = ((d.season == Y) & (d.is_F == 0)).values
    print(f"      판별력 {Y}R = {sfix(d.loc[ev, TARGET].values, z[ev]):.1f}")

print("\n  === 결과 ===")
print(f"  R: {np.mean(out[0]):+.5f}   (전체시즌 학습 구성은 -0.00413)")
if out[1]:
    print(f"  F: {np.mean(out[1]):+.5f}   (전체시즌 학습 구성은 -0.01695)")
    print(f"  F 헤지값(R과의 중간) = {(np.mean(out[0])+np.mean(out[1]))/2:+.5f}")
