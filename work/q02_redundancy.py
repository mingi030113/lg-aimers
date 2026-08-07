"""prev1/3/5 계열 중복이 실제로 해가 되는지 검증.

상관: prev3↔prev5 0.882, prev1↔prev3 0.662, prev1↔prev5 0.573
      prev3_mid↔prev5_mid 0.846, prev1_mid↔prev3_mid 0.595

변형
  (a) 전부 (현재 구성)
  (b) prev5 계열만 (prev1·prev3 및 그 파생, form_blend 제거)
  (c) prev1 계열만
  (d) 최근폼 블록 전부 제거  -> 이 블록의 총 가치
현재 제출 구성과 동일하게: train {Y-2, Y-1} -> eval Y, R채점, 레벨 고정.
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
ALL = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
del raw

G1 = ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
      "form_dev1", "mid_dev1"]
G3 = ["asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev3_game_middle_rate",
      "form_dev3", "mid_dev3"]
G5 = ["asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev5_game_middle_rate",
      "form_dev5", "mid_dev5"]
GB = ["form_blend", "form_blend_dev"]   # prev1·3·5 를 섞은 파생

VAR = {
    "(a) 전부 (현재)": ALL,
    "(b) prev5 계열만": [c for c in ALL if c not in G1 + G3 + GB],
    "(c) prev1 계열만": [c for c in ALL if c not in G3 + G5 + GB],
    "(d) 최근폼 블록 전부 제거": [c for c in ALL if c not in G1 + G3 + G5 + GB],
}


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


def run(feats, upto, Y):
    msk = ((d.season >= upto - 1) & (d.season <= upto)
           & ~((d.is_F == 1) & (d.season <= 2022))).values
    ev = ((d.season == Y) & (d.is_F == 0)).values
    X, y = d.loc[msk, feats], d.loc[msk, TARGET].values
    z = np.zeros(int(ev.sum()))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        z += lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS).predict(
            d.loc[ev, feats], raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit((A - mu) / sd, y)
    B = d.loc[ev, feats].fillna(med).to_numpy(np.float64)
    zb = (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])
    return sfix(d.loc[ev, TARGET].values, zb)


print("=" * 96)
print("### prev1/3/5 중복 검증 (train {Y-2,Y-1} -> Y, R채점)")
print("=" * 96)
print(f"  {'구성':26s} {'피처':>5} {'2023R':>9} {'2024R':>9} {'평균':>9}")
base = None
for tag, fs in VAR.items():
    sc = [run(fs, upto, Y) for upto, Y in [(2022, 2023), (2023, 2024)]]
    m = np.mean(sc)
    if base is None:
        base = m
    print(f"  {tag:26s} {len(fs):5d} {sc[0]:9.1f} {sc[1]:9.1f} {m:9.1f}   Δ{m-base:+7.1f}")

print("\n" + "=" * 96)
print("### 참고: 로지스틱 단독 계수 안정성 (다중공선성 실제 영향)")
print("=" * 96)
msk = ((d.season >= 2022) & (d.season <= 2023) & ~((d.is_F == 1) & (d.season <= 2022))).values
X, y = d.loc[msk, ALL], d.loc[msk, TARGET].values
med = X.median()
A = X.fillna(med).to_numpy(np.float64)
mu, sd = A.mean(0), A.std(0)
sd[sd == 0] = 1.0
An = (A - mu) / sd
for C in [0.01, 1.0, 100.0]:
    lr = LogisticRegression(C=C, max_iter=400, solver="lbfgs").fit(An, y)
    co = pd.Series(lr.coef_[0], index=ALL)
    grp = co[[c for c in G1 + G3 + G5 if c in co.index]]
    print(f"  C={C:<6} 계수 최대절대값 {np.abs(lr.coef_[0]).max():7.3f}  "
          f"prev계열 계수 합 {grp.sum():+.4f}  개별 최대 {grp.abs().max():.4f}")
print("  -> 우리는 C=0.01 (아주 강한 L2). 릿지는 상관 피처들에 계수를 '나눠' 갖는다.")
