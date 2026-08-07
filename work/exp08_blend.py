"""실험 8: 선형모델(L2 로지스틱) 단독/앙상블 + 시드 배깅 효과.

신호가 극도로 약한 문제에서는 저분산 모델의 가치가 크다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import TARGET, decompose, logit, sigmoid
from features import build, feature_list

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
d = build(raw)
d["season"] = raw["season"].values
del raw
F0 = feature_list(d, use_season=False, use_ids=False)

PARAMS = dict(objective="binary", verbose=-1, num_threads=6, force_row_wise=True,
              bagging_freq=1, bagging_fraction=0.8, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=1000, feature_fraction=0.7, lambda_l2=10.0)
ROUNDS = 300


def trainable(upto):
    return (d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))


for upto, Y in [(2022, 2023), (2023, 2024)]:
    tr = trainable(upto)
    evR = (d.season == Y) & (d.is_F == 0)
    yR = d.loc[evR, TARGET].values
    Xtr, ytr = d.loc[tr, F0], d.loc[tr, TARGET].values
    Xev = d.loc[evR, F0]
    print(f"\n=== train<={upto} -> {Y} (R채점, 판별력 기준) ===")

    # --- 시드 배깅 LGBM ---
    ps = []
    for seed in range(5):
        p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        m = lgb.train(p, lgb.Dataset(Xtr, ytr), num_boost_round=ROUNDS)
        ps.append(m.predict(Xev))
        if seed == 0:
            print(f"  LGBM 단일시드                  판별={decompose(yR, ps[0])['score_if_mean_fixed']:7.1f}")
    pl_avg = sigmoid(np.mean([logit(x) for x in ps], axis=0))
    print(f"  LGBM 5시드 로짓평균            판별={decompose(yR, pl_avg)['score_if_mean_fixed']:7.1f}")

    # --- L2 로지스틱 ---
    med = Xtr.median()
    sc = StandardScaler()
    A = sc.fit_transform(Xtr.fillna(med).values.astype(np.float64))
    B = sc.transform(Xev.fillna(med).values.astype(np.float64))
    for C in [0.003, 0.01, 0.1]:
        lrm = LogisticRegression(C=C, max_iter=400, solver="lbfgs", n_jobs=6)
        lrm.fit(A, ytr)
        plr = lrm.predict_proba(B)[:, 1]
        r = decompose(yR, plr)
        print(f"  로지스틱 C={C:<6}                판별={r['score_if_mean_fixed']:7.1f}")
        if C == 0.01:
            plr_keep = plr

    # --- 블렌드 ---
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        pb = sigmoid((1 - w) * logit(pl_avg) + w * logit(plr_keep))
        print(f"  블렌드 로지스틱비중 {w:<4}          판별={decompose(yR, pb)['score_if_mean_fixed']:7.1f}")
