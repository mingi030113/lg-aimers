"""최종 레시피를 train_final.py 와 동일한 코드 경로로 전방검증한다.

train <= Y-1 로 학습 -> Y 시즌 전체(R+F)를 실제 대회 채점식으로 평가.
2025 예측에서 쓸 절차(자동시프트 보정 + game_type별 로짓시프트)를 그대로 적용한다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from common import TARGET, decompose
from features import build, feature_list
from train_final import PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W, DRIFT, D_AUTO, sigmoid, solve_shift

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
del raw
FEATS = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]


def run(upto, Y):
    msk = ((d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))).values
    X, y = d.loc[msk, FEATS], d.loc[msk, TARGET].values

    z_lgb = np.zeros(len(d))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        m = lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS)
        z_lgb += m.predict(d[FEATS], raw_score=True)
    z_lgb /= N_SEEDS

    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs")
    lr.fit((A - mu) / sd, y)
    B = d[FEATS].fillna(med).to_numpy(np.float64)
    z_lr = ((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0]

    z = (1 - BLEND_W) * z_lgb + BLEND_W * z_lr

    lvl = d[d.season == upto].groupby("is_F")[TARGET].mean()
    p = np.empty(len(d))
    for g in (0, 1):
        selT = ((d.season == upto) & (d.is_F == g)).values
        # F 는 2023년부터가 유효 레짐. 학습에 쓰지 않은 레짐의 레벨을 기준으로 삼으면 안 된다.
        if g == 1 and upto <= 2022:
            print(f"  (F: {upto} 시점엔 유효 레짐 F 데이터가 없어 채점 불가)")
            p[(d.is_F == 1).values] = np.nan
            continue
        if selT.sum() < 1000:
            selT = ((d.season == upto) & (d.is_F == 0)).values
            base = float(lvl[0])
        else:
            base = float(lvl[g])
        delta = solve_shift(z[selT], base + DRIFT - D_AUTO[g])
        sel = (d.is_F == g).values
        p[sel] = sigmoid(z[sel] + delta)

    ev = (d.season == Y).values & ~np.isnan(p)
    evR = ev & (d.is_F == 0).values
    evF = ev & (d.is_F == 1).values
    print(f"\n### train<={upto} -> {Y}")
    for name, m2 in [("R", evR), ("F", evF), ("전체", ev)]:
        if m2.sum() == 0:
            continue
        r = decompose(d.loc[m2, TARGET].values, p[m2])
        print(f"  {name:4s} n={m2.sum():7d}  총점={r['score']:8.1f}  판별력={r['score_if_mean_fixed']:8.1f}  "
              f"pred_mean={r['pred_mean']:.4f}  true={r['true_mean']:.4f}  오프셋={r['mean_offset']:+.4f}")
    return d.loc[ev, TARGET].values, p[ev]


print(f"설정: rounds={ROUNDS} seeds={N_SEEDS} blend_w={BLEND_W} C={LOGREG_C} "
      f"DRIFT={DRIFT} D_AUTO={D_AUTO}")
for upto, Y in [(2022, 2023), (2023, 2024)]:
    run(upto, Y)
