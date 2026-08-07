"""Trackman 구성에서의 자동시프트 Δ_auto 재측정.

Trackman 피처는 '직전 시즌까지'의 고정값이라 시즌이 넘어가도 거의 움직이지 않는다.
드리프트를 실어 나르던 asof_* 의 상대적 영향이 줄어들면 모델의 자동 하강량이 달라진다.
기본 모델의 -0.0046 을 그대로 쓰면 Trackman 효과와 레벨 오차가 섞여 LB 비교가 무의미해진다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from common import TARGET, decompose, sigmoid
from features import build, feature_list, attach_trackman
from train_final import PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
idmap = pd.read_parquet("idmap_pitcher_v2.parquet")
sel = (idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
       .drop_duplicates("tm_pitcher_id", keep="first"))
p2t = dict(zip(sel.pitcher_id, sel.tm_pitcher_id))
t2p = {v: k for k, v in p2t.items()}
pri = pd.read_parquet("tm_prior.parquet")
pri["pitcher_id"] = pri["pitcher_trackman_id"].map(t2p)
pri = pri.dropna(subset=["pitcher_id"]).drop(columns=["pitcher_trackman_id"])
pri["pitcher_id"] = pri["pitcher_id"].astype(int)

raw = attach_trackman(raw, pri)
d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
FEATS = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
print(f"피처 {len(FEATS)}개 (Trackman 포함), 커버리지 {raw['tm_prior_n'].notna().mean()*100:.1f}%")


def blend_pred(upto):
    msk = ((d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))).values
    X, y = d.loc[msk, FEATS], d.loc[msk, TARGET].values
    z = np.zeros(len(d))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        m = lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS)
        z += m.predict(d[FEATS], raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs")
    lr.fit((A - mu) / sd, y)
    B = d[FEATS].fillna(med).to_numpy(np.float64)
    zl = ((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0]
    return (1 - BLEND_W) * z + BLEND_W * zl


print("\n" + "=" * 100)
print("### Δ_auto 측정 (Trackman 구성)")
print("=" * 100)
out = {0: [], 1: []}
for upto, Y in [(2022, 2023), (2023, 2024)]:
    z = blend_pred(upto)
    for g, name in [(0, "R"), (1, "F")]:
        selA = ((d.season == upto) & (d.is_F == g)).values
        selB = ((d.season == Y) & (d.is_F == g)).values
        if selA.sum() < 1000 or selB.sum() == 0:
            continue
        if g == 1 and upto <= 2022:
            print(f"  {upto}->{Y} F: 유효 레짐 F 학습데이터 없음 -> 측정 불가")
            continue
        a = sigmoid(z[selA]).mean()
        b = sigmoid(z[selB]).mean()
        ta = d.loc[selA, TARGET].mean()
        tb = d.loc[selB, TARGET].mean()
        out[g].append(b - a)
        print(f"  {upto}->{Y} {name}: a={a:.4f}(실제{ta:.4f})  b={b:.4f}(실제{tb:.4f})  "
              f"Δ_auto={b-a:+.4f}  Δ_true={tb-ta:+.4f}")

print("\n  === 결과 ===")
print(f"  R: {np.mean(out[0]):+.5f}   (기본 모델은 -0.00460)")
if out[1]:
    print(f"  F: {np.mean(out[1]):+.5f}   (기본 모델은 -0.01830, R 사전값과 중간값 -0.011 채택)")
