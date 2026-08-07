"""prior 분할통계를 '현재 파이프라인'(rounds=100, 5시드, 블렌드, Trackman 포함)에서 재측정.

exp06 은 단일시드/rounds=300/블렌드없음이라 기준선이 달랐다(476/681 vs 563/741).
LB 로 검증할 후보이므로 실제 제출 구성 그대로 다시 잰다.

채택 판단은 이제 '두 폴드 모두 개선'이 아니라
'2024 폴드 이득 × 전이율(Trackman 실측 78%) vs 2023 폴드 손실' 로 본다.
2025는 실측상 2024형(안정된 해)이었기 때문이다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from features import build, feature_list, attach_trackman
from prior_stats import build_prior, apply_prior
from train_final import (PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W,
                         DRIFT, D_AUTO, sigmoid, solve_shift, TARGET)

pd.set_option("display.width", 250)

RAW = pd.read_parquet("train.parquet")
idmap = pd.read_parquet("idmap_pitcher_v2.parquet")
sel = (idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
       .drop_duplicates("tm_pitcher_id", keep="first"))
t2p = {v: k for k, v in zip(sel.pitcher_id, sel.tm_pitcher_id)}
TMP = pd.read_parquet("tm_prior.parquet")
TMP["pitcher_id"] = TMP["pitcher_trackman_id"].map(t2p)
TMP = TMP.dropna(subset=["pitcher_id"]).drop(columns=["pitcher_trackman_id"])
TMP["pitcher_id"] = TMP["pitcher_id"].astype(int)


def score_fixed(y, p):
    """전역 레벨을 정답에 맞춘 뒤의 점수 = 순수 판별력."""
    r = y.mean()
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    p2 = sigmoid(z + solve_shift(z, r))
    return 1e5 * (1 - np.mean((p2 - y) ** 2) / (r * (1 - r)))


def run(upto, Y, use_prior, decay=0.9):
    src = RAW[RAW.season <= upto]
    raw = RAW[RAW.season <= Y].copy()
    if use_prior:
        tables, _ = build_prior(src, sorted(set(src.season.unique().tolist() + [Y])), decay=decay)
        raw = apply_prior(raw, tables)
    raw = attach_trackman(raw, TMP)
    d = build(raw)
    d["season"] = raw["season"].values
    d[TARGET] = raw[TARGET].values
    feats = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]

    msk = ((d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))).values
    X, y = d.loc[msk, feats], d.loc[msk, TARGET].values
    ev = ((d.season == Y) & (d.is_F == 0)).values

    z = np.zeros(ev.sum())
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
    return score_fixed(d.loc[ev, TARGET].values, sigmoid(zb)), len(feats)


print("=" * 100)
print("### 현재 파이프라인 기준 prior 분할통계 재측정 (R채점, 레벨 고정)")
print("=" * 100)
res = {}
for tag, up in [("기준 (Trackman만)", False), ("+prior decay=0.9", True)]:
    line = []
    for upto, Y in [(2022, 2023), (2023, 2024)]:
        s, nf = run(upto, Y, up)
        line.append(s)
        print(f"  {tag:22s} {Y}R = {s:7.1f}  (피처 {nf}개)")
    res[tag] = line

b, a = res["기준 (Trackman만)"], res["+prior decay=0.9"]
print(f"\n  Δ2023 = {a[0]-b[0]:+.1f}   Δ2024 = {a[1]-b[1]:+.1f}")
print(f"  Trackman 실측 전이율 78% 적용 시 LB 기대 Δ ≈ {(a[1]-b[1])*0.78:+.1f}")
