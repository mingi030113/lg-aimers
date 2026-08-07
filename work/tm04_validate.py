"""Trackman 구위 피처의 전방검증. 기존과 동일한 프로토콜(rounds=100, 5시드, 블렌드, 2폴드).

채택 규칙: 두 폴드 모두 개선될 때만. (prior 분할통계는 이 규칙으로 기각됐다)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from common import TARGET, decompose, logit, sigmoid
from features import build, feature_list
from train_final import PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
idmap = pd.read_parquet("idmap_pitcher_v2.parquet")
prior = pd.read_parquet("tm_prior.parquet")

sel = idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
sel = sel.drop_duplicates("tm_pitcher_id", keep="first")
raw["tm_pid"] = raw["pitcher_id"].map(dict(zip(sel.pitcher_id, sel.tm_pitcher_id)))

TMF = [c for c in prior.columns if c not in ("pitcher_trackman_id", "season")]
raw = raw.merge(prior.rename(columns={"pitcher_trackman_id": "tm_pid"}),
                on=["tm_pid", "season"], how="left")
print(f"trackman 피처 {len(TMF)}개 부착, 전체 커버리지 {raw['tm_prior_n'].notna().mean()*100:.1f}%")

d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
for c in TMF:
    d[c] = raw[c].values
d["tm_log_n"] = np.log1p(d["tm_prior_n"].fillna(0))
d["tm_has"] = d["tm_prior_n"].notna().astype(np.int8)
TMF2 = TMF + ["tm_log_n", "tm_has"]

BASE = [c for c in feature_list(d, use_season=False, use_ids=False)
        if c != TARGET and c not in TMF2 and c != "tm_pid"]
REL = ["tm_rel_var_rel_height", "tm_rel_var_rel_side", "tm_rel_var_extension",
       "tm_log_n", "tm_has"]
print(f"기본 피처 {len(BASE)}개")

FOLDS = [(2022, 2023), (2023, 2024)]


def run(feats, tag):
    line = []
    for upto, Y in FOLDS:
        msk = ((d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))).values
        X, y = d.loc[msk, feats], d.loc[msk, TARGET].values
        ev = ((d.season == Y) & (d.is_F == 0)).values

        z = np.zeros(ev.sum())
        for s in range(N_SEEDS):
            p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
            m = lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS)
            z += m.predict(d.loc[ev, feats], raw_score=True)
        z /= N_SEEDS

        med = X.median()
        A = X.fillna(med).to_numpy(np.float64)
        mu, sd = A.mean(0), A.std(0)
        sd[sd == 0] = 1.0
        lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs")
        lr.fit((A - mu) / sd, y)
        B = d.loc[ev, feats].fillna(med).to_numpy(np.float64)
        zl = ((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0]

        zb = (1 - BLEND_W) * z + BLEND_W * zl
        line.append(decompose(d.loc[ev, TARGET].values, sigmoid(zb))["score_if_mean_fixed"])
        if Y == 2024 and feats is not BASE:
            globals()["_last_model"] = (m, feats)
    print(f"  {tag:36s} 2023R:{line[0]:7.1f}  2024R:{line[1]:7.1f}   평균={np.mean(line):7.1f}")
    return np.array(line)


print("\n" + "=" * 100)
print("### Trackman 구위 피처 효과 (판별력, R채점)")
print("=" * 100)
b = run(BASE, "기본 78개")
r = run(BASE + REL, "+ 릴리스 반복성만 (5개)")
a = run(BASE + TMF2, "+ Trackman 전체 (22개)")
print(f"\n  릴리스만: Δ2023={r[0]-b[0]:+.1f}  Δ2024={r[1]-b[1]:+.1f}  "
      f"두 폴드 모두 개선? {bool((r > b).all())}")
print(f"  전체:     Δ2023={a[0]-b[0]:+.1f}  Δ2024={a[1]-b[1]:+.1f}  "
      f"두 폴드 모두 개선? {bool((a > b).all())}")

m, feats = _last_model
imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
imp = imp / imp.sum() * 100
print("\n### Trackman 피처 중요도 (2024 폴드, 전체 구성)")
print(imp[[c for c in TMF2 if c in imp.index]].to_string(float_format=lambda x: f"{x:.2f}%"))
print(f"\n  Trackman 피처 gain 합계: {imp[[c for c in TMF2 if c in imp.index]].sum():.2f}%")
print("\n### 전체 상위 15")
print(imp.head(15).to_string(float_format=lambda x: f"{x:.2f}%"))
