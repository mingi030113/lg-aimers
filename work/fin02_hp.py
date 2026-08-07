"""2시즌(49.9만행) 학습 구성에 맞춘 하이퍼파라미터 재탐색.

현재 상수는 전부 137만행(2019~2024 균등) 기준으로 정한 값이다.
학습셋이 1/3로 줄었으므로 최적 용량·정규화·블렌드 비중이 달라졌을 가능성이 크다.
제출 없이 폴드로 잴 수 있는 항목이므로 먼저 소진한다.

폴드는 실제 제출과 같은 구조: train {Y-2, Y-1} -> eval Y (R채점, 레벨 고정)
피처는 937 구성 그대로(Trackman 포함 100개, prior 없음).
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
print(f"피처 {len(FEATS)}개")

FOLDS = [(2022, 2023), (2023, 2024)]
CACHE = {}


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


def parts(upto, Y, params, rounds, n_seeds):
    """LGBM 로짓과 로지스틱 로짓을 따로 반환 (블렌드 비중을 나중에 스윕하려고)."""
    msk = ((d.season >= upto - 1) & (d.season <= upto)
           & ~((d.is_F == 1) & (d.season <= 2022))).values
    ev = ((d.season == Y) & (d.is_F == 0)).values
    X, y = d.loc[msk, FEATS], d.loc[msk, TARGET].values
    zl = np.zeros(int(ev.sum()))
    for s in range(n_seeds):
        p = dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        zl += lgb.train(p, lgb.Dataset(X, y), num_boost_round=rounds).predict(
            d.loc[ev, FEATS], raw_score=True)
    zl /= n_seeds
    key = ("lin", upto)
    if key not in CACHE:
        med = X.median()
        A = X.fillna(med).to_numpy(np.float64)
        mu, sd = A.mean(0), A.std(0)
        sd[sd == 0] = 1.0
        lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit((A - mu) / sd, y)
        B = d.loc[ev, FEATS].fillna(med).to_numpy(np.float64)
        CACHE[key] = ((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0]
    return zl, CACHE[key], d.loc[ev, TARGET].values


print("\n" + "=" * 108)
print("### LGBM 용량/정규화 스윕 (블렌드 0.35 고정)")
print("=" * 108)
print(f"  {'설정':46s} {'2023R':>8} {'2024R':>8} {'평균':>8}")
grid = [
    dict(tag="현재값 lv31 ml1000 ff0.7 l2=10 R100", nl=31, ml=1000, ff=0.7, l2=10, R=100),
    dict(tag="R150", nl=31, ml=1000, ff=0.7, l2=10, R=150),
    dict(tag="R200", nl=31, ml=1000, ff=0.7, l2=10, R=200),
    dict(tag="R60", nl=31, ml=1000, ff=0.7, l2=10, R=60),
    dict(tag="lv15 ml500 R150", nl=15, ml=500, ff=0.7, l2=10, R=150),
    dict(tag="lv15 ml300 R200", nl=15, ml=300, ff=0.7, l2=10, R=200),
    dict(tag="lv63 ml2000 R100", nl=63, ml=2000, ff=0.7, l2=10, R=100),
    dict(tag="ml300 (표본 1/3 반영)", nl=31, ml=300, ff=0.7, l2=10, R=100),
    dict(tag="ml500", nl=31, ml=500, ff=0.7, l2=10, R=100),
    dict(tag="ml300 R150", nl=31, ml=300, ff=0.7, l2=10, R=150),
    dict(tag="ff0.5 l2=30", nl=31, ml=500, ff=0.5, l2=30, R=150),
    dict(tag="lr0.03 R200 ml500", nl=31, ml=500, ff=0.7, l2=10, R=200, lr=0.03),
]
best = None
store = {}
for g in grid:
    p = dict(PARAMS, num_leaves=g["nl"], min_data_in_leaf=g["ml"],
             feature_fraction=g["ff"], lambda_l2=g["l2"],
             learning_rate=g.get("lr", 0.05))
    sc, keep = [], {}
    for upto, Y in FOLDS:
        zl, zlin, y = parts(upto, Y, p, g["R"], N_SEEDS)
        keep[Y] = (zl, zlin, y)
        sc.append(sfix(y, (1 - BLEND_W) * zl + BLEND_W * zlin))
    store[g["tag"]] = keep
    m = np.mean(sc)
    print(f"  {g['tag']:46s} {sc[0]:8.1f} {sc[1]:8.1f} {m:8.1f}")
    if best is None or m > best[1]:
        best = (g["tag"], m, sc)

print(f"\n  최고: {best[0]}  평균 {best[1]:.1f}  (2023R {best[2][0]:.1f} / 2024R {best[2][1]:.1f})")

print("\n" + "=" * 108)
print(f"### 블렌드 비중 스윕 (최고 설정 '{best[0]}')")
print("=" * 108)
keep = store[best[0]]
for wgt in [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 1.0]:
    sc = [sfix(keep[Y][2], (1 - wgt) * keep[Y][0] + wgt * keep[Y][1]) for _, Y in FOLDS]
    mark = "  <- 현재" if abs(wgt - BLEND_W) < 1e-9 else ""
    print(f"  로지스틱 비중 {wgt:4.2f}   2023R {sc[0]:8.1f}   2024R {sc[1]:8.1f}   "
          f"평균 {np.mean(sc):8.1f}{mark}")

print("\n" + "=" * 108)
print("### 시드 수 효과 (최고 설정)")
print("=" * 108)
gbest = [g for g in grid if g["tag"] == best[0]][0]
p = dict(PARAMS, num_leaves=gbest["nl"], min_data_in_leaf=gbest["ml"],
         feature_fraction=gbest["ff"], lambda_l2=gbest["l2"],
         learning_rate=gbest.get("lr", 0.05))
for ns in (5, 10):
    sc = []
    for upto, Y in FOLDS:
        zl, zlin, y = parts(upto, Y, p, gbest["R"], ns)
        sc.append(sfix(y, (1 - BLEND_W) * zl + BLEND_W * zlin))
    print(f"  시드 {ns:2d}개   2023R {sc[0]:8.1f}   2024R {sc[1]:8.1f}   평균 {np.mean(sc):8.1f}")
