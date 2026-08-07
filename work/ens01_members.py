"""앙상블 후보 멤버 평가 + 최적 블렌드 탐색.

현재: z = 0.65*LGBM + 0.35*L2로지스틱  (다양성 이득 +21.5 실측)

후보
  A. XGBoost        — LGBM 과 같은 계열. 대조군으로 상관을 확인해 실제로 무용한지 본다.
  B. CatBoost(수치)  — 같은 피처, ordered boosting
  C. CatBoost(+ID)  — pitcher_id/batter_id 를 네이티브 범주형으로.
                      LGBM 에서 이건 622.6 -> -145.9 로 붕괴했다. CatBoost 의 ordered target
                      statistics 가 이 누수를 구조적으로 막는지가 유일한 차별점.
  D. MLP            — 매끄러운 함수. 트리(조각별 상수)와 귀납편향이 다르다.
                      sklearn MLPClassifier (평가서버 기본 설치, requirements 추가 불필요)

폴드는 실제 제출과 동일: train {Y-2,Y-1} -> eval Y, R채점, 레벨 고정.
멤버별 예측을 저장해두고 블렌드 가중치는 뒤에서 최적화한다.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from features import build, feature_list, attach_trackman
from train_final import (PARAMS, ROUNDS, N_SEEDS, LOGREG_C, sigmoid, solve_shift, TARGET)

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
d["pitcher_id"] = raw["pitcher_id"].values
d["batter_id"] = raw["batter_id"].values
FEATS = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
del raw
print(f"피처 {len(FEATS)}개")

FOLDS = [(2022, 2023), (2023, 2024)]


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


def masks(upto, Y):
    tr = ((d.season >= upto - 1) & (d.season <= upto)
          & ~((d.is_F == 1) & (d.season <= 2022))).values
    ev = ((d.season == Y) & (d.is_F == 0)).values
    return tr, ev


def logit_clip(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


Z = {}   # Z[(멤버, Y)] = eval 로짓
for upto, Y in FOLDS:
    tr, ev = masks(upto, Y)
    X, y = d.loc[tr, FEATS], d.loc[tr, TARGET].values
    Xe = d.loc[ev, FEATS]
    ye = d.loc[ev, TARGET].values
    print(f"\n=== 폴드 {Y} (학습 {tr.sum():,}행) ===")

    # --- LGBM (현재) ---
    t0 = time.time()
    z = np.zeros(int(ev.sum()))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        z += lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS).predict(Xe, raw_score=True)
    Z[("LGBM", Y)] = z / N_SEEDS
    print(f"  LGBM        {sfix(ye, Z[('LGBM',Y)]):7.1f}  ({time.time()-t0:.0f}s)")

    # --- 로지스틱 (현재) ---
    t0 = time.time()
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    An = (A - mu) / sd
    Be = (Xe.fillna(med).to_numpy(np.float64) - mu) / sd
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit(An, y)
    Z[("Logistic", Y)] = Be @ lr.coef_[0] + lr.intercept_[0]
    print(f"  Logistic    {sfix(ye, Z[('Logistic',Y)]):7.1f}  ({time.time()-t0:.0f}s)")

    # --- XGBoost (대조군) ---
    t0 = time.time()
    import xgboost as xgb
    xm = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                           min_child_weight=200, subsample=0.8, colsample_bytree=0.7,
                           reg_lambda=10, tree_method="hist", n_jobs=6, eval_metric="logloss")
    xm.fit(X, y)
    Z[("XGBoost", Y)] = logit_clip(xm.predict_proba(Xe)[:, 1])
    print(f"  XGBoost     {sfix(ye, Z[('XGBoost',Y)]):7.1f}  ({time.time()-t0:.0f}s)")

    # --- CatBoost (수치만) ---
    t0 = time.time()
    from catboost import CatBoostClassifier, Pool
    cm = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=10,
                            min_data_in_leaf=500, random_seed=0, verbose=0, thread_count=6)
    cm.fit(X.fillna(med), y)
    Z[("CatBoost", Y)] = logit_clip(cm.predict_proba(Xe.fillna(med))[:, 1])
    print(f"  CatBoost    {sfix(ye, Z[('CatBoost',Y)]):7.1f}  ({time.time()-t0:.0f}s)")

    # --- CatBoost + 선수 ID 네이티브 범주형 ---
    t0 = time.time()
    Xc = X.fillna(med).copy()
    Xec = Xe.fillna(med).copy()
    for c in ("pitcher_id", "batter_id"):
        Xc[c] = d.loc[tr, c].astype(str).values
        Xec[c] = d.loc[ev, c].astype(str).values
    cats = ["pitcher_id", "batter_id"]
    cm2 = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=10,
                             min_data_in_leaf=500, random_seed=0, verbose=0, thread_count=6)
    cm2.fit(Pool(Xc, y, cat_features=cats))
    Z[("CatBoost+ID", Y)] = logit_clip(cm2.predict_proba(Pool(Xec, cat_features=cats))[:, 1])
    print(f"  CatBoost+ID {sfix(ye, Z[('CatBoost+ID',Y)]):7.1f}  ({time.time()-t0:.0f}s)"
          f"   <- LGBM 에서 ID 범주형은 -145.9 로 붕괴했던 자리")

    # --- MLP ---
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1.0, batch_size=4096,
                        learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                        n_iter_no_change=4, validation_fraction=0.1, random_state=0)
    mlp.fit(An, y)
    Z[("MLP", Y)] = logit_clip(mlp.predict_proba(Be)[:, 1])
    print(f"  MLP         {sfix(ye, Z[('MLP',Y)]):7.1f}  ({time.time()-t0:.0f}s, "
          f"{mlp.n_iter_} epoch)")

np.savez("ens_members.npz", **{f"{k[0]}|{k[1]}": v for k, v in Z.items()})

MEM = ["LGBM", "Logistic", "XGBoost", "CatBoost", "CatBoost+ID", "MLP"]
print("\n" + "=" * 96)
print("### 멤버 간 상관 (2024 폴드 로짓) — 1에 가까울수록 앙상블 가치가 없다")
print("=" * 96)
M = pd.DataFrame({m: Z[(m, 2024)] for m in MEM})
print(M.corr().round(3).to_string())
