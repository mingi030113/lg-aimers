"""더 작은 망 확인 + MLP 시드 평균 효과. 최종 설정 확정용."""
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from features import build, feature_list, attach_trackman
from train_final import sigmoid, solve_shift, TARGET

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
d = build(raw); d["season"] = raw["season"].values; d[TARGET] = raw[TARGET].values
FEATS = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
del raw

Z = np.load("ens_members.npz")
FOLDS = [(2022, 2023), (2023, 2024)]
PRE = {}
for upto, Y in FOLDS:
    tr = ((d.season >= upto-1) & (d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))).values
    ev = ((d.season == Y) & (d.is_F == 0)).values
    X = d.loc[tr, FEATS]; med = X.median()
    A = X.fillna(med).to_numpy(np.float64); mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1.0
    PRE[Y] = dict(An=(A-mu)/sd, y=d.loc[tr, TARGET].values,
                  Be=(d.loc[ev, FEATS].fillna(med).to_numpy(np.float64)-mu)/sd,
                  ye=d.loc[ev, TARGET].values,
                  base=0.65*Z[f"LGBM|{Y}"] + 0.35*Z[f"Logistic|{Y}"])

def sfix(y, z):
    r = y.mean(); return 1e5*(1-np.mean((sigmoid(z+solve_shift(z, r))-y)**2)/(r*(1-r)))
def fwd(m, B):
    h = B
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]):
        h = np.maximum(h @ W + b, 0.0)
    return (h @ m.coefs_[-1] + m.intercepts_[-1]).ravel()
def fit(h, a, Y, seed):
    p = PRE[Y]
    m = MLPClassifier(hidden_layer_sizes=h, alpha=a, batch_size=4096, learning_rate_init=1e-3,
                      max_iter=150, early_stopping=True, n_iter_no_change=10,
                      validation_fraction=0.1, random_state=seed).fit(p["An"], p["y"])
    return fwd(m, p["Be"])

B0 = {Y: sfix(PRE[Y]["ye"], PRE[Y]["base"]) for _, Y in FOLDS}
print(f"기준 2023R {B0[2023]:.1f}  2024R {B0[2024]:.1f}  평균 {np.mean(list(B0.values())):.1f}\n")
print("### 더 작은 망 (시드 1개)")
print(f"{'설정':22s}{'최적비중':>8}{'2023R':>9}{'2024R':>9}{'평균':>9}{'Δ':>8}")
for h, a in [((32,16),1.0), ((16,8),1.0), ((32,),1.0), ((16,),1.0), ((48,24),1.0), ((32,16),0.3)]:
    zz = {Y: fit(h, a, Y, 0) for _, Y in FOLDS}
    best = max([( wn, sfix(PRE[2023]["ye"], (1-wn)*PRE[2023]["base"]+wn*zz[2023]),
                  sfix(PRE[2024]["ye"], (1-wn)*PRE[2024]["base"]+wn*zz[2024]))
                for wn in [0.05,0.10,0.15,0.20,0.25,0.30]], key=lambda r: (r[1]+r[2])/2)
    wn, s23, s24 = best; m = (s23+s24)/2
    both = "  양쪽개선" if s23 > B0[2023] and s24 > B0[2024] else ""
    print(f"{str(h)+' a='+str(a):22s}{wn:8.2f}{s23:9.1f}{s24:9.1f}{m:9.1f}{m-np.mean(list(B0.values())):+8.1f}{both}")

print("\n### 시드 평균 효과 — (32,16) a=1.0")
for ns in (1, 3, 5):
    t0 = time.time()
    zz = {Y: np.mean([fit((32,16), 1.0, Y, s) for s in range(ns)], axis=0) for _, Y in FOLDS}
    print(f"  시드 {ns}개:")
    for wn in [0.10, 0.15, 0.20, 0.25]:
        s = [sfix(PRE[Y]["ye"], (1-wn)*PRE[Y]["base"]+wn*zz[Y]) for _, Y in FOLDS]
        both = "  양쪽개선" if s[0] > B0[2023] and s[1] > B0[2024] else ""
        print(f"    비중 {wn:.2f}  2023R {s[0]:7.1f}  2024R {s[1]:7.1f}  평균 {np.mean(s):7.1f}{both}")
    print(f"    ({time.time()-t0:.0f}s)")
