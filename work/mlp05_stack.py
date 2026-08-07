"""MLP2-A 가중치 확장 + 3번째 멤버 조합 검증. 예측을 저장해 가중치 탐색을 반복 가능하게."""
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

from features import build, feature_list, attach_trackman
from train_final import sigmoid, solve_shift, TARGET

pd.set_option("display.width", 250)
NS = 3

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
    Xtr, Xev = d.loc[tr, FEATS], d.loc[ev, FEATS]
    med = Xtr.median()
    A = Xtr.fillna(med).to_numpy(np.float64); E = Xev.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1.0
    q = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0)
    PRE[Y] = dict(Az=(A-mu)/sd, Ez=(E-mu)/sd, Aq=q.fit_transform(A), Eq=q.transform(E),
                  miss_tr=Xtr.isna().to_numpy(np.float64), miss_ev=Xev.isna().to_numpy(np.float64),
                  y=d.loc[tr, TARGET].values, ye=d.loc[ev, TARGET].values,
                  base=0.65*Z[f"LGBM|{Y}"] + 0.35*Z[f"Logistic|{Y}"])

def sfix(y, z):
    r = y.mean(); return 1e5*(1-np.mean((sigmoid(z+solve_shift(z, r))-y)**2)/(r*(1-r)))
def fwd(m, B):
    h = B
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]): h = np.maximum(h@W+b, 0.0)
    return (h@m.coefs_[-1]+m.intercepts_[-1]).ravel()
def tr_mlp(Atr, Aev, y, h=(32,16), a=1.0):
    return np.mean([fwd(MLPClassifier(hidden_layer_sizes=h, alpha=a, batch_size=4096,
                    learning_rate_init=1e-3, max_iter=150, early_stopping=True,
                    n_iter_no_change=10, validation_fraction=0.1, random_state=s).fit(Atr, y), Aev)
                    for s in range(NS)], axis=0)

rng = np.random.default_rng(0)
sub_idx = np.sort(rng.choice(len(FEATS), int(len(FEATS)*0.7), replace=False))
M = {}
for _, Y in FOLDS:
    p = PRE[Y]; t0 = time.time()
    M[("M1", Y)] = tr_mlp(p["Az"], p["Ez"], p["y"])
    M[("A", Y)]  = tr_mlp(p["Aq"], p["Eq"], p["y"])
    M[("C", Y)]  = tr_mlp(np.hstack([p["Az"], p["miss_tr"]]), np.hstack([p["Ez"], p["miss_ev"]]), p["y"])
    M[("E", Y)]  = tr_mlp(p["Az"][:, sub_idx], p["Ez"][:, sub_idx], p["y"])
    print(f"  폴드 {Y} 멤버 4종 학습 완료 ({time.time()-t0:.0f}s)")
np.savez("mlp_members.npz", **{f"{k[0]}|{k[1]}": v for k, v in M.items()})

def blend(Y, w):
    z = (1 - sum(w.values())) * PRE[Y]["base"]
    for k, v in w.items(): z = z + v * M[(k, Y)]
    return sfix(PRE[Y]["ye"], z)

CUR = {Y: blend(Y, {"M1": 0.20}) for _, Y in FOLDS}
print(f"\n현재(949) 구성: 2023R {CUR[2023]:.1f}  2024R {CUR[2024]:.1f}  평균 {np.mean(list(CUR.values())):.1f}\n")

print("=== M1 0.20 고정, A 가중치 확장 ===")
for w2 in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    s = [blend(Y, {"M1": 0.20, "A": w2}) for _, Y in FOLDS]
    ok = "  양쪽개선" if s[0] > CUR[2023] and s[1] > CUR[2024] else ""
    print(f"  A={w2:.2f}  2023R {s[0]:7.1f}  2024R {s[1]:7.1f}  평균 {np.mean(s):7.1f}{ok}")

print("\n=== M1 과 A 비중 동시 탐색 ===")
best = None
for w1 in [0.10, 0.15, 0.20, 0.25]:
    row = []
    for w2 in [0.15, 0.20, 0.25, 0.30]:
        s = [blend(Y, {"M1": w1, "A": w2}) for _, Y in FOLDS]
        ok = s[0] > CUR[2023] and s[1] > CUR[2024]
        row.append(f"{np.mean(s):7.1f}{'*' if ok else ' '}")
        if ok and (best is None or np.mean(s) > best[0]): best = (np.mean(s), w1, w2, s)
    print(f"  M1={w1:.2f} | " + "  ".join(f"A={w:.2f}:{v}" for w, v in zip([0.15,0.20,0.25,0.30], row)))
print(f"  (* = 양쪽 폴드 개선)  최고: M1={best[1]:.2f} A={best[2]:.2f} 평균 {best[0]:.1f} "
      f"(2023R {best[3][0]:.1f} / 2024R {best[3][1]:.1f})")

print("\n=== 3번째 MLP 추가 ===")
for third in ("C", "E"):
    for w3 in [0.05, 0.10, 0.15]:
        s = [blend(Y, {"M1": best[1], "A": best[2], third: w3}) for _, Y in FOLDS]
        ok = "  양쪽개선" if s[0] > best[3][0] and s[1] > best[3][1] else ""
        print(f"  +{third} {w3:.2f}  2023R {s[0]:7.1f}  2024R {s[1]:7.1f}  평균 {np.mean(s):7.1f}{ok}")
