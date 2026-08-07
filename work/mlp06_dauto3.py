"""3-MLP 블렌드 구성의 Δ_auto 측정.
z = 0.45*(0.65*LGBM+0.35*Logistic) + 0.20*MLP1(z) + 0.20*MLP2(분위수) + 0.15*MLP3(피처70%)
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from features import build, feature_list, attach_trackman
from train_final import (PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W, MLP_HIDDEN, MLP_ALPHA,
                         MLP_SEEDS, MLP_W, MLP2_W, MLP3_W, MLP3_FEAT_FRAC, MLP3_FEAT_SEED,
                         sigmoid, solve_shift, TARGET)

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

def fwd(m, B):
    h = B
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]): h = np.maximum(h@W+b, 0.0)
    return (h@m.coefs_[-1]+m.intercepts_[-1]).ravel()
def tr(Atr, Aev, y):
    return np.mean([fwd(MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, alpha=MLP_ALPHA,
        batch_size=4096, learning_rate_init=1e-3, max_iter=150, early_stopping=True,
        n_iter_no_change=10, validation_fraction=0.1, random_state=s).fit(Atr, y), Aev)
        for s in range(MLP_SEEDS)], axis=0)
def sfix(y, z):
    r = y.mean(); return 1e5*(1-np.mean((sigmoid(z+solve_shift(z, r))-y)**2)/(r*(1-r)))

rng = np.random.default_rng(MLP3_FEAT_SEED)
sub_idx = np.sort(rng.choice(len(FEATS), int(len(FEATS)*MLP3_FEAT_FRAC), replace=False))
out = {0: [], 1: []}
for Y in (2023, 2024):
    upto = Y-1
    m = ((d.season >= upto-1) & (d.season <= upto) & ~((d.is_F==1)&(d.season<=2022))).values
    X, y = d.loc[m, FEATS], d.loc[m, TARGET].values
    zl = np.zeros(len(d))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        zl += lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS).predict(d[FEATS], raw_score=True)
    zl /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64); B = d[FEATS].fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0); sd[sd==0]=1.0
    An, Bn = (A-mu)/sd, (B-mu)/sd
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit(An, y)
    base = (1-BLEND_W)*zl + BLEND_W*(Bn@lr.coef_[0]+lr.intercept_[0])
    q = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(A)
    z1 = tr(An, Bn, y); z2 = tr(q.transform(A), q.transform(B), y)
    z3 = tr(An[:, sub_idx], Bn[:, sub_idx], y)
    z = (1-MLP_W-MLP2_W-MLP3_W)*base + MLP_W*z1 + MLP2_W*z2 + MLP3_W*z3
    print(f"\n  --- train {{{upto-1},{upto}}} -> {Y}")
    for g, nm in [(0,"R"),(1,"F")]:
        if g==1 and Y==2023: continue
        a = sigmoid(z[((d.season==upto)&(d.is_F==g)).values]).mean()
        b = sigmoid(z[((d.season==Y)&(d.is_F==g)).values]).mean()
        out[g].append(b-a); print(f"      {nm}: {a:.4f} -> {b:.4f}   Δ_auto={b-a:+.4f}")
    ev = ((d.season==Y)&(d.is_F==0)).values
    print(f"      판별력 {Y}R = {sfix(d.loc[ev,TARGET].values, z[ev]):.1f}")
R = float(np.mean(out[0])); F = float((np.mean(out[0])+np.mean(out[1]))/2)
print(f"\n  === 결과 ===\n  R = {R:+.5f}   F 헤지값 = {F:+.5f}  (F 원값 {np.mean(out[1]):+.5f})")
print(f"  참고: MLP1만 구성은 R -0.0027 / F -0.0064")
