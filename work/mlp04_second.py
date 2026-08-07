"""두 번째 MLP — 다양성 확장 검증.

원리(오늘 실측): 앙상블 기여는 '잘 맞히는가'가 아니라 '다르게 틀리는가'에서 나온다.
  XGBoost 상관0.977 -> -0.9 | CatBoost 0.958 -> -0.3 | MLP 0.838 -> +12.0
따라서 MLP2 는 구조가 아니라 **전처리**를 바꿔 오차 방향을 갈라놓는다.

현재 MLP1: 중앙값 대치 + z-표준화, (32,16), relu, alpha=1.0, 3시드

MLP2 후보
  A 분위수변환(정규)      — 입력 분포 자체를 바꿔 이상치·꼬리를 다르게 취급
  B tanh 활성함수         — relu 의 조각별 선형 대신 매끄러운 포화
  C 결측 표시자 추가       — 콜드스타트를 '평균값'이 아니라 '별도 신호'로
  D 분위수변환 + 단층(64,) — 구조까지 다르게
  E 피처 70% 부분집합      — 고전적 다양성 주입

평가: z = (1-w1-w2)*base + w1*MLP1 + w2*MLP2,  base = 0.65*LGBM + 0.35*Logistic
채택: 양쪽 폴드 모두 현재 구성(base*0.8 + MLP1*0.2)보다 개선.
"""
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

from features import build, feature_list, attach_trackman
from train_final import sigmoid, solve_shift, TARGET

pd.set_option("display.width", 250)
H1, A1, NS, W1 = (32, 16), 1.0, 3, 0.20

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
    tr = ((d.season >= upto - 1) & (d.season <= upto)
          & ~((d.is_F == 1) & (d.season <= 2022))).values
    ev = ((d.season == Y) & (d.is_F == 0)).values
    Xtr, Xev = d.loc[tr, FEATS], d.loc[ev, FEATS]
    med = Xtr.median()
    A = Xtr.fillna(med).to_numpy(np.float64)
    E = Xev.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1.0
    PRE[Y] = dict(Az=(A - mu) / sd, Ez=(E - mu) / sd, Araw=A, Eraw=E,
                  miss_tr=Xtr.isna().to_numpy(np.float64),
                  miss_ev=Xev.isna().to_numpy(np.float64),
                  y=d.loc[tr, TARGET].values, ye=d.loc[ev, TARGET].values,
                  base=0.65 * Z[f"LGBM|{Y}"] + 0.35 * Z[f"Logistic|{Y}"])


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


def fwd(m, B):
    h = B
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]):
        h = np.maximum(h @ W + b, 0.0) if m.activation == "relu" else np.tanh(h @ W + b)
    return (h @ m.coefs_[-1] + m.intercepts_[-1]).ravel()


def train_mlp(Atr, Aev, y, h, a, act="relu", seeds=NS, feat_idx=None):
    if feat_idx is not None:
        Atr, Aev = Atr[:, feat_idx], Aev[:, feat_idx]
    return np.mean([fwd(MLPClassifier(hidden_layer_sizes=h, alpha=a, activation=act,
                                      batch_size=4096, learning_rate_init=1e-3, max_iter=150,
                                      early_stopping=True, n_iter_no_change=10,
                                      validation_fraction=0.1, random_state=s).fit(Atr, y), Aev)
                    for s in range(seeds)], axis=0)


# ---- MLP1 (현재 채택본) ----
M1 = {}
for _, Y in FOLDS:
    p = PRE[Y]
    M1[Y] = train_mlp(p["Az"], p["Ez"], p["y"], H1, A1)
CUR = {Y: sfix(PRE[Y]["ye"], (1 - W1) * PRE[Y]["base"] + W1 * M1[Y]) for _, Y in FOLDS}
B0 = {Y: sfix(PRE[Y]["ye"], PRE[Y]["base"]) for _, Y in FOLDS}
print(f"2멤버 기준        2023R {B0[2023]:7.1f}  2024R {B0[2024]:7.1f}  평균 {np.mean(list(B0.values())):7.1f}")
print(f"현재 (+MLP1 0.20) 2023R {CUR[2023]:7.1f}  2024R {CUR[2024]:7.1f}  평균 {np.mean(list(CUR.values())):7.1f}  <- 넘어야 할 선\n")

rng = np.random.default_rng(0)
sub_idx = np.sort(rng.choice(len(FEATS), int(len(FEATS) * 0.7), replace=False))
QT = {}
for _, Y in FOLDS:
    q = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0)
    QT[Y] = (q.fit_transform(PRE[Y]["Araw"]), q.transform(PRE[Y]["Eraw"]))

CAND = {
    "A 분위수변환 (32,16)": lambda Y: train_mlp(QT[Y][0], QT[Y][1], PRE[Y]["y"], (32, 16), 1.0),
    "B tanh (32,16)": lambda Y: train_mlp(PRE[Y]["Az"], PRE[Y]["Ez"], PRE[Y]["y"], (32, 16), 1.0, act="tanh"),
    "C 결측표시자 (32,16)": lambda Y: train_mlp(
        np.hstack([PRE[Y]["Az"], PRE[Y]["miss_tr"]]),
        np.hstack([PRE[Y]["Ez"], PRE[Y]["miss_ev"]]), PRE[Y]["y"], (32, 16), 1.0),
    "D 분위수+단층 (64,)": lambda Y: train_mlp(QT[Y][0], QT[Y][1], PRE[Y]["y"], (64,), 1.0),
    "E 피처70% (32,16)": lambda Y: train_mlp(PRE[Y]["Az"], PRE[Y]["Ez"], PRE[Y]["y"], (32, 16), 1.0,
                                           feat_idx=sub_idx),
}

print(f"{'MLP2 후보':22s}{'MLP1상관':>9}{'base상관':>9} | {'최적w2':>7}{'2023R':>9}{'2024R':>9}{'평균':>9}{'Δ':>8}")
res = []
for tag, fn in CAND.items():
    t0 = time.time()
    M2 = {Y: fn(Y) for _, Y in FOLDS}
    c1 = np.corrcoef(M1[2024], M2[2024])[0, 1]
    cb = np.corrcoef(PRE[2024]["base"], M2[2024])[0, 1]
    best = None
    for w2 in [0.05, 0.10, 0.15, 0.20]:
        s = [sfix(PRE[Y]["ye"], (1 - W1 - w2) * PRE[Y]["base"] + W1 * M1[Y] + w2 * M2[Y])
             for _, Y in FOLDS]
        if best is None or np.mean(s) > np.mean(best[1:]):
            best = (w2, s[0], s[1])
    w2, s23, s24 = best
    m = (s23 + s24) / 2
    both = "  <= 양쪽개선" if s23 > CUR[2023] and s24 > CUR[2024] else ""
    print(f"{tag:22s}{c1:9.3f}{cb:9.3f} | {w2:7.2f}{s23:9.1f}{s24:9.1f}{m:9.1f}"
          f"{m-np.mean(list(CUR.values())):+8.1f}{both}  ({time.time()-t0:.0f}s)")
    res.append((tag, best, M2, both != ""))

print("\n=== 통과 후보의 w2 별 상세 ===")
for tag, _, M2, ok in res:
    if not ok:
        continue
    print(f"\n  {tag}")
    for w2 in [0.0, 0.05, 0.10, 0.15, 0.20]:
        s = [sfix(PRE[Y]["ye"], (1 - W1 - w2) * PRE[Y]["base"] + W1 * M1[Y] + w2 * M2[Y])
             for _, Y in FOLDS]
        both = "  양쪽개선" if s[0] > CUR[2023] and s[1] > CUR[2024] else ""
        print(f"    w2={w2:.2f}  2023R {s[0]:7.1f}  2024R {s[1]:7.1f}  평균 {np.mean(s):7.1f}{both}")
