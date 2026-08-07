"""MLP 설정 스윕 — '블렌드 기여도' 기준으로 평가한다.

단독 성능이 아니라 (LGBM 0.65 + Logistic 0.35) 에 얹었을 때의 이득으로 고른다.
MLP 는 단독 368.9 로 꼴찌인데도 유일하게 기여했다(상관 0.838). 중요한 건 오차 방향의 차이다.

LGBM/Logistic 폴드 예측은 ens_members.npz 에 저장돼 있어 재학습하지 않는다 (MLP 만 재학습).
추론용 numpy 순전파도 여기서 검증한다 (sklearn 버전 의존성 제거 목적).
"""
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from features import build, feature_list, attach_trackman
from train_final import sigmoid, solve_shift, TARGET, BLEND_W

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

Z = np.load("ens_members.npz")
FOLDS = [(2022, 2023), (2023, 2024)]

PRE = {}
for upto, Y in FOLDS:
    tr = ((d.season >= upto - 1) & (d.season <= upto)
          & ~((d.is_F == 1) & (d.season <= 2022))).values
    ev = ((d.season == Y) & (d.is_F == 0)).values
    X = d.loc[tr, FEATS]
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    PRE[Y] = dict(An=(A - mu) / sd, y=d.loc[tr, TARGET].values,
                  Be=(d.loc[ev, FEATS].fillna(med).to_numpy(np.float64) - mu) / sd,
                  ye=d.loc[ev, TARGET].values,
                  base=0.65 * Z[f"LGBM|{Y}"] + 0.35 * Z[f"Logistic|{Y}"])


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


def mlp_logit(m, B):
    """numpy 순전파로 raw 로짓 계산 (sklearn predict_proba 의존 제거)."""
    h = B
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]):
        h = np.maximum(h @ W + b, 0.0)          # relu
    return (h @ m.coefs_[-1] + m.intercepts_[-1]).ravel()


BASE = {Y: sfix(PRE[Y]["ye"], PRE[Y]["base"]) for _, Y in FOLDS}
print(f"기준 (LGBM+Logistic): 2023R {BASE[2023]:.1f}  2024R {BASE[2024]:.1f}  "
      f"평균 {np.mean(list(BASE.values())):.1f}\n")

CFG = [
    dict(tag="현재값 (64,32) a=1.0 it40", h=(64, 32), a=1.0, it=40, lr=1e-3),
    dict(tag="(64,32) a=1.0 it150", h=(64, 32), a=1.0, it=150, lr=1e-3),
    dict(tag="(128,64) a=3.0 it150", h=(128, 64), a=3.0, it=150, lr=1e-3),
    dict(tag="(32,16) a=1.0 it150", h=(32, 16), a=1.0, it=150, lr=1e-3),
    dict(tag="(64,) a=1.0 it150", h=(64,), a=1.0, it=150, lr=1e-3),
    dict(tag="(256,128) a=10 it150", h=(256, 128), a=10.0, it=150, lr=1e-3),
    dict(tag="(64,32) a=10 it150", h=(64, 32), a=10.0, it=150, lr=1e-3),
    dict(tag="(64,32) a=0.1 it150", h=(64, 32), a=0.1, it=150, lr=1e-3),
]

print(f"{'설정':30s}{'단독2023':>9}{'단독2024':>9} | 최적비중  2023R    2024R     평균     Δ")
results = []
for c in CFG:
    t0 = time.time()
    zz, solo = {}, {}
    for _, Y in FOLDS:
        p = PRE[Y]
        m = MLPClassifier(hidden_layer_sizes=c["h"], alpha=c["a"], batch_size=4096,
                          learning_rate_init=c["lr"], max_iter=c["it"], early_stopping=True,
                          n_iter_no_change=10, validation_fraction=0.1, random_state=0)
        m.fit(p["An"], p["y"])
        zz[Y] = mlp_logit(m, p["Be"])
        solo[Y] = sfix(p["ye"], zz[Y])
        if c is CFG[0] and Y == 2024:      # numpy 순전파 검증
            err = np.abs(sigmoid(zz[Y]) - m.predict_proba(p["Be"])[:, 1]).max()
            print(f"  [검증] numpy 순전파 vs sklearn 최대오차 = {err:.2e}")
    best = None
    for wn in [0.05, 0.10, 0.15, 0.20, 0.25]:
        s = [sfix(PRE[Y]["ye"], (1 - wn) * PRE[Y]["base"] + wn * zz[Y]) for _, Y in FOLDS]
        both = s[0] > BASE[2023] and s[1] > BASE[2024]
        rec = (wn, s[0], s[1], np.mean(s), both)
        if best is None or (rec[3] > best[3]):
            best = rec
    wn, s23, s24, avg, both = best
    flag = "  <= 양쪽개선" if both else ""
    print(f"{c['tag']:30s}{solo[2023]:9.1f}{solo[2024]:9.1f} | {wn:6.2f} {s23:8.1f} {s24:8.1f} "
          f"{avg:8.1f} {avg-np.mean(list(BASE.values())):+7.1f}{flag}  ({time.time()-t0:.0f}s)")
    results.append((c, best, zz))

results.sort(key=lambda x: -x[1][3])
bc, bb, _ = results[0]
print(f"\n최고: {bc['tag']}  비중 {bb[0]:.2f}  평균 {bb[3]:.1f} "
      f"(Δ{bb[3]-np.mean(list(BASE.values())):+.1f})  양쪽개선={bb[4]}")

print("\n=== 상위 설정의 비중별 상세 ===")
for c, _, zz in results[:2]:
    print(f"\n  {c['tag']}")
    for wn in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        s = [sfix(PRE[Y]["ye"], (1 - wn) * PRE[Y]["base"] + wn * zz[Y]) for _, Y in FOLDS]
        both = "  양쪽개선" if s[0] > BASE[2023] and s[1] > BASE[2024] else ""
        print(f"    비중 {wn:.2f}   2023R {s[0]:7.1f}   2024R {s[1]:7.1f}   평균 {np.mean(s):7.1f}{both}")
