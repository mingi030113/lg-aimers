"""실험 10 (최종 확정용).

(a) 수정된 prior 분할통계가 두 폴드 모두에서 이득인가
(b) 모델 자동시프트 Δ_auto 정밀 측정
(c) LGBM(rounds=100, 5시드) vs L2 로지스틱 vs 블렌드  — 공정 비교
(d) 최종 레시피 end-to-end (레벨 시프트 포함, game_type 별)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import TARGET, decompose, logit, sigmoid
from features import build, feature_list
from prior_stats import build_prior, apply_prior

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
PARAMS = dict(objective="binary", verbose=-1, num_threads=6, force_row_wise=True,
              bagging_freq=1, bagging_fraction=0.8, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=1000, feature_fraction=0.7, lambda_l2=10.0)
ROUNDS = 100
SEEDS = 5
FOLDS = [(2022, 2023), (2023, 2024)]


def make(upto, evalY, with_prior):
    if with_prior:
        src = raw[raw.season <= upto]
        seasons = sorted(set(src.season.unique().tolist() + [evalY]))
        tables, _ = build_prior(src, seasons, decay=0.9, k_shrink=(400, 400, 800, 300, 200))
        a = apply_prior(raw[raw.season <= upto], tables)
        b = apply_prior(raw[raw.season == evalY], tables)
    else:
        a, b = raw[raw.season <= upto], raw[raw.season == evalY]
    A = build(a); A["season"] = a["season"].values
    B = build(b); B["season"] = b["season"].values
    return A, B


def lgb_raw(A, B, feats, seeds=SEEDS, rounds=ROUNDS):
    msk = (~((A.is_F == 1) & (A.season <= 2022))).values
    rt, re = [], []
    for s in range(seeds):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        m = lgb.train(p, lgb.Dataset(A.loc[msk, feats], A.loc[msk, TARGET]), num_boost_round=rounds)
        rt.append(m.predict(A[feats], raw_score=True))
        re.append(m.predict(B[feats], raw_score=True))
    return np.mean(rt, 0), np.mean(re, 0)


def lr_raw(A, B, feats, C):
    msk = (~((A.is_F == 1) & (A.season <= 2022))).values
    med = A.loc[msk, feats].median()
    sc = StandardScaler()
    Xtr = sc.fit_transform(A.loc[msk, feats].fillna(med).to_numpy(np.float64))
    m = LogisticRegression(C=C, max_iter=300, solver="lbfgs")
    m.fit(Xtr, A.loc[msk, TARGET].values)
    f = lambda D: m.decision_function(sc.transform(D[feats].fillna(med).to_numpy(np.float64)))
    return f(A), f(B)


def disc(B, z, mask=None):
    m = (B.is_F == 0).values if mask is None else mask
    return decompose(B.loc[m, TARGET].values, sigmoid(z[m]))["score_if_mean_fixed"]


print("=" * 118)
print("### (a) 수정 prior 분할통계 (2022이전 F 제외 + 시즌x게임타입 리그평균), rounds=100 5시드")
print("=" * 118)
CACHE = {}
for wp in (False, True):
    line = []
    for upto, Y in FOLDS:
        A, B = make(upto, Y, wp)
        feats = [c for c in feature_list(A, use_season=False, use_ids=False) if c != TARGET]
        rt, re = lgb_raw(A, B, feats)
        CACHE[(wp, Y)] = (A, B, feats, rt, re)
        line.append(disc(B, re))
    print(f"  prior={str(wp):5s}   2023R:{line[0]:7.1f}  2024R:{line[1]:7.1f}   평균={np.mean(line):7.1f}")

USE_PRIOR = False   # (a) 결과를 보고 아래에서 갱신
sa = np.mean([disc(CACHE[(False, Y)][1], CACHE[(False, Y)][4]) for _, Y in FOLDS])
sb = np.mean([disc(CACHE[(True, Y)][1], CACHE[(True, Y)][4]) for _, Y in FOLDS])
both = all(disc(CACHE[(True, Y)][1], CACHE[(True, Y)][4]) >
           disc(CACHE[(False, Y)][1], CACHE[(False, Y)][4]) for _, Y in FOLDS)
USE_PRIOR = bool(both and sb > sa)
print(f"  => 두 폴드 모두 개선? {both}  채택: {USE_PRIOR}")

print("\n" + "=" * 118)
print("### (b) 자동시프트 Δ_auto")
print("=" * 118)
autos = []
for upto, Y in FOLDS:
    A, B, feats, rt, re = CACHE[(USE_PRIOR, Y)]
    sel = ((A.season == upto) & (A.is_F == 0)).values
    a = sigmoid(rt[sel]).mean()
    b = sigmoid(re[(B.is_F == 0).values]).mean()
    tp = A.loc[sel, TARGET].mean()
    tt = B.loc[B.is_F == 0, TARGET].mean()
    autos.append(b - a)
    print(f"  {upto}->{Y}: a={a:.4f}(실제{tp:.4f})  b={b:.4f}(실제{tt:.4f})  "
          f"Δ_auto={b-a:+.4f}  Δ_true={tt-tp:+.4f}")
D_AUTO = float(np.mean(autos))
print(f"  => Δ_auto = {D_AUTO:+.4f}")

print("\n" + "=" * 118)
print("### (c) LGBM vs 로지스틱 vs 블렌드 (판별력, R채점)")
print("=" * 118)
LR = {}
for C in [0.001, 0.003, 0.01, 0.03]:
    line = []
    for upto, Y in FOLDS:
        A, B, feats, rt, re = CACHE[(USE_PRIOR, Y)]
        zt, ze = lr_raw(A, B, feats, C)
        LR[(C, Y)] = (zt, ze)
        line.append(disc(B, ze))
    print(f"  로지스틱 C={C:<6}      2023R:{line[0]:7.1f}  2024R:{line[1]:7.1f}   평균={np.mean(line):7.1f}")

bestC = max([0.001, 0.003, 0.01, 0.03],
            key=lambda C: np.mean([disc(CACHE[(USE_PRIOR, Y)][1], LR[(C, Y)][1]) for _, Y in FOLDS]))
print(f"  최적 C={bestC}")
line = []
for upto, Y in FOLDS:
    line.append(disc(CACHE[(USE_PRIOR, Y)][1], CACHE[(USE_PRIOR, Y)][4]))
print(f"  LGBM 5시드            2023R:{line[0]:7.1f}  2024R:{line[1]:7.1f}   평균={np.mean(line):7.1f}")

BEST_W, best_s = 0.0, -1e9
for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
    line = []
    for upto, Y in FOLDS:
        A, B, feats, rt, re = CACHE[(USE_PRIOR, Y)]
        z = (1 - w) * re + w * LR[(bestC, Y)][1]
        line.append(disc(B, z))
    s = np.mean(line)
    print(f"  블렌드 w(로지스틱)={w:<4}   2023R:{line[0]:7.1f}  2024R:{line[1]:7.1f}   평균={s:7.1f}")
    if s > best_s:
        best_s, BEST_W = s, w
print(f"  => 최적 w={BEST_W}  판별력={best_s:.1f}")


def solve_shift(ztr, target):
    lo, hi = -2.0, 2.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if sigmoid(ztr + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


print("\n" + "=" * 118)
print("### (d) 최종 레시피 end-to-end (DRIFT 민감도) — 블렌드 + game_type별 레벨 시프트")
print("=" * 118)
for DRIFT in [0.0, -0.004, -0.008, -0.011, -0.014, -0.018]:
    rows = []
    for upto, Y in FOLDS:
        A, B, feats, rt, re = CACHE[(USE_PRIOR, Y)]
        zt = (1 - BEST_W) * rt + BEST_W * LR[(bestC, Y)][0]
        ze = (1 - BEST_W) * re + BEST_W * LR[(bestC, Y)][1]
        p = np.empty(len(B))
        for g in (0, 1):
            selB = (B.is_F == g).values
            if selB.sum() == 0:
                continue
            selA = ((A.season == upto) & (A.is_F == g)).values
            if selA.sum() < 1000:
                selA = ((A.season == upto) & (A.is_F == 0)).values
            t = A.loc[selA, TARGET].mean() + DRIFT - D_AUTO
            p[selB] = sigmoid(ze[selB] + solve_shift(zt[selA], t))
        evR = (B.is_F == 0).values
        rR = decompose(B.loc[evR, TARGET].values, p[evR])
        rA = decompose(B[TARGET].values, p)
        rows.append((Y, rR, rA))
    s = "  ".join(f"{Y} R={r1['score']:7.1f}[{r1['pred_mean']:.4f}/{r1['true_mean']:.4f}] 전체={r2['score']:7.1f}"
                  for Y, r1, r2 in rows)
    print(f"  DRIFT={DRIFT:+.3f}  {s}  R평균={np.mean([r[1]['score'] for r in rows]):7.1f}")

print(f"\n>>> 확정: USE_PRIOR={USE_PRIOR}  blend_w={BEST_W}  logregC={bestC}  D_AUTO={D_AUTO:+.5f}")
