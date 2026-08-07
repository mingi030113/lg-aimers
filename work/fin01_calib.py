"""최종 구성(2023~2024 학습 + Trackman + prior 분할통계)의 상수 재측정.

측정 대상
  1) 판별력 — prior 를 얹었을 때 실제로 오르는지 이 구성에서 재확인
  2) Δ_auto  — 모델이 바뀌면 자동 하강량도 바뀐다
  3) 볼카운트 b-s 보정량 — 새 구성의 out-of-sample 잔차에서 다시 뽑고 λ 재검증

폴드는 실제 제출과 같은 구조로: train {Y-2, Y-1} -> eval Y
  Y=2023 -> train {2021, 2022}
  Y=2024 -> train {2022, 2023}
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from features import build, feature_list, attach_trackman
from prior_stats import build_prior, apply_prior
from train_final import (PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W,
                         sigmoid, solve_shift, TARGET)

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

PRIOR_DECAY = 0.9


def prep(upto, Y, use_prior):
    """upto 까지의 라벨로 prior 생성 -> upto 이하 학습셋 + Y 평가셋에 부착."""
    src = RAW[RAW.season <= upto]
    raw = RAW[RAW.season <= Y].copy()
    if use_prior:
        tables, _ = build_prior(src, sorted(set(src.season.unique().tolist() + [Y])),
                                decay=PRIOR_DECAY)
        raw = apply_prior(raw, tables)
    raw = attach_trackman(raw, TMP)
    d = build(raw)
    d["season"] = raw["season"].values
    d[TARGET] = raw[TARGET].values
    feats = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
    return d, feats


def fit(d, feats, upto):
    msk = ((d.season >= upto - 1) & (d.season <= upto)
           & ~((d.is_F == 1) & (d.season <= 2022))).values
    X, y = d.loc[msk, feats], d.loc[msk, TARGET].values
    z = np.zeros(len(d))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        z += lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS).predict(d[feats], raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit((A - mu) / sd, y)
    B = d[feats].fillna(med).to_numpy(np.float64)
    return (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


FOLDS = [(2022, 2023), (2023, 2024)]

print("=" * 100)
print("### 1) prior 분할통계 효과 (학습 2시즌 구성에서 재확인, R채점)")
print("=" * 100)
Zs, Ds = {}, {}
for use_prior in (False, True):
    line = []
    for upto, Y in FOLDS:
        d, feats = prep(upto, Y, use_prior)
        z = fit(d, feats, upto)
        ev = ((d.season == Y) & (d.is_F == 0)).values
        line.append(sfix(d.loc[ev, TARGET].values, z[ev]))
        if use_prior:
            Zs[Y] = (z, d)
            Ds[Y] = feats
    tag = "+prior" if use_prior else "기준(Trackman만)"
    print(f"  {tag:20s} 2023R {line[0]:7.1f}   2024R {line[1]:7.1f}   피처 {len(feats)}개")
    if not use_prior:
        base = line
print(f"\n  Δ2023 = {line[0]-base[0]:+.1f}   Δ2024 = {line[1]-base[1]:+.1f}")

print("\n" + "=" * 100)
print("### 2) 최종 구성의 Δ_auto")
print("=" * 100)
out = {0: [], 1: []}
for Y in (2023, 2024):
    z, d = Zs[Y]
    upto = Y - 1
    for g, nm in [(0, "R"), (1, "F")]:
        if g == 1 and Y == 2023:
            continue
        a = sigmoid(z[((d.season == upto) & (d.is_F == g)).values]).mean()
        b = sigmoid(z[((d.season == Y) & (d.is_F == g)).values]).mean()
        out[g].append(b - a)
        print(f"  {upto}->{Y} {nm}: {a:.4f} -> {b:.4f}   Δ_auto={b-a:+.4f}")
D_R = float(np.mean(out[0]))
D_F = float((np.mean(out[0]) + np.mean(out[1])) / 2)
print(f"\n  R = {D_R:+.5f}   F 헤지값 = {D_F:+.5f}  (F 원값 {np.mean(out[1]):+.5f})")

print("\n" + "=" * 100)
print("### 3) 볼카운트 b-s 보정량 재산출 + λ 재검증")
print("=" * 100)
F = {}
for Y in (2023, 2024):
    z, d = Zs[Y]
    ev = ((d.season == Y) & (d.is_F == 0)).values
    sub = d.loc[ev].reset_index(drop=True)
    y = sub[TARGET].values
    zz = z[ev] + solve_shift(z[ev], y.mean())     # 전역 레벨 제거
    g = (sub["balls_before"] - sub["strikes_before"]).values.astype(int).astype(str)
    F[Y] = (zz, y, g)

C = {Y: {k: solve_shift(F[Y][0][F[Y][2] == k], F[Y][1][F[Y][2] == k].mean())
         for k in np.unique(F[Y][2]) if (F[Y][2] == k).sum() >= 300} for Y in F}
ks = sorted(set(C[2023]) | set(C[2024]), key=int)
print(pd.DataFrame({"b-s": ks, "2023": [C[2023].get(k, np.nan) for k in ks],
                    "2024": [C[2024].get(k, np.nan) for k in ks]})
      .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

print(f"\n  {'λ':>5} {'2023->2024':>12} {'2024->2023':>12} {'평균':>9} {'최악':>9}")
best = None
for lam in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]:
    ds = []
    for src, dst in [(2023, 2024), (2024, 2023)]:
        zz, y, g = F[dst]
        b0 = sfix(y, zz)
        z2 = zz + np.array([lam * C[src].get(k, 0.0) for k in g])
        ds.append(sfix(y, z2) - b0)
    m, w = np.mean(ds), np.min(ds)
    print(f"  {lam:5.2f} {ds[0]:+12.1f} {ds[1]:+12.1f} {m:+9.1f} {w:+9.1f}")
    if best is None or (w > 0 and m > best[1]):
        best = (lam, m, w)
LAM = best[0]
print(f"\n  채택 λ = {LAM:.2f} (평균 {best[1]:+.1f}, 최악 {best[2]:+.1f})")

COUNT_CAL = {k: float(LAM * np.nanmean([C[2023].get(k, np.nan), C[2024].get(k, np.nan)]))
             for k in ks}
print(f"  최종 보정량(두 폴드 평균 × λ): "
      + ", ".join(f"{k}:{v:+.4f}" for k, v in COUNT_CAL.items()))

with open("fin_consts.json", "w", encoding="utf-8") as fp:
    json.dump(dict(d_auto={"R": D_R, "F": D_F}, count_cal=COUNT_CAL,
                   lam=LAM, prior_decay=PRIOR_DECAY), fp, ensure_ascii=False, indent=2)
print("\n저장: fin_consts.json")
