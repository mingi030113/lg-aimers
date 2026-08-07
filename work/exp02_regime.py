"""실험 2: 2023 폴드 붕괴의 원인(F 게임 레짐 변화) 분리 + 레벨 제어 전략 비교."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import TARGET, ID, score_bss, decompose, logit, sigmoid

pd.set_option("display.width", 250)

t = pd.read_parquet("train.parquet")
t["is_F"] = (t["game_type"] == "F").astype(np.int8)
t["top_bottom"] = (t["top_bottom"] == "B").astype(np.int8)
BASE_MAP = {"___": 0, "1__": 1, "_2_": 2, "12_": 3, "__3": 4, "1_3": 5, "_23": 6, "123": 7}
t["base_state"] = t["base_state"].map(BASE_MAP).astype(np.int8)
t = t.drop(columns=["game_type", "asof_pitcher_pitchmix_n", "away_win_expectancy"])

FEATS = [c for c in t.columns if c not in (ID, TARGET, "season")]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=6, force_row_wise=True)

print("=== 2023 폴드: R/F 분리 채점 (train<=2022, LGBM season제거) ===")
tr = t.season <= 2022
ev = t.season == 2023
m = lgb.train(PARAMS, lgb.Dataset(t.loc[tr, FEATS], t.loc[tr, TARGET]), num_boost_round=400)
p = m.predict(t.loc[ev, FEATS])
ye = t.loc[ev, TARGET].values
fe = t.loc[ev, "is_F"].values.astype(bool)
for name, msk in [("전체", np.ones(len(ye), bool)), ("R만", ~fe), ("F만", fe)]:
    d = decompose(ye[msk], p[msk])
    print(f"  {name:6s} n={msk.sum():7d}  score={d['score']:9.1f}  pred_mean={d['pred_mean']:.4f}  true={d['true_mean']:.4f}")

print("\n  -> 2023 붕괴는 F 게임(전체의 10%) 레짐 변화 때문. 2022 이전 F 라벨은 다른 체계.")

# ---------------------------------------------------------------- #
print("\n" + "=" * 110)
print("### 2024 폴드 (train<=2023): 학습셋 구성 / 레벨 제어 전략 비교")
print("=" * 110)
ev = t.season == 2024
Xte = t.loc[ev, FEATS]
yte = t.loc[ev, TARGET].values


def report(tag, p):
    d = decompose(yte, p)
    print(f"  {tag:56s} score={d['score']:8.1f} | 평균고정 {d['score_if_mean_fixed']:8.1f} | "
          f"레벨손실 {d['level_loss']:7.1f} | pred_mean={d['pred_mean']:.4f}")
    return d


def fit(mask, weight=None, rounds=400):
    w = None if weight is None else weight[mask.values]
    ds = lgb.Dataset(t.loc[mask, FEATS], t.loc[mask, TARGET], weight=w)
    return lgb.train(PARAMS, ds, num_boost_round=rounds)


variants = {
    "A. 전체 2019-2023": t.season <= 2023,
    "B. F 게임 전부 제외": (t.season <= 2023) & (t.is_F == 0),
    "C. 2022이전 F만 제외": (t.season <= 2023) & ~((t.is_F == 1) & (t.season <= 2022)),
    "D. 2021-2023만": (t.season >= 2021) & (t.season <= 2023),
    "E. 2021-2023 & 2022이전F 제외": (t.season >= 2021) & (t.season <= 2023) & ~((t.is_F == 1) & (t.season <= 2022)),
}
preds = {}
for k, msk in variants.items():
    mm = fit(msk)
    preds[k] = mm.predict(Xte)
    report(k, preds[k])

print("\n  --- 시즌 가중 (지수 감쇠) ---")
for half in [1.0, 2.0, 3.0]:
    w = 0.5 ** ((2023 - t["season"].values) / half)
    msk = (t.season <= 2023) & ~((t.is_F == 1) & (t.season <= 2022))
    mm = fit(msk, weight=pd.Series(w))
    p = mm.predict(Xte)
    report(f"F. 2022이전F제외 + 반감기{half}시즌 가중", p)

# ---------------------------------------------------------------- #
print("\n" + "=" * 110)
print("### 레벨(평균) 보정 전략 — 판별력은 그대로 두고 평균만 옮기기")
print("=" * 110)
best = preds["C. 2022이전 F만 제외"]
print(f"  보정 전 pred_mean={best.mean():.4f}, 실제 2024={yte.mean():.4f}")

# (1) train-only 추세 외삽으로 목표 평균 결정
ss = t[t.season <= 2023].groupby("season")[TARGET].mean()
for k in [2, 3, 4]:
    co = np.polyfit(ss.index.values[-k:], ss.values[-k:], 1)
    tgt = np.polyval(co, 2024)
    z = logit(best)
    lo, hi = -3, 3
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() < tgt:
            lo = mid
        else:
            hi = mid
    report(f"G. 로짓시프트 -> 최근{k}시즌 외삽평균({tgt:.4f})", sigmoid(z + (lo + hi) / 2))

# (2) 오라클 (2024 정답 평균에 맞춤)
z = logit(best)
lo, hi = -3, 3
for _ in range(60):
    mid = (lo + hi) / 2
    if sigmoid(z + mid).mean() < yte.mean():
        lo = mid
    else:
        hi = mid
report(f"H. 로짓시프트 -> 오라클평균({yte.mean():.4f})", sigmoid(z + (lo + hi) / 2))

# (3) 확률 수축 (판별력 과신 제거)
for a in [0.6, 0.8, 0.9, 1.0, 1.1, 1.2]:
    p = sigmoid(logit(best) * a)
    report(f"I. 로짓 스케일 x{a}", p)
