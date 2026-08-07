"""실험 5: 정규화 강도 스윕 (2폴드) + 로짓 스케일 보정 + 저기여 피처 제거."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import TARGET, score_bss, decompose, logit, sigmoid
from features import build, feature_list

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
d = build(raw)
d["season"] = raw["season"].values
del raw

F0 = feature_list(d, use_season=False, use_ids=False)

FOLDS = [(2022, 2023), (2023, 2024)]


def trainable(upto):
    return (d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))


def run(params, rounds, feats=F0):
    """각 폴드에서 예측 반환."""
    out = {}
    for upto, Y in FOLDS:
        tr = trainable(upto)
        ds = lgb.Dataset(d.loc[tr, feats], d.loc[tr, TARGET])
        m = lgb.train(params, ds, num_boost_round=rounds)
        ev = d.season == Y
        out[Y] = (m.predict(d.loc[ev, feats]), d.loc[ev, TARGET].values,
                  d.loc[ev, "is_F"].values.astype(bool), m, feats)
    return out


def summarize(tag, out, scale=1.0, fix_mean=True):
    line, tot = [], []
    for Y, (p, y, isf, _, _) in out.items():
        pr = sigmoid(logit(p) * scale)
        # R 게임만 (F는 레짐 이슈로 별도)
        r = decompose(y[~isf], pr[~isf])
        line.append(f"{Y}R: {r['score_if_mean_fixed'] if fix_mean else r['score']:7.1f}")
        tot.append(r["score_if_mean_fixed"] if fix_mean else r["score"])
    print(f"  {tag:56s} " + "  ".join(line) + f"   평균={np.mean(tot):7.1f}")
    return np.mean(tot)


BASE = dict(objective="binary", verbose=-1, num_threads=6, force_row_wise=True,
            bagging_freq=1, bagging_fraction=0.8)

print("=" * 118)
print("### 정규화 스윕 (판별력 = 평균고정 점수, R게임만 채점)")
print("=" * 118)
grid = [
    dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=1000, feature_fraction=0.7, lambda_l2=10, rounds=300),
    dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=1000, feature_fraction=0.7, lambda_l2=10, rounds=150),
    dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.6, lambda_l2=30, rounds=300),
    dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.6, lambda_l2=30, rounds=150),
    dict(learning_rate=0.05, num_leaves=7, min_data_in_leaf=3000, feature_fraction=0.6, lambda_l2=50, rounds=400),
    dict(learning_rate=0.05, num_leaves=7, min_data_in_leaf=3000, feature_fraction=0.6, lambda_l2=50, rounds=200),
    dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=3000, feature_fraction=0.5, lambda_l2=50, rounds=400),
    dict(learning_rate=0.03, num_leaves=7, min_data_in_leaf=5000, feature_fraction=0.5, lambda_l2=100, rounds=500),
    dict(learning_rate=0.02, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.6, lambda_l2=30, rounds=500),
    dict(learning_rate=0.1, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.6, lambda_l2=30, rounds=100),
]
best, best_s = None, -1e9
results = {}
for g in grid:
    g = dict(g)
    rounds = g.pop("rounds")
    p = dict(BASE, **g)
    out = run(p, rounds)
    tag = f"lr={g['learning_rate']} lv={g['num_leaves']} ml={g['min_data_in_leaf']} ff={g['feature_fraction']} l2={g['lambda_l2']} R={rounds}"
    s = summarize(tag, out)
    results[tag] = out
    if s > best_s:
        best_s, best, best_out = s, (p, rounds, tag), out

print(f"\n  최고: {best[2]}  판별력={best_s:.1f}")

print("\n" + "=" * 118)
print("### 로짓 스케일 보정 (과신 제거) — 최고 설정에")
print("=" * 118)
for a in [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]:
    summarize(f"scale={a}", best_out, scale=a)

print("\n" + "=" * 118)
print("### 저기여 피처 제거 효과")
print("=" * 118)
p, rounds, _ = best
m = best_out[2024][3]
imp = pd.Series(m.feature_importance("gain"), index=F0).sort_values(ascending=False)
for keep in [len(F0), 40, 30, 25, 20, 15]:
    fs = imp.head(keep).index.tolist()
    out = run(p, rounds, feats=fs)
    summarize(f"상위 {keep}개 피처", out)

print("\n" + "=" * 118)
print("### F 게임 학습셋 처리 재확인 (2024 폴드, R채점)")
print("=" * 118)
ev = d.season == 2024
yR = d.loc[ev & (d.is_F == 0), TARGET].values
for tag, msk in [("2022이전F 제외(기본)", trainable(2023)),
                 ("F 전부 제외", (d.season <= 2023) & (d.is_F == 0)),
                 ("F 전부 포함", d.season <= 2023)]:
    ds = lgb.Dataset(d.loc[msk, F0], d.loc[msk, TARGET])
    mm = lgb.train(p, ds, num_boost_round=rounds)
    pr = mm.predict(d.loc[ev & (d.is_F == 0), F0])
    r = decompose(yR, pr)
    print(f"  {tag:36s} 판별력={r['score_if_mean_fixed']:7.1f}  raw={r['score']:7.1f}  pred_mean={r['pred_mean']:.4f}")
