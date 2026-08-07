"""실험 9: 단조 제약 / 콜드스타트 처리 / 시즌보정 커리어 지표.

동기:
  asof_pitcher_success_rate 는 '커리어 누적'이라 시대(era) 오염이 있다.
  2019년(리그 .565) 비중이 큰 베테랑과 2024년(.486)만 있는 신인의 같은 값은 의미가 다르다.
  -> 투수의 커리어가 어느 시즌에 얼마나 분포하는지로 '기대 리그평균'을 만들어 차감한다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import TARGET, decompose, logit, sigmoid
from features import build, feature_list

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")

# ---- 시대 보정용: 투수별 '커리어 기대 리그평균' (직전 시즌까지 누적, train 라벨만 사용) ----
league = raw.groupby("season")[TARGET].mean()
cnt = raw.groupby(["pitcher_id", "season"]).size().rename("n").reset_index()
cnt["lg"] = cnt["season"].map(league)


def era_table(upto_seasons):
    """각 target 시즌 S에 대해 '시즌<S 까지의 투구수 가중 리그평균' 테이블."""
    rows = []
    for S in upto_seasons:
        sub = cnt[cnt.season < S]
        if not len(sub):
            continue
        g = sub.groupby("pitcher_id").apply(
            lambda x: pd.Series({"era_lg": np.average(x["lg"], weights=x["n"]),
                                 "prior_n": x["n"].sum()}), include_groups=False)
        g["season"] = S
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


ERA = era_table(sorted(raw.season.unique()))
raw = raw.merge(ERA, on=["pitcher_id", "season"], how="left")
raw["era_adj_career"] = raw["asof_pitcher_success_rate"] - raw["era_lg"]
raw["era_lg_fill"] = raw["era_lg"].fillna(league.iloc[-1])

d = build(raw)
d["season"] = raw["season"].values
for c in ["era_lg", "era_adj_career", "prior_n"]:
    d[c] = raw[c].values
d["log_prior_n"] = np.log1p(d["prior_n"].fillna(0))
del raw

PARAMS = dict(objective="binary", verbose=-1, num_threads=6, force_row_wise=True,
              bagging_freq=1, bagging_fraction=0.8, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=1000, feature_fraction=0.7, lambda_l2=10.0)
ROUNDS = 100
FOLDS = [(2022, 2023), (2023, 2024)]

NEW = ["era_lg", "era_adj_career", "prior_n", "log_prior_n"]
F_BASE = [c for c in feature_list(d, use_season=False, use_ids=False) if c not in NEW]
F_ERA = F_BASE + NEW


def evaluate(feats, params=None, rounds=ROUNDS, mono=None, tag=""):
    line = []
    for upto, Y in FOLDS:
        tr = (d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))
        p = dict(PARAMS if params is None else params)
        if mono is not None:
            p["monotone_constraints"] = [mono.get(c, 0) for c in feats]
            p["monotone_constraints_method"] = "advanced"
        m = lgb.train(p, lgb.Dataset(d.loc[tr, feats], d.loc[tr, TARGET]), num_boost_round=rounds)
        ev = (d.season == Y) & (d.is_F == 0)
        line.append(decompose(d.loc[ev, TARGET].values, m.predict(d.loc[ev, feats]))["score_if_mean_fixed"])
    print(f"  {tag:52s} 2023R:{line[0]:7.1f}  2024R:{line[1]:7.1f}   평균={np.mean(line):7.1f}")
    return np.mean(line)


print("=" * 118)
print("### 시대(era) 보정 피처 추가")
print("=" * 118)
evaluate(F_BASE, tag="기준 (rounds=100)")
evaluate(F_ERA, tag="+ era 보정 피처")

print("\n" + "=" * 118)
print("### 단조 제약")
print("=" * 118)
MONO = {
    "asof_pitcher_success_rate": 1, "p_succ_eb200": 1, "p_succ_eb1000": 1,
    "asof_pitcher_prev1_game_success_rate": 1, "asof_pitcher_prev3_game_success_rate": 1,
    "asof_pitcher_prev5_game_success_rate": 1, "form_blend": 1,
    "asof_pitcher_reverse_rate": -1, "asof_pitcher_middle_rate": -1,
    "asof_batter_success_rate": 1, "b_succ_eb500": 1,
    "asof_batter_middle_rate": -1, "era_adj_career": 1,
}
evaluate(F_ERA, mono=MONO, tag="+ era + 단조제약")
evaluate(F_BASE, mono=MONO, tag="기준 + 단조제약")

print("\n" + "=" * 118)
print("### 콜드스타트(asof_pitcher_n 작은 투수) 구간별 성능")
print("=" * 118)
tr = (d.season <= 2023) & ~((d.is_F == 1) & (d.season <= 2022))
m = lgb.train(PARAMS, lgb.Dataset(d.loc[tr, F_ERA], d.loc[tr, TARGET]), num_boost_round=ROUNDS)
ev = (d.season == 2024) & (d.is_F == 0)
p = m.predict(d.loc[ev, F_ERA])
y = d.loc[ev, TARGET].values
n = d.loc[ev, "asof_pitcher_n"].values
for lo, hi in [(0, 50), (50, 300), (300, 1000), (1000, 3000), (3000, 10 ** 9)]:
    msk = (n >= lo) & (n < hi)
    if msk.sum() < 500:
        continue
    r = decompose(y[msk], p[msk])
    print(f"  asof_pitcher_n [{lo:>5},{hi if hi<10**8 else '∞':>5}) n={msk.sum():7d} "
          f"판별={r['score_if_mean_fixed']:8.1f}  실제평균={r['true_mean']:.4f} 예측평균={r['pred_mean']:.4f}")

print("\n### 중요도 (era 포함)")
imp = pd.Series(m.feature_importance("gain"), index=F_ERA).sort_values(ascending=False)
print((imp / imp.sum() * 100).head(20).to_string(float_format=lambda x: f"{x:.2f}%"))
