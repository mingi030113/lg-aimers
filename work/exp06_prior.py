"""실험 6: 직전시즌까지 누적 분할통계(prior_stats) 추가 효과 검증."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import TARGET, decompose, logit, sigmoid
from features import build, feature_list
from prior_stats import build_prior, apply_prior

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")

PARAMS = dict(objective="binary", verbose=-1, num_threads=6, force_row_wise=True,
              bagging_freq=1, bagging_fraction=0.8, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=1000, feature_fraction=0.7, lambda_l2=10.0)
ROUNDS = 300

FOLDS = [(2022, 2023), (2023, 2024)]


def prep(upto, evalY, decay, ks):
    """upto 까지의 라벨만으로 prior 생성 -> upto 이하 학습셋 + evalY 평가셋에 부착."""
    src = raw[raw.season <= upto]
    seasons = sorted(set(src.season.unique().tolist() + [evalY]))
    tables, league = build_prior(src, seasons, decay=decay, k_shrink=ks)
    tr_raw = apply_prior(raw[raw.season <= upto], tables)
    ev_raw = apply_prior(raw[raw.season == evalY], tables)
    tr = build(tr_raw); tr["season"] = tr_raw["season"].values
    ev = build(ev_raw); ev["season"] = ev_raw["season"].values
    return tr, ev


def score(tr, ev, feats, tag):
    msk = ~((tr.is_F == 1) & (tr.season <= 2022))
    ds = lgb.Dataset(tr.loc[msk, feats], tr.loc[msk, TARGET])
    m = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)
    isf = ev["is_F"].values.astype(bool)
    p = m.predict(ev[feats])
    r = decompose(ev.loc[~isf, TARGET].values, p[~isf])
    return r["score_if_mean_fixed"], m


print("=" * 118)
print("### prior 분할통계 추가 효과 (판별력 = 평균고정 점수, R게임 채점)")
print("=" * 118)

for decay, ks in [(0.75, (400, 400, 800, 300, 200)), (0.6, (400, 400, 800, 300, 200)),
                  (0.9, (400, 400, 800, 300, 200)), (0.75, (1500, 1500, 3000, 1000, 800))]:
    line = []
    for upto, Y in FOLDS:
        tr, ev = prep(upto, Y, decay, ks)
        base_f = [c for c in feature_list(tr, use_season=False, use_ids=False)
                  if not any(c.startswith(p) for p in
                             ("p_", "b_", "pteam_")) or c.startswith("p_succ") or c.startswith("pb_")]
        all_f = feature_list(tr, use_season=False, use_ids=False)
        s0, _ = score(tr, ev, base_f, "base")
        s1, m = score(tr, ev, all_f, "with prior")
        line.append((Y, s0, s1))
        if decay == 0.75 and ks[0] == 400 and Y == 2024:
            imp = pd.Series(m.feature_importance("gain"), index=all_f).sort_values(ascending=False)
            keep_imp = imp
    print(f"  decay={decay} k={ks}")
    for Y, s0, s1 in line:
        print(f"      {Y}: base={s0:7.1f}  +prior={s1:7.1f}   Δ={s1-s0:+6.1f}")

print("\n### 새 피처 중요도 (2024 폴드, decay0.75)")
new = [c for c in keep_imp.index if any(c.startswith(p) for p in
       ("p_bh", "p_cnt", "p_run", "pteam", "b_ph")) or c in ("p_n", "p_dev", "p_logn", "b_n", "b_dev", "b_logn")]
print((keep_imp[new] / keep_imp.sum() * 100).to_string(float_format=lambda x: f"{x:.2f}%"))
print("\n전체 상위 20:")
print((keep_imp / keep_imp.sum() * 100).head(20).to_string(float_format=lambda x: f"{x:.2f}%"))
