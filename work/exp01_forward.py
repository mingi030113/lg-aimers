"""실험 1: 전방검증(train <= Y-1, eval = Y)으로 '1년 앞 예측' 상황을 그대로 재현.

test=2025 이므로 실제 대회는 train<=2024 -> 2025 예측.
따라서 검증도 반드시 1년 앞 홀드아웃이어야 한다. 랜덤 KFold는 드리프트를 못 잡는다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import TARGET, ID, score_bss, decompose, logit, sigmoid

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 100)

t = pd.read_parquet("train.parquet")

# --- 인코딩 ---
t["top_bottom"] = (t["top_bottom"] == "B").astype(np.int8)
t["game_type_F"] = (t["game_type"] == "F").astype(np.int8)
BASE_MAP = {"___": 0, "1__": 1, "_2_": 2, "12_": 3, "__3": 4, "1_3": 5, "_23": 6, "123": 7}
t["base_state"] = t["base_state"].map(BASE_MAP).astype(np.int8)
t = t.drop(columns=["game_type", "asof_pitcher_pitchmix_n", "away_win_expectancy"])

ASOF_PREV = ["asof_pitcher_prev1_game_success_rate",
             "asof_pitcher_prev3_game_success_rate",
             "asof_pitcher_prev5_game_success_rate"]

ALL_FEATS = [c for c in t.columns if c not in (ID, TARGET)]
NO_SEASON = [c for c in ALL_FEATS if c != "season"]

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=6,
              force_row_wise=True)


def run(feats, train_mask, eval_mask, rounds=400, weight=None, tag=""):
    Xtr = t.loc[train_mask, feats]
    ytr = t.loc[train_mask, TARGET].values
    Xte = t.loc[eval_mask, feats]
    yte = t.loc[eval_mask, TARGET].values
    w = None if weight is None else weight[train_mask.values]
    ds = lgb.Dataset(Xtr, ytr, weight=w, free_raw_data=True)
    m = lgb.train(PARAMS, ds, num_boost_round=rounds)
    p = m.predict(Xte)
    d = decompose(yte, p)
    print(f"  {tag:52s} score={d['score']:8.1f}  (평균고정시 {d['score_if_mean_fixed']:8.1f}, "
          f"레벨손실 {d['level_loss']:8.1f}, pred_mean={d['pred_mean']:.4f} true={d['true_mean']:.4f})")
    return m, p, d


for Y in [2023, 2024]:
    print(f"\n{'='*110}\n### 전방검증: train <= {Y-1}  ->  eval {Y}\n{'='*110}")
    tr = t["season"] <= Y - 1
    ev = t["season"] == Y
    yte = t.loc[ev, TARGET].values

    # 0) 상수 예측들
    r_tr = t.loc[tr, TARGET].mean()
    print(f"  {'[const] train 전체 평균':52s} score={score_bss(yte, np.full(len(yte), r_tr)):8.1f}  (p={r_tr:.4f})")
    r_last = t.loc[t["season"] == Y - 1, TARGET].mean()
    print(f"  {'[const] 직전시즌 평균':52s} score={score_bss(yte, np.full(len(yte), r_last)):8.1f}  (p={r_last:.4f})")
    # 선형 추세 외삽
    ss = t[t.season <= Y - 1].groupby("season")[TARGET].mean()
    co = np.polyfit(ss.index.values[-3:], ss.values[-3:], 1)
    r_ext = np.polyval(co, Y)
    print(f"  {'[const] 최근3시즌 선형외삽':52s} score={score_bss(yte, np.full(len(yte), r_ext)):8.1f}  (p={r_ext:.4f})")
    print(f"  {'[const] 오라클(정답 평균)':52s} score={score_bss(yte, np.full(len(yte), yte.mean())):8.1f}  (p={yte.mean():.4f})")
    # asof_prev5 를 그대로 확률로 (per-row, 규정 안전)
    for c in ASOF_PREV + ["asof_pitcher_success_rate"]:
        p = t.loc[ev, c].fillna(r_last).values
        print(f"  [const-free] {c[:39]:39s} score={score_bss(yte, p):8.1f}  (p={p.mean():.4f})")

    # 1) 전체 피처 + season
    run(ALL_FEATS, tr, ev, tag="[LGBM] 전체피처 (season 포함)")
    # 2) season 제거
    run(NO_SEASON, tr, ev, tag="[LGBM] 전체피처 (season 제거)")
    # 3) season 제거 + 최근 시즌 가중
    wt = np.where(t["season"] >= Y - 2, 3.0, 1.0)
    run(NO_SEASON, tr, ev, weight=wt, tag="[LGBM] season제거 + 최근2시즌 x3 가중")
    # 4) 최근 2시즌만 학습
    run(NO_SEASON, (t["season"] >= Y - 2) & (t["season"] <= Y - 1), ev,
        tag="[LGBM] 최근 2시즌만 학습")
    # 5) 최근 1시즌만
    run(NO_SEASON, t["season"] == Y - 1, ev, tag="[LGBM] 직전 1시즌만 학습")
