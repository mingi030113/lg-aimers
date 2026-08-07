"""실험 4: (a) 모델의 드리프트 자동적응력 측정  (b) 피처셋별 판별력 비교."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import TARGET, score_bss, decompose, logit, sigmoid
from features import build, feature_list

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
d = build(raw)
d["season"] = raw["season"].values
print("built:", d.shape)

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, num_threads=6, force_row_wise=True)


def trainable(upto):
    """학습 가능 마스크: <=upto 시즌, 2022 이전 F 게임 제외 (라벨 레짐 다름)."""
    return (d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))


def fit(mask, feats, rounds=500, params=None, cat=None):
    p = dict(PARAMS if params is None else params)
    ds = lgb.Dataset(d.loc[mask, feats], d.loc[mask, TARGET],
                     categorical_feature=cat or [], free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=rounds)


# ============================================================================ #
print("\n" + "=" * 112)
print("### (a) 드리프트 자동적응력: train<=2022(R기준) 모델이 2023/2024를 얼마나 스스로 따라 내려가는가")
print("=" * 112)
f0 = feature_list(d, use_season=False, use_ids=False)
m = fit(trainable(2022), f0)
for Y in [2022, 2023, 2024]:
    ev = (d.season == Y) & (d.is_F == 0)
    p = m.predict(d.loc[ev, f0])
    print(f"  {Y}:  pred_mean={p.mean():.4f}   true_mean={d.loc[ev, TARGET].mean():.4f}")
prev = {}
for Y in [2022, 2023, 2024]:
    ev = (d.season == Y) & (d.is_F == 0)
    prev[Y] = (m.predict(d.loc[ev, f0]).mean(), d.loc[ev, TARGET].mean())
for Y in [2023, 2024]:
    da = prev[Y][0] - prev[Y - 1][0]
    dt = prev[Y][1] - prev[Y - 1][1]
    print(f"  {Y-1}->{Y}: Δ_auto={da:+.4f}  Δ_true={dt:+.4f}  추종률={da/dt*100 if dt else 0:5.1f}%")

# ============================================================================ #
print("\n" + "=" * 112)
print("### (b) 피처셋 비교 (train<=2023 -> eval 2024, 판별력=평균고정점수 기준)")
print("=" * 112)
ev = d.season == 2024
yte = d.loc[ev, TARGET].values
tr = trainable(2023)


def report(tag, p):
    r = decompose(yte, p)
    print(f"  {tag:52s} score={r['score']:8.1f} | 판별력(평균고정) {r['score_if_mean_fixed']:8.1f} "
          f"| pred_mean={r['pred_mean']:.4f}")
    return r


RAW_ONLY = [c for c in raw.columns if c not in ("row_id", "control_success", "season",
                                                "game_type", "top_bottom", "base_state",
                                                "asof_pitcher_pitchmix_n", "away_win_expectancy")]
RAW_ONLY = [c for c in RAW_ONLY if c in d.columns]

sets = {
    "1. 원본 컬럼만": RAW_ONLY,
    "2. +파생 전부": f0,
    "3. +파생 +season": feature_list(d, use_season=True, use_ids=False),
    "4. +파생 +pitcher/batter id(cat)": feature_list(d, use_season=False, use_ids=True),
}
models = {}
for k, fs in sets.items():
    cat = ["pitcher_id", "batter_id"] if "id(cat)" in k else []
    mm = fit(tr, fs, cat=cat)
    models[k] = (mm, fs)
    report(k, mm.predict(d.loc[ev, fs]))

print("\n  --- 부스팅 라운드 / 정규화 스윕 (피처셋 2) ---")
for rounds, leaves, mdl in [(300, 31, 1000), (500, 63, 500), (800, 63, 1000), (1500, 31, 2000), (3000, 15, 3000)]:
    pp = dict(PARAMS, num_leaves=leaves, min_data_in_leaf=mdl)
    mm = fit(tr, f0, rounds=rounds, params=pp)
    report(f"   rounds={rounds} leaves={leaves} minleaf={mdl}", mm.predict(d.loc[ev, f0]))

print("\n  --- 학습률/깊이 보수적 세팅 ---")
for lr, rounds, leaves in [(0.02, 2000, 31), (0.03, 1200, 31), (0.01, 3000, 63)]:
    pp = dict(PARAMS, learning_rate=lr, num_leaves=leaves, min_data_in_leaf=2000)
    mm = fit(tr, f0, rounds=rounds, params=pp)
    report(f"   lr={lr} rounds={rounds} leaves={leaves}", mm.predict(d.loc[ev, f0]))

# ============================================================================ #
print("\n" + "=" * 112)
print("### (c) 피처 중요도 (gain) — 피처셋 2, train<=2023")
print("=" * 112)
mm, fs = models["2. +파생 전부"]
imp = pd.Series(mm.feature_importance("gain"), index=fs).sort_values(ascending=False)
print((imp / imp.sum() * 100).head(35).to_string(float_format=lambda x: f"{x:.2f}%"))
print("\n  하위 15개:")
print((imp / imp.sum() * 100).tail(15).to_string(float_format=lambda x: f"{x:.3f}%"))
