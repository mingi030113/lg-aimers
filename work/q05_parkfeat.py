"""park(구장) 피처를 실제 모델에 넣어 양쪽 폴드로 검증.

park = 홈팀 = (투수가 홈이면 pitcher_team_id, 아니면 batter_team_id)
  투수홈여부는 top_bottom 으로 100% 복원된다 (score_diff 교차검증 일치율 100.0000%).

사전 예측: 2023 폴드는 나빠질 가능성이 크다.
  구장 편차의 연도 상관이 2022↔2023 에서 -0.490 으로 뒤집힌다(레짐 경계).
  2023 폴드는 train {2021,2022} 라 그 반전을 가로지른다.
  반면 2023↔2024 는 +0.694 이고 2025 도 같은 레짐이다.
  -> 이번엔 '양쪽 폴드' 규칙과 메커니즘 근거가 충돌할 수 있다. 숫자를 보고 판단한다.

변형
  (a) 기준 (현재 100피처)
  (b) + park (수치)
  (c) + park (LGBM 네이티브 범주형)
  (d) + park + park×투수홈
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from features import build, feature_list, attach_trackman
from train_final import (PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W,
                         sigmoid, solve_shift, TARGET)

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

p_home = (raw["top_bottom"] == "T").values
park = np.where(p_home, raw["pitcher_team_id"].values, raw["batter_team_id"].values)

d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
d["park"] = park.astype(np.int16)
d["park_home"] = (park * 2 + p_home).astype(np.int16)
del raw

BASE_F = [c for c in feature_list(d, use_season=False, use_ids=False)
          if c not in (TARGET, "park", "park_home")]
print(f"기준 피처 {len(BASE_F)}개")

VAR = {
    "(a) 기준": (BASE_F, []),
    "(b) +park (수치)": (BASE_F + ["park"], []),
    "(c) +park (범주형)": (BASE_F + ["park"], ["park"]),
    "(d) +park +park×홈 (범주형)": (BASE_F + ["park", "park_home"], ["park", "park_home"]),
}


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


def run(feats, cats, upto, Y):
    tr = ((d.season >= upto - 1) & (d.season <= upto)
          & ~((d.is_F == 1) & (d.season <= 2022))).values
    ev = ((d.season == Y) & (d.is_F == 0)).values
    X, y = d.loc[tr, feats], d.loc[tr, TARGET].values
    Xe = d.loc[ev, feats]
    z = np.zeros(int(ev.sum()))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        ds = lgb.Dataset(X, y, categorical_feature=cats or "auto")
        z += lgb.train(p, ds, num_boost_round=ROUNDS).predict(Xe, raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit((A - mu) / sd, y)
    B = Xe.fillna(med).to_numpy(np.float64)
    zb = (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])
    return sfix(d.loc[ev, TARGET].values, zb)


print("\n" + "=" * 92)
print("### park 피처 검증 (train {Y-2,Y-1} -> eval Y, R채점, 레벨 고정)")
print("=" * 92)
print(f"  {'구성':28s} {'2023R':>9} {'2024R':>9} {'평균':>9} {'Δ평균':>8}")
b = None
for tag, (fs, cats) in VAR.items():
    s = [run(fs, cats, upto, Y) for upto, Y in [(2022, 2023), (2023, 2024)]]
    m = np.mean(s)
    if b is None:
        b, s0 = m, s
    both = "  ← 양쪽 개선" if s[0] > s0[0] and s[1] > s0[1] else ""
    print(f"  {tag:28s} {s[0]:9.1f} {s[1]:9.1f} {m:9.1f} {m-b:+8.1f}{both}")

print("\n" + "=" * 92)
print("### 참고: 폴드별 구장효과 전이 가능성 (학습창 -> 평가시즌 상관)")
print("=" * 92)
sr = d[(d.is_F == 0)].copy()
sr["dev"] = sr[TARGET] - sr.groupby("season")[TARGET].transform("mean")
piv = sr.pivot_table(index="park", columns="season", values="dev", aggfunc="mean")
piv = piv.loc[piv.index <= 21]
for upto, Y in [(2022, 2023), (2023, 2024)]:
    tr_eff = piv[[upto - 1, upto]].mean(axis=1)
    print(f"  학습창 {{{upto-1},{upto}}} 평균 vs {Y}: 상관 {tr_eff.corr(piv[Y]):+.3f}")
print("  -> 2025 는 학습창 {2023,2024} 와 같은 레짐이므로 2024 폴드 쪽이 실제와 가깝다.")
