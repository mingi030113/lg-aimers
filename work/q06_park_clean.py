"""park 피처의 '레짐 내부' 전이 테스트.

q05 에서 park 가 양쪽 폴드를 깎았는데, 원인은 학습창이 레짐 경계를 걸치기 때문이었다.
  {2021,2022} -> 2023 : 상관 -0.450  (경계 통과)
  {2022,2023} -> 2024 : 상관 +0.006  (뒤집힌 2022 와 안 뒤집힌 2023 이 상쇄)

실제 제출은 {2023,2024} -> 2025 로 **학습창 전체가 반전 이후**다.
이 구조를 재현하는 유일한 조합: train {2023} -> eval 2024 (둘 다 반전 이후).

비교군으로 train {2022} -> 2023 (둘 다 반전 이전, 역시 레짐 내부)도 같이 본다.
두 경우 모두 park 가 도움이 되면 '레짐 내부에서는 유효하다'는 근거가 된다.
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
del raw

BASE_F = [c for c in feature_list(d, use_season=False, use_ids=False)
          if c not in (TARGET, "park")]


def sfix(y, z):
    r = y.mean()
    return 1e5 * (1 - np.mean((sigmoid(z + solve_shift(z, r)) - y) ** 2) / (r * (1 - r)))


def run(feats, cats, seasons, Y):
    tr = (d.season.isin(seasons) & ~((d.is_F == 1) & (d.season <= 2022))).values
    ev = ((d.season == Y) & (d.is_F == 0)).values
    X, y = d.loc[tr, feats], d.loc[tr, TARGET].values
    Xe = d.loc[ev, feats]
    z = np.zeros(int(ev.sum()))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        z += lgb.train(p, lgb.Dataset(X, y, categorical_feature=cats or "auto"),
                       num_boost_round=ROUNDS).predict(Xe, raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit((A - mu) / sd, y)
    B = Xe.fillna(med).to_numpy(np.float64)
    zb = (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])
    return sfix(d.loc[ev, TARGET].values, zb)


# 전이 상관 먼저 확인
sr = d[d.is_F == 0].copy()
sr["dev"] = sr[TARGET] - sr.groupby("season")[TARGET].transform("mean")
piv = sr.pivot_table(index="park", columns="season", values="dev", aggfunc="mean")
piv = piv.loc[piv.index <= 21]

print("=" * 92)
print("### 레짐 내부 단일시즌 학습 테스트")
print("=" * 92)
for seasons, Y, lab in [([2023], 2024, "반전 이후 (실제 과제와 같은 구조)"),
                        ([2022], 2023, "반전 경계 통과 (대조군)"),
                        ([2021], 2022, "반전 이전 (레짐 내부 대조군)")]:
    corr = piv[seasons[0]].corr(piv[Y])
    print(f"\n  --- train {seasons} -> eval {Y}   [{lab}]")
    print(f"      구장효과 전이 상관 = {corr:+.3f}")
    b = run(BASE_F, [], seasons, Y)
    p1 = run(BASE_F + ["park"], [], seasons, Y)
    p2 = run(BASE_F + ["park"], ["park"], seasons, Y)
    print(f"      기준        {b:8.1f}")
    print(f"      +park 수치   {p1:8.1f}   Δ{p1-b:+7.1f}")
    print(f"      +park 범주형  {p2:8.1f}   Δ{p2-b:+7.1f}")

print("\n" + "=" * 92)
print("### 판단 기준")
print("=" * 92)
print("  실제 제출은 train {2023,2024} -> 2025 로 학습창 전체가 반전 이후다.")
print("  위에서 'train [2023] -> 2024' 가 그 구조에 가장 가깝다.")
print("  여기서도 park 가 마이너스면, 구장은 레짐과 무관하게 그냥 쓸모없는 것이다.")
