"""결정적 가설 검증: 우리 폴드는 전부 '레짐 전환 폴드'라서 실제 과제를 과소평가한다.

배경
  KBO 는 2024년에 ABS(자동 볼판정)를 도입했고, 2023년에도 스트라이크존 판정이 바뀌었다.
  control_success 가 존 기하에 기반한 라벨이라면 2023, 2024 는 각각 불연속점이다.

  우리 폴드:      train 2019~2023 (ABS 이전)  -> eval 2024 (ABS 원년)   = 레짐 횡단
  실제 과제:      train 2019~2024 (ABS 포함)  -> eval 2025 (ABS 2년차)  = 레짐 내부

  이게 맞다면 폴드는 구조적으로 비관적이고, LB 가 +94 높은 게 자연스럽다.

검증
  2024 를 전후반으로 쪼개, 같은 2024 후반기를 두 가지로 예측한다.
    (a) 2024 전반기만으로 학습  -> 표본은 적지만 같은 레짐
    (b) 2019~2023 으로 학습     -> 표본은 12배 많지만 다른 레짐
  (a) 가 (b) 에 필적하거나 이기면 '레짐 일치'가 표본 수보다 중요하다는 뜻.
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

d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
d["row_ord"] = np.arange(len(d))
FEATS = [c for c in feature_list(d, use_season=False, use_ids=False)
         if c not in (TARGET, "row_ord")]
del raw


def sfix(y, z):
    r = y.mean()
    z2 = z + solve_shift(z, r)
    return 1e5 * (1 - np.mean((sigmoid(z2) - y) ** 2) / (r * (1 - r)))


def fit_eval(train_mask, eval_mask, rounds=ROUNDS):
    X, y = d.loc[train_mask, FEATS], d.loc[train_mask, TARGET].values
    z = np.zeros(int(eval_mask.sum()))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        z += lgb.train(p, lgb.Dataset(X, y), num_boost_round=rounds).predict(
            d.loc[eval_mask, FEATS], raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit((A - mu) / sd, y)
    B = d.loc[eval_mask, FEATS].fillna(med).to_numpy(np.float64)
    zb = (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])
    return sfix(d.loc[eval_mask, TARGET].values, zb), int(train_mask.sum())


# 2024 R 게임을 전/후반으로 (row_id 순 = 경기 순)
r24 = ((d.season == 2024) & (d.is_F == 0)).values
idx = np.where(r24)[0]
half = idx[len(idx) // 2]
first24 = np.zeros(len(d), bool); first24[idx[:len(idx) // 2]] = True
second24 = np.zeros(len(d), bool); second24[idx[len(idx) // 2:]] = True
print(f"2024R 전반 {first24.sum():,}행 / 후반 {second24.sum():,}행")

print("\n" + "=" * 100)
print("### 같은 '2024 후반기'를 서로 다른 학습셋으로 예측")
print("=" * 100)
pre23 = ((d.season <= 2023) & ~((d.is_F == 1) & (d.season <= 2022))).values
for tag, m, rounds in [
        ("(a) 2024 전반기만 (같은 레짐, 소표본)", first24, 60),
        ("(b) 2019~2023 (다른 레짐, 대표본)", pre23, ROUNDS),
        ("(c) 2019~2023 + 2024전반 (혼합)", pre23 | first24, ROUNDS)]:
    sc, n = fit_eval(m, second24, rounds=rounds)
    print(f"  {tag:36s} 학습 {n:>9,}행 -> 판별력 {sc:7.1f}")

print("\n" + "=" * 100)
print("### 참고: 2023 도 같은 방식으로 (2023 은 존 판정 변경 해)")
print("=" * 100)
r23 = ((d.season == 2023) & (d.is_F == 0)).values
i23 = np.where(r23)[0]
f23 = np.zeros(len(d), bool); f23[i23[:len(i23) // 2]] = True
s23 = np.zeros(len(d), bool); s23[i23[len(i23) // 2:]] = True
pre22 = ((d.season <= 2022) & ~((d.is_F == 1) & (d.season <= 2022))).values
for tag, m, rounds in [
        ("(a) 2023 전반기만", f23, 60),
        ("(b) 2019~2022", pre22, ROUNDS),
        ("(c) 2019~2022 + 2023전반", pre22 | f23, ROUNDS)]:
    sc, n = fit_eval(m, s23, rounds=rounds)
    print(f"  {tag:36s} 학습 {n:>9,}행 -> 판별력 {sc:7.1f}")

print("\n" + "=" * 100)
print("### F 게임 판별력의 시간 추이 (레짐 성숙 효과)")
print("=" * 100)
Z = np.load("cal_z.npz")
for Y in (2023, 2024):
    ev = ((d.season == Y) & (d.is_F == 1)).values
    z = Z[str(Y)][ev]
    y = d.loc[ev, TARGET].values
    print(f"  {Y} F: n={ev.sum():6d}  판별력 {sfix(y, z):7.1f}")
print("  (2023 F 는 학습셋에 신 레짐 F 가 0시즌, 2024 F 는 1시즌 들어있다.")
print("   2025 F 는 2시즌 -> 계속 좋아질 것으로 예상)")
