"""부분집단 재보정이 실제로 점수를 올리는지 out-of-sample 로 검증.

cal01 의 '회수 가능 점수'는 in-sample 이라 잡음이 섞여 있다
(그룹 G개, 표본 N개면 순수 잡음만으로 약 1e5*G/N/0.25*0.25 = G/N*1e5 점이 나온다).

진짜 검증: 폴드 A(2023 R)에서 그룹별 보정을 학습 -> 폴드 B(2024 R)에 적용 -> 실제 Δ점수.
전역 레벨은 각 폴드에서 이미 제거하고(정답 평균에 맞춤) '상대적' 그룹 편향만 이전시킨다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from features import build, feature_list, attach_trackman
from train_final import (PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W,
                         DRIFT, D_AUTO, sigmoid, solve_shift, TARGET)

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
idmap = pd.read_parquet("idmap_pitcher_v2.parquet")
sel = (idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
       .drop_duplicates("tm_pitcher_id", keep="first"))
t2p = {v: k for k, v in zip(sel.pitcher_id, sel.tm_pitcher_id)}
pri = pd.read_parquet("tm_prior.parquet")
pri["pitcher_id"] = pri["pitcher_trackman_id"].map(t2p)
pri = pri.dropna(subset=["pitcher_id"]).drop(columns=["pitcher_trackman_id"])
pri["pitcher_id"] = pri["pitcher_id"].astype(int)
raw = attach_trackman(raw, pri)
d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
FEATS = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
del raw


def logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_fold(upto):
    msk = ((d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))).values
    X, y = d.loc[msk, FEATS], d.loc[msk, TARGET].values
    z = np.zeros(len(d))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        z += lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS).predict(d[FEATS], raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit((A - mu) / sd, y)
    B = d[FEATS].fillna(med).to_numpy(np.float64)
    return (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])


ZS = {}
for upto, Y in [(2022, 2023), (2023, 2024)]:
    ZS[Y] = fit_fold(upto)
    print(f"폴드 {Y} 예측 완료")
np.savez("cal_z.npz", **{str(k): v for k, v in ZS.items()})


def fold_data(Y):
    ev = ((d.season == Y) & (d.is_F == 0)).values
    sub = d.loc[ev].reset_index(drop=True)
    z = ZS[Y][ev]
    y = sub[TARGET].values
    # 전역 레벨은 정답에 맞춰 제거 -> 그룹 '상대' 편향만 본다
    z = z + solve_shift(z, y.mean())
    return z, y, sub


def score(y, p):
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


AXES = {
    "볼카운트": lambda s: (s["balls_before"].astype(str) + "-" + s["strikes_before"].astype(str)).values,
    "투수손x타자손": lambda s: (s["pitcher_hand"].astype(str) + "x" + s["batter_hand"].astype(str)).values,
    "월": lambda s: s["game_month"].astype(str).values,
    "이닝": lambda s: np.clip(s["inning"].values, 1, 10).astype(str),
    "투수경험": lambda s: pd.cut(s["asof_pitcher_n"].values, [-1, 0, 50, 200, 1000, 3000, 10**9],
                             labels=["0", "1-50", "51-200", "201-1k", "1k-3k", "3k+"]).astype(str),
    "주자상태": lambda s: s["base_code"].astype(str).values,
}

zA, yA, subA = fold_data(2023)
zB, yB, subB = fold_data(2024)
base_B = score(yB, sigmoid(zB))
print(f"\n기준: 2024R 전역레벨 정답고정 점수 = {base_B:.1f}")
print(f"       (2023R 동일 기준 = {score(yA, sigmoid(zA)):.1f})")

print("\n" + "=" * 100)
print("### 그룹 보정: 2023R 에서 학습 -> 2024R 에 적용한 실제 Δ점수")
print("=" * 100)
print(f"  {'축':16s} {'그룹수':>5} {'in-sample(2024)':>16} {'out-of-sample Δ':>17}")
for name, fn in AXES.items():
    gA, gB = fn(subA), fn(subB)
    tA = pd.DataFrame({"g": gA, "z": zA, "y": yA})
    # 그룹별 로짓 보정: 그 그룹의 예측평균을 실제평균에 맞추는 delta
    corr = {}
    for g, t in tA.groupby("g"):
        if len(t) < 300:
            continue
        corr[g] = solve_shift(t["z"].values, t["y"].values.mean())
    dB = np.array([corr.get(g, 0.0) for g in gB])
    p_oos = sigmoid(zB + dB)
    # 보정 후 전역 레벨이 흔들리므로 다시 고정 (레벨 효과와 분리)
    p_oos = sigmoid(logit(p_oos) + solve_shift(logit(p_oos), yB.mean()))
    # in-sample 상한 참고용
    tB = pd.DataFrame({"g": gB, "z": zB, "y": yB})
    corrB = {g: solve_shift(t["z"].values, t["y"].values.mean())
             for g, t in tB.groupby("g") if len(t) >= 300}
    p_is = sigmoid(zB + np.array([corrB.get(g, 0.0) for g in gB]))
    print(f"  {name:16s} {len(corr):5d} {score(yB, p_is)-base_B:+15.1f} {score(yB, p_oos)-base_B:+16.1f}")

print("\n" + "=" * 100)
print("### 전역 로짓 재척도: 2023R 에서 최적 s 를 찾아 2024R 에 적용")
print("=" * 100)
best_s, best_v = None, -1e18
for s in np.arange(0.7, 1.45, 0.05):
    zz = zA * s
    v = score(yA, sigmoid(zz + solve_shift(zz, yA.mean())))
    if v > best_v:
        best_v, best_s = v, s
print(f"  2023R 최적 s = {best_s:.2f} (그 폴드 점수 {best_v:.1f}, s=1일 때 {score(yA, sigmoid(zA)):.1f})")
for s in [0.8, 0.9, 1.0, 1.1, 1.2, best_s]:
    zz = zB * s
    v = score(yB, sigmoid(zz + solve_shift(zz, yB.mean())))
    tag = "  <- 2023 최적" if abs(s - best_s) < 1e-9 else ""
    print(f"    2024R  s={s:.2f}: {v:8.1f}  (Δ{v-base_B:+.1f}){tag}")
