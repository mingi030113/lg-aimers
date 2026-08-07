"""실험 7: '레벨을 명시적으로 분리'하는 설계 검증.

설계 A (기본)      : 원 피처로 학습, 사후에 로짓 시프트로 평균 이동
설계 B (init_score): init_score = logit(리그레벨(시즌, game_type)) 로 주고 잔차만 학습.
                     추론 시 리그레벨을 우리가 정한 2025 목표값으로 지정 -> 레벨 완전 통제.
설계 C (B + 시즌차감 피처): rate 계열 피처를 시즌 리그평균 대비 편차로 변환.

평가: 1년 앞 전방검증에서 (판별력, 실제 총점) 둘 다 확인.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import TARGET, decompose, logit, sigmoid
from features import build, feature_list

pd.set_option("display.width", 250)

raw = pd.read_parquet("train.parquet")
d = build(raw)
d["season"] = raw["season"].values
del raw

# 시즌 x game_type 리그 레벨 (train 라벨에서만 계산)
LVL = d.groupby(["season", "is_F"])[TARGET].mean()

RATE_COLS = [c for c in d.columns if c.endswith("_rate") or c.startswith("p_succ_eb")
             or c == "b_succ_eb500" or c == "form_blend"]
RATE_SUCC = [c for c in RATE_COLS if "success" in c or c.startswith("p_succ") or c == "b_succ_eb500"
             or c == "form_blend"]
print("성공률 계열(시즌차감 대상):", RATE_SUCC)

PARAMS = dict(objective="binary", verbose=-1, num_threads=6, force_row_wise=True,
              bagging_freq=1, bagging_fraction=0.8, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=1000, feature_fraction=0.7, lambda_l2=10.0)
ROUNDS = 150
F0 = feature_list(d, use_season=False, use_ids=False)

# --- 라운드 수 미세 스윕 (150 이하가 더 좋은지 확인) ---
print("=" * 118)
print("### 라운드 수 스윕 (판별력, R채점)")
print("=" * 118)
_P = dict(PARAMS)
for R in [50, 75, 100, 125, 150, 200, 250]:
    line = []
    for _upto, _Y in [(2022, 2023), (2023, 2024)]:
        _tr = (d.season <= _upto) & ~((d.is_F == 1) & (d.season <= 2022))
        _m = lgb.train(_P, lgb.Dataset(d.loc[_tr, F0], d.loc[_tr, TARGET]), num_boost_round=R)
        _ev = (d.season == _Y) & (d.is_F == 0)
        _r = decompose(d.loc[_ev, TARGET].values, _m.predict(d.loc[_ev, F0]))
        line.append(_r["score_if_mean_fixed"])
    print(f"  rounds={R:4d}   2023R:{line[0]:7.1f}  2024R:{line[1]:7.1f}   평균={np.mean(line):7.1f}")


def trainable(upto):
    return (d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))


def level_of(season_arr, isF_arr, override=None):
    """행별 리그레벨. override={(season,isF):value} 로 미래시즌 지정."""
    out = np.empty(len(season_arr))
    for i, (s, f) in enumerate(zip(season_arr, isF_arr)):
        if override and (s, f) in override:
            out[i] = override[(s, f)]
        else:
            out[i] = LVL.get((s, f), 0.52)
    return out


def lvl_vec(sub, override=None):
    key = pd.MultiIndex.from_arrays([sub["season"].values, sub["is_F"].values])
    v = LVL.reindex(key).values
    if override:
        for (s, f), val in override.items():
            m = (sub["season"].values == s) & (sub["is_F"].values == f)
            v = np.where(m, val, v)
    return np.where(np.isnan(v), 0.52, v)


def demean(sub, lv):
    x = sub.copy()
    for c in RATE_SUCC:
        if c in x.columns:
            x[c] = x[c] - lv
    return x


print("\n" + "=" * 118)
print("### 설계 비교 (전방검증). '목표평균'은 '직전시즌−0.010' 규칙으로 train만 보고 정함")
print("=" * 118)

for upto, Y in [(2022, 2023), (2023, 2024)]:
    tr = trainable(upto)
    ev = d.season == Y
    evR = ev & (d.is_F == 0)
    yR = d.loc[evR, TARGET].values
    # train 만 보고 정한 2가지 목표평균
    prevR = LVL.get((upto, 0))
    tgtR = prevR - 0.010
    print(f"\n--- train<={upto} -> eval {Y} (R게임 채점) | 직전R={prevR:.4f} 목표={tgtR:.4f} 실제={yR.mean():.4f}")

    # ===== A: 기본 =====
    m = lgb.train(PARAMS, lgb.Dataset(d.loc[tr, F0], d.loc[tr, TARGET]), num_boost_round=ROUNDS)
    pA = m.predict(d.loc[evR, F0])
    rA = decompose(yR, pA)
    print(f"  A. 기본 (보정 없음)                    총점={rA['score']:7.1f} 판별={rA['score_if_mean_fixed']:7.1f} pred_mean={rA['pred_mean']:.4f}")
    # A + 목표평균으로 시프트 (2024년도 train 행 기준으로 b 결정 -> 실전 절차 그대로)
    ref = d.season == upto
    pref = m.predict(d.loc[ref & (d.is_F == 0), F0])
    lo, hi = -3.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(logit(pref) + mid).mean() < tgtR:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    pA2 = sigmoid(logit(pA) + b)
    rA2 = decompose(yR, pA2)
    print(f"  A'. 직전시즌 기준 로짓시프트(b={b:+.4f})   총점={rA2['score']:7.1f} 판별={rA2['score_if_mean_fixed']:7.1f} pred_mean={rA2['pred_mean']:.4f}")

    # ===== B: init_score =====
    lv_tr = lvl_vec(d.loc[tr])
    ds = lgb.Dataset(d.loc[tr, F0], d.loc[tr, TARGET], init_score=logit(lv_tr))
    pB_ = dict(PARAMS, boost_from_average=False)
    mB = lgb.train(pB_, ds, num_boost_round=ROUNDS)
    lv_ev = np.full(evR.sum(), tgtR)
    pB = sigmoid(logit(lv_ev) + mB.predict(d.loc[evR, F0], raw_score=True))
    rB = decompose(yR, pB)
    print(f"  B. init_score(리그레벨) + 2025목표 주입   총점={rB['score']:7.1f} 판별={rB['score_if_mean_fixed']:7.1f} pred_mean={rB['pred_mean']:.4f}")

    # ===== C: B + 시즌차감 피처 =====
    trd = demean(d.loc[tr, F0], lvl_vec(d.loc[tr]))
    evd = demean(d.loc[evR, F0], np.full(evR.sum(), tgtR))
    dsC = lgb.Dataset(trd, d.loc[tr, TARGET], init_score=logit(lv_tr))
    mC = lgb.train(pB_, dsC, num_boost_round=ROUNDS)
    pC = sigmoid(logit(np.full(evR.sum(), tgtR)) + mC.predict(evd, raw_score=True))
    rC = decompose(yR, pC)
    print(f"  C. B + rate 시즌차감                    총점={rC['score']:7.1f} 판별={rC['score_if_mean_fixed']:7.1f} pred_mean={rC['pred_mean']:.4f}")

    # 오라클 목표평균을 줬다면?
    lv_o = np.full(evR.sum(), yR.mean())
    pBo = sigmoid(logit(lv_o) + mB.predict(d.loc[evR, F0], raw_score=True))
    print(f"     [참고] B에 오라클 목표평균 주입          총점={decompose(yR, pBo)['score']:7.1f}")
    evdo = demean(d.loc[evR, F0], lv_o)
    pCo = sigmoid(logit(lv_o) + mC.predict(evdo, raw_score=True))
    print(f"     [참고] C에 오라클 목표평균 주입          총점={decompose(yR, pCo)['score']:7.1f}")
