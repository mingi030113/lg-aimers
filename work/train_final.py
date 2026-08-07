"""최종 학습 스크립트. 산출물: model/ (부스터 5개 + 선형모델 + meta.json)

설계 요약
  1) 학습셋: TRAIN_FROM~2024, 단 2022년 이전 F(퓨처스) 행 제외 — 라벨 레짐이 다름
     ※ TRAIN_FROM 은 전방검증이 아니라 LB 로 확정한다. 폴드는 이 결정에서 부호까지 틀렸다
       (2024 폴드: 2시즌 학습이 5시즌보다 -21.3 -> 실제 LB 는 +108.9).
       모든 폴드가 최신 시즌을 1개만 가질 수 있어 실제 과제 구조를 재현하지 못하기 때문.
  2) season 컬럼 미사용 — 2025는 미학습 구간이라 트리가 외삽 불가
  3) LGBM(강한 정규화, rounds=100, 5시드) + L2 로지스틱 블렌드 (신호가 거의 선형)
  4) 레벨 통제: game_type 별로 로짓 시프트를 미리 계산해 저장
       목표 2025 평균 = (2024 실제평균) + DRIFT
       단, 모델 자체가 1년당 D_AUTO 만큼 자동으로 내려가므로 그만큼 상쇄
"""
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

from qt_numpy import qt_transform

from features import build, feature_list, attach_trackman
from prior_stats import build_prior, apply_prior

DATA = "../data"
OUT = "model"
TARGET = "control_success"

# Trackman 구위 피처 사용 여부.
#   전방검증(tm04): 2023R −14.9 / 2024R +15.8 -> '두 폴드 모두 개선' 규칙은 통과 못 함.
#   2025가 ABS 2년차로 안정된 해라는 판단에 베팅하는 실험 제출용으로 True.
#   되돌리려면 False + D_AUTO 를 기본값(R −0.0046 / F −0.011)으로.
USE_TRACKMAN = True

# 학습 시작 시즌. 2019 = 전체.
#   레짐 가설: control_success 라벨 레짐이 2023년에 끊겼다(F 성공률 .709 -> .473).
#   2023/2024/2025 가 같은 레짐이므로 2023~2024 만 쓰는 게 나을 수 있다.
#   근거: 2023 후반기를 예측할 때 2023 전반기 11만행(766.9)이 2019~2022 87만행(508.3)을
#         압도했고, 섞으면 664.3 으로 오히려 떨어졌다 (rev02).
#   반대 증거: 2024 폴드에서는 2시즌(719.9) < 5시즌(741.2). 단 그 폴드는 어느 쪽이든
#         같은 레짐 시즌이 1개뿐이라 가설을 검증하지 못한다. LB 로 직접 잰다.
TRAIN_FROM = 2023

# 지수 감쇠 가중. None 이면 균등. 값을 주면 TRAIN_FROM 은 2019 로 두고 전체를 쓴다.
#   가중 = 0.5 ** ((2024 - season) / DECAY_HALFLIFE)
#   반감기 스윕(rev05, 양쪽 폴드 R판별력):
#     0.50 -> 651.2 | 0.75 -> 656.6 | 1.00 -> 662.2 | 1.50 -> 661.4 | 2.00 -> 660.9 | 3.00 -> 656.7
#     (균등 전체 649.2 / 2023~2024 하드컷 639.3)
#   하드 컷오프와 달리 **두 폴드 모두** 개선하고 1.0~2.0 이 평평한 고원이라 견고하다.
DECAY_HALFLIFE = None

# 직접 만드는 '직전 시즌까지' 선수 분할통계 (좌우타 상대 / 볼카운트 구간 / 주자 유무).
#   리그평균을 (시즌 × game_type) 으로 정규화하므로 드리프트에 중립적이다.
#   시즌 S 행에는 시즌 < S 집계만 붙는다 (test 2025 -> 2019~2024). 누수 없음.
USE_PRIOR = False   # 실측 기각: prior 포함 시 LB 937.10 -> 931.14
PRIOR_DECAY = 0.9

# 볼카운트(b−s) 잔차 보정. fin01_calib.py 가 두 폴드의 out-of-sample 잔차에서 산출.
#   보정을 적용한 뒤 레벨 시프트를 풀어야 평균이 맞는다.
#   fin01_calib.py 산출 (λ=0.6, 두 폴드 OOS 평균). 양방향 +4.2 / +1.3
COUNT_CAL = {}      # 931 제출에 prior 와 함께 섞여 있어 단독 효과 미측정. 최고본(937)에는 미적용
# 참고값(λ=0.6): {"-2":-0.004133,"-1":0.012512,"0":-0.000616,"1":-0.012254,"2":-0.005719,"3":0.050902}

# MLP 3번째 블렌드 멤버. 단독 성능이 아니라 '다르게 틀리는' 다양성으로 기여한다.
#   (32,16) a=1.0 이 봉우리: 더 크거나(48,24 +4.6) 더 작으면(16,8 +3.2) 떨어진다.
#   시드를 평균낼수록 기여가 준다(1개 +12.0 / 3개 +11.9 / 5개 +9.8) — 매끄러워질수록
#   LGBM 과 닮아 다양성이 사라지기 때문. 시드 복불복을 줄이는 선에서 3개 채택.
#   비중 0.05~0.25 전 구간이 양쪽 폴드 개선인 넓은 고원. 0.20 채택.
USE_MLP = True
MLP_HIDDEN = (32, 16)
MLP_ALPHA = 1.0
MLP_SEEDS = 3
MLP_W = 0.20

# 두 번째 MLP — 구조가 아니라 **전처리**를 바꿔 오차 방향을 갈라놓는다.
#   분위수변환(정규)이 z-표준화 대비 이상치·꼬리를 다르게 취급 -> MLP1 과 상관 0.869.
#   후보 5종 중 4종이 양쪽 폴드 개선. 전처리를 크게 바꾼 A 가 1등(+10.9),
#   구조만 바꾼 tanh 가 꼴찌(+2.7) — 다양성의 원천이 전처리라는 근거.
#   추론은 quantiles_/references_ 만 저장해 numpy 로 재현 (sklearn 버전 의존 제거,
#   sklearn 대비 오차 0.000e+00 검증됨).
USE_MLP2 = True
MLP2_W = 0.20

# 세 번째 MLP — 피처 70% 무작위 부분집합 (random subspace).
#   전처리가 아니라 '보는 피처'를 줄여 다양성을 만든다. MLP1 상관 0.898.
#   M1 0.20 + A 0.20 위에 E 0.15 를 얹으면 평균 662.1 -> 666.2, 양쪽 폴드 개선.
#   E 가중치 0.05~0.25 전 구간이 양쪽 개선인 고원(최고 0.15).
#   4번째(C 결측표시자)는 +0.7 뿐이라 복잡도 대비 무의미 -> 3멤버에서 끊는다.
USE_MLP3 = True
MLP3_W = 0.15
MLP3_FEAT_FRAC = 0.7
MLP3_FEAT_SEED = 0

# ---- 전방검증으로 확정한 상수 ----
PARAMS = dict(objective="binary", verbose=-1, num_threads=6, force_row_wise=True,
              bagging_freq=1, bagging_fraction=0.8, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=1000, feature_fraction=0.7, lambda_l2=10.0)
ROUNDS = 100
N_SEEDS = 5
LOGREG_C = 0.01
BLEND_W = 0.35      # 로지스틱 비중 (2폴드 스윕에서 0.2~0.4가 동등, 분포이동에 강한 쪽으로 살짝 가중)
DRIFT = -0.011      # 2024 -> 2025 리그 평균 변화 예상치 (R 5년 평균 -0.012, 백테스트 최적 -0.010)
# 모델이 1년 앞 데이터에서 스스로 내려가는 양. 구성이 바뀌면 반드시 다시 측정한다
# (Trackman 피처는 직전시즌 고정값이라 드리프트를 안 실어 나른다 -> 자동 하강량이 줄어듦).
#   기본 78피처   R: -0.0053, -0.0040 -> -0.0046 | F: -0.0183 (1회) -> 헤지 -0.011
#   +Trackman     R: -0.0035, -0.0047 -> -0.0041 | F: -0.0170 (1회) -> 헤지 -0.0105
#   2023~2024만    R: -0.0028, -0.0023 -> -0.0026 | F: -0.0113 (1회) -> 헤지 -0.0069
#   (학습 구간이 짧을수록 최근 레벨에 밀착해 자동 하강량이 줄어든다)
#   2024만         R: -0.0028, -0.0048 -> -0.0038 | F: -0.0135 (1회) -> 헤지 -0.0086
#   반감기1.0 가중  R: -0.0034, -0.0045 -> -0.0040 | F: -0.0141 (1회) -> 헤지 -0.0090
#   2023~2024 + prior + 볼카운트보정 (현재 구성)
#                  R: -0.0034, -0.0042 -> -0.0038 | F: -0.0114 (1회) -> 헤지 -0.0076
#   2023~2024 + MLP 블렌드 (현재 구성) R: -0.0023, -0.0031 -> -0.0027 | F: -0.0101 -> 헤지 -0.0064
if USE_MLP and TRAIN_FROM == 2023 and not USE_PRIOR:
    D_AUTO = {0: -0.0027, 1: -0.0064}
elif TRAIN_FROM == 2023 and USE_PRIOR:
    D_AUTO = {0: -0.0038, 1: -0.0076}
elif DECAY_HALFLIFE is not None:
    D_AUTO = {0: -0.0040, 1: -0.0090}
elif TRAIN_FROM >= 2024:
    D_AUTO = {0: -0.0038, 1: -0.0086}
elif TRAIN_FROM >= 2023:
    D_AUTO = {0: -0.0026, 1: -0.0069}
elif USE_TRACKMAN:
    D_AUTO = {0: -0.0041, 1: -0.0105}
else:
    D_AUTO = {0: -0.0046, 1: -0.011}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def solve_shift(z, target):
    """sigmoid(z + delta) 의 평균이 target 이 되는 delta."""
    lo, hi = -2.0, 2.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
    print("train:", raw.shape)

    if USE_TRACKMAN:
        # pitcher_id 키의 '해당 시즌 이전' Trackman 집계표를 만들어 model/ 에 동봉한다
        idmap = pd.read_parquet("idmap_pitcher_v2.parquet")
        sel = (idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
               .drop_duplicates("tm_pitcher_id", keep="first"))
        p2t = dict(zip(sel.pitcher_id, sel.tm_pitcher_id))
        pri = pd.read_parquet("tm_prior.parquet")
        t2p = {v: k for k, v in p2t.items()}
        pri["pitcher_id"] = pri["pitcher_trackman_id"].map(t2p)
        pri = pri.dropna(subset=["pitcher_id"]).drop(columns=["pitcher_trackman_id"])
        pri["pitcher_id"] = pri["pitcher_id"].astype(int)
        # 2025 행(= test)은 2019~2024 전체 누적을 쓴다
        pri.to_csv(f"{OUT}/tm_prior.csv", index=False)
        print(f"  Trackman prior {len(pri):,}행 저장 (투수 {pri.pitcher_id.nunique()}명)")
        raw = attach_trackman(raw, pri)
        print(f"  부착 커버리지 {raw['tm_prior_n'].notna().mean()*100:.1f}%")

    if USE_PRIOR:
        # 전체 라벨로 (시즌 < S) 집계 테이블 생성. 2025용 행만 model/ 에 동봉한다.
        seasons = sorted(raw["season"].unique().tolist() + [2025])
        tables, _ = build_prior(raw, seasons, decay=PRIOR_DECAY)
        for nm, tab in tables.items():
            t25 = tab[tab["season"] == 2025]
            t25.to_csv(f"{OUT}/prior_{nm}.csv", index=False)
            print(f"  prior[{nm}] 2025용 {len(t25):,}행 저장")
        raw = apply_prior(raw, tables)

    d = build(raw)
    d["season"] = raw["season"].values
    d[TARGET] = raw[TARGET].values
    del raw

    feats = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
    print(f"피처 {len(feats)}개")

    start = 2019 if DECAY_HALFLIFE is not None else TRAIN_FROM
    msk = ((d.season >= start) & ~((d.is_F == 1) & (d.season <= 2022))).values
    X, y = d.loc[msk, feats], d.loc[msk, TARGET].values
    if DECAY_HALFLIFE is None:
        w = None
        print(f"학습 행수 {msk.sum():,} / {len(d):,}  ({start}~2024, 2022 이전 F 제외, 균등가중)")
    else:
        w = (0.5 ** ((2024 - d.loc[msk, "season"].values) / DECAY_HALFLIFE)).astype(np.float64)
        print(f"학습 행수 {msk.sum():,} / {len(d):,}  ({start}~2024, 2022 이전 F 제외, "
              f"반감기 {DECAY_HALFLIFE}시즌 가중, 실효표본 {w.sum():,.0f})")
        for s_ in sorted(d.loc[msk, "season"].unique()):
            print(f"    {s_}: 가중 {0.5 ** ((2024 - s_) / DECAY_HALFLIFE):.4f}")

    # ---------- LGBM 5시드 ----------
    z_lgb_all = np.zeros(len(d))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        m = lgb.train(p, lgb.Dataset(X, y, weight=w), num_boost_round=ROUNDS)
        m.save_model(f"{OUT}/lgb_{s}.txt")
        z_lgb_all += m.predict(d[feats], raw_score=True)
        print(f"  LGBM seed {s} 완료")
    z_lgb_all /= N_SEEDS

    # ---------- L2 로지스틱 (스케일러/중앙값을 직접 저장해 sklearn 의존 제거) ----------
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs")
    lr.fit((A - mu) / sd, y, sample_weight=w)
    print("  로지스틱 완료")
    B = d[feats].fillna(med).to_numpy(np.float64)
    z_lr_all = ((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0]

    np.savez(f"{OUT}/linear.npz", coef=lr.coef_[0], intercept=lr.intercept_,
             mu=mu, sd=sd, med=med.to_numpy(np.float64))

    # ---------- MLP (가중치만 npz 로 저장, 추론은 numpy 순전파) ----------
    z_mlp_all = None
    if USE_MLP:
        An = (A - mu) / sd
        Ball = (B - mu) / sd
        acc, packs = [], []
        for s_ in range(MLP_SEEDS):
            mm = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, alpha=MLP_ALPHA,
                               batch_size=4096, learning_rate_init=1e-3, max_iter=150,
                               early_stopping=True, n_iter_no_change=10,
                               validation_fraction=0.1, random_state=s_)
            mm.fit(An, y)
            h = Ball
            for W_, b_ in zip(mm.coefs_[:-1], mm.intercepts_[:-1]):
                h = np.maximum(h @ W_ + b_, 0.0)
            acc.append((h @ mm.coefs_[-1] + mm.intercepts_[-1]).ravel())
            packs.append(mm)
            print(f"  MLP seed {s_} 완료 ({mm.n_iter_} epoch)")
        z_mlp_all = np.mean(acc, axis=0)
        arr = {}
        for i, mm in enumerate(packs):
            for j, (W_, b_) in enumerate(zip(mm.coefs_, mm.intercepts_)):
                arr[f"s{i}_W{j}"] = np.asarray(W_, np.float64)
                arr[f"s{i}_b{j}"] = np.asarray(b_, np.float64)
        arr["n_seeds"] = np.array([MLP_SEEDS])
        arr["n_layers"] = np.array([len(packs[0].coefs_)])
        np.savez(f"{OUT}/mlp.npz", **arr)

    # ---------- MLP2 (분위수변환 입력) ----------
    z_mlp2_all = None
    if USE_MLP2:
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                 random_state=0).fit(A)
        Aq, Bq = qt.transform(A), qt_transform(B, qt.quantiles_, qt.references_)
        acc2, packs2 = [], []
        for s_ in range(MLP_SEEDS):
            mm = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, alpha=MLP_ALPHA,
                               batch_size=4096, learning_rate_init=1e-3, max_iter=150,
                               early_stopping=True, n_iter_no_change=10,
                               validation_fraction=0.1, random_state=s_)
            mm.fit(Aq, y)
            h = Bq
            for W_, b_ in zip(mm.coefs_[:-1], mm.intercepts_[:-1]):
                h = np.maximum(h @ W_ + b_, 0.0)
            acc2.append((h @ mm.coefs_[-1] + mm.intercepts_[-1]).ravel())
            packs2.append(mm)
            print(f"  MLP2 seed {s_} 완료 ({mm.n_iter_} epoch)")
        z_mlp2_all = np.mean(acc2, axis=0)
        arr2 = {"quantiles": qt.quantiles_, "references": qt.references_,
                "n_seeds": np.array([MLP_SEEDS]),
                "n_layers": np.array([len(packs2[0].coefs_)])}
        for i, mm in enumerate(packs2):
            for j, (W_, b_) in enumerate(zip(mm.coefs_, mm.intercepts_)):
                arr2[f"s{i}_W{j}"] = np.asarray(W_, np.float64)
                arr2[f"s{i}_b{j}"] = np.asarray(b_, np.float64)
        np.savez(f"{OUT}/mlp2.npz", **arr2)

    # ---------- MLP3 (피처 70% 부분집합) ----------
    z_mlp3_all = None
    if USE_MLP3:
        rng_ = np.random.default_rng(MLP3_FEAT_SEED)
        sub_idx = np.sort(rng_.choice(len(feats), int(len(feats) * MLP3_FEAT_FRAC),
                                      replace=False))
        As, Bs = ((A - mu) / sd)[:, sub_idx], ((B - mu) / sd)[:, sub_idx]
        acc3, packs3 = [], []
        for s_ in range(MLP_SEEDS):
            mm = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, alpha=MLP_ALPHA,
                               batch_size=4096, learning_rate_init=1e-3, max_iter=150,
                               early_stopping=True, n_iter_no_change=10,
                               validation_fraction=0.1, random_state=s_)
            mm.fit(As, y)
            h = Bs
            for W_, b_ in zip(mm.coefs_[:-1], mm.intercepts_[:-1]):
                h = np.maximum(h @ W_ + b_, 0.0)
            acc3.append((h @ mm.coefs_[-1] + mm.intercepts_[-1]).ravel())
            packs3.append(mm)
            print(f"  MLP3 seed {s_} 완료 ({mm.n_iter_} epoch)")
        z_mlp3_all = np.mean(acc3, axis=0)
        arr3 = {"feat_idx": sub_idx.astype(np.int64),
                "n_seeds": np.array([MLP_SEEDS]),
                "n_layers": np.array([len(packs3[0].coefs_)])}
        for i, mm in enumerate(packs3):
            for j, (W_, b_) in enumerate(zip(mm.coefs_, mm.intercepts_)):
                arr3[f"s{i}_W{j}"] = np.asarray(W_, np.float64)
                arr3[f"s{i}_b{j}"] = np.asarray(b_, np.float64)
        np.savez(f"{OUT}/mlp3.npz", **arr3)

    # ---------- 블렌드 -> 볼카운트 보정 -> game_type 별 레벨 시프트 ----------
    z = (1 - BLEND_W) * z_lgb_all + BLEND_W * z_lr_all
    w2_ = MLP2_W if USE_MLP2 else 0.0
    w3_ = MLP3_W if USE_MLP3 else 0.0
    if USE_MLP:
        z = ((1 - MLP_W - w2_ - w3_) * z + MLP_W * z_mlp_all
             + (w2_ * z_mlp2_all if USE_MLP2 else 0.0)
             + (w3_ * z_mlp3_all if USE_MLP3 else 0.0))
        print(f"  MLP 블렌드 적용 (MLP1 {MLP_W} / MLP2 {w2_} / MLP3 {w3_}, "
              f"base {1-MLP_W-w2_-w3_:.2f})")
    if COUNT_CAL:
        cd = d["cnt_diff"].to_numpy()
        z = z + np.array([COUNT_CAL.get(str(int(v)), 0.0) for v in cd])
        print(f"  볼카운트(b−s) 보정 적용: {COUNT_CAL}")
    lvl24 = d[d.season == 2024].groupby("is_F")[TARGET].mean()
    print("\n2024 실제평균:", dict(lvl24.round(4)))

    shifts = {}
    for g, name in [(0, "R"), (1, "F")]:
        sel = ((d.season == 2024) & (d.is_F == g)).values
        target = float(lvl24[g]) + DRIFT - D_AUTO[g]
        delta = solve_shift(z[sel], target)
        shifts[name] = float(delta)
        print(f"  {name}: 2024 예측평균={sigmoid(z[sel]).mean():.4f} -> 목표 {target:.4f} "
              f"(2025 기대평균 {float(lvl24[g])+DRIFT:.4f})  delta={delta:+.4f}")

    meta = dict(features=feats, n_seeds=N_SEEDS, blend_w=BLEND_W, logreg_C=LOGREG_C,
                use_trackman=USE_TRACKMAN, train_from=TRAIN_FROM,
                decay_halflife=DECAY_HALFLIFE,
                use_mlp=USE_MLP, mlp_w=MLP_W,
                use_mlp2=USE_MLP2, mlp2_w=MLP2_W,
                use_mlp3=USE_MLP3, mlp3_w=MLP3_W,
                use_prior=USE_PRIOR, prior_decay=PRIOR_DECAY,
                prior_tables=(sorted(tables) if USE_PRIOR else []),
                count_cal=COUNT_CAL,
                shift=shifts, drift=DRIFT, d_auto={"R": D_AUTO[0], "F": D_AUTO[1]},
                level_2024={"R": float(lvl24[0]), "F": float(lvl24[1])},
                expected_2025={"R": float(lvl24[0]) + DRIFT, "F": float(lvl24[1]) + DRIFT},
                train_rows=int(msk.sum()))
    with open(f"{OUT}/meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    # 검증: 2024 행에 최종 변환을 적용했을 때 평균
    for g, name in [(0, "R"), (1, "F")]:
        sel = ((d.season == 2024) & (d.is_F == g)).values
        print(f"  [확인] {name} 최종 예측평균(2024행) = {sigmoid(z[sel] + shifts[name]).mean():.4f}")
    print("\n저장 완료:", OUT)


if __name__ == "__main__":
    main()
