# -*- coding: utf-8 -*-
"""제출용 추론 스크립트 (bundle.py 가 features.py 를 인라인해 생성).

흐름
  data/test.csv, data/sample_submission.csv 읽기
  -> build()  (학습과 완전히 동일한 함수)
  -> LGBM 5개 raw_score 평균 + L2 로지스틱 decision_function 을 블렌드
  -> game_type 별로 미리 계산된 로짓 시프트 적용 (2025 리그 레벨 통제)
  -> sigmoid -> output/submission.csv

평가데이터 내부 집계·행순서 기반 피처를 일절 쓰지 않는다 (완전한 행 단위 독립 추론).
"""
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

ID_COL = "row_id"
TARGET_COL = "control_success"

TEST_DIR = "./data"
MODEL_DIR = "./model"
OUT_DIR = "./output"

# ===== features.py 인라인 시작 =====
BASE_MAP = {"___": 0, "1__": 1, "_2_": 2, "12_": 3, "__3": 4, "1_3": 5, "_23": 6, "123": 7}

# 리그 사전값 (train에서 계산한 상수. 평가데이터를 보지 않음)
LEAGUE_P = 0.52
DROP = ["asof_pitcher_pitchmix_n",   # asof_pitcher_n 과 완전 동일
        "away_win_expectancy"]        # 100 - home_win_expectancy


def attach_trackman(df, prior):
    """Trackman 구위 피처 부착. prior 는 (pitcher_id, season) 키의 '해당 시즌 이전' 집계.

    build() 이전에 호출한다. 학습·추론에서 동일한 코드를 쓰기 위해 여기에 둔다.
    """
    d = df.merge(prior, on=["pitcher_id", "season"], how="left")
    d["tm_log_n"] = np.log1p(d["tm_prior_n"].fillna(0))
    d["tm_has"] = d["tm_prior_n"].notna().astype(np.int8)
    return d


def build(df, league_p=LEAGUE_P):
    d = df.copy()
    d["is_F"] = (d["game_type"] == "F").astype(np.int8)
    d["is_bottom"] = (d["top_bottom"] == "B").astype(np.int8)
    d["base_code"] = d["base_state"].map(BASE_MAP).astype(np.int8)
    d = d.drop(columns=[c for c in ["game_type", "top_bottom", "base_state"] + DROP if c in d.columns])

    # ---- 볼카운트 파생 ----
    b, s = d["balls_before"], d["strikes_before"]
    d["cnt_code"] = (b * 3 + s).astype(np.int8)
    d["cnt_diff"] = (b - s).astype(np.int8)
    d["is_3ball"] = (b == 3).astype(np.int8)
    d["is_2strike"] = (s == 2).astype(np.int8)
    d["pitcher_ahead"] = (s > b).astype(np.int8)
    d["is_full"] = ((b == 3) & (s == 2)).astype(np.int8)
    d["pitch_of_pa_est"] = (b + s).astype(np.int8)

    # ---- 상황 ----
    d["platoon"] = (d["pitcher_hand"] == d["batter_hand"]).astype(np.int8)
    d["hand_combo"] = (d["pitcher_hand"] * 2 + d["batter_hand"]).astype(np.int8)
    d["scoring_pos"] = ((d["runner_on_2b"] == 1) | (d["runner_on_3b"] == 1)).astype(np.int8)
    d["abs_score_diff"] = d["score_diff_pitcher_team"].abs()
    d["close_game"] = (d["abs_score_diff"] <= 1).astype(np.int8)
    d["blowout"] = (d["abs_score_diff"] >= 5).astype(np.int8)
    d["late_close"] = ((d["inning"] >= 7) & (d["close_game"] == 1)).astype(np.int8)
    d["log_li"] = np.log1p(d["li"])
    d["we_pitcher"] = np.where(d["is_bottom"] == 1,   # 말 공격 = 홈 공격 -> 투수는 원정
                               100 - d["home_win_expectancy"], d["home_win_expectancy"])

    # ---- 투수 경험/피로 ----
    n = d["asof_pitcher_n"]
    d["log_pitcher_n"] = np.log1p(n)
    d["log_batter_n"] = np.log1p(d["asof_batter_n"])
    d["is_rookie_p"] = (n < 200).astype(np.int8)

    # ---- 경험적 베이즈 수축 (표본 적은 투수 안정화) ----
    for k in (200, 1000):
        d[f"p_succ_eb{k}"] = ((d["asof_pitcher_success_rate"].fillna(league_p) * n + league_p * k)
                              / (n + k))
    bn = d["asof_batter_n"]
    d["b_succ_eb500"] = ((d["asof_batter_success_rate"].fillna(league_p) * bn + league_p * 500)
                         / (bn + 500))

    # ---- 최근 폼: 커리어 대비 편차 (리그 드리프트 + 컨디션을 함께 담음) ----
    car = d["asof_pitcher_success_rate"]
    for k in (1, 3, 5):
        c = f"asof_pitcher_prev{k}_game_success_rate"
        d[f"form_dev{k}"] = d[c] - car
    d["form_blend"] = (0.2 * d["asof_pitcher_prev1_game_success_rate"]
                       + 0.3 * d["asof_pitcher_prev3_game_success_rate"]
                       + 0.5 * d["asof_pitcher_prev5_game_success_rate"])
    d["form_blend_dev"] = d["form_blend"] - car
    mcar = d["asof_pitcher_middle_rate"]
    for k in (1, 3, 5):
        d[f"mid_dev{k}"] = d[f"asof_pitcher_prev{k}_game_middle_rate"] - mcar

    # ---- 투수 성향 조합 ----
    d["rev_plus_mid"] = d["asof_pitcher_reverse_rate"] + d["asof_pitcher_middle_rate"]
    d["strike_minus_ball"] = d["asof_pitcher_strike_rate"] - d["asof_pitcher_ball_rate"]
    d["breaking_plus_off"] = d["asof_pitcher_breaking_rate"] + d["asof_pitcher_offspeed_rate"]
    mix = d[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
             "asof_pitcher_offspeed_rate"]].to_numpy(dtype=np.float64)
    mix = np.clip(mix, 1e-6, 1.0)
    d["mix_entropy"] = -np.nansum(mix * np.log(mix), axis=1)   # pandas .sum() 과 동일하게 NaN skip

    # ---- 투수 vs 타자 매치업 ----
    d["pb_succ_gap"] = d["p_succ_eb1000"] - d["b_succ_eb500"]
    d["pb_mid_gap"] = d["asof_pitcher_middle_rate"] - d["asof_batter_middle_rate"]
    return d


def feature_list(d, use_season=False, use_ids=False):
    drop = {"row_id", "control_success"}
    if not use_season:
        drop.add("season")
    if not use_ids:
        drop |= {"pitcher_id", "batter_id"}
    return [c for c in d.columns if c not in drop]


TARGET = "control_success"


def _cnt_group(b, s):
    """볼카운트 구간: 0=초구, 1=투수유리, 2=중립, 3=타자유리, 4=3볼"""
    g = np.full(len(b), 2, dtype=np.int8)
    g[(b == 0) & (s == 0)] = 0
    g[(s > b) & ~((b == 0) & (s == 0))] = 1
    g[(b > s) & (b < 3)] = 3
    g[b == 3] = 4
    return g


def _agg(df, keys, name, league):
    """(keys) 별로 시즌별 (n, 리그대비 성공 초과분 합) 집계.

    리그평균은 (시즌, game_type) 기준이라 시즌 드리프트와 F/R 레벨차가 모두 제거된다.
    """
    x = df[keys + ["season"]].copy()
    lg = league.reindex(pd.MultiIndex.from_arrays([df["season"].values, df["_gt"].values])).values
    x["dev"] = df[TARGET].values - lg
    g = x.groupby(keys + ["season"], observed=True).agg(n=("dev", "size"), sdev=("dev", "sum"))
    g.name = name
    return g


def _expand_to_prior(g, keys, seasons, decay=0.75):
    """시즌별 집계를 '직전 시즌까지 누적'(지수감쇠 가중)으로 변환."""
    out = []
    for S in seasons:
        sub = g[g.index.get_level_values("season") < S]
        if len(sub) == 0:
            continue
        w = decay ** (S - 1 - sub.index.get_level_values("season").values)
        tmp = pd.DataFrame({"n": sub["n"].values * w, "sdev": sub["sdev"].values * w},
                           index=sub.index.droplevel("season"))
        tmp = tmp.groupby(level=list(range(len(keys)))).sum()
        tmp["season"] = S
        out.append(tmp.reset_index())
    if not out:
        return pd.DataFrame(columns=keys + ["n", "sdev", "season"])
    return pd.concat(out, ignore_index=True)


def build_prior(train_df, target_seasons, decay=0.75, k_shrink=(400, 400, 800, 300, 200)):
    """train_df(라벨 보유)로부터 target_seasons 각 시즌용 prior 테이블 생성.

    반환: dict of DataFrame, 각각 key + season 으로 merge 가능.
    값은 모두 '리그평균 대비 편차(수축 적용)' 이므로 시즌 드리프트에 중립적이다.
    """
    # 2022년 이전 F(퓨처스)는 라벨 레짐이 달라 집계에서 제외한다.
    isF = (train_df["game_type"] == "F")
    df = train_df[~(isF & (train_df["season"] <= 2022))].copy()
    # 리그 평균은 (시즌 x game_type) 별로. F/R 레벨이 크게 다르다.
    df["_gt"] = isF.reindex(df.index).astype(np.int8)
    league = df.groupby(["season", "_gt"])[TARGET].mean()

    df["cnt_g"] = _cnt_group(df["balls_before"].values, df["strikes_before"].values)
    df["runner_any"] = (df["num_runners_on"] > 0).astype(np.int8)
    df["inn_g"] = np.clip(df["inning"].values, 1, 9).astype(np.int8)

    specs = [
        (["pitcher_id"], "p", k_shrink[0]),
        (["pitcher_id", "batter_hand"], "p_bh", k_shrink[1]),
        (["pitcher_id", "cnt_g"], "p_cnt", k_shrink[2]),
        (["batter_id"], "b", k_shrink[3]),
        (["batter_id", "pitcher_hand"], "b_ph", k_shrink[4]),
        (["pitcher_team_id"], "pteam", 2000),
        (["pitcher_id", "runner_any"], "p_run", 800),
    ]
    tables = {}
    for keys, name, k in specs:
        g = _agg(df, keys, name, league)
        pri = _expand_to_prior(g, keys, target_seasons, decay=decay)
        pri[f"{name}_n"] = pri["n"]
        pri[f"{name}_dev"] = pri["sdev"] / (pri["n"] + k)   # 경험적 베이즈 수축
        tables[name] = pri[keys + ["season", f"{name}_n", f"{name}_dev"]]
    return tables, league


def apply_prior(df, tables):
    d = df.copy()
    d["cnt_g"] = _cnt_group(d["balls_before"].values, d["strikes_before"].values)
    d["runner_any"] = (d["num_runners_on"] > 0).astype(np.int8)
    for name, tab in tables.items():
        keys = [c for c in tab.columns if not c.startswith(name + "_") and c != "season"]
        d = d.merge(tab, on=keys + ["season"], how="left")
    for name in tables:
        d[f"{name}_n"] = d[f"{name}_n"].fillna(0.0)
        d[f"{name}_dev"] = d[f"{name}_dev"].fillna(0.0)
        d[f"{name}_logn"] = np.log1p(d[f"{name}_n"])
    # 분할 - 전체 (순수 스플릿 효과)
    d["p_bh_split"] = d["p_bh_dev"] - d["p_dev"]
    d["p_cnt_split"] = d["p_cnt_dev"] - d["p_dev"]
    d["p_run_split"] = d["p_run_dev"] - d["p_dev"]
    d["b_ph_split"] = d["b_ph_dev"] - d["b_dev"]
    d = d.drop(columns=["cnt_g", "runner_any"])
    return d
# ===== features.py 인라인 끝 =====


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    with open(os.path.join(MODEL_DIR, "meta.json"), "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    feats = meta["features"]
    w = meta["blend_w"]
    shift = meta["shift"]
    print(f"meta: {len(feats)} feats, {meta['n_seeds']} boosters, blend_w={w}, "
          f"shift R={shift['R']:+.4f} F={shift['F']:+.4f}, "
          f"expected 2025 R={meta['expected_2025']['R']:.4f} F={meta['expected_2025']['F']:.4f}")

    test = pd.read_csv(os.path.join(TEST_DIR, "test.csv"), encoding="utf-8-sig")
    sub = pd.read_csv(os.path.join(TEST_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    print(f"test={len(test)}  sample_submission={len(sub)}")

    tm_path = os.path.join(MODEL_DIR, "tm_prior.csv")
    if meta.get("use_trackman") and os.path.exists(tm_path):
        prior = pd.read_csv(tm_path)
        test = attach_trackman(test, prior)
        print(f"trackman prior 부착: 커버리지 {test['tm_prior_n'].notna().mean()*100:.1f}%")

    if meta.get("use_prior"):
        tables = {nm: pd.read_csv(os.path.join(MODEL_DIR, f"prior_{nm}.csv"))
                  for nm in meta["prior_tables"]}
        test = apply_prior(test, tables)
        print(f"prior 분할통계 부착: {len(tables)}개 테이블, "
              f"투수 매칭률 {(test['p_n'] > 0).mean()*100:.1f}%")

    d = build(test)
    X = d.reindex(columns=feats)
    print(f"feature matrix {X.shape}")

    z_lgb = np.zeros(len(X), dtype=np.float64)
    for s in range(meta["n_seeds"]):
        b = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lgb_{s}.txt"))
        z_lgb += b.predict(X, raw_score=True)
    z_lgb /= meta["n_seeds"]

    lin = np.load(os.path.join(MODEL_DIR, "linear.npz"))
    A = X.to_numpy(np.float64)
    nan = np.isnan(A)
    if nan.any():
        A[nan] = np.take(lin["med"], np.where(nan)[1])
    z_lin = ((A - lin["mu"]) / lin["sd"]) @ lin["coef"] + lin["intercept"][0]

    z = (1 - w) * z_lgb + w * z_lin

    if meta.get("use_mlp"):
        # 학습 때 저장한 가중치로 numpy 순전파 (sklearn 버전 의존성 제거)
        mp = np.load(os.path.join(MODEL_DIR, "mlp.npz"))
        ns, nl = int(mp["n_seeds"][0]), int(mp["n_layers"][0])
        Bn = (A - lin["mu"]) / lin["sd"]
        acc = np.zeros(len(Bn), dtype=np.float64)
        for i in range(ns):
            h = Bn
            for j in range(nl - 1):
                h = np.maximum(h @ mp[f"s{i}_W{j}"] + mp[f"s{i}_b{j}"], 0.0)   # relu
            acc += (h @ mp[f"s{i}_W{nl-1}"] + mp[f"s{i}_b{nl-1}"]).ravel()
        z_mlp = acc / ns
        mw = meta["mlp_w"]
        z = (1 - mw) * z + mw * z_mlp
        print(f"MLP 블렌드 적용: {ns}시드 {nl}층, 비중 {mw}")
    cc = meta.get("count_cal") or {}
    if cc:
        z = z + np.array([cc.get(str(int(v)), 0.0) for v in d["cnt_diff"].to_numpy()])
        print(f"볼카운트(b−s) 보정 적용: {cc}")
    z = z + np.where(d["is_F"].to_numpy() == 1, shift["F"], shift["R"])
    p = np.clip(_sigmoid(z), 1e-6, 1 - 1e-6)
    print(f"pred mean={p.mean():.5f}  sd={p.std():.5f}  min={p.min():.4f}  max={p.max():.4f}")

    pred = dict(zip(test[ID_COL].to_numpy(), p))
    fallback = 0.5 * (meta["expected_2025"]["R"] + meta["expected_2025"]["F"])
    out, miss = [], 0
    for rid in sub[ID_COL].to_numpy():
        v = pred.get(rid)
        if v is None:
            miss += 1
            out.append(fallback)
        else:
            out.append(v)
    if miss:
        print(f"warning: {miss} row_id missing from test.csv")
    sub[TARGET_COL] = out

    os.makedirs(OUT_DIR, exist_ok=True)
    sub.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False, encoding="utf-8")
    print(f"saved {OUT_DIR}/submission.csv rows={len(sub)}")


if __name__ == "__main__":
    main()
