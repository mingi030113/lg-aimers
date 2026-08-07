"""피처 빌더. train/test 동일 함수를 쓴다 (per-row 변환만; 평가데이터 집계 금지 규정 준수)."""
import numpy as np
import pandas as pd

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
