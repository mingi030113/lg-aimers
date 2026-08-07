"""EDA 2: 피처별 단변량 신호 세기 (BSS 기여) + 상관/중복 구조."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 300)

t = pd.read_parquet("train.parquet")
y = t["control_success"].values.astype(np.float64)
r = y.mean()
BASE = r * (1 - r)


def bss_of_groups(key_series, min_n=200):
    """key로 그룹 평균 예측 시 얻는 BSS(×100000). in-sample 이므로 상한 추정치."""
    df = pd.DataFrame({"k": key_series.astype("object").fillna("__NA__"), "y": y})
    g = df.groupby("k")["y"].agg(["count", "mean"])
    g.loc[g["count"] < min_n, "mean"] = r
    p = df["k"].map(g["mean"]).values
    brier = np.mean((p - y) ** 2)
    return 100000 * (1 - brier / BASE), g.shape[0]


def bss_of_numeric(col, bins=20):
    s = t[col]
    q = pd.qcut(s, bins, duplicates="drop")
    q = q.cat.add_categories(["__NA__"]).fillna("__NA__")
    return bss_of_groups(q)


CAT = ["season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
       "balls_before", "strikes_before", "outs_before", "runner_on_1b", "runner_on_2b",
       "runner_on_3b", "num_runners_on", "base_state", "pitcher_hand", "batter_hand",
       "pitcher_team_id", "batter_team_id", "pitcher_id", "batter_id"]
NUM = ["run_top_before", "run_bot_before", "run_total_before", "score_diff_home",
       "score_diff_pitcher_team", "home_win_expectancy", "away_win_expectancy", "li",
       "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
       "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
       "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
       "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
       "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
       "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
       "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]

rows = []
for c in CAT:
    b, k = bss_of_groups(t[c])
    rows.append((c, "cat", k, b))
for c in NUM:
    b, k = bss_of_numeric(c)
    rows.append((c, "num20", k, b))

res = pd.DataFrame(rows, columns=["feature", "kind", "n_levels", "bss_insample"])
print("=== 단변량 BSS (in-sample, 최소 200샘플 그룹만 평균 사용) ===")
print(res.sort_values("bss_insample", ascending=False).to_string(index=False))

print("\n=== 볼카운트 (balls x strikes) 성공률 ===")
print(t.pivot_table(index="balls_before", columns="strikes_before",
                    values="control_success", aggfunc=["mean", "size"]))

print("\n=== count x season ===")
t["cnt"] = t["balls_before"].astype(str) + "-" + t["strikes_before"].astype(str)
print(t.pivot_table(index="cnt", columns="season", values="control_success", aggfunc="mean"))

print("\n=== 상관행렬 (수치형, |r|>0.5 쌍만) ===")
num_all = t[NUM + ["control_success"]].corr()
cm = num_all.where(~np.eye(len(num_all), dtype=bool))
pairs = cm.stack().reset_index()
pairs.columns = ["a", "b", "r"]
pairs = pairs[pairs["a"] < pairs["b"]]
print(pairs[pairs["r"].abs() > 0.5].sort_values("r", key=abs, ascending=False).to_string(index=False))

print("\n=== target과의 pearson 상관 ===")
print(num_all["control_success"].drop("control_success").sort_values(key=abs, ascending=False).to_string())

print("\n=== win_expectancy 합 검증 ===")
print((t["home_win_expectancy"] + t["away_win_expectancy"]).describe())
