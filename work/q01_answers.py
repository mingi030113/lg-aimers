"""질문 답변용 집계: 결측 실태 / 상황변수별 효과 / prev1·3·5 중복 검증."""
import numpy as np
import pandas as pd

from features import build, feature_list
from common import score_bss

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

raw = pd.read_parquet("train.parquet")
TARGET = "control_success"
y = raw[TARGET].values
r = y.mean()
BASE = r * (1 - r)

print("=" * 100)
print("### 1) 원본 49개 컬럼의 결측률 (%)")
print("=" * 100)
mr = (raw.isna().mean() * 100)
nz = mr[mr > 0].sort_values(ascending=False)
print(f"  결측이 있는 컬럼: {len(nz)}개 / 전체 {raw.shape[1]}개")
print(nz.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\n  결측이 0인 컬럼 {(mr == 0).sum()}개 (row_id, season, 볼카운트, 주자, 점수, 선수/팀 ID 등)")

print("\n  --- 결측의 정체: 콜드스타트 ---")
m = raw["asof_pitcher_success_rate"].isna()
print(f"  asof_pitcher_success_rate 결측 {m.sum():,}행 중 asof_pitcher_n==0 인 비율: "
      f"{(raw.loc[m, 'asof_pitcher_n'] == 0).mean()*100:.1f}%")
m2 = raw["asof_pitcher_prev5_game_success_rate"].isna()
print(f"  prev5 결측 {m2.sum():,}행의 asof_pitcher_n 중앙값: {raw.loc[m2,'asof_pitcher_n'].median():.0f}")
print(f"  -> 결측 = '아직 이력이 없다'는 정보 자체. 0으로 채우면 '성공률 0%'로 오해된다.")
print(f"\n  결측 행의 실제 성공률 vs 전체:")
print(f"    투수이력 결측행 {m.sum():>6,}행 -> {y[m.values].mean():.4f}   (전체 {r:.4f})")
print(f"    prev5 결측행   {m2.sum():>6,}행 -> {y[m2.values].mean():.4f}")

d = build(raw)
d[TARGET] = y
FE = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
dm = (d[FE].isna().mean() * 100)
print(f"\n  파생 후 {len(FE)}개 피처 중 결측 있는 것: {(dm > 0).sum()}개 "
      f"(최대 {dm.max():.2f}%)")


def eff(name, keys):
    """그룹 평균 예측 시 얻는 in-sample BSS = 그 변수 단독 설명력의 상한."""
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["count", "mean"])
    g.loc[g["count"] < 200, "mean"] = r
    p = pd.Series(keys).map(g["mean"]).values
    return 1e5 * (1 - np.mean((p - y) ** 2) / BASE), g


print("\n" + "=" * 100)
print("### 2) 초/말 (top_bottom)")
print("=" * 100)
s, g = eff("tb", raw["top_bottom"].values)
print(g.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"  단독 설명력 = {s:.2f}점   (차이 {abs(g['mean'].diff().iloc[-1])*1000:.1f}‰)")

print("\n  --- 이닝별 (초/말 분리) ---")
piv = raw.pivot_table(index="inning", columns="top_bottom", values=TARGET, aggfunc=["mean", "size"])
print(piv.head(10).to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 100)
print("### 3) 볼카운트")
print("=" * 100)
print(raw.pivot_table(index="balls_before", columns="strikes_before",
                      values=TARGET, aggfunc="mean").to_string(float_format=lambda x: f"{x:.4f}"))
for c in ["balls_before", "strikes_before"]:
    s, _ = eff(c, raw[c].values)
    print(f"  {c:16s} 단독 설명력 = {s:6.2f}점")
s, g = eff("b-s", (raw["balls_before"] - raw["strikes_before"]).values)
print(f"\n  b−s (볼−스트라이크) 단독 설명력 = {s:.2f}점")
print(g.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 100)
print("### 4) 상황 변수 총정리 — 단독 설명력 (in-sample 상한)")
print("=" * 100)
rows = []
cands = {
    "top_bottom (초/말)": raw["top_bottom"].values,
    "inning (이닝)": raw["inning"].values,
    "balls_before": raw["balls_before"].values,
    "strikes_before": raw["strikes_before"].values,
    "볼카운트 12칸": (raw.balls_before.astype(str) + "-" + raw.strikes_before.astype(str)).values,
    "outs_before": raw["outs_before"].values,
    "base_state (주자)": raw["base_state"].values,
    "num_runners_on": raw["num_runners_on"].values,
    "득점권 여부": ((raw.runner_on_2b == 1) | (raw.runner_on_3b == 1)).values,
    "li (중요도, 20분위)": pd.qcut(raw["li"], 20, duplicates="drop").astype(str).values,
    "score_diff_pitcher (10분위)": pd.qcut(raw["score_diff_pitcher_team"], 10, duplicates="drop").astype(str).values,
    "home_win_expectancy (20분위)": pd.qcut(raw["home_win_expectancy"], 20, duplicates="drop").astype(str).values,
    "pitcher_hand": raw["pitcher_hand"].values,
    "batter_hand": raw["batter_hand"].values,
    "좌우 조합 (4칸)": (raw.pitcher_hand.astype(str) + "x" + raw.batter_hand.astype(str)).values,
    "pitcher_team_id": raw["pitcher_team_id"].values,
    "batter_team_id": raw["batter_team_id"].values,
    "pitcher_id": raw["pitcher_id"].values,
    "batter_id": raw["batter_id"].values,
    "asof_pitcher_success_rate (20분위)": pd.qcut(raw["asof_pitcher_success_rate"], 20, duplicates="drop").astype(str).fillna("NA").values,
    "asof_pitcher_reverse_rate (20분위)": pd.qcut(raw["asof_pitcher_reverse_rate"], 20, duplicates="drop").astype(str).fillna("NA").values,
    "asof_batter_success_rate (20분위)": pd.qcut(raw["asof_batter_success_rate"], 20, duplicates="drop").astype(str).fillna("NA").values,
}
for k, v in cands.items():
    s, _ = eff(k, v)
    rows.append((k, s))
res = pd.DataFrame(rows, columns=["변수", "단독설명력(점)"]).sort_values("단독설명력(점)", ascending=False)
print(res.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

print("\n" + "=" * 100)
print("### 5) 좌우 조합 상세 (1=?, 2=? 는 익명 코드)")
print("=" * 100)
combo = raw.groupby(["pitcher_hand", "batter_hand"])[TARGET].agg(["size", "mean"])
combo["대비전체"] = combo["mean"] - r
print(combo.to_string(float_format=lambda x: f"{x:.4f}"))
print("\n  같은손 vs 다른손:")
same = (raw.pitcher_hand == raw.batter_hand)
print(f"    같은손 {same.sum():>7,}행 -> {y[same.values].mean():.4f}")
print(f"    다른손 {(~same).sum():>7,}행 -> {y[~same.values].mean():.4f}")
