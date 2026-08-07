"""홈/원정 및 구장(ballpark) 정보가 복원 가능한지 + 제구에 영향이 있는지.

야구 규칙: 홈팀이 말(B)에 공격한다.
  top_bottom == 'T' (초) -> 원정팀 공격 -> **투수가 홈**
  top_bottom == 'B' (말) -> 홈팀 공격   -> **투수가 원정**

독립 검증: score_diff_home 과 score_diff_pitcher_team 의 부호 관계로 확인 가능하다
  (투수가 홈이면 두 값이 같고, 원정이면 부호가 반대다)

여기서 파생되는 새 정보: **구장** = 홈팀의 홈구장
  home_team_id = 투수홈이면 pitcher_team_id, 아니면 batter_team_id
  현재 모델에는 pitcher_team_id / batter_team_id 가 따로 들어갈 뿐 '구장'은 없다.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
TARGET = "control_success"

raw = pd.read_parquet("train.parquet")
y = raw[TARGET].values
r = y.mean()
BASE = r * (1 - r)

print("=" * 96)
print("### 1) 투수 홈/원정 복원의 독립 검증")
print("=" * 96)
p_home = (raw["top_bottom"] == "T").values          # 가설
same = (raw["score_diff_home"] == raw["score_diff_pitcher_team"]).values
nz = (raw["score_diff_home"] != 0).values           # 동점이면 0==0 이라 판별 불가

print(f"  동점이 아닌 행 {nz.sum():,} 개로 검증")
print(f"    (top_bottom=='T') 와 (score_diff_home == score_diff_pitcher_team) 일치율: "
      f"{(p_home[nz] == same[nz]).mean()*100:.4f}%")
ct = pd.crosstab(pd.Series(p_home[nz], name="가설: 투수가 홈"),
                 pd.Series(same[nz], name="점수차 부호 일치"))
print(ct.to_string())
print("  -> 완전 일치. top_bottom 하나로 투수 홈/원정이 100% 복원된다.")

print("\n" + "=" * 96)
print("### 2) 투수 홈/원정이 제구 성공률에 영향을 주는가")
print("=" * 96)
g = pd.DataFrame({"홈투수": p_home, "y": y}).groupby("홈투수")["y"].agg(["count", "mean"])
print(g.to_string(float_format=lambda x: f"{x:.4f}"))
d = g.loc[True, "mean"] - g.loc[False, "mean"]
print(f"  차이 = {d*1000:+.2f}‰   ->  단독 설명력 "
      f"{1e5*(g['count']*(g['mean']-r)**2).sum()/len(y)/BASE:.2f}점")
print("  (참고: 이 정보는 이미 is_bottom 피처로 모델에 들어가 있다. 부호만 반대)")

print("\n  --- 이닝별로 나눠보면 (후반부 홈 어드밴티지?) ---")
tmp = pd.DataFrame({"홈투수": p_home, "이닝": np.clip(raw["inning"], 1, 10), "y": y})
print(tmp.pivot_table(index="이닝", columns="홈투수", values="y", aggfunc="mean")
      .to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 96)
print("### 3) 구장(ballpark) — 새 정보")
print("=" * 96)
home_team = np.where(p_home, raw["pitcher_team_id"].values, raw["batter_team_id"].values)
raw["park"] = home_team
print(f"  복원된 구장 수: {len(np.unique(home_team))}개")


def solo(keys, name):
    t = pd.DataFrame({"k": keys, "y": y})
    gg = t.groupby("k")["y"].agg(["count", "mean"])
    gg.loc[gg["count"] < 200, "mean"] = r
    p = t["k"].map(gg["mean"]).values
    s = 1e5 * (1 - np.mean((p - y) ** 2) / BASE)
    print(f"  {name:34s} 단독 설명력 = {s:7.2f}점  (그룹 {len(gg)}개)")
    return s, gg


solo(home_team, "park (구장)")
solo(raw["pitcher_team_id"].values, "pitcher_team_id (기존)")
solo(raw["batter_team_id"].values, "batter_team_id (기존)")
solo([f"{a}_{b}" for a, b in zip(home_team, p_home)], "park × 투수홈여부")
solo([f"{a}_{b}" for a, b in zip(raw["pitcher_team_id"].values, p_home)],
     "pitcher_team × 투수홈여부")

print("\n  --- 구장별 성공률 (표본 큰 순) ---")
gg = pd.DataFrame({"park": home_team, "y": y}).groupby("park")["y"].agg(["count", "mean"])
gg["대비전체"] = gg["mean"] - r
print(gg.sort_values("count", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))

print("\n  --- 같은 투수팀이 홈/원정일 때 차이 (구장 효과 vs 팀 효과 분리) ---")
t = pd.DataFrame({"team": raw["pitcher_team_id"].values, "홈": p_home, "y": y})
piv = t.pivot_table(index="team", columns="홈", values="y", aggfunc=["mean", "size"])
piv.columns = ["원정_평균", "홈_평균", "원정_n", "홈_n"]
piv["홈-원정"] = piv["홈_평균"] - piv["원정_평균"]
print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\n  홈-원정 차이의 평균 {piv['홈-원정'].mean()*1000:+.2f}‰, "
      f"표준편차 {piv['홈-원정'].std()*1000:.2f}‰")
print(f"  이항잡음만으로 기대되는 표준편차 ≈ "
      f"{np.sqrt(0.25/piv['홈_n'].mean() + 0.25/piv['원정_n'].mean())*1000:.2f}‰")

print("\n" + "=" * 96)
print("### 4) 시즌별로 구장 효과가 안정적인가 (전이 가능성)")
print("=" * 96)
t = pd.DataFrame({"park": home_team, "season": raw["season"].values, "y": y})
lg = t.groupby("season")["y"].transform("mean")
t["dev"] = t["y"] - lg
piv = t.pivot_table(index="park", columns="season", values="dev", aggfunc="mean")
print((piv * 1000).round(1).to_string())
c = piv[[2023]].join(piv[[2024]]).corr().iloc[0, 1]
print(f"\n  2023 vs 2024 구장 편차 상관 = {c:.3f}")
print(f"  2022 vs 2023 = {piv[[2022]].join(piv[[2023]]).corr().iloc[0,1]:.3f}")
