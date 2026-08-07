"""홈/원정 유의성 + 구장 효과를 '실제 쓰는 레짐'(R게임, 2023~2024)으로 재분석.

앞선 q03 은 전체 데이터라 2022 이전 F(구 레짐, 성공률 .70대)가 섞여 결과를 왜곡했다.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
TARGET = "control_success"

raw = pd.read_parquet("train.parquet")
raw["p_home"] = (raw["top_bottom"] == "T")
raw["park"] = np.where(raw["p_home"], raw["pitcher_team_id"], raw["batter_team_id"])

print("=" * 92)
print("### 1) 홈/원정 차이가 통계적으로 유의한가")
print("=" * 92)
for lab, sub in [("전체", raw), ("R게임만", raw[raw.game_type == "R"]),
                 ("R게임 2023~2024", raw[(raw.game_type == "R") & (raw.season >= 2023)])]:
    a = sub.loc[sub.p_home, TARGET]
    b = sub.loc[~sub.p_home, TARGET]
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    print(f"  [{lab:16s}] 홈투수 {a.mean():.4f} (n={len(a):,})  원정투수 {b.mean():.4f} (n={len(b):,})")
    print(f"  {'':18s} 차이 {diff*1000:+6.2f}‰   표준오차 {se*1000:.2f}‰   z={diff/se:+.2f}"
          f"   -> {'유의하지 않음' if abs(diff/se) < 1.96 else '유의'}")

print("\n" + "=" * 92)
print("### 2) 구장 정체 확인 — 표본이 이상한 구장들")
print("=" * 92)
ct = pd.crosstab(raw["park"], raw["game_type"])
ct["합계"] = ct.sum(axis=1)
ct["F비율%"] = (ct["F"] / ct["합계"] * 100).round(1)
print(ct.sort_values("합계", ascending=False).to_string())
print("\n  -> 22, 23 번은 F(퓨처스) 전용 구장. 2022 이전 F 는 라벨 레짐이 달라(성공률 .70대)")
print("     앞선 분석의 0.666 / 0.599 는 구장 효과가 아니라 그 레짐 오염이었다.")
print("     실제 학습에서는 2022 이전 F 를 이미 제외하고 있다.")

print("\n" + "=" * 92)
print("### 3) 구장 효과 — R게임, 2023~2024 (실제 학습 구간)")
print("=" * 92)
sub = raw[(raw.game_type == "R") & (raw.season >= 2023)]
r = sub[TARGET].mean()
BASE = r * (1 - r)
g = sub.groupby("park")[TARGET].agg(["count", "mean"])
g["대비평균‰"] = (g["mean"] - r) * 1000
g["표준오차‰"] = np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"]) * 1000
g["z"] = g["대비평균‰"] / g["표준오차‰"]
print(g.sort_values("대비평균‰", ascending=False).to_string(float_format=lambda x: f"{x:8.3f}"))
solo = 1e5 * (g["count"] * (g["mean"] - r) ** 2).sum() / len(sub) / BASE
print(f"\n  구장 단독 설명력 = {solo:.2f}점   (그룹 {len(g)}개, 순수잡음 바닥 ≈ "
      f"{1e5*len(g)/len(sub):.2f}점)")
for c, nm in [("pitcher_team_id", "투수팀"), ("batter_team_id", "타자팀")]:
    gg = sub.groupby(c)[TARGET].agg(["count", "mean"])
    s = 1e5 * (gg["count"] * (gg["mean"] - r) ** 2).sum() / len(sub) / BASE
    print(f"  {nm} 단독 설명력 = {s:.2f}점")

print("\n" + "=" * 92)
print("### 4) 같은 팀의 홈 vs 원정 (구장 효과와 팀 효과 분리, R게임 2023~2024)")
print("=" * 92)
t = sub.pivot_table(index="pitcher_team_id", columns="p_home", values=TARGET,
                    aggfunc=["mean", "size"])
t.columns = ["원정", "홈", "n원정", "n홈"]
t["홈−원정‰"] = (t["홈"] - t["원정"]) * 1000
t["표준오차‰"] = np.sqrt(0.25 / t["n홈"] + 0.25 / t["n원정"]) * 1000
t["z"] = t["홈−원정‰"] / t["표준오차‰"]
t = t[t["n홈"] > 3000]
print(t.to_string(float_format=lambda x: f"{x:9.3f}"))
print(f"\n  홈−원정 차이: 평균 {t['홈−원정‰'].mean():+.2f}‰, 실제 표준편차 {t['홈−원정‰'].std():.2f}‰")
print(f"  이항잡음만으로 기대되는 표준편차 {t['표준오차‰'].mean():.2f}‰")
print(f"  |z|>2 인 팀: {(t['z'].abs() > 2).sum()}개 / {len(t)}개")

print("\n" + "=" * 92)
print("### 5) 구장 효과가 연도 간 이어지는가 (R게임만, 예측에 쓸 수 있으려면 필수)")
print("=" * 92)
sr = raw[raw.game_type == "R"].copy()
sr["dev"] = sr[TARGET] - sr.groupby("season")[TARGET].transform("mean")
piv = sr.pivot_table(index="park", columns="season", values="dev", aggfunc="mean") * 1000
piv = piv.loc[piv.index <= 21]     # 1군 구장만
print(piv.round(1).to_string())
print("\n  연도쌍 상관:")
for a, b in [(2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]:
    print(f"    {a} ↔ {b}: {piv[a].corr(piv[b]):+.3f}")
print(f"    평균: {np.mean([piv[a].corr(piv[b]) for a,b in [(2019,2020),(2020,2021),(2021,2022),(2022,2023),(2023,2024)]]):+.3f}")
