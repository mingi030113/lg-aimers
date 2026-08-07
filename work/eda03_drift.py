"""EDA 3: 시즌 드리프트 정량화 + asof 피처가 드리프트를 담고 있는지 검증.

핵심 질문:
  test=2025인데 train=2019~2024. 시즌별 base rate가 0.565 -> 0.486으로 단조 하락.
  Brier Skill Score 스케일에서 평균 오프셋 d 는 -100000*d^2/(r(1-r)) 점 손해.
  => d=0.01 이면 -40점, d=0.02 이면 -160점, d=0.04 이면 -641점.
  베이스라인 LB가 549점인 대회에서 이건 전부를 좌우한다.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 200)

t = pd.read_parquet("train.parquet")

print("=== 평균 오프셋 d 가 BSS 점수에 주는 손해 (r=0.49 가정) ===")
r = 0.49
for d in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
    print(f"  d={d:.3f} -> {-100000*d*d/(r*(1-r)):9.1f} 점")

print("\n=== 시즌별: target mean vs asof 피처 mean (드리프트 추적력) ===")
cols = ["control_success",
        "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_batter_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev5_game_middle_rate"]
s = t.groupby("season")[cols].mean()
print(s.T.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n=== 시즌내 월별로 보면? (2024) ===")
t24 = t[t.season == 2024]
print(t24.groupby("game_month")[cols].mean().T.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n=== 드리프트 추적 오차: target_mean - asof_prevK_mean ===")
for k in [1, 3, 5]:
    c = f"asof_pitcher_prev{k}_game_success_rate"
    diff = s["control_success"] - s[c]
    print(f"  prev{k}: " + "  ".join(f"{y}:{v:+.4f}" for y, v in diff.items()))
c = "asof_pitcher_success_rate"
diff = s["control_success"] - s[c]
print(f"  career: " + "  ".join(f"{y}:{v:+.4f}" for y, v in diff.items()))

print("\n=== 시즌 시작 시점(3~4월) vs 시즌 후반 asof_prev5 결측률 ===")
print(t.groupby(["season", "game_month"])["asof_pitcher_prev5_game_success_rate"]
      .apply(lambda x: x.isna().mean()).unstack().to_string(float_format=lambda x: f"{x:.3f}"))

print("\n=== game_type 별 시즌 추세 (F는 별도 리그로 보임) ===")
print(t.pivot_table(index="season", columns="game_type", values="control_success",
                    aggfunc=["mean", "size"]))

print("\n=== game_type별 pitcher_id 겹침 ===")
pf = set(t.loc[t.game_type == "F", "pitcher_id"])
pr = set(t.loc[t.game_type == "R", "pitcher_id"])
print(f"  F전용 {len(pf-pr)}, R전용 {len(pr-pf)}, 공통 {len(pf&pr)}")
print("\n  F/R 공통 투수의 game_type별 성공률 차이 (2024):")
t24 = t[(t.season == 2024)]
piv = t24[t24.pitcher_id.isin(pf & pr)].pivot_table(
    index="pitcher_id", columns="game_type", values="control_success", aggfunc=["mean", "size"])
piv.columns = ["mean_F", "mean_R", "n_F", "n_R"]
piv = piv[(piv.n_F > 50) & (piv.n_R > 50)]
print(f"  n={len(piv)}  mean_F={piv.mean_F.mean():.4f}  mean_R={piv.mean_R.mean():.4f}  "
      f"corr={piv.mean_F.corr(piv.mean_R):.3f}")

print("\n=== 투수 단위 성공률 산포 (2024, n>=300) ===")
pp = t24.groupby("pitcher_id")["control_success"].agg(["count", "mean"])
pp = pp[pp["count"] >= 300]
print(pp["mean"].describe())
print(f"  이론적 이항잡음 std @n=500: {np.sqrt(0.5*0.5/500):.4f}  /  실제 std: {pp['mean'].std():.4f}")

print("\n=== 타자 단위 성공률 산포 (2024, n>=300) ===")
bb = t24.groupby("batter_id")["control_success"].agg(["count", "mean"])
bb = bb[bb["count"] >= 300]
print(bb["mean"].describe())
print(f"  이론적 이항잡음 std @n=500: {np.sqrt(0.5*0.5/500):.4f}  /  실제 std: {bb['mean'].std():.4f}")
