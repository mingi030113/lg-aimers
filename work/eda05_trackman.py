"""EDA 5: trackman_history 와 main 데이터의 연결 가능성 조사.

main: pitcher_id (20700~24633, 792명) / trackman: pitcher_trackman_id (다른 체계)
연결되면 구위(구속/회전/무브먼트/릴리스 일관성) 기반 투수 피처를 만들 수 있다.
단, '현재 투구의 trackman 측정값' 사용은 금지 -> 투수 단위 과거 집계만 사용.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

tm = pd.read_csv("../data/trackman_history.csv", encoding="utf-8-sig")
print("trackman shape:", tm.shape)
tm.to_parquet("trackman.parquet", index=False)

print("\n=== 결측률(%) ===")
print((tm.isna().mean() * 100).round(3).to_string())

print("\n=== nunique ===")
print(tm.nunique().to_string())

print("\n=== 시즌별 행수 ===")
print(tm.groupby("season").size())

print("\n=== pitch_type_group 분포 ===")
print(tm["pitch_type_group"].value_counts(dropna=False))
print("\n=== tagged vs auto pitch type (상위) ===")
print(tm["tagged_pitch_type"].value_counts(dropna=False).head(15))

print("\n=== 팀 코드 ===")
print(sorted(tm["pitcher_team"].dropna().unique()))
print("팀 수:", tm["pitcher_team"].nunique())

print("\n=== hand 표기 ===")
print(tm["pitcher_hand"].value_counts(dropna=False))

print("\n=== 수치 컬럼 요약 ===")
print(tm[["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
          "extension", "rel_height", "rel_side", "zone_speed"]].describe().T)

print("\n=== main 데이터 행 수 vs trackman 행 수 (시즌별) ===")
mn = pd.read_parquet("train.parquet")
cmp = pd.DataFrame({"main": mn.groupby("season").size(), "trackman": tm.groupby("season").size()})
cmp["ratio"] = cmp["trackman"] / cmp["main"]
print(cmp)

print("\n=== main 은 경기순서대로 정렬돼 있는가? (row_id 순서로 상태 진행 확인) ===")
h = mn.head(40)[["row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom",
                 "balls_before", "strikes_before", "outs_before", "base_state",
                 "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]]
print(h.to_string(index=False))

print("\n=== trackman 경기 하나의 투구 순서 (pitch_no 정렬) ===")
g0 = tm[tm.trackman_game_id == tm.trackman_game_id.iloc[0]].sort_values("pitch_no")
print(g0.head(20)[["pitch_no", "inning", "top_bottom", "balls_before", "strikes_before",
                   "outs_before", "pitch_of_pa", "pitcher_trackman_id", "batter_trackman_id",
                   "pitcher_team", "batter_team", "pitch_type_group"]].to_string(index=False))
print("경기 내 투구 수:", len(g0), " pitch_no 범위:", g0.pitch_no.min(), g0.pitch_no.max())

print("\n=== trackman 경기 수 / 시즌 ===")
print(tm.groupby("season")["trackman_game_id"].nunique())

print("\n=== trackman_game_id 구조 샘플 ===")
print(tm["trackman_game_id"].drop_duplicates().head(10).tolist())
