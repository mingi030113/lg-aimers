"""EDA 1: 기본 구조 / 시즌 분포 / 결측 / 타깃 분포."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

DATA = "../data"

train = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
print("shape:", train.shape)
train.to_parquet("train.parquet", index=False)

print("\n=== dtypes ===")
print(train.dtypes.to_string())

print("\n=== season x target ===")
g = train.groupby("season")["control_success"].agg(["count", "mean"])
g["brier_base"] = g["mean"] * (1 - g["mean"])
print(g)

print("\n=== overall target ===")
r = train["control_success"].mean()
print(f"r={r:.6f}  base_brier={r*(1-r):.6f}")

print("\n=== missing rate (%) ===")
mr = (train.isna().mean() * 100).sort_values(ascending=False)
print(mr[mr > 0].to_string())

print("\n=== nunique ===")
print(train.nunique().to_string())

print("\n=== game_type / top_bottom / hands ===")
for c in ["game_type", "top_bottom", "pitcher_hand", "batter_hand", "base_state"]:
    print(f"\n-- {c}")
    print(train.groupby(c)["control_success"].agg(["count", "mean"]))

print("\n=== season x game_type ===")
print(pd.crosstab(train["season"], train["game_type"]))

print("\n=== describe (numeric) ===")
print(train.describe().T.to_string())
