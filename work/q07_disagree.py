"""tagged_pitch_type vs auto_pitch_type 불일치가 쓸 만한 신호인지 조사.

가설: 자동 분류기가 태깅과 다르게 본 투구 = 구질 형태가 애매한 투구.
      투수 단위로 집계하면 '구종 구분이 불분명한 투수' 지표가 된다.

검증 순서
  1) 라벨 표기 정규화 (Changeup/ChangeUp, SInker/Sinker, Four-Seam/Fastball 등)
  2) 불일치율 전체·시즌별 (시즌별로 튀면 분류기 버전 교체 = 시스템 인공물)
  3) 투수 단위 불일치율의 참 산포 (이항잡음 대비)
  4) 시즌 간 안정성 (예측에 쓰려면 필수)
  5) 제구 성공률과의 관계 (매핑된 투수만)
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)

tm = pd.read_parquet("trackman.parquet")
print(f"trackman {len(tm):,}행")

print("\n=== 원본 라벨 분포 ===")
print("tagged_pitch_type:")
print(tm.tagged_pitch_type.value_counts(dropna=False).to_string())
print("\nauto_pitch_type:")
print(tm.auto_pitch_type.value_counts(dropna=False).to_string())

CANON = {
    "Fastball": "FA", "Four-Seam": "FA", "FourSeamFastBall": "FA",
    "Sinker": "SI", "SInker": "SI", "TwoSeamFastBall": "SI",
    "Cutter": "FC", "Slider": "SL", "Sweeper": "SL", "Slurve": "SL",
    "Curveball": "CU", "Knuckleball": "KN",
    "ChangeUp": "CH", "Changeup": "CH", "Splitter": "FS",
    "Undefined": None, "Other": None,
}


def canon(s):
    return s.map(lambda x: CANON.get(x, "?" if pd.notna(x) else None))


tm["t"] = canon(tm["tagged_pitch_type"])
tm["a"] = canon(tm["auto_pitch_type"])
print("\n  정규화 후 미매핑(?) 개수: tagged", (tm.t == "?").sum(), " auto", (tm.a == "?").sum())

ok = tm.t.notna() & tm.a.notna()
tm["dis"] = np.where(ok, (tm.t != tm.a).astype(float), np.nan)
print(f"\n=== 불일치율 (양쪽 라벨이 있는 {ok.sum():,}행 기준) ===")
print(f"  전체 불일치율 = {tm.loc[ok,'dis'].mean()*100:.2f}%")

print("\n=== 시즌별 (분류기 버전 교체 흔적 확인) ===")
g = tm[ok].groupby("season")["dis"].agg(["size", "mean"])
g["불일치%"] = (g["mean"] * 100).round(2)
print(g[["size", "불일치%"]].to_string())

print("\n=== 구종별 불일치 (tagged 기준) ===")
h = tm[ok].groupby("t")["dis"].agg(["size", "mean"])
h["불일치%"] = (h["mean"] * 100).round(2)
print(h.sort_values("size", ascending=False)[["size", "불일치%"]].to_string())

print("\n=== 혼동 행렬 (상위, tagged 행 x auto 열) ===")
ct = pd.crosstab(tm.loc[ok, "t"], tm.loc[ok, "a"])
print(ct.to_string())

print("\n=== 투수 단위 불일치율 — 참 산포가 있는가 ===")
p = tm[ok].groupby("pitcher_trackman_id")["dis"].agg(["size", "mean"])
p = p[p["size"] >= 300]
noise = np.sqrt((p["mean"] * (1 - p["mean"]) / p["size"])).mean()
print(f"  투수 {len(p)}명 (300구 이상)")
print(f"  불일치율 평균 {p['mean'].mean()*100:.2f}%  실제 표준편차 {p['mean'].std()*100:.2f}%p")
print(f"  이항잡음만으로 기대되는 표준편차 {noise*100:.2f}%p")
true_sd = np.sqrt(max(p["mean"].var() - (noise ** 2), 0))
print(f"  -> 참 산포 ≈ {true_sd*100:.2f}%p  ({'실재' if true_sd > noise else '노이즈 수준'})")
print(p["mean"].describe().apply(lambda x: f"{x*100:.2f}%").to_string())

print("\n=== 시즌 간 안정성 (2023 vs 2024) ===")
ps = tm[ok].groupby(["pitcher_trackman_id", "season"])["dis"].agg(["size", "mean"])
ps = ps[ps["size"] >= 150]["mean"].unstack()
for a, b in [(2021, 2022), (2022, 2023), (2023, 2024)]:
    if a in ps.columns and b in ps.columns:
        v = ps[[a, b]].dropna()
        print(f"  {a} ↔ {b}: 상관 {v[a].corr(v[b]):+.3f}  (n={len(v)})")

print("\n=== 제구 성공률과의 관계 (매핑된 투수, R게임 2023~2024) ===")
idmap = pd.read_parquet("idmap_pitcher_v2.parquet")
sel = (idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
       .drop_duplicates("tm_pitcher_id", keep="first"))
t2p = dict(zip(sel.tm_pitcher_id, sel.pitcher_id))
p["pitcher_id"] = p.index.map(t2p)
pm = p.dropna(subset=["pitcher_id"]).set_index("pitcher_id")
mn = pd.read_parquet("train.parquet")
sub = mn[(mn.game_type == "R") & (mn.season >= 2023)]
ps2 = sub.groupby("pitcher_id")["control_success"].agg(["size", "mean"])
ps2 = ps2[ps2["size"] >= 300]
j = ps2.join(pm[["mean"]].rename(columns={"mean": "dis"}), how="inner")
print(f"  투수 {len(j)}명")
print(f"  불일치율 vs 제구 성공률 상관 = {j['dis'].corr(j['mean']):+.4f}")
j["dis_q"] = pd.qcut(j["dis"], 5, labels=["최저", "낮음", "중간", "높음", "최고"])
print(j.groupby("dis_q", observed=True).agg(투수수=("mean", "size"), 불일치율=("dis", "mean"),
                                            제구성공률=("mean", "mean"))
      .to_string(float_format=lambda x: f"{x:.4f}"))
