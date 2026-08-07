"""Trackman 구위 피처 생성 + 전방검증.

규정: '현재 투구의 Trackman 측정값'과 '2025 Trackman'은 금지. 투수 단위 과거 집계는 허용.
설계: 시즌 S 행에는 **시즌 < S 의 trackman 집계만** 붙인다.
      test(2025)는 2019~2024 전체를 쓰게 되고, train 각 시즌도 같은 규칙이라 일관적이다.

핵심 가설: 릴리스포인트 산포(rel_height/rel_side/extension 의 구종내 표준편차)가
           반복성 = 커맨드의 대리지표일 것이다. asof_pitcher_success_rate 가 이미
           결과를 직접 측정하므로, 잔여 가치가 있는지가 관건.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

pd.set_option("display.width", 250)

mn = pd.read_parquet("train.parquet")
tm = pd.read_parquet("trackman.parquet")
idmap = pd.read_parquet("idmap_pitcher_v2.parquet")

# ---------------- 0. 구종 그룹 정의 차이 진단 ----------------
print("=" * 100)
print("### 진단: main 의 fastball 정의가 trackman 의 어느 조합인가")
print("=" * 100)
print(tm.pitch_type_group.value_counts(dropna=False).to_string())
last = mn.groupby("pitcher_id").tail(1).set_index("pitcher_id")
good = idmap[idmap.share > 0.5]
tag = tm.groupby(["pitcher_trackman_id", "tagged_pitch_type"]).size().unstack(fill_value=0)
tot = tag.sum(axis=1)
for name, cols in [("Fastball만", ["Fastball"]),
                   ("Fastball+Sinker", ["Fastball", "Sinker"]),
                   ("Fastball+Cutter", ["Fastball", "Cutter"]),
                   ("Fastball+Sinker+Cutter", ["Fastball", "Sinker", "Cutter"])]:
    c = [x for x in cols if x in tag.columns]
    share = tag[c].sum(axis=1) / tot
    v = pd.DataFrame({"m": good.pitcher_id.map(last["asof_pitcher_fastball_rate"]).values,
                      "t": good.tm_pitcher_id.map(share).values}).dropna()
    print(f"  {name:24s} 상관={v.m.corr(v.t):.4f}  MAE={(v.m-v.t).abs().mean():.4f}")

# ---------------- 1. Hungarian 으로 1:1 정제 ----------------
print("\n" + "=" * 100)
print("### 헝가리안 1:1 배정으로 정제")
print("=" * 100)
mh = mn.assign(half=mn.inning * 2 + (mn.top_bottom == "B")).copy()
# tm02 의 투표행렬을 재현하지 않고, share>0.5 인 것만 채택하고 중복만 정리한다
sel = idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
sel = sel.drop_duplicates("tm_pitcher_id", keep="first")
print(f"  share>0.5 이고 중복 제거 후 {len(sel)}명 / main 792명 "
      f"({len(sel)/792*100:.1f}% 커버)")
print(f"  좌우 일치율 {sel.hand_ok.mean()*100:.2f}%")
P2T = dict(zip(sel.pitcher_id, sel.tm_pitcher_id))

# ---------------- 2. 투수 x 시즌 trackman 요약 ----------------
NUM = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
       "extension", "rel_height", "rel_side", "zone_speed"]
tm2 = tm[tm.pitch_type_group.isin(["fastball", "breaking", "offspeed"])].copy()

# (투수, 시즌, 구종군) 단위 평균/표준편차 -> 구종내 산포가 곧 반복성
g = tm2.groupby(["pitcher_trackman_id", "season", "pitch_type_group"])
agg = g[NUM].agg(["mean", "std"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg["n"] = g.size()
agg = agg.reset_index()

# 구종군별로 펼치기 + 구종내 산포의 가중평균(=릴리스 반복성)
def per_season(df):
    out = []
    for (pid, s), sub in df.groupby(["pitcher_trackman_id", "season"]):
        n = sub["n"].to_numpy()
        w = n / n.sum()
        r = {"pitcher_trackman_id": pid, "season": s, "tm_n": n.sum()}
        # 릴리스 반복성: 구종내 std 를 투구수 가중평균
        for c in ["rel_height", "rel_side", "extension"]:
            r[f"rel_var_{c}"] = np.nansum(w * sub[f"{c}_std"].to_numpy())
        # 전체 평균(투구수 가중)
        for c in NUM:
            r[f"m_{c}"] = np.nansum(w * sub[f"{c}_mean"].to_numpy())
        # 구종군별 구속
        for grp in ["fastball", "breaking", "offspeed"]:
            m = sub.pitch_type_group == grp
            r[f"v_{grp}"] = sub.loc[m, "rel_speed_mean"].mean() if m.any() else np.nan
            r[f"sp_{grp}"] = sub.loc[m, "spin_rate_mean"].mean() if m.any() else np.nan
        r["n_types"] = int((n / n.sum() >= 0.05).sum())
        out.append(r)
    return pd.DataFrame(out)


ps = per_season(agg)
ps["velo_sep"] = ps["v_fastball"] - ps["v_offspeed"]
print(f"\n투수x시즌 요약 {len(ps):,}행")

# ---------------- 3. '시즌 < S' 누적 (투구수 가중) ----------------
VALS = [c for c in ps.columns if c not in ("pitcher_trackman_id", "season", "tm_n")]
rows = []
for S in range(2020, 2026):
    sub = ps[ps.season < S]
    if not len(sub):
        continue
    w = sub["tm_n"].to_numpy()
    d = sub[["pitcher_trackman_id"]].copy()
    for c in VALS:
        d[c] = sub[c].to_numpy() * w
    d["_w"] = w
    d["_wv"] = w * (~sub[VALS[0]].isna()).to_numpy()
    acc = d.groupby("pitcher_trackman_id").sum(min_count=1)
    for c in VALS:
        acc[c] = acc[c] / acc["_w"]
    acc["tm_prior_n"] = acc["_w"]
    acc = acc.drop(columns=["_w", "_wv"]).reset_index()
    acc["season"] = S
    rows.append(acc)
prior = pd.concat(rows, ignore_index=True)
prior.columns = [c if c in ("pitcher_trackman_id", "season")
                 else ("tm_" + c if not c.startswith("tm_") else c) for c in prior.columns]
prior.to_parquet("tm_prior.parquet", index=False)
print(f"직전시즌까지 누적 {len(prior):,}행, 피처 {len(prior.columns)-2}개")
print("  피처:", [c for c in prior.columns if c not in ("pitcher_trackman_id", "season")])

# ---------------- 4. main 에 부착 & 커버리지 ----------------
mn["tm_pid"] = mn["pitcher_id"].map(P2T)
att = mn[["row_id", "season", "tm_pid"]].merge(
    prior, left_on=["tm_pid", "season"], right_on=["pitcher_trackman_id", "season"], how="left")
cov = att.groupby("season")["tm_prior_n"].apply(lambda x: x.notna().mean())
print("\n시즌별 부착 커버리지:")
print((cov * 100).round(1).to_string())
