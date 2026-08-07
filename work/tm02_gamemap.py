"""Trackman ID 매핑 2단계: 경기 복원 -> 경기 정합 -> 이닝 단위 투표.

1단계(tm01)의 키 투표는 (시즌,월,요일) 안에 ~20경기가 뭉쳐 정밀도가 낮았다.
main 이 실제 경기·투구 순서로 정렬돼 있으므로 경기를 복원해 1:1로 붙이면
반이닝 단위까지 정합이 되어 훨씬 깨끗한 투표가 가능하다.

경기 경계 규칙: ord = inning*2 + (0 if 초 else 1) 는 경기 내에서 단조증가한다.
                ord 가 감소하거나 (시즌,월,요일)/팀쌍이 바뀌면 새 경기.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment

pd.set_option("display.width", 250)

mn = pd.read_parquet("train.parquet")
tm = pd.read_parquet("trackman.parquet")
mn["tb"] = (mn["top_bottom"] == "B").astype(np.int8)
tm["tb"] = (tm["top_bottom"] == "Bottom").astype(np.int8)

# ---------------- 1. main 경기 복원 ----------------
mn = mn.sort_values("row_id", kind="stable").reset_index(drop=True)
ordv = mn["inning"].to_numpy() * 2 + mn["tb"].to_numpy()
teams = (np.minimum(mn.pitcher_team_id, mn.batter_team_id).to_numpy() * 100
         + np.maximum(mn.pitcher_team_id, mn.batter_team_id).to_numpy())
day = (mn.season.to_numpy() * 1000 + mn.game_month.to_numpy() * 10
       + mn.game_dayofweek.to_numpy())
newg = np.zeros(len(mn), bool)
newg[0] = True
newg[1:] = (ordv[1:] < ordv[:-1]) | (teams[1:] != teams[:-1]) | (day[1:] != day[:-1])
mn["game"] = np.cumsum(newg) - 1
ng = mn["game"].nunique()
sz = mn.groupby("game").size()
print(f"main 경기 복원: {ng:,}경기,  투구/경기 중앙값 {sz.median():.0f} "
      f"(p5={sz.quantile(.05):.0f}, p95={sz.quantile(.95):.0f})")
print(mn.groupby("season")["game"].nunique().rename("main 경기수").to_frame()
      .join(tm.groupby("season")["trackman_game_id"].nunique().rename("tm 경기수")))

# ---------------- 2. 경기 서명 벡터 ----------------
# (inning<=13, half, balls, strikes, outs) 상태 히스토그램
def state_code(df):
    inn = np.clip(df["inning"].to_numpy(), 1, 13) - 1
    return (((inn * 2 + df["tb"].to_numpy()) * 4
             + df["balls_before"].to_numpy()) * 3
            + df["strikes_before"].to_numpy()) * 3 + df["outs_before"].to_numpy()


D = 13 * 2 * 4 * 3 * 3
mn["sc"] = state_code(mn)
tm["sc"] = state_code(tm)
tm["game"] = pd.factorize(tm["trackman_game_id"])[0]

def game_matrix(df, gcol):
    g = df[gcol].to_numpy()
    ng_ = g.max() + 1
    M = sparse.csr_matrix((np.ones(len(df), np.float32), (g, df["sc"].to_numpy())),
                          shape=(ng_, D))
    return M


MG, TG = game_matrix(mn, "game"), game_matrix(tm, "game")
mn_gi = mn.groupby("game").agg(season=("season", "first"), mon=("game_month", "first"),
                               dow=("game_dayofweek", "first"), n=("season", "size"))
tm_gi = tm.groupby("game").agg(season=("season", "first"), mon=("game_month", "first"),
                               dow=("game_dayofweek", "first"), n=("season", "size"))

def l2norm(M):
    n = np.sqrt(np.asarray(M.multiply(M).sum(1))).ravel()
    return sparse.diags(1.0 / np.maximum(n, 1e-9)) @ M


MGn, TGn = l2norm(MG), l2norm(TG)

# ---------------- 3. (시즌,월,요일) 버킷별 경기 1:1 배정 ----------------
pairs = []
mn_gi["key"] = list(zip(mn_gi.season, mn_gi.mon, mn_gi.dow))
tm_gi["key"] = list(zip(tm_gi.season, tm_gi.mon, tm_gi.dow))
tm_by = tm_gi.groupby("key").groups
for key, midx in mn_gi.groupby("key").groups.items():
    tidx = tm_by.get(key)
    if tidx is None:
        continue
    mi, ti = np.asarray(midx), np.asarray(tidx)
    S = np.asarray((MGn[mi] @ TGn[ti].T).todense())
    r, c = linear_sum_assignment(-S)
    for a, b in zip(r, c):
        pairs.append((mi[a], ti[b], S[a, b]))

pm = pd.DataFrame(pairs, columns=["mgame", "tgame", "sim"])
print(f"\n경기 정합: {len(pm):,}쌍  유사도 분포: " +
      "  ".join(f"p{q}={pm.sim.quantile(q/100):.3f}" for q in [5, 25, 50, 75, 95]))
for th in [0.7, 0.8, 0.85, 0.9]:
    print(f"    sim>{th}: {(pm.sim > th).sum():,}쌍 ({(pm.sim>th).mean()*100:.1f}%)")

SIM_TH = 0.80
pm = pm[pm.sim > SIM_TH]
print(f"  -> sim>{SIM_TH} 인 {len(pm):,}쌍만 사용")

# ---------------- 4. 반이닝 단위 투표 ----------------
mn["half"] = mn["inning"] * 2 + mn["tb"]
tm["half"] = tm["inning"] * 2 + tm["tb"]
mh = (mn.groupby(["game", "half", "pitcher_id"]).size().rename("cm").reset_index())
th_ = (tm.groupby(["game", "half", "pitcher_trackman_id"]).size().rename("ct").reset_index())
mh = mh.merge(pm[["mgame", "tgame"]], left_on="game", right_on="mgame")
j = mh.merge(th_, left_on=["tgame", "half"], right_on=["game", "half"], suffixes=("", "_t"))
j["w"] = np.minimum(j["cm"], j["ct"])
vote = j.groupby(["pitcher_id", "pitcher_trackman_id"])["w"].sum().reset_index()
print(f"\n투표 쌍 {len(vote):,}개")

pu = np.sort(mn.pitcher_id.unique())
tu = np.sort(tm.pitcher_trackman_id.unique())
pi = {v: i for i, v in enumerate(pu)}
ti = {v: i for i, v in enumerate(tu)}
V = np.zeros((len(pu), len(tu)))
V[vote.pitcher_id.map(pi).to_numpy(), vote.pitcher_trackman_id.map(ti).to_numpy()] = vote.w
rowsum = V.sum(1, keepdims=True)
share = V / np.maximum(rowsum, 1)          # 이 투수 표의 몇 %가 그 후보에게 갔는가
order = np.argsort(-V, axis=1)
best, second = order[:, 0], order[:, 1]
res = pd.DataFrame({
    "pitcher_id": pu,
    "tm_pitcher_id": tu[best],
    "votes": V[np.arange(len(pu)), best],
    "votes2": V[np.arange(len(pu)), second],
    "share": share[np.arange(len(pu)), best],
})
res["margin"] = res.votes / np.maximum(res.votes2, 1)
print("  최고득표 점유율 분포: " +
      "  ".join(f"p{q}={res.share.quantile(q/100):.3f}" for q in [5, 25, 50, 75]))
for th2 in [0.5, 0.7, 0.8, 0.9]:
    print(f"    share>{th2}: {(res.share > th2).sum():4d}명 ({(res.share>th2).mean()*100:5.1f}%)")
print(f"  단사성 위반: {res.tm_pitcher_id.duplicated(keep=False).sum()}건")

# ---------------- 5. 검증 ----------------
print("\n" + "=" * 100)
print("### 검증 (매칭에 쓰지 않은 정보)")
print("=" * 100)
mhand = mn.groupby("pitcher_id")["pitcher_hand"].agg(lambda x: x.mode()[0])
thand = tm.groupby("pitcher_trackman_id")["pitcher_hand"].agg(lambda x: x.mode()[0])
res["main_hand"] = res.pitcher_id.map(mhand)
res["tm_hand"] = res.tm_pitcher_id.map(thand)
ct = pd.crosstab(res.main_hand, res.tm_hand)
lut = ct.idxmax(axis=1).to_dict()
res["hand_ok"] = [lut.get(a) == b for a, b in zip(res.main_hand, res.tm_hand)]
print(ct)

last = mn.groupby("pitcher_id").tail(1).set_index("pitcher_id")
tmix = (tm[tm.pitch_type_group.isin(["fastball", "breaking", "offspeed"])]
        .groupby(["pitcher_trackman_id", "pitch_type_group"]).size().unstack(fill_value=0))
tmix = tmix.div(tmix.sum(axis=1), axis=0)
res["main_fb"] = res.pitcher_id.map(last["asof_pitcher_fastball_rate"])
res["tm_fb"] = res.tm_pitcher_id.map(tmix["fastball"])
res["main_br"] = res.pitcher_id.map(last["asof_pitcher_breaking_rate"])
res["tm_br"] = res.tm_pitcher_id.map(tmix["breaking"])
res["main_of"] = res.pitcher_id.map(last["asof_pitcher_offspeed_rate"])
res["tm_of"] = res.tm_pitcher_id.map(tmix["offspeed"])

print(f"\n  {'조건':>16} {'n':>5} {'좌우일치':>8} {'fb상관':>8} {'br상관':>8} {'of상관':>8} {'fb MAE':>8}")
for lab, s in [("전체", res),
               ("share>0.5", res[res.share > 0.5]),
               ("share>0.7", res[res.share > 0.7]),
               ("share>0.8", res[res.share > 0.8]),
               ("share>0.9", res[res.share > 0.9])]:
    v = s.dropna(subset=["main_fb", "tm_fb"])
    if len(v) < 10:
        continue
    print(f"  {lab:>16} {len(s):5d} {s.hand_ok.mean()*100:7.1f}% "
          f"{v.main_fb.corr(v.tm_fb):8.4f} {v.main_br.corr(v.tm_br):8.4f} "
          f"{v.main_of.corr(v.tm_of):8.4f} {(v.main_fb-v.tm_fb).abs().mean():8.4f}")

res.to_parquet("idmap_pitcher_v2.parquet", index=False)
pm.to_parquet("gamemap.parquet", index=False)
print("\n저장: idmap_pitcher_v2.parquet, gamemap.parquet")
