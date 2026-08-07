"""Trackman ID 매핑 복구.

main(train.csv) 의 pitcher_id 와 trackman 의 pitcher_trackman_id 는 체계가 다르다.
두 로그가 같은 실제 투구들을 담고 있다는 점을 이용해, 투구 상태 서명의
동시출현 빈도로 매핑을 복원한다.

서명 키: (season, game_month, game_dayofweek, inning, top_bottom,
          balls_before, strikes_before, outs_before)

main 에는 game_date 가 없어 같은 요일의 여러 날짜가 한 키에 뭉치지만,
한 투수는 시즌 내 20~30개의 서로 다른 등판일에 걸쳐 나타나므로
참 짝은 모든 등판일에서 함께 나타나고 가짜 짝은 몇 번만 겹친다.

검증(매칭에 쓰지 않은 정보로):
  1) 좌우 일치        main pitcher_hand(1/2)  vs trackman pitcher_hand(Right/Left)
  2) 시즌 집합 일치
  3) 단사성           한 trackman id 를 두 main id 가 차지하지 않는지
  4) 구종 구성 일치   main asof_pitcher_fastball_rate vs trackman fastball 비율
"""
import numpy as np
import pandas as pd
from scipy import sparse

pd.set_option("display.width", 250)

KEY = ["season", "game_month", "game_dayofweek", "inning", "tb",
       "balls_before", "strikes_before", "outs_before"]

mn = pd.read_parquet("train.parquet")
tm = pd.read_parquet("trackman.parquet")

mn["tb"] = (mn["top_bottom"] == "B").astype(np.int8)
tm["tb"] = (tm["top_bottom"] == "Bottom").astype(np.int8)

# 공통 키 공간 (문자열 결합 대신 정수 인코딩)
def keycode(df):
    k = df["season"].astype(np.int64) - 2019
    for col, card in [("game_month", 13), ("game_dayofweek", 7), ("inning", 20),
                      ("tb", 2), ("balls_before", 4), ("strikes_before", 3),
                      ("outs_before", 3)]:
        k = k * card + df[col].astype(np.int64)
    return k.to_numpy()


mn_k, tm_k = keycode(mn), keycode(tm)
allk, inv = np.unique(np.concatenate([mn_k, tm_k]), return_inverse=True)
K = len(allk)
mn_ki, tm_ki = inv[:len(mn_k)], inv[len(mn_k):]
print(f"키 공간 {K:,}개  (main {len(mn):,}행, trackman {len(tm):,}행)")


def build(ids, kidx, K):
    codes, uniq = pd.factorize(ids)
    M = sparse.csr_matrix((np.ones(len(codes), np.float32), (codes, kidx)),
                          shape=(len(uniq), K))
    return M, uniq


def match(main_id_col, tm_id_col, label):
    M, mu = build(mn[main_id_col].to_numpy(), mn_ki, K)
    T, tu = build(tm[tm_id_col].to_numpy(), tm_ki, K)
    nm = np.asarray(M.sum(1)).ravel()
    nt = np.asarray(T.sum(1)).ravel()

    # IDF: 흔한 상태(0-0 카운트 등)는 정보가 적다
    df = np.asarray((M > 0).sum(0)).ravel() + np.asarray((T > 0).sum(0)).ravel()
    idf = np.log(1.0 + (len(mu) + len(tu)) / np.maximum(df, 1)).astype(np.float32)
    W = sparse.diags(idf)
    O = np.asarray((M @ W @ T.T).todense())

    # 독립 가정 대비 초과 동시출현 (PMI 유사)
    S = O / np.outer(np.maximum(nm, 1), np.maximum(nt, 1))

    order = np.argsort(-S, axis=1)
    best, second = order[:, 0], order[:, 1]
    top = S[np.arange(len(mu)), best]
    sec = S[np.arange(len(mu)), second]
    margin = top / np.maximum(sec, 1e-12)

    res = pd.DataFrame({label: mu, f"tm_{label}": tu[best], "n_main": nm,
                        "n_tm": nt[best], "score": top, "margin": margin})
    print(f"\n=== {label} 매핑 ===")
    print(f"  main {len(mu)}명 -> trackman {len(tu)}명 중 선택")
    print(f"  margin 분포: " + "  ".join(
        f"p{q}={np.percentile(margin, q):.2f}" for q in [1, 5, 25, 50, 75]))
    for th in [1.5, 2, 3, 5, 10]:
        print(f"    margin>{th:<4}: {(margin > th).sum():4d}명 ({(margin>th).mean()*100:5.1f}%)")
    dup = res[f"tm_{label}"].duplicated(keep=False).sum()
    print(f"  단사성 위반(중복 배정): {dup}건")
    return res, S, mu, tu


pres, PS, pmu, ptu = match("pitcher_id", "pitcher_trackman_id", "pitcher_id")
bres, BS, bmu, btu = match("batter_id", "batter_trackman_id", "batter_id")

# ---------------- 검증 ----------------
print("\n" + "=" * 100)
print("### 검증 1: 좌우 일치 (매칭에 쓰지 않은 정보)")
print("=" * 100)
mh = mn.groupby("pitcher_id")["pitcher_hand"].agg(lambda x: x.mode()[0])
th = tm.groupby("pitcher_trackman_id")["pitcher_hand"].agg(lambda x: x.mode()[0])
chk = pres.copy()
chk["main_hand"] = chk["pitcher_id"].map(mh)
chk["tm_hand"] = chk["tm_pitcher_id"].map(th)
ct = pd.crosstab(chk["main_hand"], chk["tm_hand"])
print(ct)
# 1/2 <-> Right/Left 대응을 다수결로 결정
lut = ct.idxmax(axis=1).to_dict()
chk["hand_ok"] = chk.apply(lambda r: lut.get(r["main_hand"]) == r["tm_hand"], axis=1)
print(f"\n  전체 일치율: {chk['hand_ok'].mean()*100:.1f}%")
for th_ in [1.5, 2, 3, 5]:
    s = chk[chk.margin > th_]
    print(f"    margin>{th_:<4}: {s['hand_ok'].mean()*100:5.1f}%  (n={len(s)})")

print("\n" + "=" * 100)
print("### 검증 2: 시즌 집합 일치")
print("=" * 100)
ms = mn.groupby("pitcher_id")["season"].apply(set)
ts = tm.groupby("pitcher_trackman_id")["season"].apply(set)
chk["seasons_main"] = chk["pitcher_id"].map(ms)
chk["seasons_tm"] = chk["tm_pitcher_id"].map(ts)
chk["season_jac"] = [len(a & b) / len(a | b) for a, b in zip(chk.seasons_main, chk.seasons_tm)]
print(f"  자카드 중앙값: {chk.season_jac.median():.3f}")
for th_ in [1.5, 2, 3, 5]:
    s = chk[chk.margin > th_]
    print(f"    margin>{th_:<4}: 자카드 중앙값 {s.season_jac.median():.3f}, "
          f"완전일치 {(s.season_jac == 1).mean()*100:5.1f}%")

print("\n" + "=" * 100)
print("### 검증 3: 구종 구성 일치 (완전 독립 검증)")
print("=" * 100)
# main: 각 투수 커리어 마지막 행의 asof_pitcher_fastball_rate ≈ 커리어 fastball 비율
last = mn.sort_values("row_id").groupby("pitcher_id").tail(1).set_index("pitcher_id")
tmix = (tm[tm.pitch_type_group.isin(["fastball", "breaking", "offspeed"])]
        .groupby(["pitcher_trackman_id", "pitch_type_group"]).size().unstack(fill_value=0))
tmix = tmix.div(tmix.sum(1), axis=0)
chk["main_fb"] = chk["pitcher_id"].map(last["asof_pitcher_fastball_rate"])
chk["tm_fb"] = chk["tm_pitcher_id"].map(tmix["fastball"])
chk["main_br"] = chk["pitcher_id"].map(last["asof_pitcher_breaking_rate"])
chk["tm_br"] = chk["tm_pitcher_id"].map(tmix["breaking"])
v = chk.dropna(subset=["main_fb", "tm_fb"])
print(f"  fastball 비율 상관: {v.main_fb.corr(v.tm_fb):.4f}   (n={len(v)})")
print(f"  breaking 비율 상관: {v.main_br.corr(v.tm_br):.4f}")
for th_ in [1.5, 2, 3, 5, 10]:
    s = v[v.margin > th_]
    if len(s) < 20:
        continue
    print(f"    margin>{th_:<4}: fb상관 {s.main_fb.corr(s.tm_fb):.4f}  "
          f"br상관 {s.main_br.corr(s.tm_br):.4f}  평균절대오차 {(s.main_fb-s.tm_fb).abs().mean():.4f}  n={len(s)}")

# 랜덤 매칭 대조군
rng = np.random.default_rng(0)
shuf = v.copy(); shuf["tm_fb"] = rng.permutation(shuf["tm_fb"].values)
print(f"  [대조군] 무작위 매칭 fb상관: {shuf.main_fb.corr(shuf.tm_fb):.4f}")

chk.drop(columns=["seasons_main", "seasons_tm"]).to_parquet("idmap_pitcher.parquet", index=False)
bres.to_parquet("idmap_batter.parquet", index=False)
print("\n저장: idmap_pitcher.parquet, idmap_batter.parquet")
