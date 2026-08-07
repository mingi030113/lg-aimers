"""캘리브레이션 감사: 최종 파이프라인을 폴드에 그대로 돌려 '어디서 틀리는지' 분해.

LB 점수만으로는 레벨오차와 판별력을 분리할 수 없으므로, 실제 제출과 동일한 절차
(train<=Y-1 -> DRIFT/D_AUTO 로 시프트 -> Y 평가)를 재현해 오프라인에서 분해한다.

분해:
  Score = 1e5 * [2Cov(p,y) - Var(p) - (p̄-r)^2] / r(1-r)
          |______ 판별력 ______|   |_ 레벨손실 _|

추가로 (a) 신뢰도 곡선 -> 과신/과소신뢰 여부, (b) 부분집단별 편향 -> 국소 보정 여지.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from features import build, feature_list, attach_trackman
from train_final import (PARAMS, ROUNDS, N_SEEDS, LOGREG_C, BLEND_W,
                         DRIFT, D_AUTO, sigmoid, solve_shift, TARGET)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)

# ---------------- 데이터 (제출본과 동일: Trackman 포함) ----------------
raw = pd.read_parquet("train.parquet")
idmap = pd.read_parquet("idmap_pitcher_v2.parquet")
sel = (idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
       .drop_duplicates("tm_pitcher_id", keep="first"))
t2p = {v: k for k, v in zip(sel.pitcher_id, sel.tm_pitcher_id)}
pri = pd.read_parquet("tm_prior.parquet")
pri["pitcher_id"] = pri["pitcher_trackman_id"].map(t2p)
pri = pri.dropna(subset=["pitcher_id"]).drop(columns=["pitcher_trackman_id"])
pri["pitcher_id"] = pri["pitcher_id"].astype(int)
raw = attach_trackman(raw, pri)

d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
for c in ["game_month", "inning", "balls_before", "strikes_before", "li",
          "asof_pitcher_n", "asof_batter_n", "pitcher_hand", "batter_hand"]:
    if c not in d.columns:
        d[c] = raw[c].values
FEATS = [c for c in feature_list(d, use_season=False, use_ids=False) if c != TARGET]
del raw
print(f"피처 {len(FEATS)}개")


def full_procedure(upto):
    """제출과 동일한 절차. upto 까지 학습하고, upto 시즌을 기준으로 시프트를 잡는다."""
    msk = ((d.season <= upto) & ~((d.is_F == 1) & (d.season <= 2022))).values
    X, y = d.loc[msk, FEATS], d.loc[msk, TARGET].values
    z = np.zeros(len(d))
    for s in range(N_SEEDS):
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        m = lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS)
        z += m.predict(d[FEATS], raw_score=True)
    z /= N_SEEDS
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=LOGREG_C, max_iter=300, solver="lbfgs").fit((A - mu) / sd, y)
    B = d[FEATS].fillna(med).to_numpy(np.float64)
    zb = (1 - BLEND_W) * z + BLEND_W * (((B - mu) / sd) @ lr.coef_[0] + lr.intercept_[0])

    lvl = d[d.season == upto].groupby("is_F")[TARGET].mean()
    shifts = {}
    for g in (0, 1):
        if g not in lvl.index:
            continue
        s_ = ((d.season == upto) & (d.is_F == g)).values
        shifts[g] = solve_shift(zb[s_], float(lvl[g]) + DRIFT - D_AUTO[g])
    return zb, shifts


def decomp(p, y, base=None):
    r = y.mean()
    B = r * (1 - r) if base is None else base
    e = p.mean() - r
    cov = np.cov(p, y, bias=True)[0, 1]
    var = p.var()
    return dict(n=len(y), r=r, pbar=p.mean(), e=e, var=var, cov=cov,
                slope=cov / var if var > 0 else np.nan,
                score=1e5 * (2 * cov - var - e * e) / B,
                disc=1e5 * (2 * cov - var) / B,
                level_loss=-1e5 * e * e / B,
                best_scale=1e5 * (cov * cov / var - e * e) / B)


RES = {}
for upto, Y in [(2022, 2023), (2023, 2024)]:
    zb, shifts = full_procedure(upto)
    ev = (d.season == Y).values
    p = np.where(d.loc[ev, "is_F"].values == 1,
                 sigmoid(zb[ev] + shifts.get(1, shifts[0])), sigmoid(zb[ev] + shifts[0]))
    RES[Y] = (p, d.loc[ev].reset_index(drop=True))

    print("\n" + "=" * 112)
    print(f"### 폴드 {Y} — 제출과 동일한 절차 (train<={upto}, {upto}시즌 기준 시프트)")
    print("=" * 112)
    sub = d.loc[ev].reset_index(drop=True)
    for lab, m in [("전체", np.ones(len(sub), bool)),
                   ("R만", (sub.is_F == 0).values), ("F만", (sub.is_F == 1).values)]:
        if m.sum() < 100:
            continue
        r = decomp(p[m], sub.loc[m, TARGET].values)
        print(f"  [{lab:4s}] n={r['n']:7d}  실제평균={r['r']:.4f} 예측평균={r['pbar']:.4f} "
              f"오차 e={r['e']:+.4f}")
        print(f"          총점={r['score']:8.1f}  = 판별력 {r['disc']:8.1f} "
              f"+ 레벨손실 {r['level_loss']:7.1f}")
        print(f"          Var(p)={r['var']:.6f} Cov(p,y)={r['cov']:.6f} "
              f"기울기={r['slope']:.3f} (>1이면 과소신뢰)  최적재척도시={r['best_scale']:8.1f}")

print("\n" + "=" * 112)
print("### 신뢰도 곡선 (예측 20분위 -> 실제) — 과신/과소신뢰 진단")
print("=" * 112)
for Y in RES:
    p, sub = RES[Y]
    m = (sub.is_F == 0).values
    q = pd.qcut(p[m], 20, labels=False, duplicates="drop")
    t = pd.DataFrame({"q": q, "p": p[m], "y": sub.loc[m, TARGET].values})
    g = t.groupby("q").agg(n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
    g["편차"] = g["obs"] - g["pred"]
    print(f"\n  --- {Y} (R게임) ---")
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))


print("\n" + "=" * 112)
print("### 부분집단별 편향과 회수 가능 점수  (양수 = 그 집단에서 과대예측)")
print("=" * 112)


def groups(sub):
    n = sub["asof_pitcher_n"].values
    bn = sub["asof_batter_n"].values
    return {
        "game_type": np.where(sub.is_F.values == 1, "F", "R"),
        "월": sub["game_month"].values,
        "이닝": np.clip(sub["inning"].values, 1, 10),
        "볼카운트": (sub["balls_before"].astype(str) + "-" + sub["strikes_before"].astype(str)).values,
        "투수손x타자손": (sub["pitcher_hand"].astype(str) + "x" + sub["batter_hand"].astype(str)).values,
        "투수경험": pd.cut(n, [-1, 0, 50, 200, 1000, 3000, 10 ** 9],
                       labels=["0", "1-50", "51-200", "201-1k", "1k-3k", "3k+"]).astype(str),
        "타자경험": pd.cut(bn, [-1, 0, 200, 1000, 3000, 10 ** 9],
                       labels=["0", "1-200", "201-1k", "1k-3k", "3k+"]).astype(str),
        "예측10분위": pd.qcut(np.arange(len(sub)), 1, labels=False),  # placeholder
    }


rows = []
for Y in RES:
    p, sub = RES[Y]
    N = len(sub)
    r_all = sub[TARGET].mean()
    BASE = r_all * (1 - r_all)
    gs = groups(sub)
    gs["예측10분위"] = pd.qcut(p, 10, labels=False, duplicates="drop")
    for gname, gv in gs.items():
        t = pd.DataFrame({"g": gv, "p": p, "y": sub[TARGET].values})
        a = t.groupby("g").agg(n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
        a["bias"] = a["pred"] - a["obs"]
        a["회수점수"] = 1e5 * (a["n"] / N) * a["bias"] ** 2 / BASE
        for k, v in a.iterrows():
            rows.append(dict(fold=Y, 축=gname, 그룹=str(k), n=int(v["n"]),
                             예측=v["pred"], 실제=v["obs"], 편향=v["bias"],
                             회수점수=v["회수점수"]))
tab = pd.DataFrame(rows)

for gname in tab["축"].unique():
    t = tab[tab["축"] == gname]
    piv = t.pivot_table(index="그룹", columns="fold", values=["n", "편향", "회수점수"])
    tot = t.groupby("fold")["회수점수"].sum()
    stable = (piv[("편향", 2023)] * piv[("편향", 2024)] > 0)
    print(f"\n--- {gname} --- (축 전체 회수가능: 2023 {tot.get(2023,0):.1f}점 / 2024 {tot.get(2024,0):.1f}점)")
    out = piv.copy()
    out[("부호일치", "")] = np.where(stable, "O", "")
    print(out.to_string(float_format=lambda x: f"{x:.4f}"))

tab.to_csv("cal_subgroup.csv", index=False, encoding="utf-8-sig")
print("\n저장: cal_subgroup.csv")
