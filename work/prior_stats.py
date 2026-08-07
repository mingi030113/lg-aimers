"""'직전 시즌까지'(as-of previous seasons) 선수 집계 피처.

왜 이 설계인가:
  - 주최측 asof_* 는 시즌 내 진행분까지 포함하지만 '분할(split)'이 없다.
  - 우리가 만드는 분할 통계(좌/우타 상대, 볼카운트 구간, 주자 유무 등)를 test(2025)에
    적용하려면 2024년말 시점 값밖에 못 쓴다(평가데이터 내부 집계 금지).
  - 따라서 train/test 모두 '해당 시즌 이전까지'로 통일해 계산한다. 누수 없음 + 규정 준수.
  - 시즌마다 리그 평균이 다르므로 각 시즌 기여분을 리그평균 대비 편차로 정규화한다.
"""
import numpy as np
import pandas as pd

TARGET = "control_success"


def _cnt_group(b, s):
    """볼카운트 구간: 0=초구, 1=투수유리, 2=중립, 3=타자유리, 4=3볼"""
    g = np.full(len(b), 2, dtype=np.int8)
    g[(b == 0) & (s == 0)] = 0
    g[(s > b) & ~((b == 0) & (s == 0))] = 1
    g[(b > s) & (b < 3)] = 3
    g[b == 3] = 4
    return g


def _agg(df, keys, name, league):
    """(keys) 별로 시즌별 (n, 리그대비 성공 초과분 합) 집계.

    리그평균은 (시즌, game_type) 기준이라 시즌 드리프트와 F/R 레벨차가 모두 제거된다.
    """
    x = df[keys + ["season"]].copy()
    lg = league.reindex(pd.MultiIndex.from_arrays([df["season"].values, df["_gt"].values])).values
    x["dev"] = df[TARGET].values - lg
    g = x.groupby(keys + ["season"], observed=True).agg(n=("dev", "size"), sdev=("dev", "sum"))
    g.name = name
    return g


def _expand_to_prior(g, keys, seasons, decay=0.75):
    """시즌별 집계를 '직전 시즌까지 누적'(지수감쇠 가중)으로 변환."""
    out = []
    for S in seasons:
        sub = g[g.index.get_level_values("season") < S]
        if len(sub) == 0:
            continue
        w = decay ** (S - 1 - sub.index.get_level_values("season").values)
        tmp = pd.DataFrame({"n": sub["n"].values * w, "sdev": sub["sdev"].values * w},
                           index=sub.index.droplevel("season"))
        tmp = tmp.groupby(level=list(range(len(keys)))).sum()
        tmp["season"] = S
        out.append(tmp.reset_index())
    if not out:
        return pd.DataFrame(columns=keys + ["n", "sdev", "season"])
    return pd.concat(out, ignore_index=True)


def build_prior(train_df, target_seasons, decay=0.75, k_shrink=(400, 400, 800, 300, 200)):
    """train_df(라벨 보유)로부터 target_seasons 각 시즌용 prior 테이블 생성.

    반환: dict of DataFrame, 각각 key + season 으로 merge 가능.
    값은 모두 '리그평균 대비 편차(수축 적용)' 이므로 시즌 드리프트에 중립적이다.
    """
    # 2022년 이전 F(퓨처스)는 라벨 레짐이 달라 집계에서 제외한다.
    isF = (train_df["game_type"] == "F")
    df = train_df[~(isF & (train_df["season"] <= 2022))].copy()
    # 리그 평균은 (시즌 x game_type) 별로. F/R 레벨이 크게 다르다.
    df["_gt"] = isF.reindex(df.index).astype(np.int8)
    league = df.groupby(["season", "_gt"])[TARGET].mean()

    df["cnt_g"] = _cnt_group(df["balls_before"].values, df["strikes_before"].values)
    df["runner_any"] = (df["num_runners_on"] > 0).astype(np.int8)
    df["inn_g"] = np.clip(df["inning"].values, 1, 9).astype(np.int8)

    specs = [
        (["pitcher_id"], "p", k_shrink[0]),
        (["pitcher_id", "batter_hand"], "p_bh", k_shrink[1]),
        (["pitcher_id", "cnt_g"], "p_cnt", k_shrink[2]),
        (["batter_id"], "b", k_shrink[3]),
        (["batter_id", "pitcher_hand"], "b_ph", k_shrink[4]),
        (["pitcher_team_id"], "pteam", 2000),
        (["pitcher_id", "runner_any"], "p_run", 800),
    ]
    tables = {}
    for keys, name, k in specs:
        g = _agg(df, keys, name, league)
        pri = _expand_to_prior(g, keys, target_seasons, decay=decay)
        pri[f"{name}_n"] = pri["n"]
        pri[f"{name}_dev"] = pri["sdev"] / (pri["n"] + k)   # 경험적 베이즈 수축
        tables[name] = pri[keys + ["season", f"{name}_n", f"{name}_dev"]]
    return tables, league


def apply_prior(df, tables):
    d = df.copy()
    d["cnt_g"] = _cnt_group(d["balls_before"].values, d["strikes_before"].values)
    d["runner_any"] = (d["num_runners_on"] > 0).astype(np.int8)
    for name, tab in tables.items():
        keys = [c for c in tab.columns if not c.startswith(name + "_") and c != "season"]
        d = d.merge(tab, on=keys + ["season"], how="left")
    for name in tables:
        d[f"{name}_n"] = d[f"{name}_n"].fillna(0.0)
        d[f"{name}_dev"] = d[f"{name}_dev"].fillna(0.0)
        d[f"{name}_logn"] = np.log1p(d[f"{name}_n"])
    # 분할 - 전체 (순수 스플릿 효과)
    d["p_bh_split"] = d["p_bh_dev"] - d["p_dev"]
    d["p_cnt_split"] = d["p_cnt_dev"] - d["p_dev"]
    d["p_run_split"] = d["p_run_dev"] - d["p_dev"]
    d["b_ph_split"] = d["b_ph_dev"] - d["b_dev"]
    d = d.drop(columns=["cnt_g", "runner_any"])
    return d
