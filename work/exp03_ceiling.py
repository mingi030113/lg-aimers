"""실험 3: (a) 성능 상한 추정  (b) 시즌 평균 외삽 정책 백테스트."""
import numpy as np
import pandas as pd
from common import TARGET, score_bss, decompose, logit, sigmoid

pd.set_option("display.width", 250)
t = pd.read_parquet("train.parquet")
t["is_F"] = (t["game_type"] == "F").astype(np.int8)

print("=" * 100)
print("### (a) 오라클 상한: 2024 정답을 알고 있을 때 각 정보원의 BSS")
print("=" * 100)
ev = t[t.season == 2024]
y = ev[TARGET].values
r = y.mean()

def oracle(keys, name, shrink=0):
    g = ev.groupby(keys)[TARGET].agg(["count", "mean"])
    p = ev[keys].merge(g, left_on=keys, right_index=True, how="left") if isinstance(keys, list) else None
    idx = pd.MultiIndex.from_frame(ev[keys]) if isinstance(keys, list) else ev[keys]
    m = g["mean"].reindex(idx).values
    n = g["count"].reindex(idx).values
    if shrink:
        m = (m * n + r * shrink) / (n + shrink)
    m = np.where(np.isnan(m), r, m)
    print(f"  {name:52s} score={score_bss(y, m):8.1f}")

oracle("pitcher_id", "투수 2024 실제평균 (in-sample, 과적합 포함)")
oracle("pitcher_id", "투수 2024 실제평균 (shrink k=300)", shrink=300)
oracle("batter_id", "타자 2024 실제평균 (shrink k=300)", shrink=300)
oracle(["pitcher_id", "batter_hand"], "투수x타자손 (shrink k=300)", shrink=300)
oracle(["pitcher_id", "balls_before", "strikes_before"], "투수x볼카운트 (shrink k=300)", shrink=300)

# 홀드아웃 방식 오라클: 2024를 반으로 갈라 전반기로 후반기 예측
ev2 = ev.sort_values("row_id")
h = len(ev2) // 2
first, second = ev2.iloc[:h], ev2.iloc[h:]
y2 = second[TARGET].values
r1 = first[TARGET].mean()
for k in [0, 100, 300, 1000, 3000]:
    g = first.groupby("pitcher_id")[TARGET].agg(["count", "mean"])
    m = g["mean"].reindex(second["pitcher_id"]).values
    n = g["count"].reindex(second["pitcher_id"]).fillna(0).values
    p = np.where(np.isnan(m), r1, (np.nan_to_num(m) * n + r1 * k) / (n + k))
    print(f"  [정직] 2024전반기 투수평균(shrink {k:4d}) -> 후반기       score={score_bss(y2, p):8.1f}")

print("\n" + "=" * 100)
print("### (b) 시즌 평균 외삽 정책 백테스트 (train<=Y-1 로 Y의 평균 맞히기)")
print("=" * 100)
# R 게임만 / 전체 두 기준 모두
for sub, lab in [(t, "전체"), (t[t.is_F == 0], "R만")]:
    ss = sub.groupby("season")[TARGET].mean()
    print(f"\n  --- {lab} 시즌평균: " + "  ".join(f"{y_}:{v:.4f}" for y_, v in ss.items()))
    for Y in [2021, 2022, 2023, 2024]:
        hist = ss[ss.index <= Y - 1]
        true = ss[Y]
        outs = {"직전": hist.iloc[-1],
                "직전2평균": hist.iloc[-2:].mean(),
                "선형2": np.polyval(np.polyfit(hist.index[-2:], hist.values[-2:], 1), Y),
                "선형3": np.polyval(np.polyfit(hist.index[-3:], hist.values[-3:], 1), Y),
                "선형4": np.polyval(np.polyfit(hist.index[-4:], hist.values[-4:], 1), Y) if len(hist) >= 4 else np.nan,
                "직전+0.5*직전변화": hist.iloc[-1] + 0.5 * (hist.iloc[-1] - hist.iloc[-2]),
                }
        s = "  ".join(f"{k}={v:.4f}({(v-true)*1000:+5.1f}‰)" for k, v in outs.items() if not np.isnan(v))
        print(f"    {Y} true={true:.4f} | {s}")

print("\n  === 각 정책의 4년 평균 제곱오차 -> 예상 점수손실 (r=0.48) ===")
for sub, lab in [(t, "전체"), (t[t.is_F == 0], "R만")]:
    ss = sub.groupby("season")[TARGET].mean()
    acc = {}
    for Y in [2021, 2022, 2023, 2024]:
        hist = ss[ss.index <= Y - 1]
        true = ss[Y]
        outs = {"직전": hist.iloc[-1],
                "직전2평균": hist.iloc[-2:].mean(),
                "선형2": np.polyval(np.polyfit(hist.index[-2:], hist.values[-2:], 1), Y),
                "선형3": np.polyval(np.polyfit(hist.index[-3:], hist.values[-3:], 1), Y),
                "직전+0.5*직전변화": hist.iloc[-1] + 0.5 * (hist.iloc[-1] - hist.iloc[-2]),
                "직전-0.010": hist.iloc[-1] - 0.010,
                "직전-0.015": hist.iloc[-1] - 0.015,
                }
        for k, v in outs.items():
            acc.setdefault(k, []).append((v - true) ** 2)
    print(f"  [{lab}]")
    for k, v in sorted(acc.items(), key=lambda x: np.mean(x[1])):
        mse = np.mean(v)
        print(f"    {k:20s} RMSE={np.sqrt(mse):.4f}  기대점수손실={-100000*mse/(0.48*0.52):8.1f}")

print("\n  === 2025 예측값 (train 2019-2024 전체 사용) ===")
for sub, lab in [(t, "전체"), (t[t.is_F == 0], "R만"), (t[t.is_F == 1], "F만")]:
    ss = sub.groupby("season")[TARGET].mean()
    print(f"  [{lab}] 직전={ss.iloc[-1]:.4f}  선형2={np.polyval(np.polyfit(ss.index[-2:], ss.values[-2:],1),2025):.4f}  "
          f"선형3={np.polyval(np.polyfit(ss.index[-3:], ss.values[-3:],1),2025):.4f}  "
          f"선형4={np.polyval(np.polyfit(ss.index[-4:], ss.values[-4:],1),2025):.4f}  "
          f"직전+0.5변화={ss.iloc[-1]+0.5*(ss.iloc[-1]-ss.iloc[-2]):.4f}")
