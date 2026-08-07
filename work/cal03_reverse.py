"""볼카운트 재보정의 역방향 검증(2024 -> 2023) + 보정량 자체 확인.

cal02 에서 2023->2024 로 +11.1 이 나왔다. 한 방향 1회 관측이라
반대 방향도 이득이어야 '연도에 안정적인 구조적 편향'이라고 말할 수 있다.
cal_z.npz 에 저장된 예측을 재사용하므로 재학습 없이 즉시 확인 가능.
"""
import numpy as np
import pandas as pd

from features import build, feature_list
from train_final import sigmoid, solve_shift, TARGET

pd.set_option("display.width", 250)

Z = np.load("cal_z.npz")
raw = pd.read_parquet("train.parquet")
d = build(raw)
d["season"] = raw["season"].values
d[TARGET] = raw[TARGET].values
del raw


def fold(Y):
    ev = ((d.season == Y) & (d.is_F == 0)).values
    sub = d.loc[ev].reset_index(drop=True)
    z = Z[str(Y)][ev]
    y = sub[TARGET].values
    z = z + solve_shift(z, y.mean())          # 전역 레벨 제거
    g = (sub["balls_before"].astype(str) + "-" + sub["strikes_before"].astype(str)).values
    return z, y, g


def score(y, p):
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def learn(z, y, g):
    out = {}
    for k in np.unique(g):
        m = g == k
        if m.sum() >= 300:
            out[k] = solve_shift(z[m], y[m].mean())
    return out


def apply_(z, y, g, corr):
    dz = np.array([corr.get(k, 0.0) for k in g])
    z2 = z + dz
    return sigmoid(z2 + solve_shift(z2, y.mean()))


F = {Y: fold(Y) for Y in (2023, 2024)}
C = {Y: learn(*F[Y]) for Y in (2023, 2024)}

print("=" * 90)
print("### 볼카운트 보정량 (로짓, 양수 = 예측을 올려야 함) 과 연도간 일관성")
print("=" * 90)
keys = sorted(set(C[2023]) | set(C[2024]), key=lambda k: (int(k[0]), int(k[2])))
rows = []
for k in keys:
    z, y, g = F[2024]
    m = g == k
    rows.append(dict(카운트=k, n_2024=int(m.sum()),
                     보정_2023=C[2023].get(k, np.nan), 보정_2024=C[2024].get(k, np.nan)))
t = pd.DataFrame(rows)
t["부호일치"] = np.where(t.보정_2023 * t.보정_2024 > 0, "O", "")
print(t.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print(f"\n  상관계수 = {t.보정_2023.corr(t.보정_2024):.3f}   부호일치 {(t.부호일치=='O').sum()}/{len(t)}")

print("\n" + "=" * 90)
print("### 양방향 out-of-sample 이득")
print("=" * 90)
for src, dst in [(2023, 2024), (2024, 2023)]:
    z, y, g = F[dst]
    base = score(y, sigmoid(z))
    oos = score(y, apply_(z, y, g, C[src]))
    ins = score(y, apply_(z, y, g, C[dst]))
    print(f"  {src} 에서 학습 -> {dst} 적용:  기준 {base:7.1f}  ->  {oos:7.1f}  "
          f"(Δ{oos-base:+.1f})    [in-sample 상한 {ins-base:+.1f}]")

print("\n" + "=" * 90)
print("### 두 폴드 평균 보정량을 쓰면? (실제 제출에 넣을 형태)")
print("=" * 90)
avg = {k: np.nanmean([C[2023].get(k, np.nan), C[2024].get(k, np.nan)]) for k in keys}
for dst in (2023, 2024):
    z, y, g = F[dst]
    base = score(y, sigmoid(z))
    v = score(y, apply_(z, y, g, avg))
    print(f"  {dst}R: {base:7.1f} -> {v:7.1f}  (Δ{v-base:+.1f})   [자기 폴드 포함이라 낙관적]")
print("\n  평균 보정량:", {k: round(v, 4) for k, v in avg.items()})
