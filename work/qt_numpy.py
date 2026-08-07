"""QuantileTransformer(output_distribution='normal') 의 numpy 재구현.

왜 필요한가: MLP2 는 분위수변환 입력을 쓴다. 추론 시 sklearn 객체를 pickle 하면
평가서버(sklearn 1.8.0)와 로컬(1.9.0) 버전 불일치 위험이 있다.
-> quantiles_ / references_ 만 npz 로 저장하고 변환을 직접 계산한다.
   (로지스틱·MLP 가중치를 npz 로 저장한 것과 같은 원칙)

scipy 는 평가서버 기본 설치(1.15.3)라 ndtri 사용 가능.
"""
import numpy as np
from scipy.special import ndtri

BOUNDS_THRESHOLD = 1e-7


def qt_transform(X, quantiles, references):
    """sklearn QuantileTransformer.transform(output_distribution='normal') 과 동일.

    quantiles: (n_quantiles, n_features), references: (n_quantiles,)
    """
    X = np.asarray(X, dtype=np.float64).copy()
    clip_min = ndtri(BOUNDS_THRESHOLD - np.spacing(1))
    clip_max = ndtri(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
    for i in range(X.shape[1]):
        q = quantiles[:, i]
        col = X[:, i]
        lo_idx = col == q[0]
        hi_idx = col == q[-1]
        finite = ~np.isnan(col)
        cf = col[finite]
        # sklearn 은 정방향/역방향 보간의 평균을 쓴다 (동값 구간 처리)
        col[finite] = 0.5 * (np.interp(cf, q, references)
                             - np.interp(-cf, -q[::-1], -references[::-1]))
        col[hi_idx] = 1.0
        col[lo_idx] = 0.0
        col = ndtri(col)
        X[:, i] = np.clip(col, clip_min, clip_max)
    return X


if __name__ == "__main__":
    # sklearn 과 정확히 일치하는지 검증
    import pandas as pd
    from sklearn.preprocessing import QuantileTransformer
    from features import build, feature_list, attach_trackman

    raw = pd.read_parquet("train.parquet")
    idmap = pd.read_parquet("idmap_pitcher_v2.parquet")
    sel = (idmap[idmap.share > 0.5].sort_values("votes", ascending=False)
           .drop_duplicates("tm_pitcher_id", keep="first"))
    t2p = {v: k for k, v in zip(sel.pitcher_id, sel.tm_pitcher_id)}
    TMP = pd.read_parquet("tm_prior.parquet")
    TMP["pitcher_id"] = TMP["pitcher_trackman_id"].map(t2p)
    TMP = TMP.dropna(subset=["pitcher_id"]).drop(columns=["pitcher_trackman_id"])
    TMP["pitcher_id"] = TMP["pitcher_id"].astype(int)
    raw = attach_trackman(raw, TMP)
    d = build(raw); d["season"] = raw["season"].values
    F = [c for c in feature_list(d, use_season=False, use_ids=False) if c != "control_success"]
    tr = ((d.season >= 2023) & ~((d.is_F == 1) & (d.season <= 2022))).values
    X = d.loc[tr, F]
    med = X.median()
    A = X.fillna(med).to_numpy(np.float64)
    q = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(A)
    ref = q.transform(A[:50000])
    mine = qt_transform(A[:50000], q.quantiles_, q.references_)
    print(f"sklearn vs numpy 재구현 최대 절대오차 = {np.abs(ref - mine).max():.3e}")
    print(f"quantiles_ shape {q.quantiles_.shape}  references_ shape {q.references_.shape}")
    print(f"npz 예상 크기 ≈ {(q.quantiles_.size + q.references_.size) * 8 / 1e6:.2f} MB")
