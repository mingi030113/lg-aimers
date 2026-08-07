"""공통 유틸: 전방검증(forward validation) 하네스 + 점수 함수."""
import numpy as np
import pandas as pd

TARGET = "control_success"
ID = "row_id"


def score_bss(y, p):
    """대회 점수. Score = max(0, 100000*(1 - BS / (r(1-r))))  (음수도 보이게 clip 안 함)"""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    r = y.mean()
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1.0 - r)))


def decompose(y, p):
    """점수를 '레벨(평균 오프셋)' 손실과 '변별력' 이득으로 분해."""
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    r = y.mean()
    base = r * (1 - r)
    d = p.mean() - r
    # p를 평균만 정답으로 옮긴 버전
    p_centered = p - d
    s_all = 100000 * (1 - np.mean((p - y) ** 2) / base)
    s_cal = 100000 * (1 - np.mean((p_centered - y) ** 2) / base)
    return dict(score=s_all, score_if_mean_fixed=s_cal,
                level_loss=s_all - s_cal, mean_offset=d, pred_mean=p.mean(), true_mean=r)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load():
    return pd.read_parquet("train.parquet")
