# -*- coding: utf-8 -*-
"""제출용 추론 스크립트 (bundle.py 가 features.py 를 인라인해 생성).

흐름
  data/test.csv, data/sample_submission.csv 읽기
  -> build()  (학습과 완전히 동일한 함수)
  -> LGBM 5개 raw_score 평균 + L2 로지스틱 decision_function 을 블렌드
  -> game_type 별로 미리 계산된 로짓 시프트 적용 (2025 리그 레벨 통제)
  -> sigmoid -> output/submission.csv

평가데이터 내부 집계·행순서 기반 피처를 일절 쓰지 않는다 (완전한 행 단위 독립 추론).
"""
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

ID_COL = "row_id"
TARGET_COL = "control_success"

TEST_DIR = "./data"
MODEL_DIR = "./model"
OUT_DIR = "./output"

# ===== features.py 인라인 시작 =====
__FEATURES_SOURCE__
# ===== features.py 인라인 끝 =====


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    with open(os.path.join(MODEL_DIR, "meta.json"), "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    feats = meta["features"]
    w = meta["blend_w"]
    shift = meta["shift"]
    print(f"meta: {len(feats)} feats, {meta['n_seeds']} boosters, blend_w={w}, "
          f"shift R={shift['R']:+.4f} F={shift['F']:+.4f}, "
          f"expected 2025 R={meta['expected_2025']['R']:.4f} F={meta['expected_2025']['F']:.4f}")

    test = pd.read_csv(os.path.join(TEST_DIR, "test.csv"), encoding="utf-8-sig")
    sub = pd.read_csv(os.path.join(TEST_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    print(f"test={len(test)}  sample_submission={len(sub)}")

    tm_path = os.path.join(MODEL_DIR, "tm_prior.csv")
    if meta.get("use_trackman") and os.path.exists(tm_path):
        prior = pd.read_csv(tm_path)
        test = attach_trackman(test, prior)
        print(f"trackman prior 부착: 커버리지 {test['tm_prior_n'].notna().mean()*100:.1f}%")

    if meta.get("use_prior"):
        tables = {nm: pd.read_csv(os.path.join(MODEL_DIR, f"prior_{nm}.csv"))
                  for nm in meta["prior_tables"]}
        test = apply_prior(test, tables)
        print(f"prior 분할통계 부착: {len(tables)}개 테이블, "
              f"투수 매칭률 {(test['p_n'] > 0).mean()*100:.1f}%")

    d = build(test)
    X = d.reindex(columns=feats)
    print(f"feature matrix {X.shape}")

    z_lgb = np.zeros(len(X), dtype=np.float64)
    for s in range(meta["n_seeds"]):
        b = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lgb_{s}.txt"))
        z_lgb += b.predict(X, raw_score=True)
    z_lgb /= meta["n_seeds"]

    lin = np.load(os.path.join(MODEL_DIR, "linear.npz"))
    A = X.to_numpy(np.float64)
    nan = np.isnan(A)
    if nan.any():
        A[nan] = np.take(lin["med"], np.where(nan)[1])
    z_lin = ((A - lin["mu"]) / lin["sd"]) @ lin["coef"] + lin["intercept"][0]

    z = (1 - w) * z_lgb + w * z_lin

    if meta.get("use_mlp"):
        # 학습 때 저장한 가중치로 numpy 순전파 (sklearn 버전 의존성 제거)
        mp = np.load(os.path.join(MODEL_DIR, "mlp.npz"))
        ns, nl = int(mp["n_seeds"][0]), int(mp["n_layers"][0])
        Bn = (A - lin["mu"]) / lin["sd"]
        acc = np.zeros(len(Bn), dtype=np.float64)
        for i in range(ns):
            h = Bn
            for j in range(nl - 1):
                h = np.maximum(h @ mp[f"s{i}_W{j}"] + mp[f"s{i}_b{j}"], 0.0)   # relu
            acc += (h @ mp[f"s{i}_W{nl-1}"] + mp[f"s{i}_b{nl-1}"]).ravel()
        z_mlp = acc / ns
        mw = meta["mlp_w"]
        z_mlp2, mw2 = 0.0, 0.0
        if meta.get("use_mlp2"):
            m2 = np.load(os.path.join(MODEL_DIR, "mlp2.npz"))
            ns2, nl2 = int(m2["n_seeds"][0]), int(m2["n_layers"][0])
            Bq = qt_transform(A, m2["quantiles"], m2["references"])
            a2 = np.zeros(len(Bq), dtype=np.float64)
            for i in range(ns2):
                h = Bq
                for j in range(nl2 - 1):
                    h = np.maximum(h @ m2[f"s{i}_W{j}"] + m2[f"s{i}_b{j}"], 0.0)
                a2 += (h @ m2[f"s{i}_W{nl2-1}"] + m2[f"s{i}_b{nl2-1}"]).ravel()
            z_mlp2 = a2 / ns2
            mw2 = meta["mlp2_w"]
        z_mlp3, mw3 = 0.0, 0.0
        if meta.get("use_mlp3"):
            m3 = np.load(os.path.join(MODEL_DIR, "mlp3.npz"))
            ns3, nl3 = int(m3["n_seeds"][0]), int(m3["n_layers"][0])
            Bs = Bn[:, m3["feat_idx"]]
            a3 = np.zeros(len(Bs), dtype=np.float64)
            for i in range(ns3):
                h = Bs
                for j in range(nl3 - 1):
                    h = np.maximum(h @ m3[f"s{i}_W{j}"] + m3[f"s{i}_b{j}"], 0.0)
                a3 += (h @ m3[f"s{i}_W{nl3-1}"] + m3[f"s{i}_b{nl3-1}"]).ravel()
            z_mlp3 = a3 / ns3
            mw3 = meta["mlp3_w"]
        z = (1 - mw - mw2 - mw3) * z + mw * z_mlp + mw2 * z_mlp2 + mw3 * z_mlp3
        print(f"MLP 블렌드: MLP1 {mw} / MLP2 {mw2} / MLP3 {mw3}, base {1-mw-mw2-mw3:.2f}")
    cc = meta.get("count_cal") or {}
    if cc:
        z = z + np.array([cc.get(str(int(v)), 0.0) for v in d["cnt_diff"].to_numpy()])
        print(f"볼카운트(b−s) 보정 적용: {cc}")
    z = z + np.where(d["is_F"].to_numpy() == 1, shift["F"], shift["R"])
    p = np.clip(_sigmoid(z), 1e-6, 1 - 1e-6)
    print(f"pred mean={p.mean():.5f}  sd={p.std():.5f}  min={p.min():.4f}  max={p.max():.4f}")

    pred = dict(zip(test[ID_COL].to_numpy(), p))
    fallback = 0.5 * (meta["expected_2025"]["R"] + meta["expected_2025"]["F"])
    out, miss = [], 0
    for rid in sub[ID_COL].to_numpy():
        v = pred.get(rid)
        if v is None:
            miss += 1
            out.append(fallback)
        else:
            out.append(v)
    if miss:
        print(f"warning: {miss} row_id missing from test.csv")
    sub[TARGET_COL] = out

    os.makedirs(OUT_DIR, exist_ok=True)
    sub.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False, encoding="utf-8")
    print(f"saved {OUT_DIR}/submission.csv rows={len(sub)}")


if __name__ == "__main__":
    main()
