"""현재 배포 recipe와 동일한 경로를 재현하는 rolling validator.

기본 실행은 학습을 하지 않는 dry-run이다. 실제 LGBM/Logistic/MLP1/MLP2
재학습은 --fit을 명시했을 때만 수행한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK_DIR = HERE.parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from features import attach_trackman, build, feature_list  # noqa: E402
from recipe import (  # noqa: E402
    RecipeError,
    apply_count_calibration,
    audit_model_meta,
    audit_train_script,
    blend_logits,
    component_weights,
    fit_group_shifts,
    load_recipe,
    make_fold_masks,
    score_summary,
    sigmoid,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        return frame.head(max_rows).copy() if max_rows else frame
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, encoding="utf-8-sig", nrows=max_rows)
    raise RecipeError(f"지원하지 않는 입력 형식: {path}")


def prepare_trackman_prior(
    prior_path: str | Path, pitcher_map_path: str | Path | None
) -> pd.DataFrame:
    prior = load_table(prior_path)
    if "pitcher_id" in prior.columns:
        required = {"pitcher_id", "season"}
        if not required.issubset(prior.columns):
            raise RecipeError(f"Trackman prior 열 누락: {sorted(required-set(prior.columns))}")
        return prior

    if "pitcher_trackman_id" not in prior.columns:
        raise RecipeError("Trackman prior에 pitcher_id 또는 pitcher_trackman_id가 필요합니다")
    if pitcher_map_path is None:
        raise RecipeError("pitcher_trackman_id prior에는 --pitcher-map이 필요합니다")
    idmap = load_table(pitcher_map_path)
    required = {"pitcher_id", "tm_pitcher_id", "share", "votes"}
    if not required.issubset(idmap.columns):
        raise RecipeError(f"pitcher map 열 누락: {sorted(required-set(idmap.columns))}")
    selected = (
        idmap[idmap["share"] > 0.5]
        .sort_values("votes", ascending=False)
        .drop_duplicates("tm_pitcher_id", keep="first")
    )
    trackman_to_pitcher = dict(zip(selected["tm_pitcher_id"], selected["pitcher_id"]))
    prior = prior.copy()
    prior["pitcher_id"] = prior["pitcher_trackman_id"].map(trackman_to_pitcher)
    prior = prior.dropna(subset=["pitcher_id"]).drop(columns=["pitcher_trackman_id"])
    prior["pitcher_id"] = prior["pitcher_id"].astype(int)
    if prior.duplicated(["pitcher_id", "season"]).any():
        raise RecipeError("매핑 후 Trackman prior의 (pitcher_id, season)이 중복됩니다")
    return prior


def prepare_features(
    raw: pd.DataFrame,
    recipe: dict[str, Any],
    trackman_prior: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    target = str(recipe["target"])
    required = {"season", "game_type", target}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RecipeError(f"학습 데이터 필수 열 누락: {missing}")
    source = raw
    if recipe["trackman"]["enabled"]:
        if trackman_prior is None:
            raise RecipeError("Trackman 활성 recipe에는 prior 입력이 필요합니다")
        source = attach_trackman(source, trackman_prior)
    d = build(source)
    d["season"] = source["season"].to_numpy()
    d[target] = source[target].to_numpy()
    feats = [c for c in feature_list(d, use_season=False, use_ids=False) if c != target]
    if len(feats) != len(set(feats)):
        raise RecipeError("중복 feature 이름이 있습니다")
    return d, feats


def _forward_mlp(model: Any, values: np.ndarray) -> np.ndarray:
    hidden = values
    for weight, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        hidden = np.maximum(hidden @ weight + bias, 0.0)
    return (hidden @ model.coefs_[-1] + model.intercepts_[-1]).ravel()


def _train_mlp_ensemble(
    train_x: np.ndarray,
    pred_x: np.ndarray,
    y: np.ndarray,
    settings: dict[str, Any],
    n_seeds: int,
) -> np.ndarray:
    from sklearn.neural_network import MLPClassifier

    logits = np.zeros(len(pred_x), dtype=np.float64)
    for seed in range(n_seeds):
        model = MLPClassifier(
            hidden_layer_sizes=tuple(settings["hidden"]),
            alpha=float(settings["alpha"]),
            batch_size=int(settings["batch_size"]),
            learning_rate_init=float(settings["learning_rate_init"]),
            max_iter=int(settings["max_iter"]),
            early_stopping=True,
            n_iter_no_change=int(settings["n_iter_no_change"]),
            validation_fraction=float(settings["validation_fraction"]),
            random_state=seed,
        )
        model.fit(train_x, y)
        logits += _forward_mlp(model, pred_x)
    return logits / n_seeds


def _sample_weights(
    seasons: np.ndarray, train_end_year: int, recipe: dict[str, Any]
) -> np.ndarray | None:
    half_life = recipe["training"].get("decay_halflife")
    if half_life is None:
        return None
    return 0.5 ** ((train_end_year - seasons.astype(float)) / float(half_life))


def fit_fold(
    d: pd.DataFrame,
    feats: list[str],
    recipe: dict[str, Any],
    eval_year: int,
    quick: bool = False,
) -> dict[str, Any]:
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import QuantileTransformer

    masks = make_fold_masks(d, recipe, eval_year)
    train = masks["train"]
    predict = masks["calibration"] | masks["evaluation"]
    if int(train.sum()) == 0 or int(masks["evaluation"].sum()) == 0:
        raise RecipeError(f"{eval_year} fold에 학습 또는 평가 행이 없습니다")

    target = str(recipe["target"])
    x_train = d.loc[train, feats]
    x_pred = d.loc[predict, feats]
    y_train = d.loc[train, target].to_numpy(dtype=np.float64)
    y_pred = d.loc[predict, target].to_numpy(dtype=np.float64)
    settings = recipe["models"]
    weights = _sample_weights(
        d.loc[train, "season"].to_numpy(), eval_year - 1, recipe
    )

    lgb_settings = settings["lgbm"]
    n_lgb = 1 if quick else int(lgb_settings["n_seeds"])
    z_lgb = np.zeros(len(x_pred), dtype=np.float64)
    for seed in range(n_lgb):
        params = dict(lgb_settings["params"])
        params.update(
            seed=seed,
            bagging_seed=seed,
            feature_fraction_seed=seed,
            data_random_seed=seed,
        )
        model = lgb.train(
            params,
            lgb.Dataset(x_train, y_train, weight=weights),
            num_boost_round=int(lgb_settings["rounds"]),
        )
        z_lgb += model.predict(x_pred, raw_score=True)
    z_lgb /= n_lgb

    median = x_train.median()
    a = x_train.fillna(median).to_numpy(dtype=np.float64)
    b = x_pred.fillna(median).to_numpy(dtype=np.float64)
    mean, std = a.mean(axis=0), a.std(axis=0)
    std[std == 0] = 1.0
    a_norm, b_norm = (a - mean) / std, (b - mean) / std
    log_settings = settings["logistic"]
    logistic = LogisticRegression(
        C=float(log_settings["C"]),
        max_iter=int(log_settings["max_iter"]),
        solver="lbfgs",
    )
    logistic.fit(a_norm, y_train, sample_weight=weights)
    z_logistic = b_norm @ logistic.coef_[0] + logistic.intercept_[0]

    n_mlp = 1 if quick else int(settings["mlp1"]["n_seeds"])
    z_mlp1 = None
    if settings["mlp1"]["enabled"]:
        z_mlp1 = _train_mlp_ensemble(
            a_norm, b_norm, y_train, settings["mlp1"], n_mlp
        )

    z_mlp2 = None
    if settings["mlp2"]["enabled"]:
        # scipy/sklearn은 실제 --fit 경로에서만 필요하다. 정적 dry-run은 배포
        # 환경의 선택적 학습 의존성 없이 실행될 수 있어야 한다.
        from qt_numpy import qt_transform

        n_quantiles = min(int(settings["mlp2"]["n_quantiles"]), len(a))
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=0,
        ).fit(a)
        a_quantile = transformer.transform(a)
        b_quantile = qt_transform(b, transformer.quantiles_, transformer.references_)
        z_mlp2 = _train_mlp_ensemble(
            a_quantile, b_quantile, y_train, settings["mlp1"], n_mlp
        )

    z = blend_logits(recipe, z_lgb, z_logistic, z_mlp1, z_mlp2)
    d_pred = d.loc[predict].copy().reset_index(drop=True)
    z = apply_count_calibration(z, d_pred["cnt_diff"].to_numpy(), recipe)
    shifts = fit_group_shifts(d_pred, z, recipe, eval_year - 1)

    is_f = d_pred["is_F"].to_numpy() == 1
    is_eval = d_pred["season"].to_numpy() == eval_year
    probability = np.full(len(d_pred), np.nan, dtype=np.float64)
    clip = float(recipe["calibration"]["probability_clip"])
    for label, group_f in (("R", False), ("F", True)):
        info = shifts[label]
        if info is None:
            continue
        selected = is_eval & (is_f == group_f)
        probability[selected] = np.clip(
            sigmoid(z[selected] + float(info["shift"])), clip, 1.0 - clip
        )

    metrics: dict[str, Any] = {}
    for label, group_f in (("R", False), ("F", True)):
        selected = is_eval & (is_f == group_f) & np.isfinite(probability)
        if selected.any():
            metrics[label] = score_summary(y_pred[selected], probability[selected])
    selected = is_eval & np.isfinite(probability)
    if selected.any():
        metrics["all"] = score_summary(y_pred[selected], probability[selected])
    return {
        "eval_year": eval_year,
        "quick": quick,
        "train_rows": int(train.sum()),
        "calibration_rows": int(masks["calibration"].sum()),
        "evaluation_rows": int(masks["evaluation"].sum()),
        "shifts": shifts,
        "metrics": metrics,
    }


def _parse_years(raw: str | None, recipe: dict[str, Any]) -> list[int]:
    if raw is None:
        return [int(v) for v in recipe["validation"]["eval_years"]]
    years = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not years:
        raise RecipeError("--eval-years가 비어 있습니다")
    return years


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "recipe_current.json"))
    parser.add_argument("--data", help="train.csv 또는 train.parquet")
    parser.add_argument("--trackman-prior", help="tm_prior.csv/parquet")
    parser.add_argument("--pitcher-map", help="idmap_pitcher_v2.csv/parquet")
    parser.add_argument("--model-meta", help="배포 model/meta.json 정합성 검사")
    parser.add_argument("--train-script", help="train_final.py literal 설정 정합성 검사")
    parser.add_argument("--eval-years", help="예: 2023,2024")
    parser.add_argument("--max-rows", type=int, help="dry-run CSV 입력 행 제한")
    parser.add_argument("--fit", action="store_true", help="실제 rolling 모델 학습 실행")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="--fit에서 LGBM/MLP seed를 각각 1개만 사용(정식 점수 아님)",
    )
    parser.add_argument("--allow-feature-source-mismatch", action="store_true")
    parser.add_argument("--output", help="검사 결과 JSON 저장 경로")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recipe = load_recipe(args.config)
    result: dict[str, Any] = {
        "mode": "fit" if args.fit else "dry-run",
        "recipe": recipe["name"],
        "component_weights": component_weights(recipe),
        "audits": {},
    }

    feature_path = WORK_DIR / "features.py"
    actual_feature_hash = file_sha256(feature_path)
    expected_feature_hash = recipe.get("provenance", {}).get("features_sha256")
    feature_ok = expected_feature_hash is None or actual_feature_hash == expected_feature_hash
    result["audits"]["feature_source"] = {
        "path": str(feature_path),
        "expected_sha256": expected_feature_hash,
        "actual_sha256": actual_feature_hash,
        "ok": feature_ok,
    }
    if not feature_ok and not args.allow_feature_source_mismatch:
        raise RecipeError(
            "features.py가 recipe snapshot과 다릅니다. 의도한 변경이면 "
            "--allow-feature-source-mismatch로 명시하세요"
        )

    meta: dict[str, Any] | None = None
    if args.model_meta:
        with Path(args.model_meta).open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
        problems = audit_model_meta(recipe, meta)
        result["audits"]["model_meta"] = {"ok": not problems, "problems": problems}
        if problems:
            raise RecipeError("model/meta.json 불일치: " + "; ".join(problems))

    if args.train_script:
        problems = audit_train_script(recipe, args.train_script)
        result["audits"]["train_script"] = {"ok": not problems, "problems": problems}
        if problems:
            raise RecipeError("train script 설정 불일치: " + "; ".join(problems))

    d = None
    feats = None
    if args.data:
        raw = load_table(args.data, max_rows=args.max_rows)
        prior = None
        if recipe["trackman"]["enabled"]:
            if not args.trackman_prior:
                raise RecipeError("Trackman 활성 recipe에는 --trackman-prior가 필요합니다")
            prior = prepare_trackman_prior(args.trackman_prior, args.pitcher_map)
        d, feats = prepare_features(raw, recipe, prior)
        years = _parse_years(args.eval_years, recipe)
        folds = {}
        for year in years:
            masks = make_fold_masks(d, recipe, year)
            folds[str(year)] = {name: int(mask.sum()) for name, mask in masks.items()}
        feature_audit: dict[str, Any] = {"count": len(feats), "fold_rows": folds}
        if meta is not None:
            expected_features = meta.get("features", [])
            feature_audit["matches_model_meta"] = feats == expected_features
            if feats != expected_features:
                missing = [name for name in expected_features if name not in feats]
                extra = [name for name in feats if name not in expected_features]
                raise RecipeError(
                    f"feature schema가 model meta와 다릅니다: missing={missing}, extra={extra}"
                )
        result["audits"]["data_and_features"] = feature_audit
    elif args.fit:
        raise RecipeError("--fit에는 --data가 필요합니다")

    if args.fit:
        if args.max_rows is not None:
            raise RecipeError("--fit에서는 --max-rows를 사용할 수 없습니다")
        assert d is not None and feats is not None
        result["folds"] = [
            fit_fold(d, feats, recipe, year, quick=args.quick)
            for year in _parse_years(args.eval_years, recipe)
        ]

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecipeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
