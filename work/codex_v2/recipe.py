"""현재 최종 recipe의 순수 계산·검증 함수.

학습 라이브러리를 import하지 않아 설정 검사와 단위 테스트를 빠르게 실행할 수 있다.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


class RecipeError(ValueError):
    """서로 모순되는 recipe 설정을 발견했을 때 발생한다."""


def load_recipe(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        recipe = json.load(fp)
    validate_recipe(recipe)
    return recipe


def _active(model: Mapping[str, Any]) -> bool:
    return bool(model.get("enabled", True))


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "prediction_year", "target", "final_train_from",
        "train_window_years", "exclude_f_through", "trackman", "models",
        "calibration", "validation",
    }
    missing = sorted(required - set(recipe))
    if missing:
        raise RecipeError(f"recipe 필수 키 누락: {missing}")

    prediction_year = int(recipe["prediction_year"])
    train_from = int(recipe["final_train_from"])
    window = int(recipe["train_window_years"])
    if window <= 0:
        raise RecipeError("train_window_years는 양수여야 합니다")
    if prediction_year - train_from != window:
        raise RecipeError(
            "현재 final window와 rolling validation window가 다릅니다: "
            f"prediction_year-final_train_from={prediction_year-train_from}, "
            f"train_window_years={window}"
        )

    models = recipe["models"]
    for name in ("lgbm", "logistic", "mlp1", "mlp2", "mlp3"):
        if name not in models:
            raise RecipeError(f"models.{name} 누락")

    base_sum = (
        float(models["lgbm"]["weight_within_base"])
        + float(models["logistic"]["weight_within_base"])
    )
    if not np.isclose(base_sum, 1.0, atol=1e-12):
        raise RecipeError(f"base 내부 가중치 합이 1이 아닙니다: {base_sum}")

    if _active(models["mlp2"]) and not _active(models["mlp1"]):
        raise RecipeError("현재 blend 경로에서는 MLP2가 활성화되려면 MLP1도 활성화되어야 합니다")
    extra = sum(
        float(models[name]["weight"])
        for name in ("mlp1", "mlp2", "mlp3")
        if _active(models[name])
    )
    if extra < 0 or extra >= 1:
        raise RecipeError(f"활성 MLP 가중치 합은 [0, 1)이어야 합니다: {extra}")

    groups = set(recipe["validation"]["game_types"])
    if groups != {"R", "F"}:
        raise RecipeError(f"현재 보정은 R/F 두 game_type을 요구합니다: {sorted(groups)}")
    d_auto = recipe["calibration"]["d_auto"]
    if set(d_auto) != {"R", "F"}:
        raise RecipeError("calibration.d_auto에는 R과 F가 모두 있어야 합니다")


def component_weights(recipe: Mapping[str, Any]) -> dict[str, float]:
    """최종 logit에서 각 모델이 차지하는 실효 가중치를 반환한다."""
    validate_recipe(recipe)
    models = recipe["models"]
    active_mlp = {
        name: float(models[name]["weight"])
        for name in ("mlp1", "mlp2", "mlp3")
        if _active(models[name])
    }
    base_weight = 1.0 - sum(active_mlp.values())
    out = {
        "lgbm": base_weight * float(models["lgbm"]["weight_within_base"]),
        "logistic": base_weight * float(models["logistic"]["weight_within_base"]),
        **active_mlp,
    }
    if not np.isclose(sum(out.values()), 1.0, atol=1e-12):
        raise RecipeError(f"최종 모델 가중치 합이 1이 아닙니다: {out}")
    return out


def fold_train_bounds(recipe: Mapping[str, Any], eval_year: int) -> tuple[int, int]:
    """2025의 2023~2024 창을 과거 평가연도까지 그대로 이동한다."""
    window = int(recipe["train_window_years"])
    return int(eval_year) - window, int(eval_year) - 1


def make_fold_masks(
    frame: pd.DataFrame, recipe: Mapping[str, Any], eval_year: int
) -> dict[str, np.ndarray]:
    if "season" not in frame:
        raise RecipeError("입력 frame에 season 열이 없습니다")
    if "is_F" in frame:
        is_f = frame["is_F"].to_numpy() == 1
    elif "game_type" in frame:
        is_f = frame["game_type"].astype(str).to_numpy() == "F"
    else:
        raise RecipeError("입력 frame에 is_F 또는 game_type 열이 없습니다")

    season = frame["season"].to_numpy()
    start, end = fold_train_bounds(recipe, eval_year)
    excluded_old_f = is_f & (season <= int(recipe["exclude_f_through"]))
    train = (season >= start) & (season <= end) & ~excluded_old_f
    calibration = season == end
    evaluation = season == int(eval_year)
    return {
        "train": train,
        "calibration": calibration,
        "evaluation": evaluation,
        "evaluation_R": evaluation & ~is_f,
        "evaluation_F": evaluation & is_f,
    }


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -709.0, 709.0)))


def solve_shift(z: np.ndarray, target_mean: float) -> float:
    """mean(sigmoid(z + delta)) == target_mean이 되도록 delta를 구한다."""
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        raise RecipeError("빈 배열에는 평균 보정 shift를 계산할 수 없습니다")
    if not 0.0 < float(target_mean) < 1.0:
        raise RecipeError(f"target_mean은 (0, 1)이어야 합니다: {target_mean}")
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if float(sigmoid(z + mid).mean()) < target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def blend_logits(
    recipe: Mapping[str, Any],
    z_lgbm: np.ndarray,
    z_logistic: np.ndarray,
    z_mlp1: np.ndarray | None = None,
    z_mlp2: np.ndarray | None = None,
    z_mlp3: np.ndarray | None = None,
) -> np.ndarray:
    members = {
        "lgbm": z_lgbm,
        "logistic": z_logistic,
        "mlp1": z_mlp1,
        "mlp2": z_mlp2,
        "mlp3": z_mlp3,
    }
    weights = component_weights(recipe)
    out = np.zeros_like(np.asarray(z_lgbm, dtype=np.float64))
    for name, weight in weights.items():
        value = members[name]
        if value is None:
            raise RecipeError(f"활성 모델 {name}의 logit이 전달되지 않았습니다")
        value = np.asarray(value, dtype=np.float64)
        if value.shape != out.shape:
            raise RecipeError(f"{name} logit shape 불일치: {value.shape} != {out.shape}")
        out += weight * value
    return out


def apply_count_calibration(
    z: np.ndarray, cnt_diff: np.ndarray, recipe: Mapping[str, Any]
) -> np.ndarray:
    table = recipe.get("count_cal") or {}
    z = np.asarray(z, dtype=np.float64).copy()
    if table:
        correction = np.array(
            [float(table.get(str(int(v)), 0.0)) for v in np.asarray(cnt_diff)],
            dtype=np.float64,
        )
        z += correction
    return z


def fit_group_shifts(
    frame: pd.DataFrame,
    z: np.ndarray,
    recipe: Mapping[str, Any],
    calibration_year: int,
) -> dict[str, dict[str, float | int] | None]:
    """최종 train_final.py와 같은 in-sample 평균 보정 shift를 계산한다.

    목표 평균은 이전 시즌 실측 평균 + DRIFT - D_AUTO이다. 2022 이전 F는
    정의가 달라 유효한 보정 근거로 사용하지 않는다.
    """
    z = np.asarray(z, dtype=np.float64)
    if len(frame) != len(z):
        raise RecipeError("frame과 z 길이가 다릅니다")
    target_col = str(recipe["target"])
    if target_col not in frame:
        raise RecipeError(f"입력 frame에 target 열이 없습니다: {target_col}")
    if "is_F" in frame:
        is_f = frame["is_F"].to_numpy() == 1
    else:
        is_f = frame["game_type"].astype(str).to_numpy() == "F"
    season = frame["season"].to_numpy()
    y = frame[target_col].to_numpy(dtype=np.float64)
    drift = float(recipe["calibration"]["drift"])
    d_auto = recipe["calibration"]["d_auto"]
    min_rows = int(recipe["validation"]["minimum_calibration_rows"])
    result: dict[str, dict[str, float | int] | None] = {}

    for label, group_f in (("R", False), ("F", True)):
        if group_f and calibration_year <= int(recipe["exclude_f_through"]):
            result[label] = None
            continue
        selected = (season == calibration_year) & (is_f == group_f)
        if int(selected.sum()) < min_rows:
            raise RecipeError(
                f"{calibration_year} {label} 보정 표본 부족: "
                f"{int(selected.sum())} < {min_rows}"
            )
        observed = float(y[selected].mean())
        target_mean = observed + drift - float(d_auto[label])
        delta = solve_shift(z[selected], target_mean)
        result[label] = {
            "n": int(selected.sum()),
            "observed_mean": observed,
            "target_mean": target_mean,
            "shift": delta,
        }
    return result


def score_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if y.size == 0:
        raise RecipeError("채점 가능한 행이 없습니다")
    rate = float(y.mean())
    base = rate * (1.0 - rate)
    if base <= 0:
        raise RecipeError("target이 한 클래스뿐이라 BSS를 계산할 수 없습니다")
    offset = float(p.mean() - rate)
    centered = p - offset
    return {
        "n": int(y.size),
        "score": 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / base),
        "score_if_mean_fixed": 100000.0
        * (1.0 - float(np.mean((centered - y) ** 2)) / base),
        "pred_mean": float(p.mean()),
        "true_mean": rate,
        "mean_offset": offset,
    }


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(float(left), float(right), atol=1e-12, rtol=0.0))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return list(left) == list(right)
    return left == right


def audit_model_meta(recipe: Mapping[str, Any], meta: Mapping[str, Any]) -> list[str]:
    """배포 model/meta.json과 recipe의 불일치 목록을 반환한다."""
    models = recipe["models"]
    expected = {
        "n_seeds": models["lgbm"]["n_seeds"],
        "blend_w": models["logistic"]["weight_within_base"],
        "logreg_C": models["logistic"]["C"],
        "use_trackman": recipe["trackman"]["enabled"],
        "train_from": recipe["final_train_from"],
        "decay_halflife": recipe["training"]["decay_halflife"],
        "use_mlp": models["mlp1"]["enabled"],
        "mlp_w": models["mlp1"]["weight"],
        "use_mlp2": models["mlp2"]["enabled"],
        "mlp2_w": models["mlp2"]["weight"],
        "use_mlp3": models["mlp3"]["enabled"],
        "mlp3_w": models["mlp3"]["weight"],
        "use_prior": recipe["prior"]["enabled"],
        "prior_decay": recipe["prior"]["decay"],
        "count_cal": recipe.get("count_cal") or {},
        "drift": recipe["calibration"]["drift"],
        "d_auto": recipe["calibration"]["d_auto"],
    }
    problems = []
    for key, value in expected.items():
        if key not in meta:
            problems.append(f"meta.{key} 누락")
        elif not _same(value, meta[key]):
            problems.append(f"meta.{key}: expected={value!r}, actual={meta[key]!r}")
    return problems

def read_literal_assignments(path: str | Path) -> dict[str, Any]:
    """학습 스크립트를 실행하지 않고 상단 literal 설정만 읽는다."""
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    result: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                result[target.id] = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                pass
    return result


def audit_train_script(recipe: Mapping[str, Any], path: str | Path) -> list[str]:
    """현재 train_final.py의 정적 설정과 recipe를 비교한다."""
    values = read_literal_assignments(path)
    models = recipe["models"]
    expected = {
        "USE_TRACKMAN": recipe["trackman"]["enabled"],
        "TRAIN_FROM": recipe["final_train_from"],
        "DECAY_HALFLIFE": recipe["training"]["decay_halflife"],
        "USE_PRIOR": recipe["prior"]["enabled"],
        "PRIOR_DECAY": recipe["prior"]["decay"],
        "COUNT_CAL": recipe.get("count_cal") or {},
        "USE_MLP": models["mlp1"]["enabled"],
        "MLP_HIDDEN": models["mlp1"]["hidden"],
        "MLP_ALPHA": models["mlp1"]["alpha"],
        "MLP_SEEDS": models["mlp1"]["n_seeds"],
        "MLP_W": models["mlp1"]["weight"],
        "USE_MLP2": models["mlp2"]["enabled"],
        "MLP2_W": models["mlp2"]["weight"],
        "USE_MLP3": models["mlp3"]["enabled"],
        "MLP3_W": models["mlp3"]["weight"],
        "ROUNDS": models["lgbm"]["rounds"],
        "N_SEEDS": models["lgbm"]["n_seeds"],
        "LOGREG_C": models["logistic"]["C"],
        "BLEND_W": models["logistic"]["weight_within_base"],
        "DRIFT": recipe["calibration"]["drift"],
    }
    problems = []
    for name, value in expected.items():
        if name not in values:
            problems.append(f"{name} literal 설정을 찾지 못함")
        elif not _same(value, values[name]):
            problems.append(f"{name}: expected={value!r}, actual={values[name]!r}")
    return problems
