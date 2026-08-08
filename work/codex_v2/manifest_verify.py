#!/usr/bin/env python3
"""Build and verify immutable LG Aimers submission snapshots.

This utility deliberately does not import project modules.  It treats a trained
model directory as an immutable input, derives the bundle allowlist from
``meta.json``, validates artifact shapes, and writes a new zip through a
same-directory temporary file followed by ``os.replace``.

Commands
--------
check-model
    Validate one model directory without changing it.
build
    Create a strictly allowlisted bundle containing an artifact manifest.
verify-bundle
    Recompute all hashes and artifact checks from an existing bundle.
verify-input
    Fail unless test.csv and sample_submission.csv have a one-to-one row_id set.
verify-output
    Fail unless submission.csv exactly follows sample_submission row order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only in a broken env
    raise SystemExit("numpy is required to inspect model artifacts") from exc


MANIFEST_NAME = "artifact_manifest.json"
REQUIREMENTS_DEFAULT = "lightgbm==4.7.0\n"
SAFE_PRIOR_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
CHUNK = 1024 * 1024


class VerificationError(RuntimeError):
    """Raised for a fail-closed validation error."""


def fail(message: str) -> None:
    raise VerificationError(message)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON root must be an object: {path}")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite")
    return result


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(f"{label} must be a positive integer")
    return value


def boolean_flag(meta: dict[str, Any], key: str) -> bool:
    value = meta.get(key, False)
    if not isinstance(value, bool):
        fail(f"meta.{key} must be a JSON boolean")
    return value


def validate_meta(meta: dict[str, Any]) -> tuple[list[str], list[str]]:
    features = meta.get("features")
    if not isinstance(features, list) or not features:
        fail("meta.features must be a non-empty list")
    if any(not isinstance(v, str) or not v or any(ch.isspace() for ch in v) for v in features):
        fail("meta.features must contain non-empty names without whitespace")
    if len(set(features)) != len(features):
        fail("meta.features contains duplicates")

    n_seeds = positive_int(meta.get("n_seeds"), "meta.n_seeds")
    blend_w = finite_number(meta.get("blend_w"), "meta.blend_w")
    if not 0.0 <= blend_w <= 1.0:
        fail("meta.blend_w must be in [0, 1]")

    for label in ("shift", "expected_2025"):
        value = meta.get(label)
        if not isinstance(value, dict) or set(value) != {"R", "F"}:
            fail(f"meta.{label} must contain exactly R and F")
        finite_number(value["R"], f"meta.{label}.R")
        finite_number(value["F"], f"meta.{label}.F")

    use_mlp = boolean_flag(meta, "use_mlp")
    use_mlp2 = boolean_flag(meta, "use_mlp2")
    use_mlp3 = boolean_flag(meta, "use_mlp3")
    if (use_mlp2 or use_mlp3) and not use_mlp:
        fail("MLP2/MLP3 cannot be enabled when meta.use_mlp is false")
    mlp_weights: list[float] = []
    for enabled, key in (
        (use_mlp, "mlp_w"),
        (use_mlp2, "mlp2_w"),
        (use_mlp3, "mlp3_w"),
    ):
        if enabled:
            weight = finite_number(meta.get(key), f"meta.{key}")
            if weight < 0.0:
                fail(f"meta.{key} must be non-negative")
            mlp_weights.append(weight)
    if sum(mlp_weights) > 1.0 + 1e-12:
        fail("enabled MLP weights exceed 1.0")

    expected = ["meta.json", "linear.npz"]
    expected.extend(f"lgb_{seed}.txt" for seed in range(n_seeds))
    if boolean_flag(meta, "use_trackman"):
        expected.append("tm_prior.csv")
    if use_mlp:
        expected.append("mlp.npz")
    if use_mlp2:
        expected.append("mlp2.npz")
    if use_mlp3:
        expected.append("mlp3.npz")

    prior_names: list[str] = []
    if boolean_flag(meta, "use_prior"):
        raw_names = meta.get("prior_tables")
        if not isinstance(raw_names, list) or not raw_names:
            fail("meta.prior_tables must be non-empty when use_prior is true")
        for name in raw_names:
            if not isinstance(name, str) or not SAFE_PRIOR_NAME.fullmatch(name):
                fail(f"unsafe prior table name: {name!r}")
            prior_names.append(name)
            expected.append(f"prior_{name}.csv")
        if len(set(prior_names)) != len(prior_names):
            fail("meta.prior_tables contains duplicates")

    if len(set(expected)) != len(expected):
        fail("derived model allowlist contains duplicates")
    return expected, prior_names


def require_finite(array: np.ndarray, label: str) -> None:
    if not np.issubdtype(array.dtype, np.number):
        fail(f"{label} must be numeric, got {array.dtype}")
    if not np.isfinite(array).all():
        fail(f"{label} contains NaN or infinity")


def validate_linear(path: Path, n_features: int) -> dict[str, Any]:
    expected_keys = {"coef", "intercept", "mu", "sd", "med"}
    try:
        with np.load(path, allow_pickle=False) as pack:
            if set(pack.files) != expected_keys:
                fail(f"{path.name}: keys differ from {sorted(expected_keys)}")
            shapes = {key: list(pack[key].shape) for key in sorted(pack.files)}
            for key in expected_keys:
                require_finite(pack[key], f"{path.name}:{key}")
            for key in ("coef", "mu", "sd", "med"):
                if pack[key].shape != (n_features,):
                    fail(f"{path.name}:{key} expected shape {(n_features,)}, got {pack[key].shape}")
            if pack["intercept"].shape not in {(1,), ()}:
                fail(f"{path.name}:intercept must be scalar or shape (1,)")
            if np.any(pack["sd"] <= 0):
                fail(f"{path.name}:sd must be strictly positive")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        fail(f"cannot inspect {path}: {exc}")
    return {"kind": "linear", "arrays": shapes}


def scalar_int(pack: Any, key: str, path: Path) -> int:
    if key not in pack.files or pack[key].size != 1:
        fail(f"{path.name}:{key} must contain one integer")
    raw = pack[key].reshape(-1)[0]
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        fail(f"{path.name}:{key} is not an integer")
    if float(raw) != value or value <= 0:
        fail(f"{path.name}:{key} must be a positive integer")
    return value


def validate_mlp(path: Path, n_features: int, kind: str) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as pack:
            n_seeds = scalar_int(pack, "n_seeds", path)
            n_layers = scalar_int(pack, "n_layers", path)
            special: set[str] = set()
            input_dim = n_features
            details: dict[str, Any] = {"kind": kind, "n_seeds": n_seeds, "n_layers": n_layers}

            if kind == "mlp2":
                special = {"quantiles", "references"}
                if not special.issubset(pack.files):
                    fail(f"{path.name} lacks quantiles/references")
                q, refs = pack["quantiles"], pack["references"]
                require_finite(q, f"{path.name}:quantiles")
                require_finite(refs, f"{path.name}:references")
                if q.ndim != 2 or q.shape[1] != n_features or refs.shape != (q.shape[0],):
                    fail(f"{path.name}: quantile shapes are inconsistent with {n_features} features")
                if q.shape[0] < 2 or np.any(np.diff(refs) < 0) or np.any(np.diff(q, axis=0) < -1e-12):
                    fail(f"{path.name}: quantile grid must be monotonic")
                details["quantiles_shape"] = list(q.shape)
            elif kind == "mlp3":
                special = {"feat_idx"}
                if "feat_idx" not in pack.files:
                    fail(f"{path.name} lacks feat_idx")
                idx = pack["feat_idx"]
                if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or idx.size == 0:
                    fail(f"{path.name}:feat_idx must be a non-empty integer vector")
                if len(set(int(v) for v in idx)) != len(idx) or int(idx.min()) < 0 or int(idx.max()) >= n_features:
                    fail(f"{path.name}:feat_idx is duplicated or out of range")
                input_dim = len(idx)
                details["feature_subset_size"] = input_dim

            expected = {"n_seeds", "n_layers"} | special
            layer_shapes: list[list[int]] | None = None
            for seed in range(n_seeds):
                previous = input_dim
                current_shapes: list[list[int]] = []
                for layer in range(n_layers):
                    wk, bk = f"s{seed}_W{layer}", f"s{seed}_b{layer}"
                    expected.update({wk, bk})
                    if wk not in pack.files or bk not in pack.files:
                        fail(f"{path.name} lacks {wk}/{bk}")
                    weight, bias = pack[wk], pack[bk]
                    require_finite(weight, f"{path.name}:{wk}")
                    require_finite(bias, f"{path.name}:{bk}")
                    if weight.ndim != 2 or bias.ndim != 1:
                        fail(f"{path.name}:{wk}/{bk} must be matrix/vector")
                    if weight.shape[0] != previous or weight.shape[1] != bias.shape[0]:
                        fail(f"{path.name}:{wk}/{bk} has a broken layer chain")
                    previous = weight.shape[1]
                    current_shapes.append(list(weight.shape))
                if previous != 1:
                    fail(f"{path.name}: final layer must have one output")
                if layer_shapes is None:
                    layer_shapes = current_shapes
                elif current_shapes != layer_shapes:
                    fail(f"{path.name}: architecture differs across seeds")
            if set(pack.files) != expected:
                extra = sorted(set(pack.files) - expected)
                missing = sorted(expected - set(pack.files))
                fail(f"{path.name}: unexpected npz layout; missing={missing}, extra={extra}")
            details["layer_shapes"] = layer_shapes
            return details
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        fail(f"cannot inspect {path}: {exc}")


def validate_lgb(path: Path, features: list[str]) -> dict[str, Any]:
    max_feature_idx: int | None = None
    model_features: list[str] | None = None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(100):
                line = fh.readline()
                if not line:
                    break
                if line.startswith("max_feature_idx="):
                    max_feature_idx = int(line.split("=", 1)[1])
                elif line.startswith("feature_names="):
                    model_features = line.split("=", 1)[1].strip().split()
                if max_feature_idx is not None and model_features is not None:
                    break
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        fail(f"cannot inspect LightGBM model {path}: {exc}")
    if max_feature_idx != len(features) - 1:
        fail(f"{path.name}: max_feature_idx={max_feature_idx}, expected {len(features) - 1}")
    if model_features != features:
        fail(f"{path.name}: feature names/order differ from meta.json")
    return {"kind": "lightgbm", "n_features": len(features)}


def validate_trackman_csv(path: Path) -> dict[str, Any]:
    required = {"pitcher_id", "season", "tm_prior_n"}
    seen: set[tuple[str, str]] = set()
    rows = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                fail(f"{path.name}: missing or duplicate CSV headers")
            if not required.issubset(reader.fieldnames):
                fail(f"{path.name}: missing columns {sorted(required - set(reader.fieldnames or []))}")
            for line_no, row in enumerate(reader, start=2):
                key = (row["pitcher_id"], row["season"])
                if not key[0] or not key[1]:
                    fail(f"{path.name}:{line_no}: blank pitcher_id/season")
                if key in seen:
                    fail(f"{path.name}:{line_no}: duplicate pitcher_id/season {key}")
                seen.add(key)
                try:
                    count = float(row["tm_prior_n"])
                except (TypeError, ValueError):
                    fail(f"{path.name}:{line_no}: invalid tm_prior_n")
                if not math.isfinite(count) or count < 0:
                    fail(f"{path.name}:{line_no}: tm_prior_n must be finite and non-negative")
                rows += 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        fail(f"cannot inspect {path}: {exc}")
    if rows == 0:
        fail(f"{path.name} is empty")
    return {"kind": "trackman_prior", "rows": rows, "unique_keys": len(seen)}


def validate_generic_csv(path: Path) -> dict[str, Any]:
    rows = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header or len(header) != len(set(header)):
                fail(f"{path.name}: missing or duplicate CSV headers")
            for _ in reader:
                rows += 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        fail(f"cannot inspect {path}: {exc}")
    if rows == 0:
        fail(f"{path.name} is empty")
    return {"kind": "prior_table", "rows": rows, "columns": header}


def regular_files(directory: Path) -> set[str]:
    names: set[str] = set()
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        fail(f"cannot list model directory {directory}: {exc}")
    for path in entries:
        if path.is_symlink():
            fail(f"symlink is not allowed in model directory: {path.name}")
        if not path.is_file():
            fail(f"nested/non-file entry is not allowed in model directory: {path.name}")
        names.add(path.name)
    return names


def validate_model_dir(model_dir: Path) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    if not model_dir.is_dir():
        fail(f"model directory does not exist: {model_dir}")
    meta = load_json(model_dir / "meta.json")
    expected, prior_names = validate_meta(meta)
    actual = regular_files(model_dir)
    missing, extra = sorted(set(expected) - actual), sorted(actual - set(expected))
    if missing or extra:
        fail(f"model allowlist mismatch; missing={missing}, extra={extra}")

    features = meta["features"]
    checks: dict[str, Any] = {}
    checks["linear.npz"] = validate_linear(model_dir / "linear.npz", len(features))
    for seed in range(meta["n_seeds"]):
        name = f"lgb_{seed}.txt"
        checks[name] = validate_lgb(model_dir / name, features)
    if meta.get("use_trackman", False):
        checks["tm_prior.csv"] = validate_trackman_csv(model_dir / "tm_prior.csv")
    if meta.get("use_mlp", False):
        checks["mlp.npz"] = validate_mlp(model_dir / "mlp.npz", len(features), "mlp")
    if meta.get("use_mlp2", False):
        checks["mlp2.npz"] = validate_mlp(model_dir / "mlp2.npz", len(features), "mlp2")
    if meta.get("use_mlp3", False):
        checks["mlp3.npz"] = validate_mlp(model_dir / "mlp3.npz", len(features), "mlp3")
    for name in prior_names:
        filename = f"prior_{name}.csv"
        checks[filename] = validate_generic_csv(model_dir / filename)
    return meta, expected, checks


def validate_script(path: Path) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
        if not source.strip():
            fail(f"script is empty: {path}")
        compile(source, str(path), "exec")
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        fail(f"invalid inference script {path}: {exc}")
    return {"lines": source.count("\n") + 1, "bytes": len(source.encode("utf-8"))}


def copy_stable(source: Path, destination: Path) -> str:
    if source.is_symlink() or not source.is_file():
        fail(f"source artifact must be a regular file: {source}")
    before = sha256_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after = sha256_path(source)
    copied = sha256_path(destination)
    if before != after or before != copied:
        fail(f"source changed while snapshotting: {source}")
    return copied


def git_commit_near(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40}", value) else None


def artifact_entry(root: Path, relative: str) -> dict[str, Any]:
    path = root / Path(relative)
    return {"path": relative, "size": path.stat().st_size, "sha256": sha256_path(path)}


def write_deterministic_zip(root: Path, names: Iterable[str], output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(names):
            data = (root / Path(name)).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def atomic_write(path: Path, data: bytes, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        fail(f"refusing to overwrite existing file: {path}")
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def make_bundle(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir.resolve()
    script = args.script.resolve()
    output = args.out.resolve()
    if output.exists() and not args.replace:
        fail(f"refusing to overwrite existing bundle: {output}")
    if script == output or output.is_dir():
        fail("output must be a new zip file, separate from all inputs")

    if args.test is not None or args.sample is not None:
        if args.test is None or args.sample is None:
            fail("--test and --sample must be supplied together")
        validate_input_pair(args.test, args.sample, args.id_col)

    meta, model_names, source_checks = validate_model_dir(model_dir)
    script_check = validate_script(script)
    # Guard the whole snapshot window, not only each individual copy. This
    # rejects a training process that publishes a new seed/meta after an
    # earlier artifact has already been staged.
    source_hashes = {name: sha256_path(model_dir / name) for name in model_names}
    script_source_hash = sha256_path(script)
    requirements = args.requirements.encode("utf-8")
    if not requirements.strip():
        fail("requirements text cannot be empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_name(output.name + ".manifest.json")
    if sidecar.exists() and not args.replace:
        fail(f"refusing to overwrite existing sidecar: {sidecar}")
    stage = Path(tempfile.mkdtemp(prefix=".artifact-stage-", dir=output.parent))
    fd, raw_zip = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temp_zip = Path(raw_zip)
    try:
        copy_stable(script, stage / "script.py")
        (stage / "requirements.txt").write_bytes(requirements)
        for name in model_names:
            copy_stable(model_dir / name, stage / "model" / name)

        if sha256_path(script) != script_source_hash:
            fail("inference script changed during the snapshot window")
        for name, expected_hash in source_hashes.items():
            if sha256_path(model_dir / name) != expected_hash:
                fail(f"model artifact changed during the snapshot window: {name}")

        staged_meta, staged_names, staged_checks = validate_model_dir(stage / "model")
        if staged_meta != meta or staged_names != model_names or staged_checks != source_checks:
            fail("staged artifact validation differs from source validation")
        validate_script(stage / "script.py")

        members = ["script.py", "requirements.txt"] + [f"model/{name}" for name in model_names]
        files = [artifact_entry(stage, name) for name in sorted(members)]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit_near(model_dir),
            "bundle_allowlist": sorted(members + [MANIFEST_NAME]),
            "files": files,
            "model_meta_sha256": sha256_path(stage / "model" / "meta.json"),
            "model_summary": {
                "n_features": len(meta["features"]),
                "n_lgb_seeds": meta["n_seeds"],
                "train_rows": meta.get("train_rows"),
                "use_trackman": bool(meta.get("use_trackman", False)),
                "use_mlp": bool(meta.get("use_mlp", False)),
                "use_mlp2": bool(meta.get("use_mlp2", False)),
                "use_mlp3": bool(meta.get("use_mlp3", False)),
            },
            "script_check": script_check,
            "model_checks": staged_checks,
        }
        (stage / MANIFEST_NAME).write_bytes(json_bytes(manifest))
        write_deterministic_zip(stage, members + [MANIFEST_NAME], temp_zip)
        verify_bundle_file(temp_zip, args.max_uncompressed_mb)
        os.replace(temp_zip, output)
        atomic_write(sidecar, json_bytes(manifest), args.replace)
        return {
            "bundle": str(output),
            "bundle_sha256": sha256_path(output),
            "manifest": str(sidecar),
            "members": len(members) + 1,
        }
    finally:
        if temp_zip.exists():
            temp_zip.unlink()
        shutil.rmtree(stage, ignore_errors=True)


def safe_zip_members(archive: zipfile.ZipFile, max_uncompressed_mb: int) -> list[str]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        fail("bundle contains duplicate member names")
    total = 0
    for info in infos:
        name = info.filename
        pure = PurePosixPath(name)
        if not name or "\\" in name or pure.is_absolute() or ".." in pure.parts or name.endswith("/"):
            fail(f"unsafe/non-file zip member: {name!r}")
        unix_mode = (info.external_attr >> 16) & 0o170000
        if unix_mode == 0o120000:
            fail(f"zip symlink is not allowed: {name}")
        total += info.file_size
    if total > max_uncompressed_mb * 1024 * 1024:
        fail(f"bundle expands to {total} bytes, above configured limit")
    return names


def verify_bundle_file(bundle: Path, max_uncompressed_mb: int = 200) -> dict[str, Any]:
    if not bundle.is_file():
        fail(f"bundle does not exist: {bundle}")
    try:
        with zipfile.ZipFile(bundle, "r") as archive:
            names = safe_zip_members(archive, max_uncompressed_mb)
            if MANIFEST_NAME not in names:
                fail(f"bundle lacks {MANIFEST_NAME}")
            try:
                manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
                meta = json.loads(archive.read("model/meta.json").decode("utf-8"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                fail(f"cannot read bundle metadata: {exc}")
            if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
                fail("unsupported or malformed artifact manifest")
            model_names, _ = validate_meta(meta)
            expected = sorted(
                ["script.py", "requirements.txt", MANIFEST_NAME]
                + [f"model/{name}" for name in model_names]
            )
            if sorted(names) != expected:
                fail(f"bundle allowlist mismatch; expected={expected}, actual={sorted(names)}")
            if manifest.get("bundle_allowlist") != expected:
                fail("manifest bundle_allowlist differs from bundle contents")

            entries = manifest.get("files")
            if not isinstance(entries, list):
                fail("manifest.files must be a list")
            by_name: dict[str, dict[str, Any]] = {}
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                    fail("malformed file entry in manifest")
                if entry["path"] in by_name:
                    fail(f"duplicate manifest file entry: {entry['path']}")
                by_name[entry["path"]] = entry
            expected_hashed = set(expected) - {MANIFEST_NAME}
            if set(by_name) != expected_hashed:
                fail("manifest hash inventory differs from bundle allowlist")
            for name in sorted(expected_hashed):
                data = archive.read(name)
                entry = by_name[name]
                if entry.get("size") != len(data) or entry.get("sha256") != sha256_bytes(data):
                    fail(f"hash/size mismatch for {name}")

            temp = Path(tempfile.mkdtemp(prefix="artifact-verify-"))
            try:
                for name in expected_hashed:
                    target = temp / Path(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
                staged_meta, staged_names, checks = validate_model_dir(temp / "model")
                validate_script(temp / "script.py")
                if staged_meta != meta or staged_names != model_names:
                    fail("extracted model metadata differs from bundle metadata")
                if manifest.get("model_checks") != checks:
                    fail("manifest model_checks differ from recomputed checks")
                if manifest.get("model_meta_sha256") != sha256_path(temp / "model" / "meta.json"):
                    fail("manifest model_meta_sha256 mismatch")
            finally:
                shutil.rmtree(temp, ignore_errors=True)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        fail(f"cannot verify bundle {bundle}: {exc}")
    return {
        "bundle": str(bundle.resolve()),
        "bundle_sha256": sha256_path(bundle),
        "members": len(names),
        "status": "verified",
    }


def read_ids(path: Path, id_col: str) -> tuple[list[str], set[str]]:
    ordered: list[str] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                fail(f"{path}: missing or duplicate CSV headers")
            if id_col not in reader.fieldnames:
                fail(f"{path}: missing required column {id_col!r}")
            for line_no, row in enumerate(reader, start=2):
                value = row[id_col]
                if value is None or value == "":
                    fail(f"{path}:{line_no}: blank {id_col}")
                if value in seen:
                    fail(f"{path}:{line_no}: duplicate {id_col}={value!r}")
                seen.add(value)
                ordered.append(value)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        fail(f"cannot inspect CSV {path}: {exc}")
    if not ordered:
        fail(f"{path}: no data rows")
    return ordered, seen


def sample_values(values: Iterable[str], limit: int = 5) -> list[str]:
    return sorted(values)[:limit]


def validate_input_pair(test: Path, sample: Path, id_col: str = "row_id") -> dict[str, Any]:
    test_order, test_ids = read_ids(test, id_col)
    sample_order, sample_ids = read_ids(sample, id_col)
    if len(test_order) != len(sample_order) or test_ids != sample_ids:
        missing = sample_ids - test_ids
        unexpected = test_ids - sample_ids
        fail(
            "row_id mismatch between test and sample_submission; "
            f"test_rows={len(test_order)}, sample_rows={len(sample_order)}, "
            f"missing_in_test={len(missing)} {sample_values(missing)}, "
            f"unexpected_in_test={len(unexpected)} {sample_values(unexpected)}"
        )
    return {
        "test": str(test.resolve()),
        "sample": str(sample.resolve()),
        "rows": len(test_order),
        "same_order": test_order == sample_order,
        "status": "row_id bijection verified",
    }


def validate_output(sample: Path, output: Path, id_col: str, target_col: str) -> dict[str, Any]:
    sample_order, _ = read_ids(sample, id_col)
    output_order: list[str] = []
    values: list[float] = []
    seen: set[str] = set()
    try:
        with output.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            required = {id_col, target_col}
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                fail(f"{output}: missing or duplicate CSV headers")
            if not required.issubset(reader.fieldnames):
                fail(f"{output}: missing columns {sorted(required - set(reader.fieldnames or []))}")
            for line_no, row in enumerate(reader, start=2):
                rid = row[id_col]
                if rid is None or rid == "" or rid in seen:
                    fail(f"{output}:{line_no}: blank or duplicate {id_col}={rid!r}")
                seen.add(rid)
                output_order.append(rid)
                try:
                    value = float(row[target_col])
                except (TypeError, ValueError):
                    fail(f"{output}:{line_no}: invalid probability {row[target_col]!r}")
                if not math.isfinite(value) or not 0.0 < value < 1.0:
                    fail(f"{output}:{line_no}: probability must be finite and strictly between 0 and 1")
                values.append(value)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        fail(f"cannot inspect CSV {output}: {exc}")
    if output_order != sample_order:
        first = next(
            (idx for idx, (a, b) in enumerate(zip(output_order, sample_order)) if a != b),
            min(len(output_order), len(sample_order)),
        )
        fail(
            "submission row_id sequence must exactly match sample_submission; "
            f"output_rows={len(output_order)}, sample_rows={len(sample_order)}, first_difference={first}"
        )
    return {
        "output": str(output.resolve()),
        "rows": len(values),
        "prediction_min": min(values),
        "prediction_max": max(values),
        "prediction_mean": sum(values) / len(values),
        "status": "submission verified",
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-model", help="validate a model directory read-only")
    check.add_argument("--model-dir", type=Path, required=True)

    build = commands.add_parser("build", help="atomically create an allowlisted bundle and manifest")
    build.add_argument("--model-dir", type=Path, required=True)
    build.add_argument("--script", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--requirements", default=REQUIREMENTS_DEFAULT)
    build.add_argument("--test", type=Path)
    build.add_argument("--sample", type=Path)
    build.add_argument("--id-col", default="row_id")
    build.add_argument("--replace", action="store_true", help="explicitly permit replacing output and sidecar")
    build.add_argument("--max-uncompressed-mb", type=int, default=200)

    verify = commands.add_parser("verify-bundle", help="verify bundle allowlist, hashes, and artifact shapes")
    verify.add_argument("bundle", type=Path)
    verify.add_argument("--max-uncompressed-mb", type=int, default=200)

    inputs = commands.add_parser("verify-input", help="require exact test/sample row_id bijection")
    inputs.add_argument("--test", type=Path, required=True)
    inputs.add_argument("--sample", type=Path, required=True)
    inputs.add_argument("--id-col", default="row_id")

    output = commands.add_parser("verify-output", help="require output to match sample order and valid probabilities")
    output.add_argument("--sample", type=Path, required=True)
    output.add_argument("--output", type=Path, required=True)
    output.add_argument("--id-col", default="row_id")
    output.add_argument("--target-col", default="control_success")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "check-model":
            meta, names, checks = validate_model_dir(args.model_dir.resolve())
            result = {
                "model_dir": str(args.model_dir.resolve()),
                "n_features": len(meta["features"]),
                "allowlist": names,
                "checks": checks,
                "status": "verified",
            }
        elif args.command == "build":
            if args.max_uncompressed_mb <= 0:
                fail("--max-uncompressed-mb must be positive")
            result = make_bundle(args)
        elif args.command == "verify-bundle":
            if args.max_uncompressed_mb <= 0:
                fail("--max-uncompressed-mb must be positive")
            result = verify_bundle_file(args.bundle.resolve(), args.max_uncompressed_mb)
        elif args.command == "verify-input":
            result = validate_input_pair(args.test.resolve(), args.sample.resolve(), args.id_col)
        elif args.command == "verify-output":
            result = validate_output(args.sample.resolve(), args.output.resolve(), args.id_col, args.target_col)
        else:  # pragma: no cover
            fail(f"unknown command: {args.command}")
    except VerificationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
