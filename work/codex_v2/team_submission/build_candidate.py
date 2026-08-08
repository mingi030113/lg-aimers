"""Build an isolated team-residual candidate without touching the base workdir."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

try:
    from .runtime import _team_load_table, _team_validate_config
except ImportError:  # direct ``python build_candidate.py`` execution
    from runtime import _team_load_table, _team_validate_config


HERE = Path(__file__).resolve().parent
RUNTIME_PATH = HERE / "runtime.py"
TABLE_FILE = "team_residual.csv"
MANIFEST_FILE = "team_residual_manifest.json"
SCRIPT_NEEDLE = "    p = np.clip(_sigmoid(z), 1e-6, 1 - 1e-6)\n"
MAIN_NEEDLE = "\ndef main():\n"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_model_files(meta: dict) -> list[str]:
    n_seeds = meta.get("n_seeds")
    if isinstance(n_seeds, bool) or not isinstance(n_seeds, int) or n_seeds <= 0:
        raise ValueError("meta.n_seeds must be a positive integer")
    names = ["meta.json", "linear.npz"] + [f"lgb_{i}.txt" for i in range(n_seeds)]
    if meta.get("use_trackman", False):
        names.append("tm_prior.csv")
    if meta.get("use_mlp", False):
        names.append("mlp.npz")
    if meta.get("use_mlp2", False):
        if not meta.get("use_mlp", False):
            raise ValueError("MLP2 cannot be enabled without MLP1")
        names.append("mlp2.npz")
    if meta.get("use_mlp3", False):
        if not meta.get("use_mlp", False):
            raise ValueError("MLP3 cannot be enabled without MLP1")
        names.append("mlp3.npz")
    if meta.get("use_prior", False):
        priors = meta.get("prior_tables")
        if not isinstance(priors, list) or not priors:
            raise ValueError("use_prior requires a non-empty prior_tables list")
        for name in priors:
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError("invalid prior table name")
            names.append(f"prior_{name}.csv")
    if len(names) != len(set(names)):
        raise ValueError("duplicate model artifact in allowlist")
    return names


def requirements_from_bundle(bundle_path: Path) -> str:
    tree = ast.parse(bundle_path.read_text(encoding="utf-8"), filename=str(bundle_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "REQUIREMENTS" for t in node.targets):
                value = ast.literal_eval(node.value)
                if not isinstance(value, str) or not value.strip():
                    break
                return value
    raise ValueError("could not read a literal REQUIREMENTS value from base bundle.py")


def inject_runtime(base_script: str, runtime_source: str) -> str:
    if base_script.count(MAIN_NEEDLE) != 1:
        raise ValueError("base script does not contain exactly one main() insertion point")
    if base_script.count(SCRIPT_NEEDLE) != 1:
        raise ValueError("base script does not contain exactly one calibrated probability line")
    if "_apply_team_residual_probability" in base_script:
        raise ValueError("base script already contains a team residual correction")
    script = base_script.replace(
        MAIN_NEEDLE,
        "\n\n# ===== isolated team-residual runtime =====\n"
        + runtime_source.rstrip()
        + "\n# ===== end team-residual runtime =====\n\n\ndef main():\n",
        1,
    )
    script = script.replace(
        SCRIPT_NEEDLE,
        SCRIPT_NEEDLE
        + "    p = _apply_team_residual_probability(test, p, meta, MODEL_DIR)\n",
        1,
    )
    compile(script, "candidate_script.py", "exec")
    return script


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_artifact(artifact_dir: Path) -> tuple[dict, Path, Path]:
    table_path = artifact_dir / TABLE_FILE
    manifest_path = artifact_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("table_file") != TABLE_FILE:
        raise ValueError("artifact manifest has an unexpected table filename")
    if manifest.get("table_sha256") != sha256_path(table_path):
        raise ValueError("artifact correction table SHA-256 mismatch")
    if manifest.get("group_col") != "batter_team_id":
        raise ValueError("only batter_team_id artifacts are allowed")
    if manifest.get("applies_to_game_type") != "R":
        raise ValueError("team residual artifact must be scoped to R rows")
    if manifest.get("correction_space") != "probability":
        raise ValueError("only probability-space artifacts are allowed")
    if manifest.get("application_order") != "after_base_calibration":
        raise ValueError("artifact must run after base calibration")
    if manifest.get("unknown_correction") != 0.0:
        raise ValueError("artifact unknown correction must be zero")
    return manifest, table_path, manifest_path


def build_candidate(
    *,
    base_work: Path,
    artifact_dir: Path,
    output_dir: Path,
    weight: float,
    zip_name: str | None = None,
) -> Path | None:
    base_work = base_work.resolve()
    artifact_dir = artifact_dir.resolve()
    output_dir = output_dir.resolve()
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("weight must be numeric")
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if _inside(output_dir, base_work):
        raise ValueError("output directory must be outside the immutable base workdir")
    if zip_name is not None and (Path(zip_name).name != zip_name or not zip_name.endswith(".zip")):
        raise ValueError("zip_name must be a plain .zip filename")

    base_model = base_work / "model"
    base_meta_path = base_model / "meta.json"
    meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
    model_allowlist = required_model_files(meta)
    for name in model_allowlist:
        path = base_model / name
        if not path.is_file():
            raise FileNotFoundError(f"required base model artifact is missing: {path}")

    # Read and fingerprint all three current deployment sources.  We consume the
    # already-inlined script so the original bundle generator never writes into
    # the immutable base workdir.
    base_script_path = base_work / "script.py"
    template_path = base_work / "script_template.py"
    bundle_path = base_work / "bundle.py"
    base_script = base_script_path.read_text(encoding="utf-8")
    template = template_path.read_text(encoding="utf-8")
    if "__FEATURES_SOURCE__" not in template:
        raise ValueError("base script_template.py is missing its feature placeholder")
    if meta.get("use_mlp2", False) and "mlp2.npz" not in base_script:
        raise ValueError("base script.py does not implement the active MLP2 model")
    requirements = requirements_from_bundle(bundle_path)
    runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
    candidate_script = inject_runtime(base_script, runtime_source)

    artifact_manifest, table_path, artifact_manifest_path = _load_artifact(artifact_dir)
    if artifact_manifest.get("fit", {}).get("base_meta_sha256") != sha256_path(base_meta_path):
        raise ValueError("correction artifact was not fitted for this base meta.json")

    candidate_meta = dict(meta)
    candidate_meta["team_residual"] = {
        "enabled": True,
        "group_col": "batter_team_id",
        "applies_to_game_type": "R",
        "weight": weight,
        "correction_space": "probability",
        "application_order": "after_base_calibration",
        "unknown_correction": 0.0,
        "table_file": TABLE_FILE,
        "manifest_file": MANIFEST_FILE,
    }
    validated_config = _team_validate_config(candidate_meta)
    if validated_config is None:  # defensive: config was just constructed above
        raise ValueError("candidate team residual config unexpectedly missing")
    # Validate the complete manifest/table allowlist before copying anything.
    # This intentionally runs even for the zero-weight control; weight zero skips
    # the artifact only at inference time to preserve exact base probabilities.
    _team_load_table(str(artifact_dir), validated_config)

    temp_dir = output_dir.with_name(output_dir.name + f".tmp-{os.getpid()}")
    if temp_dir.exists():
        raise FileExistsError(f"temporary build directory already exists: {temp_dir}")
    try:
        model_out = temp_dir / "model"
        model_out.mkdir(parents=True)
        (temp_dir / "script.py").write_text(candidate_script, encoding="utf-8")
        (temp_dir / "requirements.txt").write_text(requirements, encoding="utf-8")
        for name in model_allowlist:
            if name != "meta.json":
                shutil.copy2(base_model / name, model_out / name)
        (model_out / "meta.json").write_text(
            json.dumps(candidate_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(table_path, model_out / TABLE_FILE)
        shutil.copy2(artifact_manifest_path, model_out / MANIFEST_FILE)

        copied_model_files = sorted(path.name for path in model_out.iterdir())
        expected_model_files = sorted(
            [name for name in model_allowlist if name != "meta.json"]
            + ["meta.json", TABLE_FILE, MANIFEST_FILE]
        )
        if copied_model_files != expected_model_files:
            raise ValueError("candidate model directory differs from explicit allowlist")

        build_manifest = {
            "schema_version": 1,
            "candidate": "batter_team_probability_residual",
            "weight": weight,
            "base_sources": {
                "script.py": sha256_path(base_script_path),
                "script_template.py": sha256_path(template_path),
                "bundle.py": sha256_path(bundle_path),
                "model/meta.json": sha256_path(base_meta_path),
            },
            "model_allowlist": expected_model_files,
            "candidate_script_sha256": sha256_path(temp_dir / "script.py"),
            "candidate_meta_sha256": sha256_path(model_out / "meta.json"),
            "team_table_sha256": sha256_path(model_out / TABLE_FILE),
            "zero_weight_contract": "bit-for-bit base probability before row_id mapping",
        }
        (temp_dir / "candidate_build_manifest.json").write_text(
            json.dumps(build_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_dir.rename(output_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    zip_path = None
    if zip_name is not None:
        zip_path = output_dir / zip_name
        bundle_names = ["script.py", "requirements.txt"] + [
            f"model/{name}" for name in expected_model_files
        ]
        with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED) as archive:
            for name in bundle_names:
                archive.write(output_dir / name, name)
        with zipfile.ZipFile(zip_path) as archive:
            if sorted(archive.namelist()) != sorted(bundle_names):
                raise ValueError("candidate zip differs from explicit allowlist")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-work", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--zip-name", help="optional plain .zip filename")
    args = parser.parse_args()
    zip_path = build_candidate(
        base_work=args.base_work,
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        weight=args.weight,
        zip_name=args.zip_name,
    )
    print(f"candidate staged at {args.output_dir.resolve()}")
    if zip_path is not None:
        print(f"candidate bundle written to {zip_path}")


if __name__ == "__main__":
    main()
