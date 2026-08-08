"""Runtime-only probability-space batter-team correction.

This module is injected verbatim into the candidate ``script.py``.  Keep it
self-contained and do not add any fitting or destination-data aggregation here.
"""

import hashlib as _team_hashlib
import json as _team_json
import os as _team_os

import numpy as np
import pandas as pd


_TEAM_MANIFEST_KEYS = {
    "schema_version",
    "artifact_type",
    "group_col",
    "applies_to_game_type",
    "correction_space",
    "application_order",
    "unknown_correction",
    "table_file",
    "table_sha256",
    "columns",
    "n_rows",
    "allowed_group_values",
    "max_abs_correction",
    "fit",
}
_TEAM_TABLE_COLUMNS = ["batter_team_id", "residual_correction"]


def _team_sha256(path):
    digest = _team_hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _team_plain_filename(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty filename")
    if value != _team_os.path.basename(value) or "/" in value or "\\" in value:
        raise ValueError(f"{label} must not contain a path")
    return value


def _team_integer_ids(values, label):
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain finite integer IDs")
    return numeric.astype(np.int64)


def _team_validate_config(meta):
    config = meta.get("team_residual")
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("meta.team_residual must be an object")
    expected = {
        "enabled",
        "group_col",
        "applies_to_game_type",
        "weight",
        "correction_space",
        "application_order",
        "unknown_correction",
        "table_file",
        "manifest_file",
    }
    if set(config) != expected:
        raise ValueError(
            "meta.team_residual keys differ from the deployment allowlist: "
            f"missing={sorted(expected - set(config))}, extra={sorted(set(config) - expected)}"
        )
    if not isinstance(config["enabled"], bool):
        raise ValueError("team_residual.enabled must be boolean")
    if config["group_col"] != "batter_team_id":
        raise ValueError("only batter_team_id correction is allowed")
    if config["applies_to_game_type"] != "R":
        raise ValueError("team residual correction is allowed only for R rows")
    weight = config["weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("team_residual.weight must be numeric")
    weight = float(weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("team_residual.weight must be finite and in [0, 1]")
    if config["correction_space"] != "probability":
        raise ValueError("team residual must be applied in probability space")
    if config["application_order"] != "after_base_calibration":
        raise ValueError("team residual must run after base calibration")
    if config["unknown_correction"] != 0.0:
        raise ValueError("unknown team correction must be exactly zero")
    _team_plain_filename(config["table_file"], "team_residual.table_file")
    _team_plain_filename(config["manifest_file"], "team_residual.manifest_file")
    return config


def _team_load_table(model_dir, config):
    manifest_path = _team_os.path.join(model_dir, config["manifest_file"])
    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = _team_json.load(fp)
    if not isinstance(manifest, dict) or set(manifest) != _TEAM_MANIFEST_KEYS:
        actual = set(manifest) if isinstance(manifest, dict) else set()
        raise ValueError(
            "team residual manifest keys differ from allowlist: "
            f"missing={sorted(_TEAM_MANIFEST_KEYS - actual)}, "
            f"extra={sorted(actual - _TEAM_MANIFEST_KEYS)}"
        )
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported team residual manifest schema")
    if manifest["artifact_type"] != "batter_team_probability_residual":
        raise ValueError("unexpected team residual artifact type")
    if manifest["group_col"] != config["group_col"]:
        raise ValueError("team residual group_col mismatch")
    if manifest["applies_to_game_type"] != config["applies_to_game_type"]:
        raise ValueError("team residual game_type scope mismatch")
    if manifest["correction_space"] != config["correction_space"]:
        raise ValueError("team residual correction_space mismatch")
    if manifest["application_order"] != config["application_order"]:
        raise ValueError("team residual application_order mismatch")
    if manifest["unknown_correction"] != config["unknown_correction"]:
        raise ValueError("team residual unknown correction mismatch")
    if manifest["columns"] != _TEAM_TABLE_COLUMNS:
        raise ValueError("team residual table column allowlist mismatch")
    if manifest["table_file"] != config["table_file"]:
        raise ValueError("team residual table filename mismatch")
    _team_plain_filename(manifest["table_file"], "manifest.table_file")
    table_path = _team_os.path.join(model_dir, manifest["table_file"])
    if _team_sha256(table_path) != manifest["table_sha256"]:
        raise ValueError("team residual table SHA-256 mismatch")

    table = pd.read_csv(table_path)
    if list(table.columns) != _TEAM_TABLE_COLUMNS:
        raise ValueError("team residual CSV columns/order differ from allowlist")
    if len(table) != manifest["n_rows"] or len(table) == 0:
        raise ValueError("team residual CSV row count mismatch")
    team_ids = _team_integer_ids(table["batter_team_id"], "team residual table")
    if len(np.unique(team_ids)) != len(team_ids):
        raise ValueError("team residual table contains duplicate team IDs")
    allowed = _team_integer_ids(
        manifest["allowed_group_values"], "manifest.allowed_group_values"
    )
    if sorted(team_ids.tolist()) != sorted(allowed.tolist()):
        raise ValueError("team residual table team IDs differ from manifest allowlist")

    correction = pd.to_numeric(
        table["residual_correction"], errors="coerce"
    ).to_numpy(np.float64)
    max_abs = manifest["max_abs_correction"]
    if isinstance(max_abs, bool) or not isinstance(max_abs, (int, float)):
        raise ValueError("manifest.max_abs_correction must be numeric")
    max_abs = float(max_abs)
    if not np.isfinite(max_abs) or not 0.0 < max_abs <= 0.10:
        raise ValueError("manifest.max_abs_correction must be in (0, 0.10]")
    if not np.isfinite(correction).all() or np.any(np.abs(correction) > max_abs + 1e-15):
        raise ValueError("team residual correction is non-finite or exceeds the cap")
    if not isinstance(manifest["fit"], dict):
        raise ValueError("manifest.fit must be an object")
    return pd.Series(correction, index=team_ids, dtype=np.float64)


def _apply_team_residual_probability(rows, probability, meta, model_dir):
    """Apply a fixed lookup after base calibration; never aggregate test rows."""
    config = _team_validate_config(meta)
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 1 or len(p) != len(rows):
        raise ValueError("team residual prediction length mismatch")
    if not np.isfinite(p).all() or np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("base probabilities must be finite and in (0, 1)")
    if config is None or not config["enabled"] or float(config["weight"]) == 0.0:
        # The zero-weight control must remain bit-for-bit identical and must not
        # inspect a correction table or any other test row.
        return probability
    if config["group_col"] not in rows:
        raise ValueError("test rows are missing batter_team_id")
    if "game_type" not in rows:
        raise ValueError("test rows are missing game_type")

    mapping = _team_load_table(model_dir, config)
    raw_ids = pd.to_numeric(rows[config["group_col"]], errors="coerce")
    correction = raw_ids.map(mapping).fillna(0.0).to_numpy(np.float64)
    correction = np.where(
        rows["game_type"].astype(str).eq(config["applies_to_game_type"]).to_numpy(),
        correction,
        0.0,
    )
    return np.clip(
        p + float(config["weight"]) * correction,
        1e-6,
        1.0 - 1e-6,
    )
