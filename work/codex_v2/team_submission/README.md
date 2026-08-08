# Batter-team residual candidate

This directory prepares a strictly isolated candidate submission. It never
executes the base `bundle.py`, never writes to the base `model/`, and copies only
the model files implied by `meta.json`. Disabled or stale artifacts such as an
unused `mlp3.npz` are not bundled.

The correction is a fixed lookup fitted from calibrated OOF predictions:

```text
base logits -> R/F shift -> sigmoid -> p_base
p_candidate = clip(p_base + weight * team_correction, 1e-6, 1-1e-6)
```

The correction is applied only to regular-season (`game_type == "R"`) rows.
F rows remain bit-for-bit equal to the base probability because the available
OOF evidence is R-only.

There is no fit, group-by, target mean calculation, or other aggregation over
test rows. An unknown/missing team receives exactly zero correction. A weight of
zero returns the original probability object before row-id mapping, giving a
bit-for-bit control.

## 1. Export the selected OOF table

The input prediction column must be the final calibrated OOF probability from
the exact deployable two-MLP base model. Use a new destination directory.

```powershell
python make_artifact.py `
  --oof C:\path\selected_oof.csv `
  --base-meta "C:\Users\noc07\OneDrive\Desktop\류민기\work\model\meta.json" `
  --output-dir C:\path\team_artifact `
  --prediction-col prediction `
  --shrinkage 1000 `
  --max-abs-correction 0.04 `
  --fit-description "2024 R OOF from the final 2-MLP recipe"
```

The output contains only `team_residual.csv` and
`team_residual_manifest.json`. The runtime checks the exact two-column schema,
team allowlist, row count, correction cap, application space/order, and table
SHA-256.

## 2. Stage a candidate at the chosen weight

The base directory is read-only; the output must be elsewhere. `--zip-name` is
optional, so validation can inspect the staged directory before creating a zip.

```powershell
python build_candidate.py `
  --base-work "C:\Users\noc07\OneDrive\Desktop\류민기\work" `
  --artifact-dir C:\path\team_artifact `
  --output-dir C:\path\candidate_w025 `
  --weight 0.25
```

After end-to-end verification, add `--zip-name submit_team_w025.zip` on a fresh
output directory. Do not use leaderboard results to tune the weight; choose it
from the transfer/blocked validation first.

## Selected candidate

The v2 forward validation selected `shrinkage=1000`, raw correction cap `0.04`,
and deployment weight `0.50`. The fixed table was fitted once from the 2024 R
OOF predictions of the final LGBM + Logistic + MLP1 + MLP2 recipe.

The resulting `../submit_team_residual_w050.zip` was built with the strict
allowlist/manifest verifier and then executed from a clean unpacked directory.
Its generated output passed row-id, order, probability-range, and exact-output
hash checks. This remains a challenger until an actual leaderboard submission
provides an external score. It is frozen to the 100-feature base meta SHA-256
`d4fb764b0950608f60c3b45a2ece5036ca81620d7120bce3c46fd141c100ca13`;
do not transplant its correction table onto a later base model.
