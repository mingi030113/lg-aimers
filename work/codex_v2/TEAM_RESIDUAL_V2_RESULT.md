# Batter-team residual v2 result

## Decision

Adopt the fixed batter-team residual correction as a submission challenger at
deployment weight `0.50`.

This does not replace or retrain the base model. The base remains the final
two-MLP ensemble: five LightGBM seeds, logistic regression, MLP1, and quantile
MLP2. The correction is a ten-row lookup fitted only from calibrated OOF
residuals.

## Validation

The source table used `shrinkage=1000`. Candidate weights were fixed in advance
at `0.25`, `0.50`, `0.75`, and `1.00`. At the selected weight:

| Source | Held-out target | Score delta |
|---|---|---:|
| 2023 all | 2024 all | +39.33 |
| 2023 early | 2023 late | +54.50 |
| 2023 late | 2024 early | +45.16 |
| 2024 early | 2024 late | +33.16 |

- All forward transfer views were positive: `8/8`.
- Leave-one-team-correction-out checks were positive: `47/50`.
- The annual held-out prediction mean changed by only `-0.000051`.
- Team 13 contributed most of the gain, but its residual direction repeated in
  2023 early, 2023 late, 2024 early, and 2024 late.

Weight `0.75` had the highest local average point estimate. Weight `0.50` was
selected as the practical transfer-stable setting, keeping only half of the
estimated correction without discarding a consistently positive signal.

## Deployment contract

- Fit source: 2024 regular-season calibrated OOF from the exact final two-MLP
  recipe (`223,497` rows).
- Raw correction cap: `0.04`.
- Applied weight: `0.50`.
- Application point: probability space after the existing R/F calibration.
- Scope: only rows where `game_type == "R"`; F predictions remain unchanged.
- Unknown or missing batter team: zero correction.
- No fitting, mean correction, group-by, or other aggregation on test rows.

## Submission artifact

- File: `submit_team_residual_w050.zip`
- Size: `1,899,665` bytes
- SHA-256: `878761afa9f0f4cd27e53a31f13cc2f490eeebeb053fa0422f6759528fbf6ce5`
- Members: 15 strict-allowlisted files, including MLP2 and excluding inactive
  MLP3.

The final ZIP passed strict manifest verification, model shape/schema checks,
and an end-to-end execution from a clean unpacked directory. The clean run's
five-row smoke output was byte-identical to the staged candidate output
(`SHA-256 04c96e3e110bdb2e7efa9189ce3a8f4be91fac1ae29b07ee4d14ced66fe42c21`).

## Frozen-base provenance

The candidate is intentionally frozen to the 100-feature two-MLP champion
snapshot that was current when this validation began:

- base `model/meta.json` SHA-256:
  `d4fb764b0950608f60c3b45a2ece5036ca81620d7120bce3c46fd141c100ca13`
- base `script.py` SHA-256:
  `652991d04af48a258bffc87a874351525bb1d2effcdedea66c2edb9126cd9406`

The active source workspace changed concurrently after the bundle was built.
Those later files were neither read into nor merged with this artifact. In
particular, this correction table must not be copied onto a different base
model without regenerating matching OOF predictions and revalidating it.

These are local forward-OOF results. The public score remains the recorded
champion score until this challenger is actually submitted and evaluated.
