# Trackman prior v2 (candidate only)

This implementation is intentionally isolated from `work/tm03_features.py` and
from the champion model artifacts. It fixes denominator correctness and provides
reproducible validation; it does **not** opt the corrected prior into training.

## What changes

- Every raw feature uses its own non-null observation count at both aggregation
  levels: pitch group -> pitcher-season and pitcher-season -> target-season prior.
- `tm_velo_sep` is derived from the independently accumulated fastball and
  offspeed velocities.
- Output keys are checked for `(pitcher_trackman_id, season)` uniqueness and the
  expected row count is derived independently for every target season.
- Raw and generated values must pass broad physical-unit guardrails.
- Optional main-data attachment uses Pandas `validate="many_to_one"` and must
  preserve the exact input row count.
- Output, manifest, and optional old/new comparison are different explicit files.
  Existing files are not replaced unless `--force` is supplied.

## Adoption decision

A separately executed repeated OOF test for the denominator-correction family
reported `2023R +3.5`, `2024R -4.7`, and mean `-0.6` (SD `2.4`). That is not
evidence to replace the current champion prior. Also, that experiment did not
exercise every stronger per-feature support invariant in this v2 implementation.

Treat the artifact as a correctness/audit candidate. Run the exact final pipeline
and require time-forward improvement before changing any model default.

## Full-data audit run

The implementation was run against the repository's full
`data/trackman_history.csv`, with a legacy prior reproduced from the same input:

- raw rows: 1,793,078
- supported pitch-group rows: 1,770,780; ignored groups: 22,298
- pitcher-season summary rows: 2,589
- output rows: 3,998, exactly matching the legacy key set
- rows by target season: 394 / 535 / 632 / 727 / 804 / 906 for 2020-2025
- duplicate output keys, infinite values, physical-range failures: zero

The stronger feature-specific definition materially changes the sparse
offspeed history: 497 offspeed-velocity cells differ by more than 1% (198 by
more than 10%), and 1,108 offspeed-spin cells differ by more than 1% (267 by
more than 10%). `tm_velo_sep` changes by more than 1% in 1,620 cells. This is
why the candidate needs its own exact-pipeline OOF run; correctness alone is
not a model-adoption argument.

The generated CSV, comparison JSON, and manifest live under `artifacts/`, which
is ignored by Git. They can be regenerated with the command below.

## Example

Run from `work/` in an environment with Pandas and a Parquet engine:

```powershell
python .\codex_v2\trackman_prior_v2.py `
  --trackman .\trackman.parquet `
  --output .\codex_v2\artifacts\tm_prior_v2.parquet `
  --reference-prior .\tm_prior.parquet `
  --manifest .\codex_v2\artifacts\tm_prior_v2.manifest.json `
  --comparison-report .\codex_v2\artifacts\tm_prior_v2.comparison.json `
  --main-data .\train.parquet `
  --id-map .\idmap_pitcher_v2.parquet
```

The command never edits its input files. Re-running against an existing output
fails unless `--force` is explicitly provided.

## Tests

The tests use CSV fixtures, so a Parquet engine is not required:

```powershell
python -m unittest discover -s .\codex_v2\tests -v
```
