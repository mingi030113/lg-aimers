# Codex v2 experiment plan

This directory is intentionally isolated from the actively edited source tree.
Nothing here should overwrite files under `work/` or write into the original
workspace. Every command must receive explicit input and output paths.

## Champion baseline

Keep the submitted two-MLP recipe as the champion until a challenger passes all
gates:

- train window: 2023--2024 (pre-2023 F excluded)
- recent ensemble: five-seed LightGBM + L2 logistic + MLP1 + quantile MLP2
- submitted effective weights: 0.39 / 0.21 / 0.20 / 0.20
- public score recorded in the source workspace: 952.62

## Gate 0: correctness before adoption

Rebuild the Trackman prior with a per-feature non-null weighted denominator and
validate uniqueness, row preservation, finite ranges, and feature order. The
corrected table is a correctness reference, not an automatic model upgrade.
The concurrent source experiment reported a repeat mean delta of -0.6 points
(2023 +3.5, 2024 -4.7), so corrected Trackman must be evaluated as a challenger
and must not silently replace the champion input.

## Gate 1: one end-to-end validator

The validator must exercise the same feature attachment, train-window rules,
member preprocessing, blend, and R/F calibration as the deployable recipe. It
must write OOF predictions and a machine-readable report for every fold.

Required evaluation views:

1. forward annual R folds ending in 2023 and 2024;
2. post-regime time blocks inside 2023 and 2024;
3. a separate 2024 F sanity table;
4. paired bootstrap uncertainty clustered by pitcher (and game when available);
5. weight transfer: choose a candidate weight on one block and apply it unchanged
   to another block.

## Challenger order

Run one isolated change at a time.

1. previous-season R pitcher empirical-Bayes prior;
2. corrected/recency-aware Trackman block;
3. time-decayed history branch, blended at only 0/0.05/0.10;
4. small cross-fitted residual GAM/spline model;
5. random-subspace MLP only after the earlier candidates are resolved.

The time-decayed branch must exclude pre-2023 F labels. Old target rows are never
added directly to the recent champion. The branch remains at weight zero unless
its correction improves both a forward fold and a held-out post-regime block.

## Acceptance rule

A challenger is eligible only when all conditions hold:

- positive paired score delta on both recent forward views;
- positive transferred-weight delta;
- no material F regression;
- deterministic rebuild and clean-bundle verification pass;
- the smaller weight is chosen when uncertainty intervals overlap.

Public leaderboard submissions are confirmation only, not a weight-search loop.

## Recorded challenger results

- Previous-season R pitcher EB prior (`k=500`): rejected as a direct residual
  correction to the current two-MLP OOF ensemble. The coefficient learned on
  2023 clipped to zero (raw coefficient -0.069). The coefficient learned on
  2024 was 0.028 and transferred to 2023 at -1.34 points, with pitcher-clustered
  95% interval [-4.78, +1.64]. Keep the implementation for reproducibility, but
  do not add the feature to the champion based on its strong standalone signal.
- Batter-team OOF residual correction v1 (`shrinkage=1000`, full weight): the
  initial annual-only audit was held back because there are only ten team
  clusters and its cluster intervals included zero. That result is retained as
  the conservative baseline, not treated as the final decision.
- Batter-team OOF residual correction v2 (`shrinkage=1000`, fixed weight 0.50):
  adopted as a submission challenger after expanding the validation to annual,
  chronological half-season, and forward transfer views. All 8/8 forward views
  improved. The key fixed-weight deltas were +39.33 (2023 all -> 2024 all),
  +54.50 (2023 early -> late), +45.16 (2023 late -> 2024 early), and +33.16
  (2024 early -> late); 47/50 leave-one-correction-out checks remained positive.
  The deployment table is fitted once from 2024 regular-season OOF, applied only
  to `game_type == "R"` after base calibration, and never uses test-row
  aggregation. Weight 0.75 had the stronger local point estimate, but 0.50 was
  selected before submission as the smaller transfer-stable weight. This is a
  local-validation decision, not a new public leaderboard score.
