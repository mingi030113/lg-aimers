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
- Batter-team OOF residual correction (`shrinkage=1000`): promising point
  estimates (+36.33 for 2023->2024 and +59.20 for 2024->2023), but rejected for
  now because there are only ten team clusters. Team-clustered 95% intervals
  include zero ([-35.27, +156.69] and [-29.01, +211.96]), and leave-one-team-out
  deltas can be negative. Preserve it as a research candidate; do not ship it.
