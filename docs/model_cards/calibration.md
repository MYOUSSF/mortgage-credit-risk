# Model Card — PD Calibration (Ch.7)

## 1. Identification

| | |
|---|---|
| Script | `08_calibration.py` |
| Model type | Post-hoc recalibration layer: Platt scaling, isotonic regression, temperature scaling |
| Chapter | Ch.7 (extension beyond the Sexton 2022 thesis) |
| Version | v1.0 — see `git log -- 08_calibration.py` for change history |
| Model tier | `TBD` — typically inherits the tier of whichever PD model it recalibrates |

## 2. Purpose & Intended Use

Recalibrates a PD model's raw score so predicted probabilities match observed default rates — a distinct property from rank-ordering (AUROC/KS/Gini), which the underlying PD models are validated on separately. Applies to **whichever of [pd_logistic_regression.md](pd_logistic_regression.md) / [pd_xgboost.md](pd_xgboost.md) has produced score outputs** (`pd_lr_results.csv` / `pd_xgb_results.csv`) — this is a recalibration layer, not a standalone predictive model, and it is only as good as the PD model feeding it.

**Why this matters specifically for the XGBoost model:** `scale_pos_weight=155` class rebalancing (see [pd_xgboost.md](pd_xgboost.md)) is known to systematically inflate raw predicted probabilities even when rank-ordering is excellent — this is the concrete failure mode this calibration layer exists to correct before any PD is used in `ECL = PD × LGD × EAD`.

**Not intended for:** correcting a model with poor discrimination — calibration adjusts the *level* of predicted probabilities, not their *rank order*; a poorly-discriminating model calibrated to match the population default rate is still a poor model.

## 3. Methodology

| Method | Formula | When to use |
|---|---|---|
| Platt scaling | `P_cal = σ(a·s + b)` | Default choice — stable, auditable |
| Isotonic regression | Non-parametric monotone | Large datasets (500+ events) |
| Temperature scaling | `P_cal = σ(logit(P) / T)` | When over-confidence is the problem |

Fitted on a calibration split, evaluated on a held-out OOS-eval split and OOT. Full derivation in the [README's Ch.7 section](../../README.md#ch7--pd-calibration).

## 4. Development Data

Uses the score outputs of whichever PD model(s) have already been run (`data/processed/pd_lr_results.csv`, `data/processed/pd_xgb_results.csv`) — no independent development population; its "training data" is another model's OOS scores plus the same `default_12m` target.

## 5. Key Assumptions

- The calibration mapping is fitted on one time period (the calibration split) and assumed to remain valid going forward. A macroeconomic regime shift (e.g. a recession) that changes the true default rate at a given score level would break this assumption immediately — which is exactly what the [PSI/score monitoring](../../README.md#ch8--ongoing-model-monitoring) is meant to catch before it goes unnoticed.
- Isotonic regression can overfit the calibration curve at the observed sample's specific event count (documented threshold: prefer isotonic only with 500+ events; use Platt below that).
- Platt scaling assumes a sigmoid is the correct functional form linking raw score to calibrated probability — usually a reasonable assumption for logistic-regression scores, less obviously so for tree-ensemble scores, which is why temperature scaling and isotonic regression are compared side by side rather than assuming Platt is always right.

## 6. Performance

Reported per PD model calibrated (LR and/or XGB, whichever are available), via: Brier score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Hosmer–Lemeshow p-value, and LRADR (Long-Run Average Default Rate) comparison against Basel II §461's through-the-cycle benchmark. See `data/processed/calibration_metrics.csv` after running the script.

## 7. Validation

- Reliability diagrams (actual default rate vs. mean predicted PD per decile), pre- and post-calibration, on both an OOS-eval split and OOT
- LRADR regulatory summary comparing observed default rate to calibrated mean PD, per method, per split — the standard regulatory report format for demonstrating a PD model is neither systematically over- nor under-predicting
- **TTC cycle adjustment** (`compute_ttc_pd_via_lradr()`): logit-shifts the Platt-calibrated PIT PD so the population mean matches LRADR, producing an actual per-loan TTC PD (`ttc_calibrated_pd.csv`) rather than leaving LRADR as a diagnostic gap — consumed by `10_basel_irb_capital.py` as the preferred TTC PD source for the Basel IRB capital formula

## 8. Ongoing Monitoring Plan

The calibration mapping itself should be re-validated whenever the underlying PD model's score-level PSI (from `09_monitoring.py`) flags drift — a stable score distribution with a stale calibration mapping is a plausible failure mode that pure PSI monitoring on the raw score would not by itself detect. Recommended: re-run the Brier/ECE/HL comparison each monitoring cycle, not just once at model build.

## 9. Governance

| Field | Value |
|---|---|
| Model owner | `TBD` |
| Independent validator | `TBD` |
| Approval date | `TBD` |
| Next scheduled review | `TBD` |

## 10. Change Log

| Version | Change |
|---|---|
| v1.0 | Initial calibration extension — Platt / isotonic / temperature scaling, LRADR comparison |
| v1.1 | Added `compute_ttc_pd_via_lradr()` — LRADR-anchored TTC cycle adjustment producing a per-loan capital PD, not just a diagnostic |
