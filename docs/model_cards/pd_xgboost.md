# Model Card — PD: XGBoost (Ch.2)

## 1. Identification

| | |
|---|---|
| Script | `03_pd_ensemble.py` |
| Model type | Gradient-boosted trees (`XGBClassifier`, histogram method, GPU-accelerated) |
| Chapter | Ch.2 (Sexton 2022 replication) |
| Version | v1.0 — see `git log -- 03_pd_ensemble.py` for change history |
| Model tier | `TBD` — typically higher materiality than [pd_logistic_regression.md](pd_logistic_regression.md) if used for capital/provisioning rather than as a challenger |

## 2. Purpose & Intended Use

Higher-discrimination challenger PD model for the same 12-month default target as [pd_logistic_regression.md](pd_logistic_regression.md). Feeds two downstream uses directly:
- **SHAP loan-level attribution** (Ch.4, `05_shap_explanations.py`) — BCBS 239 Principles 6 & 11 individual explanations are computed against *this* model's score, not the logistic regression's
- **IFRS 9 macro scenario ECL** (Ch.6, `07_macro_scenario_analysis.py`) — the quarterly conditional-PD engine retrains and rescoring against macro paths uses this model's feature set and architecture

**Not intended for:** use as a calibrated probability without the recalibration layer in [calibration.md](calibration.md) — `scale_pos_weight=155` class rebalancing is known to distort raw predicted probabilities even though it improves rank-ordering; individual adverse-action decisions without the SHAP waterfall in Ch.4, since raw feature importances from a boosted tree are not a compliant substitute for BCBS 239-style attribution.

## 3. Methodology

```python
XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    scale_pos_weight=155, tree_method="hist",
    device="cuda",              # auto-detected via config.detect_gpu(); CPU fallback
    early_stopping_rounds=20,
)
```
Full derivation in the [README's Ch.2 section](../../README.md#ch2--xgboost-pd).

## 4. Development Data

Same population, target, and split as [pd_logistic_regression.md](pd_logistic_regression.md) (`config.DEFAULT_CODES`, `config.OOT_CUTOFF`, `config.PD_FEATURES`), with one difference: training uses a stratified **20% subsample** of the training parquet (`TRAIN_SAMPLE_FRAC = 0.20`, ~5M of ~24M rows) to keep peak RAM within Kaggle's 30 GB limit. This assumes the subsample is representative of the full training population — not separately re-verified against the full 24M rows in this repo.

## 5. Key Assumptions

- `scale_pos_weight=155` (the neg/pos ratio) upweights the minority (default) class during training — this improves AUROC/KS/Gini but means the raw output is **not** a calibrated probability; treat it as a risk score until passed through [calibration.md](calibration.md).
- No hyperparameter tuning was performed (fixed `max_depth=6`, `learning_rate=0.05`, `n_estimators=500` with early stopping) — a cross-validated grid search could plausibly improve OOT AUROC by 1–3 points (see the README's "Known Limitations").
- GPU/CPU device selection happens automatically via `config.detect_gpu()`; results should be numerically close but not bit-identical between `tree_method="hist"` on GPU vs CPU.

## 6. Performance

| Split | AUROC | KS | Gini |
|---|---|---|---|
| OOS | ~0.91 | ~0.64 | ~0.82 |

Outperforms the logistic regression baseline (~0.87 AUROC) by capturing non-linear FICO × CLTV × HPI interactions that WoE binning misses.

## 7. Validation

- Out-of-sample and out-of-time (2017–2024) backtesting, per OCC SR 11-7
- `early_stopping_rounds=20` uses the OOS set to halt training once AUC plateaus, guarding against overfitting to the training subsample
- Downstream SHAP attribution (Ch.4) serves as an additional validation layer — reviewing whether the top SHAP drivers per risk decile are directionally sensible is a standard sanity check for tree-based PD models

## 8. Ongoing Monitoring Plan

Run `09_monitoring.py` each time a new period lands — feature-level PSI on `config.PD_FEATURES` (shared with the LR model, so a drift alert here is relevant to both models), plus this model's own score PSI once `data/processed/pd_xgb_results.csv` exists. Because this model feeds the macro scenario ECL engine (Ch.6) directly, a "major shift" alert here should also trigger a re-check of any live IFRS 9 ECL output produced downstream.

Recommended cadence: `TBD` per institution's policy.

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
| v1.0 | Initial replication of Sexton (2022) Ch.2, GPU-accelerated XGBoost PD |
