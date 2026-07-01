# Model Card — LGD Model Suite (Ch.3)

## 1. Identification

| | |
|---|---|
| Script | `04_lgd_models.py` |
| Model type | Four LGD estimators compared champion/challenger: Fractional Response (FRM), Natural Spline Regression, Random Forest, XGBoost Regressor |
| Chapter | Ch.3 (Sexton 2022 replication) |
| Version | v1.0 — see `git log -- 04_lgd_models.py` for change history |
| Model tier | `TBD` |

## 2. Purpose & Intended Use

Estimates Loss Given Default for defaulted single-family mortgages, for use in `ECL = PD × LGD × EAD` under IFRS 9 / Basel IRB. Four candidate models are trained and compared on the same target so an institution can select a champion based on its own bias/variance tradeoff preference (FRM is the most auditable; XGBoost typically the most accurate).

**Feeds `07_macro_scenario_analysis.py`:** the lowest-OOS-RMSE model (falling back to Train RMSE for small samples) is selected as champion by `select_champion()`, and its mean predicted LGD is written to `lgd_champion_summary.csv`. The macro scenario ECL engine reads this as its `base_lgd` anchor and scales it per scenario-quarter with `scenario_lgd()` (recovery scales with HPI), instead of a flat assumption disconnected from this model suite's output. This is still a **population-level** anchor, not a per-loan LGD — see Key Assumptions below for why, and the README's "Known Limitations" #4 for what a fuller fix would require.

## 3. Methodology

| Model | Key property | Hyperparameters |
|---|---|---|
| Fractional Response (FRM) | OLS on `logit(LGD)` → sigmoid predictions always in (0,1) | — |
| Natural Spline Regression | Cubic splines, non-linearity without overfitting | `n_knots=5, degree=3` |
| Random Forest | | `n_estimators=200, max_depth=6, min_samples_leaf=5` |
| XGBoost Regressor | | `n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8` |

Compared on RMSE, MAE, R², and mean bias. `select_champion()` then picks the model with the lowest RMSE on OOS (Train if OOS is empty) and records its mean predicted LGD on that split — the value consumed downstream by Ch.6. Full derivation in the [README's Ch.3 section](../../README.md#ch3--lgd-models).

## 4. Development Data

- **Population:** defaulted loans only (`zero_balance_code ∈ config.DEFAULT_CODES`) — one row per loan, the final servicer observation at resolution
- **Target:** `lgd = actual_loss / zero_balance_removal_upb`, clipped to [0, 1]
- **Sample size:** approximately 150 defaults in the sample dataset — this is small enough to materially affect all four models' variance; pre-2010 crisis vintages are recommended to increase the usable default count
- **Features:** `config.LGD_FEATURES` (17 features — HPI change since origination, mortgage insurance %, CLTV, DTI, current interest rate, property/loan characteristics)
- **Split:** same train/OOS/OOT structure as the PD models (`config.OOT_CUTOFF`, `config.OOS_FRAC`), applied to the defaulted-loan population

## 5. Key Assumptions

- **Sample size is the dominant limitation** — ~150 defaults means all four models, including the tree-based ones, are trained on a dataset small enough that performance estimates themselves carry meaningful uncertainty. Do not treat the RMSE/MAE comparison between models as decisive without a larger default sample (e.g. by including more crisis-era vintages).
- FRM assumes a fractional-logit link is an adequate functional form; the spline model assumes 5 knots is sufficient flexibility without overfitting the small sample; RF/XGBoost assume the ~150-row training set is enough for tree-based methods to generalize, which is optimistic at that sample size.
- LGD is clipped to [0, 1] post-hoc — any economic loss above 100% of UPB (e.g. from legal/foreclosure costs) is truncated rather than modelled.
- **The macro ECL anchor is population-level, not per-loan.** The PD population `07_macro_scenario_analysis.py` scores doesn't carry this suite's LGD-specific features (`hpi_change_since_orig`, `mi_pct`, `current_interest_rate`, etc.) — `pd_train/oos/oot.parquet` only persist `config.PD_FEATURES`. So every loan in a given scenario-quarter gets the same champion-derived LGD, scaled only by that scenario-quarter's aggregate HPI path, not by each loan's own CLTV/collateral trajectory. Closing this gap fully would mean persisting `config.LGD_FEATURES` alongside the PD population in `01_data_preprocessing.py` — a bigger, schema-changing fix than this one.

## 6. Performance

RMSE, MAE, R², and mean bias per model — see `data/processed/` outputs after running the script (not reproduced here since they depend on which vintages/sample size the training population was built with; the README's Ch.3 section documents the comparison methodology rather than fixed numbers given the small-sample caveat above).

## 7. Validation

- Same OOS/OOT backtesting structure as the PD models
- Four models compared side by side rather than a single model reported in isolation — a form of challenger-model validation appropriate given the small development sample

## 8. Ongoing Monitoring Plan

Not currently covered by `09_monitoring.py`, which monitors the PD feature set (`config.PD_FEATURES`) and PD scores only. Before production use, extend monitoring to LGD-specific drift: `config.LGD_FEATURES` distribution vs. the training reference, and realized LGD vs. predicted LGD on newly-resolved defaults (a directly observable actual-vs-expected check that PD monitoring cannot do, since PD outcomes take 12 months to mature but LGD outcomes are known at resolution).

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
| v1.0 | Initial replication of Sexton (2022) Ch.3, four-model LGD comparison |
| v1.1 | Added `select_champion()` and `lgd_champion_summary.csv`, wiring this suite's output into the Ch.6 macro scenario ECL engine (previously a fixed, disconnected LGD assumption) |
