# Model Card — PD: Logistic Regression (Ch.1)

## 1. Identification

| | |
|---|---|
| Script | `02_pd_logistic_regression.py` |
| Model type | Binary logistic regression, L2-regularised, WoE-encoded inputs |
| Chapter | Ch.1 (Sexton 2022 replication) |
| Version | v1.0 — see `git log -- 02_pd_logistic_regression.py` for change history |
| Model tier | `TBD` — assign per institution's model risk policy (materiality typically driven by portfolio exposure this PD feeds into) |

## 2. Purpose & Intended Use

Estimates 12-month probability of default (`default_12m`) for single-family first-lien mortgages, for:
- Basel II/III IRB PD estimation (through-the-cycle, full 2000–2020 economic cycle)
- IFRS 9 Stage 1 (12-month ECL) PD input
- The **regulatory-facing champion** model — WoE bins and coefficients are directly inspectable, which [pd_xgboost.md](pd_xgboost.md) is not

**Not intended for:** individual loan-level adverse action decisions without the SHAP attribution in Ch.4 (which explains the XGBoost score, not this one); origination-time-only PD without refreshing `delinquency_indicator` / `loan_age` monthly; any portfolio outside conforming single-family US mortgages 2000–2020 vintages without re-validation.

## 3. Methodology

WoE encoding (15 quantile bins for continuous features, raw categories for `occupancy_status`/`property_type`) fitted **on the training set only**, then L2-regularised logistic regression (`C=1.0`, `class_weight='balanced'`, upweighting defaults ~155× for the ~0.64% base rate). Full derivation in the [README's Ch.1 section](../../README.md#ch1--logistic-regression-pd).

## 4. Development Data

- **Population:** Freddie Mac Single-Family Loan-Level Dataset, origination years 2000–2020
- **Target:** `default_12m` — `zero_balance_code ∈ {02, 03, 06, 09, 15}` (3rd-party sale, short sale, repurchase, REO, note sale) within a 12-month forward window; prepayments (01) and non-loss codes (16, 96) excluded. See `config.DEFAULT_CODES`.
- **Split:** Train 2000–2017 (70% of in-sample), OOS random 30% of in-sample, OOT 2017–2024 (temporal, never seen during fitting) — see `config.OOT_CUTOFF` / `config.OOS_FRAC`
- **Features:** `config.PD_FEATURES` (12 features — delinquency status, HPI change, credit score, CLTV, DTI, loan age, unemployment lag, etc.)

## 5. Key Assumptions

- WoE quantile binning assumes the target rate is reasonably monotonic within each of the 15 bins; a feature with a non-monotonic relationship to default would have its signal partially cancelled out.
- No interaction terms — the model assumes each WoE-transformed feature contributes additively to the log-odds. [pd_xgboost.md](pd_xgboost.md) exists specifically because this assumption loses the FICO × CLTV × HPI interactions XGBoost captures.
- `hpi_change` / `ur_3m_lag` depend on the optional macro data files; when absent, these features are entirely missing and the median-imputed WoE bin (0.0) is used for every row, silently reducing the model to the 10 non-macro features for that population.

## 6. Performance

| Split | AUROC | KS | Gini |
|---|---|---|---|
| OOS | ~0.87 | ~0.58 | ~0.74 |

Top features by Information Value: `delinquency_indicator` (0.538, very strong), `loan_age` (0.374, strong), `credit_score` (0.304, strong). Full IV table in the [README](../../README.md#ch1--logistic-regression-pd). Hosmer–Lemeshow calibration test is computed per split (`hl_pval`); a raw (uncalibrated) LR score should not be assumed well-calibrated out of the box — see [calibration.md](calibration.md).

## 7. Validation

- Out-of-sample (random 30% holdout) and out-of-time (2017–2024, never seen during fitting) backtesting, per OCC SR 11-7
- Hosmer–Lemeshow goodness-of-fit test at each split
- WoE/IV computed on train only and re-applied to OOS/OOT via fixed bin edges — verified leakage-proof in `tests/test_pd_logistic_regression.py` (unseen categories and out-of-range values fall back to a neutral WoE of 0.0 rather than raising or leaking)

## 8. Ongoing Monitoring Plan

Run `09_monitoring.py` each time a new period of servicing data lands:
- Feature-level PSI (`config.PD_FEATURES`) against the training reference — alert on PSI > 0.25 ("major shift")
- Score-level PSI on this model's output (`data/processed/pd_lr_results.csv`)
- Observed default rate vs. the training-set average

Recommended cadence: `TBD` per institution's monitoring policy (quarterly is typical for retail PD models under EBA GL/2017/16). Escalate any "major shift" flag to model owner before trusting scores from that period.

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
| v1.0 | Initial replication of Sexton (2022) Ch.1, WoE + logistic regression |
