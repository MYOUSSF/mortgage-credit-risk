# Model Card — Survival: Cox Proportional Hazards (Ch.5)

> ## ⚠️ SUPERSEDED — do not use for new work
>
> This model is superseded by the discrete-time survival framework in `11_discrete_hazard.py`
> ([logit card](discrete_hazard_logit.md) · [XGBoost card](discrete_hazard_xgb.md)).
> It is retained to document the approach and as a reference point.
>
> **Two confirmed defects:**
>
> 1. **Covariate leakage.** `build_survival_df()` collapses the loan-month panel to one row
>    per loan — the row with the maximum `loan_age` — and reads the time-varying covariates
>    (`delinquency_indicator`, `hpi_change`, `ur_3m_lag`) off that last row. For a defaulter,
>    that row is the month immediately before default, so the model is given the borrower's
>    state at the brink of default and asked to predict default. `delinquency_indicator` is the
>    worst offender: on the last pre-default row it is close to a default flag. **The C-index
>    reported in §6 below is inflated by construction and is not a forecast performance
>    estimate** — it is retained only so the defect is visible, and it should not be quoted.
> 2. **Horizon PDs are not conditional on current age.** `compute_horizon_pds()` returns
>    `1 − S(h|x)` — the probability of default within `h` months *measured from origination* —
>    and assigns it to a loan that has already survived to age `a`. The correct conditional
>    quantity is `1 − S(a+h|x) / S(a|x)`. The figure produced here overstates PD for any
>    seasoned loan, because it charges the loan again for default risk it has already survived.
>
> Both are fixed in `11_discrete_hazard.py`, which fits the monthly hazard on the panel as it
> stands (each row carrying its own contemporaneous covariates) and builds horizon PDs as a
> product of one-period survival probabilities starting at the loan's current age.
>
> `10_basel_irb_capital.py` now prefers the discrete-hazard TTC PD over this model's output,
> falling back here only if neither Ch.7 nor Ch.5b has been run.


## 1. Identification

| | |
|---|---|
| Script | `06_survival_analysis.py` |
| Model type | Cox Proportional Hazards (`lifelines.CoxPHFitter`) |
| Chapter | Ch.5 (extension beyond the Sexton 2022 thesis) |
| Version | v1.0 — see `git log -- 06_survival_analysis.py` for change history |
| Model tier | `TBD` |

## 2. Purpose & Intended Use

Models time-to-default with proper right-censoring for loans that haven't defaulted by the observation cutoff (rather than treating them as confirmed non-defaulters, which biases a binary 12-month model). Produces:
- Multi-horizon PD (12m / 24m / 36m / lifetime) for a single loan from one fitted model, instead of a separate model per horizon — the lifetime figure is the natural input to the *lifetime* leg of `07_macro_scenario_analysis.py`'s Stage 2/3 staged ECL (the actual Stage 1/2/3 classification — the SICR test and DPD backstops — lives in that script, not here)
- **Point-in-time (PIT) PD** (`compute_horizon_pds()`) — scored on each loan's actual current macro state, feeding IFRS 9 provisioning
- **Through-the-cycle (TTC) PD** (`compute_ttc_pds()`) — the same fitted model re-scored with macro covariates held at their long-run average, feeding Basel IRB capital (`10_basel_irb_capital.py`) via `08_calibration.py`'s further LRADR anchoring

**Not intended for:** Stage 1 (12-month) PD in place of [pd_logistic_regression.md](pd_logistic_regression.md) / [pd_xgboost.md](pd_xgboost.md) — those are the validated 12-month champion/challenger; this model's comparative advantage is the *lifetime* horizon those two cannot produce. Not intended for use where the proportional hazards assumption (below) has failed the Schoenfeld residual test for a feature relevant to the population being scored.

## 3. Methodology

```
h(t | x) = h₀(t) · exp(x'β)
```

Fitted on `config.COX_FEATURES` — a numeric-only subset of `config.PD_FEATURES` (Cox regression requires numeric covariates; categoricals like `occupancy_status`/`property_type` are excluded rather than encoded). Full derivation and the binary-vs-Cox comparison in the [README's Ch.5 section](../../README.md#ch5--survival-analysis).

## 4. Development Data

Same population and split as the PD models (`config.OOT_CUTOFF`, `config.OOS_FRAC`), scored via `config.COX_FEATURES` (9 numeric features: credit score, CLTV, DTI, interest rate, UPB, HPI change, unemployment lag, number of borrowers, delinquency indicator).

**Duration variable:** `loan_age` at the last observed report date. This is a known approximation, not each loan's true full performance history — see Key Assumptions below.

## 5. Key Assumptions

- **Proportional hazards assumption** — the model assumes each covariate's effect on the hazard is constant over time (a fixed hazard ratio, not one that grows or shrinks with loan age). This is validated via Schoenfeld residuals, not assumed blindly, but any covariate that fails the test in a given re-fit means that covariate's coefficient should not be trusted for long-horizon extrapolation.
- **Duration truncation** — using `loan_age` at last observation means a loan observed for only a few months contributes only that partial history to the likelihood; loans whose true multi-year performance is truncated by the data window may be undercounted relative to a dataset with longer follow-up per loan.
- Right-censoring is assumed **non-informative** (a loan's exit from observation is unrelated to its unobserved future default risk) — reasonable for an OOT cutoff-driven censoring mechanism, but would not hold if, say, loans were selectively removed from servicing data for reasons correlated with credit risk.

## 6. Performance

| Split | Concordance (≈ AUROC) | Status |
|---|---|---|
| OOS | ~0.85–0.89 | **INVALID — do not quote** |

**This figure must not be used.** It is inflated by the covariate leakage described in the banner
at the top of this card: the model is scored on each loan's last pre-default row, where
`delinquency_indicator` is close to a default flag. A leakage-free concordance for this model was
never measured, and the model is superseded rather than re-measured.

For honest, like-for-like performance on an identical population, see the discrete-time hazard
cards ([logit](discrete_hazard_logit.md) · [XGBoost](discrete_hazard_xgb.md)) and
`data/processed/discrete_hazard_comparison.csv`, which scores both hazard models and the Ch.2
12-month benchmark on the same fully-observed snapshot rows.

## 7. Validation

- Concordance index (survival analogue of AUROC) on OOS
- Schoenfeld residual test for the proportional hazards assumption — this is the validation step specific to Cox models that a binary classifier's validation plan does not need, and skipping it is the most common way a Cox PH model silently fails in production
- Kaplan-Meier curves by credit score / vintage as a non-parametric cross-check against the fitted hazard

## 8. Ongoing Monitoring Plan

Not currently covered by `09_monitoring.py` (PD-feature and PD-score focused). Before production use, extend monitoring with: (1) `config.COX_FEATURES` distribution PSI vs. training reference, (2) a periodic re-run of the Schoenfeld residual test as new data accumulates — the proportional hazards assumption can degrade over time even if it held at model-build time, and this is a check specific to survival models that no generic PSI monitor would catch, (3) concordance on rolling OOT windows.

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
| v1.0 | Initial Cox PH extension — multi-horizon PD and IFRS 9 Stage 2 lifetime PD |
| v1.1 | Added `compute_ttc_pds()` — macro-neutral re-scoring producing an actual through-the-cycle PD alongside the existing point-in-time horizon PD, for Basel IRB capital use |
| v1.2 | **Superseded** by `11_discrete_hazard.py`. Disclosed the end-of-follow-up covariate leakage and the non-conditional horizon PDs; marked the reported C-index invalid. Model retained for reference only; `10_basel_irb_capital.py` now prefers the discrete-hazard TTC PD over this model's output. |
