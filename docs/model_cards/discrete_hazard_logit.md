# Model Card — Discrete-Time Hazard: Logistic Regression (Ch.5b)

## 1. Identification

| | |
|---|---|
| Script | `11_discrete_hazard.py` |
| Model type | Discrete-time survival — logistic regression on the monthly hazard (`statsmodels` GLM, Binomial family, logit link) |
| Chapter | Ch.5b (replaces the Ch.5 Cox model) |
| Version | v1.0 — see `git log -- src/11_discrete_hazard.py` for change history |
| Model tier | `TBD` |
| Supersedes | [survival_cox.md](survival_cox.md) |
| Paired challenger | [discrete_hazard_xgb.md](discrete_hazard_xgb.md) |

## 2. Purpose & Intended Use

Estimates the **monthly default hazard** `h(t | x) = P(default in month t | survived to t, x)` directly on the loan-month panel, and composes it into a **PD conditional on each loan's current age** at any horizon:

```
PD(a → a+h) = 1 − ∏_{j=1..h} [1 − ĥ(a + j | x, macro_j)]
```

Intended uses:

- **Multi-horizon conditional PD** (12m / 24m / 36m / 60m / lifetime) from one fitted model — the lifetime figure feeds the *lifetime* leg of IFRS 9 Stage 2/3 ECL.
- **Point-in-time (PIT) PD** — each loan's current macro state held flat across the horizon, for IFRS 9 provisioning.
- **Through-the-cycle (TTC) PD** — macro covariates pinned at their long-run training-sample mean, for Basel IRB capital (EBA/GL/2017/16 §6.2). This is the default capital source (`config.DISCRETE_HAZARD_CAPITAL_MODEL = "logit"`), chosen over the XGBoost challenger because capital is the most supervisory-scrutinised output in the pipeline and this model's coefficients and odds ratios are directly auditable.

**Not intended for:** replacing the Ch.1/Ch.2 12-month classifiers as the 12-month champion without a formal champion/challenger comparison — `data/processed/discrete_hazard_comparison.csv` provides that comparison on identical rows, but the decision is a governance one. Not intended for scoring a loan whose delinquency status is material to the decision: internal delinquency covariates are deliberately excluded (§5), so the model cannot distinguish a current loan from a 60-days-past-due one with identical origination characteristics.

## 3. Methodology

```
logit h(t | x) = α(t) + x'β
```

- **α(t)** — the baseline hazard, a B-spline basis on `loan_age` with `config.DISCRETE_HAZARD_SPLINE_DF` (6) degrees of freedom, knots on training-set age quantiles. This is the discrete-time analogue of Cox's non-parametric `h₀(t)`: it traces the mortgage seasoning ramp without imposing a parametric shape. Extrapolation is **constant**, so projecting past the oldest observed age holds the baseline flat rather than letting an unconstrained cubic diverge.
- **Logit link, not cloglog** — the monthly hazard is ~0.05%, and at that magnitude the two links are numerically indistinguishable. Logit is chosen because the case-control prior correction (below) is *exact* under it.
- **Why no data expansion** — the panel likelihood factorises into one Bernoulli term per loan-month at risk (Allison 1982; Singer & Willett 1993), so fitting a binary model to the panel *is* maximum likelihood for the discrete-time survival model. Each existing loan-month row is one observation.
- **Estimation** — unpenalised GLM. Unpenalised because the likelihood-ratio proportional-hazards test requires it, and with millions of rows against ~20 parameters there is nothing for a penalty to stabilise.
- **Case-control subsampling + prior correction** — every event row is kept; non-event rows are kept with probability `r` (`config.DISCRETE_HAZARD_SUBSAMPLE_RATE`). The King & Zeng (2001) correction `logit(h_true) = logit(h_sampled) + log(r)` is applied inside `predict_hazard()`, so every downstream consumer sees corrected probabilities. The shift is monotone — it changes the PD *level* (and therefore ECL and capital) but leaves AUROC, KS and Gini untouched.
- **Encoding** — continuous covariates median-imputed and standardised using training statistics only; coefficients reported back on the **original scale** (the back-transform `β_orig = β_std / s` is exact). Categoricals one-hot encoded with the alphabetically-first observed level as the documented reference, recorded in `discrete_hazard_logit_coefficients.csv`.

Full derivation in the [README's Ch.5b section](../../README.md#ch5b--discrete-time-survival-monthly-hazard).

## 4. Development Data

Same population and split as the PD models (`config.OOT_CUTOFF`, `config.OOS_FRAC`), read from `pd_{train,oos,oot}.parquet`.

**Requires `default_date` in the PD parquet files.** The target places the event on the single loan-month row immediately before default, which cannot be identified from `default_12m` (that flag marks ~12 rows per defaulter). `default_date` must also survive the split, because `split_pd()` cuts on `report_date` — a defaulting loan's rows can straddle `pd_train` and `pd_oot`, so its last row *within a file* is not necessarily the row before default.

**Covariates** (`config.DISCRETE_HAZARD_FEATURES`):

| Group | Covariates |
|---|---|
| Time | `loan_age` (as the spline baseline α(t)) |
| Static origination | `credit_score`, `orig_cltv`, `orig_dti`, `orig_interest_rate`, `orig_upb`, `num_borrowers`, `occupancy_status`, `property_type` |
| External macro | `ur_3m_lag`, `hpi_change` — taken at each row's own `report_date` |
| **Excluded** | `delinquency_indicator`, `delinquency_status` (`config.DISCRETE_HAZARD_EXCLUDED_FEATURES`) |

**Target:** `y = 1` on the last retained row of a loan whose `default_date` falls within `config.DISCRETE_HAZARD_NEXT_PERIOD_DAYS` (45 days); `y = 0` on all other at-risk rows, including every row of a censored loan. At most one event per loan is enforced by construction and asserted in the tests.

## 5. Key Assumptions

- **Proportional hazards (constant covariate effects over loan age).** The linear model constrains each covariate's effect to be constant across the life of the loan — the discrete-time counterpart of Cox's PH assumption. This is **tested, not assumed**: `proportional_hazards_lr_test()` fits a covariate × spline(`loan_age`) interaction and runs a likelihood-ratio test for `credit_score`, `orig_cltv` and `ur_3m_lag`, writing p-values to `discrete_hazard_logit_ph_tests.csv`. A rejected test means that covariate's long-horizon PD contribution should not be trusted — and is precisely the assumption the XGBoost challenger does not make.
- **Internal delinquency covariates are excluded, deliberately.** A multi-period horizon PD is a product over future months, so every covariate needs a projected value at every future month. Delinquency at month *a+j* is itself an outcome of the same deterioration process that produces default; projecting it would need a second model of delinquency transitions, and holding it flat would assert that a current loan stays current for 60 months (driving its lifetime PD towards zero). Including it would also reproduce the Cox model's leakage in subtler form, since delinquency on the row before default is near-deterministic in the target. **Cost, disclosed:** the model cannot distinguish a current from a seriously delinquent loan with identical origination characteristics. For IFRS 9 this gap is covered elsewhere — `07_macro_scenario_analysis.py`'s 30-DPD and 90-DPD backstops drive stage allocation on exactly that information.
- **PIT macro projection is flat.** "Conditions stay as they are today" — an explicit assumption, not a forecast. At long horizons a loan observed in a recession is projected as if the recession never ends, so 24m–60m PIT PDs are more dispersed across loans than a mean-reverting path would give.
- **Non-informative censoring** — a loan's exit from observation is unrelated to its unobserved future default risk. Reasonable for OOT-cutoff-driven censoring.
- **Sampling correction assumes the model estimates the sampled-population conditional probability** without distortion — which is why no class weighting is applied on top of the subsampling.

## 6. Performance

`TBD — regenerate after running on the full Freddie Mac panel.`

All performance figures are produced by `11_discrete_hazard.py` into `discrete_hazard_logit_metrics.csv` and the shared `discrete_hazard_comparison.csv`. Nothing is quoted here until that run has happened on the real data.

| Metric set | Split | Metric | Value |
|---|---|---|---|
| Monthly hazard | OOS | AUROC / log loss / predicted÷observed | `TBD` |
| Monthly hazard | OOT | AUROC / log loss / predicted÷observed | `TBD` |
| 12m conditional PD | OOT snapshot | AUROC / KS / Gini / Brier | `TBD` |
| Benchmark (Ch.2 12m XGBoost, identical rows) | OOT snapshot | AUROC / KS / Gini / Brier | `TBD` |

## 7. Validation

Only checks that are honest on this data are run — in particular, no concordance figure computed on end-of-follow-up covariates (the defect that invalidated the Cox card's C-index).

1. **Monthly hazard discrimination** — AUROC and log loss on **unsampled** OOS and OOT rows (uniformly sampled above `config.DISCRETE_HAZARD_MAX_EVAL_ROWS`, so the base rate is preserved). Log loss and the predicted÷observed ratio are the checks that confirm the prior correction worked; AUROC cannot, since the correction is monotone.
2. **Horizon validation** — the 12-month conditional PD against the realised 12-month outcome at fixed snapshot dates (`config.DISCRETE_HAZARD_SNAPSHOT_DATES`), restricted to rows whose 12-month window is **fully observed** (the same maturity rule as `filter_immature_right_censored()`). AUROC, KS, Gini, Brier and decile calibration.
3. **Benchmark on an identical population** — the Ch.2 12-month XGBoost scored on exactly the same snapshot rows. `03_pd_ensemble.py` persists no row keys in `pd_xgb_results.csv`, so the Ch.2 specification is re-fitted here to make the identical-population comparison possible (the same approach `07_macro_scenario_analysis.py` already takes); the metrics CSV records which path was used.
4. **Proportional hazards LR tests** — see §5.
5. **Observed vs predicted Kaplan-Meier** — the empirical panel hazard (events ÷ at-risk at each loan age) against the model's mean predicted hazard for the loans at risk at that age. Computed from the panel, so it does not inherit the per-loan-collapse bias of the Cox implementation.

## 8. Ongoing Monitoring Plan

Not currently covered by `09_monitoring.py` (PD-feature and PD-score focused). Before production use, extend monitoring with:

1. **PSI on `config.DISCRETE_HAZARD_FEATURES`** vs the training reference, using the fixed-reference-bin pattern `09_monitoring.py` already implements.
2. **Realised vs predicted monthly hazard by vintage and by loan-age bucket**, each period — the level check that a PSI monitor cannot catch. A drift in the predicted÷observed ratio is the first sign the prior correction or the base rate has moved.
3. **Periodic re-run of the proportional hazards LR tests** as data accumulates — the assumption can degrade over time even if it held at build time, and this is specific to the linear model; a generic monitor would not catch it.
4. **Seasoning-curve stability** — re-plot α(t) each cycle; a materially changed shape means the portfolio's seasoning behaviour has shifted.
5. **Backtest the 12-month conditional PD** on each newly-matured snapshot cohort, against the same metric set as §7.2.

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
| v1.0 | Initial discrete-time logistic hazard model, replacing the Ch.5 Cox model. Fits the monthly hazard on the loan-month panel (no end-of-follow-up covariate leakage) and produces horizon PDs conditional on each loan's current age. Adds case-control subsampling with the King & Zeng prior correction, a spline baseline hazard, proportional-hazards LR tests, and PIT/TTC macro paths. |
