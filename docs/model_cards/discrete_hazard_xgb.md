# Model Card — Discrete-Time Hazard: XGBoost (Ch.5b)

## 1. Identification

| | |
|---|---|
| Script | `11_discrete_hazard.py` |
| Model type | Discrete-time survival — gradient-boosted trees on the monthly hazard (`XGBClassifier`, `binary:logistic`) |
| Chapter | Ch.5b (replaces the Ch.5 Cox model) |
| Version | v1.0 — see `git log -- src/11_discrete_hazard.py` for change history |
| Model tier | `TBD` |
| Supersedes | [survival_cox.md](survival_cox.md) |
| Paired champion | [discrete_hazard_logit.md](discrete_hazard_logit.md) |

## 2. Purpose & Intended Use

The **challenger** to the discrete-time logistic hazard model: same target, same covariates, same case-control subsample and the same prior correction, so the two are directly comparable on identical rows. Its comparative advantage is that it makes **no proportional-hazards assumption** — `loan_age` enters as an ordinary feature, so the trees learn the baseline hazard *and its interactions with every other covariate*. The effect of `credit_score` is free to differ at month 6 and month 60 without anything being specified in advance.

Intended uses:

- **Challenger benchmark** for the linear hazard model — quantifying how much discrimination the linear-in-the-logit, proportional-hazards specification gives up.
- **Multi-horizon conditional PD** (PIT and TTC) through the identical interface, written to `discrete_hazard_xgb_pd_horizons.csv` with the same column schema.
- **Basel IRB TTC PD**, if and only if `config.DISCRETE_HAZARD_CAPITAL_MODEL` is set to `"xgb"`. It defaults to `"logit"`: capital is the most supervisory-scrutinised output in the pipeline, and odds ratios with confidence intervals are directly auditable in a way SHAP attributions on a 600-tree ensemble are not.

**Not intended for:** long-horizon or severe-scenario projection without explicit awareness of the flat-extrapolation limitation (§5) — this is the single most important reason both models are kept rather than just the better-discriminating one. Not intended for scoring where delinquency status is material (internal covariates are excluded — §5).

## 3. Methodology

```
logit h(t | x) = f(t, x)        f = gradient-boosted trees
```

- **No spline, no baseline separation** — `loan_age` is one feature among the others, and the age effect emerges from the tree structure. The learned seasoning effect is read back out via the SHAP dependence plot on `loan_age`.
- **Hyperparameters** — `config.DISCRETE_HAZARD_XGB_PARAMS`, `tree_method="hist"`, device from `config.detect_gpu()`. Early stopping on **OOS log loss** (30 rounds); the OOS split is loan-grouped by `split_pd()`, so no loan straddles fit and early-stopping sets.
- **No `scale_pos_weight`.** Class imbalance is already handled by the case-control subsampling, and reweighting on top of it would distort the predicted probabilities. That matters more here than in a pure ranking application, because the horizon PD multiplies 12–60 of these probabilities together, so a systematic level error compounds. The prior correction handles the level; weights would corrupt it. This is asserted in the tests.
- **Prior correction** — identical to the linear model, applied to the raw output **margin** before the sigmoid: `logit(h_true) = margin + log(r)`. The King & Zeng correction holds for any estimator of the sampled-population conditional probability, which is why one shared function serves both models.
- **Monotone constraints** (`config.DISCRETE_HAZARD_USE_MONOTONE_CONSTRAINTS`, on by default). Directions are economic, not fitted:

  | Covariate | Direction on the hazard |
  |---|---|
  | `credit_score` | decreasing (−1) |
  | `orig_cltv` | increasing (+1) |
  | `orig_dti` | increasing (+1) |
  | `ur_3m_lag` | increasing (+1) |

  Rationale: **auditability** — a credit committee can be told the model cannot assert "higher FICO, higher risk" on some thin slice of the data — and **more stable extrapolation** at the edges of the training range, which both the lifetime horizon and stressed macro paths reach. Unlisted covariates are unconstrained. A test sweeps each constrained covariate across a synthetic grid and asserts the predicted hazard is monotone.
- **Encoding** — categoricals ordinal-encoded from the training levels, with unseen levels mapped to the first training level (the convention `03_pd_ensemble.py` already uses).

## 4. Development Data

Identical to the [logit card's §4](discrete_hazard_logit.md#4-development-data) — same panel, same split, same target construction, same covariate set (`config.DISCRETE_HAZARD_FEATURES`), same excluded internal covariates, and the **same case-control subsample rows and seed**, so the two models differ only in functional form.

Requires `default_date` in the PD parquet files — see the logit card for why.

## 5. Key Assumptions

- **No proportional-hazards assumption.** This is the model's main advantage over both the linear hazard model and the superseded Cox model. There is correspondingly no PH test to run for it.
- **Trees extrapolate FLAT — the key limitation.** Beyond the oldest `loan_age` or the most extreme `ur_3m_lag` seen in training, the predicted hazard stops responding: a 300-month projection and a 60-month projection see the same hazard once past the training range, and a macro shock more severe than anything in the training window is treated as if it were the worst observed. The linear model extrapolates linearly in the logit (with a constant-extrapolated spline baseline). Neither is right, but they are **wrong differently**, which is the reason both are maintained. This matters specifically for: (a) the lifetime PD, which projects to `config.DISCRETE_HAZARD_LIFETIME_CAP_MONTHS`; (b) any future IFRS 9 severe-scenario projection through `compute_conditional_horizon_pd()`.
- **Internal delinquency covariates excluded** — identical reasoning and identical disclosed cost as the [logit card's §5](discrete_hazard_logit.md#5-key-assumptions).
- **Flat PIT macro projection**, **non-informative censoring**, and the **undistorted-sampled-probability** requirement of the prior correction — all as in the logit card.
- **Monotone constraints are an imposed prior, not a finding.** If the data genuinely contained a non-monotone relationship in a constrained covariate, the constrained model would not represent it. The directions above are standard credit-risk economics, but they are an assumption.

## 6. Performance

`TBD — regenerate after running on the full Freddie Mac panel.`

Produced by `11_discrete_hazard.py` into `discrete_hazard_xgb_metrics.csv` and the shared `discrete_hazard_comparison.csv`. Nothing is quoted until that run has happened on the real data.

| Metric set | Split | Metric | Value |
|---|---|---|---|
| Monthly hazard | OOS | AUROC / log loss / predicted÷observed | `TBD` |
| Monthly hazard | OOT | AUROC / log loss / predicted÷observed | `TBD` |
| 12m conditional PD | OOT snapshot | AUROC / KS / Gini / Brier | `TBD` |
| vs. linear hazard model (identical rows) | OOT snapshot | Δ AUROC / Δ Brier | `TBD` |
| vs. Ch.2 12m XGBoost (identical rows) | OOT snapshot | Δ AUROC / Δ Brier | `TBD` |

## 7. Validation

Run identically to the linear model, on the same rows, so the comparison table is like-for-like — see the [logit card's §7](discrete_hazard_logit.md#7-validation) for the full description of each check. In summary: monthly-hazard AUROC and log loss on unsampled OOS/OOT rows; 12-month conditional PD against realised outcomes on fully-observed snapshot rows (AUROC/KS/Gini/Brier + decile calibration); the Ch.2 12-month benchmark on exactly those rows; and observed vs predicted Kaplan-Meier over loan age.

Model-specific additions:

- **Global SHAP importance** (`discrete_hazard_xgb_shap_importance.png`) — exact TreeSHAP in log-odds units.
- **SHAP dependence on `loan_age`** (`discrete_hazard_xgb_shap_dependence_loan_age.png`) — the *learned* seasoning effect, the direct counterpart of the linear model's spline baseline α(t). Comparing the two is a check that the tree model has discovered the same seasoning shape rather than fitting noise.
- **Monotonicity check** — each constrained covariate swept across a grid; asserted in the test suite rather than only inspected.

> **Note on SHAP tooling:** `compute_xgb_shap()` tries `shap.TreeExplainer` first (matching `05_shap_explanations.py`), then falls back to XGBoost's native `pred_contribs`, which is the same TreeSHAP algorithm implemented inside XGBoost. The fallback exists because `shap ≤ 0.49` cannot parse the bracketed `base_score` that XGBoost ≥ 3.0 writes into its model config. Values are identical either way; only the caller differs.

## 8. Ongoing Monitoring Plan

As the [logit card's §8](discrete_hazard_logit.md#8-ongoing-monitoring-plan) (feature PSI, realised vs predicted hazard by vintage and age bucket, snapshot backtesting), **except** that the proportional-hazards re-test does not apply. Model-specific additions:

1. **SHAP importance stability** — a material re-ordering of the top covariates between cycles is a drift signal a PSI monitor on input distributions would not catch.
2. **Monotonicity re-assertion after every re-fit** — a constraint silently dropped from the params (e.g. via a hyperparameter edit) would not fail any accuracy check.
3. **Training-range coverage check** — track the share of scored loans whose `loan_age` or macro covariates fall outside the training range, where the flat-extrapolation limitation binds. Rising coverage gaps are the trigger to re-fit rather than continue scoring.

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
| v1.0 | Initial XGBoost discrete-time hazard challenger. Same target, covariates, subsample and prior correction as the linear model; adds monotone constraints, SHAP-based seasoning inspection, and no proportional-hazards assumption. |
