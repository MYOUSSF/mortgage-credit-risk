# Model Card — LGD Model Suite (Ch.3)

## 1. Identification

| | |
|---|---|
| Script | `04_lgd_models.py` |
| Model type | Three LGD estimators compared champion/challenger: Fractional Response (FRM), Two-Stage (multinomial logit + beta regression), Random Forest |
| Chapter | Ch.3 (Sexton 2022 replication + extension) |
| Version | v2.0 — see `git log -- src/04_lgd_models.py` for change history |
| Model tier | `TBD` |

## 2. Purpose & Intended Use

Estimates Loss Given Default for defaulted single-family mortgages, for use in `ECL = PD × LGD × EAD` under IFRS 9 / Basel IRB. Three candidate models are trained and compared on the same target and the same rows, so an institution can select a champion on its own bias/variance preference.

The three are chosen to be *different in kind*, not merely in flexibility:

| Model | What it estimates | Why it's in the suite |
|---|---|---|
| **Fractional Response (FRM)** | `E[LGD\|X]` | The auditable baseline — a GLM whose coefficients a credit committee can read directly |
| **Two-Stage** | The full conditional *distribution* | The only model here that represents the point masses at LGD=0 and LGD=1; supplies quantiles for downturn-LGD work |
| **Random Forest** | `E[LGD\|X]` | Non-parametric challenger — captures interactions the FRM's linear index cannot |

**Feeds `07_macro_scenario_analysis.py`:** the lowest-OOS-RMSE model (falling back to Train RMSE for small samples) is selected as champion by `select_champion()`, and its mean predicted LGD is written to `lgd_champion_summary.csv`. The macro scenario ECL engine reads this as its `base_lgd` anchor and scales it per scenario-quarter with `scenario_lgd()` (recovery scales with HPI). **That file's schema is unchanged by the move from four models to three.** This is still a **population-level** anchor, not a per-loan LGD — see Key Assumptions below, and the README's "Known Limitations" #4.

**Not intended for:** a downturn LGD for Basel capital. All three models estimate expected/average-conditions LGD. The two-stage model's `predict_quantile()` is the intended *starting point* for downturn work but is not itself a validated downturn estimate — see Key Assumptions for the constant-precision caveat.

## 3. Methodology

| Model | Key property | Hyperparameters |
|---|---|---|
| Fractional Response (FRM) | Papke-Wooldridge (1996) quasi-binomial GLM, logit link. Targets `E[LGD\|X]` directly; the quasi-likelihood is well-defined at y=0 and y=1, so no boundary clipping is needed | L2 fallback: `config.LGD_FRM_L2_ALPHA` |
| Two-Stage | Stage 1: 3-class multinomial logit over {LGD=0, interior, LGD=1}. Stage 2: beta regression on interior rows, `logit(μ)=Xγ`, constant `log φ` | `config.LGD_TWO_STAGE_STAGE1_C`, `config.LGD_TWO_STAGE_BETA_L2_ALPHA` |
| Random Forest | Non-parametric interactions | `n_estimators=200, max_depth=6, min_samples_leaf=5` |

All three are fit with IPCW sample weights (`compute_ipcw_weights()` in `01_data_preprocessing.py`) correcting for workout-period truncation bias.

### The two-stage model

Realised LGD is not smooth on [0, 1]. It is bimodal with genuine point masses at exactly 0 (the disposition covers the balance) and exactly 1 (nothing recovered). A conditional-mean model can fit the average of that distribution well while assigning essentially zero probability to either mass — predicting a confident 0.45 for a population in which almost no loan loses 45%.

```
Stage 1:  P(class | x)  over {LGD=0, interior, LGD=1}
Stage 2:  E[LGD | x, interior] = μ(x),  y ~ Beta(μφ, (1-μ)φ)

E[LGD | x] = P(LGD=1|x)·1 + P(interior|x)·μ(x) + P(LGD=0|x)·0
```

`predict()` returns that conditional mean, so the model is directly comparable to the other two on RMSE/MAE/R²/bias and in champion selection. `predict_components()` returns `P0, P_interior, P1, μ, φ` per row, and `predict_quantile(X, q)` inverts the mixture CDF (point mass at 0, beta on (0,1), point mass at 1).

**Why the beta regression is hand-written.** `statsmodels.othermod.betareg.BetaModel` does not support observation weights. Critically, it *accepts* a `weights=` keyword, emits only a `ValueWarning`, and returns bit-identical unweighted parameters — and this script suppresses warnings, so the failure would have been silent, quietly discarding the IPCW correction. `weighted_beta_nll()` therefore implements the weighted Ferrari & Cribari-Neto (2004) mean/precision log-likelihood with **analytic gradients**, maximised by L-BFGS-B, with optional L2 on non-intercept `γ`. Starting values come from a weighted FRM on the interior rows (same link) and a method-of-moments `φ`.

### Categorical encoding

Two preprocessing paths, **both fitted on the training split only**:

- **Linear path** (FRM, stage 1, stage 2): one-hot with a dropped reference level; levels with fewer than `config.LGD_MIN_LEVEL_COUNT` training rows, and levels unseen at scoring time, grouped into `"other"`; median imputation for numerics.
- **Tree path** (random forest): integer codes, fitted on train, with `config.LGD_UNSEEN_CODE` reserved for unseen levels.

Full derivation in the [README's Ch.3 section](../../README.md#ch3--lgd-models).

## 4. Development Data

- **Population:** defaulted loans only (`zero_balance_code ∈ config.DEFAULT_CODES`) — one row per loan, the final servicer observation at resolution
- **Target:** `lgd = actual_loss / zero_balance_removal_upb`, clipped to [0, 1]
- **Sample size:** approximately 150 defaults in the sample dataset — small enough to materially affect all three models' variance; pre-2010 crisis vintages are recommended to increase the usable default count
- **Features:** `config.LGD_FEATURES` (17 features — HPI change since origination, mortgage insurance %, CLTV, DTI, current interest rate, property/loan characteristics)
- **Split:** same train/OOS/OOT structure as the PD models (`config.OOT_CUTOFF`, `config.OOS_FRAC`), applied to the defaulted-loan population

## 5. Key Assumptions

- **Sample size is the dominant limitation** — ~150 defaults means all three models are trained on a dataset small enough that the performance estimates themselves carry meaningful uncertainty. Do not treat the RMSE/MAE comparison as decisive without a larger default sample. This is why the suite was reduced from four flexible mean estimators to three models with distinct purposes.
- **The two-stage model assumes constant precision.** Stage 2 shares a single `φ` across all loans, so the *spread* of partial losses is assumed not to vary with loan characteristics even though the *mean* does. `predict_quantile()` inherits this, which matters exactly where it would be used — a downturn LGD read off a high quantile. Modelling `φ` with its own covariates is the natural extension, not done here because at ~150 observations (fewer still interior) it is not identifiable.
- **Stage 1 and stage 2 are fitted independently**, not jointly. The composed conditional mean is still correct, but the standard errors of one stage do not account for estimation uncertainty in the other.
- **The FRM assumes a fractional-logit link is an adequate functional form**, and that its linear index captures the relevant structure; the random forest assumes ~150 rows suffice for a tree ensemble to generalise, which is optimistic at that sample size.
- **LGD is clipped to [0, 1] post-hoc** — any economic loss above 100% of UPB (e.g. from legal/foreclosure costs) is truncated rather than modelled.
- **The macro ECL anchor is population-level, not per-loan.** The PD population `07_macro_scenario_analysis.py` scores doesn't carry this suite's LGD-specific features (`hpi_change_since_orig`, `mi_pct`, `current_interest_rate`, etc.) — `pd_{train,oos,oot}.parquet` only persist `config.PD_FEATURES`. So every loan in a given scenario-quarter gets the same champion-derived LGD, scaled only by that scenario-quarter's aggregate HPI path. Closing this gap would mean persisting `config.LGD_FEATURES` alongside the PD population in `01_data_preprocessing.py` — a schema-changing fix.
- **Small-sample fallbacks may be active.** Each is logged; a run whose log shows a dropped class or a constant-mean stage 2 is reporting a materially simpler model than the specification above.

## 6. Performance

`TBD — regenerate after running.`

The categorical encoding fix (train-only one-hot with rare-level pooling, replacing integer label codes fed to the FRM as continuous covariates) changes every linear model's design matrix, so **no pre-fix LGD metric carries over**. Figures are produced into `lgd_metrics.csv`, `lgd_two_stage_stage_metrics.csv` and `lgd_point_mass_calibration.csv` by the script.

| Model | Split | RMSE / MAE / R² / Bias |
|---|---|---|
| FRM | OOS / OOT | `TBD` |
| Two Stage | OOS / OOT | `TBD` |
| Random Forest | OOS / OOT | `TBD` |

## 7. Validation

- Same OOS/OOT backtesting structure as the PD models
- Three models compared side by side rather than one reported in isolation — challenger-model validation appropriate to the small development sample
- **Point-mass calibration** (`lgd_point_mass_calibration.csv`): observed share of LGD=0 and LGD=1 against each model's predicted share. For the FRM and the random forest the predicted share is typically ~0 — that gap is the diagnostic, showing what the two-stage model's explicit masses buy and why a mean-only LGD model can look well-calibrated on RMSE while failing to reproduce the shape of the loss distribution
- **Stage-specific diagnostics** for the two-stage model: stage-1 multiclass log loss and confusion matrix per split; stage-2 RMSE/MAE on interior observations only
- **Distribution overlay** (`lgd_distributions.png`): actual vs each model's predicted LGD density

## 8. Ongoing Monitoring Plan

Not currently covered by `09_monitoring.py`, which monitors the PD feature set and PD scores only. Before production use, extend monitoring to:

1. **`config.LGD_FEATURES` distribution PSI** vs the training reference.
2. **Realised vs predicted LGD on newly-resolved defaults** — a directly observable actual-vs-expected check that PD monitoring cannot do, since PD outcomes take 12 months to mature but LGD outcomes are known at resolution.
3. **Point-mass share drift** — the observed share of LGD=0 and LGD=1 each period against stage 1's predicted probabilities. A shift here (e.g. rising full-recovery rates in a strong housing market) changes the shape of the loss distribution even when the mean is stable, and no mean-based monitor would catch it.
4. **Which fallbacks fired** — a period in which a boundary class drops below the minimum count, or stage 2 degenerates to a constant, is running a materially different model and should be flagged rather than silently accepted.
5. **Categorical level drift** — the share of scored loans landing in the `"other"` bucket. A rising share means the training level set no longer describes the population.

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
| v1.2 | Replaced the mislabelled "FRM" (OLS on `logit(LGD)` → sigmoid) with a genuine Papke-Wooldridge quasi-binomial GLM; added IPCW sample weights |
| v2.0 | **Suite reduced to three models.** Removed the natural spline and XGBoost regressors (redundant conditional-mean estimators). Added the two-stage multinomial-logit + beta-regression model, with a hand-written weighted beta likelihood (statsmodels' `BetaModel` silently ignores weights). Fixed categorical encoding: train-only one-hot with dropped reference levels and rare-level pooling, replacing integer label codes fed to the FRM as continuous covariates and an encoder fitted across all splits. Added L2 fallback for the FRM, small-sample fallbacks throughout, and point-mass calibration diagnostics. **Pre-v2.0 LGD metrics are invalid.** |
