# Model Card — IFRS 9 Macro Scenario ECL & Staging (Ch.6)

## 1. Identification

| | |
|---|---|
| Script | `07_macro_scenario_analysis.py` |
| Model type | PD-XGBoost re-scored under macro scenario paths, wrapped in a deterministic IFRS 9 staging rule (SICR test + DPD backstops) and ECL accumulation formula |
| Chapter | Ch.6 (extension beyond the Sexton 2022 thesis) |
| Version | v1.1 — see `git log -- 07_macro_scenario_analysis.py` for change history |
| Model tier | `TBD` |

## 2. Purpose & Intended Use

Computes probability-weighted IFRS 9 Expected Credit Loss under three macro scenarios (Base/Adverse/Severe), and classifies every loan into Stage 1 / 2 / 3 so ECL is measured at the horizon IFRS 9 actually requires per loan, rather than one horizon applied uniformly across the portfolio.

**Staging (`assign_ifrs9_stage()`):** a Stage 2 SICR test — current lifetime PD deteriorated by `config.SICR_PD_RATIO` (2.0x) or more against a PD-at-origination proxy, gated by `config.SICR_PD_ABS_FLOOR` — combined with the 30-DPD (`config.STAGE2_DPD_MONTHS`) and 90-DPD (`config.STAGE3_DPD_MONTHS`) regulatory backstops. Stage 1 loans get 12-month ECL; Stage 2/3 loans get lifetime ECL (`compute_staged_ecl()`).

**Not intended for:** Basel IRB capital — this script's PD is re-scored under scenario-specific macro conditions for ECL purposes and is point-in-time by construction, the opposite of the through-the-cycle PD `10_basel_irb_capital.py` requires (see [survival_cox.md](survival_cox.md) / [calibration.md](calibration.md) for the TTC PD sources). Not intended as a substitute for an institution's own SICR policy — the ratio/floor/backstop thresholds here are standard illustrative values (see Key Assumptions), not values calibrated to a specific portfolio's risk appetite.

## 3. Methodology

```
ECL_i = sum_q  PD_i(q) * SP_i(q-1) * LGD(q) * EAD_i(q) * DF(q)
```

per scenario, accumulated over the full 20-quarter path with survival-probability pool shrinkage, scenario-conditional LGD, and amortized EAD (see the [README's Ch.6 section](../../README.md#ch6--macro-scenario-analysis) for the full derivation). Staging is a separate post-processing step applied to the same re-scored PDs: `compute_origination_pd()` re-scores with `loan_age` forced to 0 as the origination PD proxy, `assign_ifrs9_stage()` applies the SICR/backstop rule, and `compute_staged_ecl()` selects the 12-month or lifetime ECL column per loan's stage.

## 4. Development Data

Same OOS population as the PD-XGBoost model (`config.PD_FEATURES`), re-scored under each scenario-quarter's macro path. Delinquency for the DPD backstops comes from `delinquency_status` (numeric months past due), an auxiliary column carried through `01_data_preprocessing.py` alongside the model features specifically to support this staging logic — see that script's `base_cols` comment.

## 5. Key Assumptions

- **PD-at-origination proxy, not actual origination-time PD:** `compute_origination_pd()` approximates underwriting-time risk by zeroing `loan_age` while holding every other feature (including current macro state) fixed. This pipeline persists one retrained snapshot model, not the score as it would have been computed at underwriting — see README Known Limitations #8.
- **SICR thresholds are illustrative:** `config.SICR_PD_RATIO` (2.0x) and `config.SICR_PD_ABS_FLOOR` (2pp) are standard values used in industry practice, not fitted to this portfolio's risk appetite or back-tested against realised stage transitions.
- **DPD backstops assume `delinquency_status` is a reliable proxy for regulatory DPD** — it is derived directly from the servicing file's reported delinquency (in months), the standard field for this purpose, but is a coarser signal (monthly) than a true daily DPD count.
- **LGD remains population-level, not per-loan** (README Known Limitations #4) — staging changes *which horizon* ECL is measured at per loan, not the underlying LGD assumption's granularity.

## 6. Performance

Not applicable in the discrimination/calibration sense — this script re-scores the already-validated PD-XGBoost model (see [pd_xgboost.md](pd_xgboost.md) for its OOS/OOT performance) and applies a deterministic staging rule on top.

## 7. Validation

- `compute_ifrs9_ecl()`'s accumulation formula is checked directly against a hand-derived expected value in `tests/test_macro_scenario_analysis.py` (constant-PD stub), covering survival-probability decay, scenario-conditional LGD, and EAD amortization together.
- `assign_ifrs9_stage()` is unit-tested for each trigger independently: stable PD (Stage 1), relative PD deterioration above/below the noise floor, the 30-DPD backstop, and the 90-DPD backstop overriding Stage 2.
- `plot_stage_distribution()` — portfolio count and ECL contribution by stage, the standard IFRS 9 staging disclosure pairing, for visual sanity-checking each run.

## 8. Ongoing Monitoring Plan

Not currently covered by `09_monitoring.py`. Before production use, extend monitoring with: (1) stage migration tracking period over period (the standard IFRS 9 monitoring artifact — proportion of loans moving Stage 1 -> 2, 2 -> 3, etc.), (2) periodic re-validation that the SICR ratio/floor thresholds still produce a Stage 2 population of a plausible size relative to the portfolio's actual credit deterioration, (3) re-running the accumulation-formula test whenever `config.SCENARIOS` or the amortization logic changes.

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
| v1.0 | Initial macro scenario ECL engine — quarterly path scoring, scenario-conditional LGD, amortized EAD |
| v1.1 | Added per-loan Stage 1/2/3 classification (`assign_ifrs9_stage()`) and stage-aware ECL (`compute_staged_ecl()`) — the core IFRS 9 mechanic that was previously missing |
