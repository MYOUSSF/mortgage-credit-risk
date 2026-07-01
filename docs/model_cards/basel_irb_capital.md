# Model Card — Basel IRB Rating Scale & Capital (Ch.9)

## 1. Identification

| | |
|---|---|
| Script | `10_basel_irb_capital.py` |
| Model type | Deterministic mapping + regulatory formula: PD → rating grade (`config.pd_to_rating()`); (PD, LGD, EAD) → RWA/capital (Basel IRB supervisory formula) |
| Chapter | Ch.9 (extension beyond the Sexton 2022 thesis) |
| Version | v1.0 — see `git log -- 10_basel_irb_capital.py` for change history |
| Model tier | `TBD` |

## 2. Purpose & Intended Use

Converts a continuous PD into the two artifacts that make a PD model usable for Basel II/III IRB capital, rather than just a scorecard:

- **Rating master scale** (`config.RATING_SCALE`, `assign_rating_grade()`) — a PD → letter-grade (AAA...D) mapping for credit committee reporting, portfolio limits, and disclosure.
- **Risk-weighted assets (RWA) and Pillar 1 capital**, retail residential mortgage exposure class (Basel Framework CRE31/CRE32): fixed asset correlation `R = config.BASEL_RETAIL_MORTGAGE_CORRELATION` (0.15), no maturity adjustment (that term applies only to corporate/sovereign/bank exposures under the advanced IRB approach).

**Not intended for:** IFRS 9 provisioning — that uses the point-in-time (PIT), per-loan-staged ECL from `07_macro_scenario_analysis.py`, not this script's TTC-anchored capital PD. Not intended for exposure classes other than retail residential mortgage — the fixed correlation and absent maturity adjustment are specific to that class; corporate/sovereign/bank exposures need the PD-varying correlation function and maturity adjustment this script does not implement.

## 3. Methodology

```
R = 0.15
K = LGD * N[ G(PD)/sqrt(1-R) + sqrt(R/(1-R)) * G(0.999) ] - PD * LGD
RWA = K * 12.5 * EAD
Capital = RWA * 8%  =  K * EAD
```

where `G` is the inverse standard normal CDF and `N` is the standard normal CDF. `K` is capital against **unexpected** loss only — `PD * LGD` (expected loss) is subtracted out, since expected loss is meant to be covered by IFRS 9 ECL provisions (Ch.6), not Pillar 1 capital. Full derivation in the [README's Ch.9 section](../../README.md#ch9--basel-irb-rating-scale--capital).

## 4. Development Data

No independent development population — this script is a downstream formula/mapping layer over three upstream outputs:

- **PD:** `data/processed/ttc_calibrated_pd.csv` (`08_calibration.py`'s LRADR-anchored TTC PD), falling back to `data/processed/survival_pd_horizons.csv`'s `ttc_pd_12m` column (`06_survival_analysis.py`'s macro-neutral Cox re-scoring) if Ch.7 hasn't been run.
- **LGD:** `data/processed/lgd_champion_summary.csv`'s anchor mean LGD (`04_lgd_models.py`), falling back to `config.MACRO_LGD_ASSUMPTION` with a warning.
- **EAD:** each loan's last-observed `current_upb` (falling back to `orig_upb`) from `data/processed/pd_oos.parquet`.

## 5. Key Assumptions

- **PD must be through-the-cycle**, not point-in-time — Basel IRB capital is explicitly meant not to move with the current point in the credit cycle (EBA/GL/2017/16 §6.2). This script trusts its upstream TTC PD sources rather than re-deriving TTC-ness itself; if fed a PIT PD by mistake (e.g. `pd_lr_results.csv`'s raw score), the resulting capital would understate requirements in a benign part of the cycle and overstate them in a stressed one.
- **LGD is a single population-level anchor**, applied identically to every loan regardless of grade — not per-loan, and not downturn-adjusted (Basel requires a downturn LGD for capital, distinct from the average-conditions LGD used for ECL). See Known Limitations #10 in the README.
- **Rating master scale bounds are illustrative**, not fitted to this portfolio's realised default experience (README Known Limitations #9) — an institution would calibrate its own scale before using grades for disclosure or limit-setting.
- **Retail residential mortgage exposure class only** — the fixed `R = 0.15` correlation and absent maturity adjustment do not generalise to other exposure classes.

## 6. Performance

Not applicable in the usual discrimination/calibration sense — this script applies a deterministic formula to an upstream PD, so its "performance" is inherited entirely from that PD's calibration quality (see [calibration.md](calibration.md)) and the champion LGD model's accuracy (see [lgd_models.md](lgd_models.md)).

## 7. Validation

- `basel_irb_capital_k()` is bounded: `0 <= K <= LGD` and monotonically increasing in PD — checked directly in `tests/test_basel_irb_capital.py`.
- `plot_supervisory_formula()` — the standard Basel model-validation chart: K(PD) at the fitted LGD anchor, for visual sanity-checking against published Basel supervisory formula curves.
- Grade-level RWA/capital reconciliation (`basel_irb_capital_by_grade.csv`) — total capital should rise monotonically with grade (AAA -> D) at the portfolio level, given the formula's monotonicity in PD.

## 8. Ongoing Monitoring Plan

Not currently covered by `09_monitoring.py` (PD-feature and PD-score focused; this script runs after Ch.8 in the pipeline). Before production use, extend monitoring with: (1) rating grade migration matrices over time (the standard IRB monitoring artifact — proportion of loans moving grade period over period), (2) periodic re-calibration of `config.RATING_SCALE` bounds against realised default rates per grade, (3) re-validation of the LGD anchor and confirmation it remains an appropriate (ideally downturn) estimate.

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
| v1.0 | Initial Basel IRB extension — rating master scale, retail residential mortgage RWA/capital formula |
