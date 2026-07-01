# Notebooks — Mortgage Credit Risk Modelling

These notebooks visualise and interpret the outputs produced by the Python scripts in `../src/`. They are **read-only** companions to the pipeline: run the scripts first, then open the notebooks in order. Notebooks are meant to be run with the repository root as the working directory (not `notebooks/`), matching the scripts' own relative `data/...` paths.

---

## Prerequisites

```bash
pip install -r ../requirements.txt -r ../requirements-dev.txt
```

All notebooks expect the following directories to exist and be populated:

| Path | Populated by |
|---|---|
| `data/processed/` | `src/01_data_preprocessing.py` |
| `data/figures/` | The relevant numbered script |

---

## Notebook Order

| # | Notebook | Depends on script(s) | What it covers |
|---|---|---|---|
| 1 | `01_EDA.ipynb` | `src/01_data_preprocessing.py` | Portfolio characteristics, vintage default rates, crisis signature, class imbalance |
| 2 | `02_PD_Modelling.ipynb` | `src/02_pd_logistic_regression.py`, `src/03_pd_ensemble.py` | WoE/IV feature ranking, logistic regression vs XGBoost, AUROC / KS / Gini, PSI |
| 3 | `03_LGD_Modelling.ipynb` | `src/04_lgd_models.py` | LGD distribution, FRM / Spline / RF / XGBoost comparison, ECL = PD × LGD × EAD |
| 4 | `04_SHAP_Explanations.ipynb` | `src/05_shap_explanations.py` | Global importance, beeswarm, individual waterfall plots, BCBS 239 segment report |
| 5 | `05_Survival_Analysis.ipynb` | `src/06_survival_analysis.py` | Cox PH hazard ratios, Kaplan-Meier curves, multi-horizon PD for IFRS 9 staging, PIT vs TTC PD |
| 6 | `06_Macro_Scenario_Analysis.ipynb` | `src/07_macro_scenario_analysis.py` (optionally `src/04_lgd_models.py` first) | Base / Adverse / Severe scenarios, probability-weighted ECL, macro sensitivity, per-loan IFRS 9 Stage 1/2/3 |
| 7 | `07_Calibration.ipynb` | `src/08_calibration.py` | Platt / Isotonic / Temperature scaling, reliability diagrams, LRADR validation, TTC cycle adjustment for Basel IRB |

---

## Key Output Files Referenced

```
data/processed/
  pd_train.parquet                    # PD modelling dataset — training
  pd_oos.parquet                      # PD modelling dataset — out-of-sample
  pd_oot.parquet                      # PD modelling dataset — out-of-time
  lgd_train.parquet                   # LGD modelling dataset
  lgd_champion_summary.csv            # Champion LGD model's anchor mean LGD
  survival_cox_coefs.csv              # Cox hazard ratios + confidence intervals
  survival_pd_horizons.csv            # Per-loan PIT + TTC PD at 12m / 24m / 36m / lifetime
  calibrated_scores_oos.csv           # Calibrated PD scores (all methods)
  ttc_calibrated_pd.csv               # Per-loan PIT + TTC (LRADR-shifted) PD
  ifrs9_ecl_by_loan.csv               # Per-loan ECL at fixed 1Y/2Y/3Y/lifetime horizons
  ifrs9_staged_ecl_by_loan.csv        # Per-loan IFRS 9 Stage 1/2/3 + staged ECL
  ifrs9_staged_ecl_summary.csv        # Portfolio staged ECL by stage

data/figures/
  survival_km_by_credit_score.png
  survival_km_by_vintage.png
  survival_cox_hazard_ratios.png
  survival_schoenfeld_residuals.png
  survival_pit_vs_ttc.png
  shap_beeswarm.png
  shap_waterfall_*.png
  shap_segment_report.png
  ifrs9_stage_distribution.png
```