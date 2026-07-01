# Notebooks — Mortgage Credit Risk Modelling

These notebooks visualise and interpret the outputs produced by the Python scripts in the parent directory. They are **read-only** companions to the pipeline: run the scripts first, then open the notebooks in order.

---

## Prerequisites

```bash
pip install lifelines shap xgboost scikit-learn matplotlib pandas pyarrow
```

All notebooks expect the following directories to exist and be populated:

| Path | Populated by |
|---|---|
| `data/processed/` | `01_data_preprocessing.py` |
| `data/figures/` | The relevant numbered script |

---

## Notebook Order

| # | Notebook | Depends on script(s) | What it covers |
|---|---|---|---|
| 1 | `01_EDA.ipynb` | `01_data_preprocessing.py` | Portfolio characteristics, vintage default rates, crisis signature, class imbalance |
| 2 | `02_PD_Modelling.ipynb` | `02_pd_logistic_regression.py`, `03_pd_ensemble.py` | WoE/IV feature ranking, logistic regression vs XGBoost, AUROC / KS / Gini, PSI |
| 3 | `03_LGD_Modelling.ipynb` | `04_lgd_models.py` | LGD distribution, FRM / Spline / RF / XGBoost comparison, ECL = PD × LGD × EAD |
| 4 | `04_SHAP_Explanations.ipynb` | `05_shap_explanations.py` | Global importance, beeswarm, individual waterfall plots, BCBS 239 segment report |
| 5 | `05_Survival_Analysis.ipynb` | `06_survival_analysis.py` | Cox PH hazard ratios, Kaplan-Meier curves, multi-horizon PD for IFRS 9 staging |
| 6 | `06_Macro_Scenario_Analysis.ipynb` | `07_macro_scenario_analysis.py` | Base / Adverse / Severe scenarios, probability-weighted ECL, macro sensitivity |
| 7 | `07_Calibration.ipynb` | `08_calibration.py` | Platt / Isotonic / Temperature scaling, reliability diagrams, LRADR validation |

---

## Key Output Files Referenced

```
data/processed/
  pd_train.parquet              # PD modelling dataset — training
  pd_oos.parquet                # PD modelling dataset — out-of-sample
  pd_oot.parquet                # PD modelling dataset — out-of-time
  lgd_train.parquet             # LGD modelling dataset
  survival_cox_coefs.csv        # Cox hazard ratios + confidence intervals
  survival_pd_horizons.csv      # Per-loan PD at 12m / 24m / 36m / 60m
  calibrated_scores_oos.csv     # Calibrated PD scores (all methods)

data/figures/
  survival_km_by_credit_score.png
  survival_km_by_vintage.png
  survival_cox_hazard_ratios.png
  survival_schoenfeld_residuals.png
  shap_summary_beeswarm.png
  shap_waterfall_*.png
  shap_segment_report.png
```