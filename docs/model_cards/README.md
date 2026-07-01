# Model Cards

Model documentation for each model in the pipeline, in the format an SR 11-7 / EBA GL/2017/16 model risk management (MRM) function would expect: intended use, key assumptions, performance, and an ongoing monitoring plan — one level more specific than the project-wide "Known Limitations" section in the [top-level README](../../README.md).

This is a portfolio/replication project, not a deployed model, so fields that are institution-specific (model owner, approver, materiality tier, review cadence) are marked `TBD` with a description of what an institution would fill in, rather than invented.

## Model Inventory

| Card | Script | Chapter | Role |
|---|---|---|---|
| [PD — Logistic Regression](pd_logistic_regression.md) | `02_pd_logistic_regression.py` | Ch.1 | Champion PD model — regulatory-facing, fully auditable |
| [PD — XGBoost](pd_xgboost.md) | `03_pd_ensemble.py` | Ch.2 | Challenger PD model — higher discrimination, feeds SHAP (Ch.4) and macro ECL (Ch.6) |
| [LGD Model Suite](lgd_models.md) | `04_lgd_models.py` | Ch.3 | Four LGD estimators compared champion/challenger |
| [Survival — Cox PH](survival_cox.md) | `06_survival_analysis.py` | Ch.5 | Point-in-time and through-the-cycle PD |
| [IFRS 9 Macro Scenario ECL & Staging](ifrs9_macro_scenario.md) | `07_macro_scenario_analysis.py` | Ch.6 | Stage 1/2/3 classification + staged ECL under macro scenarios |
| [PD Calibration](calibration.md) | `08_calibration.py` | Ch.7 | Recalibration layer + TTC cycle adjustment applied to the PD-XGBoost score |
| [Basel IRB Rating Scale & Capital](basel_irb_capital.md) | `10_basel_irb_capital.py` | Ch.9 | Rating master scale + retail mortgage RWA/capital |

Not separately card'd: `05_shap_explanations.py` (an explainability layer on PD-XGBoost, not a distinct model), and `09_monitoring.py` (a monitoring process, not a model — see its own [README section](../../README.md#ch8--ongoing-model-monitoring)).

## Card Template

Every card follows the same section order so models are comparable at a glance:

1. **Identification** — name, script, version/effective date, tier
2. **Purpose & Intended Use** — what decision this supports, and what it must not be used for
3. **Methodology** — one paragraph; full detail lives in the README's Methodology section
4. **Development Data** — population, period, sample size, exclusions
5. **Key Assumptions** — specific to this model, distinct from generic pipeline limitations
6. **Performance** — actual metrics from this repo's runs
7. **Validation** — how it was tested (OOS/OOT, backtesting)
8. **Ongoing Monitoring Plan** — what to watch post-deployment, thresholds, cadence
9. **Governance** — owner, approver, review cadence (`TBD` per institution)
10. **Change Log**
