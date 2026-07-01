# Mortgage Credit Risk Modelling

**Probability of Default · Loss Given Default · SHAP · Survival Analysis · Macro Stress Testing · Calibration**

A replication and four-chapter extension of Sexton, S. M. (2022), *Credit Risk Modelling Using Machine Learning Methods*, PhD Thesis, Department of Economics — implemented on the Freddie Mac Single-Family Loan Performance dataset (2000–2020, 200M+ loan-month records).

---

## Key Results

| Model | AUROC (OOS) | KS (OOS) | Gini (OOS) |
|---|---|---|---|
| Logistic Regression (Ch.1) | ~0.87 | ~0.58 | ~0.74 |
| XGBoost (Ch.2) | ~0.91 | ~0.64 | ~0.82 |
| Cox Proportional Hazards (Ch.5) | ~0.85–0.89 | — | — |

Results evaluated on a held-out 30% OOS set and a temporal OOT set (2017–2024) never seen during training.

> **Note:** these figures are illustrative and predate a fix to the OOS split (it was previously row-level rather than loan-level — see below). The same loan's monthly snapshots could land in both Train and OOS, letting models partially "recognise" training loans; the split is now grouped by `loan_seq_num` so no loan spans two splits. Treat the OOT column as the more trustworthy generalisation estimate until this table is regenerated on the corrected split.

---

## Repository Structure

```
mortgage-credit-risk/
│
├── config.py                           # Shared constants: SEED, split boundary,
│                                        #   feature lists, GPU/logging/plot setup,
│                                        #   macro scenario assumptions
│
│  ── Core pipeline ──────────────────────────────────────────────────────
├── 00_download_freddie_mac.py          # Authenticate + sequential download
├── 01_data_preprocessing.py            # Year-by-year pipeline → Parquet
├── 02_pd_logistic_regression.py        # Ch.1: WoE + logistic regression PD
├── 03_pd_ensemble.py                   # Ch.2: XGBoost PD (GPU-accelerated)
├── 04_lgd_models.py                    # Ch.3: FRM / splines / RF / XGBoost LGD
│
│  ── Extensions ─────────────────────────────────────────────────────────
├── 05_shap_explanations.py             # Ch.4: SHAP — BCBS 239 loan attribution
├── 06_survival_analysis.py             # Ch.5: Cox PH — right-censored time-to-default
├── 07_macro_scenario_analysis.py       # Ch.6: IFRS 9 stress testing + per-loan staging
├── 08_calibration.py                   # Ch.7: Platt / isotonic / temperature + TTC cycle adjustment
├── 09_monitoring.py                    # Ch.8: PSI drift monitoring vs training reference
├── 10_basel_irb_capital.py             # Ch.9: rating master scale + Basel IRB RWA/capital
│
│  ── Notebooks ──────────────────────────────────────────────────────────
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory data analysis
│   ├── 02_PD_Modelling.ipynb           # PD results: ROC / WoE / PSI / importance
│   ├── 03_LGD_Modelling.ipynb          # LGD analysis + ECL illustration
│   ├── 04_SHAP_Explanations.ipynb      # Global / beeswarm / waterfall / segments
│   ├── 05_Survival_Analysis.ipynb      # KM curves / Cox HR / multi-horizon PD
│   ├── 06_Macro_Scenario_Analysis.ipynb  # ECL by scenario / sensitivity / ECDF
│   └── 07_Calibration.ipynb            # Reliability diagrams / Brier / ECE / LRADR
│
│  ── Documentation ──────────────────────────────────────────────────────
├── docs/
│   ├── methodology.docx                # Formatted 20-page methodology report
│   ├── project_portfolio.html          # Interactive project portfolio page
│   └── model_cards/                    # Per-model documentation — see "Model Documentation"
│
│  ── Tests ───────────────────────────────────────────────────────────────
├── tests/                              # pytest suite — see "Testing" below
│
├── requirements.txt
├── requirements-dev.txt                # + pytest
└── README.md
```

---

## Data

**Source:** [Freddie Mac Single-Family Loan Performance Dataset](https://www.freddiemac.com/research/datasets) — publicly available, registration required.

| File | Columns | Contents |
|---|---|---|
| `sample_orig_YYYY.txt` | 32 | Static loan attributes at origination |
| `sample_svcg_YYYY.txt` | 32 | Monthly servicer updates (UPB, delinquency, disposition) |

Origination years 2000–2020 yield 200M+ loan-month records spanning the 2004–2008 subprime crisis (default rates 3–15% in crisis vintages).

**Macro data (optional — materially improves discrimination):**
- FHFA HPI by 3-digit ZIP → `data/raw/macro/hpi_3digit_zip.csv`  ([FHFA](https://www.fhfa.gov/data/hpi))
- BLS unemployment LNS14000000 → `data/raw/macro/unemployment_rate.csv`  ([BLS](https://data.bls.gov/timeseries/LNS14000000))

---

## How to Run

### Setup

```bash
git clone https://github.com/MYOUSSF/mortgage-credit-risk
cd mortgage-credit-risk
pip install -r requirements.txt
```

### Core pipeline

```bash
python 00_download_freddie_mac.py     # Download raw data
python 01_data_preprocessing.py       # ~20 min on Kaggle GPU
python 02_pd_logistic_regression.py   # ~5 min
python 03_pd_ensemble.py              # ~15 min (GPU auto-detected)
python 04_lgd_models.py               # ~10 min
```

### Extensions

```bash
python 05_shap_explanations.py        # SHAP — BCBS 239 attribution
python 06_survival_analysis.py        # Cox PH — right-censored time-to-default + TTC PD
python 07_macro_scenario_analysis.py  # IFRS 9 stress testing + Stage 1/2/3 staging
python 08_calibration.py              # Platt / isotonic calibration + TTC cycle adjustment
python 09_monitoring.py               # PSI drift monitoring vs training reference
python 10_basel_irb_capital.py        # Rating master scale + Basel IRB RWA/capital
```

> **Environment:** Kaggle notebooks (2×T4 GPU, 30 GB RAM) recommended for scripts 03–05. All scripts fall back to CPU gracefully.

### Configuration

Every script imports `config.py` for values that must stay identical across the pipeline: the random seed, the train/OOS/OOT split boundary, the default-event codes, the PD/LGD feature lists, GPU detection, the plot theme, the shared logging setup, PSI/IV rating thresholds, and the IFRS 9 macro scenario assumptions. Each script keeps its own local name for what it imports (e.g. `TARGET = config.TARGET_PD`), so change an assumption once in `config.py` and every script that depends on it picks it up — there's no second or third copy of `DEFAULT_CODES` or the macro scenario shocks to remember to update.

### Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

The suite (`tests/`) covers the pieces where a silent bug is most expensive: the 12-month default-event window and leakage guard, the temporal OOT / random OOS split boundaries, WoE maps fitted train-only and their safe application to unseen values at scoring time, PSI/IV drift detection (including the fixed-reference-over-time monitoring pattern), and an independent check of the IFRS 9 ECL accumulation formula. It does not require the Freddie Mac dataset — everything runs against small synthetic DataFrames.

---

## Methodology

### Data Engineering (Script 01)

Year-by-year chunked processing keeps peak RAM at ~400 MB rather than ~15 GB:

```python
for year in range(2000, 2021):
    orig = load_orig_year(year)      # ~15 MB
    svcg = load_svcg_year(year)      # ~200 MB
    merged = svcg.merge(orig, on="loan_seq_num")
    pd_chunk.to_parquet(f"chunks/pd_{year}.parquet")
    del merged; gc.collect()
pd_all = pd.concat([pd.read_parquet(f) for f in chunk_files])
```

**Default definition:** `zero_balance_code ∈ {02, 03, 06, 09, 15}` — 3rd-party sale, short sale, repurchase, REO, note sale. Prepayments (01) explicitly excluded; the rare non-standard codes 16 (reperforming) and 96 (non-standard disposition) are not treated as default events.

**Train / OOS / OOT split:**

| Split | Period | Purpose |
|---|---|---|
| Train | 2000–2017 | Model fitting |
| OOS | 2000–2017 | Random ~30% of **loans** (grouped by `loan_seq_num` — no loan spans two splits) |
| OOT | 2017–2024 | Temporal holdout — never seen during fitting |

---

### Ch.1 — Logistic Regression PD

WoE encoding with leakage-proof maps (fitted on train only, applied to OOS/OOT):

```
WoE_j = ln(p_j / q_j)      IV = Σ_j (p_j − q_j) · WoE_j
```

| Feature | IV | Strength |
|---|---|---|
| `delinquency_indicator` | 0.538 | Very strong |
| `loan_age` | 0.374 | Strong |
| `credit_score` | 0.304 | Strong |
| `orig_dti` | 0.285 | Medium |
| `orig_interest_rate` | 0.159 | Medium |
| `orig_cltv` | 0.125 | Medium |

`class_weight='balanced'` upweights defaults ~155×. Hosmer–Lemeshow calibration test included.

---

### Ch.2 — XGBoost PD

```python
XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    scale_pos_weight=155,       # neg/pos ratio for class imbalance
    tree_method="hist",         # 5–10× RAM reduction via histogram approx.
    device="cuda",              # auto-detected; CPU fallback
    early_stopping_rounds=20,   # halts when OOS AUC plateaus
)
```

GPU acceleration delivers ~15× speedup. XGBoost outperforms LR by capturing non-linear FICO × CLTV × HPI interactions missed by WoE binning.

---

### Ch.3 — LGD Models

**Target:** `LGD = actual_loss / zero_balance_removal_upb`, clipped to [0, 1]

Four models compared on RMSE, MAE, R², and mean bias:

| Model | Key Property |
|---|---|
| Fractional Response (FRM) | Papke-Wooldridge quasi-binomial GLM (logit link) — targets E[LGD\|X] directly, no boundary clipping |
| Natural Spline Regression | Cubic splines (5 knots) — non-linearity without overfitting |
| Random Forest | 200 trees, max depth 6 |
| XGBoost Regressor | Gradient boosted trees |

All four are fit with inverse-probability-of-censoring (IPCW) sample weights correcting for LGD workout-period truncation bias: `extract_lgd_rows()` correctly keeps only *resolved* defaults (no leakage), but that resolved-case sample is truncated — a loan entering workout (90+ days past due) close to the dataset's end and taking a long time to resolve (contested foreclosure, REO) is systematically missing, while fast-resolving cases (short sales) are always captured even near the cutoff. `compute_ipcw_weights()` estimates the truncation ("censoring") distribution via a reversed Kaplan-Meier fit and reweights resolved cases accordingly, implemented in `01_data_preprocessing.py`.

The lowest-RMSE model on OOS (falling back to Train for small samples) is selected as champion and its mean predicted LGD is saved to `lgd_champion_summary.csv` — this is the value Ch.6's macro scenario ECL uses as its LGD anchor, rather than a flat assumption disconnected from this model suite.

---

### Ch.4 — SHAP Explanations (BCBS 239)

Individual loan-level feature attribution using `TreeExplainer` (exact, zero approximation error for tree-based models):

**Key outputs:**
- **Waterfall chart** — starts at E[f(x)] (base rate), shows how each feature pushes the score for one specific loan
- **Beeswarm** — SHAP distribution per feature, colour-coded by raw value (direction + magnitude + distribution)
- **Segment report** — top SHAP drivers by risk decile, satisfying BCBS 239 Principles 6 & 11

Example explanation:
> *"This loan's PD = 3.2% vs 0.8% portfolio average. CLTV = 95% adds +1.4pp, FICO = 620 adds +0.9pp, rising unemployment adds +0.3pp."*

---

### Ch.5 — Survival Analysis

Cox Proportional Hazards model handles right-censoring:

```
h(t | x) = h₀(t) · exp(x'β)
```

| Problem with binary model | Cox PH solution |
|---|---|
| Active loans fabricated as non-defaulters | Censored loans contribute partial likelihood |
| 12-month window only | PD at any horizon: 12m / 24m / 36m / lifetime |
| No temporal structure | Full hazard trajectory modelled |

**Point-in-time (PIT) vs through-the-cycle (TTC) PD.** The headline horizon PD (`compute_horizon_pds()`) conditions on each loan's *actual current* macro state (`ur_3m_lag`, `hpi_change`), so it is point-in-time by construction — it moves with the cycle, which is exactly what IFRS 9 provisioning wants. Basel IRB capital wants the opposite: a PD that doesn't move with the current point in the cycle (EBA/GL/2017/16 §6.2). `compute_ttc_pds()` produces that as an actual second PD, not a diagnostic: it re-scores the same fitted Cox model with the macro covariates held at their long-run training-sample average, holding every other loan characteristic fixed. Both PIT and TTC PD are saved to `survival_pd_horizons.csv`.

Schoenfeld residuals validate the proportional hazards assumption.

---

### Ch.6 — Macro Scenario Analysis

IFRS 9 §5.5.17 multiple economic scenarios with probability-weighted ECL:

```
ECL_weighted = Σ_s (weight_s × PD_s × LGD_s(q) × EAD(q))
```

`EAD(q)` is a per-quarter amortized balance (`amortized_ead()`), projected forward from the loan's current UPB via standard declining-balance mortgage amortization at its current rate and remaining term — not a flat origination balance held constant across all 20 quarters.

| Scenario | Weight | UR Shock | HPI Shock |
|---|---|---|---|
| Base | 60% | Stable | +2% p.a. |
| Adverse | 30% | +3pp over 12m | −10% |
| Severe | 10% | +6pp over 18m | −25% |

**LGD is scenario-conditional, not fixed.** `scenario_lgd()` scales recovery at foreclosure with that scenario-quarter's collateral value (`hpi_ratio`): `recovery(q) = (1 − base_lgd) × hpi_ratio(q)`, so LGD rises under HPI stress and falls under HPI appreciation, reproducing `base_lgd` exactly when `hpi_ratio == 1.0`. `base_lgd` is Ch.3's champion LGD model's mean predicted LGD (`lgd_champion_summary.csv`), falling back to `config.MACRO_LGD_ASSUMPTION` with a warning if `04_lgd_models.py` hasn't been run yet.

Tornado chart quantifies ΔPD per unit macro shock — standard ALCO reporting format.

**Per-loan IFRS 9 staging (`assign_ifrs9_stage()`).** Every loan is classified into Stage 1 / 2 / 3, not just scored at a horizon applied uniformly to the whole portfolio:

| Stage | Trigger | ECL horizon |
|---|---|---|
| Stage 1 | No SICR, current | 12 months |
| Stage 2 | SICR: current lifetime PD ≥ `config.SICR_PD_RATIO` (2.0×) the PD-at-origination proxy (with an absolute floor, `config.SICR_PD_ABS_FLOOR`, against noise) **or** the 30-DPD backstop | Lifetime |
| Stage 3 | 90-DPD backstop (credit-impaired) — same trigger as the LGD workout-period onset (`LGD_ONSET_DPD_MONTHS`) | Lifetime |

The PD-at-origination proxy (`compute_origination_pd()`) re-scores the model with `loan_age` forced to 0, holding every other feature at its current value — the closest recoverable approximation given this pipeline persists one retrained snapshot model rather than each loan's actual underwriting-time score (see Known Limitations). `compute_staged_ecl()` then picks the 12-month or lifetime ECL column per loan according to its stage, and `ifrs9_staged_ecl_summary.csv` / `ifrs9_stage_distribution.png` report the staged portfolio ECL alongside the unstaged, uniform-horizon comparison (`ifrs9_ecl_summary.csv`) already described above.

---

### Ch.7 — PD Calibration

Three calibration methods aligned predicted PDs with observed default rates:

| Method | Formula | When to use |
|---|---|---|
| Platt scaling | `P_cal = σ(a·s + b)` | Default choice — stable, auditable |
| Isotonic regression | Non-parametric monotone | Large datasets (500+ events) |
| Temperature scaling | `P_cal = σ(logit(P) / T)` | When over-confidence is the problem |

**Metrics:** Brier score, ECE, MCE, Hosmer–Lemeshow p-value, LRADR comparison (Basel II §461).

**LRADR → TTC cycle adjustment.** The long-run average default rate (LRADR) isn't just plotted against the calibrated PIT PD as a diagnostic gap: `compute_ttc_pd_via_lradr()` logit-shifts every loan's Platt-calibrated PD by the constant offset that moves the population mean to LRADR, producing an actual per-loan TTC PD (`ttc_calibrated_pd.csv`) for Basel IRB capital — a simpler alternative to Ch.5's macro-neutral re-scoring, used here because this script only has each model's raw score column, not the fitted model object itself.

---

### Ch.8 — Ongoing Model Monitoring

01_data_preprocessing.py computes PSI once, at build time, to help choose which features to model with. This is a different, recurring check: has the population the model now sees drifted away from the population it was trained on?

Each feature's reference distribution is fitted **once** from the training set and kept fixed; every subsequent period is re-binned into those same bins so PSI stays comparable over time — refitting bin edges per period would let the bins silently drift along with the data and mask real shifts.

```
PSI = Σᵢ (pᵢ − qᵢ) · ln(pᵢ / qᵢ)        < 0.10 stable | 0.10–0.25 investigate | > 0.25 major shift
```

Monitored per period (OOS as a post-build checkpoint, then OOT bucketed by year):
- Every PD feature's distribution vs the training reference
- The model score distribution (if `02_pd_logistic_regression.py` has been run)
- Observed 12-month default rate vs the training-set average

Outputs a per-period × per-feature PSI heatmap and CSV reports, with an explicit alert log for any "major shift" breach — the artifact a model risk function would actually review each monitoring cycle.

---

### Ch.9 — Basel IRB Rating Scale & Capital

Everything through Ch.8 is a scorecard: PD/LGD models, ECL, and calibration diagnostics, but nothing converts a continuous PD into the two things that make a model "IRB" rather than just a scorecard. `10_basel_irb_capital.py` adds both:

**Rating master scale** (`config.RATING_SCALE`, `config.pd_to_rating()`) — a PD → letter-grade mapping (AAA...D), illustrative bounds that an institution would calibrate to its own realised default experience.

**Basel II/III IRB capital formula**, retail residential mortgage exposure class (Basel Framework CRE31/CRE32) — fixed asset correlation, no maturity adjustment (that term is corporate/sovereign/bank-only):

```
R = 0.15  (config.BASEL_RETAIL_MORTGAGE_CORRELATION)
K = LGD × N[ G(PD)/√(1-R) + √(R/(1-R)) × G(0.999) ] − PD × LGD
RWA = K × 12.5 × EAD
Capital = RWA × 8%  =  K × EAD
```

PD input is TTC, not PIT — Basel IRB capital wants a PD that doesn't move with the current point in the cycle, the opposite of what IFRS 9 provisioning uses. The script prefers Ch.7's LRADR-anchored `ttc_calibrated_pd.csv`, falling back to Ch.5's macro-neutral `ttc_pd_12m` if Ch.7 hasn't been run.

**Known limitation:** LGD is Ch.3's population-level champion-model anchor, the same constraint documented for Ch.6's macro ECL engine, and it is not downturn-adjusted — Basel requires a downturn LGD for capital, distinct from the average LGD used for ECL.

---

## Regulatory Alignment

| Framework | Coverage |
|---|---|
| Basel II/III IRB | PD (TTC) + LGD, rating master scale, retail mortgage RWA/capital formula (Ch.9) |
| IFRS 9 / CECL | Per-loan Stage 1/2/3 (SICR + DPD backstops), 12m/lifetime staged ECL, probability-weighted scenarios (Ch.6) |
| BCBS 239 | SHAP waterfall/segment reports — Principles 6 & 11 |
| OCC SR 11-7 | OOS/OOT backtesting, ongoing PSI monitoring (Ch.8), HL calibration test |
| EBA GL/2017/16 | Survival-based PIT and TTC PD (Ch.5); Schoenfeld residual validation |

---

## Model Documentation

Each model has its own model card under [`docs/model_cards/`](docs/model_cards/README.md) — intended use, key assumptions, performance, validation approach, and an ongoing monitoring plan, one level more specific than the shared "Known Limitations" below:

| Model | Card |
|---|---|
| PD — Logistic Regression (Ch.1) | [pd_logistic_regression.md](docs/model_cards/pd_logistic_regression.md) |
| PD — XGBoost (Ch.2) | [pd_xgboost.md](docs/model_cards/pd_xgboost.md) |
| LGD Model Suite (Ch.3) | [lgd_models.md](docs/model_cards/lgd_models.md) |
| Survival — Cox PH (Ch.5) | [survival_cox.md](docs/model_cards/survival_cox.md) |
| IFRS 9 Macro Scenario ECL & Staging (Ch.6) | [ifrs9_macro_scenario.md](docs/model_cards/ifrs9_macro_scenario.md) |
| PD Calibration (Ch.7) | [calibration.md](docs/model_cards/calibration.md) |
| Basel IRB Rating Scale & Capital (Ch.9) | [basel_irb_capital.md](docs/model_cards/basel_irb_capital.md) |

---

## Known Limitations

1. **LGD sample size:** ~150 defaults in the sample dataset. Pre-2010 crisis vintages recommended.
2. **12-month window immaturity:** `filter_immature_right_censored()` drops loan-months too close to the dataset's true end to know their 12-month outcome (never-observed-to-default AND within 365 days of the panel's max `report_date`), so a right-censored active loan isn't mislabelled as a confirmed non-default. Rows with a known (even distant) `default_date` keep their label regardless of proximity to the cutoff.
3. **No hyperparameter tuning:** Cross-validated grid search could improve OOT AUROC by 1–3 points.
4. **Scenario LGD is population-level, not per-loan:** the macro ECL engine now anchors LGD to Ch.3's champion model output and scales it with scenario HPI (`scenario_lgd()`), rather than a flat 40% assumption — but every loan in a given scenario-quarter still gets the same LGD, since the OOS population scored for PD doesn't carry the LGD-specific features (`hpi_change_since_orig`, `mi_pct`, etc.) needed for true per-loan conditioning. A production system would persist those features alongside the PD population so LGD could vary by loan, not just by scenario and quarter.
5. **Survival duration:** duration is `loan_age` at the last retained pre-default observation, plus one reporting period for actual defaulters (since `extract_pd_rows()` drops the default row itself to prevent leakage in the binary target) — the closest recoverable approximation to true time-to-default given that constraint.
6. **EAD amortization is schedule-only:** `amortized_ead()` projects a standard declining-balance schedule from current UPB/rate/remaining term; it does not model stochastic prepayment beyond that schedule, so realised future balances (and therefore realised EAD) could decline faster than projected.
7. **LGD IPCW onset trigger is a proxy:** the workout-period truncation correction (`compute_ipcw_weights()`) defines "onset" as first reaching 90+ days past due — a standard regulatory default trigger, but distinct from (and possibly earlier than) the actual start of a formal workout/foreclosure process, which isn't separately recorded in the fields this pipeline reads.
8. **PD-at-origination proxy for SICR:** `07_macro_scenario_analysis.py`'s `compute_origination_pd()` approximates each loan's PD at initial recognition by re-scoring with `loan_age` forced to 0, holding every other feature (including current macro state) fixed — not the loan's actual historical origination-time PD, which this pipeline doesn't persist per loan. A production system would snapshot and store the underwriting-time score itself.
9. **Rating master scale bounds are illustrative:** `config.RATING_SCALE`'s PD upper bounds are standard S&P/Moody's-style anchor points, not calibrated to this portfolio's realised default experience — an institution would fit its own master scale before using it for disclosure or limit-setting.
10. **Basel capital LGD is not downturn-adjusted:** `10_basel_irb_capital.py` uses the same population-level, average-conditions LGD anchor as Ch.6's ECL engine (limitation #4). Basel IRB capital formally requires a downturn LGD — the LGD expected under adverse economic conditions — which is typically higher and would increase the computed capital requirement.

---

## References

1. Sexton, S. M. (2022). *Credit Risk Modelling Using Machine Learning Methods*. PhD Thesis.
2. Freddie Mac (2024). *Single-Family Loan-Level Dataset*. https://www.freddiemac.com/research/datasets
3. FHFA (2024). *House Price Index Datasets*. https://www.fhfa.gov/data/hpi
4. BLS (2024). *CPS — Series LNS14000000*. https://data.bls.gov
5. Chen, T. & Guestrin, C. (2016). XGBoost. *KDD '16*.
6. Lundberg, S. & Lee, S. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
7. Cox, D. R. (1972). Regression models and life-tables. *JRSS-B*.
8. Basel Committee on Banking Supervision (2006). *Basel II*. BIS.
9. EBA (2017). *Guidelines on PD estimation*. EBA/GL/2017/16.
10. IASB (2014). *IFRS 9 Financial Instruments*.
11. Hosmer, D. & Lemeshow, S. (2000). *Applied Logistic Regression*, 2nd ed.
12. Platt, J. (1999). Probabilistic outputs for SVMs. *Advances in Large Margin Classifiers*.

---

*Python 3.11 · pandas · scikit-learn · XGBoost · SHAP*
