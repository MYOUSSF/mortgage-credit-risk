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

---

## Repository Structure

```
mortgage-credit-risk/
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
├── 07_macro_scenario_analysis.py       # Ch.6: IFRS 9 stress testing (Base/Adverse/Severe)
├── 08_calibration.py                   # Ch.7: Platt / isotonic / temperature calibration
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
│   └── project_portfolio.html          # Interactive project portfolio page
│
├── requirements.txt
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
python 06_survival_analysis.py        # Cox PH — right-censored time-to-default
python 07_macro_scenario_analysis.py  # IFRS 9 stress testing
python 08_calibration.py              # Platt / isotonic calibration
```

> **Environment:** Kaggle notebooks (2×T4 GPU, 30 GB RAM) recommended for scripts 03–05. All scripts fall back to CPU gracefully.

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

**Default definition:** `zero_balance_code ∈ {02, 03, 06, 09, 15, 16, 96}` — prepayments (01) explicitly excluded.

**Train / OOS / OOT split:**

| Split | Period | Purpose |
|---|---|---|
| Train | 2000–2017 | Model fitting |
| OOS | 2000–2017 | Random 30% holdout |
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
| Fractional Response (FRM) | OLS on logit(LGD) → sigmoid predictions always in (0,1) |
| Natural Spline Regression | Cubic splines (5 knots) — non-linearity without overfitting |
| Random Forest | 200 trees, max depth 6 |
| XGBoost Regressor | Gradient boosted trees |

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

**IFRS 9 staging:**

| Stage | PD Horizon |
|---|---|
| Stage 1 | 12 months |
| Stage 2 (SICR) | Lifetime |

Schoenfeld residuals validate the proportional hazards assumption.

---

### Ch.6 — Macro Scenario Analysis

IFRS 9 §5.5.17 multiple economic scenarios with probability-weighted ECL:

```
ECL_weighted = Σ_s (weight_s × PD_s × LGD × EAD)
```

| Scenario | Weight | UR Shock | HPI Shock |
|---|---|---|---|
| Base | 60% | Stable | +2% p.a. |
| Adverse | 30% | +3pp over 12m | −10% |
| Severe | 10% | +6pp over 18m | −25% |

Tornado chart quantifies ΔPD per unit macro shock — standard ALCO reporting format.

---

### Ch.7 — PD Calibration

Three calibration methods aligned predicted PDs with observed default rates:

| Method | Formula | When to use |
|---|---|---|
| Platt scaling | `P_cal = σ(a·s + b)` | Default choice — stable, auditable |
| Isotonic regression | Non-parametric monotone | Large datasets (500+ events) |
| Temperature scaling | `P_cal = σ(logit(P) / T)` | When over-confidence is the problem |

**Metrics:** Brier score, ECE, MCE, Hosmer–Lemeshow p-value, LRADR comparison (Basel II §461).

---

## Regulatory Alignment

| Framework | Coverage |
|---|---|
| Basel II/III IRB | PD + LGD through-the-cycle on full economic cycle (2000–2020) |
| IFRS 9 / CECL | 12m PD (Stage 1), lifetime PD from Cox (Stage 2), probability-weighted ECL |
| BCBS 239 | SHAP waterfall/segment reports — Principles 6 & 11 |
| OCC SR 11-7 | OOS/OOT backtesting, PSI monitoring, HL calibration test |
| EBA GL/2017/16 | Survival-based PD; Schoenfeld residual validation |

---

## Known Limitations

1. **LGD sample size:** ~150 defaults in the sample dataset. Pre-2010 crisis vintages recommended.
2. **12-month window truncation:** Year-by-year architecture means the forward window cannot cross a calendar-year boundary.
3. **No hyperparameter tuning:** Cross-validated grid search could improve OOT AUROC by 1–3 points.
4. **Scenario LGD:** Macro scenarios use fixed LGD = 40%. In production, LGD would be conditional on the scenario.
5. **Survival censoring:** Duration uses `loan_age` at last observation — may undercount for truncated multi-year loans.

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
