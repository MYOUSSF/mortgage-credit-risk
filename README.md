# Mortgage Credit Risk Modelling

**Probability of Default · Loss Given Default · SHAP · Survival Analysis · Macro Stress Testing · Calibration**

Credit Risk Modelling implemented on the Freddie Mac Single-Family Loan Performance dataset (2000–2020 vintages, ~1.05M loans, ~64M loan-month records).

---

## Key Results

All figures below come from the full-dataset run on Kaggle (notebooks 1, 2, 3, 5 and 8), not from the bundled sample data. The OOS set is a random 30% holdout; the OOT set is a later period never seen in training, with a much lower default rate (0.12% of loan-months vs 0.58% in training).

**12-month PD classifiers** (Ch.1, Ch.2): current delinquency is an input

| Model | AUROC OOS | KS OOS | Gini OOS | AUROC OOT | Gini OOT |
|---|---|---|---|---|---|
| Logistic Regression (Ch.1) | 0.946 | 0.750 | 0.891 | 0.962 | 0.923 |
| XGBoost (Ch.2) | 0.951 | 0.760 | 0.901 | 0.964 | 0.929 |

**Discrete-time hazard models** (Ch.5b): delinquency excluded, forward-looking from loan characteristics only

| Model | Monthly hazard AUROC OOS / OOT | 12m PD AUROC (OOT snapshots) | 12m KS | 12m Gini | Predicted ÷ observed 12m PD |
|---|---|---|---|---|---|
| Logit hazard (champion) | 0.916 / 0.903 | 0.886 | 0.614 | 0.772 | 1.27 |
| XGBoost hazard (challenger) | 0.919 / 0.924 | 0.906 | 0.670 | 0.812 | 0.80 |
| Ch.2 XGBoost, same rows (benchmark) | — | 0.905 | 0.658 | 0.810 | 36.2 |

The 12-month rows are scored on the same 506,733 OOT loan-months at three snapshot dates (2018-06, 2019-06, 2020-06), whose 12-month outcome is fully observed.

**LGD** (Ch.3), OOS: the Fractional Response Model is champion, with RMSE 0.234, R² 0.385 and mean predicted LGD 0.503.

**Competing risks** (Ch.10): treating prepayment as censoring overstates PD measured from origination by **+47% at 36 months** and **~5× over the loan's life** (12.9% vs 2.6%).

> **The Cox PH concordance previously reported here (~0.85–0.89) has been removed as invalid.** It was
> computed on a dataset where each loan's time-varying covariates were read off its *last* observed row
> — the month immediately before default, for a defaulter — so the model was scored on the borrower's
> state at the brink of default. See [Ch.5](#ch5--survival-analysis-superseded) for the full disclosure.
> The Ch.5b figures above replace it, and come from `discrete_hazard_comparison.csv` rather than being transcribed by hand.

---

## Findings

What the full-dataset run shows, notebook by notebook. Figures are from the saved notebook outputs.

### Data (Notebook 1)

- **Scale.** 21 origination vintages (2000–2020) × 50,000 sampled loans gives ~1.05M loans. That is 64.2M loan-months in the competing-risks panel and 42.9M in the PD panel after the right-censoring filter.
- **The crisis signature is clear.** About 8–9% of 2006–2007 loans eventually defaulted (4,066 and 4,479 of 50,000), against under 1% for every vintage from 2009 onwards and 0.2% for 2016. Cumulative 60-month default reaches 3.8% for the 2006 vintage and 5.5% for 2007, against 0.4–0.8% for 2000–2004.
- **Prepayment, not default, is how mortgages end.** 87% of loans prepay. On the competing-risks training panel there are 37 prepayments for every default.
- **Strong population shift into OOT.** PSI between training and OOT exceeds the 0.25 "major shift" threshold for `ur_3m_lag` (13.1), `orig_interest_rate` (1.55) and `hpi_change` (1.06). The OOT window covers the 2020 unemployment spike and a very different rate environment, so the OOT metrics are a genuine out-of-regime test.

### PD classifiers (Notebook 2)

- **Current delinquency dominates.** `delinquency_indicator` has an IV of 2.13 and 79% of XGBoost's total gain. The next features are `orig_interest_rate` (IV 1.31), `hpi_change` (1.18) and `loan_age` (0.80). Much of the ~0.95 AUROC comes from recognising loans that are already delinquent.
- **XGBoost's edge over logistic regression is small:** +0.005 AUROC and +0.010 KS on OOS, and less on OOT. WoE-binned logistic regression captures most of the signal.
- **Both models hold up out of time.** OOT AUROC is higher than OOS for both, despite the population shift above. The OOT default rate is about five times lower, so this is not a like-for-like comparison.
- **The raw scores are not probabilities.** Both models reweight defaults about 172× (`class_weight='balanced'` / `scale_pos_weight`), so the score distributions are bimodal and centred far above the true default rate. On identical rows the Ch.2 model's mean score is 4.6% against an observed 0.13% (36× too high). Any use as a PD level has to go through Ch.7 calibration.

### LGD (Notebook 3)

- **Sample.** 15,464 resolved defaults: 10,099 train, 4,329 OOS and 1,036 OOT. Mean LGD is 0.49 and the distribution is bimodal, with 3.6% at exactly 0 and 7.9% at exactly 1 in training.
- **The three models tie on OOS RMSE:**

  | Model | OOS RMSE | OOS R² | OOT RMSE | OOT R² |
  |---|---|---|---|---|
  | FRM (champion) | 0.234 | 0.385 | 0.288 | 0.284 |
  | Two-Stage | 0.234 | 0.385 | 0.296 | 0.241 |
  | Random Forest | 0.240 | 0.354 | 0.302 | 0.213 |

  The FRM degrades least out of time.
- **Main drivers.** House-price change since origination is the strongest (FRM coefficient +1.83 on the logit scale; 32% of RF importance), followed by original balance and mortgage-insurance coverage. The largest state effects, relative to the AK reference, are NY (+2.02), DC (+1.92), IL, PA and OH (+1.64 each).
- **Only the two-stage model reproduces the point masses.** FRM and random forest predict a 0% share at both LGD = 0 and LGD = 1. The two-stage model predicts 3.3% and 8.5% against an observed 3.7% and 7.8% (OOS). Its stage-1 classifier still assigns almost every loan to "interior", so it gets the *shares* right without *identifying which* loans hit a boundary.

### Discrete-time hazard (Notebook 5)

- **The PD level is well calibrated in-sample, less so out of time.** Predicted ÷ observed monthly hazard is 1.04 (logit) and 1.02 (XGBoost) on OOS. On OOT it is 1.09 (logit) and 0.69 (XGBoost): XGBoost under-predicts the low-default OOT period by about 30%. On the 12-month snapshot test, logit over-predicts by 27% and XGBoost under-predicts by 20%.
- **XGBoost ranks better, logit is more conservative.** XGBoost matches the Ch.2 classifier's 12-month AUROC (0.906 vs 0.905) without using delinquency. It also has a Brier score 12× lower, because its probabilities are on the right scale.
- **Odds ratios per economic step** (logit): +1pp note rate 1.93; +10pp CLTV 1.51; +0.10 HPI ratio (house prices 10% lower than at origination) 1.30; +20 FICO 0.88; two borrowers vs one 0.58; owner-occupied vs investor 0.79. Unemployment adds only 1.03 per percentage point, holding the other covariates fixed.
- **Proportional hazards is rejected** for all three tested covariates. The LR statistics are `ur_3m_lag` 287, `orig_cltv` 122 and `credit_score` 116, against a 5% critical value of 12.6. The linear model's long-horizon PDs lean on an assumption the data do not support.
- **Seasoning.** The default hazard rises from zero to a peak around loan age 100–120 months. Both models track the observed hazard and Kaplan–Meier curve closely up to ~180 months. Beyond that the models diverge: the logit spline falls back towards zero after its ~120-month peak, while XGBoost's learned curve stays flat at its peak (trees extrapolate flat).
- **That tail divergence drives lifetime PD.** For the 210,207 scored OOS loans, mean PIT PDs agree closely up to 60 months (logit 0.70% / 1.48% / 2.32% / 4.05% at 12 / 24 / 36 / 60m; XGBoost within 0.05pp). Mean lifetime PD, however, is 10.3% (logit) against 18.1% (XGBoost). Lifetime PDs from either model should not be used without resolving this.
- **The TTC PD is below the PIT PD** for the logit model at every horizon (12m: 0.55% vs 0.70%). Scoring on the long-run macro mean (unemployment 6.55%, HPI ratio 0.978) is more benign than the scored loans' actual macro state.

### Competing risks (Notebook 8)

- **The naive conversion overstates PD, and the overstatement grows with horizon.** These PDs are cumulative from origination:

  | Horizon | Naive `1 − S(t)` | CIF (Cox) | CIF (multinomial) | Overstatement |
  |---|---|---|---|---|
  | 12m | 0.0073% | 0.0067% | 0.0069% | +9% |
  | 24m | 0.128% | 0.101% | 0.112% | +27% |
  | 36m | 0.457% | 0.312% | 0.337% | +47% |
  | Lifetime (307m) | 12.93% | 2.59% | 2.40% | +399% |

- **A model-free check agrees.** The Aalen–Johansen estimator computed directly on the panel gives the same picture: at 120 months, naive Kaplan–Meier gives 5.42% against an Aalen–Johansen CIF of 2.09% (+159%). Both model specifications track the empirical CIF closely for default and prepayment.
- **The two specifications agree on direction but not quite on level.** Every covariate effect has the same sign in the Cox and multinomial models. The CIF gap is +3.0% at 12m, +11.8% at 24m, +8.2% at 36m and −7.2% at lifetime, so it is outside the script's ±5% tolerance at 24 and 36 months. The absolute differences are small (0.026pp at 36m).
- **Covariates act on the two exits differently.** A higher note rate raises both hazards: default HR 1.15 per +1pp, prepayment HR 1.32. Doubling the original balance barely changes default (1.01) but raises prepayment (1.27). Higher FICO lowers default (0.95 per +20 points) but raises prepayment (1.04). Higher CLTV and DTI raise default and slow prepayment.
- **Expected life is a quarter of the contractual term:** mean 71 months (median 69) against a mean contractual remaining term of 308 months. It is 83 months for the OOT cohort.
- **What this means for Ch.5b.** The discrete hazard model also censors prepayment, so its multi-year and lifetime PDs are the naive kind. The populations differ (Ch.5b conditions on each loan's current age), but the size of the gap here implies Ch.5b's lifetime PDs are materially overstated for provisioning purposes.

### Issues the run surfaced

- **`refi_incentive` is missing from the competing-risks models.** The PMMS rate file was not found at preprocessing (`macro/pmms_30yr_fixed.csv`), so the dominant prepayment driver was dropped from both specifications. Add the file and re-run `01` and `12`.
- **Two EDA panels are empty.** Notebook 1's crisis default/90+ DPD time series and its class-imbalance panel rendered their "run preprocessing first" placeholders instead of data.
- **Notebook 1's preprocessing log is from a different run than the data the models used.** Its PD split sizes (21.4M / 9.2M / 12.3M) and panel end date (2025-09) do not match the `pd_*` files the models loaded (22.6M / 9.7M / 8.7M; OOT panel ending 2021-04). Its per-vintage "mean LGD" values (0.0006–0.043) also contradict the 0.49 in the `lgd_*` files. Regenerate the processed data in one run before quoting data-level statistics side by side with model results.

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
├── 11_discrete_hazard.py               # Ch.5b: discrete-time survival — monthly hazard,
│                                        #   conditional PIT/TTC PD (supersedes 06)
├── 12_competing_risks.py               # Ch.10: prepayment as a competing risk —
│                                        #   cumulative incidence vs the naive 1 - S(t)
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

Origination years 2000–2020 (50,000 sampled loans per vintage) yield ~64M loan-month records spanning the 2004–2008 subprime crisis. About 8–9% of the 2006–2007 vintages eventually defaulted, against under 1% of every vintage from 2009 on.

**Macro data (optional — materially improves discrimination):**
- FHFA HPI by 3-digit ZIP → `data/raw/macro/hpi_3digit_zip.csv`  ([FHFA](https://www.fhfa.gov/data/hpi))
- BLS unemployment LNS14000000 → `data/raw/macro/unemployment_rate.csv`  ([BLS](https://data.bls.gov/timeseries/LNS14000000))
- Freddie Mac PMMS 30-year fixed rate → `pmms_30yr_fixed.csv` in the macro directory  ([PMMS](https://www.freddiemac.com/pmms)). Without it `refi_incentive`, the main prepayment driver, is dropped from the competing-risks models (Ch.10).

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
python 11_discrete_hazard.py          # Ch.5b: discrete-time hazard — logit + XGBoost
python 12_competing_risks.py          # Ch.10: competing risks — CIF vs naive 1 - S(t)
python 10_basel_irb_capital.py        # Rating master scale + Basel IRB RWA/capital
```

> **Ordering note:** run `11_discrete_hazard.py` *before* `10_basel_irb_capital.py` — Ch.9 prefers the
> discrete-hazard TTC PD over Ch.5's Cox output. `11_discrete_hazard.py` also requires `default_date`
> in the PD parquet files, which means re-running `01_data_preprocessing.py` if your parquet files
> predate that change; the script aborts with an explicit message rather than building an empty target.
>
> `12_competing_risks.py` reads the `surv_*.parquet` variant, also emitted by `01_data_preprocessing.py`,
> and should run *before* `07_macro_scenario_analysis.py` if you want IFRS 9 lifetime ECL measured over
> expected rather than contractual life (`config.IFRS9_USE_EXPECTED_LIFE`).

> **Environment:** Kaggle notebooks (2×T4 GPU, 30 GB RAM) recommended for scripts 03–05. All scripts fall back to CPU gracefully.

### Configuration

Every script imports `config.py` for values that must stay identical across the pipeline: the random seed, the train/OOS/OOT split boundary, the default-event codes, the PD/LGD feature lists, GPU detection, the plot theme, the shared logging setup, PSI/IV rating thresholds, and the IFRS 9 macro scenario assumptions. Each script keeps its own local name for what it imports (e.g. `TARGET = config.TARGET_PD`), so change an assumption once in `config.py` and every script that depends on it picks it up — there's no second or third copy of `DEFAULT_CODES` or the macro scenario shocks to remember to update.

### Running on Kaggle (or any chained-notebook setup)

All paths are resolved in `src/config.py`, so neither the notebooks nor anything else in `src/` sets paths. `config.py` detects Kaggle (`/kaggle/input` exists) and switches to the Kaggle layout. Any path can still be overridden with an environment variable:

| Variable | Local default | Kaggle default | What it holds |
|---|---|---|---|
| `MCR_DATA_DIR` | `<repo>/data` | `/kaggle/working/repo/data` | Root for the writable directories below |
| `MCR_RAW_DIR` | `$MCR_DATA_DIR/raw/freddie_mac` | `<dataset>/freddie_mac` | Raw Freddie Mac `.txt` files |
| `MCR_MACRO_DIR` | `$MCR_DATA_DIR/macro` | `<dataset>/macro` | HPI / unemployment / PMMS |
| `MCR_PROC_DIR` | `$MCR_DATA_DIR/processed` | Notebook 1's output mount if attached, else `$MCR_DATA_DIR/processed` | Processed **datasets**: the `pd_*`, `lgd_*`, `surv_*` parquet |
| `MCR_OUT_DIR` | `$MCR_DATA_DIR/outputs` | same | Derived **artifacts**: metrics, coefficients, per-loan scores |
| `MCR_FIG_DIR` | `$MCR_DATA_DIR/figures` | same | Figures |

`<dataset>` is `config.KAGGLE_DATASET_DIR` (`/kaggle/input/datasets/youssefmousaaid/freddiemacmorgatge`). Notebook 1's output mount is `config.KAGGLE_UPSTREAM_PROC_DIR`. If you fork the notebooks, change those two constants.

**How the notebook chain works.** Script `01` is slow, so Notebook 1 runs it once and publishes the processed datasets from `/kaggle/working/repo/data/processed`. Notebooks 2–8 attach Notebook 1's output as an input; Kaggle mounts it **read-only**. When that mount is present, `config.py` reads the datasets from it and writes every script's results to the writable `/kaggle/working/repo/data`. Reading and writing different directories is what avoids `OSError: [Errno 30] Read-only file system`.

The scripts run as `!python src/...` subprocesses, which import `config.py` themselves, so they resolve the same paths as the notebook without anything being passed to them. If you do override a path with an `MCR_*` variable inside a notebook, set it **before** importing any pipeline module: each module binds its paths at import time.

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

Training-set IVs from the full-dataset run:

| Feature | IV | Strength |
|---|---|---|
| `delinquency_indicator` | 2.125 | Very strong |
| `orig_interest_rate` | 1.314 | Very strong |
| `hpi_change` | 1.177 | Very strong |
| `loan_age` | 0.801 | Very strong |
| `credit_score` | 0.618 | Very strong |
| `orig_cltv` | 0.530 | Very strong |
| `ur_3m_lag` | 0.504 | Very strong |
| `orig_dti` | 0.252 | Medium |
| `num_borrowers` | 0.129 | Medium |
| `property_type` | 0.031 | Weak |
| `orig_upb` | 0.012 | Negligible |
| `occupancy_status` | 0.007 | Negligible |

An IV above 0.5 normally prompts a leakage check. Here `delinquency_indicator` is a genuine current-state predictor, not leakage (post-default rows are excluded), but it is why the Ch.1/Ch.2 AUROCs are so high.

`class_weight='balanced'` upweights defaults ~172× (the non-default:default ratio in training), so raw scores are not PD levels; Ch.7 calibrates them. Hosmer–Lemeshow calibration test included. Results: OOS AUROC 0.946, KS 0.750, Gini 0.891; OOT AUROC 0.962.

---

### Ch.2 — XGBoost PD

```python
XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    scale_pos_weight=172,       # neg/pos ratio for class imbalance (computed at run time)
    tree_method="hist",         # 5–10× RAM reduction via histogram approx.
    device="cuda",              # auto-detected; CPU fallback
    early_stopping_rounds=20,   # halts when OOS AUC plateaus
)
```

GPU acceleration delivers ~15× speedup. On the full dataset XGBoost beats LR only narrowly: OOS AUROC 0.951 vs 0.946, OOT 0.964 vs 0.962. `delinquency_indicator` alone accounts for 79% of its total gain, so there is little left for non-linear interactions to add.

---

### Ch.3 — LGD Models

**Target:** `LGD = actual_loss / zero_balance_removal_upb`, clipped to [0, 1]

Three models compared on RMSE, MAE, R², and mean bias:

| Model | Key Property |
|---|---|
| Fractional Response (FRM) | Papke-Wooldridge quasi-binomial GLM (logit link) — targets E[LGD\|X] directly, no boundary clipping |
| Two-Stage Model | 3-class multinomial logit over {LGD=0, interior, LGD=1} + beta regression on the interior — models the point masses explicitly |
| Random Forest | 200 trees, max depth 6 |

> **Previously four models** (the above plus a natural spline regression and an XGBoost regressor). The spline and XGBoost models were removed: all four estimated the same thing — the conditional *mean* of LGD — differing only in functional flexibility. On ~150 training observations that is the wrong axis to spend model risk on. The two-stage model replaces them because it answers a different question (below).

**Why a two-stage model.** Realised LGD is not smooth on [0, 1]. It is bimodal, with genuine point masses at exactly 0 (the disposition covers the balance) and exactly 1 (nothing recovered), and a diffuse spread of partial losses between them. A conditional-mean model can match the average of that distribution while placing almost no probability on either mass — reporting a confident 0.45 for a population in which hardly any loan actually loses 45%. The two-stage model represents the masses explicitly:

```
Stage 1:  P(class | x)  over {LGD=0, interior, LGD=1}     multinomial logit, L2
Stage 2:  E[LGD | x, interior] = μ(x)                      beta regression, constant φ

E[LGD | x] = P(LGD=1|x)·1 + P(interior|x)·μ(x) + P(LGD=0|x)·0
```

`predict()` returns that conditional mean, so RMSE/MAE/R²/bias and champion selection treat it exactly like the other two. Unlike them it also yields the *full conditional distribution* — `predict_components()` and `predict_quantile()` — which is what downturn/stressed-LGD work needs and a mean-only model cannot supply. `lgd_point_mass_calibration.csv` reports the observed share of LGD=0 and LGD=1 against each model's predicted share; for the FRM and the random forest that predicted share is ~0, which is the point of the comparison rather than an incidental diagnostic.

**Beta regression is hand-written, deliberately.** `statsmodels.othermod.betareg.BetaModel` does not support observation weights — and it *accepts* a `weights=` keyword, emits only a `ValueWarning` (which this pipeline suppresses), and returns bit-identical unweighted estimates. Silently dropping the IPCW weights would reintroduce exactly the truncation bias they exist to correct, so `weighted_beta_nll()` implements the weighted Ferrari–Cribari-Neto mean/precision log-likelihood with analytic gradients, maximised by L-BFGS-B.

**Categorical encoding fix.** Categoricals are now one-hot encoded with a dropped reference level, rare levels (`< config.LGD_MIN_LEVEL_COUNT`) and unseen levels grouped into `"other"`, with encoders fitted on the **training split only**. The previous implementation label-encoded categoricals into integer codes and fed those to the FRM as if continuous — asserting, for example, that `property_state` has one linear slope across alphabetically ordered states, AK→AL→AR being a unit step of equal effect each time. It also fitted the encoder on `concat([train, oos, oot])`, leaking the evaluation splits' level sets into the training encoding. **Any FRM coefficient or metric produced before this fix is invalid and should be regenerated rather than quoted.** The random forest keeps integer codes (a tree can isolate any subset by splitting, so the ordering imposes no functional-form assumption), but now fitted on train only, with a dedicated code for unseen levels.

**Small-sample fallbacks**, all logged explicitly rather than applied silently: a boundary class with fewer than `config.LGD_TWO_STAGE_MIN_CLASS_OBS` training rows is dropped from stage 1 and assigned probability 0; too few interior rows, or a beta optimiser that fails, falls back to a constant IPCW-weighted interior mean with a method-of-moments φ; and the FRM tries its unpenalised (auditable) fit first, falling back to an L2-penalised fit only on non-convergence or perfect separation. `predict()` never raises because a class was absent in training.

All three are fit with inverse-probability-of-censoring (IPCW) sample weights correcting for LGD workout-period truncation bias: `extract_lgd_rows()` correctly keeps only *resolved* defaults (no leakage), but that resolved-case sample is truncated — a loan entering workout (90+ days past due) close to the dataset's end and taking a long time to resolve (contested foreclosure, REO) is systematically missing, while fast-resolving cases (short sales) are always captured even near the cutoff. `compute_ipcw_weights()` estimates the truncation ("censoring") distribution via a reversed Kaplan-Meier fit and reweights resolved cases accordingly, implemented in `01_data_preprocessing.py`.

The lowest-RMSE model on OOS (falling back to Train for small samples) is selected as champion and its mean predicted LGD is saved to `lgd_champion_summary.csv` — this is the value Ch.6's macro scenario ECL uses as its LGD anchor, rather than a flat assumption disconnected from this model suite. That file's schema is unchanged by the move to three models.

**Metrics** (full-dataset run, 10,099 train / 4,329 OOS / 1,036 OOT resolved defaults):

| Model | OOS RMSE | OOS R² | OOS bias | OOT RMSE | OOT R² |
|---|---|---|---|---|---|
| FRM (champion) | 0.234 | 0.385 | +0.010 | 0.288 | 0.284 |
| Two-Stage | 0.234 | 0.385 | +0.013 | 0.296 | 0.241 |
| Random Forest | 0.240 | 0.354 | +0.012 | 0.302 | 0.213 |

The FRM is champion, with a mean predicted LGD of 0.503 passed to Ch.6. Only the two-stage model reproduces the boundary shares: it predicts 3.3% at LGD = 0 and 8.5% at LGD = 1, against an observed 3.7% and 7.8% on OOS. FRM and RF predict 0% for both. These figures post-date the encoding fix, so earlier LGD figures should not be quoted.

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

### Ch.5 — Survival Analysis (superseded)

> **⚠️ This chapter's Cox model is superseded by [Ch.5b](#ch5b--discrete-time-survival-monthly-hazard).**
> `06_survival_analysis.py` is retained to document the approach and as a reference point. Its outputs
> should not be used, and its reported C-index should not be quoted. Two confirmed defects:

**1. Covariate leakage from end of follow-up.** `build_survival_df()` collapses the loan-month panel to
one row per loan — the row with the maximum `loan_age` — and reads the time-varying covariates
(`delinquency_indicator`, `hpi_change`, `ur_3m_lag`) off that last row. For a loan that defaults, the last
retained row is the month immediately *before* default, so the model is handed the borrower's state at the
brink of default and asked to predict default. `delinquency_indicator` is the worst offender: on that row
it is close to a default flag. The resulting concordance (~0.85–0.89) is inflated by construction and is
**not** a forecast performance estimate.

**2. Horizon PDs are not conditional on the loan's current age.** `compute_horizon_pds()` returns
`1 − S(h|x)` — the probability of default within `h` months measured *from origination* — and assigns it to
a loan that has already survived to age `a`. The quantity actually wanted is:

```
PD(a → a+h) = 1 − S(a+h | x) / S(a | x)
```

The figure produced overstates PD for any seasoned loan, because it charges the loan again for the default
risk of the years it has already survived.

The original Cox specification, for reference:

```
h(t | x) = h₀(t) · exp(x'β)
```

Kaplan-Meier curves by FICO tertile and vintage era, hazard ratios with confidence intervals, and
Schoenfeld residual PH checks are all still produced by the script, and the KM curves in particular remain
a valid non-parametric description of the data.

---

### Ch.5b — Discrete-Time Survival (Monthly Hazard)

`11_discrete_hazard.py` replaces the Cox model with a **discrete-time survival** framework fitted directly
on the loan-month panel, in **two implementations** sharing one target, one covariate set, one subsample
and one scoring engine:

| | Linear | Gradient-boosted |
|---|---|---|
| Model | `logit h(t|x) = α(t) + x'β` (statsmodels GLM) | `logit h(t|x) = f(t, x)` (XGBoost) |
| Baseline hazard α(t) | B-spline on `loan_age`, 6 df | learned by the trees |
| Proportional hazards | assumed, **LR-tested** | not assumed |
| Role | champion — auditable, default Basel capital source | challenger — higher flexibility |
| Card | [discrete_hazard_logit.md](docs/model_cards/discrete_hazard_logit.md) | [discrete_hazard_xgb.md](docs/model_cards/discrete_hazard_xgb.md) |

**Every loan-month row is one observation — no data expansion.** A row at age `a` is a Bernoulli trial for
"does this loan default in month `a+1`". Fitting a binary model to that target on the panel *is* maximum
likelihood for the discrete-time survival model, because the panel likelihood factorises into one Bernoulli
term per loan-month at risk (Allison 1982; Singer & Willett 1993). Each row carries its own contemporaneous
covariates, so defect (1) above cannot arise.

**Horizon PD is conditional by construction:**

```
PD(a → a+h) = 1 − ∏_{j=1..h} [1 − ĥ(a + j | x, macro_j)]
```

`loan_age` advances deterministically, so the product starts at the loan's *present* age — the loan is never
charged again for months it has already survived. This fixes defect (2). Horizons: 12m / 24m / 36m / 60m,
plus a lifetime PD capped at each loan's `remaining_months`.

**Case-control subsampling + King & Zeng prior correction.** The panel is ~99.95% non-events. Every event
row is kept; non-event rows are kept with probability `r`. Under this design the sampled odds are the true
odds divided by `r`, so `logit(h_true) = logit(h_sampled) + log(r)` — an exact intercept shift under a logit
link (which is why logit is preferred to cloglog; at a ~0.05% monthly hazard the two links are numerically
indistinguishable anyway). The correction is applied *inside* `predict_hazard()`, identically for both
models (for XGBoost, to the raw margin), so every downstream consumer sees corrected probabilities. Being
monotone, it changes the PD **level** — and therefore ECL and capital — while leaving AUROC, KS and Gini
untouched.

**Internal covariates are deliberately excluded from both models.** `delinquency_indicator` and
`delinquency_status` are not used. A multi-period horizon PD needs a projected value for every covariate at
every future month; delinquency at month `a+j` is itself an *outcome* of the same deterioration process that
produces default, so projecting it would require a second model of delinquency transitions, and holding it
flat would assert that a current loan stays current for 60 months (driving its lifetime PD towards zero).
Including it would also reproduce the Cox leakage in subtler form. The cost — the models cannot distinguish
a current from a seriously delinquent loan with identical origination characteristics — is covered
elsewhere for IFRS 9: Ch.6's 30-DPD and 90-DPD backstops drive stage allocation on exactly that information.

**PIT vs TTC** come from the same fitted model, differing only in the macro path handed to the scoring
engine: PIT holds each loan's current `ur_3m_lag` / `hpi_change` flat across the horizon (an explicit
assumption, not a forecast); TTC pins them at their long-run training-sample mean, for Basel IRB capital
(EBA/GL/2017/16 §6.2). `compute_conditional_horizon_pd()` accepts an explicit per-period macro path — a
scalar, a length-`h` vector, or an `n_loans × h` array — which is the interface a future IFRS 9 scenario
projection would use.

**Validation — only what is honest.** No concordance on end-of-follow-up covariates:

- Monthly hazard AUROC and log loss on **unsampled** OOS and OOT rows (log loss and the predicted÷observed
  ratio are what confirm the prior correction worked; AUROC cannot, since the correction is monotone)
- 12-month conditional PD vs the realised 12-month outcome at fixed snapshot dates, restricted to rows whose
  12-month window is **fully observed** (the `filter_immature_right_censored()` maturity rule) — AUROC, KS,
  Gini, Brier, decile calibration
- The Ch.2 12-month XGBoost benchmark **on exactly those rows**. `03_pd_ensemble.py` persists no row keys in
  `pd_xgb_results.csv`, so the Ch.2 specification is re-fitted to make an identical-population comparison
  possible (as `07_macro_scenario_analysis.py` already does); the metrics CSV records which path was used
- Observed Kaplan-Meier over `loan_age` (events ÷ at-risk, computed from the panel) vs mean predicted survival
- Proportional-hazards LR tests (`covariate × spline(loan_age)` interaction) for the linear model only

All three models land in one comparison table, `discrete_hazard_comparison.csv`, on identical rows.

---

### Ch.10 — Competing Risks (Prepayment)

Chapters 1–5b treat voluntary prepayment (zero-balance code `01`) as **censoring**: the loan stops appearing
in the panel, and Ch.5 converts its Cox survival function to a default probability with `1 − S(t)`.
`12_competing_risks.py` treats it as a **competing risk** instead, and measures how much the old conversion
overstated lifetime default.

**Why censoring is the wrong model.** Censoring means "this loan is still at risk; we simply stopped
watching". A prepaid mortgage is not: the lien is released and it can never default. So `1 − S(t)` estimates
the probability of default in a world where prepayment has been *abolished* and prepaid loans stay exposed
forever. That is a coherent quantity — a net, cause-removed risk — but it is not the one a provision is
built on, which is the **crude** probability: the chance this loan defaults before anything else happens to
it. That is the cumulative incidence function, and it is always smaller. On the full training panel
**37 loans prepay for every one that defaults** (87.5% prepay, 2.4% default), so the gap is not a rounding detail.

```
S(t)     = exp( −( H_d(t) + H_p(t) ) )            overall survival uses BOTH hazards
CIF_d(t) = Σ_k  [ S(t_{k−1}) − S(t_k) ] · dH_d / dH        crude default probability
```

**The inequality `CIF_d(t) ≤ 1 − S_d(t)` is an identity, not a finding.** Both accumulate the same
cause-specific hazard increments; the CIF weights each by survival from *both* risks, the naive figure by
survival from default alone. A run where the naive figure comes out lower is a bug, and the script asserts
this at every grid point — as do the tests.

**Two independent specifications**, deliberately not sharing an implementation:

| | (a) Cause-specific Cox | (b) Discrete-time multinomial |
|---|---|---|
| Data shape | one row per loan | loan-month panel |
| Fit | two `lifelines` Cox models — default with prepayment censored, and vice versa | 3-class multinomial logit over {survive, default, prepay} |
| Baseline hazard | non-parametric, built in | binned months-since-origination (`config.CR_DURATION_BIN_EDGES`) — a logit has none of its own |
| Combination | `CIF_d(t) = Σ dH_d · S(t−1)` | `S(k) = Π (1 − h_d − h_p)`, `CIF_d(t) = Σ h_d(k) · S(k−1)` |
| Inference | partial-likelihood SEs | **SEs clustered by `loan_seq_num`** |

The multinomial is a multinomial rather than two binary logits for a structural reason: the shared softmax
denominator makes `h_d + h_p < 1` an algebraic guarantee. Two independent binary models are each free to
predict 0.7, and the implied per-period survival then goes negative, silently corrupting every later term of
the survival product. Clustering matters for the same kind of reason — consecutive months of one loan are
not independent draws, and unclustered SEs on a loan-month panel are optimistic by roughly √(months/loan).

**A new dataset variant, not a replacement.** `01_data_preprocessing.py` now also emits
`surv_{train,oos,oot}.parquet`, which retain prepayment as an event and carry a 3-class `event_type`
(0 censored / 1 default / 2 prepay) and a `duration_months`. `pd_*.parquet` and `survival_pd_horizons.csv`
are untouched — Ch.5 remains the naive baseline this chapter is measured against.

**`duration_months` is derived from dates, never from `loan_age`.** The Freddie Mac `loan_age` field
*resets* when a loan is modified (Modification Flag Y/P), so a loan modified at month 30 can report
`loan_age = 1` the following month. Using it as the survival clock rewinds time for exactly the distressed
loans whose timing matters most, and manufactures a spurious mass of "young" defaults. On the sample data
`loan_age` understates true seasoning by up to 35 months on modified loans. `loan_age` is still carried as
an ordinary covariate.

**IFRS 9 expected life (§5.5.19).** The per-loan expected life — `Σ_t S(t)`, the area under the
competing-risks survival curve — is written to `competing_risks_expected_life.csv` and consumed by Ch.6,
which truncates projected exposure beyond it instead of amortizing over the contractual term. Switchable
via `config.IFRS9_USE_EXPECTED_LIFE` (`False` reproduces the old behaviour exactly). On the full dataset
expected life is **71 months on average against a contractual 308** — under a quarter. Note that the
amortization *schedule* still runs on the contractual term: a loan does not pay down faster because it is
expected to prepay early, it follows its schedule and then disappears.

**Primary output**, `data/processed/competing_risks_comparison.csv`:

| horizon | naive_1_minus_S | cif_cox | cif_multinomial | overstatement_pct |
|---|---|---|---|---|
| 12m | 0.0073% | 0.0067% | 0.0069% | +9% |
| 24m | 0.128% | 0.101% | 0.112% | +27% |
| 36m | 0.457% | 0.312% | 0.337% | +47% |
| lifetime (307m) | 12.93% | 2.59% | 2.40% | +399% |

`overstatement_pct = (naive − cif_cox) / cif_cox × 100`, so it reads as "the naive figure is N% larger than
the correct one". The overstatement compounds with horizon — small at 12 months, large at lifetime — which
is precisely why it matters most for the Stage 2/3 lifetime ECL that Ch.6 computes.

---

### Ch.6 — Macro Scenario Analysis

> **Lifetime horizon:** `amortized_ead()` truncates projected exposure at each loan's IFRS 9 expected life when `config.IFRS9_USE_EXPECTED_LIFE` is on and [Ch.10](#ch10--competing-risks-prepayment) has been run — see that section. Note the binding constraint is usually `config.N_QUARTERS` (20 quarters = 5 years), which is already far shorter than either expected or contractual life.

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

PD input is TTC, not PIT — Basel IRB capital wants a PD that doesn't move with the current point in the cycle, the opposite of what IFRS 9 provisioning uses. The script prefers Ch.7's LRADR-anchored `ttc_calibrated_pd.csv`, then Ch.5b's `discrete_hazard_{model}_pd_horizons.csv` (`config.DISCRETE_HAZARD_CAPITAL_MODEL` selects which hazard model, default `logit`), and only then Ch.5's Cox `ttc_pd_12m` as a last resort — the Cox file is ranked last because of the leakage and non-conditional-horizon defects documented in [Ch.5](#ch5--survival-analysis-superseded).

**Known limitation:** LGD is Ch.3's population-level champion-model anchor, the same constraint documented for Ch.6's macro ECL engine, and it is not downturn-adjusted — Basel requires a downturn LGD for capital, distinct from the average LGD used for ECL.

---

## Regulatory Alignment

| Framework | Coverage |
|---|---|
| Basel II/III IRB | PD (TTC) + LGD, rating master scale, retail mortgage RWA/capital formula (Ch.9) |
| IFRS 9 / CECL | Per-loan Stage 1/2/3 (SICR + DPD backstops), 12m/lifetime staged ECL, probability-weighted scenarios (Ch.6); expected-life measurement §5.5.19 via competing-risks survival (Ch.10) |
| BCBS 239 | SHAP waterfall/segment reports — Principles 6 & 11 |
| OCC SR 11-7 | OOS/OOT backtesting, ongoing PSI monitoring (Ch.8), HL calibration test |
| EBA GL/2017/16 | Survival-based PIT and TTC PD (Ch.5b discrete-time hazard; Ch.5 Cox superseded); proportional-hazards LR testing |

---

## Model Documentation

Each model has its own model card under [`docs/model_cards/`](docs/model_cards/README.md) — intended use, key assumptions, performance, validation approach, and an ongoing monitoring plan, one level more specific than the shared "Known Limitations" below:

| Model | Card |
|---|---|
| PD — Logistic Regression (Ch.1) | [pd_logistic_regression.md](docs/model_cards/pd_logistic_regression.md) |
| PD — XGBoost (Ch.2) | [pd_xgboost.md](docs/model_cards/pd_xgboost.md) |
| LGD Model Suite (Ch.3) | [lgd_models.md](docs/model_cards/lgd_models.md) |
| Survival — Cox PH (Ch.5) | [survival_cox.md](docs/model_cards/survival_cox.md) — ⚠️ superseded |
| Discrete Hazard — Logistic (Ch.5b) | [discrete_hazard_logit.md](docs/model_cards/discrete_hazard_logit.md) |
| Discrete Hazard — XGBoost (Ch.5b) | [discrete_hazard_xgb.md](docs/model_cards/discrete_hazard_xgb.md) |
| IFRS 9 Macro Scenario ECL & Staging (Ch.6) | [ifrs9_macro_scenario.md](docs/model_cards/ifrs9_macro_scenario.md) |
| PD Calibration (Ch.7) | [calibration.md](docs/model_cards/calibration.md) |
| Basel IRB Rating Scale & Capital (Ch.9) | [basel_irb_capital.md](docs/model_cards/basel_irb_capital.md) |

---

## Known Limitations

1. **Prepayment is a competing risk only in Ch.10.** `12_competing_risks.py` treats voluntary prepayment properly, but chapters 1–5b still censor it: the Ch.1/Ch.2 12-month PD models, the Ch.5b discrete hazard and the Ch.5 Cox model all leave a prepaid loan out of the at-risk set without modelling why it left. For a 12-month horizon the distortion is small (the overstatement compounds with horizon), but any *lifetime* quantity taken from those chapters — including Ch.6's Stage 2/3 lifetime ECL PD input — inherits it. Ch.10 measures the size of the error: on the full dataset the naive figure is 27% too high at 24 months, 47% at 36 months and ~5× over the loan's life. It does not retrofit the correction upstream.
2. **Ch.6's "lifetime" horizon is 5 years, not a lifetime.** `config.N_QUARTERS = 20` caps every projection at 20 quarters. Switching Ch.6 from contractual maturity to expected life (71 vs 308 months on the full dataset) therefore changes ECL by very little, because the 60-month projection window binds long before either. The expected-life correction is implemented, switchable and correct, but it will only move ECL materially if `N_QUARTERS` is raised to a genuine lifetime horizon.
3. **Expected-life truncation is a hard cutoff, not a survival weighting.** Exposure drops to zero past the expected life. The fully correct treatment weights each quarter's exposure by the probability the loan is still alive, `S(q)`, which needs the per-loan survival *curve* rather than its integral.
4. **Origination-time covariates are held flat in the CIF projection.** Both competing-risks specifications project each loan's covariates — including `refi_incentive`, the dominant prepayment driver — at their origination values. A genuine refinancing wave is therefore not anticipated, the same flat-projection caveat that applies to the Ch.5b PIT macro path. In the full-dataset run `refi_incentive` was absent altogether, because the PMMS file was missing at preprocessing (see [Data](#data)).
5. **The `surv_*` OOT split is by origination date, not report date.** `split_pd()` cuts OOT by row, so a loan's months can straddle in-sample and OOT. That is harmless for a snapshot classifier but fatal for a survival model, whose duration is a property of the whole history. `split_survival()` therefore assigns whole loans by origination date. The two splits use the same `OOT_CUTOFF` but are not row-identical, so Ch.10 metrics are not directly comparable to Ch.1–5b metrics on "the same" OOT set. The train/OOS holdout is also drawn **per origination-year cohort** rather than in one pass over the combined panel — this is what lets `01` stream the panel to disk instead of concatenating ~30M loan-months in RAM, and it is safe because a Freddie Mac `sample_svcg_YYYY.txt` holds each loan's complete history, so no loan spans two cohorts. The design (loan-grouped, ~`OOS_FRAC` held out) is unchanged; which specific loans land in OOS differs from a single global draw.
6. **Sentinel handling at load is global, not field-specific.** `load_orig_year()` applies one `na_values` list (`9`, `99`, `999`, …) to every origination column, but Freddie Mac's sentinels are field-specific (9999 FICO, 999 DTI/LTV/CLTV/MI%, 99 units/borrowers, 9 occupancy/channel/purpose). A legitimate `orig_cltv` or `orig_ltv` of exactly 99 is therefore read as missing. This predates the competing-risks work and is inherited by every chapter including `surv_*`; fixing it would change `pd_*` and so every downstream chapter's output, which is why it has been left as a documented defect rather than changed in passing.
7. **LGD sample size:** ~150 defaults in the bundled sample dataset (the full dataset has 15,464 resolved defaults, of which 10,099 are in training). Pre-2010 crisis vintages recommended. On the sample data this is the binding constraint on Ch.3: it is why the suite was reduced to three models with distinct purposes rather than four flexible mean estimators, why rare categorical levels are pooled into `"other"` before one-hot encoding (`property_state` alone would otherwise contribute ~50 indicators to a ~150-row regression, most identifying a single loan), and why every model carries an explicitly logged small-sample fallback. Check the run log for which fallbacks fired before quoting any LGD metric.
8. **12-month window immaturity:** `filter_immature_right_censored()` drops loan-months too close to the dataset's true end to know their 12-month outcome (never-observed-to-default AND within 365 days of the panel's max `report_date`), so a right-censored active loan isn't mislabelled as a confirmed non-default. Rows with a known (even distant) `default_date` keep their label regardless of proximity to the cutoff.
9. **No hyperparameter tuning:** Cross-validated grid search could improve OOT AUROC by 1–3 points.
10. **Scenario LGD is population-level, not per-loan:** the macro ECL engine now anchors LGD to Ch.3's champion model output and scales it with scenario HPI (`scenario_lgd()`), rather than a flat 40% assumption — but every loan in a given scenario-quarter still gets the same LGD, since the OOS population scored for PD doesn't carry the LGD-specific features (`hpi_change_since_orig`, `mi_pct`, etc.) needed for true per-loan conditioning. A production system would persist those features alongside the PD population so LGD could vary by loan, not just by scenario and quarter.
11. **Prior LGD results are invalid under the encoding fix:** before the Ch.3 rewrite, categoricals were label-encoded into integer codes and passed to the Fractional Response Model as continuous covariates — so an FRM coefficient on `property_state` described a single linear slope across alphabetically ordered states. The encoder was also fitted on `concat([train, oos, oot])`, leaking the evaluation splits' level sets into the training encoding. Both are fixed (train-only one-hot with a dropped reference level and rare-level pooling), but any FRM coefficient or metric produced before the fix should be regenerated, not quoted.
12. **The two-stage LGD model's tail behaviour rests on a constant-precision beta:** stage 2 fits `logit(μ) = Xγ` with a single precision parameter φ shared across all loans, so the *spread* of partial losses is assumed not to vary with loan characteristics even though the *mean* does. `predict_quantile()` inherits that assumption, which matters precisely where it would be used — a downturn LGD read off a high quantile. Modelling φ with its own covariates is the natural extension and is not done here, because at ~150 observations on the sample data (of which fewer still are interior) there is not enough data to identify it. The full dataset's 8,937 interior training rows may be enough to try.
13. **Ch.5 Cox model is superseded and its metrics are invalid:** `06_survival_analysis.py` collapses the panel to one row per loan and reads time-varying covariates off that last row — the month before default, for a defaulter — so its reported C-index is inflated by construction, and its horizon PDs are measured from origination rather than conditional on the loan's current age. Both are fixed in Ch.5b (`11_discrete_hazard.py`); the Cox script is retained for reference only.
14. **Discrete hazard: PIT macro projection is flat.** The PIT horizon PD holds each loan's current `ur_3m_lag` / `hpi_change` constant across the whole projection — an explicit "conditions stay as they are today" assumption, not a forecast, since this pipeline contains no macro forecasting model. At long horizons a loan observed in a recession is projected as if the recession never ends (and one at a cyclical peak as if the expansion never does), so 24m–60m PIT PDs are more dispersed across loans than a mean-reverting path would produce. The TTC path is the mean-reverting counterpart, and `compute_conditional_horizon_pd()` accepts an explicit scenario path for anyone who wants one.
15. **Discrete hazard: internal (delinquency) covariates are excluded.** `delinquency_indicator` and `delinquency_status` are deliberately absent from both hazard models — their future path is an outcome of the default process itself and cannot be projected over a multi-period horizon without a second model of delinquency transitions, and on the row before default they are near-deterministic in the target. The cost is that the models cannot distinguish a current loan from a seriously delinquent one with otherwise identical characteristics; for IFRS 9 that information drives staging instead (Ch.6's 30/90-DPD backstops), not the PD level.
16. **Discrete hazard: the PD level depends on the sampling correction.** Both models are fitted on a case-control subsample (all events, non-events at rate `r`) and corrected by the King & Zeng logit shift `+log(r)`. The correction is exact under a logit link and is validated against the realised base rate in the metrics (`pred_obs_ratio`), but it does mean the absolute PD level — and therefore every ECL and capital figure derived from it — rests on that correction being right, where discrimination metrics would be unaffected by an error in it. Check `pred_obs_ratio` in `discrete_hazard_*_metrics.csv` before trusting any level-sensitive output.
17. **Discrete hazard (XGBoost): trees extrapolate flat.** Beyond the oldest `loan_age` or the most extreme macro values seen in training, the boosted model's predicted hazard stops responding — a 300-month projection sees the same hazard as a 60-month one once past the training range, and a macro shock more severe than anything observed is treated as if it were the worst observed. The linear model extrapolates linearly in the logit (with a constant-extrapolated spline baseline). Neither is right; they are wrong differently, which is why both are kept. On the full dataset this is not academic. The two models' mean PDs agree within 0.05pp out to 60 months, but mean lifetime PD is 10.3% for the logit model and 18.1% for XGBoost. The logit spline's hazard falls after its ~120-month peak, while XGBoost holds its peak hazard flat for the rest of the projection. Lifetime PDs should not be quoted from either model until this is resolved.
18. **Survival duration (Ch.5, historical):** in the superseded Cox script, duration is `loan_age` at the last retained pre-default observation, plus one reporting period for actual defaulters (since `extract_pd_rows()` drops the default row itself to prevent leakage in the binary target) — the closest recoverable approximation to true time-to-default given that constraint.
19. **EAD amortization is schedule-only:** `amortized_ead()` projects a standard declining-balance schedule from current UPB/rate/remaining term; it does not model stochastic prepayment beyond that schedule, so realised future balances (and therefore realised EAD) could decline faster than projected.
20. **LGD IPCW onset trigger is a proxy:** the workout-period truncation correction (`compute_ipcw_weights()`) defines "onset" as first reaching 90+ days past due — a standard regulatory default trigger, but distinct from (and possibly earlier than) the actual start of a formal workout/foreclosure process, which isn't separately recorded in the fields this pipeline reads.
21. **PD-at-origination proxy for SICR:** `07_macro_scenario_analysis.py`'s `compute_origination_pd()` approximates each loan's PD at initial recognition by re-scoring with `loan_age` forced to 0, holding every other feature (including current macro state) fixed — not the loan's actual historical origination-time PD, which this pipeline doesn't persist per loan. A production system would snapshot and store the underwriting-time score itself.
22. **Rating master scale bounds are illustrative:** `config.RATING_SCALE`'s PD upper bounds are standard S&P/Moody's-style anchor points, not calibrated to this portfolio's realised default experience — an institution would fit its own master scale before using it for disclosure or limit-setting.
23. **Basel capital LGD is not downturn-adjusted:** `10_basel_irb_capital.py` uses the same population-level, average-conditions LGD anchor as Ch.6's ECL engine (limitation #10). Basel IRB capital formally requires a downturn LGD — the LGD expected under adverse economic conditions — which is typically higher and would increase the computed capital requirement.

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
13. Allison, P. D. (1982). Discrete-time methods for the analysis of event histories. *Sociological Methodology*, 13, 61–98.
14. Singer, J. D. & Willett, J. B. (1993). It's about time: using discrete-time survival analysis to study duration and the timing of events. *Journal of Educational Statistics*, 18(2), 155–195.
15. King, G. & Zeng, L. (2001). Logistic regression in rare events data. *Political Analysis*, 9(2), 137–163.

---

*Python 3.11 · pandas · scikit-learn · XGBoost · SHAP*
