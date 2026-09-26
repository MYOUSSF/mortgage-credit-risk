"""
=============================================================================
Mortgage Credit Risk Modelling  |  Shared Configuration
=============================================================================
Single source of truth for constants that were previously copy-pasted,
identically, across the numbered pipeline scripts: the random seed, the
train/OOS/OOT split boundary, the default-event codes, the PD/LGD feature
lists, GPU detection, the shared plot theme, the logging setup boilerplate,
the PSI/IV rating thresholds, and the IFRS 9 macro scenario assumptions.

The duplication this replaces was already an active liability, not just a
readability issue: DEFAULT_CODES had drifted between the README and the
code, and PLT_STYLE / FEATURES / CAT_FEATURES / _detect_gpu() existed in
near-identical copies across 4-5 files with no guarantee an edit to one
would be reflected in the others.

Each script imports what it needs and keeps its own local name for it
(e.g. `TARGET = config.TARGET_PD`), so the ~600-900 lines of pipeline logic
in each script don't need to change — only the top "CONFIGURATION" block.

This module has no import-time side effects (no directory creation, no
logging setup) — each script still calls FIG_DIR.mkdir(...) and
config.configure_logging(...) explicitly, so merely importing config.py
does nothing to the filesystem or the logging system.
=============================================================================
"""

from __future__ import annotations
import os
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42


# =============================================================================
# PATHS
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent    # src/config.py → repo root

# Kaggle layout. Inputs are read-only mounts under /kaggle/input; only
# /kaggle/working is writable, and it is what a notebook publishes as its
# output. Detected here so that the notebooks AND the scripts they launch with
# `!python src/...` (separate processes, which import this module themselves)
# resolve the same paths without any notebook having to set them.
ON_KAGGLE = Path("/kaggle/input").is_dir()
# Raw Freddie Mac .txt files (freddie_mac/) and the macro CSVs (macro/).
KAGGLE_DATASET_DIR = Path("/kaggle/input/datasets/youssefmousaaid/freddiemacmorgatge")
# Notebook 1 (01-eda-ipynb) writes the processed pd_*/lgd_*/surv_* datasets
# under /kaggle/working/repo/data; notebooks 2-8 attach that notebook's output
# and read them from this mount.
KAGGLE_UPSTREAM_PROC_DIR = Path(
    "/kaggle/input/notebooks/youssefmousaaid/01-eda-ipynb/repo/data/processed")

if ON_KAGGLE:
    _default_data  = Path("/kaggle/working/repo/data")
    _default_raw   = KAGGLE_DATASET_DIR/"freddie_mac"
    _default_macro = KAGGLE_DATASET_DIR/"macro"
else:
    _default_data  = REPO_ROOT/"data"
    _default_raw   = None     # derived from DATA_DIR below
    _default_macro = None

# Every path can still be overridden with its MCR_* environment variable.
DATA_DIR  = Path(os.environ.get("MCR_DATA_DIR", _default_data))

RAW_DIR   = Path(os.environ.get("MCR_RAW_DIR",  _default_raw or DATA_DIR/"raw"/"freddie_mac"))
MACRO_DIR = Path(os.environ.get("MCR_MACRO_DIR", _default_macro or DATA_DIR/"macro"))
# On Kaggle, read notebook 1's published datasets when they are attached (the
# modelling notebooks); otherwise — i.e. when running notebook 1 itself —
# write them under DATA_DIR, which is where that mount is published from.
PROC_DIR  = Path(os.environ.get(
    "MCR_PROC_DIR",
    KAGGLE_UPSTREAM_PROC_DIR if ON_KAGGLE and KAGGLE_UPSTREAM_PROC_DIR.is_dir()
    else DATA_DIR/"processed"))

OUT_DIR = Path(os.environ.get("MCR_OUT_DIR", DATA_DIR/"outputs"))
FIG_DIR = Path(os.environ.get("MCR_FIG_DIR", DATA_DIR/"figures"))
# Scratch space for 01_data_preprocessing.py's per-origination-year chunks.
# Lives under PROC_DIR because 01 removes the directory once the chunks have
# been combined and split.
CHUNK_DIR = Path(os.environ.get("MCR_CHUNK_DIR", PROC_DIR/"chunks"))


# =============================================================================
# TRAIN / OOS / OOT SPLIT
# =============================================================================

# OOT cutoff: thesis §1.5.1 — last ~3 years of data held out for temporal
# out-of-time (OOT) validation. Loans originated through 2020 have servicer
# performance history through ~2024, so setting the cutoff at mid-2017
# gives ~17 years in-sample and ~7 years OOT.
OOT_CUTOFF = pd.Timestamp("2020-01-01")

# Fraction of the in-sample (pre-OOT_CUTOFF) population held out as OOS.
OOS_FRAC = 0.30


# =============================================================================
# DEFAULT-EVENT DEFINITION
# =============================================================================

# Zero-balance codes treated as default events: 3rd-party sale, short sale,
# repurchase, REO, note sale. 01 = prepayment (explicitly excluded). The
# rare codes 16 (reperforming) and 96 (non-standard disposition) are
# excluded — neither represents a credit loss event. See README's
# "Default definition" note for the regulatory framing.
DEFAULT_CODES = {"02", "03", "06", "09", "15"}

TARGET_PD  = "default_12m"
TARGET_LGD = "lgd"


# =============================================================================
# COMPETING RISKS  (12_competing_risks.py, surv_* dataset variant)
# =============================================================================
# Voluntary prepayment. Chapters 1-5 treat this as censoring: the loan simply
# stops appearing in the panel and the Cox model in 06_survival_analysis.py
# converts its survival function with 1 - S(t). That is only valid if
# prepayment cannot happen — censoring is assumed non-informative and, more
# importantly, "surviving" is assumed to mean "still exposed to default".
# For a mortgage neither holds: roughly 10x more loans leave via prepayment
# than via default, and a prepaid loan can never default afterwards. So
# 1 - S(t) answers "what fraction would default if prepayment were abolished",
# which overstates the cumulative default probability. The competing-risks
# treatment answers the question actually being asked, via the cumulative
# incidence function.
PREPAY_CODES = {"01"}

# 3-class event indicator on the surv_* dataset variant.
EVENT_TYPE_COL = "event_type"
EVENT_CENSORED = 0
EVENT_DEFAULT  = 1
EVENT_PREPAY   = 2
EVENT_TYPE_LABELS = {
    EVENT_CENSORED: "censored",
    EVENT_DEFAULT:  "default",
    EVENT_PREPAY:   "prepay",
}

# Duration column on the surv_* variant: months from origination to the
# terminating event (or to the performance cutoff if censored). Derived from
# report_date - orig_date, NOT from loan_age: the Freddie Mac loan_age field
# RESETS when a loan is modified (Modification Flag Y/P), so using it as the
# duration variable silently rewinds the clock for exactly the distressed
# loans whose timing matters most.
DURATION_COL = "duration_months"

# surv_* dataset variant — emitted ALONGSIDE pd_*.parquet, never replacing
# them: chapters 1-4 and 7-10 read the pd_* files and must keep seeing the
# identical schema and contents.
SURV_TRAIN_FILE = "surv_train.parquet"
SURV_OOS_FILE   = "surv_oos.parquet"
SURV_OOT_FILE   = "surv_oot.parquet"

# 12_competing_risks.py outputs. survival_pd_horizons.csv is deliberately NOT
# among them — chapter 5 stays untouched as the naive baseline to compare
# against.
CR_COMPARISON_FILE      = "competing_risks_comparison.csv"
CR_CIF_CURVES_FILE      = "competing_risks_cif_curves.csv"
CR_COX_COEFS_FILE       = "competing_risks_cox_coefficients.csv"
CR_MULTINOMIAL_COEFS_FILE = "competing_risks_multinomial_coefficients.csv"
CR_EXPECTED_LIFE_FILE   = "competing_risks_expected_life.csv"

# Horizons reported in the comparison table. "lifetime" is the longest
# horizon the fitted curves support and is resolved at runtime.
CR_HORIZONS_MONTHS = [12, 24, 36]

# Discrete-time baseline hazard: months-since-origination is binned rather
# than entered linearly, since a multinomial logit has no baseline hazard of
# its own and the mortgage seasoning ramp is strongly non-linear.
CR_DURATION_BIN_EDGES = [0, 6, 12, 18, 24, 36, 48, 60, 84, 120, 180, 240, 480]

# A duration bin containing too few events of the rarer cause contributes a
# dummy that is collinear with "no default ever happened here", which makes
# the multinomial design singular and kills the clustered-SE fit. Adjacent
# sparse bins are merged until each carries at least this many events of
# every modelled cause. Merging coarsens the baseline hazard only where the
# data cannot support it.
CR_MIN_BIN_EVENTS = 5

# Cap on the discrete-time panel handed to the multinomial fit. The full
# loan-month panel is tens of millions of rows; the multinomial is fitted on
# a loan-level random sample (whole loan histories, never individual rows, so
# each loan's duration stays intact).
CR_MAX_PANEL_LOANS = 200_000

# Optional input: Freddie Mac Primary Mortgage Market Survey weekly average
# 30-year fixed rate, used to build refi_incentive = orig_interest_rate -
# pmms_rate. Absent -> the feature is skipped with a warning, matching the
# HPI/unemployment optional-input pattern in 01_data_preprocessing.py.
PMMS_PATH = MACRO_DIR / "pmms_30yr_fixed.csv"

# LGD workout-period truncation bias (IPCW correction) — thesis §3.3 note.
# Onset trigger: 90+ days past due (delinquency_status in months >= 3), the
# standard regulatory proxy for "entered workout", distinct from the
# terminal zero-balance disposition code (DEFAULT_CODES) already used to
# mark LGD resolution.
LGD_ONSET_DPD_MONTHS = 3
# Floor on the estimated censoring-survival probability Ĝ(t) used to build
# inverse-probability-of-censoring weights, so a handful of very slow,
# thinly-observed resolutions can't produce an extreme weight.
LGD_IPCW_G_FLOOR = 0.05


# =============================================================================
# LGD MODEL SUITE  (04_lgd_models.py)
# =============================================================================
# The LGD training sample is small — on the order of 150 resolved defaults —
# so every constant here exists to keep a model identifiable at that size
# rather than to squeeze out accuracy.

# Categorical levels with fewer than this many TRAINING observations are
# grouped into a single "other" level before one-hot encoding. Without this,
# property_state alone would contribute ~50 indicator columns to a ~150-row
# regression, most of them identifying a single loan — guaranteed separation
# and meaningless coefficients. Unseen levels at scoring time land in the
# same bucket.
LGD_MIN_LEVEL_COUNT = 10
LGD_OTHER_LEVEL = "other"
# Code reserved for levels the tree path never saw in training. Kept distinct
# from every fitted code so a tree can split on "unknown" rather than having
# unseen levels silently masquerade as the first training level.
LGD_UNSEEN_CODE = -1

# Boundary tolerance defining the LGD point masses. The target is a ratio of
# two reported dollar amounts, so an exact 0.0 or 1.0 is the normal encoding
# of "full recovery" / "total loss"; 1e-4 of UPB is well below one dollar on
# any realistic loan, so it separates the point masses from genuine interior
# values without capturing any economically distinct case.
LGD_BOUNDARY_EPS = 1e-4

# Minimum training observations a class needs before it is modelled. Below
# this, the two-stage model drops the class from stage 1 (assigning it
# probability 0) or falls back to a constant interior mean, logging either.
LGD_TWO_STAGE_MIN_CLASS_OBS = 10

# FRM regularisation. The unpenalised quasi-binomial GLM is tried first
# because its coefficients are the auditable output; L2 is a convergence
# fallback, not the default estimator.
LGD_FRM_USE_L2_FALLBACK = True
LGD_FRM_L2_ALPHA = 1.0

# Two-stage model regularisation. C is sklearn's inverse penalty strength for
# the stage-1 multinomial logit; BETA_L2_ALPHA penalises the stage-2 beta
# mean coefficients. Both penalise non-intercept coefficients only —
# penalising the intercept would shift the predicted LGD *level*, and that
# level is exactly what feeds the ECL anchor downstream.
LGD_TWO_STAGE_STAGE1_C = 1.0
LGD_TWO_STAGE_BETA_L2_ALPHA = 0.0


# =============================================================================
# IFRS 9 STAGING  (07_macro_scenario_analysis.py)
# =============================================================================
# Per-loan Stage 1/2/3 classification, IFRS 9 §5.5. A loan moves out of
# Stage 1 (12m ECL) into Stage 2 (lifetime ECL) if EITHER SICR trigger fires:
#   - Relative test: current lifetime PD has deteriorated by >= SICR_PD_RATIO
#     against the PD implied at origination (§5.5.9), gated by an absolute
#     floor so a tiny move on a near-zero starting PD can't trip the ratio
#     test on noise alone.
#   - 30-DPD backstop (delinquency_status >= STAGE2_DPD_MONTHS, i.e. >=30
#     days past due) — the rebuttable presumption in §5.5.11.
# Stage 3 (credit-impaired / in default) is triggered once delinquency
# reaches the same 90-DPD regulatory default trigger already used to onset
# the LGD workout period (LGD_ONSET_DPD_MONTHS) — the loan is exposed to
# lifetime ECL like Stage 2, but is flagged separately since it is already
# in default rather than merely higher-risk.
SICR_PD_RATIO     = 2.0
SICR_PD_ABS_FLOOR = 0.02   # pp minimum absolute PD increase to trigger SICR
STAGE2_DPD_MONTHS = 1      # 30+ days past due backstop
STAGE3_DPD_MONTHS = LGD_ONSET_DPD_MONTHS  # 90+ days past due -> credit-impaired


# =============================================================================
# FEATURE SETS
# =============================================================================

# PD feature set — used identically by 02, 03, 05, 07 and 09.
PD_FEATURES = [
    "delinquency_indicator", "hpi_change", "occupancy_status",
    "orig_interest_rate", "orig_cltv", "num_borrowers", "credit_score",
    "property_type", "loan_age", "orig_dti", "orig_upb", "ur_3m_lag",
]
PD_CAT_FEATURES = ["occupancy_status", "property_type"]

# LGD feature set — used by 04.
LGD_FEATURES = [
    "hpi_change_since_orig", "mi_pct", "orig_cltv", "orig_dti", "orig_upb",
    "orig_interest_rate", "loan_age", "current_interest_rate", "ur_3m_lag",
    "occupancy_status", "first_time_homebuyer", "num_units", "property_type",
    "channel", "loan_purpose", "num_borrowers", "property_state",
]
LGD_CAT_FEATURES = [
    "occupancy_status", "first_time_homebuyer", "num_units",
    "property_type", "channel", "loan_purpose",
    "num_borrowers", "property_state",
]


# =============================================================================
# DISCRETE-TIME SURVIVAL (MONTHLY HAZARD)  (11_discrete_hazard.py)
# =============================================================================
# Ch.5b — fits the monthly default hazard directly on the loan-month panel,
# replacing 06_survival_analysis.py's Cox model (which collapsed the panel to
# one row per loan and read its time-varying covariates off the last row,
# leaking end-of-follow-up state into the fit).
#
# Covariate set. Deliberately NOT config.PD_FEATURES: delinquency_indicator
# and delinquency_status are excluded because a multi-period hazard projection
# has to supply every covariate's future path, and the delinquency path is
# itself an outcome of the default process — projecting it would require a
# second model of delinquency transitions. See the script docstring.
DISCRETE_HAZARD_STATIC_FEATURES = [
    "credit_score", "orig_cltv", "orig_dti", "orig_interest_rate",
    "orig_upb", "num_borrowers", "occupancy_status", "property_type",
]
# Macro covariates, read at the row's own report_date. These are the only
# covariates whose future path the horizon-PD engine has to project, which is
# why PIT/TTC differ only in what is supplied here.
DISCRETE_HAZARD_MACRO_FEATURES = ["ur_3m_lag", "hpi_change"]

DISCRETE_HAZARD_FEATURES = (
    DISCRETE_HAZARD_STATIC_FEATURES + DISCRETE_HAZARD_MACRO_FEATURES
)
# num_borrowers stays numeric here, matching its treatment in PD_FEATURES
# (it is absent from PD_CAT_FEATURES) rather than LGD_CAT_FEATURES.
DISCRETE_HAZARD_CAT_FEATURES = ["occupancy_status", "property_type"]

# Covariates that must never enter the hazard models — pinned as a constant so
# the exclusion is testable rather than merely documented.
DISCRETE_HAZARD_EXCLUDED_FEATURES = ["delinquency_indicator", "delinquency_status"]

# Target construction: a row at age a predicts default in month a+1. A default
# counts as "next period" if it falls within this many days of the row's
# report_date — 45 days spans one monthly reporting period with slack for
# month-length variation, without reaching the month after next.
DISCRETE_HAZARD_NEXT_PERIOD_DAYS = 45

# Baseline hazard alpha(t): spline basis on loan_age. 6 degrees of freedom is
# enough to trace the standard mortgage seasoning ramp (rising to a peak
# around years 3-5, then declining) without chasing month-to-month noise.
DISCRETE_HAZARD_SPLINE_DF = 6

# Case-control subsampling: keep every y=1 row, retain y=0 rows at this rate.
# The monthly hazard is ~0.05-0.1%, so the panel is overwhelmingly y=0 and
# nothing is lost by thinning it. The intercept is corrected afterwards by the
# King & Zeng (2001) prior correction, log(r) on the logit scale.
DISCRETE_HAZARD_SUBSAMPLE_RATE = 0.05

# Rows per chunk when subsampling and when scoring multi-period horizon PDs —
# bounds peak RAM on the 30 GB Kaggle instance.
DISCRETE_HAZARD_CHUNK_SIZE = 2_000_000
DISCRETE_HAZARD_SCORING_CHUNK_SIZE = 50_000

# Cap on rows scored for the unsampled monthly-hazard metrics. Above this the
# split is uniformly sampled — uniformly, so the base rate (and therefore the
# calibration check) is preserved, unlike the case-control training sample.
DISCRETE_HAZARD_MAX_EVAL_ROWS = 5_000_000

DISCRETE_HAZARD_HORIZONS = [12, 24, 36, 60]
# Hard cap on the "lifetime" horizon when remaining_months is available but
# large (a fresh 30-year loan) — keeps the scoring loop bounded.
DISCRETE_HAZARD_LIFETIME_CAP_MONTHS = 360

# Snapshot dates at which the 12-month conditional PD is validated against
# realised 12-month outcomes. Chosen to sit inside the OOT window with a full
# 365-day forward window observable after each.
DISCRETE_HAZARD_SNAPSHOT_DATES = ["2020-06-01", "2021-06-01", "2022-06-01",
                                    "2023-06-01", "2024-06-01"]

# Covariates tested for proportional hazards (covariate x spline(loan_age)
# interaction, likelihood-ratio test against the main model). Only meaningful
# for the linear model — the tree model has no PH assumption to violate.
DISCRETE_HAZARD_PH_TEST_FEATURES = ["credit_score", "orig_cltv", "ur_3m_lag"]

# Monotone constraints for the XGBoost hazard model. Direction is economic,
# not fitted: default risk falls as credit quality rises, and rises with
# leverage, affordability strain and unemployment. Constraining these buys
# auditability (a credit committee can be told the model cannot say "higher
# FICO, higher risk") and more stable extrapolation at the edges of the
# training range. Unlisted covariates are unconstrained.
DISCRETE_HAZARD_USE_MONOTONE_CONSTRAINTS = True
DISCRETE_HAZARD_MONOTONE_DIRECTIONS: dict[str, int] = {
    "credit_score":  -1,
    "orig_cltv":     +1,
    "orig_dti":      +1,
    "ur_3m_lag":     +1,
}

# XGBoost hazard-model hyperparameters. No scale_pos_weight: the class
# imbalance is already handled by the case-control subsampling above, and
# reweighting on top of it would distort the very probabilities the horizon-PD
# product relies on (the prior correction assumes an undistorted sampled-
# population probability).
DISCRETE_HAZARD_XGB_PARAMS: dict = dict(
    n_estimators          = 600,
    max_depth             = 5,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 50,
    gamma                 = 1.0,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    eval_metric           = "logloss",
    tree_method           = "hist",
    early_stopping_rounds = 30,
)

# Which discrete-hazard model 10_basel_irb_capital.py takes its TTC PD from.
# Defaults to the linear model: Basel IRB capital is the most
# supervisory-scrutinised output in the pipeline, and the logit model's
# coefficients and odds ratios are directly auditable in a way the boosted
# model's are not. "xgb" is permitted.
DISCRETE_HAZARD_CAPITAL_MODEL = "logit"


# =============================================================================
# PSI / IV RATING THRESHOLDS
# =============================================================================

def psi_flag(psi: float) -> str:
    """< 0.10 stable | 0.10-0.25 investigate | > 0.25 major shift."""
    import numpy as np
    if np.isnan(psi):  return "N/A"
    if psi < 0.10:      return "Stable"
    if psi < 0.25:      return "Investigate"
    return "Major shift"


def iv_strength(iv: float) -> str:
    if iv < 0.02:  return "Negligible"
    if iv < 0.10:  return "Weak"
    if iv < 0.30:  return "Medium"
    if iv < 0.50:  return "Strong"
    return "Very strong"


# =============================================================================
# GPU DETECTION
# =============================================================================

def detect_gpu() -> tuple[str, int]:
    """
    Probe for NVIDIA GPUs via nvidia-smi.

    Returns (device_str, n_gpus). XGBoost >= 2.0 automatically uses all
    visible GPUs when device="cuda"; callers that don't care about the GPU
    count can ignore the second element.
    """
    log = logging.getLogger("config")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = [g.strip() for g in result.stdout.strip().splitlines() if g.strip()]
            log.info("[GPU] %d GPU(s) found: %s", len(gpus), ", ".join(gpus))
            if len(gpus) > 1:
                import xgboost as xgb_ver
                ver = tuple(int(x) for x in xgb_ver.__version__.split(".")[:2])
                if ver >= (2, 0):
                    log.info("[GPU] XGBoost %s >= 2.0 — all %d GPUs active via NCCL.",
                             xgb_ver.__version__, len(gpus))
                else:
                    log.warning(
                        "[GPU] XGBoost %s < 2.0 — only 1 GPU will be used. "
                        "Upgrade: pip install -U xgboost",
                        xgb_ver.__version__,
                    )
            return "cuda", len(gpus)
    except Exception:
        pass

    log.info("[CPU] No GPU detected — using device='cpu' (hist, memory-efficient).")
    return "cpu", 0


# =============================================================================
# LOGGING
# =============================================================================

def configure_logging(log_filename: str, mode: str = "w") -> None:
    """
    Console + file logging shared by every pipeline script. Each script
    still does `log = logging.getLogger(__name__)` itself afterward, so
    log records are attributed to the calling script as before.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename, mode=mode, encoding="utf-8"),
        ],
    )


# =============================================================================
# PLOT THEME
# =============================================================================

PLT_STYLE: dict = {
    "figure.facecolor":  "#0F1117",
    "axes.facecolor":    "#0F1117",
    "axes.edgecolor":    "#2D3748",
    "axes.labelcolor":   "#E2E8F0",
    "xtick.color":       "#A0AEC0",
    "ytick.color":       "#A0AEC0",
    "text.color":        "#E2E8F0",
    "grid.color":        "#1A2035",
    "legend.facecolor":  "#1A2035",
    "legend.edgecolor":  "#2D3748",
    "font.family":       "monospace",
    "figure.dpi":        130,
}


# =============================================================================
# IFRS 9 MACRO SCENARIOS  (07_macro_scenario_analysis.py)
# =============================================================================
# Users should replace these with their institution's official stress
# scenarios from their Internal Capital Adequacy Assessment Process (ICAAP).

N_QUARTERS           = 20     # full path length: 5 years
DISCOUNT_R           = 0.05   # annual risk-free discount rate for ECL
MACRO_LGD_ASSUMPTION = 0.40   # placeholder — replace with 04_lgd_models.py output

# Expected life vs contractual maturity for lifetime ECL.
#
# amortized_ead() runs each loan's balance down over remaining_months — the
# CONTRACTUAL remaining term. IFRS 9 §5.5.19 measures lifetime ECL over the
# expected life, and for a mortgage the two differ by a lot: a 30-year loan
# with 340 contractual months left has an expected life of roughly 5-8 years
# once voluntary prepayment is accounted for. Amortizing over the contractual
# term holds exposure high for years during which most of the book has in
# fact refinanced away, overstating lifetime ECL.
#
# When True, 07_macro_scenario_analysis.py caps each loan's amortization term
# at its competing-risks expected life (12_competing_risks.py ->
# CR_EXPECTED_LIFE_FILE), falling back to the contractual term with a warning
# if that file has not been produced. False reproduces the original
# contractual-maturity behaviour exactly.
IFRS9_USE_EXPECTED_LIFE = True


def _pad(lst: list, n: int, fill: float = 0.0) -> list:
    """Extend list to length n with fill value."""
    return lst + [fill] * (n - len(lst))


_BASE_UR  = _pad([0.0] * N_QUARTERS, N_QUARTERS, 0.0)
_BASE_HPI = _pad([0.5] * N_QUARTERS, N_QUARTERS, 0.5)

# Adverse UR: rise Q1-Q4, plateau Q5-Q6, recover Q7-Q14, slight residual
_ADV_UR = _pad(
    [0.8, 1.0, 0.8, 0.4,          # rise   : cumulative +3pp by Q4
     0.0, 0.0,                     # plateau: stays at 7.0%
    -0.4,-0.5,-0.5,-0.4,          # recover: -1.8pp over Q7-Q10
    -0.3,-0.2,-0.1,-0.1,          # recover: -0.7pp over Q11-Q14
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0], N_QUARTERS, 0.0)  # flat residual +0.5pp
# Net: 3.0 - 1.8 - 0.7 = +0.5pp above start at Q20

_ADV_HPI = _pad(
    [-2.5,-3.0,-2.5,-1.5,         # fall  : ~-9.5% by Q4
     -0.5, 0.0,                   # trough
      0.3, 0.5, 0.5, 0.5,         # partial recovery
      0.4, 0.4, 0.3, 0.3,
      0.3, 0.2, 0.2, 0.2, 0.2, 0.2], N_QUARTERS, 0.2)

# Severe UR: rise Q1-Q6, plateau Q7-Q8, recover Q9-Q20
_SEV_UR = _pad(
    [1.2, 1.5, 1.5, 1.2, 0.6, 0.0,  # rise   : +6pp by Q6
     0.0, 0.0,                        # plateau: stays at 10%
    -0.5,-0.6,-0.7,-0.6,              # recover: -2.4pp Q9-Q12
    -0.5,-0.4,-0.3,-0.3,              # recover: -1.5pp Q13-Q16
    -0.2,-0.2,-0.1,-0.2], N_QUARTERS, 0.0)
# Net: 6.0 - 2.4 - 1.5 - 0.7 = +1.4pp scarring at Q20

_SEV_HPI = _pad(
    [-5.0,-6.0,-6.0,-5.0,-3.0,-1.0,  # fall ~-26%
     -0.5, 0.0,
      0.2, 0.3, 0.4, 0.4,
      0.4, 0.4, 0.3, 0.3,
      0.3, 0.2, 0.2, 0.2], N_QUARTERS, 0.2)

SCENARIOS: dict[str, dict] = {
    "Base": {
        "label":      "Base Scenario",
        "color":      "#10B981",
        "weight":     0.60,
        "ur_delta":   _BASE_UR,
        "hpi_delta":  _BASE_HPI,
    },
    "Adverse": {
        "label":      "Adverse Scenario",
        "color":      "#F59E0B",
        "weight":     0.30,
        "ur_delta":   _ADV_UR,
        "hpi_delta":  _ADV_HPI,
    },
    "Severe": {
        "label":      "Severe Scenario (GFC-level)",
        "color":      "#EF4444",
        "weight":     0.10,
        "ur_delta":   _SEV_UR,
        "hpi_delta":  _SEV_HPI,
    },
}


# =============================================================================
# RATING MASTER SCALE & BASEL IRB CAPITAL  (10_basel_irb_capital.py)
# =============================================================================
# Continuous-PD -> letter-grade master scale, S&P/Moody's-style long-run PD
# upper bounds (a loan's TTC PD maps to the first band whose bound it is
# under). Institutions calibrate their own master scale to their portfolio's
# realised default experience; these bounds are a standard illustrative
# scale, not a fitted one — replace with your institution's calibrated scale.
RATING_SCALE: list[tuple[str, float]] = [
    ("AAA", 0.0002),
    ("AA",  0.0005),
    ("A",   0.0010),
    ("BBB", 0.0030),
    ("BB",  0.0100),
    ("B",   0.0300),
    ("CCC", 0.1000),
    ("D",   1.0001),   # catch-all upper bound so PD == 1.0 still maps to D
]

# Basel II/III IRB retail residential mortgage exposure class (Basel
# Framework CRE31/CRE32): fixed asset correlation R=0.15 and no maturity
# adjustment b(PD) — that term only applies to corporate/sovereign/bank
# exposures under the advanced IRB approach, not retail.
BASEL_RETAIL_MORTGAGE_CORRELATION = 0.15
BASEL_CONFIDENCE        = 0.999   # 99.9% supervisory confidence level
BASEL_MIN_CAPITAL_RATIO = 0.08    # 8% Pillar 1 minimum capital ratio


def pd_to_rating(pd_value: float) -> str:
    """Map a continuous PD to its RATING_SCALE letter grade."""
    for grade, upper_bound in RATING_SCALE:
        if pd_value < upper_bound:
            return grade
    return RATING_SCALE[-1][0]
