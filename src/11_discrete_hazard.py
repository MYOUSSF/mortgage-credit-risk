"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.5b — Discrete-Time Survival (Hazard)
=============================================================================
Script  : 11_discrete_hazard.py
Purpose : Model the monthly default hazard directly on the loan-month panel,
          producing conditional PD at any horizon (PIT and TTC), with two
          implementations of one shared framework:

            (A) discrete-time logistic regression  — linear in the logit
            (B) XGBoost                            — same discrete-time target

Why this replaces the Cox model in 06_survival_analysis.py
-----------------------------------------------------------
  06_survival_analysis.py has two defects that this script exists to fix.

  1. Covariate leakage. build_survival_df() collapses the loan-month panel
     to ONE row per loan — the row with the maximum loan_age — and reads the
     time-varying covariates (delinquency_indicator, hpi_change, ur_3m_lag)
     off that last row. For a loan that defaults, the last retained row is
     the month immediately before default, so the model is handed the
     borrower's state *at the brink of default* and asked to predict
     default. The reported C-index (~0.85-0.89) is inflated by construction
     and is not a forecast performance estimate. The delinquency_indicator
     covariate is the worst offender: on the last pre-default row it is
     essentially a default flag.

  2. Horizon PDs are unconditional. compute_horizon_pds() returns
     1 - S(h | x), the probability of defaulting within h months *measured
     from origination*, and assigns it to a loan that has already survived
     to age a. The quantity actually wanted is the conditional PD

         PD(a -> a+h) = 1 - S(a+h | x) / S(a | x)

     For a seasoned loan the unconditional figure overstates PD, because it
     charges the loan again for the default risk of the years it has
     already survived.

  This script fits the hazard on the panel as it stands — every loan-month
  row is one observation, each row's covariates are that row's own
  contemporaneous values, and nothing from after report_date enters the
  design matrix. The horizon PD is conditional on the loan's current age by
  construction, because it is built as a product of one-period survival
  probabilities starting at the loan's current age.

Discrete-time survival
----------------------
  Let h(t | x) = P(default in month t | survived to month t, x). Each
  loan-month row at age a is a Bernoulli trial for "does this loan default
  in month a+1". Fitting a binary model to that target on the panel IS
  maximum likelihood for the discrete-time survival model: the panel
  likelihood factorises into one Bernoulli term per loan-month at risk
  (Allison 1982; Singer & Willett 1993). This is why no data expansion is
  needed — the panel is already in counting-process form.

  The baseline hazard alpha(t) is the loan_age effect:

      logit h(t | x) = alpha(t) + x'beta          (model A, spline alpha)
      logit h(t | x) = f(t, x)                    (model B, trees)

  Model A takes alpha(t) as a spline in loan_age, which is the discrete-time
  analogue of Cox's non-parametric h_0(t) — it traces the mortgage seasoning
  ramp without imposing a parametric shape. Model B lets the trees learn the
  age effect and its interactions with everything else, so it carries no
  proportional-hazards assumption at all.

  Logit link, not cloglog: the monthly hazard here is on the order of 0.05%,
  and at that magnitude logit and cloglog are numerically indistinguishable
  (they differ in the third decimal of the linear predictor only as the
  hazard approaches 1). Logit is preferred because the case-control prior
  correction below is EXACT under logit and only approximate under cloglog.

Case-control subsampling and the prior correction
--------------------------------------------------
  The panel is ~99.95% non-events. Every y=1 row is kept; y=0 rows are kept
  with probability r (config.DISCRETE_HAZARD_SUBSAMPLE_RATE). Under this
  design the sampled-population odds are the true odds divided by r, so

      logit(h_true) = logit(h_sampled) + log(r)

  which is the King & Zeng (2001) prior correction. It is a pure intercept
  shift: it changes the level of the hazard, never the ranking, and it is
  applied identically to both models (for XGBoost, to the raw output margin
  before the sigmoid). Because the correction is applied inside
  predict_hazard(), every downstream consumer — horizon PDs, calibration,
  the Basel TTC PD — sees already-corrected probabilities and never has to
  know that subsampling happened.

Why internal (delinquency) covariates are excluded from BOTH models
--------------------------------------------------------------------
  delinquency_indicator and delinquency_status are excluded deliberately,
  and their exclusion is pinned in config.DISCRETE_HAZARD_EXCLUDED_FEATURES
  and asserted in the tests.

  A multi-period horizon PD is a product over future months, so every
  covariate in the model needs a projected value at every future month. For
  loan_age the projection is deterministic. For the macro covariates it is
  an explicit scenario assumption (below). But delinquency at month a+j is
  itself an outcome of the same credit-deterioration process that produces
  default — a loan is 90 days past due *because* it is heading towards
  default. Projecting it would require a second model of delinquency
  transitions, and holding it flat at today's value (the treatment the macro
  covariates get) is indefensible: it would assert that a currently-current
  loan stays current for 60 months, which mechanically drives its lifetime
  PD towards zero, and that a currently-delinquent loan stays exactly as
  delinquent for 60 months without either curing or defaulting.

  Including it would also reproduce the Cox model's leakage in a subtler
  form: delinquency_status on the row immediately before default is close to
  a deterministic function of the target. The one-month-ahead AUROC would
  look excellent and the multi-year PD would be worthless.

  The cost of this choice is real and is disclosed: the models cannot
  distinguish a current loan from a 60-days-past-due loan with identical
  origination characteristics and macro state. For IFRS 9 that gap is
  covered elsewhere — 07_macro_scenario_analysis.py's staging applies the
  30-DPD and 90-DPD backstops to exactly that information
  (config.STAGE2_DPD_MONTHS / STAGE3_DPD_MONTHS), so delinquency drives
  stage allocation rather than the PD level.

Point-in-time vs through-the-cycle
-----------------------------------
  Both are produced from the SAME fitted model, differing only in the macro
  path handed to compute_conditional_horizon_pd():

    PIT — each loan's current ur_3m_lag / hpi_change held flat across the
          whole projection horizon. This is a documented assumption, not a
          forecast: it says "conditions stay as they are today". It is the
          right default for IFRS 9 (which wants PD to move with the cycle)
          and it is honest about the fact that this script does not contain
          a macro forecasting model.
    TTC — macro covariates set to their long-run training-sample mean, for
          Basel IRB capital (EBA/GL/2017/16 §6.2), which wants a PD that
          does not move with the current point in the cycle.

Note for a future Ch.6 integration (not done here)
---------------------------------------------------
  compute_conditional_horizon_pd() accepts an explicit per-period macro path
  (scalar, length-h vector, or n_loans x h array), which is precisely the
  interface 07_macro_scenario_analysis.py's quarterly ECL engine needs. That
  script currently re-trains the Ch.2 12-month classifier and re-scores it
  once per quarter with shocked macro inputs (score_quarter()), which gives
  a sequence of 12-month PDs rather than a set of marginal per-quarter
  default probabilities — so its ECL accumulation has to reconcile
  overlapping 12-month windows. Passing config.SCENARIOS' ur_delta /
  hpi_delta paths into this function instead would yield the marginal
  monthly hazards directly, and the quarterly marginal PD would be the
  product of three consecutive months' survival. That replacement is
  deliberately NOT made in this commit: Ch.6 is left untouched.

Inputs
------
  data/processed/pd_train.parquet      (requires default_date — re-run 01)
  data/processed/pd_oos.parquet
  data/processed/pd_oot.parquet
  data/processed/pd_xgb_results.csv    (optional — Ch.2 benchmark)

Outputs
-------
  data/processed/discrete_hazard_logit_pd_horizons.csv
  data/processed/discrete_hazard_xgb_pd_horizons.csv
  data/processed/discrete_hazard_logit_metrics.csv
  data/processed/discrete_hazard_xgb_metrics.csv
  data/processed/discrete_hazard_comparison.csv
  data/processed/discrete_hazard_logit_coefficients.csv
  data/processed/discrete_hazard_logit_ph_tests.csv
  data/figures/discrete_hazard_{logit,xgb}_seasoning.png
  data/figures/discrete_hazard_{logit,xgb}_calibration.png
  data/figures/discrete_hazard_{logit,xgb}_pit_vs_ttc.png
  data/figures/discrete_hazard_{logit,xgb}_km_check.png
  data/figures/discrete_hazard_logit_odds_ratios.png
  data/figures/discrete_hazard_xgb_shap_importance.png
  data/figures/discrete_hazard_xgb_shap_dependence_loan_age.png
  data/figures/discrete_hazard_comparison.png

Prerequisites
-------------
  Run 01_data_preprocessing.py first — with default_date persisted to the PD
  parquet files (it is in main()'s base_cols). A parquet built before that
  change will not have the column and this script will abort with a clear
  message rather than silently building an empty target.
=============================================================================
"""

from __future__ import annotations

import gc
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.special import expit, logit as _logit_fn
from scipy.stats import chi2
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in [REPO_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import config

# =============================================================================
# LOGGING
# =============================================================================

config.configure_logging("discrete_hazard.log")
log = logging.getLogger(__name__)

DEVICE, N_GPUS = config.detect_gpu()


# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = config.PROC_DIR
FIG_DIR  = config.FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = config.SEED

# The discrete-time target built by build_discrete_target(): "does this loan
# default in the next reporting period", as distinct from config.TARGET_PD
# ("does this loan default within the next 365 days"), which stays in the
# parquet and is used as the realised outcome for horizon validation.
TARGET_HAZARD = "hazard_event"
TARGET_12M    = config.TARGET_PD

FEATURES          = config.DISCRETE_HAZARD_FEATURES
STATIC_FEATURES   = config.DISCRETE_HAZARD_STATIC_FEATURES
MACRO_FEATURES    = config.DISCRETE_HAZARD_MACRO_FEATURES
CAT_FEATURES      = config.DISCRETE_HAZARD_CAT_FEATURES
EXCLUDED_FEATURES = config.DISCRETE_HAZARD_EXCLUDED_FEATURES

TIME_COL = "loan_age"
MODEL_COLS = [TIME_COL] + FEATURES

NEXT_PERIOD_DAYS = config.DISCRETE_HAZARD_NEXT_PERIOD_DAYS
SPLINE_DF        = config.DISCRETE_HAZARD_SPLINE_DF
SUBSAMPLE_RATE   = config.DISCRETE_HAZARD_SUBSAMPLE_RATE
CHUNK_SIZE       = config.DISCRETE_HAZARD_CHUNK_SIZE
SCORING_CHUNK    = config.DISCRETE_HAZARD_SCORING_CHUNK_SIZE
HORIZONS         = config.DISCRETE_HAZARD_HORIZONS
LIFETIME_CAP     = config.DISCRETE_HAZARD_LIFETIME_CAP_MONTHS
SNAPSHOT_DATES   = config.DISCRETE_HAZARD_SNAPSHOT_DATES
PH_TEST_FEATURES = config.DISCRETE_HAZARD_PH_TEST_FEATURES
MAX_EVAL_ROWS    = config.DISCRETE_HAZARD_MAX_EVAL_ROWS

PLT_STYLE: dict = config.PLT_STYLE
PALETTE = ["#38BDF8", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6",
           "#EC4899", "#14B8A6", "#F97316"]

MODEL_COLORS = {"logit": "#38BDF8", "xgb": "#F59E0B", "ch2_xgb_12m": "#8B5CF6"}

# Columns read from the PD parquet files — selective loading keeps peak RAM
# bounded on the 30 GB Kaggle instance (the panel is ~24M rows).
_ID_COLS   = ["loan_seq_num", "report_date", "default_date"]
_AUX_COLS  = ["remaining_months", TARGET_12M]
_LOAD_COLS = _ID_COLS + _AUX_COLS + MODEL_COLS


# =============================================================================
# TARGET CONSTRUCTION
# =============================================================================

def build_discrete_target(df: pd.DataFrame,
                          next_period_days: int = NEXT_PERIOD_DAYS,
                          target_col: str = TARGET_HAZARD) -> pd.DataFrame:
    """
    Build the discrete-time survival target on the loan-month panel.

    Each existing row is one observation — no data expansion. A row observed
    at loan age `a` is a Bernoulli trial for "does this loan default in month
    a+1", so:

        y = 1  on the last retained row of a loan whose default_date is known
               and falls within the next reporting period, i.e.
               0 < (default_date - report_date).days <= next_period_days
        y = 0  on every other row, including
                 - rows of a defaulting loan whose default is further away
                   than one period (they were at risk and did not default)
                 - every row of a loan that never defaults (right-censored)

    The 45-day default window (config.DISCRETE_HAZARD_NEXT_PERIOD_DAYS) spans
    one monthly reporting period with slack for month-length variation
    (28-31 days) without ever reaching the month after next.

    Why default_date and not default_12m: default_12m marks every row within
    365 days of the event, which is ~12 rows per defaulter — that is the
    Ch.1/Ch.2 12-month classification target, not a hazard target. The hazard
    target needs the single row at which the event occurs.

    Why this needs default_date persisted in the parquet (01's base_cols):
    split_pd() cuts the panel on report_date, so a defaulting loan's rows can
    straddle pd_train and pd_oot. Its last row *within pd_train* is then not
    the row before default, and taking a per-file "last row per loan" would
    mark an event on a row where no event occurred. Keying on the event date
    itself is immune to how the panel is split.

    At most one y=1 per loan is guaranteed by construction: among rows
    satisfying the window condition (normally exactly one, given monthly
    reporting), only the latest is marked. The guarantee holds across files
    too, since the condition is a property of the row's own date, and the
    winning row lives in exactly one split.

    Returns a copy with a fresh RangeIndex and the target column appended.
    """
    out = df.reset_index(drop=True).copy()

    if "default_date" not in out.columns:
        raise KeyError(
            "build_discrete_target requires a 'default_date' column. The PD "
            "parquet files carry it only if 01_data_preprocessing.py was run "
            "after default_date was added to main()'s base_cols — re-run 01."
        )

    default_date = pd.to_datetime(out["default_date"])
    report_date  = pd.to_datetime(out["report_date"])
    days_to_default = (default_date - report_date).dt.days

    candidate = (
        days_to_default.notna()
        & (days_to_default > 0)
        & (days_to_default <= next_period_days)
    )

    y = pd.Series(np.zeros(len(out), dtype=np.int8), index=out.index)
    if candidate.any():
        # Among candidate rows, keep only the latest per loan. With monthly
        # reporting there is exactly one candidate per defaulter; this guard
        # makes the "at most one event per loan" invariant hold even if the
        # reporting cadence is irregular for some servicer.
        last_rows = out.loc[candidate].groupby("loan_seq_num")["report_date"].idxmax()
        y.loc[last_rows.values] = np.int8(1)

    out[target_col] = y.astype(np.int8)
    return out


def drop_post_default_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any loan-month row at or after its loan's default_date.

    extract_pd_rows() in 01_data_preprocessing.py already does this, so on a
    correctly built panel this is a no-op. It is applied again here because a
    single post-default row is both a leakage vector (the covariates describe
    a loan that has already defaulted) and a silent corrupter of the
    at-risk denominator in the observed Kaplan-Meier check — and it is
    cheaper to enforce the invariant than to assume it.
    """
    if "default_date" not in df.columns:
        return df
    default_date = pd.to_datetime(df["default_date"])
    report_date  = pd.to_datetime(df["report_date"])
    keep = default_date.isna() | (report_date < default_date)
    n_dropped = int((~keep).sum())
    if n_dropped:
        log.warning("  Dropped %s post-default row(s) — 01's leakage guard "
                    "should already have removed these.", f"{n_dropped:,}")
    return df.loc[keep].copy()


def loans_with_multiple_events(df: pd.DataFrame,
                               target_col: str = TARGET_HAZARD) -> pd.Series:
    """
    Loans carrying more than one hazard event. Empty Series = invariant holds.

    A discrete-time survival model admits at most one terminating event per
    subject; two events for one loan would mean the loan "defaulted twice",
    double-counting it in the likelihood and inflating the baseline hazard.
    """
    counts = df.groupby("loan_seq_num")[target_col].sum()
    return counts[counts > 1]


# =============================================================================
# CASE-CONTROL SUBSAMPLING  +  KING & ZENG (2001) PRIOR CORRECTION
# =============================================================================

def subsample_case_control(df: pd.DataFrame,
                           rate: float = SUBSAMPLE_RATE,
                           target_col: str = TARGET_HAZARD,
                           seed: int = SEED,
                           chunk_size: int = CHUNK_SIZE) -> tuple[pd.DataFrame, float]:
    """
    Keep every event row; keep each non-event row with probability `rate`.

    Processed in chunks so the boolean mask and the retained slices are built
    incrementally rather than materialising a full-panel random vector
    alongside the panel itself.

    Returns (sampled_frame, realised_rate). The realised rate — actual
    non-events kept / non-events available — is what the prior correction
    should use, not the nominal rate: they agree to several decimals on a
    panel this size, but using the realised figure keeps the correction exact
    on small samples too (which is what makes it testable).
    """
    if not 0 < rate <= 1:
        raise ValueError(f"subsample rate must be in (0, 1], got {rate}")

    if rate == 1.0:
        return df.copy(), 1.0

    rng = np.random.default_rng(seed)
    kept_parts: list[pd.DataFrame] = []
    n_controls_total = 0
    n_controls_kept  = 0

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        is_event = chunk[target_col].to_numpy().astype(bool)
        n_controls_total += int((~is_event).sum())

        draw = rng.random(len(chunk))
        keep = is_event | (draw < rate)
        n_controls_kept += int((keep & ~is_event).sum())

        if keep.any():
            kept_parts.append(chunk.loc[keep])

    sampled = (pd.concat(kept_parts, ignore_index=True) if kept_parts
               else df.iloc[:0].copy())
    realised = (n_controls_kept / n_controls_total) if n_controls_total else rate

    log.info(
        "  Case-control subsample: %s rows -> %s  (all %s events kept, "
        "%s of %s non-events at realised rate %.6f, nominal %.6f)",
        f"{len(df):,}", f"{len(sampled):,}",
        f"{int(df[target_col].sum()):,}",
        f"{n_controls_kept:,}", f"{n_controls_total:,}", realised, rate,
    )
    return sampled, realised


def king_zeng_correct_logit(logit_sampled: np.ndarray, rate: float) -> np.ndarray:
    """
    King & Zeng (2001) prior correction on the logit scale.

    Under a case-control design that keeps every event and each non-event
    with probability r, the sampled-population conditional odds are the true
    odds divided by r:

        odds_sampled(x) = P(y=1|x) / [ (1 - P(y=1|x)) * r ]

    so the correction is a constant intercept shift

        logit(h_true) = logit(h_sampled) + log(r)

    Exact under a logit link, which is why the linear model uses one. The
    shift is monotone, so it changes the LEVEL of the hazard (and therefore
    every horizon PD, ECL and capital number downstream) while leaving the
    ranking — and hence AUROC, KS and Gini — untouched.

    It is model-agnostic: any estimator of the sampled-population conditional
    probability can be corrected this way, which is why the same function
    serves both the logistic regression and the XGBoost margin.
    """
    if not 0 < rate <= 1:
        raise ValueError(f"sampling rate must be in (0, 1], got {rate}")
    return np.asarray(logit_sampled, dtype=np.float64) + np.log(rate)


def apply_prior_correction(p_sampled: np.ndarray, rate: float) -> np.ndarray:
    """Probability-scale wrapper around king_zeng_correct_logit()."""
    p = np.clip(np.asarray(p_sampled, dtype=np.float64), 1e-12, 1 - 1e-12)
    return expit(king_zeng_correct_logit(_logit_fn(p), rate))


# =============================================================================
# SHARED MODEL INTERFACE
# =============================================================================

class HazardModel:
    """
    Interface every discrete-time hazard model in this script implements.

        fit(X_train, y_train, X_valid=None, y_valid=None) -> self
        predict_hazard(X) -> np.ndarray of monthly hazards in [0, 1]

    X is always a RAW DataFrame holding TIME_COL plus config's feature
    columns, with categoricals as their original string values. Each model
    owns its encoding (spline + one-hot for the linear model, ordinal codes
    for the trees), so the shared code never has to know how a model
    represents its inputs.

    predict_hazard() returns ALREADY prior-corrected probabilities — the
    King & Zeng shift is applied inside, using the sampling rate the model
    was constructed with. Everything downstream (conditional horizon PDs,
    PIT/TTC, validation, plots, the Basel TTC PD) goes through this method
    only, so adding a third model means implementing these two methods and
    nothing else.
    """

    name = "base"

    def __init__(self, sampling_rate: float = 1.0) -> None:
        self.sampling_rate = sampling_rate
        self.feature_names_: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_valid: pd.DataFrame | None = None,
            y_valid: np.ndarray | None = None) -> "HazardModel":
        raise NotImplementedError

    def predict_hazard(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# =============================================================================
# DESIGN MATRIX FOR THE LINEAR MODEL
# =============================================================================

class HazardDesign:
    """
    Fit-on-train / apply-anywhere design matrix for the logit hazard model.

    Assembles, in this order:
      1. an intercept column
      2. a B-spline basis on loan_age — the baseline hazard alpha(t). Knots
         are placed on training-set loan_age quantiles; extrapolation is
         "constant", so projecting a loan past the oldest age ever observed
         holds the baseline flat instead of letting an unconstrained cubic
         diverge. That matters here because the lifetime PD projects up to
         config.DISCRETE_HAZARD_LIFETIME_CAP_MONTHS months ahead.
      3. continuous covariates, median-imputed and standardised using
         TRAINING statistics only
      4. one-hot indicators for each categorical, dropping the
         alphabetically-first observed level as the reference

    Standardisation is internal only: it is there so IRLS converges cleanly
    when orig_upb (~1e5) sits in the same matrix as orig_dti (~30). The
    coefficients are reported on the ORIGINAL scale — for a covariate
    standardised by (x - m) / s, beta_original = beta_standardised / s, and
    the standard error and confidence bounds scale by the same factor, so
    the back-transform is exact rather than approximate.
    """

    def __init__(self, spline_df: int = SPLINE_DF,
                 cat_features: list[str] | None = None,
                 time_col: str = TIME_COL) -> None:
        self.spline_df = spline_df
        self.cat_features = list(cat_features if cat_features is not None
                                 else CAT_FEATURES)
        self.time_col = time_col

        self.continuous_: list[str] = []
        self.medians_: dict[str, float] = {}
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}
        self.categories_: dict[str, list[str]] = {}
        self.reference_levels_: dict[str, str] = {}
        self.spline_: SplineTransformer | None = None
        self.spline_cols_: list[str] = []
        self.columns_: list[str] = []

    # -- fitting ------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "HazardDesign":
        present = [c for c in X.columns if c in FEATURES or c == self.time_col]
        self.continuous_ = [c for c in present
                            if c != self.time_col and c not in self.cat_features]

        for col in self.continuous_:
            values = pd.to_numeric(X[col], errors="coerce")
            self.medians_[col] = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(self.medians_[col])
            self.means_[col] = float(filled.mean())
            std = float(filled.std(ddof=0))
            # A constant covariate would divide by zero; leave it unscaled and
            # let the fit drop it as collinear with the intercept.
            self.stds_[col] = std if std > 1e-12 else 1.0

        for col in self.cat_features:
            if col not in X.columns:
                continue
            levels = sorted(X[col].fillna("missing").astype(str).unique().tolist())
            self.categories_[col] = levels
            self.reference_levels_[col] = levels[0] if levels else "missing"

        age = pd.to_numeric(X[self.time_col], errors="coerce").fillna(0.0)
        # n_knots + degree - 1 basis columns for degree=3, so n_knots chosen
        # to land exactly on spline_df columns.
        degree = 3
        # SplineTransformer emits n_knots + degree - 2 columns when
        # include_bias=False, so invert that to land on exactly spline_df
        # basis functions (the intercept is supplied separately).
        n_knots = max(self.spline_df - degree + 2, 2)
        self.spline_ = SplineTransformer(
            n_knots=n_knots, degree=degree, include_bias=False,
            extrapolation="constant", knots="quantile",
        )
        self.spline_.fit(age.to_numpy().reshape(-1, 1))
        n_basis = self.spline_.transform(np.array([[0.0]])).shape[1]
        self.spline_cols_ = [f"age_spline_{i}" for i in range(n_basis)]

        self.columns_ = (
            ["const"]
            + self.spline_cols_
            + list(self.continuous_)
            + [f"{col}[{lvl}]" for col in self.categories_
               for lvl in self.categories_[col][1:]]
        )
        return self

    # -- application --------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        parts: dict[str, np.ndarray] = {"const": np.ones(n, dtype=np.float64)}

        age = pd.to_numeric(X[self.time_col], errors="coerce").fillna(0.0)
        basis = self.spline_.transform(age.to_numpy().reshape(-1, 1))
        for i, col in enumerate(self.spline_cols_):
            parts[col] = basis[:, i]

        for col in self.continuous_:
            values = pd.to_numeric(X[col], errors="coerce").fillna(self.medians_[col])
            parts[col] = (values.to_numpy() - self.means_[col]) / self.stds_[col]

        for col, levels in self.categories_.items():
            raw = X[col].fillna("missing").astype(str).to_numpy()
            for lvl in levels[1:]:  # levels[0] is the reference
                parts[f"{col}[{lvl}]"] = (raw == lvl).astype(np.float64)

        return pd.DataFrame(parts, columns=self.columns_, index=X.index)

    def interaction_columns(self, X: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        feature x spline(loan_age) interaction block, for the proportional
        hazards test. A non-proportional effect is exactly one whose
        coefficient varies with time, so letting the covariate interact with
        the baseline's own basis is the natural alternative hypothesis.
        """
        design = self.transform(X)
        if feature in self.continuous_:
            base = design[feature].to_numpy()
        elif feature in design.columns:
            base = design[feature].to_numpy()
        else:
            raise KeyError(f"{feature} is not part of this design")
        block = {f"{feature}_x_{sc}": base * design[sc].to_numpy()
                 for sc in self.spline_cols_}
        return pd.DataFrame(block, index=X.index)


# =============================================================================
# MODEL A — DISCRETE-TIME LOGISTIC REGRESSION
# =============================================================================

class LogitHazardModel(HazardModel):
    """
    Discrete-time logistic regression: logit h(t|x) = alpha(t) + x'beta.

    statsmodels GLM (Binomial family, logit link), unpenalised — the
    likelihood-ratio proportional-hazards test below is only valid against an
    unpenalised fit, and with millions of rows against ~20 parameters there
    is nothing for a penalty to stabilise.
    """

    name = "logit"

    def __init__(self, sampling_rate: float = 1.0,
                 spline_df: int = SPLINE_DF) -> None:
        super().__init__(sampling_rate=sampling_rate)
        self.spline_df = spline_df
        self.design = HazardDesign(spline_df=spline_df)
        self.result_ = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_valid: pd.DataFrame | None = None,
            y_valid: np.ndarray | None = None) -> "LogitHazardModel":
        import statsmodels.api as sm

        self.design.fit(X_train)
        design = self.design.transform(X_train)
        self.feature_names_ = list(design.columns)

        model = sm.GLM(np.asarray(y_train, dtype=np.float64), design,
                       family=sm.families.Binomial(sm.families.links.Logit()))
        self.result_ = model.fit()
        log.info("  [logit] converged in %s IRLS iterations | llf=%.2f | "
                 "%d parameters on %s rows",
                 getattr(self.result_, "fit_history", {}).get("iteration", "?"),
                 self.result_.llf, len(self.feature_names_), f"{len(design):,}")
        return self

    def predict_hazard(self, X: pd.DataFrame) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("LogitHazardModel.predict_hazard before fit()")
        design = self.design.transform(X)
        linear = design.to_numpy() @ self.result_.params.to_numpy()
        return expit(king_zeng_correct_logit(linear, self.sampling_rate))

    # -- reporting ----------------------------------------------------------

    def coefficient_table(self) -> pd.DataFrame:
        """
        Coefficients and odds ratios with 95% confidence intervals, reported
        on the ORIGINAL covariate scale (the internal standardisation is
        undone exactly — see HazardDesign).

        The intercept is reported as fitted on the sampled population, with
        the prior-corrected intercept alongside: the corrected figure is the
        one consistent with the hazards predict_hazard() returns.
        """
        res = self.result_
        params = res.params
        bse    = res.bse
        conf   = res.conf_int()
        conf.columns = ["lo", "hi"]

        rows = []
        for name in params.index:
            scale = self.design.stds_.get(name, 1.0)
            kind = ("baseline spline" if name in self.design.spline_cols_
                    else "intercept" if name == "const"
                    else "indicator" if name not in self.design.continuous_
                    else "continuous")
            coef = params[name] / scale
            se   = bse[name] / scale
            lo   = conf.loc[name, "lo"] / scale
            hi   = conf.loc[name, "hi"] / scale
            rows.append({
                "feature": name,
                "kind": kind,
                "coef": coef,
                "std_err": se,
                "z": res.tvalues[name],
                "p_value": res.pvalues[name],
                "odds_ratio": np.exp(coef),
                "or_lower_95": np.exp(lo),
                "or_upper_95": np.exp(hi),
                "reference_level": next(
                    (f"{c}={self.design.reference_levels_[c]}"
                     for c in self.design.reference_levels_
                     if name.startswith(f"{c}[")), ""),
            })

        table = pd.DataFrame(rows)
        corrected_intercept = (params.get("const", np.nan)
                               + np.log(self.sampling_rate))
        table.loc[table["feature"] == "const", "prior_corrected_coef"] = corrected_intercept
        return table

    def baseline_hazard_curve(self, ages: np.ndarray,
                              reference_row: pd.DataFrame) -> np.ndarray:
        """
        alpha(t): the fitted hazard over loan_age with every other covariate
        pinned at the reference row (median continuous values, reference
        categorical levels). This is the seasoning curve.
        """
        frame = pd.concat([reference_row] * len(ages), ignore_index=True)
        frame[TIME_COL] = ages
        return self.predict_hazard(frame)


def proportional_hazards_lr_test(model: LogitHazardModel,
                                 X: pd.DataFrame,
                                 y: np.ndarray,
                                 features: list[str] = PH_TEST_FEATURES) -> pd.DataFrame:
    """
    Likelihood-ratio test of the proportional-hazards assumption for each
    named covariate.

    The main model constrains every covariate's effect to be constant over
    loan age — the discrete-time counterpart of Cox's proportional hazards
    assumption, and the assumption 06_survival_analysis.py checks with
    Schoenfeld residuals. The alternative here adds a
    covariate x spline(loan_age) interaction block, so the covariate's
    coefficient is free to vary with age:

        LR = 2 * (llf_interaction - llf_main)  ~  chi2(spline_df)

    A small p-value means that covariate's effect is NOT constant over the
    life of the loan — which for the linear model is a real limitation to
    disclose (its long-horizon PD leans on the constancy), and which is
    precisely the assumption the XGBoost challenger does not make.
    """
    import statsmodels.api as sm

    base_design = model.design.transform(X)
    llf_main = model.result_.llf
    y_arr = np.asarray(y, dtype=np.float64)

    rows = []
    for feature in features:
        if feature not in model.design.continuous_:
            log.warning("  [PH test] %s not among the model's continuous "
                        "covariates — skipped.", feature)
            continue
        inter = model.design.interaction_columns(X, feature)
        augmented = pd.concat([base_design, inter], axis=1)
        aug_fit = sm.GLM(
            y_arr, augmented,
            family=sm.families.Binomial(sm.families.links.Logit()),
        ).fit()

        lr_stat = 2.0 * (aug_fit.llf - llf_main)
        df_diff = inter.shape[1]
        p_value = float(chi2.sf(max(lr_stat, 0.0), df_diff))
        rows.append({
            "feature": feature,
            "lr_statistic": lr_stat,
            "df": df_diff,
            "p_value": p_value,
            "ph_assumption": "rejected" if p_value < 0.05 else "not rejected",
        })
        log.info("  [PH test] %-20s LR=%10.2f  df=%d  p=%.4g  -> %s",
                 feature, lr_stat, df_diff, p_value, rows[-1]["ph_assumption"])

    return pd.DataFrame(rows)


# =============================================================================
# MODEL B — XGBOOST DISCRETE-TIME HAZARD
# =============================================================================

class XGBHazardModel(HazardModel):
    """
    XGBoost on the same discrete-time target and the same raw covariates,
    with loan_age as an ordinary feature.

    The trees learn the baseline hazard AND its interactions with every other
    covariate, so unlike the linear model this carries no proportional
    hazards assumption: the effect of credit_score is free to differ at month
    6 and month 60 without anything being specified in advance.

    No scale_pos_weight. The class imbalance is already handled by the
    case-control subsampling, and reweighting on top of it would distort the
    predicted probabilities — which matters more here than in a pure ranking
    application, because the horizon PD multiplies 12-60 of these
    probabilities together, so a systematic level error compounds. The prior
    correction handles the level; the weights would corrupt it.

    Monotone constraints (config.DISCRETE_HAZARD_MONOTONE_DIRECTIONS, on by
    default) force the hazard to move in the economically-signed direction
    for credit_score, orig_cltv, orig_dti and ur_3m_lag. Two reasons: a
    credit committee can be told the model cannot assert "higher FICO, higher
    risk" on some thin slice of the data, and a constrained fit extrapolates
    more stably at the edges of the training range — which the lifetime PD
    and the stressed macro paths both reach.

    Known limitation, disclosed rather than patched: trees extrapolate FLAT.
    Beyond the oldest loan_age or the most extreme ur_3m_lag in the training
    data, the predicted hazard stops responding. The linear model
    extrapolates linearly in the logit (with a constant-extrapolated spline
    baseline). Neither is right, but they are wrong differently, which is
    part of why both are kept.
    """

    name = "xgb"

    def __init__(self, sampling_rate: float = 1.0,
                 params: dict | None = None,
                 use_monotone: bool = config.DISCRETE_HAZARD_USE_MONOTONE_CONSTRAINTS,
                 device: str = DEVICE,
                 seed: int = SEED) -> None:
        super().__init__(sampling_rate=sampling_rate)
        self.params = dict(params if params is not None
                           else config.DISCRETE_HAZARD_XGB_PARAMS)
        self.use_monotone = use_monotone
        self.device = device
        self.seed = seed
        self.model_ = None
        self.categories_: dict[str, list[str]] = {}
        self.monotone_constraints_: tuple[int, ...] = ()

    # -- encoding -----------------------------------------------------------

    def _fit_encoders(self, X: pd.DataFrame) -> None:
        self.feature_names_ = [c for c in X.columns if c in MODEL_COLS]
        self.categories_ = {
            col: sorted(X[col].fillna("missing").astype(str).unique().tolist())
            for col in CAT_FEATURES if col in X.columns
        }

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.feature_names_:
            if col in self.categories_:
                levels = self.categories_[col]
                lookup = {lvl: i for i, lvl in enumerate(levels)}
                raw = X[col].fillna("missing").astype(str)
                # Unseen levels fall back to the first training level, the
                # same convention 03_pd_ensemble.py's _apply_encoder uses.
                cols.append(raw.map(lambda v: lookup.get(v, 0)).to_numpy(dtype=np.float32))
            else:
                cols.append(pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=np.float32))
        return np.column_stack(cols).astype(np.float32)

    def _build_monotone_constraints(self) -> tuple[int, ...]:
        directions = config.DISCRETE_HAZARD_MONOTONE_DIRECTIONS
        return tuple(int(directions.get(col, 0)) for col in self.feature_names_)

    # -- interface ----------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_valid: pd.DataFrame | None = None,
            y_valid: np.ndarray | None = None) -> "XGBHazardModel":
        from xgboost import XGBClassifier

        self._fit_encoders(X_train)
        X_mat = self._encode(X_train)

        params = dict(self.params)
        params.update(device=self.device, random_state=self.seed, n_jobs=-1)
        if self.use_monotone:
            self.monotone_constraints_ = self._build_monotone_constraints()
            params["monotone_constraints"] = self.monotone_constraints_
            constrained = {c: d for c, d in
                           zip(self.feature_names_, self.monotone_constraints_) if d}
            log.info("  [xgb] monotone constraints: %s", constrained or "none")

        eval_set = None
        if X_valid is not None and y_valid is not None and len(X_valid):
            eval_set = [(self._encode(X_valid), np.asarray(y_valid))]
        else:
            # early_stopping_rounds without an eval set raises in xgboost
            params.pop("early_stopping_rounds", None)

        self.model_ = XGBClassifier(**params)
        self.model_.fit(X_mat, np.asarray(y_train),
                        eval_set=eval_set, verbose=False)
        best = getattr(self.model_, "best_iteration", None)
        log.info("  [xgb] fitted on %s rows | best_iteration=%s",
                 f"{len(X_mat):,}", best if best is not None else "n/a")
        return self

    def predict_hazard(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("XGBHazardModel.predict_hazard before fit()")
        margin = self.model_.predict(self._encode(X), output_margin=True)
        return expit(king_zeng_correct_logit(np.asarray(margin, dtype=np.float64),
                                             self.sampling_rate))


# =============================================================================
# CONDITIONAL HORIZON PD
# =============================================================================

def _macro_column(path, period: int, n_loans: int):
    """
    Resolve a macro path to its value at `period` (1-indexed).

    Accepted shapes, so that the same engine serves PIT, TTC and a future
    IFRS 9 scenario projection:
      scalar          — one value for every loan and every period (TTC)
      (h,)            — a per-period path shared by all loans (scenario)
      (n_loans, 1)    — a per-loan constant (PIT flat projection)
      (n_loans, h)    — a per-loan, per-period path
    Paths shorter than the horizon hold their final value, so a 20-quarter
    scenario can be projected against a 60-month horizon without the caller
    having to pad it.
    """
    arr = np.asarray(path, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[min(period - 1, arr.shape[0] - 1)])
    if arr.ndim == 2:
        if arr.shape[0] != n_loans:
            raise ValueError(
                f"macro path has {arr.shape[0]} rows for {n_loans} loans"
            )
        return arr[:, min(period - 1, arr.shape[1] - 1)]
    raise ValueError(f"macro path must be 0-, 1- or 2-dimensional, got {arr.ndim}")


def _horizon_pd_chunk(model: HazardModel,
                      loans: pd.DataFrame,
                      macro_paths: dict,
                      horizons: list[int],
                      lifetime_months: np.ndarray | None,
                      time_col: str) -> dict[str, np.ndarray]:
    n = len(loans)
    base_age = pd.to_numeric(loans[time_col], errors="coerce").fillna(0.0).to_numpy()

    max_h = max(horizons) if horizons else 0
    if lifetime_months is not None and len(lifetime_months):
        max_h = max(max_h, int(np.nanmax(lifetime_months)))

    horizon_set = set(horizons)
    survival = np.ones(n, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    lifetime_pd = np.full(n, np.nan) if lifetime_months is not None else None

    scratch = loans.copy()
    for period in range(1, max_h + 1):
        scratch[time_col] = base_age + period
        for col, path in macro_paths.items():
            if col in scratch.columns:
                scratch[col] = _macro_column(path, period, n)

        hazard = np.clip(model.predict_hazard(scratch), 0.0, 1.0)
        survival *= (1.0 - hazard)

        if period in horizon_set:
            out[f"pd_{period}m"] = 1.0 - survival
        if lifetime_pd is not None:
            due = lifetime_months == period
            if due.any():
                lifetime_pd[due] = 1.0 - survival[due]

    if lifetime_pd is not None:
        # Loans whose cap is 0 (or missing) never get a recorded value.
        out["pd_lifetime"] = lifetime_pd
    return out


def compute_conditional_horizon_pd(model: HazardModel,
                                   loans: pd.DataFrame,
                                   macro_paths: dict | None = None,
                                   horizons: list[int] = HORIZONS,
                                   lifetime_months: np.ndarray | None = None,
                                   prefix: str = "",
                                   time_col: str = TIME_COL,
                                   chunk_size: int = SCORING_CHUNK) -> pd.DataFrame:
    """
    PD conditional on the loan's CURRENT age — the quantity Ch.5's Cox
    implementation failed to produce.

    For a loan observed at age a with covariates x:

        PD(a -> a+h) = 1 - prod_{j=1..h} [ 1 - h_hat(a + j | x, macro_j) ]

    loan_age advances deterministically (a+1, a+2, ...), which is what makes
    this conditional: the product starts at the loan's present age, so the
    loan is never charged again for the default risk of the months it has
    already survived. Equivalently, PD(a -> a+h) = 1 - S(a+h|x)/S(a|x).

    Parameters
    ----------
    model
        Anything implementing the HazardModel interface. Only
        predict_hazard() is called, so this function is model-agnostic and is
        tested against a stub.
    loans
        One row per loan to score, carrying time_col and the model's
        covariates. Typically the latest observation of each loan.
    macro_paths
        {column: path} for the covariates whose future values must be
        assumed. See _macro_column() for accepted shapes. Columns absent from
        `loans` are ignored; covariates absent from macro_paths keep their
        value in `loans`, held constant across the horizon.
    horizons
        Fixed horizons, in months, to report as pd_{h}m.
    lifetime_months
        Optional per-loan horizon cap (remaining_months, clipped) producing
        an additional pd_lifetime column.
    prefix
        Prepended to every output column — "ttc_" produces the ttc_pd_{h}m
        schema that survival_pd_horizons.csv and 10_basel_irb_capital.py
        already expect.

    Scoring is chunked over loans: the engine evaluates the model once per
    (loan chunk, future month), so a 360-month lifetime projection over a
    large OOS population is bounded by chunk_size rows in flight rather than
    by n_loans x horizon.
    """
    macro_paths = dict(macro_paths or {})
    if len(loans) == 0:
        cols = [f"{prefix}pd_{h}m" for h in horizons]
        if lifetime_months is not None:
            cols.append(f"{prefix}pd_lifetime")
        return pd.DataFrame(columns=cols)

    collected: list[dict[str, np.ndarray]] = []
    for start in range(0, len(loans), max(chunk_size, 1)):
        stop = start + max(chunk_size, 1)
        chunk = loans.iloc[start:stop]
        chunk_macro = {
            col: (path if np.asarray(path).ndim < 2
                  else np.asarray(path)[start:stop])
            for col, path in macro_paths.items()
        }
        chunk_lifetime = (None if lifetime_months is None
                          else np.asarray(lifetime_months)[start:stop])
        collected.append(_horizon_pd_chunk(
            model, chunk, chunk_macro, list(horizons), chunk_lifetime, time_col
        ))

    merged = {
        f"{prefix}{key}": np.concatenate([c[key] for c in collected])
        for key in collected[0]
    }
    return pd.DataFrame(merged, index=loans.index)


def build_pit_macro_paths(loans: pd.DataFrame,
                          macro_cols: list[str] = MACRO_FEATURES) -> dict:
    """
    Point-in-time macro path: each loan's CURRENT macro state, held flat
    across the whole projection horizon.

    This is an explicit assumption, not a forecast — "conditions stay as they
    are today". It is the standard PIT convention when the model is not
    paired with a macro forecasting model, and it is what makes the resulting
    PD move with the cycle as IFRS 9 wants. Its cost: at long horizons a loan
    observed in a recession is projected as if the recession never ends (and
    one observed at a cyclical peak as if the expansion never does), so the
    24m-60m PIT PDs are more dispersed across loans than a mean-reverting
    macro path would produce. The TTC path is the mean-reverting counterpart;
    a genuine scenario path can be passed straight to
    compute_conditional_horizon_pd() instead.

    Shape (n_loans, 1) — broadcast across periods without materialising an
    n_loans x horizon array.
    """
    return {c: loans[c].to_numpy(dtype=np.float64).reshape(-1, 1)
            for c in macro_cols if c in loans.columns}


def build_ttc_macro_paths(long_run_means: dict,
                          macro_cols: list[str] = MACRO_FEATURES) -> dict:
    """
    Through-the-cycle macro path: every macro covariate pinned at its
    long-run TRAINING-sample mean, for every loan and every period.

    Everything else about the loan — credit score, CLTV, DTI, loan age — is
    left at its actual value, so this answers "what is this loan's PD at a
    typical point in the cycle", which is the PD Basel IRB capital wants
    (EBA/GL/2017/16 §6.2).
    """
    return {c: float(long_run_means[c]) for c in macro_cols if c in long_run_means}


# =============================================================================
# VALIDATION
# =============================================================================

def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def evaluate_monthly_hazard(model: HazardModel,
                            X: pd.DataFrame,
                            y: np.ndarray,
                            label: str,
                            model_name: str) -> dict:
    """
    One-month-ahead discrimination and calibration on rows that were NOT
    subsampled, so the base rate is the true panel base rate and the log loss
    and mean-predicted-vs-observed comparison are meaningful.

    AUROC is invariant to the prior correction (a monotone shift); log loss
    and the predicted/observed ratio are not, which is exactly why they are
    reported — they are what tells you whether the correction worked.
    """
    y = np.asarray(y)
    n_events = int(y.sum())
    if n_events == 0 or n_events == len(y):
        log.warning("  [%s|%s] %d events in %s rows — metrics not computed.",
                    model_name, label, n_events, f"{len(y):,}")
        return {"model": model_name, "split": label, "metric_set": "monthly_hazard",
                "n": len(y), "n_events": n_events, "auroc": np.nan,
                "log_loss": np.nan, "observed_rate": float(y.mean()) if len(y) else np.nan,
                "mean_predicted": np.nan, "pred_obs_ratio": np.nan}

    hazard = model.predict_hazard(X)
    auroc = float(roc_auc_score(y, hazard))
    ll    = float(log_loss(y, np.clip(hazard, 1e-12, 1 - 1e-12), labels=[0, 1]))
    obs   = float(y.mean())
    pred  = float(hazard.mean())

    log.info("  [%s|%-4s] monthly hazard  n=%s  events=%s  AUROC=%.4f  "
             "logloss=%.6f  observed=%.6f  predicted=%.6f  ratio=%.3f",
             model_name, label, f"{len(y):,}", f"{n_events:,}",
             auroc, ll, obs, pred, pred / obs if obs else np.nan)

    return {"model": model_name, "split": label, "metric_set": "monthly_hazard",
            "n": len(y), "n_events": n_events, "auroc": auroc, "log_loss": ll,
            "observed_rate": obs, "mean_predicted": pred,
            "pred_obs_ratio": pred / obs if obs else np.nan}


def evaluate_horizon_pd(y_true: np.ndarray, pd_pred: np.ndarray,
                        label: str, model_name: str,
                        metric_set: str = "horizon_12m") -> dict:
    """AUROC, KS, Gini and Brier for a horizon PD against a realised outcome."""
    y_true = np.asarray(y_true)
    pd_pred = np.clip(np.asarray(pd_pred, dtype=np.float64), 0.0, 1.0)
    n_events = int(y_true.sum())

    if n_events == 0 or n_events == len(y_true):
        log.warning("  [%s|%s] %d events — horizon metrics not computed.",
                    model_name, label, n_events)
        return {"model": model_name, "split": label, "metric_set": metric_set,
                "n": len(y_true), "n_events": n_events, "auroc": np.nan,
                "ks": np.nan, "gini": np.nan, "brier": np.nan,
                "observed_rate": float(y_true.mean()) if len(y_true) else np.nan,
                "mean_predicted": float(pd_pred.mean()) if len(pd_pred) else np.nan,
                "pred_obs_ratio": np.nan}

    auroc = float(roc_auc_score(y_true, pd_pred))
    ks    = ks_statistic(y_true, pd_pred)
    brier = float(brier_score_loss(y_true, pd_pred))
    obs   = float(y_true.mean())
    pred  = float(pd_pred.mean())

    log.info("  [%s|%-12s] %s  n=%s  events=%s  AUROC=%.4f  KS=%.4f  "
             "Gini=%.4f  Brier=%.6f  observed=%.5f  predicted=%.5f",
             model_name, label, metric_set, f"{len(y_true):,}", f"{n_events:,}",
             auroc, ks, 2 * auroc - 1, brier, obs, pred)

    return {"model": model_name, "split": label, "metric_set": metric_set,
            "n": len(y_true), "n_events": n_events, "auroc": auroc, "ks": ks,
            "gini": 2 * auroc - 1, "brier": brier, "observed_rate": obs,
            "mean_predicted": pred, "pred_obs_ratio": pred / obs if obs else np.nan}


def decile_calibration(y_true: np.ndarray, pd_pred: np.ndarray,
                       n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted PD vs realised default rate by predicted-PD decile."""
    frame = pd.DataFrame({"y": np.asarray(y_true),
                          "p": np.asarray(pd_pred, dtype=np.float64)}).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["decile", "n", "mean_predicted", "observed_rate"])
    try:
        frame["decile"] = pd.qcut(frame["p"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        frame["decile"] = 0
    return (
        frame.groupby("decile")
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )


def build_snapshot_population(panel: pd.DataFrame,
                              snapshot_dates: list[str] = SNAPSHOT_DATES,
                              window_days: int = 365) -> pd.DataFrame:
    """
    Rows at fixed snapshot dates whose 12-month forward window is FULLY
    observed — the population on which a 12-month conditional PD can be
    honestly compared with a realised 12-month outcome.

    The maturity rule is the one filter_immature_right_censored() applies in
    01_data_preprocessing.py: a row is usable if the loan's default_date is
    known (the outcome is observed regardless of how close the row sits to
    the panel's end) or if a full `window_days` of panel remains after it. A
    row failing both is right-censored — "has not defaulted yet, and we will
    not know for N more months" — and scoring it would credit the model for
    predicting a non-default that has not happened yet.

    The realised outcome is the existing default_12m column, which is exactly
    "defaulted within 365 days of this row".
    """
    if panel.empty:
        return panel.copy()

    report_date = pd.to_datetime(panel["report_date"])
    global_max = report_date.max()
    cutoff = global_max - pd.Timedelta(days=window_days)
    has_default = pd.to_datetime(panel["default_date"]).notna()

    wanted = pd.to_datetime(pd.Series(list(snapshot_dates))).tolist()
    on_snapshot = report_date.isin(wanted)
    mature = has_default | (report_date <= cutoff)

    out = panel.loc[on_snapshot & mature].copy()
    log.info("  Snapshot population: %s rows at %s  (panel max %s, maturity "
             "cutoff %s, %s rows dropped as immature)",
             f"{len(out):,}",
             ", ".join(d.date().isoformat() for d in wanted),
             global_max.date(), cutoff.date(),
             f"{int((on_snapshot & ~mature).sum()):,}")
    return out


def observed_discrete_km(panel: pd.DataFrame,
                         target_col: str = TARGET_HAZARD,
                         max_age: int = 240) -> pd.DataFrame:
    """
    Non-parametric discrete-time Kaplan-Meier over loan_age, computed
    directly on the panel.

    At each age t the panel gives the at-risk count (rows observed at that
    age) and the event count, so the empirical hazard is events/at-risk and
    the survival curve is the running product of (1 - hazard). This is the
    discrete-time KM estimator, and because it is computed from the panel
    rather than from a per-loan collapse it does not inherit the
    end-of-follow-up bias that 06_survival_analysis.py's per-loan duration
    construction carries.
    """
    ages = pd.to_numeric(panel[TIME_COL], errors="coerce")
    frame = pd.DataFrame({"age": ages, "event": panel[target_col].to_numpy()}).dropna()
    frame = frame[(frame["age"] >= 0) & (frame["age"] <= max_age)]
    if frame.empty:
        return pd.DataFrame(columns=["loan_age", "n_at_risk", "n_events",
                                     "hazard", "survival"])

    agg = (frame.groupby(frame["age"].astype(int))
           .agg(n_at_risk=("event", "size"), n_events=("event", "sum"))
           .reset_index().rename(columns={"age": "loan_age"})
           .sort_values("loan_age"))
    agg["hazard"] = agg["n_events"] / agg["n_at_risk"].clip(lower=1)
    agg["survival"] = (1.0 - agg["hazard"]).cumprod()
    return agg


def predicted_mean_hazard_by_age(model: HazardModel,
                                 panel: pd.DataFrame,
                                 max_age: int = 240,
                                 sample_rows: int = 300_000,
                                 seed: int = SEED) -> pd.DataFrame:
    """
    Mean predicted hazard at each observed loan_age, with the matching
    implied survival curve, for comparison against observed_discrete_km().

    Averaging the predicted hazard over the loans actually at risk at each
    age is the like-for-like counterpart of the empirical hazard: both are
    conditional on being at risk at that age, so a gap between the two curves
    is a genuine calibration gap rather than a composition effect.
    """
    frame = panel
    if len(frame) > sample_rows:
        frame = frame.sample(n=sample_rows, random_state=seed)

    hazard = model.predict_hazard(frame)
    ages = pd.to_numeric(frame[TIME_COL], errors="coerce").astype("Int64")
    out = (pd.DataFrame({"loan_age": ages, "hazard": hazard}).dropna()
           .groupby("loan_age")["hazard"].mean().reset_index()
           .sort_values("loan_age"))
    out = out[(out["loan_age"] >= 0) & (out["loan_age"] <= max_age)]
    out["survival"] = (1.0 - out["hazard"]).cumprod()
    return out


# =============================================================================
# CH.2 BENCHMARK  (12-month XGBoost, identical rows)
# =============================================================================

def benchmark_ch2_12m(snapshot: pd.DataFrame,
                      train_panel: pd.DataFrame | None = None) -> tuple[np.ndarray | None, str]:
    """
    Score the Ch.2 12-month XGBoost model on EXACTLY the snapshot rows, so
    the comparison is on an identical population rather than on whatever
    rows each script happened to evaluate.

    03_pd_ensemble.py writes pd_xgb_results.csv as (split, default_12m,
    xgb_score) with NO loan identifier and no report_date, so its persisted
    scores cannot be joined back to specific loan-months. Two consequences:

      1. If a future revision of 03 persists loan_seq_num and report_date,
         this function joins on them and uses the real Ch.2 scores — that
         path is implemented and preferred.
      2. Otherwise the Ch.2 model is re-fitted here on pd_train with
         config.PD_FEATURES and config.TARGET_PD and scored on the snapshot
         rows. This is the same approach 07_macro_scenario_analysis.py
         already takes (retrain_xgboost()) and is the only way to get an
         identical-population comparison without modifying Ch.2. It is a
         faithful re-fit of the Ch.2 specification, NOT 03's persisted
         scores — the log says which path was taken, and the metrics CSV
         records it.

    Returns (scores aligned to snapshot.index, source description), or
    (None, reason) if neither path is available.
    """
    results_path = PROC_DIR / "pd_xgb_results.csv"
    if results_path.exists():
        head = pd.read_csv(results_path, nrows=5)
        if {"loan_seq_num", "report_date"}.issubset(head.columns):
            scores = pd.read_csv(results_path)
            scores["report_date"] = pd.to_datetime(scores["report_date"])
            key = snapshot[["loan_seq_num", "report_date"]].copy()
            key["report_date"] = pd.to_datetime(key["report_date"])
            merged = key.merge(
                scores[["loan_seq_num", "report_date", "xgb_score"]],
                on=["loan_seq_num", "report_date"], how="left",
            )
            if merged["xgb_score"].notna().any():
                log.info("  Ch.2 benchmark: joined %s of %s snapshot rows from %s",
                         f"{int(merged['xgb_score'].notna().sum()):,}",
                         f"{len(snapshot):,}", results_path.name)
                return merged["xgb_score"].to_numpy(), f"{results_path.name} (persisted Ch.2 scores)"
        else:
            log.warning(
                "  %s carries no loan_seq_num/report_date, so its scores cannot "
                "be joined to the snapshot rows — re-fitting the Ch.2 "
                "specification instead so the comparison stays on an identical "
                "population.", results_path.name,
            )

    if train_panel is None or train_panel.empty:
        return None, "unavailable (no Ch.2 scores and no training panel supplied)"

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - environment-dependent
        return None, f"unavailable ({exc})"

    feats = [f for f in config.PD_FEATURES if f in train_panel.columns
             and f in snapshot.columns]
    if not feats or TARGET_12M not in train_panel.columns:
        return None, "unavailable (Ch.2 features not present in the panel)"

    log.info("  Ch.2 benchmark: re-fitting the Ch.2 12-month specification on "
             "%s training rows over %d features …", f"{len(train_panel):,}", len(feats))

    encoders: dict[str, LabelEncoder] = {}
    for col in config.PD_CAT_FEATURES:
        if col in feats:
            enc = LabelEncoder()
            enc.fit(train_panel[col].fillna("missing").astype(str))
            encoders[col] = enc

    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[feats].copy()
        for col, enc in encoders.items():
            known = set(enc.classes_)
            fallback = enc.classes_[0]
            out[col] = enc.transform(
                out[col].fillna("missing").astype(str)
                .map(lambda v, k=known, fb=fallback: v if v in k else fb)
            )
        return out

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(_encode(train_panel)).astype(np.float32)
    y_tr = train_panel[TARGET_12M].to_numpy()

    pos = int(y_tr.sum())
    spw = (len(y_tr) - pos) / max(pos, 1)
    bench = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.5,
        colsample_bytree=0.8, min_child_weight=50, gamma=1.0, reg_alpha=0.1,
        reg_lambda=1.0, eval_metric="auc", tree_method="hist", device=DEVICE,
        random_state=SEED, n_jobs=-1, scale_pos_weight=spw,
    )
    bench.fit(X_tr, y_tr, verbose=False)

    X_snap = imputer.transform(_encode(snapshot)).astype(np.float32)
    scores = bench.predict_proba(X_snap)[:, 1]
    del X_tr, y_tr
    gc.collect()
    return scores, "re-fitted Ch.2 12-month specification (03 persists no row keys)"


# =============================================================================
# VISUALISATIONS
# =============================================================================

def _save(fig, filename: str) -> None:
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_seasoning_curve(curves: dict[str, pd.DataFrame], filename: str,
                         title: str) -> None:
    """Estimated baseline hazard (seasoning curve) against loan age."""
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (label, curve), color in zip(curves.items(), PALETTE):
        ax.plot(curve["loan_age"], curve["hazard"] * 100,
                color=MODEL_COLORS.get(label, color), linewidth=2.2, label=label)
    ax.set_xlabel("Loan Age (months)", fontsize=10)
    ax.set_ylabel("Monthly default hazard (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, filename)


def plot_odds_ratios(table: pd.DataFrame, filename: str) -> None:
    """Forest plot of odds ratios with 95% CIs for the linear hazard model."""
    show = table[table["kind"].isin(["continuous", "indicator"])].copy()
    if show.empty:
        return
    show = show.sort_values("odds_ratio")

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(10, max(5, len(show) * 0.5 + 1.5)))
    y_pos = np.arange(len(show))
    or_v = show["odds_ratio"].to_numpy()
    lo   = show["or_lower_95"].to_numpy()
    hi   = show["or_upper_95"].to_numpy()
    colors = ["#EF4444" if v > 1 else "#22C55E" for v in or_v]

    ax.errorbar(or_v, y_pos, xerr=[np.maximum(or_v - lo, 0), np.maximum(hi - or_v, 0)],
                fmt="none", ecolor="#E2E8F0", elinewidth=1.4, capsize=4, capthick=1.4)
    ax.scatter(or_v, y_pos, color=colors, s=55, zorder=5)
    ax.axvline(1.0, color="#4B5563", linewidth=1.5, linestyle="--",
               label="OR = 1 (no effect)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(show["feature"].tolist(), fontsize=9)
    ax.set_xlabel("Odds ratio on the monthly hazard  (OR > 1 = higher risk)", fontsize=10)
    ax.set_title("Discrete-Time Logistic Hazard — Odds Ratios (95% CI)\n"
                 "Per one unit of the covariate, on its original scale",
                 fontsize=11, fontweight="bold", color="white", pad=14)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, filename)


def plot_decile_calibration(tables: dict[str, pd.DataFrame], filename: str,
                            title: str) -> None:
    """Predicted vs observed by predicted-PD decile."""
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    upper = 0.0
    for (label, table), color in zip(tables.items(), PALETTE):
        if table.empty:
            continue
        ax.scatter(table["mean_predicted"] * 100, table["observed_rate"] * 100,
                   s=70, zorder=5, label=label,
                   color=MODEL_COLORS.get(label, color))
        upper = max(upper, float(table["mean_predicted"].max()),
                    float(table["observed_rate"].max()))
    if upper <= 0:
        plt.close(fig)
        return
    ax.plot([0, upper * 100], [0, upper * 100], "r--", linewidth=1.4,
            label="Perfect calibration")
    ax.set_xlabel("Mean predicted 12m PD (%)", fontsize=10)
    ax.set_ylabel("Observed 12m default rate (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, filename)


def plot_pit_vs_ttc(horizon_df: pd.DataFrame, horizons: list[int],
                    filename: str, model_label: str) -> None:
    """Mean PIT vs mean TTC PD at each horizon."""
    pit = [c for c in (f"pd_{h}m" for h in horizons) if c in horizon_df.columns]
    ttc = [c for c in (f"ttc_pd_{h}m" for h in horizons) if c in horizon_df.columns]
    if not pit or not ttc:
        return

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(
        f"Point-in-Time vs Through-the-Cycle Conditional PD — {model_label}\n"
        "PIT: each loan's current macro state held flat   "
        "TTC: macro covariates at their long-run training mean",
        fontsize=11, fontweight="bold", color="white",
    )
    x = np.arange(len(horizons))
    width = 0.35
    pit_means = [horizon_df[f"pd_{h}m"].mean() * 100 for h in horizons]
    ttc_means = [horizon_df[f"ttc_pd_{h}m"].mean() * 100 for h in horizons]

    ax.bar(x - width / 2, pit_means, width, color="#38BDF8", alpha=0.85, label="PIT PD")
    ax.bar(x + width / 2, ttc_means, width, color="#F59E0B", alpha=0.85, label="TTC PD")
    for i, (p, t) in enumerate(zip(pit_means, ttc_means)):
        ax.text(i - width / 2, p, f"{p:.3f}%", ha="center", va="bottom",
                fontsize=8, color="#E2E8F0")
        ax.text(i + width / 2, t, f"{t:.3f}%", ha="center", va="bottom",
                fontsize=8, color="#E2E8F0")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}m" for h in horizons], fontsize=10)
    ax.set_xlabel("Horizon (months ahead of the loan's current age)", fontsize=10)
    ax.set_ylabel("Mean conditional PD (%)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, filename)


def plot_km_check(observed: pd.DataFrame, predicted: pd.DataFrame,
                  filename: str, model_label: str) -> None:
    """Observed discrete Kaplan-Meier vs mean predicted survival over age."""
    if observed.empty or predicted.empty:
        return
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Observed vs Predicted — {model_label}\n"
        "Empirical panel hazard (events / at-risk at each loan age) against "
        "the model's mean predicted hazard for the loans at risk at that age",
        fontsize=11, fontweight="bold", color="white",
    )

    axes[0].plot(observed["loan_age"], observed["hazard"] * 100,
                 color="#94A3B8", linewidth=1.8, label="Observed")
    axes[0].plot(predicted["loan_age"], predicted["hazard"] * 100,
                 color=MODEL_COLORS.get(model_label, "#38BDF8"),
                 linewidth=2.0, label="Predicted")
    axes[0].set_xlabel("Loan age (months)", fontsize=10)
    axes[0].set_ylabel("Monthly hazard (%)", fontsize=10)
    axes[0].set_title("Hazard", color="#CBD5E1", fontsize=10)
    axes[0].legend(fontsize=9)

    axes[1].plot(observed["loan_age"], observed["survival"],
                 color="#94A3B8", linewidth=1.8, label="Observed KM")
    axes[1].plot(predicted["loan_age"], predicted["survival"],
                 color=MODEL_COLORS.get(model_label, "#38BDF8"),
                 linewidth=2.0, label="Predicted")
    axes[1].set_xlabel("Loan age (months)", fontsize=10)
    axes[1].set_ylabel("Survival probability", fontsize=10)
    axes[1].set_title("Survival", color="#CBD5E1", fontsize=10)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    axes[1].legend(fontsize=9)

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _save(fig, filename)


def plot_model_comparison(seasoning: dict[str, pd.DataFrame],
                          calibration: dict[str, pd.DataFrame]) -> None:
    """Overlay of both models' seasoning curves and decile calibration."""
    if not seasoning and not calibration:
        return
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Discrete-Time Hazard — Linear vs Gradient-Boosted\n"
        "Same target, same covariates, same case-control subsample and "
        "prior correction",
        fontsize=12, fontweight="bold", color="white",
    )

    for label, curve in seasoning.items():
        axes[0].plot(curve["loan_age"], curve["hazard"] * 100, linewidth=2.2,
                     color=MODEL_COLORS.get(label, "#38BDF8"), label=label)
    axes[0].set_xlabel("Loan age (months)", fontsize=10)
    axes[0].set_ylabel("Monthly hazard (%)", fontsize=10)
    axes[0].set_title("Seasoning curve", color="#CBD5E1", fontsize=10)
    axes[0].legend(fontsize=9)

    upper = 0.0
    for label, table in calibration.items():
        if table.empty:
            continue
        axes[1].scatter(table["mean_predicted"] * 100, table["observed_rate"] * 100,
                        s=65, zorder=5, color=MODEL_COLORS.get(label, "#38BDF8"),
                        label=label)
        upper = max(upper, float(table["mean_predicted"].max()),
                    float(table["observed_rate"].max()))
    if upper > 0:
        axes[1].plot([0, upper * 100], [0, upper * 100], "r--", linewidth=1.4,
                     label="Perfect calibration")
    axes[1].set_xlabel("Mean predicted 12m PD (%)", fontsize=10)
    axes[1].set_ylabel("Observed 12m default rate (%)", fontsize=10)
    axes[1].set_title("12-month conditional PD calibration", color="#CBD5E1", fontsize=10)
    axes[1].legend(fontsize=9)

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _save(fig, "discrete_hazard_comparison.png")


def compute_xgb_shap(model: XGBHazardModel, X: pd.DataFrame) -> np.ndarray | None:
    """
    Exact TreeSHAP values for the XGBoost hazard model, in log-odds units.

    Tries shap.TreeExplainer first, matching 05_shap_explanations.py. That
    path breaks on some shap/XGBoost version combinations — shap <= 0.49
    cannot parse the bracketed base_score ("[5E-1]") that XGBoost >= 3.0
    writes into its model config — so it falls back to XGBoost's own
    pred_contribs, which is the same TreeSHAP algorithm implemented inside
    XGBoost (shap delegates to it for XGBoost models anyway). The fallback
    returns identical values; only the caller differs.

    Returns an (n_rows, n_features) array, or None if neither path works.
    """
    encoded = model._encode(X)

    try:
        import shap
        explainer = shap.TreeExplainer(model.model_)
        values = explainer.shap_values(encoded)
        return values[1] if isinstance(values, list) else values
    except Exception as exc:
        log.info("  shap.TreeExplainer unavailable (%s) — using XGBoost's "
                 "native TreeSHAP (pred_contribs) instead.", exc)

    try:
        from xgboost import DMatrix
        contribs = model.model_.get_booster().predict(
            DMatrix(encoded), pred_contribs=True)
        # Last column is the bias (expected value) term, not a feature.
        return np.asarray(contribs)[:, :-1]
    except Exception as exc:  # pragma: no cover - environment-dependent
        log.warning("  SHAP computation failed (%s) — skipping SHAP outputs.", exc)
        return None


def plot_xgb_shap(model: XGBHazardModel, X: pd.DataFrame,
                  max_rows: int = 50_000) -> None:
    """
    Global SHAP importance and the loan_age dependence plot — the learned
    seasoning effect, read off the trees rather than imposed by a spline.
    """
    frame = X if len(X) <= max_rows else X.sample(n=max_rows, random_state=SEED)
    values = compute_xgb_shap(model, frame)
    if values is None:
        return

    names = model.feature_names_
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=names).sort_values()

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.4 + 1.5)))
    ax.barh(mean_abs.index, mean_abs.to_numpy(), color="#F59E0B", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|  (log-odds of the monthly hazard)", fontsize=10)
    ax.set_title("XGBoost Hazard Model — Global SHAP Importance",
                 fontsize=11, fontweight="bold", color="white", pad=14)
    ax.grid(True, axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "discrete_hazard_xgb_shap_importance.png")

    if TIME_COL in names:
        idx = names.index(TIME_COL)
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax.scatter(pd.to_numeric(frame[TIME_COL], errors="coerce"), values[:, idx],
                   s=6, alpha=0.25, color="#38BDF8", linewidths=0)
        ax.axhline(0, color="#4B5563", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Loan age (months)", fontsize=10)
        ax.set_ylabel("SHAP value for loan_age  (log-odds)", fontsize=10)
        ax.set_title("XGBoost Hazard Model — Learned Seasoning Effect\n"
                     "SHAP contribution of loan age to the monthly hazard",
                     fontsize=11, fontweight="bold", color="white", pad=14)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _save(fig, "discrete_hazard_xgb_shap_dependence_loan_age.png")


# =============================================================================
# SCORING POPULATION
# =============================================================================

def latest_observation_per_loan(panel: pd.DataFrame) -> pd.DataFrame:
    """
    One row per loan — its most recent observation, which carries the loan's
    current age and current macro state. This is the population the horizon
    PDs are computed for, matching what 06_survival_analysis.py scores.
    """
    if panel.empty:
        return panel.copy()
    idx = panel.groupby("loan_seq_num")[TIME_COL].idxmax()
    return panel.loc[idx].reset_index(drop=True)


def lifetime_horizon_months(loans: pd.DataFrame,
                            cap: int = LIFETIME_CAP) -> np.ndarray | None:
    """
    Per-loan lifetime horizon: remaining_months, clipped to [1, cap].

    Returns None when remaining_months is absent, in which case no
    pd_lifetime column is produced — better than silently substituting a
    fixed horizon and labelling it "lifetime".
    """
    if "remaining_months" not in loans.columns:
        log.warning("  remaining_months not in the scored population — "
                    "no lifetime PD will be produced.")
        return None
    months = pd.to_numeric(loans["remaining_months"], errors="coerce")
    if months.notna().sum() == 0:
        log.warning("  remaining_months is entirely missing — no lifetime PD.")
        return None
    return months.fillna(cap).clip(lower=1, upper=cap).to_numpy().astype(int)


# =============================================================================
# MAIN
# =============================================================================

def _load_panel(name: str) -> pd.DataFrame:
    """Load one split with selective columns, build the target, guard leakage."""
    path = PROC_DIR / name
    import pyarrow.parquet as pq
    available = pq.read_schema(path).names
    missing_required = [c for c in ["loan_seq_num", "report_date", "default_date"]
                        if c not in available]
    if missing_required:
        raise KeyError(
            f"{name} is missing {missing_required}. Re-run "
            f"01_data_preprocessing.py — default_date must be persisted to the "
            f"PD parquet files for the discrete hazard target to be buildable."
        )

    cols = [c for c in _LOAD_COLS if c in available]
    panel = pd.read_parquet(path, columns=cols)
    panel["report_date"] = pd.to_datetime(panel["report_date"])
    panel = drop_post_default_rows(panel)
    panel = build_discrete_target(panel)

    duplicated = loans_with_multiple_events(panel)
    if len(duplicated):
        log.warning("  %s: %s loan(s) carry more than one hazard event — "
                    "investigate the reporting cadence for these loans.",
                    name, f"{len(duplicated):,}")

    log.info("  %-16s %s rows | %s events | monthly hazard %.6f%%",
             name, f"{len(panel):,}", f"{int(panel[TARGET_HAZARD].sum()):,}",
             panel[TARGET_HAZARD].mean() * 100)
    return panel


def _eval_sample(panel: pd.DataFrame, max_rows: int = MAX_EVAL_ROWS) -> pd.DataFrame:
    """Uniform random sample when a split is too large to score whole."""
    if len(panel) <= max_rows:
        return panel
    log.info("  Uniformly sampling %s of %s rows for evaluation.",
             f"{max_rows:,}", f"{len(panel):,}")
    return panel.sample(n=max_rows, random_state=SEED)


def main() -> None:
    gpu_str = f"{N_GPUS}x GPU (CUDA)" if N_GPUS > 0 else "CPU"
    log.info("=" * 70)
    log.info("Mortgage Credit Risk  |  Ch.5b — Discrete-Time Survival (Hazard)")
    log.info("Device: %s  |  subsample rate: %.4f  |  spline df: %d",
             gpu_str, SUBSAMPLE_RATE, SPLINE_DF)
    log.info("=" * 70)

    # ── [1/8] Load panels and build the discrete-time target ─────────────
    log.info("")
    log.info("[1/8] Loading panels and building the discrete-time target …")
    train = _load_panel("pd_train.parquet")
    oos   = _load_panel("pd_oos.parquet")
    oot   = _load_panel("pd_oot.parquet")

    excluded_present = [c for c in EXCLUDED_FEATURES if c in MODEL_COLS]
    if excluded_present:
        raise RuntimeError(
            f"Internal covariates {excluded_present} leaked into the model "
            "columns — see the module docstring for why they are excluded."
        )
    # Resolve the covariate list against what the parquet actually carries,
    # the same defensive pattern 03_pd_ensemble.py uses — a macro file absent
    # at preprocessing time leaves hpi_change/ur_3m_lag out of the panel.
    model_cols = [c for c in MODEL_COLS if c in train.columns]
    dropped = [c for c in MODEL_COLS if c not in model_cols]
    if dropped:
        log.warning("  Covariates absent from the panel, dropped: %s", dropped)
    log.info("  Covariates: %s", model_cols)
    log.info("  Deliberately excluded: %s", EXCLUDED_FEATURES)

    # Long-run macro means from the FULL training panel, before subsampling.
    long_run_means = {c: float(pd.to_numeric(train[c], errors="coerce").mean())
                      for c in MACRO_FEATURES if c in train.columns}
    log.info("  Long-run macro anchor (TTC): %s",
             ", ".join(f"{c}={v:.4f}" for c, v in long_run_means.items()) or "none")

    observed_km = observed_discrete_km(train)

    # ── [2/8] Case-control subsample ──────────────────────────────────────
    log.info("")
    log.info("[2/8] Case-control subsampling (shared by both models) …")
    train_sub, realised_rate = subsample_case_control(train, SUBSAMPLE_RATE)
    oos_sub, _ = subsample_case_control(oos, SUBSAMPLE_RATE, seed=SEED + 1)

    X_tr = train_sub[model_cols]
    y_tr = train_sub[TARGET_HAZARD].to_numpy()
    X_va = oos_sub[model_cols]
    y_va = oos_sub[TARGET_HAZARD].to_numpy()
    log.info("  Prior correction: logit shift of log(%.6f) = %.4f",
             realised_rate, np.log(realised_rate))

    # ── [3/8] Fit both models on the same rows ───────────────────────────
    log.info("")
    log.info("[3/8] Fitting the linear model (discrete-time logistic regression) …")
    logit_model = LogitHazardModel(sampling_rate=realised_rate)
    logit_model.fit(X_tr, y_tr)

    coef_table = logit_model.coefficient_table()
    coef_table.to_csv(PROC_DIR / "discrete_hazard_logit_coefficients.csv", index=False)
    log.info("  Coefficients → discrete_hazard_logit_coefficients.csv")
    log.info("  Categorical reference levels: %s",
             logit_model.design.reference_levels_ or "none")

    log.info("")
    log.info("[3/8] Fitting the challenger (XGBoost discrete-time hazard) …")
    xgb_model = XGBHazardModel(sampling_rate=realised_rate)
    xgb_model.fit(X_tr, y_tr, X_va, y_va)

    models: dict[str, HazardModel] = {"logit": logit_model, "xgb": xgb_model}

    # ── [4/8] Proportional hazards test (linear model only) ──────────────
    log.info("")
    log.info("[4/8] Proportional hazards likelihood-ratio tests (linear model) …")
    ph_tests = proportional_hazards_lr_test(logit_model, X_tr, y_tr)
    if not ph_tests.empty:
        ph_tests.to_csv(PROC_DIR / "discrete_hazard_logit_ph_tests.csv", index=False)
        log.info("  PH tests → discrete_hazard_logit_ph_tests.csv")
        log.info("  The XGBoost challenger makes no proportional hazards "
                 "assumption, so no equivalent test applies to it.")

    del train_sub, oos_sub
    gc.collect()

    # ── [5/8] Monthly hazard validation on unsampled rows ────────────────
    log.info("")
    log.info("[5/8] Monthly hazard discrimination (unsampled OOS and OOT rows) …")
    oos_eval = _eval_sample(oos)
    oot_eval = _eval_sample(oot)

    metrics: list[dict] = []
    for name, model in models.items():
        for label, frame in [("OOS", oos_eval), ("OOT", oot_eval)]:
            metrics.append(evaluate_monthly_hazard(
                model, frame[model_cols], frame[TARGET_HAZARD].to_numpy(),
                label, name,
            ))

    # ── [6/8] Horizon validation on fully-observed snapshot rows ─────────
    log.info("")
    log.info("[6/8] 12-month conditional PD vs realised 12-month outcome …")
    snapshot = build_snapshot_population(oot)
    calibration_tables: dict[str, pd.DataFrame] = {}

    if snapshot.empty:
        log.warning("  No usable snapshot rows — horizon validation skipped. "
                    "Check config.DISCRETE_HAZARD_SNAPSHOT_DATES against the "
                    "panel's date range.")
    else:
        y_snapshot = snapshot[TARGET_12M].to_numpy()
        snapshot_loans = snapshot[model_cols]
        pit_paths = build_pit_macro_paths(snapshot)

        for name, model in models.items():
            pd_12m = compute_conditional_horizon_pd(
                model, snapshot_loans, pit_paths, horizons=[12],
            )["pd_12m"].to_numpy()
            metrics.append(evaluate_horizon_pd(y_snapshot, pd_12m, "OOT-snapshot", name))
            calib = decile_calibration(y_snapshot, pd_12m)
            calibration_tables[name] = calib
            calib.assign(model=name).to_csv(
                PROC_DIR / f"discrete_hazard_{name}_decile_calibration.csv", index=False)
            plot_decile_calibration(
                {name: calib},
                f"discrete_hazard_{name}_calibration.png",
                f"12-Month Conditional PD Calibration — {name}\n"
                "Decile means on fully-observed OOT snapshot rows",
            )

        bench_scores, bench_source = benchmark_ch2_12m(snapshot, train_panel=train)
        if bench_scores is not None and np.isfinite(bench_scores).any():
            valid = np.isfinite(bench_scores)
            bench_metrics = evaluate_horizon_pd(
                y_snapshot[valid], bench_scores[valid], "OOT-snapshot",
                "ch2_xgb_12m",
            )
            bench_metrics["source"] = bench_source
            metrics.append(bench_metrics)
            calibration_tables["ch2_xgb_12m"] = decile_calibration(
                y_snapshot[valid], bench_scores[valid])
            log.info("  Ch.2 benchmark source: %s", bench_source)
        else:
            log.warning("  Ch.2 benchmark unavailable: %s", bench_source)

    # ── [7/8] Conditional horizon PDs for the scored population ──────────
    log.info("")
    log.info("[7/8] Conditional horizon PDs (PIT and TTC) for the OOS population …")
    scored = latest_observation_per_loan(oos)
    log.info("  Scoring %s loans (latest observation each) at horizons %s "
             "plus lifetime …", f"{len(scored):,}", HORIZONS)

    lifetime = lifetime_horizon_months(scored)
    ttc_paths = build_ttc_macro_paths(long_run_means)
    seasoning_curves: dict[str, pd.DataFrame] = {}

    for name, model in models.items():
        log.info("  [%s] PIT …", name)
        pit = compute_conditional_horizon_pd(
            model, scored[model_cols], build_pit_macro_paths(scored),
            horizons=HORIZONS, lifetime_months=lifetime,
        )
        log.info("  [%s] TTC …", name)
        ttc = compute_conditional_horizon_pd(
            model, scored[model_cols], ttc_paths,
            horizons=HORIZONS, lifetime_months=lifetime, prefix="ttc_",
        )

        horizon_df = pd.concat(
            [scored[["loan_seq_num"]].reset_index(drop=True),
             pit.reset_index(drop=True), ttc.reset_index(drop=True)], axis=1,
        )
        horizon_df["actual_default"] = scored[TARGET_12M].to_numpy()
        out_path = PROC_DIR / f"discrete_hazard_{name}_pd_horizons.csv"
        horizon_df.to_csv(out_path, index=False)
        log.info("  [%s] Horizon PDs → %s", name, out_path.name)
        for col in [c for c in horizon_df.columns
                    if c.startswith("pd_") or c.startswith("ttc_pd_")]:
            log.info("    mean %-16s %.4f%%", col, horizon_df[col].mean() * 100)

        # Seasoning curve: the model's hazard over loan age at a reference
        # loan (median continuous covariates, reference categorical levels).
        ages = np.arange(0, 241, 1)
        reference = scored[model_cols].median(numeric_only=True).to_frame().T
        for cat in CAT_FEATURES:
            if cat in model_cols:
                reference[cat] = (logit_model.design.reference_levels_.get(cat)
                                  or scored[cat].mode().iloc[0])
        reference = reference[model_cols]
        curve = pd.DataFrame({"loan_age": ages})
        frame = pd.concat([reference] * len(ages), ignore_index=True)
        frame[TIME_COL] = ages
        curve["hazard"] = model.predict_hazard(frame)
        seasoning_curves[name] = curve

        plot_seasoning_curve(
            {name: curve}, f"discrete_hazard_{name}_seasoning.png",
            f"Estimated Baseline Hazard (Seasoning Curve) — {name}\n"
            "Monthly default hazard by loan age, other covariates at "
            "reference values",
        )
        plot_pit_vs_ttc(horizon_df, HORIZONS,
                        f"discrete_hazard_{name}_pit_vs_ttc.png", name)
        plot_km_check(observed_km,
                      predicted_mean_hazard_by_age(model, oos_eval),
                      f"discrete_hazard_{name}_km_check.png", name)

    plot_odds_ratios(coef_table, "discrete_hazard_logit_odds_ratios.png")
    plot_xgb_shap(xgb_model, X_tr)
    plot_model_comparison(seasoning_curves, calibration_tables)

    # ── [8/8] Metrics and the one comparison table ───────────────────────
    log.info("")
    log.info("[8/8] Writing metrics and the model comparison table …")
    metrics_df = pd.DataFrame(metrics)
    metrics_df["sampling_rate"] = realised_rate

    for name in models:
        sub = metrics_df[metrics_df["model"] == name]
        sub.to_csv(PROC_DIR / f"discrete_hazard_{name}_metrics.csv", index=False)
        log.info("  %s metrics → discrete_hazard_%s_metrics.csv", name, name)

    # One table, both metric sets: the monthly-hazard rows carry AUROC and
    # log loss, the horizon rows carry AUROC/KS/Gini/Brier. Columns a given
    # metric set does not produce stay blank rather than being dropped, so
    # the linear model, the XGBoost challenger and the Ch.2 benchmark are
    # readable side by side on identical rows.
    preferred = ["model", "split", "metric_set", "n", "n_events", "auroc",
                 "ks", "gini", "brier", "log_loss", "observed_rate",
                 "mean_predicted", "pred_obs_ratio", "source"]
    cols = [c for c in preferred if c in metrics_df.columns]
    cols += [c for c in metrics_df.columns if c not in cols]
    comparison = metrics_df[cols].sort_values(["metric_set", "split", "model"])
    comparison.to_csv(PROC_DIR / "discrete_hazard_comparison.csv", index=False)
    log.info("  Comparison → discrete_hazard_comparison.csv")
    log.info("\n%s", comparison.to_string(index=False))

    log.info("")
    log.info("=" * 70)
    log.info("Ch.5b complete — discrete-time hazard models fitted (logit + xgb).")
    log.info("  Capital model for 10_basel_irb_capital.py: %s "
             "(config.DISCRETE_HAZARD_CAPITAL_MODEL)",
             config.DISCRETE_HAZARD_CAPITAL_MODEL)
    log.info("  Next: python 10_basel_irb_capital.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
