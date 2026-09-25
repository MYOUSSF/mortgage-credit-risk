"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.10 — Competing Risks (Prepayment)
=============================================================================
Script  : 12_competing_risks.py
Purpose : Estimate the cumulative incidence of default when voluntary
          prepayment is treated as a COMPETING RISK rather than as censoring,
          under two independent specifications, and quantify how much the
          naive 1 - S(t) conversion overstates lifetime default probability.

The problem
-----------
  06_survival_analysis.py fits a Cox model in which a prepaid loan is
  censored, then converts the survival function to a default probability
  with 1 - S(t). Censoring encodes "this loan is still at risk; we simply
  stopped watching". For a prepaid mortgage that is false: the loan is gone,
  the lien is released, and it can never default. The two are not
  interchangeable, and the gap is not small — in this dataset roughly 10-12
  loans prepay for every one that defaults.

  Formally, 1 - S(t) estimates the probability of default in a hypothetical
  world where prepayment has been abolished and prepaid loans remain exposed
  forever. That is a coherent quantity (a "net" or "cause-removed" risk), but
  it is not the probability anyone provisioning against actually wants, which
  is the CRUDE probability: the chance this loan defaults before anything
  else happens to it. That is the cumulative incidence function (CIF), and it
  is always smaller.

  The inequality CIF_d(t) <= 1 - S_d(t) is an identity, not an empirical
  finding: both accumulate the same cause-specific hazard increments, but the
  CIF weights each increment by the probability of having survived BOTH
  risks, while 1 - S_d weights it by survival from default alone, which is
  always larger. A run in which the naive figure comes out below the CIF is a
  bug, and is asserted as such in the tests.

Two specifications, deliberately independent
---------------------------------------------
  (a) CAUSE-SPECIFIC COX (continuous time, lifelines).
      Two Cox models — one for default with prepayment censored, one for
      prepayment with default censored — combined into the CIF:

          S(t)     = exp( -( H_d(t) + H_p(t) ) )
          CIF_d(t) = sum over event times  dH_d(t_k) * S(t_k - 1)

      Note that overall survival uses BOTH cumulative hazards. Using only
      H_d here is the single most common way this calculation is got wrong,
      and it silently reproduces the naive answer.

  (b) DISCRETE-TIME MULTINOMIAL (challenger, on the loan-month panel).
      One row per loan-month at risk, the terminating row carrying the
      outcome; a 3-class multinomial logit over {survive, default, prepay}:

          S(k)     = prod_j ( 1 - h_d(j) - h_p(j) )
          CIF_d(t) = sum_k  h_d(k) * S(k-1)

      The shared softmax denominator is the point: it guarantees
      h_d + h_p < 1 by construction. Two independent binary logits do not —
      each is free to predict 0.7, and the implied "probability of surviving
      this month" then goes negative, which corrupts every subsequent term
      of the survival product without raising anything.

      Standard errors are clustered by loan_seq_num: consecutive months of
      the same loan are not independent observations, and unclustered SEs on
      a loan-month panel are optimistic by roughly sqrt(months per loan).

  The two agree only if the continuous and discrete formulations, the
  functional forms, and the baseline-hazard treatments are all approximately
  consistent. They are fitted from different data shapes (loan-level vs
  loan-month) and different likelihoods, so agreement is evidence; the
  comparison table reports the gap rather than averaging it away.

Why the multinomial needs an explicit duration term
-----------------------------------------------------
  A Cox model carries a non-parametric baseline hazard. A logistic
  regression has none — without a duration term it asserts that the monthly
  hazard is flat over the life of the loan, which for a mortgage is badly
  wrong in both directions (default peaks around years 3-5; prepayment is
  near zero in the first months, then rises sharply). Months-since-
  origination therefore enters as a binned factor
  (config.CR_DURATION_BIN_EDGES), which is the discrete-time analogue of the
  Cox baseline.

Inputs
------
  data/processed/surv_train.parquet     (01_data_preprocessing.py)
  data/processed/surv_oos.parquet
  data/processed/surv_oot.parquet

Outputs
-------
  data/processed/competing_risks_comparison.csv     — the headline table
  data/processed/competing_risks_cif_curves.csv     — full curves for plotting
  data/processed/competing_risks_cox_coefficients.csv
  data/processed/competing_risks_multinomial_coefficients.csv
  data/processed/competing_risks_expected_life.csv  — consumed by Ch.6
  data/figures/competing_risks_cif.png
  data/figures/competing_risks_hazards.png

Deliberately NOT written: survival_pd_horizons.csv. Chapter 5 stays as the
naive baseline this script is measured against.
=============================================================================
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in [REPO_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import config

warnings.filterwarnings("ignore")

try:
    from lifelines import CoxPHFitter
except ImportError:  # pragma: no cover - environment-dependent
    print("ERROR: lifelines not installed.  Run: pip install lifelines>=0.27.0")
    sys.exit(1)

# =============================================================================
# LOGGING
# =============================================================================

config.configure_logging("competing_risks.log")
log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = config.PROC_DIR
OUT_DIR  = config.OUT_DIR
FIG_DIR  = config.FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = config.SEED

EVENT_TYPE = config.EVENT_TYPE_COL
DURATION   = config.DURATION_COL
E_CENSORED = config.EVENT_CENSORED
E_DEFAULT  = config.EVENT_DEFAULT
E_PREPAY   = config.EVENT_PREPAY

HORIZONS        = config.CR_HORIZONS_MONTHS
DURATION_BINS   = config.CR_DURATION_BIN_EDGES
MAX_PANEL_LOANS = config.CR_MAX_PANEL_LOANS

PLT_STYLE: dict = config.PLT_STYLE

# Loan-level covariates for the cause-specific Cox models. Numeric only —
# lifelines requires a numeric design matrix, and one-hot expanding
# property_state across a small event count is a recipe for separation.
COX_COVARIATES = [
    "credit_score", "orig_cltv", "orig_dti", "orig_interest_rate",
    "log_orig_upb", "refi_incentive",
]

# Additional covariates for the discrete-time multinomial. The duration bin
# dummies are added separately and are not listed here.
MULTINOMIAL_COVARIATES = COX_COVARIATES


# =============================================================================
# PURE CORE — CUMULATIVE INCIDENCE
# =============================================================================

def cif_from_cumulative_hazards(cumhaz_default: np.ndarray,
                                cumhaz_prepay: np.ndarray,
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cumulative incidence functions from two cause-specific cumulative hazards.

    Given H_d(t) and H_p(t) evaluated on a shared, increasing time grid:

        S(t_k)     = exp( -( H_d(t_k) + H_p(t_k) ) )        overall survival
        dH_c(t_k)  = H_c(t_k) - H_c(t_{k-1})                hazard increment
        CIF_c(t)   = sum_{t_k <= t}  dH_c(t_k) * S(t_{k-1})

    The S(t_{k-1}) factor — survival from BOTH causes up to just before t_k —
    is what makes this a crude probability rather than a net one. Replacing
    it with exp(-H_d(t_{k-1})) reproduces the naive 1 - S_d(t) figure.

    Returns (CIF_default, CIF_prepay, S_overall), each aligned to the input
    grid. Both CIFs are monotonically non-decreasing and their sum is bounded
    by 1 - S(t) <= 1, since the increments are a partition of the total
    probability of having left the at-risk state.

    Pure and array-only, so the identities above are testable without
    fitting anything.
    """
    h_d = np.asarray(cumhaz_default, dtype=float)
    h_p = np.asarray(cumhaz_prepay, dtype=float)
    if h_d.shape != h_p.shape:
        raise ValueError(f"cumulative hazards must align: {h_d.shape} vs {h_p.shape}")
    if h_d.ndim != 1:
        raise ValueError("cumulative hazards must be 1-dimensional")

    # Cause-specific cumulative hazards are non-decreasing by construction;
    # a numerically negative increment would silently subtract incidence.
    d_default = np.diff(h_d, prepend=0.0).clip(min=0.0)
    d_prepay  = np.diff(h_p, prepend=0.0).clip(min=0.0)
    d_total = d_default + d_prepay

    survival = np.exp(-(h_d + h_p))
    # S at t_{k-1}: survival lagged one grid point, with S(t_0^-) = 1.
    survival_prev = np.concatenate([[1.0], survival[:-1]])

    # Each interval's total exit probability, split between the two causes in
    # proportion to their hazard increments.
    #
    # Writing this as the literal sum  dH_c(t_k) * S(t_{k-1})  is the usual
    # textbook form and agrees with what follows to O(dH^2), but it is a
    # left-endpoint Riemann sum: it evaluates a falling integrand at the
    # start of each interval and so overestimates. On a fine monthly grid
    # with mortgage-sized hazards the error is invisible, but it is not
    # bounded — with coarse grids or heavy hazards the two CIFs can sum to
    # more than 1, which is not a rounding artefact but a statement that
    # more than the whole cohort has left.
    #
    # Using the interval's EXACT exit probability, S(t_{k-1}) - S(t_k),
    # makes the partition identity CIF_d + CIF_p + S = 1 hold exactly by
    # telescoping, for any grid. The allocation between causes is the
    # standard proportional rule, exact when the two hazards are
    # proportional within the interval and second-order accurate otherwise.
    exit_prob = survival_prev - survival
    with np.errstate(divide="ignore", invalid="ignore"):
        share_default = np.where(d_total > 0, d_default / d_total, 0.0)

    cif_default = np.cumsum(exit_prob * share_default)
    cif_prepay  = np.cumsum(exit_prob * (1.0 - share_default))
    return cif_default, cif_prepay, survival


def naive_one_minus_survival(cumhaz_default: np.ndarray) -> np.ndarray:
    """
    The Chapter 5 conversion: 1 - S_d(t), prepayment treated as censoring.

    Uses ONLY the default cumulative hazard — exactly the assumption under
    scrutiny, that a loan leaving via prepayment stays at risk of default
    indefinitely.

    This is the exact closed form, and cif_from_cumulative_hazards() uses
    the exact interval exit probability too, so the two sides are directly
    comparable with no quadrature error between them. That matters: an
    earlier version compared a left-endpoint Riemann sum (the CIF) against
    this closed form, and the resulting O(dH^2) mismatch put the "naive"
    figure ~9e-7 BELOW the CIF at months 6-8, where the prepayment hazard is
    still near zero — tripping the naive >= CIF check for an entirely
    numerical reason and masking whether the real inequality held.

    With both sides exact, the inequality is exact and interval-by-interval:
    the naive interval probability is exp(-H_d(t_{k-1})) - exp(-H_d(t_k)),
    the CIF's is [exp(-H(t_{k-1})) - exp(-H(t_k))] * dH_d/dH, and the former
    dominates because exp(-H_d) >= exp(-(H_d + H_p)) whenever H_p >= 0. The
    remaining gap is entirely the competing risk, which is the quantity this
    script exists to measure.
    """
    return 1.0 - np.exp(-np.asarray(cumhaz_default, dtype=float))


def discrete_cif(hazard_default: np.ndarray,
                 hazard_prepay: np.ndarray,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cumulative incidence from discrete-time per-period hazards.

        S(k)     = prod_{j<=k} ( 1 - h_d(j) - h_p(j) )
        CIF_d(t) = sum_{k<=t}  h_d(k) * S(k-1)

    h_d and h_p are per-period (monthly) conditional probabilities on a
    1-indexed grid. Their sum must stay below 1 — guaranteed when they come
    from a shared softmax, which is the reason the challenger is a
    multinomial rather than two binary logits.

    Returns (CIF_default, CIF_prepay, S), aligned to the input grid.
    """
    h_d = np.asarray(hazard_default, dtype=float)
    h_p = np.asarray(hazard_prepay, dtype=float)
    if h_d.shape != h_p.shape:
        raise ValueError(f"hazards must align: {h_d.shape} vs {h_p.shape}")

    total = h_d + h_p
    if np.any(total >= 1.0):
        raise ValueError(
            "h_default + h_prepay >= 1 for at least one period — the implied "
            "per-period survival is non-positive. Two independent binary "
            "models produce this; a shared softmax cannot."
        )
    if np.any(h_d < 0) or np.any(h_p < 0):
        raise ValueError("hazards must be non-negative")

    survival = np.cumprod(1.0 - total)
    survival_prev = np.concatenate([[1.0], survival[:-1]])

    return (np.cumsum(h_d * survival_prev),
            np.cumsum(h_p * survival_prev),
            survival)


def expected_life_months(survival: np.ndarray) -> float:
    """
    Expected time to exit (either cause), as the area under the survival
    curve: E[T] = sum_k S(k) on a monthly grid.

    This is the IFRS 9 "expected life" — how long the loan is actually
    expected to remain on the balance sheet, as opposed to its contractual
    remaining term. For a 30-year mortgage the two differ by decades, because
    most loans refinance long before maturity.
    """
    return float(np.sum(np.asarray(survival, dtype=float)))


def interpolate_at(times: np.ndarray, values: np.ndarray,
                   targets: list[int] | np.ndarray) -> np.ndarray:
    """
    Step-function lookup: value at the largest grid time <= each target.

    Right-continuous step interpolation, not linear — a cumulative incidence
    curve is a step function and linearly interpolating between event times
    invents incidence that did not occur.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.empty(len(targets), dtype=float)
    for i, t in enumerate(targets):
        prior = np.searchsorted(times, t, side="right") - 1
        out[i] = values[prior] if prior >= 0 else 0.0
    return out


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_survival_panel(path, extra_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Load a surv_* panel with only the columns this script actually uses, and
    in compact dtypes.

    The panel is a loan-MONTH table: tens of millions of rows on the full
    dataset, at roughly 600 bytes/row if read naively. Reading all 33 columns
    as float64/object is an easy 15-20 GB and will take the kernel out on a
    30 GB box. Two cheap defences:

      * Selective column projection — parquet is columnar, so the unread
        columns are never touched on disk or in RAM. This script needs about
        a dozen of the 33.
      * Category dtype for the low-cardinality strings, and for
        loan_seq_num. Strings round-trip out of parquet as Python objects at
        ~50-60 bytes per row each; as categories they become small integer
        codes plus one shared dictionary. loan_seq_num matters most — it is
        one distinct value per loan but repeats once per month.
    """
    import pyarrow.parquet as pq

    wanted = (["loan_seq_num", "orig_date", "period_month", "period_event",
               EVENT_TYPE, DURATION, "remaining_months"]
              + list(COX_COVARIATES) + list(extra_columns or []))
    # log_orig_upb is derived downstream, not stored.
    wanted = [c for c in dict.fromkeys(wanted) if c != "log_orig_upb"]
    wanted.append("orig_upb")

    available = set(pq.read_schema(path).names)
    columns = [c for c in dict.fromkeys(wanted) if c in available]
    missing = [c for c in wanted if c not in available]
    if missing:
        log.warning("  %s: columns not present and skipped: %s", path.name, missing)

    panel = pd.read_parquet(path, columns=columns)

    for col in ("loan_seq_num",):
        if col in panel.columns:
            panel[col] = panel[col].astype("category")
    for col in panel.select_dtypes("float64").columns:
        panel[col] = panel[col].astype(np.float32)

    log.info("  %s: %s rows x %d columns (%.1f MB in RAM)", path.name,
             f"{len(panel):,}", panel.shape[1],
             panel.memory_usage(deep=True).sum() / 1e6)
    return panel


def build_loan_level(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the loan-month panel to one row per loan for the Cox models.

    Keeps the FIRST row of each loan, so covariates are valued at
    origination. This is deliberate: a Cox model with time-invariant
    covariates must not read them off the last row, which is the
    end-of-follow-up leakage already documented for
    06_survival_analysis.py. duration_months and event_type are loan-level
    constants, so they survive the collapse unchanged.
    """
    if panel.empty:
        return panel.copy()

    ordered = panel.sort_values(["loan_seq_num", "period_month"])
    # observed=True: loan_seq_num is a category with one level per loan, and
    # a non-observed groupby would try to build the full cartesian product.
    loans = ordered.groupby("loan_seq_num", as_index=False, observed=True).first()

    if "orig_upb" in loans.columns:
        loans["log_orig_upb"] = np.log(loans["orig_upb"].clip(lower=1.0))
    return loans


def available_covariates(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Covariates present with at least some non-null, non-constant values."""
    out = []
    for col in wanted:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            log.warning("  Covariate %s is entirely missing — dropped.", col)
            continue
        if values.nunique(dropna=True) <= 1:
            log.warning("  Covariate %s is constant — dropped.", col)
            continue
        out.append(col)
    return out


def duration_bin_labels(months: pd.Series,
                        edges: list[int] = DURATION_BINS) -> pd.Series:
    """
    Bin months-since-origination into the discrete-time baseline hazard.

    The multinomial logit has no baseline hazard of its own, so this factor
    is what lets the monthly hazard vary with seasoning instead of being
    assumed flat across the life of the loan.
    """
    return pd.cut(months, bins=edges, right=True, include_lowest=True)


def collapse_sparse_bins(months: pd.Series, events: pd.Series,
                         edges: list[int] = DURATION_BINS,
                         min_events: int = config.CR_MIN_BIN_EVENTS,
                         ) -> list[int]:
    """
    Merge adjacent duration bins until each carries at least `min_events` of
    every modelled cause, and return the surviving edges.

    A bin in which no loan ever defaults produces a dummy that perfectly
    predicts "not a default" for those rows. The multinomial's design matrix
    then becomes singular and the Newton fit dies — which on the sample
    dataset silently cost the clustered standard errors that are the whole
    reason for using statsmodels here.

    Merging trades baseline-hazard resolution for identifiability, and only
    where the data cannot support the finer grid. The default schedule is
    left untouched wherever events are plentiful.
    """
    edges = sorted(set(edges))
    causes = [E_DEFAULT, E_PREPAY]

    changed = True
    while changed and len(edges) > 2:
        changed = False
        bins = pd.cut(months, bins=edges, right=True, include_lowest=True)
        counts = pd.DataFrame({"bin": bins, "event": events})
        per_bin = {
            cause: counts[counts["event"] == cause]["bin"].value_counts()
            for cause in causes
        }
        categories = list(bins.cat.categories)

        for idx, cat in enumerate(categories):
            sparse = any(int(per_bin[c].get(cat, 0)) < min_events for c in causes)
            if not sparse:
                continue
            # Drop the edge that dissolves this bin into a neighbour: the
            # right edge normally, the left edge for the final bin.
            drop = edges[idx + 1] if idx + 1 < len(edges) - 1 else edges[idx]
            if drop in (edges[0],):
                continue
            edges.remove(drop)
            changed = True
            break

    return edges


# =============================================================================
# SPECIFICATION (a) — CAUSE-SPECIFIC COX
# =============================================================================

def fit_cause_specific_cox(loans: pd.DataFrame, covariates: list[str],
                           cause: int, label: str,
                           penalizer: float = 0.1) -> CoxPHFitter:
    """
    One cause-specific Cox model: the named cause is the event, every other
    exit (including the competing event) is censored at its observed time.

    Censoring the competing event here is correct and is NOT the error this
    script exists to fix. The cause-specific hazard is a well-defined
    quantity estimated exactly this way; the error is converting it to a
    probability with 1 - S, which ignores that the competing event removes
    loans from the at-risk set. The combination step downstream is where
    prepayment is properly accounted for.
    """
    frame = loans[[DURATION, EVENT_TYPE] + covariates].dropna().copy()
    frame["event"] = (frame[EVENT_TYPE] == cause).astype(int)
    frame = frame.drop(columns=[EVENT_TYPE])

    n_events = int(frame["event"].sum())
    log.info("  [%s] %s loans, %s events (%.2f%%)", label, f"{len(frame):,}",
             f"{n_events:,}", 100.0 * n_events / max(len(frame), 1))
    if n_events < 10:
        log.warning("  [%s] fewer than 10 events — coefficients are unreliable.",
                    label)

    cox = CoxPHFitter(penalizer=penalizer)
    cox.fit(frame, duration_col=DURATION, event_col="event")
    return cox


def cox_cumulative_hazards(cox: CoxPHFitter, loans: pd.DataFrame,
                           covariates: list[str],
                           times: np.ndarray) -> np.ndarray:
    """
    Per-loan cumulative hazard matrix (n_loans, n_times).

        H_i(t) = H_0(t) * exp(x_i'beta)

    Returned per loan rather than at the mean covariate vector, because the
    CIF is non-linear in the hazards: averaging covariates first and then
    computing one CIF is not the same as computing each loan's CIF and
    averaging, and the portfolio quantity wanted downstream is the latter.
    """
    baseline = cox.baseline_cumulative_hazard_
    base_times = baseline.index.to_numpy(dtype=float)
    base_values = baseline.iloc[:, 0].to_numpy(dtype=float)

    grid = interpolate_at(base_times, base_values, times)
    partial = cox.predict_partial_hazard(loans[covariates]).to_numpy(dtype=float)

    return np.outer(partial, grid)


def portfolio_cif_cox(cox_default: CoxPHFitter, cox_prepay: CoxPHFitter,
                      loans: pd.DataFrame, covariates: list[str],
                      times: np.ndarray) -> dict[str, np.ndarray]:
    """
    Portfolio-average CIFs, naive comparator and survival from the two
    cause-specific Cox fits.

    Each loan's CIF is computed from its own hazards and the results are
    averaged, which is the marginal cumulative incidence over the scored
    population.
    """
    frame = loans[covariates].dropna()
    kept = loans.loc[frame.index]
    log.info("  Combining CIFs over %s loans on a %d-month grid …",
             f"{len(kept):,}", len(times))

    h_default = cox_cumulative_hazards(cox_default, kept, covariates, times)
    h_prepay  = cox_cumulative_hazards(cox_prepay,  kept, covariates, times)

    n = len(kept)
    cif_d = np.zeros((n, len(times)))
    cif_p = np.zeros((n, len(times)))
    surv  = np.zeros((n, len(times)))
    naive = np.zeros((n, len(times)))

    for i in range(n):
        cif_d[i], cif_p[i], surv[i] = cif_from_cumulative_hazards(
            h_default[i], h_prepay[i])
        naive[i] = naive_one_minus_survival(h_default[i])

    return {
        "times": times,
        "cif_default": cif_d.mean(axis=0),
        "cif_prepay":  cif_p.mean(axis=0),
        "survival":    surv.mean(axis=0),
        "naive":       naive.mean(axis=0),
        "per_loan_survival": surv,
        "loan_ids": kept["loan_seq_num"].to_numpy(),
    }


# =============================================================================
# SPECIFICATION (b) — DISCRETE-TIME MULTINOMIAL
# =============================================================================

def build_discrete_panel(panel: pd.DataFrame, covariates: list[str],
                         max_loans: int = MAX_PANEL_LOANS,
                         seed: int = SEED) -> pd.DataFrame:
    """
    Loan-month rows for the multinomial fit, sampled by LOAN.

    Sampling whole loans rather than individual rows is essential: dropping
    random months out of a loan's history would silently delete the periods
    it survived and leave the terminating row, inflating every hazard.
    """
    loan_ids = panel["loan_seq_num"].drop_duplicates()
    if len(loan_ids) > max_loans:
        keep = loan_ids.sample(n=max_loans, random_state=seed)
        panel = panel[panel["loan_seq_num"].isin(set(keep))]
        log.info("  Sampled %s of %s loans for the multinomial panel.",
                 f"{max_loans:,}", f"{len(loan_ids):,}")

    frame = panel[["loan_seq_num", "period_month", "period_event"] + covariates].copy()
    frame = frame.dropna(subset=["period_month", "period_event"])
    for col in covariates:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=covariates)

    edges = collapse_sparse_bins(frame["period_month"], frame["period_event"])
    if edges != sorted(set(DURATION_BINS)):
        log.info("  Duration bins collapsed for identifiability: %d -> %d edges %s",
                 len(set(DURATION_BINS)), len(edges), edges)
    frame["duration_bin"] = duration_bin_labels(frame["period_month"], edges)
    frame.attrs["duration_edges"] = edges
    return frame


def fit_multinomial_hazard(frame: pd.DataFrame, covariates: list[str]):
    """
    3-class multinomial logit over {survive, default, prepay} on loan-months,
    with standard errors clustered by loan_seq_num.

    The shared softmax denominator guarantees h_d + h_p < 1. Clustered SEs
    are required because consecutive months of one loan are not independent
    draws — a loan contributing 60 rows contributes far less than 60 rows'
    worth of information, and unclustered SEs would overstate precision by
    roughly the square root of the average months per loan.

    Returns (result, design_columns, used_statsmodels). Falls back to
    sklearn (point estimates only, no inference) if the statsmodels fit does
    not converge, logging the downgrade rather than failing.
    """
    import statsmodels.api as sm

    dummies = pd.get_dummies(frame["duration_bin"], prefix="dur",
                             drop_first=True, dtype=float)
    design = pd.concat([frame[covariates].astype(float), dummies], axis=1)
    design = sm.add_constant(design, has_constant="add")
    y = frame["period_event"].astype(int).to_numpy()

    log.info("  Multinomial design: %s rows x %d columns, %s loans",
             f"{len(design):,}", design.shape[1],
             f"{frame['loan_seq_num'].nunique():,}")
    counts = pd.Series(y).value_counts().sort_index()
    log.info("  Outcome counts per loan-month: %s",
             ", ".join(f"{config.EVENT_TYPE_LABELS[int(k)]}={v:,}"
                       for k, v in counts.items()))

    # A constant column (e.g. a dummy no row activates after sampling) makes
    # the design rank-deficient and the Newton step singular.
    constant = [c for c in design.columns
                if c != "const" and design[c].nunique(dropna=False) <= 1]
    if constant:
        log.info("  Dropping %d constant design column(s): %s",
                 len(constant), constant)
        design = design.drop(columns=constant)

    groups = frame["loan_seq_num"].to_numpy()
    for method in ("newton", "bfgs"):
        try:
            result = sm.MNLogit(y, design).fit(
                method=method, maxiter=500, disp=False,
                cov_type="cluster", cov_kwds={"groups": groups},
            )
            if not np.all(np.isfinite(np.asarray(result.bse))):
                raise ValueError("non-finite clustered standard errors")
            log.info("  Multinomial converged via %s (clustered SEs on %s loans).",
                     method, f"{frame['loan_seq_num'].nunique():,}")
            return result, list(design.columns), True
        except Exception as exc:
            log.warning("  MNLogit (%s) failed: %s", method, exc)

    log.warning("  All statsmodels attempts failed — falling back to sklearn "
                "for point estimates. CLUSTERED STANDARD ERRORS ARE NOT "
                "AVAILABLE in this run; the coefficient CSV will be skipped.")

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(design.to_numpy(), y)
    return clf, list(design.columns), False


def multinomial_hazard_paths(result, design_columns: list[str],
                             loans: pd.DataFrame, covariates: list[str],
                             times: np.ndarray, used_statsmodels: bool,
                             edges: list[int] = DURATION_BINS,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-loan monthly hazard paths (n_loans, n_times) for default and prepay.

    Each loan's covariates are held at their origination values while
    months-since-origination advances — the same flat-covariate projection
    convention used elsewhere in this pipeline. The duration bin is the only
    thing that moves, which is precisely the baseline hazard doing its job.
    """
    n_loans, n_times = len(loans), len(times)
    bins = duration_bin_labels(pd.Series(times), edges)
    bin_dummies = pd.get_dummies(bins, prefix="dur", drop_first=True, dtype=float)

    base = loans[covariates].astype(float).reset_index(drop=True)
    h_default = np.zeros((n_loans, n_times))
    h_prepay  = np.zeros((n_loans, n_times))

    for t_idx in range(n_times):
        block = base.copy()
        block["const"] = 1.0
        for col in design_columns:
            if col.startswith("dur_"):
                value = (bin_dummies.iloc[t_idx][col]
                         if col in bin_dummies.columns else 0.0)
                block[col] = float(value)
        block = block.reindex(columns=design_columns, fill_value=0.0)

        if used_statsmodels:
            probs = result.predict(exog=block.to_numpy())
        else:
            probs = result.predict_proba(block.to_numpy())
        probs = np.asarray(probs, dtype=float)

        h_default[:, t_idx] = probs[:, E_DEFAULT]
        h_prepay[:, t_idx]  = probs[:, E_PREPAY]

    return h_default, h_prepay


def portfolio_cif_multinomial(h_default: np.ndarray, h_prepay: np.ndarray,
                              ) -> dict[str, np.ndarray]:
    """Average the per-loan discrete CIFs into portfolio curves."""
    n = h_default.shape[0]
    cif_d = np.zeros_like(h_default)
    cif_p = np.zeros_like(h_prepay)
    surv  = np.zeros_like(h_default)

    for i in range(n):
        cif_d[i], cif_p[i], surv[i] = discrete_cif(h_default[i], h_prepay[i])

    # The naive discrete comparator ignores the prepayment hazard entirely,
    # exactly as 1 - S_d does in continuous time.
    naive = 1.0 - np.cumprod(1.0 - h_default, axis=1)

    return {
        "cif_default": cif_d.mean(axis=0),
        "cif_prepay":  cif_p.mean(axis=0),
        "survival":    surv.mean(axis=0),
        "naive":       naive.mean(axis=0),
        "per_loan_survival": surv,
    }


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def build_comparison(times: np.ndarray, cox: dict, multinomial: dict,
                     horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """
    The headline table: one row per horizon, naive vs both CIF
    specifications, and the overstatement the naive conversion carries.

    overstatement_pct is measured against the Cox CIF (specification (a),
    the primary) and expressed as a percentage OF the CIF:

        (naive - cif_cox) / cif_cox * 100

    so "+45%" reads as "the naive figure is 45% larger than the correct
    one", which is the number that matters when it is feeding a provision.
    """
    labels = [f"{h}m" for h in horizons] + ["lifetime"]
    points = list(horizons) + [int(times[-1])]

    naive_v = interpolate_at(times, cox["naive"], points)
    cox_v   = interpolate_at(times, cox["cif_default"], points)
    mn_v    = interpolate_at(times, multinomial["cif_default"], points)

    with np.errstate(divide="ignore", invalid="ignore"):
        overstatement = np.where(cox_v > 0, (naive_v - cox_v) / cox_v * 100.0, np.nan)
        spec_gap = np.where(cox_v > 0, (mn_v - cox_v) / cox_v * 100.0, np.nan)

    return pd.DataFrame({
        "horizon": labels,
        "horizon_months": points,
        "naive_1_minus_S": np.round(naive_v, 6),
        "cif_cox": np.round(cox_v, 6),
        "cif_multinomial": np.round(mn_v, 6),
        "overstatement_pct": np.round(overstatement, 2),
        "spec_gap_pct": np.round(spec_gap, 2),
    })


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_cif(times: np.ndarray, cox: dict, multinomial: dict) -> None:
    """Naive 1 - S(t) against both CIF specifications."""
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Competing Risks — Cumulative Incidence of Default vs the Naive 1 - S(t)\n"
        "The gap is the default probability attributed to loans that had "
        "already prepaid",
        fontsize=12, fontweight="bold", color="white",
    )

    ax = axes[0]
    ax.plot(times, cox["naive"] * 100, color="#EF4444", linewidth=2.4,
            label="Naive  1 - S(t)  (prepayment censored)")
    ax.plot(times, cox["cif_default"] * 100, color="#38BDF8", linewidth=2.4,
            label="CIF default — cause-specific Cox")
    ax.plot(times, multinomial["cif_default"] * 100, color="#F59E0B",
            linewidth=2.0, linestyle="--", label="CIF default — multinomial")
    ax.fill_between(times, cox["cif_default"] * 100, cox["naive"] * 100,
                    color="#EF4444", alpha=0.12, label="Overstatement")
    ax.set_xlabel("Months since origination", fontsize=10)
    ax.set_ylabel("Cumulative default probability (%)", fontsize=10)
    ax.set_title("Default", color="#CBD5E1", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")

    ax = axes[1]
    ax.plot(times, cox["cif_default"] * 100, color="#38BDF8", linewidth=2.4,
            label="CIF default")
    ax.plot(times, cox["cif_prepay"] * 100, color="#10B981", linewidth=2.4,
            label="CIF prepayment")
    ax.plot(times, cox["survival"] * 100, color="#94A3B8", linewidth=2.0,
            linestyle=":", label="Still alive  S(t)")
    total = (cox["cif_default"] + cox["cif_prepay"] + cox["survival"]) * 100
    ax.plot(times, total, color="#E2E8F0", linewidth=1.0, alpha=0.6,
            label="Sum (must be 100%)")
    ax.set_xlabel("Months since origination", fontsize=10)
    ax.set_ylabel("Share of the original cohort (%)", fontsize=10)
    ax.set_title("Where the cohort goes", color="#CBD5E1", fontsize=10)
    ax.legend(fontsize=8.5, loc="center right")

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "competing_risks_cif.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_hazards(times: np.ndarray, h_default: np.ndarray,
                 h_prepay: np.ndarray) -> None:
    """Mean monthly cause-specific hazards from the multinomial fit."""
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(times, h_default.mean(axis=0) * 100, color="#38BDF8",
            linewidth=2.2, label="Default hazard")
    ax.plot(times, h_prepay.mean(axis=0) * 100, color="#10B981",
            linewidth=2.2, label="Prepayment hazard")
    ax.set_xlabel("Months since origination", fontsize=10)
    ax.set_ylabel("Monthly hazard (%)", fontsize=10)
    ax.set_title("Discrete-Time Cause-Specific Hazards (multinomial)\n"
                 "Binned months-since-origination is the baseline a logit "
                 "otherwise lacks",
                 fontsize=11, fontweight="bold", color="white", pad=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = FIG_DIR / "competing_risks_hazards.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 70)
    log.info("Mortgage Credit Risk  |  Ch.10 — Competing Risks (Prepayment)")
    log.info("=" * 70)

    # ── [1/6] Load ───────────────────────────────────────────────────────
    log.info("")
    log.info("[1/6] Loading the competing-risks panel …")
    train_path = OUT_DIR / config.SURV_TRAIN_FILE
    if not train_path.exists():
        log.error("  %s not found. Re-run 01_data_preprocessing.py — it now "
                  "emits the surv_* variant alongside pd_*.", train_path)
        return

    train = load_survival_panel(train_path)
    log.info("  Train panel: %s loan-months, %s loans",
             f"{len(train):,}", f"{train['loan_seq_num'].nunique():,}")

    loans = build_loan_level(train)
    counts = loans[EVENT_TYPE].value_counts().sort_index()
    for code, label in config.EVENT_TYPE_LABELS.items():
        n = int(counts.get(code, 0))
        log.info("    %-9s %s loans (%.2f%%)", label, f"{n:,}",
                 100.0 * n / max(len(loans), 1))
    n_prepay = int((loans[EVENT_TYPE] == E_PREPAY).sum())
    n_default = int((loans[EVENT_TYPE] == E_DEFAULT).sum())
    if n_default:
        log.info("    prepay:default = %.1f:1 — the competing risk 1 - S(t) ignores",
                 n_prepay / n_default)

    covariates = available_covariates(loans, COX_COVARIATES)
    log.info("  Covariates: %s", covariates)

    max_month = int(loans[DURATION].max())
    times = np.arange(1, max_month + 1)
    log.info("  Time grid: 1 … %d months", max_month)

    # ── [2/6] Cause-specific Cox ─────────────────────────────────────────
    log.info("")
    log.info("[2/6] Specification (a): cause-specific Cox models …")
    cox_default = fit_cause_specific_cox(loans, covariates, E_DEFAULT, "default")
    cox_prepay  = fit_cause_specific_cox(loans, covariates, E_PREPAY,  "prepay")

    coef_frames = []
    for name, model in [("default", cox_default), ("prepay", cox_prepay)]:
        summary = model.summary.copy()
        summary.index.name = "covariate"
        summary = summary.reset_index()
        summary.insert(0, "cause", name)
        coef_frames.append(summary)
    pd.concat(coef_frames, ignore_index=True).to_csv(
        OUT_DIR / config.CR_COX_COEFS_FILE, index=False)
    log.info("  Coefficients → %s", config.CR_COX_COEFS_FILE)

    log.info("")
    log.info("[3/6] Combining the cause-specific hazards into CIFs …")
    cox_curves = portfolio_cif_cox(cox_default, cox_prepay, loans, covariates, times)

    # The core identity. A violation is a bug, not a finding.
    violation = cox_curves["naive"] < cox_curves["cif_default"] - 1e-9
    if violation.any():
        log.error("  naive 1 - S(t) fell BELOW the CIF at %d of %d grid points — "
                  "this is impossible and indicates a bug in the combination step.",
                  int(violation.sum()), len(times))
    else:
        log.info("  Check passed: naive 1 - S(t) >= CIF at all %d grid points.",
                 len(times))

    # ── [4/6] Discrete-time multinomial ──────────────────────────────────
    log.info("")
    log.info("[4/6] Specification (b): discrete-time multinomial …")
    panel_covariates = available_covariates(train, MULTINOMIAL_COVARIATES)
    if "log_orig_upb" in panel_covariates and "log_orig_upb" not in train.columns:
        panel_covariates.remove("log_orig_upb")
    panel = train.copy()
    if "orig_upb" in panel.columns:
        panel["log_orig_upb"] = np.log(panel["orig_upb"].clip(lower=1.0))
    panel_covariates = available_covariates(panel, MULTINOMIAL_COVARIATES)

    discrete = build_discrete_panel(panel, panel_covariates)
    result, design_columns, used_sm = fit_multinomial_hazard(discrete, panel_covariates)

    if used_sm:
        params = result.params
        bse = result.bse
        rows = []
        # MNLogit drops the base outcome (0 = survive); remaining columns are
        # the log-odds of default and prepay against surviving that month.
        outcome_names = [config.EVENT_TYPE_LABELS[E_DEFAULT],
                         config.EVENT_TYPE_LABELS[E_PREPAY]]
        for j in range(params.shape[1]):
            for i, col in enumerate(design_columns):
                rows.append({
                    "outcome_vs_survive": outcome_names[j] if j < len(outcome_names) else f"class_{j+1}",
                    "covariate": col,
                    "coef": float(np.asarray(params)[i, j]),
                    "std_err_clustered": float(np.asarray(bse)[i, j]),
                })
        pd.DataFrame(rows).to_csv(
            OUT_DIR / config.CR_MULTINOMIAL_COEFS_FILE, index=False)
        log.info("  Coefficients (clustered SEs) → %s",
                 config.CR_MULTINOMIAL_COEFS_FILE)

    loans_for_paths = loans.dropna(subset=panel_covariates).reset_index(drop=True)
    h_default, h_prepay = multinomial_hazard_paths(
        result, design_columns, loans_for_paths, panel_covariates, times, used_sm,
        edges=discrete.attrs.get("duration_edges", DURATION_BINS))
    log.info("  Max h_default + h_prepay across all loan-months: %.4f  "
             "(softmax guarantees < 1)", float((h_default + h_prepay).max()))

    mn_curves = portfolio_cif_multinomial(h_default, h_prepay)

    # ── [5/6] Comparison ─────────────────────────────────────────────────
    log.info("")
    log.info("[5/6] Building the comparison table …")
    comparison = build_comparison(times, cox_curves, mn_curves)
    comparison.to_csv(OUT_DIR / config.CR_COMPARISON_FILE, index=False)
    log.info("\n%s", comparison.to_string(index=False))
    log.info("  → %s", config.CR_COMPARISON_FILE)

    at_36 = comparison[comparison["horizon"] == "36m"]
    if not at_36.empty:
        gap = float(at_36["spec_gap_pct"].iloc[0])
        if abs(gap) <= 5.0:
            log.info("  The two specifications agree at 36 months (gap %.1f%%).", gap)
        else:
            log.warning(
                "  The two specifications DISAGREE at 36 months (gap %.1f%%). "
                "Likely causes, in order of plausibility: (1) the discrete "
                "baseline bins are too coarse where the hazard moves fastest; "
                "(2) the multinomial holds covariates at origination values "
                "while the Cox partial hazard does the same but on a "
                "different functional form; (3) too few default events to "
                "identify both models comparably. Investigate before "
                "quoting either figure.", gap)

    curves = pd.DataFrame({
        "month": times,
        "naive_1_minus_S": cox_curves["naive"],
        "cif_default_cox": cox_curves["cif_default"],
        "cif_prepay_cox": cox_curves["cif_prepay"],
        "survival_cox": cox_curves["survival"],
        "cif_default_multinomial": mn_curves["cif_default"],
        "cif_prepay_multinomial": mn_curves["cif_prepay"],
        "survival_multinomial": mn_curves["survival"],
    })
    curves.to_csv(OUT_DIR / config.CR_CIF_CURVES_FILE, index=False)
    log.info("  Curves → %s", config.CR_CIF_CURVES_FILE)

    # ── [6/6] Expected life for Chapter 6 ────────────────────────────────
    log.info("")
    log.info("[6/6] Expected life (IFRS 9) …")

    # Scored across ALL splits, not just the training population. Chapter 6
    # scores pd_oos, so an expected-life file built from surv_train alone
    # covers only the loans that happen to appear in both — roughly two
    # thirds — and every unmatched loan silently falls back to its
    # contractual term, which is the behaviour being corrected. The Cox
    # models are fitted on train but can score any loan.
    life_frames = []
    for split_file in (config.SURV_TRAIN_FILE, config.SURV_OOS_FILE,
                       config.SURV_OOT_FILE):
        path = PROC_DIR / split_file
        if not path.exists():
            continue
        split_loans = build_loan_level(load_survival_panel(path))
        scored = split_loans.dropna(subset=covariates)
        if scored.empty:
            continue
        curves = portfolio_cif_cox(cox_default, cox_prepay, scored, covariates, times)
        life_frames.append(pd.DataFrame({
            "loan_seq_num": curves["loan_ids"],
            "expected_life_months": np.round(
                [expected_life_months(s) for s in curves["per_loan_survival"]], 2),
            "split": split_file.replace("surv_", "").replace(".parquet", ""),
        }))

    life = (pd.concat(life_frames, ignore_index=True).drop_duplicates("loan_seq_num")
            if life_frames else pd.DataFrame(columns=["loan_seq_num",
                                                      "expected_life_months", "split"]))
    per_loan_life = life["expected_life_months"].to_numpy(dtype=float)
    log.info("  Scored %s loans across %d split(s).", f"{len(life):,}", len(life_frames))
    life.to_csv(OUT_DIR / config.CR_EXPECTED_LIFE_FILE, index=False)
    log.info("  Expected life: mean=%.1f  median=%.1f  months (grid capped at %d)",
             per_loan_life.mean(), np.median(per_loan_life), max_month)
    if "remaining_months" in loans.columns:
        contractual = pd.to_numeric(loans["remaining_months"], errors="coerce").mean()
        log.info("  Mean CONTRACTUAL remaining term: %.1f months — Chapter 6 "
                 "amortizes over this unless config.IFRS9_USE_EXPECTED_LIFE.",
                 contractual)
    log.info("  → %s", config.CR_EXPECTED_LIFE_FILE)

    plot_cif(times, cox_curves, mn_curves)
    plot_hazards(times, h_default, h_prepay)

    log.info("")
    log.info("=" * 70)
    log.info("Ch.10 complete — competing-risks CIFs estimated under both specs.")
    log.info("  Chapter 5 (survival_pd_horizons.csv) is untouched and remains "
             "the naive baseline.")
    log.info("  Next: python 07_macro_scenario_analysis.py  (picks up expected life)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
