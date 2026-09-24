"""
=============================================================================
Mortgage Credit Risk Modelling  |  Data Engineering Pipeline
=============================================================================
Script  : 01_data_preprocessing.py
Purpose : Build model-ready PD and LGD datasets from raw Freddie Mac
          single-family loan performance files.

Memory Strategy
---------------
Loading all origination years simultaneously requires ~15 GB of RAM at the
merge step — exceeding the 30 GB Kaggle limit after pandas overhead.  Instead
the pipeline processes one origination year at a time:

    for year in 2000 … 2020:
        1. Load  sample_orig_YYYY.txt  (~50k rows,  ~15 MB in RAM)
        2. Load  sample_svcg_YYYY.txt  (~1–5M rows, ~200 MB in RAM)
        3. Clean → merge → engineer features
        4. Extract PD rows  → data/processed/chunks/pd_YYYY.parquet
        5. Extract LGD rows → data/processed/chunks/lgd_YYYY.parquet
        6. del merged; gc.collect()   # free RAM before next year

    Peak RAM ≈ 400 MB per year instead of ~15 GB for the full dataset.

Inputs
------
  data/raw/freddie_mac/sample_orig_YYYY.txt
  data/raw/freddie_mac/sample_svcg_YYYY.txt
  data/raw/macro/hpi_3digit_zip.csv          (optional — FHFA HPI)
  data/raw/macro/unemployment_rate.csv       (optional — BLS LNS14000000)

Outputs
-------
  data/processed/pd_train.parquet
  data/processed/pd_oos.parquet
  data/processed/pd_oot.parquet
  data/processed/lgd_train.parquet
  data/processed/lgd_oos.parquet
  data/processed/lgd_oot.parquet
  data/processed/pd_iv_summary.csv
  data/processed/pd_psi_summary.csv
=============================================================================
"""

from __future__ import annotations

import gc
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in [REPO_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import config

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING
# =============================================================================

config.configure_logging("preprocessing.log")
log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DIR   = config.RAW_DIR
MACRO_DIR = config.MACRO_DIR
OUT_DIR   = config.PROC_DIR
CHUNK_DIR = config.CHUNK_DIR

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2000
END_YEAR   = 2020

OOT_CUTOFF = config.OOT_CUTOFF
SEED       = config.SEED
OOS_FRAC   = config.OOS_FRAC


# =============================================================================
# COLUMN DEFINITIONS  (Freddie Mac data guide — 32 columns per file)
# =============================================================================

ORIG_COLS = [
    "credit_score", "first_payment_date", "first_time_homebuyer",
    "maturity_date", "msa", "mi_pct", "num_units", "occupancy_status",
    "orig_cltv", "orig_dti", "orig_upb", "orig_ltv", "orig_interest_rate",
    "channel", "ppm_flag", "amortization_type", "property_state",
    "property_type", "postal_code", "loan_seq_num", "loan_purpose",
    "orig_loan_term", "num_borrowers", "seller_name", "servicer_name",
    "super_conforming_flag", "pre_harp_seq_num", "program_indicator",
    "harp_indicator", "property_valuation_method", "io_indicator",
    "mi_cancellation_indicator",
]

SVCG_COLS = [
    "loan_seq_num", "monthly_reporting_period", "current_upb",
    "delinquency_status", "loan_age", "remaining_months",
    "defect_settlement_date", "modification_flag", "zero_balance_code",
    "zero_balance_date", "current_interest_rate", "current_deferred_upb",
    "ddlpi", "mi_recoveries", "net_sale_proceeds", "non_mi_recoveries",
    "expenses", "legal_costs", "maintenance_costs", "taxes_insurance",
    "misc_expenses", "actual_loss", "modification_cost",
    "step_modification_flag", "deferred_payment_plan", "eltv",
    "zero_balance_removal_upb", "delinquent_accrued_interest",
    "delinquency_due_to_disaster", "borrower_assistance_status",
    "current_month_modification_cost", "interest_bearing_upb",
]

# Servicer columns required by the pipeline (saves ~50% read time)
# Servicer columns required by the pipeline (saves ~50% read time).
# modification_flag is read for the competing-risks dataset: loan_age RESETS
# when a loan is modified, so the flag both explains the reset and is a
# legitimate covariate for the prepayment/default hazards. It is additive —
# pd_*/lgd_* chunks select their columns by explicit list in main().
SVCG_USECOLS = [
    "loan_seq_num", "monthly_reporting_period", "current_upb",
    "delinquency_status", "loan_age", "remaining_months", "zero_balance_code",
    "zero_balance_date", "current_interest_rate", "mi_recoveries",
    "net_sale_proceeds", "non_mi_recoveries", "expenses", "actual_loss",
    "zero_balance_removal_upb", "delinquent_accrued_interest",
    "interest_bearing_upb", "modification_flag",
]

# Origination columns propagated to each performance row after the merge
ORIG_KEEP = [
    "loan_seq_num", "orig_date", "zip3",
    "credit_score", "first_time_homebuyer", "mi_pct", "num_units",
    "occupancy_status", "orig_cltv", "orig_dti", "orig_upb",
    "orig_interest_rate", "channel", "property_state",
    "property_type", "loan_purpose", "num_borrowers",
]

# PD feature set — thesis Table 3 (Chapter 1)
PD_FEATURES = config.PD_FEATURES

# LGD feature set — thesis Chapter 3, §3.4
LGD_FEATURES = config.LGD_FEATURES

# Zero-balance codes treated as default events — see config.DEFAULT_CODES
# for the full rationale (01 = prepayment, 16/96 excluded as non-loss events).
DEFAULT_CODES = config.DEFAULT_CODES

_ZBC_LABELS = {
    "01": "prepayment",      "02": "3rd-party sale",
    "03": "short sale",      "06": "repurchase",
    "09": "REO",             "15": "note sale",
    "16": "reperforming",    "96": "non-standard",
}


# =============================================================================
# DATE PARSING HELPER
# =============================================================================

# Freddie Mac servicer files use "MM/YYYY" for monthly_reporting_period and
# zero_balance_date.  However the exact format can vary across dataset
# vintages (e.g. some exports use "YYYYMM" without a separator).  This helper
# tries each known format in sequence so the pipeline is robust to both.
#
# FIX: The original code used format="%m/%Y" directly inside clean_perf(),
# which is correct for "MM/YYYY" strings.  If the on-disk format is actually
# "YYYYMM" (no separator), every value parses as NaT — silently — causing
# split_pd() to receive an empty in_sample and crashing sklearn with:
#   ValueError: With n_samples=0, test_size=0.3 … the resulting train set
#   will be empty.

_PERIOD_FORMATS = [
    "%m/%Y",    # "01/2000"  — standard Freddie Mac servicer format
    "%Y%m",     # "200001"   — compact format sometimes used in older files
    "%m-%Y",    # "01-2000"  — dash-separated variant
    "%Y-%m",    # "2000-01"  — ISO-like variant
    "%Y-%m-%d", # "2000-01-01" — full date variant
]


def _parse_period(series: pd.Series, col_name: str = "date",
                  warn_threshold: float = 0.01) -> pd.Series:
    """
    Robustly parse a Freddie Mac period column (monthly_reporting_period or
    zero_balance_date) by trying each format in _PERIOD_FORMATS in order.

    The first format that produces a non-trivial parse rate (>50% non-NaT)
    is adopted for the whole column.  If no format clears that bar, pandas'
    own inference is used as a last resort.

    A WARNING is emitted when the NaT rate exceeds warn_threshold so silent
    failures become immediately visible in the log.  Pass warn_threshold=1.0
    to suppress the warning entirely for structurally sparse columns such as
    zero_balance_date, which is blank for all active (non-exited) loans and
    therefore legitimately has a very high NaT rate.
    """
    s = series.astype(str).str.strip()
    best: pd.Series | None = None
    best_valid = -1

    for fmt in _PERIOD_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        n_valid = parsed.notna().sum()
        if n_valid > best_valid:
            best_valid = n_valid
            best = parsed
        # Accept the first format that parses more than half the rows
        if n_valid / max(len(s), 1) > 0.5:
            break

    # Final fallback: pandas inference (slowest but most flexible)
    if best is None or best_valid == 0:
        best = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

    nat_rate = best.isna().mean()
    if nat_rate > warn_threshold:
        log.warning(
            "  %s: %.1f%% of values parsed as NaT — check raw date format "
            "(sample values: %s)",
            col_name,
            nat_rate * 100,
            series.dropna().head(3).tolist(),
        )

    return best


# =============================================================================
# FILE LOADING
# =============================================================================

def _read_csv_pipe(path: Path, names: list[str], usecols: list[int] | None = None,
                   na_values: list[str] | None = None) -> pd.DataFrame:
    """Shared CSV reader for Freddie Mac's pipe-delimited latin-1 files."""
    kwargs: dict = dict(
        sep="|", header=None, names=names,
        dtype=str, encoding="latin-1", low_memory=False,
        na_values=na_values or ["", " "],
    )
    if usecols is not None:
        kwargs["usecols"] = usecols
    return pd.read_csv(path, **kwargs)


def load_orig_year(year: int) -> pd.DataFrame:
    """Load one origination file.  Returns empty DataFrame if missing."""
    path = RAW_DIR / f"sample_orig_{year}.txt"
    if not path.exists():
        log.warning("sample_orig_%d.txt not found — skipping year.", year)
        return pd.DataFrame(columns=ORIG_COLS)

    # Freddie Mac encodes missing numerics as sentinel strings (9, 99, …)
    numeric_na = ["", " ", "9", "99", "999", "9999", "99999", "999999", "9999999"]
    return _read_csv_pipe(path, ORIG_COLS, na_values=numeric_na)


def load_svcg_year(year: int) -> pd.DataFrame:
    """Load one servicer file, reading only the columns the pipeline needs."""
    path = RAW_DIR / f"sample_svcg_{year}.txt"
    if not path.exists():
        log.warning("sample_svcg_%d.txt not found — skipping year.", year)
        return pd.DataFrame(columns=SVCG_USECOLS)

    usecol_idx = [SVCG_COLS.index(c) for c in SVCG_USECOLS]
    df = _read_csv_pipe(path, SVCG_COLS, usecols=usecol_idx)
    return df.dropna(subset=["loan_seq_num", "monthly_reporting_period"])


# =============================================================================
# MACRO DATA  (optional — pipeline runs without these files)
# =============================================================================

def load_hpi() -> pd.DataFrame | None:
    """
    FHFA 3-digit ZIP-code HPI.

    Download : https://www.fhfa.gov/data/hpi/datasets?tab=additional-data
    Save as  : data/raw/macro/hpi_3digit_zip.csv
    Columns  : zip3, year, quarter, hpi_index
    """
    path = MACRO_DIR / "hpi_3digit_zip.csv"
    if not path.exists():
        log.warning("HPI file not found — hpi_change features will be NaN.")
        return None

    hpi = pd.read_csv(path, dtype=str)
    hpi.columns      = hpi.columns.str.lower().str.strip()
    hpi["year"]      = hpi["year"].astype(int)
    hpi["quarter"]   = hpi["quarter"].astype(int)
    hpi["hpi_index"] = pd.to_numeric(hpi["hpi_index"], errors="coerce")
    hpi["date"] = pd.to_datetime(
        hpi["year"].astype(str) + "-"
        + (hpi["quarter"] * 3).astype(str).str.zfill(2) + "-01"
    )
    log.info("  HPI loaded: %d zip-quarter records.", len(hpi))
    return hpi


def load_unemployment() -> pd.DataFrame | None:
    """
    BLS national unemployment rate, series LNS14000000.

    Download : https://data.bls.gov/timeseries/LNS14000000
    Save as  : data/raw/macro/unemployment_rate.csv
    Columns  : date (YYYY-MM-01), unemployment_rate
    """
    for fname in ["unemployment_rate.csv", "unemployment.csv"]:
        path = MACRO_DIR / fname
        if path.exists():
            break
    else:
        log.warning("Unemployment file not found — ur_3m_lag feature will be NaN.")
        return None

    ur = pd.read_csv(path, dtype=str)
    ur.columns = ur.columns.str.lower().str.strip()

    # Accept either a pre-formatted date column or year+period columns
    if "date" in ur.columns:
        ur["date"] = pd.to_datetime(ur["date"])
    elif "year" in ur.columns and "period" in ur.columns:
        ur["month"] = ur["period"].str.replace("M", "").str.zfill(2)
        ur["date"]  = pd.to_datetime(ur["year"] + "-" + ur["month"] + "-01")
    else:
        log.warning("Unemployment file has unexpected columns: %s", list(ur.columns))
        return None

    # Standardise the rate column name
    rate_col = next((c for c in ur.columns if "rate" in c or "value" in c), None)
    if rate_col is None:
        log.warning("Cannot find unemployment rate column.")
        return None

    ur = ur.rename(columns={rate_col: "unemployment_rate"})
    ur["unemployment_rate"] = pd.to_numeric(ur["unemployment_rate"], errors="coerce")
    log.info("  Unemployment loaded: %d monthly records.", len(ur))
    return ur[["date", "unemployment_rate"]].dropna()


def load_pmms() -> pd.DataFrame | None:
    """
    Freddie Mac Primary Mortgage Market Survey — weekly average 30-year
    fixed commitment rate, used by the competing-risks dataset to build
    refi_incentive = orig_interest_rate - pmms_rate_at_period.

    The refinancing incentive is the single strongest driver of voluntary
    prepayment: a borrower holding a 7% note when the market offers 4% has
    an obvious reason to refinance, and one holding 3% when the market
    offers 7% is locked in. Without it the prepayment hazard is close to
    unidentifiable from loan characteristics alone.

    Download : https://www.freddiemac.com/pmms
    Save as  : data/raw/macro/pmms_30yr_fixed.csv
    Columns  : date, rate   (any column containing "rate" is accepted)

    Optional — returns None with a warning if absent, matching the
    load_hpi() / load_unemployment() pattern. Callers skip the feature
    rather than failing.
    """
    path = config.PMMS_PATH
    if not path.exists():
        log.warning(
            "PMMS file not found at %s — refi_incentive will be skipped in "
            "the surv_* dataset. Download the 30-year fixed weekly series "
            "from https://www.freddiemac.com/pmms to enable it.", path,
        )
        return None

    pmms = pd.read_csv(path, dtype=str)
    pmms.columns = pmms.columns.str.lower().str.strip()

    date_col = next((c for c in pmms.columns if "date" in c or c == "week"), None)
    rate_col = next((c for c in pmms.columns
                     if "rate" in c or "pmms" in c or c in {"value", "frm30"}), None)
    if date_col is None or rate_col is None:
        log.warning("PMMS file has unexpected columns %s — skipping refi_incentive.",
                    list(pmms.columns))
        return None

    out = pd.DataFrame({
        "date": pd.to_datetime(pmms[date_col], errors="coerce"),
        "pmms_rate": pd.to_numeric(pmms[rate_col], errors="coerce"),
    }).dropna().sort_values("date")

    if out.empty:
        log.warning("PMMS file parsed to zero usable rows — skipping refi_incentive.")
        return None

    log.info("  PMMS loaded: %d weekly observations (%s – %s).",
             len(out), out["date"].min().date(), out["date"].max().date())
    return out


# =============================================================================
# CLEANING
# =============================================================================

def clean_orig(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast and derive fields for the origination file."""
    out = df.copy()

    # Parse origination date from YYYYMM
    out["orig_date"] = pd.to_datetime(
        out["first_payment_date"].str.strip(), format="%Y%m", errors="coerce"
    ) - pd.DateOffset(months=1)

    # Derive 3-digit ZIP for HPI join
    out["zip3"] = out["postal_code"].str.strip().str.zfill(5).str[:3]

    # Numeric casts
    for col in ["credit_score", "orig_cltv", "orig_ltv", "orig_dti",
                "orig_upb", "orig_interest_rate", "mi_pct", "num_borrowers",
                "orig_loan_term", "num_units"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Clip to economically valid ranges
    out["credit_score"]      = out["credit_score"].clip(300, 850)
    out["orig_cltv"]         = out["orig_cltv"].clip(0, 200)
    out["orig_dti"]          = out["orig_dti"].clip(0, 100)
    out["orig_interest_rate"]= out["orig_interest_rate"].clip(0, 30)

    return out


def clean_perf(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast and derive fields for the servicer (performance) file.

    FIX: replaced bare pd.to_datetime(…, format="%m/%Y") with the robust
    _parse_period() helper for both monthly_reporting_period (→ report_date)
    and zero_balance_date.  The helper tries all known Freddie Mac date
    formats in sequence and warns loudly if the NaT rate is high, preventing
    the silent all-NaT parse that caused split_pd() to receive an empty
    in_sample and crash sklearn with n_samples=0.
    """
    out = df.copy()

    # ── FIX: use robust multi-format parser instead of a single hard-coded fmt ──
    out["report_date"] = _parse_period(
        out["monthly_reporting_period"], col_name="monthly_reporting_period"
    )

    for col in ["loan_age", "current_upb", "actual_loss",
                "zero_balance_removal_upb", "mi_recoveries",
                "net_sale_proceeds", "non_mi_recoveries", "expenses",
                "delinquent_accrued_interest", "interest_bearing_upb",
                "current_interest_rate", "remaining_months"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # zero_balance_date is blank for all active (non-exited) loans, so a high
    # NaT rate is structurally expected — suppress the warning entirely.
    out["zero_balance_date"] = _parse_period(
        out["zero_balance_date"], col_name="zero_balance_date", warn_threshold=1.0
    )

    # Delinquency: 'X' = current (0 months past due), numeric strings otherwise.
    #
    # The raw field is a STRING and carries non-numeric states that are not
    # "missing": "RA" = REO acquisition (the loan is in the lender's REO
    # inventory — a terminal credit state, not an unknown one) and "XX" =
    # not available. The coercion below maps both to NaN, which is safe (it
    # does not raise) but lossy for RA. The stripped raw string is preserved
    # first so the competing-risks dataset can distinguish "REO acquired"
    # from "unknown" without re-reading the servicer files.
    #
    # This column is additive: pd_*/lgd_* chunks select their columns by
    # explicit list in main(), so carrying it here cannot alter their schema.
    out["delinquency_status_raw"] = out["delinquency_status"].str.strip()

    out["delinquency_status"] = (
        out["delinquency_status"].str.strip()
        .replace({"X": "0", "R": np.nan})
    )
    out["delinquency_status"] = pd.to_numeric(out["delinquency_status"], errors="coerce")

    # Binary indicator used as a model feature (thesis §1.5.2)
    out["delinquency_indicator"] = (out["delinquency_status"] > 0).astype(np.int8)

    out["zero_balance_code"] = out["zero_balance_code"].str.strip().str.zfill(2)
    out["loan_age"]          = out["loan_age"].clip(0, 480)

    return out


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def _build_hpi_lookup(hpi: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute a (zip3, year, quarter) → hpi_index lookup table.

    Called once after load_hpi() so that engineer_features() can resolve
    all HPI values with two vectorized merges instead of a Python loop.
    """
    return (
        hpi[["zip3", "year", "quarter", "hpi_index"]]
        .drop_duplicates(subset=["zip3", "year", "quarter"])
        .reset_index(drop=True)
    )


def _hpi_keys(date_series: pd.Series, zip3_series: pd.Series) -> pd.DataFrame:
    """
    Derive (zip3, year, quarter) join keys from date and zip3 columns.
    Returns a DataFrame aligned to the input index.
    """
    return pd.DataFrame({
        "zip3":    zip3_series.str.zfill(3).where(zip3_series.notna(), ""),
        "year":    date_series.dt.year,
        "quarter": (date_series.dt.month - 1) // 3 + 1,
    })


def engineer_features(merged: pd.DataFrame,
                       hpi: pd.DataFrame | None,
                       ur: pd.DataFrame | None) -> pd.DataFrame:
    """
    Add macroeconomic and derived features to the merged loan dataset.

    Features added
    --------------
    hpi_change           : origination HPI / current HPI (PD)
    hpi_change_since_orig: same ratio stored separately for LGD
    ur_3m_lag            : unemployment rate lagged 3 months

    HPI strategy
    ------------
    Two vectorized left merges on a pre-built (zip3, year, quarter) lookup
    table. A pandas left merge preserves the left frame's row order, and the
    lookup is de-duplicated on its keys, so the result stays row-aligned
    with df (checked below).

    Unemployment strategy
    ---------------------
    merge_asof requires both frames sorted on the join key, which reorders
    the rows. The original row position is carried through the merge in
    `_pos` and the result is re-sorted on it before assignment, so each
    loan-month receives the rate for its own lag date. (The previous version
    assigned the date-sorted result straight back by position, which
    scrambled ur_3m_lag across rows.)
    """
    df = merged.copy()

    # ── HPI change ratio ──────────────────────────────────────────────────
    if hpi is not None and "zip3" in df.columns and "orig_date" in df.columns:
        log.debug("  Computing HPI change ratios (vectorized merge) …")

        hpi_lookup = _build_hpi_lookup(hpi)

        # Derive join keys for origination date and current report date
        orig_keys = _hpi_keys(df["orig_date"],   df["zip3"]).add_suffix("_orig")
        curr_keys = _hpi_keys(df["report_date"], df["zip3"]).add_suffix("_curr")

        work = pd.concat([orig_keys, curr_keys], axis=1)
        work.index = df.index

        # Merge origination HPI
        work = work.merge(
            hpi_lookup.rename(columns={
                "zip3": "zip3_orig", "year": "year_orig",
                "quarter": "quarter_orig", "hpi_index": "hpi_orig",
            }),
            on=["zip3_orig", "year_orig", "quarter_orig"],
            how="left",
        )

        # Merge current HPI
        work = work.merge(
            hpi_lookup.rename(columns={
                "zip3": "zip3_curr", "year": "year_curr",
                "quarter": "quarter_curr", "hpi_index": "hpi_curr",
            }),
            on=["zip3_curr", "year_curr", "quarter_curr"],
            how="left",
        )

        if len(work) != len(df):
            raise RuntimeError(
                f"HPI merge changed row count ({len(df):,} → {len(work):,}); "
                "check the HPI file for duplicate (zip3, year, quarter) keys."
            )

        df["hpi_orig"] = work["hpi_orig"].to_numpy()
        df["hpi_curr"] = work["hpi_curr"].to_numpy()

        # hpi_change = HPI at origination / HPI now.
        #   > 1  → prices have FALLEN since origination (equity eroded,
        #          higher default risk)
        #   < 1  → prices have RISEN since origination (equity built up)
        # Any code that overrides this feature (e.g. the stress-test engine
        # in 07_macro_scenario_analysis.py) must use the same direction.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                df["hpi_curr"] > 0,
                df["hpi_orig"] / df["hpi_curr"],
                np.nan,
            )
        df["hpi_change"]            = ratio
        df["hpi_change_since_orig"] = ratio
    else:
        df["hpi_change"]            = np.nan
        df["hpi_change_since_orig"] = np.nan

    # ── Unemployment rate (3-month lag) ───────────────────────────────────
    if ur is not None and "report_date" in df.columns:
        lag_df = pd.DataFrame({
            "lag_date": (df["report_date"] - pd.DateOffset(months=3)).to_numpy(),
            "_pos":     np.arange(len(df)),
        })

        ur_sorted = (
            ur[["date", "unemployment_rate"]]
            .dropna(subset=["date"])
            .drop_duplicates(subset=["date"])
            .rename(columns={"date": "lag_date"})
            .sort_values("lag_date")
        )

        # merge_asof cannot take null keys: merge only rows with a valid
        # date; rows with a missing report_date keep NaN.
        valid = lag_df["lag_date"].notna()
        joined = pd.merge_asof(
            lag_df[valid].sort_values("lag_date"),
            ur_sorted,
            on="lag_date",
            direction="nearest",
            tolerance=pd.Timedelta(days=45),
        )

        ur_values = np.full(len(df), np.nan)
        ur_values[joined["_pos"].to_numpy()] = joined["unemployment_rate"].to_numpy()
        df["ur_3m_lag"] = ur_values

        n_missing = int(np.isnan(ur_values).sum())
        if n_missing:
            last_ur = ur_sorted["lag_date"].max()
            log.warning(
                "  ur_3m_lag missing for %s of %s rows (unemployment data ends "
                "%s) — these rows will be dropped from the PD set. Extend "
                "the BLS LNS14000000 file to cover the full panel.",
                f"{n_missing:,}", f"{len(df):,}", f"{last_ur:%Y-%m}",
            )
    else:
        df["ur_3m_lag"] = np.nan

    return df


# =============================================================================
# PD TARGET CONSTRUCTION  (thesis §1.5.2)
# =============================================================================

def extract_pd_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 12-month forward default indicator.

    Definition (thesis §1.5.2):
        default_12m = 1  if  0 < days_to_default ≤ 365
                    = 0  otherwise

    Rows at or after the default event are dropped to prevent data leakage.
    """
    df = df.copy()
    df["is_default"] = df["zero_balance_code"].isin(DEFAULT_CODES)

    # Earliest default date per loan
    default_dates = (
        df[df["is_default"]][["loan_seq_num", "report_date"]]
        .groupby("loan_seq_num")["report_date"]
        .min()
        .rename("default_date")
    )
    df = df.merge(default_dates, on="loan_seq_num", how="left")

    # Drop post-default rows (leakage prevention)
    df = df[df["default_date"].isna() | (df["report_date"] < df["default_date"])].copy()

    # 12-month forward window
    df["days_to_default"] = (df["default_date"] - df["report_date"]).dt.days
    df["default_12m"] = (
        (df["days_to_default"] > 0) & (df["days_to_default"] <= 365)
    ).astype(np.int8)

    # Whether this loan is ever observed to default (used downstream by
    # filter_immature_right_censored() to distinguish a confirmed
    # default_12m=0 — where the eventual default_date is known and simply
    # falls outside the 365-day window — from a right-censored row whose
    # true 12-month outcome is unknown because the dataset ends too soon).
    df["has_default"] = df["default_date"].notna()

    # Keep only pre-default, non-zero-balance rows
    mask = df["zero_balance_code"].isna() | ~df["is_default"]
    return df[mask].copy()


def filter_immature_right_censored(df: pd.DataFrame,
                                    window_days: int = 365) -> pd.DataFrame:
    """
    Drop right-censored PD rows too close to the dataset's true end to
    know their 12-month outcome.

    extract_pd_rows() labels default_12m=0 for any row lacking a full
    365-day forward window — including active loans near the overall
    performance-history cutoff, which "hasn't defaulted yet but we don't
    know for N more months" — treating it the same as a confirmed
    non-default.  That biases the default rate downward, concentrated in
    the most recent reporting periods.

    Only rows where the loan is NEVER observed to default (has_default
    False) and whose report_date falls within `window_days` of the
    dataset's global max report_date are dropped: for those rows the
    365-day outcome genuinely cannot be known yet.  Rows with a known
    (possibly distant) default_date keep their default_12m=0 label as-is
    — that label is already correct regardless of how close report_date
    is to the dataset's end, since the true outcome is observed.

    Must be called after combining all yearly chunks (needs the true
    global max report_date, not a single origination-year cohort's max).
    """
    global_max_date = df["report_date"].max()
    cutoff = global_max_date - pd.Timedelta(days=window_days)
    immature = (~df["has_default"]) & (df["report_date"] > cutoff)

    log.info(
        "  filter_immature_right_censored: dropping %s of %s rows "
        "(report_date > %s, never observed to default) — panel max date: %s",
        f"{int(immature.sum()):,}", f"{len(df):,}",
        cutoff.date(), global_max_date.date(),
    )
    return df[~immature].drop(columns=["has_default"]).copy()


# =============================================================================
# COMPETING-RISKS SURVIVAL DATASET  (surv_* variant)
# =============================================================================

PREPAY_CODES   = config.PREPAY_CODES
EVENT_TYPE     = config.EVENT_TYPE_COL
DURATION       = config.DURATION_COL

# Static origination covariates carried onto every loan-month row.
SURV_STATIC_COLS = [
    "credit_score", "orig_cltv", "orig_dti", "orig_upb", "orig_interest_rate",
    "occupancy_status", "property_type", "loan_purpose", "channel",
    "num_borrowers", "first_time_homebuyer", "property_state", "mi_pct",
]
# Time-varying covariates, valued at each row's own reporting period.
SURV_TIME_VARYING_COLS = [
    "current_upb", "current_interest_rate", "remaining_months",
    "delinquency_status", "delinquency_status_raw", "is_reo_acquisition",
    "modification_flag", "is_modified", "loan_age",
    "ur_3m_lag", "hpi_change", "pmms_rate", "refi_incentive",
]


# Explicit dtypes for the survival panel. Pandas defaults every numeric to
# float64/int64, which on a 24M-row, 33-column panel is roughly twice the RAM
# it needs. These are applied at chunk-build time so the saving stage never
# materialises a float64 copy, and they are fixed rather than inferred so
# every yearly chunk writes an identical schema (a year in which a column
# happens to be all-NaN must not silently write a different type).
SURV_FLOAT32_COLS = [
    "credit_score", "orig_cltv", "orig_dti", "orig_upb", "orig_interest_rate",
    "mi_pct", "current_upb", "current_interest_rate", "remaining_months",
    "delinquency_status", "loan_age", "ur_3m_lag", "hpi_change",
    "pmms_rate", "refi_incentive",
]
SURV_INT8_COLS  = ["period_event", "event_type", "is_reo_acquisition", "is_modified"]
SURV_INT32_COLS = ["period_month", "duration_months"]
# Low-cardinality strings: category dtype turns a per-row Python object
# pointer into a small integer code plus one shared dictionary.
SURV_CATEGORY_COLS = [
    "occupancy_status", "property_type", "loan_purpose", "channel",
    "num_borrowers", "first_time_homebuyer", "property_state",
    "delinquency_status_raw", "modification_flag",
]


def normalise_surv_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the survival panel to compact, FIXED dtypes.

    Roughly halves the panel's memory footprint. Fixed rather than inferred
    so every origination-year chunk has an identical schema, which is what
    lets the saving stage stream chunks straight into one parquet file
    instead of concatenating the whole panel in RAM first.
    """
    for col in SURV_FLOAT32_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    for col in SURV_INT8_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)
    for col in SURV_INT32_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)
    for col in SURV_CATEGORY_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").astype("category")
    return df


def months_since(start: pd.Series, end: pd.Series) -> pd.Series:
    """
    Whole calendar months between two date series.

    Computed from the date fields rather than differencing days/30, so a
    loan reporting on the 1st of every month advances by exactly 1 per
    period with no drift.
    """
    return ((end.dt.year - start.dt.year) * 12
            + (end.dt.month - start.dt.month))


def clean_delinquency_status(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Parse the raw delinquency-status string into (months_past_due, is_reo).

    The field is NOT numeric. Freddie Mac uses:
        "0".."N"  months past due
        "X"       current (0 months past due) in some vintages
        "XX"      not available
        "R"/"RA"  REO acquisition — a terminal credit state, NOT missing

    A bare pd.to_numeric() maps X, XX, R and RA alike to NaN, which
    conflates "this loan is in REO" with "we do not know". This returns the
    REO state as its own flag so the competing-risks dataset can use it.
    """
    s = raw.astype(str).str.strip().str.upper()

    is_reo = s.isin({"R", "RA"})
    months = s.replace({"X": "0", "XX": np.nan, "R": np.nan, "RA": np.nan})
    months = pd.to_numeric(months, errors="coerce")

    return months, is_reo.astype(np.int8)


def attach_refi_incentive(df: pd.DataFrame,
                          pmms: pd.DataFrame | None) -> pd.DataFrame:
    """
    Join the PMMS market rate at each reporting period and derive

        refi_incentive = orig_interest_rate - pmms_rate

    Positive means the borrower's note rate is above the market — an
    in-the-money refinance, the dominant driver of voluntary prepayment.

    PMMS is weekly and reporting periods are monthly, so the join is a
    backward merge_asof: each loan-month takes the most recent survey on or
    before its reporting date. No PMMS file -> both columns are NaN and the
    caller logs that the feature is unavailable.
    """
    out = df.copy()
    if pmms is None or "report_date" not in out.columns:
        out["pmms_rate"] = np.nan
        out["refi_incentive"] = np.nan
        return out

    joined = pd.merge_asof(
        out[["report_date"]].sort_values("report_date"),
        pmms.rename(columns={"date": "report_date"}),
        on="report_date", direction="backward",
        tolerance=pd.Timedelta(days=31),
    )
    out["pmms_rate"] = joined["pmms_rate"].to_numpy()

    if "orig_interest_rate" in out.columns:
        out["refi_incentive"] = out["orig_interest_rate"] - out["pmms_rate"]
    else:
        out["refi_incentive"] = np.nan
    return out


def extract_survival_rows(df: pd.DataFrame,
                          pmms: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the competing-risks loan-month panel for one origination cohort.

    Differs from extract_pd_rows() in three ways that matter:

    1. PREPAYMENT IS AN EVENT, NOT CENSORING. extract_pd_rows() keeps the
       zero-balance-code-01 row, labels it default_12m = 0, and the loan
       then simply stops appearing. Every downstream model therefore treats
       a prepaid loan as "still at risk, outcome unknown". It is not: a
       prepaid loan can never default afterwards. Here code 01 terminates
       the loan with event_type = 2.

    2. DURATION IS DERIVED FROM DATES, NOT loan_age. The Freddie Mac
       loan_age field resets when a loan is modified (Modification Flag
       Y/P), so a loan modified at month 40 can report loan_age = 1 the
       following month. Using it as the survival clock rewinds time for
       precisely the distressed loans whose timing matters most, and
       produces a spurious mass of "young" defaults. duration_months is
       months(orig_date -> event/censoring date) and is immune to this.
       loan_age is still carried as an ordinary covariate.

    3. ROWS RUN UP TO AND INCLUDING THE TERMINATING ROW. extract_pd_rows()
       drops the default row itself to avoid leaking the binary label; a
       discrete-time hazard model needs that row, because it is the one
       carrying the event.

    Returns one row per loan-month at risk, with:
        event_type      loan-level terminal event (0 censored / 1 default /
                        2 prepay) — constant within a loan
        duration_months loan-level time to that event — constant within a loan
        period_month    this row's month index since origination (1-based)
        period_event    0 on every row except the terminating one, which
                        carries event_type — the discrete-time target
    """
    out = df.copy()

    required = {"loan_seq_num", "report_date", "orig_date"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"extract_survival_rows requires {sorted(missing)}")

    out = out.dropna(subset=["report_date", "orig_date"])
    if out.empty:
        return pd.DataFrame()

    # ── Terminal events, by cause ────────────────────────────────────────
    out["is_default_row"] = out["zero_balance_code"].isin(DEFAULT_CODES)
    out["is_prepay_row"]  = out["zero_balance_code"].isin(PREPAY_CODES)

    def _first_date(mask: pd.Series, name: str) -> pd.DataFrame:
        """Earliest date per loan for one cause, as a 2-column frame.

        Returned as a frame and merged rather than concatenated on the index:
        when a cause has NO events in a cohort (common — many origination
        years contain zero defaults), an empty Series carries an unnamed
        index, and concatenating it strips the "loan_seq_num" index name off
        the result, so the subsequent reset_index() yields a column called
        "index" and every later reference raises KeyError.
        """
        sub = out.loc[mask, ["loan_seq_num", "report_date"]]
        if sub.empty:
            return pd.DataFrame({"loan_seq_num": pd.Series(dtype=out["loan_seq_num"].dtype),
                                 name: pd.Series(dtype="datetime64[ns]")})
        return (sub.groupby("loan_seq_num", as_index=False)["report_date"]
                .min().rename(columns={"report_date": name}))

    loans = (out.groupby("loan_seq_num", as_index=False)["report_date"]
             .max().rename(columns={"report_date": "last_obs_date"}))
    loans = loans.merge(_first_date(out["is_default_row"], "default_date"),
                        on="loan_seq_num", how="left")
    loans = loans.merge(_first_date(out["is_prepay_row"], "prepay_date"),
                        on="loan_seq_num", how="left")
    for col in ("default_date", "prepay_date"):
        if col not in loans.columns:
            loans[col] = pd.NaT
        loans[col] = pd.to_datetime(loans[col])

    # Whichever cause fires first terminates the loan. A loan carrying both
    # codes is a data error, but taking the earlier of the two is the
    # defensible resolution rather than letting it contribute twice.
    both = loans["default_date"].notna() & loans["prepay_date"].notna()
    if both.any():
        log.warning("  %s loan(s) carry BOTH a default and a prepayment code — "
                    "using whichever occurred first.", f"{int(both.sum()):,}")

    d, p = loans["default_date"], loans["prepay_date"]
    default_first = d.notna() & (p.isna() | (d <= p))
    prepay_first  = p.notna() & (d.isna() | (p < d))

    loans[EVENT_TYPE] = np.select(
        [default_first, prepay_first],
        [config.EVENT_DEFAULT, config.EVENT_PREPAY],
        default=config.EVENT_CENSORED,
    ).astype(np.int8)
    loans["event_date"] = np.where(default_first, d,
                                   np.where(prepay_first, p, loans["last_obs_date"]))
    loans["event_date"] = pd.to_datetime(loans["event_date"])

    # ── Duration, from dates ─────────────────────────────────────────────
    orig_dates = out.groupby("loan_seq_num")["orig_date"].min().rename("orig_date_loan")
    loans = loans.merge(orig_dates, on="loan_seq_num", how="left")
    loans[DURATION] = months_since(loans["orig_date_loan"], loans["event_date"])
    # A loan terminating in its first reporting period has duration 1, not 0:
    # it was at risk for one period. Negative values would mean an event
    # dated before origination, which is a data error.
    bad = loans[DURATION] < 0
    if bad.any():
        log.warning("  %s loan(s) have an event dated before origination — dropped.",
                    f"{int(bad.sum()):,}")
        loans = loans[~bad]
    loans[DURATION] = loans[DURATION].clip(lower=1).astype(int)

    # ── Trim the panel to the at-risk window ─────────────────────────────
    out = out.merge(
        loans[["loan_seq_num", EVENT_TYPE, DURATION, "event_date", "orig_date_loan"]],
        on="loan_seq_num", how="inner",
    )
    out = out[out["report_date"] <= out["event_date"]].copy()

    # NOT clipped at 1: clipping would collapse a genuine month-0 row into
    # month 1, giving two distinct reporting periods the same period index
    # and double-counting one of them in the discrete-time panel. In Freddie
    # Mac data orig_date is first_payment_date - 1 month, so the first
    # servicing row already lands at month 1.
    out["period_month"] = months_since(out["orig_date_loan"], out["report_date"])

    # The terminating row carries the event; every earlier row is a survival.
    is_terminal = out["report_date"] == out["event_date"]
    out["period_event"] = np.where(is_terminal, out[EVENT_TYPE],
                                   config.EVENT_CENSORED).astype(np.int8)

    # ── Covariates ───────────────────────────────────────────────────────
    if "delinquency_status_raw" in out.columns:
        _months, is_reo = clean_delinquency_status(out["delinquency_status_raw"])
        out["is_reo_acquisition"] = is_reo
    else:
        out["is_reo_acquisition"] = np.int8(0)

    if "modification_flag" in out.columns:
        flag = out["modification_flag"].astype(str).str.strip().str.upper()
        out["is_modified"] = flag.isin({"Y", "P"}).astype(np.int8)
    else:
        out["modification_flag"] = np.nan
        out["is_modified"] = np.int8(0)

    out = attach_refi_incentive(out, pmms)

    keep = (["loan_seq_num", "report_date", "orig_date_loan", "period_month",
             "period_event", EVENT_TYPE, DURATION]
            + [c for c in SURV_STATIC_COLS if c in out.columns]
            + [c for c in SURV_TIME_VARYING_COLS if c in out.columns])
    keep = list(dict.fromkeys(keep))
    out = out[keep].rename(columns={"orig_date_loan": "orig_date"})
    return normalise_surv_dtypes(out)


def stream_survival_splits(surv_files: list, out_dir: Path) -> dict:
    """
    Split and write the survival panel one origination-year chunk at a time,
    never holding the whole panel in memory.

    Why this can be done per chunk: a Freddie Mac sample_svcg_YYYY.txt file
    contains the COMPLETE servicing history of the loans originated in YYYY,
    so every loan lives entirely inside exactly one chunk. Both split rules
    are therefore chunk-local — the OOT assignment keys on each loan's own
    origination date, and the train/OOS GroupShuffleSplit groups by
    loan_seq_num within the chunk — and no loan can end up spanning two
    splits, which is the property the survival models actually depend on.

    The one behavioural difference from splitting the combined panel in a
    single pass: which specific in-sample loans land in train vs OOS. Each
    chunk draws its own ~30% holdout instead of one draw over all loans. The
    design (loan-grouped, ~OOS_FRAC of loans held out, same OOT_CUTOFF) is
    unchanged, and the aggregate proportions are the same.

    Category columns are written as plain strings so every chunk produces an
    identical Arrow schema; parquet still dictionary-encodes them on disk, so
    the file size is unaffected.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    targets = {"train": config.SURV_TRAIN_FILE,
               "oos":   config.SURV_OOS_FILE,
               "oot":   config.SURV_OOT_FILE}
    writers: dict[str, pq.ParquetWriter] = {}
    stats = {
        "events": {config.EVENT_CENSORED: 0, config.EVENT_DEFAULT: 0,
                   config.EVENT_PREPAY: 0},
        "rows": {k: 0 for k in targets},
        "loans": {k: 0 for k in targets},
        "refi_sum": 0.0, "refi_n": 0, "total_rows": 0,
    }

    def _arrow_ready(frame: pd.DataFrame) -> pd.DataFrame:
        cats = frame.select_dtypes(include="category").columns
        if len(cats):
            frame = frame.copy()
            for col in cats:
                frame[col] = frame[col].astype("string")
        return frame

    try:
        for path in surv_files:
            chunk = pd.read_parquet(path)
            stats["total_rows"] += len(chunk)

            # One row per loan for the event tally — each loan is confined to
            # this chunk, so per-chunk counts sum to the panel total.
            terminal = chunk[["loan_seq_num", EVENT_TYPE]].drop_duplicates("loan_seq_num")
            for code in stats["events"]:
                stats["events"][code] += int((terminal[EVENT_TYPE] == code).sum())
            if "refi_incentive" in chunk.columns:
                values = chunk["refi_incentive"]
                stats["refi_sum"] += float(values.sum(skipna=True))
                stats["refi_n"]  += int(values.notna().sum())

            parts = split_survival(chunk, quiet=True)
            for name, part in zip(("train", "oos", "oot"), parts):
                if part.empty:
                    continue
                stats["rows"][name]  += len(part)
                stats["loans"][name] += part["loan_seq_num"].nunique()

                table = pa.Table.from_pandas(_arrow_ready(part), preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(out_dir / targets[name], table.schema)
                else:
                    table = table.cast(writers[name].schema)
                writers[name].write_table(table)

            del chunk, terminal, parts
            gc.collect()
    finally:
        for writer in writers.values():
            writer.close()

    return stats


def split_survival(df: pd.DataFrame,
                   quiet: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train / OOS / OOT split for the competing-risks panel.

    Uses the same OOT_CUTOFF constant and the same GroupShuffleSplit on
    loan_seq_num as split_pd(), with ONE deliberate difference, which is a
    requirement of survival data rather than a preference:

        split_pd() cuts OOT by ROW (report_date >= OOT_CUTOFF), so a single
        loan's monthly rows can land in both the in-sample and OOT files.
        That is harmless for a snapshot classifier. It is fatal for a
        survival model: a loan's duration is a property of its whole
        history, so splitting that history across two files would truncate
        every straddling loan's duration at the cutoff in one file and
        start it mid-flight in the other, biasing both.

    So OOT is assigned by ORIGINATION date: loans originated on or after
    OOT_CUTOFF form a genuine out-of-time cohort with their histories
    intact, and the remaining loans are split train/OOS by loan, exactly as
    split_pd() does. Whole loan histories never span two splits.
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    orig = df.groupby("loan_seq_num")["orig_date"].min()
    oot_loans = set(orig[orig >= OOT_CUTOFF].index)

    oot = df[df["loan_seq_num"].isin(oot_loans)].copy()
    in_sample = df[~df["loan_seq_num"].isin(oot_loans)].copy()

    if in_sample.empty:
        if not quiet:
            log.warning("  split_survival: no in-sample loans before %s — "
                        "returning everything as OOT.", OOT_CUTOFF.date())
        return in_sample, in_sample.copy(), oot

    gss = GroupShuffleSplit(n_splits=1, test_size=OOS_FRAC, random_state=SEED)
    train_idx, oos_idx = next(gss.split(in_sample, groups=in_sample["loan_seq_num"]))
    train = in_sample.iloc[train_idx]
    oos   = in_sample.iloc[oos_idx]

    def _loans(d):
        return d["loan_seq_num"].nunique()

    if quiet:
        return train, oos, oot

    log.info(
        "  Survival split — Train: %s rows / %s loans  |  OOS: %s / %s  |  "
        "OOT: %s / %s  (OOT = originated on or after %s)",
        f"{len(train):,}", f"{_loans(train):,}", f"{len(oos):,}", f"{_loans(oos):,}",
        f"{len(oot):,}", f"{_loans(oot):,}", OOT_CUTOFF.date(),
    )
    return train, oos, oot


# =============================================================================
# LGD TARGET CONSTRUCTION  (thesis §3.3)
# =============================================================================

def extract_lgd_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the LGD target: actual_loss / zero_balance_removal_upb, clipped [0, 1].

    One row is retained per defaulted loan — the final servicer observation
    at the point of resolution, which contains the realised loss amount.
    """
    defaults = df[df["zero_balance_code"].isin(DEFAULT_CODES)].copy()
    if defaults.empty:
        return pd.DataFrame()

    # Keep the last observation per loan (resolution row)
    defaults = (
        defaults.sort_values("report_date")
        .groupby("loan_seq_num")
        .last()
        .reset_index()
    )

    upb  = defaults["zero_balance_removal_upb"].fillna(defaults["current_upb"])
    # Freddie Mac reports realised losses as negative amounts; flip the sign.
    # Missing losses stay NaN (dropped below) rather than being treated as zero.
    loss = -pd.to_numeric(defaults["actual_loss"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        lgd_raw = np.where(upb > 0, loss / upb, np.nan)

    defaults["lgd_raw"] = lgd_raw
    defaults["lgd"]     = np.clip(lgd_raw, 0, 1)

    return defaults.dropna(subset=["lgd"])


# =============================================================================
# LGD WORKOUT-PERIOD TRUNCATION BIAS  (IPCW correction — thesis §3.3 note)
# =============================================================================

def extract_lgd_onset_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per loan that ever enters workout (90+ days past due, per
    config.LGD_ONSET_DPD_MONTHS), recording when it entered and whether /
    when it resolved.

    extract_lgd_rows() correctly keeps only *resolved* defaults (no
    leakage), but that resolved-case sample is truncated: a loan that
    entered workout close to the dataset's end and takes a long time to
    resolve (contested foreclosure, REO) is systematically missing,
    while a loan that resolves quickly (short sale) is always captured
    even near the cutoff. This under-the-hood sample is the input to
    compute_ipcw_weights(), which corrects for that truncation.
    """
    onset_mask = df["delinquency_status"] >= config.LGD_ONSET_DPD_MONTHS
    if not onset_mask.any():
        return pd.DataFrame(columns=["loan_seq_num", "onset_date", "resolved", "resolution_date"])

    onset_date = (
        df[onset_mask][["loan_seq_num", "report_date"]]
        .groupby("loan_seq_num")["report_date"]
        .min()
        .rename("onset_date")
    )

    is_default = df["zero_balance_code"].isin(DEFAULT_CODES)
    resolution_date = (
        df[is_default][["loan_seq_num", "report_date"]]
        .groupby("loan_seq_num")["report_date"]
        .min()
        .rename("resolution_date")
    )

    out = onset_date.reset_index().merge(
        resolution_date.reset_index(), on="loan_seq_num", how="left"
    )
    out["resolved"] = out["resolution_date"].notna()
    return out


def compute_ipcw_weights(onset_df: pd.DataFrame,
                          global_max_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Inverse-probability-of-censoring weights correcting the LGD training
    sample for workout-period truncation.

    Resolved cases whose onset is close to the dataset's observation
    cutoff can only be in the sample if they resolved quickly — slow
    resolutions with the same onset timing are still open and (correctly)
    excluded from extract_lgd_rows(), so the resolved-case sample
    under-represents long resolution times. This is a random-truncation
    problem; the standard correction is to estimate the truncation
    ("censoring") distribution G(t) = P(available follow-up > t) via a
    reversed Kaplan-Meier fit — treating still-open cases as the "event"
    and resolved cases as "censored" in that auxiliary fit — and weight
    each resolved case by 1 / Ĝ(its resolution time).

    global_max_date should be the true panel-wide max report_date (e.g.
    from the combined PD dataset, which spans every loan-month — not just
    onset/resolution events) — pass it in explicitly when available.
    Falls back to the max onset/resolution date seen in onset_df itself
    (an underestimate whenever still-current loans that never hit the
    onset trigger extend the panel further) so this function stays
    usable/testable standalone.

    Returns a [loan_seq_num, ipcw_weight] frame (mean-normalised to 1
    across resolved cases). If there are no still-open cases at all,
    there is no truncation to correct for and every resolved case gets
    weight 1.0.
    """
    if onset_df.empty:
        return pd.DataFrame(columns=["loan_seq_num", "ipcw_weight"])

    df = onset_df.copy()
    if global_max_date is None:
        global_max_date = df["onset_date"].combine(
            df["resolution_date"], lambda a, b: b if pd.notna(b) else a
        ).max()
    # available_followup: how long an as-yet-unresolved case *could* have
    # been observed to resolve, given the dataset's true end.
    available_followup_days = (global_max_date - df["onset_date"]).dt.days
    resolution_days = (df["resolution_date"] - df["onset_date"]).dt.days

    df["duration"] = np.where(df["resolved"], resolution_days, available_followup_days)
    df["duration"] = df["duration"].clip(lower=1)

    resolved = df["resolved"].to_numpy()
    if not (~resolved).any():
        # No still-open cases at all -> nothing is truncated.
        return pd.DataFrame({
            "loan_seq_num": df.loc[resolved, "loan_seq_num"],
            "ipcw_weight": 1.0,
        })

    from lifelines import KaplanMeierFitter

    censor_event = (~resolved).astype(int)  # still-open cases are the "event" for G
    kmf = KaplanMeierFitter()
    kmf.fit(df["duration"], censor_event)

    resolved_durations = df.loc[resolved, "duration"]
    g_hat = kmf.survival_function_at_times(resolved_durations).values
    g_hat = np.clip(g_hat, config.LGD_IPCW_G_FLOOR, 1.0)

    weights = 1.0 / g_hat
    weights = weights / weights.mean()

    return pd.DataFrame({
        "loan_seq_num": df.loc[resolved, "loan_seq_num"].values,
        "ipcw_weight":  weights,
    })


# =============================================================================
# TRAIN / OOS / OOT SPLIT
# =============================================================================

def split_pd(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal OOT split + random, loan-grouped OOS split on the in-sample portion.

    OOT (out-of-time)      : report_date >= OOT_CUTOFF
    OOS (out-of-sample)    : random ~30% of loans (grouped by loan_seq_num)
                             with report_date < OOT_CUTOFF
    Train                  : remaining ~70% of in-sample loans

    FIX: added a NaT-rate diagnostic and a descriptive ValueError when
    in_sample is empty, replacing the cryptic sklearn n_samples=0 crash.
    The most common cause is _parse_period() silently producing all-NaT
    values, which makes every NaT < OOT_CUTOFF comparison evaluate to
    False, leaving in_sample with zero rows.

    FIX: switched from row-level train_test_split(shuffle=True) to
    GroupShuffleSplit grouped by loan_seq_num.  df holds one row per
    loan-month, so a plain row shuffle scattered a single loan's monthly
    snapshots randomly across Train and OOS — the same loan (largely
    static credit score, CLTV, DTI, orig rate) could appear in both,
    letting models partially "recognise" training loans and inflating
    reported OOS AUROC/KS/Gini.  GroupShuffleSplit keeps every row of a
    given loan in the same split.  Because loan-month counts vary by
    loan, the achieved row-level OOS fraction only approximates
    OOS_FRAC (unlike the old exact row-level split) — logged below.
    """
    nat_rate = df["report_date"].isna().mean()
    if nat_rate > 0.01:
        log.warning(
            "  split_pd: %.1f%% of report_date values are NaT — "
            "date parsing in clean_perf() may have failed.",
            nat_rate * 100,
        )

    in_sample = df[df["report_date"] < OOT_CUTOFF].copy()
    oot        = df[df["report_date"] >= OOT_CUTOFF].copy()

    # ── FIX: guard against empty in_sample before calling sklearn ────────
    if in_sample.empty:
        raise ValueError(
            f"split_pd: in_sample is empty after applying OOT_CUTOFF "
            f"({OOT_CUTOFF.date()}).  "
            f"report_date range in data: "
            f"{df['report_date'].min()} – {df['report_date'].max()}  "
            f"(NaT rate: {nat_rate:.1%}).  "
            "Check that _parse_period() is correctly parsing "
            "monthly_reporting_period from the raw servicer files."
        )

    gss = GroupShuffleSplit(n_splits=1, test_size=OOS_FRAC, random_state=SEED)
    train_idx, oos_idx = next(
        gss.split(in_sample, groups=in_sample["loan_seq_num"])
    )
    train = in_sample.iloc[train_idx]
    oos   = in_sample.iloc[oos_idx]

    log.info(
        "  PD split — Train: %s  OOS: %s  OOT: %s  (OOS row fraction of "
        "in-sample: %.1f%%, target %.1f%% — approximate since loans have "
        "differing month counts)",
        f"{len(train):,}", f"{len(oos):,}", f"{len(oot):,}",
        len(oos) / len(in_sample) * 100, OOS_FRAC * 100,
    )
    return train, oos, oot


def split_lgd(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same temporal / random split applied to the LGD dataset.

    FIX: added matching NaT-rate diagnostic and empty in_sample guard,
    consistent with the fix applied to split_pd().

    Unlike split_pd(), this does NOT need a GroupShuffleSplit: df is
    already one row per loan (extract_lgd_rows() dedupes to the final
    resolution row per loan_seq_num before this is called), so a plain
    row-level train_test_split is already loan-level.
    """
    nat_rate = df["zero_balance_date"].isna().mean()
    if nat_rate > 0.01:
        log.warning(
            "  split_lgd: %.1f%% of zero_balance_date values are NaT — "
            "date parsing in clean_perf() may have failed.",
            nat_rate * 100,
        )

    in_sample = df[df["zero_balance_date"] < OOT_CUTOFF].copy()
    oot        = df[df["zero_balance_date"] >= OOT_CUTOFF].copy()

    if in_sample.empty:
        log.warning(
            "  split_lgd: in_sample is empty after OOT cutoff — "
            "zero_balance_date range: %s – %s  (NaT rate: %.1f%%).  "
            "OOS split skipped.",
            df["zero_balance_date"].min(),
            df["zero_balance_date"].max(),
            nat_rate * 100,
        )
        return in_sample, pd.DataFrame(), oot

    if len(in_sample) < 5:
        log.warning("  Very few in-sample LGD rows — OOS split skipped.")
        return in_sample, pd.DataFrame(), oot

    train, oos = train_test_split(in_sample, test_size=OOS_FRAC,
                                  random_state=SEED, shuffle=True)
    return train, oos, oot


# =============================================================================
# INFORMATION VALUE  (thesis §1.5.2, eq. 5-6)
# =============================================================================

def _compute_iv(df: pd.DataFrame, feature: str,
                target: str, n_bins: int = 15) -> float:
    """
    Compute the Information Value (IV) for a single feature.

    WoE_j = ln(p_j / q_j)   where p_j = fraction of goods in bin j
                                   q_j = fraction of bads  in bin j

    IV = Σ_j (p_j - q_j) * WoE_j
    """
    data = df[[feature, target]].dropna()
    good_total = max((data[target] == 0).sum(), 1)
    bad_total  = max((data[target] == 1).sum(), 1)

    is_cat = data[feature].dtype == object or data[feature].nunique() <= 10
    if is_cat:
        data = data.copy()
        data["bin"] = data[feature].astype(str)
    else:
        try:
            data = data.copy()
            data["bin"] = pd.qcut(data[feature], q=n_bins,
                                  duplicates="drop").astype(str)
        except Exception:
            return np.nan

    iv = 0.0
    for _, grp in data.groupby("bin", observed=True):
        p = max((grp[target] == 0).sum() / good_total, 1e-9)
        q = max((grp[target] == 1).sum() / bad_total,  1e-9)
        iv += (p - q) * np.log(p / q)

    return round(iv, 6)


_iv_strength = config.iv_strength


def compute_all_iv(train: pd.DataFrame, features: list[str],
                   target: str = "default_12m") -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in train.columns:
            continue
        iv = _compute_iv(train, feat, target)
        rows.append({"feature": feat, "iv": iv, "strength": _iv_strength(iv)})
    return (
        pd.DataFrame(rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# POPULATION STABILITY INDEX  (thesis §1.6)
# =============================================================================

def _compute_psi(ref: pd.Series, test: pd.Series, n_bins: int = 10) -> float:
    """
    PSI = Σ_i (p̂_i - q̂_i) * ln(p̂_i / q̂_i)

    Thresholds: < 0.10 stable | 0.10–0.25 investigate | > 0.25 major shift.
    """
    ref  = ref.dropna()
    test = test.dropna()
    if len(ref) == 0 or len(test) == 0:
        return np.nan

    # Proportions are counts / len(test), not value_counts(normalize=True)
    # (which divides by the count of values landing in a *known* bin — if
    # test falls entirely outside ref's bins that denominator is 0, giving
    # NaN proportions that Series.sum(skipna=True) then silently drops,
    # reporting PSI ~ 0 for what should be the largest possible shift).
    is_cat = ref.dtype == object or ref.nunique() <= 10
    if is_cat:
        cats = ref.value_counts(normalize=True)
        p = cats
        q = test.value_counts().reindex(cats.index, fill_value=0) / len(test)
    else:
        try:
            _, edges = pd.qcut(ref, q=n_bins, duplicates="drop", retbins=True)
            p = pd.cut(ref,  bins=edges, include_lowest=True).value_counts(normalize=True).sort_index()
            test_counts = pd.cut(test, bins=edges, include_lowest=True).value_counts().sort_index()
            q = test_counts.reindex(p.index, fill_value=0) / len(test)
        except Exception:
            return np.nan

    p = p.clip(lower=1e-9)
    q = q.clip(lower=1e-9)
    return float(((p - q) * np.log(p / q)).sum())


_psi_flag = config.psi_flag


def compute_all_psi(ref: pd.DataFrame, test: pd.DataFrame,
                    features: list[str], label: str) -> pd.DataFrame:
    if test.empty:
        return pd.DataFrame(columns=["feature", f"psi_{label}", f"flag_{label}"])
    rows = []
    for feat in features:
        if feat not in ref.columns or feat not in test.columns:
            continue
        psi = _compute_psi(ref[feat], test[feat])
        rows.append({"feature": feat, f"psi_{label}": round(psi, 6),
                     f"flag_{label}": _psi_flag(psi)})
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Data Engineering Pipeline")
    log.info("Memory strategy: year-by-year chunked processing")
    log.info("=" * 65)

    # ── Macro data (small — load once and keep in RAM) ───────────────────
    log.info("")
    log.info("[1/4] Loading macro data …")
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    hpi = load_hpi()
    ur  = load_unemployment()
    pmms = load_pmms()   # optional — competing-risks refi_incentive only

    # ── Year-by-year processing ──────────────────────────────────────────
    log.info("")
    log.info("[2/4] Processing origination years …")
    years_found    = 0
    pd_row_total   = 0
    lgd_row_total  = 0
    surv_row_total = 0

    for year in range(START_YEAR, END_YEAR + 1):
        if not (RAW_DIR / f"sample_orig_{year}.txt").exists():
            continue
        if not (RAW_DIR / f"sample_svcg_{year}.txt").exists():
            continue

        log.info("")
        log.info("  ── %d ─────────────────────────────────────────────", year)

        # Load
        orig_raw = load_orig_year(year)
        svcg_raw = load_svcg_year(year)
        if orig_raw.empty or svcg_raw.empty:
            continue
        log.info("  orig: %s rows  |  svcg: %s rows",
                 f"{len(orig_raw):>7,}", f"{len(svcg_raw):>9,}")

        # Clean
        orig_clean = clean_orig(orig_raw)
        svcg_clean = clean_perf(svcg_raw)
        del orig_raw, svcg_raw
        gc.collect()

        # Merge
        keep = [c for c in ORIG_KEEP if c in orig_clean.columns]
        merged = svcg_clean.merge(orig_clean[keep], on="loan_seq_num", how="inner")
        del orig_clean, svcg_clean
        gc.collect()

        # Feature engineering
        merged = engineer_features(merged, hpi, ur)
        log.info("  Merged + engineered: %s rows", f"{len(merged):,}")

        # Disposition diagnostics
        zbc = merged["zero_balance_code"].dropna().value_counts().sort_index()
        if not zbc.empty:
            parts = [
                f"{'★' if c in DEFAULT_CODES else ' '}{c}={n:,}"
                f"({_ZBC_LABELS.get(c, 'other')})"
                for c, n in zbc.items()
            ]
            log.info("  Dispositions: %s", "  ".join(parts))
            n_def = merged["zero_balance_code"].isin(DEFAULT_CODES).sum()
            log.info("  Default events (★): %s", f"{int(n_def):,}")
        else:
            log.info("  Dispositions: all NaN (loans still active)")

        # PD chunk
        pd_chunk = extract_pd_rows(merged)
        pd_cols  = [c for c in PD_FEATURES if c in pd_chunk.columns]
        required = [c for c in pd_cols if pd_chunk[c].notna().any()]
        # has_default: auxiliary flag consumed by filter_immature_right_censored()
        # after chunks are combined, then dropped — not a model feature.
        # current_upb / current_interest_rate / remaining_months: auxiliary
        # EAD-amortization inputs for 07_macro_scenario_analysis.py — also not
        # model features (not in config.PD_FEATURES).
        # delinquency_status: numeric months-past-due (delinquency_indicator
        # only carries the binary >0 flag) — the IFRS 9 stage-2/stage-3 DPD
        # backstops in 07_macro_scenario_analysis.py need the actual DPD count
        # to distinguish 30-DPD from 90-DPD, not just "any delinquency".
        # default_date: the loan's earliest observed default report_date (NaT
        # for loans never observed to default) — 11_discrete_hazard.py needs
        # the actual event date, not just the 365-day default_12m flag, to
        # place its monthly hazard target on the correct loan-month row. It
        # must survive the split because split_pd() cuts on report_date, so a
        # defaulting loan's rows can straddle pd_train and pd_oot: the last
        # row in pd_train is then NOT the row before default, and the event
        # date cannot be recovered from either file alone.
        base_cols = [c for c in ["loan_seq_num", "report_date", "default_12m",
                                  "has_default", "default_date", "current_upb",
                                  "current_interest_rate", "remaining_months",
                                  "delinquency_status"]
                     if c in pd_chunk.columns]
        pd_chunk = pd_chunk[list(dict.fromkeys(pd_cols + base_cols))]
        if required:
            pd_chunk = pd_chunk.dropna(subset=required)
        if not pd_chunk.empty:
            pd_chunk.to_parquet(CHUNK_DIR / f"pd_{year}.parquet", index=False)
            pd_row_total += len(pd_chunk)
            log.info("  PD rows: %s  (default rate: %.4f%%)",
                     f"{len(pd_chunk):,}", pd_chunk["default_12m"].mean() * 100)

        # LGD chunk
        lgd_chunk = extract_lgd_rows(merged)
        if not lgd_chunk.empty:
            lgd_cols     = [c for c in LGD_FEATURES if c in lgd_chunk.columns]
            lgd_required = [c for c in lgd_cols if lgd_chunk[c].notna().any()]
            lgd_base     = [c for c in ["loan_seq_num", "zero_balance_date", "lgd", "lgd_raw"]
                            if c in lgd_chunk.columns]
            lgd_chunk = lgd_chunk[list(dict.fromkeys(lgd_cols + lgd_base))]
            lgd_chunk = lgd_chunk.dropna(subset=lgd_required + ["lgd"]) if lgd_required \
                        else lgd_chunk.dropna(subset=["lgd"])
            if not lgd_chunk.empty:
                lgd_chunk.to_parquet(CHUNK_DIR / f"lgd_{year}.parquet", index=False)
                lgd_row_total += len(lgd_chunk)
                log.info("  LGD rows: %s  (mean LGD: %.4f)",
                         f"{len(lgd_chunk):,}", lgd_chunk["lgd"].mean())

        # Onset chunk — feeds the IPCW correction for LGD workout-period
        # truncation bias (compute_ipcw_weights(), applied after combining).
        onset_chunk = extract_lgd_onset_rows(merged)
        if not onset_chunk.empty:
            onset_chunk.to_parquet(CHUNK_DIR / f"onset_{year}.parquet", index=False)

        # Competing-risks panel — retains prepayment (zero-balance code 01)
        # as an EVENT rather than letting the loan silently leave the panel.
        # Written to its own chunk family so the pd_*/lgd_* outputs above are
        # untouched.
        surv_chunk = extract_survival_rows(merged, pmms)
        if not surv_chunk.empty:
            surv_chunk.to_parquet(CHUNK_DIR / f"surv_{year}.parquet", index=False)
            surv_row_total += len(surv_chunk)
            terminal = surv_chunk.drop_duplicates("loan_seq_num")[config.EVENT_TYPE_COL]
            log.info("  Survival rows: %s  (loans: %s — default %s / prepay %s / censored %s)",
                     f"{len(surv_chunk):,}", f"{len(terminal):,}",
                     f"{int((terminal == config.EVENT_DEFAULT).sum()):,}",
                     f"{int((terminal == config.EVENT_PREPAY).sum()):,}",
                     f"{int((terminal == config.EVENT_CENSORED).sum()):,}")

        del merged, pd_chunk, lgd_chunk, onset_chunk, surv_chunk
        gc.collect()
        years_found += 1

    if years_found == 0:
        raise FileNotFoundError(
            f"No data files found in {RAW_DIR}\n"
            "Expected: sample_orig_YYYY.txt and sample_svcg_YYYY.txt\n"
            "Run 00_download_freddie_mac.py first."
        )

    log.info("")
    log.info("  Total PD rows : %s", f"{pd_row_total:,}")
    log.info("  Total LGD rows: %s", f"{lgd_row_total:,}")
    log.info("  Total survival rows: %s", f"{surv_row_total:,}")

    # ── Combine chunks ───────────────────────────────────────────────────
    log.info("")
    log.info("[3/4] Combining chunks and splitting train / OOS / OOT …")

    pd_files    = sorted(CHUNK_DIR.glob("pd_*.parquet"))
    lgd_files   = sorted(CHUNK_DIR.glob("lgd_*.parquet"))
    onset_files = sorted(CHUNK_DIR.glob("onset_*.parquet"))
    surv_files  = sorted(CHUNK_DIR.glob("surv_*.parquet"))

    if not pd_files:
        raise RuntimeError("No PD chunk files produced — inspect year-by-year output above.")

    pd_all  = pd.concat([pd.read_parquet(f) for f in pd_files],  ignore_index=True)
    lgd_all = pd.concat([pd.read_parquet(f) for f in lgd_files], ignore_index=True) \
              if lgd_files else pd.DataFrame()
    onset_all = pd.concat([pd.read_parquet(f) for f in onset_files], ignore_index=True) \
                if onset_files else pd.DataFrame(columns=["loan_seq_num", "onset_date", "resolved", "resolution_date"])

    log.info("  PD  combined: %s rows  (before immaturity filter)", f"{len(pd_all):,}")
    log.info("  LGD combined: %s rows", f"{len(lgd_all):,}")

    global_max_report_date = pd_all["report_date"].max()
    pd_all = filter_immature_right_censored(pd_all)
    log.info("  PD  after immaturity filter: %s rows  (overall default rate: %.4f%%)",
             f"{len(pd_all):,}", pd_all["default_12m"].mean() * 100)

    pd_cols = [c for c in PD_FEATURES if c in pd_all.columns]
    pd_train, pd_oos, pd_oot = split_pd(pd_all)
    # split_pd() returns independent frames, so the combined panel is dead
    # weight from here on — and it is one of the largest objects alive when
    # the saving stage runs.
    del pd_all
    gc.collect()

    def _attach_ipcw(d: pd.DataFrame, ipcw: pd.DataFrame) -> pd.DataFrame:
        if d.empty or "loan_seq_num" not in d.columns:
            return d
        d = d.merge(ipcw, on="loan_seq_num", how="left")
        d["ipcw_weight"] = d["ipcw_weight"].fillna(1.0)
        return d

    if not lgd_all.empty:
        lgd_train, lgd_oos, lgd_oot = split_lgd(lgd_all)
        ipcw = compute_ipcw_weights(onset_all, global_max_date=global_max_report_date)
        log.info("  IPCW weights (LGD workout-truncation correction): %s resolved "
                 "loans weighted  (min=%.3f  mean=%.3f  max=%.3f)",
                 f"{len(ipcw):,}",
                 ipcw["ipcw_weight"].min() if not ipcw.empty else 1.0,
                 ipcw["ipcw_weight"].mean() if not ipcw.empty else 1.0,
                 ipcw["ipcw_weight"].max() if not ipcw.empty else 1.0)
        lgd_train = _attach_ipcw(lgd_train, ipcw)
        lgd_oos   = _attach_ipcw(lgd_oos,   ipcw)
        lgd_oot   = _attach_ipcw(lgd_oot,   ipcw)
    else:
        log.warning("No LGD data — no defaults recorded in dataset.")
        lgd_train = lgd_oos = lgd_oot = pd.DataFrame()

    # ── Information Value ────────────────────────────────────────────────
    log.info("")
    log.info("[+] Computing Information Values on PD training set …")
    iv_summary = compute_all_iv(pd_train, pd_cols)
    log.info("\n%s", iv_summary.to_string(index=False))

    # ── Population Stability Index ───────────────────────────────────────
    log.info("")
    log.info("[+] Computing PSI (Train vs OOS and Train vs OOT) …")
    psi_oos = compute_all_psi(pd_train, pd_oos, pd_cols, "OOS")
    psi_oot = compute_all_psi(pd_train, pd_oot, pd_cols, "OOT")
    psi_all = psi_oos.merge(psi_oot, on="feature", how="outer")
    log.info("\n%s", psi_all.to_string(index=False))

    # ── Save ────────────────────────────────────────────────────────────
    log.info("")
    log.info("[4/4] Saving outputs to %s …", OUT_DIR.resolve())

    pd_train.to_parquet(OUT_DIR / "pd_train.parquet",   index=False)
    pd_oos.to_parquet(  OUT_DIR / "pd_oos.parquet",     index=False)
    pd_oot.to_parquet(  OUT_DIR / "pd_oot.parquet",     index=False)
    iv_summary.to_csv(  OUT_DIR / "pd_iv_summary.csv",  index=False)
    psi_all.to_csv(     OUT_DIR / "pd_psi_summary.csv", index=False)

    if not lgd_all.empty:
        lgd_train.to_parquet(OUT_DIR / "lgd_train.parquet", index=False)
        lgd_oos.to_parquet(  OUT_DIR / "lgd_oos.parquet",   index=False)
        lgd_oot.to_parquet(  OUT_DIR / "lgd_oot.parquet",   index=False)

    # ── Competing-risks variant ─────────────────────────────────────────
    # Emitted alongside pd_* / lgd_*, never in place of them.
    #
    # Runs LAST, and streams. Both matter on a 30 GB box: the PD and LGD
    # frames above are released first so the survival panel never shares RAM
    # with them, and the panel is split and written one origination-year
    # chunk at a time rather than being concatenated whole. Concatenating it
    # here — while pd_all, the three PD splits and the LGD frames were all
    # still live — is what exhausted memory on Kaggle at the saving stage.
    del pd_train, pd_oos, pd_oot
    if not lgd_all.empty:
        del lgd_train, lgd_oos, lgd_oot
    del lgd_all, onset_all, iv_summary, psi_all, psi_oos, psi_oot
    gc.collect()

    if surv_files:
        log.info("")
        log.info("[+] Writing the competing-risks variant (streamed per "
                 "origination-year chunk) …")
        stats = stream_survival_splits(surv_files, OUT_DIR)

        n_loans = sum(stats["events"].values())
        log.info("  Competing-risks panel: %s loan-months across %s loans",
                 f"{stats['total_rows']:,}", f"{n_loans:,}")
        for code, label in config.EVENT_TYPE_LABELS.items():
            n = stats["events"][code]
            log.info("      %-9s %s loans (%.2f%%)", label, f"{n:,}",
                     100.0 * n / max(n_loans, 1))
        n_default = stats["events"][config.EVENT_DEFAULT]
        if n_default:
            log.info("      prepayment:default event ratio = %.1f:1 — this is the "
                     "magnitude of the competing risk that 1 - S(t) ignores.",
                     stats["events"][config.EVENT_PREPAY] / n_default)
        if stats["refi_n"]:
            log.info("      refi_incentive: mean=%.3f  (PMMS joined)",
                     stats["refi_sum"] / stats["refi_n"])
        else:
            log.warning("      refi_incentive unavailable (no PMMS file) — the "
                        "prepayment hazard will be weakly identified.")
        for name in ("train", "oos", "oot"):
            log.info("      surv_%-6s %s rows / %s loans", name,
                     f"{stats['rows'][name]:,}", f"{stats['loans'][name]:,}")
    else:
        log.warning("  No survival chunks produced — surv_*.parquet not written.")

    # Clean up chunk files
    for f in CHUNK_DIR.glob("*.parquet"):
        f.unlink()
    CHUNK_DIR.rmdir()

    for f in sorted(OUT_DIR.iterdir()):
        log.info("  %-35s  (%s KB)", f.name, f"{f.stat().st_size / 1024:,.0f}")

    log.info("")
    log.info("=" * 65)
    log.info("Preprocessing complete.")
    log.info("  Next: python 02_pd_logistic_regression.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()