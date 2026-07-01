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

import src.config as config

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
SVCG_USECOLS = [
    "loan_seq_num", "monthly_reporting_period", "current_upb",
    "delinquency_status", "loan_age", "remaining_months", "zero_balance_code",
    "zero_balance_date", "current_interest_rate", "mi_recoveries",
    "net_sale_proceeds", "non_mi_recoveries", "expenses", "actual_loss",
    "zero_balance_removal_upb", "delinquent_accrued_interest",
    "interest_bearing_upb",
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

    # Delinquency: 'X' = current (0 months past due), numeric strings otherwise
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
    hpi_change          : ratio of origination HPI to current HPI (PD)
    hpi_change_since_orig: same ratio stored separately for LGD
    ur_3m_lag           : unemployment rate lagged 3 months

    HPI strategy
    ------------
    Previously used a Python ``iterrows`` loop with a per-row dict cache —
    O(n) Python overhead for every row.  Replaced with two vectorized merges
    on a pre-built (zip3, year, quarter) lookup table: zero Python-level
    iteration regardless of dataset size.
    """
    df = merged.copy()

    # ── HPI change ratio ──────────────────────────────────────────────────
    if hpi is not None and "zip3" in df.columns and "orig_date" in df.columns:
        log.debug("  Computing HPI change ratios (vectorized merge) …")

        hpi_lookup = _build_hpi_lookup(hpi)

        # Derive join keys for origination date and current report date
        orig_keys = _hpi_keys(df["orig_date"],   df["zip3"]).add_suffix("_orig")
        curr_keys = _hpi_keys(df["report_date"],  df["zip3"]).add_suffix("_curr")

        # Attach keys to a slim working frame (preserves df index)
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

        df["hpi_orig"] = work["hpi_orig"].values
        df["hpi_curr"] = work["hpi_curr"].values

        # Ratio > 1 means prices have risen since origination (positive equity)
        # Ratio < 1 means prices have fallen (negative equity → higher default risk)
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
        ur_indexed = ur.set_index("date")["unemployment_rate"]  # noqa: F841
        lag_dates  = df["report_date"] - pd.DateOffset(months=3)
        # Nearest available observation (tolerance = 45 days)
        df["ur_3m_lag"] = pd.merge_asof(
            pd.DataFrame({"lag_date": lag_dates}).sort_values("lag_date"),
            ur.rename(columns={"date": "lag_date"}).sort_values("lag_date"),
            on="lag_date",
            direction="nearest",
            tolerance=pd.Timedelta(days=45),
        )["unemployment_rate"].values
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

    upb = defaults["zero_balance_removal_upb"].fillna(defaults["current_upb"])
    loss = pd.to_numeric(defaults["actual_loss"], errors="coerce").fillna(0)

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

    # ── Year-by-year processing ──────────────────────────────────────────
    log.info("")
    log.info("[2/4] Processing origination years …")
    years_found   = 0
    pd_row_total  = 0
    lgd_row_total = 0

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
        base_cols = [c for c in ["loan_seq_num", "report_date", "default_12m",
                                  "has_default", "current_upb",
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

        del merged, pd_chunk, lgd_chunk, onset_chunk
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

    # ── Combine chunks ───────────────────────────────────────────────────
    log.info("")
    log.info("[3/4] Combining chunks and splitting train / OOS / OOT …")

    pd_files    = sorted(CHUNK_DIR.glob("pd_*.parquet"))
    lgd_files   = sorted(CHUNK_DIR.glob("lgd_*.parquet"))
    onset_files = sorted(CHUNK_DIR.glob("onset_*.parquet"))

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