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

RAW_DIR   = Path("/kaggle/input/datasets/youssefmousaaid/freddie-mac-credit-risk/freddie_mac")
MACRO_DIR = Path("/kaggle/input/datasets/youssefmousaaid/freddie-mac-credit-risk/macro")
PROC_DIR  = Path("data/processed")
CHUNK_DIR = PROC_DIR / "chunks"
FIG_DIR   = Path("data/figures")


# =============================================================================
# TRAIN / OOS / OOT SPLIT
# =============================================================================

# OOT cutoff: thesis §1.5.1 — last ~3 years of data held out for temporal
# out-of-time (OOT) validation. Loans originated through 2020 have servicer
# performance history through ~2024, so setting the cutoff at mid-2017
# gives ~17 years in-sample and ~7 years OOT.
OOT_CUTOFF = pd.Timestamp("2017-06-01")

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
