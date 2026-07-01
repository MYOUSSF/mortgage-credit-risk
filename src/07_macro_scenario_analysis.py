"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.6 — Macro Scenario Analysis (IFRS 9)
=============================================================================
Script  : 07_macro_scenario_analysis.py
Purpose : IFRS 9 ECL calculation under three probability-weighted macro
          scenarios, with full quarterly path scoring rather than a single
          peak-quarter approximation.

Design
------
  1. Quarterly scoring loop
     Each loan is scored at EVERY quarter of the macro path, not just
     the peak quarter. PD varies with the macro environment each quarter.

  2. At-risk pool shrinkage
     Loans that default in quarter q are removed from the at-risk pool
     in q+1. ECL is accumulated over a geometrically declining population.

  3. Recovery path
     All three scenarios include an explicit recovery phase (delta_k < 0)
     after the stress peak, with UR mean-reverting toward its long-run
     level rather than staying permanently elevated.

  4. Loan feature ageing
     loan_age is incremented each quarter. This allows the model to
     reflect that ageing loans have different risk profiles — e.g. seasoned
     loans approaching the peak default risk window (typically years 3-5)
     behave differently from newly originated ones.

  5. IFRS 9 ECL formula
     For each loan i and quarter q:

         ECL_i = sum_{q=1}^{Q}  PD_i(q) * SP_i(q-1) * LGD(q) * EAD_i * DF(q)

     where:
         PD_i(q)    = conditional default probability in quarter q
                      given macro conditions at q
         SP_i(q-1)  = survival probability to the START of quarter q
                    = product_{k=1}^{q-1} (1 - PD_i(k))
         LGD(q)     = scenario-conditional LGD (see below) — not a flat
                      assumption
         EAD_i      = exposure at default (current UPB, amortised)
         DF(q)      = discount factor = 1 / (1 + r)^(q/4)
                      where r = annual risk-free rate

  6. Scenario-conditional LGD
     LGD(q) scales with that scenario-quarter's collateral value
     (config.SCENARIOS' hpi_delta paths), via scenario_lgd(): recovery at
     foreclosure is assumed proportional to current HPI relative to
     origination, so LGD rises when HPI falls and falls when HPI
     appreciates. The base_lgd anchor (LGD at hpi_ratio == 1.0) comes from
     04_lgd_models.py's champion model — data/processed/lgd_champion_summary.csv
     — falling back to config.MACRO_LGD_ASSUMPTION if that file doesn't
     exist yet (run 04_lgd_models.py first for the model-derived value).

  7. Horizon flexibility
     ECL is computed at 12m, 24m, 36m, and lifetime (full path) horizons,
     matching IFRS 9 Stage 1, Stage 2 output requirements.

  8. Per-loan IFRS 9 staging
     Every loan is classified into Stage 1 / 2 / 3 (assign_ifrs9_stage()) via
     a SICR test — current lifetime PD vs a PD-at-origination proxy — plus
     the 30-DPD and 90-DPD regulatory backstops (IFRS 9 §5.5.9-5.5.11).
     Stage 1 loans are measured at the 12-month ECL horizon; Stage 2/3 loans
     at the lifetime horizon (compute_staged_ecl()). This is the core IFRS 9
     mechanic: applying one horizon uniformly to the whole portfolio (as the
     unstaged ecl_by_loan/ecl_summary outputs still do, for horizon
     comparison purposes) is not IFRS 9-compliant on its own.

Scenario assumptions
--------------------
  Base    (60%): UR stable, HPI +2% p.a., recovery by definition flat
  Adverse (30%): UR +3pp over 4Q, plateau 2Q, recovery over 6Q
  Severe  (10%): UR +6pp over 6Q, plateau 2Q, recovery over 8Q

  HPI shocks are symmetric: falls during stress, partial recovery thereafter.
  All paths are 20 quarters (5 years) long.

Inputs
------
  data/processed/pd_oos.parquet
  data/processed/pd_train.parquet
  data/processed/lgd_champion_summary.csv    (optional — 04_lgd_models.py;
                                               falls back to
                                               config.MACRO_LGD_ASSUMPTION)

Outputs
-------
  data/processed/ifrs9_ecl_by_loan.csv       — per-loan ECL at each horizon
  data/processed/ifrs9_ecl_summary.csv       — portfolio ECL summary
  data/processed/ifrs9_macro_paths.csv       — full quarterly macro paths
  data/processed/ifrs9_staged_ecl_by_loan.csv    — per-loan stage + staged ECL
  data/processed/ifrs9_staged_ecl_summary.csv    — portfolio staged ECL summary
  data/figures/ifrs9_macro_paths.png
  data/figures/ifrs9_pd_paths.png
  data/figures/ifrs9_ecl_by_horizon.png
  data/figures/ifrs9_survival_curves.png
  data/figures/ifrs9_stage_distribution.png
=============================================================================
"""

from __future__ import annotations

import gc
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed.")
    sys.exit(1)

import src.config as config

# =============================================================================
# LOGGING
# =============================================================================

config.configure_logging("macro_ifrs9.log")
log = logging.getLogger(__name__)


# =============================================================================
# GPU DETECTION
# =============================================================================

DEVICE, _N_GPUS = config.detect_gpu()

# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = config.PROC_DIR
FIG_DIR  = config.FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET      = config.TARGET_PD
SEED        = config.SEED
LGD         = config.MACRO_LGD_ASSUMPTION  # placeholder — replace with script 04 output
DISCOUNT_R  = config.DISCOUNT_R            # annual risk-free discount rate for ECL
N_QUARTERS  = config.N_QUARTERS            # full path length: 5 years

FEATURES: list[str] = config.PD_FEATURES
CAT_FEATURES = config.PD_CAT_FEATURES

PLT_STYLE: dict = config.PLT_STYLE

# =============================================================================
# SCENARIO DEFINITIONS  — 20-quarter paths with explicit recovery
# =============================================================================
#
# Each scenario specifies:
#   ur_delta  : quarterly increment to UR (pp). Negative = recovery.
#   hpi_delta : quarterly % change in HPI. Negative = price fall.
#
# Constraint: sum(ur_delta) defines the net UR change by Q20.
# All paths are exactly N_QUARTERS = 20 quarters long.
#
# Base    : UR flat throughout. HPI +0.5% per quarter (+2% p.a.).
# Adverse : UR rises +3pp over Q1-Q4, plateaus Q5-Q6, recovers Q7-Q14,
#           returns to +0.5pp above start by Q20.
#           HPI falls ~10%, partial recovery.
# Severe  : UR rises +6pp over Q1-Q6, plateaus Q7-Q8, recovers Q9-Q20,
#           returns to +1pp above start by Q20 (scarring effect).
#           HPI falls ~25%, partial recovery.
#
# Defined once in config.py (config.SCENARIOS) alongside N_QUARTERS, since
# the scenario paths and the path length are coupled. Replace them there
# with your institution's official ICAAP stress scenarios.

SCENARIOS: dict[str, dict] = config.SCENARIOS


# =============================================================================
# MACRO PATH CONSTRUCTION
# =============================================================================

def build_macro_paths(current_ur: float = 4.0,
                       current_hpi: float = 1.0) -> pd.DataFrame:
    """
    Build full N_QUARTERS quarterly macro paths for each scenario.

    Returns DataFrame indexed by (scenario, quarter) with:
        ur        : unemployment rate (%)
        hpi_ratio : HPI ratio relative to origination value
    """
    rows = []
    for scenario, cfg in SCENARIOS.items():
        ur  = current_ur
        hpi = current_hpi
        rows.append({"scenario": scenario, "quarter": 0,
                      "ur": ur, "hpi_ratio": hpi})

        for q in range(N_QUARTERS):
            ur  = max(ur  + cfg["ur_delta"][q],  0.5)
            hpi = hpi * (1 + cfg["hpi_delta"][q] / 100)
            hpi = max(hpi, 0.1)   # floor: HPI cannot go negative
            rows.append({"scenario": scenario, "quarter": q + 1,
                          "ur": round(ur, 4), "hpi_ratio": round(hpi, 6)})

    return pd.DataFrame(rows)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def _encode_categoricals(train: pd.DataFrame,
                           oos: pd.DataFrame,
                           cat_cols: list[str]
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in cat_cols:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        all_vals = pd.concat([train[col], oos[col]]).fillna("missing").astype(str)
        le.fit(all_vals)
        known    = set(le.classes_)
        fallback = le.classes_[0]
        for df in [train, oos]:
            df[col] = le.transform(
                df[col].fillna("missing").astype(str)
                .map(lambda x, k=known, fb=fallback: x if x in k else fb)
            )
    return train, oos


def prepare(train: pd.DataFrame,
            oos:   pd.DataFrame) -> tuple:
    feats = [f for f in FEATURES if f in train.columns]
    train_enc, oos_enc = _encode_categoricals(
        train.copy(), oos.copy(), CAT_FEATURES
    )
    feats = [f for f in feats if train_enc[f].notna().any()]

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(train_enc[feats])
    X_oo = imputer.transform(oos_enc[feats])

    return (X_tr, X_oo,
            train_enc[TARGET].values,
            oos_enc[TARGET].values,
            imputer, feats, oos_enc)


def retrain_xgboost(X_tr, y_tr, X_oo, y_oo) -> XGBClassifier:
    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = neg / max(pos, 1)
    log.info("  scale_pos_weight = %.1f", spw)
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.5, colsample_bytree=0.8, min_child_weight=50,
        gamma=1.0, reg_alpha=0.1, reg_lambda=1.0, eval_metric="auc",
        tree_method="hist", device=DEVICE, random_state=SEED,
        early_stopping_rounds=20, scale_pos_weight=spw,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_oo, y_oo)], verbose=50)
    log.info("  Best iteration: %d", xgb.best_iteration)
    return xgb


# =============================================================================
# SCENARIO-CONDITIONAL LGD
# =============================================================================

def scenario_lgd(base_lgd: float, hpi_ratio: float) -> float:
    """
    LGD conditional on collateral value under macro stress.

    Recovery at foreclosure is assumed to scale linearly with current
    collateral value relative to the anchor point (hpi_ratio == 1.0
    reproduces base_lgd exactly):

        recovery(q) = (1 - base_lgd) * hpi_ratio(q)
        LGD(q)      = clip(1 - recovery(q), 0, 1)

    So a Severe-scenario HPI trough of hpi_ratio=0.74 (-26%) raises LGD
    above base_lgd, while HPI appreciation (hpi_ratio > 1.0) lowers it.
    base_lgd is the champion LGD model's mean predicted LGD from
    04_lgd_models.py (data/processed/lgd_champion_summary.csv), falling
    back to config.MACRO_LGD_ASSUMPTION when that file doesn't exist yet.
    """
    recovery = (1.0 - base_lgd) * hpi_ratio
    return float(np.clip(1.0 - recovery, 0.0, 1.0))


# =============================================================================
# AMORTIZED EAD
# =============================================================================

def amortized_ead(oos_df: pd.DataFrame, n_quarters: int) -> np.ndarray:
    """
    Project each loan's exposure at default forward across the macro path
    via standard declining-balance mortgage amortization, rather than
    holding origination UPB flat for all n_quarters.

    Anchored at the loan's current_upb (falling back to orig_upb when
    absent — e.g. parquets built before this column was added, or
    synthetic test data), amortized at current_interest_rate (falling
    back to orig_interest_rate) over remaining_months (falling back to
    max(360 - loan_age, 12), a 30-year-term proxy):

        B(m) = B0 * [(1+i)^N0 - (1+i)^m] / [(1+i)^N0 - 1]   0 < m <= N0
        B(m) = B0 * (N0 - m) / N0                            i == 0
        B(m) = 0                                             m > N0

    where B0 = anchor balance, i = monthly rate, N0 = remaining term in
    months, m = months elapsed (quarter * 3).

    Returns an (n_loans, n_quarters) array, column q-1 = EAD at quarter q.
    """
    n_loans = len(oos_df)

    b0 = (oos_df["current_upb"].values if "current_upb" in oos_df.columns
          else oos_df["orig_upb"].values if "orig_upb" in oos_df.columns
          else np.full(n_loans, 200_000.0))

    annual_rate = (oos_df["current_interest_rate"].values
                   if "current_interest_rate" in oos_df.columns
                   else oos_df["orig_interest_rate"].values
                   if "orig_interest_rate" in oos_df.columns
                   else np.full(n_loans, 6.0))
    annual_rate = np.nan_to_num(annual_rate, nan=6.0)
    i = annual_rate / 1200.0

    if "remaining_months" in oos_df.columns:
        n0 = oos_df["remaining_months"].values
    elif "loan_age" in oos_df.columns:
        n0 = np.maximum(360.0 - oos_df["loan_age"].values, 12.0)
    else:
        n0 = np.full(n_loans, 360.0)
    n0 = np.nan_to_num(n0, nan=360.0)
    n0 = np.maximum(n0, 1.0)

    b0 = np.nan_to_num(b0, nan=200_000.0)

    ead = np.zeros((n_loans, n_quarters))
    for q in range(1, n_quarters + 1):
        m = q * 3
        matured = m > n0
        zero_rate = i <= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            growth_n0 = (1 + i) ** n0
            growth_m  = (1 + i) ** np.minimum(m, n0)
            declining = b0 * (growth_n0 - growth_m) / (growth_n0 - 1)
        straight_line = b0 * (n0 - np.minimum(m, n0)) / n0
        balance = np.where(zero_rate, straight_line, declining)
        balance = np.where(matured, 0.0, balance)
        ead[:, q - 1] = np.clip(balance, 0.0, None)

    return ead


# =============================================================================
# QUARTERLY SCORING ENGINE
# =============================================================================

def score_quarter(xgb:       XGBClassifier,
                  oos_df:    pd.DataFrame,
                  macro_row: pd.Series,
                  feats:     list[str],
                  imputer:   SimpleImputer,
                  quarter:   int) -> np.ndarray:
    """
    Score the OOS portfolio under one quarter's macro conditions,
    with loan_age incremented by `quarter` months (3 months per quarter).

    Parameters
    ----------
    quarter : number of quarters elapsed since t=0.
              Used to age loan_age by quarter*3 months.

    Returns
    -------
    pd_q : (n_loans,) array of conditional PD for this quarter.
    """
    df_q = oos_df.copy()

    # Apply macro overrides
    if "ur_3m_lag" in df_q.columns:
        df_q["ur_3m_lag"] = macro_row["ur"]
    if "hpi_change" in df_q.columns:
        df_q["hpi_change"] = macro_row["hpi_ratio"]

    # Age the loan: each quarter = 3 additional months
    if "loan_age" in df_q.columns:
        df_q["loan_age"] = df_q["loan_age"] + quarter * 3

    X_q = imputer.transform(df_q[[f for f in feats if f in df_q.columns]])
    return xgb.predict_proba(X_q)[:, 1]


# =============================================================================
# IFRS 9 ECL COMPUTATION
# =============================================================================

def compute_ifrs9_ecl(xgb:       XGBClassifier,
                       oos_df:    pd.DataFrame,
                       macro_df:  pd.DataFrame,
                       imputer:   SimpleImputer,
                       feats:     list[str],
                       horizons:  list[int] = [4, 8, 12, N_QUARTERS],
                       base_lgd:  float = LGD,
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute IFRS 9 ECL for each loan under each scenario.

    The ECL at horizon H for loan i under scenario s is:

        ECL_i^s(H) = sum_{q=1}^{H}  PD_i^s(q) * SP_i^s(q-1) * LGD^s(q) * EAD_i(q) * DF(q)

    where:
        PD_i^s(q)  = conditional PD in quarter q under scenario s
        SP_i^s(q-1) = P(survived quarters 1..q-1)
                    = product_{k=1}^{q-1} (1 - PD_i^s(k))
        SP_i^s(0)  = 1  (all loans survive to start)
        LGD^s(q)   = scenario_lgd(base_lgd, hpi_ratio) — conditional on that
                     scenario-quarter's collateral value, not a flat constant
        EAD_i(q)   = amortized_ead() — declining balance projected from
                     current_upb via standard mortgage amortization, not a
                     flat orig_upb held constant across all quarters
        DF(q)      = 1 / (1 + DISCOUNT_R)^(q/4)  quarterly discount factor

    Parameters
    ----------
    horizons : list of quarters at which to report ECL.
               e.g. [4, 8, 12, 20] = 1Y, 2Y, 3Y, lifetime
    base_lgd : LGD at hpi_ratio == 1.0 (no HPI stress) — the champion LGD
               model's mean predicted LGD (04_lgd_models.py), or
               config.MACRO_LGD_ASSUMPTION as a fallback.

    Returns
    -------
    ecl_by_loan : DataFrame of per-loan ECL at each horizon under each scenario
    ecl_summary : DataFrame of portfolio-level ECL and weighted ECL
    """
    n_loans  = len(oos_df)
    ead_mat  = amortized_ead(oos_df, N_QUARTERS)   # (n_loans, N_QUARTERS): EAD_i(q)
    orig_upb = oos_df["orig_upb"].values if "orig_upb" in oos_df.columns \
               else np.full(n_loans, 200_000.0)

    # Discount factors: DF(q) = 1/(1+r)^(q/4)
    discount = np.array([1.0 / (1 + DISCOUNT_R) ** (q / 4)
                          for q in range(N_QUARTERS + 1)])

    all_results = {}   # scenario -> (n_loans, N_QUARTERS) PD matrix
    all_sp      = {}   # scenario -> (n_loans, N_QUARTERS+1) survival prob matrix
    all_lgd     = {}   # scenario -> (N_QUARTERS,) scenario-conditional LGD vector

    for scenario, cfg in SCENARIOS.items():
        log.info("  Computing quarterly PDs: %s ...", scenario)
        scenario_macro = macro_df[macro_df["scenario"] == scenario].sort_values("quarter")

        pd_matrix  = np.zeros((n_loans, N_QUARTERS))   # PD_i(q)  q=1..N_QUARTERS
        sp_matrix  = np.ones((n_loans, N_QUARTERS + 1)) # SP_i(q)  q=0..N_QUARTERS
        lgd_vector = np.zeros(N_QUARTERS)               # LGD(q)   q=1..N_QUARTERS

        for q in range(1, N_QUARTERS + 1):
            macro_row = scenario_macro[scenario_macro["quarter"] == q].iloc[0]

            # Conditional PD for this quarter
            pd_q = score_quarter(xgb, oos_df, macro_row, feats, imputer, q)
            pd_q = np.clip(pd_q, 0.0, 1.0)
            pd_matrix[:, q - 1] = pd_q

            # Survival probability to END of quarter q
            # SP(q) = SP(q-1) * (1 - PD(q))
            sp_matrix[:, q] = sp_matrix[:, q - 1] * (1 - pd_q)

            # LGD conditional on this scenario-quarter's collateral value
            lgd_vector[q - 1] = scenario_lgd(base_lgd, macro_row["hpi_ratio"])

            if q % 4 == 0:
                log.info("    %s Q%02d: mean PD=%.4f%%  mean survival=%.4f%%  LGD=%.4f",
                          scenario, q,
                          pd_q.mean() * 100,
                          sp_matrix[:, q].mean() * 100,
                          lgd_vector[q - 1])

        all_results[scenario] = pd_matrix
        all_sp[scenario]      = sp_matrix
        all_lgd[scenario]     = lgd_vector

    # ── Build per-loan ECL at each horizon ────────────────────────────────────
    loan_id = oos_df["loan_seq_num"].values \
              if "loan_seq_num" in oos_df.columns \
              else np.arange(n_loans)

    ecl_rows = {"loan_seq_num": loan_id, "orig_upb": orig_upb}

    horizon_labels = {h: f"{h//4}Y" if h % 4 == 0 else f"Q{h}" for h in horizons}

    for scenario in SCENARIOS:
        pd_mat  = all_results[scenario]   # (n_loans, N_QUARTERS)
        sp_mat  = all_sp[scenario]        # (n_loans, N_QUARTERS+1)
        lgd_vec = all_lgd[scenario]       # (N_QUARTERS,)

        for h in horizons:
            h_idx = min(h, N_QUARTERS)
            # ECL_i(H) = sum_{q=1}^{H} PD_i(q) * SP_i(q-1) * LGD(q) * EAD_i(q) * DF(q)
            ecl_i = np.zeros(n_loans)
            for q in range(1, h_idx + 1):
                ecl_i += (pd_mat[:, q - 1]     # PD in quarter q
                          * sp_mat[:, q - 1]    # survived to start of q
                          * lgd_vec[q - 1]      # LGD conditional on this quarter
                          * ead_mat[:, q - 1]   # amortized EAD at this quarter
                          * discount[q])

            col = f"ecl_{scenario.lower()}_{horizon_labels[h]}"
            ecl_rows[col] = ecl_i

    ecl_by_loan = pd.DataFrame(ecl_rows)

    # ── Portfolio summary ─────────────────────────────────────────────────────
    summary_rows = []
    for scenario, cfg in SCENARIOS.items():
        row = {"scenario": cfg["label"], "weight": cfg["weight"]}
        for h in horizons:
            hl = horizon_labels[h]
            col = f"ecl_{scenario.lower()}_{hl}"
            total_ecl = ecl_by_loan[col].sum() / 1e6
            row[f"total_ecl_{hl}_$M"] = round(total_ecl, 3)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Probability-weighted ECL
    weighted_row = {"scenario": "PROBABILITY-WEIGHTED", "weight": 1.0}
    for h in horizons:
        hl = horizon_labels[h]
        w_ecl = 0.0
        for scenario, cfg in SCENARIOS.items():
            col = f"total_ecl_{hl}_$M"
            w_ecl += cfg["weight"] * summary_df.loc[
                summary_df["scenario"] == cfg["label"], col
            ].values[0]
        weighted_row[f"total_ecl_{hl}_$M"] = round(w_ecl, 3)
    summary_df = pd.concat([summary_df,
                             pd.DataFrame([weighted_row])],
                            ignore_index=True)

    return ecl_by_loan, summary_df, all_results, all_sp


# =============================================================================
# IFRS 9 STAGING  (per-loan Stage 1 / 2 / 3)
# =============================================================================

def compute_origination_pd(xgb:     XGBClassifier,
                            oos_df:  pd.DataFrame,
                            feats:   list[str],
                            imputer: SimpleImputer) -> np.ndarray:
    """
    Proxy for PD at initial recognition — the SICR baseline IFRS 9 §5.5.9
    compares the current PD against.

    Scores every loan with loan_age forced to 0, holding every other
    feature (credit score, CLTV, DTI, current macro state, etc.) at its
    current value. This is not the loan's actual historical origination-time
    PD — that would require the model and macro state as they existed at
    underwriting, which this pipeline doesn't persist per loan — it's the
    closest recoverable proxy from a single retrained snapshot model, in the
    same spirit as amortized_ead()'s remaining_months fallback.
    """
    df0 = oos_df.copy()
    if "loan_age" in df0.columns:
        df0["loan_age"] = 0
    X0 = imputer.transform(df0[[f for f in feats if f in df0.columns]])
    return xgb.predict_proba(X0)[:, 1]


def assign_ifrs9_stage(pd_current:         np.ndarray,
                        pd_origination:     np.ndarray,
                        delinquency_status: np.ndarray) -> np.ndarray:
    """
    Per-loan IFRS 9 Stage 1 / 2 / 3 classification (IFRS 9 §5.5).

    Stage 2 (SICR) triggers on EITHER:
      - Relative test: pd_current / pd_origination >= config.SICR_PD_RATIO,
        gated by an absolute floor (config.SICR_PD_ABS_FLOOR) so a tiny
        move off a near-zero starting PD can't trip the ratio test alone.
      - 30-DPD backstop: delinquency_status >= config.STAGE2_DPD_MONTHS —
        the rebuttable presumption in §5.5.11.

    Stage 3 (credit-impaired) overrides Stage 2 once delinquency reaches the
    90-DPD regulatory default trigger (config.STAGE3_DPD_MONTHS) — the same
    trigger used to onset the LGD workout period (LGD_ONSET_DPD_MONTHS).

    Returns an (n_loans,) int array of 1 / 2 / 3.
    """
    pd_origination_safe    = np.maximum(pd_origination, 1e-6)
    relative_deterioration = pd_current / pd_origination_safe
    absolute_deterioration = pd_current - pd_origination

    sicr = (
        (relative_deterioration >= config.SICR_PD_RATIO)
        & (absolute_deterioration >= config.SICR_PD_ABS_FLOOR)
    ) | (delinquency_status >= config.STAGE2_DPD_MONTHS)

    stage = np.where(sicr, 2, 1)
    stage = np.where(delinquency_status >= config.STAGE3_DPD_MONTHS, 3, stage)
    return stage.astype(int)


def compute_staged_ecl(ecl_by_loan:    pd.DataFrame,
                        stage:          np.ndarray,
                        horizon_labels: dict[int, str]) -> pd.DataFrame:
    """
    IFRS 9 staged ECL: Stage 1 loans are measured at the 12-month ECL
    horizon; Stage 2/3 loans (SICR-triggered or already credit-impaired) are
    measured at the lifetime horizon (IFRS 9 §5.5.3-5.5.4) — the actual core
    staging mechanic, as opposed to compute_ifrs9_ecl()'s horizon columns,
    which apply one horizon uniformly across the whole portfolio.
    """
    stage12_label   = horizon_labels[4]
    lifetime_label  = horizon_labels[max(horizon_labels)]

    out = pd.DataFrame({
        "loan_seq_num": ecl_by_loan["loan_seq_num"].values,
        "stage":        stage,
    })
    for scenario, cfg in SCENARIOS.items():
        col_12m  = f"ecl_{scenario.lower()}_{stage12_label}"
        col_life = f"ecl_{scenario.lower()}_{lifetime_label}"
        out[f"staged_ecl_{scenario.lower()}"] = np.where(
            stage == 1, ecl_by_loan[col_12m].values, ecl_by_loan[col_life].values
        )

    out["staged_ecl_weighted"] = sum(
        cfg["weight"] * out[f"staged_ecl_{scenario.lower()}"]
        for scenario, cfg in SCENARIOS.items()
    )
    return out


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_macro_paths(macro_df: pd.DataFrame) -> None:
    """Four-panel chart: UR path, HPI path, UR delta, HPI delta."""
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        "Macro Scenario Paths — Full 20-Quarter (5-Year) Projection\n"
        "Including explicit recovery phase — replaces original peak-only approximation",
        fontsize=12, fontweight="bold", color="white"
    )

    quarters = sorted(macro_df["quarter"].unique())
    x_labels = [f"Q{q}" if q % 4 == 0 else "" for q in quarters]

    for scenario, cfg in SCENARIOS.items():
        sub = macro_df[macro_df["scenario"] == scenario].sort_values("quarter")
        col = cfg["color"]
        lbl = f"{cfg['label']}  (w={cfg['weight']:.0%})"

        axes[0,0].plot(sub["quarter"], sub["ur"],
                       color=col, linewidth=2.5, marker="o", markersize=3, label=lbl)
        axes[0,1].plot(sub["quarter"], (sub["hpi_ratio"] - 1) * 100,
                       color=col, linewidth=2.5, marker="o", markersize=3, label=cfg["label"])

        # Quarterly increments
        ur_vals  = sub["ur"].values
        hpi_vals = (sub["hpi_ratio"] - 1).values * 100
        ur_delta  = np.diff(ur_vals,  prepend=ur_vals[0])
        hpi_delta = np.diff(hpi_vals, prepend=hpi_vals[0])
        axes[1,0].bar(sub["quarter"] + (list(SCENARIOS.keys()).index(scenario) - 1) * 0.25,
                       ur_delta, width=0.25, color=col, alpha=0.7, label=cfg["label"])
        axes[1,1].bar(sub["quarter"] + (list(SCENARIOS.keys()).index(scenario) - 1) * 0.25,
                       hpi_delta, width=0.25, color=col, alpha=0.7, label=cfg["label"])

    for ax, title, ylabel, fmt in [
        (axes[0,0], "Unemployment Rate Path", "UR (%)", "%.1f%%"),
        (axes[0,1], "HPI Change from Origination", "HPI Change (%)", "%.1f%%"),
        (axes[1,0], "Quarterly UR Increment (delta_k)", "delta UR (pp)", "%+.2f"),
        (axes[1,1], "Quarterly HPI Increment", "delta HPI (%)", "%+.1f%%"),
    ]:
        ax.set_title(title, color="#CBD5E1", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Quarter", fontsize=9)
        ax.set_xticks(quarters[::2])
        ax.set_xticklabels([f"Q{q}" for q in quarters[::2]], fontsize=7)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(fmt))

    axes[0,0].axhline(4.0, color="#4B5563", linewidth=1, linestyle=":", alpha=0.6)
    axes[0,1].axhline(0,   color="#4B5563", linewidth=1, linestyle="--")
    axes[1,0].axhline(0,   color="#4B5563", linewidth=1, linestyle="--")
    axes[1,1].axhline(0,   color="#4B5563", linewidth=1, linestyle="--")

    fig.tight_layout()
    path = FIG_DIR / "ifrs9_macro_paths.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_pd_paths(all_results: dict) -> None:
    """Portfolio mean PD path per scenario across all 20 quarters."""
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        "Portfolio Mean Conditional PD — Quarterly Path per Scenario\n"
        "Left: PD level   Right: PD relative to baseline",
        fontsize=12, fontweight="bold", color="white"
    )

    quarters = np.arange(1, N_QUARTERS + 1)
    base_pd  = all_results["Base"].mean(axis=0)

    for scenario, cfg in SCENARIOS.items():
        pd_mean = all_results[scenario].mean(axis=0)
        axes[0].plot(quarters, pd_mean * 100,
                     color=cfg["color"], linewidth=2.5,
                     label=f"{cfg['label']} (w={cfg['weight']:.0%})")
        axes[1].plot(quarters, (pd_mean - base_pd) * 100,
                     color=cfg["color"], linewidth=2.5,
                     label=cfg["label"])

    axes[0].set_ylabel("Mean Portfolio PD (%)", fontsize=10)
    axes[0].set_title("Conditional PD by Quarter", color="#CBD5E1", fontsize=10)
    axes[0].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f%%"))

    axes[1].axhline(0, color="#4B5563", linewidth=1, linestyle="--")
    axes[1].set_ylabel("PD Uplift vs Base (pp)", fontsize=10)
    axes[1].set_title("PD Uplift Relative to Base Scenario", color="#CBD5E1", fontsize=10)
    axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter("%+.3f%%"))

    for ax in axes:
        ax.set_xlabel("Quarter", fontsize=10)
        ax.set_xticks(quarters[::2])
        ax.set_xticklabels([f"Q{q}" for q in quarters[::2]], fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Mark recovery onset with vertical lines
        ax.axvline(4,  color="#F59E0B", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.axvline(6,  color="#EF4444", linewidth=0.8, linestyle=":", alpha=0.5)

    fig.tight_layout()
    path = FIG_DIR / "ifrs9_pd_paths.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_survival_curves(all_sp: dict) -> None:
    """Portfolio mean survival probability path — the shrinking at-risk pool."""
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(
        "Portfolio Mean Survival Probability — At-Risk Pool Shrinkage\n"
        "SP(q) = product_{k=1}^{q} (1 - PD(k))   reflects cumulative defaults removed",
        fontsize=12, fontweight="bold", color="white"
    )

    quarters = np.arange(0, N_QUARTERS + 1)
    for scenario, cfg in SCENARIOS.items():
        sp_mean = all_sp[scenario].mean(axis=0)
        ax.plot(quarters, sp_mean * 100,
                color=cfg["color"], linewidth=2.5,
                label=f"{cfg['label']}  (w={cfg['weight']:.0%})")
        ax.fill_between(quarters, sp_mean * 100, 100,
                         color=cfg["color"], alpha=0.05)

    ax.set_xlabel("Quarter", fontsize=10)
    ax.set_ylabel("Mean Survival Probability (%)", fontsize=10)
    ax.set_xticks(quarters[::2])
    ax.set_xticklabels([f"Q{q}" for q in quarters[::2]], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "ifrs9_survival_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_ecl_by_horizon(summary_df: pd.DataFrame,
                          horizons:   list[int]) -> None:
    """Grouped bar chart: ECL at each horizon for each scenario."""
    plt.rcParams.update(PLT_STYLE)
    horizon_labels = {h: f"{h//4}Y" if h % 4 == 0 else f"Q{h}" for h in horizons}

    scen_rows = summary_df[summary_df["scenario"] != "PROBABILITY-WEIGHTED"].copy()
    weighted  = summary_df[summary_df["scenario"] == "PROBABILITY-WEIGHTED"].copy()

    n_horizons = len(horizons)
    n_scenarios = len(scen_rows)
    width = 0.18
    x = np.arange(n_horizons)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "IFRS 9 ECL by Horizon and Scenario\n"
        "ECL = sum_{q} PD(q) * SP(q-1) * LGD(q) * EAD * DF(q)   "
        "with full quarterly path scoring and at-risk pool shrinkage",
        fontsize=11, fontweight="bold", color="white"
    )

    # Left: ECL per scenario
    colors_map = {cfg["label"]: cfg["color"] for cfg in SCENARIOS.values()}
    for i, (_, row) in enumerate(scen_rows.iterrows()):
        vals = [row.get(f"total_ecl_{horizon_labels[h]}_$M", 0) for h in horizons]
        offset = (i - n_scenarios / 2 + 0.5) * width
        bars = axes[0].bar(x + offset, vals, width=width,
                            color=colors_map.get(row["scenario"], "#38BDF8"),
                            alpha=0.85, label=row["scenario"])
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.002,
                          f"${v:.2f}M", ha="center", va="bottom",
                          fontsize=7.5, color="#E2E8F0")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([horizon_labels[h] for h in horizons], fontsize=10)
    axes[0].set_ylabel("Total ECL ($M)", fontsize=10)
    axes[0].set_title("Scenario ECL by Horizon", color="#CBD5E1", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: probability-weighted ECL by horizon
    if not weighted.empty:
        w_vals = [weighted[f"total_ecl_{horizon_labels[h]}_$M"].values[0]
                   for h in horizons]
        bars = axes[1].bar(x, w_vals, width=0.5, color="#3B82F6", alpha=0.85)
        for bar, v in zip(bars, w_vals):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.002,
                          f"${v:.3f}M", ha="center", va="bottom",
                          fontsize=9, color="white", fontweight="bold")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([horizon_labels[h] for h in horizons], fontsize=10)
    axes[1].set_ylabel("Probability-Weighted ECL ($M)", fontsize=10)
    axes[1].set_title(
        f"Probability-Weighted ECL by Horizon\n"
        f"Base {SCENARIOS['Base']['weight']:.0%} / "
        f"Adverse {SCENARIOS['Adverse']['weight']:.0%} / "
        f"Severe {SCENARIOS['Severe']['weight']:.0%}",
        color="#CBD5E1", fontsize=10
    )
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "ifrs9_ecl_by_horizon.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_stage_distribution(staged_df: pd.DataFrame) -> None:
    """
    Two-panel chart: portfolio count by IFRS 9 stage, and probability-
    weighted staged ECL contribution by stage — the standard IFRS 9
    disclosure pairing (staging distribution alongside its ECL impact).
    """
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "IFRS 9 Staging — Portfolio Distribution and ECL Contribution\n"
        "Stage 1: 12m ECL   Stage 2 (SICR) / Stage 3 (credit-impaired): lifetime ECL",
        fontsize=12, fontweight="bold", color="white"
    )

    stage_colors = {1: "#10B981", 2: "#F59E0B", 3: "#EF4444"}
    stage_labels = {1: "Stage 1", 2: "Stage 2 (SICR)", 3: "Stage 3 (Impaired)"}

    counts = staged_df["stage"].value_counts().sort_index()
    stages = counts.index.tolist()
    axes[0].bar([stage_labels[s] for s in stages], counts.values,
                color=[stage_colors[s] for s in stages], alpha=0.85)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}\n({v / counts.sum():.1%})",
                     ha="center", va="bottom", fontsize=9, color="#E2E8F0")
    axes[0].set_ylabel("Number of Loans", fontsize=10)
    axes[0].set_title("Portfolio Distribution by Stage", color="#CBD5E1", fontsize=10)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    ecl_by_stage = (staged_df.groupby("stage")["staged_ecl_weighted"].sum() / 1e6)
    axes[1].bar([stage_labels[s] for s in ecl_by_stage.index],
                ecl_by_stage.values,
                color=[stage_colors[s] for s in ecl_by_stage.index], alpha=0.85)
    for i, v in enumerate(ecl_by_stage.values):
        axes[1].text(i, v, f"${v:.3f}M", ha="center", va="bottom",
                     fontsize=9, color="#E2E8F0")
    axes[1].set_ylabel("Probability-Weighted ECL ($M)", fontsize=10)
    axes[1].set_title("ECL Contribution by Stage", color="#CBD5E1", fontsize=10)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "ifrs9_stage_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def load_base_lgd() -> tuple[float, str]:
    """
    Load the base LGD anchor from 04_lgd_models.py's champion selection.

    Falls back to config.MACRO_LGD_ASSUMPTION (with a warning) if
    lgd_champion_summary.csv doesn't exist yet — the same
    optional-input-with-fallback pattern used for the macro data files in
    01_data_preprocessing.py.

    Returns (base_lgd, source_description) for logging/traceability.
    """
    path = PROC_DIR / "lgd_champion_summary.csv"
    if not path.exists():
        log.warning(
            "  %s not found — run 04_lgd_models.py first to anchor LGD on "
            "an actual fitted model. Falling back to config.MACRO_LGD_ASSUMPTION.",
            path,
        )
        return config.MACRO_LGD_ASSUMPTION, "config.MACRO_LGD_ASSUMPTION (fallback)"

    champion = pd.read_csv(path)
    if champion.empty or pd.isna(champion.iloc[0]["anchor_mean_lgd"]):
        log.warning(
            "  %s has no usable anchor_mean_lgd — falling back to "
            "config.MACRO_LGD_ASSUMPTION.", path,
        )
        return config.MACRO_LGD_ASSUMPTION, "config.MACRO_LGD_ASSUMPTION (fallback)"

    row = champion.iloc[0]
    source = f"{row['champion_model']} champion ({row['anchor_split']} mean predicted LGD)"
    return float(row["anchor_mean_lgd"]), source


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Ch.6 — IFRS 9 ECL (Full Path)")
    log.info("=" * 65)

    # Current macro state
    current_ur = 4.0

    # ── Load ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("[1/5] Loading data ...")
    train = pd.read_parquet(PROC_DIR / "pd_train.parquet")
    oos   = pd.read_parquet(PROC_DIR / "pd_oos.parquet")
    log.info("  Train: %s  |  OOS: %s",
             f"{len(train):,}", f"{len(oos):,}")

    # ── Prepare ───────────────────────────────────────────────────────────
    log.info("")
    log.info("[2/5] Preparing data and retraining XGBoost ...")
    X_tr, X_oo, y_tr, y_oo, imputer, feats, oos_enc = prepare(train, oos)
    xgb = retrain_xgboost(X_tr, y_tr, X_oo, y_oo)
    del X_tr, X_oo, y_tr, y_oo
    gc.collect()

    # Baseline PD
    X_base = imputer.transform(oos_enc[[f for f in feats if f in oos_enc.columns]])
    pd_baseline = xgb.predict_proba(X_base)[:, 1].mean()
    log.info("  Baseline portfolio mean PD: %.4f%%", pd_baseline * 100)
    del X_base
    gc.collect()

    # ── Macro paths ───────────────────────────────────────────────────────
    log.info("")
    log.info("[3/5] Building %d-quarter macro paths ...", N_QUARTERS)
    macro_df = build_macro_paths(current_ur=current_ur)
    macro_df.to_csv(PROC_DIR / "ifrs9_macro_paths.csv", index=False)

    # Log path summary
    for scenario in SCENARIOS:
        sub = macro_df[macro_df["scenario"] == scenario]
        peak_q   = sub.loc[sub["ur"].idxmax(), "quarter"]
        peak_ur  = sub["ur"].max()
        final_ur = sub[sub["quarter"] == N_QUARTERS]["ur"].values[0]
        log.info("  %-10s  peak UR=%.1f%% at Q%d  |  final UR=%.1f%%",
                  scenario, peak_ur, peak_q, final_ur)

    plot_macro_paths(macro_df)

    # ── IFRS 9 ECL ────────────────────────────────────────────────────────
    log.info("")
    log.info("[4/6] Computing IFRS 9 ECL with full quarterly path scoring ...")
    log.info("  Horizons: 1Y (Q4), 2Y (Q8), 3Y (Q12), lifetime (%dQ)", N_QUARTERS)

    base_lgd, lgd_source = load_base_lgd()
    log.info("  Base LGD: %.4f (%s)  |  Discount rate: %.1f%% p.a.",
             base_lgd, lgd_source, DISCOUNT_R * 100)
    log.info("  LGD is scenario-conditional per quarter via scenario_lgd() "
             "(recovery scales with HPI, not a flat assumption)")
    log.info("  Loan ageing: loan_age incremented 3 months per quarter")
    log.info("  At-risk pool: SP(q) = product_{k=1}^{q} (1 - PD(k))")

    horizons = [4, 8, 12, N_QUARTERS]
    ecl_by_loan, ecl_summary, all_results, all_sp = compute_ifrs9_ecl(
        xgb, oos_enc, macro_df, imputer, feats, horizons=horizons, base_lgd=base_lgd
    )

    ecl_by_loan.to_csv(PROC_DIR / "ifrs9_ecl_by_loan.csv", index=False)
    ecl_summary.to_csv(PROC_DIR / "ifrs9_ecl_summary.csv", index=False)

    log.info("")
    log.info("  IFRS 9 ECL Summary:")
    log.info("\n%s", ecl_summary.to_string(index=False))

    # ── IFRS 9 staging ────────────────────────────────────────────────────
    log.info("")
    log.info("[5/6] Classifying loans into IFRS 9 Stage 1 / 2 / 3 ...")

    pd_origination = compute_origination_pd(xgb, oos_enc, feats, imputer)
    X_base = imputer.transform(oos_enc[[f for f in feats if f in oos_enc.columns]])
    pd_current = xgb.predict_proba(X_base)[:, 1]
    del X_base
    gc.collect()

    delinquency_status = (
        oos_enc["delinquency_status"].fillna(0).values
        if "delinquency_status" in oos_enc.columns
        else np.zeros(len(oos_enc))
    )
    stage = assign_ifrs9_stage(pd_current, pd_origination, delinquency_status)

    horizon_labels = {h: f"{h//4}Y" if h % 4 == 0 else f"Q{h}" for h in horizons}
    staged_df = compute_staged_ecl(ecl_by_loan, stage, horizon_labels)
    staged_df["pd_origination"] = pd_origination
    staged_df["pd_current"] = pd_current

    stage_counts = pd.Series(stage).value_counts().sort_index()
    for s in [1, 2, 3]:
        n = int(stage_counts.get(s, 0))
        log.info("  Stage %d: %s loans (%.2f%%)", s, f"{n:,}", n / len(stage) * 100)

    staged_summary = pd.DataFrame([{
        "stage": s,
        "n_loans": int((stage == s).sum()),
        "pct_loans": round(float((stage == s).mean() * 100), 3),
        "total_staged_ecl_weighted_$M": round(
            float(staged_df.loc[staged_df["stage"] == s, "staged_ecl_weighted"].sum() / 1e6), 3
        ),
    } for s in [1, 2, 3]])
    staged_summary.loc["total"] = {
        "stage": "ALL",
        "n_loans": len(stage),
        "pct_loans": 100.0,
        "total_staged_ecl_weighted_$M": round(float(staged_df["staged_ecl_weighted"].sum() / 1e6), 3),
    }

    staged_df.to_csv(PROC_DIR / "ifrs9_staged_ecl_by_loan.csv", index=False)
    staged_summary.to_csv(PROC_DIR / "ifrs9_staged_ecl_summary.csv", index=False)

    log.info("")
    log.info("  IFRS 9 Staged ECL Summary (probability-weighted):")
    log.info("\n%s", staged_summary.to_string(index=False))

    # ── Plots and outputs ─────────────────────────────────────────────────
    log.info("")
    log.info("[6/6] Generating outputs ...")
    plot_pd_paths(all_results)
    plot_survival_curves(all_sp)
    plot_ecl_by_horizon(ecl_summary, horizons)
    plot_stage_distribution(staged_df)

    # Survival pool stats
    log.info("")
    log.info("  At-risk pool survival at end of horizon (portfolio mean):")
    for scenario, cfg in SCENARIOS.items():
        sp_final = all_sp[scenario][:, -1].mean() * 100
        sp_peak  = all_sp[scenario][:, 4].mean() * 100
        log.info("    %-10s  SP at Q4=%.3f%%  SP at Q%d=%.3f%%",
                  scenario, sp_peak, N_QUARTERS, sp_final)

    log.info("")
    log.info("=" * 65)
    log.info("IFRS 9 ECL computation complete.")
    log.info("  %d quarterly predictions per loan per scenario,", N_QUARTERS)
    log.info("  with at-risk pool shrinkage, loan ageing, and recovery path,")
    log.info("  and per-loan Stage 1/2/3 classification.")
    log.info("=" * 65)
    log.info("  Next: python 08_calibration.py")


if __name__ == "__main__":
    main()
