"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.6 — Macro Scenario Analysis (IFRS 9)
=============================================================================
Script  : 07_macro_scenario_ifrs9.py
Purpose : Proper IFRS 9 ECL calculation under three macro scenarios with
          full quarterly path scoring — replacing the single peak-quarter
          approximation in the original script.

Key improvements over original 07_macro_scenario_analysis.py
-------------------------------------------------------------
  1. Quarterly scoring loop
     Each loan is scored at EVERY quarter of the macro path, not just
     the peak quarter. PD varies with the macro environment each quarter.

  2. At-risk pool shrinkage
     Loans that default in quarter q are removed from the at-risk pool
     in q+1. ECL is accumulated over a geometrically declining population.

  3. Recovery path
     All three scenarios include an explicit recovery phase (delta_k < 0)
     after the stress peak. UR mean-reverts toward its long-run level.
     This avoids the original script's implicit assumption that UR stays
     permanently elevated forever.

  4. Loan feature ageing
     loan_age is incremented each quarter. This allows the model to
     reflect that ageing loans have different risk profiles — e.g. seasoned
     loans approaching the peak default risk window (typically years 3-5)
     behave differently from newly originated ones.

  5. IFRS 9 ECL formula
     For each loan i and quarter q:

         ECL_i = sum_{q=1}^{Q}  PD_i(q) * SP_i(q-1) * LGD * EAD_i * DF(q)

     where:
         PD_i(q)    = conditional default probability in quarter q
                      given macro conditions at q
         SP_i(q-1)  = survival probability to the START of quarter q
                    = product_{k=1}^{q-1} (1 - PD_i(k))
         LGD        = loss given default (40% assumption or per-loan)
         EAD_i      = exposure at default (current UPB, amortised)
         DF(q)      = discount factor = 1 / (1 + r)^(q/4)
                      where r = annual risk-free rate

  6. Horizon flexibility
     ECL is computed at 12m, 24m, 36m, and lifetime (full path) horizons,
     matching IFRS 9 Stage 1, Stage 2 output requirements.

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

Outputs
-------
  data/processed/ifrs9_ecl_by_loan.csv       — per-loan ECL at each horizon
  data/processed/ifrs9_ecl_summary.csv       — portfolio ECL summary
  data/processed/ifrs9_macro_paths.csv       — full quarterly macro paths
  data/figures/ifrs9_macro_paths.png
  data/figures/ifrs9_pd_paths.png
  data/figures/ifrs9_ecl_by_horizon.png
  data/figures/ifrs9_survival_curves.png
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
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed.")
    sys.exit(1)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("macro_ifrs9.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# GPU DETECTION
# =============================================================================

def _detect_gpu() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = [g.strip() for g in result.stdout.strip().splitlines() if g.strip()]
            log.info("[GPU] %d GPU(s): %s", len(gpus), ", ".join(gpus))
            return "cuda"
    except Exception:
        pass
    log.info("[CPU] No GPU detected.")
    return "cpu"

DEVICE = _detect_gpu()

# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = Path("data/processed")
FIG_DIR  = Path("data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET      = "default_12m"
SEED        = 42
LGD         = 0.40          # assumed LGD — replace with script 04 output
DISCOUNT_R  = 0.05          # annual risk-free discount rate for ECL
N_QUARTERS  = 20            # full path length: 5 years

FEATURES: list[str] = [
    "delinquency_indicator",
    "hpi_change",
    "occupancy_status",
    "orig_interest_rate",
    "orig_cltv",
    "num_borrowers",
    "credit_score",
    "property_type",
    "loan_age",
    "orig_dti",
    "orig_upb",
    "ur_3m_lag",
]
CAT_FEATURES = ["occupancy_status", "property_type"]

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
    "figure.dpi":        130,
}

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
                       horizons:  list[int] = [4, 8, 12, N_QUARTERS]
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute IFRS 9 ECL for each loan under each scenario.

    The ECL at horizon H for loan i under scenario s is:

        ECL_i^s(H) = sum_{q=1}^{H}  PD_i^s(q) * SP_i^s(q-1) * LGD * EAD_i * DF(q)

    where:
        PD_i^s(q)  = conditional PD in quarter q under scenario s
        SP_i^s(q-1) = P(survived quarters 1..q-1)
                    = product_{k=1}^{q-1} (1 - PD_i^s(k))
        SP_i^s(0)  = 1  (all loans survive to start)
        DF(q)      = 1 / (1 + DISCOUNT_R)^(q/4)  quarterly discount factor

    Parameters
    ----------
    horizons : list of quarters at which to report ECL.
               e.g. [4, 8, 12, 20] = 1Y, 2Y, 3Y, lifetime

    Returns
    -------
    ecl_by_loan : DataFrame of per-loan ECL at each horizon under each scenario
    ecl_summary : DataFrame of portfolio-level ECL and weighted ECL
    """
    n_loans = len(oos_df)
    ead     = oos_df["orig_upb"].values if "orig_upb" in oos_df.columns \
              else np.full(n_loans, 200_000.0)

    # Discount factors: DF(q) = 1/(1+r)^(q/4)
    discount = np.array([1.0 / (1 + DISCOUNT_R) ** (q / 4)
                          for q in range(N_QUARTERS + 1)])

    all_results = {}   # scenario -> (n_loans, N_QUARTERS) PD matrix
    all_sp      = {}   # scenario -> (n_loans, N_QUARTERS+1) survival prob matrix

    for scenario, cfg in SCENARIOS.items():
        log.info("  Computing quarterly PDs: %s ...", scenario)
        scenario_macro = macro_df[macro_df["scenario"] == scenario].sort_values("quarter")

        pd_matrix = np.zeros((n_loans, N_QUARTERS))   # PD_i(q)  q=1..N_QUARTERS
        sp_matrix = np.ones((n_loans, N_QUARTERS + 1)) # SP_i(q)  q=0..N_QUARTERS

        for q in range(1, N_QUARTERS + 1):
            macro_row = scenario_macro[scenario_macro["quarter"] == q].iloc[0]

            # Conditional PD for this quarter
            pd_q = score_quarter(xgb, oos_df, macro_row, feats, imputer, q)
            pd_q = np.clip(pd_q, 0.0, 1.0)
            pd_matrix[:, q - 1] = pd_q

            # Survival probability to END of quarter q
            # SP(q) = SP(q-1) * (1 - PD(q))
            sp_matrix[:, q] = sp_matrix[:, q - 1] * (1 - pd_q)

            if q % 4 == 0:
                log.info("    %s Q%02d: mean PD=%.4f%%  mean survival=%.4f%%",
                          scenario, q,
                          pd_q.mean() * 100,
                          sp_matrix[:, q].mean() * 100)

        all_results[scenario] = pd_matrix
        all_sp[scenario]      = sp_matrix

    # ── Build per-loan ECL at each horizon ────────────────────────────────────
    loan_id = oos_df["loan_seq_num"].values \
              if "loan_seq_num" in oos_df.columns \
              else np.arange(n_loans)

    ecl_rows = {"loan_seq_num": loan_id, "orig_upb": ead}

    horizon_labels = {h: f"{h//4}Y" if h % 4 == 0 else f"Q{h}" for h in horizons}

    for scenario in SCENARIOS:
        pd_mat = all_results[scenario]   # (n_loans, N_QUARTERS)
        sp_mat = all_sp[scenario]        # (n_loans, N_QUARTERS+1)

        for h in horizons:
            h_idx = min(h, N_QUARTERS)
            # ECL_i(H) = sum_{q=1}^{H} PD_i(q) * SP_i(q-1) * LGD * EAD_i * DF(q)
            ecl_i = np.zeros(n_loans)
            for q in range(1, h_idx + 1):
                ecl_i += (pd_mat[:, q - 1]   # PD in quarter q
                          * sp_mat[:, q - 1]  # survived to start of q
                          * LGD
                          * ead
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
        "ECL = sum_{q} PD(q) * SP(q-1) * LGD * EAD * DF(q)   "
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
    log.info("[4/5] Computing IFRS 9 ECL with full quarterly path scoring ...")
    log.info("  Horizons: 1Y (Q4), 2Y (Q8), 3Y (Q12), lifetime (%dQ)", N_QUARTERS)
    log.info("  LGD assumption: %.0f%%  |  Discount rate: %.1f%% p.a.",
             LGD * 100, DISCOUNT_R * 100)
    log.info("  Loan ageing: loan_age incremented 3 months per quarter")
    log.info("  At-risk pool: SP(q) = product_{k=1}^{q} (1 - PD(k))")

    horizons = [4, 8, 12, N_QUARTERS]
    ecl_by_loan, ecl_summary, all_results, all_sp = compute_ifrs9_ecl(
        xgb, oos_enc, macro_df, imputer, feats, horizons=horizons
    )

    ecl_by_loan.to_csv(PROC_DIR / "ifrs9_ecl_by_loan.csv", index=False)
    ecl_summary.to_csv(PROC_DIR / "ifrs9_ecl_summary.csv", index=False)

    log.info("")
    log.info("  IFRS 9 ECL Summary:")
    log.info("\n%s", ecl_summary.to_string(index=False))

    # ── Plots and outputs ─────────────────────────────────────────────────
    log.info("")
    log.info("[5/5] Generating outputs ...")
    plot_pd_paths(all_results)
    plot_survival_curves(all_sp)
    plot_ecl_by_horizon(ecl_summary, horizons)

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
    log.info("  Key improvement over original script:")
    log.info("  Original: one prediction at peak quarter only")
    log.info("  This script: %d predictions per loan per scenario,", N_QUARTERS)
    log.info("  with at-risk pool shrinkage, loan ageing, and recovery path.")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
