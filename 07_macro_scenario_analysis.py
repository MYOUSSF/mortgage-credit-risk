"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.6 — Macro Scenario Analysis
=============================================================================
Script  : 07_macro_scenario_analysis.py
Purpose : Stress-test the XGBoost PD model under adverse macroeconomic
          paths — as required under IFRS 9 multiple economic scenarios (MES)
          and the EBA/Fed stress-testing frameworks.

Regulatory context
-------------------
  IFRS 9 paragraph 5.5.17 requires "forward-looking information" and
  "multiple economic scenarios" for ECL measurement.  Banks must weight
  at least three scenarios:

    Base     — most likely macro path (central forecast)
    Adverse  — moderate stress (1-in-7 year event)
    Severe   — severe stress (1-in-25 year event, akin to 2008 GFC)

  The probability-weighted ECL = Σ_s (weight_s × ECL_s)

  This script implements:
    1. Scenario construction — shock paths for unemployment rate and HPI
    2. Conditional PD surface — PD(t) under each scenario at each horizon
    3. ECL calculation        — probability-weighted across scenarios
    4. Portfolio loss distribution — VaR and CVaR at 99%
    5. Feature sensitivity    — marginal PD impact per unit macro shock

Scenario assumptions (user-configurable)
-----------------------------------------
  Base     : UR stays flat, HPI grows 2% p.a.
  Adverse  : UR rises +3pp over 12m, HPI falls 10%
  Severe   : UR rises +6pp over 18m (2008-level), HPI falls 25%

  Users should replace these with their institution's official stress
  scenarios from their Internal Capital Adequacy Assessment Process (ICAAP).

Inputs
------
  data/processed/pd_oos.parquet
  data/processed/pd_xgb_results.csv      (baseline XGBoost scores)
  data/raw/macro/unemployment_rate.csv   (optional — current UR level)

Outputs
-------
  data/processed/scenario_pd_by_loan.csv   — per-loan PD under each scenario
  data/processed/scenario_ecl_summary.csv  — portfolio ECL by scenario
  data/figures/scenario_macro_paths.png
  data/figures/scenario_pd_distributions.png
  data/figures/scenario_ecl_waterfall.png
  data/figures/scenario_sensitivity.png
=============================================================================
"""

from __future__ import annotations

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
        logging.FileHandler("macro_scenario.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# GPU DETECTION  (mirrors 03_pd_ensemble.py)
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
            log.info("[GPU] %d GPU(s) found: %s", len(gpus), ", ".join(gpus))
            return "cuda"
    except Exception:
        pass
    log.info("[CPU] No GPU detected — using device='cpu'.")
    return "cpu"


DEVICE = _detect_gpu()


# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = Path("data/processed")
MACRO_DIR = Path("data/raw/macro")
FIG_DIR  = Path("data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "default_12m"
SEED     = 42

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

# ── Scenario definitions ──────────────────────────────────────────────────────
# All shocks are *additive* on top of the current macro state.
# current_ur  : current national unemployment rate (%)
# current_hpi : index value at t=0 (normalised to 1.0)

SCENARIOS: dict[str, dict] = {
    "Base": {
        "label":       "Base Scenario",
        "description": "Unemployment stable, HPI appreciates 2% p.a.",
        "color":       "#10B981",
        "weight":      0.60,          # probability weight for ECL
        "ur_shock":    [0.0,  0.0,  0.0,  0.0,  0.0],   # pp change per quarter
        "hpi_shock":   [0.50, 0.50, 0.50, 0.50, 0.50],  # % HPI change per quarter
    },
    "Adverse": {
        "label":       "Adverse Scenario",
        "description": "UR +3pp over 12m, HPI −10% over 18m then partial recovery",
        "color":       "#F59E0B",
        "weight":      0.30,
        "ur_shock":    [0.8,  1.0,  0.8,  0.4,  0.0],
        "hpi_shock":   [-2.5, -3.0, -2.5, -1.5, 0.5],
    },
    "Severe": {
        "label":       "Severe Scenario  (GFC-level)",
        "description": "UR +6pp over 18m, HPI −25%, protracted recovery",
        "color":       "#EF4444",
        "weight":      0.10,
        "ur_shock":    [1.2,  1.5,  1.5,  1.2,  0.6],
        "hpi_shock":   [-5.0, -6.0, -6.0, -5.0, -3.0],
    },
}

# Quarterly horizon for scenario projection
N_QUARTERS = 5    # 5 quarters = 15 months

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
# MACRO PATH CONSTRUCTION
# =============================================================================

def build_macro_paths(current_ur: float = 4.0,
                       current_hpi: float = 1.0) -> pd.DataFrame:
    """
    Build quarterly macro paths for each scenario.

    Returns a DataFrame indexed by (scenario, quarter) with columns:
        ur        : unemployment rate (%)
        hpi_ratio : HPI ratio (current/origination) — used as hpi_change feature
    """
    rows = []
    for scenario, cfg in SCENARIOS.items():
        ur_path  = current_ur
        hpi_path = current_hpi

        rows.append({"scenario": scenario, "quarter": 0,
                     "ur": ur_path, "hpi_ratio": hpi_path})

        for q in range(N_QUARTERS):
            ur_path  += cfg["ur_shock"][q] if q < len(cfg["ur_shock"])  else 0
            hpi_pct   = cfg["hpi_shock"][q] if q < len(cfg["hpi_shock"]) else 0
            hpi_path *= (1 + hpi_pct / 100)
            rows.append({"scenario": scenario, "quarter": q + 1,
                         "ur": max(ur_path, 0.5),   # floor at 0.5%
                         "hpi_ratio": hpi_path})

    return pd.DataFrame(rows)


def plot_macro_paths(macro_df: pd.DataFrame) -> None:
    """Two-panel chart: unemployment path and HPI ratio path per scenario."""
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Macro Scenario Paths — IFRS 9 Multiple Economic Scenarios\n"
                 "Quarterly projections under Base / Adverse / Severe scenarios",
                 fontsize=12, fontweight="bold", color="white")

    quarters = macro_df["quarter"].unique()
    x_labels = [f"Q{q}" for q in quarters]

    for scenario, cfg in SCENARIOS.items():
        sub = macro_df[macro_df["scenario"] == scenario].sort_values("quarter")
        col = cfg["color"]

        axes[0].plot(sub["quarter"], sub["ur"], color=col,
                     linewidth=2.5, marker="o", markersize=6,
                     label=f"{cfg['label']}  (w={cfg['weight']:.0%})")
        axes[1].plot(sub["quarter"], (sub["hpi_ratio"] - 1) * 100,
                     color=col, linewidth=2.5, marker="o", markersize=6,
                     label=cfg["label"])

    axes[0].set_xticks(quarters)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_ylabel("Unemployment Rate (%)", fontsize=10)
    axes[0].set_title("Unemployment Rate Path", color="#CBD5E1", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].axhline(0, color="#4B5563", linewidth=1, linestyle="--")
    axes[1].set_xticks(quarters)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_ylabel("HPI Change from Origination (%)", fontsize=10)
    axes[1].set_title("House Price Index Path", color="#CBD5E1", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)
    axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "scenario_macro_paths.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


# =============================================================================
# DATA PREPARATION  (mirrors 03_pd_ensemble.py)
# =============================================================================

def _encode_categoricals(train: pd.DataFrame, oos: pd.DataFrame,
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
            oos:   pd.DataFrame) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray,
                                           SimpleImputer, list[str],
                                           pd.DataFrame]:
    feats = [f for f in FEATURES if f in train.columns]
    train_enc, oos_enc = _encode_categoricals(train.copy(), oos.copy(), CAT_FEATURES)

    # Drop columns that are entirely NaN in training — SimpleImputer cannot
    # compute a median for them and silently drops them from the output matrix,
    # causing shape mismatches downstream.  hpi_change and ur_3m_lag are the
    # typical culprits when macro files are absent.
    feats = [f for f in feats if train_enc[f].notna().any()]

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(train_enc[feats])
    X_oo = imputer.transform(oos_enc[feats])

    # Return oos_enc so callers can pass the encoded DataFrame to functions
    # that call imputer.transform() again (e.g. apply_scenario_shock).
    # Using the raw oos would fail because categoricals are still strings.
    return X_tr, X_oo, train_enc[TARGET].values, oos_enc[TARGET].values, imputer, feats, oos_enc


def retrain_xgboost(X_tr, y_tr, X_oo, y_oo) -> XGBClassifier:
    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = neg / max(pos, 1)
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
# SCENARIO PD COMPUTATION
# =============================================================================

def apply_scenario_shock(oos_df:     pd.DataFrame,
                           scenario:   str,
                           macro_row:  pd.Series,
                           feats:      list[str],
                           imputer:    SimpleImputer) -> np.ndarray:
    """
    Shock the OOS feature matrix with the macro values from one scenario quarter.

    Modifies ur_3m_lag and hpi_change in place, then scores with the XGBoost model.
    All other loan-level features (FICO, CLTV, DTI, etc.) remain unchanged —
    the model is conditioned on *current* loan characteristics but stressed
    macro environment.
    """
    df_shocked = oos_df.copy()

    if "ur_3m_lag" in df_shocked.columns:
        df_shocked["ur_3m_lag"] = macro_row["ur"]

    if "hpi_change" in df_shocked.columns:
        df_shocked["hpi_change"] = macro_row["hpi_ratio"]

    X_shocked = imputer.transform(df_shocked[feats])
    return X_shocked


def compute_scenario_pds(xgb:      XGBClassifier,
                          oos_df:   pd.DataFrame,
                          macro_df: pd.DataFrame,
                          imputer:  SimpleImputer,
                          feats:    list[str]) -> pd.DataFrame:
    """
    Compute conditional PD for each loan under each scenario × quarter combination.

    Returns a wide DataFrame: rows = loans, columns = scenario_Q metrics.
    """
    results: dict[str, np.ndarray] = {}

    for scenario in SCENARIOS:
        scenario_sub = macro_df[macro_df["scenario"] == scenario]

        # Peak stress quarter = max UR (worst quarter per scenario)
        peak_row = scenario_sub.loc[scenario_sub["ur"].idxmax()]
        X_stressed = apply_scenario_shock(oos_df, scenario, peak_row,
                                          feats, imputer)
        pds = xgb.predict_proba(X_stressed)[:, 1]
        results[f"pd_{scenario.lower()}"] = pds
        log.info("  Scenario %-10s  peak UR=%.1f%%  mean PD=%.4f%%",
                 scenario, peak_row["ur"], pds.mean() * 100)

    result_df = pd.DataFrame(results)
    result_df["loan_seq_num"]   = oos_df["loan_seq_num"].values if "loan_seq_num" in oos_df else np.arange(len(oos_df))
    result_df["actual_default"] = oos_df[TARGET].values
    result_df["orig_upb"]       = oos_df["orig_upb"].values if "orig_upb" in oos_df else 200_000

    return result_df


def compute_weighted_ecl(scenario_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute probability-weighted ECL across scenarios.

        ECL_weighted = Σ_s (weight_s × PD_s × LGD × EAD)

    LGD is assumed to be 0.40 (40%) — typical for well-collateralised
    residential mortgages in the US.  Replace with model output from
    04_lgd_models.py in production.

    EAD = original UPB (simplification; in production use current UPB
    adjusted for amortisation and drawn-down commitments).
    """
    LGD_ASSUMPTION = 0.40

    ecl_rows = []
    for scenario, cfg in SCENARIOS.items():
        pd_col = f"pd_{scenario.lower()}"
        if pd_col not in scenario_df.columns:
            continue

        ecl = scenario_df[pd_col] * LGD_ASSUMPTION * scenario_df["orig_upb"]
        weighted_ecl = ecl * cfg["weight"]
        ecl_rows.append({
            "scenario":          cfg["label"],
            "weight":            cfg["weight"],
            "mean_pd_pct":       scenario_df[pd_col].mean() * 100,
            "total_ecl_$M":      ecl.sum() / 1_000_000,
            "weighted_ecl_$M":   weighted_ecl.sum() / 1_000_000,
        })

    ecl_df = pd.DataFrame(ecl_rows)
    ecl_df.loc[len(ecl_df)] = {
        "scenario":        "PROBABILITY-WEIGHTED TOTAL",
        "weight":           1.0,
        "mean_pd_pct":      None,
        "total_ecl_$M":     None,
        "weighted_ecl_$M":  ecl_df["weighted_ecl_$M"].sum(),
    }
    return ecl_df


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(xgb:     XGBClassifier,
                           oos_df:  pd.DataFrame,
                           imputer: SimpleImputer,
                           feats:   list[str]) -> pd.DataFrame:
    """
    Marginal sensitivity of portfolio PD to unit shocks in each macro feature.

    For each feature f:
        ΔPD_f = mean PD(x + δ_f * e_f) − mean PD(x)

    where δ_f is a standardised shock:
        ur_3m_lag  : +1 percentage point
        hpi_change : +0.10 (10% price fall)

    This is the primary sensitivity table reported to Asset-Liability
    Committees (ALCOs) and risk committees.
    """
    feats_present = [f for f in feats if f in oos_df.columns]
    df_base = oos_df.copy()
    X_base  = imputer.transform(df_base[feats_present])
    pd_base = xgb.predict_proba(X_base)[:, 1].mean()

    shocks  = {
        "ur_3m_lag":  +1.0,    # +1pp unemployment
        "hpi_change": +0.10,   # 10pp decline in HPI ratio (prices fall)
    }

    rows = []
    for feat, shock in shocks.items():
        if feat not in feats_present:
            continue
        feat_idx      = feats_present.index(feat)
        X_shocked     = X_base.copy()
        X_shocked[:, feat_idx] += shock
        pd_shocked    = xgb.predict_proba(X_shocked)[:, 1].mean()
        delta         = (pd_shocked - pd_base) * 100   # in pp

        rows.append({
            "feature":         feat,
            "shock":           f"+{shock}",
            "pd_base_pct":     round(pd_base * 100, 5),
            "pd_shocked_pct":  round(pd_shocked * 100, 5),
            "delta_pd_pp":     round(delta, 5),
        })
        log.info("  Sensitivity %-18s  shock=%+.2f  ΔPD=%+.4f pp",
                 feat, shock, delta)

    return pd.DataFrame(rows)


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_pd_distributions(scenario_df: pd.DataFrame) -> None:
    """Overlapping PD histograms for each scenario — shows tail risk clearly."""
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PD Distribution Under IFRS 9 Macro Scenarios\n"
                 "Severe scenario shifts the entire distribution rightward",
                 fontsize=12, fontweight="bold", color="white")

    pd_cols = [c for c in scenario_df.columns if c.startswith("pd_")]

    # Histogram
    for col in pd_cols:
        scen_key = col.replace("pd_", "").title()
        cfg      = SCENARIOS.get(scen_key, {})
        color    = cfg.get("color", "#38BDF8")
        label    = cfg.get("label", scen_key)
        axes[0].hist(scenario_df[col] * 100, bins=60, alpha=0.5,
                     color=color, label=label, density=True,
                     range=(0, scenario_df[pd_cols].max().max() * 100 * 1.1))

    axes[0].set_xlabel("12-Month PD (%)", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].set_title("PD Distribution by Scenario", color="#CBD5E1", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)
    axes[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Cumulative distribution (portfolio loss)
    for col in pd_cols:
        scen_key = col.replace("pd_", "").title()
        cfg      = SCENARIOS.get(scen_key, {})
        color    = cfg.get("color", "#38BDF8")
        label    = cfg.get("label", scen_key)
        vals     = np.sort(scenario_df[col] * 100)
        cdf      = np.arange(1, len(vals) + 1) / len(vals)
        axes[1].plot(vals, cdf, color=color, linewidth=2.5, label=label)

    axes[1].set_xlabel("12-Month PD (%)", fontsize=10)
    axes[1].set_ylabel("Cumulative Fraction of Portfolio", fontsize=10)
    axes[1].set_title("Cumulative PD Distribution (ECDF)", color="#CBD5E1", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)
    axes[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "scenario_pd_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_ecl_waterfall(ecl_df: pd.DataFrame) -> None:
    """ECL waterfall: scenario ECL bars with weighted total highlighted."""
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))

    rows_plot = ecl_df[ecl_df["scenario"] != "PROBABILITY-WEIGHTED TOTAL"].copy()
    weighted_total = ecl_df[ecl_df["scenario"] == "PROBABILITY-WEIGHTED TOTAL"]

    scen_keys = [r for r in SCENARIOS if f"pd_{r.lower()}" in rows_plot.get("scenario", pd.Series()).values or True]
    colors_map = {cfg["label"]: cfg["color"] for cfg in SCENARIOS.values()}

    x_pos    = np.arange(len(rows_plot) + 1)
    labels   = list(rows_plot["scenario"]) + ["Probability-Weighted ECL"]
    ecl_vals = list(rows_plot["total_ecl_$M"].fillna(0)) + [
        float(weighted_total["weighted_ecl_$M"].values[0]) if not weighted_total.empty else 0
    ]
    bar_cols = [colors_map.get(s, "#38BDF8") for s in rows_plot["scenario"]] + ["#3B82F6"]

    bars = ax.bar(x_pos, ecl_vals, color=bar_cols, alpha=0.85, edgecolor="none", width=0.6)

    for bar, val, row in zip(bars[:-1], ecl_vals[:-1], rows_plot.itertuples()):
        w = getattr(row, "weight", 0)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"${val:.1f}M\n(w={w:.0%})",
                ha="center", va="bottom", fontsize=9.5, color="#E2E8F0",
                fontweight="bold")

    # Weighted total annotation
    if ecl_vals:
        ax.text(bars[-1].get_x() + bars[-1].get_width() / 2,
                bars[-1].get_height() + 0.02,
                f"${ecl_vals[-1]:.2f}M",
                ha="center", va="bottom", fontsize=11, color="white",
                fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9, rotation=10, ha="right")
    ax.set_ylabel("Expected Credit Loss ($M)", fontsize=10)
    ax.set_title("ECL by Scenario — Probability-Weighted IFRS 9 Provision\n"
                 "ECL = PD × LGD (40%) × EAD  |  Weights: Base 60% / Adverse 30% / Severe 10%",
                 fontsize=11, fontweight="bold", color="white", pad=14)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "scenario_ecl_waterfall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_sensitivity(sensitivity_df: pd.DataFrame,
                      pd_base: float) -> None:
    """Tornado chart of macro sensitivity — standard ALCO reporting format."""
    if sensitivity_df.empty:
        return

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(9, max(4, len(sensitivity_df) * 1.2 + 2)))

    y_pos  = np.arange(len(sensitivity_df))
    deltas = sensitivity_df["delta_pd_pp"].values
    labels = [f"{r['feature']}\n(shock {r['shock']})"
              for _, r in sensitivity_df.iterrows()]
    colors = ["#EF4444" if d > 0 else "#22C55E" for d in deltas]

    bars = ax.barh(y_pos, deltas, color=colors, alpha=0.85,
                   edgecolor="none", height=0.55)
    ax.axvline(0, color="#4B5563", linewidth=1.5)

    for bar, val in zip(bars, deltas):
        ax.text(val + (0.0002 if val >= 0 else -0.0002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f} pp",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=9.5, color="#E2E8F0", fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Change in Portfolio Mean PD (percentage points)", fontsize=10)
    ax.set_title(f"Macro Sensitivity — ΔPD per Unit Shock\n"
                 f"Base portfolio mean PD = {pd_base*100:.4f}%  |  "
                 f"Positive = higher risk",
                 fontsize=11, fontweight="bold", color="white", pad=14)
    ax.grid(True, axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "scenario_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Ch.6 — Macro Scenario Analysis (IFRS 9)")
    log.info("=" * 65)

    # Current macro state — read from file if available, else use defaults
    current_ur  = 4.0
    ur_path = MACRO_DIR / "unemployment_rate.csv"
    if ur_path.exists():
        try:
            ur_df = pd.read_csv(ur_path, dtype=str)
            ur_df.columns = ur_df.columns.str.lower().str.strip()
            col = next((c for c in ur_df.columns if "rate" in c or "value" in c or "ur" in c), None)
            if col:
                current_ur = float(pd.to_numeric(ur_df[col], errors="coerce").dropna().iloc[-1])
                log.info("  Current unemployment rate from file: %.1f%%", current_ur)
        except Exception:
            pass
    log.info("  Using current UR = %.1f%%", current_ur)

    # ── Load ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("[1/5] Loading processed data …")
    train = pd.read_parquet(PROC_DIR / "pd_train.parquet")
    oos   = pd.read_parquet(PROC_DIR / "pd_oos.parquet")
    log.info("  Train: %s  |  OOS: %s",
             f"{len(train):,}", f"{len(oos):,}")

    # ── Prepare and retrain ───────────────────────────────────────────────
    log.info("")
    log.info("[2/5] Preparing data and retraining XGBoost …")
    X_tr, X_oo, y_tr, y_oo, imputer, feats, oos_enc = prepare(train, oos)
    xgb = retrain_xgboost(X_tr, y_tr, X_oo, y_oo)

    # ── Macro paths ───────────────────────────────────────────────────────
    log.info("")
    log.info("[3/5] Building macro scenario paths …")
    macro_df = build_macro_paths(current_ur=current_ur, current_hpi=1.0)
    log.info("  Scenarios: %s", list(SCENARIOS.keys()))
    plot_macro_paths(macro_df)

    # ── Scenario PD computation ───────────────────────────────────────────
    log.info("")
    log.info("[4/5] Computing conditional PD under each scenario …")
    pd_baseline = xgb.predict_proba(X_oo)[:, 1].mean()
    log.info("  Baseline portfolio mean PD: %.4f%%", pd_baseline * 100)

    # Use oos_enc (label-encoded) not raw oos — apply_scenario_shock calls
    # imputer.transform() which requires numeric-only columns.
    scenario_df = compute_scenario_pds(xgb, oos_enc, macro_df, imputer, feats)
    scenario_df.to_csv(PROC_DIR / "scenario_pd_by_loan.csv", index=False)
    log.info("  Per-loan scenario PD → data/processed/scenario_pd_by_loan.csv")

    ecl_df = compute_weighted_ecl(scenario_df)
    ecl_df.to_csv(PROC_DIR / "scenario_ecl_summary.csv", index=False)
    log.info("")
    log.info("  ECL Summary:")
    log.info("\n%s", ecl_df.to_string(index=False))

    sensitivity_df = sensitivity_analysis(xgb, oos_enc, imputer, feats)
    sensitivity_df.to_csv(PROC_DIR / "scenario_sensitivity.csv", index=False)

    plot_pd_distributions(scenario_df)
    plot_ecl_waterfall(ecl_df)
    plot_sensitivity(sensitivity_df, pd_baseline)

    # VaR / CVaR
    log.info("")
    log.info("[5/5] Portfolio loss statistics (Severe scenario) …")
    pd_col = "pd_severe"
    if pd_col in scenario_df.columns:
        losses = scenario_df[pd_col] * 0.40 * scenario_df["orig_upb"] / 1e6
        var99  = np.percentile(losses, 99)
        cvar99 = losses[losses >= var99].mean()
        log.info("  VaR  (99%%, severe): $%.2fM per loan", var99)
        log.info("  CVaR (99%%, severe): $%.2fM per loan", cvar99)

    log.info("")
    log.info("=" * 65)
    log.info("Ch.6 complete — Macro scenario analysis generated.")
    log.info("  Next: python 08_calibration.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()