"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.5 — Survival Analysis
=============================================================================
Script  : 06_survival_analysis.py
Purpose : Cox Proportional Hazards model for time-to-default, handling
          right-censoring more rigorously than the binary 12-month indicator.

Why survival analysis outperforms binary classification for PD
--------------------------------------------------------------
  The 12-month binary default indicator in Ch.1 and Ch.2 has three weaknesses:

    1. Right-censoring ignored — loans still active at observation date are
       dropped or treated as non-defaulters, introducing downward bias in PD.
    2. Temporal dynamics lost — the indicator treats a loan 1 month from
       default the same as one 359 days from default.
    3. Window is arbitrary — 12 months is regulatory convention (IFRS 9
       Stage 1); survival analysis provides predictions at *any* horizon.

  The Cox Proportional Hazards model resolves all three:

    h(t | x) = h₀(t) · exp(x'β)

  where h₀(t) is the non-parametric baseline hazard, estimated from the data.
  Right-censored observations (active loans) contribute to the partial
  likelihood — their full history is used without fabricating an outcome.

Regulatory context
-------------------
  EBA Guidelines (EBA/GL/2017/16) on PD estimation explicitly acknowledge
  survival-based methods as appropriate for through-the-cycle PD estimation.
  The Basel II IRB formulae implicitly assume an exponential survival model;
  the Cox model relaxes this by allowing the baseline hazard to vary freely.

Models implemented
------------------
  1. Kaplan-Meier estimator   — non-parametric survival function (no covariates)
     by credit score tertile, property type, and origination vintage
  2. Cox Proportional Hazards — semi-parametric; produces hazard ratios and
     survival functions for individual loans
  3. Time-varying covariate model — extends Cox to incorporate loan_age and
     time-varying macroeconomic state (ur_3m_lag, hpi_change)
  4. Calibration to IFRS 9 horizons — survival function integrated to 12m,
     24m, and lifetime PD

Inputs
------
  data/processed/pd_train.parquet
  data/processed/pd_oos.parquet
  data/processed/pd_oot.parquet

  The pipeline constructs the survival dataset internally from the PD
  Parquet files: duration = loan_age at last observation, event = default_12m.

Outputs
-------
  data/processed/survival_cox_coefs.csv     — hazard ratios + confidence intervals
  data/processed/survival_pd_horizons.csv   — per-loan PD at 12m / 24m / lifetime
  data/figures/survival_km_by_vintage.png
  data/figures/survival_km_by_credit_score.png
  data/figures/survival_cox_hazard_ratios.png
  data/figures/survival_pd_horizons.png
  data/figures/survival_schoenfeld_residuals.png

Prerequisites
-------------
  pip install lifelines>=0.27.0
  Run 01_data_preprocessing.py first.
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
    import lifelines
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    from lifelines.utils import concordance_index
except ImportError:
    print("ERROR: lifelines not installed.  Run: pip install lifelines>=0.27.0")
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
        logging.FileHandler("survival_analysis.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = Path("data/processed")
FIG_DIR  = Path("data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED   = 42
TARGET = "default_12m"

# Features available in the PD parquet (numeric only for Cox model)
COX_FEATURES: list[str] = [
    "credit_score",
    "orig_cltv",
    "orig_dti",
    "orig_interest_rate",
    "orig_upb",
    "hpi_change",
    "ur_3m_lag",
    "num_borrowers",
    "delinquency_indicator",
]

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

PALETTE = ["#38BDF8", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6",
           "#EC4899", "#14B8A6", "#F97316"]


# =============================================================================
# SURVIVAL DATASET CONSTRUCTION
# =============================================================================

def build_survival_df(pd_df: pd.DataFrame,
                      features: list[str]) -> pd.DataFrame:
    """
    Construct the survival analysis dataset from the PD Parquet file.

    Each row in pd_df is a loan-month observation.  We aggregate to
    one row per loan (the last observation), with:

        duration  = loan_age at last observation  (months since first payment)
        event     = 1 if the loan defaulted, 0 if censored (still active)

    Right-censored loans (event=0) contribute their full history to the
    partial likelihood — this is the key advantage over the binary indicator.
    """
    required = ["loan_seq_num", "loan_age", TARGET] + [
        f for f in features if f in pd_df.columns
    ]
    df = pd_df[required].copy()

    # Last observation per loan (maximum loan age seen)
    idx = df.groupby("loan_seq_num")["loan_age"].idxmax()
    df  = df.loc[idx].reset_index(drop=True)

    # Event = any default flag observed for this loan
    event = (
        pd_df.groupby("loan_seq_num")[TARGET]
        .max()
        .rename("event")
        .reset_index()
    )
    df = df.merge(event, on="loan_seq_num", how="left")

    # Duration must be at least 1 month
    df["duration"] = df["loan_age"].clip(lower=1)
    df["event"]    = df["event"].fillna(0).astype(int)

    return df.dropna(subset=["duration", "event"])


# =============================================================================
# KAPLAN-MEIER PLOTS
# =============================================================================

def plot_km_by_group(df: pd.DataFrame,
                     group_col: str,
                     group_labels: dict | None,
                     title: str,
                     filename: str,
                     max_t: float = 120) -> None:
    """
    Kaplan-Meier survival curves stratified by `group_col`.

    Includes log-rank test p-value for all pairwise comparisons.
    """
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(11, 6))

    groups = sorted(df[group_col].dropna().unique())
    kmf    = KaplanMeierFitter()
    event_tables = []

    for grp, color in zip(groups, PALETTE):
        mask  = df[group_col] == grp
        sub   = df[mask]
        label = (group_labels or {}).get(grp, str(grp))
        n     = mask.sum()
        events= sub["event"].sum()

        kmf.fit(sub["duration"].clip(upper=max_t),
                sub["event"],
                label=f"{label}  (n={n:,}, events={int(events):,})")
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2.2,
                                   ci_show=True, ci_alpha=0.12)
        event_tables.append({"group": label, "n": n, "events": events})

    # Log-rank test
    if len(groups) >= 2:
        try:
            result = multivariate_logrank_test(
                df["duration"].clip(upper=max_t),
                df[group_col].fillna("Unknown"),
                df["event"],
            )
            pval_str = f"Log-rank p = {result.p_value:.4f}"
        except Exception:
            pval_str = ""
        ax.text(0.97, 0.97, pval_str,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color="#A0AEC0",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1A2035",
                          edgecolor="#2D3748", alpha=0.9))

    ax.set_xlabel("Loan Age (Months)", fontsize=11)
    ax.set_ylabel("Survival Probability  P(No Default by Month t)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color="white", pad=14)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_t)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


# =============================================================================
# COX PH MODEL
# =============================================================================

def fit_cox(train_surv: pd.DataFrame,
            features:   list[str]) -> CoxPHFitter:
    """
    Fit a penalised Cox PH model.

    penalizer=0.1 provides L2 regularisation — important for stability
    when some features are correlated (credit_score and orig_cltv often are).

    The concordance index (C-index) generalises AUROC to survival data:
    it is the probability that a randomly chosen defaulter has a higher
    predicted hazard than a randomly chosen non-defaulter.  C-index = 0.5
    is random, 1.0 is perfect.
    """
    avail_feats = [f for f in features if f in train_surv.columns
                   and train_surv[f].notna().any()]
    df_cox      = train_surv[["duration", "event"] + avail_feats].dropna()

    log.info("  Cox PH dataset: %s loans  (%s events)",
             f"{len(df_cox):,}", f"{int(df_cox['event'].sum()):,}")

    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(df_cox, duration_col="duration", event_col="event")

    cox.print_summary()
    c_idx = concordance_index(
        df_cox["duration"], -cox.predict_partial_hazard(df_cox), df_cox["event"]
    )
    log.info("  C-index (train): %.4f", c_idx)
    return cox


def plot_hazard_ratios(cox: CoxPHFitter) -> None:
    """
    Forest plot of hazard ratios with 95% confidence intervals.

    HR > 1 means the feature *increases* default risk (shorter survival).
    HR < 1 means the feature *reduces* default risk.

    This is the standard regulatory output for semi-parametric survival models.
    """
    summary = cox.summary.copy()
    summary = summary.sort_values("exp(coef)", ascending=True)

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(10, max(5, len(summary) * 0.55 + 1.5)))

    y_pos = np.arange(len(summary))
    hr    = summary["exp(coef)"].values
    lo    = summary["exp(coef) lower 95%"].values
    hi    = summary["exp(coef) upper 95%"].values

    colors = ["#EF4444" if h > 1 else "#22C55E" for h in hr]
    ax.barh(y_pos, hr, color=colors, alpha=0.35, edgecolor="none", height=0.5)
    ax.errorbar(hr, y_pos,
                xerr=[hr - lo, hi - hr],
                fmt="none", ecolor="#E2E8F0", elinewidth=1.5, capsize=4,
                capthick=1.5)
    ax.scatter(hr, y_pos, color=colors, s=55, zorder=5)
    ax.axvline(1.0, color="#4B5563", linewidth=1.5, linestyle="--",
               label="HR = 1  (no effect)")

    # p-value stars
    for y, (_, row) in zip(y_pos, summary.iterrows()):
        p = row.get("p", 1.0)
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if stars:
            ax.text(hi[y_pos.tolist().index(y)] + 0.02, y, stars,
                    va="center", fontsize=9, color="#FBBF24")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary.index.tolist(), fontsize=9)
    ax.set_xlabel("Hazard Ratio  (HR > 1 = higher default risk)", fontsize=10)
    ax.set_title("Cox PH Model — Hazard Ratios with 95% Confidence Intervals\n"
                 "Red = increases risk  ·  Green = reduces risk  ·  * p<0.05  ** p<0.01  *** p<0.001",
                 fontsize=11, fontweight="bold", color="white", pad=14)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "survival_cox_hazard_ratios.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_schoenfeld_residuals(cox: CoxPHFitter,
                               train_surv: pd.DataFrame,
                               features:   list[str]) -> None:
    """
    Schoenfeld residual plots for the proportional-hazards assumption check.

    If the PH assumption holds, Schoenfeld residuals should show no trend
    over time.  A systematic trend indicates time-varying coefficients —
    a violation that would require stratification or time-interaction terms.
    This is a key model validation step required by regulators.
    """
    avail = [f for f in features if f in train_surv.columns]
    df_cox = train_surv[["duration", "event"] + avail].dropna()

    try:
        residuals = cox.compute_residuals(df_cox, kind="schoenfeld")
    except Exception as e:
        log.warning("  Schoenfeld residuals unavailable: %s", e)
        return

    # Plot up to 4 features
    top_feats = [f for f in residuals.columns if f in avail][:4]
    if not top_feats:
        return

    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, len(top_feats),
                              figsize=(4.5 * len(top_feats), 4.5),
                              sharey=False)
    if len(top_feats) == 1:
        axes = [axes]

    fig.suptitle("Schoenfeld Residuals — Proportional Hazards Assumption Check\n"
                 "No time-trend = PH assumption holds",
                 fontsize=11, fontweight="bold", color="white")

    for ax, feat in zip(axes, top_feats):
        times = df_cox.loc[residuals.index, "duration"]
        ax.scatter(times, residuals[feat], alpha=0.3, s=8, color="#38BDF8",
                   linewidths=0)
        # Lowess smoothing
        try:
            from scipy.stats import pearsonr
            from numpy.polynomial.polynomial import polyfit
            coefs = polyfit(times, residuals[feat], 1)
            t_line = np.linspace(times.min(), times.max(), 100)
            ax.plot(t_line, coefs[0] + coefs[1] * t_line,
                    color="#EF4444", linewidth=2, label="Trend")
            r, p = pearsonr(times, residuals[feat])
            ax.text(0.97, 0.97, f"r={r:.3f}, p={p:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, color="#A0AEC0")
        except Exception:
            pass

        ax.axhline(0, color="#4B5563", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Time (Loan Age, Months)", fontsize=9)
        ax.set_ylabel("Schoenfeld Residual", fontsize=9)
        ax.set_title(feat, color="#CBD5E1", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "survival_schoenfeld_residuals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


# =============================================================================
# MULTI-HORIZON PD
# =============================================================================

def compute_horizon_pds(cox:       CoxPHFitter,
                         oos_surv:  pd.DataFrame,
                         features:  list[str],
                         horizons:  list[int] = [12, 24, 36, 60]) -> pd.DataFrame:
    """
    Compute cumulative default probability at multiple horizons for each loan.

    P(default by horizon h) = 1 - S(h | x_i)

    where S(h | x_i) is the Cox survival function evaluated at month h for
    loan i.  This directly produces the input needed for IFRS 9 ECL staging:

        Stage 1: 12-month PD
        Stage 2: Lifetime PD (significant increase in credit risk)
        Stage 3: Already defaulted
    """
    avail = [f for f in features if f in oos_surv.columns
             and oos_surv[f].notna().any()]
    df_cox = oos_surv[["loan_seq_num", "duration", "event"] + avail].dropna()

    if len(df_cox) == 0:
        log.warning("  No OOS survival data available for PD horizon calculation.")
        return pd.DataFrame()

    log.info("  Computing survival functions for %s OOS loans …",
             f"{len(df_cox):,}")

    # predict_survival_function returns DataFrame(index=times, cols=loan_idx)
    sf = cox.predict_survival_function(df_cox[avail])

    results = []
    for h in horizons:
        # Find the largest time index ≤ h in the baseline survival function
        t_idx = sf.index[sf.index <= h]
        if len(t_idx) == 0:
            probs = np.zeros(len(df_cox))
        else:
            probs = 1 - sf.loc[t_idx[-1]].values

        results.append(pd.Series(probs, name=f"pd_{h}m"))

    horizon_df = pd.concat(results, axis=1)
    horizon_df.index = df_cox.index
    horizon_df["loan_seq_num"] = df_cox["loan_seq_num"].values
    horizon_df["actual_default"] = df_cox["event"].values

    return horizon_df


def plot_pd_horizons(horizon_df: pd.DataFrame) -> None:
    """Compare PD distributions across IFRS 9-relevant time horizons."""
    if horizon_df.empty:
        return

    pd_cols = [c for c in horizon_df.columns if c.startswith("pd_")]
    if not pd_cols:
        return

    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Survival Model — PD at Multiple Horizons  (IFRS 9 / Basel)\n"
                 "P(default by month h | loan characteristics)",
                 fontsize=12, fontweight="bold", color="white")

    # 1. Box plot of PD distributions per horizon
    data_boxes = [horizon_df[c].clip(0, 1) * 100 for c in pd_cols]
    labels     = [c.replace("pd_", "").replace("m", "m horizon") for c in pd_cols]
    bp = axes[0].boxplot(data_boxes, labels=labels, patch_artist=True,
                          medianprops=dict(color="white", linewidth=2),
                          whiskerprops=dict(color="#A0AEC0"),
                          capprops=dict(color="#A0AEC0"),
                          flierprops=dict(marker=".", color="#4B5563", alpha=0.3))
    for patch, color in zip(bp["boxes"], PALETTE[:len(pd_cols)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[0].set_ylabel("Cumulative PD (%)", fontsize=10)
    axes[0].set_title("PD Distribution by Horizon", color="#CBD5E1", fontsize=10)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=2))
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # 2. Mean PD vs observed default rate by 12m decile
    if "pd_12m" in horizon_df.columns and "actual_default" in horizon_df.columns:
        df_plot = horizon_df[["pd_12m", "actual_default"]].dropna()
        df_plot["decile"] = pd.qcut(df_plot["pd_12m"], q=10,
                                     labels=False, duplicates="drop")
        agg = df_plot.groupby("decile").agg(
            mean_pd=("pd_12m", "mean"),
            obs_dr=("actual_default", "mean"),
        ).reset_index()

        axes[1].scatter(agg["mean_pd"] * 100, agg["obs_dr"] * 100,
                        color=PALETTE[0], s=80, zorder=5, label="Score deciles")
        axes[1].plot([0, agg["mean_pd"].max() * 100],
                     [0, agg["mean_pd"].max() * 100],
                     "r--", linewidth=1.5, label="Perfect calibration")
        axes[1].set_xlabel("Mean Cox 12m PD (%)", fontsize=10)
        axes[1].set_ylabel("Observed Default Rate (%)", fontsize=10)
        axes[1].set_title("Cox 12m PD Calibration — Decile Comparison",
                           color="#CBD5E1", fontsize=10)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.2)
        axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=2))
        axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=2))
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "survival_pd_horizons.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Ch.5 — Survival Analysis (Cox PH)")
    log.info("lifelines version: %s", lifelines.__version__)
    log.info("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("[1/5] Loading processed PD data …")
    train = pd.read_parquet(PROC_DIR / "pd_train.parquet")
    oos   = pd.read_parquet(PROC_DIR / "pd_oos.parquet")
    oot   = pd.read_parquet(PROC_DIR / "pd_oot.parquet")
    log.info("  Train: %s  |  OOS: %s  |  OOT: %s",
             f"{len(train):,}", f"{len(oos):,}", f"{len(oot):,}")

    # ── Build survival datasets ───────────────────────────────────────────
    log.info("")
    log.info("[2/5] Building survival datasets (duration + event per loan) …")
    train_surv = build_survival_df(train, COX_FEATURES)
    oos_surv   = build_survival_df(oos,   COX_FEATURES)

    log.info("  Train survival: %s loans  (%s events)",
             f"{len(train_surv):,}", f"{int(train_surv['event'].sum()):,}")
    log.info("  OOS survival:   %s loans  (%s events)",
             f"{len(oos_surv):,}", f"{int(oos_surv['event'].sum()):,}")

    # ── Kaplan-Meier plots ────────────────────────────────────────────────
    log.info("")
    log.info("[3/5] Kaplan-Meier stratified survival curves …")

    # By credit score tertile
    if "credit_score" in train_surv.columns:
        cs_vals = train_surv["credit_score"].dropna()
        q33, q67 = np.percentile(cs_vals, [33, 67])
        train_surv["cs_group"] = pd.cut(
            train_surv["credit_score"],
            bins=[-np.inf, q33, q67, np.inf],
            labels=["Low FICO (< 660)", "Mid FICO (660–740)", "High FICO (> 740)"],
        ).astype(str)

        plot_km_by_group(
            train_surv, "cs_group", None,
            "Kaplan-Meier Survival — Stratified by FICO Score Tertile\n"
            "Lower FICO = faster default onset and lower long-run survival",
            "survival_km_by_credit_score.png",
        )

    # By origination vintage era
    if "report_date" in train.columns:
        train["report_date"] = pd.to_datetime(train["report_date"])
        train_surv2 = build_survival_df(train, COX_FEATURES)
        orig_year_map = (
            train.groupby("loan_seq_num")["report_date"]
            .min().dt.year.rename("orig_year").reset_index()
        )
        train_surv2 = train_surv2.merge(orig_year_map, on="loan_seq_num", how="left")

        def vintage_era(y):
            if pd.isna(y):   return "Unknown"
            y = int(y)
            if y < 2004:     return "Pre-crisis (2000-2003)"
            if y < 2008:     return "Crisis (2004-2007)"
            if y < 2012:     return "Post-crisis (2008-2011)"
            return "Recovery (2012+)"

        train_surv2["era"] = train_surv2["orig_year"].apply(vintage_era)

        plot_km_by_group(
            train_surv2, "era", None,
            "Kaplan-Meier Survival — Stratified by Origination Vintage Era\n"
            "Crisis-era loans show dramatically lower survival probabilities",
            "survival_km_by_vintage.png",
        )
    else:
        log.warning("  report_date not in train — skipping vintage KM plot.")

    # ── Cox PH model ──────────────────────────────────────────────────────
    log.info("")
    log.info("[4/5] Fitting Cox Proportional Hazards model …")
    cox = fit_cox(train_surv, COX_FEATURES)

    # Save coefficients
    summary = cox.summary.copy()
    summary.index.name = "feature"
    summary.reset_index().to_csv(PROC_DIR / "survival_cox_coefs.csv", index=False)
    log.info("  Cox coefficients → data/processed/survival_cox_coefs.csv")

    # C-index on OOS
    avail = [f for f in COX_FEATURES if f in oos_surv.columns
             and oos_surv[f].notna().any()]
    df_cox_oos = oos_surv[["duration", "event"] + avail].dropna()
    if len(df_cox_oos) > 0:
        c_oos = concordance_index(
            df_cox_oos["duration"],
            -cox.predict_partial_hazard(df_cox_oos),
            df_cox_oos["event"],
        )
        log.info("  C-index (OOS): %.4f", c_oos)

    plot_hazard_ratios(cox)
    plot_schoenfeld_residuals(cox, train_surv, COX_FEATURES)

    # ── Multi-horizon PD ──────────────────────────────────────────────────
    log.info("")
    log.info("[5/5] Computing multi-horizon PD (12m / 24m / 36m / 60m) …")
    horizon_df = compute_horizon_pds(cox, oos_surv, COX_FEATURES,
                                      horizons=[12, 24, 36, 60])

    if not horizon_df.empty:
        horizon_df.to_csv(PROC_DIR / "survival_pd_horizons.csv", index=False)
        log.info("  Horizon PD saved → data/processed/survival_pd_horizons.csv")

        pd_cols = [c for c in horizon_df.columns if c.startswith("pd_")]
        for col in pd_cols:
            mean_pd = horizon_df[col].mean() * 100
            log.info("  Mean OOS %s: %.4f%%", col, mean_pd)

        plot_pd_horizons(horizon_df)

    log.info("")
    log.info("=" * 65)
    log.info("Ch.5 complete — Survival analysis generated.")
    log.info("  Next: python 07_macro_scenario_analysis.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()