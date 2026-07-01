"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.9 — Basel IRB Rating Scale & Capital
=============================================================================
Script  : 10_basel_irb_capital.py
Purpose : Map continuous PD to a rating master scale and compute Basel
          II/III IRB risk-weighted assets (RWA) and Pillar 1 capital for the
          retail residential mortgage exposure class.

Why this script exists
-----------------------
  A PD model is a scorecard, not an IRB model, until two more things exist:

    1. A rating master scale — a mapping from continuous PD to a small set
       of discrete grades (AAA...D) that a credit committee, loan-level
       disclosure, or portfolio limit can actually be built around. Nothing
       upstream in this pipeline produces one.

    2. A capital calculation — the Basel IRB formula that turns (PD, LGD,
       EAD) into risk-weighted assets and a minimum capital requirement.
       Chapters 1-8 stop at PD/LGD/ECL; they never convert those into RWA.

  This script is the first place in the pipeline both exist.

Basel IRB retail residential mortgage formula (CRE31/CRE32)
-------------------------------------------------------------
  Unlike corporate/sovereign/bank exposures, retail residential mortgages
  under the Basel IRB framework use a FIXED asset correlation (no PD-varying
  correlation function) and carry NO maturity adjustment b(PD) — that term
  only applies outside the retail asset classes.

      R = 0.15                                     (config.BASEL_RETAIL_MORTGAGE_CORRELATION)
      K = LGD * N[ G(PD)/sqrt(1-R) + sqrt(R/(1-R)) * G(0.999) ]  -  PD * LGD
      RWA = K * 12.5 * EAD
      Capital = RWA * 8%  =  K * EAD

  where G = inverse standard normal CDF, N = standard normal CDF, and the
  0.999 is the 99.9% supervisory confidence level (config.BASEL_CONFIDENCE).
  K is the capital requirement as a fraction of EAD net of expected loss
  (PD * LGD is already subtracted out, since Basel Pillar 1 capital covers
  UNEXPECTED loss — expected loss is meant to be covered by provisions,
  i.e. the IFRS 9 ECL this pipeline already computes in
  07_macro_scenario_analysis.py).

PD input: through-the-cycle, not point-in-time
------------------------------------------------
  Basel IRB capital wants a PD that does not move with the current point in
  the credit cycle (EBA/GL/2017/16 §6.2) — the opposite of the point-in-time
  PD IFRS 9 provisioning uses. This script therefore prefers the actual TTC
  PD outputs already produced upstream, in order:

    1. data/processed/ttc_calibrated_pd.csv   (08_calibration.py's LRADR-
       anchored TTC PD — logit-shifts the calibrated PIT PD so its
       population mean matches the long-run average default rate)
    2. data/processed/survival_pd_horizons.csv's ttc_pd_12m column
       (06_survival_analysis.py's macro-neutral Cox re-scoring)

  falling back to config.MACRO_LGD_ASSUMPTION-style flat assumptions with a
  warning if neither upstream script has been run yet.

Known limitation
------------------
  LGD is a population-level anchor (04_lgd_models.py's champion mean
  predicted LGD), not per-loan — the same constraint documented in
  07_macro_scenario_analysis.py's known limitations, and for the same
  reason: the PD population doesn't carry the LGD-specific features needed
  for per-loan LGD conditioning. It is also not downturn-adjusted (Basel
  requires a downturn LGD for capital, distinct from the expected/average
  LGD used for ECL) — a production system would maintain both.

Inputs
------
  data/processed/pd_oos.parquet                    (EAD: current_upb / orig_upb)
  data/processed/ttc_calibrated_pd.csv              (preferred TTC PD source)
  data/processed/survival_pd_horizons.csv           (fallback TTC PD source)
  data/processed/lgd_champion_summary.csv           (optional — LGD anchor)

Outputs
-------
  data/processed/basel_irb_capital_by_loan.csv      — per-loan grade, RWA, capital
  data/processed/basel_irb_capital_by_grade.csv     — portfolio RWA/capital by grade
  data/figures/basel_irb_rating_distribution.png
  data/figures/basel_irb_capital_by_grade.png
  data/figures/basel_irb_supervisory_formula.png
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
from scipy.stats import norm

import src.config as config

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING
# =============================================================================

config.configure_logging("basel_irb_capital.log")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = config.PROC_DIR
FIG_DIR  = config.FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

RATING_SCALE  = config.RATING_SCALE
CORRELATION   = config.BASEL_RETAIL_MORTGAGE_CORRELATION
CONFIDENCE    = config.BASEL_CONFIDENCE
MIN_CAP_RATIO = config.BASEL_MIN_CAPITAL_RATIO

PLT_STYLE: dict = config.PLT_STYLE

GRADE_ORDER  = [g for g, _ in RATING_SCALE]
GRADE_COLORS = {
    "AAA": "#10B981", "AA": "#34D399", "A": "#6EE7B7",
    "BBB": "#FBBF24", "BB": "#F59E0B", "B": "#FB923C",
    "CCC": "#EF4444", "D": "#991B1B",
}


# =============================================================================
# RATING MASTER SCALE
# =============================================================================

def assign_rating_grade(pd_values: np.ndarray) -> np.ndarray:
    """Vectorised config.pd_to_rating() over an array of PDs."""
    return np.array([config.pd_to_rating(float(p)) for p in pd_values])


# =============================================================================
# BASEL IRB CAPITAL FORMULA  (retail residential mortgage, CRE31/CRE32)
# =============================================================================

def basel_irb_capital_k(pd_values: np.ndarray,
                         lgd:       float,
                         correlation: float = CORRELATION,
                         confidence: float = CONFIDENCE) -> np.ndarray:
    """
    Basel IRB capital requirement K, as a fraction of EAD, for the retail
    residential mortgage exposure class.

        K = LGD * N[ G(PD)/sqrt(1-R) + sqrt(R/(1-R)) * G(confidence) ] - PD * LGD

    No maturity adjustment b(PD) is applied — that term is specific to
    corporate/sovereign/bank exposures under the advanced IRB approach, not
    retail (Basel Framework CRE32.10).

    PD is clipped away from {0, 1} before the inverse-normal transform to
    avoid +/-inf; a PD of exactly 0 or 1 is not a meaningful IRB input.
    """
    eps = 1e-6
    pd_clipped = np.clip(pd_values, eps, 1 - eps)

    g_pd   = norm.ppf(pd_clipped)
    g_conf = norm.ppf(confidence)

    conditional_pd = norm.cdf(
        g_pd / np.sqrt(1 - correlation)
        + np.sqrt(correlation / (1 - correlation)) * g_conf
    )
    k = lgd * conditional_pd - pd_clipped * lgd
    return np.clip(k, 0.0, 1.0)


def compute_rwa(k: np.ndarray, ead: np.ndarray) -> np.ndarray:
    """RWA = K * 12.5 * EAD.  12.5 = 1 / 8% Pillar 1 minimum capital ratio."""
    return k * (1.0 / MIN_CAP_RATIO) * ead


# =============================================================================
# TTC PD LOADING  (prefers 08_calibration.py, falls back to 06_survival_analysis.py)
# =============================================================================

def load_ttc_pd() -> tuple[pd.DataFrame, str]:
    """
    Load a per-loan TTC PD, preferring 08_calibration.py's LRADR-anchored
    output and falling back to 06_survival_analysis.py's macro-neutral Cox
    re-scoring if 08 hasn't been run yet.

    Returns (DataFrame[loan_seq_num, ttc_pd], source_description).
    """
    calib_path = PROC_DIR / "ttc_calibrated_pd.csv"
    if calib_path.exists():
        df = pd.read_csv(calib_path)
        if not df.empty and "ttc_pd" in df.columns:
            out = df[["loan_seq_num", "ttc_pd"]].drop_duplicates("loan_seq_num")
            return out, f"{calib_path.name} (LRADR-anchored calibrated PD)"

    survival_path = PROC_DIR / "survival_pd_horizons.csv"
    if survival_path.exists():
        df = pd.read_csv(survival_path)
        if not df.empty and "ttc_pd_12m" in df.columns:
            out = df[["loan_seq_num", "ttc_pd_12m"]].rename(
                columns={"ttc_pd_12m": "ttc_pd"}
            ).drop_duplicates("loan_seq_num")
            return out, f"{survival_path.name} (macro-neutral Cox re-scoring)"

    log.warning(
        "  Neither %s nor %s found — run 08_calibration.py or "
        "06_survival_analysis.py first for a real TTC PD. Returning empty.",
        calib_path, survival_path,
    )
    return pd.DataFrame(columns=["loan_seq_num", "ttc_pd"]), "none available"


def load_base_lgd() -> tuple[float, str]:
    """
    Load the base LGD anchor from 04_lgd_models.py's champion selection —
    the same optional-input-with-fallback pattern used in
    07_macro_scenario_analysis.py's load_base_lgd().
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
# VISUALISATIONS
# =============================================================================

def plot_rating_distribution(by_loan: pd.DataFrame) -> None:
    """Portfolio count and EAD share by rating grade."""
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Rating Master Scale — Portfolio Distribution\n"
        "PD -> rating grade mapping (config.RATING_SCALE)",
        fontsize=12, fontweight="bold", color="white"
    )

    counts = by_loan["grade"].value_counts().reindex(GRADE_ORDER).fillna(0)
    axes[0].bar(counts.index, counts.values,
                color=[GRADE_COLORS[g] for g in counts.index], alpha=0.85)
    axes[0].set_ylabel("Number of Loans", fontsize=10)
    axes[0].set_title("Loan Count by Grade", color="#CBD5E1", fontsize=10)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    ead_by_grade = by_loan.groupby("grade")["ead"].sum().reindex(GRADE_ORDER).fillna(0) / 1e6
    axes[1].bar(ead_by_grade.index, ead_by_grade.values,
                color=[GRADE_COLORS[g] for g in ead_by_grade.index], alpha=0.85)
    axes[1].set_ylabel("Total EAD ($M)", fontsize=10)
    axes[1].set_title("Exposure by Grade", color="#CBD5E1", fontsize=10)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "basel_irb_rating_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_capital_by_grade(by_grade: pd.DataFrame) -> None:
    """RWA and capital requirement by rating grade."""
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle(
        "Basel IRB Capital by Rating Grade  (Retail Residential Mortgage)\n"
        "RWA = K x 12.5 x EAD   |   Capital = RWA x 8%",
        fontsize=12, fontweight="bold", color="white"
    )

    x = np.arange(len(by_grade))
    width = 0.35
    ax.bar(x - width / 2, by_grade["total_rwa_$M"], width,
           color="#38BDF8", alpha=0.85, label="RWA ($M)")
    ax.bar(x + width / 2, by_grade["total_capital_$M"], width,
           color="#F59E0B", alpha=0.85, label="Capital requirement ($M)")

    ax.set_xticks(x)
    ax.set_xticklabels(by_grade["grade"], fontsize=10)
    ax.set_ylabel("$ Millions", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "basel_irb_capital_by_grade.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_supervisory_formula(lgd: float) -> None:
    """K(PD) supervisory formula curve — the standard Basel validation chart."""
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    pd_grid = np.linspace(0.0001, 0.20, 400)
    k_grid  = basel_irb_capital_k(pd_grid, lgd)

    ax.plot(pd_grid * 100, k_grid * 100, color="#38BDF8", linewidth=2.5)
    ax.set_xlabel("PD (%)", fontsize=10)
    ax.set_ylabel("Capital Requirement K (% of EAD)", fontsize=10)
    ax.set_title(
        f"Basel IRB Supervisory Formula — Retail Residential Mortgage\n"
        f"R={CORRELATION}, confidence={CONFIDENCE:.1%}, LGD={lgd:.2f}",
        fontsize=11, fontweight="bold", color="white", pad=14
    )
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))

    fig.tight_layout()
    path = FIG_DIR / "basel_irb_supervisory_formula.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Ch.9 — Basel IRB Rating Scale & Capital")
    log.info("=" * 65)

    # ── Load TTC PD ───────────────────────────────────────────────────────
    log.info("")
    log.info("[1/4] Loading through-the-cycle (TTC) PD ...")
    ttc_pd_df, pd_source = load_ttc_pd()
    if ttc_pd_df.empty:
        log.error("  No TTC PD available — run 06_survival_analysis.py and/or "
                   "08_calibration.py first. Aborting.")
        return
    log.info("  TTC PD source: %s  (%s loans)", pd_source, f"{len(ttc_pd_df):,}")

    # ── Load EAD ──────────────────────────────────────────────────────────
    log.info("")
    log.info("[2/4] Loading exposure at default (EAD) ...")
    oos = pd.read_parquet(PROC_DIR / "pd_oos.parquet")
    ead_col = "current_upb" if "current_upb" in oos.columns else "orig_upb"
    ead_df = (
        oos.sort_values("loan_age")
        .groupby("loan_seq_num", as_index=False)
        .last()[["loan_seq_num", ead_col]]
        .rename(columns={ead_col: "ead"})
    )
    log.info("  EAD anchor column: %s  (%s loans)", ead_col, f"{len(ead_df):,}")

    base_lgd, lgd_source = load_base_lgd()
    log.info("  Base LGD: %.4f (%s)", base_lgd, lgd_source)

    # ── Merge and compute ─────────────────────────────────────────────────
    log.info("")
    log.info("[3/4] Assigning rating grades and computing Basel IRB capital ...")
    by_loan = ttc_pd_df.merge(ead_df, on="loan_seq_num", how="inner")
    by_loan["ead"] = by_loan["ead"].fillna(0.0).clip(lower=0.0)
    by_loan["lgd"] = base_lgd

    by_loan["grade"] = assign_rating_grade(by_loan["ttc_pd"].values)
    by_loan["capital_k"] = basel_irb_capital_k(by_loan["ttc_pd"].values, base_lgd)
    by_loan["rwa"] = compute_rwa(by_loan["capital_k"].values, by_loan["ead"].values)
    by_loan["capital_required"] = by_loan["rwa"] * MIN_CAP_RATIO

    by_loan.to_csv(PROC_DIR / "basel_irb_capital_by_loan.csv", index=False)
    log.info("  Per-loan capital → data/processed/basel_irb_capital_by_loan.csv")

    by_grade = (
        by_loan.groupby("grade")
        .agg(
            n_loans=("loan_seq_num", "count"),
            mean_pd=("ttc_pd", "mean"),
            total_ead_M=("ead", lambda s: s.sum() / 1e6),
            total_rwa_M=("rwa", lambda s: s.sum() / 1e6),
            total_capital_M=("capital_required", lambda s: s.sum() / 1e6),
        )
        .reindex(GRADE_ORDER)
        .dropna(how="all")
        .reset_index()
        .rename(columns={
            "total_ead_M": "total_ead_$M",
            "total_rwa_M": "total_rwa_$M",
            "total_capital_M": "total_capital_$M",
        })
    )
    by_grade.to_csv(PROC_DIR / "basel_irb_capital_by_grade.csv", index=False)
    log.info("  Grade-level summary → data/processed/basel_irb_capital_by_grade.csv")

    total_ead      = by_loan["ead"].sum()
    total_rwa      = by_loan["rwa"].sum()
    total_capital  = by_loan["capital_required"].sum()
    avg_risk_weight = total_rwa / total_ead if total_ead > 0 else float("nan")

    log.info("")
    log.info("  Portfolio Basel IRB Summary:")
    log.info("    Total EAD:            $%.2fM", total_ead / 1e6)
    log.info("    Total RWA:            $%.2fM", total_rwa / 1e6)
    log.info("    Total capital (8%%):   $%.2fM", total_capital / 1e6)
    log.info("    Average risk weight:  %.1f%%", avg_risk_weight * 100)
    log.info("")
    log.info("  By grade:")
    log.info("\n%s", by_grade.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("[4/4] Generating outputs ...")
    plot_rating_distribution(by_loan)
    plot_capital_by_grade(by_grade)
    plot_supervisory_formula(base_lgd)

    log.info("")
    log.info("=" * 65)
    log.info("Ch.9 complete — rating master scale and Basel IRB capital generated.")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
