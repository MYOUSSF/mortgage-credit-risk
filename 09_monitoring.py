"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.8 — Ongoing Model Monitoring
=============================================================================
Script  : 09_monitoring.py
Purpose : Population Stability Index (PSI) monitoring of feature and score
          distributions over time — the periodic check a model risk
          management function runs after a model goes live, distinct from
          the one-off IV/PSI feature screen performed at build time in
          01_data_preprocessing.py.

Why this is a separate script from feature selection
------------------------------------------------------
  01_data_preprocessing.py computes PSI once, comparing OOS/OOT against
  Train, as part of *choosing* which features to build the model with.

  This script answers a different, recurring question: "has the world the
  model now sees drifted away from the world it was trained on?" It fixes
  the reference (training) distribution once and re-tests every subsequent
  period against that same fixed reference — the standard SR 11-7 /
  EBA GL/2017/16 ongoing-monitoring pattern. Feature bin edges must stay
  fixed across periods for PSI trends to be comparable over time; recomputing
  edges from each new period (as a naive re-run of the build-time PSI check
  would) breaks that comparability.

Method
------
  1. Fit each feature's reference distribution once from the training set
     (10 quantile bins for continuous features, raw categories for
     categorical / low-cardinality features).
  2. Bucket the temporal out-of-time (OOT) population into yearly monitoring
     periods by report_date, plus the OOS split as a single "post-build"
     checkpoint.
  3. For every (period, feature) pair, bin the period's values into the
     *reference* bins and compute:
         PSI = Σ_i (p_i − q_i) · ln(p_i / q_i)
     using the same thresholds as the build-time check: < 0.10 stable,
     0.10–0.25 investigate, > 0.25 major shift.
  4. Repeat for the model's score distribution (if score outputs from
     02_pd_logistic_regression.py are available), and for the observed
     12-month default rate vs the training-set average.

Inputs
------
  data/processed/pd_train.parquet
  data/processed/pd_oos.parquet
  data/processed/pd_oot.parquet
  data/processed/pd_lr_results.csv   (optional — per-loan scores; if
                                       absent, score-drift monitoring is
                                       skipped but feature monitoring
                                       still runs)

Outputs
-------
  data/processed/monitoring_feature_psi.csv
  data/processed/monitoring_score_psi.csv     (only if scores are available)
  data/processed/monitoring_default_rate.csv
  data/figures/monitoring_psi_heatmap.png

Next Step
---------
  Re-run this script each time a new period of servicing data lands, and
  alert on any "major shift" (PSI > 0.25) row before trusting model output
  for that period.
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
from pathlib import Path

import config

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING
# =============================================================================

config.configure_logging("monitoring.log")
log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = config.PROC_DIR
FIG_DIR  = config.FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = config.TARGET_PD
N_BINS = 10

# Same feature set the PD models are built on (config.PD_FEATURES) — drift
# in any of these is what would actually move the score.
MONITORED_FEATURES = config.PD_FEATURES

PLT_STYLE: dict = config.PLT_STYLE


# =============================================================================
# PSI AGAINST A FIXED REFERENCE DISTRIBUTION
# =============================================================================

def fit_reference_distribution(ref: pd.Series, n_bins: int = N_BINS) -> dict:
    """
    Fit a feature's reference (training) distribution once.

    Returns a dict describing how to bin any future period's values into
    these same bins, so PSI stays comparable period over period.
    """
    ref = ref.dropna()
    is_cat = ref.dtype == object or ref.nunique() <= 10
    if is_cat:
        p = ref.value_counts(normalize=True)
        return {"type": "cat", "p": p}

    _, edges = pd.qcut(ref, q=n_bins, duplicates="drop", retbins=True)
    p = (
        pd.cut(ref, bins=edges, include_lowest=True)
        .value_counts(normalize=True)
        .sort_index()
    )
    return {"type": "num", "edges": edges, "p": p}


def psi_against_reference(test: pd.Series, ref_dist: dict) -> tuple[float, float]:
    """
    PSI of `test` against a pre-fitted reference distribution.

    Returns (psi, pct_outside_reference_range). The second value flags
    values the model has never been trained on — itself a governance signal
    even before PSI crosses a threshold.
    """
    test = test.dropna()
    n = len(test)
    if n == 0:
        return np.nan, np.nan

    # Raw counts normalised by the full period length (not by
    # value_counts(normalize=True), which divides by the count of values
    # that landed in a known bin — if a period's values fall entirely
    # outside the reference range that denominator is 0, producing NaN
    # proportions that pandas' .sum(skipna=True) then silently drops,
    # masking a complete population shift as PSI ~ 0).
    if ref_dist["type"] == "cat":
        counts = test.value_counts().reindex(ref_dist["p"].index, fill_value=0)
        pct_outside = 1.0 - test.isin(ref_dist["p"].index).mean()
    else:
        binned = pd.cut(test, bins=ref_dist["edges"], include_lowest=True)
        counts = binned.value_counts().reindex(ref_dist["p"].index, fill_value=0)
        pct_outside = binned.isna().mean()

    p = ref_dist["p"].clip(lower=1e-9)
    q = (counts / n).clip(lower=1e-9)
    psi = float(((p - q) * np.log(p / q)).sum())
    return psi, float(pct_outside)


psi_flag = config.psi_flag


# =============================================================================
# MONITORING PERIODS
# =============================================================================

def build_monitoring_periods(oos: pd.DataFrame, oot: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    OOS is reported as a single post-build checkpoint; OOT is split into
    yearly periods by report_date so drift can be tracked over time rather
    than collapsed into one large temporal holdout.
    """
    periods: dict[str, pd.DataFrame] = {"OOS (post-build)": oos}

    years = sorted(oot["report_date"].dt.year.dropna().unique())
    for year in years:
        periods[f"OOT {year}"] = oot[oot["report_date"].dt.year == year]

    return periods


# =============================================================================
# FEATURE-LEVEL MONITORING
# =============================================================================

def monitor_features(train: pd.DataFrame, periods: dict[str, pd.DataFrame],
                      features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in train.columns:
            continue
        ref_dist = fit_reference_distribution(train[feat])

        for period_name, period_df in periods.items():
            if feat not in period_df.columns:
                continue
            psi, pct_outside = psi_against_reference(period_df[feat], ref_dist)
            rows.append({
                "period":  period_name,
                "feature": feat,
                "n":       period_df[feat].notna().sum(),
                "psi":     round(psi, 4) if not np.isnan(psi) else np.nan,
                "flag":    psi_flag(psi),
                "pct_outside_reference_range": round(pct_outside, 4)
                           if not np.isnan(pct_outside) else np.nan,
            })

    return pd.DataFrame(rows)


# =============================================================================
# SCORE-LEVEL MONITORING
# =============================================================================

def monitor_score(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Score PSI per monitoring period, using pd_lr_results.csv output from
    02_pd_logistic_regression.py (columns: report_date, split, score).
    """
    train_scores = scores.loc[scores["split"] == "train", "score"]
    ref_dist = fit_reference_distribution(train_scores, n_bins=N_BINS)

    oos = scores[scores["split"] == "oos"]
    oot = scores[scores["split"] == "oot"].copy()

    rows = []
    psi, pct_outside = psi_against_reference(oos["score"], ref_dist)
    rows.append({"period": "OOS (post-build)", "n": len(oos),
                  "psi": round(psi, 4), "flag": psi_flag(psi),
                  "pct_outside_reference_range": round(pct_outside, 4)})

    for year in sorted(oot["report_date"].dt.year.dropna().unique()):
        sub = oot[oot["report_date"].dt.year == year]
        psi, pct_outside = psi_against_reference(sub["score"], ref_dist)
        rows.append({"period": f"OOT {year}", "n": len(sub),
                      "psi": round(psi, 4), "flag": psi_flag(psi),
                      "pct_outside_reference_range": round(pct_outside, 4)})

    return pd.DataFrame(rows)


# =============================================================================
# DEFAULT RATE DRIFT
# =============================================================================

def monitor_default_rate(train: pd.DataFrame,
                          periods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    train_rate = train[TARGET].mean()
    rows = []
    for period_name, period_df in periods.items():
        if TARGET not in period_df.columns or len(period_df) == 0:
            continue
        obs_rate = period_df[TARGET].mean()
        rows.append({
            "period":                period_name,
            "n":                     len(period_df),
            "observed_default_rate": round(obs_rate, 6),
            "train_default_rate":    round(train_rate, 6),
            "delta_pp":              round((obs_rate - train_rate) * 100, 4),
        })
    return pd.DataFrame(rows)


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_psi_heatmap(feature_psi: pd.DataFrame, path: Path) -> None:
    if feature_psi.empty:
        return

    pivot = feature_psi.pivot(index="feature", columns="period", values="psi")
    # keep OOS first, then OOT years chronologically
    period_order = sorted(pivot.columns, key=lambda c: (c != "OOS (post-build)", c))
    pivot = pivot[period_order]

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 3, 0.5 * len(pivot) + 2))
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=0.5, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=8)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="black" if v < 0.35 else "white")

        ax.set_title("Feature PSI vs Training Reference\n"
                      "green < 0.10 stable  |  amber 0.10–0.25 investigate  |  red > 0.25 major shift",
                      fontsize=10, color="white")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(colors="#A0AEC0")

        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
        plt.close(fig)
    log.info("  Figure -> %s", path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Ch.8 — Ongoing Model Monitoring")
    log.info("=" * 65)

    log.info("")
    log.info("[1/4] Loading Train / OOS / OOT populations …")
    import pyarrow.parquet as pq
    available_cols = pq.read_schema(PROC_DIR / "pd_train.parquet").names
    features_present = [f for f in MONITORED_FEATURES if f in available_cols]
    load_cols = features_present + [TARGET, "report_date"]

    train = pd.read_parquet(PROC_DIR / "pd_train.parquet", columns=load_cols)
    oos   = pd.read_parquet(PROC_DIR / "pd_oos.parquet",   columns=load_cols)
    oot   = pd.read_parquet(PROC_DIR / "pd_oot.parquet",   columns=load_cols)
    for df in (train, oos, oot):
        df["report_date"] = pd.to_datetime(df["report_date"])

    periods = build_monitoring_periods(oos, oot)
    log.info("  Monitoring periods: %s", ", ".join(periods.keys()))
    log.info("  Features monitored: %s", features_present)

    log.info("")
    log.info("[2/4] Computing feature-level PSI against training reference …")
    feature_psi = monitor_features(train, periods, features_present)
    feature_psi.to_csv(PROC_DIR / "monitoring_feature_psi.csv", index=False)

    breaches = feature_psi[feature_psi["flag"] != "Stable"]
    if not breaches.empty:
        log.info("\n%s", breaches.sort_values(["flag", "psi"], ascending=[True, False])
                                   .to_string(index=False))
    else:
        log.info("  All monitored features remain within the stable PSI band (< 0.10).")

    log.info("")
    log.info("[3/4] Computing score-level PSI and default-rate drift …")

    score_path = PROC_DIR / "pd_lr_results.csv"
    if score_path.exists():
        scores = pd.read_csv(score_path, parse_dates=["report_date"])
        score_psi = monitor_score(scores)
        score_psi.to_csv(PROC_DIR / "monitoring_score_psi.csv", index=False)
        log.info("\n%s", score_psi.to_string(index=False))
    else:
        log.warning("  %s not found — run 02_pd_logistic_regression.py first "
                    "to enable score-level monitoring. Skipping.", score_path)

    default_rate = monitor_default_rate(train, periods)
    default_rate.to_csv(PROC_DIR / "monitoring_default_rate.csv", index=False)
    log.info("\n%s", default_rate.to_string(index=False))

    log.info("")
    log.info("[4/4] Generating PSI heatmap …")
    plot_psi_heatmap(feature_psi, FIG_DIR / "monitoring_psi_heatmap.png")

    major_shifts = feature_psi[feature_psi["flag"] == "Major shift"]
    log.info("")
    log.info("=" * 65)
    if not major_shifts.empty:
        log.warning("ALERT: %d feature/period combinations show a major "
                    "population shift (PSI > 0.25):", len(major_shifts))
        for _, row in major_shifts.iterrows():
            log.warning("  %-20s  %-22s  PSI=%.3f", row["period"], row["feature"], row["psi"])
    else:
        log.info("No major population shifts detected.")
    log.info("Monitoring run complete.")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
