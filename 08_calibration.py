"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.7 — PD Calibration
=============================================================================
Script  : 08_calibration.py
Purpose : Align predicted default probabilities with observed default rates
          using Platt scaling and isotonic regression — essential for
          Expected Credit Loss (ECL) calculations under IFRS 9 / Basel IRB.

Why calibration matters in credit risk
----------------------------------------
  Discrimination (AUROC, KS, Gini) measures rank-ordering — whether the
  model correctly identifies *who* is riskier.  Calibration measures
  *how much* riskier — whether predicted probabilities match realised rates.

  A model can discriminate perfectly (AUROC = 1.0) but still be badly
  mis-calibrated.  Example: XGBoost with class_weight=155 scores well on
  AUROC but its raw probabilities are systematically inflated by the
  overweighting of defaults during training.

  For regulatory use:
    Basel II/III IRB — Long-Run Average Default Rate (LRADR) must be
      estimated with sufficient accuracy; calibration is tested annually.
    IFRS 9 ECL       — ECL = PD × LGD × EAD; mis-calibrated PD directly
      mis-states provisions and affects Tier 1 capital ratios.
    OCC SR 11-7      — Backtesting and benchmarking require calibration
      evidence as part of model validation documentation.

Methods
--------
  1. Reliability diagram         — actual default rate vs mean predicted PD
     per decile, both pre- and post-calibration (the standard visual test)

  2. Platt scaling (sigmoid)     — fits a 1D logistic regression on raw
     scores: P_cal = σ(a · f(x) + b).  Preserves rank order.  Fast and
     stable with small calibration sets (≥100 events).

  3. Isotonic regression         — non-parametric monotone function fitted
     directly on the score → observed default mapping.  More flexible than
     Platt but requires more data (≥500 events) and can overfit.

  4. Temperature scaling         — single-parameter variant of Platt:
     P_cal = σ(f(x) / T).  Used in NLP; included for completeness.

  5. Calibration metrics
     - Brier score = mean(p_hat - y)²  (lower = better; 0 = perfect)
     - Expected Calibration Error (ECE) = Σ_b (|B_b|/n) × |acc_b - conf_b|
     - Maximum Calibration Error (MCE) = max_b |acc_b - conf_b|
     - Hosmer-Lemeshow χ² (as in Ch.1)

  6. Long-Run Average PD (LRADR) — regulatory comparison of point-in-time
     calibrated PD vs through-the-cycle target rate

Inputs
------
  data/processed/pd_train.parquet
  data/processed/pd_oos.parquet
  data/processed/pd_oot.parquet
  data/processed/pd_lr_results.csv      (LR raw scores — 02_pd_logistic.py)
  data/processed/pd_xgb_results.csv     (XGB raw scores — 03_pd_ensemble.py)

Outputs
-------
  data/processed/calibration_metrics.csv      — Brier, ECE, MCE before/after
  data/processed/calibrated_scores_oos.csv    — per-loan calibrated PDs
  data/figures/calibration_reliability_lr.png
  data/figures/calibration_reliability_xgb.png
  data/figures/calibration_comparison.png
  data/figures/calibration_lradr.png
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
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("calibration.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = Path("data/processed")
FIG_DIR  = Path("data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "default_12m"
N_BINS   = 10     # calibration diagram bins

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
# CALIBRATION METHODS
# =============================================================================

class PlattCalibrator:
    """
    Platt scaling: fits σ(a·s + b) on a calibration set.

    Uses sklearn LogisticRegression on a 1D feature (the raw score),
    which is exactly equivalent to the classic Platt (1999) procedure.

    The calibration set should be separate from both training and evaluation
    sets to avoid overfitting the calibrator.  Typically the OOS set is split:
    half for calibration, half for evaluation.
    """

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(scores.reshape(-1, 1), y)
        self.a_ = float(self._lr.coef_[0, 0])
        self.b_ = float(self._lr.intercept_[0])
        log.info("  Platt calibrator: a=%.4f  b=%.4f", self.a_, self.b_)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]

    def __repr__(self) -> str:
        return f"PlattCalibrator(a={self.a_:.4f}, b={self.b_:.4f})"


class IsotonicCalibrator:
    """
    Isotonic regression calibrator.

    Fits a piecewise-constant monotone non-decreasing function mapping
    raw scores to calibrated probabilities.  More flexible than Platt
    but requires ≥500 observed events for stability.
    """

    def __init__(self) -> None:
        self._ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._ir.fit(scores, y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._ir.predict(scores)


class TemperatureCalibrator:
    """
    Temperature scaling with bias correction: P_cal = σ((logit(P_raw) − b) / T).

    Pure temperature scaling (T only) cannot fix absolute bias caused by
    class reweighting (scale_pos_weight / class_weight='balanced'), because
    it scales all logits by the same factor without shifting them.  A model
    trained with scale_pos_weight=155 produces logits whose mean is ~+1.2
    (≈23% predicted PD) on a population with a 0.64% true base rate.
    Dividing by any T > 0 preserves the rank order but cannot shift the
    distribution to match the observed base rate.

    The fix adds a bias term b (the intercept) to the temperature model,
    making it P_cal = σ((logit(P_raw) / T) + b).  This is equivalent to
    fitting a 1D logistic regression on logit(P_raw)/T, which is exactly
    what Platt scaling does — but we retain the temperature interpretation
    by first finding the optimal T via NLL grid search, then solving for b
    analytically as the log-odds of the observed base rate minus the mean
    scaled logit.

    T > 1 → softens probabilities (reduces overconfidence)
    T < 1 → sharpens probabilities
    b     → corrects absolute bias from class reweighting
    """

    def __init__(self) -> None:
        self.T_: float = 1.0
        self.b_: float = 0.0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        eps = 1e-7
        scores_c = np.clip(scores, eps, 1 - eps)
        logits   = np.log(scores_c / (1 - scores_c))

        best_T   = 1.0
        best_nll = np.inf

        # Grid-search T; for each T compute optimal b analytically
        # b* = logit(base_rate) - mean(logits / T)  → centres the distribution
        base_rate  = np.clip(y.mean(), eps, 1 - eps)
        target_log_odds = np.log(base_rate / (1 - base_rate))

        for T in np.linspace(0.1, 5.0, 500):
            scaled = logits / T
            b      = target_log_odds - scaled.mean()   # bias correction
            p      = np.clip(1 / (1 + np.exp(-(scaled + b))), eps, 1 - eps)
            nll    = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if nll < best_nll:
                best_nll = nll
                best_T   = T

        # Store optimal T and compute corresponding b
        self.T_ = best_T
        scaled   = logits / best_T
        self.b_  = target_log_odds - scaled.mean()
        log.info("  Temperature calibrator: T=%.4f  b=%+.4f  (NLL=%.6f)",
                 best_T, self.b_, best_nll)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        eps    = 1e-7
        scores = np.clip(scores, eps, 1 - eps)
        logits = np.log(scores / (1 - scores))
        return 1 / (1 + np.exp(-(logits / self.T_ + self.b_)))


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def expected_calibration_error(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b| / n) × |mean_predicted_b − observed_rate_b|

    Weighted average of calibration gap across bins; weights are bin size.
    Lower is better.  A perfectly calibrated model has ECE = 0.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    # Match lengths (calibration_curve may drop empty bins)
    n = len(frac_pos)
    bin_counts_valid = bin_counts[bin_counts > 0][:n]
    weights = bin_counts_valid / bin_counts_valid.sum()
    return float(np.sum(weights * np.abs(frac_pos - mean_pred)))


def hosmer_lemeshow(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     g: int = 10) -> dict:
    hl_df = pd.DataFrame({"y": y_true, "p": y_pred})
    hl_df["decile"] = pd.qcut(hl_df["p"], q=g, duplicates="drop", labels=False)
    grp = hl_df.groupby("decile").agg(
        obs_bad=("y", "sum"),
        obs_good=("y", lambda x: (x == 0).sum()),
        exp_bad=("p", "sum"),
        n=("y", "count"),
    )
    grp["exp_good"] = grp["n"] - grp["exp_bad"]
    hl_stat = (
        ((grp["obs_bad"]  - grp["exp_bad"])**2  / grp["exp_bad"].clip(lower=1e-9)).sum()
      + ((grp["obs_good"] - grp["exp_good"])**2 / grp["exp_good"].clip(lower=1e-9)).sum()
    )
    p_val = 1 - chi2.cdf(hl_stat, df=g - 2)
    return {"hl_stat": round(float(hl_stat), 4), "hl_pval": round(float(p_val), 4)}


def compute_metrics(y_true: np.ndarray,
                     y_prob: np.ndarray,
                     label:  str) -> dict:
    n_pos = y_true.sum()
    if n_pos == 0:
        return {"label": label, "n": len(y_true), "n_events": 0}

    auroc  = roc_auc_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    ece    = expected_calibration_error(y_true, y_prob, N_BINS)
    hl     = hosmer_lemeshow(y_true, y_prob)
    bias   = float(np.mean(y_prob) - np.mean(y_true))

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=N_BINS)
    mce = float(np.max(np.abs(frac_pos - mean_pred)))

    log.info("  %-30s  AUROC=%.4f  Brier=%.6f  ECE=%.6f  Bias=%+.6f  HL-p=%.4f",
             label, auroc, brier, ece, bias, hl["hl_pval"])

    return {
        "label":    label,
        "n":        len(y_true),
        "n_events": int(n_pos),
        "auroc":    round(auroc, 6),
        "brier":    round(brier, 8),
        "ece":      round(ece, 8),
        "mce":      round(mce, 8),
        "bias":     round(bias, 8),
        "hl_stat":  hl["hl_stat"],
        "hl_pval":  hl["hl_pval"],
    }


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_reliability_diagram(y_true: np.ndarray,
                              scores_dict: dict[str, np.ndarray],
                              title: str,
                              filename: str) -> None:
    """
    Reliability diagram: observed default rate vs mean predicted PD per decile.

    Perfect calibration = all points on the diagonal.
    Points above diagonal = model under-predicts (too conservative).
    Points below diagonal = model over-predicts (too aggressive).

    The histogram at the bottom shows the score distribution (confidence).
    """
    plt.rcParams.update(PLT_STYLE)
    fig = plt.figure(figsize=(11, 8))
    gs  = fig.add_gridspec(3, 1, hspace=0.08)
    ax_main = fig.add_subplot(gs[:2])
    ax_hist  = fig.add_subplot(gs[2], sharex=ax_main)

    palette = {"Raw":       "#F59E0B",
               "Platt":     "#38BDF8",
               "Isotonic":  "#10B981",
               "Temperature": "#A78BFA"}

    all_probs = []
    for label, probs in scores_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=N_BINS)
        color = palette.get(label, "#E2E8F0")
        ax_main.plot(mean_pred * 100, frac_pos * 100,
                     marker="o", markersize=6, linewidth=2,
                     color=color, label=label)
        all_probs.append(probs)

    # Perfect calibration diagonal
    ax_main.plot([0, 100], [0, 100], "k--", linewidth=1.2, alpha=0.5,
                 label="Perfect calibration")

    ax_main.set_ylabel("Observed Default Rate (%)", fontsize=11)
    ax_main.set_title(title, fontsize=12, fontweight="bold",
                      color="white", pad=14)
    ax_main.legend(fontsize=10, loc="upper left")
    ax_main.grid(True, alpha=0.2)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))
    ax_main.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Score distribution histogram
    raw_scores = list(scores_dict.values())[0]
    ax_hist.hist(raw_scores * 100, bins=50, color="#38BDF8",
                 alpha=0.7, edgecolor="none")
    ax_hist.set_xlabel("Predicted Default Probability (%)", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.grid(True, alpha=0.2)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))

    path = FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_calibration_comparison(metrics_df: pd.DataFrame) -> None:
    """
    Side-by-side bar charts: Brier score and ECE before/after calibration
    for both LR and XGBoost models.
    """
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Calibration Improvement — Before vs After  (OOS Evaluation Set)\n"
                 "Lower Brier Score and ECE = better calibration",
                 fontsize=12, fontweight="bold", color="white")

    for ax, metric, ylabel in zip(
        axes,
        ["brier", "ece"],
        ["Brier Score  (lower = better)", "Expected Calibration Error  (lower = better)"],
    ):
        labels = metrics_df["label"].tolist()
        values = metrics_df[metric].tolist()

        palette = []
        for lbl in labels:
            if "Raw"  in lbl: palette.append("#F59E0B")
            elif "Platt"      in lbl: palette.append("#38BDF8")
            elif "Isotonic"   in lbl: palette.append("#10B981")
            elif "Temperature"in lbl: palette.append("#A78BFA")
            else: palette.append("#E2E8F0")

        bars = ax.bar(range(len(labels)), values, color=palette,
                      alpha=0.85, edgecolor="none", width=0.6)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.6f}", ha="center", va="bottom",
                    fontsize=8.5, color="#E2E8F0")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "calibration_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_lradr(y_true_by_split: dict[str, np.ndarray],
               pd_by_split:     dict[str, dict[str, np.ndarray]]) -> None:
    """
    Long-Run Average Default Rate (LRADR) comparison.

    Regulators require that calibrated PD estimates be compared against the
    observed long-run average default rate (Basel II §461).  This chart
    shows mean predicted PD vs observed default rate by year / split,
    with and without calibration — a standard model validation output.
    """
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))

    x_pos = np.arange(len(y_true_by_split))
    width = 0.22
    offsets = {"Raw": -width, "Platt": 0, "Isotonic": width}
    palette  = {"Raw": "#F59E0B", "Platt": "#38BDF8", "Isotonic": "#10B981"}

    splits = list(y_true_by_split.keys())
    obs_rates = [y_true_by_split[s].mean() * 100 for s in splits]

    ax.bar(x_pos, obs_rates, width=width * 3.5, color="#4B5563",
           alpha=0.4, edgecolor="none", label="Observed default rate")

    for method, offset in offsets.items():
        pds = [pd_by_split[s].get(method, np.array([np.nan])).mean() * 100
               for s in splits]
        ax.bar(x_pos + offset, pds, width=width, color=palette[method],
               alpha=0.85, edgecolor="none", label=f"Mean PD ({method})")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.upper() for s in splits], fontsize=10)
    ax.set_ylabel("Default Rate / Mean PD (%)", fontsize=10)
    ax.set_title("LRADR Comparison — Calibrated PD vs Observed Default Rate\n"
                 "Basel II §461: Calibrated PD should approximate the long-run average",
                 fontsize=11, fontweight="bold", color="white", pad=14)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f%%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "calibration_lradr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Ch.7 — PD Calibration")
    log.info("  Methods: Platt scaling · Isotonic regression · Temperature scaling")
    log.info("=" * 65)

    # ── Load pre-computed scores ──────────────────────────────────────────
    log.info("")
    log.info("[1/5] Loading raw model scores …")

    models_available = {}
    for model_name, path, score_col in [
        ("LR",  PROC_DIR / "pd_lr_results.csv",  "score"),
        ("XGB", PROC_DIR / "pd_xgb_results.csv", "xgb_score"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            models_available[model_name] = {"df": df, "score_col": score_col}
            log.info("  ✓ Loaded %s scores (%s rows)", model_name, f"{len(df):,}")
        else:
            log.warning("  %s scores not found — run 0%d_pd_*.py first",
                        model_name, 2 if model_name == "LR" else 3)

    if not models_available:
        log.error("No model score files found.  Run 02_ and 03_ scripts first.")
        return

    all_metrics: list[dict] = []
    calibrated_dfs: list[pd.DataFrame] = []

    # ── Calibrate each model ───────────────────────────────────────────────
    log.info("")
    log.info("[2/5] Calibrating models …")

    for model_name, info in models_available.items():
        df         = info["df"]
        score_col  = info["score_col"]

        # Split OOS into calibration and evaluation halves
        oos_df = df[df["split"] == "oos"].copy()
        oot_df = df[df["split"] == "oot"].copy()

        if oos_df.empty:
            log.warning("  No OOS rows for %s — skipping.", model_name)
            continue

        n_cal   = len(oos_df) // 2
        cal_df  = oos_df.iloc[:n_cal]   # calibration set (fit calibrators)
        eval_df = oos_df.iloc[n_cal:]   # evaluation set  (compute metrics)

        y_cal   = cal_df[TARGET].values
        s_cal   = cal_df[score_col].values
        y_eval  = eval_df[TARGET].values
        s_eval  = eval_df[score_col].values
        y_oot   = oot_df[TARGET].values
        s_oot   = oot_df[score_col].values

        if y_cal.sum() < 5:
            log.warning("  Fewer than 5 defaults in calibration set for %s — "
                        "calibrators will be unreliable.", model_name)

        # Raw metrics (uncalibrated)
        log.info("")
        log.info("  %s — Uncalibrated:", model_name)
        raw_metric = compute_metrics(y_eval, s_eval, f"{model_name} Raw (OOS-eval)")
        all_metrics.append(raw_metric)

        # Fit calibrators on cal_df
        platt = PlattCalibrator().fit(s_cal, y_cal)
        iso   = IsotonicCalibrator().fit(s_cal, y_cal)
        temp  = TemperatureCalibrator().fit(s_cal, y_cal)

        # Evaluate on eval_df
        log.info("  %s — After calibration (eval half of OOS):", model_name)
        scores_eval = {
            "Raw":         s_eval,
            "Platt":       platt.predict(s_eval),
            "Isotonic":    iso.predict(s_eval),
            "Temperature": temp.predict(s_eval),
        }

        for label, probs in scores_eval.items():
            if label == "Raw":
                continue
            m = compute_metrics(y_eval, probs, f"{model_name} {label} (OOS-eval)")
            all_metrics.append(m)

        # Reliability diagrams
        plot_reliability_diagram(
            y_eval, scores_eval,
            f"Reliability Diagram — {model_name}  (OOS Evaluation Half)\n"
            "Observed default rate vs mean predicted PD per decile",
            f"calibration_reliability_{model_name.lower()}.png",
        )

        # Save calibrated OOS scores for downstream use
        cal_out = eval_df[["loan_seq_num", "report_date", TARGET, score_col]].copy() \
                  if "loan_seq_num" in eval_df.columns \
                  else eval_df[[TARGET, score_col]].copy()
        cal_out[f"{model_name.lower()}_platt"]       = scores_eval["Platt"]
        cal_out[f"{model_name.lower()}_isotonic"]    = scores_eval["Isotonic"]
        cal_out[f"{model_name.lower()}_temperature"] = scores_eval["Temperature"]
        calibrated_dfs.append(cal_out)

    # ── Save metrics ───────────────────────────────────────────────────────
    log.info("")
    log.info("[3/5] Saving calibration metrics …")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(PROC_DIR / "calibration_metrics.csv", index=False)
    log.info("\n  Calibration metrics summary:")
    display_cols = [c for c in ["label", "brier", "ece", "mce", "bias", "hl_pval"]
                    if c in metrics_df.columns]
    log.info("\n%s", metrics_df[display_cols].to_string(index=False))

    if calibrated_dfs:
        pd.concat(calibrated_dfs, ignore_index=True).to_csv(
            PROC_DIR / "calibrated_scores_oos.csv", index=False
        )
        log.info("  Calibrated scores → data/processed/calibrated_scores_oos.csv")

    # ── Comparison chart ───────────────────────────────────────────────────
    log.info("")
    log.info("[4/5] Generating calibration comparison charts …")
    if not metrics_df.empty:
        plot_calibration_comparison(metrics_df)

    # ── LRADR ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("[5/5] Long-Run Average Default Rate comparison …")
    # Use first available model for LRADR
    if models_available:
        first_model, first_info = next(iter(models_available.items()))
        df        = first_info["df"]
        sc        = first_info["score_col"]
        oos_sub   = df[df["split"] == "oos"].iloc[len(df[df["split"]=="oos"]) // 2:]
        oot_sub   = df[df["split"] == "oot"]

        # Refit calibrators on first half of OOS
        oos_cal   = df[df["split"] == "oos"].iloc[:len(df[df["split"]=="oos"]) // 2]
        platt_lr  = PlattCalibrator().fit(oos_cal[sc].values, oos_cal[TARGET].values)
        iso_lr    = IsotonicCalibrator().fit(oos_cal[sc].values, oos_cal[TARGET].values)

        y_true_by_split = {
            "oos-eval": oos_sub[TARGET].values,
            "oot":      oot_sub[TARGET].values,
        }
        pd_by_split = {
            "oos-eval": {
                "Raw":      oos_sub[sc].values,
                "Platt":    platt_lr.predict(oos_sub[sc].values),
                "Isotonic": iso_lr.predict(oos_sub[sc].values),
            },
            "oot": {
                "Raw":      oot_sub[sc].values,
                "Platt":    platt_lr.predict(oot_sub[sc].values),
                "Isotonic": iso_lr.predict(oot_sub[sc].values),
            },
        }
        plot_lradr(y_true_by_split, pd_by_split)

        # Regulatory summary table
        log.info("")
        log.info("  LRADR Regulatory Summary:")
        for split, y_true in y_true_by_split.items():
            obs_dr = y_true.mean() * 100
            for method, pds in pd_by_split[split].items():
                bias = (pds.mean() - y_true.mean()) * 100
                log.info("    %-10s  %-12s  Observed DR=%.4f%%  Mean PD=%.4f%%  Bias=%+.4f pp",
                         split, method, obs_dr, pds.mean() * 100, bias)

    log.info("")
    log.info("=" * 65)
    log.info("Ch.7 complete — PD calibration analysis generated.")
    log.info("  Pipeline complete.  All four extensions written.")
    log.info("=" * 65)


if __name__ == "__main__":
    main()