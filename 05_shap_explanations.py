"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.4 — SHAP Explanations
=============================================================================
Script  : 05_shap_explanations.py
Purpose : Individual loan-level feature attribution for the XGBoost PD model
          using SHAP (SHapley Additive exPlanations), aligned with the
          BCBS 239 principle of risk data aggregation and reporting.

Why SHAP matters in banking
----------------------------
  Regulators (PRA, ECB, OCC) and the Basel Committee expect banks to be able
  to explain *why* a model produced a given score for any individual loan.
  Simple feature importances (gain, permutation) are global — they describe
  the model on average.  SHAP provides *local* explanations:

    "This mortgage was assigned PD = 3.2% instead of the portfolio average
     of 0.8%.  The main drivers were a CLTV of 95% (+1.4pp), FICO of 620
     (+0.9pp), and rising unemployment in the ZIP code (+0.3pp)."

  This is directly reportable to credit committees and model risk reviewers.

BCBS 239 alignment
-------------------
  Principle 2  — Data architecture and IT infrastructure
  Principle 6  — Completeness
  Principle 11 — Accuracy and integrity of risk reports

  SHAP waterfall and force plots provide auditable, loan-level attribution
  that satisfies the completeness and accuracy principles.

Methods
-------
  1. Global importance — SHAP mean |φ_i| across all training loans
  2. Summary beeswarm  — SHAP value distribution per feature (colour = raw value)
  3. Dependence plots  — φ_i vs raw feature value, coloured by interaction term
  4. Waterfall plots   — individual loan attribution (high-risk and low-risk)
  5. Force plots       — saved as HTML for interactive review
  6. SHAP-based segment report — mean SHAP per portfolio decile

Inputs
------
  data/processed/pd_train.parquet
  data/processed/pd_oos.parquet
  data/processed/pd_xgb_results.csv   (scores produced by 03_pd_ensemble.py)

Outputs
-------
  data/figures/shap_global_importance.png
  data/figures/shap_beeswarm.png
  data/figures/shap_dependence_credit_score.png
  data/figures/shap_dependence_orig_cltv.png
  data/figures/shap_waterfall_high_risk.png
  data/figures/shap_waterfall_low_risk.png
  data/figures/shap_segment_report.png
  data/processed/shap_values_oos.parquet   (full SHAP matrix for OOS set)
  data/processed/shap_segment_report.csv

Prerequisites
-------------
  pip install shap>=0.42.0
  Run 03_pd_ensemble.py first.
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
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    import shap
    shap.initjs()
except ImportError:
    print("ERROR: shap not installed.  Run: pip install shap>=0.42.0")
    sys.exit(1)

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
        logging.FileHandler("shap_explanations.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = Path("data/processed")
FIG_DIR  = Path("data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "default_12m"
SEED   = 42

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

# Number of OOS loans to compute SHAP values for (full set can be slow on CPU)
SHAP_SAMPLE_N = 5_000

# Plot style — dark, professional
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

FEATURE_LABELS: dict[str, str] = {
    "delinquency_indicator": "Current Delinquency",
    "hpi_change":            "HPI Change (Orig/Curr)",
    "occupancy_status":      "Occupancy Type",
    "orig_interest_rate":    "Note Rate (%)",
    "orig_cltv":             "Combined LTV (%)",
    "num_borrowers":         "Number of Borrowers",
    "credit_score":          "FICO Score",
    "property_type":         "Property Type",
    "loan_age":              "Loan Age (Months)",
    "orig_dti":              "Debt-to-Income (%)",
    "orig_upb":              "Original UPB ($)",
    "ur_3m_lag":             "Unemployment Rate (3m lag)",
}


# =============================================================================
# DATA PREPARATION  (mirrors 03_pd_ensemble.py exactly so XGBoost re-fits
# on identical inputs — in production you would pickle the fitted model)
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
                                           list[str]]:
    feats = [f for f in FEATURES if f in train.columns]
    train, oos = _encode_categoricals(train.copy(), oos.copy(), CAT_FEATURES)

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(train[feats])
    X_oo = imputer.transform(oos[feats])

    return (
        X_tr, X_oo,
        train[TARGET].values, oos[TARGET].values,
        feats,
    )


# =============================================================================
# RETRAIN XGBOOST  (identical hyper-parameters to 03_pd_ensemble.py)
# =============================================================================

def retrain_xgboost(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_oo: np.ndarray, y_oo: np.ndarray) -> XGBClassifier:
    """
    Re-fit the XGBoost model used in Ch.2.

    In a production system this would be replaced by loading a pickled model.
    We retrain here to keep scripts self-contained and avoid large binary
    artefacts in the repository.
    """
    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = neg / max(pos, 1)
    log.info("  scale_pos_weight = %.1f  (%s neg / %s pos)",
             spw, f"{neg:,}", f"{pos:,}")

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.5,
        colsample_bytree=0.8,
        min_child_weight=50,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="auc",
        tree_method="hist",
        device="cpu",          # SHAP TreeExplainer runs on CPU
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=20,
        scale_pos_weight=spw,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_oo, y_oo)], verbose=50)
    log.info("  Best iteration: %d", xgb.best_iteration)
    return xgb


# =============================================================================
# SHAP COMPUTATION
# =============================================================================

def compute_shap(model: XGBClassifier,
                 X: np.ndarray,
                 feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values using the exact TreeExplainer (zero approximation error
    for tree-based models).

    Returns
    -------
    shap_values : (n_samples, n_features)  — φ_i for each observation
    expected_value : float                  — E[f(X)] = base rate in log-odds
    """
    log.info("  Computing SHAP values for %d observations …", X.shape[0])
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary XGBoost, shap_values is (n, p) — log-odds contributions
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    log.info("  SHAP matrix: %s", shap_values.shape)
    return shap_values, float(explainer.expected_value)


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_global_importance(shap_values: np.ndarray,
                            feature_names: list[str]) -> None:
    """Bar chart of mean |SHAP| — global feature importance in probability units."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)
    labels   = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
    vals     = mean_abs[order]

    # Convert log-odds SHAP to approximate probability-scale via sigmoid derivative
    # at base rate ~0.8%: σ'(logit(0.008)) ≈ 0.008*(1-0.008) ≈ 0.00794
    # Multiply by 100 to express as pp
    vals_pp = vals * 0.008 * 0.992 * 100

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.YlOrRd(np.linspace(0.35, 0.95, len(vals)))
    bars   = ax.barh(labels, vals_pp, color=colors, edgecolor="none", height=0.65)

    for bar, v in zip(bars, vals_pp):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"+{v:.3f} pp", va="center", fontsize=9, color="#E2E8F0")

    ax.set_xlabel("Mean |SHAP|  (approx. impact on default probability, pp)",
                  fontsize=10, color="#CBD5E1")
    ax.set_title("SHAP Global Feature Importance — XGBoost PD Model (OOS)\n"
                 "Mean absolute Shapley value across held-out loans",
                 fontsize=12, fontweight="bold", color="white", pad=14)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(0, vals_pp.max() * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / "shap_global_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_beeswarm(shap_values: np.ndarray,
                   X:           np.ndarray,
                   feature_names: list[str]) -> None:
    """
    SHAP summary beeswarm plot.

    Each dot is one loan.  Position on x-axis = SHAP value (log-odds impact).
    Colour = raw feature value (red = high, blue = low).
    Shows simultaneously: direction, magnitude, and distribution of each
    feature's influence — the most information-dense SHAP visualisation.
    """
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(11, 7))

    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]   # top features first

    top_n  = min(10, len(feature_names))
    order  = order[:top_n]
    sv_sub = shap_values[:, order]
    X_sub  = X[:, order]
    labs   = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]

    for row_idx, (feat_shap, feat_raw, label) in enumerate(
        zip(sv_sub.T[::-1], X_sub.T[::-1], labs[::-1])
    ):
        # Normalise raw feature to [0,1] for colour mapping
        fmin, fmax = np.nanpercentile(feat_raw, [2, 98])
        norm = np.clip((feat_raw - fmin) / max(fmax - fmin, 1e-9), 0, 1)

        # Jitter on y-axis to show density
        jitter = np.random.default_rng(row_idx).uniform(-0.3, 0.3, len(feat_shap))

        sc = ax.scatter(feat_shap,
                        np.full(len(feat_shap), row_idx) + jitter,
                        c=norm, cmap="coolwarm", s=6, alpha=0.5,
                        linewidths=0, vmin=0, vmax=1)

    ax.axvline(0, color="#4B5563", linewidth=1.2, linestyle="--")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labs[::-1], fontsize=10)
    ax.set_xlabel("SHAP Value (log-odds impact on default probability)", fontsize=10)
    ax.set_title("SHAP Summary Plot — Top Features by Impact on Default Probability\n"
                 "Each point = one loan  ·  Colour: red = high feature value, blue = low",
                 fontsize=11, fontweight="bold", color="white", pad=14)
    ax.grid(True, axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Feature value\n(normalised)", fontsize=9, color="#CBD5E1")
    cbar.ax.yaxis.set_tick_params(color="#A0AEC0")

    fig.tight_layout()
    path = FIG_DIR / "shap_beeswarm.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_dependence(shap_values: np.ndarray,
                    X:           np.ndarray,
                    feature_names: list[str],
                    feature:       str,
                    interaction:   str | None = None) -> None:
    """
    SHAP dependence plot: φ_i vs raw value of `feature`.

    Colour = raw value of `interaction` feature (if provided) — reveals
    how two features jointly drive default risk.  Regulators often ask:
    "Does a high DTI hurt less-risky borrowers differently than high-LTV
    borrowers?"  This plot answers that directly.
    """
    if feature not in feature_names:
        log.warning("  Feature %s not in list — skipping dependence plot.", feature)
        return

    idx   = feature_names.index(feature)
    sv    = shap_values[:, idx]
    fval  = X[:, idx]

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))

    if interaction and interaction in feature_names:
        inter_idx  = feature_names.index(interaction)
        inter_vals = X[:, inter_idx]
        fmin, fmax = np.nanpercentile(inter_vals, [2, 98])
        norm  = np.clip((inter_vals - fmin) / max(fmax - fmin, 1e-9), 0, 1)
        sc    = ax.scatter(fval, sv, c=norm, cmap="RdYlGn_r",
                           s=8, alpha=0.4, linewidths=0, vmin=0, vmax=1)
        inter_label = FEATURE_LABELS.get(interaction, interaction)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(inter_label + "\n(normalised)", fontsize=9, color="#CBD5E1")
    else:
        ax.scatter(fval, sv, color="#5DADE2", s=8, alpha=0.4, linewidths=0)

    ax.axhline(0, color="#4B5563", linewidth=1.2, linestyle="--",
               label="Zero SHAP = baseline")
    ax.set_xlabel(FEATURE_LABELS.get(feature, feature), fontsize=11)
    ax.set_ylabel("SHAP Value (log-odds contribution to default risk)", fontsize=10)
    title_feat = FEATURE_LABELS.get(feature, feature)
    ax.set_title(f"SHAP Dependence Plot — {title_feat}\n"
                 "Positive SHAP → increases default probability",
                 fontsize=11, fontweight="bold", color="white", pad=12)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / f"shap_dependence_{feature}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)


def plot_waterfall(shap_values:   np.ndarray,
                   expected_value: float,
                   X:              np.ndarray,
                   feature_names:  list[str],
                   loan_idx:       int,
                   title_suffix:   str,
                   filename:       str) -> None:
    """
    SHAP waterfall chart for a single loan.

    Starts at E[f(X)] (portfolio average log-odds) and shows how each
    feature pushes the score up or down to reach the loan's final PD.
    This is the primary regulatory explanation output.
    """
    sv   = shap_values[loan_idx]
    base = expected_value

    # Sort features by absolute SHAP magnitude for this loan
    order  = np.argsort(np.abs(sv))[::-1]
    top_n  = min(10, len(feature_names))
    others = sv[order[top_n:]].sum()

    feats_shown = [feature_names[i] for i in order[:top_n]]
    sv_shown    = sv[order[:top_n]]
    xvals_shown = X[loan_idx, order[:top_n]]

    # Add "All other features" row
    if len(order) > top_n:
        feats_shown = feats_shown + ["(all other features)"]
        sv_shown    = np.append(sv_shown, others)
        xvals_shown = np.append(xvals_shown, np.nan)

    # Build cumulative waterfall
    cumulative = base
    running    = [base]
    for s in sv_shown:
        cumulative += s
        running.append(cumulative)

    final_log_odds = running[-1]
    final_prob     = 1 / (1 + np.exp(-final_log_odds))

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(11, 6))

    y_pos  = np.arange(len(feats_shown) + 2)  # base + features + final
    colors = []
    lefts  = []
    widths = []

    # Base bar
    colors.append("#4B5563")
    lefts.append(0)
    widths.append(running[0])

    # Feature bars
    for i, s in enumerate(sv_shown):
        colors.append("#EF4444" if s > 0 else "#22C55E")
        start = running[i]
        lefts.append(min(start, start + s))
        widths.append(abs(s))

    # Final bar
    colors.append("#3B82F6")
    lefts.append(0)
    widths.append(running[-1])

    labels_y = (["E[f(X)] = Base rate"]
                + [f"{FEATURE_LABELS.get(f, f)}"
                   + (f" = {xvals_shown[j]:.2f}" if not np.isnan(xvals_shown[j]) else "")
                   for j, f in enumerate(feats_shown)]
                + [f"f(x) = {final_prob:.4f}  (PD)"])

    ax.barh(y_pos, widths, left=lefts, color=colors,
            edgecolor="none", height=0.65, alpha=0.92)

    # Connector lines
    for i in range(1, len(running)):
        ax.plot([running[i], running[i]], [y_pos[i] - 0.35, y_pos[i] + 0.35],
                color="#4B5563", linewidth=0.8, linestyle=":")

    # Value annotations
    for i, (left, width, s) in enumerate(zip(lefts, widths, [running[0]] + list(sv_shown) + [running[-1]])):
        val = running[0] if i == 0 else (s if i <= len(sv_shown) else running[-1])
        xpos = left + width / 2 if i in (0, len(y_pos) - 1) else left + width + 0.01 * (1 if sv_shown[i-1] > 0 else -1)
        ax.text(lefts[i] + widths[i] + 0.015,
                y_pos[i], f"{sv_shown[i-1]:+.3f}" if 0 < i <= len(sv_shown) else "",
                va="center", fontsize=8.5,
                color="#EF4444" if (i > 0 and i <= len(sv_shown) and sv_shown[i-1] > 0)
                else "#22C55E" if i > 0 and i <= len(sv_shown) else "#E2E8F0")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_y, fontsize=9)
    ax.set_xlabel("Log-odds contribution", fontsize=10)
    ax.set_title(f"SHAP Waterfall — {title_suffix}\n"
                 f"Final PD = {final_prob:.4f}  |  Base E[f(X)] = {1/(1+np.exp(-base)):.4f}  "
                 f"|  Red = raises risk  ·  Green = lowers risk",
                 fontsize=10, fontweight="bold", color="white", pad=12)
    ax.axvline(base, color="#4B5563", linewidth=1, linestyle="--", alpha=0.6)
    ax.grid(True, axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s  (PD = %.4f)", path, final_prob)


def plot_segment_report(shap_values: np.ndarray,
                         y_scores:    np.ndarray,
                         y_true:      np.ndarray,
                         feature_names: list[str]) -> pd.DataFrame:
    """
    Segment-level SHAP report: portfolio deciles ranked by predicted PD.

    For each decile shows mean SHAP per feature — helps credit teams
    understand which risk drivers dominate each rating grade.
    """
    # Convert log-odds scores to probability
    probs   = 1 / (1 + np.exp(-y_scores))
    deciles = pd.qcut(probs, q=10, labels=False, duplicates="drop")

    rows = []
    for d in sorted(np.unique(deciles)):
        mask  = deciles == d
        n     = mask.sum()
        dr    = y_true[mask].mean() * 100
        p_avg = probs[mask].mean() * 100
        sv_mean = np.abs(shap_values[mask]).mean(axis=0)
        top3  = sorted(zip(feature_names, sv_mean), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{FEATURE_LABELS.get(f,f)}({v:.3f})" for f, v in top3)
        rows.append({
            "decile":          int(d) + 1,
            "n_loans":         int(n),
            "mean_pd_pct":     round(p_avg, 4),
            "observed_dr_pct": round(dr, 4),
            "top_3_drivers":   top3_str,
        })

    report_df = pd.DataFrame(rows)

    # Plot
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SHAP Segment Report — Portfolio Deciles by Predicted PD\n"
                 "BCBS 239: Risk Data Aggregation and Reporting",
                 fontsize=12, fontweight="bold", color="white")

    # Decile PD vs observed
    x = report_df["decile"].values
    axes[0].bar(x, report_df["mean_pd_pct"], color="#3B82F6", alpha=0.8,
                label="Mean predicted PD (%)", width=0.4, align="edge")
    axes[0].bar(x + 0.4, report_df["observed_dr_pct"], color="#EF4444",
                alpha=0.8, label="Observed default rate (%)", width=0.4, align="edge")
    axes[0].set_xlabel("Score Decile (1=lowest risk, 10=highest risk)", fontsize=10)
    axes[0].set_ylabel("Rate (%)", fontsize=10)
    axes[0].set_title("Predicted PD vs Observed Default Rate by Decile",
                      color="#CBD5E1", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].set_xticks(x + 0.4)
    axes[0].set_xticklabels(x.astype(str))

    # Stacked SHAP contributions per decile (top 5 features)
    mean_abs_global = np.abs(shap_values).mean(axis=0)
    top5_idx        = np.argsort(mean_abs_global)[::-1][:5]
    top5_names      = [FEATURE_LABELS.get(feature_names[i], feature_names[i])
                       for i in top5_idx]
    palette = ["#3B82F6","#F59E0B","#10B981","#EF4444","#8B5CF6"]

    bottoms = np.zeros(len(report_df))
    for feat_pos, (fi, fname, col) in enumerate(zip(top5_idx, top5_names, palette)):
        # Mean |SHAP| for this feature per decile
        vals = []
        for d in sorted(np.unique(deciles)):
            mask = deciles == d
            vals.append(np.abs(shap_values[mask, fi]).mean())
        vals = np.array(vals)
        axes[1].bar(x, vals, bottom=bottoms, color=col, alpha=0.85,
                    label=fname, width=0.7)
        bottoms += vals

    axes[1].set_xlabel("Score Decile (1=lowest risk, 10=highest risk)", fontsize=10)
    axes[1].set_ylabel("Mean |SHAP| (log-odds)", fontsize=10)
    axes[1].set_title("Mean |SHAP| per Feature by Risk Decile",
                      color="#CBD5E1", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x.astype(str))

    fig.tight_layout()
    path = FIG_DIR / "shap_segment_report.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    log.info("  → %s", path)

    return report_df


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Ch.4 — SHAP Explanations (BCBS 239)")
    log.info("SHAP version: %s", shap.__version__)
    log.info("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("[1/6] Loading processed data …")
    train = pd.read_parquet(PROC_DIR / "pd_train.parquet")
    oos   = pd.read_parquet(PROC_DIR / "pd_oos.parquet")
    log.info("  Train: %s  |  OOS: %s",
             f"{len(train):,}", f"{len(oos):,}")

    # ── Prepare ───────────────────────────────────────────────────────────
    log.info("")
    log.info("[2/6] Preparing feature matrices …")
    X_tr, X_oo, y_tr, y_oo, feats = prepare(train, oos)
    log.info("  Features: %s", feats)

    # ── Retrain XGBoost ───────────────────────────────────────────────────
    log.info("")
    log.info("[3/6] Retraining XGBoost PD model …")
    xgb = retrain_xgboost(X_tr, y_tr, X_oo, y_oo)

    # ── Sample for SHAP ───────────────────────────────────────────────────
    log.info("")
    log.info("[4/6] Computing SHAP values (n = %s) …",
             f"{min(SHAP_SAMPLE_N, len(X_oo)):,}")
    rng      = np.random.default_rng(SEED)
    idx_sample = rng.choice(len(X_oo), size=min(SHAP_SAMPLE_N, len(X_oo)),
                            replace=False)
    X_shap   = X_oo[idx_sample]
    y_shap   = y_oo[idx_sample]

    shap_values, expected_value = compute_shap(xgb, X_shap, feats)

    # Raw log-odds scores for sampled loans
    log_odds = xgb.get_booster().predict(
        __import__("xgboost").DMatrix(X_shap), output_margin=True
    )

    # ── Save SHAP matrix ──────────────────────────────────────────────────
    sv_df = pd.DataFrame(shap_values, columns=[f"shap_{f}" for f in feats])
    sv_df["log_odds"]    = log_odds
    sv_df["predicted_pd"] = 1 / (1 + np.exp(-log_odds))
    sv_df["actual_default"] = y_shap
    sv_df.to_parquet(PROC_DIR / "shap_values_oos.parquet", index=False)
    log.info("  SHAP matrix saved → data/processed/shap_values_oos.parquet")

    # ── Plots ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("[5/6] Generating SHAP visualisations …")

    plot_global_importance(shap_values, feats)
    plot_beeswarm(shap_values, X_shap, feats)

    # Dependence plots for the two most important features
    mean_abs   = np.abs(shap_values).mean(axis=0)
    top2_feats = [feats[i] for i in np.argsort(mean_abs)[::-1][:2]]
    inter_feat  = feats[np.argsort(mean_abs)[::-1][2]] if len(feats) > 2 else None

    for feat in top2_feats:
        plot_dependence(shap_values, X_shap, feats, feat, interaction=inter_feat)

    # Waterfall for highest-risk and lowest-risk loan in sample
    probs_all   = 1 / (1 + np.exp(-log_odds))
    high_idx    = int(np.argmax(probs_all))
    low_idx     = int(np.argmin(probs_all))

    plot_waterfall(shap_values, expected_value, X_shap, feats,
                   high_idx,
                   f"Highest-Risk Loan  (Predicted PD = {probs_all[high_idx]:.4f})",
                   "shap_waterfall_high_risk.png")

    plot_waterfall(shap_values, expected_value, X_shap, feats,
                   low_idx,
                   f"Lowest-Risk Loan  (Predicted PD = {probs_all[low_idx]:.4f})",
                   "shap_waterfall_low_risk.png")

    # ── Segment report ────────────────────────────────────────────────────
    log.info("")
    log.info("[6/6] Building SHAP segment report …")
    report_df = plot_segment_report(shap_values, log_odds, y_shap, feats)
    report_df.to_csv(PROC_DIR / "shap_segment_report.csv", index=False)
    log.info("  Segment report:\n%s", report_df.to_string(index=False))

    log.info("")
    log.info("=" * 65)
    log.info("Ch.4 complete — SHAP explanations generated.")
    log.info("  Next: python 06_survival_analysis.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
