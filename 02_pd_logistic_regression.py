"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.1 — Logistic Regression PD Model
=============================================================================
Script  : 02_pd_logistic_regression.py
Purpose : Implements the baseline Probability of Default (PD) model from
          Sexton (2022) Chapter 1 using Weight-of-Evidence encoding and
          L2-regularised logistic regression.

Methodology
-----------
  WoE encoding (§1.5.2)
    Each feature is binned and replaced by its Weight-of-Evidence score.
    WoE maps are fitted on the training set only and applied to OOS/OOT
    to prevent data leakage.

        WoE_j = ln(p_j / q_j)
        IV    = Σ_j (p_j - q_j) · WoE_j

  Logistic regression (§1.5.3)
    Binary logistic regression with L2 regularisation (C = 1.0) and
    class_weight='balanced' to handle the ~0.64% default rate.

  Validation metrics (§1.6)
    AUROC  : overall discrimination
    KS     : max separation between good and bad score distributions
    Gini   : 2 × AUROC − 1
    HL test: Hosmer–Lemeshow calibration test (p > 0.05 → acceptable)

  Score banding (§1.7)
    Log-odds scores are banded into deciles and reported with observed
    default rates and cumulative capture rates.

Inputs
------
  data/processed/pd_train.parquet
  data/processed/pd_oos.parquet
  data/processed/pd_oot.parquet

Outputs
-------
  data/processed/pd_lr_results.csv    — per-row predictions across all splits
  data/processed/pd_lr_metrics.csv    — AUROC, KS, Gini, HL per split
  data/processed/pd_lr_coefs.csv      — model coefficients
  data/processed/pd_lr_scorecard.csv  — score bands + default rates
  data/processed/pd_lr_iv.csv         — IV per feature
  data/figures/pd_lr_roc.png          — ROC curves (Train / OOS / OOT)

Next Step
---------
  python 03_pd_ensemble.py
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
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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
        logging.FileHandler("pd_logistic.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = Path("data/processed")
OUT_DIR  = Path("data/processed")
FIG_DIR  = Path("data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "default_12m"
SEED   = 42

# Feature set — thesis Table 3 (Chapter 1)
# hpi_change and ur_3m_lag will be NaN when macro files are absent;
# the imputer handles this gracefully using the training-set median.
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
NUM_FEATURES = [f for f in FEATURES if f not in CAT_FEATURES]


# =============================================================================
# WEIGHT-OF-EVIDENCE ENCODING
# =============================================================================

def _compute_woe_table(df: pd.DataFrame, feature: str,
                       target: str, n_bins: int = 15) -> pd.DataFrame:
    """
    Compute the WoE / IV table for a single feature.

    Returns an empty DataFrame if the feature has only one class in the target.
    """
    data = df[[feature, target]].dropna().copy()
    good_total = max((data[target] == 0).sum(), 1)
    bad_total  = max((data[target] == 1).sum(), 1)
    if good_total == 0 or bad_total == 0:
        return pd.DataFrame()

    is_cat = (data[feature].dtype == object or data[feature].nunique() <= 10)
    if is_cat:
        data["bin"] = data[feature].astype(str)
    else:
        try:
            data["bin"] = pd.qcut(
                data[feature], q=n_bins, duplicates="drop"
            ).astype(str)
        except Exception:
            data["bin"] = pd.cut(
                data[feature], bins=n_bins, duplicates="drop"
            ).astype(str)

    rows = []
    for bin_label, grp in data.groupby("bin", observed=True):
        n_good = (grp[target] == 0).sum()
        n_bad  = (grp[target] == 1).sum()
        p_ij   = max(n_good / good_total, 1e-9)
        q_ij   = max(n_bad  / bad_total,  1e-9)
        woe    = np.log(p_ij / q_ij)
        rows.append({
            "feature":      feature,
            "bin":          bin_label,
            "n_total":      len(grp),
            "n_good":       n_good,
            "n_bad":        n_bad,
            "default_rate": n_bad / len(grp),
            "woe":          round(woe, 6),
            "iv_component": round((p_ij - q_ij) * woe, 8),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["iv_total"] = round(result["iv_component"].sum(), 6)
    return result


def fit_woe_maps(train: pd.DataFrame, features: list[str],
                 n_bins: int = 15) -> tuple[dict, pd.DataFrame]:
    """
    Learn WoE bin maps from the training set only.

    Returns
    -------
    woe_maps : dict  — {feature: {"type": "cat"|"num", "edges"?, "map": {bin: woe}}}
    iv_df    : DataFrame — IV summary sorted descending
    """
    woe_maps: dict = {}
    iv_rows:  list = []

    for feat in features:
        if feat not in train.columns:
            continue
        tbl = _compute_woe_table(train, feat, TARGET, n_bins)
        if tbl.empty:
            continue

        is_cat = (train[feat].dtype == object or train[feat].nunique() <= 10)
        if is_cat:
            woe_maps[feat] = {
                "type": "cat",
                "map":  tbl.set_index("bin")["woe"].to_dict(),
            }
        else:
            try:
                _, edges = pd.qcut(
                    train[feat].dropna(), q=n_bins,
                    duplicates="drop", retbins=True
                )
                woe_maps[feat] = {
                    "type":  "num",
                    "edges": edges,
                    "map":   tbl.set_index("bin")["woe"].to_dict(),
                }
            except Exception:
                woe_maps[feat] = {
                    "type": "cat",
                    "map":  tbl.set_index("bin")["woe"].to_dict(),
                }

        iv_rows.append({
            "feature": feat,
            "iv":      round(tbl["iv_component"].sum(), 4),
        })

    iv_df = (
        pd.DataFrame(iv_rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    return woe_maps, iv_df


def apply_woe(df: pd.DataFrame, woe_maps: dict) -> pd.DataFrame:
    """Apply pre-fitted WoE maps to any data split (no fitting on OOS/OOT)."""
    out = df.copy()
    for feat, info in woe_maps.items():
        if feat not in out.columns:
            continue
        col_name = f"{feat}_woe"
        if info["type"] == "cat":
            out[col_name] = out[feat].astype(str).map(info["map"]).fillna(0.0)
        else:
            binned = pd.cut(
                out[feat], bins=info["edges"], include_lowest=True
            ).astype(str)
            out[col_name] = binned.map(info["map"]).fillna(0.0)
    return out


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS = max |CDF_bad(t) − CDF_good(t)| across all score thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def hosmer_lemeshow(y_true: np.ndarray, y_pred: np.ndarray, g: int = 10) -> dict:
    """
    Hosmer–Lemeshow calibration test (thesis §1.6).

    H₀ : model is well-calibrated.
    p > 0.05 → fail to reject H₀ (acceptable calibration).

    Parameters
    ----------
    y_true : observed binary outcomes
    y_pred : predicted default probabilities
    g      : number of score deciles (default 10)
    """
    from scipy.stats import chi2

    hl_df = pd.DataFrame({"y": y_true, "p": y_pred})
    hl_df["decile"] = pd.qcut(hl_df["p"], q=g, duplicates="drop", labels=False)

    grp = hl_df.groupby("decile").agg(
        obs_bad=  ("y", "sum"),
        obs_good= ("y", lambda x: (x == 0).sum()),
        exp_bad=  ("p", "sum"),
        n=        ("y", "count"),
    )
    grp["exp_good"] = grp["n"] - grp["exp_bad"]

    hl_stat = (
        ((grp["obs_bad"]  - grp["exp_bad"])**2  / grp["exp_bad"].clip(lower=1e-9)).sum()
      + ((grp["obs_good"] - grp["exp_good"])**2 / grp["exp_good"].clip(lower=1e-9)).sum()
    )
    p_val = 1 - chi2.cdf(hl_stat, df=g - 2)

    return {
        "hl_stat": round(float(hl_stat), 4),
        "hl_pval": round(float(p_val),   4),
        "hl_dof":  g - 2,
    }


def evaluate(name: str, y_true: np.ndarray,
             y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute and log AUROC, KS, Gini, and Hosmer–Lemeshow for one split."""
    n_pos = int(y_true.sum())
    if n_pos == 0:
        log.warning("  [%s] No positive cases — metrics not computed.", name)
        return {"split": name, "n": len(y_true), "n_defaults": 0,
                "auroc": np.nan, "ks": np.nan, "gini": np.nan}

    auroc = roc_auc_score(y_true, y_score)
    ks    = ks_statistic(y_true, y_score)
    gini  = 2 * auroc - 1
    hl    = hosmer_lemeshow(y_true, y_pred)

    log.info(
        "  [%-5s]  n=%s  defaults=%s  AUROC=%.4f  KS=%.4f  "
        "Gini=%.4f  HL-stat=%.2f  HL-p=%.4f",
        name, f"{len(y_true):,}", f"{n_pos:,}",
        auroc, ks, gini, hl["hl_stat"], hl["hl_pval"],
    )
    return {
        "split": name, "n": len(y_true), "n_defaults": n_pos,
        "auroc": round(auroc, 4), "ks": round(ks, 4), "gini": round(gini, 4),
        **hl,
    }


# =============================================================================
# SCORE BANDING  (thesis §1.7 scorecard output)
# =============================================================================

def build_scorecard(df: pd.DataFrame, score_col: str = "score",
                    n_bands: int = 10) -> pd.DataFrame:
    """
    Bin log-odds scores into deciles and compute:
        - observed default rate per band
        - cumulative capture rate (sorted high-risk first)
    """
    tmp = df[[score_col, TARGET]].copy()
    tmp["band"] = pd.qcut(tmp[score_col], q=n_bands, duplicates="drop", labels=False)

    agg = tmp.groupby("band").agg(
        score_min=(score_col, "min"),
        score_max=(score_col, "max"),
        n=(TARGET, "count"),
        n_default=(TARGET, "sum"),
    ).reset_index()

    agg["default_rate"] = agg["n_default"] / agg["n"]
    agg = agg.sort_values("score_min", ascending=False).reset_index(drop=True)

    total_defaults = agg["n_default"].sum()
    agg["cum_capture"] = agg["n_default"].cumsum() / max(total_defaults, 1)

    return agg


# =============================================================================
# ROC PLOT
# =============================================================================

def plot_roc(y_true_dict: dict, y_score_dict: dict, path: Path) -> None:
    """Plot ROC curves for Train, OOS, and OOT splits on a single axis."""
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {"Train": "#2563EB", "OOS": "#D97706", "OOT": "#059669"}

    for name, y_true in y_true_dict.items():
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score_dict[name])
        auroc = roc_auc_score(y_true, y_score_dict[name])
        ax.plot(fpr, tpr,
                label=f"{name}  (AUROC = {auroc:.3f})",
                color=palette.get(name, "gray"),
                linewidth=2.0)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Random classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — Ch.1 Logistic Regression PD Model", fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  ROC plot → %s", path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Ch.1 — Logistic Regression PD Model")
    log.info("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("[1/5] Loading processed data …")
    train = pd.read_parquet(PROC_DIR / "pd_train.parquet")
    oos   = pd.read_parquet(PROC_DIR / "pd_oos.parquet")
    oot   = pd.read_parquet(PROC_DIR / "pd_oot.parquet")

    log.info("  Train: %s  |  OOS: %s  |  OOT: %s",
             f"{len(train):,}", f"{len(oos):,}", f"{len(oot):,}")
    log.info("  Default rate — Train: %.4f%%  OOS: %.4f%%  OOT: %.4f%%",
             train[TARGET].mean() * 100,
             oos[TARGET].mean() * 100,
             oot[TARGET].mean() * 100)

    feats_present = [f for f in FEATURES if f in train.columns]
    log.info("  Features present: %s", feats_present)

    # ── WoE encoding  (fit on train, apply to OOS/OOT) ───────────────────
    log.info("")
    log.info("[2/5] Fitting WoE maps on training data …")
    woe_maps, iv_df = fit_woe_maps(train, feats_present)
    log.info("\n%s", iv_df.to_string(index=False))

    train_woe = apply_woe(train, woe_maps)
    oos_woe   = apply_woe(oos,   woe_maps)
    oot_woe   = apply_woe(oot,   woe_maps)

    woe_cols = [f"{f}_woe" for f in woe_maps if f"{f}_woe" in train_woe.columns]
    log.info("  WoE columns: %s", woe_cols)

    # ── Impute (median from training set — handles absent macro features) ─
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_woe[woe_cols])
    X_oos   = imputer.transform(oos_woe[woe_cols])
    X_oot   = imputer.transform(oot_woe[woe_cols])

    y_train = train[TARGET].values
    y_oos   = oos[TARGET].values
    y_oot   = oot[TARGET].values

    # ── Standardise ───────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_oos   = scaler.transform(X_oos)
    X_oot   = scaler.transform(X_oot)

    # ── Logistic Regression  (thesis §1.5.3) ─────────────────────────────
    log.info("")
    log.info("[3/5] Fitting logistic regression (L2, liblinear, balanced) …")
    lr = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=SEED,
        class_weight="balanced",   # upweights defaults ~155× for 0.64% rate
    )
    lr.fit(X_train, y_train)

    coef_df = pd.DataFrame({
        "feature":     woe_cols,
        "coefficient": lr.coef_[0],
        "abs_coef":    np.abs(lr.coef_[0]),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    log.info("\n  Coefficients (WoE scale, descending |coef|):")
    log.info("\n%s", coef_df[["feature", "coefficient"]].to_string(index=False))

    # ── Evaluate ──────────────────────────────────────────────────────────
    log.info("")
    log.info("[4/5] Evaluating model across Train / OOS / OOT …")

    splits = {
        "Train": (y_train, X_train),
        "OOS":   (y_oos,   X_oos),
        "OOT":   (y_oot,   X_oot),
    }
    metrics:    list[dict]  = []
    score_data: dict[str, np.ndarray] = {}

    for name, (y, X) in splits.items():
        y_score = lr.predict_proba(X)[:, 1]
        metrics.append(evaluate(name, y, y_score, y_score))
        score_data[name] = y_score

    metrics_df = pd.DataFrame(metrics)

    # ── Scorecard + outputs ───────────────────────────────────────────────
    log.info("")
    log.info("[5/5] Building scorecard and saving outputs …")

    def _score_df(df: pd.DataFrame, X: np.ndarray, split_label: str) -> pd.DataFrame:
        out = df[["loan_seq_num", "report_date", TARGET]].copy()
        out["score"] = lr.predict_proba(X)[:, 1]
        out["split"] = split_label
        return out

    all_scores = pd.concat([
        _score_df(train, X_train, "train"),
        _score_df(oos,   X_oos,   "oos"),
        _score_df(oot,   X_oot,   "oot"),
    ], ignore_index=True)

    train_scores_df = all_scores[all_scores["split"] == "train"]
    scorecard = build_scorecard(train_scores_df, "score")

    log.info("\n  Scorecard (training set — high-risk to low-risk):")
    log.info("\n%s", scorecard.to_string(index=False))

    # ROC plot
    plot_roc(
        {"Train": y_train, "OOS": y_oos, "OOT": y_oot},
        score_data,
        FIG_DIR / "pd_lr_roc.png",
    )

    # Persist outputs
    all_scores.to_csv(  OUT_DIR / "pd_lr_results.csv",   index=False)
    metrics_df.to_csv(  OUT_DIR / "pd_lr_metrics.csv",   index=False)
    coef_df.to_csv(     OUT_DIR / "pd_lr_coefs.csv",     index=False)
    scorecard.to_csv(   OUT_DIR / "pd_lr_scorecard.csv", index=False)
    iv_df.to_csv(       OUT_DIR / "pd_lr_iv.csv",        index=False)

    log.info("")
    log.info("  Files saved to %s", OUT_DIR.resolve())
    log.info("")
    log.info("  Final Metrics:")
    log.info(
        "\n%s",
        metrics_df[["split", "n", "n_defaults", "auroc", "ks", "gini"]].to_string(index=False)
    )

    log.info("")
    log.info("=" * 65)
    log.info("Ch.1 complete.  Next: python 03_pd_ensemble.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
