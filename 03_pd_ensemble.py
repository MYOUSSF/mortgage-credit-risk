"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.2 — XGBoost PD Model
=============================================================================
Script  : 03_pd_ensemble.py
Purpose : Gradient-boosted tree ensemble for Probability of Default, extending
          the Ch.1 logistic regression baseline per Sexton (2022) Chapter 2.

GPU Strategy
------------
  XGBoost is preferred over Random Forest for large-scale PD modelling because:

    • tree_method="hist" + device="cuda" uses a fraction of RF's RAM via
      histogram approximation of split search.
    • Native row/column subsampling means the full feature matrix never
      needs to be materialised in a single dense array.
    • GPU acceleration delivers ~10–20× speedup on T4/A100 hardware.

  Auto-detection at startup:
    GPU present → device="cuda",  tree_method="hist"
    GPU absent  → device="cpu",   tree_method="hist"  (still memory-efficient)

Class Imbalance
---------------
  With ~0.64% default rate, scale_pos_weight = neg/pos ≈ 155 is computed
  dynamically from the training data and passed to XGBoost, equivalent to
  oversampling the minority class in the gradient computation.

Early Stopping
--------------
  Training halts when OOS AUC does not improve for 20 consecutive rounds,
  preventing overfitting without manual n_estimators tuning.  Typical
  convergence is at 200–350 rounds on the Freddie Mac sample dataset.

Inputs
------
  data/processed/pd_train.parquet
  data/processed/pd_oos.parquet
  data/processed/pd_oot.parquet
  data/processed/pd_lr_metrics.csv   (Ch.1 baseline — optional comparison)

Outputs
-------
  data/processed/pd_ensemble_metrics.csv
  data/processed/pd_xgb_importance.csv
  data/processed/pd_xgb_results.csv
  data/figures/pd_xgb_importance.png
  data/figures/pd_ensemble_roc.png

Next Step
---------
  python 04_lgd_models.py
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
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pd_ensemble.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# GPU DETECTION
# =============================================================================

def _detect_gpu() -> tuple[str, int]:
    """
    Probe for NVIDIA GPUs via nvidia-smi.

    Returns (device_str, n_gpus).
    XGBoost ≥ 2.0 automatically uses all visible GPUs when device="cuda".
    """
    try:
        import subprocess
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
                    log.info("[GPU] XGBoost %s ≥ 2.0 — all %d GPUs active via NCCL.",
                             xgb_ver.__version__, len(gpus))
                else:
                    log.warning(
                        "[GPU] XGBoost %s < 2.0 — only 1 GPU will be used.  "
                        "Upgrade: pip install -U xgboost",
                        xgb_ver.__version__,
                    )
            return "cuda", len(gpus)
    except Exception:
        pass

    log.info("[CPU] No GPU detected — using device='cpu' (hist, memory-efficient).")
    return "cpu", 0


DEVICE, N_GPUS = _detect_gpu()


# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = Path("data/processed")
OUT_DIR  = Path("data/processed")
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

# Hyperparameters — informed by thesis defaults (§2.4)
# scale_pos_weight is computed dynamically from training class ratio
XGB_PARAMS: dict = dict(
    n_estimators          = 500,
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.5,      # row subsampling per tree
    colsample_bytree      = 0.8,      # feature subsampling per tree
    min_child_weight      = 50,       # minimum sum of instance weights in a leaf
    gamma                 = 1.0,      # minimum loss reduction for a split
    reg_alpha             = 0.1,      # L1 regularisation
    reg_lambda            = 1.0,      # L2 regularisation
    eval_metric           = "auc",
    tree_method           = "hist",   # histogram approximation — GPU + CPU
    device                = DEVICE,   # auto-detected above
    random_state          = SEED,
    n_jobs                = -1,
    early_stopping_rounds = 20,       # halt when OOS AUC plateaus
)


# =============================================================================
# PREPROCESSING
# =============================================================================

def _encode_categoricals(train: pd.DataFrame, oos: pd.DataFrame,
                          oot: pd.DataFrame, cat_cols: list[str]
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Label-encode categorical features.

    The encoder is fitted on the union of all splits so that labels present
    only in OOS/OOT do not raise unseen-label errors at inference time.
    """
    for col in cat_cols:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        all_vals = (
            pd.concat([train[col], oos[col], oot[col]])
            .fillna("missing")
            .astype(str)
        )
        le.fit(all_vals)
        known = set(le.classes_)
        fallback = le.classes_[0]

        for df in [train, oos, oot]:
            df[col] = le.transform(
                df[col]
                .fillna("missing")
                .astype(str)
                .map(lambda x, k=known, fb=fallback: x if x in k else fb)
            )
    return train, oos, oot


def prepare(train: pd.DataFrame, oos: pd.DataFrame,
            oot: pd.DataFrame) -> tuple:
    """Return feature matrices, label vectors, and the active feature list."""
    feats = [f for f in FEATURES if f in train.columns]
    train, oos, oot = _encode_categoricals(
        train.copy(), oos.copy(), oot.copy(), CAT_FEATURES
    )
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(train[feats])
    X_oo = imputer.transform(oos[feats])
    X_ot = imputer.transform(oot[feats])
    return (
        X_tr, X_oo, X_ot,
        train[TARGET].values, oos[TARGET].values, oot[TARGET].values,
        feats,
    )


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def _ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def evaluate(name: str, y_true: np.ndarray, y_score: np.ndarray) -> dict:
    n_pos = int(y_true.sum())
    if n_pos == 0:
        log.warning("  [%s] No defaults — metrics not computed.", name)
        return {"split": name, "n": len(y_true), "n_defaults": 0,
                "auroc": np.nan, "ks": np.nan, "gini": np.nan}

    auroc = roc_auc_score(y_true, y_score)
    ks    = _ks(y_true, y_score)
    gini  = 2 * auroc - 1

    log.info(
        "  [%-5s]  n=%s  defaults=%s  AUROC=%.4f  KS=%.4f  Gini=%.4f",
        name, f"{len(y_true):,}", f"{n_pos:,}", auroc, ks, gini,
    )
    return {
        "split": name, "n": len(y_true), "n_defaults": n_pos,
        "auroc": round(auroc, 4), "ks": round(ks, 4), "gini": round(gini, 4),
    }


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_importance(imp: pd.Series, title: str, path: Path,
                    top_n: int = 12) -> None:
    """Horizontal bar chart of XGBoost gain-based feature importances."""
    fig, ax = plt.subplots(figsize=(9, 5))
    imp.head(top_n).sort_values().plot(
        kind="barh", ax=ax, color="#2563EB", edgecolor="white"
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Mean Gain", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Feature importance plot → %s", path)


def plot_roc(results: dict, path: Path) -> None:
    """
    Three-panel ROC plot (Train | OOS | OOT) comparing all models.

    Parameters
    ----------
    results : {model_name: {split_name: {"y_true": ..., "y_score": ...}}}
    """
    palette = ["#2563EB", "#D97706", "#059669", "#DC2626"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, split in zip(axes, ["Train", "OOS", "OOT"]):
        for (model_name, data), color in zip(results.items(), palette):
            if split not in data:
                continue
            y_true  = data[split]["y_true"]
            y_score = data[split]["y_score"]
            if y_true.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auroc = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, label=f"{model_name}  ({auroc:.3f})",
                    color=color, linewidth=2.0)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0)
        ax.set_title(split, fontsize=11)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate",  fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("ROC Curves — Ch.2 Model Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  ROC comparison plot → %s", path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    gpu_str = f"{N_GPUS}× GPU (CUDA)" if N_GPUS > 0 else "CPU"
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Ch.2 — XGBoost PD Model")
    log.info("Device: %s", gpu_str)
    log.info("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("[1/5] Loading data …")
    train = pd.read_parquet(PROC_DIR / "pd_train.parquet")
    oos   = pd.read_parquet(PROC_DIR / "pd_oos.parquet")
    oot   = pd.read_parquet(PROC_DIR / "pd_oot.parquet")

    log.info("  Train: %s  |  OOS: %s  |  OOT: %s",
             f"{len(train):,}", f"{len(oos):,}", f"{len(oot):,}")
    log.info("  Default rate — Train: %.4f%%  OOS: %.4f%%  OOT: %.4f%%",
             train[TARGET].mean() * 100,
             oos[TARGET].mean() * 100,
             oot[TARGET].mean() * 100)

    # ── Prepare ───────────────────────────────────────────────────────────
    log.info("")
    log.info("[2/5] Preparing feature matrices …")
    X_tr, X_oo, X_ot, y_tr, y_oo, y_ot, feats = prepare(train, oos, oot)
    log.info("  Feature matrix: %s  Active features: %s", X_tr.shape, feats)

    del train, oos, oot
    gc.collect()

    # Class imbalance weight (computed from training data)
    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = neg / max(pos, 1)
    log.info("  scale_pos_weight = %.1f  (%s neg / %s pos)", spw,
             f"{neg:,}", f"{pos:,}")

    # ── Train XGBoost ──────────────────────────────────────────────────────
    log.info("")
    log.info("[3/5] Training XGBoost (early stopping on OOS AUC) …")
    params = {**XGB_PARAMS, "scale_pos_weight": spw}
    xgb = XGBClassifier(**params)
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_oo, y_oo)],
        verbose=50,
    )
    log.info("  Best iteration: %d", xgb.best_iteration)
    gc.collect()

    # ── Evaluate ───────────────────────────────────────────────────────────
    log.info("")
    log.info("[4/5] Evaluating XGBoost …")
    all_preds: dict = {"XGBoost": {}}
    metrics:   list = []

    for split_name, (X, y) in [("Train", (X_tr, y_tr)),
                                ("OOS",   (X_oo, y_oo)),
                                ("OOT",   (X_ot, y_ot))]:
        score = xgb.predict_proba(X)[:, 1]
        m = evaluate(split_name, y, score)
        m["model"] = "XGBoost"
        metrics.append(m)
        all_preds["XGBoost"][split_name] = {"y_true": y, "y_score": score}

    # Load Ch.1 LR results for side-by-side comparison
    lr_results_path = PROC_DIR / "pd_lr_results.csv"
    lr_metrics_path = PROC_DIR / "pd_lr_metrics.csv"

    if lr_results_path.exists():
        log.info("")
        log.info("  Loading Ch.1 LR results for comparison …")
        lr_res   = pd.read_csv(lr_results_path)
        lr_preds = {}
        for raw_split, canonical in [("train", "Train"), ("oos", "OOS"), ("oot", "OOT")]:
            sub = lr_res[lr_res["split"] == raw_split]
            if not sub.empty:
                lr_preds[canonical] = {
                    "y_true":  sub[TARGET].values,
                    "y_score": sub["score"].values,
                }
        if lr_preds:
            all_preds["Logistic Regression"] = lr_preds

    if lr_metrics_path.exists():
        lr_m = pd.read_csv(lr_metrics_path)
        lr_m["model"] = "Logistic Regression"
        metrics.extend(lr_m.to_dict("records"))

    metrics_df = pd.DataFrame(metrics)

    log.info("")
    log.info("  Model comparison (OOS + OOT):")
    summary = (
        metrics_df[metrics_df["split"].isin(["OOS", "OOT"])]
        [["model", "split", "auroc", "ks", "gini"]]
        .sort_values(["split", "auroc"], ascending=[True, False])
    )
    log.info("\n%s", summary.to_string(index=False))

    # ── Feature importance ────────────────────────────────────────────────
    log.info("")
    log.info("[5/5] Feature importance and saving outputs …")
    imp_raw = pd.Series(xgb.get_booster().get_score(importance_type="gain"))

    # XGBoost may return f0, f1, … names — map back to feature names
    if imp_raw.index.str.startswith("f").all():
        idx_map = {f"f{i}": name for i, name in enumerate(feats)}
        imp_raw.index = imp_raw.index.map(lambda x: idx_map.get(x, x))

    imp_gain = imp_raw.sort_values(ascending=False)
    log.info("\n  Feature Importance (gain):\n%s", imp_gain.to_string())

    plot_importance(
        imp_gain,
        "XGBoost Feature Importance — Gain (Ch.2)",
        FIG_DIR / "pd_xgb_importance.png",
    )

    if all_preds:
        plot_roc(all_preds, FIG_DIR / "pd_ensemble_roc.png")

    # ── Save ──────────────────────────────────────────────────────────────
    metrics_df.to_csv(OUT_DIR / "pd_ensemble_metrics.csv", index=False)
    imp_gain.reset_index().rename(
        columns={"index": "feature", 0: "gain"}
    ).to_csv(OUT_DIR / "pd_xgb_importance.csv", index=False)

    # Persist per-row XGBoost predictions for downstream LGD or portfolio use
    pred_rows = (
        list(zip(["train"] * len(y_tr), y_tr, xgb.predict_proba(X_tr)[:, 1]))
      + list(zip(["oos"]   * len(y_oo), y_oo, xgb.predict_proba(X_oo)[:, 1]))
      + list(zip(["oot"]   * len(y_ot), y_ot, xgb.predict_proba(X_ot)[:, 1]))
    )
    pd.DataFrame(pred_rows, columns=["split", TARGET, "xgb_score"]).to_csv(
        OUT_DIR / "pd_xgb_results.csv", index=False
    )

    log.info("")
    log.info("  Files saved to %s", OUT_DIR.resolve())
    log.info("")
    log.info("=" * 65)
    log.info("Ch.2 complete.  Next: python 04_lgd_models.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
