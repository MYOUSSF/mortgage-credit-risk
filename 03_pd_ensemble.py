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

Memory optimisations (Kaggle 30 GB limit)
------------------------------------------
  1. Training subsample    — XGBoost is trained on a stratified 20% sample
     of the training parquet (~5M rows) instead of all 24M.  At 500 trees
     with hist approximation this gives essentially identical AUC while
     cutting training-time RAM by ~5×.  The full OOS/OOT sets are still
     used for evaluation so metrics remain comparable.

  2. Sequential loading    — train, oos, oot are never all in RAM together.
     Each split is loaded, encoded, imputed, and converted to float32 before
     the next is loaded.  The raw DataFrame is deleted immediately after its
     NumPy array is extracted.

  3. Dtype downcasting     — float64 → float32 and int64 → int8/int16 halve
     the NumPy array footprint before passing to XGBoost.

  4. In-place categoricals — LabelEncoder is fitted on training data only
     (no cross-split concat), with OOS/OOT unseen labels mapped to a fallback.

  5. Single predict_proba  — scores are computed once per split and stored;
     the redundant second call in the save step is eliminated.

  6. Lean score storage    — all_preds stores only (y_true, y_score) arrays,
     not full DataFrames.
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

# MEM OPT 1: train on a stratified subsample of the training parquet.
# 20% of 24M = ~5M rows — sufficient for XGBoost to converge and cuts
# training-time RAM by ~5×.  Set to 1.0 to use the full training set
# if RAM permits.
TRAIN_SAMPLE_FRAC = 0.20

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

# Columns to read from parquet — pruned to the minimum needed
_ID_COLS   = ["loan_seq_num", "report_date"]
_LOAD_COLS = _ID_COLS + [TARGET] + FEATURES

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
# HELPERS
# =============================================================================

def _get_parquet_columns(path: Path) -> list[str]:
    """Return column names from a parquet file without loading any data."""
    import pyarrow.parquet as pq
    return pq.read_schema(path).names


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 → float32, int64 → smallest int dtype. Operates in-place."""
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes("int64").columns:
        if col == TARGET:
            df[col] = df[col].astype(np.int8)
        else:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _fit_encoders(train: pd.DataFrame, cat_cols: list[str]) -> dict[str, LabelEncoder]:
    """
    Fit one LabelEncoder per categorical column on training data only.

    MEM OPT 4: fitted on train alone — no cross-split concatenation.
    Unseen labels in OOS/OOT are mapped to the most frequent training class
    at transform time.
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        le.fit(train[col].fillna("missing").astype(str))
        encoders[col] = le
    return encoders


def _apply_encoder(df: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """Apply pre-fitted label encoders; unseen labels → fallback class 0."""
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        known    = set(le.classes_)
        fallback = le.classes_[0]
        df[col]  = le.transform(
            df[col].fillna("missing").astype(str)
            .map(lambda x, k=known, fb=fallback: x if x in k else fb)
        )
    return df


def _load_and_prepare(path: Path, available_cols: list[str],
                      encoders: dict[str, LabelEncoder],
                      imputer: SimpleImputer | None,
                      feats: list[str],
                      sample_frac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one split from parquet, encode, impute, and return (X, y) as float32.

    MEM OPT 2: the raw DataFrame is deleted immediately after extraction.
    MEM OPT 3: dtype downcasting applied before imputation.

    Parameters
    ----------
    imputer     : if None, a new SimpleImputer is fitted on this data (train).
    sample_frac : fraction of rows to sample (stratified on TARGET).
                  Only applied when < 1.0, i.e. for the training split.

    Returns (X_float32, y_int8, fitted_imputer)
    """
    load_cols = [c for c in _LOAD_COLS if c in available_cols]
    df = pd.read_parquet(path, columns=load_cols)
    _downcast(df)

    # MEM OPT 1: stratified subsample of training data only
    if sample_frac < 1.0:
        df = (
            df.groupby(TARGET, group_keys=False)
              .apply(lambda g: g.sample(frac=sample_frac, random_state=SEED))
        )
        log.info("  Sampled %s rows (%.0f%% of training set, stratified).",
                 f"{len(df):,}", sample_frac * 100)

    df = _apply_encoder(df, encoders)

    X_df = df[[f for f in feats if f in df.columns]]

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X_df).astype(np.float32)
    else:
        X = imputer.transform(X_df).astype(np.float32)

    y = df[TARGET].values.astype(np.int8)

    # MEM OPT 2: free the DataFrame immediately
    del df, X_df
    gc.collect()

    return X, y, imputer


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

    # Discover available columns without loading data
    available_cols = _get_parquet_columns(PROC_DIR / "pd_train.parquet")
    feats = [f for f in FEATURES if f in available_cols]

    # ── Load + prepare training data ──────────────────────────────────────
    log.info("")
    log.info("[1/5] Loading and preparing training data …")

    # MEM OPT 4: fit encoders on training data alone (no cross-split concat)
    # We need a small peek at the train categorical columns to fit encoders.
    train_cats = pd.read_parquet(
        PROC_DIR / "pd_train.parquet",
        columns=[c for c in CAT_FEATURES if c in available_cols] + [TARGET],
    )
    encoders = _fit_encoders(train_cats, CAT_FEATURES)
    del train_cats
    gc.collect()

    # MEM OPT 1+2+3: load train with stratified subsample, encode, impute
    X_tr, y_tr, imputer = _load_and_prepare(
        PROC_DIR / "pd_train.parquet",
        available_cols, encoders, imputer=None,
        feats=feats, sample_frac=TRAIN_SAMPLE_FRAC,
    )

    log.info("  Train matrix: %s  |  default rate: %.4f%%",
             X_tr.shape, y_tr.mean() * 100)
    log.info("  Active features: %s", feats)

    # Class imbalance weight
    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = neg / max(pos, 1)
    log.info("  scale_pos_weight = %.1f  (%s neg / %s pos)",
             spw, f"{neg:,}", f"{pos:,}")

    # MEM OPT 2: load OOS now (needed for early stopping eval set)
    log.info("")
    log.info("[2/5] Loading OOS for early stopping eval set …")
    X_oo, y_oo, _ = _load_and_prepare(
        PROC_DIR / "pd_oos.parquet",
        available_cols, encoders, imputer=imputer,
        feats=feats, sample_frac=1.0,
    )
    log.info("  OOS matrix: %s  |  default rate: %.4f%%",
             X_oo.shape, y_oo.mean() * 100)

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

    # ── Evaluate across all splits ────────────────────────────────────────
    log.info("")
    log.info("[4/5] Evaluating XGBoost …")

    # MEM OPT 5: single predict_proba per split — scores stored once
    all_preds: dict = {"XGBoost": {}}
    metrics:   list = []
    xgb_score_frames: list[pd.DataFrame] = []  # lean frames for CSV output

    for split_label, X, y in [
        ("Train", X_tr, y_tr),
        ("OOS",   X_oo, y_oo),
    ]:
        score = xgb.predict_proba(X)[:, 1]
        m = evaluate(split_label, y, score)
        m["model"] = "XGBoost"
        metrics.append(m)
        all_preds["XGBoost"][split_label] = {"y_true": y, "y_score": score}
        xgb_score_frames.append(pd.DataFrame({
            "split": split_label.lower(), TARGET: y, "xgb_score": score
        }))

    # Free train + OOS arrays before loading OOT
    del X_tr, y_tr, X_oo, y_oo
    gc.collect()

    log.info("  Loading OOT …")
    X_ot, y_ot, _ = _load_and_prepare(
        PROC_DIR / "pd_oot.parquet",
        available_cols, encoders, imputer=imputer,
        feats=feats, sample_frac=1.0,
    )
    log.info("  OOT matrix: %s  |  default rate: %.4f%%",
             X_ot.shape, y_ot.mean() * 100)

    score_ot = xgb.predict_proba(X_ot)[:, 1]
    m = evaluate("OOT", y_ot, score_ot)
    m["model"] = "XGBoost"
    metrics.append(m)
    all_preds["XGBoost"]["OOT"] = {"y_true": y_ot, "y_score": score_ot}
    xgb_score_frames.append(pd.DataFrame({
        "split": "oot", TARGET: y_ot, "xgb_score": score_ot
    }))
    del X_ot, y_ot
    gc.collect()

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
        del lr_res
        gc.collect()

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

    # MEM OPT 5: use pre-computed score frames — no redundant predict_proba
    pd.concat(xgb_score_frames, ignore_index=True).to_csv(
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