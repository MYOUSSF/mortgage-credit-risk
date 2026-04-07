"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.3 — Loss Given Default Models
=============================================================================
Script  : 04_lgd_models.py
Purpose : Fit and evaluate four LGD models from Sexton (2022) Chapter 3.

Models
------
  1. Fractional Response Model (FRM)   — OLS on logit(LGD), predictions via
                                         inverse-logit → [0, 1] guaranteed
  2. Natural Spline Regression         — cubic splines (5 knots, degree 3)
                                         via sklearn SplineTransformer + OLS
  3. Random Forest Regressor           — 200 trees, max depth 6
  4. XGBoost Regressor                 — gradient-boosted trees (GPU if available)

LGD Target
----------
    LGD = actual_loss / zero_balance_removal_upb,  clipped to [0, 1]

    One row per defaulted loan (the final servicer observation at resolution).

Validation Metrics (§3.7)
--------------------------
    RMSE : root mean squared error
    MAE  : mean absolute error
    R²   : coefficient of determination
    Bias : mean(predicted − actual)

Note on Sample Size
-------------------
  The sample dataset (50,000 loans / year) yields only ~150 LGD observations
  from post-2010 vintages where crisis-era defaults are absent.  Metric
  variance is therefore high.  For publication-quality results, download the
  full (non-sample) Freddie Mac dataset or focus on 2004–2009 vintages where
  default rates reached 3–15%.

Inputs
------
  data/processed/lgd_train.parquet
  data/processed/lgd_oos.parquet
  data/processed/lgd_oot.parquet

Outputs
-------
  data/processed/lgd_metrics.csv
  data/processed/lgd_predictions.csv
  data/figures/lgd_frm_actual_vs_pred.png
  data/figures/lgd_rf_actual_vs_pred.png
  data/figures/lgd_xgb_actual_vs_pred.png
  data/figures/lgd_distributions.png
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("lgd_models.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

if not XGB_AVAILABLE:
    log.warning("xgboost not installed — XGBoost LGD model will be skipped.")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = Path("data/processed")
OUT_DIR  = Path("data/processed")
FIG_DIR  = Path("data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "lgd"
SEED   = 42

# LGD feature set — thesis §3.4
FEATURES: list[str] = [
    "hpi_change_since_orig",
    "mi_pct",
    "orig_cltv",
    "orig_dti",
    "orig_upb",
    "orig_interest_rate",
    "loan_age",
    "current_interest_rate",
    "ur_3m_lag",
    "occupancy_status",
    "first_time_homebuyer",
    "num_units",
    "property_type",
    "channel",
    "loan_purpose",
    "num_borrowers",
    "property_state",
]

CAT_FEATURES: list[str] = [
    "occupancy_status", "first_time_homebuyer", "num_units",
    "property_type", "channel", "loan_purpose",
    "num_borrowers", "property_state",
]

RF_PARAMS: dict = dict(
    n_estimators   = 200,
    max_depth      = 6,
    min_samples_leaf = 5,
    n_jobs         = -1,
    random_state   = SEED,
)

XGB_PARAMS: dict = dict(
    n_estimators     = 200,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = SEED,
    n_jobs           = -1,
)


# =============================================================================
# PREPROCESSING
# =============================================================================

def _encode_categoricals(train: pd.DataFrame, oos: pd.DataFrame,
                          oot: pd.DataFrame, cat_cols: list[str]
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Label-encode categorical features.

    Encoder fitted on union of all splits to avoid unseen-label errors.
    """
    for col in cat_cols:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        all_vals = (
            pd.concat([train[col], oos[col], oot[col]])
            .dropna()
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
    """Encode, impute, and return arrays for training and evaluation."""
    feats = [f for f in FEATURES if f in train.columns]
    train, oos, oot = _encode_categoricals(
        train.copy(), oos.copy(), oot.copy(), CAT_FEATURES
    )
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(train[feats])
    X_oo = imputer.transform(oos[feats])  if len(oos) else np.empty((0, len(feats)))
    X_ot = imputer.transform(oot[feats])  if len(oot) else np.empty((0, len(feats)))

    y_tr = train[TARGET].values
    y_oo = oos[TARGET].values   if len(oos) else np.array([])
    y_ot = oot[TARGET].values   if len(oot) else np.array([])

    return X_tr, X_oo, X_ot, y_tr, y_oo, y_ot, feats


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def evaluate_lgd(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                 model: str) -> dict:
    """Compute and log RMSE, MAE, R², and mean bias for one split."""
    if len(y_true) == 0:
        log.info("  [%s] %s — no rows, skipping.", model, name)
        return {"model": model, "split": name, "n": 0}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    bias = float(np.mean(y_pred - y_true))

    log.info(
        "  [%-5s] %s  n=%s  RMSE=%.4f  MAE=%.4f  R²=%.4f  Bias=%+.4f",
        model, name, f"{len(y_true):,}", rmse, mae, r2, bias,
    )
    return {
        "model": model, "split": name, "n": len(y_true),
        "rmse": round(rmse, 4), "mae": round(mae, 4),
        "r2":   round(r2, 4),  "bias": round(bias, 4),
    }


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class FractionalResponseModel:
    """
    Thesis §3.3 — OLS regression on logit-transformed LGD.

    Predictions are mapped back via the sigmoid (inverse logit) function,
    guaranteeing LGD_hat ∈ (0, 1).

        logit(y) = log(y / (1 - y))
        LGD_hat  = σ(Xβ) = 1 / (1 + exp(-Xβ))
    """

    def __init__(self) -> None:
        self._model = LinearRegression()

    @staticmethod
    def _logit(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        y = np.clip(y, eps, 1 - eps)
        return np.log(y / (1 - y))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FractionalResponseModel":
        self._model.fit(X, self._logit(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(self._model.predict(X))

    @property
    def coef_(self) -> np.ndarray:
        return self._model.coef_

    @property
    def intercept_(self) -> float:
        return float(self._model.intercept_)


def build_spline_model(degree: int = 3, n_knots: int = 5) -> Pipeline:
    """
    Thesis §3.6 — Natural cubic spline regression.

    SplineTransformer (sklearn) generates the basis functions; OLS
    estimates the coefficients.  Predictions are clipped to [0, 1].
    """
    return Pipeline([
        ("spline", SplineTransformer(n_knots=n_knots, degree=degree,
                                     include_bias=False)),
        ("lr",     LinearRegression()),
    ])


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str, path: Path) -> None:
    """Scatter plot of actual vs predicted LGD with a 45° calibration line."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.45, s=18, color="#2563EB", edgecolors="white",
               linewidths=0.3)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Actual LGD", fontsize=11)
    ax.set_ylabel("Predicted LGD", fontsize=11)
    ax.set_title(f"{model_name} — Actual vs Predicted LGD", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Plot → %s", path)


def plot_lgd_distributions(y_true: np.ndarray, pred_dict: dict,
                            path: Path) -> None:
    """
    Overlay histogram of actual LGD and each model's predicted distribution.

    Thesis §3.7 — visual calibration check.
    """
    palette = ["#2563EB", "#D97706", "#059669", "#DC2626"]
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(y_true, bins=30, alpha=0.4, label="Actual (train)",
            color="gray", density=True)

    for (model_name, preds), color in zip(pred_dict.items(), palette):
        ax.hist(preds, bins=30, alpha=0.6, label=model_name,
                color=color, density=True, histtype="step", linewidth=2.0)

    ax.set_xlabel("LGD", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("LGD Distribution — Actual vs Model Predictions (Training Set)",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  LGD distribution plot → %s", path)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("Mortgage Credit Risk  |  Ch.3 — Loss Given Default Models")
    log.info("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("[1/6] Loading LGD data …")
    train = pd.read_parquet(PROC_DIR / "lgd_train.parquet")
    oos   = pd.read_parquet(PROC_DIR / "lgd_oos.parquet")
    oot   = pd.read_parquet(PROC_DIR / "lgd_oot.parquet")

    log.info("  Train: %s  |  OOS: %s  |  OOT: %s",
             f"{len(train):,}", f"{len(oos):,}", f"{len(oot):,}")

    if len(train) < 10:
        log.warning("Very few LGD training rows (< 10).")
        log.warning(
            "LGD modelling requires defaults.  Download pre-2010 Freddie Mac "
            "vintages (2004–2009, default rates 3–15%%) for robust results."
        )
        if len(train) == 0:
            log.error("No training data — exiting.")
            return

    lgd_desc = train[TARGET].describe()
    log.info(
        "  LGD (train): mean=%.4f  median=%.4f  std=%.4f  min=%.4f  max=%.4f",
        lgd_desc["mean"], lgd_desc["50%"], lgd_desc["std"],
        lgd_desc["min"],  lgd_desc["max"],
    )

    # ── Prepare ───────────────────────────────────────────────────────────
    log.info("")
    log.info("[2/6] Preparing features …")
    X_tr, X_oo, X_ot, y_tr, y_oo, y_ot, feats = prepare(train, oos, oot)
    log.info("  Feature matrix: %s  Active features: %s", X_tr.shape, feats)

    all_metrics: list[dict] = []
    train_preds: dict       = {}

    def _eval_model(model_obj, model_name: str) -> None:
        """Evaluate a fitted model on all splits and collect metrics."""
        for split_name, (X, y) in [("Train", (X_tr, y_tr)),
                                    ("OOS",   (X_oo, y_oo)),
                                    ("OOT",   (X_ot, y_ot))]:
            if len(y) == 0:
                continue
            pred = model_obj.predict(X)
            # Clip to [0, 1] for models that do not enforce bounds natively
            pred = np.clip(pred, 0, 1)
            m = evaluate_lgd(split_name, y, pred, model_name)
            all_metrics.append(m)
            if split_name == "Train":
                train_preds[model_name] = pred

    # ── Model 1: Fractional Response Model ───────────────────────────────
    log.info("")
    log.info("[3/6] Fractional Response Model (logit-OLS) …")
    frm = FractionalResponseModel().fit(X_tr, y_tr)
    _eval_model(frm, "FRM")

    log.info("\n  FRM Coefficients (logit scale):")
    for fname, coef in zip(feats, frm.coef_):
        log.info("    %-35s  %+.4f", fname, coef)

    plot_actual_vs_pred(y_tr, frm.predict(X_tr), "Fractional Response Model",
                        FIG_DIR / "lgd_frm_actual_vs_pred.png")

    # ── Model 2: Spline Regression ────────────────────────────────────────
    log.info("")
    log.info("[4/6] Natural Spline Regression (degree=3, knots=5) …")

    class ClippedSpline:
        """Wraps the spline Pipeline and clips predictions to [0, 1]."""
        def __init__(self, pipe): self._pipe = pipe
        def fit(self, X, y):   self._pipe.fit(X, y); return self
        def predict(self, X):  return np.clip(self._pipe.predict(X), 0, 1)

    spline = ClippedSpline(build_spline_model(degree=3, n_knots=5))
    spline.fit(X_tr, y_tr)
    _eval_model(spline, "Spline")

    # ── Model 3: Random Forest ────────────────────────────────────────────
    log.info("")
    log.info("[5/6] Random Forest Regressor (200 trees, max_depth=6) …")
    rf = RandomForestRegressor(**RF_PARAMS).fit(X_tr, y_tr)
    _eval_model(rf, "Random Forest")

    imp_rf = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    log.info("\n  Random Forest Feature Importances (top 10):\n%s", imp_rf.head(10).to_string())

    plot_actual_vs_pred(y_tr, rf.predict(X_tr), "Random Forest",
                        FIG_DIR / "lgd_rf_actual_vs_pred.png")

    # ── Model 4: XGBoost ──────────────────────────────────────────────────
    if XGB_AVAILABLE:
        log.info("")
        log.info("  XGBoost Regressor …")
        xgb = XGBRegressor(**XGB_PARAMS)
        xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_oo, y_oo)] if len(y_oo) > 0 else None,
            verbose=False,
        )
        _eval_model(xgb, "XGBoost")
        plot_actual_vs_pred(y_tr, xgb.predict(X_tr), "XGBoost",
                            FIG_DIR / "lgd_xgb_actual_vs_pred.png")
    else:
        log.info("  XGBoost not available — skipping.")

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("")
    log.info("[6/6] Summary and saving outputs …")
    metrics_df = pd.DataFrame(all_metrics)

    oos_oot = metrics_df[metrics_df["split"].isin(["OOS", "OOT"])].copy()
    if not oos_oot.empty:
        log.info("\n  Model comparison (OOS + OOT):")
        log.info(
            "\n%s",
            oos_oot[["model", "split", "rmse", "mae", "r2", "bias"]]
            .sort_values(["split", "rmse"])
            .to_string(index=False),
        )

    if train_preds:
        plot_lgd_distributions(y_tr, train_preds, FIG_DIR / "lgd_distributions.png")

    # Collect per-row predictions from all models
    pred_rows = []
    model_objects = {"FRM": frm, "Spline": spline, "Random Forest": rf}
    if XGB_AVAILABLE:
        model_objects["XGBoost"] = xgb

    for split_label, df_split, X in [
        ("train", train, X_tr),
        ("oos",   oos,   X_oo),
        ("oot",   oot,   X_ot),
    ]:
        if len(df_split) == 0 or len(X) == 0:
            continue
        y = {"train": y_tr, "oos": y_oo, "oot": y_ot}[split_label]
        row = df_split[["loan_seq_num"]].copy() if "loan_seq_num" in df_split.columns \
              else pd.DataFrame(index=df_split.index)
        row["split"]      = split_label
        row["actual_lgd"] = y
        for mname, mobj in model_objects.items():
            row[f"{mname.lower().replace(' ', '_')}_pred"] = np.clip(mobj.predict(X), 0, 1)
        pred_rows.append(row)

    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(
            OUT_DIR / "lgd_predictions.csv", index=False
        )

    metrics_df.to_csv(OUT_DIR / "lgd_metrics.csv", index=False)
    log.info("  Files saved to %s", OUT_DIR.resolve())

    if len(train) < 100:
        log.warning(
            "LGD metrics should be interpreted cautiously (small sample: %d rows).  "
            "Download pre-2010 Freddie Mac vintages for robust results.",
            len(train),
        )

    log.info("")
    log.info("=" * 65)
    log.info("Ch.3 complete.  Pipeline finished.")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
