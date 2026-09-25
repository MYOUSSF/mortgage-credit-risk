"""
=============================================================================
Mortgage Credit Risk Modelling  |  Ch.3 — Loss Given Default Models
=============================================================================
Script  : 04_lgd_models.py
Purpose : Fit and evaluate three LGD models on the resolved-default sample.

Models
------
  1. Fractional Response Model (FRM)   — Papke-Wooldridge (1996) quasi-binomial
                                         GLM (logit link); E[LGD|X] estimated
                                         directly, no boundary-value clipping
  2. Two-Stage Model                   — 3-class multinomial logit over
                                         {LGD=0, interior, LGD=1} combined with
                                         a beta regression on the interior
                                         observations; models the point masses
                                         explicitly rather than only the mean
  3. Random Forest Regressor           — 200 trees, max depth 6

  All three are fit with inverse-probability-of-censoring (IPCW) sample
  weights (see 01_data_preprocessing.py::compute_ipcw_weights) correcting
  for LGD workout-period truncation bias: fast-resolving defaults (short
  sales) are otherwise overrepresented relative to slow-resolving ones
  (contested foreclosure, REO) near the end of the observation window.

Why these three, and why the spline and XGBoost models were removed
--------------------------------------------------------------------
  The previous suite fit four models (FRM, natural spline regression, random
  forest, XGBoost) that all estimated the same thing — the conditional MEAN
  of LGD — and differed only in functional flexibility. On ~150 training
  observations that is the wrong axis to spend model risk on: two additional
  flexible mean estimators add variance and review burden without answering
  a question the first two do not already answer.

  The two-stage model replaces them because it answers a different question.
  The LGD distribution here is strongly bimodal with point masses at exactly
  0 (full recovery, no loss after disposition) and exactly 1 (total loss).
  A conditional-mean model can fit the average of that distribution well
  while assigning essentially zero probability to either mass — predicting
  0.45 for a population where almost no loan actually loses 45%. The
  two-stage model represents the masses explicitly and yields the full
  conditional distribution, which is what any downturn-LGD or stressed-LGD
  work needs (see predict_quantile()).

  The spread between the three is reported in lgd_point_mass_calibration.csv:
  the observed share of LGD=0 and LGD=1 against each model's predicted share.
  For the FRM and the random forest that predicted share is typically ~0 —
  which is the point of the comparison, not an incidental diagnostic.

Categorical encoding
--------------------
  Two preprocessing paths, BOTH fitted on the training split only:

    Linear path (FRM, stage 1, stage 2) — one-hot with a dropped reference
      level, rare levels (< config.LGD_MIN_LEVEL_COUNT training rows) and
      unseen levels grouped into "other", median imputation for numerics.
    Tree path (random forest) — integer codes, mapping fitted on train, with
      a dedicated code for unseen levels.

  This replaces a single label-encoding path that fed integer codes to the
  FRM as if they were continuous. That treatment asserted, for example, that
  property_state has one linear slope across alphabetically ordered states —
  AK to AL to AR being a unit step of equal effect each time. Any FRM
  coefficient on a categorical produced before this fix is meaningless, and
  the previously reported FRM metrics should be regenerated rather than
  quoted. The old encoder was also fitted on the union of train, OOS and OOT,
  which leaked the evaluation splits' level sets into the training encoding.

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

    Plus, for the two-stage model specifically: stage-1 multiclass log loss
    and confusion matrix, stage-2 RMSE/MAE on interior rows only, and
    point-mass calibration for all three models.

Note on Sample Size
-------------------
  The sample dataset (50,000 loans / year) yields only ~150 LGD observations
  from post-2010 vintages where crisis-era defaults are absent.  Metric
  variance is therefore high.  For publication-quality results, download the
  full (non-sample) Freddie Mac dataset or focus on 2004–2009 vintages where
  default rates reached 3–15%.

  Every small-sample fallback in this script is logged explicitly rather than
  applied silently — see the WARNING lines for dropped classes, constant-mean
  stage 2, and penalised FRM fits.

Inputs
------
  data/processed/lgd_train.parquet
  data/processed/lgd_oos.parquet
  data/processed/lgd_oot.parquet

Outputs
-------
  data/processed/lgd_metrics.csv
  data/processed/lgd_predictions.csv
  data/processed/lgd_champion_summary.csv   — lowest-RMSE model's mean predicted
                                               LGD, consumed by
                                               07_macro_scenario_analysis.py as
                                               the scenario-conditional ECL
                                               LGD anchor (schema unchanged)
  data/processed/lgd_two_stage_components.csv
  data/processed/lgd_two_stage_stage1_coefficients.csv
  data/processed/lgd_two_stage_stage2_coefficients.csv
  data/processed/lgd_point_mass_calibration.csv
  data/figures/lgd_frm_actual_vs_pred.png
  data/figures/lgd_two_stage_actual_vs_pred.png
  data/figures/lgd_rf_actual_vs_pred.png
  data/figures/lgd_distributions.png
  data/figures/lgd_point_mass_calibration.png
=============================================================================
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, logit as _logit_fn
from scipy.stats import beta as beta_dist
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score)

import statsmodels.api as sm

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in [REPO_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import config

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING
# =============================================================================

config.configure_logging("lgd_models.log")
log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROC_DIR = config.PROC_DIR
OUT_DIR  = config.OUT_DIR
FIG_DIR  = config.FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = config.TARGET_LGD
SEED   = config.SEED

# LGD feature set — thesis §3.4
FEATURES: list[str] = config.LGD_FEATURES

CAT_FEATURES: list[str] = config.LGD_CAT_FEATURES

MIN_LEVEL_COUNT = config.LGD_MIN_LEVEL_COUNT
OTHER_LEVEL     = config.LGD_OTHER_LEVEL
UNSEEN_CODE     = config.LGD_UNSEEN_CODE
BOUNDARY_EPS    = config.LGD_BOUNDARY_EPS
MIN_CLASS_OBS   = config.LGD_TWO_STAGE_MIN_CLASS_OBS

PLT_STYLE: dict = config.PLT_STYLE

RF_PARAMS: dict = dict(
    n_estimators   = 200,
    max_depth      = 6,
    min_samples_leaf = 5,
    n_jobs         = -1,
    random_state   = SEED,
)

# Class labels for the two-stage model. Ordered 0 < interior < 1 so that the
# integer codes are monotone in loss severity.
CLASS_ZERO     = 0
CLASS_INTERIOR = 1
CLASS_ONE      = 2
CLASS_LABELS = {CLASS_ZERO: "LGD=0", CLASS_INTERIOR: "interior", CLASS_ONE: "LGD=1"}


# =============================================================================
# PREPROCESSING
# =============================================================================

class LinearPreprocessor:
    """
    Design matrix for the models that are linear in their covariates (the
    FRM, the two-stage stage-1 multinomial logit, and the stage-2 beta mean
    model). Fitted on the TRAINING split only.

    Steps, in order:
      1. Drop features that are entirely NaN in training — SimpleImputer has
         no median to compute for them and would silently drop them from its
         output, desynchronising feature names from matrix columns.
      2. Categorical levels seen fewer than config.LGD_MIN_LEVEL_COUNT times
         in training are folded into config.LGD_OTHER_LEVEL, as are levels
         never seen in training at all. On ~150 rows, property_state would
         otherwise contribute ~50 indicators, most identifying one loan.
      3. One-hot encode, dropping the first surviving level of each
         categorical as the reference. The dropped level is recorded in
         reference_levels_ so coefficients can be read against it.
      4. Median-impute numeric features using training medians.

    feature_names_ is always aligned with the transformed matrix's columns.
    """

    def __init__(self, features: list[str], cat_features: list[str],
                 min_level_count: int = MIN_LEVEL_COUNT) -> None:
        self.features = list(features)
        self.cat_features = list(cat_features)
        self.min_level_count = min_level_count

        self.numeric_: list[str] = []
        self.categorical_: list[str] = []
        self.kept_levels_: dict[str, list[str]] = {}
        self.reference_levels_: dict[str, str] = {}
        self.indicator_names_: list[str] = []
        self.feature_names_: list[str] = []
        self._imputer: SimpleImputer | None = None

    # -- internals ----------------------------------------------------------

    def _clean_levels(self, series: pd.Series) -> pd.Series:
        return series.fillna("missing").astype(str)

    def _map_levels(self, series: pd.Series, col: str) -> pd.Series:
        kept = set(self.kept_levels_[col])
        return self._clean_levels(series).map(
            lambda v, k=kept: v if v in k else OTHER_LEVEL
        )

    # -- interface ----------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "LinearPreprocessor":
        available = [f for f in self.features if f in train.columns]
        # Step 1: a column that is all-NaN in training carries no information
        # and cannot be imputed.
        available = [f for f in available if train[f].notna().any()]

        self.categorical_ = [f for f in available if f in self.cat_features]
        self.numeric_ = [f for f in available if f not in self.cat_features]

        # Step 2 + 3: level grouping and reference selection.
        self.kept_levels_ = {}
        self.reference_levels_ = {}
        self.indicator_names_ = []
        for col in self.categorical_:
            counts = self._clean_levels(train[col]).value_counts()
            kept = sorted(counts[counts >= self.min_level_count].index.tolist())
            self.kept_levels_[col] = kept

            grouped = self._map_levels(train[col], col)
            levels = sorted(grouped.unique().tolist())
            if not levels:
                self.reference_levels_[col] = OTHER_LEVEL
                continue
            self.reference_levels_[col] = levels[0]
            self.indicator_names_ += [f"{col}[{lvl}]" for lvl in levels[1:]]

        self.feature_names_ = list(self.numeric_) + list(self.indicator_names_)

        if self.numeric_:
            self._imputer = SimpleImputer(strategy="median")
            self._imputer.fit(train[self.numeric_])

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if n == 0:
            return np.empty((0, len(self.feature_names_)))

        blocks = []
        if self.numeric_:
            blocks.append(np.asarray(self._imputer.transform(df[self.numeric_]),
                                     dtype=float))

        if self.indicator_names_:
            indicators = np.zeros((n, len(self.indicator_names_)), dtype=float)
            position = {name: i for i, name in enumerate(self.indicator_names_)}
            for col in self.categorical_:
                grouped = self._map_levels(df[col], col).to_numpy()
                for lvl in set(grouped):
                    key = f"{col}[{lvl}]"
                    if key in position:  # the reference level has no column
                        indicators[:, position[key]] = (grouped == lvl).astype(float)
            blocks.append(indicators)

        if not blocks:
            return np.empty((n, 0))
        return np.column_stack(blocks)

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)


class TreePreprocessor:
    """
    Design matrix for the random forest. Integer codes are acceptable for a
    tree — a split on "code <= 3" can isolate any subset given enough splits,
    so the ordering imposes no functional-form assumption the way it does for
    a linear model.

    Fitted on training only; levels never seen in training map to
    config.LGD_UNSEEN_CODE, kept distinct from every fitted code so the tree
    can split on "unknown" rather than having unseen levels silently take on
    the identity of the first training level.

    NaNs in numeric features are median-imputed on training medians, matching
    the linear path so the two models see the same information.
    """

    def __init__(self, features: list[str], cat_features: list[str]) -> None:
        self.features = list(features)
        self.cat_features = list(cat_features)

        self.numeric_: list[str] = []
        self.categorical_: list[str] = []
        self.level_codes_: dict[str, dict[str, int]] = {}
        self.feature_names_: list[str] = []
        self._imputer: SimpleImputer | None = None

    def fit(self, train: pd.DataFrame) -> "TreePreprocessor":
        available = [f for f in self.features if f in train.columns]
        available = [f for f in available if train[f].notna().any()]

        self.categorical_ = [f for f in available if f in self.cat_features]
        self.numeric_ = [f for f in available if f not in self.cat_features]

        self.level_codes_ = {}
        for col in self.categorical_:
            levels = sorted(train[col].fillna("missing").astype(str).unique().tolist())
            self.level_codes_[col] = {lvl: i for i, lvl in enumerate(levels)}

        # Column order: numerics then categoricals, matching feature_names_.
        self.feature_names_ = list(self.numeric_) + list(self.categorical_)

        if self.numeric_:
            self._imputer = SimpleImputer(strategy="median")
            self._imputer.fit(train[self.numeric_])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if n == 0:
            return np.empty((0, len(self.feature_names_)))

        blocks = []
        if self.numeric_:
            blocks.append(np.asarray(self._imputer.transform(df[self.numeric_]),
                                     dtype=float))
        if self.categorical_:
            codes = np.zeros((n, len(self.categorical_)), dtype=float)
            for j, col in enumerate(self.categorical_):
                lookup = self.level_codes_[col]
                raw = df[col].fillna("missing").astype(str).to_numpy()
                codes[:, j] = [lookup.get(v, UNSEEN_CODE) for v in raw]
            blocks.append(codes)

        if not blocks:
            return np.empty((n, 0))
        return np.column_stack(blocks)

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)


class LGDData(NamedTuple):
    """
    Both preprocessing paths for all three splits, plus the targets and the
    IPCW training weights.

    A NamedTuple rather than a bare tuple because there are now two design
    matrices per split: positional unpacking of ten-plus elements is exactly
    where a silent mis-wiring (linear matrix into the tree model) would hide.
    """
    X_lin: dict[str, np.ndarray]
    X_tree: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    w_train: np.ndarray
    linear_features: list[str]
    tree_features: list[str]
    linear_pre: LinearPreprocessor
    tree_pre: TreePreprocessor


def prepare(train: pd.DataFrame, oos: pd.DataFrame, oot: pd.DataFrame) -> LGDData:
    """
    Fit both preprocessing paths on TRAINING data and transform every split.

    Returns an LGDData bundle. IPCW weights default to 1.0 when the
    ipcw_weight column is absent (older parquets, or synthetic test data) —
    unchanged behaviour.
    """
    linear_pre = LinearPreprocessor(FEATURES, CAT_FEATURES).fit(train)
    tree_pre = TreePreprocessor(FEATURES, CAT_FEATURES).fit(train)

    splits = {"train": train, "oos": oos, "oot": oot}
    X_lin = {k: linear_pre.transform(df) for k, df in splits.items()}
    X_tree = {k: tree_pre.transform(df) for k, df in splits.items()}
    y = {k: (df[TARGET].to_numpy() if len(df) else np.array([]))
         for k, df in splits.items()}

    w_train = (train["ipcw_weight"].to_numpy() if "ipcw_weight" in train.columns
               else np.ones(len(train)))

    return LGDData(X_lin=X_lin, X_tree=X_tree, y=y, w_train=w_train,
                   linear_features=linear_pre.feature_names_,
                   tree_features=tree_pre.feature_names_,
                   linear_pre=linear_pre, tree_pre=tree_pre)


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
        "  [%-13s] %s  n=%s  RMSE=%.4f  MAE=%.4f  R²=%.4f  Bias=%+.4f",
        model, name, f"{len(y_true):,}", rmse, mae, r2, bias,
    )
    return {
        "model": model, "split": name, "n": len(y_true),
        "rmse": round(rmse, 4), "mae": round(mae, 4),
        "r2":   round(r2, 4),  "bias": round(bias, 4),
    }


def select_champion(metrics_df: pd.DataFrame, preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick the model with the lowest RMSE on OOS (falling back to Train if
    OOS has no rows — the small-sample case this script already warns
    about) and compute its mean predicted LGD on that ranking split.

    This anchor value is what 07_macro_scenario_analysis.py uses as the
    base LGD for its scenario-conditional ECL calculation, in place of a
    fixed assumption disconnected from this model suite's actual output.

    Returns a one-row DataFrame, or an empty DataFrame if no metrics are
    available to rank models on (e.g. zero LGD training rows).

    The output schema is load-bearing: 07_macro_scenario_analysis.py's
    load_base_lgd() reads champion_model, anchor_split and anchor_mean_lgd
    by name. It is unchanged by the move to a three-model suite.
    """
    if metrics_df.empty or "split" not in metrics_df.columns:
        return pd.DataFrame()

    oos_metrics = metrics_df[metrics_df["split"] == "OOS"]
    rank_split = "OOS" if not oos_metrics.empty else "Train"
    rank_df = metrics_df[metrics_df["split"] == rank_split]
    if rank_df.empty:
        return pd.DataFrame()

    champion_row  = rank_df.sort_values("rmse").iloc[0]
    champion_name = champion_row["model"]
    pred_col      = f"{champion_name.lower().replace(' ', '_')}_pred"
    split_key     = rank_split.lower()

    anchor_rows = preds_df[preds_df["split"] == split_key] if not preds_df.empty else pd.DataFrame()
    anchor_mean_lgd = (
        float(anchor_rows[pred_col].mean())
        if pred_col in anchor_rows.columns and not anchor_rows.empty
        else np.nan
    )

    return pd.DataFrame([{
        "champion_model":  champion_name,
        "anchor_split":    rank_split,
        "anchor_mean_lgd": round(anchor_mean_lgd, 6) if not np.isnan(anchor_mean_lgd) else np.nan,
        "n_anchor_obs":    len(anchor_rows),
        "rmse": champion_row["rmse"], "mae": champion_row["mae"],
        "r2":   champion_row["r2"],   "bias": champion_row["bias"],
    }])


# =============================================================================
# MODEL 1 — FRACTIONAL RESPONSE MODEL
# =============================================================================

class FractionalResponseModel:
    """
    Thesis §3.3 — Papke-Wooldridge (1996) fractional response model: a
    quasi-binomial GLM with a logit link, fit by quasi-maximum-likelihood.

        E[LGD | X] = Λ(Xβ) = 1 / (1 + exp(-Xβ))

    This is NOT the same as OLS on logit(LGD) followed by a sigmoid
    back-transform (a common mislabelling). The Binomial quasi-likelihood
    is well-defined at the boundary values y=0 and y=1 themselves, so no
    epsilon-clipping of the target is needed (LGD of exactly 0% or 100%
    loss is common and legitimate here). It also targets E[LGD|X]
    directly through the GLM's mean function, avoiding the systematic
    bias an OLS-then-sigmoid back-transform introduces via Jensen's
    inequality: E[sigmoid(z)] != sigmoid(E[z]) for a linear-in-X z.

    Small-sample behaviour: with ~150 rows and a one-hot design matrix, the
    unpenalised fit can fail to converge or return coefficients of enormous
    magnitude where a level perfectly separates the outcome. fit() therefore
    tries the unpenalised fit first — its coefficients are the auditable
    output and the thing the thesis specifies — and falls back to an L2
    penalty only when that fails, logging the fallback. The penalty applies
    to slope coefficients only: penalising the intercept would shift the
    predicted LGD level, which is exactly the quantity feeding the ECL
    anchor downstream.
    """

    def __init__(self) -> None:
        self._result = None
        self.penalised_ = False
        self.converged_ = None

    @staticmethod
    def _with_const(X: np.ndarray) -> np.ndarray:
        return sm.add_constant(X, has_constant="add")

    def fit(self, X: np.ndarray, y: np.ndarray,
            freq_weights: np.ndarray | None = None,
            l2_alpha: float | None = None,
            use_l2_fallback: bool = config.LGD_FRM_USE_L2_FALLBACK,
            ) -> "FractionalResponseModel":
        design = self._with_const(X)
        model = sm.GLM(y, design, family=sm.families.Binomial(),
                       freq_weights=freq_weights)

        try:
            result = model.fit()
            converged = bool(getattr(result, "converged", True))
            extreme = (np.isfinite(np.asarray(result.params)).all()
                       and np.max(np.abs(np.asarray(result.params))) > 1e6)
            if converged and not extreme:
                self._result = result
                self.converged_ = True
                return self
            reason = ("did not converge" if not converged
                      else "produced extreme coefficients (|beta| > 1e6, "
                           "typically perfect separation)")
        except Exception as exc:
            result, reason = None, f"raised {type(exc).__name__}: {exc}"

        if not use_l2_fallback:
            if result is None:
                raise RuntimeError(f"FRM fit failed and L2 fallback is disabled: {reason}")
            log.warning("  [FRM] Unpenalised fit %s; L2 fallback disabled — "
                        "using the unpenalised result as-is.", reason)
            self._result = result
            self.converged_ = False
            return self

        alpha = config.LGD_FRM_L2_ALPHA if l2_alpha is None else l2_alpha
        log.warning(
            "  [FRM] Unpenalised quasi-binomial GLM %s — falling back to an "
            "L2-penalised fit (alpha=%.3g on slopes only). Coefficients are "
            "shrunk and their standard errors are not valid for inference.",
            reason, alpha,
        )
        # alpha=0 on the intercept column keeps the predicted level unpenalised.
        alpha_vec = np.full(design.shape[1], float(alpha))
        alpha_vec[0] = 0.0
        self._result = model.fit_regularized(alpha=alpha_vec, L1_wt=0.0)
        self.penalised_ = True
        self.converged_ = False
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._result.predict(self._with_const(X)))

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(self._result.params)[1:]

    @property
    def intercept_(self) -> float:
        return float(np.asarray(self._result.params)[0])


# =============================================================================
# MODEL 2 — TWO-STAGE MODEL
# =============================================================================

def assign_lgd_class(y: np.ndarray, eps: float = BOUNDARY_EPS) -> np.ndarray:
    """
    Map continuous LGD onto the three-class support used by stage 1.

        class 0 (CLASS_ZERO)     : LGD <= eps          — full recovery
        class 1 (CLASS_INTERIOR) : eps < LGD < 1 - eps — partial loss
        class 2 (CLASS_ONE)      : LGD >= 1 - eps      — total loss

    eps (config.LGD_BOUNDARY_EPS) exists because the target is a ratio of two
    reported dollar amounts: an exact 0.0 or 1.0 is how "full recovery" and
    "total loss" are encoded, and a value 1e-4 away from either is a rounding
    artefact rather than an economically distinct outcome. Keeping the
    interior strictly inside (0, 1) also keeps the beta log-likelihood
    defined — it involves log(y) and log(1-y).
    """
    y = np.asarray(y, dtype=float)
    out = np.full(y.shape, CLASS_INTERIOR, dtype=int)
    out[y <= eps] = CLASS_ZERO
    out[y >= 1.0 - eps] = CLASS_ONE
    return out


def weighted_beta_nll(params: np.ndarray, X: np.ndarray, y: np.ndarray,
                      w: np.ndarray, l2_alpha: float = 0.0
                      ) -> tuple[float, np.ndarray]:
    """
    Negative weighted beta-regression log-likelihood and its analytic gradient.

    Ferrari & Cribari-Neto (2004) mean/precision parametrisation: for
    y ~ Beta(p, q) with mean mu and precision phi, p = mu*phi and
    q = (1-mu)*phi, giving Var(y) = mu(1-mu)/(1+phi).

        logit(mu_i) = X_i @ gamma          (mean model)
        log(phi)    = delta                (constant precision)

        ll_i = lgamma(phi) - lgamma(mu_i phi) - lgamma((1-mu_i) phi)
               + (mu_i phi - 1) log(y_i) + ((1-mu_i) phi - 1) log(1-y_i)

    and the objective is -sum_i w_i * ll_i (+ the L2 penalty).

    Weights are the reason this exists rather than a call to
    statsmodels.othermod.betareg.BetaModel: that class does not support
    observation weights, and — worse for a pipeline that suppresses warnings
    — it ACCEPTS a `weights=` keyword, emits only a ValueWarning, and returns
    bit-identical unweighted estimates. Silently dropping the IPCW weights
    would reintroduce exactly the workout-period truncation bias
    compute_ipcw_weights() exists to correct.

    params is [gamma (k), delta (1)]. The L2 penalty applies to gamma's
    non-intercept entries only, on the convention that the design matrix's
    first column is the intercept.

    Returns (negative log-likelihood, gradient) for scipy.optimize with
    jac=True.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    gamma, delta = params[:-1], params[-1]
    phi = float(np.exp(delta))

    eta = X @ gamma
    mu = expit(eta)
    # Guard the digamma/log terms against a mean pinned at an endpoint.
    mu = np.clip(mu, 1e-10, 1 - 1e-10)

    mu_phi = mu * phi
    one_minus_mu_phi = (1.0 - mu) * phi
    log_y = np.log(y)
    log_1my = np.log1p(-y)

    ll = (gammaln(phi) - gammaln(mu_phi) - gammaln(one_minus_mu_phi)
          + (mu_phi - 1.0) * log_y + (one_minus_mu_phi - 1.0) * log_1my)
    nll = -float(np.sum(w * ll))

    # y_star / mu_star are Ferrari & Cribari-Neto's score components.
    y_star = log_y - log_1my
    mu_star = digamma(mu_phi) - digamma(one_minus_mu_phi)

    # d ll / d eta = phi * (y_star - mu_star) * dmu/deta, with dmu/deta = mu(1-mu)
    dll_deta = phi * (y_star - mu_star) * mu * (1.0 - mu)
    grad_gamma = -(X.T @ (w * dll_deta))

    # d ll / d phi, then chain through phi = exp(delta)
    dll_dphi = (mu * (y_star - mu_star) + log_1my
                - digamma(one_minus_mu_phi) + digamma(phi))
    grad_delta = -float(np.sum(w * dll_dphi * phi))

    if l2_alpha:
        penalty_mask = np.ones_like(gamma)
        penalty_mask[0] = 0.0  # never penalise the intercept
        nll += l2_alpha * float(np.sum((gamma * penalty_mask) ** 2))
        grad_gamma += 2.0 * l2_alpha * gamma * penalty_mask

    return nll, np.concatenate([grad_gamma, [grad_delta]])


class TwoStageLGDModel:
    """
    Two-stage LGD model: a discrete choice over the shape of the loss, then a
    continuous model of its severity given a partial loss.

    Motivation
    ----------
    Realised LGD on resolved mortgage defaults is not a smooth variable on
    [0, 1]. It is bimodal with genuine point masses at exactly 0 (the
    disposition covers the balance — full recovery) and exactly 1 (nothing is
    recovered — total loss), with a diffuse spread of partial losses between
    them. The FRM and the random forest both model only E[LGD | x]. A
    conditional-mean model can match that average well while placing almost
    no probability on either mass — reporting a confident 0.45 for a
    population in which hardly any loan actually loses 45% of its balance.

    This model represents the masses explicitly:

        Stage 1 : P(class | x) over {LGD=0, interior, LGD=1}
                  — multinomial (softmax) logistic regression, L2-penalised
        Stage 2 : E[LGD | x, interior] = mu(x) via beta regression on the
                  interior rows only, with constant precision phi

    and composes them into the conditional mean

        E[LGD | x] = P(LGD=1 | x) * 1 + P(interior | x) * mu(x) + P(LGD=0 | x) * 0

    which is what predict() returns, so RMSE/MAE/R²/bias and champion
    selection treat it exactly like the other two models. Unlike them, it
    also yields the full conditional distribution — see predict_components()
    and predict_quantile(), which is what downturn/stressed-LGD work needs
    and what a mean-only model cannot supply.

    Small-sample behaviour
    ----------------------
    Every degenerate case that ~150 observations can produce is handled by an
    explicitly logged fallback rather than an exception:

      * A boundary class with fewer than config.LGD_TWO_STAGE_MIN_CLASS_OBS
        training rows is dropped from stage 1 (which then fits over the
        remaining classes, or is skipped entirely if only one class remains)
        and is assigned probability 0 at prediction time.
      * Fewer interior rows than that minimum, or a beta optimiser that fails
        to converge, falls back to a constant stage-2 mean: the IPCW-weighted
        mean of the interior observations, or 0.5 if there are none.
      * predict() never raises because a class was absent in training.
    """

    def __init__(self, stage1_C: float = config.LGD_TWO_STAGE_STAGE1_C,
                 beta_l2_alpha: float = config.LGD_TWO_STAGE_BETA_L2_ALPHA,
                 min_class_obs: int = MIN_CLASS_OBS,
                 boundary_eps: float = BOUNDARY_EPS,
                 seed: int = SEED) -> None:
        self.stage1_C = stage1_C
        self.beta_l2_alpha = beta_l2_alpha
        self.min_class_obs = min_class_obs
        self.boundary_eps = boundary_eps
        self.seed = seed

        # Stage 1
        self.stage1_ = None
        self.modelled_classes_: list[int] = []
        self.dropped_classes_: list[int] = []
        self.single_class_: int | None = None
        # Stage 2
        self.gamma_: np.ndarray | None = None
        self.phi_: float = 1.0
        self.constant_mu_: float | None = None
        self.stage2_converged_: bool = False
        self.n_interior_: int = 0

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _with_const(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])

    # -- stage 2 ------------------------------------------------------------

    def _fit_stage2(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """Weighted beta regression on the interior observations."""
        self.n_interior_ = len(y)
        design = self._with_const(X)

        weighted_mean = float(np.average(y, weights=w)) if len(y) else 0.5
        # Method-of-moments precision: Var = mu(1-mu)/(1+phi)  =>
        # phi = mu(1-mu)/Var - 1. Floored at a small positive value so a
        # degenerate variance cannot produce phi <= 0.
        if len(y) > 1:
            var = float(np.average((y - weighted_mean) ** 2, weights=w))
            phi_mom = (weighted_mean * (1 - weighted_mean) / var - 1.0) if var > 0 else 1.0
        else:
            phi_mom = 1.0
        phi_mom = float(np.clip(phi_mom, 0.1, 1e4))

        if len(y) < self.min_class_obs:
            self.constant_mu_ = weighted_mean
            self.phi_ = phi_mom
            self.stage2_converged_ = False
            log.warning(
                "  [Two Stage] Only %d interior observation(s) (< %d) — stage 2 "
                "falls back to a constant IPCW-weighted interior mean of %.4f "
                "(phi=%.3f by method of moments). No covariates enter the "
                "severity model.",
                len(y), self.min_class_obs, weighted_mean, phi_mom,
            )
            return

        # Start gamma from a weighted FRM on the interior rows — same link, so
        # it lands in the right region of the parameter space.
        try:
            start_frm = FractionalResponseModel().fit(X, y, freq_weights=w)
            gamma0 = np.concatenate([[start_frm.intercept_], start_frm.coef_])
            if not np.isfinite(gamma0).all():
                raise ValueError("non-finite FRM start values")
        except Exception:
            gamma0 = np.zeros(design.shape[1])
            gamma0[0] = _logit_fn(np.clip(weighted_mean, 1e-6, 1 - 1e-6))

        x0 = np.concatenate([gamma0, [np.log(phi_mom)]])

        try:
            result = minimize(
                weighted_beta_nll, x0, args=(design, y, w, self.beta_l2_alpha),
                method="L-BFGS-B", jac=True,
                options={"maxiter": 1000},
            )
            if not result.success or not np.isfinite(result.x).all():
                raise RuntimeError(result.message)
            self.gamma_ = result.x[:-1]
            self.phi_ = float(np.exp(result.x[-1]))
            self.constant_mu_ = None
            self.stage2_converged_ = True
            log.info("  [Two Stage] Stage 2 beta regression converged on %d "
                     "interior rows (phi=%.3f, %d mean coefficients).",
                     len(y), self.phi_, len(self.gamma_))
        except Exception as exc:
            self.gamma_ = None
            self.constant_mu_ = weighted_mean
            self.phi_ = phi_mom
            self.stage2_converged_ = False
            log.warning(
                "  [Two Stage] Stage 2 beta optimiser failed (%s) — falling "
                "back to a constant interior mean of %.4f (phi=%.3f).",
                exc, weighted_mean, phi_mom,
            )

    # -- interface ----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "TwoStageLGDModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = (np.ones(len(y)) if sample_weight is None
             else np.asarray(sample_weight, dtype=float))

        classes = assign_lgd_class(y, self.boundary_eps)
        counts = {c: int((classes == c).sum()) for c in
                  (CLASS_ZERO, CLASS_INTERIOR, CLASS_ONE)}
        log.info("  [Two Stage] Training class counts — %s",
                 ", ".join(f"{CLASS_LABELS[c]}: {counts[c]}" for c in sorted(counts)))

        # ── Stage 1 ────────────────────────────────────────────────────────
        # Boundary classes below the minimum are dropped; the interior class
        # is never dropped from stage 1 (stage 2 has its own fallback), since
        # dropping it would leave no way to express a partial loss at all.
        self.dropped_classes_ = [
            c for c in (CLASS_ZERO, CLASS_ONE)
            if 0 < counts[c] < self.min_class_obs or counts[c] == 0
        ]
        for c in self.dropped_classes_:
            log.warning(
                "  [Two Stage] Class %s has %d training observation(s) "
                "(< %d) — dropped from stage 1 and assigned probability 0 at "
                "prediction time.", CLASS_LABELS[c], counts[c], self.min_class_obs,
            )

        keep_mask = ~np.isin(classes, self.dropped_classes_)
        kept_classes = classes[keep_mask]
        self.modelled_classes_ = sorted(set(kept_classes.tolist()))

        if len(self.modelled_classes_) <= 1:
            self.single_class_ = (self.modelled_classes_[0]
                                  if self.modelled_classes_ else CLASS_INTERIOR)
            self.stage1_ = None
            log.warning(
                "  [Two Stage] Only one class survives the minimum-count "
                "filter (%s) — stage 1 is skipped and that class gets "
                "probability 1.", CLASS_LABELS[self.single_class_],
            )
        else:
            self.single_class_ = None
            self.stage1_ = LogisticRegression(
                C=self.stage1_C, max_iter=5000, random_state=self.seed,
            )
            self.stage1_.fit(X[keep_mask], kept_classes,
                             sample_weight=w[keep_mask])

        # ── Stage 2 ────────────────────────────────────────────────────────
        interior = classes == CLASS_INTERIOR
        self._fit_stage2(X[interior], y[interior], w[interior])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        P(class | x) as an (n, 3) array ordered [P(LGD=0), P(interior), P(LGD=1)].

        Classes dropped in training get exactly 0. Rows always sum to 1.
        """
        X = np.asarray(X, dtype=float)
        n = len(X)
        out = np.zeros((n, 3), dtype=float)

        if self.single_class_ is not None:
            out[:, self.single_class_] = 1.0
            return out

        probs = self.stage1_.predict_proba(X)
        for j, cls in enumerate(self.stage1_.classes_):
            out[:, int(cls)] = probs[:, j]
        return out

    def predict_interior_mean(self, X: np.ndarray) -> np.ndarray:
        """mu(x) — E[LGD | x, interior], from stage 2."""
        X = np.asarray(X, dtype=float)
        if self.gamma_ is None:
            fallback = 0.5 if self.constant_mu_ is None else self.constant_mu_
            return np.full(len(X), float(fallback))
        return expit(self._with_const(X) @ self.gamma_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Conditional mean — the quantity RMSE/MAE/R²/bias and champion
        selection use, so this model is directly comparable to the FRM and
        the random forest:

            E[LGD | x] = P(LGD=1|x)*1 + P(interior|x)*mu(x) + P(LGD=0|x)*0
        """
        proba = self.predict_proba(X)
        mu = self.predict_interior_mean(X)
        return proba[:, CLASS_ONE] + proba[:, CLASS_INTERIOR] * mu

    def predict_components(self, X: np.ndarray) -> pd.DataFrame:
        """
        The full conditional distribution, per row: the three class
        probabilities, the interior mean and the precision.

        This is what a mean-only model cannot give, and the reason the
        two-stage model is in the suite.
        """
        proba = self.predict_proba(X)
        mu = self.predict_interior_mean(X)
        return pd.DataFrame({
            "p_lgd_zero":     proba[:, CLASS_ZERO],
            "p_lgd_interior": proba[:, CLASS_INTERIOR],
            "p_lgd_one":      proba[:, CLASS_ONE],
            "interior_mu":    mu,
            "phi":            self.phi_,
        })

    def predict_quantile(self, X: np.ndarray, q: float) -> np.ndarray:
        """
        Quantile q of the mixture distribution: a point mass at 0, a beta
        density on (0, 1), and a point mass at 1.

        The mixture CDF is

            F(t) = P0 + P_int * F_beta(t; mu*phi, (1-mu)*phi)   for 0 <= t < 1
            F(1) = 1

        so the quantile is 0 while q <= P0, 1 once q > P0 + P_int, and the
        beta quantile of the rescaled probability (q - P0) / P_int in between.

        Not used for champion selection — that uses the conditional mean.
        This exists for downturn / stressed-LGD work, where the regulatory
        question is a high quantile of the loss distribution rather than its
        average (Basel requires a downturn LGD distinct from the average LGD
        used for ECL — see the README's Known Limitations).
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q}")

        proba = self.predict_proba(X)
        mu = np.clip(self.predict_interior_mean(X), 1e-9, 1 - 1e-9)
        p0, p_int = proba[:, CLASS_ZERO], proba[:, CLASS_INTERIOR]

        out = np.ones(len(mu), dtype=float)          # default: the LGD=1 mass
        out[q <= p0] = 0.0

        in_interior = (q > p0) & (q <= p0 + p_int) & (p_int > 0)
        if in_interior.any():
            scaled = (q - p0[in_interior]) / p_int[in_interior]
            a = mu[in_interior] * self.phi_
            b = (1.0 - mu[in_interior]) * self.phi_
            out[in_interior] = beta_dist.ppf(np.clip(scaled, 0.0, 1.0), a, b)

        return np.clip(out, 0.0, 1.0)

    # -- reporting ----------------------------------------------------------

    def stage1_coefficients(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Stage-1 coefficients on the log-odds scale, per modelled class,
        expressed RELATIVE to the reference class (the lowest-numbered
        modelled class).

        Two sklearn representations have to be handled, and conflating them
        is a live bug rather than a theoretical one — the small-sample
        fallbacks routinely drop a boundary class, which turns stage 1
        binary:

          * Binary (2 modelled classes): coef_ holds ONE row, describing the
            log-odds of classes_[1] against classes_[0]. classes_ still has
            two entries, so indexing coef_ by class position overruns it.
          * Multinomial (3 classes): coef_ holds one row per class under an
            over-parameterised softmax. Raw rows are not directly readable as
            log-odds; subtracting the reference row gives the multinomial
            logit coefficients the reference-class framing implies.

        The reference class itself is emitted with zero coefficients, so the
        table is complete and the baseline is explicit rather than absent.
        """
        if self.stage1_ is None:
            return pd.DataFrame(columns=["class", "feature", "coef",
                                         "reference_class"])

        names = ["intercept"] + list(feature_names)
        classes = [int(c) for c in self.stage1_.classes_]
        reference = classes[0]
        coef = np.atleast_2d(self.stage1_.coef_)
        intercept = np.atleast_1d(self.stage1_.intercept_)

        if coef.shape[0] == 1 and len(classes) == 2:
            # Binary: the single row belongs to the non-reference class.
            relative = {classes[1]: np.concatenate([[intercept[0]], coef[0]])}
        else:
            base = np.concatenate([[intercept[0]], coef[0]])
            relative = {
                cls: np.concatenate([[intercept[j]], coef[j]]) - base
                for j, cls in enumerate(classes)
            }

        rows = []
        for cls in classes:
            values = relative.get(cls, np.zeros(len(names)))
            for name, value in zip(names, values):
                rows.append({
                    "class": CLASS_LABELS[cls],
                    "feature": name,
                    "coef": float(value),
                    "reference_class": CLASS_LABELS[reference],
                })
        return pd.DataFrame(rows)

    def stage2_coefficients(self, feature_names: list[str]) -> pd.DataFrame:
        """Stage-2 beta mean coefficients (logit scale) plus the precision."""
        if self.gamma_ is None:
            return pd.DataFrame([{
                "feature": "constant_interior_mean",
                "coef": float(self.constant_mu_ if self.constant_mu_ is not None else 0.5),
                "note": "stage 2 fell back to a constant mean — see the log",
            }, {"feature": "phi", "coef": float(self.phi_), "note": "method of moments"}])

        names = ["intercept"] + list(feature_names)
        rows = [{"feature": n, "coef": float(c), "note": ""}
                for n, c in zip(names, self.gamma_)]
        rows.append({"feature": "phi", "coef": float(self.phi_),
                     "note": "constant precision, log-parametrised"})
        return pd.DataFrame(rows)


# =============================================================================
# MODEL-SPECIFIC DIAGNOSTICS
# =============================================================================

def stage1_diagnostics(model: TwoStageLGDModel, X: np.ndarray, y: np.ndarray,
                       split: str) -> tuple[dict, pd.DataFrame]:
    """Multiclass log loss and confusion matrix for stage 1 on one split."""
    if len(y) == 0:
        return {}, pd.DataFrame()

    true_classes = assign_lgd_class(y, model.boundary_eps)
    proba = model.predict_proba(X)
    all_classes = [CLASS_ZERO, CLASS_INTERIOR, CLASS_ONE]

    # log_loss needs strictly positive probabilities for the observed labels;
    # a dropped class has probability exactly 0 by design, so clip first.
    safe = np.clip(proba, 1e-12, 1.0)
    safe = safe / safe.sum(axis=1, keepdims=True)
    ll = float(log_loss(true_classes, safe, labels=all_classes))

    predicted = np.asarray(all_classes)[proba.argmax(axis=1)]
    cm = confusion_matrix(true_classes, predicted, labels=all_classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{CLASS_LABELS[c]}" for c in all_classes],
        columns=[f"pred_{CLASS_LABELS[c]}" for c in all_classes],
    )
    log.info("  [Two Stage] Stage 1 %s — multiclass log loss %.4f", split, ll)
    log.info("  [Two Stage] Stage 1 %s confusion matrix:\n%s", split, cm_df.to_string())
    return {"model": "Two Stage", "split": split, "stage1_log_loss": round(ll, 4)}, cm_df


def stage2_diagnostics(model: TwoStageLGDModel, X: np.ndarray, y: np.ndarray,
                       split: str) -> dict:
    """RMSE and MAE for stage 2, on interior observations only."""
    if len(y) == 0:
        return {}
    interior = assign_lgd_class(y, model.boundary_eps) == CLASS_INTERIOR
    if not interior.any():
        log.info("  [Two Stage] Stage 2 %s — no interior observations.", split)
        return {"model": "Two Stage", "split": split, "n_interior": 0}

    mu = model.predict_interior_mean(X[interior])
    rmse = float(np.sqrt(mean_squared_error(y[interior], mu)))
    mae = float(mean_absolute_error(y[interior], mu))
    log.info("  [Two Stage] Stage 2 %s — n_interior=%d  RMSE=%.4f  MAE=%.4f",
             split, int(interior.sum()), rmse, mae)
    return {"model": "Two Stage", "split": split,
            "n_interior": int(interior.sum()),
            "stage2_interior_rmse": round(rmse, 4),
            "stage2_interior_mae": round(mae, 4)}


def point_mass_calibration(y_true: np.ndarray, predictions: dict,
                           two_stage: TwoStageLGDModel | None = None,
                           two_stage_X: np.ndarray | None = None,
                           split: str = "Train",
                           eps: float = BOUNDARY_EPS) -> pd.DataFrame:
    """
    Observed share of LGD=0 and LGD=1 against each model's predicted share.

    For the FRM and the random forest, "predicted share" is the fraction of
    point predictions falling within eps of each boundary — which is normally
    near zero, because a conditional-mean model has no mechanism for placing
    mass on a boundary. That gap IS the comparison: it shows what the
    two-stage model's explicit point masses buy, and it is the reason a
    mean-only LGD model can look well-calibrated on RMSE while being unable
    to reproduce the shape of the loss distribution at all.

    For the two-stage model the predicted share is the mean predicted class
    probability, which is the model's actual statement about the masses.
    """
    if len(y_true) == 0:
        return pd.DataFrame()

    observed_zero = float(np.mean(np.asarray(y_true) <= eps))
    observed_one = float(np.mean(np.asarray(y_true) >= 1.0 - eps))

    rows = []
    for name, preds in predictions.items():
        preds = np.asarray(preds, dtype=float)
        if name == "Two Stage" and two_stage is not None and two_stage_X is not None:
            proba = two_stage.predict_proba(two_stage_X)
            pred_zero = float(proba[:, CLASS_ZERO].mean())
            pred_one = float(proba[:, CLASS_ONE].mean())
            basis = "mean predicted class probability"
        else:
            pred_zero = float(np.mean(preds <= eps))
            pred_one = float(np.mean(preds >= 1.0 - eps))
            basis = "share of point predictions within eps of the boundary"

        rows.append({
            "split": split, "model": name,
            "observed_share_lgd_zero": round(observed_zero, 6),
            "predicted_share_lgd_zero": round(pred_zero, 6),
            "observed_share_lgd_one": round(observed_one, 6),
            "predicted_share_lgd_one": round(pred_one, 6),
            "basis": basis,
        })
    return pd.DataFrame(rows)


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

    Thesis §3.7 — visual calibration check. With the point masses at 0 and 1
    the actual histogram is strongly bimodal; the conditional-mean models
    concentrate in the middle, which is the visual counterpart of the
    point-mass calibration table.
    """
    palette = ["#2563EB", "#D97706", "#059669"]
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


def plot_point_mass_calibration(calib: pd.DataFrame, path: Path) -> None:
    """Observed vs predicted share of LGD=0 and LGD=1, by model."""
    if calib.empty:
        return
    train = calib[calib["split"] == "Train"]
    if train.empty:
        train = calib

    models = train["model"].tolist()
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5 * width, train["observed_share_lgd_zero"] * 100, width,
           label="Observed LGD=0", color="#94A3B8")
    ax.bar(x - 0.5 * width, train["predicted_share_lgd_zero"] * 100, width,
           label="Predicted LGD=0", color="#2563EB")
    ax.bar(x + 0.5 * width, train["observed_share_lgd_one"] * 100, width,
           label="Observed LGD=1", color="#CBD5E1")
    ax.bar(x + 1.5 * width, train["predicted_share_lgd_one"] * 100, width,
           label="Predicted LGD=1", color="#DC2626")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Share of observations (%)", fontsize=11)
    ax.set_title("LGD Point-Mass Calibration — Observed vs Predicted\n"
                 "Conditional-mean models place almost no mass on either boundary",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Point-mass calibration plot → %s", path)


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
    for split_name, df_split in [("Train", train), ("OOS", oos), ("OOT", oot)]:
        if len(df_split) == 0:
            continue
        classes = assign_lgd_class(df_split[TARGET].to_numpy())
        log.info("  %s class counts — %s", split_name, ", ".join(
            f"{CLASS_LABELS[c]}: {int((classes == c).sum())}"
            for c in (CLASS_ZERO, CLASS_INTERIOR, CLASS_ONE)))

    # ── Prepare ───────────────────────────────────────────────────────────
    log.info("")
    log.info("[2/6] Preparing features (train-only encoders, two paths) …")
    data = prepare(train, oos, oot)
    X_tr, X_oo, X_ot = data.X_lin["train"], data.X_lin["oos"], data.X_lin["oot"]
    T_tr, T_oo, T_ot = data.X_tree["train"], data.X_tree["oos"], data.X_tree["oot"]
    y_tr, y_oo, y_ot = data.y["train"], data.y["oos"], data.y["oot"]
    w_tr = data.w_train

    log.info("  Linear path: %s  (%d features, one-hot with dropped reference)",
             X_tr.shape, len(data.linear_features))
    log.info("  Tree path:   %s  (%d features, integer codes)",
             T_tr.shape, len(data.tree_features))
    for col, ref in data.linear_pre.reference_levels_.items():
        n_kept = len(data.linear_pre.kept_levels_.get(col, []))
        log.info("    %-20s reference level '%s'  (%d level(s) kept, rest -> '%s')",
                 col, ref, n_kept, OTHER_LEVEL)
    log.info("  IPCW sample weights: min=%.3f  mean=%.3f  max=%.3f",
             w_tr.min(), w_tr.mean(), w_tr.max())

    all_metrics: list[dict] = []
    train_preds: dict = {}

    def _eval_model(model_obj, model_name: str, matrices: tuple) -> None:
        """Evaluate a fitted model on all splits and collect metrics."""
        X_train, X_oos, X_oot = matrices
        for split_name, (X, y) in [("Train", (X_train, y_tr)),
                                   ("OOS",   (X_oos, y_oo)),
                                   ("OOT",   (X_oot, y_ot))]:
            if len(y) == 0:
                continue
            pred = np.clip(model_obj.predict(X), 0, 1)
            all_metrics.append(evaluate_lgd(split_name, y, pred, model_name))
            if split_name == "Train":
                train_preds[model_name] = pred

    linear_matrices = (X_tr, X_oo, X_ot)
    tree_matrices = (T_tr, T_oo, T_ot)

    # ── Model 1: Fractional Response Model ───────────────────────────────
    log.info("")
    log.info("[3/6] Fractional Response Model (Papke-Wooldridge quasi-binomial GLM) …")
    frm = FractionalResponseModel().fit(X_tr, y_tr, freq_weights=w_tr)
    _eval_model(frm, "FRM", linear_matrices)

    log.info("\n  FRM Coefficients (logit scale%s):",
             ", L2-penalised" if frm.penalised_ else "")
    for fname, coef in zip(data.linear_features, frm.coef_):
        log.info("    %-35s  %+.4f", fname, coef)

    plot_actual_vs_pred(y_tr, np.clip(frm.predict(X_tr), 0, 1),
                        "Fractional Response Model",
                        FIG_DIR / "lgd_frm_actual_vs_pred.png")

    # ── Model 2: Two-Stage Model ─────────────────────────────────────────
    log.info("")
    log.info("[4/6] Two-Stage Model (multinomial logit + beta regression) …")
    two_stage = TwoStageLGDModel().fit(X_tr, y_tr, sample_weight=w_tr)
    _eval_model(two_stage, "Two Stage", linear_matrices)

    stage_diag_rows: list[dict] = []
    for split_name, X, y in [("Train", X_tr, y_tr), ("OOS", X_oo, y_oo),
                             ("OOT", X_ot, y_ot)]:
        if len(y) == 0:
            continue
        s1, _cm = stage1_diagnostics(two_stage, X, y, split_name)
        s2 = stage2_diagnostics(two_stage, X, y, split_name)
        merged = {**s1, **s2}
        if merged:
            stage_diag_rows.append(merged)

    plot_actual_vs_pred(y_tr, np.clip(two_stage.predict(X_tr), 0, 1),
                        "Two-Stage Model",
                        FIG_DIR / "lgd_two_stage_actual_vs_pred.png")

    # ── Model 3: Random Forest ───────────────────────────────────────────
    log.info("")
    log.info("[5/6] Random Forest Regressor (200 trees, max_depth=6) …")
    rf = RandomForestRegressor(**RF_PARAMS).fit(T_tr, y_tr, sample_weight=w_tr)
    _eval_model(rf, "Random Forest", tree_matrices)

    imp_rf = pd.Series(rf.feature_importances_,
                       index=data.tree_features).sort_values(ascending=False)
    log.info("\n  Random Forest Feature Importances (top 10):\n%s",
             imp_rf.head(10).to_string())

    plot_actual_vs_pred(y_tr, np.clip(rf.predict(T_tr), 0, 1), "Random Forest",
                        FIG_DIR / "lgd_rf_actual_vs_pred.png")

    # ── Summary ──────────────────────────────────────────────────────────
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

    # Per-row predictions from all three models.
    model_objects = {"FRM": (frm, linear_matrices),
                     "Two Stage": (two_stage, linear_matrices),
                     "Random Forest": (rf, tree_matrices)}

    pred_rows = []
    component_rows = []
    for idx, (split_label, df_split) in enumerate(
            [("train", train), ("oos", oos), ("oot", oot)]):
        if len(df_split) == 0:
            continue
        y = {"train": y_tr, "oos": y_oo, "oot": y_ot}[split_label]
        row = df_split[["loan_seq_num"]].copy() if "loan_seq_num" in df_split.columns \
              else pd.DataFrame(index=df_split.index)
        row = row.reset_index(drop=True)
        row["split"]      = split_label
        row["actual_lgd"] = y
        for mname, (mobj, matrices) in model_objects.items():
            X_split = matrices[idx]
            row[f"{mname.lower().replace(' ', '_')}_pred"] = np.clip(
                mobj.predict(X_split), 0, 1)
        pred_rows.append(row)

        components = two_stage.predict_components(linear_matrices[idx])
        components.insert(0, "split", split_label)
        if "loan_seq_num" in df_split.columns:
            components.insert(0, "loan_seq_num",
                              df_split["loan_seq_num"].to_numpy())
        components["actual_lgd"] = y
        component_rows.append(components)

    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    if not preds_df.empty:
        preds_df.to_csv(OUT_DIR / "lgd_predictions.csv", index=False)

    metrics_df.to_csv(OUT_DIR / "lgd_metrics.csv", index=False)

    if component_rows:
        pd.concat(component_rows, ignore_index=True).to_csv(
            OUT_DIR / "lgd_two_stage_components.csv", index=False)
        log.info("  Two-stage components → lgd_two_stage_components.csv")

    two_stage.stage1_coefficients(data.linear_features).to_csv(
        OUT_DIR / "lgd_two_stage_stage1_coefficients.csv", index=False)
    two_stage.stage2_coefficients(data.linear_features).to_csv(
        OUT_DIR / "lgd_two_stage_stage2_coefficients.csv", index=False)
    log.info("  Two-stage coefficients → lgd_two_stage_stage{1,2}_coefficients.csv")

    if stage_diag_rows:
        pd.DataFrame(stage_diag_rows).to_csv(
            OUT_DIR / "lgd_two_stage_stage_metrics.csv", index=False)

    # ── Point-mass calibration across all three models ──────────────────
    calib_frames = []
    for split_name, X_split, T_split, y_split in [
        ("Train", X_tr, T_tr, y_tr), ("OOS", X_oo, T_oo, y_oo),
        ("OOT", X_ot, T_ot, y_ot),
    ]:
        if len(y_split) == 0:
            continue
        split_preds = {
            "FRM": np.clip(frm.predict(X_split), 0, 1),
            "Two Stage": np.clip(two_stage.predict(X_split), 0, 1),
            "Random Forest": np.clip(rf.predict(T_split), 0, 1),
        }
        calib_frames.append(point_mass_calibration(
            y_split, split_preds, two_stage=two_stage, two_stage_X=X_split,
            split=split_name))

    calib_df = pd.concat(calib_frames, ignore_index=True) if calib_frames else pd.DataFrame()
    if not calib_df.empty:
        calib_df.to_csv(OUT_DIR / "lgd_point_mass_calibration.csv", index=False)
        log.info("\n  Point-mass calibration (observed vs predicted share):\n%s",
                 calib_df.to_string(index=False))
        plot_point_mass_calibration(calib_df,
                                    FIG_DIR / "lgd_point_mass_calibration.png")

    # ── Champion selection for the macro scenario ECL engine ────────────
    champion_summary = select_champion(metrics_df, preds_df)
    if not champion_summary.empty:
        champion_summary.to_csv(OUT_DIR / "lgd_champion_summary.csv", index=False)
        row = champion_summary.iloc[0]
        log.info(
            "  Champion: %s (%s RMSE=%.4f, n=%d) -> mean predicted LGD=%.4f "
            "saved as the macro scenario ECL anchor (lgd_champion_summary.csv)",
            row["champion_model"], row["anchor_split"], row["rmse"],
            row["n_anchor_obs"], row["anchor_mean_lgd"],
        )
    else:
        log.warning(
            "  No LGD metrics available for champion selection — "
            "lgd_champion_summary.csv not written. "
            "07_macro_scenario_analysis.py will fall back to config.MACRO_LGD_ASSUMPTION."
        )

    log.info("  Files saved to %s", OUT_DIR.resolve())

    if len(train) < 100:
        log.warning(
            "LGD metrics should be interpreted cautiously (small sample: %d rows).  "
            "Download pre-2010 Freddie Mac vintages for robust results.",
            len(train),
        )

    log.info("")
    log.info("=" * 65)
    log.info("Ch.3 complete.")
    log.info("  Next: python 07_macro_scenario_analysis.py  (refreshes the ECL "
             "LGD anchor from this run's champion)")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
