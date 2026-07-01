"""
Tests for 04_lgd_models.py's champion selection — the anchor value
07_macro_scenario_analysis.py uses in place of a fixed LGD assumption.
"""
import numpy as np
import pandas as pd
import pytest


def _metrics_row(model, split, rmse):
    return {"model": model, "split": split, "n": 100,
            "rmse": rmse, "mae": rmse * 0.8, "r2": 0.5, "bias": 0.01}


def test_select_champion_picks_lowest_oos_rmse(lgd_models):
    metrics_df = pd.DataFrame([
        _metrics_row("FRM", "Train", 0.20),
        _metrics_row("FRM", "OOS", 0.22),
        _metrics_row("XGBoost", "Train", 0.10),
        _metrics_row("XGBoost", "OOS", 0.15),   # lower OOS RMSE than FRM
    ])
    preds_df = pd.DataFrame({
        "split":         ["oos", "oos", "oos"],
        "frm_pred":      [0.30, 0.32, 0.34],
        "xgboost_pred":  [0.40, 0.42, 0.44],
    })

    result = lgd_models.select_champion(metrics_df, preds_df)

    assert not result.empty
    row = result.iloc[0]
    assert row["champion_model"] == "XGBoost"
    assert row["anchor_split"] == "OOS"
    assert row["anchor_mean_lgd"] == pytest.approx(np.mean([0.40, 0.42, 0.44]))
    assert row["n_anchor_obs"] == 3


def test_select_champion_falls_back_to_train_when_oos_is_empty(lgd_models):
    metrics_df = pd.DataFrame([
        _metrics_row("FRM", "Train", 0.20),
        _metrics_row("Random Forest", "Train", 0.18),
    ])
    preds_df = pd.DataFrame({
        "split":            ["train", "train"],
        "frm_pred":         [0.30, 0.32],
        "random_forest_pred": [0.25, 0.27],
    })

    result = lgd_models.select_champion(metrics_df, preds_df)

    row = result.iloc[0]
    assert row["champion_model"] == "Random Forest"
    assert row["anchor_split"] == "Train"
    assert row["anchor_mean_lgd"] == pytest.approx(np.mean([0.25, 0.27]))


def test_select_champion_returns_empty_when_no_metrics(lgd_models):
    result = lgd_models.select_champion(pd.DataFrame(), pd.DataFrame())
    assert result.empty


def test_select_champion_handles_missing_prediction_column_gracefully(lgd_models):
    # preds_df doesn't have a column for the champion model — should not
    # raise, and should report NaN rather than crash the pipeline.
    metrics_df = pd.DataFrame([_metrics_row("FRM", "OOS", 0.20)])
    preds_df = pd.DataFrame({"split": ["oos"], "some_other_pred": [0.5]})

    result = lgd_models.select_champion(metrics_df, preds_df)

    assert not result.empty
    assert np.isnan(result.iloc[0]["anchor_mean_lgd"])


# ── FractionalResponseModel: Papke-Wooldridge quasi-binomial GLM ──────────────

def test_frm_fits_boundary_lgd_values_without_clipping(lgd_models):
    # LGD of exactly 0% or 100% loss is common and legitimate. The old
    # OLS-on-logit implementation needed an epsilon-clip to avoid
    # log(0/1); the GLM quasi-likelihood is well-defined at these values
    # directly, so fitting must succeed with no NaNs/errors.
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])

    frm = lgd_models.FractionalResponseModel().fit(X, y)
    preds = frm.predict(X)

    assert np.isfinite(preds).all()
    assert (preds >= 0).all() and (preds <= 1).all()


def test_frm_coef_and_intercept_align_with_feature_count(lgd_models):
    rng = np.random.default_rng(1)
    n, k = 150, 3
    X = rng.normal(size=(n, k))
    y = rng.uniform(0, 1, n)

    frm = lgd_models.FractionalResponseModel().fit(X, y)

    assert frm.coef_.shape == (k,)
    assert np.isfinite(frm.intercept_)


def test_frm_accepts_freq_weights(lgd_models):
    rng = np.random.default_rng(2)
    n = 200
    X = rng.normal(size=(n, 2))
    y = rng.uniform(0, 1, n)
    w = rng.uniform(0.5, 2.0, n)

    frm = lgd_models.FractionalResponseModel().fit(X, y, freq_weights=w)
    preds = frm.predict(X)
    assert np.isfinite(preds).all()
