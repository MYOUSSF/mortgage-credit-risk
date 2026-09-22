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


# =============================================================================
# PREPROCESSING — train-only fitting, rare/unseen level grouping
# =============================================================================

def _cat_frame(states, lgd=None, n_per=1, extra_numeric=True):
    """Build a small LGD frame with one categorical and one numeric feature."""
    rows = []
    for state in states:
        for _ in range(n_per):
            rows.append({"property_state": state, "orig_cltv": 80.0})
    df = pd.DataFrame(rows)
    if extra_numeric:
        df["orig_dti"] = 35.0
    df["lgd"] = lgd if lgd is not None else np.linspace(0.1, 0.9, len(df))
    return df


def test_linear_preprocessor_groups_rare_levels_into_other(lgd_models):
    # CA appears 12 times (>= min count 10), TX twice (rare) — TX must be
    # folded into "other" rather than getting its own indicator column, which
    # on a ~150-row regression would identify individual loans.
    train = _cat_frame(["CA"] * 12 + ["TX"] * 2)
    pre = lgd_models.LinearPreprocessor(
        ["property_state", "orig_cltv"], ["property_state"], min_level_count=10
    ).fit(train)

    assert pre.kept_levels_["property_state"] == ["CA"]
    assert all("TX" not in name for name in pre.feature_names_)


def test_linear_preprocessor_maps_unseen_levels_to_other(lgd_models):
    train = _cat_frame(["CA"] * 12 + ["NY"] * 12)
    pre = lgd_models.LinearPreprocessor(
        ["property_state", "orig_cltv"], ["property_state"], min_level_count=10
    ).fit(train)

    # FL was never seen in training; it must not raise and must not silently
    # take on the identity of a real state.
    scoring = _cat_frame(["FL"] * 3)
    X = pre.transform(scoring)

    assert X.shape == (3, len(pre.feature_names_))
    assert np.isfinite(X).all()
    # "other" is the reference level here (it sorts before CA/NY only if
    # present); whichever indicators exist, FL must not activate CA or NY.
    for name in ["property_state[CA]", "property_state[NY]"]:
        if name in pre.feature_names_:
            assert (X[:, pre.feature_names_.index(name)] == 0).all()


def test_linear_preprocessor_is_fitted_on_train_only(lgd_models):
    """
    The previous implementation fitted its encoder on
    pd.concat([train, oos, oot]), leaking the evaluation splits' level sets
    into the training encoding. Levels that appear only outside training must
    not create columns.
    """
    train = _cat_frame(["CA"] * 12)
    oos = _cat_frame(["WA"] * 12)
    pre = lgd_models.LinearPreprocessor(
        ["property_state", "orig_cltv"], ["property_state"], min_level_count=10
    ).fit(train)

    assert all("WA" not in name for name in pre.feature_names_)
    assert pre.transform(oos).shape[1] == len(pre.feature_names_)


def test_linear_path_has_no_integer_coded_categorical_column(lgd_models):
    """
    The core encoding fix: categoricals must reach the linear models as
    one-hot indicators, never as integer codes standing in for a continuous
    slope across alphabetically ordered levels.
    """
    train = _cat_frame(["CA"] * 12 + ["NY"] * 12 + ["TX"] * 12)
    pre = lgd_models.LinearPreprocessor(
        ["property_state", "orig_cltv"], ["property_state"], min_level_count=10
    )
    X = pre.fit_transform(train)

    # No column is named for the raw categorical itself …
    assert "property_state" not in pre.feature_names_
    # … and every categorical-derived column is a 0/1 indicator.
    for i, name in enumerate(pre.feature_names_):
        if name.startswith("property_state["):
            assert set(np.unique(X[:, i])).issubset({0.0, 1.0})


def test_reference_level_is_dropped_exactly_once(lgd_models):
    train = _cat_frame(["CA"] * 12 + ["NY"] * 12 + ["TX"] * 12)
    pre = lgd_models.LinearPreprocessor(
        ["property_state", "orig_cltv"], ["property_state"], min_level_count=10
    ).fit(train)

    ref = pre.reference_levels_["property_state"]
    indicators = [n for n in pre.feature_names_ if n.startswith("property_state[")]
    assert f"property_state[{ref}]" not in indicators
    assert len(indicators) == 2  # 3 levels, one dropped as reference


@pytest.mark.parametrize("path", ["linear", "tree"])
def test_feature_names_match_matrix_width(lgd_models, path):
    train = _cat_frame(["CA"] * 12 + ["NY"] * 12)
    cls = (lgd_models.LinearPreprocessor if path == "linear"
           else lgd_models.TreePreprocessor)
    pre = cls(["property_state", "orig_cltv", "orig_dti"], ["property_state"])
    X = pre.fit_transform(train)
    assert X.shape[1] == len(pre.feature_names_)


def test_all_nan_training_column_is_dropped_from_both_paths(lgd_models):
    """The pre-existing fix: SimpleImputer silently drops all-NaN columns,
    which would desynchronise feature names from matrix columns."""
    train = _cat_frame(["CA"] * 12)
    train["mi_pct"] = np.nan

    for cls in (lgd_models.LinearPreprocessor, lgd_models.TreePreprocessor):
        pre = cls(["property_state", "orig_cltv", "mi_pct"], ["property_state"])
        X = pre.fit_transform(train)
        assert "mi_pct" not in pre.feature_names_
        assert X.shape[1] == len(pre.feature_names_)


def test_tree_preprocessor_maps_unseen_levels_to_dedicated_code(lgd_models):
    train = _cat_frame(["CA"] * 5 + ["NY"] * 5)
    pre = lgd_models.TreePreprocessor(
        ["property_state", "orig_cltv"], ["property_state"]).fit(train)

    X = pre.transform(_cat_frame(["FL"] * 2))
    col = pre.feature_names_.index("property_state")
    assert (X[:, col] == lgd_models.UNSEEN_CODE).all()
    # The reserved code must not collide with any fitted level's code.
    assert lgd_models.UNSEEN_CODE not in pre.level_codes_["property_state"].values()


# =============================================================================
# TWO-STAGE MODEL — class assignment
# =============================================================================

def test_assign_lgd_class_at_and_near_the_boundaries(lgd_models):
    eps = lgd_models.BOUNDARY_EPS
    y = np.array([0.0, eps / 2, eps, 0.3, 0.5, 1 - eps, 1 - eps / 2, 1.0])
    classes = lgd_models.assign_lgd_class(y)

    expected = [
        lgd_models.CLASS_ZERO,      # exactly 0 — full recovery
        lgd_models.CLASS_ZERO,      # within eps of 0
        lgd_models.CLASS_ZERO,      # exactly at eps (<=)
        lgd_models.CLASS_INTERIOR,
        lgd_models.CLASS_INTERIOR,
        lgd_models.CLASS_ONE,       # exactly at 1-eps (>=)
        lgd_models.CLASS_ONE,       # within eps of 1
        lgd_models.CLASS_ONE,       # exactly 1 — total loss
    ]
    assert classes.tolist() == expected


def test_interior_class_is_strictly_inside_the_unit_interval(lgd_models):
    """The beta log-likelihood involves log(y) and log(1-y), so any row the
    interior class admits must be strictly inside (0, 1)."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(20), np.ones(20), rng.uniform(0, 1, 200)])
    interior = y[lgd_models.assign_lgd_class(y) == lgd_models.CLASS_INTERIOR]
    assert (interior > 0).all() and (interior < 1).all()


# =============================================================================
# TWO-STAGE MODEL — composition identities
# =============================================================================

def _two_stage_frame(n=300, seed=0):
    """Synthetic data from a KNOWN two-stage process."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = x.reshape(-1, 1)
    # Known class probabilities (constant, so shares are recoverable).
    classes = rng.choice([0, 1, 2], size=n, p=[0.25, 0.55, 0.20])
    y = np.empty(n)
    y[classes == 0] = 0.0
    y[classes == 2] = 1.0
    interior = classes == 1
    # Known interior mean ~0.4 with covariate dependence.
    mu = 1 / (1 + np.exp(-(-0.4 + 0.8 * x[interior])))
    y[interior] = rng.beta(mu * 12, (1 - mu) * 12)
    return X, y, classes


def test_two_stage_mean_equals_p1_plus_pinterior_times_mu(lgd_models):
    X, y, _ = _two_stage_frame()
    model = lgd_models.TwoStageLGDModel().fit(X, y)

    comp = model.predict_components(X)
    expected = comp["p_lgd_one"] + comp["p_lgd_interior"] * comp["interior_mu"]

    np.testing.assert_allclose(model.predict(X), expected.to_numpy())


def test_two_stage_predictions_stay_within_unit_interval(lgd_models):
    X, y, _ = _two_stage_frame()
    model = lgd_models.TwoStageLGDModel().fit(X, y)
    preds = model.predict(X)
    assert ((preds >= 0) & (preds <= 1)).all()


def test_stage1_probabilities_sum_to_one(lgd_models):
    X, y, _ = _two_stage_frame()
    model = lgd_models.TwoStageLGDModel().fit(X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)
    assert (proba >= 0).all()


def test_two_stage_recovers_known_class_shares_and_interior_mean(lgd_models):
    X, y, classes = _two_stage_frame(n=4_000, seed=3)
    model = lgd_models.TwoStageLGDModel().fit(X, y)

    proba = model.predict_proba(X)
    observed_shares = [float((classes == c).mean()) for c in (0, 1, 2)]
    predicted_shares = proba.mean(axis=0)
    np.testing.assert_allclose(predicted_shares, observed_shares, atol=0.03)

    interior = classes == 1
    assert model.predict_interior_mean(X[interior]).mean() == pytest.approx(
        y[interior].mean(), abs=0.05)


# =============================================================================
# WEIGHTED BETA LIKELIHOOD
# =============================================================================

def _beta_sample(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    mu = 1 / (1 + np.exp(-(X @ np.array([0.2, 0.6]))))
    return X, rng.beta(mu * 10, (1 - mu) * 10)


def test_weighted_beta_nll_reduces_to_unweighted_when_weights_are_one(lgd_models):
    X, y = _beta_sample()
    params = np.array([0.1, 0.4, np.log(8.0)])

    weighted, gw = lgd_models.weighted_beta_nll(params, X, y, np.ones(len(y)))
    # The unweighted likelihood is the plain sum of per-observation terms.
    from scipy.special import gammaln, expit
    mu = expit(X @ params[:-1]); phi = np.exp(params[-1])
    manual = -np.sum(gammaln(phi) - gammaln(mu * phi) - gammaln((1 - mu) * phi)
                     + (mu * phi - 1) * np.log(y)
                     + ((1 - mu) * phi - 1) * np.log1p(-y))
    assert weighted == pytest.approx(manual)


def test_weighted_beta_nll_doubling_all_weights_doubles_the_objective(lgd_models):
    X, y = _beta_sample()
    params = np.array([0.1, 0.4, np.log(8.0)])
    single, _ = lgd_models.weighted_beta_nll(params, X, y, np.ones(len(y)))
    double, _ = lgd_models.weighted_beta_nll(params, X, y, 2 * np.ones(len(y)))
    assert double == pytest.approx(2 * single)


def test_weighted_beta_nll_gradient_matches_numerical_derivative(lgd_models):
    """An analytic gradient that disagrees with the objective would make the
    optimiser converge to the wrong point silently."""
    from scipy.optimize import check_grad
    X, y = _beta_sample()
    w = np.linspace(0.5, 2.0, len(y))
    p0 = np.array([0.1, 0.3, np.log(6.0)])

    err = check_grad(lambda p: lgd_models.weighted_beta_nll(p, X, y, w)[0],
                     lambda p: lgd_models.weighted_beta_nll(p, X, y, w)[1], p0)
    assert err < 1e-4


def test_beta_weights_move_the_fit_toward_the_upweighted_observations(lgd_models):
    """
    Weighting must actually change the estimate — this is the property
    statsmodels' BetaModel silently fails to provide (it accepts a `weights`
    kwarg, warns, and returns identical unweighted parameters).
    """
    rng = np.random.default_rng(7)
    n = 400
    X = np.ones((n, 1))
    low = rng.beta(2, 8, n // 2)     # mean ~0.2
    high = rng.beta(8, 2, n // 2)    # mean ~0.8
    y = np.concatenate([low, high])

    w_low = np.concatenate([np.full(n // 2, 5.0), np.full(n // 2, 0.2)])
    w_high = np.concatenate([np.full(n // 2, 0.2), np.full(n // 2, 5.0)])

    def _fit(w):
        model = lgd_models.TwoStageLGDModel(min_class_obs=1)
        model._fit_stage2(np.empty((n, 0)), y, w)
        return float(model.predict_interior_mean(np.empty((n, 0))).mean())

    mean_when_low_upweighted = _fit(w_low)
    mean_when_high_upweighted = _fit(w_high)

    assert mean_when_low_upweighted < mean_when_high_upweighted


# =============================================================================
# TWO-STAGE MODEL — small-sample fallbacks
# =============================================================================

def test_fallback_when_training_has_no_lgd_one_observations(lgd_models):
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(size=(n, 1))
    y = np.concatenate([np.zeros(60), rng.uniform(0.05, 0.95, n - 60)])

    model = lgd_models.TwoStageLGDModel().fit(X, y)
    proba = model.predict_proba(X)

    assert lgd_models.CLASS_ONE in model.dropped_classes_
    assert (proba[:, lgd_models.CLASS_ONE] == 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)
    assert np.isfinite(model.predict(X)).all()


def test_fallback_when_training_has_no_lgd_zero_observations(lgd_models):
    rng = np.random.default_rng(2)
    n = 200
    X = rng.normal(size=(n, 1))
    y = np.concatenate([np.ones(60), rng.uniform(0.05, 0.95, n - 60)])

    model = lgd_models.TwoStageLGDModel().fit(X, y)
    proba = model.predict_proba(X)

    assert lgd_models.CLASS_ZERO in model.dropped_classes_
    assert (proba[:, lgd_models.CLASS_ZERO] == 0).all()
    assert np.isfinite(model.predict(X)).all()


def test_fallback_when_interior_rows_are_below_the_minimum(lgd_models):
    rng = np.random.default_rng(3)
    # Only 3 interior observations, below the default minimum of 10.
    y = np.concatenate([np.zeros(40), np.ones(40), np.array([0.3, 0.5, 0.7])])
    X = rng.normal(size=(len(y), 2))

    model = lgd_models.TwoStageLGDModel().fit(X, y)

    assert not model.stage2_converged_
    assert model.constant_mu_ is not None
    # Stage 2 degenerates to a constant, so mu is the same for every row.
    mu = model.predict_interior_mean(X)
    assert np.allclose(mu, mu[0])
    assert np.isfinite(model.predict(X)).all()


def test_fallback_when_only_one_class_is_present(lgd_models):
    """Degenerate but possible on a tiny sample: every resolved default is a
    total loss. The model must still fit and predict."""
    rng = np.random.default_rng(4)
    y = np.ones(50)
    X = rng.normal(size=(50, 2))

    model = lgd_models.TwoStageLGDModel().fit(X, y)
    proba = model.predict_proba(X)

    assert model.single_class_ == lgd_models.CLASS_ONE
    assert (proba[:, lgd_models.CLASS_ONE] == 1.0).all()
    np.testing.assert_allclose(model.predict(X), 1.0)


def test_predict_never_raises_for_a_class_missing_in_training(lgd_models):
    rng = np.random.default_rng(5)
    y = np.concatenate([np.zeros(50), rng.uniform(0.1, 0.9, 50)])  # no LGD=1
    X_train = rng.normal(size=(len(y), 2))
    model = lgd_models.TwoStageLGDModel().fit(X_train, y)

    # Score rows unlike anything in training — must not raise.
    X_new = rng.normal(size=(10, 2)) * 50
    preds = model.predict(X_new)
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()
    assert ((preds >= 0) & (preds <= 1)).all()


# =============================================================================
# TWO-STAGE MODEL — quantiles
# =============================================================================

def test_predict_quantile_is_monotone_in_q_and_within_unit_interval(lgd_models):
    X, y, _ = _two_stage_frame(n=600, seed=11)
    model = lgd_models.TwoStageLGDModel().fit(X, y)

    quantiles = [model.predict_quantile(X, q)
                 for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]]
    stacked = np.column_stack(quantiles)

    assert ((stacked >= 0) & (stacked <= 1)).all()
    assert (np.diff(stacked, axis=1) >= -1e-9).all()


def test_predict_quantile_returns_the_point_masses_at_the_extremes(lgd_models):
    X, y, _ = _two_stage_frame(n=600, seed=13)
    model = lgd_models.TwoStageLGDModel().fit(X, y)

    proba = model.predict_proba(X)
    # A quantile below P(LGD=0) must sit on the lower point mass.
    q_low = float(proba[:, lgd_models.CLASS_ZERO].min()) * 0.5
    assert (model.predict_quantile(X, q_low) == 0.0).all()
    # A quantile above P(LGD=0) + P(interior) must sit on the upper mass.
    assert (model.predict_quantile(X, 1.0) == 1.0).all()


def test_predict_quantile_rejects_q_outside_unit_interval(lgd_models):
    X, y, _ = _two_stage_frame(n=100)
    model = lgd_models.TwoStageLGDModel().fit(X, y)
    with pytest.raises(ValueError):
        model.predict_quantile(X, 1.5)


# =============================================================================
# SUITE COMPOSITION — three models, no spline or XGBoost
# =============================================================================

def test_select_champion_works_with_the_three_current_model_names(lgd_models):
    metrics_df = pd.DataFrame([
        _metrics_row("FRM", "OOS", 0.22),
        _metrics_row("Two Stage", "OOS", 0.18),      # lowest OOS RMSE
        _metrics_row("Random Forest", "OOS", 0.20),
    ])
    preds_df = pd.DataFrame({
        "split": ["oos", "oos"],
        "frm_pred": [0.30, 0.32],
        "two_stage_pred": [0.41, 0.43],
        "random_forest_pred": [0.25, 0.27],
    })

    result = lgd_models.select_champion(metrics_df, preds_df)
    row = result.iloc[0]

    assert row["champion_model"] == "Two Stage"
    # The derived column name must match the name.lower().replace(' ','_')
    # rule select_champion() applies — "Two Stage" -> two_stage_pred.
    assert row["anchor_mean_lgd"] == pytest.approx(np.mean([0.41, 0.43]))


def test_spline_and_xgboost_models_are_gone(lgd_models):
    for removed in ["build_spline_model", "ClippedSpline", "XGB_PARAMS",
                    "XGB_AVAILABLE", "_encode_categoricals"]:
        assert not hasattr(lgd_models, removed), f"{removed} should be removed"


# =============================================================================
# FRM — L2 fallback
# =============================================================================

def test_frm_falls_back_to_l2_on_perfect_separation(lgd_models):
    """
    A one-hot column that perfectly separates the outcome sends the
    unpenalised MLE to infinity — the realistic failure mode on ~150 rows
    with property_state indicators. The fallback must engage rather than
    propagating an extreme coefficient into the ECL anchor.
    """
    n = 60
    separator = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    X = separator.reshape(-1, 1)
    y = separator.copy()  # y is exactly the separating column

    frm = lgd_models.FractionalResponseModel().fit(X, y)
    preds = frm.predict(X)

    assert np.isfinite(preds).all()
    assert ((preds >= 0) & (preds <= 1)).all()


def test_frm_unpenalised_fit_is_preferred_when_it_converges(lgd_models):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(300, 2))
    y = rng.uniform(0.1, 0.9, 300)

    frm = lgd_models.FractionalResponseModel().fit(X, y)

    assert frm.converged_ is True
    assert frm.penalised_ is False


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def test_point_mass_calibration_reports_observed_and_predicted_shares(lgd_models):
    y = np.concatenate([np.zeros(20), np.ones(30), np.full(50, 0.5)])
    preds = {"FRM": np.full(100, 0.5)}  # a mean-only model: no mass anywhere

    calib = lgd_models.point_mass_calibration(y, preds, split="Train")
    row = calib.iloc[0]

    assert row["observed_share_lgd_zero"] == pytest.approx(0.20)
    assert row["observed_share_lgd_one"] == pytest.approx(0.30)
    # The point of the comparison: a conditional-mean model places no mass
    # on either boundary even when 50% of observations sit there.
    assert row["predicted_share_lgd_zero"] == pytest.approx(0.0)
    assert row["predicted_share_lgd_one"] == pytest.approx(0.0)


def test_stage_diagnostics_run_on_a_fitted_two_stage_model(lgd_models):
    X, y, _ = _two_stage_frame(n=400, seed=17)
    model = lgd_models.TwoStageLGDModel().fit(X, y)

    s1, cm = lgd_models.stage1_diagnostics(model, X, y, "Train")
    s2 = lgd_models.stage2_diagnostics(model, X, y, "Train")

    assert 0 <= s1["stage1_log_loss"] < 10
    assert cm.shape == (3, 3)
    assert cm.to_numpy().sum() == len(y)
    assert s2["n_interior"] > 0
    assert s2["stage2_interior_rmse"] >= 0


def test_stage1_coefficients_handle_a_binary_stage1(lgd_models):
    """
    Regression test. When a fallback drops a boundary class, stage 1 becomes
    binary, and sklearn then stores ONE coefficient row while classes_ still
    has two entries — indexing coef_ by class position raised IndexError and
    crashed the whole script at the CSV-writing step, after every model had
    already been fitted.
    """
    rng = np.random.default_rng(21)
    # No LGD=1 observations at all -> that class is dropped -> binary stage 1.
    y = np.concatenate([np.zeros(40), rng.uniform(0.1, 0.9, 40)])
    X = rng.normal(size=(len(y), 3))

    model = lgd_models.TwoStageLGDModel().fit(X, y)
    table = model.stage1_coefficients(["f0", "f1", "f2"])

    assert not table.empty
    assert set(table["feature"]) == {"intercept", "f0", "f1", "f2"}
    # The reference class is present with zero coefficients and is labelled.
    ref = table["reference_class"].iloc[0]
    ref_rows = table[table["class"] == ref]
    assert (ref_rows["coef"] == 0).all()
    assert np.isfinite(table["coef"]).all()


def test_stage1_coefficients_are_relative_to_the_reference_class(lgd_models):
    X, y, _ = _two_stage_frame(n=600, seed=23)
    model = lgd_models.TwoStageLGDModel().fit(X, y)
    table = model.stage1_coefficients(["f0"])

    # All three classes represented, reference row identically zero.
    assert table["class"].nunique() == 3
    ref = table["reference_class"].iloc[0]
    assert (table[table["class"] == ref]["coef"] == 0).all()
