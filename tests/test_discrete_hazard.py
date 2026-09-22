"""
Tests for 11_discrete_hazard.py — the discrete-time survival framework that
replaces 06_survival_analysis.py's Cox model.

The pieces covered here are the ones where a silent bug would be both easy to
introduce and expensive to miss:

  * the target — an event placed on the wrong loan-month row, or two events
    for one loan, corrupts the likelihood itself, and the symptom is a model
    that trains fine and is quietly wrong;
  * the King & Zeng prior correction — get it backwards and every PD, ECL and
    capital number in the pipeline is off by a constant factor of ~20 while
    every AUROC stays identical, because the correction is monotone;
  * the conditional horizon PD — the specific defect the Cox model had, where
    a loan is charged again for default risk it has already survived;
  * the exclusion of internal (delinquency) covariates, which is a modelling
    decision the docstring argues for and which therefore ought to be pinned.

Everything runs on small synthetic DataFrames — no Freddie Mac data required.
"""
import numpy as np
import pandas as pd
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _panel(rows):
    """Build a loan-month panel frame from (loan, report_date, default_date)."""
    return pd.DataFrame(rows).assign(
        report_date=lambda d: pd.to_datetime(d["report_date"]),
        default_date=lambda d: pd.to_datetime(d["default_date"]),
    )


def _monthly_rows(loan, start, n, default_date=None, start_age=0):
    dates = pd.date_range(start, periods=n, freq="MS")
    return [{"loan_seq_num": loan, "report_date": d, "default_date": default_date,
             "loan_age": start_age + i} for i, d in enumerate(dates)]


class _ConstantHazardModel:
    """
    Stub implementing the shared HazardModel interface with a constant
    hazard, so the horizon-PD tests exercise compute_conditional_horizon_pd()
    itself rather than either fitted model.
    """

    def __init__(self, hazard):
        self.hazard = hazard
        self.seen_ages = []

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        return self

    def predict_hazard(self, X):
        self.seen_ages.append(np.asarray(X["loan_age"]).copy())
        return np.full(len(X), self.hazard, dtype=float)


class _MacroSensitiveModel:
    """Stub whose hazard depends only on ur_3m_lag, for the TTC test."""

    def __init__(self):
        self.seen = []

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        return self

    def predict_hazard(self, X):
        self.seen.append(X.copy())
        return np.clip(np.asarray(X["ur_3m_lag"], dtype=float) / 1000.0, 0, 1)


# ── build_discrete_target: one event, on the correct row ────────────────────

def test_one_event_per_defaulter_on_the_row_before_default(discrete_hazard):
    default_date = pd.Timestamp("2015-06-01")
    panel = _panel(_monthly_rows("L1", "2015-01-01", 5, default_date))

    out = discrete_hazard.build_discrete_target(panel)
    events = out[out[discrete_hazard.TARGET_HAZARD] == 1]

    assert len(events) == 1
    # 2015-05-01 is the only row within 45 days before the default date.
    assert events["report_date"].iloc[0] == pd.Timestamp("2015-05-01")


def test_censored_loan_contributes_no_event(discrete_hazard):
    panel = _panel(_monthly_rows("L2", "2015-01-01", 6, default_date=pd.NaT))
    out = discrete_hazard.build_discrete_target(panel)
    assert out[discrete_hazard.TARGET_HAZARD].sum() == 0
    assert len(out) == 6  # every row is retained as a censored observation


def test_rows_further_than_one_period_from_default_are_not_events(discrete_hazard):
    default_date = pd.Timestamp("2015-06-01")
    panel = _panel(_monthly_rows("L1", "2015-01-01", 5, default_date))
    out = discrete_hazard.build_discrete_target(panel)

    early = out[out["report_date"] <= pd.Timestamp("2015-04-01")]
    # These loan-months were at risk and did not default in the next period,
    # even though the loan eventually defaults — they are y=0, not dropped.
    assert (early[discrete_hazard.TARGET_HAZARD] == 0).all()
    assert len(early) == 4


def test_loan_split_across_train_and_oot_gets_exactly_one_event(discrete_hazard):
    """
    split_pd() cuts on report_date, so a defaulting loan's rows can land in
    both pd_train and pd_oot. The event must be placed once, on the row
    before default, wherever that row ended up — and the other file's rows
    must carry no event. This is the failure mode that persisting
    default_date in 01_data_preprocessing.py exists to prevent: a per-file
    "last row per loan" rule would mark a spurious event on the last train
    row.
    """
    default_date = pd.Timestamp("2017-11-01")
    rows = _monthly_rows("L1", "2017-01-01", 10, default_date)
    panel = _panel(rows)

    cutoff = pd.Timestamp("2017-06-01")
    train = panel[panel["report_date"] < cutoff].copy()
    oot = panel[panel["report_date"] >= cutoff].copy()
    assert len(train) and len(oot)  # the loan genuinely straddles the split

    train_out = discrete_hazard.build_discrete_target(train)
    oot_out = discrete_hazard.build_discrete_target(oot)

    assert train_out[discrete_hazard.TARGET_HAZARD].sum() == 0
    assert oot_out[discrete_hazard.TARGET_HAZARD].sum() == 1
    event_row = oot_out[oot_out[discrete_hazard.TARGET_HAZARD] == 1]
    assert event_row["report_date"].iloc[0] == pd.Timestamp("2017-10-01")

    combined = pd.concat([train_out, oot_out], ignore_index=True)
    assert len(discrete_hazard.loans_with_multiple_events(combined)) == 0


def test_no_loan_carries_more_than_one_event(discrete_hazard):
    default_date = pd.Timestamp("2015-06-01")
    panel = _panel(
        _monthly_rows("L1", "2015-01-01", 5, default_date)
        + _monthly_rows("L2", "2015-01-01", 5, pd.NaT)
    )
    out = discrete_hazard.build_discrete_target(panel)
    assert len(discrete_hazard.loans_with_multiple_events(out)) == 0


def test_build_discrete_target_requires_default_date(discrete_hazard):
    panel = pd.DataFrame({
        "loan_seq_num": ["L1"],
        "report_date": pd.to_datetime(["2015-01-01"]),
    })
    with pytest.raises(KeyError, match="default_date"):
        discrete_hazard.build_discrete_target(panel)


# ── no post-default rows reach the model ────────────────────────────────────

def test_post_default_rows_are_dropped(discrete_hazard):
    default_date = pd.Timestamp("2015-04-01")
    # Rows deliberately extending past the default date, as a panel built by
    # a future edit to 01 that lost its leakage guard would contain.
    panel = _panel(_monthly_rows("L1", "2015-01-01", 6, default_date))

    out = discrete_hazard.drop_post_default_rows(panel)

    assert (out["report_date"] < default_date).all()
    assert len(out) == 3  # Jan, Feb, Mar


def test_drop_post_default_rows_keeps_every_censored_row(discrete_hazard):
    panel = _panel(_monthly_rows("L2", "2015-01-01", 4, pd.NaT))
    assert len(discrete_hazard.drop_post_default_rows(panel)) == 4


# ── King & Zeng prior correction ────────────────────────────────────────────

def test_prior_correction_is_a_log_rate_shift_on_the_logit_scale(discrete_hazard):
    logits = np.array([-6.0, -3.0, 0.0])
    rate = 0.05
    out = discrete_hazard.king_zeng_correct_logit(logits, rate)
    np.testing.assert_allclose(out, logits + np.log(rate))


def test_prior_correction_recovers_the_true_base_rate(discrete_hazard):
    """
    Generate a population with a KNOWN constant hazard, apply the same
    case-control subsampling the script uses, fit the intercept on the
    subsample, correct it, and check the corrected probability recovers the
    population base rate rather than the (much higher) sampled one.
    """
    rng = np.random.default_rng(0)
    n = 400_000
    true_hazard = 0.004

    y = rng.binomial(1, true_hazard, size=n)
    panel = pd.DataFrame({
        "loan_seq_num": np.arange(n).astype(str),
        discrete_hazard.TARGET_HAZARD: y.astype(np.int8),
    })

    sampled, realised_rate = discrete_hazard.subsample_case_control(
        panel, rate=0.05, seed=1,
    )

    sampled_rate = sampled[discrete_hazard.TARGET_HAZARD].mean()
    # Sanity: subsampling really did inflate the event share it has to undo.
    assert sampled_rate > 10 * true_hazard

    corrected = discrete_hazard.apply_prior_correction(
        np.array([sampled_rate]), realised_rate,
    )[0]

    assert corrected == pytest.approx(true_hazard, rel=0.05)


def test_subsample_keeps_every_event_and_thins_non_events(discrete_hazard):
    rng = np.random.default_rng(3)
    n = 50_000
    y = rng.binomial(1, 0.01, size=n)
    panel = pd.DataFrame({
        "loan_seq_num": np.arange(n).astype(str),
        discrete_hazard.TARGET_HAZARD: y.astype(np.int8),
    })

    sampled, realised = discrete_hazard.subsample_case_control(
        panel, rate=0.10, seed=7, chunk_size=1_000,
    )

    assert sampled[discrete_hazard.TARGET_HAZARD].sum() == y.sum()
    assert realised == pytest.approx(0.10, abs=0.01)
    n_controls_kept = (sampled[discrete_hazard.TARGET_HAZARD] == 0).sum()
    assert n_controls_kept == pytest.approx(0.10 * (n - y.sum()), rel=0.1)


def test_subsample_rate_of_one_is_a_no_op(discrete_hazard):
    panel = pd.DataFrame({
        "loan_seq_num": ["a", "b", "c"],
        discrete_hazard.TARGET_HAZARD: np.array([0, 1, 0], dtype=np.int8),
    })
    sampled, rate = discrete_hazard.subsample_case_control(panel, rate=1.0)
    assert len(sampled) == 3
    assert rate == 1.0
    # A rate of 1 must leave the hazard untouched, not shift it by log(1)=0
    # only by accident — check the probability path too.
    np.testing.assert_allclose(
        discrete_hazard.apply_prior_correction(np.array([0.2]), 1.0), [0.2]
    )


# ── conditional horizon PD ──────────────────────────────────────────────────

def _loans(ages, **extra):
    frame = pd.DataFrame({"loan_age": np.asarray(ages, dtype=float)})
    for key, value in extra.items():
        frame[key] = value
    return frame


def test_constant_hazard_gives_the_closed_form_horizon_pd(discrete_hazard):
    h = 0.01
    model = _ConstantHazardModel(h)
    loans = _loans([0.0, 24.0])

    out = discrete_hazard.compute_conditional_horizon_pd(
        model, loans, macro_paths={}, horizons=[12, 24],
    )

    np.testing.assert_allclose(out["pd_12m"].to_numpy(), 1 - (1 - h) ** 12)
    np.testing.assert_allclose(out["pd_24m"].to_numpy(), 1 - (1 - h) ** 24)


def test_horizon_pd_increases_with_horizon_and_stays_in_unit_interval(discrete_hazard):
    model = _ConstantHazardModel(0.02)
    loans = _loans([5.0, 60.0, 120.0])

    out = discrete_hazard.compute_conditional_horizon_pd(
        model, loans, macro_paths={}, horizons=[12, 24, 36, 60],
    )

    values = out[["pd_12m", "pd_24m", "pd_36m", "pd_60m"]].to_numpy()
    assert (np.diff(values, axis=1) > 0).all()
    assert ((values >= 0) & (values <= 1)).all()


def test_horizon_pd_is_conditional_on_the_starting_age(discrete_hazard):
    """
    The defect this whole script exists to fix: the PD handed to a loan must
    depend on the age it has already reached, because the hazard varies with
    age. A model whose hazard rises with age must give a seasoned loan a
    higher 12-month PD than a brand-new one.
    """
    class _AgeRisingModel:
        def fit(self, *a, **k):
            return self

        def predict_hazard(self, X):
            return np.clip(np.asarray(X["loan_age"], dtype=float) / 10_000.0, 0, 1)

    out = discrete_hazard.compute_conditional_horizon_pd(
        _AgeRisingModel(), _loans([0.0, 100.0]), macro_paths={}, horizons=[12],
    )
    young, seasoned = out["pd_12m"].to_numpy()
    assert seasoned > young


def test_horizon_pd_advances_loan_age_deterministically(discrete_hazard):
    model = _ConstantHazardModel(0.001)
    discrete_hazard.compute_conditional_horizon_pd(
        model, _loans([7.0]), macro_paths={}, horizons=[3],
    )
    # Ages seen by the model are a+1, a+2, a+3 — never a itself.
    assert [float(a[0]) for a in model.seen_ages] == [8.0, 9.0, 10.0]


def test_horizon_pd_chunking_does_not_change_the_answer(discrete_hazard):
    model = _ConstantHazardModel(0.005)
    loans = _loans(np.arange(0, 25, dtype=float))

    whole = discrete_hazard.compute_conditional_horizon_pd(
        model, loans, macro_paths={}, horizons=[12], chunk_size=10_000)
    chunked = discrete_hazard.compute_conditional_horizon_pd(
        model, loans, macro_paths={}, horizons=[12], chunk_size=4)

    np.testing.assert_allclose(whole["pd_12m"].to_numpy(),
                               chunked["pd_12m"].to_numpy())


def test_lifetime_pd_is_capped_at_remaining_months(discrete_hazard):
    h = 0.01
    model = _ConstantHazardModel(h)
    loans = _loans([0.0, 0.0])
    lifetime = np.array([6, 18])

    out = discrete_hazard.compute_conditional_horizon_pd(
        model, loans, macro_paths={}, horizons=[12], lifetime_months=lifetime,
    )

    np.testing.assert_allclose(out["pd_lifetime"].to_numpy(),
                               [1 - (1 - h) ** 6, 1 - (1 - h) ** 18])


def test_horizon_pd_accepts_a_per_period_macro_vector(discrete_hazard):
    """
    A length-h macro path is the shape an IFRS 9 scenario projection would
    supply — the reason compute_conditional_horizon_pd() takes an explicit
    path rather than deriving one internally.
    """
    model = _MacroSensitiveModel()
    loans = _loans([0.0], ur_3m_lag=4.0)
    path = np.array([1.0, 2.0, 3.0])

    discrete_hazard.compute_conditional_horizon_pd(
        model, loans, macro_paths={"ur_3m_lag": path}, horizons=[3],
    )

    seen = [float(frame["ur_3m_lag"].iloc[0]) for frame in model.seen]
    assert seen == [1.0, 2.0, 3.0]


def test_horizon_pd_accepts_a_per_loan_per_period_macro_array(discrete_hazard):
    model = _MacroSensitiveModel()
    loans = _loans([0.0, 0.0], ur_3m_lag=4.0)
    path = np.array([[1.0, 2.0], [8.0, 9.0]])

    discrete_hazard.compute_conditional_horizon_pd(
        model, loans, macro_paths={"ur_3m_lag": path}, horizons=[2],
    )

    assert model.seen[0]["ur_3m_lag"].tolist() == [1.0, 8.0]
    assert model.seen[1]["ur_3m_lag"].tolist() == [2.0, 9.0]


def test_empty_population_returns_the_expected_schema(discrete_hazard):
    out = discrete_hazard.compute_conditional_horizon_pd(
        _ConstantHazardModel(0.01), _loans([]), macro_paths={}, horizons=[12, 24],
    )
    assert out.empty
    assert list(out.columns) == ["pd_12m", "pd_24m"]


# ── PIT vs TTC macro paths ──────────────────────────────────────────────────

def test_ttc_replaces_macro_with_long_run_mean_leaving_others_untouched(discrete_hazard):
    """
    TTC must differ from PIT in exactly one respect: the macro covariates are
    pinned at their long-run training mean. Every other loan characteristic
    stays at its actual value — that is what makes it "this loan at a typical
    point in the cycle" rather than "an average loan".
    """
    model = _MacroSensitiveModel()
    loans = pd.DataFrame({
        "loan_age": [10.0, 10.0],
        "ur_3m_lag": [9.0, 2.0],
        "hpi_change": [0.7, 1.4],
        "credit_score": [640.0, 780.0],
    })
    long_run = {"ur_3m_lag": 5.5, "hpi_change": 1.0}

    ttc_paths = discrete_hazard.build_ttc_macro_paths(long_run)
    discrete_hazard.compute_conditional_horizon_pd(
        model, loans, macro_paths=ttc_paths, horizons=[1], prefix="ttc_",
    )

    seen = model.seen[0]
    assert (seen["ur_3m_lag"] == 5.5).all()
    assert (seen["hpi_change"] == 1.0).all()
    # Non-macro covariates are real loan characteristics — left alone.
    assert seen["credit_score"].tolist() == [640.0, 780.0]


def test_pit_holds_each_loans_own_macro_state_flat(discrete_hazard):
    model = _MacroSensitiveModel()
    loans = pd.DataFrame({
        "loan_age": [10.0, 10.0],
        "ur_3m_lag": [9.0, 2.0],
        "hpi_change": [0.7, 1.4],
    })

    pit_paths = discrete_hazard.build_pit_macro_paths(loans)
    discrete_hazard.compute_conditional_horizon_pd(
        model, loans, macro_paths=pit_paths, horizons=[3],
    )

    # Each loan keeps its OWN value, and it does not drift across periods.
    for frame in model.seen:
        assert frame["ur_3m_lag"].tolist() == [9.0, 2.0]
        assert frame["hpi_change"].tolist() == [0.7, 1.4]


def test_ttc_prefix_produces_the_downstream_column_schema(discrete_hazard):
    out = discrete_hazard.compute_conditional_horizon_pd(
        _ConstantHazardModel(0.01), _loans([0.0]), macro_paths={},
        horizons=[12, 24], prefix="ttc_",
    )
    # 10_basel_irb_capital.py's load_ttc_pd() reads ttc_pd_12m by name.
    assert "ttc_pd_12m" in out.columns
    assert "ttc_pd_24m" in out.columns


# ── internal covariates are excluded from both models ───────────────────────

def test_internal_covariates_are_absent_from_the_shared_feature_set(discrete_hazard):
    """
    delinquency_indicator / delinquency_status are excluded deliberately:
    their future path is an outcome of the default process, so they cannot be
    projected over a multi-period horizon without a second model, and on the
    row before default they are close to a deterministic function of the
    target. See the module docstring.
    """
    for excluded in discrete_hazard.EXCLUDED_FEATURES:
        assert excluded not in discrete_hazard.FEATURES
        assert excluded not in discrete_hazard.MODEL_COLS
        assert excluded not in discrete_hazard.STATIC_FEATURES
        assert excluded not in discrete_hazard.MACRO_FEATURES


def test_excluded_covariates_never_reach_either_fitted_model(discrete_hazard):
    frame = _fit_frame(discrete_hazard, n=4_000, seed=11)
    frame["delinquency_indicator"] = 1
    frame["delinquency_status"] = 3

    X = frame[discrete_hazard.MODEL_COLS]
    y = frame[discrete_hazard.TARGET_HAZARD].to_numpy()

    logit = discrete_hazard.LogitHazardModel(sampling_rate=1.0).fit(X, y)
    assert not any(
        excluded in name
        for name in logit.feature_names_
        for excluded in discrete_hazard.EXCLUDED_FEATURES
    )

    xgb = discrete_hazard.XGBHazardModel(sampling_rate=1.0).fit(X, y)
    assert not any(
        excluded in xgb.feature_names_
        for excluded in discrete_hazard.EXCLUDED_FEATURES
    )


# ── shared interface conformance ────────────────────────────────────────────

def _fit_frame(discrete_hazard, n=4_000, seed=5, hazard_intercept=-4.0,
               credit_beta=-1.2):
    """
    Synthetic loan-month rows with a known data-generating process:
    logit h = intercept + credit_beta * z(credit_score) + age effect.
    """
    rng = np.random.default_rng(seed)
    credit = rng.normal(700, 50, n)
    age = rng.integers(0, 120, n).astype(float)
    z_credit = (credit - 700) / 50
    z_age = age / 120.0
    linear = hazard_intercept + credit_beta * z_credit + 1.5 * z_age
    p = 1 / (1 + np.exp(-linear))
    y = rng.binomial(1, p)

    return pd.DataFrame({
        "loan_seq_num": np.arange(n).astype(str),
        "loan_age": age,
        "credit_score": credit,
        "orig_cltv": rng.uniform(40, 100, n),
        "orig_dti": rng.uniform(15, 50, n),
        "orig_interest_rate": rng.uniform(3, 8, n),
        "orig_upb": rng.uniform(80_000, 500_000, n),
        "num_borrowers": rng.integers(1, 3, n).astype(float),
        "occupancy_status": rng.choice(["O", "I", "S"], n),
        "property_type": rng.choice(["SF", "CO", "PU"], n),
        "ur_3m_lag": rng.uniform(3, 10, n),
        "hpi_change": rng.uniform(0.6, 1.5, n),
        discrete_hazard.TARGET_HAZARD: y.astype(np.int8),
    })


@pytest.mark.parametrize("model_name", ["logit", "xgb"])
def test_both_models_satisfy_the_shared_interface(discrete_hazard, model_name):
    frame = _fit_frame(discrete_hazard)
    X = frame[discrete_hazard.MODEL_COLS]
    y = frame[discrete_hazard.TARGET_HAZARD].to_numpy()

    model = (discrete_hazard.LogitHazardModel(sampling_rate=1.0)
             if model_name == "logit"
             else discrete_hazard.XGBHazardModel(sampling_rate=1.0))

    returned = model.fit(X, y)
    assert returned is model  # fit() returns self, so calls can chain

    hazard = model.predict_hazard(X)
    assert isinstance(hazard, np.ndarray)
    assert hazard.shape == (len(X),)
    assert ((hazard >= 0) & (hazard <= 1)).all()
    assert model.name == model_name


@pytest.mark.parametrize("model_name", ["logit", "xgb"])
def test_prior_correction_flows_through_predict_hazard(discrete_hazard, model_name):
    """
    Both models must apply the correction inside predict_hazard(), so that
    every downstream consumer sees corrected probabilities without knowing
    subsampling happened. Halving the sampling rate must scale the predicted
    odds by the same factor for either model.
    """
    frame = _fit_frame(discrete_hazard)
    X = frame[discrete_hazard.MODEL_COLS]
    y = frame[discrete_hazard.TARGET_HAZARD].to_numpy()

    def _build(rate):
        return (discrete_hazard.LogitHazardModel(sampling_rate=rate)
                if model_name == "logit"
                else discrete_hazard.XGBHazardModel(sampling_rate=rate))

    uncorrected = _build(1.0).fit(X, y).predict_hazard(X)
    corrected = _build(0.5).fit(X, y).predict_hazard(X)

    odds_ratio = (corrected / (1 - corrected)) / (uncorrected / (1 - uncorrected))
    np.testing.assert_allclose(odds_ratio, 0.5, rtol=1e-6)


# ── linear model: recovers a known DGP ──────────────────────────────────────

def test_logit_model_recovers_a_known_coefficient(discrete_hazard):
    """
    The synthetic hazard is logit h = -4 - 1.2*z(credit_score) + age effect,
    where z uses mean 700 / sd 50. On the ORIGINAL credit_score scale that is
    a coefficient of -1.2/50 = -0.024 per FICO point, which is what
    coefficient_table() must report — this exercises the exact back-transform
    of the internal standardisation.
    """
    frame = _fit_frame(discrete_hazard, n=60_000, seed=17, credit_beta=-1.2)
    X = frame[discrete_hazard.MODEL_COLS]
    y = frame[discrete_hazard.TARGET_HAZARD].to_numpy()

    model = discrete_hazard.LogitHazardModel(sampling_rate=1.0).fit(X, y)
    table = model.coefficient_table().set_index("feature")

    assert table.loc["credit_score", "coef"] == pytest.approx(-1.2 / 50, rel=0.15)
    # Lower FICO, higher hazard — the odds ratio must be below 1.
    assert table.loc["credit_score", "odds_ratio"] < 1.0
    assert (table.loc["credit_score", "or_lower_95"]
            <= table.loc["credit_score", "odds_ratio"]
            <= table.loc["credit_score", "or_upper_95"])


def test_logit_spline_recovers_a_known_baseline_hazard_shape(discrete_hazard):
    """
    Generate a hump-shaped baseline hazard in loan_age — the mortgage
    seasoning shape — and check the spline baseline traces it: the fitted
    hazard must peak in the interior of the age range, not at an endpoint,
    and must be materially higher at the peak than at age 0.
    """
    rng = np.random.default_rng(23)
    n = 120_000
    age = rng.integers(0, 121, n).astype(float)
    # Hump peaking at age 60.
    age_effect = 2.5 * np.exp(-((age - 60) ** 2) / (2 * 25.0 ** 2))
    linear = -5.0 + age_effect
    y = rng.binomial(1, 1 / (1 + np.exp(-linear)))

    frame = pd.DataFrame({
        "loan_age": age,
        "credit_score": rng.normal(700, 50, n),
        "orig_cltv": rng.uniform(40, 100, n),
        "orig_dti": rng.uniform(15, 50, n),
        "orig_interest_rate": rng.uniform(3, 8, n),
        "orig_upb": rng.uniform(80_000, 500_000, n),
        "num_borrowers": rng.integers(1, 3, n).astype(float),
        "occupancy_status": rng.choice(["O", "I"], n),
        "property_type": rng.choice(["SF", "CO"], n),
        "ur_3m_lag": rng.uniform(3, 10, n),
        "hpi_change": rng.uniform(0.6, 1.5, n),
    })
    X = frame[discrete_hazard.MODEL_COLS]

    model = discrete_hazard.LogitHazardModel(sampling_rate=1.0).fit(X, y)

    ages = np.arange(0, 121, 5)
    reference = X.median(numeric_only=True).to_frame().T
    reference["occupancy_status"] = "O"
    reference["property_type"] = "SF"
    curve = model.baseline_hazard_curve(ages, reference[discrete_hazard.MODEL_COLS])

    peak_age = ages[int(np.argmax(curve))]
    assert 35 <= peak_age <= 85          # recovered the hump's location
    assert curve.max() > 3 * curve[0]    # and its amplitude vs. age 0


def test_logit_reference_levels_are_recorded_for_each_categorical(discrete_hazard):
    frame = _fit_frame(discrete_hazard)
    X = frame[discrete_hazard.MODEL_COLS]
    model = discrete_hazard.LogitHazardModel(sampling_rate=1.0).fit(
        X, frame[discrete_hazard.TARGET_HAZARD].to_numpy())

    for cat in discrete_hazard.CAT_FEATURES:
        assert cat in model.design.reference_levels_
        # The dropped level must NOT appear as its own indicator column.
        ref = model.design.reference_levels_[cat]
        assert f"{cat}[{ref}]" not in model.feature_names_


def test_proportional_hazards_test_reports_a_p_value_per_feature(discrete_hazard):
    frame = _fit_frame(discrete_hazard, n=20_000, seed=29)
    X = frame[discrete_hazard.MODEL_COLS]
    y = frame[discrete_hazard.TARGET_HAZARD].to_numpy()
    model = discrete_hazard.LogitHazardModel(sampling_rate=1.0).fit(X, y)

    tests = discrete_hazard.proportional_hazards_lr_test(
        model, X, y, features=["credit_score", "orig_cltv"])

    assert set(tests["feature"]) == {"credit_score", "orig_cltv"}
    assert ((tests["p_value"] >= 0) & (tests["p_value"] <= 1)).all()
    # The interaction block adds one column per spline basis function.
    assert (tests["df"] == discrete_hazard.SPLINE_DF).all()
    # The alternative nests the null, so its likelihood cannot be lower.
    assert (tests["lr_statistic"] >= -1e-6).all()


# ── XGBoost model specifics ─────────────────────────────────────────────────

def test_xgb_does_not_set_scale_pos_weight(discrete_hazard):
    """
    Imbalance is handled by case-control subsampling; reweighting on top of
    it would distort the probabilities the horizon PD multiplies together.
    """
    frame = _fit_frame(discrete_hazard)
    model = discrete_hazard.XGBHazardModel(sampling_rate=1.0).fit(
        frame[discrete_hazard.MODEL_COLS],
        frame[discrete_hazard.TARGET_HAZARD].to_numpy(),
    )
    assert "scale_pos_weight" not in discrete_hazard.config.DISCRETE_HAZARD_XGB_PARAMS
    assert model.model_.get_params().get("scale_pos_weight") in (None, 1, 1.0)


def test_xgb_monotone_constraints_are_applied_in_feature_order(discrete_hazard):
    frame = _fit_frame(discrete_hazard)
    model = discrete_hazard.XGBHazardModel(sampling_rate=1.0, use_monotone=True).fit(
        frame[discrete_hazard.MODEL_COLS],
        frame[discrete_hazard.TARGET_HAZARD].to_numpy(),
    )

    directions = discrete_hazard.config.DISCRETE_HAZARD_MONOTONE_DIRECTIONS
    constraints = dict(zip(model.feature_names_, model.monotone_constraints_))
    for feature, expected in directions.items():
        if feature in constraints:
            assert constraints[feature] == expected
    # Unlisted covariates stay unconstrained.
    assert constraints.get("orig_upb", 0) == 0


def test_xgb_hazard_is_monotone_in_the_constrained_covariates(discrete_hazard):
    """
    With constraints on, sweeping a constrained covariate across a synthetic
    grid (holding everything else fixed) must move the predicted hazard in
    one direction only — the auditability property the constraints buy.
    """
    frame = _fit_frame(discrete_hazard, n=20_000, seed=31)
    model = discrete_hazard.XGBHazardModel(sampling_rate=1.0, use_monotone=True).fit(
        frame[discrete_hazard.MODEL_COLS],
        frame[discrete_hazard.TARGET_HAZARD].to_numpy(),
    )

    base = frame[discrete_hazard.MODEL_COLS].median(numeric_only=True).to_frame().T
    base["occupancy_status"] = "O"
    base["property_type"] = "SF"
    base = base[discrete_hazard.MODEL_COLS]

    for feature, direction in discrete_hazard.config.DISCRETE_HAZARD_MONOTONE_DIRECTIONS.items():
        if feature not in base.columns:
            continue
        grid = np.linspace(frame[feature].min(), frame[feature].max(), 25)
        sweep = pd.concat([base] * len(grid), ignore_index=True)
        sweep[feature] = grid
        hazard = model.predict_hazard(sweep)
        diffs = np.diff(hazard)
        if direction > 0:
            assert (diffs >= -1e-9).all(), f"{feature} not non-decreasing"
        else:
            assert (diffs <= 1e-9).all(), f"{feature} not non-increasing"


def test_xgb_monotone_constraints_can_be_disabled(discrete_hazard):
    frame = _fit_frame(discrete_hazard, n=2_000)
    model = discrete_hazard.XGBHazardModel(sampling_rate=1.0, use_monotone=False).fit(
        frame[discrete_hazard.MODEL_COLS],
        frame[discrete_hazard.TARGET_HAZARD].to_numpy(),
    )
    assert model.monotone_constraints_ == ()


# ── validation helpers ──────────────────────────────────────────────────────

def test_snapshot_population_drops_immature_rows_only(discrete_hazard, monkeypatch):
    """
    Same maturity rule as filter_immature_right_censored(): a row is usable
    if its loan's outcome is observed (default_date known) or if a full
    365-day window remains in the panel after it.
    """
    panel = _panel([
        # Mature: 365+ days of panel remain after it.
        {"loan_seq_num": "L1", "report_date": "2019-06-01", "default_date": pd.NaT,
         "loan_age": 10},
        # Immature: too close to the panel end AND never observed to default.
        {"loan_seq_num": "L2", "report_date": "2020-06-01", "default_date": pd.NaT,
         "loan_age": 10},
        # Immature date, but the outcome IS observed — keep it.
        {"loan_seq_num": "L3", "report_date": "2020-06-01",
         "default_date": "2020-09-01", "loan_age": 10},
        # Not a snapshot date at all.
        {"loan_seq_num": "L4", "report_date": "2019-07-01", "default_date": pd.NaT,
         "loan_age": 10},
        # Extends the panel so 2019-06-01 clears the 365-day cutoff.
        {"loan_seq_num": "L5", "report_date": "2020-12-01", "default_date": pd.NaT,
         "loan_age": 10},
    ])

    out = discrete_hazard.build_snapshot_population(
        panel, snapshot_dates=["2019-06-01", "2020-06-01"])

    assert set(out["loan_seq_num"]) == {"L1", "L3"}


def test_observed_discrete_km_matches_hand_computed_hazard(discrete_hazard):
    panel = pd.DataFrame({
        "loan_age": [0, 0, 0, 0, 1, 1, 1],
        discrete_hazard.TARGET_HAZARD: [0, 0, 0, 1, 0, 0, 1],
    })
    km = discrete_hazard.observed_discrete_km(panel).set_index("loan_age")

    assert km.loc[0, "hazard"] == pytest.approx(0.25)   # 1 of 4 at risk
    assert km.loc[1, "hazard"] == pytest.approx(1 / 3)  # 1 of 3 at risk
    assert km.loc[1, "survival"] == pytest.approx(0.75 * (2 / 3))


def test_decile_calibration_reports_predicted_against_observed(discrete_hazard):
    rng = np.random.default_rng(41)
    p = rng.uniform(0, 0.5, 5_000)
    y = rng.binomial(1, p)

    table = discrete_hazard.decile_calibration(y, p)

    assert len(table) == 10
    assert {"n", "mean_predicted", "observed_rate"}.issubset(table.columns)
    # A well-specified DGP must come out roughly on the diagonal.
    assert np.corrcoef(table["mean_predicted"], table["observed_rate"])[0, 1] > 0.9


def test_evaluate_horizon_pd_reports_the_standard_metric_set(discrete_hazard):
    rng = np.random.default_rng(43)
    p = rng.uniform(0.01, 0.4, 2_000)
    y = rng.binomial(1, p)

    out = discrete_hazard.evaluate_horizon_pd(y, p, "OOT-snapshot", "logit")

    assert {"auroc", "ks", "gini", "brier"}.issubset(out)
    assert out["gini"] == pytest.approx(2 * out["auroc"] - 1)
    assert 0 <= out["brier"] <= 1
