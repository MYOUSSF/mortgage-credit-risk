"""
Tests for 01_data_preprocessing.py — the default-event target, the
train/OOS/OOT split, and the IV/PSI diagnostics used for monitoring.

These are the pieces where a silent bug is most expensive: an off-by-one
in the default window or a leaky split would quietly inflate every
downstream model's reported performance.
"""
import numpy as np
import pandas as pd
import pytest


# ── extract_pd_rows: default-event target construction ──────────────────────

def test_default_codes_match_documented_definition(preprocessing):
    # README documents this exact set — see 01_data_preprocessing.py:163-167.
    # Codes 01 (prepayment), 16 (reperforming) and 96 (non-standard) are
    # deliberately excluded; pin the set so a future edit doesn't silently
    # widen or narrow the default definition without updating docs.
    assert preprocessing.DEFAULT_CODES == {"02", "03", "06", "09", "15"}


def test_extract_pd_rows_labels_12m_forward_window(preprocessing):
    loan_id = "L1"
    rows = []
    # Monthly performance rows: 400 days before default, then 30 days before.
    default_date = pd.Timestamp("2010-06-01")
    for days_before, code in [(400, np.nan), (30, np.nan), (0, "03")]:
        rows.append({
            "loan_seq_num": loan_id,
            "report_date": default_date - pd.Timedelta(days=days_before),
            "zero_balance_code": code,
        })
    df = pd.DataFrame(rows)

    out = preprocessing.extract_pd_rows(df)

    # The default event row itself must be dropped (it directly encodes the
    # outcome — keeping it would leak the label into the feature rows).
    assert (out["report_date"] == default_date).sum() == 0

    row_400 = out.loc[out["report_date"] == default_date - pd.Timedelta(days=400)]
    row_30  = out.loc[out["report_date"] == default_date - pd.Timedelta(days=30)]
    assert row_400["default_12m"].item() == 0  # outside the 365-day window
    assert row_30["default_12m"].item() == 1   # inside the 365-day window


def test_extract_pd_rows_prepayment_is_not_a_default(preprocessing):
    df = pd.DataFrame({
        "loan_seq_num": ["L1", "L1"],
        "report_date":  pd.to_datetime(["2010-01-01", "2010-02-01"]),
        "zero_balance_code": [np.nan, "01"],  # 01 = prepayment
    })
    out = preprocessing.extract_pd_rows(df)
    assert (out["default_12m"] == 1).sum() == 0


def test_extract_pd_rows_never_defaulting_loan_is_all_zero(preprocessing):
    df = pd.DataFrame({
        "loan_seq_num": ["L1"] * 5,
        "report_date":  pd.date_range("2010-01-01", periods=5, freq="MS"),
        "zero_balance_code": [np.nan] * 5,
    })
    out = preprocessing.extract_pd_rows(df)
    assert len(out) == 5
    assert (out["default_12m"] == 0).all()


# ── split_pd: temporal OOT + random OOS ──────────────────────────────────────

def test_split_pd_respects_temporal_boundary(preprocessing):
    dates = pd.date_range("2000-01-01", "2020-01-01", freq="MS")
    df = pd.DataFrame({
        "loan_seq_num": [f"L{i}" for i in range(len(dates))],
        "report_date": dates,
        "x": np.arange(len(dates)),
    })
    train, oos, oot = preprocessing.split_pd(df)

    cutoff = preprocessing.OOT_CUTOFF
    assert (train["report_date"] < cutoff).all()
    assert (oos["report_date"] < cutoff).all()
    assert (oot["report_date"] >= cutoff).all()
    assert len(train) + len(oos) + len(oot) == len(df)


def test_split_pd_train_oos_are_disjoint_and_oos_fraction_is_correct(preprocessing):
    dates = pd.date_range("2000-01-01", "2016-01-01", freq="D")
    df = pd.DataFrame({
        "loan_seq_num": [f"L{i}" for i in range(len(dates))],
        "report_date": dates,
        "x": np.arange(len(dates)),
    }).reset_index(drop=True)
    train, oos, oot = preprocessing.split_pd(df)

    assert set(train.index).isdisjoint(set(oos.index))
    in_sample_n = len(train) + len(oos)
    # GroupShuffleSplit only approximates OOS_FRAC at the row level (groups
    # vary in size) — this single-row-per-group df is degenerate (every row
    # is its own "group"), so the fraction should still come out close.
    assert abs(len(oos) / in_sample_n - preprocessing.OOS_FRAC) < 0.05


def test_split_pd_no_loan_spans_both_train_and_oos(preprocessing):
    # Regression test for the loan-level leakage bug: split_pd() used to
    # shuffle individual loan-month rows, so a single loan's monthly
    # snapshots (largely static features — credit score, CLTV, DTI, orig
    # rate) could land in both Train and OOS, letting models partially
    # "recognise" training loans. Build a multi-row-per-loan panel and
    # assert every loan_seq_num is wholly contained in one split.
    rng = np.random.default_rng(0)
    n_loans = 200
    rows = []
    for i in range(n_loans):
        loan_id = f"L{i}"
        n_months = rng.integers(3, 24)
        start = pd.Timestamp("2005-01-01") + pd.DateOffset(months=int(rng.integers(0, 120)))
        for m in range(n_months):
            rows.append({
                "loan_seq_num": loan_id,
                "report_date": start + pd.DateOffset(months=m),
                "x": i,
            })
    df = pd.DataFrame(rows)

    train, oos, oot = preprocessing.split_pd(df)

    train_loans = set(train["loan_seq_num"])
    oos_loans   = set(oos["loan_seq_num"])
    assert train_loans.isdisjoint(oos_loans)


def test_split_pd_raises_on_empty_in_sample(preprocessing):
    # All rows past the OOT cutoff -> in_sample is empty. This should raise
    # a descriptive error rather than crash inside sklearn with n_samples=0.
    df = pd.DataFrame({
        "report_date": pd.date_range("2018-01-01", periods=10, freq="MS"),
        "x": np.arange(10),
    })
    with pytest.raises(ValueError, match="in_sample is empty"):
        preprocessing.split_pd(df)


# ── filter_immature_right_censored: 12m window immaturity ────────────────────

def test_filter_immature_right_censored_drops_recent_never_defaulting_rows(preprocessing):
    # Panel's max report_date is 2020-01-01. A never-defaulting loan's last
    # row is only 60 days before that — its true 12m outcome is unknown.
    df = pd.DataFrame({
        "loan_seq_num": ["L_OLD", "L_RECENT"],
        "report_date":  pd.to_datetime(["2010-01-01", "2019-11-02"]),
        "has_default":  [False, False],
    })
    # Ensure the panel's global max date is 2020-01-01 via another row.
    df = pd.concat([df, pd.DataFrame({
        "loan_seq_num": ["L_ANCHOR"], "report_date": pd.to_datetime(["2020-01-01"]),
        "has_default": [False],
    })], ignore_index=True)

    out = preprocessing.filter_immature_right_censored(df)

    assert "L_OLD" in set(out["loan_seq_num"])
    assert "L_RECENT" not in set(out["loan_seq_num"])
    assert "has_default" not in out.columns


def test_filter_immature_right_censored_keeps_rows_with_known_default_date(preprocessing):
    # This loan is known to NOT default within 12m (its eventual default_date
    # is far in the future) even though its row is close to the panel's end —
    # that label is already correct, not censored, so it must be kept.
    df = pd.DataFrame({
        "loan_seq_num": ["L_KNOWN", "L_ANCHOR"],
        "report_date":  pd.to_datetime(["2019-11-02", "2020-01-01"]),
        "has_default":  [True, False],
    })
    out = preprocessing.filter_immature_right_censored(df)
    assert "L_KNOWN" in set(out["loan_seq_num"])


# ── LGD workout-period truncation bias: IPCW weighting ────────────────────────

def test_extract_lgd_onset_rows_identifies_onset_and_resolution(preprocessing):
    df = pd.DataFrame({
        "loan_seq_num": ["L1", "L1", "L1", "L2", "L2"],
        "report_date": pd.to_datetime([
            "2010-01-01", "2010-04-01", "2010-05-01",
            "2010-01-01", "2010-04-01",
        ]),
        "delinquency_status": [0, 3, 4, 0, 3],
        "zero_balance_code": [np.nan, np.nan, "03", np.nan, np.nan],
    })
    out = preprocessing.extract_lgd_onset_rows(df)
    out = out.set_index("loan_seq_num")

    assert out.loc["L1", "onset_date"] == pd.Timestamp("2010-04-01")
    assert out.loc["L1", "resolved"] == True
    assert out.loc["L1", "resolution_date"] == pd.Timestamp("2010-05-01")
    assert out.loc["L2", "resolved"] == False
    assert pd.isna(out.loc["L2", "resolution_date"])


def test_compute_ipcw_weights_upweights_slow_resolutions(preprocessing):
    rng = np.random.default_rng(1)
    n = 300
    onset_dates = pd.Series(
        pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D")
    )
    # True resolution time is highly variable; global_max_date truncates
    # the sample so slow-resolving, late-onset cases go unresolved.
    true_resolution_days = rng.exponential(200, n).astype(int) + 1
    global_max_date = pd.Timestamp("2020-01-01")
    available_days = (global_max_date - onset_dates).dt.days

    resolved = true_resolution_days <= available_days
    resolution_date = onset_dates + pd.to_timedelta(true_resolution_days, unit="D")

    onset_df = pd.DataFrame({
        "loan_seq_num": [f"L{i}" for i in range(n)],
        "onset_date": onset_dates,
        "resolved": resolved,
        "resolution_date": resolution_date.where(resolved),
    })

    weights = preprocessing.compute_ipcw_weights(onset_df, global_max_date=global_max_date)

    assert set(weights["loan_seq_num"]) == set(onset_df.loc[resolved, "loan_seq_num"])
    assert weights["ipcw_weight"].mean() == pytest.approx(1.0, abs=0.05)

    merged = weights.merge(
        onset_df[["loan_seq_num"]].assign(
            resolution_days=(resolution_date - onset_dates).dt.days
        ),
        on="loan_seq_num",
    )
    slow = merged[merged["resolution_days"] > merged["resolution_days"].quantile(0.9)]
    fast = merged[merged["resolution_days"] < merged["resolution_days"].quantile(0.5)]
    assert slow["ipcw_weight"].mean() > fast["ipcw_weight"].mean()


def test_compute_ipcw_weights_all_resolved_yields_uniform_weight(preprocessing):
    onset_df = pd.DataFrame({
        "loan_seq_num": ["L1", "L2", "L3"],
        "onset_date": pd.to_datetime(["2015-01-01", "2015-02-01", "2015-03-01"]),
        "resolved": [True, True, True],
        "resolution_date": pd.to_datetime(["2015-06-01", "2015-08-01", "2015-05-01"]),
    })
    weights = preprocessing.compute_ipcw_weights(onset_df)
    assert (weights["ipcw_weight"] == 1.0).all()


def test_compute_ipcw_weights_empty_input_returns_empty(preprocessing):
    out = preprocessing.compute_ipcw_weights(pd.DataFrame())
    assert out.empty


# ── IV / PSI diagnostics ──────────────────────────────────────────────────────

def test_iv_is_high_for_a_perfectly_separating_feature(preprocessing):
    n = 400
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, n)
    # Feature is a noisy but strongly separating signal: bads cluster high.
    feature = target * 5 + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"f": feature, "y": target})
    iv = preprocessing._compute_iv(df, "f", "y")
    assert iv > 0.5  # "Very strong" per _iv_strength()


def test_iv_is_near_zero_for_an_unrelated_feature(preprocessing):
    n = 2000
    rng = np.random.default_rng(1)
    target = rng.integers(0, 2, n)
    feature = rng.normal(0, 1, n)  # independent of target
    df = pd.DataFrame({"f": feature, "y": target})
    iv = preprocessing._compute_iv(df, "f", "y")
    assert iv < 0.02  # "Negligible" per _iv_strength()


def test_psi_is_near_zero_for_identical_distributions(preprocessing):
    rng = np.random.default_rng(2)
    ref = pd.Series(rng.normal(0, 1, 5000))
    test = pd.Series(rng.normal(0, 1, 5000))
    psi = preprocessing._compute_psi(ref, test)
    assert psi < 0.05


def test_psi_flags_a_major_population_shift(preprocessing):
    rng = np.random.default_rng(3)
    ref = pd.Series(rng.normal(0, 1, 5000))
    test = pd.Series(rng.normal(5, 1, 5000))  # fully shifted distribution
    psi = preprocessing._compute_psi(ref, test)
    assert psi > 0.25  # "major shift" threshold per _psi_flag()


def test_psi_flags_a_population_entirely_outside_the_reference_range(preprocessing):
    # Regression test: a test population that falls completely outside
    # every reference bin must report a very high PSI (the worst-case
    # shift), not PSI ~ 0. The naive value_counts(normalize=True) approach
    # divides by the count of values landing in a *known* bin, which is 0
    # here — the resulting NaN proportions get silently dropped by
    # Series.sum(skipna=True), masking a total population shift as "Stable".
    rng = np.random.default_rng(4)
    ref = pd.Series(rng.normal(0, 1, 5000))
    test = pd.Series(rng.normal(20, 1, 2000))  # zero overlap with ref's range
    psi = preprocessing._compute_psi(ref, test)
    assert psi > 0.25
