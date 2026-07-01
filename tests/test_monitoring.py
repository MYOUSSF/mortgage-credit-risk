"""
Tests for 09_monitoring.py — PSI drift monitoring against a fixed
reference distribution, plus default-rate and period-bucketing logic.
"""
import numpy as np
import pandas as pd
import pytest


def test_reference_bins_are_fixed_across_repeated_calls(monitoring):
    # The whole point of fitting the reference once is that PSI stays
    # comparable across periods -- refitting edges per period would let
    # the bins silently drift along with the data, masking real shifts.
    ref = pd.Series(np.random.default_rng(0).normal(0, 1, 5000))
    dist_a = monitoring.fit_reference_distribution(ref)
    dist_b = monitoring.fit_reference_distribution(ref)
    np.testing.assert_array_equal(dist_a["edges"], dist_b["edges"])


def test_psi_is_near_zero_for_an_unchanged_population(monitoring):
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 5000))
    ref_dist = monitoring.fit_reference_distribution(ref)

    test = pd.Series(rng.normal(0, 1, 2000))
    psi, pct_outside = monitoring.psi_against_reference(test, ref_dist)
    assert psi < 0.05
    assert monitoring.psi_flag(psi) == "Stable"


def test_psi_flags_a_major_shift_and_out_of_range_values(monitoring):
    rng = np.random.default_rng(2)
    ref = pd.Series(rng.normal(0, 1, 5000))
    ref_dist = monitoring.fit_reference_distribution(ref)

    # Fully shifted and outside anything the reference bins ever saw.
    test = pd.Series(rng.normal(20, 1, 2000))
    psi, pct_outside = monitoring.psi_against_reference(test, ref_dist)
    assert monitoring.psi_flag(psi) == "Major shift"
    assert pct_outside > 0.9  # almost none of these values fall in a reference bin


def test_psi_flag_thresholds(monitoring):
    assert monitoring.psi_flag(0.05) == "Stable"
    assert monitoring.psi_flag(0.15) == "Investigate"
    assert monitoring.psi_flag(0.30) == "Major shift"
    assert monitoring.psi_flag(np.nan) == "N/A"


def test_categorical_reference_and_unseen_category(monitoring):
    ref = pd.Series(["A", "A", "B", "B", "B", "C"] * 100)
    ref_dist = monitoring.fit_reference_distribution(ref)
    assert ref_dist["type"] == "cat"

    # A category never seen in training shows up as fully out-of-range,
    # not as a crash or a silently-ignored row.
    test = pd.Series(["D"] * 50)
    psi, pct_outside = monitoring.psi_against_reference(test, ref_dist)
    assert pct_outside == 1.0
    assert not np.isnan(psi)


def test_build_monitoring_periods_buckets_oot_by_year(monitoring):
    oos = pd.DataFrame({"report_date": pd.to_datetime(["2016-01-01"])})
    oot = pd.DataFrame({
        "report_date": pd.to_datetime(
            ["2017-06-01", "2017-09-01", "2018-01-01", "2019-01-01"]
        ),
    })
    periods = monitoring.build_monitoring_periods(oos, oot)

    assert set(periods.keys()) == {"OOS (post-build)", "OOT 2017", "OOT 2018", "OOT 2019"}
    assert len(periods["OOT 2017"]) == 2
    assert len(periods["OOT 2018"]) == 1


def test_monitor_default_rate_reports_delta_against_train(monitoring):
    train = pd.DataFrame({"default_12m": [0] * 90 + [1] * 10})  # 10% base rate
    period_df = pd.DataFrame({"default_12m": [0] * 80 + [1] * 20})  # 20% observed
    periods = {"OOT 2020": period_df}

    result = monitoring.monitor_default_rate(train, periods)

    row = result.iloc[0]
    assert row["train_default_rate"] == pytest.approx(0.10)
    assert row["observed_default_rate"] == pytest.approx(0.20)
    assert row["delta_pp"] == pytest.approx(10.0)


def test_monitor_features_end_to_end_on_synthetic_populations(monitoring):
    rng = np.random.default_rng(3)
    train = pd.DataFrame({
        "credit_score": rng.integers(600, 800, 3000),
        "channel":      rng.choice(["R", "B", "C"], 3000),
    })
    stable_period = pd.DataFrame({
        "credit_score": rng.integers(600, 800, 500),
        "channel":      rng.choice(["R", "B", "C"], 500),
    })
    shifted_period = pd.DataFrame({
        "credit_score": rng.integers(300, 500, 500),  # entirely different range
        "channel":      rng.choice(["R", "B", "C"], 500),
    })
    periods = {"Stable period": stable_period, "Shifted period": shifted_period}

    result = monitoring.monitor_features(train, periods, ["credit_score", "channel"])

    cs_stable = result.query("feature == 'credit_score' and period == 'Stable period'")["flag"].item()
    cs_shifted = result.query("feature == 'credit_score' and period == 'Shifted period'")["flag"].item()
    assert cs_stable == "Stable"
    assert cs_shifted == "Major shift"
