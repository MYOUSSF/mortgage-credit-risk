"""
Tests for 06_survival_analysis.py's build_survival_df() — specifically the
Cox PH duration calculation.

extract_pd_rows() (01_data_preprocessing.py) drops the actual default row
itself to prevent leakage in the binary 12-month target. That means the
last retained row for a defaulted loan is exactly one reporting period (1
month) before the true default — build_survival_df() must add that period
back for event=1 loans so duration reflects true time-to-default rather
than time-to-last-observation-before-it. Censored (event=0) loans need no
adjustment: their last retained row *is* the true censoring time.
"""
import numpy as np
import pandas as pd


def test_duration_adds_one_period_for_an_actual_defaulter(survival_analysis):
    # L1's last retained row has loan_age=11 and default_12m=1 (it defaults
    # the following month, the row extract_pd_rows() already dropped).
    df = pd.DataFrame({
        "loan_seq_num": ["L1", "L1", "L1"],
        "loan_age":     [9, 10, 11],
        "default_12m":  [0, 0, 1],
    })
    out = survival_analysis.build_survival_df(df, features=[])
    row = out.set_index("loan_seq_num").loc["L1"]

    assert row["event"] == 1
    assert row["duration"] == 12  # last retained loan_age (11) + 1


def test_duration_is_unchanged_for_a_censored_loan(survival_analysis):
    # L2 never defaults — its last retained row's loan_age IS the true
    # censoring time, no adjustment needed.
    df = pd.DataFrame({
        "loan_seq_num": ["L2", "L2", "L2"],
        "loan_age":     [5, 6, 7],
        "default_12m":  [0, 0, 0],
    })
    out = survival_analysis.build_survival_df(df, features=[])
    row = out.set_index("loan_seq_num").loc["L2"]

    assert row["event"] == 0
    assert row["duration"] == 7  # unchanged last retained loan_age


# ── compute_ttc_pds: macro-neutral (through-the-cycle) re-scoring ────────────

def test_compute_ttc_pds_overrides_macro_covariates_to_long_run_mean(survival_analysis):
    """
    compute_ttc_pds() must re-score the same fitted Cox model with the
    time-varying macro covariates replaced by their long-run training-set
    average — that's what makes the output through-the-cycle rather than
    point-in-time — while leaving every other loan characteristic (here,
    credit_score) at its actual current value.
    """
    captured = {}

    class StubCox:
        def predict_survival_function(self, X):
            captured["X"] = X.copy()
            return pd.DataFrame(0.9, index=[12, 24], columns=X.index)

    oos_surv = pd.DataFrame({
        "loan_seq_num": ["L1", "L2"],
        "duration":     [12, 24],
        "event":        [0, 0],
        "ur_3m_lag":    [8.0, 2.0],
        "hpi_change":   [0.5, 1.5],
        "credit_score": [700, 750],
    })
    train_surv = pd.DataFrame({
        "ur_3m_lag":  [4.0, 4.0, 6.0, 2.0],
        "hpi_change": [1.0, 1.0, 1.0, 1.0],
    })
    features = ["ur_3m_lag", "hpi_change", "credit_score"]

    ttc_df = survival_analysis.compute_ttc_pds(
        StubCox(), oos_surv, features, train_surv, horizons=[12, 24]
    )

    X_seen = captured["X"]
    assert (X_seen["ur_3m_lag"] == train_surv["ur_3m_lag"].mean()).all()
    assert (X_seen["hpi_change"] == train_surv["hpi_change"].mean()).all()
    # Non-macro feature is a real loan characteristic — left untouched.
    assert X_seen["credit_score"].tolist() == [700, 750]

    assert list(ttc_df["loan_seq_num"]) == ["L1", "L2"]
    assert "ttc_pd_12m" in ttc_df.columns
    assert "ttc_pd_24m" in ttc_df.columns
    np.testing.assert_allclose(ttc_df["ttc_pd_12m"].values, [0.1, 0.1])


def test_compute_ttc_pds_returns_empty_when_no_oos_survival_data(survival_analysis):
    empty = pd.DataFrame(columns=["loan_seq_num", "duration", "event"])
    train_surv = pd.DataFrame({"ur_3m_lag": [4.0], "hpi_change": [1.0]})

    class StubCox:
        def predict_survival_function(self, X):
            raise AssertionError("should not be called on empty input")

    result = survival_analysis.compute_ttc_pds(
        StubCox(), empty, features=["ur_3m_lag"], train_surv=train_surv
    )
    assert result.empty
