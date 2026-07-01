"""
Tests for 02_pd_logistic_regression.py — WoE encoding leakage-proofing and
the validation metrics (KS, Hosmer-Lemeshow) used to grade the PD model.
"""
import numpy as np
import pandas as pd
import pytest


def _make_train_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    # credit_score separates the target: low score -> more defaults.
    credit_score = rng.integers(500, 800, n)
    default = (credit_score < 620).astype(np.int8)
    # flip a few labels so bins aren't perfectly pure
    flip = rng.random(n) < 0.05
    default = np.where(flip, 1 - default, default)
    return pd.DataFrame({
        "loan_seq_num": [f"L{i}" for i in range(n)],
        "report_date": pd.date_range("2005-01-01", periods=n, freq="D"),
        "credit_score": credit_score,
        "channel": rng.choice(["R", "B", "C"], n),  # categorical feature
        "default_12m": default,
    })


# ── WoE maps: fitted on train only, safe to apply to unseen data ────────────

def test_fit_woe_maps_uses_only_the_training_target_rate(pd_logistic):
    train = _make_train_df()
    woe_maps, iv_df = pd_logistic.fit_woe_maps(train, ["credit_score", "channel"])

    assert set(woe_maps.keys()) == {"credit_score", "channel"}
    assert woe_maps["credit_score"]["type"] == "num"
    assert "edges" in woe_maps["credit_score"]
    assert woe_maps["channel"]["type"] == "cat"

    # credit_score should carry meaningfully more separating power than
    # channel, which was assigned independently of the target.
    iv = iv_df.set_index("feature")["iv"]
    assert iv["credit_score"] > iv["channel"]


def test_extract_arrays_handles_values_outside_the_training_range(pd_logistic):
    train = _make_train_df()
    woe_maps, _ = pd_logistic.fit_woe_maps(train, ["credit_score", "channel"])

    X_train, y_train, _, imputer, scaler = pd_logistic._extract_arrays(
        train, woe_maps, imputer=None, scaler=None, fit=True
    )
    assert not np.isnan(X_train).any()

    # OOS row with a credit_score far outside anything seen in training and
    # an unseen categorical value. This must not raise or produce NaNs —
    # unseen bins should fall back to a neutral WoE of 0.0, not crash the
    # pipeline the way an unhandled KeyError/IndexError would.
    oos = pd.DataFrame({
        "loan_seq_num": ["L_new"],
        "report_date": pd.Timestamp("2019-01-01"),
        "credit_score": [10_000],       # outside every training bin edge
        "channel": ["UNSEEN_CHANNEL"],  # never seen in training
        "default_12m": [0],
    })
    X_oos, y_oos, id_df, _, _ = pd_logistic._extract_arrays(
        oos, woe_maps, imputer=imputer, scaler=scaler, fit=False
    )
    assert X_oos.shape == (1, 2)
    assert not np.isnan(X_oos).any()


# ── Validation metrics ────────────────────────────────────────────────────────

def test_ks_statistic_is_one_for_perfect_separation(pd_logistic):
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.array([0.1] * 50 + [0.9] * 50)  # bads always score higher
    ks = pd_logistic.ks_statistic(y_true, y_score)
    assert ks == pytest.approx(1.0)


def test_ks_statistic_is_near_zero_for_uninformative_scores(pd_logistic):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 2000)
    y_score = rng.random(2000)  # scores carry no information about y_true
    ks = pd_logistic.ks_statistic(y_true, y_score)
    assert ks < 0.15


def test_hosmer_lemeshow_returns_a_valid_test_statistic(pd_logistic):
    rng = np.random.default_rng(0)
    n = 500
    p = rng.uniform(0.01, 0.5, n)
    y = (rng.random(n) < p).astype(int)

    result = pd_logistic.hosmer_lemeshow(y, p, g=10)

    assert set(result.keys()) == {"hl_stat", "hl_pval", "hl_dof"}
    assert result["hl_dof"] == 8
    assert result["hl_stat"] >= 0
    assert 0.0 <= result["hl_pval"] <= 1.0
