"""
Tests for 10_basel_irb_capital.py — the rating master scale (PD -> grade)
and the Basel II/III IRB retail residential mortgage capital formula,
which nothing upstream in this pipeline previously computed.
"""
import numpy as np
import pandas as pd
import pytest


# ── Rating master scale ───────────────────────────────────────────────────────

def test_assign_rating_grade_maps_low_pd_to_top_grade(basel_irb_capital):
    grades = basel_irb_capital.assign_rating_grade(np.array([0.00001]))
    assert grades[0] == "AAA"


def test_assign_rating_grade_maps_high_pd_to_default_grade(basel_irb_capital):
    grades = basel_irb_capital.assign_rating_grade(np.array([0.5, 1.0]))
    assert (grades == "D").all()


def test_assign_rating_grade_is_monotonic_in_pd(basel_irb_capital):
    pds = np.array([0.0001, 0.0004, 0.0008, 0.002, 0.02, 0.2])
    grades = basel_irb_capital.assign_rating_grade(pds)
    grade_rank = {g: i for i, (g, _) in enumerate(basel_irb_capital.RATING_SCALE)}
    ranks = [grade_rank[g] for g in grades]
    assert ranks == sorted(ranks)


# ── Basel IRB capital formula ─────────────────────────────────────────────────

def test_capital_k_increases_with_pd(basel_irb_capital):
    k = basel_irb_capital.basel_irb_capital_k(
        np.array([0.001, 0.01, 0.05, 0.20]), lgd=0.35
    )
    assert (np.diff(k) > 0).all()


def test_capital_k_is_zero_at_zero_pd(basel_irb_capital):
    k = basel_irb_capital.basel_irb_capital_k(np.array([1e-9]), lgd=0.35)
    assert k[0] == pytest.approx(0.0, abs=1e-3)


def test_capital_k_stays_within_zero_and_lgd(basel_irb_capital):
    # K = LGD * N[...] - PD*LGD is always <= LGD (unexpected loss capital
    # can never exceed total loss-given-default) and never negative.
    pds = np.linspace(0.0001, 0.99, 50)
    k = basel_irb_capital.basel_irb_capital_k(pds, lgd=0.35)
    assert (k >= 0.0).all()
    assert (k <= 0.35 + 1e-9).all()


def test_compute_rwa_matches_12_5_times_k_times_ead(basel_irb_capital):
    k = np.array([0.08])
    ead = np.array([100_000.0])
    rwa = basel_irb_capital.compute_rwa(k, ead)
    assert rwa[0] == pytest.approx(0.08 * 12.5 * 100_000.0)


def test_capital_required_is_8pct_of_rwa_by_construction(basel_irb_capital):
    # RWA * MIN_CAP_RATIO (8%) should reproduce K * EAD exactly, since
    # RWA = K * (1/MIN_CAP_RATIO) * EAD.
    k = np.array([0.05, 0.10])
    ead = np.array([200_000.0, 50_000.0])
    rwa = basel_irb_capital.compute_rwa(k, ead)
    capital = rwa * basel_irb_capital.MIN_CAP_RATIO
    np.testing.assert_allclose(capital, k * ead)


# ── load_ttc_pd: source preference and fallback ──────────────────────────────

def test_load_ttc_pd_prefers_calibration_output(basel_irb_capital, tmp_path, monkeypatch):
    monkeypatch.setattr(basel_irb_capital, "PROC_DIR", tmp_path)
    pd.DataFrame({
        "loan_seq_num": ["L1", "L2"], "split": ["oos-eval", "oot"],
        "pit_pd_platt": [0.02, 0.03], "ttc_pd": [0.015, 0.025],
    }).to_csv(tmp_path / "ttc_calibrated_pd.csv", index=False)
    pd.DataFrame({
        "loan_seq_num": ["L1"], "ttc_pd_12m": [0.999],
    }).to_csv(tmp_path / "survival_pd_horizons.csv", index=False)

    out, source = basel_irb_capital.load_ttc_pd()

    assert "ttc_calibrated_pd.csv" in source
    assert set(out["loan_seq_num"]) == {"L1", "L2"}
    assert out.set_index("loan_seq_num").loc["L1", "ttc_pd"] == pytest.approx(0.015)


def test_load_ttc_pd_falls_back_to_survival_analysis(basel_irb_capital, tmp_path, monkeypatch):
    monkeypatch.setattr(basel_irb_capital, "PROC_DIR", tmp_path)
    pd.DataFrame({
        "loan_seq_num": ["L1"], "ttc_pd_12m": [0.04],
    }).to_csv(tmp_path / "survival_pd_horizons.csv", index=False)

    out, source = basel_irb_capital.load_ttc_pd()

    assert "survival_pd_horizons.csv" in source
    assert out.set_index("loan_seq_num").loc["L1", "ttc_pd"] == pytest.approx(0.04)


def test_load_ttc_pd_returns_empty_when_neither_source_exists(basel_irb_capital, tmp_path, monkeypatch):
    monkeypatch.setattr(basel_irb_capital, "PROC_DIR", tmp_path)
    out, source = basel_irb_capital.load_ttc_pd()
    assert out.empty
    assert source == "none available"
