"""
Tests for 08_calibration.py's compute_ttc_pd_via_lradr() — the LRADR
anchoring cycle-adjustment that turns a calibrated point-in-time (PIT) PD
into an actual per-loan through-the-cycle (TTC) PD, rather than leaving
LRADR as a diagnostic comparison between two aggregate numbers.
"""
import numpy as np


def test_ttc_pd_mean_moves_toward_lradr(calibration):
    rng = np.random.default_rng(0)
    y_true = np.zeros(1000)
    y_true[:10] = 1  # 1% long-run default rate
    pit_pd = rng.uniform(0.01, 0.05, 1000)  # mean well above 1%

    ttc_pd, lradr, shift = calibration.compute_ttc_pd_via_lradr(y_true, pit_pd)

    assert lradr == y_true.mean()
    # TTC mean must land much closer to LRADR than the raw PIT mean did.
    assert abs(ttc_pd.mean() - lradr) < abs(pit_pd.mean() - lradr)
    assert ttc_pd.mean() < pit_pd.mean()  # shifted down, since LRADR < mean(PIT)


def test_ttc_pd_preserves_rank_order(calibration):
    pit_pd = np.array([0.01, 0.05, 0.02, 0.20, 0.001])
    y_true = np.zeros(100)
    y_true[:10] = 1  # 10% LRADR, deliberately a different sample than pit_pd

    ttc_pd, _, _ = calibration.compute_ttc_pd_via_lradr(y_true, pit_pd)

    assert np.argsort(pit_pd).tolist() == np.argsort(ttc_pd).tolist()


def test_ttc_pd_is_identity_when_lradr_equals_mean_pit(calibration):
    pit_pd = np.full(100, 0.05)
    y_true = np.zeros(100)
    y_true[:5] = 1  # 5% observed rate == mean(pit_pd)

    ttc_pd, lradr, shift = calibration.compute_ttc_pd_via_lradr(y_true, pit_pd)

    assert abs(shift) < 1e-9
    np.testing.assert_allclose(ttc_pd, pit_pd, atol=1e-9)


def test_ttc_pd_stays_within_unit_interval(calibration):
    rng = np.random.default_rng(1)
    pit_pd = rng.uniform(0.0, 1.0, 500)
    y_true = rng.integers(0, 2, 500).astype(float)

    ttc_pd, _, _ = calibration.compute_ttc_pd_via_lradr(y_true, pit_pd)

    assert (ttc_pd >= 0.0).all() and (ttc_pd <= 1.0).all()
