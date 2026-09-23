"""
Tests for the competing-risks treatment of voluntary prepayment
(12_competing_risks.py) and the surv_* dataset variant that feeds it
(01_data_preprocessing.py).

The central claim under test is an identity, not an empirical result:

    CIF_default(t)  <=  naive 1 - S_default(t)     for every t

Both sides accumulate the same cause-specific hazard increments. The CIF
weights each increment by the probability of having survived BOTH risks; the
naive figure weights it by survival from default alone, which is always at
least as large. So the naive conversion — the one 06_survival_analysis.py
performs — can only ever overstate cumulative default probability, and a run
in which it does not is a bug in the combination step rather than a finding.

Everything here runs on small synthetic arrays and frames; no Freddie Mac
data is required.
"""
import numpy as np
import pandas as pd
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _increasing_hazard(n, rate, seed=0):
    """A non-decreasing cumulative hazard on an n-point grid."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.uniform(0, rate, n))


# =============================================================================
# CUMULATIVE INCIDENCE — core identities
# =============================================================================

def test_cif_is_monotonically_non_decreasing(competing_risks):
    h_d = _increasing_hazard(60, 0.01, seed=1)
    h_p = _increasing_hazard(60, 0.05, seed=2)

    cif_d, cif_p, _ = competing_risks.cif_from_cumulative_hazards(h_d, h_p)

    assert (np.diff(cif_d) >= -1e-12).all(), "CIF_default decreased"
    assert (np.diff(cif_p) >= -1e-12).all(), "CIF_prepay decreased"


def test_cifs_sum_to_at_most_one(competing_risks):
    # Large hazards — the regime where a naive implementation overshoots.
    h_d = _increasing_hazard(200, 0.05, seed=3)
    h_p = _increasing_hazard(200, 0.20, seed=4)

    cif_d, cif_p, surv = competing_risks.cif_from_cumulative_hazards(h_d, h_p)

    assert (cif_d + cif_p <= 1.0 + 1e-9).all()
    assert (cif_d >= 0).all() and (cif_p >= 0).all()
    # The cohort is partitioned: still alive, defaulted, or prepaid.
    np.testing.assert_allclose(cif_d + cif_p + surv, 1.0, atol=1e-6)


def test_cif_never_exceeds_the_naive_one_minus_survival(competing_risks):
    """
    THE CORE CLAIM. Checked across a range of competing-hazard intensities,
    including h_p = 0 where the two must coincide exactly.
    """
    h_d = _increasing_hazard(120, 0.02, seed=5)

    for prepay_rate in [0.0, 0.001, 0.01, 0.05, 0.2, 0.5]:
        h_p = (np.zeros_like(h_d) if prepay_rate == 0
               else _increasing_hazard(120, prepay_rate, seed=6))
        cif_d, _, _ = competing_risks.cif_from_cumulative_hazards(h_d, h_p)
        naive = competing_risks.naive_one_minus_survival(h_d)

        assert (naive >= cif_d - 1e-12).all(), (
            f"naive fell below the CIF at prepay_rate={prepay_rate} — "
            f"max violation {np.max(cif_d - naive):.3e}"
        )


def test_naive_equals_cif_exactly_when_there_is_no_competing_risk(competing_risks):
    """
    With no prepayment the competing-risks correction must vanish: the two
    quantities are then the same integral. This is what pins the two sides
    to a common discretisation — comparing a left-endpoint sum against the
    closed form 1 - exp(-H) leaves an O(dH^2) gap that has nothing to do
    with competing risks and can flip the inequality.
    """
    h_d = _increasing_hazard(80, 0.03, seed=7)
    zero = np.zeros_like(h_d)

    cif_d, cif_p, _ = competing_risks.cif_from_cumulative_hazards(h_d, zero)
    naive = competing_risks.naive_one_minus_survival(h_d)

    np.testing.assert_allclose(cif_d, naive, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cif_p, 0.0, atol=1e-15)


def test_overstatement_grows_with_the_competing_hazard(competing_risks):
    """The heavier the prepayment, the more the naive figure overstates."""
    h_d = _increasing_hazard(100, 0.02, seed=8)
    naive = competing_risks.naive_one_minus_survival(h_d)

    gaps = []
    for rate in [0.01, 0.05, 0.15, 0.40]:
        h_p = _increasing_hazard(100, rate, seed=9)
        cif_d, _, _ = competing_risks.cif_from_cumulative_hazards(h_d, h_p)
        gaps.append(float(naive[-1] - cif_d[-1]))

    assert gaps == sorted(gaps), f"overstatement not monotone in prepay hazard: {gaps}"


def test_cif_rejects_misaligned_hazard_grids(competing_risks):
    with pytest.raises(ValueError, match="align"):
        competing_risks.cif_from_cumulative_hazards(np.zeros(10), np.zeros(11))


# =============================================================================
# DISCRETE-TIME CIF — the multinomial constraint
# =============================================================================

def test_discrete_hazards_from_a_softmax_always_sum_below_one(competing_risks):
    """
    The structural reason the challenger is a multinomial rather than two
    independent binary logits: a shared softmax denominator makes
    h_d + h_p < 1 an algebraic guarantee, whatever the coefficients.
    """
    rng = np.random.default_rng(11)
    logits = rng.normal(0, 5.0, size=(500, 3))   # deliberately extreme
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    h_d, h_p = probs[:, 1], probs[:, 2]

    assert (h_d + h_p < 1.0).all()
    # And the CIF machinery accepts them without complaint.
    cif_d, cif_p, surv = competing_risks.discrete_cif(h_d, h_p)
    assert (cif_d + cif_p <= 1.0 + 1e-9).all()
    assert (surv >= 0).all()


def test_discrete_cif_rejects_hazards_that_sum_to_one_or_more(competing_risks):
    """
    Two independent binary models CAN produce this — each is free to predict
    0.7 — and the implied per-period survival then goes negative, silently
    corrupting every later term of the product. It must raise, not coerce.
    """
    h_d = np.array([0.10, 0.70, 0.10])
    h_p = np.array([0.10, 0.45, 0.10])   # period 2 sums to 1.15

    with pytest.raises(ValueError, match="softmax|>= 1"):
        competing_risks.discrete_cif(h_d, h_p)


def test_discrete_cif_matches_a_hand_computed_two_period_example(competing_risks):
    h_d = np.array([0.10, 0.20])
    h_p = np.array([0.30, 0.10])

    cif_d, cif_p, surv = competing_risks.discrete_cif(h_d, h_p)

    # Period 1: S(0) = 1 -> CIF_d(1) = 0.10, CIF_p(1) = 0.30, S(1) = 0.60
    # Period 2: CIF_d(2) = 0.10 + 0.20*0.60 = 0.22
    #           CIF_p(2) = 0.30 + 0.10*0.60 = 0.36
    #           S(2) = 0.60 * (1 - 0.30) = 0.42
    np.testing.assert_allclose(cif_d, [0.10, 0.22])
    np.testing.assert_allclose(cif_p, [0.30, 0.36])
    np.testing.assert_allclose(surv, [0.60, 0.42])
    np.testing.assert_allclose(cif_d + cif_p + surv, 1.0)


def test_discrete_cif_is_non_decreasing_and_bounded(competing_risks):
    rng = np.random.default_rng(13)
    h_d = rng.uniform(0, 0.02, 200)
    h_p = rng.uniform(0, 0.10, 200)

    cif_d, cif_p, surv = competing_risks.discrete_cif(h_d, h_p)

    assert (np.diff(cif_d) >= -1e-15).all()
    assert (np.diff(cif_p) >= -1e-15).all()
    assert (np.diff(surv) <= 1e-15).all()          # survival only falls
    assert (cif_d + cif_p <= 1.0 + 1e-9).all()


def test_discrete_naive_also_dominates_the_discrete_cif(competing_risks):
    """The same inequality must hold in the discrete formulation."""
    rng = np.random.default_rng(17)
    h_d = rng.uniform(0, 0.02, 120)
    h_p = rng.uniform(0, 0.15, 120)

    cif_d, _, _ = competing_risks.discrete_cif(h_d, h_p)
    naive = 1.0 - np.cumprod(1.0 - h_d)

    assert (naive >= cif_d - 1e-12).all()


# =============================================================================
# EXPECTED LIFE
# =============================================================================

def test_expected_life_is_the_area_under_the_survival_curve(competing_risks):
    # A loan certain to survive 5 months then certain to exit has an
    # expected life of exactly 5 months.
    survival = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    assert competing_risks.expected_life_months(survival) == pytest.approx(5.0)


def test_expected_life_shortens_as_the_prepayment_hazard_rises(competing_risks):
    """
    The whole point for IFRS 9: heavier prepayment means the loan leaves the
    balance sheet sooner, so lifetime ECL should be accumulated over a
    shorter horizon than the contractual term.
    """
    h_d = np.full(240, 0.001)
    lives = []
    for prepay in [0.001, 0.005, 0.02]:
        _, _, surv = competing_risks.discrete_cif(h_d, np.full(240, prepay))
        lives.append(competing_risks.expected_life_months(surv))

    assert lives == sorted(lives, reverse=True), f"expected life not falling: {lives}"
    # And all are far below the 240-month contractual horizon.
    assert max(lives) < 240


# =============================================================================
# STEP INTERPOLATION
# =============================================================================

def test_interpolate_at_is_a_right_continuous_step_lookup(competing_risks):
    times = np.array([1, 5, 10, 20])
    values = np.array([0.1, 0.2, 0.3, 0.4])

    out = competing_risks.interpolate_at(times, values, [0, 1, 4, 5, 12, 25])

    # Before the first grid point -> 0; otherwise the value at the largest
    # time <= target. Linear interpolation here would invent incidence
    # between event times that did not occur.
    np.testing.assert_allclose(out, [0.0, 0.1, 0.1, 0.2, 0.3, 0.4])


# =============================================================================
# EVENT TYPE CONSTRUCTION  (01_data_preprocessing.py)
# =============================================================================

def _perf_rows(loan, start, n_months, terminal_code=None):
    """
    Servicing rows for one loan.

    orig_date sits one month BEFORE the first reporting period, matching how
    clean_orig() derives it (first_payment_date - 1 month). The first
    servicing row therefore lands at period_month = 1.
    """
    dates = pd.date_range(start, periods=n_months, freq="MS")
    orig = pd.Timestamp(start) - pd.DateOffset(months=1)
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "loan_seq_num": loan,
            "report_date": d,
            "orig_date": orig,
            "zero_balance_code": (terminal_code if (terminal_code and i == n_months - 1)
                                  else np.nan),
            "orig_interest_rate": 5.0,
        })
    return rows


def test_prepayment_code_01_maps_to_event_type_2(preprocessing):
    df = pd.DataFrame(_perf_rows("L1", "2015-01-01", 6, terminal_code="01"))
    out = preprocessing.extract_survival_rows(df)
    loan = out.drop_duplicates("loan_seq_num").iloc[0]

    assert loan["event_type"] == 2, "code 01 must be a prepayment EVENT, not censoring"
    assert loan["duration_months"] == 6   # 6 reporting periods at risk


@pytest.mark.parametrize("code", ["02", "03", "06", "09", "15"])
def test_default_codes_map_to_event_type_1(preprocessing, code):
    df = pd.DataFrame(_perf_rows("L1", "2015-01-01", 4, terminal_code=code))
    out = preprocessing.extract_survival_rows(df)
    assert out.drop_duplicates("loan_seq_num").iloc[0]["event_type"] == 1


def test_still_active_loan_maps_to_event_type_0(preprocessing):
    df = pd.DataFrame(_perf_rows("L1", "2015-01-01", 8, terminal_code=None))
    out = preprocessing.extract_survival_rows(df)
    loan = out.drop_duplicates("loan_seq_num").iloc[0]

    assert loan["event_type"] == 0
    assert loan["duration_months"] == 8
    assert len(out) == 8, "a censored loan contributes all of its at-risk months"


def test_non_terminal_codes_do_not_create_an_event(preprocessing):
    # 16 (reperforming) and 96 (non-standard) are neither default nor
    # prepayment — config excludes both from DEFAULT_CODES and PREPAY_CODES.
    for code in ["16", "96"]:
        df = pd.DataFrame(_perf_rows("L1", "2015-01-01", 5, terminal_code=code))
        out = preprocessing.extract_survival_rows(df)
        assert out.drop_duplicates("loan_seq_num").iloc[0]["event_type"] == 0


def test_terminal_row_carries_the_event_and_earlier_rows_do_not(preprocessing):
    df = pd.DataFrame(_perf_rows("L1", "2015-01-01", 5, terminal_code="03"))
    out = preprocessing.extract_survival_rows(df).sort_values("period_month")

    assert (out["period_event"].to_numpy() == [0, 0, 0, 0, 1]).all()
    assert out["period_month"].tolist() == [1, 2, 3, 4, 5]


def test_duration_is_derived_from_dates_not_loan_age(preprocessing):
    """
    loan_age RESETS when a loan is modified, so a loan modified at month 30
    can report loan_age = 1 the following month. Using it as the survival
    clock rewinds time for exactly the distressed loans whose timing matters
    most. duration_months must come from report_date - orig_date.
    """
    rows = _perf_rows("L1", "2015-01-01", 36, terminal_code="03")
    for i, r in enumerate(rows):
        # loan_age resets to 0 at month 30 — the modification
        r["loan_age"] = i if i < 30 else i - 30
        r["modification_flag"] = "N" if i < 30 else "Y"
    out = preprocessing.extract_survival_rows(pd.DataFrame(rows))
    loan = out.drop_duplicates("loan_seq_num").iloc[0]

    assert loan["duration_months"] == 36, "duration must ignore the loan_age reset"
    # The reset is visible in the covariate but not in the clock.
    last = out.sort_values("period_month").iloc[-1]
    assert last["period_month"] == 36
    assert last["loan_age"] == 5


def test_earliest_cause_wins_when_a_loan_carries_both_codes(preprocessing):
    rows = _perf_rows("L1", "2015-01-01", 6)
    rows[2]["zero_balance_code"] = "01"   # prepay at month 3
    rows[5]["zero_balance_code"] = "03"   # default later — impossible in reality
    out = preprocessing.extract_survival_rows(pd.DataFrame(rows))
    loan = out.drop_duplicates("loan_seq_num").iloc[0]

    assert loan["event_type"] == 2, "the earlier event terminates the loan"
    assert loan["duration_months"] == 3


def test_no_rows_survive_past_the_terminating_event(preprocessing):
    rows = _perf_rows("L1", "2015-01-01", 10)
    rows[3]["zero_balance_code"] = "01"
    out = preprocessing.extract_survival_rows(pd.DataFrame(rows))

    assert len(out) == 4, "rows after the event must be dropped"
    assert out["report_date"].max() == pd.Timestamp("2015-04-01")


def test_reo_acquisition_is_preserved_not_silently_nulled(preprocessing):
    """
    delinquency_status is a STRING carrying "RA" (REO acquisition) and "XX"
    (not available). A bare to_numeric maps both to NaN, conflating a
    terminal credit state with an unknown one.
    """
    months, is_reo = preprocessing.clean_delinquency_status(
        pd.Series(["0", "X", "3", "RA", "XX", "R", " ra "])
    )
    assert months.tolist()[:3] == [0.0, 0.0, 3.0]
    assert np.isnan(months.iloc[3]) and np.isnan(months.iloc[4])
    assert is_reo.tolist() == [0, 0, 0, 1, 0, 1, 1]


def test_event_type_codes_match_config(preprocessing, competing_risks):
    import config
    assert config.PREPAY_CODES == {"01"}
    assert config.EVENT_CENSORED == 0
    assert config.EVENT_DEFAULT == 1
    assert config.EVENT_PREPAY == 2
    # Prepayment must not be inside the default set, or the competing risk
    # would be counted as a credit loss.
    assert not (config.PREPAY_CODES & config.DEFAULT_CODES)


# =============================================================================
# SPLIT INTEGRITY
# =============================================================================

def test_no_loan_history_spans_two_survival_splits(preprocessing):
    """
    A survival duration is a property of a loan's whole history. split_pd()
    cuts OOT by row, which is harmless for a snapshot classifier but would
    truncate a straddling loan's duration in one file and start it mid-flight
    in the other. split_survival() therefore assigns whole loans.
    """
    frames = []
    for i, start in enumerate(["2014-01-01", "2016-01-01", "2018-01-01"]):
        for j in range(12):
            rows = _perf_rows(f"L{i}{j}", start, 8, terminal_code="01")
            frames.extend(rows)
    panel = preprocessing.extract_survival_rows(pd.DataFrame(frames))

    train, oos, oot = preprocessing.split_survival(panel)

    sets = [set(d["loan_seq_num"]) for d in (train, oos, oot) if len(d)]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            assert not (sets[a] & sets[b]), "a loan appears in two splits"


def test_survival_split_uses_the_same_oot_cutoff_constant(preprocessing):
    import config
    frames = []
    for i, start in enumerate(["2010-01-01", "2020-01-01"]):
        for j in range(12):
            frames.extend(_perf_rows(f"L{i}{j}", start, 6, terminal_code="01"))
    panel = preprocessing.extract_survival_rows(pd.DataFrame(frames))

    _, _, oot = preprocessing.split_survival(panel)

    assert len(oot) > 0
    assert (oot["orig_date"] >= config.OOT_CUTOFF).all()


# =============================================================================
# CH.6 EXPECTED-LIFE INTEGRATION
# =============================================================================

def test_amortized_ead_truncates_exposure_beyond_expected_life(macro_scenario):
    loans = pd.DataFrame({
        "current_upb": [300_000.0],
        "current_interest_rate": [5.0],
        "remaining_months": [340.0],     # contractual: ~28 years
    })

    contractual = macro_scenario.amortized_ead(loans, n_quarters=20)
    truncated = macro_scenario.amortized_ead(
        loans, n_quarters=20, expected_life_months=np.array([24.0]))

    # Quarters 1-8 are within the 24-month expected life; both agree there.
    np.testing.assert_allclose(contractual[0, :8], truncated[0, :8])
    # Beyond it, exposure is zero under expected life but still large under
    # the contractual term — this is the IFRS 9 §5.5.19 correction.
    assert (truncated[0, 8:] == 0).all()
    assert (contractual[0, 8:] > 0).all()


def test_amortized_ead_is_unchanged_when_expected_life_is_not_supplied(macro_scenario):
    """config.IFRS9_USE_EXPECTED_LIFE = False must reproduce the old
    behaviour exactly."""
    loans = pd.DataFrame({
        "current_upb": [250_000.0],
        "current_interest_rate": [4.5],
        "remaining_months": [300.0],
    })
    a = macro_scenario.amortized_ead(loans, n_quarters=20)
    b = macro_scenario.amortized_ead(loans, n_quarters=20, expected_life_months=None)
    np.testing.assert_array_equal(a, b)


def test_expected_life_truncation_lowers_total_projected_exposure(macro_scenario):
    loans = pd.DataFrame({
        "current_upb": [400_000.0, 150_000.0],
        "current_interest_rate": [6.0, 3.5],
        "remaining_months": [355.0, 280.0],
    })
    contractual = macro_scenario.amortized_ead(loans, 20)
    truncated = macro_scenario.amortized_ead(
        loans, 20, expected_life_months=np.array([36.0, 48.0]))

    assert truncated.sum() < contractual.sum()
    assert (truncated <= contractual + 1e-9).all()
