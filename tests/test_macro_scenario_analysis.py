"""
Tests for 07_macro_scenario_analysis.py — the macro path construction and
the IFRS 9 ECL accumulation formula (quarterly PD x survival probability x
LGD x EAD x discount factor), which is the core regulatory-facing output
of this script.

compute_ifrs9_ecl() needs a fitted XGBClassifier internally via
score_quarter(); we monkeypatch score_quarter() to return a fixed PD per
quarter so the test isolates the accumulation math from model behaviour.
"""
import numpy as np
import pandas as pd
import pytest


def test_load_base_lgd_falls_back_when_champion_summary_is_missing(macro_scenario, tmp_path, monkeypatch):
    monkeypatch.setattr(macro_scenario, "PROC_DIR", tmp_path)  # empty dir — no CSV
    base_lgd, source = macro_scenario.load_base_lgd()
    assert base_lgd == macro_scenario.config.MACRO_LGD_ASSUMPTION
    assert "fallback" in source


def test_load_base_lgd_uses_champion_summary_when_present(macro_scenario, tmp_path, monkeypatch):
    monkeypatch.setattr(macro_scenario, "PROC_DIR", tmp_path)
    pd.DataFrame([{
        "champion_model": "XGBoost", "anchor_split": "OOS",
        "anchor_mean_lgd": 0.28, "n_anchor_obs": 42,
        "rmse": 0.15, "mae": 0.12, "r2": 0.4, "bias": 0.01,
    }]).to_csv(tmp_path / "lgd_champion_summary.csv", index=False)

    base_lgd, source = macro_scenario.load_base_lgd()

    assert base_lgd == pytest.approx(0.28)
    assert "XGBoost" in source


def test_build_macro_paths_shape_and_scenario_weights_sum_to_one(macro_scenario):
    macro_df = macro_scenario.build_macro_paths()

    n_scenarios = len(macro_scenario.SCENARIOS)
    assert len(macro_df) == n_scenarios * (macro_scenario.N_QUARTERS + 1)

    weights = [cfg["weight"] for cfg in macro_scenario.SCENARIOS.values()]
    assert sum(weights) == pytest.approx(1.0)

    # quarter 0 is the current/starting point for every scenario
    q0 = macro_df[macro_df["quarter"] == 0]
    assert (q0["ur"] == 4.0).all()
    assert (q0["hpi_ratio"] == 1.0).all()


def test_build_macro_paths_unemployment_never_goes_below_floor(macro_scenario):
    macro_df = macro_scenario.build_macro_paths()
    assert (macro_df["ur"] >= 0.5).all()
    assert (macro_df["hpi_ratio"] >= 0.1).all()


def test_severe_scenario_stresses_unemployment_more_than_base(macro_scenario):
    macro_df = macro_scenario.build_macro_paths()
    peak_ur = macro_df.groupby("scenario")["ur"].max()
    assert peak_ur["Severe"] > peak_ur["Adverse"] > peak_ur["Base"]


def test_scenario_lgd_reproduces_base_lgd_at_hpi_ratio_one(macro_scenario):
    assert macro_scenario.scenario_lgd(0.35, 1.0) == pytest.approx(0.35)


def test_scenario_lgd_rises_when_collateral_value_falls(macro_scenario):
    stressed = macro_scenario.scenario_lgd(0.35, 0.74)   # -26% HPI, Severe trough
    assert stressed > 0.35


def test_scenario_lgd_falls_when_collateral_value_appreciates(macro_scenario):
    appreciated = macro_scenario.scenario_lgd(0.35, 1.10)  # +10% HPI
    assert appreciated < 0.35


def test_scenario_lgd_stays_within_unit_interval(macro_scenario):
    assert 0.0 <= macro_scenario.scenario_lgd(0.99, 0.01) <= 1.0
    assert 0.0 <= macro_scenario.scenario_lgd(0.01, 5.0) <= 1.0


def test_ecl_accumulation_matches_the_documented_formula(macro_scenario, monkeypatch):
    # Constant conditional PD every quarter, independent of macro conditions,
    # isolates the ECL loop's use of PD from the survival-probability,
    # scenario-conditional LGD, EAD amortization, and discounting logic that
    # wraps it.
    constant_pd = 0.02

    def stub_score_quarter(xgb, oos_df, macro_row, feats, imputer, quarter):
        return np.full(len(oos_df), constant_pd)

    monkeypatch.setattr(macro_scenario, "score_quarter", stub_score_quarter)

    oos_df = pd.DataFrame({
        "loan_seq_num": ["L1", "L2"],
        "orig_upb": [200_000.0, 100_000.0],
        "current_upb": [190_000.0, 95_000.0],
        "current_interest_rate": [5.5, 6.0],
        "remaining_months": [300.0, 180.0],
    })
    macro_df = macro_scenario.build_macro_paths()
    horizons = [4, 8]  # 1Y, 2Y
    base_lgd = macro_scenario.LGD

    ecl_by_loan, ecl_summary, all_results, all_sp = macro_scenario.compute_ifrs9_ecl(
        xgb=None, oos_df=oos_df, macro_df=macro_df,
        imputer=None, feats=[], horizons=horizons, base_lgd=base_lgd,
    )

    r = macro_scenario.DISCOUNT_R
    # Track the documented behavior via the module's own amortized_ead(),
    # not a hand-rolled re-derivation of the formula.
    ead_mat = macro_scenario.amortized_ead(oos_df, macro_scenario.N_QUARTERS)

    base_macro = macro_df[macro_df["scenario"] == "Base"].sort_values("quarter")
    hpi_by_q = {int(row["quarter"]): row["hpi_ratio"] for _, row in base_macro.iterrows()}

    for h, label in [(4, "1Y"), (8, "2Y")]:
        sp = 1.0
        expected = np.zeros(len(oos_df))
        for q in range(1, h + 1):
            df_q  = 1.0 / (1 + r) ** (q / 4)
            lgd_q = macro_scenario.scenario_lgd(base_lgd, hpi_by_q[q])
            expected += constant_pd * sp * lgd_q * ead_mat[:, q - 1] * df_q
            sp *= (1 - constant_pd)

        actual = ecl_by_loan[f"ecl_base_{label}"].values
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # ECL must grow with horizon (every added term is non-negative) ...
    assert (ecl_by_loan["ecl_base_2Y"] >= ecl_by_loan["ecl_base_1Y"]).all()

    # ... but stay strictly below the naive no-survival/no-discount/no-HPI/
    # no-amortization bound, since SP(q-1) <= 1, DF(q) <= 1, EAD(q) <=
    # current_upb, and Base-scenario HPI only appreciates (so
    # scenario_lgd(base_lgd, hpi_ratio) <= base_lgd throughout).
    naive_upper_bound = constant_pd * base_lgd * oos_df["current_upb"].values * 8
    assert (ecl_by_loan["ecl_base_2Y"].values < naive_upper_bound).all()

    # Survival probability must be monotonically non-increasing each quarter.
    sp_base = all_sp["Base"]
    assert (np.diff(sp_base, axis=1) <= 1e-12).all()


# ── amortized_ead: EAD amortization ───────────────────────────────────────────

def test_amortized_ead_declines_monotonically(macro_scenario):
    oos_df = pd.DataFrame({
        "current_upb": [200_000.0],
        "current_interest_rate": [5.0],
        "remaining_months": [300],
    })
    ead = macro_scenario.amortized_ead(oos_df, n_quarters=20)
    assert (np.diff(ead[0]) <= 1e-6).all()
    assert ead[0, 0] < 200_000.0


def test_amortized_ead_hits_zero_after_remaining_months(macro_scenario):
    oos_df = pd.DataFrame({
        "current_upb": [200_000.0],
        "current_interest_rate": [5.0],
        "remaining_months": [10],   # matures within ~3 quarters
    })
    ead = macro_scenario.amortized_ead(oos_df, n_quarters=20)
    assert ead[0, -1] == 0.0


def test_amortized_ead_falls_back_when_columns_missing(macro_scenario):
    # No current_upb / current_interest_rate / remaining_months / loan_age
    # at all — must not raise, and should still amortize (not stay flat).
    oos_df = pd.DataFrame({"orig_upb": [150_000.0]})
    ead = macro_scenario.amortized_ead(oos_df, n_quarters=20)
    assert ead[0, 0] <= 150_000.0
    assert ead[0, -1] < ead[0, 0]


# ── IFRS 9 staging: assign_ifrs9_stage ────────────────────────────────────────

def test_stage1_when_pd_stable_and_current(macro_scenario):
    stage = macro_scenario.assign_ifrs9_stage(
        pd_current=np.array([0.01]),
        pd_origination=np.array([0.01]),
        delinquency_status=np.array([0.0]),
    )
    assert stage[0] == 1


def test_stage2_on_relative_pd_deterioration(macro_scenario):
    # PD has more than doubled (config.SICR_PD_RATIO=2.0) and the absolute
    # move clears the noise floor (config.SICR_PD_ABS_FLOOR=0.02) — SICR
    # triggers even though the loan is fully current (no delinquency).
    stage = macro_scenario.assign_ifrs9_stage(
        pd_current=np.array([0.10]),
        pd_origination=np.array([0.02]),
        delinquency_status=np.array([0.0]),
    )
    assert stage[0] == 2


def test_stage1_when_relative_ratio_trips_but_absolute_floor_does_not(macro_scenario):
    # Ratio is 3x (>= SICR_PD_RATIO) but the absolute move is only 0.002pp,
    # well under SICR_PD_ABS_FLOOR=0.02 — noise, not a real SICR event.
    stage = macro_scenario.assign_ifrs9_stage(
        pd_current=np.array([0.003]),
        pd_origination=np.array([0.001]),
        delinquency_status=np.array([0.0]),
    )
    assert stage[0] == 1


def test_stage2_backstop_on_30_dpd_even_without_pd_deterioration(macro_scenario):
    # PD is unchanged since origination, but the loan is 30+ days past due
    # (delinquency_status >= config.STAGE2_DPD_MONTHS=1) — the rebuttable
    # presumption backstop fires regardless of the relative PD test.
    stage = macro_scenario.assign_ifrs9_stage(
        pd_current=np.array([0.01]),
        pd_origination=np.array([0.01]),
        delinquency_status=np.array([1.0]),
    )
    assert stage[0] == 2


def test_stage3_on_90_dpd_backstop_overrides_stage2(macro_scenario):
    stage = macro_scenario.assign_ifrs9_stage(
        pd_current=np.array([0.01]),
        pd_origination=np.array([0.01]),
        delinquency_status=np.array([3.0]),
    )
    assert stage[0] == 3


def test_assign_ifrs9_stage_is_vectorised_across_loans(macro_scenario):
    stage = macro_scenario.assign_ifrs9_stage(
        pd_current=np.array([0.01, 0.10, 0.01]),
        pd_origination=np.array([0.01, 0.02, 0.01]),
        delinquency_status=np.array([0.0, 0.0, 3.0]),
    )
    np.testing.assert_array_equal(stage, [1, 2, 3])


# ── IFRS 9 staging: compute_origination_pd ────────────────────────────────────

def test_compute_origination_pd_forces_loan_age_to_zero(macro_scenario):
    seen_ages = {}

    class StubXGB:
        def predict_proba(self, X):
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    class StubImputer:
        def transform(self, df):
            seen_ages["loan_age"] = df["loan_age"].tolist()
            return df.values.astype(float)

    oos_df = pd.DataFrame({"loan_age": [36.0, 120.0]})
    macro_scenario.compute_origination_pd(
        StubXGB(), oos_df, feats=["loan_age"], imputer=StubImputer()
    )

    assert seen_ages["loan_age"] == [0, 0]
    # Original frame must be untouched (a defensive copy, not a mutation).
    assert oos_df["loan_age"].tolist() == [36.0, 120.0]


# ── IFRS 9 staging: compute_staged_ecl ────────────────────────────────────────

def test_compute_staged_ecl_picks_12m_for_stage1_and_lifetime_for_stage23(macro_scenario):
    horizon_labels = {4: "1Y", macro_scenario.N_QUARTERS: "5Y"}
    ecl_by_loan = pd.DataFrame({
        "loan_seq_num": ["L1", "L2", "L3"],
        "ecl_base_1Y":   [10.0, 20.0, 30.0],
        "ecl_base_5Y":   [100.0, 200.0, 300.0],
        "ecl_adverse_1Y": [11.0, 21.0, 31.0],
        "ecl_adverse_5Y": [110.0, 210.0, 310.0],
        "ecl_severe_1Y":  [12.0, 22.0, 32.0],
        "ecl_severe_5Y":  [120.0, 220.0, 320.0],
    })
    stage = np.array([1, 2, 3])

    staged = macro_scenario.compute_staged_ecl(ecl_by_loan, stage, horizon_labels)

    # Stage 1 (L1) takes the 1Y column; Stage 2/3 (L2, L3) take the 5Y column.
    assert staged.loc[0, "staged_ecl_base"] == 10.0
    assert staged.loc[1, "staged_ecl_base"] == 200.0
    assert staged.loc[2, "staged_ecl_base"] == 300.0

    # Weighted ECL matches the documented probability weights.
    weights = {s: cfg["weight"] for s, cfg in macro_scenario.SCENARIOS.items()}
    expected_l1 = (weights["Base"] * 10.0 + weights["Adverse"] * 11.0
                   + weights["Severe"] * 12.0)
    assert staged.loc[0, "staged_ecl_weighted"] == pytest.approx(expected_l1)
