"""인과 엔진: 추정기는 독립 구현과 대조하고, 판정 규칙은 분기마다 고정한다."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from jverdict.causal import PanelDesign, SeriesDesign, analyze_panel, analyze_series
from jverdict.causal.checks import (PlaceboSpace, PlaceboTime, Sensitivity, DoubleCompute, double_compute_2x2,
                                    placebo_space, placebo_time, series_means_sql)
from jverdict.causal.estimate import Estimate, did, event_study, gap_series, its_fit
from jverdict.causal.power import assess
from jverdict.causal.stats import absorb, ar1_phi, fe_ols, fixedb_cv, wild_cluster
from jverdict.causal.verdict import judge
from jverdict.quality import check_coverage, exclusion_mask, to_wide
from jverdict.synth.generator import Effect, MeasurementBreak, PreTrend, Shock, SynthSpec, generate
from jverdict.synth.scenarios import BUS_GROUPS

D = "2024-03-04"


def _wide(inj, seed=5):
    sd = generate(SynthSpec("2021-01-01", "2025-10-31", BUS_GROUPS, injections=inj), seed)
    cov = check_coverage(sd.df, "time", "group", "value", freq="D")
    w = to_wide(sd.df, "time", "group", "value", "D")
    return w, exclusion_mask(cov, w.index)


def _design(**kw):
    return SeriesDesign("q", "청소년", "일반", D, "D", expected="increase", placebo_groups=["경로", "어린이"], **kw)


def _random_panel(seed=0, n_units=30, n_t=12, drop=0.1):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        tr = int(u < n_units // 3)
        a = rng.normal()
        for t in range(n_t):
            if rng.random() < drop:
                continue
            rows.append({"unit": f"u{u}", "t": t, "tr": tr, "x2": rng.normal(),
                         "y": a + 0.1 * t + 0.4 * tr * (t >= 6) + rng.normal(0, 0.3)})
    d = pd.DataFrame(rows)
    d["time"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(d.t * 31, unit="D")
    d["time"] = d.time.dt.to_period("M").dt.to_timestamp()
    return d


# ------------------------------------------------------------------ 통계 핵심부
def test_fe_ols_matches_dummy_regression_unbalanced():
    d = _random_panel()
    d["tp"] = d.tr * (d.t >= 6)
    ref = smf.ols("y ~ tp + x2 + C(unit) + C(t)", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.unit)[0]})
    r = fe_ols(d.y.to_numpy(), d[["tp", "x2"]].to_numpy(float), ["tp", "x2"], [d.unit.to_numpy(), d.t.to_numpy()],
               d.unit.to_numpy())
    assert r.coef("tp")[0] == pytest.approx(ref.params["tp"], abs=1e-8)
    assert r.coef("tp")[1] == pytest.approx(ref.bse["tp"], rel=1e-6)
    assert r.coef("x2")[1] == pytest.approx(ref.bse["x2"], rel=1e-6)


def test_absorb_removes_both_fixed_effects():
    d = _random_panel(1)
    z = absorb(d.y.to_numpy(), [d.unit.to_numpy(), d.t.to_numpy()])[:, 0]
    g = pd.Series(z).groupby(d.unit.to_numpy()).mean()
    h = pd.Series(z).groupby(d.t.to_numpy()).mean()
    assert np.abs(g).max() < 1e-8 and np.abs(h).max() < 1e-8


def test_wild_bootstrap_ci_contains_estimate_and_rejects_real_effect():
    d = _random_panel(2)
    d["tp"] = (d.tr * (d.t >= 6)).astype(float)
    r = fe_ols(d.y.to_numpy(), d[["tp"]].to_numpy(), ["tp"], [d.unit.to_numpy(), d.t.to_numpy()], d.unit.to_numpy())
    wb = wild_cluster(r, "tp", B=1999)
    b = r.coef("tp")[0]
    assert wb["ci"][0] < b < wb["ci"][1] and wb["p"] < 0.01 and wb["treated_clusters"] == 10


def test_fixedb_and_ar1_helpers():
    assert fixedb_cv(0.0) == pytest.approx(1.96, abs=1e-3)
    assert fixedb_cv(0.1) > fixedb_cv(0.05) > 1.96
    rng = np.random.default_rng(0)
    x = np.zeros(400)
    for i in range(1, 400):
        x[i] = 0.6 * x[i - 1] + rng.normal()
    phi, n = ar1_phi(x)
    assert n == 400 and phi == pytest.approx(0.6, abs=0.08)


# ------------------------------------------------------------------ 추정기
def test_its_effect_equals_post_minus_pre_mean_and_sql_agrees():
    w, m = _wide([Effect("청소년", D, 0.2)])
    gs = gap_series(w, "청소년", "일반", "D", mask=m)
    i0 = gs.pos(D)
    e = its_fit(gs, i0, pre=600, post=150)
    assert e.effect == pytest.approx(e.details["post_mean"] - e.details["pre_mean"], abs=1e-12)
    assert series_means_sql(gs.g_adj, i0, i0 - 600, i0 + 150) == pytest.approx(e.effect, abs=1e-9)
    assert e.ci[0] < 0.2 < e.ci[1]


def test_its_trend_adjustment_removes_linear_pretrend_in_level_series():
    idx = pd.date_range("2020-01-01", "2024-12-01", freq="MS")
    t = np.arange(len(idx))
    rng = np.random.default_rng(3)
    y = np.exp(5 + 0.02 * t + rng.normal(0, 0.01, len(t)))
    w = pd.DataFrame({"a": y, "b": np.full(len(t), np.e ** 5)}, index=idx)
    gs = gap_series(w, "a", "b", "MS", season="none")
    i0 = gs.pos("2023-01-01")
    naive = its_fit(gs, i0, pre=24, post=6)
    adj = its_fit(gs, i0, pre=24, post=6, trend=True)
    assert naive.effect > 0.2 and abs(adj.effect) < 0.02


def test_did_and_event_study_on_panel():
    d = _random_panel(4, n_units=40, drop=0.0)
    e = did(d, "y", "unit", "time", "tr", d.time.sort_values().unique()[6])
    assert e.ci[0] < 0.4 < e.ci[1] and e.details["n_treat_units"] == 13
    es = event_study(d, "y", "unit", "time", "tr", d.time.sort_values().unique()[6], d.time.sort_values().unique()[5])
    assert es.pretrend_q == 5 and es.pretrend_p > 0.01
    dc = double_compute_2x2(d.assign(post=(d.t >= 6).astype(int)), "y", "unit", "tr", "post")
    assert dc.match


# ------------------------------------------------------------------ 검증 툴
def test_placebo_time_skips_forbidden_windows():
    calls = []

    def fit(i):
        calls.append(i)
        return Estimate("x", 0.01 * (i % 5), 0.1, (-1, 1), 0.5, 10, 0.05, 10, "t")
    pt = placebo_time(fit, 200, 150, real=0.5, window=(20, 10), step=1, forbidden=list(range(150, 160)))
    assert all(not (i - 20 <= 155 < i + 10) for i in calls)
    assert pt.n == len(calls) and pt.p_two == pytest.approx(1 / (pt.n + 1))


def test_placebo_space_rejects_comparable_fake_effect():
    real = Estimate("x", 0.20, 0.02, (0.16, 0.24), 0.0, 100, 0.05, 100, "t")
    big = Estimate("x", 0.15, 0.02, (0.11, 0.19), 0.0, 100, 0.05, 100, "t")
    small = Estimate("x", 0.03, 0.02, (-0.01, 0.07), 0.2, 100, 0.05, 100, "t")
    assert not placebo_space(real, {"a": big}).passed
    assert placebo_space(real, {"a": small}).passed


# ------------------------------------------------------------------ 판정 규칙 (분기별)
def _est(eff, lo, hi, p=0.001):
    return Estimate("its", eff, (hi - lo) / 3.92, (lo, hi), p, 100, 0.05, 300, "t", n_pre=200, n_post=100)


def _kw(**over):
    base = dict(pretrend_p=0.5, trend_adjusted=_est(0.19, 0.1, 0.3), placebo_time=None,
                placebo_space=PlaceboSpace(0.2, {"g": {"effect": 0.0}}, True, ""),
                sens=Sensitivity({}, 3, 3, 3), double=DoubleCompute("q", 1.0, 1.0, 1e-9, True),
                power=assess(0.01, None, 0.10, 100, "일"))
    base.update(over)
    return base


def test_rule_effect_when_all_checks_pass():
    v = judge(_est(0.2, 0.15, 0.25), **_kw())
    assert v.label == "효과 있음(증가)" and v.light == "🟢"


def test_rule_severe_pretrend_holds_unless_trend_adjusted_survives():
    v = judge(_est(0.2, 0.15, 0.25), **_kw(pretrend_p=0.001, trend_adjusted=_est(0.01, -0.1, 0.12, 0.8)))
    assert v.level == "판정 보류"
    v = judge(_est(0.2, 0.15, 0.25), **_kw(pretrend_p=0.001, trend_adjusted=_est(0.15, 0.05, 0.25)))
    assert v.label == "조건부(증가)"


def test_rule_pretrend_drift_share_fails_c1():
    v = judge(_est(0.2, 0.15, 0.25), **_kw(pretrend_p=0.3, trend_adjusted=_est(0.05, -0.05, 0.15, 0.3)))
    assert v.level == "조건부" and any(c.code == "C1" and c.status == "실패" for c in v.checks)


def test_rule_c1_and_c2_fail_holds():
    v = judge(_est(0.2, 0.15, 0.25), **_kw(pretrend_p=0.05, placebo_space=PlaceboSpace(0.2, {}, False, "")))
    assert v.level == "판정 보류"


def test_rule_null_with_power_is_no_effect_and_without_power_is_hold():
    assert judge(_est(0.01, -0.02, 0.04, 0.5), **_kw()).level == "효과 없음"
    weak = assess(0.08, None, 0.10, 100, "일")
    v = judge(_est(0.01, -0.15, 0.17, 0.9), **_kw(power=weak))
    assert v.level == "판정 보류" and v.needed_data


def test_rule_weak_signal_is_conditional():
    weak = assess(0.08, None, 0.10, 100, "일")
    v = judge(_est(0.12, -0.01, 0.25, 0.07), **_kw(power=weak))
    assert v.label == "조건부(증가)"


def test_rule_placebo_ci_widens_decision_interval():
    pt = PlaceboTime(0.2, [0.0] * 30, [], 0.2, 0.1, 0.1, (-0.05, 0.4), 30, 7)
    v = judge(_est(0.2, 0.15, 0.25), **_kw(placebo_time=pt))
    assert v.ci_used == (-0.05, 0.4) and v.level != "효과 있음"


def test_rule_quality_caps():
    assert judge(_est(0.2, 0.15, 0.25), **_kw(), quality_cap="판정 보류", quality_reason="측정 단절").level == "판정 보류"
    assert judge(_est(0.2, 0.15, 0.25), **_kw(), quality_cap="조건부", quality_reason="확인 필요").level == "조건부"


def test_rule_double_compute_mismatch_blocks_verdict():
    v = judge(_est(0.2, 0.15, 0.25), **_kw(double=DoubleCompute("q", 1.0, 1.1, 1e-9, False)))
    assert v.level == "판정 없음"


# ------------------------------------------------------------------ 파이프라인 (합성 정답)
def test_pipeline_recovers_effect():
    w, m = _wide([Effect("청소년", D, 0.22)])
    a = analyze_series(w, _design(), mask=m)
    assert a.verdict.label == "효과 있음(증가)"
    assert a.main.ci[0] < 0.22 < a.main.ci[1]


def test_pipeline_null_is_no_effect():
    w, m = _wide([], seed=6)
    a = analyze_series(w, _design(), mask=m)
    assert a.verdict.level == "효과 없음"


def test_pipeline_measurement_break_is_held_with_needed_data():
    w, m = _wide([MeasurementBreak("청소년", D, 0.2, 10)])
    a = analyze_series(w, _design(), mask=m)
    assert a.verdict.level == "판정 보류" and a.verdict.needed_data


def test_pipeline_reference_shock_is_not_called_effect():
    w, m = _wide([Shock(["일반"], D, "2024-08-01", -0.2)])
    a = analyze_series(w, _design(), mask=m)
    assert a.verdict.level != "효과 있음"


def test_panel_group_level_inference_is_not_narrower_than_cluster():
    from jverdict.synth.accuracy_causal import panel_design, panel_frame
    from jverdict.synth.scenarios_causal import _card
    sd = generate(_card([Effect("중국", "2023-03-01", 0.6)]), 7)
    a = analyze_panel(panel_frame(sd), panel_design("2023-03-01"))
    assert a.main.se >= a.main.details["cr1_se"]
    assert a.main.effect > 0 and a.verdict.level in ("효과 있음", "조건부")
