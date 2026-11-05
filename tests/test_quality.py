"""데이터 품질 툴: 정답을 아는 합성데이터로 동작을 고정한다."""
import numpy as np
import pandas as pd
import pytest

from jverdict.quality._panel import to_wide
from jverdict.quality.breaks import diagnose, scan_breaks
from jverdict.quality.coverage import check_coverage, exclusion_mask
from jverdict.quality.external import ExternalStat, external_check
from jverdict.synth.generator import (Effect, MeasurementBreak, Missing, Outage, PartialDays,
                                      SynthSpec, generate)
from jverdict.synth.scenarios import BUS_GROUPS, CARD_GROUPS, FOREIGN

DATE = "2024-09-02"


def _bus(inj, seed=11):
    sd = generate(SynthSpec("2021-01-01", "2025-10-31", BUS_GROUPS, injections=inj), seed)
    cov = check_coverage(sd.df, "time", "group", "value", freq="D")
    w = to_wide(sd.df, "time", "group", "value", "D")
    return sd, cov, w, exclusion_mask(cov, w.index)


# ------------------------------------------------------------------ 커버리지
def test_clean_data_has_no_findings():
    _, cov, _, _ = _bus([])
    assert cov.findings == []


def test_global_gap_and_partial_tail_exact():
    sd, cov, w, m = _bus([Missing("2023-05-10", "2023-07-01", None, partial_tail=4)])
    gaps = cov.by_kind("전체 결측")
    assert [(f.start, f.end, f.length) for f in gaps] == [("2023-05-10", "2023-07-01", 53)]
    assert m[w.index.get_loc(pd.Timestamp("2023-07-02"))]          # 재개 직후 부분 집계일은 제외 대상
    assert not m[w.index.get_loc(pd.Timestamp("2023-08-01"))]


def test_scattered_partial_days_found():
    sd, cov, w, m = _bus([PartialDays(6, share_high=0.25)])
    truth = next(i for i in sd.truth["injections"] if i["kind"] == "partial_days")["dates"]
    flagged = {str(d.date()) for d in w.index[m]}
    assert set(truth) <= flagged


def test_group_missing_monthly_exact():
    spec = SynthSpec("2018-01-01", "2025-06-01", CARD_GROUPS, "MS", units=10, unit_sd=1.0,
                     holidays=False, injections=[Missing("2023-05-01", "2023-06-01", FOREIGN)])
    sd = generate(spec, 3)
    cov = check_coverage(sd.df, "time", "group", "value", unit="unit", freq="MS")
    f = cov.by_kind("집단 결측")
    assert len(f) == 1 and (f[0].start, f[0].end) == ("2023-05", "2023-06")
    assert set(f[0].groups) == set(FOREIGN) and f[0].severity == "높음"
    assert cov.affects("2023-03-01", "2023-08-01", ["중국"]) == f


def test_new_category_reported():
    _, cov, _, _ = _bus([MeasurementBreak("어린이", DATE, 0.5, reclass_to="기타")])
    f = cov.by_kind("분류 신설")
    assert len(f) == 1 and f[0].groups == ["기타"] and f[0].start == DATE


# ------------------------------------------------------------------ 측정 단절
def test_no_event_no_suspicion():
    _, _, w, m = _bus([])
    dg = diagnose(w, "청소년", "일반", DATE, "D", expected="increase", mask=m)
    assert not dg.suspected and dg.judgement_cap is None


def test_true_effect_is_not_called_measurement_break():
    _, _, w, m = _bus([Effect("청소년", DATE, 0.25)])
    dg = diagnose(w, "청소년", "일반", DATE, "D", expected="increase", mask=m)
    assert dg.verdict == "측정 단절 증거 없음"
    assert dg.test.ci[0] <= 0.25 <= dg.test.ci[1]


def test_measurement_break_with_external_is_strong():
    sd, _, w, m = _bus([MeasurementBreak("청소년", DATE, keep=0.2, adoption=12)])
    total = float(sd.latent.loc[DATE:"2024-12-31", "청소년"].sum())
    ext = [ExternalStat("공식 집계", total, DATE, "2024-12-31", ["청소년"], same_definition=True)]
    dg = diagnose(w, "청소년", "일반", DATE, "D", expected="increase", external=ext, mask=m)
    assert dg.verdict == "측정 단절 의심(강)" and dg.judgement_cap == "판정 보류"
    met = {e.code for e in dg.evidence if e.met}
    assert {"E2", "E3", "E4"} <= met
    assert dg.test.onset["shape"] == "점진 이탈"
    assert dg.needed_data


def test_backfire_is_only_a_caution():
    _, _, w, m = _bus([Effect("청소년", DATE, -0.25)])
    dg = diagnose(w, "청소년", "일반", DATE, "D", expected="increase", mask=m)
    assert dg.verdict == "확인 필요" and dg.judgement_cap == "조건부"


def test_reclass_detected_without_policy_info():
    _, _, w, m = _bus([MeasurementBreak("어린이", DATE, keep=0.4, reclass_to="기타")])
    dg = diagnose(w, "어린이", "일반", DATE, "D", mask=m)
    assert dg.suspected and any(r["kind"] == "신규 분류 출현" for r in dg.reclass)


def test_outage_flags_whole_system():
    _, _, w, m = _bus([Outage(DATE, "2024-11-15", keep=0.3)])
    dg = diagnose(w, "청소년", "일반", DATE, "D", expected="increase", mask=m)
    assert dg.verdict == "전체 집계 이상"


def test_scan_finds_unknown_break_date():
    _, _, w, m = _bus([MeasurementBreak("청소년", DATE, keep=0.3, adoption=0)])
    c = scan_breaks(w, "청소년", "일반", "D", mask=m)
    assert c, "단절 후보가 없음"
    assert abs((pd.Timestamp(c[0].date) - pd.Timestamp(DATE)).days) <= 7
    assert c[0].step < 0


def test_scan_quiet_on_clean_data():
    _, _, w, m = _bus([], seed=12)
    assert len(scan_breaks(w, "청소년", "일반", "D", mask=m)) <= 1


# ------------------------------------------------------------------ 외부 통계
def test_external_tolerance_depends_on_definition():
    idx = pd.date_range("2024-01-01", "2024-01-31")
    w = pd.DataFrame({"a": np.full(len(idx), 100.0)}, index=idx)
    same = external_check(w, ExternalStat("x", 31 * 150, "2024-01-01", "2024-01-31", ["a"], same_definition=True), "D")
    loose = external_check(w, ExternalStat("x", 31 * 150, "2024-01-01", "2024-01-31", ["a"]), "D")
    assert same.contradiction and not loose.contradiction
    assert same.ratio == pytest.approx(1.5)


def test_external_period_mismatch_uses_period_mean():
    idx = pd.date_range("2024-01-01", "2024-01-31")
    w = pd.DataFrame({"a": np.full(len(idx), 100.0)}, index=idx)
    r = external_check(w, ExternalStat("x", 60 * 100, "2024-01-01", "2024-02-29", ["a"]), "D")
    assert r.period_mismatch and r.ratio == pytest.approx(1.0)
