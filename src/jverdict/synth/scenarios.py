"""정확도 검증용 시나리오 모음.

각 시나리오는 시드마다 정책 시점, 효과 크기, 단절 정도 등을 무작위로 바꾼다.
정답(라벨)과 '올바른 판정'을 미리 적어 두고, 툴의 출력과 대조한다.

라벨
- none     : 아무 일도 없음
- effect   : 실제 정책 효과 (측정은 정상)
- mbreak   : 측정 단절 (실제 행동은 그대로인데 기록 방식이 바뀜)
- outage   : 전체 집계 이상
- coverage : 결측·부분 집계 (커버리지 점검 대상)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .generator import (Effect, GroupSpec, MeasurementBreak, Missing, Outage, PartialDays,
                        SynthSpec)

# 실데이터(2023-03 ~ 2025-06 청소년/일반 비율)의 변동에 맞춘 버스형 집단 구성.
# 맞춘 기준: 일별 잔차 표준편차 약 0.10, 하루 자기상관 약 0.45, 단절 통계 MAD 약 0.05.
BUS_GROUPS = [
    GroupSpec("일반", 30000, vacation=-0.03, weekend=-0.15, noise_sd=0.05),
    GroupSpec("청소년", 10000, vacation=-0.35, weekend=-0.45, noise_sd=0.07),
    GroupSpec("어린이", 2500, vacation=-0.30, weekend=-0.25, noise_sd=0.20),
    GroupSpec("경로", 14000, vacation=0.0, weekend=-0.30, noise_sd=0.07),
]

CARD_GROUPS = [
    GroupSpec("제주도민", 6.0e10, noise_sd=0.05), GroupSpec("내국인", 4.5e10, noise_sd=0.08),
    GroupSpec("법인", 4.0e10, noise_sd=0.06), GroupSpec("중국", 9.0e8, noise_sd=0.15),
    GroupSpec("일본", 1.5e8, noise_sd=0.15), GroupSpec("동남아", 5.5e8, noise_sd=0.15),
    GroupSpec("기타외국", 1.3e9, noise_sd=0.12),
]
FOREIGN = ["중국", "일본", "동남아", "기타외국"]


@dataclass
class Case:
    """툴에 넘길 정책 카드에 해당하는 정보와 정답."""
    scenario: str
    label: str
    date: str | None
    target: str | None
    reference: list[str]
    expected: str | None
    true_log_step: float | None = None     # 계단형 실제 효과의 로그 크기 (복원 정확도용)
    external_groups: list[str] | None = None
    external_window: tuple[str, str] | None = None
    ideal: list[str] = field(default_factory=list)   # 올바른 진단 결과
    tolerable: list[str] = field(default_factory=list)  # 틀리진 않았지만 보수적인 결과
    extra: dict = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    title: str
    label: str
    build: Callable[[np.random.Generator], tuple[SynthSpec, Case]]
    kind: str = "break"     # break = 측정 단절 진단, coverage = 커버리지 점검


def _policy_date(rng: np.random.Generator, lo="2023-03-01", hi="2025-08-31") -> str:
    days = (pd.Timestamp(hi) - pd.Timestamp(lo)).days
    return str((pd.Timestamp(lo) + pd.Timedelta(days=int(rng.integers(0, days + 1)))).date())


def _bus(injections, **kw) -> SynthSpec:
    return SynthSpec("2021-01-01", "2025-10-31", BUS_GROUPS, "D", injections=injections, **kw)


def _ext_window(date: str, days: int = 120) -> tuple[str, str]:
    d = pd.Timestamp(date)
    return str(d.date()), str((d + pd.Timedelta(days=days)).date())


SUSPECT = ["측정 단절 의심", "측정 단절 의심(강)"]
CLEAN = ["단절 없음", "측정 단절 증거 없음"]


def s_null(rng):
    d = _policy_date(rng)
    return _bus([]), Case("null", "none", d, "청소년", ["일반"], "increase",
                          ideal=CLEAN, tolerable=["확인 필요"])


def s_effect_step(rng):
    d, size = _policy_date(rng), float(rng.uniform(0.15, 0.35))
    return _bus([Effect("청소년", d, size)]), Case(
        "effect_step", "effect", d, "청소년", ["일반"], "increase", true_log_step=size,
        ideal=["측정 단절 증거 없음"], tolerable=["확인 필요"])


def s_effect_ramp(rng):
    d, size = _policy_date(rng), float(rng.uniform(0.15, 0.35))
    return _bus([Effect("청소년", d, size, ramp=int(rng.integers(30, 91)))]), Case(
        "effect_ramp", "effect", d, "청소년", ["일반"], "increase",
        ideal=CLEAN, tolerable=["확인 필요"])


def s_effect_ext(rng):
    d, size = _policy_date(rng), float(rng.uniform(0.15, 0.35))
    return _bus([Effect("청소년", d, size)]), Case(
        "effect_ext", "effect", d, "청소년", ["일반"], "increase", true_log_step=size,
        external_groups=["청소년"], external_window=_ext_window(d),
        ideal=["측정 단절 증거 없음"], tolerable=["확인 필요"])


def s_backfire(rng):
    d, size = _policy_date(rng), -float(rng.uniform(0.15, 0.35))
    return _bus([Effect("청소년", d, size)]), Case(
        "backfire", "effect", d, "청소년", ["일반"], "increase", true_log_step=size,
        ideal=["확인 필요", "측정 단절 증거 없음"], tolerable=[])


def s_mbreak(rng):
    d = _policy_date(rng)
    keep, adopt = float(rng.uniform(0.10, 0.40)), int(rng.integers(3, 22))
    return _bus([MeasurementBreak("청소년", d, keep, adopt)]), Case(
        "mbreak", "mbreak", d, "청소년", ["일반"], "increase",
        ideal=SUSPECT, tolerable=["확인 필요"], extra={"keep": keep, "adoption": adopt})


def s_mbreak_mild(rng):
    d = _policy_date(rng)
    keep, adopt = float(rng.uniform(0.55, 0.75)), int(rng.integers(3, 22))
    return _bus([MeasurementBreak("청소년", d, keep, adopt)]), Case(
        "mbreak_mild", "mbreak", d, "청소년", ["일반"], "increase",
        ideal=SUSPECT, tolerable=["확인 필요"], extra={"keep": keep, "adoption": adopt})


def s_mbreak_mild_ext(rng):
    spec, case = s_mbreak_mild(rng)
    case.scenario = "mbreak_mild_ext"
    case.external_groups, case.external_window = ["청소년"], _ext_window(case.date)
    return spec, case


def s_reclass(rng):
    d = _policy_date(rng)
    keep, share = float(rng.uniform(0.30, 0.60)), float(rng.uniform(0.6, 1.0))
    adopt = int(rng.integers(0, 8))
    return _bus([MeasurementBreak("어린이", d, keep, adopt, reclass_to="기타",
                                  reclass_share=share)]), Case(
        "reclass", "mbreak", d, "어린이", ["일반"], None,
        ideal=SUSPECT, tolerable=["확인 필요"],
        extra={"keep": keep, "reclass_share": share, "adoption": adopt})


def s_reclass_existing(rng):
    """어린이 일부가 기존 '청소년' 분류로 잘못 집계되기 시작함 (정책 정보 없음)."""
    d = _policy_date(rng)
    keep = float(rng.uniform(0.30, 0.60))
    return _bus([MeasurementBreak("어린이", d, keep, int(rng.integers(0, 8)), reclass_to="청소년")]), Case(
        "reclass_existing", "mbreak", d, "어린이", ["일반"], None,
        ideal=SUSPECT, tolerable=["확인 필요"], extra={"keep": keep})


def s_outage(rng):
    d = _policy_date(rng)
    end = str((pd.Timestamp(d) + pd.Timedelta(days=int(rng.integers(45, 90)))).date())
    keep = float(rng.uniform(0.2, 0.5))
    return _bus([Outage(d, end, keep)]), Case(
        "outage", "outage", d, "청소년", ["일반"], "increase",
        ideal=["전체 집계 이상"], tolerable=[], extra={"keep": keep, "end": end})


def s_gap_partial(rng):
    start = _policy_date(rng, "2022-01-01", "2024-12-31")
    length = int(rng.integers(30, 121))
    end = str((pd.Timestamp(start) + pd.Timedelta(days=length - 1)).date())
    tail = int(rng.integers(3, 8))
    spec = _bus([Missing(start, end, None, partial_tail=tail), PartialDays(8, share_high=0.25)])
    return spec, Case("gap_partial", "coverage", start, None, [], None,
                      extra={"gap": (start, end), "tail": tail})


def s_group_missing(rng):
    m = pd.Timestamp(_policy_date(rng, "2019-06-01", "2024-10-01")).replace(day=1)
    end = m + pd.offsets.MonthBegin(1)
    spec = SynthSpec("2018-01-01", "2025-06-01", CARD_GROUPS, "MS", units=10, unit_sd=1.0,
                     unit_noise=0.08, common_sd=0.03, holidays=False,
                     injections=[Missing(str(m.date()), str(end.date()), FOREIGN)])
    return spec, Case("group_missing", "coverage", str(m.date()), None, [], None,
                      extra={"gap": (str(m.date()), str(end.date())), "groups": FOREIGN})


SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("null", "아무 일도 없음", "none", s_null),
    Scenario("effect_step", "실제 효과(즉시, +16~42%)", "effect", s_effect_step),
    Scenario("effect_ramp", "실제 효과(1~3개월에 걸쳐)", "effect", s_effect_ramp),
    Scenario("effect_ext", "실제 효과 + 공식 통계 대조", "effect", s_effect_ext),
    Scenario("backfire", "실제 역효과(−14~−30%)", "effect", s_backfire),
    Scenario("mbreak", "측정 단절(60~90% 누락)", "mbreak", s_mbreak),
    Scenario("mbreak_mild", "약한 측정 단절(25~45% 누락)", "mbreak", s_mbreak_mild),
    Scenario("mbreak_mild_ext", "약한 측정 단절 + 공식 통계 대조", "mbreak", s_mbreak_mild_ext),
    Scenario("reclass", "분류 변경(새 분류로 이동, 정책 정보 없음)", "mbreak", s_reclass),
    Scenario("reclass_existing", "분류 변경(기존 분류로 이동, 정책 정보 없음)", "mbreak", s_reclass_existing),
    Scenario("outage", "전체 집계 이상", "outage", s_outage),
    Scenario("gap_partial", "전체 결측 + 부분 집계", "coverage", s_gap_partial, kind="coverage"),
    Scenario("group_missing", "외국인 집단만 2개월 결측(월 단위)", "coverage", s_group_missing, kind="coverage"),
]}
