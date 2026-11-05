"""인과 엔진 정확도 시나리오 (사례 C의 2부).

각 시나리오는 시드마다 정책 시점·효과 크기·함정의 세기를 무작위로 바꾸고, 판정의 정답을 미리 적어 둔다.
    ideal     : 올바른 판정
    tolerable : 틀리지는 않았지만 보수적인 판정 (예: 효과가 있는데 '조건부')
    그 밖     : 오판. 특히 '없는 효과를 있다고 함'(효과 있음 오판)을 따로 센다.

시계열(버스형 일 단위)과 패널(카드형 월 단위, 집단 × 업종) 두 경로를 모두 시험한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .generator import (Effect, MeasurementBreak, Missing, Outage, PartialDays, PreTrend, Shock, SynthSpec)
from .scenarios import BUS_GROUPS, CARD_GROUPS, FOREIGN, _policy_date

BUS_START, BUS_END = "2021-01-01", "2025-10-31"
CARD_START, CARD_END = "2018-01-01", "2025-06-01"
POST_DAYS = 182


@dataclass
class CCase:
    scenario: str
    kind: str                       # series / panel
    date: str
    true_effect: float | None       # 사후 창 평균 실제 효과(로그). 효과가 없으면 0
    ideal: list[str]
    tolerable: list[str] = field(default_factory=list)
    has_effect: bool = False        # 정책 효과가 실제로 있는가 (오판 분류용)
    extra: dict = field(default_factory=dict)


@dataclass
class CScenario:
    name: str
    title: str
    kind: str
    build: Callable[[np.random.Generator], tuple[SynthSpec, CCase]]


def _bus(inj) -> SynthSpec:
    return SynthSpec(BUS_START, BUS_END, BUS_GROUPS, "D", injections=inj)


def _card(inj) -> SynthSpec:
    return SynthSpec(CARD_START, CARD_END, CARD_GROUPS, "MS", units=10, unit_sd=1.0, unit_noise=0.08,
                     common_sd=0.03, holidays=False, injections=inj)


def _ramp_avg(size: float, ramp: int, n: int) -> float:
    k = np.arange(n)
    return float(size * np.mean(np.clip((k + 1) / ramp, 0, 1)))


def _post_len(date: str, end: str = BUS_END, n: int = POST_DAYS) -> int:
    return int(min(n, (pd.Timestamp(end) - pd.Timestamp(date)).days + 1))


UP, DOWN = "효과 있음(증가)", "효과 있음(감소)"
C_UP, C_DOWN = "조건부(증가)", "조건부(감소)"
NONE, HOLD = "효과 없음", "판정 보류"


# ---------------------------------------------------------------- 시계열 (버스형)
def s_null(rng):
    d = _policy_date(rng)
    return _bus([]), CCase("null", "series", d, 0.0, [NONE], [HOLD])


def s_effect(rng):
    d, size = _policy_date(rng), float(rng.uniform(0.10, 0.35))
    return _bus([Effect("청소년", d, size)]), CCase("effect", "series", d, size, [UP], [C_UP], True,
                                                    {"size": size})


def s_ramp(rng):
    d, size, ramp = _policy_date(rng), float(rng.uniform(0.15, 0.35)), int(rng.integers(30, 91))
    return _bus([Effect("청소년", d, size, ramp=ramp)]), CCase(
        "ramp", "series", d, _ramp_avg(size, ramp, _post_len(d)), [UP], [C_UP], True,
        {"size": size, "ramp": ramp})


def s_backfire(rng):
    d, size = _policy_date(rng), -float(rng.uniform(0.10, 0.35))
    # 기대(증가)와 반대 방향이라 품질 진단이 '확인 필요'(상한 조건부)를 내는 것이 설계상 정상
    return _bus([Effect("청소년", d, size)]), CCase("backfire", "series", d, size, [DOWN, C_DOWN], [], True,
                                                    {"size": size})


def s_effect_dirty(rng):
    d, size = _policy_date(rng, "2023-09-01", "2025-08-31"), float(rng.uniform(0.10, 0.35))
    gs = pd.Timestamp(d) - pd.Timedelta(days=int(rng.integers(150, 400)))
    ge = gs + pd.Timedelta(days=int(rng.integers(20, 60)))
    inj = [Effect("청소년", d, size), Missing(str(gs.date()), str(ge.date()), None, partial_tail=4),
           PartialDays(10, share_high=0.25)]
    return _bus(inj), CCase("effect_dirty", "series", d, size, [UP], [C_UP], True,
                            {"size": size, "gap": (str(gs.date()), str(ge.date()))})


def s_pretrend(rng):
    d = _policy_date(rng)
    start = pd.Timestamp(d) - pd.Timedelta(days=int(rng.integers(180, 366)))
    slope = float(rng.choice([-1, 1]) * rng.uniform(0.15, 0.30))
    return _bus([PreTrend("청소년", slope, str(start.date()), BUS_END)]), CCase(
        "pretrend", "series", d, 0.0, [HOLD, NONE], [C_UP, C_DOWN], False,
        {"slope": slope, "trend_start": str(start.date())})


def s_shock_ref(rng):
    d = _policy_date(rng)
    end = pd.Timestamp(d) + pd.Timedelta(days=int(rng.integers(90, 181)))
    size = -float(rng.uniform(0.10, 0.25))
    return _bus([Shock(["일반"], d, str(end.date()), size)]), CCase(
        "shock_ref", "series", d, 0.0, [HOLD], [NONE, C_UP, C_DOWN], False,
        {"shock": size, "end": str(end.date())})


def s_mbreak(rng):
    d = _policy_date(rng)
    keep, adopt = float(rng.uniform(0.10, 0.60)), int(rng.integers(3, 22))
    return _bus([MeasurementBreak("청소년", d, keep, adopt)]), CCase(
        "mbreak", "series", d, 0.0, [HOLD], [], False, {"keep": keep, "adoption": adopt})


def s_outage(rng):
    d = _policy_date(rng)
    end = pd.Timestamp(d) + pd.Timedelta(days=int(rng.integers(45, 90)))
    keep = float(rng.uniform(0.2, 0.5))
    return _bus([Outage(d, str(end.date()), keep)]), CCase(
        "outage", "series", d, 0.0, [HOLD], [NONE], False, {"keep": keep})


# ---------------------------------------------------------------- 패널 (카드형)
def _month(rng) -> str:
    return str(pd.Timestamp(_policy_date(rng, "2021-06-01", "2024-10-01")).replace(day=1).date())


def p_null(rng):
    return _card([]), CCase("panel_null", "panel", _month(rng), 0.0, [NONE], [HOLD])


def p_effect(rng):
    d, size = _month(rng), float(rng.uniform(0.40, 0.80))
    return _card([Effect("중국", d, size)]), CCase("panel_effect", "panel", d, size, [UP], [C_UP], True,
                                                  {"size": size})


def p_pretrend(rng):
    d = _month(rng)
    start = pd.Timestamp(d) - pd.DateOffset(months=int(rng.integers(9, 19)))
    slope = float(rng.uniform(0.40, 0.80))
    return _card([PreTrend("중국", slope, str(start.date()), CARD_END)]), CCase(
        "panel_pretrend", "panel", d, 0.0, [HOLD, NONE], [C_UP], False,
        {"slope": slope, "trend_start": str(start.date())})


CSCENARIOS: dict[str, CScenario] = {s.name: s for s in [
    CScenario("null", "아무 일도 없음", "series", s_null),
    CScenario("effect", "실제 효과(즉시, +11~42%)", "series", s_effect),
    CScenario("ramp", "실제 효과(1~3개월에 걸쳐 나타남)", "series", s_ramp),
    CScenario("backfire", "실제 역효과(−10~−30%, 기대와 반대)", "series", s_backfire),
    CScenario("effect_dirty", "실제 효과 + 결측·부분 집계", "series", s_effect_dirty),
    CScenario("pretrend", "효과 없음 + 정책 6~12개월 전 시작된 추세", "series", s_pretrend),
    CScenario("shock_ref", "효과 없음 + 대조 집단에만 생긴 사건", "series", s_shock_ref),
    CScenario("mbreak", "효과 없음 + 측정 단절(40~90% 누락)", "series", s_mbreak),
    CScenario("outage", "효과 없음 + 전체 집계 이상", "series", s_outage),
    CScenario("panel_null", "패널: 아무 일도 없음", "panel", p_null),
    CScenario("panel_effect", "패널: 실제 효과(+49~123%)", "panel", p_effect),
    CScenario("panel_pretrend", "패널: 효과 없음 + 처치 집단만 회복 추세", "panel", p_pretrend),
]}

PANEL_GROUPS = FOREIGN
