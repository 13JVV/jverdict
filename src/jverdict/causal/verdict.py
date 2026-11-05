"""4단계 판정 규칙 (코드로 고정).

판정은 통계 툴의 결과만으로 정해진다. LLM은 이 함수를 바꿀 수 없고, 결과를 말로 옮기기만 한다.

검증 항목
    C0 데이터 품질 : 측정 단절 진단의 판정 상한(보류/조건부), 커버리지(사전·사후 관측 충분)
    C1 사전추세    : p ≥ 0.10 이고 '사전 추세를 연장하면 설명되는 몫'(주분석 − 추세 보정 추정)이 효과의 50% 미만이면 통과.
                     p < 0.01이면 '심각'
    C2 위약검정    : 시점 위약(가짜 시행일 19개 이상일 때, 양측 순위 p < 0.05)과
                     공간 위약(정책 대상이 아닌 집단, 실제 효과가 모든 가짜 효과보다 큼) 중 가능한 것 모두 통과
    C3 민감도      : 사양 3개 중 2개 이상이 같은 방향·유의
    C4 이중 계산   : 핵심 수치의 두 계산 경로가 일치

판정에 쓰는 신뢰구간 = 분석적 구간 ∪ 시점 위약 경험적 구간 (위약을 만들 수 있을 때).
    처치 집단이 하나뿐이거나 자기상관이 강하면 분석적 구간이 지나치게 좁을 수 있기 때문이다.

규칙 (위에서부터 처음 맞는 것)
    C4 불일치                               → 판정 없음 (계산 오류부터 고친다)
    커버리지 부족 · 품질 상한 '보류'          → 판정 보류
    구간이 0을 포함하지 않음:
        C1 심각(p < 0.01)                  → 추세 보정 추정도 같은 방향·유의하면 조건부, 아니면 판정 보류
        C1 실패 그리고 C2 실패               → 판정 보류
        C1·C2·C3 모두 통과                  → 효과 있음
        그 밖                               → 조건부
    구간이 0을 포함:
        MDE ≤ 기준 효과, C1 심각 아님         → 효과 없음
        90% 수준 신호(p < 0.10) + 민감도 전부 같은 방향 + C1 심각 아님 → 조건부 (약한 신호)
        그 밖                               → 판정 보류 (검정력 부족)
    품질 상한 '조건부'이면 '효과 있음'을 '조건부'로 내린다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .checks import DoubleCompute, PlaceboSpace, PlaceboTime, Sensitivity
from .estimate import Estimate
from .power import Power

RULES_VERSION = "v2"
RULE_HISTORY = [
    {"version": "v1", "date": "2026-11-02",
     "change": "기획서 3.3 원안. 사례 D 사전 등록(2026-11-03 잠금)에 사용."},
    {"version": "v2", "date": "2026-11-03",
     "change": ("사례 D에서 사전추세가 크게 깨져도(p<0.001) '조건부'가 나오는 문제를 보완: "
                "사전추세 p<0.01이면 상한을 '판정 보류'로 두고 추세 보정 추정이 유의할 때만 '조건부'. "
                "판정용 신뢰구간을 분석적 구간과 위약 경험적 구간의 합집합으로. "
                "'방향은 일관되나 유의하지 않음'(조건부)을 90% 수준 신호 + 민감도 전부 같은 방향으로 명시. "
                "이미 기록한 사례 D의 판정은 바꾸지 않고 v1·v2 판정을 함께 보고한다.")},
    {"version": "v2", "date": "2026-11-03", "change": (
        "합성데이터 정확도 측정(사례 C) 결과 반영: (1) 공간 위약 통과 기준을 '가짜 처치 효과가 모두 실제 효과의 절반 미만'으로 "
        "강화(대조 집단에만 생긴 사건에서 '효과 있음' 오판 10% → 0%). (2) C1에 '사전 추세 연장으로 설명되는 몫 50% 미만' 조건 추가"
        "(사전추세 검정의 검정력이 낮은 패널에서 오판 감소). (3) 처치 집단이 하나인 패널은 집단 수준 격차 시계열(AR(1))로 "
        "표준오차·사전추세를 계산(업종 군집 표준오차의 구간 포함률 20%). 실데이터 결과를 보기 전에 정한 변경이다.")},
]
LIGHT = {"효과 있음": "🟢", "효과 없음": "⚪", "조건부": "🟡", "판정 보류": "🔴", "판정 없음": "⛔"}
PASS, FAIL, NA = "통과", "실패", "해당 없음"
DRIFT_SHARE = 0.5      # 사전 추세 연장이 효과의 이 비율 이상을 설명하면 C1 실패


@dataclass
class Check:
    code: str
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    level: str
    direction: str | None
    headline: str
    reasons: list[str]
    checks: list[Check]
    ci_used: tuple[float, float]
    p_used: float
    power: Power | None
    quality_cap: str | None
    needed_data: list[str] = field(default_factory=list)
    rules_version: str = RULES_VERSION

    @property
    def light(self) -> str:
        return LIGHT[self.level]

    @property
    def label(self) -> str:
        if self.level in ("효과 있음", "조건부") and self.direction:
            return f"{self.level}({self.direction})"
        return self.level

    def to_dict(self) -> dict:
        d = asdict(self)
        d["light"], d["label"] = self.light, self.label
        return d


def _pct(x: float) -> str:
    return f"{np.expm1(x) * 100:+.1f}%"


def judge(main: Estimate | None, *, pretrend_p: float | None, trend_adjusted: Estimate | None,
          placebo_time: PlaceboTime | None, placebo_space: PlaceboSpace | None,
          sens: Sensitivity | None, double: DoubleCompute | None, power: Power | None,
          quality_cap: str | None = None, quality_reason: str = "", coverage_reason: str | None = None,
          needed_data: list[str] | None = None, window_label: str = "사후 기간") -> Verdict:
    needed = list(needed_data or [])
    checks: list[Check] = []

    # ---------------- C0 데이터
    if coverage_reason or main is None:
        why = coverage_reason or "추정에 필요한 관측이 부족함"
        checks.append(Check("C0", "데이터 커버리지", FAIL, why))
        return Verdict("판정 보류", None, f"판정 보류: {why}", [why], checks,
                       (float("nan"), float("nan")), float("nan"), power, quality_cap, needed)
    checks.append(Check("C0", "데이터 품질",
                        FAIL if quality_cap == "판정 보류" else (("주의" if quality_cap else PASS)),
                        quality_reason or "측정 단절 증거 없음"))

    # ---------------- 판정용 구간
    ci, p = tuple(main.ci), main.p
    if placebo_time is not None and placebo_time.available and placebo_time.ci:
        ci = (min(ci[0], placebo_time.ci[0]), max(ci[1], placebo_time.ci[1]))
        p = max(p, placebo_time.p_two)
    sig = bool(ci[0] > 0 or ci[1] < 0)
    direction = "증가" if main.effect > 0 else "감소"

    # ---------------- C1
    if pretrend_p is None or not np.isfinite(pretrend_p):
        c1, c1_severe = None, False
        checks.append(Check("C1", "사전추세", NA, "사전 기간이 짧아 검정 불가"))
    else:
        c1, c1_severe = pretrend_p >= 0.10, pretrend_p < 0.01
        detail = f"사전 추세 검정 p={pretrend_p:.3g}" + (" (심각: p<0.01)" if c1_severe else "")
        if trend_adjusted is not None and np.isfinite(trend_adjusted.effect) and abs(main.effect) > 0:
            # 사전 추세를 그대로 연장했을 때 설명되는 몫. 검정력이 낮아 p가 통과해도 크기로 걸러 낸다
            drift = main.effect - trend_adjusted.effect
            share = drift / main.effect
            detail += f", 사전 추세 연장으로 설명되는 몫 {share * 100:.0f}%"
            if share >= DRIFT_SHARE:
                c1 = False
                detail += f"(≥{DRIFT_SHARE:.0%}, 실패)"
        checks.append(Check("C1", "사전추세", PASS if c1 else FAIL, detail))

    # ---------------- C2
    parts = []
    if placebo_time is not None and placebo_time.available:
        ok_t = placebo_time.p_two < 0.05
        parts.append(ok_t)
        checks.append(Check("C2a", "시점 위약검정", PASS if ok_t else FAIL,
                            f"가짜 시행일 {placebo_time.n}개 중 순위 p={placebo_time.p_two:.3f}"))
    else:
        n = 0 if placebo_time is None else placebo_time.n
        checks.append(Check("C2a", "시점 위약검정", NA, f"가짜 시행일 {n}개(19개 미만)"))
    if placebo_space is not None and placebo_space.passed is not None:
        parts.append(placebo_space.passed)
        fk = ", ".join(f"{k} {np.expm1(v['effect']) * 100:+.1f}%" for k, v in placebo_space.fakes.items())
        checks.append(Check("C2b", "공간 위약검정", PASS if placebo_space.passed else FAIL,
                            f"실제 {_pct(placebo_space.real)} vs 가짜 처치 {fk}"))
    else:
        checks.append(Check("C2b", "공간 위약검정", NA, "가짜 처치로 쓸 집단 없음"))
    c2 = all(parts) if parts else None

    # ---------------- C3
    if sens is not None and sens.n > 0:
        c3 = sens.passed
        checks.append(Check("C3", "민감도", (PASS if c3 else FAIL) if sig else "참고",
                            f"{sens.n}개 중 {sens.n_pass}개 같은 방향·유의 (같은 방향 {sens.same_sign}개)"))
    else:
        c3 = None
        checks.append(Check("C3", "민감도", NA, "민감도 사양 없음"))

    # ---------------- C4
    if double is not None:
        checks.append(Check("C4", "이중 계산", PASS if double.match else FAIL,
                            f"{double.quantity}: pandas {double.pandas:.9f} / SQL {double.sql:.9f}"))
        if not double.match:
            return Verdict("판정 없음", None, "판정 없음: 두 계산 경로의 결과가 다름(계산 오류 점검 필요)",
                           ["C4 이중 계산 불일치"], checks, ci, p, power, quality_cap, needed)
    else:
        checks.append(Check("C4", "이중 계산", NA, "이중 계산 경로 없음"))

    size = f"{window_label} 평균 {_pct(main.effect)} (판정 구간 {_pct(ci[0])} ~ {_pct(ci[1])})"

    # ---------------- 품질 상한 보류
    if quality_cap == "판정 보류":
        reasons = [f"데이터 품질: {quality_reason}", f"참고로 계산한 변화는 {size}이지만 측정 문제와 구분할 수 없음"]
        return Verdict("판정 보류", None, f"판정 보류: {quality_reason}", reasons, checks, ci, p, power,
                       quality_cap, needed)

    reasons: list[str] = []
    if sig:
        if c1_severe:
            if trend_adjusted is not None and trend_adjusted.significant and \
                    np.sign(trend_adjusted.effect) == np.sign(main.effect):
                level = "조건부"
                reasons.append(f"사전추세가 크게 깨졌지만(p={pretrend_p:.3g}) 추세를 보정해도 "
                               f"{_pct(trend_adjusted.effect)}로 같은 방향·유의")
            else:
                level = "판정 보류"
                ta = "계산 불가" if trend_adjusted is None else trend_adjusted.short_ko()
                reasons.append(f"사전추세가 크게 깨짐(p={pretrend_p:.3g}): 정책 효과와 기존 추세를 구분할 수 없음. "
                               f"추세 보정 추정 {ta}")
        elif c1 is False and c2 is False:
            level = "판정 보류"
            reasons.append("사전추세(C1)와 위약검정(C2)이 모두 실패")
        elif c1 and c2 and c3:
            level = "효과 있음"
            reasons.append(f"{size}, C1·C2·C3 모두 통과")
        else:
            level = "조건부"
            fails = [n for n, ok in (("C1 사전추세", c1), ("C2 위약검정", c2), ("C3 민감도", c3)) if not ok]
            reasons.append(f"{size}. 통과하지 못한 검증: {', '.join(fails)}")
    else:
        weak = (p < 0.10 and (sens is None or sens.n == 0 or sens.same_sign == sens.n) and not c1_severe)
        if power is not None and power.enough and not c1_severe:
            level = "효과 없음"
            reasons.append(f"{size}. 구간이 0을 포함하고, 구분 가능한 최소 효과 ±{power.mde_pct * 100:.1f}%가 "
                           f"기준 효과 {power.reference_pct * 100:.0f}% 이하")
        elif weak:
            level = "조건부"
            reasons.append(f"{size}. 95% 수준에서는 유의하지 않지만 90% 수준 신호(p={p:.3f})이고 민감도 사양이 모두 같은 방향")
        else:
            level = "판정 보류"
            if c1_severe:
                reasons.append(f"사전추세가 크게 깨짐(p={pretrend_p:.3g})")
            if power is not None:
                reasons.append(f"{size}. 검정력 부족: {power.sentence_ko()}")
                if power.post_needed:
                    needed.append(power.needed_ko())
            else:
                reasons.append(f"{size}. 검정력 부족")
    if quality_cap == "조건부" and level == "효과 있음":
        level = "조건부"
        reasons.append(f"데이터 품질 주의로 상한 '조건부': {quality_reason}")
    elif quality_cap == "조건부":
        reasons.append(f"데이터 품질 주의: {quality_reason}")

    d = direction if level in ("효과 있음", "조건부") else None
    head = f"{level}{f'({d})' if d else ''}: {reasons[0]}"
    return Verdict(level, d, head, reasons, checks, ci, p, power, quality_cap, needed)


def judge_v1(main: Estimate, *, pretrend_p: float | None, placebo_space_pass: bool | None,
             sens: Sensitivity | None, mde_log: float, reference_pct: float) -> str:
    """기획서 원안 규칙(v1). 사례 D 사전 등록 판정을 재현하는 데만 쓴다."""
    lo, hi = main.ci
    if not (lo > 0 or hi < 0):
        return "효과 없음" if mde_log <= np.log1p(reference_pct) else "판정 보류"
    c1 = pretrend_p is not None and pretrend_p >= 0.10
    c2 = bool(placebo_space_pass)
    c3 = sens is not None and sens.n_pass >= 2
    if not c1 and not c2:
        return "판정 보류"
    d = "(증가)" if lo > 0 else "(감소)"
    return f"효과 있음 {d}" if (c1 and c2 and c3) else f"조건부 {d}"
