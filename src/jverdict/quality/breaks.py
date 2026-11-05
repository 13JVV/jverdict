"""측정 단절 탐지: 정책 시점의 큰 변화가 '정책 효과'인지 '집계 방식 변경'인지 가린다.

기본 신호 (정책 대상 집단 ÷ 대조 집단의 로그 비율)
    r[t] = log(1 + 대상[t]) − log(1 + 대조[t])
    날씨, 요일, 관광 성수기처럼 두 집단에 함께 작용하는 변화는 비율에서 상쇄된다.

계절 보정
    r_adj[t] = r[t] − median_k r[t − k·주기]   (일 단위 주기 364일, 요일이 맞음 / 월 단위 12개월)
    방학처럼 해마다 같은 때 반복되는 변화를 지운다. 과거 3개 주기의 중앙값이라,
    한 해에 있었던 단절이 다음 해 보정값을 오염시키는 '메아리'도 대부분 막는다.

단절 크기
    step[t] = median(r_adj[t+정착 … t+정착+창)) − median(r_adj[t−창 … t))
    정착 기간(일 단위 14일)은 카드 등록이 퍼지듯 서서히 바뀌는 변화도 다 담기 위한 여유다.

유의성
    다른 모든 시점의 step(위약 분포)과 비교한 순위 p값. 신뢰구간도 위약 분포의 분위수로 만든다.
    정책 시점과 창이 겹치는 시점, 그 시점의 계절 보정 메아리는 위약 분포에서 뺀다.

측정 단절 판단 근거 (정책 시점 단절이 유의할 때만 본다)
    E2 방향 모순    정책이 늘릴 것으로 기대되는데 줄었음(또는 그 반대)        [보통]
    E3 비현실적 크기 몇 주 만에 절반 이하로 줄거나 두 배 넘게 늘었음          [보통]
    E4 외부 통계 불일치 공식 발표 수치와 2배 넘게 차이                       [강함]
    E5 분류 변경    같은 때 새 분류가 생김, 또는 기존 분류가 감소분을 흡수하고 둘의 합계가 그대로임 [강함]
                    기존 분류가 반대로 크게 움직였지만 합계 보존은 확인되지 않음 [보통]
    대조 집단 자체에도 같은 때 단절이 있으면 '전체 집계 이상'으로 따로 본다.

원인은 단정하지 않는다. 판정 문구는 언제나 '의심'이다.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ._panel import (FreqParams, fmt, freq_params, rolling_median_back, rolling_median_fwd)
from .external import ExternalCheck, ExternalStat, external_check

LOG2 = float(np.log(2.0))


# --------------------------------------------------------------------------- 신호 만들기
@dataclass
class Signal:
    index: pd.DatetimeIndex
    r: np.ndarray          # 로그 비율 (보정 전)
    r_adj: np.ndarray      # 계절 보정 로그 비율 (보정 불가 시점은 NaN)
    step: np.ndarray       # 시점별 단절 크기 (계절 보정)
    pre: np.ndarray
    post: np.ndarray
    fp: FreqParams


def _series(wide: pd.DataFrame, cols: list[str]) -> np.ndarray:
    missing = [c for c in cols if c not in wide.columns]
    if missing:
        raise KeyError(f"데이터에 없는 집단: {missing}")
    return wide[cols].sum(axis=1, min_count=len(cols)).to_numpy(dtype=float)


def _seasonal_adjust(x: np.ndarray, fp: FreqParams, min_years: int = 1) -> np.ndarray:
    lagged = []
    for k in range(1, fp.years_back + 1):
        s = np.full_like(x, np.nan)
        lag = k * fp.season
        if lag < len(x):
            s[lag:] = x[:-lag]
        lagged.append(s)
    m = np.vstack(lagged)
    cnt = np.sum(np.isfinite(m), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 전부 NaN인 열의 중앙값 경고
        base = np.nanmedian(m, axis=0)
    base[cnt < min_years] = np.nan
    return x - base


def build_signal(wide: pd.DataFrame, target: str | list[str], reference: str | list[str],
                 freq: str, mask: np.ndarray | None = None, log_level: bool = False) -> Signal:
    """log_level=True면 비율이 아니라 대상 집단 자체의 로그 수준을 쓴다(대조 집단 점검용)."""
    fp = freq_params(freq)
    tcols = [target] if isinstance(target, str) else list(target)
    t = _series(wide, tcols)
    if log_level:
        r = np.log1p(t)
    else:
        rcols = [reference] if isinstance(reference, str) else list(reference)
        r = np.log1p(t) - np.log1p(_series(wide, rcols))
    r = r.astype(float)
    if mask is not None:
        r[np.asarray(mask, dtype=bool)] = np.nan
    r_adj = _seasonal_adjust(r, fp)
    minp = max(2, int(np.ceil(0.6 * fp.window)))   # 창의 60% 이상 관측이 있어야 계산
    pre = rolling_median_back(r_adj, fp.window, minp)
    post = rolling_median_fwd(r_adj, fp.window, minp, lag=fp.settle)
    return Signal(wide.index, r, r_adj, post - pre, pre, post, fp)


def _excluded_near(n: int, i0: int, fp: FreqParams) -> np.ndarray:
    """위약 분포에서 뺄 시점: 정책 시점과 창이 겹치는 시점과 그 계절 메아리."""
    ex = np.zeros(n, dtype=bool)
    rad = fp.window + fp.settle
    for k in range(0, fp.years_back + 1):
        c = i0 + k * fp.season
        ex[max(0, c - rad):min(n, c + rad + 1)] = True
    return ex


# --------------------------------------------------------------------------- 정해진 시점 검정
@dataclass
class BreakTest:
    date: str
    step: float | None            # 계절 보정 로그 단절 크기
    pct: float | None             # exp(step) − 1
    ci: tuple[float, float] | None
    p_rank: float | None
    z: float | None
    n_placebo: int
    significant: bool
    seasonal_adjusted: bool
    onset: dict = field(default_factory=dict)
    persistence: str = "알 수 없음"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _onset(sig: Signal, i0: int, total: float) -> dict:
    fp = sig.fp
    if not np.isfinite(total) or abs(total) < 1e-9:
        return {}
    x = sig.r_adj if np.isfinite(sig.r_adj[i0:i0 + fp.onset_horizon]).sum() > 0 else sig.r
    seg = pd.Series(x[i0:i0 + fp.onset_horizon])
    sm = seg.rolling(7 if fp.code == "D" else 1, center=True, min_periods=1).median().to_numpy()
    prog = (sm - sig.pre[i0]) / total
    def first(th):
        hit = np.flatnonzero(np.nan_to_num(prog, nan=-np.inf) >= th)
        return int(hit[0]) if len(hit) else None
    d50, d90 = first(0.5), first(0.9)
    day0 = float(prog[0]) if np.isfinite(prog[0]) else None
    if d90 is None:
        shape = "불명확"
    elif fp.code == "D":
        shape = "즉시 전환" if d90 <= 2 else ("점진 이탈" if d90 >= 5 else "빠른 전환")
    else:
        shape = "즉시 전환" if d90 == 0 else "점진 이탈"
    return {"days_to_50pct": d50, "days_to_90pct": d90, "progress_at_start": None if day0 is None else round(day0, 3),
            "shape": shape, "unit": fp.label}


def _persistence(sig: Signal, i0: int, step: float) -> str:
    a, b = sig.fp.persist
    if i0 + b > len(sig.r_adj) or not np.isfinite(step) or abs(step) < 1e-9:
        return "알 수 없음(기간 부족)"
    seg = sig.r_adj[i0 + a:i0 + b]
    if np.isfinite(seg).sum() < 0.5 * (b - a):
        return "알 수 없음(결측)"
    long = float(np.nanmedian(seg)) - sig.pre[i0]
    ratio = long / step
    if ratio >= 0.5:
        return "지속"
    if ratio <= 0.25:
        return "일시적(복귀)"
    return "부분 복귀"


def test_break_at(sig: Signal, date, alpha: float = 0.05, min_step: float = 0.05) -> BreakTest:
    fp = sig.fp
    t0 = pd.Timestamp(date)
    if t0 < sig.index[0] or t0 > sig.index[-1]:
        raise ValueError(f"데이터 기간 밖의 시점: {date}")
    i0 = int(sig.index.get_indexer([t0], method="bfill")[0])
    step = sig.step[i0]
    adjusted = bool(np.isfinite(step))
    if not adjusted:
        return BreakTest(fmt(sig.index[i0], fp.code), None, None, None, None, None, 0, False, False,
                         note="계절 보정에 필요한 과거 주기나 정책 전후 창의 데이터가 부족함")
    ex = _excluded_near(len(sig.step), i0, fp)
    plac = sig.step[(~ex) & np.isfinite(sig.step)]
    n = len(plac)
    if n < 20:
        return BreakTest(fmt(sig.index[i0], fp.code), float(step), float(np.expm1(step)), None, None, None, n,
                         False, True, note="위약 비교에 쓸 시점이 20개 미만")
    p = (1 + np.sum(np.abs(plac) >= abs(step))) / (1 + n)
    med = np.median(plac)
    mad = 1.4826 * np.median(np.abs(plac - med))
    z = (step - med) / mad if mad > 0 else np.inf
    lo, hi = np.quantile(plac - med, [0.025, 0.975])
    ci = (float(step - hi), float(step - lo))
    sig_ok = bool(p < alpha and abs(step) >= min_step)
    return BreakTest(fmt(sig.index[i0], fp.code), float(step), float(np.expm1(step)),
                     (round(ci[0], 4), round(ci[1], 4)), float(p), float(z), n, sig_ok, True,
                     onset=_onset(sig, i0, sig.post[i0] - sig.pre[i0]) if sig_ok else {},
                     persistence=_persistence(sig, i0, step) if sig_ok else "해당 없음")


# --------------------------------------------------------------------------- 분류 변경(E5)
def _reclass_evidence(wide: pd.DataFrame, target: str, refs: list[str], i0: int, step_t: float,
                      freq: str, mask: np.ndarray | None, alpha: float = 0.05) -> list[dict]:
    """같은 때 다른 분류가 대상의 변화를 흡수했는지 본다.

    - 신규 분류: 단절 시점 ±허용 오차 안에 처음 나타난 분류. 생긴 뒤 수준을 대상의 변화량과 비교한다.
    - 기존 분류로 이동(합계 보존): 그 분류가 반대 방향으로 뚜렷하게 바뀌었고(p<0.01, 강건 z≥3), 두 분류를 합친 값은
      정책 시점에 거의 변하지 않았다(독립적인 변화라면 기대되는 크기의 25% 이하). 강한 근거.
    - 기존 분류로 이동: 그 분류도 대조 집단 대비(계절 보정) 유의하게 반대 방향으로 바뀌었고,
      바뀐 양이 대상 변화량의 30~150%일 때만 인정한다. 유의수준은 분류 수로 나누고(상한 0.01),
      강건 z도 3.5 이상이어야 한다. 방학처럼 두 분류가 함께 움직이는 계절 변화를
      '이동'으로 오인하지 않기 위해서다.
    """
    fp = freq_params(freq)
    w, s = fp.window, fp.settle
    n = len(wide)
    pre_sl = slice(max(0, i0 - w), i0)
    post_sl = slice(min(n, i0 + s), min(n, i0 + s + w))
    if not np.isfinite(step_t) or step_t == 0:
        return []

    def level(col, sl):
        x = wide[col].to_numpy(dtype=float)[sl].copy()
        if mask is not None:
            x[np.asarray(mask)[sl]] = np.nan
        return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan

    t_pre = level(target, pre_sl)
    d_target = t_pre * np.expm1(step_t)          # 계절 보정한 대상 변화량(원 단위)
    if not np.isfinite(d_target) or d_target == 0:
        return []
    out = []
    date = wide.index[min(i0, n - 1)]
    n_others = sum(1 for g in wide.columns if g != target and g not in refs)
    for g in wide.columns:
        if g == target or g in refs:
            continue
        col = wide[g].to_numpy(dtype=float)
        valid = np.flatnonzero(np.isfinite(col) & (col > 0))
        if len(valid) == 0:
            continue
        first = int(valid[0])
        born = (first > 2 * fp.near) and abs(first - i0) <= fp.near
        if born:
            g_post = level(g, post_sl)
            offset = g_post / (-d_target) if np.isfinite(g_post) else np.nan
            if np.isfinite(offset) and offset > 0.05:
                out.append({"group": g, "kind": "신규 분류 출현", "first_seen": fmt(wide.index[first], fp.code),
                            "offset_share": round(float(offset), 3),
                            "text": f"같은 때 '{g}' 분류가 새로 생김({fmt(wide.index[first], fp.code)}). "
                                    f"'{target}' 변화량의 {offset * 100:.0f}%에 해당"})
            continue
        if first >= i0:
            continue
        try:
            gt = test_break_at(build_signal(wide, g, refs, freq, mask), date, alpha)
        except (ValueError, KeyError):
            continue
        if gt.step is None or gt.p_rank is None or np.sign(gt.step) == np.sign(step_t):
            continue
        # 합계 보존: 진짜 재분류라면 두 분류를 합친 값은 대조 집단 대비 변하지 않는다
        g_pre = level(g, pre_sl)
        if gt.p_rank < 0.01 and abs(gt.z or 0) >= 3 and np.isfinite(g_pre) and g_pre > 0:
            comb = test_break_at(build_signal(wide, [target, g], refs, freq, mask), date, alpha)
            expected_if_independent = np.log1p(d_target / (t_pre + g_pre))
            if (comb.step is not None and not comb.significant
                    and abs(comb.step) <= 0.25 * abs(expected_if_independent)):
                offset = g_pre * np.expm1(gt.step) / (-d_target)
                out.append({"group": g, "kind": "기존 분류로 이동(합계 보존)", "offset_share": round(float(offset), 3),
                            "combined_pct": round(float(np.expm1(comb.step)), 4),
                            "text": f"같은 때 '{g}'가 반대 방향으로 {gt.pct * 100:+.0f}% 바뀌었고(p={gt.p_rank:.3f}), "
                                    f"두 분류의 합계는 {np.expm1(comb.step) * 100:+.1f}%로 거의 그대로임. "
                                    + (f"'{target}'에서 '{g}'로" if step_t < 0 else f"'{g}'에서 '{target}'로")
                                    + " 옮겨 집계됐을 가능성"})
                continue
        # 여러 분류를 함께 보므로 유의수준을 분류 수로 나눈다(본페로니), 기본 상한 0.01
        if (gt.p_rank is None or gt.p_rank >= min(0.01, alpha / max(1, n_others))
                or abs(gt.z or 0) < 3.5 or np.sign(gt.step) == np.sign(step_t)):
            continue
        d_g = level(g, pre_sl) * np.expm1(gt.step)
        offset = d_g / (-d_target)
        # 옮겨 간 양은 빠진 양의 30~150% 범위여야 '이동'으로 본다
        if np.isfinite(offset) and 0.3 <= offset <= 1.5:
            out.append({"group": g, "kind": "다른 분류로 이동 의심", "offset_share": round(float(offset), 3),
                        "text": f"같은 때 '{g}'가 반대 방향으로 {gt.pct * 100:+.0f}% 바뀜(계절 보정, p={gt.p_rank:.3f}). "
                                f"'{target}' 변화량의 {offset * 100:.0f}%에 해당"})
    return out


# --------------------------------------------------------------------------- 종합 진단
@dataclass
class Evidence:
    code: str
    name: str
    strength: str     # 강함 / 보통
    met: bool | None  # None = 판단할 정보 없음
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Diagnosis:
    target: str
    reference: list[str]
    date: str
    freq: str
    verdict: str          # 점검 불가 / 단절 없음 / 측정 단절 증거 없음 / 확인 필요 / 측정 단절 의심 / 측정 단절 의심(강) / 전체 집계 이상
    judgement_cap: str | None   # 이후 효과 판정의 상한 (None = 제한 없음)
    message: str
    test: BreakTest
    reference_test: BreakTest
    evidence: list[Evidence]
    external: list[ExternalCheck]
    reclass: list[dict]
    needed_data: list[str]

    @property
    def suspected(self) -> bool:
        return self.verdict.startswith("측정 단절 의심") or self.verdict == "전체 집계 이상"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["suspected"] = self.suspected
        return d

    def report_ko(self) -> str:
        t = self.test
        lines = [f"[{self.verdict}] {self.message}"]
        if t.step is not None:
            ci = f", 95% 구간 {np.expm1(t.ci[0]) * 100:+.0f}% ~ {np.expm1(t.ci[1]) * 100:+.0f}%" if t.ci else ""
            lines.append(f"- 정책 시점 변화(대조 집단 대비, 계절 보정): {t.pct * 100:+.1f}%{ci}, "
                         f"위약 순위 p={t.p_rank:.3f} (비교 시점 {t.n_placebo}개)")
        if t.onset:
            o = t.onset
            lines.append(f"- 전환 모양: {o['shape']} (90% 도달 {o['days_to_90pct']}{o['unit']}), 지속성: {t.persistence}")
        for e in self.evidence:
            mark = "●" if e.met else ("○" if e.met is False else "–")
            lines.append(f"  {mark} {e.code} {e.name}[{e.strength}]: {e.detail}")
        if self.needed_data:
            lines.append("- 판정에 필요한 데이터: " + " / ".join(self.needed_data))
        return "\n".join(lines)


def diagnose(wide: pd.DataFrame, target: str, reference: str | list[str], date, freq: str, *,
             expected: str | None = None, external: list[ExternalStat] | None = None,
             plausible_max: float = LOG2, mask: np.ndarray | None = None,
             alpha: float = 0.05, needed_hint: list[str] | None = None) -> Diagnosis:
    """정책 시점(date)에서 대상 집단(target)의 측정 단절 여부를 진단한다.

    expected: 정책이 대상 집단의 값을 늘릴 것으로 기대하면 'increase', 줄일 것으로 기대하면 'decrease'
    external: 대조할 공식 통계 목록
    plausible_max: 몇 주 사이 로그 변화가 이보다 크면 '비현실적 크기'(기본 log 2 = 절반 또는 두 배)
    mask: 분석에서 뺄 시점(부분 집계일 등)
    """
    refs = [reference] if isinstance(reference, str) else list(reference)
    fp = freq_params(freq)
    sig = build_signal(wide, target, refs, freq, mask)
    test = test_break_at(sig, date, alpha)
    ref_sig = build_signal(wide, refs, refs, freq, mask, log_level=True)
    ref_test = test_break_at(ref_sig, date, alpha, min_step=0.15)
    i0 = int(sig.index.get_indexer([pd.Timestamp(date)], method="bfill")[0])

    ext = [external_check(wide, s, freq) for s in (external or [])]
    reclass = (_reclass_evidence(wide, target, refs, i0, test.step, freq, mask, alpha)
               if test.significant else [])

    ev: list[Evidence] = []
    step = test.step if test.step is not None else np.nan
    # E2 방향
    if expected in ("increase", "decrease") and test.significant:
        opposite = (step < 0) if expected == "increase" else (step > 0)
        ev.append(Evidence("E2", "방향 모순", "보통", bool(opposite),
                           f"기대 방향은 {'증가' if expected == 'increase' else '감소'}, 관측은 "
                           f"{'감소' if step < 0 else '증가'} ({np.expm1(step) * 100:+.0f}%)"))
    else:
        ev.append(Evidence("E2", "방향 모순", "보통", None,
                           "기대 방향 정보 없음" if expected is None else "정책 시점 단절이 유의하지 않음"))
    # E3 크기
    if test.significant:
        ev.append(Evidence("E3", "비현실적 크기", "보통", bool(abs(step) > plausible_max),
                           f"변화 {np.expm1(step) * 100:+.0f}% (기준: {np.expm1(-plausible_max) * 100:.0f}% ~ "
                           f"{np.expm1(plausible_max) * 100:+.0f}% 밖이면 해당)"))
    else:
        ev.append(Evidence("E3", "비현실적 크기", "보통", None, "정책 시점 단절이 유의하지 않음"))
    # E4 외부 통계
    if ext:
        bad = [x for x in ext if x.contradiction]
        ev.append(Evidence("E4", "외부 통계 불일치", "강함", bool(bad),
                           " / ".join(x.message for x in ext)))
    else:
        ev.append(Evidence("E4", "외부 통계 불일치", "강함", None, "대조할 공식 통계 없음"))
    # E5 분류 변경: 새 분류 출현은 강한 근거, 기존 분류로의 이동은 우연히 겹친 변화일 수 있어 보통 근거
    if test.significant:
        born = [x for x in reclass if x["kind"] in ("신규 분류 출현", "기존 분류로 이동(합계 보존)")]
        strength = "강함" if born or not reclass else "보통"
        ev.append(Evidence("E5", "분류 변경", strength, bool(reclass),
                           " / ".join(x["text"] for x in reclass) if reclass else "같은 때 새로 생기거나 감소분을 흡수한 분류 없음"))
    else:
        ev.append(Evidence("E5", "분류 변경", "강함", None, "정책 시점 단절이 유의하지 않음"))

    strong = sum(1 for e in ev if e.met and e.strength == "강함")
    moderate = sum(1 for e in ev if e.met and e.strength == "보통")
    needed = list(needed_hint or [])
    d = fmt(sig.index[i0], fp.code)

    if test.step is None:
        verdict, cap = "점검 불가", None
        msg = f"{d} 전후 데이터가 부족해 측정 단절을 점검할 수 없음 ({test.note})"
        needed = needed or [f"{d} 전후 각 {fp.window}{fp.label} 이상의 자료와 과거 같은 시기 자료"]
    elif ref_test.significant:
        verdict, cap = "전체 집계 이상", "판정 보류"
        msg = (f"{d} 무렵 대조 집단({', '.join(refs)}) 자체도 {ref_test.pct * 100:+.0f}% 바뀜. "
               f"정책과 무관한 집계 이상(누락·시스템 변경)일 수 있어 집단 간 비교가 성립하는지 먼저 확인해야 함")
        needed = needed or [f"{d} 전후 데이터 제공처의 집계 이력(누락·시스템 변경 여부)"]
    elif not test.significant:
        if ext and any(x.contradiction for x in ext):
            verdict, cap = "확인 필요", "조건부"
            msg = f"{d}에 뚜렷한 단절은 없지만 공식 통계와 규모가 맞지 않음. 집계 기준 확인 필요"
        else:
            verdict, cap = "단절 없음", None
            msg = f"{d}에 '{target}'의 뚜렷한 단절이 없음. 측정 체계 변경의 신호 없음"
    elif strong >= 2 or (strong >= 1 and moderate >= 1):
        verdict, cap = "측정 단절 의심(강)", "판정 보류"
        msg = (f"{d}부터 '{target}'가 {test.pct * 100:+.0f}% 바뀌었지만, 근거 {strong + moderate}개가 "
               f"정책 효과보다 집계 방식 변경을 가리킴")
    elif strong == 1 or moderate >= 2:
        verdict, cap = "측정 단절 의심", "판정 보류"
        msg = (f"{d}부터 '{target}'가 {test.pct * 100:+.0f}% 바뀜. 집계 방식 변경일 가능성이 있어 "
               f"효과 판정을 보류함")
    elif moderate == 1:
        verdict, cap = "확인 필요", "조건부"
        which = next(e for e in ev if e.met)
        msg = (f"{d}부터 '{target}'가 {test.pct * 100:+.0f}% 바뀜. 근거 1개({which.name})만 집계 변경을 가리킴. "
               f"실제 효과일 수도 있어 공식 통계 대조가 필요함")
        if not needed:
            needed = ["같은 기간의 공식 집계(행정 통계·보도자료)"]
    else:
        verdict, cap = "측정 단절 증거 없음", None
        msg = (f"{d}부터 '{target}'가 {test.pct * 100:+.0f}% 바뀜. 방향·크기·분류 모두 정상 범위라 "
               f"효과 추정으로 넘어감")

    if verdict.startswith("측정 단절 의심") and not needed:
        needed = [f"정책 전후 같은 기준으로 집계한 '{target}' 자료",
                  "데이터 제공처의 집계 기준 변경 이력"]
    return Diagnosis(target, refs, d, fp.code, verdict, cap, msg, test, ref_test, ev, ext, reclass, needed)


# --------------------------------------------------------------------------- 시점을 모를 때: 전 기간 탐색
@dataclass
class BreakCandidate:
    date: str          # 전환 시작일 추정
    peak: str          # 단절 통계가 가장 큰 시점
    step: float
    pct: float
    z: float
    shape: str
    persistence: str
    reclass: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def scan_breaks(wide: pd.DataFrame, target: str, reference: str | list[str], freq: str, *,
                z_min: float = 4.0, min_step: float = 0.15, top: int = 10,
                mask: np.ndarray | None = None) -> list[BreakCandidate]:
    """시점을 모르는 단절을 찾는다. 계절 보정 단절 통계의 강건 z값이 큰 지점을 고른다."""
    refs = [reference] if isinstance(reference, str) else list(reference)
    fp = freq_params(freq)
    sig = build_signal(wide, target, refs, freq, mask)
    st = sig.step
    ok = np.isfinite(st)
    if ok.sum() < 30:
        return []
    med = np.median(st[ok])
    mad = 1.4826 * np.median(np.abs(st[ok] - med)) or 1e-9
    z = np.where(ok, (st - med) / mad, 0.0)
    hit = (np.abs(z) >= z_min) & (np.abs(np.nan_to_num(st)) >= min_step)
    order = np.argsort(-np.abs(z))
    rad = fp.window + fp.settle
    taken = np.zeros(len(st), dtype=bool)
    found: list[tuple[int, int]] = []    # (peak, onset)
    for i in order:
        if not hit[i] or taken[i]:
            continue
        # 같은 단절의 이웃 시점은 하나로 묶는다
        taken[max(0, i - rad):i + rad + 1] = True
        # 메아리 제거: 더 큰 단절의 k주기 뒤이고 부호가 반대면 계절 보정의 잔상으로 본다
        echo = any(abs(abs(i - p) - k * fp.season) <= rad and np.sign(st[i]) != np.sign(st[p])
                   for p, _ in found for k in range(1, fp.years_back + 1))
        if echo:
            continue
        found.append((i, _refine_onset(sig, i)))
        if len(found) >= top:
            break
    out = []
    for i, j in found:
        total = sig.post[i] - sig.pre[i]
        on = _onset(sig, j, total)
        out.append(BreakCandidate(fmt(sig.index[j], fp.code), fmt(sig.index[i], fp.code), round(float(st[i]), 4),
                                  round(float(np.expm1(st[i])), 4), round(float(z[i]), 2), on.get("shape", "불명확"),
                                  _persistence(sig, j, st[i]),
                                  _reclass_evidence(wide, target, refs, j, float(st[i]), freq, mask)))
    return out


def _refine_onset(sig: Signal, i: int) -> int:
    """단절 통계가 큰 지점 i 주변에서 전환이 시작된 시점을 찾는다.

    1) 변화점: i를 둘러싼 고정 구간을 두 조각으로 나눠 각자의 중앙값으로 설명할 때
       절대 오차 합이 가장 작은 분할점
    2) 시작일: 변화점에서 거슬러 올라가며, 7기간 중앙값이 전체 변화의 15% 이상인 상태가
       끊김 없이 이어지는 가장 이른 시점 (점진적으로 빠지는 단절의 시작을 잡는다)
    """
    fp = sig.fp
    x = sig.r_adj if np.isfinite(sig.r_adj).sum() else sig.r
    w = fp.window
    n = len(x)
    # 고정 구간 [L, R)을 두 조각으로 나눌 때 각 조각의 중앙값 절대 오차 합이 최소인 분할점
    L, R = max(0, i - 2 * w - fp.settle), min(n, i + 2 * w + fp.settle)
    minp = max(2, w // 4)
    best, bj = np.inf, i
    for j in range(L + minp, R - minp + 1):
        a, b = x[L:j], x[j:R]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < minp or len(b) < minp:
            continue
        cost = np.abs(a - np.median(a)).sum() + np.abs(b - np.median(b)).sum()
        if cost < best:
            best, bj = cost, j
    seg_a, seg_b = x[max(0, bj - w):bj], x[bj:bj + w]
    if not (np.isfinite(seg_a).any() and np.isfinite(seg_b).any()):
        return i
    pre, post = np.nanmedian(seg_a), np.nanmedian(seg_b)
    total = post - pre
    if not np.isfinite(total) or total == 0:
        return bj
    short = 7 if fp.code == "D" else 1
    sm = pd.Series(x).rolling(short, center=True, min_periods=1).median().to_numpy()
    j = bj
    while j - 1 >= bj - w and np.isfinite(sm[j - 1]) and (sm[j - 1] - pre) / total >= 0.15:
        j -= 1
    return j
