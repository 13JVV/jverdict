"""데이터 커버리지 점검: 효과를 추정하기 전에 '데이터가 제대로 있는가'를 본다.

찾는 것
- 전체 결측: 모든 집단의 행이 없는 기간
- 부분 집계: 행은 있지만 합계가 평소의 일부(기본 30% 미만)뿐인 기간
- 집단 결측: 다른 집단은 있는데 특정 집단만 행이 없는 기간 (같은 기간에 빠진 집단은 하나로 묶음)
- 분류 신설·소멸: 데이터 중간에 새 집단(분류)이 생기거나 사라짐. 분류 체계 변경의 흔적
- 단위 누락: 집계 단위(정류소·업종 등) 수가 평소보다 크게 줄어든 기간

원칙: 원인을 단정하지 않는다. "무엇이, 언제부터 언제까지, 얼마나"만 사실로 적는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ._panel import fmt, freq_params, infer_freq, ranges, to_wide

SEVERITY_ORDER = {"높음": 0, "중간": 1, "참고": 2}


@dataclass
class Finding:
    kind: str
    severity: str
    groups: list[str]
    start: str
    end: str
    length: int
    message: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CoverageReport:
    freq: str
    period: tuple[str, str]
    n_periods: int
    groups: list[str]
    findings: list[Finding]

    def by_kind(self, kind: str) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]

    def affects(self, start: str, end: str, groups: list[str] | None = None) -> list[Finding]:
        """분석 기간·집단과 겹치는 발견만 고른다 (Agent가 설계 단계에서 씀)."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        out = []
        for f in self.findings:
            if pd.Timestamp(f.end) < s or pd.Timestamp(f.start) > e:
                continue
            if groups and f.groups and not (set(groups) & set(f.groups)):
                continue
            out.append(f)
        return out

    def summary_ko(self) -> str:
        if not self.findings:
            return f"커버리지 이상 없음 ({self.period[0]} ~ {self.period[1]}, {self.n_periods}개 기간)"
        lines = [f"커버리지 점검 {self.period[0]} ~ {self.period[1]}: 발견 {len(self.findings)}건"]
        for f in self.findings:
            lines.append(f"- [{f.severity}] {f.kind}: {f.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"freq": self.freq, "period": list(self.period), "n_periods": self.n_periods,
                "groups": self.groups, "findings": [f.to_dict() for f in self.findings]}


def _typical_level(total: np.ndarray, w: int) -> np.ndarray:
    """각 시점의 '평소 수준' = 앞쪽 유효값 w개 중앙값과 뒤쪽 유효값 w개 중앙값 중 큰 값.

    공백 직후처럼 한쪽이 비정상이어도 다른 쪽이 기준을 잡아 준다.
    """
    valid = ~np.isnan(total)
    idx = np.flatnonzero(valid)
    v = total[valid]
    back = pd.Series(v).rolling(w, min_periods=max(3, w // 3)).median().shift(1).to_numpy()
    fwd = pd.Series(v[::-1]).rolling(w, min_periods=max(3, w // 3)).median().shift(1).to_numpy()[::-1]
    ref = np.fmax(back, fwd)
    out = np.full(len(total), np.nan)
    out[idx] = ref
    return out


def check_coverage(df: pd.DataFrame, time: str, group: str, value: str, *,
                   unit: str | None = None, freq: str | None = None,
                   partial_share: float = 0.30, unit_share: float = 0.70,
                   min_group_share: float = 0.0005) -> CoverageReport:
    """긴 형식 데이터의 커버리지를 점검한다.

    min_group_share: 전체에서 차지하는 비중이 이보다 작은 집단은 '집단 결측' 점검에서 뺀다
                     (아주 작은 집단은 원래 행이 없는 날이 많다).
    """
    freq = freq or infer_freq(df[time])
    fp = freq_params(freq)
    w = to_wide(df, time, group, value, freq)
    idx = w.index
    groups = list(w.columns)
    findings: list[Finding] = []

    present = w.notna().to_numpy()
    any_row = present.any(axis=1)

    # 1) 전체 결측
    for s, e, n in ranges(idx, ~any_row):
        findings.append(Finding(
            "전체 결측", "높음", [], fmt(s, freq), fmt(e, freq), n,
            f"{fmt(s, freq)} ~ {fmt(e, freq)} {n}{fp.label} 동안 모든 집단의 행이 없음",
            {"n_periods": n}))

    # 2) 부분 집계: 합계가 평소의 partial_share 미만, 또는 합계 60% 미만이면서 집계 단위도 함께 줄어든 기간
    total = w.sum(axis=1, min_count=1).to_numpy()
    typical = _typical_level(total, fp.partial_ref)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = total / typical
    ushare = None
    if unit is not None:
        u = df[[time, unit]].copy()
        u[time] = pd.to_datetime(u[time])
        nu = u.groupby(time)[unit].nunique().reindex(idx).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ushare = nu / _typical_level(nu, fp.partial_ref)
    partial = any_row & (share < partial_share)
    if ushare is not None:
        partial |= any_row & (share < 0.60) & (ushare < 0.80)
    short = []
    for s, e, n in ranges(idx, partial):
        sel = (idx >= s) & (idx <= e)
        seg = share[sel]
        ev = {"min_share": round(float(np.nanmin(seg)), 4), "max_share": round(float(np.nanmax(seg)), 4)}
        if ushare is not None:
            ev["unit_share_min"] = round(float(np.nanmin(ushare[sel])), 4)
        if n >= 3:
            findings.append(Finding(
                "부분 집계", "높음", [], fmt(s, freq), fmt(e, freq), n,
                f"{fmt(s, freq)} ~ {fmt(e, freq)} {n}{fp.label}: 합계가 평소의 "
                f"{np.nanmin(seg) * 100:.0f}~{np.nanmax(seg) * 100:.0f}% 수준", ev))
        else:
            short.append((s, e, n, ev))
    if short:
        days = sum(x[2] for x in short)
        lows = [x[3]["min_share"] for x in short]
        listed = [fmt(s, freq) if n == 1 else f"{fmt(s, freq)}~{fmt(e, freq)}" for s, e, n, _ in short]
        findings.append(Finding(
            "부분 집계", "중간", [], listed[0].split("~")[0], fmt(short[-1][1], freq), days,
            f"짧은 부분 집계 {len(short)}건({days}{fp.label}): 합계가 평소의 "
            f"{min(lows) * 100:.0f}~{max(x[3]['max_share'] for x in short) * 100:.0f}%. "
            f"해당 {fp.label}은 분석에서 빼거나 가중치를 낮춰야 함",
            {"periods": listed, "n_runs": len(short)}))

    # 3) 분류 신설·소멸
    first_ok = 2 * fp.near
    lifespans: dict[str, tuple[int, int]] = {}
    for j, g in enumerate(groups):
        pos = np.flatnonzero(present[:, j])
        if len(pos) == 0:
            continue
        a, b = int(pos[0]), int(pos[-1])
        lifespans[g] = (a, b)
        if a > first_ok:
            first_vals = w[g].iloc[a:a + fp.window].dropna()
            findings.append(Finding(
                "분류 신설", "중간", [g], fmt(idx[a], freq), fmt(idx[a], freq), 1,
                f"'{g}' 분류가 {fmt(idx[a], freq)}에 처음 나타남 (데이터 시작은 {fmt(idx[0], freq)}). "
                f"분류 체계 변경 여부 확인 필요",
                {"first_window_median": float(first_vals.median()) if len(first_vals) else None}))
        if b < len(idx) - 1 - first_ok:
            findings.append(Finding(
                "분류 소멸", "중간", [g], fmt(idx[b], freq), fmt(idx[b], freq), 1,
                f"'{g}' 분류가 {fmt(idx[b], freq)} 이후 나타나지 않음 (데이터 끝은 {fmt(idx[-1], freq)})", {}))

    # 4) 집단 결측 (집단의 존속 기간 안에서, 다른 집단은 있는데 이 집단만 없음)
    grand = np.nansum(w.to_numpy())
    masks: dict[str, np.ndarray] = {}
    for j, g in enumerate(groups):
        if g not in lifespans or np.nansum(w[g].to_numpy()) < min_group_share * grand:
            continue
        a, b = lifespans[g]
        m = np.zeros(len(idx), dtype=bool)
        m[a:b + 1] = (~present[a:b + 1, j]) & any_row[a:b + 1]
        masks[g] = m
    blocks: dict[tuple[int, int], list[str]] = {}
    for g, m in masks.items():
        for s, e, n in ranges(idx, m):
            blocks.setdefault((idx.get_loc(s), idx.get_loc(e)), []).append(g)
    for (a, b), gs in sorted(blocks.items()):
        n = b - a + 1
        others = [g for g in groups if g not in gs and present[a:b + 1, groups.index(g)].any()]
        if n >= 2 and (len(gs) >= 2 or n >= fp.near):
            sev = "높음"
        elif n >= 2 or len(gs) >= 2:
            sev = "중간"
        else:
            sev = "참고"
        findings.append(Finding(
            "집단 결측", sev, sorted(gs), fmt(idx[a], freq), fmt(idx[b], freq), n,
            f"{fmt(idx[a], freq)} ~ {fmt(idx[b], freq)} {n}{fp.label} 동안 {', '.join(sorted(gs))}의 행이 없음 "
            f"(같은 기간 {', '.join(others[:5])}{' 등' if len(others) > 5 else ''}은 있음)",
            {"n_groups": len(gs), "present_groups": others}))

    # 5) 단위 누락
    if ushare is not None:
        low = any_row & (ushare < unit_share) & ~partial
        for s, e, n in ranges(idx, low):
            seg = ushare[(idx >= s) & (idx <= e)]
            findings.append(Finding(
                "단위 누락", "중간", [], fmt(s, freq), fmt(e, freq), n,
                f"{fmt(s, freq)} ~ {fmt(e, freq)} {n}{fp.label}: 집계 {unit} 수가 평소의 "
                f"{np.nanmin(seg) * 100:.0f}~{np.nanmax(seg) * 100:.0f}%",
                {"min_share": round(float(np.nanmin(seg)), 4)}))

    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.start))
    return CoverageReport(freq=freq, period=(fmt(idx[0], freq), fmt(idx[-1], freq)),
                          n_periods=len(idx), groups=groups, findings=findings)


def exclusion_mask(report: CoverageReport, index: pd.DatetimeIndex,
                   kinds: tuple[str, ...] = ("부분 집계",)) -> np.ndarray:
    """커버리지 발견 중 분석에서 뺄 기간을 참/거짓 배열로 만든다 (기본: 부분 집계일)."""
    m = np.zeros(len(index), dtype=bool)
    for f in report.findings:
        if f.kind not in kinds:
            continue
        spans = f.evidence.get("periods") or [f"{f.start}~{f.end}"]
        for sp in spans:
            a, _, b = sp.partition("~")
            s, e = pd.Timestamp(a), pd.Timestamp(b or a)
            m |= (index >= s) & (index <= e)
    return m
