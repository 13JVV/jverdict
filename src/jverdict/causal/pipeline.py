"""판정 파이프라인: 품질 점검 → 추정 → 반론 검증 → 검정력 → 판정.

Agent(3단계)와 템플릿 모드가 같은 함수를 호출한다. 설계(Design)만 다르게 넘긴다.

analyze_series(wide, SeriesDesign)  집계 시계열 (버스 이용자 유형별 일 합계, 카드 업종군 월 합계 등)
analyze_panel(df, PanelDesign)      단위 패널 (집단 × 업종 × 월 등)
"""
from __future__ import annotations

import time as _time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from jverdict.quality._panel import fmt
from jverdict.quality.breaks import Diagnosis, diagnose
from jverdict.quality.external import ExternalStat

from . import checks as C
from .estimate import (DEFAULTS, Estimate, EventStudy, apply_group_level, did, did_trend, event_study, gap_series,
                       group_level, its_fit, series_pretrend)
from .power import Power, assess, pre_outliers
from .stats import t_ci
from .verdict import RULES_VERSION, Verdict, judge

MIN_PRE = {"D": 90, "MS": 3}


@dataclass
class Analysis:
    kind: str
    question: str
    design: dict
    main: Estimate | None
    verdict: Verdict
    pretrend: dict
    trend_adjusted: Estimate | None
    placebo_time: C.PlaceboTime | None
    placebo_space: C.PlaceboSpace | None
    sensitivity: C.Sensitivity | None
    double: C.DoubleCompute | None
    power: Power | None
    quality: Diagnosis | None
    event_study: EventStudy | None = None
    curve: dict | None = None
    seconds: float = 0.0
    verdict_v3: Verdict | None = None      # 개정 규칙 v3로 같은 결과를 다시 판정(공식 판정은 verdict, v2)

    def to_dict(self) -> dict:
        def dd(x):
            return None if x is None else x.to_dict()
        return {"kind": self.kind, "question": self.question, "design": self.design,
                "rules_version": RULES_VERSION, "verdict": self.verdict.to_dict(), "main": dd(self.main),
                "pretrend": self.pretrend, "trend_adjusted": dd(self.trend_adjusted),
                "placebo_time": dd(self.placebo_time), "placebo_space": dd(self.placebo_space),
                "sensitivity": dd(self.sensitivity), "double_compute": dd(self.double),
                "power": dd(self.power), "quality": dd(self.quality), "event_study": dd(self.event_study),
                "curve": self.curve, "seconds": self.seconds,
                "verdict_v3": None if self.verdict_v3 is None else self.verdict_v3.to_dict()}

    def report_ko(self) -> str:
        v = self.verdict
        L = [f"{v.light} {v.label}", f"질문: {self.question}", f"- {v.headline}"]
        for r in v.reasons[1:]:
            L.append(f"- {r}")
        if self.main is not None:
            L.append(f"- 주분석({self.main.method}, {self.main.inference}): {self.main.short_ko()}, "
                     f"사전 {self.main.n_pre} · 사후 {self.main.n_post}")
        if self.placebo_time is not None and self.placebo_time.n:
            pt = self.placebo_time
            L.append(f"- 시점 위약: 가짜 시행일 {pt.n}개, 양측 순위 p={pt.p_two:.3f}, 가짜 효과 표준편차 {pt.sd:.3f}")
        for c in v.checks:
            L.append(f"  [{c.status}] {c.code} {c.name}: {c.detail}")
        if self.power is not None:
            L.append(f"- 검정력: {self.power.sentence_ko()}")
        if v.needed_data:
            L.append("- 판정에 필요한 데이터: " + " / ".join(v.needed_data))
        return "\n".join(L)


def _quality_reason(dg: Diagnosis | None) -> tuple[str | None, str]:
    if dg is None:
        return None, ""
    return dg.judgement_cap, f"{dg.verdict}. {dg.message}" if dg.judgement_cap else dg.verdict


# =========================================================================== 시계열
@dataclass
class SeriesDesign:
    question: str
    target: str | list[str]
    reference: str | list[str] | None
    date: str
    freq: str = "D"
    expected: str | None = None              # increase / decrease (측정 단절 진단의 방향 모순 판단)
    reference_effect: float = 0.10           # 기준 효과: 정책적으로 의미 있는 최소 크기(비율)
    pre: int | None = None
    post: int | None = None
    season: str = "sdiff"
    season_years: int = 3
    placebo_groups: list = field(default_factory=list)   # 집단 이름, 또는 (가짜 대상, 그 대조) 쌍
    alt_references: dict[str, list[str]] = field(default_factory=dict)
    exclude: list[tuple[str, str]] = field(default_factory=list)   # 시점 위약에서 뺄 기간(다른 정책 등)
    external: list[ExternalStat] | None = None
    needed_hint: list[str] | None = None
    quality: bool = True
    transition: int | None = None            # 민감도 S1에서 뺄 전환기 (기본: 일 28, 월 1)
    alpha: float = 0.05

    def to_dict(self) -> dict:
        d = asdict(self)
        d["external"] = [asdict(e) for e in (self.external or [])]
        return d


def analyze_series(wide: pd.DataFrame, ds: SeriesDesign, *, mask: np.ndarray | None = None,
                   placebo: bool = True) -> Analysis:
    t_start = _time.time()
    freq = "D" if ds.freq == "D" else "MS"
    dflt = DEFAULTS[freq]
    unit = "일" if freq == "D" else "개월"

    dg = None
    if ds.quality and isinstance(ds.target, str) and ds.reference is not None:
        dg = diagnose(wide, ds.target, ds.reference, ds.date, freq, expected=ds.expected,
                      external=ds.external, mask=mask, needed_hint=ds.needed_hint)
    cap, qreason = _quality_reason(dg)
    needed = list(dg.needed_data) if (dg and dg.judgement_cap == "판정 보류") else []

    gs = gap_series(wide, ds.target, ds.reference, freq, season=ds.season, season_years=ds.season_years, mask=mask)
    i0 = gs.pos(ds.date)
    finite = np.flatnonzero(np.isfinite(gs.g_adj[:i0]))
    avail_pre = int(i0 - finite[0]) if len(finite) else 0
    pre = min(ds.pre or dflt["pre"], avail_pre)
    post = ds.post or dflt["post"]
    if ds.season == "sdiff":
        post = min(post, gs.fp.season)
    post_eff = min(post, len(gs.index) - i0)
    cover = None
    if pre < MIN_PRE[freq]:
        cover = f"사전 관측이 {pre}{unit}로 최소 {MIN_PRE[freq]}{unit}보다 짧음"
    main = None if cover else its_fit(gs, i0, pre=pre, post=post, alpha=ds.alpha, label="주분석")
    if main is None and cover is None:
        cover = "정책 전후 창에 유효한 관측이 부족함(결측·제외 기간 확인)"
    if main is None:
        v = judge(None, pretrend_p=None, trend_adjusted=None, placebo_time=None, placebo_space=None, sens=None,
                  double=None, power=None, quality_cap=cap, quality_reason=qreason, coverage_reason=cover,
                  needed_data=needed)
        return Analysis("series", ds.question, ds.to_dict(), None, v, {}, None, None, None, None, None, None, dg,
                        seconds=round(_time.time() - t_start, 2))

    pt_info = series_pretrend(gs, i0, pre=pre)
    trend_adj = its_fit(gs, i0, pre=pre, post=post, trend=True, alpha=ds.alpha, label="추세 보정")

    # 시점 위약
    pl = None
    if placebo:
        # 실제 효과 구간과, 계절 차분이 그 구간을 기준값으로 쓰는 1~K주기 뒤 구간(메아리)은 가짜 창에 들어가면 안 된다
        spans = [0] + ([k * gs.fp.season for k in range(1, ds.season_years + 1)] if ds.season == "sdiff" else [])
        forb = sorted({i0 + s0 + j for s0 in spans for j in range(post_eff)})
        ex = np.zeros(len(gs.index), bool)
        for a, b in ds.exclude:
            ex |= (gs.index >= pd.Timestamp(a)) & (gs.index <= pd.Timestamp(b))
        pl = C.placebo_time(lambda i: its_fit(gs, i, pre=pre, post=post_eff, alpha=ds.alpha, min_frac=0.5),
                            len(gs.index), i0, real=main.effect, window=(pre, post_eff),
                            step=7 if freq == "D" else 1, forbidden=forb, exclude=ex, dates=gs.index,
                            alpha=ds.alpha)

    # 공간 위약
    ps = None
    if ds.placebo_groups:
        fakes = {}
        for g in ds.placebo_groups:
            tg, rf = (g if isinstance(g, (tuple, list)) else (g, ds.reference))
            if tg in wide.columns:
                fg = gap_series(wide, tg, rf, freq, season=ds.season, season_years=ds.season_years, mask=mask)
                fakes[tg] = its_fit(fg, i0, pre=pre, post=post, alpha=ds.alpha)
        ps = C.placebo_space(main, fakes)

    # 민감도
    tr = ds.transition if ds.transition is not None else (28 if freq == "D" else 1)
    specs: dict[str, Estimate | None] = {}
    specs[f"S1 전환기 {tr}{unit} 제외"] = its_fit(gs, i0, pre=pre, post=post, skip=tr, alpha=ds.alpha) \
        if post_eff > tr else None
    if ds.alt_references:
        for k, (name, refs) in enumerate(ds.alt_references.items()):
            ag = gap_series(wide, ds.target, refs, freq, season=ds.season, season_years=ds.season_years, mask=mask)
            specs[f"S2{'abcdef'[k]} 대조 집단 {name}"] = its_fit(ag, i0, pre=pre, post=post, alpha=ds.alpha)
    elif ds.season == "sdiff" and ds.season_years > 1:
        g1 = gap_series(wide, ds.target, ds.reference, freq, season="sdiff", season_years=1, mask=mask)
        specs["S2 계절 기준 직전 1년"] = its_fit(g1, i0, pre=pre, post=post, alpha=ds.alpha)
    half = max(MIN_PRE[freq], pre // 2)
    specs[f"S3 사전 기간 절반({half}{unit})"] = its_fit(gs, i0, pre=half, post=post, alpha=ds.alpha) \
        if half < pre else None
    sens = C.sensitivity(main, specs)

    # 이중 계산: 추세 없는 시계열 효과 = 사후 평균 − 사전 평균 (SQL로 따로)
    a, b = max(0, i0 - pre), min(len(gs.index), i0 + post)
    sql_v = C.series_means_sql(gs.g_adj, i0, a, b)
    dc = C.DoubleCompute("사후 평균 − 사전 평균(" + ("계절 차분 격차" if ds.season == "sdiff" else "격차") + ")",
                         main.effect, sql_v, 1e-9,
                         bool(abs(main.effect - sql_v) <= 1e-9))

    # 월 단위만: 한 달이 기준선을 좌우할 수 있다. 일 단위의 하루짜리 이상(명절·부분 집계)은 HAC 추론이 다룬다.
    outl = pre_outliers(gs.g_adj[max(0, i0 - pre):i0], [fmt(t, freq) for t in gs.index[max(0, i0 - pre):i0]]) \
        if freq == "MS" else []
    pw = assess(main.se, pl.sd if (pl is not None and pl.available) else None, ds.reference_effect,
                main.n_post, unit, ds.alpha, outliers=outl, n_pre=main.n_pre)
    kw = dict(pretrend_p=pt_info.get("p"), trend_adjusted=trend_adj, placebo_time=pl, placebo_space=ps,
              sens=sens, double=dc, power=pw, quality_cap=cap, quality_reason=qreason,
              window_label=f"사후 {main.n_post}{unit}")
    v = judge(main, needed_data=needed, **kw)
    v3 = judge(main, needed_data=needed, rules="v3", **kw)

    win = slice(a, b)
    curve = {"dates": [fmt(t, freq) for t in gs.index[win]], "g_adj": [None if not np.isfinite(x) else float(x)
                                                                         for x in gs.g_adj[win]],
             "counterfactual": main.details["pre_mean"], "policy": fmt(gs.index[i0], freq)}
    return Analysis("series", ds.question, ds.to_dict(), main, v, pt_info, trend_adj, pl, ps, sens, dc, pw, dg,
                    curve=curve, seconds=round(_time.time() - t_start, 2), verdict_v3=v3)


# =========================================================================== 패널
@dataclass
class PanelDesign:
    question: str
    y: str
    unit: str
    time: str
    group: str
    treated: str                              # 처치 집단 이름 (group 열의 값)
    date: str
    ref: str | None = None                    # 이벤트스터디 기준 시점 (기본: 정책 직전 시점)
    fe: list[str] | None = None
    cluster: str | None = None
    weights: str | None = None
    reference_effect: float = 0.20
    placebo_groups: list[str] | None = None   # 기본: 대조 집단 전체
    sens: dict[str, dict] = field(default_factory=dict)   # 이름 → {"filter": callable, "weights": 열}
    placebo_window: tuple[int, int] | None = None          # 시점 위약 (사전, 사후) 시점 수
    window: tuple[int, int] | None = None                  # 주분석 창 (사전, 사후) 시점 수. 없으면 전체
    group_inference: bool = True                           # 처치 집단이 하나: 집단 수준 격차 시계열로 표준오차 보완
    alpha: float = 0.05

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sens"] = {k: {kk: (vv if isinstance(vv, (str, int, float)) else "함수") for kk, vv in v.items()}
                     for k, v in self.sens.items()}
        return d


def _did(df, pd_: PanelDesign, treated_name: str, *, weights=None, label="", full=None) -> Estimate:
    d = df.assign(_tr=(df[pd_.group] == treated_name).astype(int))
    e = did(d, pd_.y, pd_.unit, pd_.time, "_tr", pd_.date, fe=pd_.fe, cluster=pd_.cluster,
            weights=weights, alpha=pd_.alpha, label=label)
    if pd_.group_inference and full is not None:
        f = full[full[pd_.group].isin(d[pd_.group].unique())]
        f = f.assign(_tr=(f[pd_.group] == treated_name).astype(int))
        e = apply_group_level(e, group_level(d, f, pd_.y, pd_.unit, pd_.time, "_tr", pd_.date, pd_.alpha))
    return e


def analyze_panel(df: pd.DataFrame, pd_: PanelDesign, *, quality_cap: str | None = None,
                  quality_reason: str = "") -> Analysis:
    t_start = _time.time()
    df = df.dropna(subset=[pd_.y]).copy()
    df[pd_.time] = pd.to_datetime(df[pd_.time])
    t0 = pd.Timestamp(pd_.date)
    full = df
    all_times = sorted(full[pd_.time].unique())
    if pd_.window:
        wi = int(pd.DatetimeIndex(all_times).searchsorted(t0))
        lo = all_times[max(0, wi - pd_.window[0])]
        hi = all_times[min(len(all_times) - 1, wi + pd_.window[1] - 1)]
        df = full[(full[pd_.time] >= lo) & (full[pd_.time] <= hi)].copy()
    times = sorted(df[pd_.time].unique())
    pre_times = [t for t in times if t < t0]
    cover = None
    if len(pre_times) < 2 or not any(t >= t0 for t in times):
        cover = "정책 전후 시점이 부족함"
    if (df[pd_.group] == pd_.treated).sum() == 0:
        cover = "처치 집단 관측 없음"
    if cover:
        v = judge(None, pretrend_p=None, trend_adjusted=None, placebo_time=None, placebo_space=None, sens=None,
                  double=None, power=None, coverage_reason=cover)
        return Analysis("panel", pd_.question, pd_.to_dict(), None, v, {}, None, None, None, None, None, None, None)

    main = _did(df, pd_, pd_.treated, weights=pd_.weights, label="주분석", full=full)
    d1 = df.assign(_tr=(df[pd_.group] == pd_.treated).astype(int))
    ref = pd.Timestamp(pd_.ref) if pd_.ref else pd.Timestamp(pre_times[-1])
    es = event_study(d1, pd_.y, pd_.unit, pd_.time, "_tr", t0, ref, fe=pd_.fe, cluster=pd_.cluster,
                     weights=pd_.weights, alpha=pd_.alpha)
    pretrend = {"available": es.pretrend_p is not None, "method": f"이벤트스터디 사전 계수 {es.pretrend_q}개 결합 검정(업종 군집)",
                "p": es.pretrend_p, "F": es.pretrend_F, "ref": es.ref}
    ta = did_trend(d1, pd_.y, pd_.unit, pd_.time, "_tr", t0, fe=pd_.fe, cluster=pd_.cluster,
                   weights=pd_.weights, alpha=pd_.alpha)
    if pd_.group_inference:
        f1 = full.assign(_tr=(full[pd_.group] == pd_.treated).astype(int))
        glp = group_level(d1, f1, pd_.y, pd_.unit, pd_.time, "_tr", t0, pd_.alpha)
        if glp and "pretrend" in glp:
            # 집단 공통 충격이 있으면 업종 군집 결합 검정은 과도하게 기각한다. 사전 6시점 이상이면 집단 수준 추세 검정을 쓴다
            pretrend = {"available": True, "method": f"집단 수준 격차의 사전 선형 추세(AR(1), φ={glp['phi']:.2f})",
                        "p": glp["pretrend"]["p"], "slope_per_year": glp["pretrend"]["slope_per_year"],
                        "event_study_p": es.pretrend_p, "ref": es.ref}
        if ta is not None and glp and "trend_adjusted" in glp and glp["trend_adjusted"]["se"] > ta.se:
            g2 = glp["trend_adjusted"]
            ta.details["cr1_se"] = ta.se
            ta.se, ta.dof = g2["se"], g2["dof"]
            ta.ci, ta.p = t_ci(ta.effect, ta.se, ta.dof, ta.alpha)
            ta.inference = f"집단 수준 AR(1) 표준오차(φ={glp['phi']:.2f})"

    controls = [g for g in df[pd_.group].unique() if g != pd_.treated]
    fakes = {}
    for g in (pd_.placebo_groups if pd_.placebo_groups is not None else controls):
        sub = df[df[pd_.group] != pd_.treated]
        if (sub[pd_.group] == g).any() and sub[pd_.group].nunique() > 1:
            fakes[g] = _did(sub, pd_, g, weights=pd_.weights, full=full[full[pd_.group] != pd_.treated])
    ps = C.placebo_space(main, fakes)

    pl = None
    if pd_.placebo_window:
        wpre, wpost = pd_.placebo_window
        tix = pd.DatetimeIndex(all_times)
        i0 = int(tix.searchsorted(t0))

        def fit(i):
            if i - wpre < 0 or i + wpost > len(tix):
                return None
            lo, hi = tix[i - wpre], tix[i + wpost - 1]
            sub = full[(full[pd_.time] >= lo) & (full[pd_.time] <= hi)]
            if sub[pd_.time].nunique() < 0.8 * (wpre + wpost):
                return None
            sub = sub.assign(_tr=(sub[pd_.group] == pd_.treated).astype(int))
            try:
                return did(sub, pd_.y, pd_.unit, pd_.time, "_tr", tix[i], fe=pd_.fe, cluster=pd_.cluster,
                           weights=pd_.weights, alpha=pd_.alpha, bootstrap=False)
            except Exception:
                return None
        # 실제 분석과 같은 창 길이로 맞춘다
        real_pre = len(pre_times)
        pl = C.placebo_time(fit, len(tix), i0, real=main.effect, window=(wpre, wpost), step=1,
                            forbidden=[i0], dates=tix, alpha=pd_.alpha)
        pl.note = f"창: 사전 {wpre} · 사후 {wpost}시점 (실제 분석의 사전 시점 {real_pre}개)"

    specs = {}
    for name, spec in pd_.sens.items():
        sub = spec["filter"](df) if "filter" in spec else df
        specs[name] = _did(sub, pd_, pd_.treated, weights=spec.get("weights", pd_.weights), full=full)
    sens = C.sensitivity(main, specs) if specs else None

    dpost = d1.assign(_post=(d1[pd_.time] >= t0).astype(int))
    dc = C.double_compute_2x2(dpost, pd_.y, pd_.unit, "_tr", "_post")

    n_post_t = len([t for t in times if t >= t0])
    pw = assess(main.se, pl.sd if (pl is not None and pl.available) else None, pd_.reference_effect,
                n_post_t, "개월", n_pre=len([t for t in times if t < t0]))
    kw = dict(placebo_time=pl, placebo_space=ps, sens=sens, double=dc, power=pw, quality_cap=quality_cap,
              quality_reason=quality_reason, window_label=f"사후 {n_post_t}개월")
    v = judge(main, pretrend_p=pretrend.get("p"), trend_adjusted=ta, **kw)
    few = len([t for t in times if t < t0]) < 6          # v3: 사전 시점이 적으면 군집 결합검정(과대 기각)을 쓰지 않음
    v3 = judge(main, pretrend_p=None if few else pretrend.get("p"), trend_adjusted=None if few else ta, rules="v3", **kw)
    return Analysis("panel", pd_.question, pd_.to_dict(), main, v, pretrend, ta, pl, ps, sens, dc, pw, None,
                    event_study=es, seconds=round(_time.time() - t_start, 2), verdict_v3=v3)
