"""Agent 툴 레지스트리.

각 툴은 입력·출력 스키마가 정해진 파이썬 함수다. Claude 툴콜링과 템플릿 모드가 같은 함수를 쓴다.
- 툴마다 쓸 수 있는 단계(stage)가 정해져 있다. 다른 단계에서 부르면 오류를 돌려준다.
- 설계 전 단계 툴은 '효과 가림'을 지킨다: 정책 시행일 이후 결과 값은 돌려주지 않는다.
- 효과 추정이 들어가는 툴(extra_check)은 사전 등록 잠금 뒤에만 쓸 수 있다.
- 출력은 집계 요약 JSON뿐이다. 원자료 행은 돌려주지 않는다.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from jverdict.quality.breaks import diagnose

from . import catalog
from . import design as DZ
from .library import EVENTS, OFFICIAL_STATS, events_between
from .prereg import GuardError

_S = {"type": "string"}
_SL = {"type": "array", "items": {"type": "string"}}
_NS = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_NI = {"anyOf": [{"type": "integer"}, {"type": "null"}]}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable
    stages: tuple[str, ...]
    needs_lock: bool = False

    strict: bool = True

    def spec(self, strict: bool | None = None) -> dict:
        st = self.strict if strict is None else strict
        return {"name": self.name, "description": self.description, "input_schema": self.schema, "strict": st}


def _round(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


# --------------------------------------------------------------------------- 툴 구현
def t_catalog_search(run, query: str) -> dict:
    q = (query or "").strip()
    out = []
    for d in catalog.DATASETS.values():
        text = " ".join([d.id, d.title, d.group_label, d.unit_label] + d.notes)
        if not q or any(w in text for w in q.split()):
            out.append(d.card())
    return {"datasets": out or [d.card() for d in catalog.DATASETS.values()]}


def t_dataset_profile(run, dataset: str) -> dict:
    # 설계 전 단계에서는 정책 시행일 이전만 보여 준다
    return catalog.profile(dataset, before=run.card["start_date"], seed=run.seed)


def t_official_stats(run) -> dict:
    if run.card.get("library_case") == "C" or "합성" in run.card.get("source", ""):
        return {"official_stats": [], "note": "합성데이터에는 대조할 공식 통계가 없음"}
    return {"official_stats": [{"id": k, "label": v.label, "period": [v.start, v.end], "groups": v.groups,
                                "same_definition": v.same_definition, "source": v.source}
                               for k, v in OFFICIAL_STATS.items()]}


def _series_arg(series: list[dict]) -> dict:
    return {s["name"]: {"groups": s["groups"], "units": s.get("units_tier") or (s.get("units_list") or None)}
            for s in series}


def t_quality_check(run, dataset: str, series: list[dict], target: str, reference: list[str],
                    official_stats: list[str]) -> dict:
    """측정 단절 진단과 커버리지 점검. 효과 가림: 단절이 의심될 때만 변화 크기를 알려 준다."""
    _, wide = catalog.series_frame(dataset, _series_arg(series), run.seed)
    freq = catalog.DATASETS[dataset].freq
    cov = DZ.coverage(dataset, run.seed)
    from jverdict.quality.coverage import exclusion_mask
    mask = exclusion_mask(cov, wide.index)
    ext = [OFFICIAL_STATS[o] for o in official_stats if o in OFFICIAL_STATS]
    dg = diagnose(wide, target, reference, run.card["start_date"], freq, expected=run.card.get("expected"),
                  external=ext or None, mask=mask, needed_hint=run.card.get("needed_hint") or None)
    run.scout_diag = dg
    out = {"verdict": dg.verdict, "judgement_cap": dg.judgement_cap,
           "evidence": [{"code": e.code, "name": e.name, "strength": e.strength, "met": e.met} for e in dg.evidence],
           "reclassification": [{"kind": r["kind"], "group": r["group"]} for r in dg.reclass],
           "needed_data": dg.needed_data, "coverage": DZ.coverage_summary(cov.to_dict())}
    if dg.suspected:
        out["message"] = dg.message             # 단절이 의심되면 크기까지 공개(어차피 효과 판정은 보류)
        out["external"] = [x.message for x in dg.external]
    else:
        out["message"] = "효과 가림: 정책 시점 변화의 크기는 설계를 잠근 뒤에 공개됩니다."
    return out


def t_pick_controls(run, dataset: str, series: list[dict], target: str, candidates: list[str]) -> dict:
    """대조 후보의 정책 전 유사도. 정책 전 2년의 계절 차분 로그 수준으로 상관과 추세 차이를 잰다."""
    _, wide = catalog.series_frame(dataset, _series_arg(series), run.seed)
    t0 = pd.Timestamp(run.card["start_date"])
    freq = catalog.DATASETS[dataset].freq
    per = 364 if freq == "D" else 12
    win = 728 if freq == "D" else 24
    pre = wide[wide.index < t0].iloc[-(win + per):]
    lg = np.log1p(pre)
    sd = (lg - lg.shift(per)).iloc[per:]
    out = []
    for c in candidates:
        if c not in sd or c == target:
            continue
        g = (sd[target] - sd[c]).dropna()
        x = np.arange(len(g)) / (365.25 if freq == "D" else 12)
        slope = float(np.polyfit(x, g.to_numpy(), 1)[0]) if len(g) > 3 else float("nan")
        out.append({"candidate": c, "corr_pre": float(sd[[target, c]].dropna().corr().iloc[0, 1]),
                    "gap_sd_pre": float(g.std()), "gap_slope_per_year": slope,
                    "mean_level_ratio": float(pre[c].mean() / max(pre[target].mean(), 1e-9))})
    out.sort(key=lambda r: (-(r["corr_pre"] if np.isfinite(r["corr_pre"]) else -9), r["gap_sd_pre"]))
    return {"target": target, "window": f"정책 전 {win}{'일' if freq == 'D' else '개월'}", "candidates": out,
            "note": "효과 가림: 정책 시행일 이전 자료만 사용"}


def t_event_calendar(run, start: str, end: str) -> dict:
    ds = (run.design or run.scout or {}).get("dataset")
    return {"dataset": ds, "events": [{k: v for k, v in ev.items() if k not in ("added", "in_estimand")}
                                      for ev in events_between(start, end, ds, as_of=run.as_of)],
            "note": "데이터 결측·부분 집계는 품질 점검이 자동으로 제외하므로 설계의 exclude에 넣지 않는다"}


def t_extra_check(run, kind: str, reference: list[str], placebo_target: str | None, post: int | None,
                  start: str | None, end: str | None, reason: str) -> dict:
    """Skeptic의 반론 검증. 잠긴 설계를 복사해 한 가지만 바꾸고 다시 추정한다(판정 규칙은 그대로).

    결과 판정은 코드가 한다: 주분석과 같은 방향·유의하고 크기가 절반 이상 유지되면 '반론 기각', 아니면 '반론 성립'.
    """
    if run.analysis is None or run.analysis.main is None:
        return {"error": "주분석 결과가 없어 반론 검증을 할 수 없음"}
    d = copy.deepcopy(run.design)
    if d["method"] == "did_panel":
        return _extra_check_panel(run, d, kind, reference, placebo_target, post, start, end, reason)
    if d["method"] != "its":
        return {"error": f"반론 검증 툴이 지원하지 않는 설계: {d['method']}"}
    names = {s["name"] for s in d["series"]}
    label = kind
    if kind == "alt_reference":
        bad = [r for r in reference if r not in names]
        if bad or not reference:
            return {"error": f"series에 없는 대조 계열: {bad}. 가능: {sorted(names)}"}
        d["reference"] = reference
        label = f"대조 집단을 {'+'.join(reference)}로"
    elif kind == "alt_window":
        if not post or post < 1:
            return {"error": "post(사후 기간)가 필요"}
        d["post"] = int(post)
        label = f"사후 기간 {post}"
    elif kind == "exclude_period":
        if not start or not end:
            return {"error": "start, end가 필요"}
        d["exclude"] = d.get("exclude", []) + [{"start": start, "end": end, "reason": reason}]
        label = f"{start}~{end} 제외"
    elif kind == "placebo_group":
        if placebo_target not in names:
            return {"error": f"series에 없는 계열: {placebo_target}"}
        d["target"] = placebo_target
        d["reference"] = reference or d["reference"]
        label = f"가짜 처치 {placebo_target}"
    else:
        return {"error": f"알 수 없는 kind: {kind}"}
    try:
        if kind == "exclude_period":
            a = _excluded_estimate(run, d, start, end)
        else:
            a = DZ.execute(run.card, d, seed=run.seed, placebo=False)[0].main
    except Exception as e:  # noqa: BLE001
        return {"error": f"추정 실패: {e}"}
    if a is None:
        return {"error": "관측이 부족해 추정할 수 없음"}
    return _record(run, kind, label, reason, a, note=_estimand_note(run, kind, start, end))


def _estimand_note(run, kind, start, end) -> str:
    """사전 등록에서 '추정 대상에 포함'한 사건을 빼는 반론은 질문 자체를 바꾸므로 판정에 쓰지 않는다."""
    if kind != "exclude_period" or not start or not end:
        return ""
    case = run.card.get("library_case")
    for ev in EVENTS:
        if case and case in ev.get("in_estimand", ()) and ev["start"] <= end and ev["end"] >= start:
            return f"'{ev['name']}'은(는) 사전 등록에서 추정 대상에 포함한 사건이라 판정에 쓰지 않음(참고 계산)"
    return ""


def _record(run, kind, label, reason, a, note: str = "") -> dict:
    """반론 결과 판정(코드).

    주분석이 유의할 때: 다시 추정한 점추정이 절반 넘게 줄거나 방향이 바뀌면 '반론 성립'.
    크기는 유지됐는데 관측이 줄어 유의성만 사라진 경우는 반론이 맞았다는 증거가 아니므로 '판단 불가(검정력 부족)'.
    (2026-11-09 보완: 이전에는 이 경우도 '반론 성립'으로 적었다. 규칙 이력 v2.2)
    판정 규칙 결과가 '조건부'처럼 방향을 말했으면 주분석이 유의하지 않아도 크기 기준으로 본다(v2.3).
    """
    main = run.analysis.main
    status = objection_status(kind, main, a, claimed=run.analysis.verdict.level in ("효과 있음", "조건부"))
    if note:
        status = "해당 없음"
    rec = {"id": f"X{len(run.extra_checks) + 1}", "kind": kind, "label": label, "reason": reason,
           "effect": a.effect, "pct": a.pct, "ci": list(a.ci), "p": a.p, "significant": a.significant,
           "main_effect": main.effect, "status": status}
    if note:
        rec["note"] = note
    elif status == "판단 불가":
        rec["note"] = "효과 크기는 유지됐지만 기간을 빼 관측이 줄어 유의하지 않음(검정력 부족). 반론의 증거로 보지 않음"
    run.extra_checks.append(rec)
    return rec


def _extra_check_panel(run, d, kind, reference, placebo_target, post, start, end, reason) -> dict:
    """패널(이중차분) 설계의 반론 검증: 비교 국적, 사후 개월, 제외 기간, 가짜 처치 국적 가운데 하나만 바꾼다."""
    p = d["panel"]
    groups, treated = list(p["groups"]), p["treated"]
    others = [g for g in groups if g != treated]
    t0 = pd.Timestamp(run.card["start_date"])
    months = [pd.Timestamp(m) for m in p["months"]]
    if kind == "alt_reference":
        bad = [r for r in reference if r not in others]
        if bad or not reference:
            return {"error": f"비교 집단으로 쓸 수 없는 값: {bad}. 가능: {others}"}
        p["groups"] = [treated] + list(reference)
        label = f"비교 국적을 {'+'.join(reference)}로"
    elif kind == "alt_window":
        if not post or post < 1:
            return {"error": "post(사후 개월 수)가 필요"}
        post_m = [m for m in months if m >= t0][: int(post)]
        p["months"] = [m.strftime("%Y-%m-%d") for m in months if m < t0] + [m.strftime("%Y-%m-%d") for m in post_m]
        label = f"사후 {len(post_m)}개월"
    elif kind == "exclude_period":
        if not start or not end:
            return {"error": "start, end가 필요"}
        keep = [m for m in months if not (pd.Timestamp(start) <= m <= pd.Timestamp(end))]
        if sum(m < t0 for m in keep) < 2 or not any(m >= t0 for m in keep):
            return {"error": "그 기간을 빼면 정책 전·후 관측이 부족함"}
        p["months"] = [m.strftime("%Y-%m-%d") for m in keep]
        if p.get("ref") and pd.Timestamp(p["ref"]) not in keep:
            p["ref"] = max(m for m in keep if m < t0).strftime("%Y-%m-%d")
        label = f"{start}~{end} 제외"
    elif kind == "placebo_group":
        if placebo_target not in others:
            return {"error": f"가짜 처치로 쓸 수 없는 값: {placebo_target}. 가능: {others}"}
        p["groups"] = [g for g in others]
        p["treated"] = placebo_target
        label = f"가짜 처치 {placebo_target}(실제 처치 {treated} 제외)"
    else:
        return {"error": f"알 수 없는 kind: {kind}"}
    try:
        a = DZ.execute(run.card, d, seed=run.seed, placebo=False)[0].main
    except Exception as e:  # noqa: BLE001
        return {"error": f"추정 실패: {e}"}
    if a is None:
        return {"error": "관측이 부족해 추정할 수 없음"}
    return _record(run, kind, label, reason, a)


def _excluded_estimate(run, d, start, end):
    """사후 창에서 특정 기간(다른 사건)을 뺀 추정: 해당 기간을 결측으로 두고 다시 계산."""
    sm = DZ._series_map(d)
    _, wide = catalog.series_frame(d["dataset"], sm, run.seed)
    freq = catalog.DATASETS[d["dataset"]].freq
    from jverdict.quality.coverage import exclusion_mask
    mask = exclusion_mask(DZ.coverage(d["dataset"], run.seed), wide.index)
    return exclusion_refit(wide, d["target"], d["reference"], freq, mask, run.card["start_date"], start, end,
                           season=d.get("season", "sdiff"), season_years=int(d["season_years"]), pre=d.get("pre"),
                           post=d.get("post"))


def exclusion_refit(wide, target, reference, freq: str, mask, policy_date: str, start: str, end: str, *,
                    season: str = "sdiff", season_years: int = 3, pre: int | None = None, post: int | None = None):
    """기간 제외 재추정의 계산부(반론 검증과 반론 성능 시험이 함께 쓴다). 계절 처리는 잠긴 설계 그대로."""
    from jverdict.causal.estimate import DEFAULTS, gap_series, its_fit
    mask = mask | ((wide.index >= pd.Timestamp(start)) & (wide.index <= pd.Timestamp(end)))
    gs = gap_series(wide, target, reference, freq, season=season, season_years=season_years, mask=mask)
    i0 = gs.pos(policy_date)
    fin = np.flatnonzero(np.isfinite(gs.g_adj[:i0]))
    pre = min(pre or DEFAULTS[freq]["pre"], int(i0 - fin[0]) if len(fin) else 0)
    return its_fit(gs, i0, pre=pre, post=post)


def objection_status(kind: str, main, a, claimed: bool = False) -> str:
    """반론 결과 판정 규칙(코드). main·a는 effect·significant를 가진 추정치.

    claimed: 판정 규칙 결과가 방향을 말했는가('효과 있음'·'조건부'). 그렇다면 주분석이 유의하지 않은 약한 신호(조건부)여도
    '크기가 절반 넘게 줄거나 방향이 바뀌면 성립'으로 본다(v2.3, 반론 단계 성능 시험 결과를 본 뒤 채택).
    """
    same = np.sign(a.effect) == np.sign(main.effect)
    kept = same and abs(a.effect) >= 0.5 * abs(main.effect)
    if kind == "placebo_group":
        return "반론 성립" if (a.significant and kept) else "반론 기각"
    if not main.significant and claimed:
        return "반론 기각" if kept else "반론 성립"
    if not main.significant:
        return "반론 성립" if a.significant else "해당 없음"
    if not kept:
        return "반론 성립"
    return "반론 기각" if a.significant else "판단 불가"


SERIES_ITEM = DZ.SERIES_SCHEMA

TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("catalog_search", "적재된 데이터셋 목록과 설명(기간, 집단, 단위)을 찾는다. query가 빈 문자열이면 전체.",
         _obj({"query": _S}), t_catalog_search, ("scout", "design")),
    Tool("dataset_profile", "데이터셋의 집단별 기간·관측 수·평균 규모를 정책 시행일 이전 기간만으로 요약한다(효과 가림).",
         _obj({"dataset": {"type": "string", "enum": list(catalog.DATASETS)}}), t_dataset_profile, ("scout", "design")),
    Tool("official_stats", "외부 통계 대조에 쓸 수 있는 등록된 공식 발표 수치 목록.", _obj({}), t_official_stats, ("scout", "design")),
    Tool("quality_check", "결과지표의 측정 단절(집계 방식 변경)과 커버리지를 점검한다. 반드시 한 번 실행한다. "
         "단절이 의심되지 않으면 변화 크기는 가려진다.",
         _obj({"dataset": {"type": "string", "enum": list(catalog.DATASETS)}, "series": {"type": "array", "items": SERIES_ITEM},
               "target": _S, "reference": _SL, "official_stats": _SL}), t_quality_check, ("scout",)),
    Tool("pick_controls", "대조 집단 후보들이 정책 전 기간에 대상 집단과 얼마나 비슷하게 움직였는지(상관, 격차 변동, 격차 추세) 잰다.",
         _obj({"dataset": {"type": "string", "enum": list(catalog.DATASETS)}, "series": {"type": "array", "items": SERIES_ITEM},
               "target": _S, "candidates": _SL}), t_pick_controls, ("design",)),
    Tool("event_calendar", "기간 안의 알려진 사건(명절, 방학, 코로나19, 다른 지원금, 데이터 결측 등)과 출처.",
         _obj({"start": _S, "end": _S}), t_event_calendar, ("design", "skeptic")),
    Tool("extra_check", "반론 검증: 잠긴 설계에서 한 가지만 바꿔 다시 추정한다(시계열·패널 모두). kind = alt_reference(reference에 다른 대조 계열·국적), "
         "alt_window(post에 사후 기간), exclude_period(start~end를 빼고), placebo_group(placebo_target을 가짜 처치로). "
         "쓰지 않는 인자는 null 또는 빈 배열. 결과의 status(반론 기각/성립)는 코드가 정한다.",
         _obj({"kind": {"type": "string", "enum": ["alt_reference", "alt_window", "exclude_period", "placebo_group"]},
               "reference": _SL, "placebo_target": _NS, "post": _NI, "start": _NS, "end": _NS, "reason": _S}),
         t_extra_check, ("skeptic",), needs_lock=True),
]}


def specs(stage: str, strict: bool = True) -> list[dict]:
    """strict=False면 인자 문법 보장을 끈다(출력 스키마가 큰 단계에서 문법 크기 제한을 피하려고). 인자는 함수가 검증한다."""
    return [t.spec(strict) for t in TOOLS.values() if stage in t.stages]


def call(run, stage: str, name: str, args: dict) -> tuple[str, bool]:
    """툴 실행. (JSON 문자열, 오류 여부). 단계·잠금 가드를 여기서 강제한다."""
    t = TOOLS.get(name)
    if t is None:
        return json.dumps({"error": f"없는 툴: {name}"}, ensure_ascii=False), True
    if stage not in t.stages:
        return json.dumps({"error": f"{name}은(는) {stage} 단계에서 쓸 수 없음"}, ensure_ascii=False), True
    if t.needs_lock and not run.locked:
        raise GuardError(f"{name}: 사전 등록 잠금 전에는 효과 추정 툴을 쓸 수 없음")
    try:
        out = t.fn(run, **args)
        err = isinstance(out, dict) and "error" in out
    except GuardError:
        raise
    except Exception as e:  # noqa: BLE001  모델이 고칠 수 있게 오류를 돌려준다
        from .llm import sanitize
        out, err = {"error": sanitize(f"{type(e).__name__}: {e}")}, True
    run.log(stage, f"툴 {name}", {"args": args, "error": err})
    return json.dumps(_round(out), ensure_ascii=False, default=str), err
