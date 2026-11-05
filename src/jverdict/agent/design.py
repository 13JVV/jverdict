"""분석 설계: 스키마, 검증, 사전 등록 문서, 실행.

설계는 JSON 하나로 표현한다. Design Agent(LLM 또는 템플릿)가 만들고, 잠근 뒤에는 Executor가 그대로 실행한다.
설계 JSON에는 효과 값이 들어갈 자리가 없다(결과를 보고 설계를 고칠 수 없게).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from jverdict.causal import PanelDesign, SeriesDesign, analyze_panel, analyze_series
from jverdict.quality.coverage import check_coverage, exclusion_mask

from . import catalog
from .library import OFFICIAL_STATS

_STR_OR_NULL = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_INT_OR_NULL = {"anyOf": [{"type": "integer"}, {"type": "null"}]}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props), "additionalProperties": False}


SERIES_SCHEMA = _obj({
    "name": {"type": "string", "description": "계열 이름(분석에서 부르는 이름)"},
    "groups": {"type": "array", "items": {"type": "string"}, "description": "데이터셋의 집단 값(이용자 유형·구분)"},
    "units_tier": {**_STR_OR_NULL, "description": "카드 업종 묶음 'tier:가능', 'tier:불가', 'tier:불가+불확실' 등. 없으면 null"},
    "units_list": {"type": "array", "items": {"type": "string"}, "description": "특정 단위(업종)만 쓸 때. 비우면 전체"},
})

PANEL_SCHEMA = _obj({
    "groups": {"type": "array", "items": {"type": "string"}},
    "treated": {"type": "string"},
    "months": {"type": "array", "items": {"type": "string"}, "description": "분석에 쓰는 월(YYYY-MM-01), 사전·사후 모두"},
    "base_year": {"type": "integer"},
    "sector_month_fe": {"type": "boolean", "description": "업종×월 고정효과(같은 업종끼리 비교)"},
    "ref": {"type": "string", "description": "이벤트스터디 기준 월"},
    "sens": {"type": "array", "items": _obj({
        "label": {"type": "string"}, "drop_months": {"type": "array", "items": {"type": "string"}},
        "drop_biz": {"type": "array", "items": {"type": "string"}}, "weighted": {"type": "boolean"}})},
})

DESIGN_SCHEMA = _obj({
    "dataset": {"type": "string", "enum": list(catalog.DATASETS)},
    "method": {"type": "string", "enum": ["its", "did_panel"],
               "description": "its = 대상/대조 합계 격차의 단절 시계열(계절 차분), did_panel = 단위 패널 이중차분"},
    "series": {"type": "array", "items": SERIES_SCHEMA},
    "target": {"type": "string"},
    "reference": {"type": "array", "items": {"type": "string"}},
    "placebo": {"type": "array", "items": _obj({"target": {"type": "string"},
                                                "reference": {"type": "array", "items": {"type": "string"}}})},
    "alt_references": {"type": "array", "items": _obj({"label": {"type": "string"},
                                                       "reference": {"type": "array", "items": {"type": "string"}}})},
    "pre": {**_INT_OR_NULL, "description": "사전 기간(일 또는 월). null이면 기본(일 728, 월 24)"},
    "post": {**_INT_OR_NULL, "description": "사후 기간(일 또는 월). null이면 기본(일 182, 월 6)"},
    "season_years": {"type": "integer", "description": "계절 차분에 쓸 과거 연수(1~3)"},
    "exclude": {"type": "array", "items": _obj({"start": {"type": "string"}, "end": {"type": "string"},
                                                "reason": {"type": "string"}}),
                "description": "시점 위약검정에서 가짜 시행일 창이 겹치면 안 되는 기간(다른 정책·큰 사건). 데이터 결측은 자동 처리되므로 넣지 않음"},
    "reference_effect": {"type": "number", "description": "정책적으로 의미 있는 최소 효과(비율, 0.10 = 10%)"},
    "expected": {"anyOf": [{"type": "string", "enum": ["increase", "decrease"]}, {"type": "null"}]},
    "official_stats": {"type": "array", "items": {"type": "string", "enum": list(OFFICIAL_STATS) or ["none"]}},
    "panel": {"anyOf": [PANEL_SCHEMA, {"type": "null"}]},
    "choices": {"type": "array", "items": _obj({"decision": {"type": "string"}, "reason": {"type": "string"}}),
                "description": "설계 선택과 이유(사전 등록 문서에 그대로 들어감)"},
})


# LLM이 새로 설계할 때 쓰는 스키마(집계 시계열 전용). 패널 설계는 사전 등록 사례에만 쓴다.
# 출력 문법이 너무 커지면 API가 거부하므로(compiled grammar too large) 필요한 항목만 둔다.
DESIGN_SCHEMA_LLM = _obj({k: v for k, v in DESIGN_SCHEMA["properties"].items() if k not in ("panel", "method")})


# 계절 처리. 'none'은 등록 설계에서만 쓴다(LLM 스키마에는 넣지 않음: 기존 Claude 기록의 재실행 지문을 지키기 위해).
#   sdiff: 과거 같은 달(같은 요일) 값의 중앙값을 뺀다. 과거 해가 정상일 때 맞다.
#   none : 계절 차분 없이 격차 그대로. 과거 같은 시기도 비정상(예: 코로나19 방역 기간)이라 기준값이 오염될 때 쓴다.
#          계절에서 오는 차이는 사전 기간을 1년 단위로 잡고, 정상 기간 가짜 시행일 분포로 보정한다.
SEASONS = {"sdiff": "계절 차분", "none": "계절 차분 없음"}


# --------------------------------------------------------------------------- 검증
def validate(d: dict, card: dict) -> list[str]:
    errs = []
    if d.get("dataset") not in catalog.DATASETS:
        return [f"dataset은 {list(catalog.DATASETS)} 중 하나여야 함"]
    ds = catalog.DATASETS[d["dataset"]]
    if d.get("method") == "did_panel":
        p = d.get("panel")
        if not p:
            return ["did_panel에는 panel 설정이 필요함"]
        if d["dataset"] != "card_sector":
            errs.append("did_panel은 card_sector 데이터셋에서만 지원")
        if p["treated"] not in p["groups"]:
            errs.append("panel.treated가 panel.groups에 없음")
        months = pd.to_datetime(p["months"])
        if not (months < pd.Timestamp(card["start_date"])).any() or not (months >= pd.Timestamp(card["start_date"])).any():
            errs.append("panel.months에 정책 전·후 월이 모두 있어야 함")
        return errs
    names = [s["name"] for s in d.get("series", [])]
    if len(set(names)) != len(names):
        errs.append("계열 이름이 중복됨")
    for k in [d.get("target")] + list(d.get("reference", [])):
        if k not in names:
            errs.append(f"'{k}'는 series에 정의되지 않음")
    for pl in d.get("placebo", []):
        for k in [pl["target"]] + pl["reference"]:
            if k not in names:
                errs.append(f"위약 계열 '{k}'가 series에 없음")
    for alt in d.get("alt_references", []):
        for k in alt["reference"]:
            if k not in names:
                errs.append(f"대체 대조 계열 '{k}'가 series에 없음")
    if not 1 <= int(d.get("season_years", 3)) <= 3:
        errs.append("season_years는 1~3")
    if d.get("season", "sdiff") not in SEASONS:
        errs.append(f"season은 {list(SEASONS)} 중 하나")
    if not errs:
        try:
            prof = catalog.profile(d["dataset"])
            known = {g["group"] for g in prof["groups"]}
            for s in d["series"]:
                bad = [g for g in s["groups"] if g not in known]
                if bad:
                    errs.append(f"계열 '{s['name']}'의 집단 {bad}가 데이터에 없음. 가능: {sorted(known)}")
            lo, hi = pd.Timestamp(prof["period"][0]), pd.Timestamp(prof["period"][1])
            if not lo < pd.Timestamp(card["start_date"]) <= hi:
                errs.append(f"정책 시행일 {card['start_date']}이 데이터 기간 {prof['period']} 밖")
        except Exception as e:     # noqa: BLE001  데이터 문제도 설계 오류로 돌려준다
            errs.append(f"데이터 확인 실패: {e}")
    for o in d.get("official_stats", []):
        if o not in OFFICIAL_STATS:
            errs.append(f"등록되지 않은 공식 통계: {o}")
    _ = ds
    return errs


def _series_map(d: dict) -> dict[str, dict]:
    out = {}
    for s in d["series"]:
        units = s.get("units_tier") or (s.get("units_list") or None)
        out[s["name"]] = {"groups": s["groups"], "units": units}
    return out


# --------------------------------------------------------------------------- 사전 등록 문서
def prereg_markdown(card: dict, d: dict, scout: dict, run_id: str) -> str:
    L = [f"# 사전 등록: {card['policy']}", "",
         "- 이 문서는 효과 값을 보기 전에 만들어졌다. Design 단계에 전달된 자료는 정책 전 기간 요약, 커버리지 점검, 측정 단절 진단의 판정 이름뿐이다.",
         "", "## 1. 질문", f"- {card['question']}", f"- 정책 시행일: {card['start_date']}",
         f"- 처치 대상: {card['treated']}", f"- 결과지표: {card['outcome']}",
         f"- 기대 방향: {d.get('expected') or '미지정'} · 기준 효과: {d['reference_effect'] * 100:.0f}%",
         "", "## 2. 데이터와 설계", f"- 데이터셋: {catalog.DATASETS[d['dataset']].title} (`{d['dataset']}`)",
         f"- 방법: {'단절 시계열(대상/대조 격차, 계절 차분)' if d['method'] == 'its' else '단위 패널 이중차분'}"]
    if d["method"] == "its":
        sm = _series_map(d)
        L += [f"- 계열 정의: " + "; ".join(f"{k} = {json.dumps(v, ensure_ascii=False)}" for k, v in sm.items()),
              f"- 대상: {d['target']} / 대조: {', '.join(d['reference'])}",
              f"- 사전 {d.get('pre') or '기본'} · 사후 {d.get('post') or '기본'} · "
              + (f"계절 차분 {d['season_years']}년" if d.get("season", "sdiff") == "sdiff" else "계절 차분 없음(격차 그대로)"),
              f"- 공간 위약: " + (", ".join(f"{p['target']} vs {'+'.join(p['reference'])}" for p in d["placebo"]) or "없음"),
              f"- 대체 대조(민감도): " + (", ".join(a["label"] for a in d["alt_references"]) or "기본 사양")]
    else:
        p = d["panel"]
        L += [f"- 집단: {', '.join(p['groups'])} (처치: {p['treated']})", f"- 분석 월: {', '.join(p['months'])}",
              f"- 결과 변수: log(금액 / {p['base_year']}년 같은 달 금액), 업종×월 고정효과 {'사용' if p['sector_month_fe'] else '미사용'}",
              f"- 이벤트스터디 기준 월: {p['ref']}", "- 민감도: " + ", ".join(s["label"] for s in p["sens"])]
    if d.get("exclude"):
        L.append("- 시점 위약 제외 기간: " + "; ".join(f"{e['start']}~{e['end']}({e['reason']})" for e in d["exclude"]))
    if d.get("official_stats"):
        L.append("- 대조할 공식 통계: " + ", ".join(OFFICIAL_STATS[o].label for o in d["official_stats"]))
    L += ["", "## 3. 데이터 점검 결과(Data Scout)", f"- 측정 단절 진단: {scout.get('quality_verdict', '-')}",
          f"- 커버리지: {scout.get('coverage_summary', '-')}", "",
          "## 4. 설계 선택과 이유"] + [f"- {c['decision']}: {c['reason']}" for c in d.get("choices", [])] + [
          "", "## 5. 판정", "- 판정 규칙 v2(`jverdict.causal.verdict`)를 그대로 적용한다. 결과를 본 뒤 설계를 바꾸면 변경 기록에 남기고 원안 판정을 함께 보고한다.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- 실행
def execute(card: dict, d: dict, *, seed: int = 20261104, placebo: bool = True):
    """잠긴 설계를 엔진으로 실행한다. 반환: (Analysis, 부가정보)."""
    if d["method"] == "did_panel":
        p = d["panel"]
        df = catalog.card_panel(p["groups"], p["months"], p["base_year"])
        sens = {}
        for s in p["sens"]:
            spec = {}
            dm = [pd.Timestamp(x) for x in s["drop_months"]]
            db = list(s["drop_biz"])
            if dm or db:
                spec["filter"] = (lambda dm, db: lambda x: x[~x.time.isin(dm) & ~x.biz.isin(db)])(dm, db)
            if s["weighted"]:
                spec["weights"] = "w_base"
            sens[s["label"]] = spec
        pdz = PanelDesign(card["question"], "y", "unit", "time", "group", p["treated"], card["start_date"], ref=p["ref"],
                          fe=["unit", "sm"] if p["sector_month_fe"] else None, reference_effect=d["reference_effect"],
                          sens=sens)
        return analyze_panel(df, pdz), {"n_rows": int(len(df))}

    sm = _series_map(d)
    long, wide = catalog.series_frame(d["dataset"], sm, seed)
    freq = catalog.DATASETS[d["dataset"]].freq
    # 커버리지는 분석 계열이 아니라 데이터셋 전체 집단으로 점검한다(부분 집계일은 모든 집단이 함께 줄어든 날)
    cov = coverage(d["dataset"], seed)
    mask = exclusion_mask(cov, wide.index)
    ext = [OFFICIAL_STATS[o] for o in d.get("official_stats", [])]
    ds = SeriesDesign(card["question"], d["target"], d["reference"], card["start_date"], freq,
                      expected=d.get("expected"), reference_effect=d["reference_effect"], pre=d.get("pre"),
                      post=d.get("post"), season=d.get("season", "sdiff"), season_years=int(d["season_years"]),
                      placebo_groups=[(p["target"], p["reference"]) for p in d.get("placebo", [])],
                      alt_references={a["label"]: a["reference"] for a in d.get("alt_references", [])},
                      exclude=[(e["start"], e["end"]) for e in d.get("exclude", [])],
                      external=ext or None, needed_hint=card.get("needed_hint") or None)
    a = analyze_series(wide, ds, mask=mask, placebo=placebo)
    return a, {"coverage": cov.to_dict(), "n_periods": int(len(wide))}


_COV: dict = {}


def coverage(dataset_id: str, seed: int = 20261104):
    key = (dataset_id, seed)
    if key not in _COV:
        full = catalog.load(dataset_id, seed)
        _COV[key] = check_coverage(full, "time", "group", "value", unit="unit", freq=catalog.DATASETS[dataset_id].freq)
    return _COV[key]


def coverage_summary(cov: dict) -> str:
    """커버리지 점검 요약. 메시지를 그대로 쓴다(부분 집계는 '흩어진 날짜 수'로 표현됨)."""
    f = cov.get("findings", [])
    if not f:
        return "이상 없음"
    parts = [f"[{x['kind']}] {x['message']}" for x in f[:6]]
    return " / ".join(parts) + (f" 외 {len(f) - 6}건" if len(f) > 6 else "") + " (결측·부분 집계일은 분석에서 자동 제외)"


def numbers_of(a) -> dict:
    """판정에 쓰인 핵심 수치(재현성 대조와 서술 숫자 검사에 쓴다)."""
    v = a.verdict
    out = {"verdict": v.label}
    if a.main is not None:
        out.update(effect=round(a.main.effect, 6), se=round(a.main.se, 6), ci_lo=round(v.ci_used[0], 6),
                   ci_hi=round(v.ci_used[1], 6), p=round(float(v.p_used), 6))
    if a.power is not None:
        out["mde_pct"] = round(a.power.mde_pct, 6)
    return out


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)
