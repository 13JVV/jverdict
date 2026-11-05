"""6개 역할: 각 역할은 LLM 판(claude)과 템플릿 판(LLM 없음)을 함께 가진다.

LLM이 판단하는 것: 정책 카드 추출, 지표·대조 집단·설계 선택, 대안 가설과 검증 방법 선택, 판정서 서술.
코드가 강제하는 것: 스키마 검증, 품질 점검 필수 실행, 효과 가림, 잠금, 판정 규칙, 반론 결과 판정, 숫자 대조.
"""
from __future__ import annotations

import copy
import json
import re

import numpy as np
import pandas as pd

from jverdict.causal.verdict import LIGHT

from . import catalog
from . import design as DZ
from . import tools as T
from .library import CARDS, OFFICIAL_STATS, events_between, match_card

_S, _SL = {"type": "string"}, {"type": "array", "items": {"type": "string"}}
_NS = {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


COMMON = ("당신은 제주도 정책 효과 판정 Agent '정책 판정관'의 한 역할입니다. 한국어로 답합니다. "
          "숫자는 툴 결과에 있는 것만 씁니다. 모르는 사실(시행일, 대상)은 지어내지 말고 assumptions/notes에 '확인 필요'로 적습니다. "
          "원인을 단정하지 않습니다. 데이터 품질 문제는 '의심'으로 씁니다.")

# =========================================================================== ① Policy Parser
CARD_SCHEMA = _obj({
    "policy": _S, "question": _S, "start_date": {"type": "string", "description": "YYYY-MM-DD"},
    "end_date": _NS, "treated": _S, "outcome": _S,
    "expected": {"anyOf": [{"type": "string", "enum": ["increase", "decrease"]}, {"type": "null"}]},
    "region": _S, "source": _S, "reference_effect": {"type": "number", "description": "의미 있는 최소 효과 비율(기본 0.10)"},
    "assumptions": _SL,
})


def _lib_card(key: str) -> dict:
    c = copy.deepcopy(CARDS[key]["card"])
    c["library_case"] = key
    c.setdefault("assumptions", [])
    return c


def parser_template(run, question: str) -> dict:
    key = match_card(question)
    if key is None:
        raise ValueError("등록된 정책 카드와 맞는 질문이 아님. 템플릿 모드에서는 정책 카드를 직접 입력해야 함")
    return _lib_card(key)


# Policy Parser 프롬프트에 참고로 보여 주는 등록 사례. 목록을 바꾸면 기존 Claude 기록의 재실행 지문이 바뀌므로 고정한다.
# 이후에 추가한 등록 사례(E 등)는 코드가 핵심어로 찾아 연결하고, 시행일·대상을 등록본으로 맞춘다(LLM이 보지 못했으므로).
PARSER_REFERENCE_CASES = ("A", "B", "D", "C")


def parser_llm(run, question: str) -> dict:
    lib = [{"case": k, **{f: CARDS[k]["card"][f] for f in ("policy", "start_date", "treated", "outcome", "source")}}
           for k in PARSER_REFERENCE_CASES]
    sys = (COMMON + " 역할: ① Policy Parser. 공무원의 질문을 정책 카드로 바꿉니다. 등록된 사례와 같은 정책이면 그 사례의 시행일·대상을 따릅니다. "
           "질문에 기간 단위가 월이면 시행일을 그 달 1일로 둡니다.")
    user = (f"질문: {question}\n\n등록된 정책 사례(참고):\n{json.dumps(lib, ensure_ascii=False, indent=1)}\n\n"
            "분석 가능한 데이터셋:\n" + json.dumps([d.card() for d in catalog.DATASETS.values()], ensure_ascii=False))
    card = run.llm.json_call("parser", sys, user, CARD_SCHEMA)
    key = match_card(question) or match_card(card.get("policy", "") + card.get("question", ""))
    if key:     # 등록 사례면 사전 등록·필요 데이터 정보를 붙인다(사람 확인 화면에서 바뀐 값은 사람이 정함)
        lib_c = CARDS[key]["card"]
        card["library_case"] = key
        card["prereg"] = lib_c.get("prereg")
        card["needed_hint"] = lib_c.get("needed_hint", [])
        if key not in PARSER_REFERENCE_CASES:
            for f in ("policy", "start_date", "end_date", "treated", "outcome", "source", "reference_effect"):
                card[f] = lib_c.get(f)
            card.setdefault("assumptions", []).append(f"등록 사례 {key}와 같은 정책으로 보고 시행일·대상을 등록본으로 맞춤")
    else:
        card.update(library_case=None, prereg=None, needed_hint=[])
    return card


# =========================================================================== ② Data Scout
SCOUT_SCHEMA = _obj({
    "dataset": {"type": "string", "enum": list(catalog.DATASETS)},
    "series": {"type": "array", "items": DZ.SERIES_SCHEMA},
    "target": _S, "reference_candidates": _SL, "official_stats": _SL, "notes": _SL,
})


def _mandatory_quality(run, scout: dict) -> dict:
    """품질 점검은 LLM이 불렀든 아니든 코드가 반드시 실행한다(가드레일)."""
    if run.scout_diag is None and scout.get("method") != "did_panel":
        T.t_quality_check(run, scout["dataset"], scout["series"], scout["target"],
                          scout["reference_candidates"][:1], scout.get("official_stats", []))
    dg = run.scout_diag
    scout["quality_verdict"] = dg.verdict if dg else "패널 설계: 시점별 결측 점검만 수행"
    scout["quality_cap"] = dg.judgement_cap if dg else None
    scout["coverage_summary"] = DZ.coverage_summary(DZ.coverage(scout["dataset"], run.seed).to_dict())
    return scout


def manual_series(m: dict) -> tuple[list[dict], str, list[str]]:
    """직접 만든 정책 카드의 계열 정의 → (series, 대상 이름, 대조 후보 이름들)."""
    def one(x):
        return {"name": x["name"], "groups": list(x["groups"]), "units_tier": None, "units_list": list(x.get("units") or [])}
    series = [one(m["target"])] + [one(r) for r in m["references"]]
    return series, m["target"]["name"], [r["name"] for r in m["references"]]


def scout_template(run) -> dict:
    key = run.card.get("library_case")
    if key is None and run.card.get("manual"):
        m = run.card["manual"]
        series, target, cands = manual_series(m)
        scout = {"dataset": m["dataset"], "series": series, "target": target, "reference_candidates": cands,
                 "official_stats": [], "method": "its",
                 "notes": ["사용자가 직접 만든 정책 카드의 계열 사용",
                           "대조 후보는 설계 단계에서 정책 전 유사도로 고름" if m.get("auto") else "대조 집단은 사용자가 정함"]}
        return _mandatory_quality(run, scout)
    if key is None:
        raise ValueError("템플릿 모드: 등록 사례가 아니면 정책 카드를 직접 만들어야 함(화면의 '정책 카드 직접 만들기')")
    d = CARDS[key]["design"]
    scout = {"dataset": d["dataset"], "series": copy.deepcopy(d["series"]), "target": d["target"],
             "reference_candidates": list(d["reference"]) + [p["target"] for p in d["placebo"]],
             "official_stats": list(d["official_stats"]), "notes": ["등록 사례의 기본 지표 사용"],
             "method": d["method"]}
    return _mandatory_quality(run, scout)


def scout_llm(run) -> dict:
    sys = (COMMON + " 역할: ② Data Scout. 결과지표를 확정하고 데이터를 점검합니다. catalog_search와 dataset_profile로 데이터를 고르고, "
           "분석할 계열(대상·대조 후보·정책 대상이 아닌 집단)을 series로 정의한 뒤, quality_check를 반드시 한 번 실행합니다. "
           "공식 발표 수치가 있으면 official_stats로 확인해 quality_check에 넣습니다. 대조 후보는 정책 대상이 아니고 규모가 큰 집단을 우선합니다.")
    user = f"정책 카드:\n{json.dumps(run.card, ensure_ascii=False, indent=1)}"
    out = run.llm.agent_call("scout", sys, user, SCOUT_SCHEMA, T.specs("scout"),
                             lambda n, a: T.call(run, "scout", n, a))
    out["method"] = "its"
    if run.card.get("library_case") and CARDS[run.card["library_case"]]["design"]["method"] == "did_panel":
        out["method"] = "did_panel"
    return _mandatory_quality(run, out)


# =========================================================================== ③ Design Agent
def _rank_controls(run, refs: list[str]) -> tuple[list[str], list[dict]]:
    """직접 만든 카드에서 대조 후보를 정책 전 유사도로 정렬(효과 가림: 시행일 이전 자료만). 실패하면 원래 순서."""
    s = run.scout
    out, err = T.call(run, "design", "pick_controls", {"dataset": s["dataset"], "series": s["series"],
                                                       "target": s["target"], "candidates": refs})
    if err:
        return refs, []
    ranked = json.loads(out)["candidates"]
    order = [r["candidate"] for r in ranked] + [r for r in refs if r not in {x["candidate"] for x in ranked}]
    return order, ranked


def _generic_design(run) -> dict:
    s = run.scout
    refs = s["reference_candidates"]
    m = run.card.get("manual") or {}
    choices = [{"decision": f"대조 집단: {refs[0]}", "reason": "Data Scout의 첫 번째 후보"}]
    if m and m.get("auto") and len(refs) > 1:
        refs, ranked = _rank_controls(run, refs)
        if ranked:
            b = ranked[0]
            choices = [{"decision": f"대조 집단: {refs[0]}",
                        "reason": f"후보 {len(ranked)}개 중 정책 전 움직임이 가장 비슷함(계절 차분 상관 {b['corr_pre']:.2f}, "
                                  f"격차 변동 {b['gap_sd_pre']:.3f}). 효과 값은 보지 않고 시행일 이전 자료로만 고름"}]
            if len(refs) > 1:
                choices.append({"decision": f"공간 위약: {', '.join(refs[1:3])}",
                                "reason": "다음으로 비슷한 후보. 정책 대상이 아니므로 같은 때 변하면 대조 쪽 사건을 의심"})
    elif m:
        choices = [{"decision": f"대조 집단: {refs[0]}", "reason": "사용자가 정한 대조 집단"}]
        if len(refs) > 1:
            choices.append({"decision": f"공간 위약: {', '.join(refs[1:3])}", "reason": "사용자가 정한 나머지 대조 후보"})
    return {"dataset": s["dataset"], "method": "its", "series": s["series"], "target": s["target"],
            "reference": refs[:1], "placebo": [{"target": r, "reference": refs[:1]} for r in refs[1:3]],
            "alt_references": [], "pre": None, "post": None, "season_years": 3, "exclude": [],
            "reference_effect": float(run.card.get("reference_effect") or 0.10), "expected": run.card.get("expected"),
            "official_stats": s.get("official_stats", []), "panel": None, "choices": choices}


def design_template(run) -> dict:
    key = run.card.get("library_case")
    d = copy.deepcopy(CARDS[key]["design"]) if key else _generic_design(run)
    d["expected"] = run.card.get("expected")
    d["reference_effect"] = float(run.card.get("reference_effect") or d["reference_effect"])
    return d


def design_llm(run) -> dict:
    if run.card.get("prereg"):          # 기존 사전 등록 사례는 등록 설계를 그대로 쓴다(설계 재선택 금지)
        return design_template(run)
    sys = (COMMON + " 역할: ③ Design Agent. 효과 값을 보지 않은 상태에서 비교 설계를 정합니다. "
           "pick_controls로 정책 전 기간에 대상과 비슷하게 움직인 대조 집단을 고르고, event_calendar로 정책 전후 사건을 확인합니다. "
           "method는 집계 시계열이면 its. 대조 1개, 공간 위약 1~2개(정책 대상이 아닌 집단), 필요하면 대체 대조를 둡니다. "
           "pre/post는 특별한 이유가 없으면 null(기본값). exclude에는 시점 위약에서 피해야 할 다른 정책·큰 사회적 사건만 넣고, "
           "데이터 결측·부분 집계는 넣지 않습니다(자동 처리). "
           "choices에는 선택과 이유를 공무원이 읽을 수 있게 씁니다.")
    user = (f"정책 카드:\n{json.dumps(run.card, ensure_ascii=False, indent=1)}\n\nData Scout 결과:\n"
            f"{json.dumps({k: v for k, v in run.scout.items()}, ensure_ascii=False, indent=1, default=str)}")
    for attempt in range(3):
        d = run.llm.agent_call("design", sys, user, DZ.DESIGN_SCHEMA_LLM, T.specs("design", strict=False),
                               lambda n, a: T.call(run, "design", n, a))
        d = {**d, "method": "its", "panel": None}
        errs = DZ.validate(d, run.card)
        if not errs:
            return d
        run.log("design", "설계 검증 실패 → 다시 요청", {"errors": errs})
        user += "\n\n이전 설계의 오류(고쳐서 다시 내세요): " + "; ".join(errs)
    raise ValueError("Design Agent가 3회 안에 유효한 설계를 내지 못함")


# =========================================================================== ④ Executor 복구
def repair_template(run, d: dict, error: str) -> dict | None:
    names = {s["name"] for s in d.get("series", [])}
    nd = copy.deepcopy(d)
    nd["placebo"] = [p for p in nd.get("placebo", []) if p["target"] in names]
    nd["alt_references"] = [a for a in nd.get("alt_references", []) if all(r in names for r in a["reference"])]
    if "사전" in error and nd.get("pre"):
        nd["pre"] = None
    return nd if nd != d else None


def repair_llm(run, d: dict, error: str) -> dict | None:
    sys = COMMON + " 역할: ④ Executor. 잠긴 설계가 실행 오류를 냈습니다. 오류를 고치는 최소한의 변경만 한 설계를 냅니다(대조 집단·기간을 결과를 보고 바꾸지 말 것)."
    user = f"설계:\n{json.dumps(d, ensure_ascii=False)}\n\n오류: {error}"
    if d.get("method") == "did_panel":
        return repair_template(run, d, error)
    nd = run.llm.json_call("executor", sys, user, DZ.DESIGN_SCHEMA_LLM)
    nd = {**nd, "method": "its", "panel": None}
    return nd if not DZ.validate(nd, run.card) else None


# =========================================================================== ⑤ Skeptic
SKEPTIC_SCHEMA = _obj({
    "hypotheses": {"type": "array", "items": _obj({
        "hypothesis": _S, "why_plausible": _S,
        "check_ids": {"type": "array", "items": _S, "description": "extra_check 결과의 id(X1, X2…). 기존 검증으로 충분하면 C1~C4, 검증 못 했으면 빈 배열"}})},
    "summary": _S,
})


def rule_hypotheses(run) -> list[dict]:
    """판정 규칙의 검증 항목을 '반론 → 검증 결과'로 다시 읽는다(추가 하향 없음, 규칙에 이미 반영)."""
    a, v = run.analysis, run.analysis.verdict
    st = {c.code: c for c in v.checks}

    def stat(*codes):
        cs = [st[c] for c in codes if c in st]
        if any(c.status == "실패" for c in cs):
            return "반론 성립"
        if cs and all(c.status in ("통과", "참고") for c in cs):
            return "반론 기각"
        return "미검증"
    hs = [
        {"hypothesis": "측정 방식이 바뀌어 숫자만 달라졌다", "check": "C0 데이터 품질", "status": stat("C0"),
         "detail": st["C0"].detail if "C0" in st else ""},
        {"hypothesis": "정책 전부터 이미 변하고 있었다(기존 추세)", "check": "C1 사전추세", "status": stat("C1"),
         "detail": st["C1"].detail if "C1" in st else ""},
        {"hypothesis": "해마다 이맘때 생기는 변화다(방학·계절·명절)", "check": "계절 차분 + C2a 시점 위약",
         "status": stat("C2a"), "detail": st["C2a"].detail if "C2a" in st else ""},
        {"hypothesis": "대조 집단 쪽에 다른 일이 생겼다", "check": "C2b 공간 위약", "status": stat("C2b"),
         "detail": st["C2b"].detail if "C2b" in st else ""},
        {"hypothesis": "분석 기간·대조 선택에 따라 달라지는 우연이다", "check": "C3 민감도", "status": stat("C3"),
         "detail": st["C3"].detail if "C3" in st else ""},
    ]
    for h in hs:
        h["source"] = "판정 규칙"
    return hs


def _post_end(run) -> str:
    a = run.analysis
    if a.main is not None and a.main.window:
        return a.main.window[1] if len(a.main.window[1]) > 7 else a.main.window[1] + "-28"
    return run.card["start_date"]


def skeptic_template(run) -> list[dict]:
    hs = []
    if run.analysis.main is None:
        return hs
    evs = events_between(run.card["start_date"], _post_end(run), run.design["dataset"], as_of=run.as_of)
    seen = set()
    for ev in evs:
        if ev["name"] in seen or "방학" in ev["name"] or "명절" in ev["name"] or "설·추석" in ev["name"]:
            continue          # 방학·명절은 계절 차분과 시점 위약이 다룬다
        if ev.get("case") == run.card.get("library_case") or ev["start"] == run.card["start_date"] or "(데이터)" in ev["name"]:
            continue          # 정책 자체, 데이터 결측은 품질 점검이 다룬다
        seen.add(ev["name"])
        h = {"hypothesis": f"같은 시기의 다른 사건 때문이다: {ev['name']}", "why_plausible": f"영향: {ev['affects']} (출처: {ev['source']})",
             "check_ids": [], "source": "사건 달력"}
        if run.design["method"] == "its":
            out, err = T.call(run, "skeptic", "extra_check", {
                "kind": "exclude_period", "reference": [], "placebo_target": None, "post": None,
                "start": ev["start"], "end": ev["end"], "reason": ev["name"]})
            if not err:
                h["check_ids"] = [json.loads(out)["id"]]
        hs.append(h)
    return hs


def skeptic_llm(run) -> list[dict]:
    a = run.analysis
    sys = (COMMON + " 역할: ⑤ Skeptic. 판정 결과를 반박합니다. 정책 맥락(방학, 연휴, 관광 회복, 유가, 다른 지원금·정책, 코로나19, 데이터 결측)을 보고 "
           "그럴듯한 대안 가설을 2~4개 세우고, event_calendar로 사실을 확인한 뒤 extra_check로 검증합니다. "
           "판정 규칙이 이미 본 검증(C0~C4)과 같은 것은 다시 하지 말고 check_ids에 C코드를 적습니다. 반론의 성립 여부는 툴 결과의 status를 따릅니다.")
    facts = {"card": run.card, "design": {k: run.design[k] for k in ("dataset", "method", "target", "reference", "placebo",
                                                                     "alt_references", "pre", "post", "exclude")},
             "verdict": a.verdict.label, "checks": [c.to_dict() for c in a.verdict.checks],
             "main": None if a.main is None else {"effect_pct": a.main.pct, "ci_pct": a.main.ci_pct, "window": a.main.window},
             "series": [s["name"] for s in run.design.get("series", [])]}
    out = run.llm.agent_call("skeptic", sys, json.dumps(facts, ensure_ascii=False, indent=1, default=str),
                             SKEPTIC_SCHEMA, T.specs("skeptic"), lambda n, x: T.call(run, "skeptic", n, x))
    hs = []
    for h in out.get("hypotheses", []):
        h["source"] = "Skeptic(LLM)"
        h["rule_checks"] = [c for c in h.get("check_ids", []) if c.startswith("C")]
        h["check_ids"] = [c for c in h.get("check_ids", []) if c.startswith("X")]
        hs.append(h)
    run.skeptic_summary = out.get("summary", "")
    return hs


def resolve_skeptic(run, hs: list[dict]) -> dict:
    """반론 결과를 코드로 판정하고, 판정 하향을 적용한다.

    하향 규칙: 판정 규칙 밖의 반론(사건 달력·LLM 가설)이 검증에서 '성립'하면
      효과 있음 → 조건부, 효과 없음 → 판정 보류. (조건부·보류는 그대로)
    """
    checks = {x["id"]: x for x in run.extra_checks}
    vc = {c.code: c for c in run.analysis.verdict.checks}
    for h in hs:
        res = [checks[c] for c in h.get("check_ids", []) if c in checks]
        h["results"] = res
        rule = [vc[c] for c in h.get("rule_checks", []) if c in vc and vc[c].status in ("통과", "실패")]
        h["rule_results"] = [f"{c.code} {c.name} {c.status}" for c in rule]
        if any(r["status"] == "반론 성립" for r in res):
            h["status"] = "반론 성립"
        elif res:
            h["status"] = "반론 기각"
        elif rule:
            # 판정 규칙의 검증으로 답한 반론: 규칙에 이미 반영되어 추가 하향은 하지 않는다
            h["status"] = "규칙 검증: 성립" if any(c.status == "실패" for c in rule) else "규칙 검증: 기각"
        else:
            h["status"] = "미검증"
    v = run.analysis.verdict
    level, label = v.level, v.label
    standing = [h for h in hs if h["status"] == "반론 성립"]
    downgraded = None
    if standing and level == "효과 있음":
        downgraded = ("효과 있음", "조건부")
    elif standing and level == "효과 없음":
        downgraded = ("효과 없음", "판정 보류")
    if downgraded:
        level = downgraded[1]
        label = f"{level}({v.direction})" if level == "조건부" and v.direction else level
    return {"rule_hypotheses": rule_hypotheses(run), "hypotheses": hs, "extra_checks": run.extra_checks,
            "standing": [h["hypothesis"] for h in standing], "downgraded": downgraded,
            "rule_standing": [h["hypothesis"] for h in rule_hypotheses(run) if h["status"] == "반론 성립"],
            "final_level": level, "final_label": label, "light": LIGHT[level],
            "summary": getattr(run, "skeptic_summary", "")}


# =========================================================================== ⑥ Reporter
REPORT_SCHEMA = _obj({"headline": _S, "summary": _S, "evidence": _SL, "limits": _SL, "cannot_say": _SL,
                      "needed_data": _SL})
_NUM = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?")
_MONTH = re.compile(r"(?<![\d.])(?:1[0-2]|[1-9])월")


def _norm(tok: str) -> str:
    t = tok.replace(",", "").replace("−", "-").lstrip("+")
    try:
        f = float(t)
    except ValueError:
        return t
    return f"{abs(f):.4g}"


def number_check(texts: list[str], facts: str) -> list[str]:
    """서술 속 숫자가 사실 목록(툴 결과)에 있는지 대조한다. 없는 숫자 목록을 돌려준다."""
    allowed = {_norm(t) for t in _NUM.findall(facts)} | {_norm(str(i)) for i in range(0, 6)} | {"95", "90"}
    bad = []
    for t in texts:
        t = _MONTH.sub("", t)          # '8월' 같은 달 이름은 통계 수치가 아니다
        for tok in _NUM.findall(t):
            if _norm(tok) not in allowed:
                bad.append(tok)
    return sorted(set(bad))


def facts_sheet(run) -> str:
    a, sk = run.analysis, run.skeptic
    v = a.verdict
    L = [f"최종 판정: {sk['light']} {sk['final_label']} (판정 규칙 결과: {v.label})",
         f"정책: {run.card['policy']} / 시행일 {run.card['start_date']} / 질문: {run.card['question']}",
         f"데이터: {catalog.DATASETS[run.design['dataset']].title}"]
    if sk.get("downgraded"):
        L.append(f"Skeptic 하향: {sk['downgraded'][0]} → {sk['downgraded'][1]} (성립한 반론: {'; '.join(sk['standing'])})")
    L += [f"판정 근거: {r}" for r in v.reasons]
    if a.main is not None:
        m = a.main
        lo, hi = np.expm1(v.ci_used[0]) * 100, np.expm1(v.ci_used[1]) * 100
        L.append(f"추정 효과: {m.pct * 100:+.1f}% (판정 구간 {lo:+.1f}% ~ {hi:+.1f}%, p={v.p_used:.3f}), "
                 f"분석 기간 {m.window[0]} ~ {m.window[1]}, 사전 {m.n_pre} · 사후 {m.n_post}")
    L += [f"검증 {c.code} {c.name}: {c.status} — {c.detail}" for c in v.checks]
    for h in sk["rule_hypotheses"] + sk["hypotheses"]:
        res = "; ".join(f"{r['label']}: {r['pct'] * 100:+.1f}% ({r['status']})" for r in h.get("results", []))
        L.append(f"반론: {h['hypothesis']} → {h['status']}" + (f" [{h.get('check', '')} {h.get('detail', '')}]" if h.get("check") else "")
                 + (f" [{res}]" if res else ""))
    if a.power is not None:
        L.append(f"검정력: {a.power.sentence_ko()}")
    if v.needed_data:
        L += [f"필요한 데이터: {x}" for x in v.needed_data]
    if a.quality is not None and a.quality.needed_data:
        L += [f"필요한 데이터: {x}" for x in a.quality.needed_data if x not in v.needed_data]
    L.append(f"사전 등록 지문: {run.lock_rec['sha256'][:16]}")
    return "\n".join(L)


def _template_summary(run) -> str:
    a, sk, v = run.analysis, run.skeptic, run.analysis.verdict
    d = run.design
    if a.main is None:
        return f"판정 보류입니다. {v.reasons[0]}"
    m = a.main
    lo, hi = np.expm1(v.ci_used[0]) * 100, np.expm1(v.ci_used[1]) * 100
    who = (f"'{d['target']}'의 변화를 대조 집단 '{'+'.join(d['reference'])}'와 비교했습니다" if d["method"] == "its"
           else f"'{d['panel']['treated']}'의 업종별 변화를 다른 집단과 비교했습니다")
    S = [f"{run.card['start_date']} 시행 전후를 놓고 {who}.",
         f"추정 변화는 {m.pct * 100:+.1f}%이고, 판정에 쓴 구간은 {lo:+.1f}% ~ {hi:+.1f}%입니다."]
    failed = [f"{c.code} {c.name}" for c in v.checks if c.status == "실패"]
    if failed:
        S.append(f"통과하지 못한 검증: {', '.join(failed)}.")
    elif v.level == "판정 보류":
        S.append("검증 항목은 모두 통과했지만, 판정에 쓴 구간이 넓어(검정력 부족) 결론을 미룹니다.")
    else:
        S.append("검증 항목을 모두 통과했습니다.")
    if sk.get("downgraded"):
        S.append(f"Skeptic이 세운 반론({'; '.join(sk['standing'])})이 성립해 판정을 {sk['downgraded'][1]}(으)로 낮췄습니다.")
    S.append(f"최종 판정은 {sk['final_label']}입니다.")
    return " ".join(S)


def reporter_template(run, facts: str) -> dict:
    a, sk = run.analysis, run.skeptic
    v = a.verdict
    head = f"{sk['final_label']}: {v.reasons[0]}"
    ev = [f"{c.code} {c.name} {c.status}: {c.detail}" for c in v.checks]
    lim = [f"{c.code} {c.name}을(를) 하지 못했습니다: {c.detail}" for c in v.checks if c.status == "해당 없음"]
    lim += [f"검증하지 못한 반론: {h['hypothesis']}" for h in sk["hypotheses"] if h["status"] == "미검증"]
    lim += [f"정책 카드 가정(확인 필요): {x}" for x in run.card.get("assumptions", [])]
    if a.main is not None and a.main.inference.startswith("집단 수준"):
        lim.append("처치 집단이 하나라 업종별 표준오차 대신 집단 수준 격차 시계열로 불확실성을 계산했습니다.")
    ds = catalog.DATASETS[run.design["dataset"]]
    if ds.synthetic:
        lim.append("합성데이터 시연입니다. 실제 정책 효과에 대한 결론이 아닙니다.")
    elif ds.freq == "MS":
        lim.append("월 단위·단일 카드사(비씨카드) 실적이라 제주 전체 소비가 아니며, 검정력이 낮습니다.")
    cannot = ["이 분석은 비교 집단이 정책이 없었다면 대상 집단과 같은 방향으로 움직였을 것(평행추세)을 가정합니다. 이 가정은 검정으로 완전히 증명할 수 없습니다.",
              "효과의 원인이 정책의 어떤 요소(요금, 홍보, 동시 시행 정책)인지는 구분하지 못합니다."]
    if v.quality_cap:
        cannot.insert(0, "데이터 품질 문제의 원인은 단정하지 않습니다. 데이터 제공처 확인이 필요합니다.")
    need = list(v.needed_data) + ([x for x in a.quality.needed_data if x not in v.needed_data] if a.quality else [])
    summ = _template_summary(run)
    return {"headline": head, "summary": summ, "evidence": ev, "limits": lim, "cannot_say": cannot, "needed_data": need}


def reporter_llm(run, facts: str) -> dict:
    sys = (COMMON + " 역할: ⑥ Reporter. 아래 사실 목록만으로 1장 판정서 문장을 씁니다. 결재 문서에 붙일 수 있게 쉬운 말로 씁니다. "
           "숫자는 사실 목록의 숫자를 그대로 쓰고 새로 계산하지 않습니다. 판정은 바꾸지 않습니다. "
           "headline은 한 문장 결론, summary는 3~4문장, evidence는 핵심 근거 3~6개, limits는 한계, "
           "cannot_say에는 '이 분석이 말할 수 없는 것'을 반드시 1개 이상, needed_data에는 결론에 필요한 데이터(보류일 때 필수)를 씁니다.")
    user = "사실 목록:\n" + facts
    for attempt in range(2):
        out = run.llm.json_call("reporter", sys, user, REPORT_SCHEMA)
        texts = [out["headline"], out["summary"]] + out["evidence"] + out["limits"] + out["cannot_say"] + out["needed_data"]
        bad = number_check(texts, facts)
        if not bad and out["cannot_say"]:
            return out
        run.log("report", "서술 숫자 대조 실패 → 다시 쓰기", {"numbers": bad})
        user += f"\n\n이전 서술에 사실 목록에 없는 숫자가 있었습니다: {bad}. 사실 목록의 숫자만 쓰세요."
    raise ValueError(f"서술 속 숫자가 툴 결과와 맞지 않음: {bad}")


def render_markdown(run, rep: dict, facts: str) -> str:
    a, sk, v = run.analysis, run.skeptic, run.analysis.verdict
    L = [f"# 판정서: {run.card['policy']}", "", f"## {sk['light']} {sk['final_label']}", "", f"**{rep['headline']}**", "",
         rep["summary"], "", "## 정책 카드", "",
         f"- 질문: {run.card['question']}", f"- 시행일: {run.card['start_date']} · 대상: {run.card['treated']}",
         f"- 결과지표: {run.card['outcome']}", f"- 데이터: {catalog.DATASETS[run.design['dataset']].title}", "",
         "## 근거", ""] + [f"- {x}" for x in rep["evidence"]] + ["", "## 검증 결과", "", "| 항목 | 결과 | 내용 |", "|---|---|---|"]
    L += [f"| {c.code} {c.name} | {c.status} | {c.detail} |" for c in v.checks]
    L += ["", "## 반론 검증 로그", "", "| 반론 | 검증 | 결과 |", "|---|---|---|"]
    for h in sk["rule_hypotheses"]:
        L.append(f"| {h['hypothesis']} | {h['check']} | {h['status']} |")
    for h in sk["hypotheses"]:
        res = "; ".join([f"{r['label']} → {r['pct'] * 100:+.1f}% ({r['status']})" for r in h.get("results", [])]
                        + h.get("rule_results", [])) or "-"
        L.append(f"| {h['hypothesis']} | {res} | {h['status']} |")
    if sk.get("downgraded"):
        L += ["", f"> Skeptic 하향: {sk['downgraded'][0]} → {sk['downgraded'][1]}"]
    if a.power is not None:
        L += ["", "## 검정력", "", f"- {a.power.sentence_ko()}"]
    L += ["", "## 한계", ""] + ([f"- {x}" for x in rep["limits"]] or ["- (없음)"])
    L += ["", "## 이 분석이 말할 수 없는 것", ""] + [f"- {x}" for x in rep["cannot_say"]]
    if rep["needed_data"]:
        L += ["", "## 판정에 필요한 데이터", ""] + [f"- {x}" for x in rep["needed_data"]]
    L += ["", "## 재현 정보", "", f"- 실행 번호 `{run.run_id}` · 모드 {run.mode} · 판정 규칙 {v.rules_version}",
          f"- 사전 등록 지문 `{run.lock_rec['sha256']}` (잠금 {run.lock_rec['locked_at']})"]
    if run.lock_rec.get("existing_prereg"):
        e = run.lock_rec["existing_prereg"]
        L.append(f"- 기존 사전 등록 `{e['file']}` 지문 확인 `{e['sha256'][:16]}…`")
    if run.truth:
        t = run.truth
        inside = a.main is not None and v.ci_used[0] <= t["effect_log"] <= v.ci_used[1]
        L += ["", "## 정답 대조 (합성데이터, 판정 뒤 공개)", "",
              f"- 심어 둔 효과: {t['group']} {t['effect_pct'] * 100:+.1f}% ({t['date']}부터)",
              f"- 추정: {a.main.pct * 100:+.1f}%, 판정 구간이 정답을 {'포함함' if inside else '포함하지 않음'}"]
    return "\n".join(L) + "\n"
