"""'Agent의 판단' 기록: 단계마다 누가(Claude·규칙·사용자) 무엇을 왜 정했는지, 어떤 툴을 불렀는지.

API 키가 없어도 Claude 실행 기록(이미 저장된 판단과 이유)을 그대로 보여 준다. 새로 계산하지 않는다.
숫자는 툴이 계산하고 LLM은 판단·서술만 한다는 원칙이 기록으로 보이게 하는 화면이다.
"""
from __future__ import annotations

import json
from collections import Counter

STAGES = [("parse", "① 정책 카드 (Policy Parser)"), ("scout", "② 데이터 점검 (Data Scout)"),
          ("design", "③ 비교 설계 (Design Agent)"), ("lock", "잠금 (사전 등록)"), ("execute", "④ 추정·판정 규칙 (Executor)"),
          ("skeptic", "⑤ 반론 검증 (Skeptic)"), ("report", "⑥ 판정서 (Reporter)")]
TOOL_KO = {"catalog_search": "데이터 찾기", "dataset_profile": "데이터 요약(정책 전만)", "official_stats": "공식 통계 목록",
           "quality_check": "측정 단절·커버리지 점검", "pick_controls": "대조 후보 유사도(정책 전만)",
           "event_calendar": "같은 시기 사건 찾기", "extra_check": "반론 검증 재추정"}
LLM_STAGES = {"parse", "scout", "design", "skeptic", "report"}


def _who(v, stage: str) -> str:
    man = v.manifest
    if stage == "parse" and v.card.get("manual"):
        return "사용자(정책 카드 직접 작성)"
    if stage not in LLM_STAGES:
        return "코드(고정 규칙)"
    fb = {f["stage"] for f in man.get("fallbacks", [])}
    if man["mode"] in ("claude", "replay") and stage not in fb and v.manifest.get("llm", {}).get("calls"):
        return "Claude"
    if man["mode"] in ("claude", "replay") and stage in fb:
        return "템플릿(Claude 실패로 대체)"
    return "템플릿(규칙 기반)"


def _tools(v, stage: str) -> list[str]:
    out = []
    for e in v.events:
        if e.get("stage") == stage and e.get("msg", "").startswith("툴 "):
            name = e["msg"][2:]
            args = (e.get("data") or {}).get("args", {})
            why = args.get("reason") or args.get("query") or ""
            rng = f"{args['start']}~{args['end']}" if args.get("start") and args.get("end") else ""
            extra = " · ".join(x for x in (args.get("kind", ""), rng, why) if x)
            out.append(f"{TOOL_KO.get(name, name)} (`{name}`)" + (f": {extra}" if extra else "")
                       + (" — 오류" if (e.get("data") or {}).get("error") else ""))
    return out


def llm_usage(v) -> dict:
    p = v.run_dir / "llm_cache.json"
    if not p.exists():
        return {}
    try:
        cache = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    c = Counter(r.get("role") for r in cache)
    return {"calls": dict(c), "cost_usd": round(sum(float(r.get("cost_usd") or 0) for r in cache), 4),
            "model": sorted({r.get("model") for r in cache if r.get("model")})}


def steps(v) -> list[dict]:
    a, sk, rep = v.analysis, v.skeptic, v.report
    scout = {}
    sp = v.run_dir / "scout.json"
    if sp.exists():
        scout = json.loads(sp.read_text(encoding="utf-8"))
    out = []
    for key, title in STAGES:
        items: list[str] = []
        if key == "parse":
            items = [f"정책: {v.card.get('policy', '')} · 시행일 {v.card.get('start_date', '')} · 대상 {v.card.get('treated', '')}"]
            items += [f"가정: {x}" for x in v.card.get("assumptions", [])]
            items.append("사람 확인 1회: 정책 카드를 승인한 뒤에 다음 단계로 넘어감")
        elif key == "scout":
            items = [f"측정 단절 진단: {scout.get('quality_verdict', '-')}"
                     + (f" → 판정 상한 '{scout['quality_cap']}'" if scout.get("quality_cap") else "")]
            items += list(scout.get("notes", []))
        elif key == "design":
            items = [f"{c['decision']} — {c['reason']}" for c in v.design.get("choices", [])]
            items.append("이 단계에서는 정책 이후 값(효과)을 볼 수 없음(효과 가림)")
        elif key == "lock":
            pre = v.manifest["prereg"]
            items = [f"설계 잠금 SHA-256 {pre['sha256'][:16]}… · {pre['locked_at'][:16].replace('T', ' ')}"]
            if pre.get("existing"):
                items.append(f"기존 사전 등록 문서 `{pre['existing']['file']}` 지문 확인")
            items += [f"잠금 뒤 변경 {x.get('n')}: {x.get('reason')}" for x in pre.get("amendments", [])]
        elif key == "execute":
            vd = a["verdict"]
            items = [f"판정 규칙 {v.manifest.get('rules_version', '')} 결과: {vd['label']}"]
            items += [f"{c['code']} {c['name']}: {c['status']}" for c in vd.get("checks", [])]
            items.append("숫자는 모두 통계 툴이 계산함(LLM은 숫자를 만들지 않음)")
        elif key == "skeptic":
            for h in sk.get("hypotheses", []):
                why = h.get("why_plausible", "")
                items.append(f"[{h.get('source', '')}] {h['hypothesis']} → {h['status']}" + (f" (근거: {why})" if why else ""))
            if sk.get("summary"):
                items.append(f"정리: {sk['summary']}")
            if sk.get("downgraded"):
                items.append(f"판정 하향: {sk['downgraded'][0]} → {sk['downgraded'][1]}")
            if not items:
                items = ["추가 반론 없음(판정 규칙의 기본 반론만 검증)"]
        elif key == "report":
            items = [f"한 줄 결론: {rep.get('headline', '')}",
                     "서술 속 숫자를 툴 결과와 대조하는 검사를 통과한 판정서만 저장함"]
        out.append({"stage": key, "title": title, "who": _who(v, key), "items": items, "tools": _tools(v, key)})
    return out


def compare_with_template(v, runs_dir) -> dict | None:
    """Claude 실행이면 같은 사례의 템플릿 실행과 판단을 나란히 비교한다."""
    key = v.card.get("library_case")
    if v.manifest["mode"] == "template" or not key:
        return None
    t = runs_dir / f"case-{key}-template"
    if not (t / "manifest.json").exists():
        return None
    from jverdict.report.view import load_run
    tv = load_run(t)

    def ref(x):
        d = x.design
        return "패널(" + ", ".join(d["panel"]["groups"]) + ")" if d.get("method") == "did_panel" else ", ".join(d.get("reference", []))
    m1, m2 = v.analysis.get("main") or {}, tv.analysis.get("main") or {}
    return {"template_run": t.name,
            "rows": [("대조 집단", ref(v), ref(tv)),
                     ("추정 변화", f"{m1.get('pct', float('nan')) * 100:+.1f}%" if m1 else "-",
                      f"{m2.get('pct', float('nan')) * 100:+.1f}%" if m2 else "-"),
                     ("추가 반론 수", str(len(v.skeptic.get("hypotheses", []))), str(len(tv.skeptic.get("hypotheses", [])))),
                     ("최종 판정", v.label, tv.label)],
            "same": v.label == tv.label}
