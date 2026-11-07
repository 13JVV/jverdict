"""사례 A·B·C·D·E를 Agent로 실행하고 모드별 결과·비용·재현 여부를 한 표로 정리한다.

    python -m jverdict.agent.demo                     # 템플릿 모드만 (오프라인, 무료)
    python -m jverdict.agent.demo --claude            # Claude 모드도 실행 (사례당 약 $0.1)
결과: outputs/runs/case-<사례>-<모드>/ (verdict.pdf 1장 판정서 포함), outputs/runs/_summary.md, _summary.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings

from .library import CARDS
from .orchestrator import RUNS, replay, run


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", action="store_true")
    ap.add_argument("--cases", nargs="*", default=list("ABCDEF"))
    a = ap.parse_args(argv)
    modes = ["template"] + (["claude"] if a.claude else [])
    for case in a.cases:
        for mode in modes:
            rid = f"case-{case}-{mode}"
            for p in (RUNS / rid, RUNS / f"{rid}-replay"):
                shutil.rmtree(p, ignore_errors=True)
            t0 = time.time()
            r = run(CARDS[case]["card"]["question"], mode=mode, run_id=rid)
            sec = time.time() - t0
            rep = replay(r.run_dir)
            man = json.loads((r.run_dir / "manifest.json").read_text(encoding="utf-8"))
            pdf = _pdf(r.run_dir)
            print(f"{case} {mode:<8} {man['final_verdict']:<12} 효과(로그) {man['numbers'].get('effect')} "
                  f"재현 {rep['reproduced']} ${man['llm']['cost_usd']} {sec:.0f}초 판정서 {pdf}", flush=True)
    summarize()


def _pdf(run_dir) -> str:
    """1장 판정서. 브라우저가 없거나 실패해도 시연은 계속한다."""
    from jverdict.report.onepage import make
    try:
        r = make(run_dir)
        return f"{r['pages']}쪽 PDF" if r["pdf"] else "HTML(브라우저 없음)"
    except Exception as e:  # noqa: BLE001
        return f"실패({type(e).__name__})"


def summarize() -> str:
    """모든 실행 폴더(case-*)의 manifest에서 요약표를 다시 만든다(일부 사례만 다시 돌려도 표가 완전함)."""
    rows = []
    for d in sorted(RUNS.glob("case-*")):
        if d.name.endswith("-replay") or not (d / "manifest.json").exists():
            continue
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        ev = [json.loads(x) for x in (d / "events.jsonl").read_text(encoding="utf-8").splitlines() if x]
        t = [time.strptime(e["t"], "%H:%M:%S") for e in ev]
        sec = (time.mktime(t[-1]) - time.mktime(t[0])) if t else 0
        rc = RUNS / f"{d.name}-replay" / "replay_check.json"
        rcj = json.loads(rc.read_text(encoding="utf-8")) if rc.exists() else {}
        rep = rcj.get("reproduced")
        design = json.loads((d / "design.json").read_text(encoding="utf-8"))
        rule = json.loads((d / "analysis.json").read_text(encoding="utf-8"))["verdict"]["label"]
        rows.append({"run": d.name, "mode": man["mode"], "verdict": man["final_verdict"], "rule_verdict": rule,
                     "effect": man["numbers"].get("effect"), "reference": design.get("reference") or ["패널"],
                     "llm_calls": man["llm"]["calls"], "cost_usd": man["llm"]["cost_usd"], "seconds": sec,
                     "reproduced": rep, "replay_error": rcj.get("error"), "fallbacks": len(man["fallbacks"]),
                     "created": man["created"]})
    L = ["# Agent 실행 요약", "", f"생성 {time.strftime('%Y-%m-%d %H:%M')}", "",
         "| 실행 | 모드 | 최종 판정 | 판정 규칙 결과 | 효과(로그) | 대조 집단 | LLM 호출 | 비용 | 소요 | 재실행 일치 | 템플릿 대체 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        eff = "-" if x["effect"] is None else f"{x['effect']:+.4f}"
        rp = {True: "예", False: "아니오", None: "-"}[x["reproduced"]]
        L.append(f"| {x['run']} | {x['mode']} | {x['verdict']} | {x['rule_verdict']} | {eff} | {', '.join(x['reference'])} | "
                 f"{x['llm_calls']} | ${x['cost_usd']:.3f} | {x['seconds']:.0f}초 | {rp} | {x['fallbacks']} |")
    L += ["", "- 재실행 일치: 같은 manifest로 다시 실행해 핵심 수치·판정·판정서 문장이 같은지(Claude 모드는 API 없이 응답 캐시로).",
          "- 템플릿 대체: Claude 단계가 실패해 그 단계만 템플릿으로 대신한 횟수(`case-C-claude-apifail`은 API 장애 대응 시연: 잘못된 키로 API 실패를 일으켜 전 단계 대체. 처음에는 2026-11-03 크레딧 소진 때 실측).",
          "- 등록 설계가 없는 사례(A·C)는 Claude가 대조 집단을 고르므로 효과 값이 템플릿과 다를 수 있다. 사전 등록 사례(B·D·E)는 같다."]
    L += [f"- `{x['run']}` 재실행 실패: {x['replay_error']}" for x in rows if x.get("replay_error")]
    if any("호출의 입력이" in (x.get("replay_error") or "") for x in rows):
        L.append("  (정규화 지문 도입 전 기록: 요청 속 소수의 마지막 비트 차이에 걸림. 새로 실행하면 해결됨)")
    L.append("")
    txt = "\n".join(L)
    (RUNS / "_summary.md").write_text(txt, encoding="utf-8")
    (RUNS / "_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(txt)
    return txt


if __name__ == "__main__":
    main()
