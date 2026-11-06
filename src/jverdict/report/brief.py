"""판정서 맨 위의 '한눈 요약': 한 줄 결론과 판정 이유 몇 줄.

긴 판정 문장(report.headline)을 화면 첫머리에 그대로 두면 읽기 어렵다. 같은 기록에서
결론의 핵심 원인을 골라 짧은 제목과 '추정 · 검증 · 원인 · 다음' 네 줄로 나눈다.
새 숫자를 만들지 않고 분석 결과(analysis.json)의 값만 옮긴다.
"""
from __future__ import annotations

import math
import re


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _first_sentence(t: str) -> str:
    t = (t or "").strip()
    m = re.match(r"(.+?\.)(\s|$)", t)
    return m.group(1) if m else t


def _fix_josa(t: str) -> str:
    """엔진 문구의 "'청소년'가" 같은 조사를 받침에 맞게(엔진 문구는 재실행 지문 때문에 그대로 둔다)."""
    from jverdict.agent.cardbuilder import _josa
    return re.sub(r"'([^']+)'(가|이) ", lambda m: f"'{m.group(1)}'{_josa(m.group(1), '이', '가')} ", t or "")


def brief(v) -> dict:
    """{title, cause, points: [(라벨, 내용)]}"""
    from jverdict.report import actions as AC
    a = v.analysis
    vd, m, pw = a["verdict"], a.get("main") or {}, a.get("power") or {}
    checks = {c["code"]: c for c in vd.get("checks", [])}
    failed = [c for c in vd.get("checks", []) if c["status"] == "실패"]
    level = v.level

    def st(code):
        return (checks.get(code) or {}).get("status")

    cause, title, why = "other", "", ""
    if st("C0") == "실패":
        cause = "quality"
        title = "데이터 집계 방식이 바뀐 것으로 의심되어 판정을 미룹니다"
        why = checks["C0"]["detail"].split(". ", 1)[-1]
    elif level == "판정 보류" and pw.get("pre_outliers"):
        cause = "outlier"
        o = pw["pre_outliers"][0]
        title = "비교 기준 기간에 크게 벗어난 시점이 있어 판정을 미룹니다"
        why = (f"{o['time']}의 대상/대조 격차가 기준 기간의 보통 수준보다 {abs(o['dev']):.2f}(로그) "
               f"{'낮아' if o['dev'] < 0 else '높아'} 기준선과 구간이 함께 흔들림")
    elif level == "판정 보류" and st("C1") == "실패" and "심각" in checks["C1"]["detail"]:
        cause = "pretrend"
        title = "정책 전부터 있던 변화와 구분할 수 없어 판정을 미룹니다"
        ta = a.get("trend_adjusted") or {}
        pm = re.search(r"p=([0-9.e+-]+)", checks["C1"]["detail"])
        why = "정책 전부터 뚜렷한 추세가 있었음" + (f"(사전추세 검정 p={float(pm.group(1)):.1g})" if pm else "")
        if ta.get("pct") is not None:
            why += f". 기존 추세를 빼면 {_pct(ta['pct'])}"
    elif level == "판정 보류" and pw.get("mde_pct") is not None and not pw.get("enough", True):
        cause = "power"
        title = "작은 효과를 가려낼 만큼 자료가 충분하지 않아 판정을 미룹니다"
        why = (f"이 자료로 구분할 수 있는 최소 효과는 ±{pw['mde_pct'] * 100:.1f}%로, "
               f"의미 있다고 본 {pw['reference_pct'] * 100:.0f}%보다 큼")
    elif level == "효과 있음":
        cause = "effect"
        title = f"정책 이후 {_pct(m.get('pct', 0))}의 변화가 확인되었습니다"
        c2 = checks.get("C2a") or {}
        why = (f"시점 위약검정: {c2['detail']}. 정책이 없던 때의 변화보다 뚜렷함" if c2.get("status") == "통과"
               else "대조 집단·계절 보정 뒤에도 변화가 유지됨")
    elif level == "효과 없음":
        cause = "null"
        title = "의미 있는 크기의 변화는 없었습니다"
        why = _first_sentence(vd.get("reasons", [""])[0])
    elif level == "조건부":
        cause = "partial"
        title = "변화의 방향은 보이지만 검증 일부를 통과하지 못했습니다"
        why = _first_sentence(vd.get("reasons", [""])[0])
    else:
        title = _first_sentence(vd.get("reasons", [""])[0]) if vd.get("reasons") else v.label

    points = []
    if m:
        lo, hi = vd["ci_used"]
        points.append(("추정", f"정책 이후 {_pct(m['pct'])} · 판정 구간 {_pct(math.expm1(lo))} ~ {_pct(math.expm1(hi))}"))
    n = len(vd.get("checks", []))
    if n:
        ok = sum(1 for c in vd["checks"] if c["status"] == "통과")
        txt = f"{n}개 중 {ok}개 통과"
        if failed:
            txt += " · 통과하지 못함: " + ", ".join(f"{c['code']} {c['name']}" for c in failed)
        points.append(("검증", txt))
    if why:
        points.append(("원인" if level != "효과 있음" else "근거", why))
    dg = v.skeptic.get("downgraded")
    if dg:
        points.append(("반론", f"Skeptic의 반론이 성립해 {dg[0]} → {dg[1]}로 낮춤"))
    rj = AC.rejudge(v)
    if rj["kind"] != "none":
        points.append(("다음", _first_sentence(rj["text"])))
    return {"title": title, "cause": cause, "points": [(k, _fix_josa(t)) for k, t in points]}
