"""판정 뒤 '다음 할 일': 결재용 개조식 요약, 재판정 예정, 자료 요청 공문 초안.

보류도 쓸모 있게 만든다. 모든 숫자는 분석 결과(analysis.json)와 판정서에서 그대로 옮긴다(새 숫자를 만들지 않음).
공문 초안은 수신처·기한·작성자 칸을 비워 둔다(사람이 채움).
"""
from __future__ import annotations

import math

import pandas as pd

REJUDGE_MAX = {"개월": 36, "일": 3 * 365}
FINE = "더 세밀한 자료(일 단위·전체 카드사 등)를 확보해 다시 판정하세요."          # 이보다 오래 기다려야 하면 기간이 아니라 자료로 풀어야 한다


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def rejudge(v) -> dict:
    """재판정 예정: {kind: date|data|none, text}. 기간을 늘리면 풀리는 경우에만 날짜를 준다."""
    a, card = v.analysis, v.card
    pw, main = a.get("power") or {}, a.get("main")
    level = v.level
    if level in ("효과 있음", "효과 없음"):
        return {"kind": "none", "text": "결론이 났습니다. 정책을 바꾸거나 새 자료가 나오면 같은 설계로 다시 판정할 수 있습니다."}
    if (getattr(v, "skeptic", None) or {}).get("downgraded"):
        return {"kind": "data", "text": "같은 시기 다른 사건의 영향을 가려낼 자료가 필요합니다(기간을 늘려도 풀리지 않음). 사건 기간·대상을 "
                                        "확인하고, 사건과 겹치지 않는 비교 대상이나 기간으로 새 설계를 등록해 다시 판정하세요."}
    if a["verdict"].get("checks") and any(c["code"] == "C0" and c["status"] == "실패" for c in a["verdict"]["checks"]):
        return {"kind": "data", "text": "데이터 측정 방식부터 확인해야 합니다(기간을 늘려도 풀리지 않음). 필요한 자료를 확보한 뒤 다시 판정하세요."}
    if main is None:
        return {"kind": "data", "text": "분석할 관측이 부족합니다. 필요한 자료를 확보한 뒤 다시 판정하세요."}
    if pw.get("pre_outliers"):
        t = ", ".join(o["time"] for o in pw["pre_outliers"])
        return {"kind": "data", "text": f"정책 전 기준 기간의 이상 시점({t}) 원인을 먼저 확인해야 합니다(기간을 늘려도 풀리지 않음). "
                                        "원인이 확인되면 그 시점을 다루는 설계를 새로 등록해 다시 판정하되, 이 판정은 그대로 기록에 남깁니다."}
    need, n_post, unit = pw.get("post_needed"), pw.get("n_post") or 0, pw.get("unit") or "개월"
    if pw.get("post_reachable") is False:
        return {"kind": "data", "text": f"정책 전 기간이 짧아 기다려서는 풀 수 없습니다. 더 긴 정책 전 기간이나 {FINE}"}
    if not need:
        return {"kind": "data", "text": "검증 실패 항목을 해결할 자료나 설계가 필요합니다('판정에 필요한 데이터' 참고)."}
    extra = max(0, int(need) - int(n_post))
    if card.get("end_date"):
        return {"kind": "data", "text": f"끝난 정책이라 기다려서는 풀 수 없습니다. {FINE}"}
    if extra > REJUDGE_MAX.get(unit, 36):
        yrs = need / (12 if unit == "개월" else 365)
        return {"kind": "data", "text": f"기준 효과를 가리려면 사후 관측이 약 {yrs:.0f}년 치 필요해 기다려서는 풀 수 없습니다. {FINE}"}
    end = pd.Timestamp(main["window"][1] + ("-01" if len(main["window"][1]) == 7 else ""))
    when = end + (pd.DateOffset(months=extra) if unit == "개월" else pd.Timedelta(days=extra))
    if when <= pd.Timestamp.now().normalize():
        return {"kind": "data", "text": "필요한 사후 기간은 이미 지났지만 이 설계의 기간은 정책 기간에 맞춰 고정되어 있어, "
                                        f"기다려서는 풀 수 없습니다. {FINE}"}
    return {"kind": "date", "date": when.strftime("%Y-%m"),
            "text": f"정책이 계속되면 자료가 약 {extra}{unit} 더 쌓인 {when.strftime('%Y년 %m월')} 이후 같은 설계로 다시 판정하세요"
                    f"(필요한 사후 관측 약 {need}{unit}, 현재 {n_post}{unit}, 근사)."}


def _dot(d: str) -> str:
    """공문 날짜 표기: 2025-08-01 → 2025. 8. 1."""
    parts = [x for x in str(d).split("-") if x]
    return ". ".join(str(int(x)) if i else x for i, x in enumerate(parts)) + "."


def _next_noun(v) -> str:
    """향후 조치(개조식 명사형). rejudge()의 안내문을 결재 문체로 줄인다."""
    rj = rejudge(v)
    needed = needed_for_request(v)
    if rj["kind"] == "none":
        return "결론 도출. 정책이 바뀌거나 새 자료가 나오면 같은 설계로 재판정 가능"
    if rj["kind"] == "date":
        return f"자료가 더 쌓이는 {rj['date'].replace('-', '. ')}. 이후 같은 설계로 재판정 예정"
    t = "기간 연장으로는 해소되지 않아 자료 확보 후 재판정 필요"
    if needed:
        t += f"(필요 자료: {needed[0]}" + (f" 외 {len(needed) - 1}건)" if len(needed) > 1 else ")")
    return t


def _src(v) -> str:
    """자료 출처(파일 이름은 쓰지 않는다)."""
    if v.synthetic:
        return "합성데이터(정답을 아는 시연용)"
    return "제주데이터허브"


def source_line(v) -> str:
    """출처 줄: 자료 이름·제공처·분석 기간·분석일·분석 번호."""
    main = v.analysis.get("main")
    win = f", 분석 기간 {main['window'][0]} ~ {main['window'][1]}" if main and main.get("window") else ""
    return (f"※ 출처: {_src(v)} 「{v.dataset_title}」" + win
            + f" / 정책 판정관 분석 번호 {v.analysis_no}, 분석일 {v.analysis_day}")


def memo(v) -> list[str]:
    """결재 보고서에 붙일 개조식 요약(□ 제목·결론, ○ 요지·근거·단순 비교와 차이·향후 조치, ※ 출처). 명사형으로 끝낸다."""
    from jverdict.report.brief import brief
    a = v.analysis
    main, vd = a.get("main"), a["verdict"]
    b = brief(v)
    tag = f"({b['tag']})" if b["tag"] and v.level == "판정 보류" else ""
    L = [f"□ 「{v.card.get('policy', '')}」 효과 분석 결과: {v.label}{tag}"]
    if b["say"]:
        L.append(f"  ○ (요지) {b['say']}")
    n = len(vd.get("checks", []))
    passed = sum(1 for c in vd.get("checks", []) if c["status"] == "통과")
    failed = [c["name"] for c in vd.get("checks", []) if c["status"] == "실패"]
    chk = f"검증 {n}개 중 {passed}개 통과" + (f", 미통과: {', '.join(failed)}" if failed else "")
    if main and v.level != "판정 보류":
        lo, hi = vd["ci_used"]
        L.append(f"  ○ (근거) 대조 집단 대비 추정 변화 {_pct(main['pct'])}(판정 구간 {_pct(math.expm1(lo))} ~ "
                 f"{_pct(math.expm1(hi))}), {chk}")
    else:
        L.append(f"  ○ (근거) {chk}")
    dg = (v.skeptic or {}).get("downgraded")
    if dg:
        L.append(f"  ○ (반론) 판정 규칙 결과는 '{dg[0]}'이나, 같은 시기 다른 사건 반론이 검증되어 '{dg[1]}'로 낮춤")
    if v.naive:
        from jverdict.agent.cardbuilder import _josa
        r = {"역효과": "정책 실패(역효과)"}.get(v.naive.get("naive_level"), v.naive.get("naive_level") or "")
        same = (r == v.level)
        L.append(f"  ○ (단순 비교와 차이) 단순 전후 비교({_pct(v.naive['pct'])})로는 '{r}'{_josa(r, '으로', '로')} 보고될 수 있었음"
                 + (" (판정관 결론과 같은 방향)" if same else ""))
    L.append(f"  ○ (향후 조치) {_next_noun(v)}")
    L.append(f"  {source_line(v)}")
    return L


def needed_for_request(v) -> list[str]:
    """요청할 자료. 측정 단절·이상 시점이 원인이면 '사후 기간을 늘린 자료'는 답이 아니므로 뺀다(안내끼리 모순 방지)."""
    needed = [x.rstrip(".") for x in (v.report.get("needed_data") or [])]
    a = v.analysis
    c0 = any(c["code"] == "C0" and c["status"] == "실패" for c in a["verdict"].get("checks", []))
    if c0 or (a.get("power") or {}).get("pre_outliers"):
        needed = [x for x in needed if not any(k in x for k in ("사후 기간", "사후 약", "사후 관측"))]
    return needed


def data_request(v) -> str | None:
    """자료 요청 공문 초안(행정 공문 서식). 결론이 난 판정, 합성데이터 시연 사례면 None. 빈칸(○)은 사람이 채운다."""
    if v.level in ("효과 있음", "효과 없음") or v.synthetic:
        return None
    needed = needed_for_request(v) or ["더 세밀한 단위의 집계 자료(일 단위 또는 전체 집계)"]
    card, man, a = v.card, v.manifest, v.analysis
    main = a.get("main")
    win = f"{_dot(main['window'][0])} ~ {_dot(main['window'][1])}" if main else "-"
    items = "\n".join(f"      {i + 1}) {x}" for i, x in enumerate(needed[:7]))
    pol = card.get("policy", "")
    return f"""수신  ○○○○○장(「{v.dataset_title}」 제공 부서장)
(경유)
제목  「{pol}」 효과 분석에 필요한 자료 제공 요청

1. 관련: ○○과-○○○○(20  .  .  .)
2. 우리 과에서는 「{pol}」의 효과를 공개 자료로 분석하였으나, 현재 자료로는
   효과 여부를 확정하기 어려워 아래와 같이 자료 제공을 요청하오니 협조하여 주시기 바랍니다.
   가. 분석에 사용한 자료: {_src(v)} 「{v.dataset_title}」, 분석 기간 {win}
   나. 요청 자료
{items}
   다. 형식: 개인을 알아볼 수 없는 집계 자료(일 또는 월 단위, 집단별 합계)
       ※ 집계 기준이 바뀐 적이 있으면 바뀐 날짜와 내용을 함께 알려 주시기 바랍니다.
   라. 활용: 미리 정해 기록한 분석 방법으로 재분석하고 결과를 공유
   마. 회신 기한: 20  .  .  .

붙임  효과 분석 결과(분석 번호 {v.analysis_no}) 1부.  끝.
"""
