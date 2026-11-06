"""판정 뒤 '다음 할 일': 결재용 3줄 요약, 재판정 예정, 자료 요청 공문 초안.

보류도 쓸모 있게 만든다. 모든 숫자는 분석 결과(analysis.json)와 판정서에서 그대로 옮긴다(새 숫자를 만들지 않음).
공문 초안은 수신처·기한·작성자 칸을 비워 둔다(사람이 채움).
"""
from __future__ import annotations

import math

import pandas as pd

REJUDGE_MAX = {"개월": 36, "일": 3 * 365}          # 이보다 오래 기다려야 하면 기간이 아니라 자료로 풀어야 한다


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def rejudge(v) -> dict:
    """재판정 예정: {kind: date|data|none, text}. 기간을 늘리면 풀리는 경우에만 날짜를 준다."""
    a, card = v.analysis, v.card
    pw, main = a.get("power") or {}, a.get("main")
    level = v.level
    if level in ("효과 있음", "효과 없음"):
        return {"kind": "none", "text": "결론이 났습니다. 정책을 바꾸거나 새 자료가 나오면 같은 설계로 다시 판정할 수 있습니다."}
    if a["verdict"].get("checks") and any(c["code"] == "C0" and c["status"] == "실패" for c in a["verdict"]["checks"]):
        return {"kind": "data", "text": "데이터 측정 방식부터 확인해야 합니다. 기간을 늘려도 풀리지 않습니다. 필요한 자료를 확보한 뒤 다시 판정하세요."}
    if main is None:
        return {"kind": "data", "text": "분석할 관측이 부족합니다. 필요한 자료를 확보한 뒤 다시 판정하세요."}
    if pw.get("pre_outliers"):
        t = ", ".join(o["time"] for o in pw["pre_outliers"])
        return {"kind": "data", "text": f"정책 전 기준 기간의 이상 시점({t}) 원인을 먼저 확인해야 합니다. 원인이 확인되면 그 시점을 뺀 설계를 새로 등록해 다시 판정하세요."}
    need, n_post, unit = pw.get("post_needed"), pw.get("n_post") or 0, pw.get("unit") or "개월"
    if not need:
        return {"kind": "data", "text": "검증 실패 항목을 해결할 자료나 설계가 필요합니다('판정에 필요한 데이터' 참고)."}
    extra = max(0, int(need) - int(n_post))
    if card.get("end_date"):
        return {"kind": "data", "text": f"끝난 정책이라 기다려도 관측이 늘지 않습니다. 더 세밀한 자료(일 단위·전체 카드사 등)가 있어야 다시 판정할 수 있습니다."}
    if extra > REJUDGE_MAX.get(unit, 36):
        return {"kind": "data", "text": f"기준 효과를 가리려면 사후 관측이 약 {need}{unit} 필요해, 기다려서 풀기에는 깁니다(3년 넘음). 더 세밀한 자료가 필요합니다."}
    end = pd.Timestamp(main["window"][1] + ("-01" if len(main["window"][1]) == 7 else ""))
    when = end + (pd.DateOffset(months=extra) if unit == "개월" else pd.Timedelta(days=extra))
    if when <= pd.Timestamp.now().normalize():
        return {"kind": "data", "text": "그만큼의 사후 기간은 이미 지났지만, 이 설계의 사후 기간은 정책 기간에 맞춰 정해져 있어 늘릴 수 "
                                        "없습니다. 더 세밀한 자료(일 단위·전체 카드사 등)가 있어야 다시 판정할 수 있습니다."}
    return {"kind": "date", "date": when.strftime("%Y-%m"),
            "text": f"정책이 계속되면 자료가 약 {extra}{unit} 더 쌓인 {when.strftime('%Y년 %m월')} 이후 같은 설계로 다시 판정하세요"
                    f"(필요한 사후 관측 약 {need}{unit}, 현재 {n_post}{unit}, 근사)."}


def memo(v) -> list[str]:
    """결재 보고서에 붙일 3줄."""
    a, rep = v.analysis, v.report
    main, vd = a.get("main"), a["verdict"]
    L1 = f"① 판정: {v.label}. {v.meaning}"
    if main:
        lo, hi = vd["ci_used"]
        passed = sum(1 for c in vd["checks"] if c["status"] == "통과")
        L2 = (f"② 근거: 추정 변화 {_pct(main['pct'])}(판정 구간 {_pct(math.expm1(lo))} ~ {_pct(math.expm1(hi))}), "
              f"검증 {len(vd['checks'])}개 중 {passed}개 통과")
        if v.naive:
            L2 += f". 단순 전후 비교({_pct(v.naive['pct'])})와 구분해 판단"
    else:
        L2 = f"② 근거: {vd['reasons'][0] if vd.get('reasons') else rep.get('headline', '')}"
    rj = rejudge(v)
    needed = rep.get("needed_data") or []
    nxt = rj["text"]
    if rj["kind"] == "data" and needed:
        nxt += f" 필요 자료: {needed[0]}" + (f" 외 {len(needed) - 1}건" if len(needed) > 1 else "")
    L3 = f"③ 다음 조치: {nxt}"
    return [L1, L2, L3]


def data_request(v) -> str | None:
    """자료 요청 공문 초안. 필요한 자료가 없으면 None."""
    needed = list(v.report.get("needed_data") or [])
    if not needed or v.level in ("효과 있음", "효과 없음"):
        return None
    card, man, a = v.card, v.manifest, v.analysis
    main = a.get("main")
    win = f"{main['window'][0]} ~ {main['window'][1]}" if main else "-"
    items = "\n".join(f"   {'가나다라마바사'[i]}. {x}" for i, x in enumerate(needed[:7]))
    return f"""수신: (데이터 제공 기관·부서)
제목: 정책 효과 판정에 필요한 자료 요청({card.get('policy', '')})

1. 목적
   "{card.get('question', '')}"에 답하기 위해 정책 판정관(J-Verdict)으로 분석했습니다.
   현재 자료로는 결론을 낼 수 없어(판정: {v.label}) 아래 자료를 요청합니다.

2. 지금까지 쓴 자료
   {v.dataset_title} ({man['data'].get('source', '')}), 분석 기간 {win}

3. 요청 자료
{items}

4. 형식
   개인을 알아볼 수 없는 집계 자료(일 또는 월 단위, 집단별 합계)면 됩니다. 원자료 행은 필요하지 않습니다.
   집계 기준이 바뀐 적이 있으면 바뀐 날짜와 내용을 함께 알려 주십시오.

5. 활용
   결과를 보기 전에 잠가 둔 설계(사전 등록 지문 {man['prereg']['sha256'][:16]}…)로 다시 판정하고, 판정서를 공유하겠습니다.

6. 회신 기한: (작성자 기입)

붙임: 판정서 1부(실행 번호 {v.run_id}).  끝.
"""
