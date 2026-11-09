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


def _p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _first_sentence(t: str) -> str:
    t = (t or "").strip()
    m = re.match(r"(.+?\.)(\s|$)", t)
    return m.group(1) if m else t


def _fix_josa(t: str) -> str:
    """엔진 문구의 "'청소년'가" 같은 조사를 받침에 맞게(엔진 문구는 재실행 지문 때문에 그대로 둔다)."""
    from jverdict.agent.cardbuilder import _josa
    return re.sub(r"'([^']+)'(가|이) ", lambda m: f"'{m.group(1)}'{_josa(m.group(1), '이', '가')} ", t or "")


# 보류 이유의 이름. 문서(기획서·결과 설명자료)와 같은 말을 쓴다
TAGS = {"skeptic": "같은 시기 다른 사건", "quality": "측정 단절 의심", "outlier": "기준 기간 이상",
        "pretrend": "정책 전 추세와 구분 불가", "power": "검정력 부족", "effect": "", "null": "", "partial": "검증 일부 실패",
        "other": ""}


def say(v, cause: str) -> str:
    """'지금 말할 수 있는 것' 한 줄: 보류여도 판정 구간·진단에서 바로 나오는 말(새 숫자를 만들지 않음)."""
    a = v.analysis
    m, vd = a.get("main") or {}, a["verdict"]
    nv = getattr(v, "naive", None) or {}
    npct = f"{nv['pct'] * 100:+.0f}%" if nv.get("pct") is not None else "단순 비교의 변화"
    if cause == "skeptic":
        return f"정책 이후의 {_pct(m.get('pct', 0))}는 같은 시기 다른 사건으로 설명되어 정책 효과라고 말할 수 없음"
    if cause == "quality":
        return f"공개 자료의 {npct} 변화는 집계 방식 변경 의심 때문에 정책 효과(또는 실패)의 근거로 쓸 수 없음"
    if cause == "outlier":
        return "기준 기간이 흔들려 효과 크기는 말할 수 없음. 이상 시점을 확인하기 전에는 효과 크기를 인용하지 말 것"
    if cause == "pretrend":
        return f"정책 이후의 {_pct(m.get('pct', 0))}는 정책 전부터 이어진 추세와 구분되지 않아 정책 효과라고 말할 수 없음"
    if not m or not vd.get("ci_used"):
        return ""
    lo, hi = (math.expm1(x) for x in vd["ci_used"])
    if v.level == "효과 있음":
        return f"정책 이후 {_pct(m.get('pct', 0))}(구간 {_pct(lo)} ~ {_pct(hi)})의 변화는 대조·계절 보정과 검증을 거친 값으로 인용할 수 있음"
    if v.level == "조건부":
        return (f"{'증가' if m.get('pct', 0) > 0 else '감소'} 방향은 보이지만 검증 일부를 통과하지 못함. "
                "성과 근거로 단독 인용하지 말 것(합성 시험에서 '조건부'의 적중률은 약 50%)")
    if v.level in ("판정 보류", "효과 없음") and lo < 0 < hi:
        return f"효과가 있었더라도 {_pct(lo)} ~ {_pct(hi)} 사이: 이보다 큰 증가·감소는 자료상 확인되지 않음"
    return ""


def speech(v, cause: str) -> dict:
    """대외적으로 '말해도 되는 문장'과 '말하면 안 되는 문장'(보고·답변 작성용). 새 숫자는 만들지 않는다."""
    a = v.analysis
    m, pw = a.get("main") or {}, a.get("power") or {}
    nv = getattr(v, "naive", None) or {}
    pol = v.card.get("policy", "이 정책")
    npct = f"{nv['pct'] * 100:+.0f}%" if nv.get("pct") is not None else None
    ok, no = [], []
    if v.level == "효과 있음":
        ok.append(f"「{pol}」 이후 대상 집단의 지표가 대조 집단보다 {_pct(m.get('pct', 0))} 변했고, 검증을 모두 통과했다.")
        no.append("이 변화가 전부 정책 하나 때문이라고 단정하는 말(같은 시기 다른 요인은 사건 달력에 있는 것만 확인함).")
    elif v.level == "조건부":
        ok.append(f"「{pol}」 이후 {'증가' if m.get('pct', 0) > 0 else '감소'} 방향의 신호가 있으나 검증 일부를 통과하지 못했다.")
        no.append("\"정책 효과가 확인되었다\"(조건부는 성과 근거로 단독 인용하지 않음).")
    elif v.level == "효과 없음":
        ok.append(f"「{pol}」 이후 의미 있다고 본 크기 이상의 변화는 확인되지 않았다.")
        no.append("\"정책이 실패했다\"(작은 효과는 이 자료로 가려낼 수 없음).")
    else:
        ok.append(f"공개 자료만으로는 「{pol}」의 효과를 확정할 수 없어 판단을 보류했다.")
        if cause == "quality":
            ok.append("정책 시점에 공개 자료의 집계 방식이 바뀐 것으로 의심되어 제공처에 확인을 요청했다.")
            if npct:
                no.append(f"\"이용(소비)이 {npct} 변했다\" 또는 \"정책이 실패했다\"(집계 변경 의심 숫자).")
        elif cause == "pretrend":
            ok.append("정책 전부터 이어진 추세와 정책 효과를 가를 수 있는 자료가 부족하다.")
            if npct:
                no.append(f"\"정책으로 {npct} 늘었다\"(정책 전부터의 추세가 섞인 숫자).")
        elif cause == "outlier":
            ok.append("비교 기준 기간에 크게 벗어난 시점이 있어 효과 크기를 계산할 수 없었다.")
            no.append(f"\"{_pct(m.get('pct', 0))} 늘었다\" 같은 효과 크기(기준선이 흔들린 참고 계산).")
        elif cause == "skeptic":
            ok.append("같은 시기의 다른 사건이 변화를 설명할 수 있어 정책 효과를 확정하지 않았다.")
            no.append(f"\"정책으로 {_pct(m.get('pct', 0))} 늘었다\"(같은 시기 사건을 빼면 사라지는 변화).")
        elif cause == "power":
            from jverdict.causal.power import mde_text
            if pw.get("mde_log") is not None:
                ok.append(f"이 자료로 구분할 수 있는 최소 효과는 {mde_text(pw['mde_log'])}이며, 그보다 작은 효과는 판단할 수 없다.")
            no.append("\"효과가 없었다\"(효과가 없다는 증거가 아니라 자료가 부족하다는 뜻).")
        if npct and cause not in ("quality", "pretrend"):
            no.append(f"단순 전후 비교 {npct}를 정책 효과로 인용하는 말.")
    return {"ok": [_fix_josa(x) for x in ok], "no": [_fix_josa(x) for x in no]}


def foot(v, cause: str) -> str:
    """'단순 비교 → 판정관' 대비를 사례에 맞게 한 문장으로."""
    nv = getattr(v, "naive", None) or {}
    if not nv:
        return ""
    reading = {"역효과": "정책 실패(역효과)"}.get(nv.get("naive_level"), nv.get("naive_level") or "")
    from jverdict.agent.cardbuilder import _josa
    head = f"단순 비교라면 '{reading}'{_josa(reading, '으로', '로')} 보고했을 숫자입니다. "
    why = {"skeptic": "판정관은 같은 시기 다른 사건을 빼고 다시 계산하자 변화가 사라져 결론을 미뤘습니다.",
           "quality": "판정관은 같은 숫자가 집계 방식 변경 때문일 수 있어 결론을 미뤘습니다.",
           "outlier": "판정관은 비교 기준 기간에 크게 벗어난 달이 있어 숫자를 그대로 믿지 않았습니다.",
           "pretrend": "판정관은 정책 전부터 이어진 추세를 빼면 같은 결론이 나오지 않아 결론을 미뤘습니다.",
           "power": "판정관은 이 자료로는 의미 있는 크기의 효과를 가려낼 수 없어 결론을 미뤘습니다.",
           "partial": "판정관은 검증 일부를 통과하지 못해 방향만 인정했습니다.",
           "null": "판정관은 대조·계절 보정 뒤 의미 있는 크기의 변화가 없다고 봤습니다.",
           "effect": "판정관은 대조·계절 보정과 검증을 모두 거친 뒤에도 변화가 남아 효과를 인정했습니다."}
    return head + why.get(cause, "판정관은 대조·보정·검증을 거쳐 다르게 판단했습니다.")


def brief(v) -> dict:
    """{title, cause, tag, say, foot, points: [(라벨, 내용)]}"""
    from jverdict.report import actions as AC
    a = v.analysis
    vd, m, pw = a["verdict"], a.get("main") or {}, a.get("power") or {}
    checks = {c["code"]: c for c in vd.get("checks", [])}
    failed = [c for c in vd.get("checks", []) if c["status"] == "실패"]
    level = v.level

    def st(code):
        return (checks.get(code) or {}).get("status")

    cause, title, why = "other", "", ""
    dg = v.skeptic.get("downgraded")
    stand = [h for h in v.skeptic.get("hypotheses", []) if h.get("status") == "반론 성립"]
    if dg and stand:
        cause = "skeptic"
        h = stand[0]
        title = ("같은 시기 다른 사건이 변화를 설명해 판정을 미룹니다" if level == "판정 보류"
                 else "같은 시기 다른 사건이 변화의 일부를 설명해 판정을 낮췄습니다")
        r = next((x for x in h.get("results", []) if x.get("status") == "반론 성립"), None)
        why = h["hypothesis"].replace("같은 시기의 다른 사건 때문이다: ", "같은 시기 사건: ").rstrip(". ")
        if r:
            why += (f". 그 기간을 빼고 다시 추정하니 {_pct(m.get('pct', 0))} → {_pct(r['pct'])}"
                    + ("(유의하지 않음)" if not r.get("significant") else ""))
    elif st("C0") == "실패":
        cause = "quality"
        title = "데이터 집계 방식이 바뀐 것으로 의심되어 판정을 미룹니다"
        why = checks["C0"]["detail"].split(". ", 1)[-1]
    elif level == "판정 보류" and pw.get("pre_outliers"):
        cause = "outlier"
        o = pw["pre_outliers"][0]
        title = "비교 기준 기간에 크게 벗어난 시점이 있어 판정을 미룹니다"
        from jverdict.causal.power import ratio_text
        why = (f"{o['time']}의 대상/대조 격차가 기준 기간 보통 때의 {ratio_text(o['dev'])}{'으로' if ratio_text(o['dev']).endswith('수준') else '로'} "
               f"{'떨어져' if o['dev'] < 0 else '올라'} 기준선과 구간이 함께 흔들림(영업 중단·집계 변경 등 확인 필요)")
    elif level == "판정 보류" and st("C1") == "실패" and "심각" in checks["C1"]["detail"]:
        cause = "pretrend"
        title = "정책 전부터 있던 변화와 구분할 수 없어 판정을 미룹니다"
        ta = a.get("trend_adjusted") or {}
        pm = re.search(r"p=([0-9.e+-]+)", checks["C1"]["detail"])
        why = "정책 전부터 뚜렷한 추세가 있었음" + (f"(사전추세 검정 {_p(float(pm.group(1)))})" if pm else "")
        if pw.get("n_pre") is not None and pw["n_pre"] < 6:
            why += f". 정책 전 시점이 {pw['n_pre']}개뿐이라 추세를 빼고 다시 계산한 값도 믿기 어려움"
        elif ta.get("pct") is not None:
            why += f". 기존 추세를 빼면 {_pct(ta['pct'])}"
    elif level == "판정 보류" and pw.get("mde_pct") is not None and not pw.get("enough", True):
        cause = "power"
        title = "작은 효과를 가려낼 만큼 자료가 충분하지 않아 판정을 미룹니다"
        from jverdict.causal.power import mde_text
        why = (f"이 자료로 구분할 수 있는 최소 효과는 {mde_text(pw['mde_log'])}로, "
               f"의미 있다고 본 {pw['reference_pct'] * 100:.0f}%보다 큼")
    elif level == "효과 있음":
        cause = "effect"
        title = f"정책 이후 {_pct(m.get('pct', 0))}의 변화가 확인되었습니다"
        c2 = checks.get("C2a") or {}
        pt = a.get("placebo_time") or {}
        why = (f"정책이 없던 가짜 시행일 {pt.get('n')}개로 같은 계산을 해 봐도 실제 변화가 더 뚜렷함(순위 {_p(pt['p_two'])})"
               if c2.get("status") == "통과" and pt.get("n")
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
    sy = say(v, cause)
    if sy:
        points.append(("요지", sy))
    if why:
        points.append(("원인" if level != "효과 있음" else "근거", why))
    n = len(vd.get("checks", []))
    if n:
        ok = sum(1 for c in vd["checks"] if c["status"] == "통과")
        txt = f"{n}개 중 {ok}개 통과"
        if failed:
            txt += f" · {len(failed)}개 실패(" + ", ".join(f"{c['code']} {c['name']}" for c in failed) + ")"
        if n - ok - len(failed):
            txt += f" · {n - ok - len(failed)}개 참고·해당 없음"
        points.append(("검증", txt))
    if dg:
        from jverdict.agent.cardbuilder import _josa
        points.append(("반론", f"판정 규칙은 '{dg[0]}'{_josa(dg[0], '이었으나', '였으나')} Skeptic의 반론이 성립해 "
                               f"'{dg[1]}'{_josa(dg[1], '으로', '로')} 낮춤"))
    rj = AC.rejudge(v)
    if rj["kind"] != "none":
        points.append(("다음", rj["text"]))
    return {"title": title, "cause": cause, "tag": TAGS.get(cause, ""), "say": _fix_josa(sy), "foot": foot(v, cause),
            "points": [(k, _fix_josa(t)) for k, t in points], "speech": speech(v, cause)}
