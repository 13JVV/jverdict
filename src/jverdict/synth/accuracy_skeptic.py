"""반론(Skeptic) 단계 성능 시험: 같은 시기 다른 사건이 있을 때 판정을 제대로 낮추고, 없을 때 낮추지 않는가.

정확도 시험(accuracy_causal·accuracy_plasmode)은 판정 규칙만 잰다. 이 시험은 그 뒤의 반론 단계를 잰다.
오프라인(규칙) 반론 단계와 같은 동작을 같은 함수로 재현한다: 사건 달력에 있는 사건 기간을 빼고 다시 추정
(tools.exclusion_refit) → 반론 성립 여부(tools.objection_status) → 판정 하향(roles.apply_downgrade).
Claude 반론 단계는 비용 때문에 이 시험에 넣지 않는다(사례 F 1건의 기록만 있음).

    사전 등록: docs/prereg/plasmode_skeptic_accuracy.md (실행 전에 잠금)
    .venv/bin/python -m jverdict.synth.accuracy_skeptic
결과: outputs/accuracy/skeptic_accuracy.json, .md, _runs.csv

상황(시드마다 정책일·행사 시작(정책 7~28일 뒤)·길이(42~84일)·크기를 무작위로)
- K1 정책 효과 0 + 같은 시기 청소년 행사(로그 0.25~0.45), 달력에 있음    → 정답: '효과 있음'이 아니어야 함
- K2 실제 효과(로그 0.15~0.30) + 달력에만 있고 이용에 영향 없는 행사     → 정답: 낮추지 않아야 함
- K3 실제 효과(로그 0.15~0.30) + 같은 시기 청소년 행사, 달력에 있음       → 정답: 방향은 유지('효과 있음' 또는 '조건부')
- K4 정책 효과 0 + 같은 시기 청소년 행사, 달력에 없음                     → 반론 단계가 알 수 없는 경우(한계 확인)
- K5 실제 효과, 행사 없음                                               → 기준선(반론 단계가 할 일 없음)
비교: 11-09 이전 반론 규칙(크기가 유지돼도 유의성이 사라지면 '성립')으로 같은 데이터를 판정한 값.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from jverdict.config import OUTPUT_DIR

OUT = OUTPUT_DIR / "accuracy"
KS = {"K1": "정책 효과 0 + 같은 시기 행사(달력에 있음)",
      "K2": "실제 효과 + 달력에만 있는 무관한 행사",
      "K3": "실제 효과 + 같은 시기 행사(달력에 있음)",
      "K4": "정책 효과 0 + 같은 시기 행사(달력에 없음)",
      "K5": "실제 효과, 행사 없음"}
HAS_EFFECT = {"K1": False, "K2": True, "K3": True, "K4": False, "K5": True}


def old_status(main, a) -> str:
    """11-09 이전 규칙: 주분석이 유의하면, 같은 방향·유의·크기 절반 이상일 때만 '기각', 아니면 '성립'."""
    same = np.sign(a.effect) == np.sign(main.effect)
    if not main.significant:
        return "반론 성립" if a.significant else "해당 없음"
    return "반론 기각" if (same and a.significant and abs(a.effect) >= 0.5 * abs(main.effect)) else "반론 성립"


def proposed(level: str, direction: str | None, main, a) -> tuple[str, str, bool]:
    """[사후 추가 · 개정안] 방향을 말한 판정(효과 있음·조건부)이면 주분석의 유의성과 상관없이 '크기가 절반 넘게 줄거나
    방향이 바뀌면 성립'으로 보고, 성립하면 조건부도 판정 보류로 낮춘다. 공식 결과는 현재 규칙(new)이다."""
    if level in ("효과 있음", "조건부") and direction:
        same = np.sign(a.effect) == np.sign(main.effect)
        standing = not (same and abs(a.effect) >= 0.5 * abs(main.effect))
    else:
        from jverdict.agent.tools import objection_status
        standing = objection_status("exclude_period", main, a) == "반론 성립"
    if not standing:
        lb = f"{level}({direction})" if level in ("효과 있음", "조건부") and direction else level
        return level, lb, False
    if level == "효과 있음" and a.significant:
        return "조건부", f"조건부({direction})", True
    if level in ("효과 있음", "조건부", "효과 없음"):
        return "판정 보류", "판정 보류", True
    return level, level, False


def run_one(args: tuple[str, int]) -> dict:
    k, seed = args
    warnings.filterwarnings("ignore")
    from jverdict.agent.roles import apply_downgrade
    from jverdict.agent.tools import exclusion_refit, objection_status
    from jverdict.causal import analyze_series
    from jverdict.quality._panel import to_wide
    from jverdict.quality.coverage import check_coverage, exclusion_mask
    from jverdict.synth.accuracy_causal import series_design
    from jverdict.synth.generator import Effect, PartialDays, SynthSpec, generate
    from jverdict.synth.scenarios import BUS_GROUPS, _policy_date
    t0 = time.time()
    rng = np.random.default_rng(seed)
    date = _policy_date(rng, "2023-03-01", "2025-03-01")
    ev_s = pd.Timestamp(date) + pd.Timedelta(days=int(rng.integers(7, 29)))
    ev_e = ev_s + pd.Timedelta(days=int(rng.integers(42, 85)))
    ev_size = float(rng.uniform(0.25, 0.45))
    eff = float(rng.uniform(0.15, 0.30)) if HAS_EFFECT[k] else 0.0
    inj = [PartialDays(6, share_high=0.25)]
    if eff:
        inj.append(Effect("청소년", date, eff))
    if k in ("K1", "K3", "K4"):
        inj.append(Effect("청소년", str(ev_s.date()), ev_size, end=str(ev_e.date())))
    sd = generate(SynthSpec("2021-01-01", "2025-10-31", BUS_GROUPS, "D", injections=inj), seed)
    cov = check_coverage(sd.df, "time", "group", "value", freq="D")
    w = to_wide(sd.df, "time", "group", "value", "D")
    mask = exclusion_mask(cov, w.index)
    a = analyze_series(w, series_design(date), mask=mask)
    v, m = a.verdict, a.main
    row = {"k": k, "seed": seed, "date": date, "event_start": str(ev_s.date()), "event_end": str(ev_e.date()),
           "event_size": ev_size if k in ("K1", "K3", "K4") else 0.0, "true_effect": eff, "has_effect": HAS_EFFECT[k],
           "rule_level": v.level, "rule_verdict": v.label, "est": m.effect if m is not None else None,
           "sig": bool(m.significant) if m is not None else None}
    in_calendar = k in ("K1", "K2", "K3")
    final = {"new": (v.level, v.label, None), "old": (v.level, v.label, None), "prop": (v.level, v.label, None)}
    if in_calendar and m is not None:
        x = exclusion_refit(w, "청소년", "일반", "D", mask, date, str(ev_s.date()), str(ev_e.date()))
        if x is not None:
            row.update(ex_est=x.effect, ex_sig=bool(x.significant))
            for rule, st_ in (("new", objection_status("exclude_period", m, x)), ("old", old_status(m, x))):
                row[f"{rule}_status"] = st_
                res = [{"significant": bool(x.significant), "status": st_}] if st_ == "반론 성립" else []
                final[rule] = apply_downgrade(v.level, v.direction, res, st_ == "반론 성립")
            pl, pb, pd_ = proposed(v.level, v.direction, m, x)
            final["prop"] = (pl, pb, pd_ or None)
    for rule, (lv, lb, dg) in final.items():
        row[f"{rule}_level"], row[f"{rule}_verdict"], row[f"{rule}_downgraded"] = lv, lb, dg is not None
    row["direction"] = v.direction
    row["seconds"] = round(time.time() - t0, 2)
    return row


def summarize(df: pd.DataFrame) -> list[dict]:
    out = []
    for k, g in df.groupby("k", sort=True):
        inc = g.direction.eq("증가")
        r = {"k": k, "title": KS[k], "runs": int(len(g)), "has_effect": HAS_EFFECT[k],
             "rule_effect_yes": float(g.rule_level.eq("효과 있음").mean()),
             "rule_verdicts": dict(Counter(g.rule_verdict).most_common())}
        for rule in ("new", "old", "prop"):
            lv = g[f"{rule}_level"]
            r[rule] = {"effect_yes": float(lv.eq("효과 있음").mean()),
                       "effect_or_partial_inc": float((lv.isin(["효과 있음", "조건부"]) & inc).mean()),
                       "downgraded": float(g[f"{rule}_downgraded"].mean()),
                       "hold": float(lv.eq("판정 보류").mean()),
                       "verdicts": dict(Counter(g[f"{rule}_verdict"]).most_common())}
            if f"{rule}_status" in g:
                r[rule]["status"] = dict(Counter(g[f"{rule}_status"].dropna()).most_common())
        out.append(r)
    return out


def headline(summary: list[dict]) -> dict:
    s = {r["k"]: r for r in summary}
    h = {}
    for rule in ("new", "old", "prop"):
        h[rule] = {
            # 행사 때문에 생긴 변화를 '효과 있음'으로 남긴 비율(낮을수록 좋음): 달력에 있는 경우
            "event_false_effect": s["K1"][rule]["effect_yes"] if "K1" in s else None,
            # 실제 효과인데 무관한 행사 때문에 낮춘 비율(낮을수록 좋음)
            "wrong_downgrade": s["K2"][rule]["downgraded"] if "K2" in s else None,
            # 실제 효과 + 행사: 방향을 유지한 비율(높을수록 좋음)
            "kept_direction_with_event": s["K3"][rule]["effect_or_partial_inc"] if "K3" in s else None,
        }
    h["rule_only_event_false_effect"] = s["K1"]["rule_effect_yes"] if "K1" in s else None
    h["not_in_calendar_false_effect"] = s["K4"]["new"]["effect_yes"] if "K4" in s else None
    return h


def to_markdown(res: dict) -> str:
    def pc(x):
        return "-" if x is None else f"{x * 100:.1f}%"
    h, n, o, pp = res["headline"], res["headline"]["new"], res["headline"]["old"], res["headline"]["prop"]
    L = ["# 반론(Skeptic) 단계 성능 시험", "",
         f"생성 {res['generated']} · 판정 규칙 {res['rules_version']} · 상황당 {res['runs_per_k']}회 · 소요 {res['seconds']:.0f}초 · "
         f"사전 등록 `{res['prereg']['file']}` (지문 `{res['prereg']['sha256'][:16]}…`, 잠금 {res['prereg']['locked_at']})", "",
         "합성 버스 데이터에 같은 시기 청소년 행사를 심고, 오프라인 반론 단계와 같은 함수로 '행사 기간을 빼고 다시 추정 → "
         "반론 성립 여부 → 판정 하향'을 거쳤습니다. Claude 반론 단계는 비용 때문에 넣지 않았습니다.", "",
         "## 요약", "",
         "| 지표 | 판정 규칙만 | 반론 단계 뒤(현재 규칙) | 반론 단계 뒤(11-09 이전 규칙) | 개정안(사후 추가) |", "|---|---|---|---|---|",
         f"| 정책 효과 0 + 같은 시기 행사(달력에 있음): '효과 있음' 비율 (낮을수록 좋음) | {pc(h['rule_only_event_false_effect'])} | "
         f"{pc(n['event_false_effect'])} | {pc(o['event_false_effect'])} | {pc(pp['event_false_effect'])} |",
         f"| 실제 효과 + 무관한 행사: 잘못 낮춘 비율 (낮을수록 좋음) | 0.0% | {pc(n['wrong_downgrade'])} | {pc(o['wrong_downgrade'])} | {pc(pp['wrong_downgrade'])} |",
         f"| 실제 효과 + 같은 시기 행사: 증가 방향을 유지한 비율 (높을수록 좋음) | - | {pc(n['kept_direction_with_event'])} | "
         f"{pc(o['kept_direction_with_event'])} | {pc(pp['kept_direction_with_event'])} |",
         f"| 정책 효과 0 + 행사가 달력에 없음: '효과 있음' 비율(한계) | - | {pc(h['not_in_calendar_false_effect'])} | - | - |",
         f"| 정책 효과 0 + 같은 시기 행사(달력에 있음): 증가 방향을 주장한 비율('효과 있음'·'조건부') | - | "
         f"{pc(res['k1_claim']['new'])} | {pc(res['k1_claim']['old'])} | {pc(res['k1_claim']['prop'])} |", "",
         "개정안(사후 추가): 실행 뒤 K1에서 '조건부(증가)'가 남는 것을 보고 추가한 비교 열입니다. 방향을 말한 판정이면 "
         "주분석의 유의성과 상관없이 '크기가 절반 넘게 줄거나 방향이 바뀌면 반론 성립'으로 보고, 성립하면 '조건부'도 '판정 보류'로 낮춥니다. "
         "공식 결과는 '현재 규칙' 열입니다(사전 등록 5장 변경 기록).", "",
         "## 상황별 판정 분포", "", "| 상황 | 판정 규칙만 | 반론 단계 뒤(현재 규칙) | 반론 결과(현재 규칙) |", "|---|---|---|---|"]
    for r in res["summary"]:
        top = lambda d: ", ".join(f"{k} {v}" for k, v in list(d.items())[:4])  # noqa: E731
        L.append(f"| {r['k']} {r['title']} | {top(r['rule_verdicts'])} | {top(r['new']['verdicts'])} | "
                 f"{top(r['new'].get('status', {})) or '-'} |")
    L += ["", "## 읽는 법과 한계", "",
          "- 반론 단계는 사건 달력에 있는 사건만 검증합니다. 달력에 없는 사건(K4)은 판정 규칙의 결과가 그대로 남습니다. "
          "그래서 판정서마다 '이 분석이 말할 수 없는 것'에 같은 시기 다른 요인을 적고, 사건 달력에 출처와 함께 사건을 쌓습니다.",
          "- 행사 기간을 빼면 관측이 줄어 유의성이 사라질 수 있습니다. 현재 규칙은 이것만으로는 반론이 맞았다고 보지 않습니다"
          "('판단 불가'). 이전 규칙과의 차이가 표의 오른쪽 두 열입니다.",
          "- 합성데이터 시험이라 실데이터의 함정은 들어 있지 않습니다. 행사 크기(로그 0.25~0.45)와 길이(6~12주)는 사례 F와 비슷하게 잡았습니다.", ""]
    return "\n".join(L)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=20261109)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true", help="상황마다 1건만(결과는 출력하지 않음)")
    a = ap.parse_args(argv)
    from jverdict.causal.verdict import RULES_VERSION
    t0 = time.time()
    runs = 1 if a.smoke else a.runs
    jobs = [(k, a.seed0 + 10_000 * i + j) for i, k in enumerate(KS) for j in range(runs)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_one, jobs, chunksize=4))
    if a.smoke:
        return {"jobs": len(rows), "seconds": [r["seconds"] for r in rows]}
    df = pd.DataFrame(rows)
    lock = json.loads((OUTPUT_DIR.parent / "docs" / "prereg" / "plasmode_skeptic_accuracy.lock.json").read_text(encoding="utf-8"))
    summ = summarize(df)
    k1 = df[df.k == "K1"]
    k1_claim = {r: float((k1[f"{r}_level"].isin(["효과 있음", "조건부"]) & k1.direction.eq("증가")).mean())
                for r in ("new", "old", "prop")}
    res = {"k1_claim": k1_claim, "generated": time.strftime("%Y-%m-%d %H:%M"), "rules_version": RULES_VERSION, "runs_per_k": runs,
           "seconds": time.time() - t0, "platform": f"{platform.system()} {platform.machine()}, 작업자 {a.workers}",
           "prereg": {k: lock[k] for k in ("file", "sha256", "locked_at")}, "summary": summ, "headline": headline(summ)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "skeptic_accuracy.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "skeptic_accuracy.md").write_text(to_markdown(res), encoding="utf-8")
    df.to_csv(OUT / "skeptic_accuracy_runs.csv", index=False, encoding="utf-8-sig")
    return res


if __name__ == "__main__":
    r = main()
    print(to_markdown(r) if "summary" in r else r)
