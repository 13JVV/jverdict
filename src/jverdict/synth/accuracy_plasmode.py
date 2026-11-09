"""반합성(plasmode) 정확도 시험: 실데이터에 정답을 심어 판정 규칙을 잰다.

합성데이터 시험(accuracy_causal)은 잡음을 실데이터 크기에 맞췄지만, 실데이터의 이상 시점·구조 변화·코로나19 같은
충격까지 흉내 내지는 못한다. 이 시험은 생성기를 쓰지 않는다. 제주데이터허브 실데이터에서 정책이 없던 날짜를
'가짜 시행일'로 고르고, 그날부터 대상 계열에 정해진 크기의 효과를 곱한 뒤(효과 0이면 그대로) 실제 판정과 같은
엔진·같은 설계로 판정한다. 대상 계열의 실제 잡음·계절·사건이 그대로 들어간다.

    사전 등록: docs/prereg/plasmode_skeptic_accuracy.md (실행 전에 잠금)
    .venv/bin/python -m jverdict.synth.accuracy_plasmode
결과: outputs/accuracy/plasmode_accuracy.json, .md, _runs.csv

계열(설계는 사례 A·B의 등록 설계와 같은 계열·대조를 쓰고, 기간은 엔진 기본값)
- 버스(일): 청소년 대 일반, 공간 위약 경로·장애 일반, 계절 차분 3년. 실데이터는 2025-07-31까지만(사례 A 정책 전).
  가짜 시행일: 2022-01-03 ~ 2025-01-27 사이 2주 간격 월요일.
- 카드(월): 도민 × 지원금 사용 가능 업종 대 사용 불가 업종, 공간 위약 법인, 계절 차분 1년.
  가짜 시행일: 2019-07 ~ 2024-12의 매월 1일 중, 그 3개월 전부터 6개월 뒤까지가 실제 지원금 기간
  (1차 2020-05~08, 상생 2021-09~12)과 겹치지 않는 날.
효과: 로그 0, 0.10, 0.20, 0.35 (0%, +10.5%, +22.1%, +41.9%), 가짜 시행일부터 끝까지 일정한 크기.

한계: 가짜 시행일끼리 분석 창이 겹쳐 서로 독립인 표본이 아니다. '효과 0'은 '알려진 정책이 없던 날'이라는 뜻이며,
그날 대상 집단에 실제로 아무 일도 없었다는 보장은 아니다(그래서 실데이터의 함정이 그대로 들어간다).
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
EFFECTS = [0.0, 0.10, 0.20, 0.35]
RELIEF = [("2020-05-01", "2020-08-31"), ("2021-09-01", "2021-12-31")]
SEED = 20261104

FAMILIES = {
    "bus": {"title": "버스 일 단위(정류소 파일): 청소년 대 일반", "case": "A", "dataset": "bus_station", "freq": "D",
            "target": "청소년", "reference": ["일반"], "placebo": [("경로", ["일반"]), ("장애 일반", ["일반"])],
            "season_years": 3, "cutoff": "2025-07-31"},
    "card": {"title": "카드 월 단위: 도민 × 사용 가능 업종 대 사용 불가 업종", "case": "B", "dataset": "card_sector", "freq": "MS",
             "target": "도민_가능", "reference": ["도민_불가"], "placebo": [("법인_가능", ["법인_불가"])],
             "season_years": 1, "cutoff": "2025-06-01"},
}


def fake_dates(fam: str) -> list[str]:
    if fam == "bus":
        return [str(d.date()) for d in pd.date_range("2022-01-03", "2025-01-27", freq="14D")]
    out = []
    for d in pd.date_range("2019-07-01", "2024-12-01", freq="MS"):
        lo, hi = d - pd.DateOffset(months=3), d + pd.DateOffset(months=6)
        if any(lo <= pd.Timestamp(b) and pd.Timestamp(a) < hi for a, b in RELIEF):
            continue
        out.append(str(d.date()))
    return out


_CACHE: dict = {}


def _frame(fam: str):
    """(넓은 형식 실데이터, 커버리지 제외 마스크). 정책 전 기간(cutoff)까지만."""
    if fam not in _CACHE:
        from jverdict.agent import catalog
        from jverdict.agent import design as DZ
        from jverdict.agent.library import CARDS
        from jverdict.quality.coverage import exclusion_mask
        f = FAMILIES[fam]
        d = CARDS[f["case"]]["design"]
        _, wide = catalog.series_frame(f["dataset"], DZ._series_map(d), SEED)
        wide = wide[wide.index <= pd.Timestamp(f["cutoff"])]
        mask = exclusion_mask(DZ.coverage(f["dataset"], SEED), wide.index)
        _CACHE[fam] = (wide, mask)
    return _CACHE[fam]


def run_one(args: tuple[str, str, float]) -> dict:
    fam, date, eff = args
    warnings.filterwarnings("ignore")
    from jverdict.causal import SeriesDesign, analyze_series
    f = FAMILIES[fam]
    t0 = time.time()
    wide, mask = _frame(fam)
    w = wide.copy()
    if eff:
        w.loc[w.index >= pd.Timestamp(date), f["target"]] *= float(np.exp(eff))
    ds = SeriesDesign("반합성 시험", f["target"], f["reference"], date, f["freq"], expected="increase",
                      reference_effect=0.10, season="sdiff", season_years=f["season_years"], placebo_groups=f["placebo"])
    a = analyze_series(w, ds, mask=mask)
    v, m = a.verdict, a.main
    row = {"family": fam, "date": date, "effect": eff, "level": v.level, "verdict": v.label, "direction": v.direction,
           "quality_cap": v.quality_cap, "seconds": round(time.time() - t0, 2),
           "checks": ";".join(f"{c.code}:{c.status}" for c in v.checks)}
    if m is not None:
        row.update(est=m.effect, sig=bool(m.significant), ci_lo=v.ci_used[0], ci_hi=v.ci_used[1],
                   cover=bool(v.ci_used[0] <= eff <= v.ci_used[1]))
    row["mde_pct"] = a.power.mde_pct if a.power else None
    row["placebo_n"] = a.placebo_time.n if a.placebo_time is not None else 0
    row["placebo_sd"] = float(a.placebo_time.sd) if a.placebo_time is not None and a.placebo_time.available else None
    row["main_se"] = float(m.se) if m is not None else None
    if a.verdict_v3 is not None:
        row["v3_level"], row["v3_verdict"] = a.verdict_v3.level, a.verdict_v3.label
    return row


def summarize(df: pd.DataFrame) -> list[dict]:
    out = []
    for (fam, eff), g in df.groupby(["family", "effect"], sort=True):
        inc = g.direction.eq("증가")
        r = {"family": fam, "title": FAMILIES[fam]["title"], "effect_log": float(eff), "effect_pct": float(np.expm1(eff)),
             "runs": int(len(g)), "verdicts": dict(Counter(g.verdict).most_common()),
             "effect_yes": float((g.level.eq("효과 있음") & inc).mean()),
             "effect_or_partial": float((g.level.isin(["효과 있음", "조건부"]) & inc).mean()),
             "any_effect_claim": float(g.level.eq("효과 있음").mean()),
             "direction_claim": float(g.level.isin(["효과 있음", "조건부"]).mean()),
             "null": float(g.level.eq("효과 없음").mean()), "hold": float(g.level.eq("판정 보류").mean()),
             "wrong_sign": float((g.level.isin(["효과 있음", "조건부"]) & g.direction.eq("감소")).mean()) if eff else None,
             "mde_mean": float(g.mde_pct.dropna().mean()) if g.mde_pct.notna().any() else None,
             "cover": float(g.cover.dropna().mean()) if "cover" in g and g.cover.notna().any() else None,
             "c0_fail": float(g.checks.str.contains("C0:실패").mean())}
        if "est" in g and g.est.notna().any():
            err = g.est.dropna() - eff
            r.update(bias=float(err.mean()), rmse=float(np.sqrt((err ** 2).mean())))
        ok = g.dropna(subset=["placebo_sd", "main_se"])
        if len(ok):
            r["placebo_sd_mean"] = float(ok.placebo_sd.mean())
        if "v3_level" in g:
            r["v3_effect_yes"] = float((g.v3_level.eq("효과 있음") & inc).mean())
            r["v3_any_effect_claim"] = float(g.v3_level.eq("효과 있음").mean())
        out.append(r)
    return out


def synthetic_reference() -> dict:
    """같은 판정 규칙의 합성데이터 결과(비교용): 검정력 곡선과 '아무 일도 없음' 시나리오."""
    p = OUT / "causal_accuracy.json"
    if not p.exists():
        return {}
    j = json.loads(p.read_text(encoding="utf-8"))
    pw = {round(x["size"], 3): x for x in j.get("power", [])}
    null = j["scenarios"].get("null", {})
    return {"null_false_effect": null.get("false_effect"), "null_verdicts": null.get("verdicts"),
            "power": {k: {"detect": v["detect"], "detect_any": v["detect_any"], "null": v["null"], "mde": v["mde"]}
                      for k, v in pw.items()}}


def to_markdown(res: dict) -> str:
    def pc(x, d=1):
        return "-" if x is None else f"{x * 100:.{d}f}%"
    syn = res.get("synthetic") or {}
    L = ["# 반합성(plasmode) 정확도 시험: 실데이터에 정답을 심어 잰 판정 규칙", "",
         f"생성 {res['generated']} · 판정 규칙 {res['rules_version']} · 총 {res['runs']:,}회 · 소요 {res['seconds']:.0f}초 · "
         f"사전 등록 `{res['prereg']['file']}` (지문 `{res['prereg']['sha256'][:16]}…`, 잠금 {res['prereg']['locked_at']})", "",
         "실데이터의 정책이 없던 날짜를 가짜 시행일로 두고 그날부터 대상 계열에 효과를 곱해 넣은 뒤(0이면 그대로), "
         "실제 판정과 같은 엔진·설계로 판정했습니다. 잡음·계절·코로나19·결측 같은 실데이터의 함정이 그대로 들어 있습니다.", "",
         "| 계열 | 심은 효과 | 횟수 | '효과 있음(증가)' | '효과 있음'·'조건부'(증가) | '효과 없음' | 판정 보류 | 방향 반대 | 판정 구간이 정답 포함 | 평균 MDE | 주요 판정 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["summary"]:
        top = ", ".join(f"{k} {v}" for k, v in list(r["verdicts"].items())[:3])
        L.append(f"| {r['title']} | {r['effect_pct'] * 100:+.1f}% | {r['runs']} | {pc(r['effect_yes'])} | "
                 f"{pc(r['effect_or_partial'])} | {pc(r['null'])} | {pc(r['hold'])} | {pc(r['wrong_sign'])} | {pc(r['cover'])} | "
                 f"{'±' + format(r['mde_mean'] * 100, '.1f') + '%' if r['mde_mean'] is not None else '-'} | {top} |")
    zero = [r for r in res["summary"] if r["effect_log"] == 0]
    L += ["", "## 효과를 심지 않았을 때(효과 0)", ""]
    for r in zero:
        L.append(f"- {r['title']}: '효과 있음' {pc(r['any_effect_claim'])}, '효과 있음'·'조건부'로 방향 주장 "
                 f"{pc(r['direction_claim'])}, 판정 보류 {pc(r['hold'])}, 측정 단절 경고(C0 실패) {pc(r['c0_fail'])}")
    if syn:
        L += ["", "## 합성데이터 시험과 비교(같은 판정 규칙)", "",
              f"- 합성 '아무 일도 없음'에서 '효과 있음' 오판 {pc(syn.get('null_false_effect'))}"]
        for k in (0.1, 0.2):
            p = (syn.get("power") or {}).get(k)
            b = next((r for r in res["summary"] if r["family"] == "bus" and abs(r["effect_log"] - k) < 1e-9), None)
            if p and b:
                L.append(f"- 효과 +{np.expm1(k) * 100:.1f}%: 합성 버스 '효과 있음(증가)' {pc(p['detect'])}(평균 MDE ±{p['mde'] * 100:.1f}%) "
                         f"→ 실데이터 버스 {pc(b['effect_yes'])}(평균 MDE ±{b['mde_mean'] * 100:.1f}%)")
    L += ["", "## 읽는 법과 한계", "",
          "- 가짜 시행일끼리 분석 창이 겹쳐 서로 독립인 표본이 아닙니다. 비율은 '이 기간의 실데이터에서 이런 정도'로 읽습니다.",
          "- '효과 0'은 알려진 정책이 없던 날이라는 뜻입니다. 그날 대상 집단에 실제로 다른 일이 있었을 수 있고, "
          "그런 날 판정관이 무엇이라고 하는지가 이 시험이 보려는 것입니다.",
          "- 효과는 가짜 시행일부터 일정한 크기로 심었습니다. 서서히 나타나는 효과나 일시적 효과는 합성 시험에서 따로 봤습니다.",
          "- 이 시험은 판정 규칙(통계 엔진)만 잽니다. 반론 단계는 반론 성능 시험(skeptic_accuracy.md)에서 따로 봅니다.", ""]
    return "\n".join(L)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--smoke", action="store_true", help="계열마다 1건만 돌려 오류·시간만 본다(결과는 출력하지 않음)")
    a = ap.parse_args(argv)
    from jverdict.causal.verdict import RULES_VERSION
    t0 = time.time()
    jobs = [(f, d, e) for f in a.families for d in fake_dates(f) for e in EFFECTS]
    if a.smoke:
        jobs = [next(j for j in jobs if j[0] == f) for f in a.families]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_one, jobs, chunksize=2))
    if a.smoke:
        return {"jobs": len(rows), "seconds": [r["seconds"] for r in rows],
                "full_jobs": {f: len(fake_dates(f)) * len(EFFECTS) for f in a.families}}
    df = pd.DataFrame(rows)
    lock = json.loads((OUTPUT_DIR.parent / "docs" / "prereg" / "plasmode_skeptic_accuracy.lock.json").read_text(encoding="utf-8"))
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "rules_version": RULES_VERSION, "runs": int(len(df)),
           "seconds": time.time() - t0, "platform": f"{platform.system()} {platform.machine()}, 작업자 {a.workers}",
           "prereg": {k: lock[k] for k in ("file", "sha256", "locked_at")},
           "dates": {f: fake_dates(f) for f in a.families}, "effects": EFFECTS,
           "summary": summarize(df), "synthetic": synthetic_reference()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plasmode_accuracy.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "plasmode_accuracy.md").write_text(to_markdown(res), encoding="utf-8")
    df.to_csv(OUT / "plasmode_accuracy_runs.csv", index=False, encoding="utf-8-sig")
    return res


if __name__ == "__main__":
    r = main()
    print(to_markdown(r) if "summary" in r else r)
