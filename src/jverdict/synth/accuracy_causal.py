"""인과 엔진 정확도 리포트 (사례 C의 2부).

시나리오마다 시드를 바꿔 합성데이터를 만들고, 실제 판정과 같은 순서로 엔진 전체를 돌린 뒤 정답과 대조한다.
    커버리지 점검 → 부분 집계일 제외 → 측정 단절 진단 → 추정 → 위약·사전추세·민감도·이중 계산 → 검정력 → 판정

재는 것
- 판정 정확도: 이상적 판정 비율, 허용 범위 비율, 오판 비율
- '없는 효과를 있다고 한' 비율(효과 있음 오판)과 '있는 효과를 없다고 한' 비율(효과 없음 오판)
- 효과 크기 복원: 편향, RMSE, 95% 구간 포함률 (분석적 구간 / 판정용 구간)
- 검정력 곡선: 실제 효과 크기별 '효과 있음' 판정 비율

실행:
    .venv/bin/python -m jverdict.synth.accuracy_causal              # 시나리오당 200회 + 검정력 곡선
    .venv/bin/python -m jverdict.synth.accuracy_causal --runs 20    # 빠른 점검
결과: outputs/accuracy/causal_accuracy.json, causal_accuracy.md, causal_accuracy_runs.csv
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

from jverdict.causal import PanelDesign, SeriesDesign, analyze_panel, analyze_series
from jverdict.causal.verdict import RULES_VERSION
from jverdict.config import OUTPUT_DIR
from jverdict.quality._panel import to_wide
from jverdict.quality.coverage import check_coverage, exclusion_mask
from jverdict.synth.generator import Effect, generate
from jverdict.synth.scenarios import _policy_date
from jverdict.synth.scenarios_causal import CSCENARIOS, PANEL_GROUPS, _bus

OUT = OUTPUT_DIR / "accuracy"
POWER_SIZES = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]


def series_design(date: str) -> SeriesDesign:
    return SeriesDesign("청소년 이용이 정책 때문에 늘었는가(합성)", "청소년", "일반", date, "D", expected="increase",
                        reference_effect=0.10, placebo_groups=["경로", "어린이"])


def panel_frame(sd) -> pd.DataFrame:
    df = sd.df[sd.df.group.isin(PANEL_GROUPS)].copy()
    df["uid"] = df.group + "|" + df.unit
    df["y"] = np.log(df.value.astype(float))
    return df


def panel_design(date: str) -> PanelDesign:
    def first_month_out(d):
        return d[d.time != pd.Timestamp(date)]

    def no_japan(d):
        return d[d.group != "일본"]

    def pre12(d):
        return d[d.time >= pd.Timestamp(date) - pd.DateOffset(months=12)]

    return PanelDesign("중국인 소비가 정책 때문에 늘었는가(합성)", "y", "uid", "time", "group", "중국", date,
                       reference_effect=0.20, window=(24, 6), placebo_window=(24, 6),
                       sens={"S1 사후 첫 달 제외": {"filter": first_month_out},
                             "S2 대조 집단에서 일본 제외": {"filter": no_japan},
                             "S3 사전 12개월": {"filter": pre12}})


def _row(name: str, seed: int, case, a, t0: float) -> dict:
    v = a.verdict
    out = {"scenario": name, "seed": seed, "kind": case.kind, "date": case.date, "verdict": v.label,
           "level": v.level, "ideal": v.label in case.ideal,
           "acceptable": v.label in case.ideal or v.label in case.tolerable,
           "false_effect": (not case.has_effect) and v.level == "효과 있음",
           "false_null": case.has_effect and v.level == "효과 없음",
           "wrong_sign": case.has_effect and v.level in ("효과 있음", "조건부") and v.direction is not None and
           np.sign(case.true_effect or 0) != (1 if v.direction == "증가" else -1),
           "true_effect": case.true_effect, "quality_cap": v.quality_cap,
           "seconds": round(time.time() - t0, 3)}
    w = a.verdict_v3                     # 개정 규칙 v3로 같은 결과를 판정한 값(나란히 보고)
    if w is not None:
        out.update(v3_verdict=w.label, v3_level=w.level, v3_ideal=w.label in case.ideal,
                   v3_acceptable=w.label in case.ideal or w.label in case.tolerable,
                   v3_false_effect=(not case.has_effect) and w.level == "효과 있음",
                   v3_false_null=case.has_effect and w.level == "효과 없음",
                   has_effect=case.has_effect, true_sign=float(np.sign(case.true_effect or 0)))
    m = a.main
    if m is not None:
        out.update(effect=m.effect, ci_lo=m.ci[0], ci_hi=m.ci[1], ci_used_lo=v.ci_used[0], ci_used_hi=v.ci_used[1],
                   sig_analytic=m.significant)
        if case.true_effect is not None and case.has_effect:
            out["cover_analytic"] = bool(m.ci[0] <= case.true_effect <= m.ci[1])
            out["cover_used"] = bool(v.ci_used[0] <= case.true_effect <= v.ci_used[1])
        if not case.has_effect and name in ("null", "panel_null"):
            out["cover_analytic"] = bool(m.ci[0] <= 0 <= m.ci[1])
            out["cover_used"] = bool(v.ci_used[0] <= 0 <= v.ci_used[1])
        if "cr1_se" in m.details and "cover_analytic" in out:
            # 비교용: 업종 군집 표준오차만 썼다면의 포함률
            truth = case.true_effect if case.has_effect else 0.0
            out["cover_cr1"] = bool(abs(m.effect - truth) <= 1.96 * m.details["cr1_se"])
    out["placebo_n"] = a.placebo_time.n if a.placebo_time is not None else 0
    out["pretrend_p"] = a.pretrend.get("p") if a.pretrend else None
    out["mde_pct"] = a.power.mde_pct if a.power else None
    out["checks"] = ";".join(f"{c.code}:{c.status}" for c in v.checks)
    return out


def run_case(args: tuple[str, int]) -> dict:
    name, seed = args
    warnings.filterwarnings("ignore")
    t0 = time.time()
    sc = CSCENARIOS[name]
    spec, case = sc.build(np.random.default_rng(seed))
    sd = generate(spec, seed)
    if sc.kind == "series":
        cov = check_coverage(sd.df, "time", "group", "value", freq="D")
        w = to_wide(sd.df, "time", "group", "value", "D")
        a = analyze_series(w, series_design(case.date), mask=exclusion_mask(cov, w.index))
    else:
        a = analyze_panel(panel_frame(sd), panel_design(case.date))
    return _row(name, seed, case, a, t0)


def run_power(args: tuple[float, int]) -> dict:
    size, seed = args
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    d = _policy_date(rng)
    sd = generate(_bus([Effect("청소년", d, size)] if size else []), seed)
    cov = check_coverage(sd.df, "time", "group", "value", freq="D")
    w = to_wide(sd.df, "time", "group", "value", "D")
    a = analyze_series(w, series_design(d), mask=exclusion_mask(cov, w.index))
    v, w = a.verdict, a.verdict_v3
    return {"size": size, "seed": seed, "level": v.level, "direction": v.direction,
            "detect": v.level == "효과 있음" and v.direction == "증가",
            "detect_any": v.level in ("효과 있음", "조건부") and v.direction == "증가",
            "null": v.level == "효과 없음", "mde_pct": a.power.mde_pct if a.power else None,
            "detect_v3": w.level == "효과 있음" and w.direction == "증가", "null_v3": w.level == "효과 없음"}


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for name, g in df.groupby("scenario", sort=False):
        sc = CSCENARIOS[name]
        r = {"title": sc.title, "kind": sc.kind, "runs": int(len(g)),
             "verdicts": dict(Counter(g.verdict).most_common()),
             "ideal": float(g.ideal.mean()), "acceptable": float(g.acceptable.mean()),
             "false_effect": float(g.false_effect.mean()), "false_null": float(g.false_null.mean()),
             "wrong_sign": float(g.wrong_sign.mean()), "seconds_mean": float(g.seconds.mean())}
        has = g.dropna(subset=["effect"]) if "effect" in g else g.iloc[0:0]
        if len(has) and has.true_effect.notna().any():
            err = has.effect - has.true_effect
            r.update(bias=float(err.mean()), rmse=float(np.sqrt((err ** 2).mean())))
        if "cover_analytic" in g and g.cover_analytic.notna().any():
            r.update(cover_analytic=float(g.cover_analytic.dropna().mean()),
                     cover_used=float(g.cover_used.dropna().mean()))
        if "cover_cr1" in g and g.cover_cr1.notna().any():
            r["cover_cr1"] = float(g.cover_cr1.dropna().mean())
        r["placebo_available"] = float((g.placebo_n >= 19).mean())
        if "v3_level" in g:
            r.update(v3_ideal=float(g.v3_ideal.mean()), v3_acceptable=float(g.v3_acceptable.mean()),
                     v3_false_effect=float(g.v3_false_effect.mean()), v3_false_null=float(g.v3_false_null.mean()),
                     v3_verdicts=dict(Counter(g.v3_verdict).most_common()))
        out[name] = r
    return out


def _rates(df: pd.DataFrame, level: str, prefix: str = "") -> dict:
    """규칙별 보조 지표: 보류 비율, 판정별 적중률(PPV), 효과 없는 상황의 방향 주장 비율."""
    lv = df[f"{prefix}level"]
    no_eff = df[~df.has_effect.astype(bool)]
    eff_only = df[df.has_effect.astype(bool)]
    claim = lv.isin(["효과 있음", "조건부"])
    out = {"hold_rate": float(lv.eq("판정 보류").mean()),
           "direction_claim_no_effect": float(claim[no_eff.index].mean()) if len(no_eff) else float("nan"),
           "effect_detect_any": float(lv[eff_only.index].isin(["효과 있음", "조건부"]).mean()) if len(eff_only) else float("nan")}
    for name, key in (("효과 있음", "ppv_effect"), ("조건부", "ppv_partial"), ("효과 없음", "ppv_null")):
        sel = df[lv.eq(name)]
        if len(sel):
            ok = sel.has_effect.astype(bool) if name != "효과 없음" else ~sel.has_effect.astype(bool)
            out[key], out[key + "_n"] = float(ok.mean()), int(len(sel))
    return out


def headline(summary: dict, df: pd.DataFrame) -> dict:
    no_eff = df[df.scenario.isin([n for n, s in CSCENARIOS.items() if not _has_effect(n)])]
    eff = df[df.scenario.isin([n for n in CSCENARIOS if _has_effect(n)])]
    extra = {}
    if "v3_level" in df:
        extra = {"v2": _rates(df, "", ""), "v3": {**_rates(df, "", "v3_"),
                 "acceptable_all": float(df.v3_acceptable.mean()), "ideal_all": float(df.v3_ideal.mean()),
                 "false_effect_rate_no_effect_scenarios": float(no_eff.v3_false_effect.mean()),
                 "false_null_rate_effect_scenarios": float(eff.v3_false_null.mean()),
                 "effect_detect_series": float(df[df.scenario == "effect"].v3_verdict.eq("효과 있음(증가)").mean()),
                 "effect_detect_panel": float(df[df.scenario == "panel_effect"].v3_verdict.eq("효과 있음(증가)").mean())}}
    return {**extra,
        "runs": int(len(df)),
        "acceptable_all": float(df.acceptable.mean()),
        "ideal_all": float(df.ideal.mean()),
        "false_effect_rate_no_effect_scenarios": float(no_eff.false_effect.mean()),
        "false_null_rate_effect_scenarios": float(eff.false_null.mean()),
        "wrong_sign_rate": float(eff.wrong_sign.mean()),
        "effect_detect_series": float(df[df.scenario == "effect"].verdict.eq("효과 있음(증가)").mean()),
        "effect_detect_panel": float(df[df.scenario == "panel_effect"].verdict.eq("효과 있음(증가)").mean()),
    }


def _has_effect(name: str) -> bool:
    return CSCENARIOS[name].build(np.random.default_rng(0))[1].has_effect


NOTES = [
    "판정용 구간은 분석적 구간(일 단위 HAC+fixed-b, 월 단위 AR(1))과 시점 위약 경험적 구간의 합집합이다. 위약 시행일이 19개 미만이면 분석적 구간만 쓴다.",
    "'역효과' 시나리오는 정책이 늘릴 것으로 기대했는데 줄어든 경우라, 측정 단절 진단이 '확인 필요'(상한 조건부)를 내는 것이 설계상 정상이다. 따라서 '조건부(감소)'도 이상적 판정으로 본다.",
    "'사전추세' 시나리오는 실제 정책 효과가 0이다. '효과 없음'은 결론으로는 맞지만, 추세 때문에 효과를 구분하기 어려운 상황이라 '판정 보류'와 함께 이상적 판정으로 본다. '효과 있음'은 오판이다.",
    "패널 시나리오는 처치 집단이 1개(중국)이고 집단 공통 충격이 업종 단위 전체에 걸친다. 업종 군집 표준오차는 이 충격을 반영하지 못해 지나치게 좁아진다(괄호 안 포함률). 엔진은 집단 수준 격차 시계열의 AR(1) 표준오차와 시점 위약 구간으로 보완한다.",
    "'측정 단절' 시나리오의 '조건부(감소)'는 40~60% 누락의 약한 단절이다. 공식 통계가 없으면 실제 역효과와 구분되지 않는다(1부 정확도 리포트와 같은 한계). 품질 상한 때문에 '효과 있음'으로는 가지 않는다.",
    "'사전추세' 시나리오에서 남는 '효과 있음' 오판은 정책 직전 몇 달 사이에 시작된 완만한 추세다. 사전 2년(패널 24개월)의 선형 추세 검정과 '추세 연장으로 설명되는 몫' 기준으로도 일부는 걸러지지 않는다. 판정서의 '이 분석이 말할 수 없는 것' 문단에 평행추세 가정을 항상 적는다.",
    "검정력 곡선: 기준 효과(10%) 미만의 작은 실제 효과는 '효과 없음'(기준 효과보다 큰 효과는 없음)으로 판정되는 것이 규칙상 정상이다.",
]


def to_markdown(res: dict) -> str:
    h, s = res["headline"], res["scenarios"]
    L = [f"# 인과 엔진 정확도 리포트 (사례 C · 2부)", "",
         f"생성 {res['generated']} · 판정 규칙 {res['rules_version']} · 총 {h['runs']:,}회 · 소요 {res['seconds']:.0f}초 · "
         f"{res['platform']}", "",
         "## 요약", "",
         f"- 판정이 허용 범위 안: **{h['acceptable_all'] * 100:.1f}%** (이상적 판정 {h['ideal_all'] * 100:.1f}%)",
         f"- 효과가 없는 시나리오에서 '효과 있음'이라고 한 비율: **{h['false_effect_rate_no_effect_scenarios'] * 100:.2f}%**",
         f"- 효과가 있는 시나리오에서 '효과 없음'이라고 한 비율: **{h['false_null_rate_effect_scenarios'] * 100:.2f}%**",
         f"- 효과 방향을 반대로 판정한 비율: {h['wrong_sign_rate'] * 100:.2f}%",
         f"- 실제 효과(+11~42%) '효과 있음(증가)' 판정: 시계열 {h['effect_detect_series'] * 100:.1f}%, "
         f"패널(+49~123%) {h['effect_detect_panel'] * 100:.1f}%", "",
         "## 시나리오별 판정", "",
         "| 시나리오 | 이상적 | 허용 범위 | 효과 있음 오판 | 효과 없음 오판 | 주요 판정 분포 |",
         "|---|---|---|---|---|---|"]
    for n, r in s.items():
        top = ", ".join(f"{k} {v}" for k, v in list(r["verdicts"].items())[:4])
        L.append(f"| {r['title']} | {r['ideal'] * 100:.1f}% | {r['acceptable'] * 100:.1f}% | "
                 f"{r['false_effect'] * 100:.1f}% | {r['false_null'] * 100:.1f}% | {top} |")
    L += ["", "## 효과 크기 복원", "",
          "| 시나리오 | 편향(로그) | RMSE | 95% 구간 포함률(분석적) | 포함률(판정용) | 위약 19개 이상 |", "|---|---|---|---|---|---|"]
    for n, r in s.items():
        if "cover_analytic" not in r:
            continue
        b = f"{r['bias']:+.4f}" if "bias" in r else "-"
        rm = f"{r['rmse']:.4f}" if "rmse" in r else "-"
        cr1 = f" (업종 군집만 {r['cover_cr1'] * 100:.0f}%)" if "cover_cr1" in r else ""
        L.append(f"| {r['title']} | {b} | {rm} | {r['cover_analytic'] * 100:.1f}%{cr1} | {r['cover_used'] * 100:.1f}% | "
                 f"{r['placebo_available'] * 100:.0f}% |")
    if res.get("power"):
        L += ["", "## 검정력 곡선 (시계열, 사후 6개월, 기준 효과 10%)", "",
              "| 실제 효과 | '효과 있음(증가)' | '효과 있음' 또는 '조건부'(증가) | '효과 없음' | 평균 MDE |", "|---|---|---|---|---|"]
        for p in res["power"]:
            L.append(f"| {np.expm1(p['size']) * 100:+.1f}% | {p['detect'] * 100:.1f}% | {p['detect_any'] * 100:.1f}% | "
                     f"{p['null'] * 100:.1f}% | ±{p['mde'] * 100:.1f}% |")
    if h.get("v3"):
        a2, a3 = h["v2"], h["v3"]

        def pc(x):
            return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"
        L += ["", "## 판정 규칙 v3(개정안)와 나란히", "",
              "공식 판정은 v2입니다. v3는 외부 검토(동등성 방식의 '효과 없음', 사전 시점 4개 미만 보류, 위약검정 체제 점검)를 "
              "반영한 개정안으로, 같은 2,400회 결과를 다시 판정한 값입니다.", "",
              "| 지표 | v2(공식) | v3(개정안) |", "|---|---|---|",
              f"| 이상적 판정 | {pc(h['ideal_all'])} | {pc(a3['ideal_all'])} |",
              f"| 효과 없는데 '효과 있음' | {pc(h['false_effect_rate_no_effect_scenarios'])} | {pc(a3['false_effect_rate_no_effect_scenarios'])} |",
              f"| 효과 있는데 '효과 없음' | {pc(h['false_null_rate_effect_scenarios'])} | {pc(a3['false_null_rate_effect_scenarios'])} |",
              f"| 실제 효과에 '효과 있음'(시계열/패널) | {pc(h['effect_detect_series'])} / {pc(h['effect_detect_panel'])} | "
              f"{pc(a3['effect_detect_series'])} / {pc(a3['effect_detect_panel'])} |",
              f"| 판정 보류 비율 | {pc(a2['hold_rate'])} | {pc(a3['hold_rate'])} |",
              f"| 효과 없는 상황에서 방향 주장('효과 있음'·'조건부') | {pc(a2['direction_claim_no_effect'])} | {pc(a3['direction_claim_no_effect'])} |",
              f"| '효과 있음' 판정의 적중률(PPV) | {pc(a2.get('ppv_effect'))} | {pc(a3.get('ppv_effect'))} |",
              f"| '조건부' 판정의 적중률(PPV) | {pc(a2.get('ppv_partial'))} | {pc(a3.get('ppv_partial'))} |",
              f"| '효과 없음' 판정의 적중률 | {pc(a2.get('ppv_null'))} | {pc(a3.get('ppv_null'))} |", "",
              "| 시나리오 | 이상적 v2 → v3 | v3 주요 판정 분포 |", "|---|---|---|"]
        for n, r in s.items():
            if "v3_ideal" in r:
                top = ", ".join(f"{k} {v}" for k, v in list(r["v3_verdicts"].items())[:4])
                L.append(f"| {r['title']} | {r['ideal'] * 100:.1f}% → {r['v3_ideal'] * 100:.1f}% | {top} |")
        if res.get("power") and "null_v3" in res["power"][0]:
            L += ["", "검정력 곡선(v3): " + ", ".join(
                f"{np.expm1(p['size']) * 100:+.1f}% → 효과 있음 {p['detect_v3'] * 100:.0f}%·효과 없음 {p['null_v3'] * 100:.0f}%"
                for p in res["power"])]
        L += ["", "PPV는 시나리오 구성(효과 있는 시나리오와 없는 시나리오의 비율)에 따라 달라지므로 실제 현장의 적중률로 읽지 않습니다."]
    L += ["", "## 해석과 한계", ""] + [f"- {x}" for x in NOTES] + [""]
    return "\n".join(L)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--power-runs", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=20261104)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    t0 = time.time()
    names = a.only or list(CSCENARIOS)
    order = list(CSCENARIOS)
    jobs = [(n, a.seed0 + 10_000 * order.index(n) + k) for n in names for k in range(a.runs)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_case, jobs, chunksize=4))
        pr = a.power_runs if a.power_runs is not None else a.runs
        pjobs = [(s, a.seed0 + 500_000 + 10_000 * j + k) for j, s in enumerate(POWER_SIZES) for k in range(pr)] \
            if not a.only else []
        prow = list(ex.map(run_power, pjobs, chunksize=4))
    df = pd.DataFrame(rows)
    power = []
    if prow:
        pdf = pd.DataFrame(prow)
        for s, g in pdf.groupby("size"):
            power.append({"size": float(s), "runs": int(len(g)), "detect": float(g.detect.mean()),
                          "detect_any": float(g.detect_any.mean()), "null": float(g.null.mean()),
                          "mde": float(g.mde_pct.mean()), "detect_v3": float(g.detect_v3.mean()),
                          "null_v3": float(g.null_v3.mean())})
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "rules_version": RULES_VERSION,
           "platform": f"{platform.system()} {platform.machine()}, 작업자 {a.workers}",
           "seconds": time.time() - t0, "seed0": a.seed0, "runs_per_scenario": a.runs,
           "headline": headline(summarize(df), df), "scenarios": summarize(df), "power": power}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if not a.only else "_partial"
    (OUT / f"causal_accuracy{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"causal_accuracy{tag}.md").write_text(to_markdown(res), encoding="utf-8")
    df.to_csv(OUT / f"causal_accuracy{tag}_runs.csv", index=False, encoding="utf-8-sig")
    return res


if __name__ == "__main__":
    r = main()
    print(to_markdown(r))
