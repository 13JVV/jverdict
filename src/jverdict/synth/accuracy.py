"""데이터 품질 툴 정확도 리포트 (사례 C의 1부).

시나리오마다 시드를 바꿔 합성데이터를 여러 번 만들고, 실제 분석과 같은 순서로 툴을 돌린 뒤
정답과 대조한다.
    커버리지 점검 → 부분 집계일 제외 → 측정 단절 진단(정책 시점) → 시점 모르는 단절 탐색

실행:
    .venv/bin/python -m jverdict.synth.accuracy            # 시나리오당 200회
    .venv/bin/python -m jverdict.synth.accuracy --runs 30  # 빠른 점검
결과: outputs/accuracy/quality_accuracy.json, quality_accuracy.md
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from jverdict.config import OUTPUT_DIR
from jverdict.quality._panel import to_wide
from jverdict.quality.breaks import diagnose, scan_breaks
from jverdict.quality.coverage import check_coverage, exclusion_mask
from jverdict.quality.external import ExternalStat
from jverdict.synth.generator import generate
from jverdict.synth.scenarios import SCENARIOS, Case

OUT = OUTPUT_DIR / "accuracy"
NEAR_DAYS = 7


def _external(case: Case, latent: pd.DataFrame, seed: int) -> list[ExternalStat]:
    if not case.external_groups:
        return []
    s, e = case.external_window
    e = min(pd.Timestamp(e), latent.index[-1]).strftime("%Y-%m-%d")   # 공식 집계 기간은 데이터 기간 안
    total = float(latent.loc[s:e, case.external_groups].sum().sum())
    # 행정 통계도 집계 오차가 있다: 실제 값에 ±5% 안팎의 오차를 넣는다
    total *= float(np.exp(np.random.default_rng(seed).normal(0, 0.05)))
    return [ExternalStat("공식 집계(합성, 같은 정의)", total, s, e, case.external_groups, "합성 정답",
                         same_definition=True)]


def run_break_case(name: str, seed: int) -> dict:
    sc = SCENARIOS[name]
    spec, case = sc.build(np.random.default_rng(seed))
    sd = generate(spec, seed)
    cov = check_coverage(sd.df, "time", "group", "value", freq=spec.freq)
    w = to_wide(sd.df, "time", "group", "value", spec.freq)
    mask = exclusion_mask(cov, w.index)
    dg = diagnose(w, case.target, case.reference, case.date, spec.freq, expected=case.expected,
                  external=_external(case, sd.latent, seed), mask=mask)
    cands = scan_breaks(w, case.target, case.reference, spec.freq, mask=mask)
    t0 = pd.Timestamp(case.date)
    near = [c for c in cands if abs((pd.Timestamp(c.date) - t0).days) <= NEAR_DAYS]
    # 서서히 빠지는 단절은 전환 기간 중 어디를 시작으로 볼지 모호하다: 전환 기간 안이면 '구간 적중'
    span = max(NEAR_DAYS, int(case.extra.get("adoption", 0)))
    in_span = [c for c in cands if -NEAR_DAYS <= (pd.Timestamp(c.date) - t0).days <= span]
    # 기존 분류에 없는 신규 분류는 정답상 정상이므로(재분류 시나리오) 커버리지 오탐에서 뺀다
    cov_false = [f for f in cov.findings if f.kind != "분류 신설"]
    t = dg.test
    out = {
        "scenario": name, "seed": seed, "label": case.label, "verdict": dg.verdict,
        "ideal": dg.verdict in case.ideal,
        "acceptable": dg.verdict in case.ideal or dg.verdict in case.tolerable,
        "suspected": dg.suspected, "step": t.step, "ci": t.ci, "p": t.p_rank,
        "true_log_step": case.true_log_step,
        "scan_hit": bool(near), "scan_hit_span": bool(in_span), "scan_top1": bool(cands) and cands[0] in in_span,
        "scan_n": len(cands),
        "scan_date_error": (min(abs((pd.Timestamp(c.date) - t0).days) for c in near) if near else None),
        "scan_reclass": any(c.reclass for c in near),
        "coverage_false": len(cov_false),
        "new_category_found": any(f.kind == "분류 신설" for f in cov.findings),
    }
    if case.true_log_step is not None and t.ci:
        out["ci_cover"] = bool(t.ci[0] <= case.true_log_step <= t.ci[1])
    return out


def _overlap(a: tuple[str, str], b: tuple[str, str]) -> bool:
    return pd.Timestamp(a[0]) <= pd.Timestamp(b[1]) and pd.Timestamp(b[0]) <= pd.Timestamp(a[1])


def run_coverage_case(name: str, seed: int) -> dict:
    sc = SCENARIOS[name]
    spec, case = sc.build(np.random.default_rng(seed))
    sd = generate(spec, seed)
    cov = check_coverage(sd.df, "time", "group", "value", unit="unit" if spec.units > 1 else None,
                         freq=spec.freq)
    gap = case.extra["gap"]
    if name == "gap_partial":
        full = [f for f in cov.by_kind("전체 결측")]
        exact = any(f.start == gap[0] and f.end == gap[1] for f in full)
        inj = next(i for i in sd.truth["injections"] if i.get("kind") == "missing")
        tail = inj.get("tail_dates", [])
        pd_inj = next(i for i in sd.truth["injections"] if i.get("kind") == "partial_days")["dates"]
        idx_all = pd.date_range(spec.start, spec.end, freq="D")
        m_all = exclusion_mask(cov, idx_all)
        flagged_days = {str(d.date()) for d in idx_all[m_all]}
        # 결측 직후 재개일 중 행이 남아 있는 날만 정답으로 센다
        present = set(sd.df["time"].dt.strftime("%Y-%m-%d").unique())
        truth_partial = {d for d in set(tail) | set(pd_inj) if d in present}
        hit = truth_partial & flagged_days
        false = flagged_days - truth_partial
        return {"scenario": name, "seed": seed, "label": "coverage", "gap_exact": exact,
                "gap_found": bool(full) and any(_overlap((f.start, f.end), gap) for f in full),
                "partial_recall": len(hit) / len(truth_partial) if truth_partial else None,
                "partial_false_days": len(false),
                "other_false": len([f for f in cov.findings if f.kind not in ("전체 결측", "부분 집계")])}
    gm = cov.by_kind("집단 결측")
    want = (pd.Timestamp(gap[0]).strftime("%Y-%m"), pd.Timestamp(gap[1]).strftime("%Y-%m"))
    exact = any(f.start == want[0] and f.end == want[1] and set(f.groups) == set(case.extra["groups"]) for f in gm)
    return {"scenario": name, "seed": seed, "label": "coverage", "gap_exact": exact,
            "gap_found": any(_overlap((f.start, f.end), want) for f in gm),
            "other_false": len([f for f in cov.findings if f not in gm])}


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out = {}
    for name, g in df.groupby("scenario", sort=False):
        sc = SCENARIOS[name]
        s = {"title": sc.title, "label": sc.label, "runs": int(len(g))}
        if sc.kind == "break":
            s["verdicts"] = dict(Counter(g["verdict"]).most_common())
            s["ideal_rate"] = float(g["ideal"].mean())
            s["acceptable_rate"] = float(g["acceptable"].mean())
            s["suspected_rate"] = float(g["suspected"].mean())
            s["scan_hit_rate"] = float(g["scan_hit"].mean())
            s["scan_hit_span_rate"] = float(g["scan_hit_span"].mean())
            s["scan_top1_rate"] = float(g["scan_top1"].mean())
            s["scan_candidates_mean"] = float(g["scan_n"].mean())
            de = g["scan_date_error"].dropna()
            s["scan_date_error_median"] = float(de.median()) if len(de) else None
            s["coverage_false_rate"] = float((g["coverage_false"] > 0).mean())
            if "ci_cover" in g and g["ci_cover"].notna().any():
                cc = g.dropna(subset=["ci_cover"])
                s["ci_coverage"] = float(cc["ci_cover"].astype(bool).mean())
                err = cc["step"] - cc["true_log_step"]
                s["bias_log"] = float(err.mean())
                s["rmse_log"] = float(np.sqrt((err ** 2).mean()))
            if name == "reclass":
                s["scan_reclass_rate"] = float(g["scan_reclass"].mean())
                s["new_category_rate"] = float(g["new_category_found"].mean())
        else:
            s["gap_exact_rate"] = float(g["gap_exact"].mean())
            s["gap_found_rate"] = float(g["gap_found"].mean())
            s["other_false_rate"] = float((g["other_false"] > 0).mean())
            if "partial_recall" in g:
                s["partial_recall_mean"] = float(g["partial_recall"].dropna().mean())
                s["partial_false_days_mean"] = float(g["partial_false_days"].mean())
        out[name] = s
    return out


def headline(summary: dict) -> dict:
    """정확도 리포트의 핵심 지표 (기획서 성공 기준과 같은 이름)."""
    def rate(names, key):
        vals = [summary[n][key] for n in names if n in summary]
        runs = [summary[n]["runs"] for n in names if n in summary]
        return float(np.average(vals, weights=runs)) if vals else None
    return {
        "측정 단절 탐지율(강한 단절)": summary.get("mbreak", {}).get("suspected_rate"),
        "측정 단절 탐지율(약한 단절 + 공식 통계)": summary.get("mbreak_mild_ext", {}).get("suspected_rate"),
        "분류 변경 탐지율(새 분류)": summary.get("reclass", {}).get("suspected_rate"),
        "분류 변경 탐지율(기존 분류로 이동, '확인 필요' 포함)": summary.get("reclass_existing", {}).get("acceptable_rate"),
        "전체 집계 이상 탐지율": summary.get("outage", {}).get("ideal_rate"),
        "오탐률(실제 효과를 측정 단절로)": rate(["effect_step", "effect_ramp", "effect_ext", "backfire"], "suspected_rate"),
        "오탐률(아무 일 없음)": summary.get("null", {}).get("suspected_rate"),
        "효과 크기 95% 구간 포함률": rate(["effect_step", "effect_ext", "backfire"], "ci_coverage"),
        "단절 시점 탐색 적중률(측정 단절, 전환 기간 안)": rate(["mbreak", "mbreak_mild", "mbreak_mild_ext", "reclass", "reclass_existing"], "scan_hit_span_rate"),
        "전체 결측 구간 정확 일치율": summary.get("gap_partial", {}).get("gap_exact_rate"),
        "집단 결측 정확 일치율": summary.get("group_missing", {}).get("gap_exact_rate"),
        "부분 집계일 재현율": summary.get("gap_partial", {}).get("partial_recall_mean"),
    }


def to_markdown(result: dict) -> str:
    h = result["headline"]
    pct = lambda v: "-" if v is None or v != v else f"{v * 100:.1f}%"
    L = ["# 데이터 품질 툴 정확도 리포트", "",
         f"- 실행: {result['meta']['finished']} · 시나리오당 {result['meta']['runs']}회 · 시드 시작 {result['meta']['seed0']}"
         f" · 소요 {result['meta']['seconds']:.0f}초",
         "- 방법: 정답을 심은 합성데이터 → 커버리지 점검 → 부분 집계일 제외 → 측정 단절 진단 → 단절 시점 탐색",
         "- 합성데이터 잡음은 실데이터(2023-03~2025-06 청소년/일반 비율)의 변동 크기에 맞췄다", "",
         "## 핵심 지표", "", "| 지표 | 값 |", "|---|---|"]
    L += [f"| {k} | {pct(v)} |" for k, v in h.items()]
    L += ["", "## 시나리오별 결과 (측정 단절 진단)", "",
          "| 시나리오 | 정답 | 올바른 진단 | 허용 포함 | '의심' 비율 | 시점 탐색 ±7일 | 시점 탐색 전환 기간 안 | 진단 분포 |",
          "|---|---|---|---|---|---|---|---|"]
    for n, s in result["summary"].items():
        if "verdicts" not in s:
            continue
        dist = ", ".join(f"{k} {v}" for k, v in s["verdicts"].items())
        no_scan = s["label"] in ("none", "outage") or n == "effect_ramp"
        scan = "-" if no_scan else pct(s["scan_hit_rate"])
        span = "-" if no_scan else pct(s["scan_hit_span_rate"])
        L.append(f"| {s['title']} | {s['label']} | {pct(s['ideal_rate'])} | {pct(s['acceptable_rate'])} | "
                 f"{pct(s['suspected_rate'])} | {scan} | {span} | {dist} |")
    L += ["", "## 효과 크기 복원 (실제 효과 시나리오)", "", "| 시나리오 | 95% 구간 포함률 | 평균 편향(로그) | RMSE(로그) |",
          "|---|---|---|---|"]
    for n, s in result["summary"].items():
        if "ci_coverage" in s:
            L.append(f"| {s['title']} | {pct(s['ci_coverage'])} | {s['bias_log']:+.4f} | {s['rmse_log']:.4f} |")
    L += ["", "## 커버리지 점검", "", "| 시나리오 | 구간 정확 일치 | 구간 겹침 | 그 밖의 오탐 | 부분 집계일 재현율 |", "|---|---|---|---|---|"]
    for n, s in result["summary"].items():
        if "gap_exact_rate" in s:
            L.append(f"| {s['title']} | {pct(s['gap_exact_rate'])} | {pct(s['gap_found_rate'])} | "
                     f"{pct(s['other_false_rate'])} | {pct(s.get('partial_recall_mean'))} |")
    null = result["summary"].get("null", {})
    if null:
        L += ["", f"- 아무 일 없는 데이터에서 시점 탐색이 낸 단절 후보: 평균 {null['scan_candidates_mean']:.2f}개 (약 5년 일 단위)",
              f"- 아무 일 없는 데이터에서 커버리지 점검 오탐: {pct(null['coverage_false_rate'])}"]
    L += ["", "## 해석과 한계", "", *result.get("notes", [])]
    return "\n".join(L) + "\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=20261103)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    t0 = time.time()
    rows = []
    names = a.only or list(SCENARIOS)
    for k, name in enumerate(names):
        fn = run_coverage_case if SCENARIOS[name].kind == "coverage" else run_break_case
        for i in range(a.runs):
            rows.append(fn(name, a.seed0 + 100_000 * k + i))
        print(f"{name}: {a.runs}회 완료 ({time.time() - t0:.0f}초)", flush=True)
    summary = summarize(rows)
    result = {"meta": {"runs": a.runs, "seed0": a.seed0, "seconds": time.time() - t0,
                       "finished": time.strftime("%Y-%m-%d %H:%M"), "python": platform.python_version()},
              "headline": headline(summary), "summary": summary, "notes": NOTES}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "quality_accuracy.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                               encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "quality_accuracy_runs.csv", index=False)
    (OUT / "quality_accuracy.md").write_text(to_markdown(result), encoding="utf-8")
    return result


NOTES = [
    "- '올바른 진단'은 시나리오마다 미리 정한 정답 판정이고, '허용'은 틀리지는 않았지만 보수적인 판정(예: 실제 효과를 '확인 필요'로 둠)이다.",
    "- 측정 단절이 정책이 기대한 방향과 같고(예: 기록이 부풀려짐) 공식 통계도 없으면, 이 툴은 원리상 실제 효과와 구별할 수 없다. 공식 통계 대조가 그 빈틈을 메운다.",
    "- 시점 탐색은 대상 ÷ 대조 비율을 보므로, 모든 집단이 함께 줄어드는 전체 집계 이상은 시점 탐색이 아니라 대조 집단 자체 점검과 커버리지 점검이 잡는다. 서서히 생기는 실제 효과(1~3개월)도 '단절'이 아니므로 탐색 대상에서 뺐다.",
    "- 2~3주에 걸쳐 서서히 빠지는 측정 단절은 누락이 15% 안팎 진행된 시점부터 데이터에 드러난다. 그래서 추정 시작일이 실제보다 1~10일 늦을 수 있고, 엄격 기준(±7일)과 '전환 기간 안' 기준을 함께 보고한다.",
    "- 기존 분류 사이의 이동(예: 어린이 일부가 청소년으로 집계)은 '두 분류의 합계가 그대로인가'로 확인한다. 합계 보존이 확인되면 강한 근거, 반대로 움직이기만 했으면 우연일 수 있어 보통 근거로만 쓴다.",
    "- 실제 역효과는 '방향 모순' 근거 하나만 충족해 '확인 필요'가 된다. 측정 단절로 단정하지 않도록 근거 1개로는 '의심'을 내지 않게 설계했다.",
]


if __name__ == "__main__":
    r = main()
    print(json.dumps(r["headline"], ensure_ascii=False, indent=2))
