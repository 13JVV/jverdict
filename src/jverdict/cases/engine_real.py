"""실데이터 사례를 인과 엔진(판정 규칙 v2)으로 판정한다.

사례 A 청소년 버스 무료(2025-08-01)   : 시계열 경로, 품질 점검 포함 (정류소 파일·노선 파일)
사례 B 1차 긴급재난지원금(2020-05)    : 시계열 경로(월, 전년 동월 차분). 사전 등록 docs/prereg/case_relief2020_engine.md
사례 D 중국 단체관광 재개(2023-08-10) : 패널 경로. 사전 등록 docs/prereg/case_china_group_tour_2023.md (수정 1)
                                        기록된 판정(규칙 v1)은 바꾸지 않고, v1·v2 판정을 나란히 보고한다.

각 사례에서 엔진 수치를 사례 스크립트의 독립 구현과 대조한다(교차 계산).

실행: .venv/bin/python -m jverdict.cases.engine_real
결과: outputs/cases/engine/real_verdicts.json, real_verdicts.md
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings

import duckdb
import numpy as np
import pandas as pd

from jverdict.causal import PanelDesign, SeriesDesign, analyze_panel, analyze_series
from jverdict.causal.verdict import RULE_HISTORY, RULES_VERSION, judge_v1
from jverdict.cases import china2023 as D
from jverdict.cases import relief2020 as B
from jverdict.cases.quality_real import NEEDED, OFFICIAL, POLICY, load_route, load_station
from jverdict.config import OUTPUT_DIR, RAW_DIR, ROOT
from jverdict.quality._panel import to_wide
from jverdict.quality.coverage import check_coverage, exclusion_mask

OUT = OUTPUT_DIR / "cases" / "engine"
B_PREREG = ROOT / "docs" / "prereg" / "case_relief2020_engine.md"
B_LOCK = ROOT / "docs" / "prereg" / "case_relief2020_engine.lock.json"


def _check_lock(doc, lock_path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if hashlib.sha256(doc.read_bytes()).hexdigest() != lock["sha256"]:
        raise SystemExit(f"사전 등록 문서가 잠금 이후 바뀌었습니다: {doc.name}")
    return lock


# --------------------------------------------------------------------------- 사례 A
def case_a() -> dict:
    out = {}
    for label, loader, unit in (("정류소 파일", load_station, "station_name"), ("노선 파일", load_route, "bus_number")):
        df = loader()
        cov = check_coverage(df, "base_date", "user_type", "user_count", unit=unit, freq="D")
        w = to_wide(df, "base_date", "user_type", "user_count", "D")
        m = exclusion_mask(cov, w.index)
        ds = SeriesDesign("청소년 버스 무료화(2025-08-01)로 청소년의 버스 이용이 늘었는가?", "청소년", "일반", POLICY, "D",
                          expected="increase", reference_effect=0.10, placebo_groups=["경로", "장애 일반"],
                          external=[OFFICIAL], needed_hint=NEEDED)
        a = analyze_series(w, ds, mask=m)
        out[label] = a
    return out


# --------------------------------------------------------------------------- 사례 B
def _b_wide() -> pd.DataFrame:
    tiers = B.load_tiers()
    can = tiers.loc[tiers.tier == "가능", "biz"].tolist()
    sure_no = tiers.loc[tiers.tier == "불가", "biz"].tolist()
    df = duckdb.execute(
        """
        SELECT strptime(ym || '01', '%Y%m%d')::DATE AS month, replace(biz, '?', '') AS biz, ut, sum(amt)::DOUBLE AS amt
        FROM read_csv(?, header=false, skip=1, delim=',',
             columns={'ym':'VARCHAR','biz':'VARCHAR','ut':'VARCHAR','sex':'VARCHAR',
                      'users':'BIGINT','cnt':'BIGINT','amt':'BIGINT'})
        WHERE ut IN ('제주도민', '법인') GROUP BY ALL
        """, [str(B.SRC)]).df()
    df["month"] = pd.to_datetime(df["month"])

    def s(ut, sectors):
        return df[(df.ut == ut) & df.biz.isin(sectors)].groupby("month").amt.sum()

    w = pd.DataFrame({
        "도민_가능": s("제주도민", can), "도민_불가": s("제주도민", sure_no),
        "도민_대형종합": s("제주도민", ["기타 대형 종합 소매업"]),
        "도민_불가면세": s("제주도민", sure_no + ["면세점"]),
        "법인_가능": s("법인", can), "법인_불가": s("법인", sure_no),
    }).sort_index()
    return w.asfreq("MS")


def case_b() -> dict:
    lock = _check_lock(B_PREREG, B_LOCK)
    w = _b_wide()
    ds = SeriesDesign("1차 긴급재난지원금(2020-05)이 사용 가능 업종의 제주도민 카드 소비를 늘렸는가?", "도민_가능", "도민_불가",
                      "2020-05-01", "MS", reference_effect=0.10, pre=3, post=4, season="sdiff", season_years=1,
                      placebo_groups=[("법인_가능", "법인_불가")],
                      alt_references={"대형종합만": ["도민_대형종합"], "대형종합+유흥+면세": ["도민_불가면세"]},
                      exclude=[("2021-09-01", "2021-12-01"), ("2022-09-01", "2022-12-01")], quality=False)
    a = analyze_series(w, ds)
    # 교차 계산: 사례 스크립트의 '대형종합+유흥, 기준 3개월' 5월 점프와 같은 정의(사후 5월 1개월)로 엔진을 돌린다
    one = analyze_series(w, SeriesDesign("교차 계산", "도민_가능", "도민_불가", "2020-05-01", "MS", pre=3, post=1,
                                         season="sdiff", season_years=1, quality=False), placebo=False)
    card = B.load_card()
    tiers = B.load_tiers()
    treat = tiers.loc[tiers.tier == "가능", "biz"].tolist()
    ctrl = tiers.loc[tiers.tier == "불가", "biz"].tolist()
    j3 = float(B.jump(B.triple_diff(card, treat, ctrl, use_tourist=False), 3).loc[B.POLICY_MONTH])
    cross = {"engine": one.main.effect, "case_script": j3, "match": bool(abs(one.main.effect - j3) < 1e-6),
             "definition": "사후 2020-05, 사전 2020-02~04, 전년 동월 차분, 대조 = 대형종합+유흥"}
    return {"analysis": a, "cross": cross, "lock": lock}


# --------------------------------------------------------------------------- 사례 D
def case_d() -> dict:
    lock = D.check_lock()
    p = D.build_panel(D.load(), D.PRE_AMEND1)
    pdz = PanelDesign("중국 단체관광 재개(2023-08-10) 이후 제주의 중국인 카드 소비가 다른 외국인 대비 늘었는가?",
                      "y", "unit", "month", "grp", "중국", "2023-08-01", ref="2023-07-01", fe=["unit", "sm"],
                      reference_effect=D.MDE_LIMIT,
                      sens={"S1 사후 2023-09부터(8월 제외)": {"filter": lambda d: d[d.month != pd.Timestamp("2023-08-01")]},
                            "S2 2019 연간 금액 가중": {"weights": "w2019"},
                            "S3 면세점 제외": {"filter": lambda d: d[d.biz != D.DUTY_FREE]}})
    a = analyze_panel(p, pdz)
    # 규칙 v1 (사전 등록 원안 규칙): 업종 군집 표준오차·이벤트스터디 결합 검정·'모든 가짜 효과보다 큼' 기준
    pz = p.assign(_tr=(p.grp == "중국").astype(int))
    from jverdict.causal.estimate import did, event_study
    m_cr1 = did(pz, "y", "unit", "month", "_tr", "2023-08-01", fe=["unit", "sm"], bootstrap=True)
    es = event_study(pz, "y", "unit", "month", "_tr", "2023-08-01", "2023-07-01", fe=["unit", "sm"])
    space_v1 = all(m_cr1.effect > v["effect"] for v in a.placebo_space.fakes.values())
    rec = json.loads((OUTPUT_DIR / "cases" / "china2023" / "summary.json").read_text(encoding="utf-8"))["amend1"]
    sens_v1 = rec["C3_sensitivity"]["n_pass"]      # v1 민감도는 업종 군집 구간 기준(기록값)
    v1 = judge_v1(m_cr1, pretrend_p=es.pretrend_p, placebo_space_pass=space_v1,
                  sens=type("S", (), {"n_pass": sens_v1})(), mde_log=2.8 * m_cr1.se, reference_pct=D.MDE_LIMIT)
    cross = {"engine_beta": m_cr1.effect, "engine_se": m_cr1.se, "recorded_beta": rec["main"]["beta"],
             "recorded_se": rec["main"]["se"],
             "match": bool(abs(m_cr1.effect - rec["main"]["beta"]) < 1e-9 and abs(m_cr1.se - rec["main"]["se"]) < 1e-9),
             "wild_bootstrap": m_cr1.details.get("wild_bootstrap")}
    return {"analysis": a, "v1_engine": v1, "v1_recorded": rec["verdict"], "cross": cross, "lock": lock,
            "event_study": es}


# --------------------------------------------------------------------------- 보고
def run() -> dict:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    a, b, d = case_a(), case_b(), case_d()
    res = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "rules_version": RULES_VERSION, "rule_history": RULE_HISTORY,
        "A": {k: v.to_dict() for k, v in a.items()},
        "B": {"analysis": b["analysis"].to_dict(), "cross": b["cross"], "prereg_sha256": b["lock"]["sha256"]},
        "D": {"analysis": d["analysis"].to_dict(), "v1_engine": d["v1_engine"], "v1_recorded": d["v1_recorded"],
              "cross": d["cross"], "prereg_sha256": d["lock"]["sha256"]},
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "real_verdicts.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "real_verdicts.md").write_text(report_md(a, b, d, res), encoding="utf-8")
    return res


def report_md(a, b, d, res) -> str:
    L = ["# 실데이터 사례 판정 (인과 엔진, 판정 규칙 v2)", "", f"생성 {res['generated']} · 소요 {res['seconds']}초", "",
         "| 사례 | 판정 | 핵심 근거 |", "|---|---|---|"]
    st = a["정류소 파일"]
    L.append(f"| A 청소년 버스 무료 | {st.verdict.light} {st.verdict.label} | {st.verdict.reasons[0][:90]} |")
    L.append(f"| B 1차 재난지원금 | {b['analysis'].verdict.light} {b['analysis'].verdict.label} | "
             f"{b['analysis'].verdict.reasons[0][:90]} |")
    L.append(f"| D 중국 단체관광 재개 | {d['analysis'].verdict.light} {d['analysis'].verdict.label} "
             f"(기록된 v1 판정: {d['v1_recorded']}) | {d['analysis'].verdict.reasons[0][:90]} |")
    L += ["", "## 사례 A: 청소년 버스 무료(2025-08-01)", ""]
    for k, an in a.items():
        L += [f"### {k}", "", "```", an.report_ko(), "```", ""]
    L += ["## 사례 B: 1차 긴급재난지원금(2020-05)", "",
          f"사전 등록 `{B_PREREG.relative_to(ROOT)}` (SHA-256 {b['lock']['sha256'][:16]}…, F5 결과 열람 후 작성임을 고지)", "",
          "```", b["analysis"].report_ko(), "```", "",
          f"- 교차 계산({b['cross']['definition']}): 엔진 {b['cross']['engine']:.6f} / 사례 스크립트 {b['cross']['case_script']:.6f} → "
          f"{'일치' if b['cross']['match'] else '불일치'}", ""]
    c = d["cross"]
    wb = c.get("wild_bootstrap") or {}
    L += ["## 사례 D: 중국 단체관광 재개(2023-08-10)", "",
          f"- 기록된 판정(사전 등록 원안 규칙 v1): **{d['v1_recorded']}** — 엔진으로 v1 규칙을 다시 적용한 결과: {d['v1_engine']}",
          f"- 교차 계산: 엔진 β {c['engine_beta']:.6f}, SE {c['engine_se']:.6f} / 기록 β {c['recorded_beta']:.6f}, "
          f"SE {c['recorded_se']:.6f} → {'일치' if c['match'] else '불일치'}",
          f"- 와일드 군집 부트스트랩(업종 군집 {wb.get('treated_clusters', '-')}개 처치): p={wb.get('p', float('nan')):.4f}", "",
          "```", d["analysis"].report_ko(), "```", "",
          "> 사례 D의 기록된 판정은 바꾸지 않는다. 규칙 v2 판정은 규칙을 보완한 뒤의 재판정으로 나란히 보고한다.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    r = run()
    print((OUT / "real_verdicts.md").read_text(encoding="utf-8"))
