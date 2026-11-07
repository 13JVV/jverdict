"""실데이터 품질 점검 (사례 A와 사례 D의 데이터 점검 단계).

허브 공개 데이터에 품질 툴을 돌려, 효과를 추정하기 전에 무엇을 믿을 수 있는지 정리한다.
1. 버스 정류소 파일 커버리지 (전체 결측, 부분 집계, 분류 신설)
2. 청소년 무료화(2025-08-01) 시점 측정 단절 진단: 정책 대상(청소년·어린이)과 비대상(경로·장애)
3. 같은 진단을 노선별 파일, 일별 시간대별 파일, 월별 패널티 파일에서 반복 (파일 간 일관성)
4. 시점을 모르는 단절 탐색 (전 집단)
5. 카드 파일 커버리지 (사례 D의 2023-05·06 외국인 결측)

실행: .venv/bin/python -m jverdict.cases.quality_real
결과: outputs/cases/quality/real_quality.json, real_quality.md
"""
from __future__ import annotations

import json
import time

import duckdb
import pandas as pd

from jverdict.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR
from jverdict.quality._panel import to_wide
from jverdict.quality.breaks import diagnose, scan_breaks
from jverdict.quality.coverage import check_coverage, exclusion_mask
from jverdict.quality.external import ExternalStat

OUT = OUTPUT_DIR / "cases" / "quality"
POLICY = "2025-08-01"
OFFICIAL = ExternalStat(
    "어린이·청소년 무료 이용 누적 483만 건(도 발표, 2025-08~2026-02)", 4_830_000,
    "2025-08-01", "2026-02-28", ["청소년", "어린이"], "제주도 2026-04-05 발표(헤드라인제주·제주일보 보도)",
    same_definition=False)
NEEDED = ["교통복지카드(무료) 승차 기록이 포함된 이용자 유형별 승차 자료",
          "2025-08-01 전후 이용자 유형 분류 기준 변경 이력(데이터 제공처 확인)"]


def _q(sql: str) -> pd.DataFrame:
    return duckdb.sql(sql).df()


def load_station() -> pd.DataFrame:
    return _q(f"select base_date, user_type, station_name, sum(user_count) user_count "
              f"from '{PROCESSED_DIR / 'bus_station_usertype_daily.parquet'}' group by all")


def load_route() -> pd.DataFrame:
    return _q(f"select base_date, user_type, bus_number, sum(user_count) user_count "
              f"from '{PROCESSED_DIR / 'bus_route_usertype_daily.parquet'}' where act_type = '승차' group by all")


def load_hourband() -> pd.DataFrame:
    """일별 시간대별 파일: 시간대를 단위로 두고 하루 합계를 본다(2023-12-25부터라 사전 기간이 짧다)."""
    return _q(f"select base_date, user_type, base_hour, sum(user_count) user_count "
              f"from '{PROCESSED_DIR / 'bus_hourband_usertype_daily.parquet'}' group by all")


def hour_profile(pre=("2025-05-01", "2025-07-31"), post=("2025-08-01", "2025-10-25")) -> list[dict]:
    """시간대별로 청소년 변화(일반 대비). 측정 단절이면 통학 시간대뿐 아니라 모든 시간대가 비슷하게 줄어든다."""
    df = _q(f"""select base_hour h, user_type u,
        avg(case when base_date between '{pre[0]}' and '{pre[1]}' then c end) pre,
        avg(case when base_date between '{post[0]}' and '{post[1]}' then c end) post
      from (select base_date, base_hour, user_type, sum(user_count) c
            from '{PROCESSED_DIR / 'bus_hourband_usertype_daily.parquet'}' group by all)
      where u in ('청소년', '일반') group by all order by h""")
    w = df.pivot(index="h", columns="u", values=["pre", "post"])
    out = []
    for h in w.index:
        t = w.loc[h, ("post", "청소년")] / w.loc[h, ("pre", "청소년")]
        g = w.loc[h, ("post", "일반")] / w.loc[h, ("pre", "일반")]
        out.append({"hour": h, "teen_change": round(float(t - 1), 4), "general_change": round(float(g - 1), 4),
                    "teen_vs_general": round(float(t / g - 1), 4)})
    return out


def load_penalty() -> pd.DataFrame:
    return _q(f"select base_month, user_type, trans_type, sum(user_count) user_count "
              f"from '{PROCESSED_DIR / 'bus_penalty_monthly.parquet'}' group by all")


def load_card() -> pd.DataFrame:
    return _q(f"select strptime(base_year_month, '%Y%m') t, user_type, biz_nm, sum(cast(use_amount as bigint)) amt "
              f"from read_csv('{RAW_DIR / '업종별성별카드이용정보.csv'}', all_varchar=true) group by all")


def _policy_block(df, time_col, unit_col, freq, label, with_external=True):
    cov = check_coverage(df, time_col, "user_type", "user_count", unit=unit_col, freq=freq)
    w = to_wide(df, time_col, "user_type", "user_count", freq)
    m = exclusion_mask(cov, w.index)
    out = {"file": label, "coverage": cov.to_dict(), "diagnoses": {}}
    for g, exp in [("청소년", "increase"), ("어린이", "increase"), ("경로", None), ("장애 일반", None)]:
        if g not in w.columns:
            continue
        dg = diagnose(w, g, "일반", POLICY, freq, expected=exp,
                      external=[OFFICIAL] if (with_external and exp) else None, mask=m,
                      needed_hint=NEEDED if exp else None)
        out["diagnoses"][g] = dg
    return out, cov, w, m


def run() -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    st, st_cov, st_w, st_m = _policy_block(load_station(), "base_date", "station_name", "D",
                                           "이용자 유형별 버스정류소 이용 인원 현황")
    rt, *_ = _policy_block(load_route(), "base_date", "bus_number", "D", "일별 버스 승차 인원 수 현황")
    hb, *_ = _policy_block(load_hourband(), "base_date", "base_hour", "D", "일별 시간대별 버스 승차 인원 수 현황")
    pn, *_ = _policy_block(load_penalty(), "base_month", "trans_type", "MS", "월별 버스 이용자 유형별 패널티 현황",
                           with_external=False)

    scans = {}
    for g in st_w.columns:
        # 하루 평균 100명 미만의 아주 작은 집단은 잡음이 커서 탐색에서 뺀다
        if g == "일반" or st_w[g].notna().sum() < 400 or st_w[g].median() < 100:
            continue
        scans[g] = [c.to_dict() for c in scan_breaks(st_w, g, "일반", "D", mask=st_m)]
    jan = diagnose(st_w, "어린이", "일반", "2025-01-01", "D", mask=st_m)

    card = load_card()
    card_cov = check_coverage(card, "t", "user_type", "amt", unit="biz_nm", freq="MS")

    res = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "policy_date": POLICY,
        "files": [
            {**{k: v for k, v in blk.items() if k != "diagnoses"},
             "diagnoses": {g: d.to_dict() for g, d in blk["diagnoses"].items()}}
            for blk in (st, rt, hb, pn)],
        "scans_station": scans,
        "child_2025_01": jan.to_dict(),
        "card_coverage": card_cov.to_dict(),
        "teen_by_hour": hour_profile(),
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "real_quality.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "real_quality.md").write_text(report_md(st, rt, hb, pn, scans, jan, card_cov, res), encoding="utf-8")
    return res


def report_md(st, rt, hb, pn, scans, jan, card_cov, res) -> str:
    L = ["# 실데이터 품질 점검 결과", "", f"생성 {res['generated']} · 소요 {res['seconds']}초", "",
         "## 1. 버스 정류소 파일 커버리지", ""]
    L += [f"- [{f['severity']}] {f['kind']}: {f['message']}" for f in st["coverage"]["findings"]]
    L += ["", f"## 2. 청소년 무료화({POLICY}) 시점 측정 단절 진단", "",
          "| 파일 | 집단 | 정책 대상 | 진단 | 변화(대조 대비) | 위약 p | 판정 상한 |", "|---|---|---|---|---|---|---|"]
    for blk in (st, rt, hb, pn):
        for g, d in blk["diagnoses"].items():
            t = d.test
            chg = "-" if t.pct is None else f"{t.pct * 100:+.0f}%"
            p = "-" if t.p_rank is None else f"{t.p_rank:.3f}"
            L.append(f"| {blk['file']} | {g} | {'예' if g in ('청소년', '어린이') else '아니오'} | {d.verdict} | {chg} | {p} | {d.judgement_cap or '-'} |")
    L += ["", "### 시간대별 청소년 변화 (일별 시간대별 파일, 시행 전 2025-05~07 대 시행 후 08~10-25 하루 평균)", "",
          "측정 단절이라면 통학 시간대만이 아니라 모든 시간대가 비슷하게 줄어든다.", "",
          "| 시간대 | 청소년 | 일반 | 청소년(일반 대비) |", "|---|---|---|---|"]
    L += [f"| {x['hour']} | {x['teen_change'] * 100:+.0f}% | {x['general_change'] * 100:+.0f}% | {x['teen_vs_general'] * 100:+.0f}% |"
          for x in res["teen_by_hour"]]
    L += ["", "### 청소년 상세 (정류소 파일)", "", "```", st["diagnoses"]["청소년"].report_ko(), "```", "",
          "## 3. 시점을 모르는 단절 탐색 (정류소 파일, 대조: 일반)", "",
          "| 집단 | 시작 추정 | 변화 | 강건 z | 모양 | 지속성 | 분류 변경 근거 |", "|---|---|---|---|---|---|---|"]
    for g, cs in scans.items():
        for c in cs:
            rc = "; ".join(r["kind"] + f"({r['group']})" for r in c["reclass"]) or "-"
            L.append(f"| {g} | {c['date']} | {c['pct'] * 100:+.0f}% | {c['z']:.1f} | {c['shape']} | {c['persistence']} | {rc} |")
    L += ["", "- 2020-03 ~ 2021-03 후보는 코로나19로 등교가 중단·재개되던 시기와 겹친다. 실제 이동 변화일 수 있어 원인 확인이 필요하다.",
          "- 하루 평균 100명 미만인 집단(유공 동반 등)은 잡음이 커서 탐색에서 뺐다."]
    L += ["", "### 2025-01-01 어린이 (정책 정보 없이 진단)", "", "```", jan.report_ko(), "```", "",
          "## 4. 카드 파일 커버리지 (사례 D)", ""]
    L += [f"- [{f['severity']}] {f['kind']}: {f['message']}" for f in card_cov.to_dict()["findings"]] or ["- 이상 없음"]
    L += ["", "> 원인은 단정하지 않는다. 모든 항목은 데이터 제공처 확인이 필요한 '의심'이다."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    r = run()
    print((OUT / "real_quality.md").read_text(encoding="utf-8"))
