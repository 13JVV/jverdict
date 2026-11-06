"""사례 E 보조 확인(사전 등록 밖, 판정에 쓰지 않음): 제주도민 카드 소비 중 밤·저녁 시간대 비중.

영업시간 제한이 풀리면 늦은 시간 소비 비중이 늘 것이다. 업종 파일과 다른 파일(시간대별 카드 이용 금액 현황)로
같은 방향이 보이는지, 그리고 사례 E의 기준 기간에 섞인 2021-08이 이 파일에서도 이상한지 본다.
시간대 구분(오전·점심·오후·저녁·밤)의 정확한 시각은 허브 파일에 적혀 있지 않다.

실행: .venv/bin/python -m jverdict.cases.distancing_night
결과: outputs/cases/distancing2022/night_share.json, night_share.md
"""
from __future__ import annotations

import json

import duckdb
import pandas as pd

from jverdict.config import OUTPUT_DIR, RAW_DIR

OUT = OUTPUT_DIR / "cases" / "distancing2022"
SRC = RAW_DIR / "시간대별카드이용금액현황.csv"


def shares() -> pd.DataFrame:
    df = duckdb.sql(f"""
        select strptime(cast(base_year_month as varchar), '%Y%m') t, time_type, sum(use_amout_by_million) amt
        from read_csv('{SRC}', header=true, all_varchar=false)
        where user_type = '제주도민' group by all""").df()
    w = df.pivot(index="t", columns="time_type", values="amt").sort_index()
    tot = w.sum(axis=1)
    return pd.DataFrame({"night": w["밤"] / tot, "evening_night": (w["밤"] + w["저녁"]) / tot})


def summarize() -> dict:
    s = shares()

    def avg(a, b, col):
        return float(s.loc[a:b, col].mean())
    rows = [("2019 같은 달(5~9월, 코로나 전)", "2019-05-01", "2019-09-01"),
            ("2021-08 (거리두기 4단계 무렵)", "2021-08-01", "2021-08-01"),
            ("2022-01~03 (해제 직전, 단계적 완화)", "2022-01-01", "2022-03-01"),
            ("2022-05~09 (전면 해제 뒤)", "2022-05-01", "2022-09-01"),
            ("2023-05~09 (1년 뒤)", "2023-05-01", "2023-09-01")]
    out = [{"period": n, "night": round(avg(a, b, "night"), 4), "evening_night": round(avg(a, b, "evening_night"), 4)}
           for n, a, b in rows]
    return {"source": SRC.name, "group": "제주도민", "note": "사전 등록 밖 보조 확인. 판정에 쓰지 않음", "rows": out}


def report_md(r: dict) -> str:
    L = ["# 사례 E 보조 확인: 제주도민 카드 소비 중 밤·저녁 비중", "",
         f"자료: 「시간대별 카드 이용 금액 현황」({r['source']}), 이용자 구분 제주도민. **사전 등록 밖의 보조 확인이며 판정에 쓰지 않는다.**", "",
         "| 기간 | 밤 비중 | 저녁+밤 비중 |", "|---|---|---|"]
    L += [f"| {x['period']} | {x['night'] * 100:.1f}% | {x['evening_night'] * 100:.1f}% |" for x in r["rows"]]
    L += ["", "- 시간대 구분의 정확한 시각은 허브 파일에 적혀 있지 않다. 월 합계 금액 기준 비중이다.",
          "- 업종 파일과 다른 파일에서 같은 방향(영업시간 제한 해제 뒤 늦은 시간 소비 회복)이 보이는지만 확인한다."]
    return "\n".join(L) + "\n"


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    r = summarize()
    (OUT / "night_share.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "night_share.md").write_text(report_md(r), encoding="utf-8")
    return r


if __name__ == "__main__":
    print(report_md(run()))
