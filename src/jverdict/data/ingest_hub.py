"""제주데이터허브 공개 CSV → 정제된 parquet.

허브 파일은 따옴표 없이 쉼표가 들어간 값이 있다(예: 정류소명 '제주국제공항(대정,화순,일주서로)').
그래서 CSV 파서 대신 한 줄 전체를 읽고, 앞뒤 고정 필드를 정규식으로 잘라낸다.
가운데 필드(쉼표를 품을 수 있는 필드)는 '.*'로 받는다.

실행:  python -m jverdict.data.ingest_hub
결과:  data/processed/*.parquet, data/processed/_ingest_report.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

from jverdict.config import PROCESSED_DIR, RAW_DIR


@dataclass(frozen=True)
class HubFile:
    key: str                 # 출력 parquet 이름
    source: str              # 허브 원본 파일명
    title: str               # 허브 데이터명 (데이터 활용 명세에 그대로 쓴다)
    pattern: str             # 한 줄 전체에 맞는 정규식 (그룹 순서 = columns 순서)
    columns: tuple[str, ...]
    casts: dict[str, str]    # 컬럼별 DuckDB 변환식, {c}가 원본 문자열


DATE = "strptime({c}, '%Y%m%d')::DATE"
YM = "strptime({c} || '01', '%Y%m%d')::DATE"
INT = "TRY_CAST({c} AS BIGINT)"

HUB_FILES: tuple[HubFile, ...] = (
    HubFile(
        key="bus_station_usertype_daily",
        source="이용자유형별버스정류소이용인원현황.csv",
        title="이용자 유형별 버스정류소 이용 인원 현황",
        pattern=r"^(\d{8}),(.*),([^,]+),(\d+)$",
        columns=("base_date", "station_name", "user_type", "user_count"),
        casts={"base_date": DATE, "user_count": INT},
    ),
    HubFile(
        key="bus_route_usertype_daily",
        source="일별버스승차인원수현황.csv",
        title="일별 버스 승차 인원 수 현황",
        pattern=r"^(\d{8}),([^,]+),(.*),([^,]+),(\d+)$",
        columns=("base_date", "act_type", "bus_number", "user_type", "user_count"),
        casts={"base_date": DATE, "user_count": INT},
    ),
    HubFile(
        key="bus_hourband_usertype_daily",
        source="일별시간대별버스승차인원수현황.csv",
        title="일별 시간대별 버스 승차 인원 수 현황",
        pattern=r"^(\d{8}),([^,]+),([^,]+),([^,]+),(\d+)$",
        columns=("base_date", "day_of_week", "base_hour", "user_type", "user_count"),
        casts={"base_date": DATE, "user_count": INT},
    ),
    HubFile(
        key="bus_penalty_monthly",
        source="월별버스이용자유형별패널티현황.csv",
        title="월별 버스 이용자 유형별 패널티 현황",
        pattern=r"^(\d{6}),(.*),([^,]+),(\d+),(\d+)$",
        columns=("base_month", "trans_type", "user_type", "penalty_count", "user_count"),
        casts={"base_month": YM, "penalty_count": INT, "user_count": INT},
    ),
)


def ingest_file(con: duckdb.DuckDBPyConnection, spec: HubFile, raw_dir: Path, out_dir: Path) -> dict:
    src = raw_dir / spec.source
    if not src.exists():
        return {"key": spec.key, "status": "원본 없음", "source": spec.source}

    names = [f"g{i}" for i in range(len(spec.columns))]
    select_parts = []
    for i, col in enumerate(spec.columns):
        expr = spec.casts.get(col, "{c}").format(c=f"m.{names[i]}")
        select_parts.append(f"{expr} AS {col}")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE parsed AS
        WITH raw AS (
            SELECT line FROM read_csv(?, delim='\\x1f', quote='', escape='', header=false,
                                      skip=1, columns={{'line': 'VARCHAR'}})
        )
        SELECT regexp_extract(line, ?, {names!r}) AS m, line FROM raw
        """,
        [str(src), spec.pattern],
    )
    total, bad = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE m.{names[0]} = '') FROM parsed"
    ).fetchone()
    bad_samples = [r[0][:120] for r in con.execute(
        f"SELECT line FROM parsed WHERE m.{names[0]} = '' LIMIT 5").fetchall()]

    out = out_dir / f"{spec.key}.parquet"
    con.execute(
        f"COPY (SELECT {', '.join(select_parts)} FROM parsed WHERE m.{names[0]} <> '') "
        f"TO '{out.as_posix()}' (FORMAT PARQUET)"
    )
    first_col = spec.columns[0]
    lo, hi = con.execute(f"SELECT min({first_col}), max({first_col}) FROM '{out.as_posix()}'").fetchone()
    return {
        "key": spec.key,
        "status": "완료",
        "source": spec.source,
        "title": spec.title,
        "rows_read": int(total),
        "rows_rejected": int(bad),
        "rejected_samples": bad_samples,
        "period": [str(lo), str(hi)],
        "output": out.name,
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    report = [ingest_file(con, spec, RAW_DIR, PROCESSED_DIR) for spec in HUB_FILES]
    (PROCESSED_DIR / "_ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for r in report:
        if r["status"] != "완료":
            print(f"- {r['key']}: {r['status']} ({r['source']})")
            continue
        print(f"- {r['key']}: {r['rows_read']:,}행 읽음, 제외 {r['rows_rejected']:,}행, "
              f"기간 {r['period'][0]} ~ {r['period'][1]}")


if __name__ == "__main__":
    main()
