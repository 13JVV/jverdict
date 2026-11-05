"""제주데이터허브 '버스 정류소 기본 정보' API를 내려받아 CSV로 저장한다.

사용법:
  1) 제주데이터허브에서 '버스 정류소 기본 정보' → [프로젝트 담기] → 마이페이지에서 프로젝트 키 확인
  2) 프로젝트 루트의 .env 파일에 한 줄 추가:  JEJU_HUB_PROJECT_KEY=발급받은키
  3) .venv/bin/python tools/fetch_stations.py
결과: data/processed/bus_stations.csv
"""
import csv
import json
import os
import sys
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://open.jejudatahub.net/api/proxy/DD11ab6a6t11D16baaa1a2tD26ata161/{key}"
OUT = ROOT / "data" / "processed" / "bus_stations.csv"


def load_key() -> str:
    key = os.environ.get("JEJU_HUB_PROJECT_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("JEJU_HUB_PROJECT_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit(".env에 JEJU_HUB_PROJECT_KEY가 없습니다. 파일 상단 사용법을 참고하세요.")
    return key


def get(url: str) -> dict:
    """요청 URL에 프로젝트 키가 들어가므로, 오류가 나도 URL과 원본 예외를 출력하지 않는다."""
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"허브 API 오류: HTTP {e.code}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise SystemExit(f"허브 API 연결 실패: {type(e).__name__}") from None


def rows_of(payload) -> list:
    """응답 형식이 문서로 확인되지 않아, 첫 번째 '딕셔너리 목록'을 찾아 쓴다."""
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload
    if isinstance(payload, dict):
        for v in payload.values():
            found = rows_of(v)
            if found:
                return found
    return []


def main() -> None:
    base = API.format(key=urllib.parse.quote(load_key()))
    # 페이지 나눔: number(페이지 번호, 1부터) + limit(페이지 크기). 응답: {totCnt, hasMore, data}
    # 일일 쿼터가 1,000회라 페이지 크기를 크게 잡는다.
    limit, number, rows, total = 100, 1, {}, None
    while number <= 100:
        payload = get(f"{base}?{urllib.parse.urlencode({'number': number, 'limit': limit})}")
        page = rows_of(payload)
        total = payload.get("totCnt", total) if isinstance(payload, dict) else total
        for r in page:
            rows[r.get("stationId") or json.dumps(r, sort_keys=True)] = r
        if not page or not (isinstance(payload, dict) and payload.get("hasMore")):
            break
        number += 1
    rows = list(rows.values())
    if not rows:
        print("응답에서 데이터 행을 찾지 못했습니다. 응답 앞부분:")
        print(json.dumps(payload, ensure_ascii=False)[:800])
        sys.exit(1)
    if total and len(rows) != total:
        print(f"경고: 응답 총건수 {total}건, 저장 {len(rows)}건")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}개 정류소 저장: {OUT}")


if __name__ == "__main__":
    main()
