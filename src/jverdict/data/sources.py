"""원본 CSV 지문 기록 (data/processed/_sources.json).

배포 묶음에는 1GB가 넘는 버스 원본 CSV 대신, 그 원본에서 만든 가공 파일(parquet)을 싣는다.
재실행 기록(manifest)의 데이터 지문은 원본 CSV 기준이므로, 원본이 없는 PC에서도 같은 지문을 확인할 수 있게
'원본 지문 ↔ 가공 파일 지문' 짝을 원본이 있는 PC에서 미리 적어 둔다.

원본이 없을 때 catalog.fingerprint는 가공 파일의 지문을 다시 계산해 이 기록과 맞춰 본다.
맞으면 기록된 원본 지문을 쓰고, 다르면(가공 파일이 바뀌었거나 손상됨) 오류로 멈춘다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jverdict.config import PROCESSED_DIR, RAW_DIR, now_iso

SOURCES_FILE = PROCESSED_DIR / "_sources.json"

# 원본 CSV → Agent가 실제로 읽는 가공 파일 (catalog._load_bus와 같은 짝)
PROCESSED_OF = {
    "이용자유형별버스정류소이용인원현황.csv": "bus_station_usertype_daily.parquet",
    "일별버스승차인원수현황.csv": "bus_route_usertype_daily.parquet",
}


class SourceMismatch(RuntimeError):
    pass


_HASHES: dict[tuple, str] = {}


def sha256_file(p: Path) -> str:
    """파일 지문. 같은 파일(경로·크기·수정 시각이 같음)은 한 번만 계산한다(원본 CSV는 수백 MB)."""
    st = p.stat()
    k = (str(Path(p).resolve()), st.st_size, st.st_mtime_ns)
    if k not in _HASHES:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        _HASHES[k] = h.hexdigest()
    return _HASHES[k]


def build() -> dict:
    """원본이 있는 PC에서: 원본·가공 파일 지문 짝을 만든다(원본이 없으면 그 항목은 건너뜀)."""
    rec: dict = {"note": "원본 CSV 지문과, 그 원본에서 만든 가공 파일 지문의 짝. jverdict.data.sources 참고",
                 "created": now_iso(), "sources": {}}
    for src, proc in PROCESSED_OF.items():
        rp, pp = RAW_DIR / src, PROCESSED_DIR / proc
        if not (rp.exists() and pp.exists()):
            continue
        rec["sources"][src] = {"bytes": rp.stat().st_size, "sha256": sha256_file(rp),
                               "processed": {"file": proc, "bytes": pp.stat().st_size, "sha256": sha256_file(pp)}}
    return rec


def write(path: Path | None = None) -> Path:
    """기록을 새로 쓴다. 기존 기록과 원본 지문이 다르면(원본이 바뀜) 그대로 덮어쓰되 바뀐 항목을 알려 준다."""
    path = path or SOURCES_FILE
    new = build()
    old = read(path)
    if old:
        for src, r in new["sources"].items():
            o = old.get("sources", {}).get(src)
            if o and (o["sha256"] != r["sha256"] or o["processed"]["sha256"] != r["processed"]["sha256"]):
                print(f"[주의] {src}: 지문이 바뀜(원본 또는 가공 파일을 다시 만든 것으로 보임)")
        # 이 PC에 원본이 없어 다시 잴 수 없는 항목은 예전 기록을 그대로 둔다
        new["sources"] = {**{k: v for k, v in old.get("sources", {}).items() if k not in new["sources"]},
                          **new["sources"]}
    path.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read(path: Path | None = None) -> dict | None:
    path = path or SOURCES_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def raw_fingerprint_via_processed(source: str) -> dict:
    """원본 없이 원본 지문을 확인한다: 가공 파일 지문이 기록과 같을 때만 기록된 원본 지문을 돌려준다."""
    rec = read()
    if rec is None or source not in rec.get("sources", {}):
        raise FileNotFoundError(f"원본 파일이 없고, 가공 파일 지문 기록(_sources.json)에도 없음: {source}")
    r = rec["sources"][source]
    pp = PROCESSED_DIR / r["processed"]["file"]
    if not pp.exists():
        raise FileNotFoundError(f"원본 파일과 가공 파일이 모두 없음: {source}")
    if pp.stat().st_size != r["processed"]["bytes"] or sha256_file(pp) != r["processed"]["sha256"]:
        raise SourceMismatch(f"가공 파일 지문이 기록과 다름(파일이 바뀌었거나 손상됨): {r['processed']['file']}")
    return {"bytes": r["bytes"], "sha256": r["sha256"]}


if __name__ == "__main__":
    p = write()
    rec = read(p)
    for src, r in rec["sources"].items():
        print(f"{src}: 원본 {r['bytes']:,} B · 가공 {r['processed']['file']} {r['processed']['bytes']:,} B")
    print(f"기록: {p}")
