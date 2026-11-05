"""사전 등록 잠금(prereg.lock): 설계 문서와 설계 JSON의 지문(SHA-256)과 시각을 남긴다.

- 잠근 뒤에만 효과 추정을 실행할 수 있다(Run.require_lock).
- 실행 직전에 지문을 다시 계산해 다르면 멈춘다.
- 잠근 뒤 설계를 바꾸면(오류 복구 등) 변경 기록에 이유와 새 지문을 남기고, 원안 결과를 함께 보고한다.
- 기존 사전 등록 문서(docs/prereg/*.md)를 쓰는 사례는 그 문서의 잠금 파일 지문도 확인한다.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from jverdict.config import ROOT, now_iso


class GuardError(RuntimeError):
    """상태머신 가드 위반: 순서를 건너뛰거나 잠금이 깨짐."""


def _digest(md: str, design: dict) -> str:
    blob = md.encode("utf-8") + b"\n---\n" + json.dumps(design, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def verify_existing(prereg_rel: str) -> dict:
    """docs/prereg의 기존 등록 문서가 잠금 이후 바뀌지 않았는지 확인한다."""
    doc = ROOT / prereg_rel
    lock = doc.with_suffix(".lock.json")
    if not doc.exists() or not lock.exists():
        raise GuardError(f"사전 등록 문서 또는 잠금 파일이 없음: {prereg_rel}")
    rec = json.loads(lock.read_text(encoding="utf-8"))
    now = hashlib.sha256(doc.read_bytes()).hexdigest()
    if now != rec["sha256"]:
        raise GuardError(f"사전 등록 문서가 잠금 이후 바뀜: {prereg_rel}")
    return {"file": prereg_rel, "sha256": now, "locked_at": rec.get("locked_at")}


def lock(run_dir: Path, md: str, design: dict, existing: dict | None = None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prereg.md").write_text(md, encoding="utf-8")
    (run_dir / "design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")
    rec = {"sha256": _digest(md, design), "locked_at": now_iso(),
           "existing_prereg": existing, "amendments": []}
    (run_dir / "prereg.lock.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def verify(run_dir: Path) -> dict:
    rec = json.loads((run_dir / "prereg.lock.json").read_text(encoding="utf-8"))
    md = (run_dir / "prereg.md").read_text(encoding="utf-8")
    design = json.loads((run_dir / "design.json").read_text(encoding="utf-8"))
    expect = rec["amendments"][-1]["sha256"] if rec["amendments"] else rec["sha256"]
    if _digest(md, design) != expect:
        raise GuardError("설계 문서가 잠금 이후 기록 없이 바뀜. 분석을 멈춤")
    if rec.get("existing_prereg"):
        verify_existing(rec["existing_prereg"]["file"])
    return rec


def amend(run_dir: Path, new_design: dict, reason: str) -> dict:
    """잠금 뒤 설계 변경: 변경 기록을 문서 끝에 붙이고 새 지문을 남긴다."""
    rec = json.loads((run_dir / "prereg.lock.json").read_text(encoding="utf-8"))
    old = json.loads((run_dir / "design.json").read_text(encoding="utf-8"))
    changed = sorted(k for k in set(old) | set(new_design) if old.get(k) != new_design.get(k))
    md = (run_dir / "prereg.md").read_text(encoding="utf-8")
    n = len(rec["amendments"]) + 1
    md += (f"\n## 변경 기록 {n} ({time.strftime('%Y-%m-%d %H:%M')}, 잠금 이후)\n- 이유: {reason}\n"
           f"- 바뀐 항목: {', '.join(changed) or '없음'}\n- 원안 설계의 결과도 판정서에 함께 보고한다.\n")
    (run_dir / "prereg.md").write_text(md, encoding="utf-8")
    (run_dir / "design.json").write_text(json.dumps(new_design, ensure_ascii=False, indent=2), encoding="utf-8")
    rec["amendments"].append({"n": n, "reason": reason, "changed": changed, "sha256": _digest(md, new_design),
                              "at": now_iso()})
    (run_dir / "prereg.lock.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec
