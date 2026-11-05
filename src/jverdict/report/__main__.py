"""1장 판정서 PDF 만들기.

    python -m jverdict.report outputs/runs/case-C-template      # 한 실행
    python -m jverdict.report --all                            # 끝난 실행 전부(재실행 폴더 제외)
결과: <실행 폴더>/verdict.pdf (브라우저가 없으면 verdict.html만)
"""
import argparse
import sys
from pathlib import Path

from jverdict.config import OUTPUT_DIR, ROOT

from .onepage import make
from .view import list_runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jverdict.report")
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    dirs = list(a.run_dirs)
    if a.all:
        dirs += [d for d in list_runs(OUTPUT_DIR / "runs") if not d.name.endswith("-replay")]
    if not dirs:
        ap.print_help()
        return 1
    bad = 0
    for d in sorted(set(dirs)):
        d = d if d.is_absolute() else ROOT / d
        try:
            r = make(d)
        except Exception as e:  # noqa: BLE001
            print(f"{d.name}: 실패 - {type(e).__name__}: {e}", file=sys.stderr)
            bad += 1
            continue
        tgt = r["pdf"] or r["html"]
        warn = "" if (r["pages"] or 1) <= 1 else f"  (경고: {r['pages']}쪽)"
        print(f"{tgt.relative_to(ROOT)}  {r['pages'] or '-'}쪽 · 글자 {r['font_pt']}pt{warn}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
