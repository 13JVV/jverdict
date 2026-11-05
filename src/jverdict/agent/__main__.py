"""명령줄 실행.

    python -m jverdict.agent "청소년 버스 무료화로 청소년 버스 이용이 늘었나?"        # 템플릿 모드(기본, 오프라인)
    python -m jverdict.agent --case A --mode claude --yes                          # Claude 사용, 카드 자동 승인
    python -m jverdict.agent --replay outputs/runs/<실행 번호>                       # 같은 조건으로 재실행·대조
    python -m jverdict.agent --backfill-key-norm outputs/runs/<실행 번호>            # 옛 Claude 기록에 정규화 지문 채우기
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from .library import CARDS
from .llm import sanitize
from .orchestrator import backfill_key_norm, replay, run


def _print_event(ev: dict):
    print(f"[{ev['t']}] {ev['stage']:<8} {ev['msg']}", flush=True)


def _confirm_interactive(card: dict) -> dict:
    print("\n정책 카드(확인 후 진행):")
    for k in ("policy", "question", "start_date", "end_date", "treated", "outcome", "expected", "reference_effect"):
        print(f"  {k:>16}: {card.get(k)}")
    ans = input("이 정책 카드로 진행할까요? [Y/n] ").strip().lower()
    if ans in ("n", "no", "아니오"):
        sys.exit("사용자가 정책 카드를 승인하지 않아 멈춤. --card 파일로 고쳐서 다시 실행하세요.")
    return card


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="jverdict.agent")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--case", choices=list(CARDS), help="등록 사례의 질문으로 실행")
    ap.add_argument("--card", type=Path, help="정책 카드 JSON 파일")
    ap.add_argument("--mode", choices=["template", "claude"], default="template")
    ap.add_argument("--yes", action="store_true", help="정책 카드 자동 승인")
    ap.add_argument("--replay", type=Path, help="이전 실행 폴더")
    ap.add_argument("--backfill-key-norm", type=Path, nargs="+", metavar="RUN_DIR",
                    help="옛 Claude 기록에 정규화 지문(key_norm)을 채움(재현될 때만)")
    ap.add_argument("--run-id")
    a = ap.parse_args(argv)
    try:
        if a.backfill_key_norm:
            res = [backfill_key_norm(d) for d in a.backfill_key_norm]
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return 0 if all(r["status"] in ("채움", "채울 것 없음") for r in res) else 1
        if a.replay:
            out = replay(a.replay, run_id=a.run_id)
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0 if out["reproduced"] else 1
        q = a.question or (CARDS[a.case]["card"]["question"] if a.case else None)
        card = json.loads(a.card.read_text(encoding="utf-8")) if a.card else None
        confirm = (lambda c: c) if a.yes else _confirm_interactive
        r = run(q, card=card, mode=a.mode, confirm=confirm, on_event=_print_event, run_id=a.run_id)
    except Exception as e:  # noqa: BLE001
        print("오류:", sanitize(str(e)), file=sys.stderr)
        return 2
    print()
    print((r.run_dir / "report.md").read_text(encoding="utf-8"))
    print(f"결과 폴더: {r.run_dir}")
    m = json.loads((r.run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"LLM 호출 {m['llm']['calls']}회, 비용 ${m['llm']['cost_usd']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
