"""화면용 백그라운드 실행기.

Agent 상태머신(orchestrator.run)을 별도 스레드에서 돌린다. 정책 카드 확인(confirm) 단계에 오면
스레드가 멈추고 화면이 카드를 보여 준다. 사람이 승인하면 그 카드로 이어서 진행하고, 취소하면 실행을 멈춘다.
확인 단계를 상태머신 밖으로 빼지 않기 때문에 재실행 기록(LLM 응답 순서)이 CLI 실행과 똑같다.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from jverdict.agent import orchestrator as O
from jverdict.agent.llm import sanitize


class Cancelled(RuntimeError):
    pass


class Job:
    """status: running → confirm → running → done | error | cancelled"""

    def __init__(self, question: str | None, mode: str = "template", run_id: str | None = None,
                 confirm_timeout: float = 3600.0, card: dict | None = None):
        self.question, self.mode = question, mode
        self.card_in = card                  # 정책 카드를 직접 만든 경우(질문 해석 단계를 건너뜀)
        self.run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{mode}"
        self.events: list[dict] = []
        self.status = "running"
        self.card: dict | None = None
        self.run_dir: Path | None = None
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None
        self._timeout = confirm_timeout
        self._decided = threading.Event()
        self._decision: dict | None = None
        self._thread = threading.Thread(target=self._work, name=f"jverdict-{self.run_id}", daemon=True)

    # ------------------------------------------------------------------ 화면에서 부르는 것
    def start(self) -> "Job":
        self._thread.start()
        return self

    def approve(self, card: dict) -> None:
        if self.status != "confirm":
            raise RuntimeError("정책 카드 확인 단계가 아님")
        self._decision = card
        self.status = "running"          # 화면이 바로 진행 표시로 넘어가도록(스레드가 깨어나기 전에)
        self._decided.set()

    def cancel(self) -> None:
        self._decision = None
        self._decided.set()

    def wait(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def stage(self) -> str | None:
        return self.events[-1]["stage"] if self.events else None

    @property
    def done_stages(self) -> list[str]:
        started = [e["stage"] for e in self.events if e["msg"].startswith("단계 시작")]
        cur = self.stage
        out = []
        for s in started:
            if s not in out:
                out.append(s)
        if self.status != "done" and cur in out:
            out.remove(cur)
        return out

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    # ------------------------------------------------------------------ 스레드
    def _confirm(self, card: dict) -> dict:
        self.card = card
        self.status = "confirm"
        if not self._decided.wait(self._timeout) or self._decision is None:
            raise Cancelled("정책 카드 확인 단계에서 취소함")
        self.status = "running"
        return self._decision

    def _work(self) -> None:
        try:
            r = O.run(self.question, card=self.card_in, mode=self.mode, confirm=self._confirm,
                      on_event=self.events.append, run_id=self.run_id)
            self.run_dir = r.run_dir
            self.status = "done"
        except Cancelled:
            self.status = "cancelled"
            shutil.rmtree(O.RUNS / self.run_id, ignore_errors=True)     # 이 실행이 만든 미완성 폴더만 지운다
        except Exception as e:  # noqa: BLE001
            self.error = sanitize(f"{type(e).__name__}: {e}")
            self.status = "error"
        finally:
            self.finished = time.time()
