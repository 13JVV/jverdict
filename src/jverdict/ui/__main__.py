"""화면 실행: python -m jverdict.ui [--port 8501] [--no-browser]

- 이 PC 안(localhost)에서만 열린다. Streamlit 사용 통계 전송은 끈다.
- 처음 실행할 때 Streamlit이 터미널에서 이메일을 묻고 멈추는 일이 없게 한다(server.showEmailPrompt=false).
- 지정한 포트를 이미 쓰고 있으면(예: 화면을 두 번 켬) 다음 빈 포트를 찾아 연다.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

from jverdict.config import ROOT


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(start: int, tries: int = 20) -> int:
    for p in range(start, start + tries):
        if port_free(p):
            return p
    raise RuntimeError(f"{start}~{start + tries - 1} 포트가 모두 사용 중입니다. 열려 있는 화면 창을 닫고 다시 실행하세요.")


def build_cmd(port: int, open_browser: bool) -> list[str]:
    app = Path(__file__).with_name("app.py")
    return [sys.executable, "-X", "utf8", "-m", "streamlit", "run", str(app),
            "--server.address=localhost", f"--server.port={port}",
            "--browser.gatherUsageStats=false", "--client.showErrorDetails=type",
            "--server.showEmailPrompt=false", "--server.fileWatcherType=none", "--server.runOnSave=false",
            f"--server.headless={'false' if open_browser else 'true'}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jverdict.ui")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않음")
    a = ap.parse_args(argv)
    try:
        port = pick_port(a.port)
    except RuntimeError as e:
        print(f"[오류] {e}", file=sys.stderr)
        return 1
    if port != a.port:
        print(f"[안내] {a.port} 포트를 이미 쓰고 있어 {port} 포트로 엽니다(화면이 이미 열려 있는지 확인하세요).")
    env = dict(os.environ, PYTHONUTF8="1")
    print(f"정책 판정관 화면: http://localhost:{port}  (브라우저가 자동으로 열리지 않으면 이 주소를 여세요. 끝내려면 Ctrl+C)",
          flush=True)
    try:
        return subprocess.call(build_cmd(port, not a.no_browser), cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
