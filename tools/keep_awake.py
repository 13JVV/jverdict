"""웹 데모(Streamlit Community Cloud) 깨워 두기: 브라우저로 데모를 열고, 잠들어 있으면 깨운다.

GitHub Actions(.github/workflows/keep-awake.yml)가 6시간마다 실행한다. 손으로 돌릴 때:

    pip install playwright==1.48.0 && python -m playwright install chromium
    python tools/keep_awake.py https://jverdict.streamlit.app

접속만 하고 아무것도 입력하지 않는다. 앱 화면 글자('판정관')가 보이면 성공(종료 코드 0).
"""
from __future__ import annotations

import sys
import time

WAKE_TEXTS = ("Yes, get this app back up", "get this app back up")
APP_TEXT = "판정관"


def main(url: str, wait_s: int = 240) -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page()
        page.goto(url + ("&" if "?" in url else "?") + "guide=0", wait_until="domcontentloaded", timeout=120_000)
        t0, woke = time.time(), False
        while time.time() - t0 < wait_s:
            if not woke:
                for t in WAKE_TEXTS:
                    btn = page.get_by_role("button", name=t)
                    if btn.count():
                        btn.first.click()
                        woke = True
                        print("잠든 앱을 깨우는 버튼을 눌렀습니다.", flush=True)
                        break
            for f in page.frames:
                try:
                    if APP_TEXT in (f.inner_text("body", timeout=2_000) or ""):
                        print(f"앱 화면 확인({time.time() - t0:.0f}초){' · 깨움' if woke else ''}", flush=True)
                        b.close()
                        return 0
                except Exception:  # noqa: BLE001  프레임이 아직 준비되지 않음
                    pass
            time.sleep(5)
        print(f"{wait_s}초 안에 앱 화면을 확인하지 못했습니다.", flush=True)
        b.close()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "https://jverdict.streamlit.app"))
