"""경로와 비밀값 설정.

비밀값 원칙:
- 비밀값은 .env 또는 환경변수로만 받는다.
- 값 자체를 로그·예외 메시지·manifest에 쓰지 않는다. 노출용으로는 `secret_status()`만 쓴다.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR              # 허브에서 내려받은 원본 CSV 위치
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"

SECRET_KEYS = ("ANTHROPIC_API_KEY", "JEJU_HUB_PROJECT_KEY")


def now_iso() -> str:
    """지금 시각(현지 시간대 오프셋 포함, 예: 2026-11-05T10:00:00+0900).

    time.strftime("%z")는 Windows에서 시간대 이름(지역화된 글자)을 낼 수 있어 datetime으로 만든다.
    """
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _read_dotenv() -> dict[str, str]:
    env_file = ROOT / ".env"
    values: dict[str, str] = {}
    if not env_file.exists():
        return values
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def get_secret(name: str) -> str | None:
    """환경변수 우선, 없으면 .env. 값은 호출한 쪽에서만 쓰고 출력하지 않는다."""
    return os.environ.get(name) or _read_dotenv().get(name) or None


def secret_status() -> dict[str, str]:
    """비밀값의 '설정 여부'만 돌려준다. 화면 표시·진단용."""
    return {k: ("설정됨" if get_secret(k) else "없음") for k in SECRET_KEYS}
