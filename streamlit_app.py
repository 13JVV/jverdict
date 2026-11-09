"""Streamlit Community Cloud 진입점: 정책 판정관 화면을 그대로 연다.

웹 데모에는 제주데이터허브 원자료가 없다. 실데이터 사례는 기록된 판정서를 열고,
합성데이터 사례와 '정책 카드 직접 만들기'(합성데이터)는 그 자리에서 판정한다.
"""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

# 저장소를 새로 받아도 서버 프로세스는 이미 읽은 jverdict 모듈을 그대로 쓴다. 파일이 바뀐 모듈은 버리고 새로 읽게 한다
# (그렇지 않으면 새 화면 코드와 옛 계산·보고 코드가 섞여 오류가 난다). 이 진입 파일이 처음 도는 프로세스면 모두 새로 읽는다.
import time as _time  # noqa: E402

# 하나라도 바뀌었으면 jverdict 전체를 버린다: 바뀐 모듈만 버리면 'from jverdict.report import brief'처럼
# 상위 패키지 속성으로 가져오는 import가 옛 모듈을 계속 쓴다.
_first = not getattr(sys, "_jv_entry_seen", False)
sys._jv_entry_seen = True
_ours = {n: m for n, m in list(sys.modules.items()) if n == "jverdict" or n.startswith("jverdict.")}


def _changed(m) -> bool:
    _f = getattr(m, "__file__", None)
    if not _f:
        return False
    try:
        return Path(_f).stat().st_mtime > getattr(m, "_jv_loaded", float("inf"))
    except OSError:
        return True


if _first or any(_changed(m) for m in _ours.values()):
    for _name in _ours:
        del sys.modules[_name]


def _mark():
    for _n, _m in list(sys.modules.items()):
        if _n.startswith("jverdict") and not hasattr(_m, "_jv_loaded"):
            _m._jv_loaded = _time.time()


try:
    runpy.run_path(str(SRC / "jverdict" / "ui" / "app.py"), run_name="__main__")
finally:
    _mark()
