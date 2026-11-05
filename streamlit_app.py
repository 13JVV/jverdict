"""Streamlit Community Cloud 진입점: 정책 판정관 화면을 그대로 연다.

웹 데모에는 제주데이터허브 원자료가 없다. 실데이터 사례는 기록된 판정서를 열고,
합성데이터 사례와 '정책 카드 직접 만들기'(합성데이터)는 그 자리에서 판정한다.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
runpy.run_path(str(Path(__file__).resolve().parent / "src" / "jverdict" / "ui" / "app.py"), run_name="__main__")
