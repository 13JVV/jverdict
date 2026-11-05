"""공개 저장소(GitHub)·웹 데모용 내보내기: 허용 목록의 파일만 dist/public/jverdict/ 에 모은다.

    .venv/bin/python tools/export_public.py            # 내보내기 + 검사 + 데이터 없이 시험
    .venv/bin/python tools/export_public.py --no-test

싣지 않는 것 (이유)
- 제주데이터허브 원자료(data/raw)와 가공 파일(data/processed/*.parquet, 정류소 목록): 허브의 재배포 조건을 확인하지 못함.
  공개 저장소에는 지문 기록(_sources.json)과 적재 보고서만 싣고, 받는 방법을 data/README.md에 적는다.
- 업종별 금액이 들어간 사례 파일(heterogeneity.csv): 원자료 금액에 가까움.
- .env, 실행 환경(runtime), 가상환경, 점검 보고서(outputs/doctor), 내부 계획(ROADMAP), 센터 방문 준비물(docs/center-visit),
  옛 기획서 초안(v0.1).
검사 (하나라도 걸리면 실패)
- .env 파일 0개, 실제 키 값 0건(메모리에서만 비교, 출력 안 함), 키 모양 문자열 0건(시험용 가짜 제외)
- 이 PC의 홈 경로·사용자 이메일 문자열 0건
- 허브 원자료·가공 파일 0개
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "public" / "jverdict"

FILES = ["README.md", "pyproject.toml", ".env.example", "실행.bat", "점검.bat", "설치.bat"]
TREES = ["src/jverdict", "tests", "tools", ".streamlit", "requirements", "docs/proposal", "docs/prereg", "docs/research",
         "docs/manual", "docs/test", "docs/report", "outputs/accuracy", "outputs/cases", "outputs/docs"]
DATA_FILES = ["data/processed/_sources.json", "data/processed/_ingest_report.json"]
EXCLUDE = {"docs/proposal/기획서_v0.1_센터데이터안.md", "outputs/cases/relief2020/heterogeneity.csv",
           "tools/export_public.py.bak"}
FORBIDDEN_SUFFIX = {".parquet"}
DATA_NAMES = {"이용자유형별버스정류소이용인원현황.csv", "일별버스승차인원수현황.csv", "업종별성별카드이용정보.csv",
              "bus_stations.csv"}

STREAMLIT_APP = '''"""Streamlit Community Cloud 진입점: 정책 판정관 화면을 그대로 연다.

웹 데모에는 제주데이터허브 원자료가 없다. 실데이터 사례는 기록된 판정서를 열고,
합성데이터 사례와 '정책 카드 직접 만들기'(합성데이터)는 그 자리에서 판정한다.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
runpy.run_path(str(Path(__file__).resolve().parent / "src" / "jverdict" / "ui" / "app.py"), run_name="__main__")
'''

GITIGNORE = """.env
.venv/
.venv.nosync/
__pycache__/
*.pyc
.DS_Store
data/raw/
data/processed/*.parquet
data/processed/bus_stations.csv
runtime/
dist/
outputs/doctor/
outputs/runs/_*/
outputs/runs/2*/
"""

DATA_README = """# 데이터 받는 방법

이 저장소에는 제주데이터허브 원자료와 가공 파일을 싣지 않았습니다(허브의 재배포 조건을 확인하지 못함).
아래 파일을 제주데이터허브(https://www.jejudatahub.net, 로그인 후 내려받기)에서 받아 `data/raw/`에 두면
모든 사례를 이 PC에서 다시 판정할 수 있습니다.

| 파일 | 허브 데이터명 |
|---|---|
| 이용자유형별버스정류소이용인원현황.csv | 이용자 유형별 버스정류소 이용 인원 현황 |
| 일별버스승차인원수현황.csv | 일별 버스 승차 인원 수 현황 |
| 일별시간대별버스승차인원수현황.csv | 일별 시간대별 버스 승차 인원 수 현황 |
| 월별버스이용자유형별패널티현황.csv | 월별 버스 이용자 유형별 패널티 현황 |
| 업종별성별카드이용정보.csv | 업종별 성별 카드 이용 정보 |

받은 뒤:

```bash
python -m jverdict.data.ingest_hub      # data/processed/*.parquet 만들기
python -m jverdict.data.sources         # 지문 기록(_sources.json)과 대조
python -m jverdict.doctor               # 사례를 새로 판정해 기록과 같은지 확인
```

`data/processed/_sources.json`에는 원자료와 가공 파일의 SHA-256 지문이 있습니다. 같은 파일을 받았는지 확인할 수 있습니다.
"""


def _builder():
    spec = importlib.util.spec_from_file_location("build_bundle", ROOT / "tools" / "build_bundle.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def collect(out: Path) -> int:
    B = _builder()
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():                     # 공개 저장소 기록(.git)은 남기고 나머지를 새로 만든다
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    n = 0

    def put(src: Path):
        nonlocal n
        rel = src.relative_to(ROOT)
        if B.skip(rel) or rel.as_posix() in EXCLUDE or src.suffix in FORBIDDEN_SUFFIX or src.name in DATA_NAMES:
            return
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    for f in FILES + DATA_FILES:
        if (ROOT / f).exists():
            put(ROOT / f)
    for t in TREES:
        for p in sorted((ROOT / t).rglob("*")):
            if p.is_file():
                put(p)
    runs = ROOT / "outputs" / "runs"
    for d in B.official_runs(runs):
        for p in sorted(d.rglob("*")):
            if p.is_file():
                put(p)
    for f in ("_summary.md", "_summary.json"):
        if (runs / f).exists():
            put(runs / f)
    cfg = out / ".streamlit" / "config.toml"          # 웹 데모: localhost 고정·headless=false는 서버에서 못 씀
    if cfg.exists():                                     # (로컬 실행기는 같은 값을 명령줄 인자로 따로 넘긴다)
        keep = [ln for ln in cfg.read_text(encoding="utf-8").splitlines()
                if not ln.strip().startswith(("address", "headless"))]
        cfg.write_text("\n".join(keep) + "\n", encoding="utf-8")
    (out / "streamlit_app.py").write_text(STREAMLIT_APP, encoding="utf-8")
    (out / "requirements.txt").write_text((ROOT / "requirements" / "base.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (out / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (out / "data" / "README.md").write_text(DATA_README, encoding="utf-8")
    return n + 4


def check(out: Path) -> list[str]:
    B = _builder()
    errs = []
    sc = B.secret_scan(out)
    if sc["env_files"]:
        errs.append(f".env 파일 {len(sc['env_files'])}개")
    if sc["value_hits"]:
        errs.append(f"실제 키 값이 든 파일 {len(sc['value_hits'])}개(경로만): {sc['value_hits']}")
    if sc["pattern_hits"]:
        errs.append(f"키 모양 문자열이 든 파일: {sc['pattern_hits']}")
    personal = [s.encode() for s in {str(Path.home()), os.environ.get("USER", "") and f"/Users/{os.environ['USER']}"} if s]
    mail = re.compile(rb"[A-Za-z0-9._%+-]+@(naver|gmail|daum|hanmail|kakao)\.(com|net)")
    for p in out.rglob("*"):
        if not p.is_file():
            continue
        b = p.read_bytes()
        rel = p.relative_to(out).as_posix()
        if any(x in b for x in personal):
            errs.append(f"홈 경로 문자열: {rel}")
        if mail.search(b):
            errs.append(f"개인 이메일 모양: {rel}")
        if p.suffix in FORBIDDEN_SUFFIX or p.name in DATA_NAMES:
            errs.append(f"허브 자료 파일: {rel}")
    return errs


def test_without_data(out: Path) -> tuple[bool, str]:
    """데이터 없는 상태로 시험: 자료가 필요한 시험은 건너뛰어야 하고 실패는 0이어야 한다."""
    env = dict(os.environ, PYTHONPATH=str(out / "src"), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(out / "tests")],
                       cwd=out, env=env, capture_output=True, text=True, timeout=1800)
    tail = "\n".join((r.stdout or "").strip().splitlines()[-3:])
    for p in out.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    return r.returncode == 0, tail


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args(argv)
    n = collect(OUT)
    errs = check(OUT)
    print(f"내보낸 파일 {n}개 → {OUT.relative_to(ROOT)}")
    if errs:
        print("검사 실패:")
        for e in errs:
            print(" -", e)
        return 1
    print("검사 통과: .env 0 · 실제 키 값 0 · 키 모양 0 · 홈 경로·이메일 0 · 허브 자료 파일 0")
    if not a.no_test:
        ok, tail = test_without_data(OUT)
        print("데이터 없이 시험:", "통과" if ok else "실패", "|", tail.replace("\n", " "))
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
