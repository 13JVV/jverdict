"""Phase 5: Windows 설치 묶음·설치 점검·재현성 보강 시험."""
import importlib.util
import json
import re
import shutil
import socket

import pytest

from jverdict.agent import catalog
from jverdict.agent import orchestrator as O
from jverdict.agent.library import CARDS
from jverdict.agent.llm import LLM
from jverdict.config import ROOT
from jverdict.data import sources

BATS = ["실행.bat", "점검.bat", "설치.bat"]


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_bundle", ROOT / "tools" / "build_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load_builder()


# ------------------------------------------------------------------------------------------ 배치 파일
@pytest.mark.parametrize("name", BATS)
def test_bat_files_are_windows_safe(name):
    b = (ROOT / name).read_bytes()
    assert not b.startswith(b"\xef\xbb\xbf"), "BOM이 있으면 cmd가 첫 줄(@echo off)을 못 읽음"
    assert not re.search(rb"(?<!\r)\n", b), "줄바꿈은 CRLF"
    txt = b.decode("utf-8")
    assert txt.startswith("@echo off\r\nchcp 65001 >nul\r\n")
    assert "/dev/null" not in txt and "cd /d \"%~dp0\"" in txt
    labels = set(re.findall(r"^:([A-Za-z_]+)\s*$", txt, re.M))
    for target in re.findall(r"goto :([A-Za-z_]+)", txt):
        assert target in labels or target == "eof", f"없는 label: {target}"
    for line in txt.splitlines():                        # 한글은 echo·rem 줄에만(명령 해석에 끼지 않게)
        if re.search(r"[가-힣]", line):
            assert re.match(r"\s*(echo|rem)\b", line), line
    assert B.check_bats(ROOT) == []


def test_bat_prefers_bundled_runtime():
    for name in ("실행.bat", "점검.bat"):
        txt = (ROOT / name).read_text(encoding="utf-8")
        assert txt.index("runtime\\python.exe") < txt.index(".venv\\Scripts\\python.exe")
        assert "-X utf8" in txt and "PYTHONUTF8=1" in txt


# ------------------------------------------------------------------------------------------ 코드 지문
def test_code_fingerprint_ignores_line_endings(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "src" / "jverdict", tmp_path / "src" / "jverdict",
                    ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(O, "ROOT", tmp_path)
    a = O._code_fingerprint()
    assert a == (monkeypatch.undo() or O._code_fingerprint())      # 복사본과 원본이 같은 지문
    monkeypatch.setattr(O, "ROOT", tmp_path)
    p = tmp_path / "src" / "jverdict" / "config.py"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))           # 압축 도구가 CRLF로 바꾼 경우
    assert O._code_fingerprint() == a
    p.write_bytes(p.read_bytes() + b"# changed\n")
    assert O._code_fingerprint() != a


# ------------------------------------------------------------------------------------------ 데이터 지문(원본 없이)
@pytest.fixture
def fake_sources(tmp_path, monkeypatch):
    raw, proc = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    proc.mkdir()
    (raw / "원본.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (proc / "가공.parquet").write_bytes(b"PAR1-fake")
    monkeypatch.setattr(sources, "RAW_DIR", raw)
    monkeypatch.setattr(sources, "PROCESSED_DIR", proc)
    monkeypatch.setattr(sources, "SOURCES_FILE", proc / "_sources.json")
    monkeypatch.setattr(sources, "PROCESSED_OF", {"원본.csv": "가공.parquet"})
    return raw, proc


def test_sources_fallback_matches_raw_and_detects_tampering(fake_sources):
    raw, proc = fake_sources
    sources.write()
    want = {"bytes": (raw / "원본.csv").stat().st_size, "sha256": sources.sha256_file(raw / "원본.csv")}
    (raw / "원본.csv").unlink()                                       # 배포 묶음: 원본 없음
    assert sources.raw_fingerprint_via_processed("원본.csv") == want
    sources.write()                                                   # 원본이 없는 PC에서 다시 써도 기록 유지
    assert sources.read()["sources"]["원본.csv"]["sha256"] == want["sha256"]
    (proc / "가공.parquet").write_bytes(b"PAR1-changed")
    with pytest.raises(sources.SourceMismatch):
        sources.raw_fingerprint_via_processed("원본.csv")
    with pytest.raises(FileNotFoundError):
        sources.raw_fingerprint_via_processed("없는.csv")


@pytest.mark.skipif(not sources.SOURCES_FILE.exists(), reason="_sources.json 없음")
def test_catalog_fingerprint_same_without_raw(tmp_path, monkeypatch):
    rec = sources.read()["sources"]
    if not rec or not all((sources.PROCESSED_DIR / r["processed"]["file"]).exists() for r in rec.values()):
        pytest.skip("가공 파일 없음")
    want = {ds: catalog.fingerprint(ds) for ds in ("bus_station", "bus_route")}
    monkeypatch.setattr(catalog, "RAW_DIR", tmp_path)                 # 원본 CSV가 없는 PC
    for ds, fp in want.items():
        assert catalog.fingerprint(ds) == fp
    man = json.loads((O.RUNS / "case-A-template" / "manifest.json").read_text(encoding="utf-8")) \
        if (O.RUNS / "case-A-template" / "manifest.json").exists() else None
    if man:
        assert catalog.fingerprint(man["data"]["dataset"]) == man["data"]


# ------------------------------------------------------------------------------------------ 재실행 보강
def test_replay_backfills_key_norm_only_on_exact_match():
    llm = LLM(mode="replay")
    payload = {"model": llm.model, "effort": llm.effort, "system": "s", "messages": [{"role": "user", "content": "0.12345678901234"}],
               "tools": [], "schema": {}}
    resp = {"content": [], "usage": {}}
    llm.cache = [{"role": "parser", "key": LLM._key(payload), "response": resp},
                 {"role": "skeptic", "key": "다른 입력", "key_norm": LLM._key_norm(payload), "response": resp}]
    llm._create("parser", "s", payload["messages"], None, {})
    llm._create("skeptic", "s", payload["messages"], None, {})       # key_norm으로만 맞음 → 채울 대상 아님
    assert llm.norm_backfill == {0: LLM._key_norm(payload)}


def test_replay_folder_follows_run_folder_name(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "RUNS", tmp_path)
    O.run(CARDS["C"]["card"]["question"], mode="template", run_id="orig")
    (tmp_path / "orig").rename(tmp_path / "renamed")
    out = O.replay(tmp_path / "renamed")
    assert out["reproduced"] and out["replay"] == "renamed-replay" and (tmp_path / "renamed-replay").exists()
    assert O.backfill_key_norm(tmp_path / "renamed")["status"] == "채울 것 없음"     # 템플릿 실행: LLM 기록 없음
    from jverdict.report.view import load_run
    v = load_run(tmp_path / "renamed")
    assert v.run_id == "renamed" and v.replay_check["reproduced"]


# ------------------------------------------------------------------------------------------ 화면 실행기
def test_ui_launcher_flags_and_free_port():
    from jverdict.ui import __main__ as L
    cmd = " ".join(L.build_cmd(8501, True))
    for flag in ("--server.address=localhost", "--browser.gatherUsageStats=false", "--server.showEmailPrompt=false",
                 "--client.showErrorDetails=type", "--server.headless=false", "-X utf8"):
        assert flag in cmd
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        busy = s.getsockname()[1]
        s.listen()
        assert L.pick_port(busy) != busy


# ------------------------------------------------------------------------------------------ 묶음 만들기 도구
def test_builder_selects_official_runs(tmp_path):
    for d in ("case-A-template", "case-A-template-replay", "case-X-replay", "_doctor-1", "ui-c"):
        (tmp_path / d).mkdir()
    (tmp_path / "case-A-template" / "manifest.json").write_text("{}")
    assert [p.name for p in B.official_runs(tmp_path)] == ["case-A-template", "case-A-template-replay"]
    assert B.skip(ROOT.relative_to(ROOT) / "src" / "__pycache__" / "x.pyc") and B.skip(type(ROOT)(".env"))
    assert not B.skip(type(ROOT)("src") / "jverdict" / "config.py")


def test_builder_finds_windows_only_dependencies(tmp_path):
    for name, reqs in (("app", ["colorama; platform_system == 'Windows'", "appnope; sys_platform == 'darwin'",
                                "lib>=1"]), ("lib", [])):
        d = tmp_path / f"{name}-1.0.dist-info"
        d.mkdir()
        (d / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n" + "".join(f"Requires-Dist: {r}\n" for r in reqs))
    assert B.missing_win_requirements(tmp_path) == ["colorama"]      # macOS 전용은 빼고 Windows 전용만


def test_builder_secret_scan_blocks_keys(tmp_path):
    (tmp_path / "ok.md").write_text("키는 .env에만 둔다", encoding="utf-8")
    assert B.secret_scan(tmp_path)["pattern_hits"] == [] and B.secret_scan(tmp_path)["env_files"] == []
    (tmp_path / "leak.py").write_text("K = '" + "sk-ant-" + "api03-" + "x" * 30 + "'\n")
    (tmp_path / ".env").write_text("A=1\n")
    r = B.secret_scan(tmp_path)
    assert r["pattern_hits"] == ["leak.py"] and r["env_files"] == [".env"]


def test_builder_pth_points_to_source_and_site_packages():
    lines = [ln for ln in B.PTH.splitlines() if ln and not ln.startswith("#")]
    assert lines == [f"python{B.PY_TAG[2:]}.zip", ".", "Lib\\site-packages", "..\\src", "import site"]
    assert re.fullmatch(r"[0-9a-f]{64}", B.PY_EMBED_SHA256) and B.PY_EMBED_URL.startswith("https://www.python.org/")


# ------------------------------------------------------------------------------------------ 설치 점검
def test_doctor_masks_paths_and_reads_markers(tmp_path):
    from jverdict import doctor as D
    s = D.mask(f"{ROOT}/outputs 와 {D.Path.home()}/x")
    assert str(ROOT) not in s and "<설치 폴더>" in s and str(D.Path.home()) not in s
    lock = tmp_path / "lock.txt"
    lock.write_text('# 주석\nnumpy==2.5.4\ncolorama==0.4.6; sys_platform == "win32"\n', encoding="utf-8")
    got = D.read_lock(lock)
    assert got["numpy"] == "2.5.4" and (("colorama" in got) == (D.os.name == "nt"))


@pytest.mark.skipif(not all(catalog.available(k) for k in catalog.DATASETS),
                    reason="허브 자료가 없는 환경(공개 저장소): 설치 점검은 자료가 있을 때만")
def test_doctor_quick_has_no_failures():
    from jverdict import doctor as D
    checks = D.Doctor(run_cases=False, verbose=False).run()
    fails = [f"{c.name}: {c.detail}" for c in checks if c.status == D.FAIL]
    assert not fails, fails
    assert {c.section for c in checks} >= {"환경", "패키지", "데이터", "판정서·화면", "보안"}
