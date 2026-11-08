"""Windows 설치 묶음 만들기 (개발 PC에서 실행).

    python tools/build_bundle.py                  # dist/JVerdict/ 와 dist/JVerdict_win64_<날짜>.zip
    python tools/build_bundle.py --no-runtime     # Python 실행 환경 없이 코드·데이터·결과만(설치.bat로 설치)
    python tools/build_bundle.py --with-raw       # 1GB 버스 원본 CSV까지 싣기(보통은 가공 파일만)
    python tools/build_bundle.py --no-zip         # 폴더까지만

묶음을 받은 PC는 인터넷도, 설치도, 관리자 권한도 필요 없다. 압축을 풀고 실행.bat를 누르면 된다.
- runtime/          python.org 공식 Windows 임베디드 Python(SHA-256 대조) + 패키지(Windows용 wheel을 미리 풀어 둠)
- 패키지 버전은 개발 PC와 같게 고정하고(requirements/lock-win-cp314.txt), 판정 수치가 같은지 점검.bat로 확인한다.
- 원본 CSV 대신 가공 파일(parquet)을 싣고, 원본 지문은 data/processed/_sources.json으로 확인한다.
- .env(비밀값)는 절대 싣지 않는다. 실제 키 값과 키 모양 문자열이 묶음 어디에도 없는지 검사한 뒤에야 압축한다.
- SHA256SUMS.txt: 묶음의 모든 파일 지문(받은 PC에서 점검.bat가 대조).

받는 PC에서 할 일은 docs/manual/설치_실행.md 참고.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jverdict.config import SECRET_KEYS, get_secret, now_iso  # noqa: E402

BUNDLE = "JVerdict"
PY_VERSION = "3.14.3"
PY_TAG = "cp314"
PLATFORM = "win_amd64"
PY_EMBED = f"python-{PY_VERSION}-embed-amd64.zip"
PY_EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/{PY_EMBED}"
# python.org 공개 SHA-256 (https://www.python.org/api/v2/downloads/release_file/?release=1090)
PY_EMBED_SHA256 = "ad4961a479dedbeb7c7d113253f8db1b1935586b73c27488712beec4f2c894e6"
LOCK = Path("requirements") / f"lock-win-{PY_TAG}.txt"

# 묶음에 싣는 것(화이트리스트). 그 밖의 것은 싣지 않는다.
ROOT_FILES = ["실행.bat", "점검.bat", "설치.bat", "README.md", "pyproject.toml", ".env.example"]
TREES = ["src/jverdict", "tests", "tools", "docs", ".streamlit", "requirements"]
OUTPUT_TREES = ["outputs/accuracy", "outputs/cases", "outputs/docs"]
SMALL_RAW_MAX = 10 * 1024 * 1024           # 이보다 작은 원본 CSV는 그대로 싣는다(카드 데이터 등)
SKIP_NAMES = {"__pycache__", ".DS_Store", ".pytest_cache", ".env", ".git", ".venv", "Thumbs.db"}
SKIP_SUFFIX = {".pyc", ".pyo"}
# 패키지에서 빼는 것: 패키지 자체 시험 코드(실행에 쓰이지 않음), Jupyter 확장, macOS용 실행 스크립트
PRUNE_TESTS_IN = ("numpy", "pandas", "scipy", "statsmodels", "pyarrow")
PRUNE_TOP = ("bin", "share")

KEY_PAT = re.compile(rb"sk-ant-[A-Za-z0-9_\-]{10,}")
KEY_ALLOW = {("tests/test_agent.py", b"sk-ant-" + b"api03-ABCD")}      # sanitize 시험용 가짜 문자열(이 파일엔 이어 쓰지 않음)
TEXT_SUFFIX = {".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".bat", ".csv", ".html", ".sql", ".example", ".cfg",
               ".ini", ".yaml", ".yml"}


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def skip(p: Path) -> bool:
    return any(part in SKIP_NAMES for part in p.parts) or p.suffix in SKIP_SUFFIX or p.name.startswith("._")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> int:
    n = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir() or skip(rel):
            continue
        copy_file(p, dst / rel)
        n += 1
    return n


# ------------------------------------------------------------------------------------------------ 1. 프로젝트 파일
def official_runs(runs: Path) -> list[Path]:
    """싣는 실행 기록: 사례 실행(case-*)과, 원본이 있는 재실행(-replay)만. 임시·점검용 실행은 뺀다."""
    out = []
    for p in sorted(runs.iterdir()) if runs.exists() else []:
        if not p.is_dir() or not p.name.startswith("case-"):
            continue
        if p.name.endswith("-replay"):
            if (runs / p.name[: -len("-replay")] / "manifest.json").exists():
                out.append(p)
        elif (p / "manifest.json").exists():
            out.append(p)
    return out


def stage_project(stage: Path, with_raw: bool) -> dict:
    from jverdict.data import sources
    counts = {}
    for f in ROOT_FILES:
        if (ROOT / f).exists():
            copy_file(ROOT / f, stage / f)
        else:
            raise SystemExit(f"[오류] 묶음에 꼭 필요한 파일이 없음: {f}")
    for t in TREES:
        counts[t] = copy_tree(ROOT / t, stage / t)
    # 데이터: 가공 파일 + 원본 지문 기록 + 작은 원본 CSV(카드 등). 큰 버스 원본은 --with-raw일 때만
    if sources.build()["sources"]:
        sources.write()
    proc = ROOT / "data" / "processed"
    counts["data/processed"] = copy_tree(proc, stage / "data" / "processed")
    raw = []
    for p in sorted((ROOT / "data").glob("*.csv")):
        if with_raw or p.stat().st_size <= SMALL_RAW_MAX:
            copy_file(p, stage / "data" / p.name)
            raw.append(p.name)
    counts["data/*.csv"] = len(raw)
    # 결과: 사례 실행 기록, 정확도 리포트, 사례 분석, 기획서
    runs = ROOT / "outputs" / "runs"
    keep = official_runs(runs)
    for p in keep:
        copy_tree(p, stage / "outputs" / "runs" / p.name)
    for f in ("_summary.md", "_summary.json"):
        if (runs / f).exists():
            copy_file(runs / f, stage / "outputs" / "runs" / f)
    counts["outputs/runs"] = len(keep)
    dropped = sorted(p.name for p in runs.iterdir() if p.is_dir() and p not in keep) if runs.exists() else []
    for t in OUTPUT_TREES:
        if (ROOT / t).exists():
            counts[t] = copy_tree(ROOT / t, stage / t)
    return {"counts": counts, "raw_csv": raw, "runs": [p.name for p in keep], "runs_dropped": dropped}


# ------------------------------------------------------------------------------------------------ 2. 실행 환경
def fetch_embed(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    z = cache / PY_EMBED
    if not z.exists() or sha256(z) != PY_EMBED_SHA256:
        log(f"  Python {PY_VERSION} 임베디드 내려받기: {PY_EMBED_URL}")
        tmp = z.with_suffix(".part")
        with urllib.request.urlopen(PY_EMBED_URL, timeout=120) as r, open(tmp, "wb") as f:   # noqa: S310 (고정 https 주소)
            shutil.copyfileobj(r, f)
        tmp.replace(z)
    got = sha256(z)
    if got != PY_EMBED_SHA256:
        z.unlink()
        raise SystemExit(f"[오류] 내려받은 Python의 SHA-256이 공식 값과 다름: {got}")
    return z


PTH = f"""python{PY_TAG[2:]}.zip
.
Lib\\site-packages
..\\src
# 정책 판정관: 위 경로만 쓴다(사용자·시스템 Python 설정과 섞이지 않게). site는 .pth 처리를 위해 켠다.
import site
"""


def stage_runtime(stage: Path, cache: Path) -> Path:
    rt = stage / "runtime"
    with zipfile.ZipFile(fetch_embed(cache)) as zf:
        zf.extractall(rt)
    pth = rt / f"python{PY_TAG[2:]}._pth"
    if not pth.exists():
        raise SystemExit(f"[오류] 임베디드 Python에 {pth.name}이 없음")
    pth.write_bytes(PTH.replace("\n", "\r\n").encode("utf-8"))
    return rt


def win_marker_env() -> dict:
    return {"implementation_name": "cpython", "implementation_version": PY_VERSION, "os_name": "nt",
            "platform_machine": "AMD64", "platform_python_implementation": "CPython", "platform_release": "10",
            "platform_system": "Windows", "platform_version": "10.0.19045", "python_full_version": PY_VERSION,
            "python_version": ".".join(PY_VERSION.split(".")[:2]), "sys_platform": "win32", "extra": ""}


def installed(sp: Path) -> dict[str, tuple[str, Path]]:
    out = {}
    for d in sp.glob("*.dist-info"):
        meta = (d / "METADATA").read_text(encoding="utf-8", errors="replace")
        name = re.search(r"^Name: (.+)$", meta, re.M).group(1).strip()
        ver = re.search(r"^Version: (.+)$", meta, re.M).group(1).strip()
        out[canon(name)] = (ver, d)
    return out


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def missing_win_requirements(sp: Path) -> list[str]:
    """설치된 패키지들의 의존성을 'Windows 기준'으로 다시 따져 빠진 것을 찾는다.
    pip의 --platform 설치는 의존성 조건(marker)을 개발 PC(macOS) 기준으로 판단하므로, Windows에서만 필요한
    의존성(예: platform_system == "Windows")이 빠질 수 있다."""
    from packaging.requirements import Requirement
    env = win_marker_env()
    have = installed(sp)
    need = set()
    for _name, (_ver, d) in have.items():
        meta = (d / "METADATA").read_text(encoding="utf-8", errors="replace")
        for line in re.findall(r"^Requires-Dist: (.+)$", meta, re.M):
            req = Requirement(line)
            if req.marker is not None and not req.marker.evaluate(env):
                continue
            if canon(req.name) not in have:
                need.add(str(Requirement(f"{req.name}{req.specifier}")))
    return sorted(need)


def pip_target(sp: Path, reqs: list[str], constraints: Path) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-compile",
           "--target", str(sp), "--platform", PLATFORM, "--python-version", PY_VERSION.rsplit(".", 1)[0],
           "--implementation", "cp", "--abi", PY_TAG, "--only-binary=:all:", "--upgrade", "-c", str(constraints), *reqs]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[오류] Windows용 패키지 설치 실패:\n{r.stderr[-3000:]}")


def stage_packages(rt: Path, tmp: Path) -> dict:
    """개발 PC와 같은 버전(constraints)으로 Windows용 wheel을 runtime/Lib/site-packages에 미리 푼다."""
    sp = rt / "Lib" / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    cons = tmp / "constraints.txt"
    r = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable", "--disable-pip-version-check"],
                       capture_output=True, text=True, check=True)
    cons.write_text("\n".join(l for l in r.stdout.splitlines() if not l.lower().startswith("jverdict")) + "\n",
                    encoding="utf-8")
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    reqs = proj["dependencies"] + proj["optional-dependencies"]["dev"]
    log(f"  패키지 {len(reqs)}개(+의존성) Windows용으로 설치: {', '.join(r.split('>')[0] for r in reqs)}")
    pip_target(sp, reqs, cons)
    added = []
    for _ in range(5):                          # Windows 전용 의존성까지 닫힐 때까지
        more = missing_win_requirements(sp)
        if not more:
            break
        log(f"  Windows 전용 의존성 추가: {', '.join(more)}")
        added += more
        pip_target(sp, more, cons)
    else:
        raise SystemExit("[오류] Windows 의존성이 닫히지 않음")
    pruned = 0
    for top in PRUNE_TOP:
        if (sp / top).exists():
            pruned += sum(f.stat().st_size for f in (sp / top).rglob("*") if f.is_file())
            shutil.rmtree(sp / top)
    for pkg in PRUNE_TESTS_IN:
        for t in sorted((sp / pkg).rglob("tests"), key=lambda p: len(p.parts)):
            if t.is_dir():
                pruned += sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
                shutil.rmtree(t)
    have = installed(sp)
    lines = [f"# 정책 판정관 Windows 설치 묶음 잠금 파일 (Python {PY_VERSION}, {PLATFORM}). tools/build_bundle.py가 만듦",
             f"# 이 버전들은 개발 PC(판정 기록을 만든 환경)와 같다. 생성 {now_iso()}"]
    win_only = {canon(re.split(r"[<>=!~ ;\[]", r, maxsplit=1)[0]) for r in added}
    lines += [f"{n}=={v}" + ('; sys_platform == "win32"' if n in win_only else "") for n, (v, _d) in sorted(have.items())]
    (ROOT / LOCK).parent.mkdir(exist_ok=True)
    (ROOT / LOCK).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"packages": len(have), "windows_only_added": added, "pruned_mb": round(pruned / 1e6, 1)}


def check_wheels(sp: Path) -> list[str]:
    """모든 패키지가 Windows 64bit(또는 순수 파이썬)용인지, macOS·Linux 바이너리가 섞이지 않았는지."""
    bad = []
    ok_plat = {PLATFORM, "any"}
    for d in sp.glob("*.dist-info"):
        w = d / "WHEEL"
        tags = re.findall(r"^Tag: (.+)$", w.read_text(encoding="utf-8"), re.M) if w.exists() else []
        if not tags or not any(t.split("-")[-1] in ok_plat for t in tags):
            bad.append(f"{d.name}: {tags}")
            continue
        for t in tags:
            py = t.split("-")[0]
            if py.startswith("cp") and py not in (PY_TAG, "cp3") and not t.split("-")[1] == "abi3":
                bad.append(f"{d.name}: {t}")
    for ext in ("*.so", "*.dylib"):
        bad += [f"macOS/Linux 바이너리: {p.relative_to(sp)}" for p in sp.rglob(ext)]
    return bad


# ------------------------------------------------------------------------------------------------ 3. 검사
def secret_scan(stage: Path) -> dict:
    """비밀값 검사: (1) .env 파일이 없어야 함 (2) 실제 키 값이 어느 파일에도 없어야 함 (3) 키 모양 문자열(허용 목록 외) 없음.
    실제 키 값은 메모리에서만 비교하고 출력하지 않는다."""
    envs = [str(p.relative_to(stage)) for p in stage.rglob(".env")]
    values = [v.encode() for v in (get_secret(k) for k in SECRET_KEYS) if v and len(v) >= 8]
    value_hits, pat_hits, n = [], [], 0
    for p in stage.rglob("*"):
        if not p.is_file():
            continue
        n += 1
        b = p.read_bytes()
        rel = p.relative_to(stage).as_posix()
        if any(v in b for v in values):
            value_hits.append(rel)
        if not rel.startswith("runtime/") and (p.suffix in TEXT_SUFFIX or p.suffix == ""):
            for m in KEY_PAT.finditer(b):
                if not any(rel == f and m.group().startswith(pre) for f, pre in KEY_ALLOW):
                    pat_hits.append(rel)
    return {"files": n, "keys_checked": len(values), "env_files": envs, "value_hits": sorted(set(value_hits)),
            "pattern_hits": sorted(set(pat_hits))}


def check_bats(stage: Path) -> list[str]:
    bad = []
    for p in sorted(stage.glob("*.bat")):
        b = p.read_bytes()
        if b.startswith(b"\xef\xbb\xbf"):
            bad.append(f"{p.name}: BOM 있음(cmd가 첫 줄을 못 읽음)")
        if re.search(rb"(?<!\r)\n", b):
            bad.append(f"{p.name}: 줄바꿈이 CRLF가 아님")
        if b"/dev/null" in b:
            bad.append(f"{p.name}: /dev/null 사용(Windows에는 없음, nul을 써야 함)")
        if b"chcp 65001" not in b:
            bad.append(f"{p.name}: chcp 65001 없음(한글 출력 깨짐)")
        b.decode("utf-8")
    return bad


def longest(stage: Path) -> tuple[int, str]:
    best = (0, "")
    for p in stage.rglob("*"):
        rel = f"{BUNDLE}\\" + str(p.relative_to(stage)).replace("/", "\\")
        if len(rel) > best[0]:
            best = (len(rel), rel)
    return best


# ------------------------------------------------------------------------------------------------ 4. 기록·압축
def code_sha() -> str:
    from jverdict.agent.orchestrator import _code_fingerprint
    return _code_fingerprint()


def nfc(rel: str) -> str:
    """zip 안 이름은 NFC(완성형)로 둔다. Mac에서 내려받은 파일은 이름이 NFD(자모 분리)인 경우가 있다.
    Mac은 둘을 같은 이름으로 보지만 Windows는 다른 이름으로 봐서, 코드가 찾는 '업종별성별카드이용정보.csv'가 없다고 나온다."""
    return unicodedata.normalize("NFC", rel)


def write_meta(stage: Path, info: dict) -> None:
    (stage / "BUILD_INFO.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for p in sorted(stage.rglob("*"), key=lambda p: p.relative_to(stage).as_posix()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            rows.append(f"{sha256(p)}  {nfc(p.relative_to(stage).as_posix())}")
    (stage / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_zip(stage: Path, zpath: Path) -> dict:
    files = sorted((p for p in stage.rglob("*") if p.is_file()), key=lambda p: p.relative_to(stage).as_posix())
    tmp = zpath.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            zf.write(p, nfc(f"{BUNDLE}/{p.relative_to(stage).as_posix()}"))  # 한글 이름은 NFC + UTF-8 표시(0x800)
    tmp.replace(zpath)
    with zipfile.ZipFile(zpath) as zf:
        badcrc = zf.testzip()
        infos = zf.infolist()
        nonascii_noflag = [i.filename for i in infos if not i.filename.isascii() and not i.flag_bits & 0x800]
        not_nfc = [i.filename for i in infos if i.filename != nfc(i.filename)]
        names = {i.filename for i in infos}
    if badcrc or nonascii_noflag or not_nfc or len(names) != len(files):
        raise SystemExit(f"[오류] 압축 검사 실패: crc={badcrc}, utf8표시없음={nonascii_noflag[:3]}, "
                         f"NFC아님={not_nfc[:3]}, 항목 {len(names)}/{len(files)}")
    return {"entries": len(infos), "bytes": zpath.stat().st_size, "sha256": sha256(zpath)}


# ------------------------------------------------------------------------------------------------ 5. 받은 PC 흉내
def simulate(zpath: Path) -> int:
    """압축을 새 폴더에 풀고, 원본 CSV가 없는 상태로 그 폴더의 코드로 점검을 돌린다(이 PC의 Python 사용).
    Windows 실행 파일 자체는 Windows에서만 시험할 수 있다(점검.bat)."""
    with tempfile.TemporaryDirectory(prefix="jv-sim-") as td:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(td)
        root = Path(td) / BUNDLE
        bad = [ln for ln in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
               if sha256(root / ln.split("  ", 1)[1]) != ln.split("  ", 1)[0]]
        if bad:
            log(f"  [실패] 풀어 낸 파일 지문 불일치 {len(bad)}개")
            return 1
        env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "jverdict.doctor"], cwd=root, env=env)
        for rep in sorted((root / "outputs" / "doctor").glob("점검_*.txt")):
            shutil.copy2(rep, ROOT / "dist" / f"simulate_{rep.name}")
        return r.returncode


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python tools/build_bundle.py", description="Windows 설치 묶음 만들기")
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--no-runtime", action="store_true", help="Python 실행 환경을 넣지 않음(설치.bat로 설치)")
    ap.add_argument("--with-raw", action="store_true", help="큰 원본 CSV까지 넣음")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-simulate", action="store_true", help="압축을 풀어 점검하는 단계를 건너뜀")
    a = ap.parse_args(argv)
    t0 = time.time()
    out: Path = a.out.resolve()
    stage = out / BUNDLE
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    info = {"name": "정책 판정관 J-Verdict", "version": proj["version"], "built": now_iso(),
            "python": PY_VERSION if not a.no_runtime else None, "platform": PLATFORM if not a.no_runtime else None,
            "lock": LOCK.as_posix(), "code_sha256": code_sha()}

    log("[1/6] 프로젝트 파일 모으기")
    info["contents"] = stage_project(stage, a.with_raw)
    if info["contents"]["runs_dropped"]:
        log(f"  싣지 않은 실행 기록(임시·짝 없는 재실행): {', '.join(info['contents']['runs_dropped'])}")

    if not a.no_runtime:
        log(f"[2/6] Windows 실행 환경(Python {PY_VERSION} 임베디드 + 패키지)")
        rt = stage_runtime(stage, out / "_cache")
        with tempfile.TemporaryDirectory() as td:
            info["runtime"] = stage_packages(rt, Path(td))
        copy_file(ROOT / LOCK, stage / LOCK)            # 방금 만든 잠금 파일로 교체
        bad = check_wheels(rt / "Lib" / "site-packages")
        if bad:
            raise SystemExit("[오류] Windows용이 아닌 패키지:\n  " + "\n  ".join(bad[:20]))
        log(f"  패키지 {info['runtime']['packages']}개 · 모두 {PLATFORM}/순수 파이썬 · 정리 {info['runtime']['pruned_mb']}MB")
    else:
        log("[2/6] 실행 환경 생략(--no-runtime)")

    log("[3/6] 검사: 비밀값·배치 파일·경로 길이")
    sec = secret_scan(stage)
    info["secret_scan"] = {k: (v if not isinstance(v, list) else len(v)) for k, v in sec.items()}
    if sec["env_files"] or sec["value_hits"] or sec["pattern_hits"]:
        raise SystemExit(f"[중단] 비밀값 검사 실패: .env {sec['env_files']}, 실제 값 포함 {sec['value_hits']}, "
                         f"키 모양 {sec['pattern_hits']}")
    log(f"  파일 {sec['files']}개 · 실제 키 {sec['keys_checked']}개 대조 0건 · 키 모양 문자열 0건 · .env 없음")
    bats = check_bats(stage)
    if bats:
        raise SystemExit("[오류] 배치 파일:\n  " + "\n  ".join(bats))
    n, where = longest(stage)
    info["longest_path"] = n
    log(f"  가장 긴 경로 {n}자(묶음 폴더 기준): {where}")
    info["unpack_path_budget"] = 259 - n
    log(f"  → 압축을 푸는 위치의 경로가 {259 - n}자보다 짧으면 Windows 경로 한도(260자) 안에 들어감")
    if n > 200:
        raise SystemExit("[오류] 경로가 너무 김: Windows 260자 한도에 여유가 부족")

    log("[4/6] BUILD_INFO.json · SHA256SUMS.txt")
    write_meta(stage, info)
    size = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    log(f"  묶음 크기 {size / 1e6:.0f}MB")

    zinfo = None
    if not a.no_zip:
        name = f"{BUNDLE}_{'win64' if not a.no_runtime else 'src'}_{time.strftime('%Y%m%d')}.zip"
        log(f"[5/6] 압축: {name}")
        zinfo = make_zip(stage, out / name)
        log(f"  {zinfo['entries']}개 항목 · {zinfo['bytes'] / 1e6:.0f}MB · CRC 검사 통과 · 한글 이름 UTF-8 표시")
        (out / f"{name}.sha256").write_text(f"{zinfo['sha256']}  {name}\n", encoding="utf-8")
        if not a.no_simulate:
            log("[6/6] 받은 PC 흉내: 새 폴더에 풀고(원본 CSV 없음) 점검")
            rc = simulate(out / name)
            if rc != 0:
                raise SystemExit("[실패] 풀어 낸 묶음의 점검에 실패 항목이 있음(dist/simulate_점검_*.txt 참고)")
    log(f"\n완료 ({time.time() - t0:.0f}초): {stage}" + (f"\n압축: {out / name}" if zinfo else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
