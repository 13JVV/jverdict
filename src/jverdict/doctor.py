"""설치 점검: python -m jverdict.doctor [--full] (Windows에서는 점검.bat)

이 PC에서 정책 판정관이 제대로 돌아가는지 한 번에 확인하고, 결과를 outputs/doctor/ 에 보고서로 남긴다.
보고서에는 비밀값·사용자 이름이 들어가지 않는다(설치 폴더는 <설치 폴더>, 홈 폴더는 ~ 로 가림).

표준 점검(약 1분):
  환경        Python·운영체제·실행 방식, 폴더 경로 길이, 쓰기 권한, 한글 입출력
  패키지      잠금 파일(requirements/lock-*.txt)과 설치된 버전 대조, 주요 패키지 불러오기
  묶음 무결성  SHA256SUMS.txt의 파일이 모두 있는지(압축 해제 중 한글 이름이 깨지면 여기서 드러남)
  데이터      데이터셋별 지문 확인(원본 또는 가공 파일 기록 대조)
  판정 재현    사례 A·B·C·D를 템플릿 모드로 새로 판정해 제출본 기록과 수치·판정이 같은지
              Claude 실행 기록을 API 없이 재실행해 재현되는지
  판정서·화면  1장 PDF 인쇄, 화면 첫 페이지 그리기
  보안        API 키는 설정 여부만, 화면이 이 PC 안에서만 열리는지
--full: 묶음 파일 전체의 SHA-256 대조와 템플릿 기록 재실행까지 더 한다(수 분).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from jverdict.config import OUTPUT_DIR, ROOT, now_iso, secret_status

PASS, WARN, FAIL, INFO = "통과", "주의", "실패", "정보"
MARK = {PASS: "✓", WARN: "!", FAIL: "✗", INFO: "·"}
CORE_PKGS = ("numpy", "pandas", "scipy", "statsmodels", "duckdb", "pyarrow", "streamlit", "plotly", "anthropic", "jinja2")


@dataclass
class Check:
    section: str
    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0


def mask(s) -> str:
    """보고서용: 설치 폴더·홈 폴더 경로와 비밀값을 가린다."""
    from jverdict.agent.llm import sanitize
    s = str(s)
    for p, label in ((str(ROOT), "<설치 폴더>"), (str(Path.home()), "~")):
        if p and len(p) > 3:
            s = s.replace(p, label).replace(p.replace("\\", "/"), label)
    return sanitize(s)


def build_info() -> dict:
    p = ROOT / "BUILD_INFO.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def runtime_kind() -> str:
    exe = Path(sys.executable).resolve()
    if (ROOT / "runtime") in exe.parents:
        return "설치 묶음(runtime)"
    if sys.prefix != sys.base_prefix:
        return "가상환경(.venv)"
    return "시스템 Python"


def lock_file() -> Path | None:
    info = build_info()
    if info.get("lock") and (ROOT / info["lock"]).exists():
        return ROOT / info["lock"]
    cands = sorted((ROOT / "requirements").glob("lock-*.txt")) if (ROOT / "requirements").exists() else []
    return cands[0] if cands else None


def read_lock(p: Path) -> dict[str, str]:
    """잠금 파일의 '이름==버전'. 이 PC에 해당하지 않는 항목(예: Windows 전용 ; sys_platform == "win32")은 뺀다."""
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s;#]+)\s*(?:;(.*))?$", line)
        if not m:
            continue
        if m.group(3):
            from packaging.markers import Marker
            if not Marker(m.group(3).strip()).evaluate():
                continue
        out[_norm(m.group(1))] = m.group(2)
    return out


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def longest_path(root: Path) -> tuple[int, str]:
    best, where = 0, ""
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            if len(p) > best:
                best, where = len(p), p
    return best, where


def long_paths_enabled() -> bool | None:
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
    except OSError:
        return False


# ------------------------------------------------------------------------------------------------ 점검 항목
class Doctor:
    def __init__(self, full: bool = False, run_cases: bool = True, verbose: bool = True):
        self.full, self.run_cases, self.verbose = full, run_cases, verbose
        self.checks: list[Check] = []
        self.tag = time.strftime("%Y%m%d-%H%M%S")

    def add(self, section, name, status, detail="", seconds=0.0):
        c = Check(section, name, status, mask(detail), round(seconds, 1))
        self.checks.append(c)
        if self.verbose:
            t = f" ({c.seconds:.0f}초)" if c.seconds >= 1 else ""
            print(f"  {MARK[status]} [{status}] {name}{t}" + (f" — {c.detail}" if c.detail else ""), flush=True)
        return c

    def guard(self, section, name, fn):
        """점검 하나가 예외로 죽어도 다음 점검은 계속한다."""
        t0 = time.time()
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)[-1]
            return self.add(section, name, FAIL, f"{type(e).__name__}: {e} ({Path(tb.filename).name}:{tb.lineno})",
                            time.time() - t0)
        if res is None:
            return None
        status, detail = res
        return self.add(section, name, status, detail, time.time() - t0)

    # -------------------------------------------------------------------- 환경
    def env(self):
        S = "환경"
        info = build_info()
        want = info.get("python")
        have = platform.python_version()
        self.add(S, "Python", PASS if not want or want == have else WARN,
                 f"{have} ({platform.architecture()[0]})" + (f", 묶음 기준 {want}" if want and want != have else ""))
        self.add(S, "운영체제", INFO, f"{platform.system()} {platform.release()} ({platform.version()}) {platform.machine()}")
        self.add(S, "실행 방식", INFO, f"{runtime_kind()} · {sys.executable}")
        self.add(S, "설치 폴더", INFO, f"{ROOT} (경로 길이 {len(str(ROOT))}자)")

        def paths():
            n, where = longest_path(ROOT / "runtime" if (ROOT / "runtime").exists() else ROOT)
            lp = long_paths_enabled()
            if os.name == "nt" and n >= 259 and not lp:
                return FAIL, (f"가장 긴 파일 경로 {n}자 ≥ 260자(Windows 한도). 'C:\\JVerdict'처럼 짧은 폴더에 다시 "
                              f"푸세요: {where}")
            if os.name == "nt" and n >= 230 and not lp:
                return WARN, f"가장 긴 파일 경로 {n}자(한도 260자에 가까움). 짧은 폴더로 옮기는 것을 권장"
            return PASS, f"가장 긴 파일 경로 {n}자" + ("" if lp is None else f", 긴 경로 허용 {'켜짐' if lp else '꺼짐'}")
        self.guard(S, "파일 경로 길이", paths)

        def writable():
            d = OUTPUT_DIR / "doctor"
            d.mkdir(parents=True, exist_ok=True)
            t = d / f".write-test-{os.getpid()}"
            t.write_text("한글 쓰기 시험", encoding="utf-8")
            ok = t.read_text(encoding="utf-8") == "한글 쓰기 시험"
            t.unlink()
            return (PASS, "outputs 폴더에 쓰고 읽기 가능") if ok else (FAIL, "쓴 내용과 읽은 내용이 다름")
        self.guard(S, "결과 폴더 쓰기 권한", writable)
        utf8 = sys.flags.utf8_mode
        enc = getattr(sys.stdout, "encoding", None)
        self.add(S, "한글 입출력", PASS if utf8 else WARN,
                 f"UTF-8 모드 {'켜짐' if utf8 else '꺼짐(실행.bat로 실행하면 켜짐)'}, 화면 출력 {enc}")

    # -------------------------------------------------------------------- 패키지
    def packages(self):
        S = "패키지"
        lf = lock_file()
        if lf is None:
            self.add(S, "잠금 파일 대조", INFO, "requirements/lock-*.txt 없음(개발 환경)")
        else:
            lock = read_lock(lf)
            diff, missing = [], []
            for name, ver in lock.items():
                try:
                    have = md.version(name)
                except md.PackageNotFoundError:
                    missing.append(name)
                    continue
                if have != ver:
                    diff.append(f"{name} {have}(잠금 {ver})")
            st = FAIL if missing else (WARN if diff else PASS)
            det = f"{lf.name}: {len(lock)}개 중 일치 {len(lock) - len(diff) - len(missing)}"
            if missing:
                det += f", 없음 {', '.join(missing[:8])}"
            if diff:
                det += f", 버전 다름 {', '.join(diff[:8])} (판정 수치가 마지막 자리에서 다를 수 있음)"
            self.add(S, "잠금 파일 대조", st, det)
        bad = []
        t0 = time.time()
        for m in CORE_PKGS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    importlib.import_module(m)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{m}({type(e).__name__}: {str(e)[:120]})")
        self.add(S, "주요 패키지 불러오기", FAIL if bad else PASS,
                 ("불러오기 실패: " + "; ".join(bad)) if bad else ", ".join(f"{m} {md.version(m)}" for m in CORE_PKGS),
                 time.time() - t0)

    # -------------------------------------------------------------------- 묶음 무결성
    def integrity(self):
        S = "묶음 무결성"
        sums = ROOT / "SHA256SUMS.txt"
        if not sums.exists():
            self.add(S, "파일 목록 대조", INFO, "SHA256SUMS.txt 없음(개발 폴더에서 실행 중)")
            return

        def check():
            from jverdict.data.sources import sha256_file
            rows = [ln.split("  ", 1) for ln in sums.read_text(encoding="utf-8").splitlines() if "  " in ln]
            missing, changed = [], []
            for h, rel in rows:
                p = ROOT / rel
                if not p.exists():
                    missing.append(rel)
                elif self.full and sha256_file(p) != h:
                    changed.append(rel)
            if missing:
                hint = (" — 한글 파일 이름이 깨졌을 수 있습니다. 반디집·7-Zip 등으로 다시 압축을 푸세요"
                        if any(re.search(r"[가-힣]", m) for m in missing) else "")
                return FAIL, f"{len(rows)}개 중 {len(missing)}개 없음: {', '.join(missing[:5])}{hint}"
            if changed:
                return FAIL, f"내용이 바뀐 파일 {len(changed)}개: {', '.join(changed[:5])}"
            return PASS, f"{len(rows)}개 파일 모두 있음" + (" · 전체 SHA-256 일치" if self.full else
                                                     " (내용 대조는 --full)")
        self.guard(S, "파일 목록 대조", check)

    # -------------------------------------------------------------------- 데이터
    def data(self):
        S = "데이터"
        from jverdict.agent import catalog
        from jverdict.config import RAW_DIR
        for ds in catalog.DATASETS.values():
            if ds.synthetic:
                continue

            def fp(ds=ds):
                f = catalog.fingerprint(ds.id)
                basis = "원본 CSV" if (RAW_DIR / ds.source).exists() else "가공 파일 ↔ 원본 지문 기록 대조"
                return PASS, f"{basis} · 지문 {f['sha256'][:16]}…"
            self.guard(S, ds.title, fp)

    # -------------------------------------------------------------------- 판정 재현
    def reproduce(self):
        S = "판정 재현"
        from jverdict.agent import design as DZ
        from jverdict.agent import orchestrator as O
        from jverdict.agent.library import CARDS
        if not self.run_cases:
            self.add(S, "사례 실행", INFO, "건너뜀(--no-run)")
            return
        warnings.filterwarnings("ignore")
        made: list[Path] = []
        try:
            for case in CARDS:
                ref = O.RUNS / f"case-{case}-template" / "manifest.json"

                def fresh(case=case, ref=ref):
                    rid = f"_doctor-{self.tag}-{case}"
                    made.append(O.RUNS / rid)
                    r = O.run(CARDS[case]["card"]["question"], mode="template", run_id=rid)
                    label = r.skeptic["final_label"]
                    if not ref.exists():
                        return INFO, f"판정 '{label}' (비교할 제출본 기록 없음)"
                    man = json.loads(ref.read_text(encoding="utf-8"))
                    nums_ok = DZ.numbers_of(r.analysis) == man["numbers"]
                    ok = nums_ok and label == man["final_verdict"]
                    if ok:
                        return PASS, f"판정 '{label}' · 핵심 수치까지 제출본 기록과 같음"
                    return FAIL, (f"판정 '{label}'(기록 '{man['final_verdict']}') · 수치 {'같음' if nums_ok else '다름'}: "
                                  f"{DZ.numbers_of(r.analysis)} vs {man['numbers']}")
                self.guard(S, f"사례 {case} 새로 판정(템플릿)", fresh)

            runs = sorted(p for p in O.RUNS.glob("case-*") if (p / "manifest.json").exists()
                          and not p.name.endswith("-replay"))
            for rd in runs:
                man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
                if man["mode"] == "template" and not self.full:
                    continue

                def rep(rd=rd, man=man):
                    rid = f"_doctor-{self.tag}-{rd.name}"
                    made.append(O.RUNS / rid)
                    out = O.replay(rd, run_id=rid)
                    shipped = rd.parent / f"{rd.name}-replay" / "replay_check.json"
                    known_bad = shipped.exists() and not json.loads(shipped.read_text(encoding="utf-8"))["reproduced"]
                    extra = "" if out.get("code_equal") else " · 코드 지문은 기록 당시와 다름(정보)"
                    if out["reproduced"]:
                        n = (man.get("llm") or {}).get("calls")
                        return PASS, (f"수치·판정·판정서 같음" + (f" · Claude 응답 {n}건을 API 없이 재사용" if n else "")
                                      + extra)
                    if known_bad:
                        return WARN, f"원래 PC에서도 재현되지 않는 기록(알려진 문제): {out.get('error', '')}"
                    return FAIL, f"재현 안 됨: {out.get('error') or out}"
                self.guard(S, f"{rd.name} 재실행(API 없이)", rep)
        finally:
            for d in made:
                shutil.rmtree(d, ignore_errors=True)

    # -------------------------------------------------------------------- 판정서·화면
    def outputs(self):
        S = "판정서·화면"
        from jverdict.agent import orchestrator as O
        from jverdict.report import onepage

        def pdf():
            b = onepage.find_browser()
            src = O.RUNS / "case-C-template"
            if not src.exists():
                return INFO, "시험할 판정 기록 없음"
            with tempfile.TemporaryDirectory() as td:
                res = onepage.make(src, out_dir=Path(td))
                if res["pdf"] is None:
                    return WARN, "Chrome·Edge를 찾지 못해 HTML 판정서만 만듦(브라우저에서 인쇄하면 같은 판정서)"
                ok = res["pages"] == 1
                return (PASS if ok else FAIL), f"{res['pages']}쪽 · 글자 {res['font_pt']}pt · 브라우저 {Path(b).name}"
        self.guard(S, "1장 판정서 PDF", pdf)

        def ui():
            from streamlit import logger as st_logger
            from streamlit.testing.v1 import AppTest
            st_logger.set_log_level("error")          # 화면 없이 그릴 때 나오는 무해한 경고를 숨김
            app = Path(onepage.__file__).resolve().parents[1] / "ui" / "app.py"
            at = AppTest.from_file(str(app), default_timeout=90).run()
            if at.exception:
                return FAIL, f"화면 오류: {at.exception[0].message[:200]}"
            return PASS, f"첫 화면 그리기 성공 ('{at.title[0].value}')"
        self.guard(S, "화면(첫 페이지)", ui)

    # -------------------------------------------------------------------- 보안
    def security(self):
        S = "보안"
        st = secret_status()
        has_env = (ROOT / ".env").exists()
        self.add(S, "API 키", INFO, f".env {'있음' if has_env else '없음'} · " +
                 ", ".join(f"{k} {v}" for k, v in st.items()) +
                 (" (키가 없으면 템플릿 모드로 판정. 오프라인에서도 동작)" if st.get("ANTHROPIC_API_KEY") != "설정됨" else ""))

        def cfg():
            p = ROOT / ".streamlit" / "config.toml"
            if not p.exists():
                return FAIL, "화면 설정 파일(.streamlit/config.toml) 없음"
            txt = p.read_text(encoding="utf-8")
            ok = re.search(r'^address\s*=\s*"localhost"', txt, re.M) and re.search(
                r"^gatherUsageStats\s*=\s*false", txt, re.M)
            return (PASS, "화면은 이 PC 안(localhost)에서만 열림 · 사용 통계 전송 꺼짐") if ok else (
                FAIL, "화면 설정이 바뀌어 있음(localhost·통계 끔 확인 필요)")
        self.guard(S, "화면 접속 범위", cfg)

    # -------------------------------------------------------------------- 전체
    def run(self) -> list[Check]:
        steps = [("환경", self.env), ("패키지", self.packages), ("묶음 무결성", self.integrity), ("데이터", self.data),
                 ("판정 재현", self.reproduce), ("판정서·화면", self.outputs), ("보안", self.security)]
        for title, fn in steps:
            if self.verbose:
                print(f"\n[{title}]", flush=True)
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self.add(title, "점검 실행", FAIL, f"{type(e).__name__}: {e}")
        return self.checks

    def report(self) -> tuple[Path, Path, dict]:
        cnt = {s: sum(c.status == s for c in self.checks) for s in (PASS, WARN, FAIL, INFO)}
        info = build_info()
        head = {"created": now_iso(), "mode": "full" if self.full else "standard", "summary": cnt,
                "build": {k: info.get(k) for k in ("version", "built", "python", "code_sha256") if k in info},
                "jverdict_src": mask(Path(__file__).resolve().parent)}
        d = OUTPUT_DIR / "doctor"
        d.mkdir(parents=True, exist_ok=True)
        jp, tp = d / f"점검_{self.tag}.json", d / f"점검_{self.tag}.txt"
        jp.write_text(json.dumps({**head, "checks": [asdict(c) for c in self.checks]}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        L = [f"정책 판정관 설치 점검 보고서 ({head['created']}, {'전체' if self.full else '표준'} 점검)",
             f"결과: 통과 {cnt[PASS]} · 주의 {cnt[WARN]} · 실패 {cnt[FAIL]} · 정보 {cnt[INFO]}", ""]
        if head["build"]:
            L += [f"묶음: {json.dumps(head['build'], ensure_ascii=False)}", ""]
        sec = None
        for c in self.checks:
            if c.section != sec:
                sec = c.section
                L.append(f"[{sec}]")
            L.append(f"  {MARK[c.status]} [{c.status}] {c.name}" + (f" — {c.detail}" if c.detail else ""))
        tp.write_text("\n".join(L) + "\n", encoding="utf-8")
        return tp, jp, cnt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jverdict.doctor", description="정책 판정관 설치 점검")
    ap.add_argument("--full", action="store_true", help="묶음 파일 전체 SHA-256 대조, 템플릿 기록 재실행까지")
    ap.add_argument("--no-run", action="store_true", help="사례 판정·재실행을 건너뜀(빠른 점검)")
    a = ap.parse_args(argv)
    print("정책 판정관 설치 점검을 시작합니다." + (" (전체 점검: 몇 분 걸립니다)" if a.full else " (1~2분)"), flush=True)
    doc = Doctor(full=a.full, run_cases=not a.no_run)
    doc.run()
    tp, _jp, cnt = doc.report()
    print(f"\n결과: 통과 {cnt[PASS]} · 주의 {cnt[WARN]} · 실패 {cnt[FAIL]} · 정보 {cnt[INFO]}")
    print(f"보고서: {mask(tp)}")
    if cnt[FAIL]:
        print("실패 항목이 있습니다. 위 보고서 파일을 개발자에게 보내 주세요.")
    else:
        print("이 PC에서 정책 판정관을 쓸 수 있습니다. 실행.bat 로 화면을 여세요.")
    return 1 if cnt[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
