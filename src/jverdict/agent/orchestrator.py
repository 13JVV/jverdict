"""상태머신: 순서는 코드가 고정하고, 단계를 건너뛸 수 없다.

    parse → confirm(사람 확인 1회) → scout → design → lock → execute → skeptic → report → manifest

모드
- template : LLM 없이 등록 사례·규칙 기반으로 동작(오프라인). 판정 수치와 등급은 claude 모드와 같다.
- claude   : 각 단계 안에서 Claude가 판단한다. API 오류가 나면 그 단계만 템플릿으로 대체하고 기록한다.
- replay   : 이전 실행의 LLM 응답 캐시로 똑같이 다시 실행한다(API 호출 없음). 입력이 다르면 멈춘다.

결과 폴더 outputs/runs/<실행 번호>/
    events.jsonl, card.json, scout.json, prereg.md, design.json, prereg.lock.json, analysis.json,
    skeptic.json, report.md, report.json, llm_cache.json, manifest.json
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from jverdict.causal.verdict import RULES_VERSION
from jverdict.config import OUTPUT_DIR, ROOT, now_iso

from . import catalog
from . import design as DZ
from . import prereg as PR
from . import roles as R
from .llm import LLM, LLMError, ReplayMismatch, sanitize
from .prereg import GuardError

STAGES = ["parse", "confirm", "scout", "design", "lock", "execute", "skeptic", "report", "manifest"]
RUNS = OUTPUT_DIR / "runs"


@dataclass
class Run:
    run_id: str
    run_dir: Path
    mode: str
    seed: int = 20261104
    llm: LLM | None = None
    on_event: Callable[[dict], None] | None = None
    events: list = field(default_factory=list)
    done: list = field(default_factory=list)
    card: dict | None = None
    scout: dict | None = None
    design: dict | None = None
    lock_rec: dict | None = None
    analysis: object = None
    skeptic: dict | None = None
    report: dict | None = None
    extra_checks: list = field(default_factory=list)
    scout_diag: object = None
    truth: dict | None = None
    fallbacks: list = field(default_factory=list)
    amendments: list = field(default_factory=list)
    original_result: dict | None = None
    skeptic_summary: str = ""
    question: str | None = None
    llm_off: bool = False
    as_of: str | None = None        # 사건 달력 기준일(새 실행은 오늘, 재실행은 원래 실행 날짜)

    @property
    def locked(self) -> bool:
        return "lock" in self.done

    def log(self, stage: str, msg: str, data=None):
        ev = {"t": time.strftime("%H:%M:%S"), "stage": stage, "msg": sanitize(msg)}
        if data is not None:
            ev["data"] = json.loads(sanitize(json.dumps(data, ensure_ascii=False, default=str)))
        self.events.append(ev)
        with open(self.run_dir / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if self.on_event:
            self.on_event(ev)

    def enter(self, stage: str):
        i = STAGES.index(stage)
        missing = [s for s in STAGES[:i] if s not in self.done]
        if missing:
            raise GuardError(f"'{stage}' 단계 전에 {missing} 단계를 거쳐야 함")
        self.log(stage, f"단계 시작: {stage}")

    def finish(self, stage: str):
        self.done.append(stage)

    def write(self, name: str, obj):
        txt = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        (self.run_dir / name).write_text(txt, encoding="utf-8")

    def role(self, stage: str, llm_fn, tmpl_fn, *args):
        """LLM 판을 시도하고, 실패하면 템플릿으로 대체(claude 모드). replay에서 불일치면 멈춘다."""
        if self.mode == "template" or self.llm is None or self.llm_off:
            if self.llm_off and self.mode != "template":
                self.fallbacks.append({"stage": stage, "error": "LLM 사용 불가(앞 단계의 크레딧·인증·연결 오류)"})
            return tmpl_fn(self, *args)
        try:
            return llm_fn(self, *args)
        except ReplayMismatch:
            raise
        except GuardError:
            raise
        except (LLMError, ValueError, KeyError) as e:
            # 재실행도 원래 실행과 같은 곳에서 같은 이유로 대체된다(입력 불일치만 멈춤)
            msg = sanitize(str(e))
            self.fallbacks.append({"stage": stage, "error": msg})
            self.log(stage, f"LLM 단계 실패 → 템플릿으로 대체: {msg}")
            if any(k in msg for k in ("credit balance", "401", "403", "연결할 수 없음", "설정되지 않음")):
                self.llm_off = True        # 다시 시도해도 소용없는 오류: 남은 단계는 바로 템플릿으로
                self.log(stage, "LLM을 쓸 수 없는 오류 → 남은 단계는 템플릿 모드로 진행")
            return tmpl_fn(self, *args)


def _code_fingerprint() -> str:
    """src/jverdict 아래 .py 전체의 지문. 운영체제와 무관하게 같은 값이 나오도록
    경로는 '/' 문자열로 정렬하고(Windows 경로 비교는 대소문자를 무시함), 줄바꿈은 LF로 맞춘다
    (압축·복사 도구가 CRLF로 바꿔도 같은 코드로 본다). LF 파일만 있으면 예전 지문과 같다."""
    h = hashlib.sha256()
    files = sorted(((p.relative_to(ROOT).as_posix(), p) for p in (ROOT / "src" / "jverdict").rglob("*.py")),
                   key=lambda t: t[0])
    for rel, p in files:
        h.update(rel.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _versions() -> dict:
    import importlib.metadata as md
    out = {"python": sys.version.split()[0], "platform": f"{platform.system()} {platform.machine()}"}
    for pkg in ("numpy", "pandas", "scipy", "duckdb", "statsmodels", "anthropic"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def run(question: str | None = None, *, card: dict | None = None, mode: str = "template",
        confirm: Callable[[dict], dict] | None = None, on_event=None, replay_dir: Path | None = None,
        run_id: str | None = None, seed: int = 20261104, as_of: str | None = None) -> Run:
    """질문(또는 정책 카드)에서 판정서까지. confirm은 정책 카드를 사람이 확인·수정하는 함수(없으면 그대로 승인)."""
    rid = run_id or time.strftime("%Y%m%d-%H%M%S")
    rdir = RUNS / rid
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "events.jsonl").write_text("", encoding="utf-8")
    llm = None
    if mode == "claude":
        llm = LLM("claude")
    elif mode == "replay":
        cache = json.loads((replay_dir / "llm_cache.json").read_text(encoding="utf-8"))
        llm = LLM("replay", cache=cache) if cache else None
    r = Run(rid, rdir, mode, seed, llm, on_event)
    r.question = question
    r.as_of = as_of or now_iso()[:10]
    r.log("parse", f"실행 시작: 모드 {mode}, 판정 규칙 {RULES_VERSION}")

    # ① 정책 카드
    r.enter("parse")
    if card is None:
        if not question:
            raise ValueError("질문 또는 정책 카드가 필요함")
        card = r.role("parse", R.parser_llm, R.parser_template, question)
    r.card = card
    r.finish("parse")

    r.enter("confirm")
    if confirm is not None:
        r.card = confirm(r.card)
    r.card.setdefault("needed_hint", [])
    r.card.setdefault("prereg", None)
    r.write("card.json", r.card)
    r.log("confirm", "정책 카드 확인 완료", {k: r.card.get(k) for k in ("policy", "start_date", "treated", "outcome")})
    r.finish("confirm")

    # ② Data Scout
    r.enter("scout")
    r.scout = r.role("scout", R.scout_llm, R.scout_template)
    r.write("scout.json", r.scout)
    r.log("scout", f"데이터 점검: {r.scout['quality_verdict']}",
          {"dataset": r.scout["dataset"], "cap": r.scout.get("quality_cap"), "coverage": r.scout["coverage_summary"]})
    if r.scout.get("quality_cap") == "판정 보류":
        r.log("scout", "측정 단절 의심 → 판정 상한 '판정 보류'로 고정")
    r.finish("scout")

    # ③ Design (효과 가림)
    r.enter("design")
    existing = PR.verify_existing(r.card["prereg"]) if r.card.get("prereg") else None
    if existing:
        r.log("design", f"기존 사전 등록 문서 사용: {existing['file']} (지문 확인)")
    d = r.role("design", R.design_llm, R.design_template)
    errs = DZ.validate(d, r.card)
    if errs:
        r.log("design", "설계 검증 실패 → 템플릿 설계로 대체", {"errors": errs})
        r.fallbacks.append({"stage": "design", "error": "; ".join(errs)})
        d = R.design_template(r)
    r.design = d
    r.finish("design")

    # 잠금
    r.enter("lock")
    md = DZ.prereg_markdown(r.card, d, r.scout, rid)
    r.lock_rec = PR.lock(rdir, md, d, existing)
    r.lock_rec["run_id"] = rid
    r.log("lock", f"설계 잠금 SHA-256 {r.lock_rec['sha256'][:16]}…")
    r.finish("lock")

    # ④ Executor
    r.enter("execute")
    PR.verify(rdir)
    tries, last = 0, None
    while tries <= 3:
        try:
            r.analysis, info = DZ.execute(r.card, r.design, seed=seed)
            break
        except Exception as e:  # noqa: BLE001
            last = sanitize(f"{type(e).__name__}: {e}")
            tries += 1
            r.log("execute", f"실행 오류({tries}/3): {last}")
            if tries > 3:
                raise RuntimeError(f"설계를 실행하지 못함: {last}") from None
            nd = r.role("execute", R.repair_llm, R.repair_template, r.design, last)
            if nd is None:
                raise RuntimeError(f"설계를 실행하지 못하고 고칠 방법도 없음: {last}") from None
            if r.original_result is None:
                r.original_result = {"design": r.design, "error": last}
            r.lock_rec = PR.amend(rdir, nd, f"실행 오류 복구: {last}")
            r.amendments.append(r.lock_rec["amendments"][-1])
            r.design = nd
            PR.verify(rdir)
    if catalog.DATASETS[r.design["dataset"]].synthetic:
        r.truth = catalog.synthetic_truth(seed, r.design["dataset"])
    r.write("analysis.json", r.analysis.to_dict())
    r.log("execute", f"판정 규칙 결과: {r.analysis.verdict.label}", DZ.numbers_of(r.analysis))
    r.finish("execute")

    # ⑤ Skeptic
    r.enter("skeptic")
    hs = r.role("skeptic", R.skeptic_llm, R.skeptic_template)
    r.skeptic = R.resolve_skeptic(r, hs)
    r.write("skeptic.json", r.skeptic)
    sk = r.skeptic
    r.log("skeptic", f"반론 {len(sk['rule_hypotheses']) + len(hs)}개 검증: 판정 규칙이 이미 반영한 성립 {len(sk['rule_standing'])}개, "
                     f"추가 반론 성립 {len(sk['standing'])}개 → 최종 {sk['final_label']}"
                     + (f" (Skeptic 하향 {sk['downgraded'][0]} → {sk['downgraded'][1]})" if sk["downgraded"] else ""))
    r.finish("skeptic")

    # ⑥ Reporter
    r.enter("report")
    facts = R.facts_sheet(r)
    rep = r.role("report", R.reporter_llm, R.reporter_template, facts)
    bad = R.number_check([rep["headline"], rep["summary"]] + rep["evidence"] + rep["limits"] + rep["cannot_say"]
                         + rep["needed_data"], facts)
    if bad and r.mode != "template":
        r.log("report", "서술 숫자 대조 실패 → 템플릿 서술로 대체", {"numbers": bad})
        rep = R.reporter_template(r, facts)
    r.report = rep
    r.write("facts.txt", facts)
    r.write("report.json", rep)
    r.write("report.md", R.render_markdown(r, rep, facts))
    from jverdict.report.naive import _safe as _naive
    nv = _naive(r.card, r.design, r.seed)          # 대비용 '단순 비교였다면'(판정에는 쓰지 않음, LLM에도 보내지 않음)
    if nv:
        r.write("naive.json", nv)
    r.log("report", "판정서 작성 완료")
    r.finish("report")

    # manifest
    r.enter("manifest")
    write_manifest(r)
    r.finish("manifest")
    r.log("manifest", "재실행 기록 저장")
    return r


def write_manifest(r: Run):
    llm_sum = r.llm.summary() if r.llm else {"calls": 0, "cost_usd": 0.0, "mode": r.mode}
    if r.llm is None:
        cache = []
    elif r.mode == "replay":
        cache = r.llm.cache                 # 재실행은 원래 기록을 그대로 보존
    else:
        cache = r.llm.cache_dump()
    r.write("llm_cache.json", cache)
    man = {"run_id": r.run_id, "created": now_iso(), "calendar_as_of": r.as_of, "mode": r.mode,
           "rules_version": RULES_VERSION, "seed": r.seed, "question_input": r.question, "question": r.card.get("question"),
           "card": r.card, "data": catalog.fingerprint(r.design["dataset"]),
           "prereg": {"sha256": r.lock_rec["sha256"], "locked_at": r.lock_rec["locked_at"],
                      "existing": r.lock_rec.get("existing_prereg"), "amendments": r.amendments},
           "code_sha256": _code_fingerprint(), "versions": _versions(),
           "numbers": DZ.numbers_of(r.analysis), "final_verdict": r.skeptic["final_label"],
           "llm": llm_sum, "fallbacks": r.fallbacks,
           "files": sorted(p.name for p in r.run_dir.iterdir())}
    r.write("manifest.json", man)


def _rerun(run_dir: Path, man: dict, rid: str) -> "Run":
    mode = "replay" if man["mode"] in ("claude", "replay") else "template"
    confirmed = man["card"]
    q = man.get("question_input")
    # 원래 질문으로 Parser부터 다시 돌리고(캐시 순서 유지), 사람이 확인한 카드는 확인 단계에서 그대로 적용한다
    return run(q, card=None if q else confirmed, mode=mode, replay_dir=run_dir, confirm=lambda c: confirmed,
               run_id=rid, seed=man["seed"], as_of=man.get("calendar_as_of") or man["created"][:10])


def backfill_key_norm(run_dir: Path) -> dict:
    """정규화 지문(key_norm)이 생기기 전에 만든 Claude 기록에 key_norm을 채운다.

    같은 조건으로 다시 돌려서 모든 호출이 '정확한 지문'(입력이 비트 단위로 같음)으로 맞고, 수치·판정·판정서까지
    재현될 때만 쓴다. 이때 지금 계산한 key_norm은 원래 입력의 key_norm과 같다. 응답·지문(key) 등 기존 값은
    건드리지 않고 key_norm 칸만 더한다. 이후에는 다른 PC(예: Windows)에서 부동소수 마지막 비트가 달라도 재현된다.
    """
    run_dir = Path(run_dir)
    cache_p = run_dir / "llm_cache.json"
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else []
    need = [i for i, rec in enumerate(cache) if rec.get("key_norm") is None]
    out = {"run": run_dir.name, "calls": len(cache), "missing": len(need), "filled": 0}
    if not need:
        return {**out, "status": "채울 것 없음"}
    rid = f"_backfill-{run_dir.name}"
    try:
        new = _rerun(run_dir, man, rid)
        ok = (DZ.numbers_of(new.analysis) == man["numbers"] and new.skeptic["final_label"] == man["final_verdict"])
        got = dict(new.llm.norm_backfill) if new.llm is not None else {}
    except ReplayMismatch as e:
        return {**out, "status": "재현 안 됨(채우지 않음)", "error": sanitize(str(e))}
    finally:
        import shutil
        shutil.rmtree(RUNS / rid, ignore_errors=True)
    if not ok or sorted(got) != need:
        return {**out, "status": "정확한 지문으로 다 맞지 않음(채우지 않음)"}
    for i in need:
        cache[i]["key_norm"] = got[i]
    cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**out, "filled": len(need), "status": "채움"}


def replay(run_dir: Path, *, run_id: str | None = None) -> dict:
    """이전 실행을 같은 조건으로 다시 돌려 핵심 수치·판정·판정서가 같은지 확인한다."""
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # 재실행 폴더 이름은 원래 실행 '폴더' 이름을 따른다(폴더 이름을 바꾼 기록도 판정서·요약표가 짝을 찾도록)
    rid = run_id or f"{Path(run_dir).name}-replay"
    try:
        new = _rerun(run_dir, man, rid)
    except ReplayMismatch as e:
        # 실패도 기록한다(예전 성공 기록이 남아 '재현됨'으로 보이지 않게)
        out = {"original": man["run_id"], "replay": rid, "numbers_equal": None, "verdict_equal": None,
               "report_equal": None, "data_equal": catalog.fingerprint(man["data"]["dataset"]) == man["data"],
               "code_equal": _code_fingerprint() == man["code_sha256"], "reproduced": False,
               "error": sanitize(str(e))}
        (RUNS / rid).mkdir(parents=True, exist_ok=True)
        (RUNS / rid / "replay_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
    old_md = (run_dir / "report.md").read_text(encoding="utf-8")
    new_md = (new.run_dir / "report.md").read_text(encoding="utf-8")
    strip = lambda s: "\n".join(l for l in s.splitlines() if "실행 번호" not in l and "사전 등록 지문" not in l)  # noqa: E731
    out = {"original": man["run_id"], "replay": new.run_id,
           "numbers_equal": DZ.numbers_of(new.analysis) == man["numbers"],
           "verdict_equal": new.skeptic["final_label"] == man["final_verdict"],
           "report_equal": strip(old_md) == strip(new_md),
           "data_equal": catalog.fingerprint(new.design["dataset"]) == man["data"],
           "code_equal": _code_fingerprint() == man["code_sha256"]}
    out["reproduced"] = all(out[k] for k in ("numbers_equal", "verdict_equal", "report_equal"))
    new.write("replay_check.json", out)
    return out
