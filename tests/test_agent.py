"""Agent: 상태머신 가드레일, 효과 가림, 잠금, 숫자 대조, 비밀값 보호, 재현성 (API 호출 없음)."""
import json
import shutil

import pandas as pd
import pytest

from jverdict.agent import catalog
from jverdict.agent import design as DZ
from jverdict.agent import orchestrator as O
from jverdict.agent import prereg as PR
from jverdict.agent import roles as R
from jverdict.agent import tools as T
from jverdict.agent.library import CARDS, events_between, match_card
from jverdict.agent.llm import LLM, ReplayMismatch, sanitize
from jverdict.config import RAW_DIR, ROOT

HAS_CARD = (RAW_DIR / "업종별성별카드이용정보.csv").exists()


@pytest.fixture
def runs_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "RUNS", tmp_path)
    return tmp_path


def _run_obj(tmp_path, case="C"):
    r = O.Run("t", tmp_path, "template")
    (tmp_path / "events.jsonl").write_text("")
    r.card = R._lib_card(case)
    return r


# ------------------------------------------------------------------ 정책 카드
def test_card_matching():
    assert match_card("청소년 버스 무료 효과 있었어?") == "A"
    assert match_card("재난지원금 효과는?") == "B"
    assert match_card("유커가 돌아와서 소비가 늘었나") == "D"
    assert match_card(CARDS["C"]["card"]["question"]) == "C"
    assert match_card("버스 요금을 올리면?") is None


# ------------------------------------------------------------------ 가드레일
def test_stage_order_cannot_be_skipped(tmp_path):
    r = _run_obj(tmp_path)
    r.done = ["parse", "confirm", "scout", "design"]
    with pytest.raises(PR.GuardError):
        r.enter("execute")            # 잠금 없이 실행 불가
    r.done.append("lock")
    r.enter("execute")
    with pytest.raises(PR.GuardError):
        r.enter("report")             # Skeptic을 거치지 않으면 판정서 불가


def test_effect_tool_requires_lock(tmp_path):
    r = _run_obj(tmp_path)
    with pytest.raises(PR.GuardError):
        T.call(r, "skeptic", "extra_check", {"kind": "alt_window", "reference": [], "placebo_target": None,
                                             "post": 90, "start": None, "end": None, "reason": "x"})
    out, err = T.call(r, "scout", "extra_check", {})
    assert err and "단계에서 쓸 수 없음" in out


def test_prereg_tamper_is_detected(tmp_path):
    d = CARDS["C"]["design"]
    PR.lock(tmp_path, "# 문서", d)
    PR.verify(tmp_path)
    bad = dict(d, reference=["경로"])
    (tmp_path / "design.json").write_text(json.dumps(bad, ensure_ascii=False))
    with pytest.raises(PR.GuardError):
        PR.verify(tmp_path)
    PR.lock(tmp_path, "# 문서", d)
    PR.amend(tmp_path, bad, "시험용 변경")      # 기록을 남기면 통과
    rec = PR.verify(tmp_path)
    assert rec["amendments"][0]["changed"] == ["reference"]
    assert "변경 기록 1" in (tmp_path / "prereg.md").read_text()


def test_existing_prereg_hash_checked(tmp_path):
    PR.verify_existing("docs/prereg/case_relief2020_engine.md")
    src = ROOT / "docs" / "prereg"
    (tmp_path / "docs" / "prereg").mkdir(parents=True)
    for f in ("case_relief2020_engine.md", "case_relief2020_engine.lock.json"):
        shutil.copy(src / f, tmp_path / "docs" / "prereg" / f)
    (tmp_path / "docs/prereg/case_relief2020_engine.md").write_text("바뀐 문서", encoding="utf-8")
    import jverdict.agent.prereg as P
    old = P.ROOT
    P.ROOT = tmp_path
    try:
        with pytest.raises(PR.GuardError):
            PR.verify_existing("docs/prereg/case_relief2020_engine.md")
    finally:
        P.ROOT = old


# ------------------------------------------------------------------ 효과 가림
def test_design_stage_sees_only_pre_policy_data(tmp_path):
    r = _run_obj(tmp_path)
    prof = T.t_dataset_profile(r, "synthetic_bus")
    assert pd.Timestamp(prof["period"][1]) < pd.Timestamp(r.card["start_date"])
    q = T.t_quality_check(r, "synthetic_bus", CARDS["C"]["design"]["series"], "청소년", ["일반"], [])
    assert "효과 가림" in q["message"] and "%" not in q["message"]


def test_events_are_scoped_to_dataset():
    names = {e["name"] for e in events_between("2022-01-01", "2022-12-31", "synthetic_bus")}
    assert not any("결측" in n or "코로나" in n for n in names)
    assert any("결측" in e["name"] for e in events_between("2022-01-01", "2022-12-31", "bus_station"))


# ------------------------------------------------------------------ 서술·비밀값
def test_number_check_flags_invented_numbers():
    facts = "추정 효과: +19.6% (판정 구간 +15.6% ~ +25.2%, p=0.028)"
    assert R.number_check(["효과는 +19.6%이고 구간은 +15.6% ~ +25.2%입니다."], facts) == []
    assert R.number_check(["효과는 약 20%입니다."], facts) == ["20"]
    assert R.number_check(["8월 방학과 겹칩니다."], facts) == []          # 달 이름은 통계 수치가 아님
    assert R.number_check(["8일 동안"], facts) == ["8"]


def test_sanitize_strips_secrets():
    s = sanitize("오류 sk-ant-api03-ABCDEFGHIJKLMNOP 와 https://x.kr/api?serviceKey=SECRET123&a=1")
    assert "sk-ant" not in s and "SECRET123" not in s and "[비밀값 제거]" in s


def test_replay_detects_changed_input():
    llm = LLM("replay", cache=[{"role": "parser", "key": "다른 입력", "response": {}}])
    with pytest.raises(ReplayMismatch):
        llm.json_call("parser", "시스템", "질문", {"type": "object", "properties": {}, "required": [],
                                                 "additionalProperties": False})


# ------------------------------------------------------------------ 설계 검증
def test_design_validation_catches_errors():
    c = CARDS["C"]
    assert DZ.validate(c["design"], c["card"]) == []
    bad = dict(c["design"], reference=["없는계열"])
    assert any("없는계열" in e for e in DZ.validate(bad, c["card"]))
    bad2 = dict(c["design"], series=[dict(c["design"]["series"][0], groups=["외계인"])] + c["design"]["series"][1:])
    assert any("외계인" in e for e in DZ.validate(bad2, c["card"]))


@pytest.mark.skipif(not HAS_CARD, reason="카드 원본 없음")
def test_card_panel_reproduces_case_d_panel():
    from jverdict.cases import china2023 as D
    p0 = D.build_panel(D.load(), D.PRE_AMEND1)
    p1 = catalog.card_panel(["중국", "일본", "동남아", "기타외국"], CARDS["D"]["design"]["panel"]["months"])
    assert len(p0) == len(p1)
    assert p0.y.sum() == pytest.approx(p1.y.sum(), abs=1e-9)


# ------------------------------------------------------------------ Skeptic 하향
def test_skeptic_downgrades_only_on_new_standing_objection(tmp_path, runs_tmp):
    r = O.run(CARDS["C"]["card"]["question"], mode="template", run_id="sk")
    assert r.skeptic["final_level"] == "효과 있음"
    r.extra_checks.append({"id": "X9", "status": "반론 성립", "label": "가짜", "pct": 0.0})
    res = R.resolve_skeptic(r, [{"hypothesis": "다른 사건", "check_ids": ["X9"]}])
    assert res["downgraded"] == ("효과 있음", "조건부") and res["final_label"] == "조건부(증가)"


# ------------------------------------------------------------------ 전체 흐름 (템플릿 모드)
def test_template_run_end_to_end_and_replay(runs_tmp):
    r = O.run(CARDS["C"]["card"]["question"], mode="template", run_id="e2e")
    files = {p.name for p in r.run_dir.iterdir()}
    assert {"card.json", "prereg.md", "prereg.lock.json", "analysis.json", "skeptic.json", "report.md",
            "manifest.json", "events.jsonl"} <= files
    md = (r.run_dir / "report.md").read_text(encoding="utf-8")
    assert "이 분석이 말할 수 없는 것" in md and "정답 대조" in md
    man = json.loads((r.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["final_verdict"] == "효과 있음(증가)" and man["llm"]["calls"] == 0
    assert man["prereg"]["sha256"] and man["code_sha256"] and man["question_input"]
    out = O.replay(r.run_dir)
    assert out["reproduced"] and out["code_equal"]


def test_replay_tolerates_last_bit_float_noise():
    """같은 입력인데 부동소수 마지막 비트만 다르면 재실행이 통과해야 한다(정규화 지문)."""
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    a, b = '{"effect_pct": 1.7087942595751344}', '{"effect_pct": 1.708794259575135}'
    pa = {"model": "m", "effort": "medium", "system": "s", "messages": [{"role": "user", "content": a}], "tools": [],
          "schema": schema}
    pb = dict(pa, messages=[{"role": "user", "content": b}])
    assert LLM._key(pa) != LLM._key(pb) and LLM._key_norm(pa) == LLM._key_norm(pb)
    pc = dict(pa, messages=[{"role": "user", "content": '{"effect_pct": 1.7087952}'}])
    assert LLM._key_norm(pa) != LLM._key_norm(pc)                  # 실제로 다른 값은 구분한다
    rec = {"role": "parser", "key": "옛 지문", "key_norm": LLM._key_norm(dict(pb, model="claude-sonnet-5-5")),
           "response": {"stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}]}}
    llm = LLM("replay", cache=[rec])
    llm.model = "claude-sonnet-5-5"
    assert llm.json_call("parser", "s", a, schema) == {}
