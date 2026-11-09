"""Agent: 상태머신 가드레일, 효과 가림, 잠금, 숫자 대조, 비밀값 보호, 재현성 (API 호출 없음)."""
import json
import shutil

import numpy as np
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


def test_extra_check_supports_panel_design(tmp_path, monkeypatch):
    """사례 D(패널)에서도 Skeptic의 반론 검증이 오류 없이 다시 추정한다."""
    import pytest as _pt
    from jverdict.agent import catalog as _cat
    from jverdict.agent import orchestrator as _O
    from jverdict.agent import tools as _T
    from jverdict.agent.library import CARDS as _C
    if not _cat.available(_C["D"]["design"]["dataset"]):
        _pt.skip("카드 데이터 없음")
    monkeypatch.setattr(_O, "RUNS", tmp_path)
    r = _O.run(_C["D"]["card"]["question"], mode="template", run_id="d")
    for kind, kw in [("alt_reference", {"reference": ["일본"]}), ("alt_window", {"post": 3}),
                     ("exclude_period", {"start": "2023-08-01", "end": "2023-08-31"}),
                     ("placebo_group", {"placebo_target": "일본"})]:
        args = {"reference": [], "placebo_target": None, "post": None, "start": None, "end": None, "reason": "시험", **kw}
        out = _T.t_extra_check(r, kind, **args)
        assert "error" not in out, (kind, out)
        assert out["status"] in ("반론 기각", "반론 성립", "해당 없음", "판단 불가")


# ------------------------------------------------------------------ 외부 검토 반영(2026-11-09, 규칙 이력 v2.2)
def _fake_run(main_eff, main_sig, case="C"):
    class A:  # noqa: D401
        pass
    r = type("R", (), {})()
    r.extra_checks, r.card = [], {"library_case": case}
    r.analysis = A()
    r.analysis.main = type("M", (), {"effect": main_eff, "significant": main_sig})()
    return r


def _est(eff, sig):
    return type("E", (), {"effect": eff, "pct": float(np.expm1(eff)), "ci": (eff - 1, eff + 1), "p": 0.01 if sig else 0.4,
                          "significant": sig})()


def test_skeptic_power_loss_is_not_objection():
    """크기는 유지됐는데 유의성만 사라지면 '반론 성립'이 아니라 '판단 불가'. 절반 넘게 줄면 '반론 성립'."""
    r = _fake_run(1.0, True)
    assert T._record(r, "alt_window", "x", "", _est(1.03, False))["status"] == "판단 불가"   # 사례 D 크루즈 제외(+181%)와 같은 모양
    assert T._record(r, "alt_window", "x", "", _est(0.9, True))["status"] == "반론 기각"
    assert T._record(r, "exclude_period", "x", "", _est(0.3, True))["status"] == "반론 성립"
    assert T._record(r, "exclude_period", "x", "", _est(-0.1, False))["status"] == "반론 성립"   # 사례 F 행사 제외


def test_estimand_event_is_not_objection():
    """사전 등록에서 추정 대상에 포함한 사건(사례 D 크루즈 재입항)을 빼는 반론은 판정에 쓰지 않는다."""
    r = _fake_run(1.0, True, case="D")
    note = T._estimand_note(r, "exclude_period", "2023-08-31", "2023-09-30")
    assert "추정 대상" in note
    assert T._record(r, "exclude_period", "x", "", _est(0.2, True), note=note)["status"] == "해당 없음"
    assert T._estimand_note(_fake_run(1.0, True, case="B"), "exclude_period", "2023-08-31", "2023-09-30") == ""
    ev = [e for e in events_between("2023-08-01", "2023-10-01", "card_sector") if "크루즈" in e["name"]]
    assert ev and T.t_event_calendar(type("X", (), {"design": {"dataset": "card_sector"}, "scout": None, "as_of": None})(),
                                     "2023-08-01", "2023-10-01")["events"][0].get("in_estimand") is None


def test_c2a_is_reference_when_effect_not_significant():
    """효과가 유의하지 않으면 시점 위약검정은 '실패'가 아니라 '참고'(계절 반론이 맞았다고 읽히지 않게)."""
    from jverdict.causal.estimate import Estimate  # noqa: F401
    from jverdict.causal.verdict import judge
    main = type("M", (), {"ci": (-0.2, 0.15), "p": 0.6, "effect": -0.03, "se": 0.09, "significant": False})()
    pl = type("P", (), {"available": True, "n": 40, "ci": None, "p_two": 0.32, "sd": 0.1})()
    v = judge(main, pretrend_p=0.5, trend_adjusted=None, placebo_time=pl, placebo_space=None, sens=None, double=None,
              power=None)
    c2a = next(c for c in v.checks if c.code == "C2a")
    assert c2a.status == "참고" and "유의하지 않아" in c2a.detail


def test_excluded_estimate_keeps_season_setting():
    """반론 재계산(기간 제외)은 잠긴 설계의 계절 처리 설정을 따른다(사례 E: 계절 차분 없음)."""
    import inspect
    src = inspect.getsource(T._excluded_estimate)
    assert 'season=d.get("season"' in src


@pytest.mark.skipif(not HAS_CARD, reason="카드 데이터 없음")
def test_case_e_exclusion_uses_locked_season(runs_tmp):
    """사례 E의 오미크론 기간 제외 재추정은 +200% 안팎(계절 차분을 잘못 하면 +1496%가 나왔다)."""
    r = O.run(CARDS["E"]["card"]["question"], mode="template", run_id="e")
    xs = [x for x in r.extra_checks if x["kind"] == "exclude_period"]
    assert xs and all(x["pct"] < 5 for x in xs), [x["pct"] for x in xs]
