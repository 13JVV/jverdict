"""판정서 한눈 요약: 원인별 제목, 짧은 이유 줄, 조사 처리."""
import pytest

from jverdict.agent import orchestrator as O
from jverdict.agent.library import CARDS
from jverdict.report.brief import _first_sentence, _fix_josa, brief
from jverdict.report.view import load_run


def test_sentence_and_josa_helpers():
    assert _first_sentence("가나다. 라마") == "가나다."
    assert _first_sentence("정책 효과보다 집계 방식") == "정책 효과보다 집계 방식"     # '다'에서 자르지 않음
    assert _fix_josa("'청소년'가 줄었다") == "'청소년'이 줄었다"
    assert _fix_josa("'어린이'가 줄었다") == "'어린이'가 줄었다"


def test_brief_for_effect_case(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "RUNS", tmp_path)
    O.run(CARDS["C"]["card"]["question"], mode="template", run_id="c")
    b = brief(load_run(tmp_path / "c"))
    assert b["cause"] == "effect" and b["title"].startswith("정책 이후 +")
    labels = [k for k, _ in b["points"]]
    assert labels == ["근거", "검증"] and b["tag"] == "" and "효과를 인정" in b["foot"]
    assert all(len(t) < 140 for _, t in b["points"])


@pytest.mark.parametrize("run,cause", [("case-A-template", "quality"), ("case-B-template", "power"),
                                       ("case-D-template", "pretrend"), ("case-E-template", "outlier")])
def test_brief_causes_of_recorded_runs(run, cause):
    d = O.RUNS / run
    if not (d / "manifest.json").exists():
        pytest.skip("기록 없음")
    b = brief(load_run(d))
    assert b["cause"] == cause and "미룹니다" in b["title"]
    labels = [k for k, _ in b["points"]]
    assert labels[0] == "요지" and labels[-1] == "다음" and b["tag"]
    assert "추정" not in labels                       # 보류에서는 점추정을 맨 앞에 두지 않는다


def test_skeptic_downgrade_case_f(tmp_path, monkeypatch):
    """시연 사례 F: 규칙은 '효과 있음'이지만 Skeptic이 같은 시기 행사를 빼고 다시 추정하자 효과가 사라져 '판정 보류'."""
    monkeypatch.setattr(O, "RUNS", tmp_path)
    O.run(CARDS["F"]["card"]["question"], mode="template", run_id="f")
    v = load_run(tmp_path / "f")
    assert v.analysis["verdict"]["level"] == "효과 있음" and v.skeptic["downgraded"] == ["효과 있음", "판정 보류"]
    b = brief(v)
    assert b["cause"] == "skeptic" and any(k == "반론" and "이었으나" in t for k, t in b["points"])
    assert "행사 기간을 빼고" in v.truth_line()
