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
    assert labels[:2] == ["추정", "검증"] and "근거" in labels and "다음" not in labels
    assert all(len(t) < 140 for _, t in b["points"])


@pytest.mark.parametrize("run,cause", [("case-A-template", "quality"), ("case-B-template", "power"),
                                       ("case-D-template", "pretrend"), ("case-E-template", "outlier")])
def test_brief_causes_of_recorded_runs(run, cause):
    d = O.RUNS / run
    if not (d / "manifest.json").exists():
        pytest.skip("기록 없음")
    b = brief(load_run(d))
    assert b["cause"] == cause and "미룹니다" in b["title"]
    assert [k for k, _ in b["points"]][-1] == "다음"
