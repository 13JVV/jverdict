"""정책 카드 직접 만들기(P0-a): 등록 사례가 아닌 정책을 API 키 없이 판정."""
import json

import pytest

from jverdict.agent import cardbuilder as CB
from jverdict.agent import orchestrator as O
from jverdict.config import RAW_DIR

HAS_CARD = (RAW_DIR / "업종별성별카드이용정보.csv").exists()


def test_options_have_groups_and_safe_start_range():
    o = CB.options("synthetic_bus")
    assert o["groups"] and o["units"] == [] and o["freq"] == "D"
    assert o["period"][0] < o["start_range"][0] < o["start_range"][1] < o["period"][1]


def test_build_card_validates_inputs():
    card, errs = CB.build_card("synthetic_bus", "", "2010-01-01", [])
    assert card is None and len(errs) >= 3
    _, errs = CB.build_card("synthetic_bus", "할인", "2024-03-04", ["청소년"], references=[{"groups": ["청소년"]}])
    assert any("같습니다" in e for e in errs)
    _, errs = CB.build_card("synthetic_bus", "할인", "2024-03-04", ["없는집단"])
    assert any("없는" in e for e in errs)


def test_build_card_auto_and_manual_references():
    card, errs = CB.build_card("synthetic_bus", "청소년 할인", "2024-03-04", ["청소년"])
    assert not errs and card["manual"]["auto"] and card["library_case"] is None
    names = [r["name"] for r in card["manual"]["references"]]
    assert "청소년" not in names and names
    card2, _ = CB.build_card("synthetic_bus", "청소년 할인", "2024-03-04", ["청소년"],
                             references=[{"groups": ["일반"]}, {"groups": ["경로"]}])
    assert not card2["manual"]["auto"] and [r["name"] for r in card2["manual"]["references"]] == ["일반", "경로"]


@pytest.mark.skipif(not HAS_CARD, reason="카드 원본 없음")
def test_monthly_start_is_moved_to_month_start():
    card, errs = CB.build_card("card_sector", "거리두기", "2022-04-18", ["제주도민"], ["기타 주점업"],
                               references=[{"groups": ["제주도민"], "units": ["슈퍼마켓"]}])
    assert not errs and card["start_date"] == "2022-04-01"
    assert any("1일로" in a for a in card["assumptions"])
    assert card["manual"]["references"][0]["name"] == "제주도민·슈퍼마켓"


def test_manual_card_runs_offline_with_agent_chosen_control(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "RUNS", tmp_path)
    card, errs = CB.build_card("synthetic_bus", "청소년 할인", "2024-03-04", ["청소년"])
    assert not errs
    r = O.run(None, card=card, mode="template", run_id="m")
    design = json.loads((tmp_path / "m" / "design.json").read_text(encoding="utf-8"))
    assert "정책 전 움직임이 가장 비슷함" in design["choices"][0]["reason"]
    assert design["reference"][0] in [x["name"] for x in card["manual"]["references"]]
    ev = [json.loads(x) for x in (tmp_path / "m" / "events.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert any(e["msg"] == "툴 pick_controls" and e["stage"] == "design" for e in ev)    # 설계 단계, 잠금 전
    assert r.skeptic["final_label"]
    assert O.replay(tmp_path / "m")["reproduced"]


def test_template_parser_error_points_to_builder():
    from jverdict.agent import roles as R
    with pytest.raises(ValueError, match="직접"):
        R.scout_template(type("Run", (), {"card": {"library_case": None}})())
