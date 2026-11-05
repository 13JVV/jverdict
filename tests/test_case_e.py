"""사례 E(양성 대조) 보강: 사전 등록 잠금, 계절 처리 선택, 사전 기간 이상 시점 설명, 기록 재현(달력·참고 사례 고정)."""
import json

import numpy as np
import pytest

from jverdict.agent import design as DZ
from jverdict.agent import orchestrator as O
from jverdict.agent import prereg as PR
from jverdict.agent import roles as R
from jverdict.agent.library import CARDS, EVENTS, events_between, match_card
from jverdict.causal.power import assess, pre_outliers
from jverdict.config import RAW_DIR

HAS_CARD = (RAW_DIR / "업종별성별카드이용정보.csv").exists()


def test_case_e_is_registered_and_locked():
    assert match_card("거리두기 해제로 주점 소비가 늘었나?") == "E"
    assert match_card(CARDS["E"]["card"]["question"]) == "E"
    PR.verify_existing(CARDS["E"]["card"]["prereg"])          # 잠금 뒤 문서가 바뀌지 않음
    d = CARDS["E"]["design"]
    assert d["season"] == "none" and d["pre"] == 12 and d["post"] == 6
    assert d["exclude"][0]["start"] == "2020-02-01"         # 방역 기간은 가짜 시행일이 될 수 없음


def test_season_option_is_validated_and_documented():
    c = CARDS["C"]
    d = dict(c["design"], season="몰라")
    assert any("season" in e for e in DZ.validate(d, c["card"]))
    md_old = DZ.prereg_markdown(c["card"], c["design"], {}, "x")          # 기존 설계: 문구가 그대로(지문 유지)
    assert "계절 차분 3년" in md_old
    md_none = DZ.prereg_markdown(c["card"], dict(c["design"], season="none"), {}, "x")
    assert "계절 차분 없음" in md_none


def test_pre_outliers_only_when_they_dominate_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, 12)
    t = [f"2021-{m:02d}" for m in range(1, 13)]
    assert pre_outliers(x, t) == []
    y = x.copy()
    y[7] = -4.0                                                   # 영업 중단 달처럼 한 달이 크게 벗어남
    out = pre_outliers(y, t)
    assert [o["time"] for o in out] == ["2021-08"] and out[0]["dev"] < 0
    z = rng.normal(0, 1.0, 400)
    z[5] = 8.0                                                    # 튀지만 전체 변동을 좌우하지 않음
    assert pre_outliers(z, [str(i) for i in range(400)]) == []


def test_power_sentence_explains_outliers_instead_of_period():
    p = assess(1.0, 0.05, 0.10, 6, "개월", outliers=[{"time": "2021-08", "dev": -4.4, "robust_z": 9.7}])
    s = p.sentence_ko()
    assert "2021-08" in s and "사후 기간을 늘리는 것만으로는 해결하기 어렵" in s
    assert "이상 시점(2021-08)" in p.needed_ko()
    plain = assess(1.0, 0.05, 0.10, 6, "개월")
    assert "2021-08" not in plain.sentence_ko() and "사후 기간 약" in plain.sentence_ko()


def test_calendar_as_of_keeps_old_records_reproducible():
    new = [e for e in EVENTS if e.get("added")]
    assert new, "추가된 사건에는 added가 있어야 함"
    names_new = {e["name"] for e in events_between("2022-01-01", "2025-06-30", "card_sector", as_of="2026-11-06")}
    names_old = {e["name"] for e in events_between("2022-01-01", "2025-06-30", "card_sector", as_of="2026-11-03")}
    assert names_old < names_new and not any("거리두기" in n for n in names_old)


def test_parser_reference_list_is_frozen():
    assert R.PARSER_REFERENCE_CASES == ("A", "B", "D", "C")      # 바꾸면 기존 Claude 기록의 첫 호출 지문이 바뀜
    assert "E" in CARDS and "E" not in R.PARSER_REFERENCE_CASES


@pytest.mark.skipif(not HAS_CARD, reason="카드 원본 없음")
def test_case_e_template_run(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "RUNS", tmp_path)
    r = O.run(CARDS["E"]["card"]["question"], mode="template", run_id="e")
    man = json.loads((tmp_path / "e" / "manifest.json").read_text(encoding="utf-8"))
    assert man["final_verdict"] == "판정 보류" and man["calendar_as_of"]
    assert r.analysis.power.pre_outliers[0]["time"] == "2021-08"
    names = [h["hypothesis"] for h in r.skeptic["hypotheses"]]
    assert any("오미크론" in n for n in names)                     # 사건 달력 반론을 검증함
    assert O.replay(tmp_path / "e")["reproduced"]
