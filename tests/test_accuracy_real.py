"""반합성 정확도 시험·반론 단계 성능 시험의 설계가 사전 등록과 같은지, 한 건씩 돌아가는지."""
import hashlib
import json

import numpy as np
import pytest

from jverdict.agent import catalog
from jverdict.config import ROOT
from jverdict.synth import accuracy_plasmode as P
from jverdict.synth import accuracy_skeptic as S


def test_prereg_lock_matches_document():
    lk = json.loads((ROOT / "docs/prereg/plasmode_skeptic_accuracy.lock.json").read_text(encoding="utf-8"))
    doc = (ROOT / lk["file"]).read_bytes()
    assert hashlib.sha256(doc).hexdigest() == lk["sha256"] and lk["history"]      # 사후 추가는 이력에 남김


def test_plasmode_dates_follow_prereg():
    bus, card = P.fake_dates("bus"), P.fake_dates("card")
    assert len(bus) == 81 and bus[0] == "2022-01-03" and bus[-1] == "2025-01-27"
    assert len(card) == 42 and not any("2020-03" <= d[:7] <= "2020-08" or "2021-07" <= d[:7] <= "2021-12" for d in card)
    assert P.EFFECTS == [0.0, 0.10, 0.20, 0.35]


@pytest.mark.skipif(not catalog.available("card_sector"), reason="카드 데이터 없음")
def test_plasmode_one_run_card():
    r = P.run_one(("card", P.fake_dates("card")[0], 0.2))
    assert r["level"] in ("효과 있음", "조건부", "효과 없음", "판정 보류") and r["family"] == "card"


def test_skeptic_benchmark_runs_offline():
    r = S.run_one(("K1", 1))                    # 합성데이터라 자료 없이 돈다
    assert r["new_status"] in ("반론 성립", "반론 기각", "판단 불가", "해당 없음") and "prop_level" in r


def test_proposed_rule_downgrades_partial_when_objection_stands():
    m = type("M", (), {"effect": 0.2, "significant": False})()
    gone = type("A", (), {"effect": 0.01, "significant": False})()
    kept = type("A", (), {"effect": 0.19, "significant": False})()
    assert S.proposed("조건부", "증가", m, gone)[0] == "판정 보류"
    assert S.proposed("조건부", "증가", m, kept)[0] == "조건부"
    m2 = type("M", (), {"effect": 0.2, "significant": True})()
    assert S.proposed("효과 있음", "증가", m2, type("A", (), {"effect": 0.05, "significant": True})())[0] == "조건부"
    assert np.isclose(S.old_status(m2, kept) == "반론 성립", True)       # 옛 규칙: 크기가 유지돼도 유의성이 사라지면 성립
