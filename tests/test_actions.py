"""판정서 보강(P1): 단순 비교 대비, 다음 할 일(3줄 요약·재판정·공문 초안), Agent의 판단 기록."""
import json

import pytest

from jverdict.agent import orchestrator as O
from jverdict.agent.library import CARDS
from jverdict.report import actions as AC
from jverdict.report import trace as TR
from jverdict.report.naive import _reading, contrast, naive_compare
from jverdict.report.view import load_run


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs")
    old = O.RUNS
    O.RUNS = d
    try:
        O.run(CARDS["C"]["card"]["question"], mode="template", run_id="case-C-template")
    finally:
        O.RUNS = old
    return d


def test_naive_reading_depends_on_expected_direction():
    assert _reading(0.3, "increase")[1] == "효과 있음"
    assert _reading(-0.3, "increase")[1] == "역효과"
    assert _reading(0.3, "decrease")[1] == "역효과"
    assert _reading(0.01, "increase")[1] == "효과 없음"
    assert "같은 방향" in contrast({"naive_level": "효과 있음"}, "효과 있음")
    assert "그대로 믿지 않고" in contrast({"naive_level": "역효과"}, "판정 보류")


def test_naive_is_written_and_differs_from_engine(runs):
    d = runs / "case-C-template"
    nv = json.loads((d / "naive.json").read_text(encoding="utf-8"))
    card = json.loads((d / "card.json").read_text(encoding="utf-8"))
    design = json.loads((d / "design.json").read_text(encoding="utf-8"))
    assert nv == naive_compare(card, design)                         # 기록과 다시 계산한 값이 같음(결정적)
    a = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
    assert abs(nv["pct"] - a["main"]["pct"]) > 0.05                  # 대조·계절 보정이 결과를 바꿈
    assert "naive" not in (d / "facts.txt").read_text(encoding="utf-8")   # LLM 입력에는 넣지 않음


def test_memo_and_rejudge_use_recorded_numbers(runs):
    v = load_run(runs / "case-C-template")
    m = AC.memo(v)
    assert m[0].startswith("□ 「") and "효과 분석 결과: 효과 있음" in m[0] and all(x.startswith("  ○ ") for x in m[1:-1])
    assert f"{v.analysis['main']['pct'] * 100:+.1f}%" in m[1] and "(향후 조치)" in m[-2] and m[-1].startswith("  ※ 출처:")
    assert not any("하세요" in x for x in m) and v.analysis_no in m[-1]          # 결재 문체(명사형), 출처 줄
    assert AC.rejudge(v)["kind"] == "none" and AC.data_request(v) is None


def test_rejudge_paths_for_withheld_verdicts():
    class V:
        pass
    v = V()
    v.level, v.card = "판정 보류", {"end_date": None}
    v.analysis = {"verdict": {"checks": []}, "main": {"window": ["2025-04", "2026-09"]},
                  "power": {"post_needed": 20, "n_post": 6, "unit": "개월"}}
    r = AC.rejudge(v)
    assert r["kind"] == "date" and r["date"] == "2027-11"
    v.analysis["main"]["window"] = ["2021-04", "2022-09"]          # 계산한 날짜가 이미 지났으면 기간이 아니라 자료 문제
    assert AC.rejudge(v)["kind"] == "data" and "이미 지났" in AC.rejudge(v)["text"]
    v.analysis["main"]["window"] = ["2025-04", "2026-09"]
    v.analysis["power"]["pre_outliers"] = [{"time": "2021-08", "dev": -4.4}]
    assert AC.rejudge(v)["kind"] == "data" and "2021-08" in AC.rejudge(v)["text"]
    v.analysis["power"] = {"post_needed": 500, "n_post": 6, "unit": "개월"}
    assert "약 42년 치" in AC.rejudge(v)["text"] and "기다려서는 풀 수 없" in AC.rejudge(v)["text"]
    v.analysis["power"] = {"post_needed": None, "post_reachable": False, "n_post": 4, "unit": "개월"}
    assert AC.rejudge(v)["kind"] == "data" and "정책 전 기간이 짧아" in AC.rejudge(v)["text"]
    v.analysis["power"] = {"post_needed": 500, "n_post": 6, "unit": "개월"}
    v.card = {"end_date": "2020-08-31"}
    v.analysis["power"] = {"post_needed": 10, "n_post": 4, "unit": "개월"}
    assert "끝난 정책" in AC.rejudge(v)["text"]
    v.analysis["verdict"]["checks"] = [{"code": "C0", "status": "실패"}]
    assert "측정 방식" in AC.rejudge(v)["text"]


def test_data_request_letter_lists_needed_data(runs):
    v = load_run(runs / "case-C-template")
    v.report = dict(v.report, needed_data=["교통복지카드 승차 기록"])
    v.skeptic = dict(v.skeptic, final_level="판정 보류")
    assert AC.data_request(v) is None                                    # 합성데이터 시연에는 공문을 만들지 않음
    v.truth, v.design = None, dict(v.design, dataset="bus_station")
    txt = AC.data_request(v)
    assert "1) 교통복지카드 승차 기록" in txt and v.analysis_no in txt and txt.rstrip().endswith("끝.")
    assert txt.startswith("수신  ") and "(경유)" in txt
    for bad in ("J-Verdict", "지문", ".csv", "작성자 기입", v.run_id, "판정 보류"):  # 외부 기관이 받는 문서에 내부 용어·파일 이름을 쓰지 않음
        assert bad not in txt, bad


def test_trace_marks_who_decided(runs):
    v = load_run(runs / "case-C-template")
    st = {s["stage"]: s for s in TR.steps(v)}
    assert st["design"]["who"].startswith("템플릿") and st["execute"]["who"].startswith("코드")
    assert any("효과 가림" in x for x in st["design"]["items"])
    assert TR.compare_with_template(v, runs) is None                 # 템플릿 실행은 비교 대상 없음


def test_trace_of_recorded_claude_run_if_present():
    d = O.RUNS / "case-A-claude"
    if not (d / "manifest.json").exists():
        pytest.skip("Claude 기록 없음")
    v = load_run(d)
    st = {s["stage"]: s for s in TR.steps(v)}
    assert st["design"]["who"] == "Claude" and st["design"]["tools"]
    assert TR.llm_usage(v)["calls"]
    cmp_ = TR.compare_with_template(v, O.RUNS)
    if cmp_:
        assert cmp_["rows"][0][0] == "대조 집단"


def test_power_post_needed_accounts_for_pre_period_variance():
    from jverdict.causal.power import assess, mde_text, ratio_text
    # 사례 B 형태: 사전 3, 사후 4. 사후를 무한히 늘려도 사전 평균의 분산이 남아 도달 불가
    pw = assess(0.0529, None, 0.10, 4, "개월", n_pre=3)
    assert not pw.enough and pw.post_reachable is False and pw.post_needed is None
    assert "아무리 늘려도" in pw.sentence_ko() and "해결 불가" in pw.needed_ko()
    # 사전이 길면 도달 가능하고, 옛 근사(1/n_post)보다 길게 나온다
    pw = assess(0.05, None, 0.10, 82, "일", n_pre=683)
    assert pw.post_reachable and pw.post_needed > int(82 * (pw.mde_log / 0.0953) ** 2)
    assert mde_text(2.89).startswith("약 18.0배 증가 또는 94% 감소") and mde_text(0.1).startswith("±")
    assert ratio_text(-4.44) == "약 1/85 수준" and ratio_text(0.69) == "약 2.0배"


def test_request_drops_period_extension_when_measurement_broke(runs):
    v = load_run(runs / "case-C-template")
    v.report = dict(v.report, needed_data=["교통복지카드 승차 기록", "정책이 계속되는 경우 사후 기간 약 806일의 추가 자료."])
    v.analysis = json.loads(json.dumps(v.analysis))
    v.analysis["verdict"]["checks"] = [dict(c, status="실패") if c["code"] == "C0" else c for c in v.analysis["verdict"]["checks"]]
    assert AC.needed_for_request(v) == ["교통복지카드 승차 기록"]
