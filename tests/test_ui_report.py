"""화면·1장 판정서: 그래프 데이터, HTML 판정서, PDF 한 장, 백그라운드 실행(카드 확인·취소), 화면 흐름."""
import math
import time

import pytest

from jverdict.agent import orchestrator as O
from jverdict.agent.library import CARDS
from jverdict.config import ROOT
from jverdict.report import onepage
from jverdict.report.view import chart_data, load_run, pct_ticks
from jverdict.ui.jobs import Job

APP = str(ROOT / "src" / "jverdict" / "ui" / "app.py")


@pytest.fixture(scope="module")
def c_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs")
    old = O.RUNS
    O.RUNS = d
    try:
        r = O.run(CARDS["C"]["card"]["question"], mode="template", run_id="ui-c")
    finally:
        O.RUNS = old
    return r.run_dir


def test_pct_ticks_are_log_positions_with_zero():
    t = pct_ticks(math.log(0.2), math.log(4.0))
    labels = [s for _, s in t]
    assert "0%" in labels and all(lo < hi for (lo, _), (hi, _) in zip(t, t[1:]))
    for x, s in t:
        if s != "0%":
            assert math.isclose((math.exp(x) - 1) * 100, float(s.rstrip("%")), abs_tol=1e-9)


def test_chart_line_equals_estimate(c_run):
    v = load_run(c_run)
    cd = chart_data(v.analysis)
    assert cd["kind"] == "series" and cd["policy"] == "2024-03-04"
    i0 = cd["dates"].index(cd["policy"])
    post = [x for x in cd["raw"][i0:] if x is not None]
    pre = [x for x in cd["raw"][:i0] if x is not None]
    assert sum(post) / len(post) - sum(pre) / len(pre) == pytest.approx(v.analysis["main"]["effect"], abs=1e-9)
    assert cd["effect"] == pytest.approx(v.analysis["main"]["effect"])


def test_onepage_html_content(c_run):
    v = load_run(c_run)
    h = onepage.build_html(v)
    assert "효과 있음(증가)" in h and "<svg" in h and "정답 대조" in h
    for c in ("C0", "C1", "C2a", "C2b", "C3", "C4"):
        assert c in h
    assert v.manifest["prereg"]["sha256"][:12] in h and "sk-ant" not in h and v.analysis_no in h


@pytest.mark.skipif(onepage.find_browser() is None, reason="Chrome·Edge 없음")
def test_onepage_pdf_is_one_page(c_run):
    r = onepage.make(c_run)
    assert r["pdf"].exists() and r["pages"] == 1


def test_job_waits_for_card_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "RUNS", tmp_path)
    job = Job(CARDS["C"]["card"]["question"], "template", run_id="job-ok").start()
    for _ in range(200):
        if job.status == "confirm":
            break
        time.sleep(0.05)
    assert job.status == "confirm" and job.done_stages == ["parse"]
    assert not (tmp_path / "job-ok" / "card.json").exists()          # 사람이 확인하기 전에는 진행하지 않는다
    job.approve(dict(job.card, reference_effect=0.15))
    assert job.wait(180) and job.status == "done"
    assert load_run(job.run_dir).card["reference_effect"] == 0.15


def test_job_cancel_removes_partial_run(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "RUNS", tmp_path)
    job = Job(CARDS["C"]["card"]["question"], "template", run_id="job-no").start()
    for _ in range(200):
        if job.status == "confirm":
            break
        time.sleep(0.05)
    job.cancel()
    assert job.wait(30) and job.status == "cancelled" and not (tmp_path / "job-no").exists()


def test_job_reports_unknown_question_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "RUNS", tmp_path)
    job = Job("버스 요금을 올리면 어떻게 되나?", "template", run_id="job-err").start()
    assert job.wait(30) and job.status == "error" and "등록된 정책 카드" in job.error


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown) + " ".join(t.value for t in at.title)


def test_app_start_and_verdict_pages(c_run, monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr(O, "RUNS", c_run.parent)
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and "어떤 정책" in _text(at)
    at = st_testing.AppTest.from_file(APP, default_timeout=60)
    at.query_params["run"] = "ui-c"
    at.run()
    assert not at.exception and "합성데이터" in _text(at)
    assert "+19.6%" in _text(at) and "정책 이후 +19.6%의 변화가 확인되었습니다" in _text(at)
    t = _text(at)
    assert "Agent가 정한 것" in t and "말해도 되는 것" in t and "말하면 안 되는 것" in t and "분석 번호" in t
    assert "문서번호" not in t and "끝." not in t.split("공문")[0]           # 판정서 머리·꼬리는 공문서 흉내를 내지 않음
    at2 = st_testing.AppTest.from_file(APP, default_timeout=60)
    at2.query_params["run"] = "../../etc"                    # 목록에 없는 값은 무시
    at2.run()
    assert not at2.exception and "어떤 정책" in _text(at2)


def test_app_accuracy_page():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    [b for b in at.button if b.label.startswith("정확도 리포트")][0].click().run()
    assert not at.exception and at.title[0].value == "정확도 리포트"


def test_app_guide_shows_once_and_reopens(c_run, monkeypatch):
    """첫 화면에 처음 들어오면 사용 안내가 한 번 뜨고, 넘기기·건너뛰기·다시 열기가 된다. 주소로 연 판정서·?guide=0에는 안 뜬다."""
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr(O, "RUNS", c_run.parent)
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and "정책이 정말 효과가" in _text(at)
    for _ in range(3):
        [b for b in at.button if b.key == "g_next"][0].click().run()
    assert "판정서 읽는 법" in _text(at) and any(b.key == "g_try" for b in at.button)
    [b for b in at.button if b.key == "g_close"][0].click().run()
    assert not at.session_state["guide_open"]
    at.run()                                                   # 다시 그려도 저절로 다시 뜨지 않음
    assert "정책이 정말 효과가" not in _text(at)
    [b for b in at.button if b.key == "guide_btn"][0].click().run()
    assert "정책이 정말 효과가" in _text(at)
    for q in ({"run": "ui-c"}, {"guide": "0"}):
        a2 = st_testing.AppTest.from_file(APP, default_timeout=60)
        for k, v in q.items():
            a2.query_params[k] = v
        a2.run()
        assert not a2.exception and "정책이 정말 효과가" not in _text(a2)
