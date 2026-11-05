"""정책 판정관 J-Verdict 화면 (Streamlit 한 페이지).

    python -m jverdict.ui          # 브라우저에서 http://localhost:8501

흐름: ① 질문 → ② 정책 카드 확인(사람 1회) → ③ 진행 로그 → ④ 판정서 → ⑤ 재실행 대조 · PDF 내려받기
- 계산은 모두 이 PC 안에서 한다. Claude 모드에서만 집계 통계를 Claude API로 보낸다(원자료 행은 보내지 않음).
- API 키 값은 화면·기록 어디에도 보여 주지 않는다(설정 여부만 표시).
"""
from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

import streamlit as st

try:
    import jverdict  # noqa: F401
except ModuleNotFoundError:                          # 설치 없이 소스 폴더에서 바로 실행한 경우
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from jverdict.agent import cardbuilder as CB
from jverdict.agent import catalog
from jverdict.agent import orchestrator as O
from jverdict.agent.library import CARDS
from jverdict.agent.llm import sanitize
from jverdict.config import get_secret
from jverdict.report import actions as AC
from jverdict.report import onepage
from jverdict.report import trace as TR
from jverdict.report.view import (LEVEL_MEANING, LEVEL_STYLE, STAGE_KO, STATUS_MARK, chart_data, list_runs,
                                  load_run)
from jverdict.config import OUTPUT_DIR
from jverdict.ui.charts import figure, placebo_figure, power_figure
from jverdict.ui.jobs import Job

st.set_page_config(page_title="정책 판정관 J-Verdict", page_icon="⚖️", layout="wide",
                   initial_sidebar_state="expanded")

E = html.escape
ss = st.session_state
ss.setdefault("job", None)
ss.setdefault("view_dir", None)
ss.setdefault("question", "")
ss.setdefault("replay_result", {})
ss.setdefault("page", "main")

EXAMPLES = [("A", "청소년 버스 무료화"), ("B", "1차 재난지원금"), ("D", "중국 단체관광 재개"),
            ("E", "거리두기 전면 해제"), ("C", "(시연) 정답 있는 합성데이터")]

st.markdown("""
<style>
.jv-badge{display:inline-flex;align-items:center;gap:.5rem;border-radius:.5rem;padding:.45rem .9rem;
  font-weight:800;font-size:1.6rem;white-space:nowrap}
.jv-badge .ic{display:inline-flex;width:1.6rem;height:1.6rem;border-radius:50%;border:2px solid currentColor;
  align-items:center;justify-content:center;font-size:1rem}
.jv-head{font-size:1.15rem;font-weight:700;line-height:1.45;margin:.2rem 0}
.jv-mean{color:#52514e}
.jv-steps{display:flex;flex-wrap:wrap;gap:.35rem;margin:.3rem 0 .6rem}
.jv-step{border:1px solid #c3c2b7;border-radius:999px;padding:.15rem .65rem;font-size:.88rem;color:#52514e;background:#fff}
.jv-step.done{border-color:#0ca30c;color:#006300;background:#eef8ee}
.jv-step.now{border-color:#2a78d6;color:#0b3d5c;background:#e8f1fc;font-weight:700}
.jv-step.wait{border-color:#fab219;color:#6b4a00;background:#fff6df;font-weight:700}
.jv-log{font-family:Menlo,Consolas,monospace;font-size:.82rem;line-height:1.5;background:#f6f6f3;border-radius:.4rem;
  padding:.5rem .7rem;max-height:18rem;overflow:auto;white-space:pre-wrap}
.jv-ok{color:#006300;font-weight:700}.jv-bad{color:#b42323;font-weight:700}.jv-open{color:#8a5a00;font-weight:700}
.jv-ok,.jv-bad,.jv-open{white-space:nowrap}
.jv-gap{margin-top:.6rem}
.jv-nv{display:flex;gap:.6rem;align-items:stretch;margin:.8rem 0 .4rem;flex-wrap:wrap}
.jv-nv>.a,.jv-nv>.b{flex:1 1 16rem;border:1px solid #c3c2b7;border-radius:.5rem;padding:.55rem .8rem}
.jv-nv>.a{background:#f6f6f3}.jv-nv>.b{background:#fff}
.jv-nv .h{font-size:.8rem;color:#52514e;font-weight:700}.jv-nv .n{font-size:1.35rem;font-weight:800;margin:.1rem 0}
.jv-nv .s{font-size:.78rem;color:#6b6a65;margin-top:.2rem}.jv-nv>.ar{align-self:center;font-size:1.4rem;color:#52514e}
</style>""", unsafe_allow_html=True)


# ============================================================================ 공통 조각
def badge(level: str, label: str) -> str:
    s = LEVEL_STYLE.get(level, LEVEL_STYLE["판정 없음"])
    return (f'<span class="jv-badge" style="background:{s["bg"]};color:{s["fg"]}">'
            f'<span class="ic">{E(s["icon"])}</span>{E(label)}</span>')


def status_cls(s: str) -> str:
    return ("jv-ok" if s in ("통과", "반론 기각", "규칙 검증: 기각") else
            "jv-bad" if s in ("실패", "반론 성립", "규칙 검증: 성립") else "jv-open" if s == "미검증" else "")


def stepper(done: list[str], now: str | None, waiting: bool = False) -> str:
    out = []
    for s in O.STAGES:
        cls, mark = ("done", "✓") if s in done else (("wait" if waiting else "now"), "…") if s == now else ("", "")
        out.append(f'<span class="jv-step {cls}">{mark} {E(STAGE_KO[s])}</span>')
    return '<div class="jv-steps">' + "".join(out) + "</div>"


def log_html(events: list[dict], last: int | None = None) -> str:
    ev = events[-last:] if last else events
    def msg(m: str) -> str:
        if m.startswith("단계 시작: "):
            return f"단계 시작: {STAGE_KO.get(m[7:], m[7:])}"
        return m
    lines = [f"{e['t']}  [{STAGE_KO.get(e['stage'], e['stage'])}] {msg(e['msg'])}" for e in ev]
    return '<div class="jv-log">' + E("\n".join(lines)) + "</div>"


def go_home():
    ss.job, ss.view_dir, ss.page = None, None, "main"


# ============================================================================ 사이드바
key_ok = bool(get_secret("ANTHROPIC_API_KEY"))
DEMO = not all(catalog.available(k) for k in catalog.DATASETS)      # 허브 자료가 없는 환경(공개 웹 데모 등)
with st.sidebar:
    st.markdown("## ⚖️ 정책 판정관")
    st.caption("도정 정책의 효과를 스스로 설계·검증·판정하는 AI Agent")
    mode_label = st.radio("실행 방식", ["오프라인(템플릿)", "Claude"], index=0,
                          help="오프라인: 인터넷·API 없이 등록된 사례를 판정합니다. "
                               "Claude: 각 단계의 판단(대조 집단 선택, 반론 제기, 서술)을 Claude가 합니다. 실행당 약 $0.1.")
    if mode_label == "Claude" and not key_ok:
        st.warning("Claude API 키가 설정되지 않아 오프라인으로 실행합니다.")
    mode = "claude" if (mode_label == "Claude" and key_ok) else "template"
    st.caption(f"Claude API 키: {'설정됨' if key_ok else '없음'} · 키 값은 화면과 기록에 표시하지 않습니다.")

    st.divider()
    st.markdown("**지난 판정서**")
    runs = [d for d in list_runs(O.RUNS) if not d.name.endswith("-replay")]
    if runs:
        def _lbl(d: Path) -> str:
            try:
                m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                how = "Claude 판단" if m["mode"] in ("claude", "replay") and m.get("llm", {}).get("calls") else "오프라인"
                pol = (m.get("card") or {}).get("policy") or d.name
                return f"{pol} · {how} · {m['final_verdict']}"
            except Exception:  # noqa: BLE001
                return d.name
        pick = st.selectbox("실행 기록", runs, format_func=_lbl, label_visibility="collapsed")
        if st.button("열기", width="stretch", disabled=ss.job is not None and ss.job.alive):
            ss.view_dir, ss.job, ss.page = str(pick), None, "main"
            st.rerun()
    else:
        st.caption("아직 실행 기록이 없습니다.")

    st.divider()
    if st.button("정확도 리포트(합성데이터)", width="stretch", disabled=ss.job is not None and ss.job.alive):
        ss.page, ss.view_dir = "accuracy", None
        st.rerun()
    with st.expander("판정 등급의 뜻"):
        for lv in ("효과 있음", "조건부", "효과 없음", "판정 보류"):
            st.markdown(badge(lv, lv).replace("1.6rem", "1rem"), unsafe_allow_html=True)
            st.caption(LEVEL_MEANING[lv])


# ============================================================================ ① 질문
def render_start():
    st.title("어떤 정책의 효과가 궁금하신가요?")
    st.markdown("질문을 적으면 Agent가 **정책 카드 → 데이터 점검 → 비교 설계(잠금) → 추정 → 반론 검증 → 판정서** 순서로 "
                "스스로 분석합니다. 판정은 네 가지입니다: 효과 있음 · 조건부 · 효과 없음 · **판정 보류**(자료가 부족하면 "
                "억지로 결론 내지 않습니다).")
    t1, t2 = st.tabs(["질문으로 시작", "정책 카드 직접 만들기 (새 정책 · API 키 없이)"])
    with t1:
        render_question()
    with t2:
        render_builder()


def _recorded(k: str) -> Path | None:
    """자료가 없는 환경(웹 데모)에서 예시 사례를 누르면 열어 줄 기록된 판정서. Claude 판단 기록을 먼저."""
    for rid in (f"case-{k}-claude", f"case-{k}-template"):
        if (O.RUNS / rid / "manifest.json").exists():
            return O.RUNS / rid
    return None


def render_question():
    st.markdown("**예시 질문**")
    cols = st.columns(len(EXAMPLES))
    for col, (k, name) in zip(cols, EXAMPLES):
        live = catalog.available(CARDS[k]["design"]["dataset"])
        rec = None if live else _recorded(k)
        if live or rec is not None:
            if col.button(name if live else f"{name} (기록 보기)", width="stretch", key=f"ex_{k}"):
                if live:
                    ss.question = CARDS[k]["card"]["question"]
                    ss["q_input"] = ss.question
                else:
                    ss.view_dir, ss.page = str(rec), "main"
                st.rerun()
    if DEMO:
        st.caption("이 화면에는 제주데이터허브 원자료를 싣지 않았습니다. 실데이터 사례는 이 PC에서 만든 판정서 기록을 엽니다"
                   "(숫자·판정은 기록 그대로). 합성데이터 사례와 '정책 카드 직접 만들기'는 여기서 바로 판정합니다.")
    q = st.text_area("질문", key="q_input", height=90, placeholder="예: 청소년 버스 무료화로 청소년의 버스 이용이 늘었는가?",
                     help="Claude 모드에서는 보도자료 문단을 그대로 붙여 넣어도 정책 카드로 바꿉니다.")
    if mode == "template":
        st.caption("오프라인 모드는 등록된 사례(위 예시)의 질문을 해석합니다. 새 정책은 옆 탭 **정책 카드 직접 만들기**에서 "
                   "API 키 없이 판정할 수 있습니다. 문장으로 된 새 질문은 Claude 모드에서 해석합니다.")
    if st.button("판정 시작", type="primary", disabled=not (q or "").strip()):
        ss.job = Job(q.strip(), mode).start()
        ss.view_dir = None
        st.rerun()


@st.cache_data(show_spinner=False)
def _builder_options(ds: str) -> dict:
    return CB.options(ds)


def render_builder():
    st.caption("등록 사례가 아닌 정책도 판정합니다. **무엇을 · 언제 · 누구에게**만 고르면 대조 집단 선택, 설계 잠금, "
               "반론 검증, 판정서는 Agent가 합니다. 인터넷과 API 키가 없어도 됩니다.")
    titles = {k: v.title for k, v in catalog.DATASETS.items() if catalog.available(k)}
    if DEMO:
        st.info("이 화면(웹 데모)에는 허브 실데이터가 없어 합성데이터로만 직접 판정할 수 있습니다. "
                "실데이터로 판정하려면 설치 묶음(실행.bat)을 쓰세요.")
    ds = st.selectbox("① 데이터", list(titles), format_func=lambda k: titles[k], key="mk_ds")
    o = _builder_options(ds)
    unit = "일" if o["freq"] == "D" else "월"
    st.caption(f"기간 {o['period'][0]} ~ {o['period'][1]} · {unit} 단위 · 값: {o['value_label']}")
    lo, hi = pd.Timestamp(o["start_range"][0]).date(), pd.Timestamp(o["start_range"][1]).date()
    default = max(lo, (pd.Timestamp(hi) - pd.DateOffset(years=1)).date())
    with st.form(f"mk_form_{ds}"):
        c1, c2 = st.columns([2, 1])
        policy = c1.text_input("② 정책 이름", placeholder="예: 청소년 버스 요금 할인")
        start = c2.date_input("③ 시행일", value=default, min_value=lo, max_value=hi,
                              help=f"정책 전 비교 기간과 정책 후 자료가 모두 있어야 해서 {lo} ~ {hi} 사이만 고를 수 있습니다."
                                   + (" 월 단위 자료는 그 달 1일로 맞춥니다." if o["freq"] == "MS" else ""))
        tg = st.multiselect(f"④ 정책 대상 {o['group_label']}", o["groups"])
        tu = st.multiselect(f"정책 대상 {o['unit_label']} (비우면 전체)", o["units"]) if o["units"] else []
        how = st.radio("⑤ 비교 대상", ["Agent가 고르기", "직접 고르기"], horizontal=True,
                       help="Agent가 고르기: 정책 대상이 아닌 집단 가운데 정책 전 움직임이 가장 비슷한 집단을 효과 값을 보지 않고 고릅니다.")
        rc1, rc2 = st.columns(2)
        rg = rc1.multiselect(f"대조 {o['group_label']} (직접 고르기일 때, 첫 번째가 주 대조)", o["groups"])
        ru = rc2.multiselect(f"대조 {o['unit_label']} (비우면 정책 대상과 같게)", o["units"]) if o["units"] else []
        e1, e2 = st.columns(2)
        exp_opts = {"increase": "늘었는가", "decrease": "줄었는가", None: "방향을 정하지 않음"}
        expected = e1.selectbox("⑥ 기대하는 변화", list(exp_opts), format_func=lambda k: exp_opts[k])
        ref = e2.number_input("의미 있다고 볼 최소 효과(%)", min_value=1, max_value=100, value=10)
        ok = st.form_submit_button("이 카드로 판정 시작", type="primary")
    if ok:
        refs = None
        if how == "직접 고르기":
            refs = [{"groups": [g], "units": ru or tu or None} for g in rg] or [{"groups": [], "units": None}]
        card, errs = CB.build_card(ds, policy, str(start), tg, tu, refs, expected, ref / 100)
        if errs:
            for e in errs:
                st.error(e)
            return
        ss.job = Job(None, mode, card=card).start()
        ss.view_dir = None
        st.rerun()


# ============================================================================ ②③ 진행 · 정책 카드 확인
def render_card_form(job: Job):
    c = job.card
    st.subheader("② 정책 카드 확인")
    st.caption("Agent가 질문을 이렇게 이해했습니다. 확인한 카드로 비교 설계를 정하고 잠급니다. 잠근 뒤에는 결과를 보고 설계를 바꿀 수 없습니다.")
    lib = c.get("library_case")
    with st.form("card_form"):
        a, b = st.columns(2)
        with a:
            st.markdown(f"**정책** {E(c.get('policy', ''))}")
            st.markdown(f"**질문** {E(c.get('question', ''))}")
            if lib:
                st.markdown(f"**시행일** {E(c.get('start_date', ''))}")
                st.markdown(f"**대상** {E(c.get('treated', ''))}")
                st.caption(f"등록 사례 {lib}: 시행일·대상은 공식 발표 기준으로 고정됩니다."
                           + (" 사전 등록 문서가 있어 설계도 등록본을 그대로 씁니다." if c.get("prereg") else ""))
                start, treated = c.get("start_date"), c.get("treated")
            else:
                start = st.text_input("시행일 (YYYY-MM-DD)", c.get("start_date", ""))
                treated = st.text_input("대상", c.get("treated", ""))
            st.markdown(f"**결과지표** {E(c.get('outcome', ''))}")
            st.markdown(f"**출처** {E(c.get('source', ''))}")
            if c.get("manual"):
                mm = c["manual"]
                st.markdown(f"**비교 설정** 대상 `{E(mm['target']['name'])}` · 대조 후보 "
                            + ", ".join(f"`{E(r['name'])}`" for r in mm["references"])
                            + (" (Agent가 정책 전 유사도로 고름)" if mm.get("auto") else " (첫 번째가 주 대조)"))
        with b:
            exp_opts = {"increase": "늘었는가", "decrease": "줄었는가", None: "방향을 정하지 않음"}
            cur = c.get("expected") if c.get("expected") in exp_opts else None
            expected = st.selectbox("기대하는 변화 방향", list(exp_opts), index=list(exp_opts).index(cur),
                                    format_func=lambda k: exp_opts[k])
            ref = st.number_input("의미 있다고 볼 최소 효과(%)", min_value=1, max_value=100,
                                  value=int(round(float(c.get("reference_effect") or 0.1) * 100)),
                                  help="이 크기의 효과를 구분할 수 있는지(검정력)를 판정에 함께 씁니다.")
            if c.get("assumptions"):
                st.markdown("**Agent가 세운 가정**")
                for x in c["assumptions"]:
                    st.markdown(("⚠️ " if "확인 필요" in x else "· ") + E(x))
        ok = st.form_submit_button("이 카드로 판정 시작", type="primary")
        no = st.form_submit_button("취소")
    if ok:
        new = dict(c, expected=expected, reference_effect=ref / 100, start_date=start, treated=treated)
        job.approve(new)
        st.rerun()
    if no:
        job.cancel()
        job.wait(10)
        go_home()
        st.rerun()


def render_job():
    job: Job = ss.job
    st.title("판정 진행 중")
    st.caption(f"질문: {job.question or (job.card_in or {}).get('question', '')} · 실행 방식 {'Claude' if job.mode == 'claude' else '오프라인'} · 실행 번호 {job.run_id}")
    if job.status == "confirm":
        st.markdown(stepper(job.done_stages, "confirm", waiting=True), unsafe_allow_html=True)
        render_card_form(job)
        with st.expander("진행 기록"):
            st.markdown(log_html(job.events), unsafe_allow_html=True)
        return
    if job.status == "running":
        live_progress()
        return
    if job.status == "done":
        ss.view_dir, ss.job = str(job.run_dir), None
        st.rerun()
    if job.status == "error":
        st.error(f"실행을 끝내지 못했습니다: {job.error}")
        if "등록된 정책 카드와 맞는 질문이 아님" in (job.error or ""):
            st.info("오프라인 모드는 등록된 사례의 질문만 해석합니다. 새 정책은 첫 화면의 '정책 카드 직접 만들기' 탭에서 "
                    "API 키 없이 판정할 수 있습니다.")
        st.markdown(log_html(job.events), unsafe_allow_html=True)
    if job.status == "cancelled":
        st.info("취소했습니다.")
    st.button("처음으로", on_click=go_home)


@st.fragment(run_every=0.8)
def live_progress():
    job: Job = ss.job
    if job is None:
        return
    st.markdown(stepper(job.done_stages, job.stage), unsafe_allow_html=True)
    st.markdown(log_html(job.events, last=14), unsafe_allow_html=True)
    st.caption(f"경과 {job.elapsed:.0f}초")
    if job.status != "running":
        st.rerun(scope="app")


# ============================================================================ ④⑤ 판정서
def render_verdict(run_dir: str):
    try:
        v = load_run(run_dir)
    except Exception as e:  # noqa: BLE001
        st.error(sanitize(f"판정서를 읽지 못했습니다: {e}"))
        st.button("처음으로", on_click=go_home)
        return
    man, rep, a = v.manifest, v.report, v.analysis
    top_l, top_r = st.columns([6, 1])
    with top_l:
        st.caption(f"판정서 · {v.dataset_title} · 실행 {v.run_id}")
        st.title(v.card.get("policy", ""))
        st.caption(f"질문: {v.card.get('question', '')} · 시행 {v.card.get('start_date', '')} · 대상 {v.card.get('treated', '')}")
    with top_r:
        st.button("새 질문", on_click=go_home, width="stretch")

    c1, c2 = st.columns([1.1, 3])
    with c1:
        st.markdown(badge(v.level, v.label), unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="jv-head">{E(rep.get("headline", ""))}</div>'
                    f'<div class="jv-mean">{E(v.meaning)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="jv-gap">{E(rep.get("summary", ""))}</div>', unsafe_allow_html=True)
    if v.naive:
        nv = v.naive
        st.markdown(
            f'<div class="jv-nv"><div class="a"><div class="h">단순 전후 비교였다면</div>'
            f'<div class="n">{nv["pct"] * 100:+.1f}%</div><div>{E(nv["reading"])}</div>'
            f'<div class="s">{E(nv["target"])} · 시행 후 {nv["n_post"]}{nv["unit"]} 평균 대 직전 {nv["n_pre"]}{nv["unit"]} '
            f'(대조 집단·계절 보정·검증 없음)</div></div>'
            f'<div class="ar">→</div><div class="b"><div class="h">판정관</div>'
            f'<div class="n">{E(v.label)}</div><div>{E(v.contrast() or "")}</div></div></div>', unsafe_allow_html=True)

    m, pw = a.get("main") or {}, a.get("power") or {}
    k = st.columns(4)
    if m:
        lo, hi = a["verdict"]["ci_used"]
        k[0].metric("추정 변화", f"{m['pct'] * 100:+.1f}%")
        k[1].metric("판정 구간", f"{(math.exp(lo) - 1) * 100:+.1f}~{(math.exp(hi) - 1) * 100:+.1f}%",
                    help="판정에 쓴 구간(분석 구간과 위약검정 구간을 합친 것). 이 구간이 0을 지나면 방향을 단정하지 않습니다.")
    if pw.get("mde_pct") is not None:
        k[2].metric("구분 가능한 최소 효과", f"±{pw['mde_pct'] * 100:.1f}%",
                    help=f"이보다 작은 효과는 이 자료로 가려내기 어렵습니다(기준 {pw['reference_pct'] * 100:.0f}%).")
    llm = man.get("llm", {})
    k[3].metric("실행 방식", {"template": "오프라인", "claude": "Claude", "replay": "재실행"}.get(man["mode"], man["mode"]))
    if llm.get("calls"):
        k[3].caption(f"LLM 호출 {llm['calls']}회 · ${llm.get('cost_usd', 0):.3f}")

    cd = chart_data(a, v.design)
    fig = figure(cd)
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption(cd["note"])
    if v.truth_line():
        st.info("정답 대조(합성데이터, 판정 뒤 공개) · " + v.truth_line())

    tabs = st.tabs(["다음 할 일", "검증 결과", "반론 검증", "Agent의 판단", "한계·필요 데이터", "설계 문서(잠금)",
                    "진행 기록", "재현·내려받기"])
    with tabs[0]:
        render_actions(v)
    with tabs[3]:
        render_trace(v)
    tabs = [tabs[1], tabs[2], tabs[4], tabs[5], tabs[6], tabs[7]]
    with tabs[0]:
        st.markdown("| 항목 | 결과 | 내용 |\n|---|---|---|\n" + "\n".join(
            f"| {c['code']} {c['name']} | <span class='{status_cls(c['status'])}'>{STATUS_MARK.get(c['status'], '')} "
            f"{c['status']}</span> | {E(c['detail'])} |" for c in v.checks), unsafe_allow_html=True)
        pf = placebo_figure(a)
        if pf is not None:
            st.markdown("**시점 위약검정**: 정책이 없던 가짜 시행일에도 같은 방법으로 '효과'를 계산해, 실제 효과가 그 사이에서 "
                        "얼마나 튀는지 봅니다. 실제 값(주황)이 회색 점들 바깥에 있을수록 우연·계절로 설명하기 어렵습니다.")
            st.plotly_chart(pf, width="stretch", config={"displaylogo": False})
            pt = a["placebo_time"]
            st.caption(f"순위 p={pt['p_two']:.3f} (가짜 {pt['n']}개 중 실제보다 크게 튄 비율)")
        if a.get("power"):
            st.caption("검정력: 자료가 이 정도 크기의 효과를 구분할 수 있는지 미리 계산한 값입니다.")
    with tabs[1]:
        st.caption("판정 규칙이 늘 검사하는 반론과, Skeptic이 이 사례에 맞게 새로 세운 반론입니다. 성립 여부는 코드가 다시 추정해서 정합니다.")
        st.markdown("| 반론 | 검증 방법·결과 | 판정 | 출처 |\n|---|---|---|---|\n" + "\n".join(
            f"| {E(o['hypothesis'])} | {E(o['how'])} | <span class='{status_cls(o['status'])}'>{E(o['status'])}</span> | "
            f"{E(o['source'])} |" for o in v.objections()), unsafe_allow_html=True)
        if v.skeptic.get("downgraded"):
            d0, d1 = v.skeptic["downgraded"]
            st.warning(f"Skeptic 하향: {d0} → {d1} (판정 규칙이 보지 못한 반론이 성립)")
    with tabs[2]:
        x, y = st.columns(2)
        with x:
            st.markdown("**이 분석이 말할 수 없는 것**")
            for s in rep.get("cannot_say", []):
                st.markdown(f"- {s}")
            st.markdown("**한계**")
            for s in rep.get("limits", []) or ["없음"]:
                st.markdown(f"- {s}")
        with y:
            st.markdown("**판정에 필요한 데이터**")
            for s in rep.get("needed_data", []) or ["없음"]:
                st.markdown(f"- {s}")
    with tabs[3]:
        pre = man["prereg"]
        st.caption(f"효과를 보기 전에 잠근 설계입니다. SHA-256 `{pre['sha256']}` · 잠금 {pre['locked_at']}"
                   + (f" · 기존 등록 문서 `{pre['existing']['file']}` 지문 확인" if pre.get("existing") else ""))
        if pre.get("amendments"):
            st.warning(f"설계 변경 기록 {len(pre['amendments'])}건(실행 오류 복구). 원래 설계와 변경 이유가 문서에 남아 있습니다.")
        st.markdown(v.prereg_md)
    with tabs[4]:
        st.markdown(log_html(v.events), unsafe_allow_html=True)
        if man.get("fallbacks"):
            st.caption("템플릿으로 대신한 단계: " + ", ".join(f"{f['stage']}({sanitize(f['error'])[:60]})"
                                                       for f in man["fallbacks"]))
    with tabs[5]:
        render_repro(v)


def render_actions(v):
    st.markdown("**결재 보고서에 붙일 3줄 요약**")
    st.code("\n".join(AC.memo(v)), language=None, wrap_lines=True)
    rj = AC.rejudge(v)
    st.markdown("**다시 판정할 때**")
    (st.info if rj["kind"] == "date" else st.success if rj["kind"] == "none" else st.warning)(rj["text"])
    letter = AC.data_request(v)
    if letter:
        st.markdown("**자료 요청 공문 초안**")
        st.caption("판정에 필요한 데이터를 데이터 제공 기관에 요청하는 문서 초안입니다. 수신처·기한은 비워 두었습니다.")
        st.code(letter, language=None, wrap_lines=True)
        st.download_button("공문 초안 내려받기(.txt)", letter.encode("utf-8"), f"자료요청_{v.run_id}.txt", "text/plain")


def render_trace(v):
    st.caption("단계마다 누가(Claude·규칙·사용자) 무엇을 왜 정했는지 보여 줍니다. 저장된 기록을 그대로 보여 주므로 "
               "API 키가 없어도 Claude가 한 판단을 볼 수 있습니다. 숫자는 통계 툴이 계산하고, LLM은 판단과 서술만 합니다.")
    use = TR.llm_usage(v)
    if use.get("calls"):
        st.markdown(f"LLM 호출 {sum(use['calls'].values())}회 ("
                    + ", ".join(f"{k} {n}" for k, n in use["calls"].items()) + f") · 비용 ${use['cost_usd']:.3f} · "
                    + ", ".join(use["model"]))
    cmp_ = TR.compare_with_template(v, O.RUNS)
    if cmp_:
        st.markdown(f"**같은 사례의 오프라인(규칙) 실행과 비교** (`{cmp_['template_run']}`)")
        st.markdown("| 항목 | 이 실행(Claude) | 오프라인 |\n|---|---|---|\n"
                    + "\n".join(f"| {E(a)} | {E(b)} | {E(c)} |" for a, b, c in cmp_["rows"]))
        st.caption("판단이 달라도 같은 판정 규칙을 적용합니다." + (" 이 사례는 최종 판정이 같습니다." if cmp_["same"] else ""))
    for stp in TR.steps(v):
        with st.expander(f"{stp['title']} · {stp['who']}", expanded=stp["stage"] in ("design", "skeptic")):
            for x in stp["items"]:
                st.markdown(f"- {E(x)}")
            if stp["tools"]:
                st.markdown("툴 호출: " + "; ".join(stp["tools"]))


def render_repro(v):
    man = v.manifest
    st.markdown("**같은 판정을 다시 만들 수 있는지**")
    st.caption("같은 질문·데이터·코드·LLM 응답 기록으로 처음부터 다시 실행해 숫자·판정·판정서 문장이 같은지 비교합니다. "
               "Claude 모드 실행도 API를 부르지 않고 기록으로 재실행합니다.")
    rc = ss.replay_result.get(v.run_id) or v.replay_check
    is_replay = v.run_id.endswith("-replay") or man["mode"] == "replay"
    have = catalog.available(v.design.get("dataset", ""))
    if not have:
        st.caption("이 환경에는 이 판정에 쓴 원자료가 없어 지금 재실행할 수 없습니다. 아래는 원래 PC에서 재실행한 대조 결과입니다.")
    if st.button("지금 재실행해서 확인", disabled=is_replay or not have):
        with st.spinner("재실행 중…"):
            try:
                rc = O.replay(v.run_dir)
                ss.replay_result[v.run_id] = rc
            except Exception as e:  # noqa: BLE001
                st.error(sanitize(f"재실행 실패: {type(e).__name__}: {e}"))
    if rc:
        names = {"numbers_equal": "핵심 수치", "verdict_equal": "판정", "report_equal": "판정서 문장",
                 "data_equal": "데이터 지문", "code_equal": "코드 지문"}
        def mark(x):
            return "✓ 같음" if x else ("– 비교 전 멈춤" if x is None else "✗ 다름")
        st.markdown(" · ".join(f"{n} {mark(rc.get(k))}" for k, n in names.items()))
        if rc.get("error"):
            st.caption(f"멈춘 이유: {sanitize(rc['error'])}")
        if rc.get("reproduced") and rc.get("code_equal") is False:
            st.caption("코드 지문은 원래 실행 뒤에 코드가 고쳐져서 다릅니다. 숫자·판정·문장이 같으므로 재현으로 봅니다.")
        (st.success if rc.get("reproduced") else st.error)(
            "재현됨: 같은 판정이 다시 나왔습니다." if rc.get("reproduced") else "재현되지 않음: 위에서 다른 항목을 확인하세요.")
    st.markdown("**재현 정보**")
    st.code(f"실행 번호   {v.run_id}\n판정 규칙   {man['rules_version']}\n데이터      {man['data'].get('source')}"
            f"  {man['data'].get('sha256', '(생성기 씨앗 ' + str(man.get('seed')) + ')')}\n"
            f"코드 지문   {man['code_sha256']}\n설계 지문   {man['prereg']['sha256']}\n"
            f"재실행 명령 python -m jverdict.agent --replay outputs/runs/{v.run_id}", language=None)

    with st.expander("manifest 전체 보기(재실행에 쓰는 기록)"):
        st.json(man, expanded=False)
    st.markdown("**내려받기**")
    d = v.run_dir
    pdf, md = d / "verdict.pdf", d / "report.md"
    fresh = pdf.exists() and pdf.stat().st_mtime >= (d / "manifest.json").stat().st_mtime
    b1, b2, b3 = st.columns(3)
    if fresh:
        b1.download_button("1장 판정서 PDF", pdf.read_bytes(), file_name=f"판정서_{v.run_id}.pdf", mime="application/pdf",
                           type="primary", width="stretch")
    elif b1.button("1장 판정서 PDF 만들기", type="primary", width="stretch"):
        with st.spinner("PDF 만드는 중…"):
            try:
                r = onepage.make(d)
                if r["pdf"] is None:
                    st.warning("Chrome·Edge를 찾지 못해 HTML로 만들었습니다. 내려받아 브라우저에서 인쇄하면 같은 판정서입니다.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(sanitize(f"PDF를 만들지 못했습니다: {e}"))
    if (d / "verdict.html").exists():
        b2.download_button("판정서 HTML", (d / "verdict.html").read_bytes(), file_name=f"판정서_{v.run_id}.html",
                           mime="text/html", width="stretch")
    if md.exists():
        b3.download_button("판정서 원문(.md)", md.read_bytes(), file_name=f"판정서_{v.run_id}.md", mime="text/markdown",
                           width="stretch")


# ============================================================================ 정확도 리포트
def render_accuracy():
    st.title("정확도 리포트")
    st.caption("정답을 아는 합성데이터에 판정 규칙을 그대로 적용해, 판정이 얼마나 맞는지 잰 결과입니다.")
    st.button("처음으로", on_click=go_home)
    p = OUTPUT_DIR / "accuracy" / "causal_accuracy.json"
    if not p.exists():
        st.info("아직 정확도 시험 결과가 없습니다: python -m jverdict.synth.accuracy_causal")
        return
    j = json.loads(p.read_text(encoding="utf-8"))
    h = j["headline"]
    st.markdown(f"**인과 판정 정확도** · {h['runs']:,}회 · 판정 규칙 {j['rules_version']} · 생성 {j['generated']}")
    k = st.columns(4)
    k[0].metric("받아들일 만한 판정", f"{h['acceptable_all'] * 100:.1f}%",
                help="정답에 비추어 틀리지 않은 판정(맞는 판정 + 조건부·보류로 물러선 판정)의 비율")
    k[1].metric("효과 없는데 '효과 있음'", f"{h['false_effect_rate_no_effect_scenarios'] * 100:.2f}%")
    k[2].metric("효과 있는데 '효과 없음'", f"{h['false_null_rate_effect_scenarios'] * 100:.2f}%")
    k[3].metric("방향을 반대로", f"{h['wrong_sign_rate'] * 100:.1f}%")
    rows = ["| 시나리오 | 정답 | 받아들일 만함 | 이상적 판정 | 잘못된 '효과 있음' | 판정 분포 |", "|---|---|---|---|---|---|"]
    for key, sc in j["scenarios"].items():
        truth = ("효과 있음(감소)" if "역효과" in sc["title"] else "효과 있음") if "실제" in sc["title"] else "효과 없음"
        dist = ", ".join(f"{k2} {v2}" for k2, v2 in sorted(sc["verdicts"].items(), key=lambda t: -t[1]))
        rows.append(f"| {E(sc['title'])} | {truth} | {sc['acceptable'] * 100:.1f}% | {sc['ideal'] * 100:.1f}% | "
                    f"{sc['false_effect'] * 100:.1f}% | {E(dist)} |")
    st.markdown("\n".join(rows))
    if j.get("power"):
        st.markdown("**검정력 곡선**: 심어 둔 효과가 커질수록 '효과 있음'으로 잡아내는 비율")
        st.plotly_chart(power_figure(j["power"]), width="stretch", config={"displaylogo": False})
    q = OUTPUT_DIR / "accuracy" / "quality_accuracy.json"
    if q.exists():
        qj = json.loads(q.read_text(encoding="utf-8"))
        st.markdown("**데이터 품질 점검(측정 단절·결측) 정확도**")
        st.markdown("| 항목 | 값 |\n|---|---|\n" + "\n".join(f"| {E(k2)} | {v2 * 100:.1f}% |" for k2, v2 in qj["headline"].items()))
    md = OUTPUT_DIR / "accuracy" / "causal_accuracy.md"
    if md.exists():
        with st.expander("상세 보고서(원문)"):
            st.markdown(md.read_text(encoding="utf-8"))


# ============================================================================ 라우팅
_q = st.query_params.get("run")
if _q and ss.job is None and not ss.view_dir:
    _hit = [d for d in list_runs(O.RUNS) if d.name == _q]      # 목록에 있는 실행 번호만(경로 조작 차단)
    if _hit:
        ss.view_dir = str(_hit[0])
if ss.job is not None:
    render_job()
elif ss.page == "accuracy":
    render_accuracy()
elif ss.view_dir:
    render_verdict(ss.view_dir)
else:
    render_start()
