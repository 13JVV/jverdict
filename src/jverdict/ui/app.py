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
from jverdict.causal import power as PW
from jverdict.config import get_secret
from jverdict.report import actions as AC
from jverdict.report import brief as BR
from jverdict.report import onepage
from jverdict.report import trace as TR
from jverdict.report.view import (LEVEL_MEANING, LEVEL_STYLE, STAGE_KO, STATUS_MARK, chart_data, list_runs,
                                  load_run)
from jverdict.config import OUTPUT_DIR
from jverdict.ui.charts import figure, placebo_figure, power_figure
from jverdict.ui.jobs import Job

st.set_page_config(page_title="정책 판정관 J-Verdict", page_icon="⚖️", layout="wide",
                   initial_sidebar_state="auto")          # 폰에서는 접힌 채 시작(본문을 가리지 않게)


def _no_strike(fn):
    """Streamlit 마크다운은 물결표 하나도 취소선으로 그린다('13~18세 … 2025-08-01~10-25'의 사이가 지워져 보임).
    HTML이 아닌 마크다운 문자열의 '~'를 이스케이프한다."""
    import re as _re

    def wrapped(body="", *a, **k):
        if isinstance(body, str) and "~" in body:
            if not k.get("unsafe_allow_html"):
                body = _re.sub(r"(?<!\\)~", r"\\~", body)
            elif "<style" not in body and "<script" not in body:      # HTML 섞인 마크다운(표 등)은 문자 참조로
                body = body.replace("~", "&#126;")
        return fn(body, *a, **k)
    wrapped._jv_no_strike = True
    return wrapped


for _name in ("markdown", "caption", "info", "warning", "success", "error"):   # 재실행마다 한 번만 감싼다
    if not getattr(getattr(st, _name), "_jv_no_strike", False):
        setattr(st, _name, _no_strike(getattr(st, _name)))

E = html.escape
ss = st.session_state
ss.setdefault("job", None)
ss.setdefault("view_dir", None)
ss.setdefault("question", "")
ss.setdefault("replay_result", {})
ss.setdefault("page", "main")

REF_HELP = ("사업계획서의 목표치, 또는 '이 정도는 변해야 성과라 할 수 있다'는 크기를 넣으세요. 모르면 10%. "
            "자료가 이 크기의 변화를 가려낼 수 있어야 '효과 없음'을 말할 수 있고, 결과를 본 뒤에는 바꿀 수 없습니다.")
EXAMPLES = [("A", "청소년 버스 무료화"), ("B", "1차 재난지원금"), ("D", "중국 단체관광 재개"),
            ("E", "거리두기 전면 해제"), ("C", "(시연) 정답 있는 합성데이터"), ("F", "(시연) Skeptic 반론")]

st.markdown("""
<style>
:root{--ink:#1b1f24;--ink2:#4a5059;--muted:#5f6670;--line:#e3e5e8;--soft:#f6f7f9;--navy:#0b3d5c;--navy2:#e9f0f5;
  --ok:#1f7a3a;--warn:#9a5b00;--bad:#a32525;--null:#4a5059;--hold:#4b3f8c;--seal:#b3261e;
  --serif:"Noto Serif KR","Nanum Myeongjo","AppleMyungjo","Batang","바탕",serif}
html,body,[data-testid="stAppViewContainer"],[data-testid="stSidebar"]{
  font-family:"Pretendard","Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",sans-serif;color:var(--ink)}
[data-testid="stMainBlockContainer"]{max-width:1180px;padding-top:5rem}  /* 상단 고정 바(약 3.75rem)에 가리지 않게 */
[data-testid="stSidebar"]{background:#f8f9fb;border-right:1px solid var(--line)}
h1,h2,h3{letter-spacing:-.02em}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--muted)!important}
/* 탭 안 문서(사전 등록 등)의 제목이 페이지 제목보다 커지지 않게 */
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] h1{font-size:1.3rem;font-weight:800;margin:.8rem 0 .4rem;padding:0}
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] h2{font-size:1.06rem;font-weight:800;border-bottom:1px solid var(--line);
  padding:0 0 .25rem;margin:1.2rem 0 .5rem}
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] h3{font-size:1rem}
[data-testid="stMarkdownContainer"] table{width:100%;border-collapse:collapse;font-size:.9rem}
[data-testid="stMarkdownContainer"] th{background:var(--soft);color:var(--ink2);font-weight:700;white-space:nowrap}
[data-testid="stMarkdownContainer"] th,[data-testid="stMarkdownContainer"] td{padding:.45rem .65rem;border-color:var(--line)}
/* 결재 메모·공문: 터미널 글꼴 대신 문서 글꼴 */
.st-key-memo pre,.st-key-memo code,.st-key-letter pre,.st-key-letter code{font-family:"Pretendard","Apple SD Gothic Neo",
  "Malgun Gothic",sans-serif!important;font-size:.93rem!important;line-height:1.75!important;letter-spacing:0;background:#fff!important}
.st-key-memo [data-testid="stCode"],.st-key-letter [data-testid="stCode"]{border:1px solid var(--line);border-radius:6px}
.st-key-letter [data-testid="stCode"]{border-top:3px double var(--ink)}
/* 알림 상자: 밝은 원색 대신 문서 톤 */
[data-testid="stAlertContainer"]{background:#fff!important;border:1px solid var(--line);border-left:3px solid var(--navy);
  border-radius:6px;color:var(--ink)!important}
button[role=tab] p{font-weight:600}
[data-testid="stExpander"] details{border-color:var(--line)!important;border-radius:6px}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:6px}
/* 브랜드 */
.jv-brand{display:flex;gap:.65rem;align-items:center;margin:.2rem 0 .3rem}
.jv-brand .seal{width:38px;height:38px;border:1.5px solid var(--navy);border-radius:50%;display:flex;align-items:center;
  justify-content:center;color:var(--navy);flex:0 0 auto}
.jv-brand .nm{font-weight:800;font-size:1.12rem;line-height:1.15;color:var(--ink)}
.jv-brand .en{font-size:.68rem;letter-spacing:.14em;color:var(--muted);font-weight:700}
.jv-brand-sub{font-size:.8rem;color:var(--ink2);margin:.1rem 0 .4rem}
/* 첫 화면 */
.jv-eyebrow{font-size:.76rem;letter-spacing:.12em;color:var(--navy);font-weight:700;text-transform:uppercase}
.jv-hero h1{font-size:2.05rem;font-weight:800;margin:.25rem 0 .45rem;padding:0;color:var(--ink)}
.jv-hero p{color:var(--ink2);margin:0 0 .9rem;font-size:.98rem;line-height:1.6}
.jv-principles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.6rem;margin:.3rem 0 1.1rem}
.jv-principles>div{border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;background:#fff}
.jv-principles b{display:block;font-size:.88rem;color:var(--ink)}
.jv-principles span{font-size:.78rem;color:var(--muted)}
@media(max-width:720px){.jv-principles{grid-template-columns:1fr}}
/* 판정서 머리 */
.jv-doc{border-top:3px double var(--ink);padding-top:.55rem;margin-bottom:1rem}
.jv-doc-top{display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;font-size:.8rem;color:var(--ink2)}
.jv-doc-top .kind{font-weight:800;letter-spacing:.32em;color:var(--ink)}
.jv-doc h1{font-family:var(--serif);font-size:1.85rem;font-weight:700;margin:.45rem 0 .5rem;padding:0;color:var(--ink);letter-spacing:-.01em}
.jv-case{display:grid;grid-template-columns:5.4rem 1fr;gap:.22rem .8rem;margin:0;font-size:.9rem}
.jv-case dt{font-weight:700;color:var(--ink2);letter-spacing:.12em;white-space:nowrap}
.jv-case dd{margin:0;color:var(--ink)}
.jv-chips{display:flex;flex-wrap:wrap;gap:.35rem}
.jv-chips span{font-size:.8rem;color:var(--ink2);background:var(--soft);border:1px solid var(--line);border-radius:999px;padding:.12rem .6rem}
.jv-ruling{display:grid;grid-template-columns:190px 1fr;border:1px solid var(--line);border-top:3px solid var(--lv);
  border-radius:8px;background:#fff;margin:.2rem 0 .9rem}
.jv-ruling .stamp{padding:1.1rem 1rem;border-right:1px solid var(--line);display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center}
.jv-ruling .stamp .k{font-size:.74rem;letter-spacing:.3em;color:var(--muted);font-weight:700}
.jv-ruling .stamp .v{margin-top:.4rem;border:3px double var(--lv);color:var(--lv);border-radius:4px;padding:.38rem .9rem;
  font-family:var(--serif);font-weight:700;font-size:1.32rem;letter-spacing:.08em;white-space:nowrap}
.jv-ruling.lv-warn .stamp .v{border:2px solid var(--lv)}
.jv-ruling.lv-hold .stamp .v{border:2px dashed var(--lv)}
.jv-ruling .stamp .d{font-size:.8rem;color:var(--lv);margin-top:.4rem;font-weight:700}
.jv-ruling .stamp .mn{font-size:.74rem;color:var(--muted);margin-top:.45rem;line-height:1.45}
.jv-ruling .body{padding:1rem 1.2rem}
.jv-ruling .t{font-size:1.18rem;font-weight:800;line-height:1.4;color:var(--ink)}
.jv-ruling dl{display:grid;grid-template-columns:3.2rem 1fr;gap:.36rem .65rem;margin:.65rem 0 0}
.jv-ruling dt{font-size:.8rem;font-weight:700;color:var(--navy);background:var(--navy2);border-radius:4px;text-align:center;
  padding:.08rem 0;height:fit-content;margin-top:.12rem}
.jv-ruling dd{margin:0;font-size:.94rem;color:var(--ink);line-height:1.55}
.jv-ruling dt.say{background:var(--lv);color:#fff}
.jv-ruling dd.say{font-weight:700}
@media(max-width:720px){.jv-ruling{grid-template-columns:1fr}.jv-ruling .stamp{border-right:0;border-bottom:1px solid var(--line)}}
/* 숫자 띠: 단순 비교 → 판정관 추정 → 최소 효과 → 가짜 시행일 시험, 맨 아래 사례별 대비 문장 */
.jv-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));border:1px solid var(--line);border-radius:8px;
  background:#fff;margin:0 0 1rem;overflow:hidden}
.jv-strip>div{padding:.65rem .95rem;border-left:1px solid var(--line)}
.jv-strip>div:first-child{border-left:0}
.jv-strip .naive{background:var(--soft)}
.jv-strip .l{font-size:.8rem;font-weight:700;color:var(--ink2)}
.jv-strip .v{font-size:1.38rem;font-weight:800;color:var(--ink);margin:.12rem 0;font-variant-numeric:tabular-nums}
.jv-strip .naive .v{color:var(--ink2);font-weight:700}
.jv-strip .ref .v{color:var(--muted)}
.jv-strip .s{font-size:.8rem;color:var(--muted);line-height:1.45}
.jv-strip .foot{grid-column:1/-1;border-left:0;border-top:1px solid var(--line);font-size:.9rem;color:var(--ink);background:#fff}
@media(max-width:720px){.jv-strip>div{border-left:0;border-top:1px solid var(--line)}.jv-strip>div:first-child{border-top:0}}
.jv-v3{font-size:.86rem;color:var(--ink2);margin:-.4rem 0 .9rem;padding:.4rem .7rem;border-left:3px solid var(--line)}
.jv-v3 span{color:var(--muted)}
.jv-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.6rem;margin:0 0 1rem}
.jv-kpis>div{border:1px solid var(--line);border-radius:8px;padding:.55rem .8rem;background:#fff}
.jv-kpis .l{font-size:.8rem;color:var(--ink2);font-weight:700}
.jv-kpis .v{font-size:1.28rem;font-weight:800;color:var(--ink);margin-top:.1rem;font-variant-numeric:tabular-nums}
.jv-kpis .s{font-size:.78rem;color:var(--muted)}
@media(max-width:720px){.jv-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
/* 첫 화면: 네 사례의 반전 */
.jv-cases{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.6rem;margin:.2rem 0 .4rem}
.jv-case-card{border:1px solid var(--line);border-radius:8px;background:#fff;padding:.7rem .85rem;min-height:13.2rem;margin-bottom:.4rem}
.jv-case-card .k{font-size:.72rem;font-weight:700;color:var(--muted);letter-spacing:.06em;margin-right:.35rem}
.jv-case-card .p{font-weight:800;font-size:.95rem;color:var(--ink)}
.jv-case-card .w{font-size:.78rem;color:var(--muted);margin:.05rem 0 .45rem}
.jv-case-card .naive{font-size:.84rem;color:var(--ink2);text-decoration:line-through;text-decoration-color:#b9bec5}
.jv-case-card .why{font-size:.8rem;color:var(--ink2);margin-top:.35rem;line-height:1.45}
@media(max-width:720px){.jv-cases{grid-template-columns:1fr 1fr}}
/* 판정서 꼬리: 직인 */
.jv-tail{display:flex;justify-content:flex-end;align-items:center;gap:.8rem;margin:1.6rem 0 .4rem;color:var(--ink2);font-size:.85rem}
.jv-tail .seal{width:58px;height:58px;border:2px solid var(--seal);border-radius:50%;color:var(--seal);display:flex;
  flex-direction:column;align-items:center;justify-content:center;font-family:var(--serif);font-weight:700;line-height:1.1;
  font-size:.8rem;transform:rotate(-4deg)}
.jv-tail .seal i{font-style:normal;font-size:.55rem;font-family:Menlo,Consolas,monospace;font-weight:400;margin-top:.15rem}
/* 공통 조각 */
.jv-badge{display:inline-flex;align-items:center;gap:.4rem;border:1.5px solid currentColor;border-radius:6px;padding:.15rem .6rem;
  font-weight:800;font-size:.95rem;white-space:nowrap;background:#fff}
.jv-steps{display:flex;flex-wrap:wrap;gap:.35rem;margin:.3rem 0 .6rem}
.jv-step{border:1px solid var(--line);border-radius:999px;padding:.15rem .65rem;font-size:.84rem;color:var(--ink2);background:#fff}
.jv-step.done{border-color:#9cc9a9;color:var(--ok);background:#f1f8f3}
.jv-step.now{border-color:var(--navy);color:var(--navy);background:var(--navy2);font-weight:700}
.jv-step.wait{border-color:#e0b25c;color:var(--warn);background:#fff8ea;font-weight:700}
.jv-log{font-family:Menlo,Consolas,monospace;font-size:.8rem;line-height:1.5;background:var(--soft);border:1px solid var(--line);
  border-radius:6px;padding:.5rem .7rem;max-height:18rem;overflow:auto;white-space:pre-wrap}
.jv-ok{color:var(--ok);font-weight:700}.jv-bad{color:var(--bad);font-weight:700}.jv-open{color:var(--warn);font-weight:700}
.jv-ok,.jv-bad,.jv-open{white-space:nowrap}
.jv-nv{display:none}
[data-testid="stElementContainer"]:has(.jv-viewkey){display:none}
[data-testid="stElementContainer"]:has(iframe[srcdoc*="jv-scroll-top"]),
[data-testid="stElementContainer"]:has(iframe[height="0"]){position:absolute;height:0;overflow:hidden;margin:0;padding:0}
</style>""", unsafe_allow_html=True)


# ============================================================================ 공통 조각
LEVEL_LINE = {"효과 있음": "var(--ok)", "조건부": "var(--warn)", "효과 없음": "var(--null)", "판정 보류": "var(--hold)",
              "판정 없음": "var(--ink)"}
LEVEL_CLS = {"효과 있음": "lv-ok", "조건부": "lv-warn", "효과 없음": "lv-null", "판정 보류": "lv-hold"}


def badge(level: str, label: str) -> str:
    return f'<span class="jv-badge" style="color:{LEVEL_LINE.get(level, "var(--ink)")}">{E(label)}</span>'


def status_cls(s: str) -> str:
    return ("jv-ok" if s in ("통과", "반론 기각", "규칙 검증: 기각") else
            "jv-bad" if s in ("실패", "반론 성립", "규칙 검증: 성립") else "jv-open" if s in ("미검증", "해당 없음") else "")


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
        return m.replace("모드 template", "모드 오프라인").replace("모드 claude", "모드 Claude")
    lines = [f"{e['t']}  [{STAGE_KO.get(e['stage'], e['stage'])}] {msg(e['msg'])}" for e in ev]
    return '<div class="jv-log">' + E("\n".join(lines)) + "</div>"


def go_home():
    ss.job, ss.view_dir, ss.page = None, None, "main"
    st.query_params.clear()          # 주소의 ?run= 이 남아 있으면 같은 판정서가 다시 열린다


# ============================================================================ 사이드바
key_ok = bool(get_secret("ANTHROPIC_API_KEY"))
DEMO = not all(catalog.available(k) for k in catalog.DATASETS)      # 허브 자료가 없는 환경(공개 웹 데모 등)
def _open_picked():
    d = ss.get("pick_run")
    if d:
        ss.view_dir, ss.job, ss.page = str(d), None, "main"
        st.query_params.clear()
    ss["pick_run"] = None             # 같은 판정서를 다시 골라도 열리게


with st.sidebar:
    st.markdown('<div class="jv-brand"><div class="seal"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
                'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18M6 21h12'
                'M4 7h16M7 7l-3 6a3 3 0 0 0 6 0zM17 7l-3 6a3 3 0 0 0 6 0z"/></svg></div><div><div class="nm">정책 판정관</div>'
                '<div class="en">J-VERDICT</div></div></div><div class="jv-brand-sub">공공데이터로 정책 효과를 판정하는 AI Agent</div>',
                unsafe_allow_html=True)
    busy = ss.job is not None and ss.job.alive
    st.button("＋ 새 판정", type="primary", width="stretch", on_click=go_home, disabled=busy)

    st.markdown("**지난 판정서**")
    runs = [d for d in list_runs(O.RUNS) if not d.name.endswith("-replay")]
    if runs:
        def _lbl(d) -> str:
            if d is None:
                return "판정서 고르기…"
            try:
                m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                how = "Claude" if m["mode"] in ("claude", "replay") and m.get("llm", {}).get("calls") else "오프라인"
                pol = (m.get("card") or {}).get("policy") or d.name
                return f"{m['final_verdict']} · {pol} · {how}"
            except Exception:  # noqa: BLE001
                return d.name
        st.selectbox("실행 기록", [None] + runs, format_func=_lbl, label_visibility="collapsed", key="pick_run",
                     on_change=_open_picked, disabled=busy)
    else:
        st.caption("아직 실행 기록이 없습니다.")

    if st.button("정확도 리포트(합성데이터)", width="stretch", disabled=busy):
        ss.page, ss.view_dir = "accuracy", None
        st.rerun()
    with st.expander("판정 등급의 뜻"):
        for lv in ("효과 있음", "조건부", "효과 없음", "판정 보류"):
            st.markdown(badge(lv, lv), unsafe_allow_html=True)
            st.caption(LEVEL_MEANING[lv])
    with st.expander("실행 설정"):
        mode_label = st.radio("실행 방식", ["기본(오프라인)", "Claude 판단"], index=0,
                              help="기본: 인터넷·API 없이 등록된 사례와 '직접 만들기' 카드를 판정합니다. "
                                   "Claude 판단: 질문 해석, 대조 집단 선택, 반론 제기, 서술을 Claude가 합니다. "
                                   "원자료 행이 아니라 집계 통계만 보내며, 실행당 약 $0.1입니다.")
        if mode_label == "Claude 판단" and not key_ok:
            st.warning("Claude API 키가 설정되지 않아 기본(오프라인)으로 실행합니다.")
        st.caption(f"Claude API 키: {'설정됨' if key_ok else '없음'} · 키 값은 화면과 기록에 표시하지 않습니다.")
    mode = "claude" if (mode_label == "Claude 판단" and key_ok) else "template"


# ============================================================================ ① 질문
@st.cache_data(show_spinner=False)
def _case_summary(run_dir: str, mtime: float) -> dict | None:
    """첫 화면 사례 카드: 단순 비교 대 판정관(기록된 판정서에서 그대로)."""
    try:
        v = load_run(run_dir)
        b = BR.brief(v)
        return {"policy": v.card.get("policy", ""), "when": v.card.get("start_date", ""), "level": v.level,
                "label": v.label, "tag": b["tag"], "title": b["title"],
                "naive": (f"{v.naive['pct'] * 100:+.0f}% → '" + {"역효과": "정책 실패"}.get(v.naive.get("naive_level"),
                          v.naive.get("naive_level", "")) + "'") if v.naive else ""}
    except Exception:  # noqa: BLE001
        return None


def render_start():
    st.markdown(
        '<div class="jv-hero"><div class="jv-eyebrow">J-Verdict · 제주 정책 효과 판정</div>'
        '<h1>어떤 정책의 효과가 궁금하신가요?</h1>'
        '<p>Agent가 데이터부터 의심하고, 결과를 보기 전에 비교 방법을 확정한 뒤, 반론까지 검증해 판정서를 냅니다. '
        '근거가 부족하면 결론을 미루고, 무엇이 있으면 판정할 수 있는지 알려 줍니다.</p></div>',
        unsafe_allow_html=True)
    cases = [(k, _recorded(k)) for k in ("A", "B", "D", "E")]
    cases = [(k, d, _case_summary(str(d), (d / "manifest.json").stat().st_mtime)) for k, d in cases if d is not None]
    cases = [c for c in cases if c[2]]
    if cases:
        st.markdown("**실제 제주 정책 4건: 단순 비교였다면 → 정책 판정관**")
        cols = st.columns(len(cases))
        for col, (k, d, c) in zip(cols, cases):
            with col:
                st.markdown(
                    f'<div class="jv-case-card"><div class="p">{E(c["policy"])}</div><div class="w">시행 {E(c["when"])}</div>'
                    f'<div><span class="k">단순 비교</span><span class="naive">{E(c["naive"])}</span></div>'
                    f'<div style="margin-top:.4rem"><span class="k">판정관</span>{badge(c["level"], c["level"])}'
                    + (f' <span style="font-size:.8rem;color:{LEVEL_LINE.get(c["level"])};font-weight:700">{E(c["tag"])}</span>'
                       if c["tag"] else "")
                    + f'</div><div class="why">{E(c["title"])}</div></div>', unsafe_allow_html=True)
                if st.button("▶ 진행 재생 · 판정서", key=f"case_{k}", width="stretch"):
                    ss.view_dir, ss.page = str(d), "replay"
                    st.rerun()
        st.caption("판정 보류는 실패 판정이 아닙니다. 틀린 결론을 막고, 결론에 필요한 자료와 공문 초안을 함께 냅니다.")
    t1, t2 = st.tabs(["질문으로 시작", "직접 만들기(새 정책)"])
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
                    ss.view_dir, ss.page = str(rec), "replay"
                st.rerun()
    if DEMO:
        st.caption("이 화면에는 제주데이터허브 원자료를 싣지 않았습니다. 실데이터 사례는 개발 PC에서 만든 판정서 기록을 엽니다"
                   "(숫자·판정은 기록 그대로). 합성데이터 사례와 '정책 카드 직접 만들기'는 여기서 바로 판정합니다.")
    q = st.text_area("질문", key="q_input", height=90, placeholder="예: 청소년 버스 무료화로 청소년의 버스 이용이 늘었는가?",
                     help="Claude 모드에서는 보도자료 문단을 그대로 붙여 넣어도 정책 카드로 바꿉니다.")
    if mode == "template":
        st.info("기본(오프라인) 모드는 위 예시 질문을 해석합니다. 새 정책은 옆 탭 **직접 만들기**에서 API 키 없이 판정할 수 있고, "
                "문장으로 된 새 질문은 사이드바 '실행 설정'에서 Claude 판단을 켜면 해석합니다.")
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
        tg = st.multiselect(f"④ 정책 대상 {o['group_label']}", o["groups"], placeholder="선택하세요")
        tu = st.multiselect(f"정책 대상 {o['unit_label']} (비우면 전체)", o["units"], placeholder="선택하세요") if o["units"] else []
        how = st.radio("⑤ 비교 대상", ["Agent가 고르기", "직접 고르기"], horizontal=True,
                       help="Agent가 고르기: 정책 대상이 아닌 집단 가운데 정책 전 움직임이 가장 비슷한 집단을 효과 값을 보지 않고 고릅니다.")
        rc1, rc2 = st.columns(2)
        rg = rc1.multiselect(f"대조 {o['group_label']} ('직접 고르기'를 골랐을 때만 씀, 첫 번째가 주 대조)", o["groups"],
                             placeholder="선택하세요")
        ru = rc2.multiselect(f"대조 {o['unit_label']} (비우면 정책 대상과 같게)", o["units"], placeholder="선택하세요") \
            if o["units"] else []
        e1, e2 = st.columns(2)
        exp_opts = {"increase": "늘었는가", "decrease": "줄었는가", None: "방향을 정하지 않음"}
        expected = e1.selectbox("⑥ 기대하는 변화", list(exp_opts), format_func=lambda k: exp_opts[k])
        ref = e2.number_input("의미 있다고 볼 최소 효과(%)", min_value=1, max_value=100, value=10, help=REF_HELP)
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
    st.caption("Agent가 질문을 이렇게 이해했습니다. 확인하면 Agent가 비교 방법을 정해 결과를 보기 전에 확정(잠금)합니다. "
               "결과에 맞춰 분석을 고칠 수 없게 하는 장치라, 감사·재검토 때 그대로 설명할 수 있습니다.")
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
                                  help=REF_HELP)
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
    st.markdown('<div class="jv-doc"><div class="jv-doc-top"><span class="kind">판 정 진 행</span>'
                f'<span>실행 번호 {E(job.run_id)}</span></div><h1>판정 진행 중</h1></div>', unsafe_allow_html=True)
    st.caption(f"질문: {job.question or (job.card_in or {}).get('question', '')} · 실행 방식 {'Claude 판단' if job.mode == 'claude' else '기본(오프라인)'}")
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
            st.info("이 질문은 기본(오프라인) 모드에서 바로 해석할 수 없습니다. 첫 화면의 **직접 만들기** 탭에서 데이터·시행일·대상을 "
                    "고르면 API 키 없이 판정할 수 있고, 사이드바 '실행 설정'에서 Claude 판단을 켜면 문장 그대로 해석합니다.")
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
    llm = man.get("llm", {})
    how = (f"Claude 판단 {llm['calls']}회(${llm.get('cost_usd', 0):.3f})" if llm.get("calls")
           else "오프라인(규칙·템플릿)") + f" · 판정 규칙 {man.get('rules_version', '')}"
    pre = man.get("prereg") or {}
    day = (pre.get("locked_at") or "")[:10]
    case = v.card.get("library_case") or "N"
    doc_no = f"JV-{day.replace('-', '')}-{case}{'C' if llm.get('calls') else 'T'}"
    top_l, top_r = st.columns([5, 1.25])
    with top_l:
        st.markdown(
            f'<div class="jv-doc"><div class="jv-doc-top"><span class="kind">정 책 효 과 판 정 서</span>'
            f'<span>문서번호 {E(doc_no)} · 판정일 {E(day)}</span></div>'
            f'<h1>{E(v.card.get("policy", ""))}</h1><dl class="jv-case">'
            f'<dt>질 문</dt><dd>{E(v.card.get("question", ""))}</dd>'
            f'<dt>사건 정보</dt><dd>시행 {E(v.card.get("start_date", ""))} · 대상 {E(v.card.get("treated", ""))} · '
            f'자료 {E(v.dataset_title)}</dd>'
            f'<dt>판정 방식</dt><dd>{E(how)} · 설계 잠금 {E((pre.get("sha256") or "")[:8])}</dd></dl></div>',
            unsafe_allow_html=True)
    with top_r:
        pdf_button(v, key="pdf_top")
        st.button("새 판정", on_click=go_home, width="stretch", key="new_top")

    b = BR.brief(v)
    lv = LEVEL_LINE.get(v.level, "var(--ink)")
    pts = "".join(f'<dt class="{"say" if k == "요지" else ""}">{E(k)}</dt>'
                  f'<dd class="{"say" if k == "요지" else ""}">{E(t)}</dd>' for k, t in b["points"])
    sub = v.label.replace(v.level, "").strip("() ") if v.label != v.level else ""
    tag = b["tag"] or sub
    mn = {"판정 보류": "근거 부족으로 미룸(실패 아님)", "효과 있음": "검증을 모두 통과",
          "효과 없음": "의미 있는 크기의 변화 없음", "조건부": "방향은 보이나 검증 일부 실패"}.get(v.level, "")
    st.markdown(
        f'<div class="jv-ruling {LEVEL_CLS.get(v.level, "")}" style="--lv:{lv}"><div class="stamp"><div class="k">판 정</div>'
        f'<div class="v">{E(v.level)}</div>' + (f'<div class="d">{E(tag)}</div>' if tag else "")
        + f'<div class="mn">{E(mn)}</div></div>'
        f'<div class="body"><div class="t">{E(b["title"])}</div><dl>{pts}</dl></div></div>',
        unsafe_allow_html=True)

    v3 = a.get("verdict_v3")
    if v3:
        r2 = a["verdict"]["label"]                     # 규칙끼리 비교(Skeptic 반론은 두 규칙에 똑같이 적용)
        dg = v.skeptic.get("downgraded")
        if v3["label"] == r2:
            msg = f'판정 규칙 <b>v3(개정안)</b>로 다시 판정해도 규칙 결과는 v2와 같은 <b>{E(r2)}</b>입니다.'
        else:
            msg = (f'판정 규칙 <b>v3(개정안)</b>의 결과는 <b style="color:{LEVEL_LINE.get(v3["level"])}">{E(v3["label"])}</b>로, '
                   f'v2 규칙 결과 {E(r2)}와 다릅니다({E(v3["reasons"][0] if v3.get("reasons") else "")}).')
        if dg:
            msg += f' Skeptic 반론은 두 규칙에 똑같이 적용되어 최종 판정은 {E(v.label)}입니다.'
        st.markdown(f'<div class="jv-v3">{msg} <span>v3는 외부 검토를 반영한 개정안이고, 공식 판정은 사전에 정한 v2를 따릅니다. '
                    "차이는 '검증·반론' 탭에.</span></div>", unsafe_allow_html=True)

    m, pw, pt = a.get("main") or {}, a.get("power") or {}, a.get("placebo_time") or {}
    held = v.level == "판정 보류"
    cells = []
    if v.naive:
        nv = v.naive
        cells.append(("naive", "단순 전후 비교였다면", f"{nv['pct'] * 100:+.1f}%",
                      f"{nv['target'].replace('_', '·')} · 시행 후 {nv['n_post']}{nv['unit']} 대 직전 {nv['n_pre']}{nv['unit']} · 대조·보정·검증 없음"))
    if m:
        lo, hi = a["verdict"]["ci_used"]
        big = math.expm1(hi) > 3 or math.expm1(hi) - math.expm1(lo) > 2       # +300% 초과나 폭 200%p 초과
        rng = ("범위가 매우 넓음" if big else f"구간 {math.expm1(lo) * 100:+.1f}% ~ {math.expm1(hi) * 100:+.1f}%")
        cells.append(("ref" if held else "", "판정관 추정" + (" (참고치)" if held else ""), f"{m['pct'] * 100:+.1f}%",
                      ("보류라 인용하지 않음 · " if held else "대조 대비 · ") + rng))
    if pw.get("mde_log") is not None:
        big = pw["mde_log"] > 0.7
        cells.append(("", "구분 가능한 최소 변화", PW.mde_short(pw["mde_log"]),
                      ("사실상 구분 불가 · " if big else "") + f"의미 있다고 본 기준 {pw['reference_pct'] * 100:.0f}%"
                      + ("" if pw.get("enough") else " → 자료 부족")))
    if pt.get("n"):
        rank = "통과" if pt["p_two"] < 0.05 else "통과 못 함"
        cells.append(("", "가짜 시행일 시험", rank,
                      f"정책이 없던 날 {pt['n']}개로 같은 계산 · 순위 {BR._p(pt['p_two'])}"))
    if cells:
        html_cells = "".join(f'<div class="{c}"><div class="l">{E(l)}</div><div class="v">{E(val)}</div>'
                             f'<div class="s">{E(sv)}</div></div>' for c, l, val, sv in cells)
        if b["foot"]:
            html_cells += f'<div class="foot">{E(b["foot"])}</div>'
        st.markdown(f'<div class="jv-strip">{html_cells}</div>', unsafe_allow_html=True)

    cd = chart_data(a, v.design)
    fig = figure(cd)
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption(cd["note"])
    if v.truth_line():
        st.info("정답 대조(합성데이터, 판정 뒤 공개) · " + v.truth_line())

    tabs = st.tabs(["다음 할 일", "검증·반론", "Agent의 판단", "설계 문서(잠금)", "재현·내려받기"])
    with tabs[0]:
        render_actions(v)
        x, y = st.columns(2)
        with x:
            st.markdown("**판정에 필요한 데이터**")
            for s_ in AC.needed_for_request(v) or ["없음"]:
                st.markdown(f"- {s_}")
            st.markdown("**이 분석이 말할 수 없는 것**")
            for s_ in rep.get("cannot_say", []):
                st.markdown(f"- {s_}")
        with y:
            st.markdown("**한계**")
            for s_ in rep.get("limits", []) or ["없음"]:
                st.markdown(f"- {s_}")
        with st.expander("판정 문장 전문"):
            st.markdown(f"**{rep.get('headline', '')}**")
            st.markdown(rep.get("summary", ""))
    with tabs[1]:
        st.markdown("**검증 결과**")
        st.markdown("| 항목 | 결과 | 내용 |\n|---|---|---|\n" + "\n".join(
            f"| {c['code']} {c['name']} | <span class='{status_cls(c['status'])}'>{STATUS_MARK.get(c['status'], '')} "
            f"{c['status']}</span> | {E(c['detail'])} |" for c in v.checks), unsafe_allow_html=True)
        pf = placebo_figure(a)
        if pf is not None:
            st.markdown("**가짜 시행일 시험(시점 위약검정)**: 정책이 없던 날을 시행일로 놓고 같은 방법으로 '효과'를 계산해, "
                        "실제 효과가 그 사이에서 얼마나 튀는지 봅니다. 실제 값(주황)이 회색 점들 바깥에 있을수록 우연·계절로 설명하기 어렵습니다.")
            st.plotly_chart(pf, width="stretch", config={"displaylogo": False})
            st.caption(f"순위 {BR._p(pt['p_two'])} (가짜 {pt['n']}개 중 실제보다 크게 튄 비율)")
        st.markdown("**반론 검증**")
        st.caption("판정 규칙이 늘 검사하는 반론과, Skeptic이 이 사례에 맞게 새로 세운 반론입니다. 성립 여부는 코드가 다시 추정해서 정합니다. "
                   "'반론 성립'은 판정에 불리한 반론이 맞았다는 뜻, '반론 기각'은 반론이 틀렸다는 뜻입니다.")
        st.markdown("| 반론 | 검증 방법·결과 | 판정 | 출처 |\n|---|---|---|---|\n" + "\n".join(
            f"| {E(o['hypothesis'])} | {E(o['how'])} | <span class='{status_cls(o['status'])}'>{E(o['status'])}</span> | "
            f"{E(o['source'])} |" for o in v.objections()), unsafe_allow_html=True)
        if v.skeptic.get("downgraded"):
            d0, d1 = v.skeptic["downgraded"]
            st.warning(f"반론으로 판정 하향: {d0} → {d1} (판정 규칙이 보지 못한 반론이 성립)")
        if v3:
            st.markdown("**판정 규칙 v2(공식)와 v3(개정안)**")
            from jverdict.causal.verdict import RULE_V3_CHANGES
            st.caption("v3는 외부 검토(통계·정책평가 관점)에서 나온 지적을 반영한 개정안입니다. 이미 잠근 판정은 바꾸지 않고 "
                       "같은 분석 결과를 두 규칙으로 판정해 나란히 보여 줍니다.")
            c2 = {c["code"]: c for c in v.checks}
            rows = [f"| 규칙 결과 | {E(a['verdict']['label'])} | {E(v3['label'])} |"]
            for c in v3["checks"]:
                o = c2.get(c["code"]) or {}
                if o.get("status") != c["status"]:
                    rows.append(f"| {c['code']} {E(c['name'])} | {E(o.get('status', '-'))} | {E(c['status'])}: {E(c['detail'])} |")
            st.markdown("| 항목 | v2(공식) | v3(개정안) |\n|---|---|---|\n" + "\n".join(rows))
            st.markdown("v3에서 바뀐 규칙: " + " / ".join(f"({i + 1}) {x}" for i, x in enumerate(RULE_V3_CHANGES)))
    with tabs[2]:
        if st.button("▶ 진행 과정 다시 재생", key="replay_btn"):
            ss.page = "replay"
            st.rerun()
        render_trace(v)
        with st.expander("진행 기록(원문 로그)"):
            st.markdown(log_html(v.events), unsafe_allow_html=True)
            if man.get("fallbacks"):
                st.caption("템플릿으로 대신한 단계: " + ", ".join(f"{f['stage']}({sanitize(f['error'])[:60]})"
                                                           for f in man["fallbacks"]))
    with tabs[3]:
        st.caption(f"효과를 보기 전에 잠근 설계입니다. SHA-256 `{pre['sha256']}` · 잠금 {pre['locked_at']}"
                   + (f" · 기존 등록 문서 `{pre['existing']['file']}` 지문 확인" if pre.get("existing") else ""))
        if pre.get("amendments"):
            st.warning(f"설계 변경 기록 {len(pre['amendments'])}건(실행 오류 복구). 원래 설계와 변경 이유가 문서에 남아 있습니다.")
        st.markdown(v.prereg_md)
    with tabs[4]:
        render_repro(v)
    st.markdown(
        f'<div class="jv-tail"><div>이 판정서는 결과를 보기 전 잠근 설계({E((pre.get("sha256") or "")[:8])})에 따라 작성되었습니다. 끝.'
        f'<br><span style="color:var(--muted)">정책 판정관 J-Verdict · 참고용 분석 자료</span></div>'
        f'<div class="seal">판정관<i>{E((pre.get("sha256") or "")[:6])}</i></div></div>', unsafe_allow_html=True)


def pdf_button(v, key: str):
    """1장 판정서 PDF: 있으면 내려받기, 없으면 만들기."""
    d = v.run_dir
    pdf = d / "verdict.pdf"
    fresh = pdf.exists() and pdf.stat().st_mtime >= (d / "manifest.json").stat().st_mtime
    if fresh:
        st.download_button("1장 판정서 PDF", pdf.read_bytes(), file_name=f"판정서_{v.run_id}.pdf", mime="application/pdf",
                           type="primary", width="stretch", key=key)
    elif st.button("1장 판정서 PDF 만들기", type="primary", width="stretch", key=key):
        with st.spinner("PDF 만드는 중…"):
            try:
                r = onepage.make(d)
                if r["pdf"] is None:
                    st.warning("Chrome·Edge를 찾지 못해 HTML로 만들었습니다. 내려받아 브라우저에서 인쇄하면 같은 판정서입니다.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(sanitize(f"PDF를 만들지 못했습니다: {e}"))


def render_actions(v):
    rj = AC.rejudge(v)
    st.markdown("**다시 판정할 때**")
    st.info(rj["text"])
    st.markdown("**결재 보고서에 붙일 요약(개조식)**")
    st.caption("오른쪽 위 복사 버튼으로 복사해 보고서에 붙여 넣으세요. 숫자는 모두 분석 결과에서 옮겼습니다.")
    with st.container(key="memo"):
        st.code("\n".join(AC.memo(v)), language=None, wrap_lines=True)
    letter = AC.data_request(v)
    if letter:
        st.markdown("**자료 요청 공문 초안**")
        st.caption("판정에 필요한 자료를 제공 기관에 요청하는 공문 초안입니다. 관련 문서 번호·회신 기한은 비워 두었습니다.")
        with st.container(key="letter"):
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
    md = d / "report.md"
    b1, b2, b3 = st.columns(3)
    with b1:
        pdf_button(v, key="pdf_repro")
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


def render_replay(run_dir: str):
    """기록된 실행의 진행 과정을 몇 초 동안 다시 보여 준다(웹 데모처럼 원자료가 없어 다시 계산할 수 없는 환경에서도
    Agent가 단계를 밟는 모습을 볼 수 있게). 숫자·판정은 기록 그대로이고, 끝나면 판정서를 연다."""
    import time as _t
    try:
        v = load_run(run_dir)
    except Exception:  # noqa: BLE001
        ss.page = "main"
        st.rerun()
    ev = v.events
    st.markdown('<div class="jv-doc"><div class="jv-doc-top"><span class="kind">진 행 재 생</span>'
                f'<span>실행 번호 {E(v.run_id)} · 기록 그대로</span></div><h1>{E(v.card.get("policy", ""))}</h1></div>',
                unsafe_allow_html=True)
    st.caption(f"질문: {v.card.get('question', '')} · 저장된 진행 기록을 다시 보여 줍니다(다시 계산하지 않음).")
    box_s, box_l = st.empty(), st.empty()
    done, now = [], None
    step = max(1, len(ev) // 60)
    for i, e in enumerate(ev):
        if e["stage"] != now:
            if now is not None and now not in done:
                done.append(now)
            now = e["stage"]
        if i % step == 0 or i == len(ev) - 1:
            box_s.markdown(stepper(done, now), unsafe_allow_html=True)
            box_l.markdown(log_html(ev[: i + 1], last=12), unsafe_allow_html=True)
            _t.sleep(0.09)
    box_s.markdown(stepper(list(O.STAGES), None), unsafe_allow_html=True)
    _t.sleep(0.6)
    ss.page = "main"
    st.rerun()


# ============================================================================ 라우팅
_q = st.query_params.get("run")
if _q and ss.job is None and not ss.view_dir and ss.get("_q_applied") != _q:
    ss["_q_applied"] = _q                                       # 주소로 연 판정서는 한 번만 적용
    _hit = [d for d in list_runs(O.RUNS) if d.name == _q]      # 목록에 있는 실행 번호만(경로 조작 차단)
    if _hit:
        ss.view_dir = str(_hit[0])
_view_key = (ss.page, ss.view_dir, getattr(ss.job, "run_id", None), getattr(ss.job, "status", None) == "confirm")
# 다른 화면으로 넘어가면 맨 위에서 시작한다(Streamlit은 다시 그릴 때 스크롤 위치를 유지해 새 화면 위쪽이 잘려 보임).
# 화면마다 보이지 않는 표식을 그리고, 늘 같은 내용으로 그리는 감시 스크립트가 표식이 바뀔 때 맨 위로 올린다.
st.markdown(f'<div class="jv-viewkey" data-k="{abs(hash(_view_key)) % 10**9}"></div>', unsafe_allow_html=True)
if True:                                             # 매번 같은 내용으로 그려야 조각이 재사용되어 스크립트가 살아 있다
    _js = ("<!-- jv-scroll-top --><script>(function(){const w=window.parent;if(w.__jvWatch)return;w.__jvWatch=1;"
           "let last=null;const d=w.document;const up=()=>{const m=d.querySelector('[data-testid=stMain]');"
           "if(m)m.scrollTo({top:0});w.scrollTo(0,0);};"
           "new w.MutationObserver(()=>{const e=d.querySelector('.jv-viewkey');if(!e)return;const k=e.getAttribute('data-k');"
           "if(last!==null&&k!==last){up();setTimeout(up,200);setTimeout(up,600);}last=k;})"
           ".observe(d.body,{subtree:true,childList:true,attributes:true,attributeFilter:['data-k']});})();</script>")
    if hasattr(st, "iframe"):
        st.iframe(_js, height=1)
    else:                                            # 옛 Streamlit
        import streamlit.components.v1 as _components
        _components.html(_js, height=0)
if ss.job is not None:
    render_job()
elif ss.page == "accuracy":
    render_accuracy()
elif ss.page == "replay" and ss.view_dir:
    render_replay(ss.view_dir)
elif ss.view_dir:
    render_verdict(ss.view_dir)
else:
    render_start()
