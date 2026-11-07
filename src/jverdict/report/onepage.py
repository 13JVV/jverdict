"""1장 판정서: 실행 폴더 → A4 한 장 HTML → PDF(Chrome·Edge 헤드리스 인쇄).

- 인터넷을 쓰지 않는다. 그래프는 SVG로 직접 그려 HTML 안에 넣는다(외부 라이브러리·글꼴 없음).
- 한 장에 맞을 때까지 글자 크기를 줄여 다시 인쇄한다. 그래도 넘치면 쪽수를 알리고 멈추지 않는다.
- 브라우저가 없으면 HTML만 만든다(브라우저에서 열어 인쇄하면 같은 판정서).
"""
from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .view import STATUS_MARK, RunView, chart_data, load_run, pct_ticks

E = html.escape

# 그래프 색(참조 팔레트) · 잉크
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"      # 관측(파랑), 정책 후 평균(주황)


# ============================================================================ SVG 그래프
def svg_chart(cd: dict | None, w: int = 720, h: int = 230) -> str:
    """시계열이면 선 + 정책 후 평균(구간 띠), 사건연구면 점 + 95% 막대."""
    if not cd:
        return ""
    L, R, T, B = 52, 150, 14, 30          # 오른쪽 여백은 직접 라벨 자리
    pw, ph = w - L - R, h - T - B
    n = len(cd["dates"])
    vals = [y for y in cd["y"] if y is not None]
    if cd["kind"] == "series":
        vals += [0.0, cd["effect"]] + [b for b in cd["band"] if b is not None]
    else:
        vals += [0.0] + [x for x in cd["lo"] + cd["hi"] if x is not None]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 0.05
    lo, hi = lo - pad, hi + pad
    ticks = pct_ticks(lo, hi)              # 위치는 로그 차이, 글자는 %

    def X(i: float) -> float:
        return L + (i / max(n - 1, 1)) * pw if cd["kind"] == "series" else L + (i + 0.5) / n * pw

    def Y(v: float) -> float:
        return T + (hi - v) / (hi - lo) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" role="img" '
           f'aria-label="{E(cd["y_title"])}" font-family="inherit" font-size="10.5">']
    for t, lab in ticks:                                # 가로 격자 + y 눈금
        y = Y(t)
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{AXIS if t == 0 else GRID}" '
                   f'stroke-width="{1.2 if t == 0 else 0.6}"/>')
        out.append(f'<text x="{L - 6}" y="{y + 3.5:.1f}" text-anchor="end" fill="{MUTED}">{lab}</text>')
    # x 눈금: 해 또는 달이 바뀌는 곳에서 최대 7개
    key = (lambda s: s[:4]) if (cd["kind"] == "series" and n > 400) else (lambda s: s[:7])
    marks = [i for i in range(n) if i == 0 or key(cd["dates"][i]) != key(cd["dates"][i - 1])]
    stride = max(1, -(-len(marks) // 7))
    for i in marks[::stride]:
        out.append(f'<text x="{X(i):.1f}" y="{T + ph + 16}" text-anchor="middle" fill="{MUTED}">'
                   f'{E(cd["dates"][i][:7])}</text>')
    # 정책 시행선
    if cd["policy"] in cd["dates"]:
        i0 = cd["dates"].index(cd["policy"])
        xp = X(i0) if cd["kind"] == "series" else (X(i0 - 1) + X(i0)) / 2
        out.append(f'<line x1="{xp:.1f}" x2="{xp:.1f}" y1="{T}" y2="{T + ph}" stroke="{INK2}" stroke-width="1" '
                   f'stroke-dasharray="3 3"/>')
        out.append(f'<text x="{xp - 4:.1f}" y="{T + 10}" text-anchor="end" fill="{INK2}">시행 {E(cd["policy"])}</text>')
    else:
        i0, xp = n, L + pw

    if cd["kind"] == "series":
        b0, b1 = cd["band"]
        p0, p1 = cd["band_pct"]
        xe = X(n - 1)
        if b0 is not None and b1 is not None:          # 판정 구간 띠
            out.append(f'<rect x="{xp:.1f}" y="{Y(b1):.1f}" width="{xe - xp:.1f}" height="{max(Y(b0) - Y(b1), 1):.1f}" '
                       f'fill="{S2}" fill-opacity="0.14"/>')
        for seg in _segments(cd["y"]):     # 관측 선(결측에서 끊김)
            pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in seg)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{S1}" stroke-width="1.5" '
                       f'stroke-linejoin="round" stroke-linecap="round"/>')
        out.append(f'<line x1="{xp:.1f}" x2="{xe:.1f}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="{INK2}" '
                   f'stroke-width="1.5" stroke-dasharray="5 3"/>')
        out.append(f'<line x1="{xp:.1f}" x2="{xe:.1f}" y1="{Y(cd["effect"]):.1f}" y2="{Y(cd["effect"]):.1f}" '
                   f'stroke="{S2}" stroke-width="2.5" stroke-linecap="round"/>')
        ly = _apart(Y(cd["effect"]), Y(0), 22, T + 8, T + ph - 4)
        out.append(f'<text x="{xe + 8:.1f}" y="{ly[0] + 4:.1f}" fill="{INK}" font-weight="700">정책 후 평균 '
                   f'{cd["effect_pct"]:+.1f}%</text>')
        if b0 is not None:
            out.append(f'<text x="{xe + 8:.1f}" y="{ly[0] + 16:.1f}" fill="{INK2}" font-size="9.5">판정 구간 '
                       f'{p0:+.1f}~{p1:+.1f}%</text>')
        out.append(f'<text x="{xe + 8:.1f}" y="{ly[1] + 4:.1f}" fill="{INK2}">정책이 없었다면(0%)</text>')
        if i0 > n * 0.2:
            out.append(f'<text x="{L + 4}" y="{T + ph - 6}" fill="{S1}" font-weight="700">'
                       f'관측: {E(cd["target"])} − {E(cd["reference"])}</text>')
    else:
        for i in range(n):                               # 95% 막대 + 점
            x, y = X(i), cd["y"][i]
            if cd["lo"][i] is not None:
                out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(cd["hi"][i]):.1f}" y2="{Y(cd["lo"][i]):.1f}" '
                           f'stroke="{S1 if not cd["pre"][i] else MUTED}" stroke-width="2" stroke-linecap="round"/>')
            if cd["ref"][i]:
                out.append(f'<circle cx="{x:.1f}" cy="{Y(0):.1f}" r="4" fill="#fcfcfb" stroke="{INK2}" stroke-width="1.5"/>')
                out.append(f'<text x="{x:.1f}" y="{Y(0) - 8:.1f}" text-anchor="middle" fill="{INK2}" font-size="9.5">기준월</text>')
            else:
                c = S1 if not cd["pre"][i] else MUTED
                out.append(f'<circle cx="{x:.1f}" cy="{Y(y):.1f}" r="4.5" fill="{c}" stroke="#fcfcfb" stroke-width="2"/>')
        out.append(f'<text x="{L + pw + 8}" y="{T + 14}" fill="{MUTED}" font-weight="700">● 정책 전</text>')
        out.append(f'<text x="{L + pw + 8}" y="{T + 30}" fill="{S1}" font-weight="700">● 정책 후</text>')
        out.append(f'<text x="{L + pw + 8}" y="{T + 46}" fill="{INK2}">{E(cd["target"])}, 기준월 대비</text>')
    out.append(f'<text x="{L}" y="{h - 2}" fill="{MUTED}" font-size="9">{E(cd["y_title"])}</text>')
    out.append("</svg>")
    return "".join(out)


def _segments(ys):
    seg, out = [], []
    for i, v in enumerate(ys):
        if v is None:
            if len(seg) > 1:
                out.append(seg)
            seg = []
        else:
            seg.append((i, v))
    if len(seg) > 1:
        out.append(seg)
    return out


def _apart(a: float, b: float, gap: float, top: float, bottom: float) -> tuple[float, float]:
    """두 라벨의 y가 겹치지 않게 벌린다."""
    if abs(a - b) >= gap:
        return min(max(a, top), bottom), min(max(b, top), bottom)
    mid = (a + b) / 2
    a2, b2 = (mid - gap / 2, mid + gap / 2) if a <= b else (mid + gap / 2, mid - gap / 2)
    shift = max(0, top - min(a2, b2)) - max(0, max(a2, b2) - bottom)
    return a2 + shift, b2 + shift


# ============================================================================ HTML
CSS = """
@page { size: A4; margin: 11mm 12mm 10mm 12mm; }
* { box-sizing: border-box; }
html { font-size: var(--fs, 9pt); }
body { font-family: "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif; color: #0b0b0b;
       background: #fff; margin: 0; line-height: 1.45; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.top { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1.6pt solid #0b3d5c;
       padding-bottom: 2pt; margin-bottom: 5pt; }
.brand { font-weight: 800; color: #1b1f24; font-size: 1.05rem; letter-spacing: .3em; }
.brand small { letter-spacing: 0; font-weight: 600; color: #52514e; font-size: .8rem; margin-left: .4em; }
.sign { color: #52514e; font-size: .82rem; }
.meta { color: #52514e; font-size: .85rem; }
h1 { font-size: 1.55rem; margin: 0 0 1pt; line-height: 1.25; }
.q { color: #52514e; margin: 0 0 5pt; }
.verdict { display: flex; gap: 8pt; align-items: stretch; margin: 0 0 5pt; }
.badge { flex: 0 0 auto; border-radius: 5pt; padding: 5pt 10pt; display: flex; align-items: center; gap: 6pt;
         font-weight: 800; font-size: 1.45rem; white-space: nowrap; }
.badge .ic { display: inline-flex; width: 1.25em; height: 1.25em; border-radius: 50%; align-items: center;
             justify-content: center; font-size: .8em; border: 1.4pt solid currentColor; }
.vtext { flex: 1; border: .6pt solid #c3c2b7; border-radius: 5pt; padding: 4pt 7pt; }
.vtext b { display: block; font-size: 1.02rem; line-height: 1.35; }
.vtext span { color: #52514e; font-size: .88rem; }
.summary { margin: 0 0 4pt; }
dl.pts { display: grid; grid-template-columns: 2.6em 1fr; gap: 1pt 5pt; margin: 3pt 0 0; font-size: .95em; }
dl.pts dt { font-weight: 700; color: #0b3d5c; }
dl.pts dd { margin: 0; }
.naive { display: flex; gap: 6pt; align-items: stretch; margin: 2pt 0 5pt; font-size: .95em; }
.naive > div { flex: 1; border: 1px solid #d6d5cc; border-radius: 4pt; padding: 3pt 6pt; }
.naive .nv { background: #f6f6f3; }
.naive .nv b, .naive .jv b { display: block; font-size: .85em; color: #52514e; }
.naive .arrow { flex: 0 0 auto; border: 0; padding: 0; align-self: center; color: #52514e; }
figure { margin: 0 0 4pt; border: .6pt solid #e1e0d9; border-radius: 4pt; padding: 3pt 4pt 1pt; background: #fcfcfb; }
figcaption { color: #52514e; font-size: .8rem; margin: 0 2pt 2pt; }
h2 { font-size: 1.02rem; color: #0b3d5c; margin: 5pt 0 2pt; border-bottom: .8pt solid #c3c2b7; padding-bottom: 1pt; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 0 10pt; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; }
th, td { border-bottom: .5pt solid #e1e0d9; padding: 1.6pt 3pt; text-align: left; vertical-align: top; }
th { color: #52514e; font-weight: 600; }
td.st, td.nm { white-space: nowrap; font-weight: 700; }
td.nm { font-weight: 600; }
.ok { color: #006300; } .bad { color: #b42323; } .na { color: #898781; } .open { color: #8a5a00; }
ul { margin: 1pt 0 2pt; padding-left: 12pt; } li { margin: 0 0 1pt; }
.truth { border: 1pt dashed #2a78d6; border-radius: 4pt; padding: 3pt 6pt; margin: 4pt 0; font-size: .9rem; }
.foot { margin-top: 5pt; border-top: .6pt solid #c3c2b7; padding-top: 2pt; color: #52514e; font-size: .74rem;
        line-height: 1.4; word-break: break-all; }
code { font-family: "Menlo", "Consolas", monospace; font-size: .95em; }
"""


def _st_class(s: str) -> str:
    if s in ("통과", "반론 기각", "규칙 검증: 기각"):
        return "ok"
    if s in ("실패", "반론 성립", "규칙 검증: 성립"):
        return "bad"
    if s == "미검증":
        return "open"
    return "na"


def _short(x: str, n: int) -> str:
    return x if len(x) <= n else x[: n - 1].rstrip() + "…"


def build_html(v: RunView, font_pt: float = 9.0, compact: bool = False) -> str:
    """compact=True: 1쪽에 안 들어갈 때 반론 문장을 줄이고 목록을 앞쪽 몇 개만 싣는다(전문은 화면·report.md에 있음)."""
    cut = (lambda x: _short(x, 70)) if compact else (lambda x: x)
    st, rep, card, man = v.style, v.report, v.card, v.manifest
    cd = chart_data(v.analysis, v.design)
    def detail(c):   # 결재 붙임에는 계산 경로 원값 대신 일치 여부만(원값은 화면·report.md)
        if c["code"] == "C4" and c["status"] == "통과":
            return "두 계산 경로(파이썬·SQL)의 결과가 일치"
        return c["detail"]
    checks = "".join(
        f'<tr><td class="nm">{E(c["code"])} {E(c["name"])}</td><td class="st {_st_class(c["status"])}">'
        f'{STATUS_MARK.get(c["status"], "")} {E(c["status"])}</td><td>{E(detail(c))}</td></tr>' for c in v.checks)
    objs = "".join(
        f'<tr><td>{E(cut(o["hypothesis"]))}</td><td class="st {_st_class(o["status"])}">{E(o["status"])}</td></tr>'
        for o in v.objections())
    dg = v.skeptic.get("downgraded")
    dg_html = f'<p class="bad"><b>반론으로 판정 하향: {E(dg[0])} → {E(dg[1])}</b></p>' if dg else ""
    power = (v.analysis.get("power") or {})
    from jverdict.causal.power import mde_text
    pw_line = (f"구분 가능한 최소 효과 {mde_text(power['mde_log'])} (의미 있는 최소 효과 기준 "
               f"{power['reference_pct'] * 100:.0f}%)") if power.get("mde_pct") is not None else ""

    def ul(xs):
        if compact and len(xs) > 3:
            xs = [cut(x) for x in xs[:3]] + [f"외 {len(xs) - 3}개(화면의 판정서에 전문)"]
        return "<ul>" + "".join(f"<li>{E(x)}</li>" for x in xs) + "</ul>" if xs else "<p class='na'>없음</p>"

    from jverdict.report.actions import needed_for_request
    needed = needed_for_request(v)
    from jverdict.report.brief import brief
    bf = brief(v)
    pts_html = '<dl class="pts">' + "".join(f"<dt>{E(k)}</dt><dd>{E(t)}</dd>" for k, t in bf["points"]) + "</dl>"
    nv = v.naive
    naive_html = (f'<div class="naive"><div class="nv"><b>단순 전후 비교였다면</b>{E(nv["sentence"])}</div>'
                  f'<div class="arrow">→</div><div class="jv"><b>판정관</b>{E(v.label)}. {E(bf["foot"] or v.contrast() or "")}</div></div>'
                  ) if nv else ""
    tl = v.truth_line()
    pre = man["prereg"]
    ex = pre.get("existing")
    llm = man.get("llm", {})
    mode_ko = {"template": "오프라인(템플릿)", "claude": "Claude", "replay": "재실행"}.get(man["mode"], man["mode"])
    rc = v.replay_check
    rep_txt = ("재실행 일치 확인됨" if rc and rc.get("reproduced") else
               "재실행 불일치" if rc else "재실행 대조 전")
    foot = (f"재현 정보 · 실행 <code>{E(v.run_id)}</code> · 모드 {E(mode_ko)} · 판정 규칙 {E(man['rules_version'])} · "
            f"{E(rep_txt)} · 생성 {E(man['created'][:16].replace('T', ' '))}<br>"
            f"사전 등록 지문 <code>{E(pre['sha256'][:24])}…</code> (잠금 {E(pre['locked_at'][:16].replace('T', ' '))})"
            + (f" · 기존 등록 문서 {E(ex['file'])} 지문 확인" if ex else "")
            + f" · 데이터 <code>{E(man['data'].get('source', ''))}</code>"
            + (f" 지문 <code>{E(man['data']['sha256'][:16])}…</code>" if man['data'].get('sha256') else f" 씨앗 {man.get('seed')}")
            + f" · 코드 지문 <code>{E(man['code_sha256'][:16])}…</code>"
            + (f" · LLM 호출 {llm.get('calls', 0)}회 ${llm.get('cost_usd', 0):.3f}" if llm.get("calls") else "")
            + f"<br>재현 명령: <code>python -m jverdict.agent --replay outputs/runs/{E(v.run_id)}</code>"
            " · 이 판정서는 판정 규칙에 따라 자동 작성된 참고용 분석 자료이며, 숫자는 모두 분석 결과에서 옮겼습니다. "
            "정책 판단의 책임은 작성 부서에 있습니다.")
    return f"""<!doctype html><html lang="ko" style="--fs:{font_pt}pt"><head><meta charset="utf-8">
<title>판정서 {E(v.run_id)}</title><style>{CSS}</style></head>
<body>
<div class="top"><span class="brand">정책효과 판정서<small>참고용 분석 자료 · 정책 판정관 J-Verdict</small></span>
<span class="sign">작성일 20&nbsp;&nbsp;.&nbsp;&nbsp;&nbsp;.&nbsp;&nbsp;&nbsp;. · 작성 부서 ________ · 담당 ________</span></div>
<h1>{E(card.get('policy', ''))}</h1>
<p class="q">질문: {E(card.get('question', ''))} · 시행 {E(card.get('start_date', ''))} · 대상 {E(card.get('treated', ''))} · 데이터 {E(v.dataset_title)}</p>
<div class="verdict">
 <div class="badge" style="background:#fff;color:{st['bg']};border:2px solid {st['bg']}"><span class="ic">{E(st['icon'])}</span>{E(v.label)}</div>
 <div class="vtext"><b>{E(bf['title'])}</b>{pts_html}</div>
</div>
{naive_html}
{f'<figure>{svg_chart(cd)}<figcaption>{E(cd["note"])}</figcaption></figure>' if cd else ''}
{f'<div class="truth"><b>정답 대조(합성데이터, 판정 뒤 공개)</b> · {E(tl)}</div>' if tl else ''}
<div class="cols">
 <div><h2>검증 결과</h2><table><tr><th>항목</th><th>결과</th><th>내용</th></tr>{checks}</table></div>
 <div><h2>반론 검증</h2><table><tr><th>반론</th><th>결과</th></tr>{objs}</table>{dg_html}</div>
</div>
<div class="cols">
 <div><h2>이 분석이 말할 수 없는 것</h2>{ul(rep.get('cannot_say', []))}<h2>한계</h2>{ul(rep.get('limits', []))}</div>
 <div><h2>판정에 필요한 데이터</h2>{ul(needed)}{f'<h2>검정력</h2><p>{E(pw_line)}</p>' if pw_line else ''}</div>
</div>
<div class="foot">{foot}</div>
</body></html>"""


# ============================================================================ PDF
def _windows_browsers() -> list[str]:
    """Windows: Chrome(모든 사용자·이 사용자 설치), Edge(Windows 10·11 기본 탑재). 설치 드라이브가 C:가 아니어도 찾는다."""
    roots = [os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "LOCALAPPDATA")]
    roots += [r"C:\Program Files", r"C:\Program Files (x86)"]
    out = []
    for r in dict.fromkeys(x for x in roots if x):
        out += [os.path.join(r, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(r, "Microsoft", "Edge", "Application", "msedge.exe")]
    return out


def find_browser() -> str | None:
    """판정서 PDF 인쇄에 쓸 Chrome·Edge. JVERDICT_BROWSER 환경변수로 직접 지정할 수도 있다."""
    env = os.environ.get("JVERDICT_BROWSER")
    cands = ([env] if env else []) + [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ] + (_windows_browsers() if os.name == "nt" else [])
    for c in cands:
        if c and os.path.exists(c):
            return c
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge", "microsoft-edge"):
        p = shutil.which(name)
        if p:
            return p
    return None


def count_pages(pdf: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


def print_pdf(html_path: Path, pdf_path: Path, browser: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
               "--disable-extensions", "--disable-background-networking", f"--user-data-dir={Path(td) / 'p'}",
               f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
        # 출력은 쓰지 않는다(Windows에서 브라우저 출력의 글자 코드가 달라도 깨지지 않게 바이트로만 받음)
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0 or not pdf_path.exists():
            raise RuntimeError(f"PDF 인쇄 실패(브라우저 종료 코드 {r.returncode})")


def make(run_dir: Path | str, out_dir: Path | None = None) -> dict:
    """판정서 HTML·PDF를 만든다. 반환: {'html', 'pdf'(없으면 None), 'pages', 'font_pt'}"""
    v = load_run(run_dir)
    out = Path(out_dir) if out_dir else v.run_dir
    out.mkdir(parents=True, exist_ok=True)
    h, p = out / "verdict.html", out / "verdict.pdf"
    browser = find_browser()
    sizes = [(9.0, False), (8.6, False), (8.2, False), (7.8, False), (8.2, True), (7.8, True), (7.4, True)]
    pages = None
    for fs, compact in sizes:
        h.write_text(build_html(v, fs, compact), encoding="utf-8")
        if browser is None:
            return {"html": h, "pdf": None, "pages": None, "font_pt": fs}
        p.unlink(missing_ok=True)
        print_pdf(h, p, browser)
        pages = count_pages(p)
        if pages <= 1:
            break
    return {"html": h, "pdf": p, "pages": pages, "font_pt": fs, "compact": compact}
