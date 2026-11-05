"""기획서·결과 설명자료용 그림(SVG)을 실데이터에서 만든다.

    .venv/bin/python tools/make_figures.py      # → docs/proposal/figures/*.svg

- trap_A.svg: 사례 A의 함정. 청소년·일반 승차(7일 평균, 시행 전 평균 = 100)와 단순 비교 대 판정관의 결론.
그림 속 숫자는 모두 기록된 판정서(outputs/runs/case-A-template)와 데이터에서 계산한다.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs" / "proposal" / "figures"

INK, INK2, MUTED, GRID = "#1f1f1e", "#52514e", "#8a8984", "#e4e3dc"
C_TEEN, C_GEN, C_BAD = "#c2410c", "#3b6e8f", "#b42323"


def _series():
    from jverdict.agent import catalog
    _, wide = catalog.series_frame("bus_station", {"청소년": {"groups": ["청소년"], "units": None},
                                                   "일반": {"groups": ["일반"], "units": None}})
    w = wide["2025-05-01":"2025-10-25"].copy()
    tot = w.sum(axis=1)
    w[tot < 0.5 * tot.median()] = np.nan              # 부분 집계일 제외(판정관과 같은 기준의 단순화)
    pre = w[:"2025-07-31"].mean()
    idx = (w / pre * 100).rolling(7, min_periods=4).mean()
    return idx


def trap_a() -> str:
    idx = _series()
    run = ROOT / "outputs" / "runs" / "case-A-template"
    nv = json.loads((run / "naive.json").read_text(encoding="utf-8"))
    W, H = 720, 250
    L, R, T, B = 46, 250, 20, 34
    pw, ph = W - L - R, H - T - B
    t0, t1 = idx.index[0], idx.index[-1]
    span = (t1 - t0).days

    def X(t):
        return L + pw * (t - t0).days / span

    def Y(v):
        return T + ph * (1 - v / 120)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" '
         f'font-family="Apple SD Gothic Neo, Malgun Gothic, Noto Sans KR, sans-serif">']
    for v in (0, 25, 50, 75, 100):
        s.append(f'<line x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L - 6}" y="{Y(v) + 3.5:.1f}" font-size="10" fill="{MUTED}" text-anchor="end">{v}</text>')
    for m in pd.date_range("2025-05-01", "2025-10-01", freq="MS"):
        s.append(f'<text x="{X(m):.1f}" y="{T + ph + 15}" font-size="10" fill="{MUTED}" text-anchor="middle">{m.month}월</text>')
    tp = pd.Timestamp("2025-08-01")
    s.append(f'<line x1="{X(tp):.1f}" x2="{X(tp):.1f}" y1="{T}" y2="{T + ph}" stroke="{INK2}" stroke-width="1" stroke-dasharray="3 3"/>')
    s.append(f'<text x="{X(tp) + 4:.1f}" y="{T + 9}" font-size="10" fill="{INK2}">청소년 무료 시행 8-1</text>')
    for col, color in (("일반", C_GEN), ("청소년", C_TEEN)):
        y = idx[col]
        pts, seg = [], []
        for t, v in y.items():
            if np.isfinite(v):
                seg.append(f"{X(t):.1f},{Y(v):.1f}")
            elif seg:
                pts.append(seg)
                seg = []
        if seg:
            pts.append(seg)
        for sg in pts:
            s.append(f'<polyline points="{" ".join(sg)}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        last = y.dropna()
        s.append(f'<text x="{X(last.index[-1]) + 5:.1f}" y="{Y(last.iloc[-1]) + 4:.1f}" font-size="11" font-weight="700" '
                 f'fill="{color}">{col}</text>')
    s.append(f'<text x="{L}" y="{T + ph + 29}" font-size="9.5" fill="{MUTED}">일 승차 인원, 7일 평균, 시행 전(5~7월) 평균 = 100 · '
             f'자료: 제주데이터허브 「이용자 유형별 버스정류소 이용 인원 현황」</text>')
    bx = L + pw + 44
    s.append(f'<rect x="{bx}" y="{T}" width="{W - bx - 4}" height="78" rx="5" fill="#f6f6f3" stroke="#c3c2b7"/>')
    s.append(f'<text x="{bx + 10}" y="{T + 17}" font-size="10" font-weight="700" fill="{INK2}">단순 전후 비교라면</text>')
    s.append(f'<text x="{bx + 10}" y="{T + 41}" font-size="20" font-weight="800" fill="{INK}">{nv["pct"] * 100:+.0f}%</text>')
    s.append(f'<text x="{bx + 10}" y="{T + 61}" font-size="10.5" fill="{INK}">"무료화했더니 이용 급감,</text>')
    s.append(f'<text x="{bx + 10}" y="{T + 74}" font-size="10.5" fill="{INK}">정책 실패"로 보고</text>')
    y2 = T + 90
    s.append(f'<rect x="{bx}" y="{y2}" width="{W - bx - 4}" height="{T + ph - y2 + 12}" rx="5" fill="#fff" stroke="{C_BAD}" stroke-width="1.5"/>')
    s.append(f'<text x="{bx + 10}" y="{y2 + 17}" font-size="10" font-weight="700" fill="{INK2}">정책 판정관</text>')
    s.append(f'<text x="{bx + 10}" y="{y2 + 38}" font-size="16" font-weight="800" fill="{C_BAD}">판정 보류</text>')
    lines = ["측정 단절 의심(강)", "· 대상만 급감, 일반은 그대로", "· 도 발표와 8.3배 차이", "· 같은 날 무료 확대된", "  어린이도 함께 급감"]
    for i, ln in enumerate(lines):
        ind = 8 if ln.startswith("  ") else 0
        s.append(f'<text x="{bx + 10 + ind}" y="{y2 + 56 + i * 13}" font-size="10" fill="{INK}">{html.escape(ln.strip())}</text>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trap_A.svg").write_text(trap_a(), encoding="utf-8")
    print("docs/proposal/figures/trap_A.svg")


if __name__ == "__main__":
    main()
