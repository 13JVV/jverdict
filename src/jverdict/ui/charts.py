"""화면용 plotly 그래프. 수치는 PDF와 같은 report.view.chart_data에서 온다(로그 위치, % 눈금)."""
from __future__ import annotations

import math

import plotly.graph_objects as go

from jverdict.report.view import pct_ticks

S1, S2, INK2, MUTED, GRID, AXIS = "#2a78d6", "#eb6834", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _pct_txt(x):
    return None if x is None else f"{(math.exp(x) - 1) * 100:+.1f}%"


def figure(cd: dict | None) -> go.Figure | None:
    if not cd:
        return None
    fig = go.Figure()
    d = cd["dates"]
    if cd["kind"] == "series":
        i0 = d.index(cd["policy"]) if cd["policy"] in d else len(d)
        b0, b1 = cd["band"]
        if b0 is not None and b1 is not None and i0 < len(d):
            fig.add_trace(go.Scatter(x=[d[i0], d[-1], d[-1], d[i0], d[i0]], y=[b0, b0, b1, b1, b0], fill="toself", mode="lines",
                                     fillcolor="rgba(235,104,52,0.14)", line=dict(width=0), hoverinfo="skip",
                                     name="판정 구간", showlegend=True))
        fig.add_trace(go.Scatter(x=d, y=cd["y"], mode="lines", line=dict(color=S1, width=1.6),
                                 name=f"관측: {cd['target']} − {cd['reference']}" + (" (7일 평균)" if cd["daily"] else ""),
                                 customdata=[_pct_txt(v) for v in cd["y"]],
                                 hovertemplate="%{x}<br>차이 변화 %{customdata}<extra></extra>"))
        if i0 < len(d):
            fig.add_trace(go.Scatter(x=[d[i0], d[-1]], y=[0, 0], mode="lines", name="정책이 없었다면(0%)",
                                     line=dict(color=INK2, width=1.5, dash="dash"), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[d[i0], d[-1]], y=[cd["effect"]] * 2, mode="lines",
                                     name=f"정책 후 평균 {cd['effect_pct']:+.1f}%", line=dict(color=S2, width=3),
                                     hovertemplate=f"정책 후 평균 {cd['effect_pct']:+.1f}%<extra></extra>"))
        vals = [v for v in cd["y"] if v is not None] + [0, cd["effect"]] + [b for b in cd["band"] if b is not None]
    else:
        pre = [i for i in range(len(d)) if cd["pre"][i] and not cd["ref"][i]]
        post = [i for i in range(len(d)) if not cd["pre"][i]]
        for idx, name, color in ((pre, "정책 전", MUTED), (post, "정책 후", S1)):
            fig.add_trace(go.Scatter(
                x=[d[i] for i in idx], y=[cd["y"][i] for i in idx], mode="markers", name=name,
                marker=dict(color=color, size=10, line=dict(color="#fcfcfb", width=2)),
                error_y=dict(type="data", symmetric=False, color=color, thickness=2, width=0,
                             array=[cd["hi"][i] - cd["y"][i] for i in idx],
                             arrayminus=[cd["y"][i] - cd["lo"][i] for i in idx]),
                customdata=[[_pct_txt(cd["y"][i]), _pct_txt(cd["lo"][i]), _pct_txt(cd["hi"][i])] for i in idx],
                hovertemplate="%{x}<br>%{customdata[0]} (95% 구간 %{customdata[1]} ~ %{customdata[2]})<extra></extra>"))
        r = [i for i in range(len(d)) if cd["ref"][i]]
        fig.add_trace(go.Scatter(x=[d[i] for i in r], y=[0] * len(r), mode="markers", name="기준월",
                                 marker=dict(color="#fcfcfb", size=10, line=dict(color=INK2, width=2)),
                                 hovertemplate="%{x} 기준월(0%)<extra></extra>"))
        vals = [v for v in cd["y"] + cd["lo"] + cd["hi"] if v is not None] + [0]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 0.05
    ticks = pct_ticks(lo - pad, hi + pad)
    if cd["policy"]:
        x_pol = cd["policy"] if cd["kind"] == "series" else _between(d, cd["policy"])
        fig.add_vline(x=x_pol, line=dict(color=INK2, width=1, dash="dot"))
        fig.add_annotation(x=x_pol, y=1, yref="paper", text=f"시행 {cd['policy']}", showarrow=False,
                           xanchor="right", yanchor="top", font=dict(color=INK2, size=12))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="x unified" if cd["kind"] == "series"
        else "closest",
        yaxis=dict(tickvals=[t for t, _ in ticks], ticktext=[s for _, s in ticks], gridcolor=GRID,
                   zeroline=True, zerolinecolor=AXIS, zerolinewidth=1.5, range=[lo - pad, hi + pad],
                   title=dict(text=cd["y_title"], font=dict(size=12, color=MUTED))),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", type="category" if cd["kind"] == "event" else "date",
                   tickformat=None if cd["kind"] == "event" else "%Y-%m",
                   categoryorder="array" if cd["kind"] == "event" else None,
                   categoryarray=d if cd["kind"] == "event" else None, hoverformat=None if cd["kind"] == "event"
                   else ("%Y-%m-%d" if cd.get("daily") else "%Y-%m")),
        font=dict(family="Apple SD Gothic Neo, Malgun Gothic, Noto Sans KR, sans-serif"))
    return fig


def _between(dates: list[str], policy: str):
    """범주 축에서 시행월 바로 앞(기준월과 첫 사후월 사이)에 선을 긋기 위한 위치."""
    if policy in dates:
        return dates.index(policy) - 0.5
    return len(dates) - 0.5


def placebo_figure(a: dict) -> go.Figure | None:
    """시점 위약검정: 가짜 시행일들의 '효과'(회색) 사이에서 실제 효과(주황)가 얼마나 튀는지."""
    pt = a.get("placebo_time") or {}
    if not pt.get("available") or not pt.get("effects"):
        return None
    eff, real = [e for e in pt["effects"] if e is not None and math.isfinite(e)], pt["real"]
    dates = pt.get("dates") or [""] * len(eff)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eff, y=[0] * len(eff), mode="markers", name=f"가짜 시행일 {len(eff)}개",
                             marker=dict(color=MUTED, size=11, opacity=0.6, line=dict(color="#fcfcfb", width=1.5)),
                             customdata=[[str(d), _pct_txt(e)] for d, e in zip(dates, eff)],
                             hovertemplate="가짜 시행일 %{customdata[0]}<br>'효과' %{customdata[1]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[real], y=[0], mode="markers", name=f"실제 시행일 {_pct_txt(real)}",
                             marker=dict(color=S2, size=16, symbol="diamond", line=dict(color="#fcfcfb", width=2)),
                             hovertemplate=f"실제 시행일 {_pct_txt(real)}<extra></extra>"))
    vals = eff + [real, 0.0]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.1 or 0.05
    ticks = pct_ticks(lo - pad, hi + pad, n=7)
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.25, x=0),
                      xaxis=dict(tickvals=[t for t, _ in ticks], ticktext=[s for _, s in ticks], gridcolor=GRID,
                                 range=[lo - pad, hi + pad], zeroline=True, zerolinecolor=AXIS),
                      yaxis=dict(visible=False, range=[-1, 1]),
                      font=dict(family="Apple SD Gothic Neo, Malgun Gothic, Noto Sans KR, sans-serif"))
    return fig


def power_figure(rows: list[dict]) -> go.Figure:
    """합성데이터 검정력 곡선: 심어 둔 효과 크기별 '효과 있음' 판정 비율."""
    x = [r["size"] * 100 for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[r["detect"] * 100 for r in rows], mode="lines+markers", name="효과 있음",
                             line=dict(color=S1, width=2), marker=dict(size=8),
                             hovertemplate="심은 효과 %{x:.1f}%<br>효과 있음 %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=[r["detect_any"] * 100 for r in rows], mode="lines+markers",
                             name="효과 있음 + 조건부", line=dict(color=S2, width=2), marker=dict(size=8),
                             hovertemplate="심은 효과 %{x:.1f}%<br>효과 있음·조건부 %{y:.1f}%<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1, x=0),
                      xaxis=dict(title="심어 둔 실제 효과 크기(%)", gridcolor=GRID),
                      yaxis=dict(title="판정 비율(%)", range=[0, 102], gridcolor=GRID),
                      font=dict(family="Apple SD Gothic Neo, Malgun Gothic, Noto Sans KR, sans-serif"))
    return fig
