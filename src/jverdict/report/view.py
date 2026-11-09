"""실행 폴더(outputs/runs/<id>/)를 화면·PDF가 함께 쓰는 하나의 보기 모델로 읽는다.

그래프 데이터도 여기서 만든다. 화면(plotly)과 PDF(SVG)가 같은 수치를 그리도록 하기 위해서다.
- 시계열(ITS): 대상−대조 격차(계절 차분)를 정책 전 평균이 0%가 되도록 맞춘 뒤 %로 바꾼다.
  정책 후 평균선의 높이가 곧 추정 효과(main.pct)이고, 0% 선이 '정책이 없었다면'(정책 전 수준)이다.
- 패널(사건연구): 월별 계수와 95% 구간을 %로 바꾼다. 기준월은 0%.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# 판정 등급별 색(상태 색, 아이콘·글자와 함께만 쓴다)과 쉬운 설명
LEVEL_STYLE = {
    "효과 있음": {"bg": "#1f7a3a", "fg": "#ffffff", "icon": "✓"},
    "조건부": {"bg": "#9a5b00", "fg": "#ffffff", "icon": "!"},
    "효과 없음": {"bg": "#4a5059", "fg": "#ffffff", "icon": "0"},
    "판정 보류": {"bg": "#4b3f8c", "fg": "#ffffff", "icon": "‖"},      # 보류=미룸(실패가 아님): 빨강 대신 남보라
    "판정 없음": {"bg": "#0b0b0b", "fg": "#ffffff", "icon": "×"},
}
LEVEL_MEANING = {
    "효과 있음": "정책 이후의 변화가 우연·계절·기존 추세·측정 방식 변경으로 설명되지 않는지 확인하는 검증을 모두 통과했습니다.",
    "조건부": "변화의 방향은 보이지만 검증 일부를 통과하지 못했거나 신호가 약합니다. 성과 근거로 단독 인용하지 마세요(합성 시험에서 조건부의 절반만 실제 효과).",
    "효과 없음": "의미 있는 크기의 변화가 있었다면 잡아낼 수 있을 만큼 자료가 충분했지만, 그런 변화가 보이지 않았습니다.",
    "판정 보류": "현재 자료로는 효과가 있다고도 없다고도 말할 수 없어 결론을 미룹니다. 실패 판정이 아니며, 필요한 자료를 확보하면 다시 판정할 수 있습니다.",
    "판정 없음": "분석을 끝내지 못했습니다.",
}
STATUS_MARK = {"통과": "✓", "실패": "✗", "해당 없음": "–"}
STAGE_KO = {"parse": "질문 해석", "confirm": "정책 카드 확인", "scout": "데이터 점검", "design": "비교 방법 정하기(결과 안 봄)",
            "lock": "분석 방법 확정(잠금)", "execute": "추정", "skeptic": "반론 검증", "report": "판정서 작성",
            "manifest": "재실행 기록"}


def _read(p: Path, default=None):
    if not p.exists():
        return default
    txt = p.read_text(encoding="utf-8")
    return json.loads(txt) if p.suffix == ".json" else txt


@dataclass
class RunView:
    run_dir: Path
    manifest: dict
    card: dict
    design: dict
    analysis: dict
    skeptic: dict
    report: dict
    prereg_md: str = ""
    events: list = field(default_factory=list)
    replay_check: dict | None = None
    dataset_title: str = ""
    truth: dict | None = None
    naive: dict | None = None

    def contrast(self) -> str | None:
        from jverdict.report.naive import contrast
        return contrast(self.naive, self.level)

    @property
    def run_id(self) -> str:
        """실행 폴더 이름(outputs/runs/<이것>). 화면 주소·재실행 명령·내려받기 파일 이름에 쓴다.
        폴더 이름을 바꾼 기록(예: 이름을 바꾼 대체 시연 기록)은 manifest의 run_id와 다를 수 있어 폴더 이름을 기준으로 한다."""
        return self.run_dir.name

    @property
    def level(self) -> str:
        return self.skeptic.get("final_level") or self.analysis["verdict"]["level"]

    @property
    def label(self) -> str:
        return self.skeptic.get("final_label") or self.analysis["verdict"]["label"]

    @property
    def style(self) -> dict:
        return LEVEL_STYLE.get(self.level, LEVEL_STYLE["판정 없음"])

    @property
    def analysis_day(self) -> str:
        """분석일: 설계를 잠근 날(판정이 확정된 날)."""
        return ((self.manifest.get("prereg") or {}).get("locked_at") or "")[:10]

    @property
    def analysis_no(self) -> str:
        """분석 번호(공문서 번호가 아님): JV-분석일-사례-모드. 파일·폴더 이름 대신 문서에 쓴다."""
        case = self.card.get("library_case") or "N"
        llm = (self.manifest.get("llm") or {}).get("calls")
        return f"JV-{self.analysis_day.replace('-', '')}-{case}{'C' if llm else 'T'}"

    @property
    def synthetic(self) -> bool:
        """정답을 심어 둔 합성데이터 사례(시연용). 공문 초안 같은 대외 문서는 만들지 않는다."""
        return bool(self.truth) or str(self.design.get("dataset", "")).startswith("synthetic")

    @property
    def policy_date(self) -> str:
        """화면 표시용 시행일. 월 자료라 분석 기준월(1일)과 실제 시행일이 다르면 둘 다 적는다(예: 사례 D 2023-08-10)."""
        from jverdict.agent.library import EVENTS
        sd, case = self.card.get("start_date", ""), self.card.get("library_case")
        real = next((e["start"] for e in EVENTS if case and e.get("case") == case), None)
        if real and real != sd:
            return f"{real} (월 자료: {int(sd[5:7])}월부터 비교)"
        return sd

    @property
    def meaning(self) -> str:
        return LEVEL_MEANING.get(self.level, "")

    @property
    def checks(self) -> list[dict]:
        return self.analysis["verdict"]["checks"]

    def objections(self) -> list[dict]:
        """반론 검증 표: 판정 규칙 반론 + Skeptic이 세운 추가 반론."""
        rows = [{"hypothesis": h["hypothesis"], "how": h["check"], "status": h["status"], "source": "판정 규칙"}
                for h in self.skeptic.get("rule_hypotheses", [])]
        for h in self.skeptic.get("hypotheses", []):
            how = "; ".join([f"{r['label']} → {r['pct'] * 100:+.1f}%" + (f" ({_xnote(r)})" if r.get("note") else "")
                             for r in h.get("results", [])]
                            + h.get("rule_results", [])) or "-"
            rows.append({"hypothesis": h["hypothesis"], "how": how, "status": h["status"],
                         "source": h.get("source", "Skeptic")})
        return rows

    def truth_line(self) -> str | None:  # noqa: D401
        if not self.truth:
            return None
        t, v, m = self.truth, self.analysis["verdict"], self.analysis["main"]
        inside = v["ci_used"][0] <= t["effect_log"] <= v["ci_used"][1]
        if t.get("note"):        # 사례 F: 정책 효과 0, 같은 시기 행사
            return (f"심어 둔 정답: {t['note']} · 판정 규칙 결과 {v['label']}, 최종 판정 {self.label}"
                    + (" · Skeptic이 행사 기간을 빼고 다시 추정해 판정을 낮춤" if self.skeptic.get("downgraded") else ""))
        return (f"심어 둔 정답 {t['group']} {t['effect_pct'] * 100:+.1f}% ({t['date']}부터) · 추정 {m['pct'] * 100:+.1f}% · "
                f"판정 구간이 정답을 {'포함함' if inside else '포함하지 않음'}")


def load_run(run_dir: Path | str) -> RunView:
    d = Path(run_dir)
    man = _read(d / "manifest.json")
    if man is None:
        raise FileNotFoundError(f"{d.name}: 끝나지 않은 실행(manifest.json 없음)")
    design = _read(d / "design.json", {})
    ev = [json.loads(x) for x in (_read(d / "events.jsonl", "") or "").splitlines() if x.strip()]
    rc = _read(d.parent / f"{d.name}-replay" / "replay_check.json") or _read(d / "replay_check.json")
    title, truth = design.get("dataset", ""), None
    try:
        from jverdict.agent import catalog
        ds = catalog.DATASETS.get(design.get("dataset"))
        if ds is not None:
            title = ds.title
            if ds.synthetic:
                truth = catalog.synthetic_truth(man.get("seed", 20261104), design.get("dataset", "synthetic_bus"))
    except Exception:  # noqa: BLE001  (보기 전용: 카탈로그를 못 읽어도 판정서는 보여 준다)
        pass
    card = _read(d / "card.json", man.get("card", {}))
    naive = _read(d / "naive.json")
    if naive is None:                      # 예전 기록: 잠긴 설계와 카드로 같은 값을 계산(파일은 쓰지 않음)
        from jverdict.report.naive import _safe
        naive = _safe(card, design, man.get("seed", 20261104))
    return RunView(d, man, card, design, _read(d / "analysis.json"),
                   _read(d / "skeptic.json", {}), _read(d / "report.json", {}), _read(d / "prereg.md", ""), ev, rc,
                   title, truth, naive)


def list_runs(runs_dir: Path) -> list[Path]:
    """끝난 실행만, 최근 것부터."""
    out = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "manifest.json").exists()] if runs_dir.exists() else []
    return sorted(out, key=lambda p: (p / "manifest.json").stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------- 그래프 데이터
def _pct(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else (math.exp(x) - 1) * 100


def _num(x) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


_NICE = (-95, -90, -80, -75, -60, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100,
         150, 200, 300, 400, 500, 700, 1000, 2000)


def pct_ticks(lo: float, hi: float, n: int = 6) -> list[tuple[float, str]]:
    """로그 축(위치=log(1+변화율))에 붙일 % 눈금: [(위치, '+50%'), ...]. 0%는 늘 넣는다."""
    cand = [(math.log1p(p / 100), p) for p in _NICE]
    inside = [(x, p) for x, p in cand if lo <= x <= hi]
    if len(inside) < 2:
        return [(0.0, "0%")]
    gap = (hi - lo) / n
    keep = [(0.0, 0)] if lo <= 0 <= hi else []
    for x, p in sorted(inside, key=lambda t: abs(t[0])):
        if all(abs(x - k[0]) >= gap * 0.8 for k in keep):
            keep.append((x, p))
    return [(x, f"{p:+d}%" if p else "0%") for x, p in sorted(keep)]


def _rolling(vals: list[float | None], k: int) -> list[float | None]:
    """가운데 맞춘 이동 평균(결측은 건너뛰고, 창의 절반 이상이 있어야 값)."""
    if k <= 1:
        return vals
    h, out = k // 2, []
    for i in range(len(vals)):
        w = [v for v in vals[max(0, i - h): i + h + 1] if v is not None]
        out.append(sum(w) / len(w) if len(w) * 2 >= k else None)
    return out


def nice(name: str) -> str:
    """계열 이름의 밑줄(도민_주점)을 화면용으로(도민·주점)."""
    return str(name).replace("_", "·")


def _names(x) -> str:
    return nice(x) if isinstance(x, str) else "·".join(nice(n) for n in (x or []))


def chart_data(a: dict, design: dict | None = None) -> dict | None:
    """화면·PDF 공용 그래프 데이터(값은 로그 차이, 눈금은 pct_ticks로 % 표시). 그릴 것이 없으면 None."""
    v, m = a["verdict"], a.get("main") or {}
    if a.get("curve"):
        cv = a["curve"]
        base = cv["counterfactual"]
        raw = [None if g is None else _num(g - base) for g in cv["g_adj"]]
        daily = len(cv["dates"][0]) == 10
        lo, hi = v["ci_used"]
        dz = a.get("design", {})
        return {"kind": "series", "dates": cv["dates"], "policy": cv["policy"], "daily": daily,
                "raw": raw, "y": _rolling(raw, 7) if daily else raw,
                "effect": m.get("effect", 0.0), "band": [_num(lo), _num(hi)],
                "effect_pct": m.get("pct", 0) * 100, "band_pct": [_pct(lo), _pct(hi)],
                "target": _names(dz.get("target")), "reference": _names(dz.get("reference")),
                "y_title": "대상과 대조 집단의 차이 변화(정책 전 평균 = 0%)",
                "note": ("선은 7일 이동 평균. " if daily else "") +
                        ("계절(작년 같은 때)을 뺀 값. " if (design or {}).get("season", "sdiff") == "sdiff" else "계절 보정 없음(사전 기간을 한 해 전체로 잡음). ") + "0% 선은 '정책이 없었다면'의 기준, 굵은 선은 정책 후 평균(추정 효과), "
                        "띠는 판정 구간."}
    es = a.get("event_study")
    if es and es.get("table"):
        rows = sorted(es["table"] + [{"time": es["ref"], "coef": 0.0, "ci_lo": None, "ci_hi": None, "pre": True,
                                      "ref": True}], key=lambda r: r["time"])
        dz = a.get("design", {})
        return {"kind": "event", "dates": [r["time"][:7] for r in rows], "policy": str(dz.get("date", ""))[:7],
                "y": [_num(r["coef"]) for r in rows], "lo": [_num(r["ci_lo"]) for r in rows],
                "hi": [_num(r["ci_hi"]) for r in rows], "ref": [bool(r.get("ref")) for r in rows],
                "pre": [bool(r["pre"]) for r in rows], "effect": m.get("effect", 0.0),
                "effect_pct": m.get("pct", 0) * 100,
                "target": str(dz.get("treated", "")), "reference": "다른 국적",
                "y_title": f"기준월({es['ref'][:7]}) 대비 대상의 상대 변화",
                "note": f"점은 월별 추정치, 세로 막대는 95% 구간. 정책 전 점들이 0%에서 멀면 정책 전부터 달랐다는 뜻"
                        f"(사전추세 p={es['pretrend_p']:.2g})."}
    return None


def _xnote(r: dict) -> str:
    """반론 결과 메모. '판단 불가'는 바꾼 조건과 상관없이 같은 뜻이라 화면에서는 일반 문장으로 보여 준다."""
    if r.get("status") == "판단 불가":
        return "효과 크기는 유지됐지만 유의하지 않음(검정력 부족). 반론의 증거로 보지 않음"
    return r.get("note", "")
