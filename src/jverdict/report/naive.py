"""'단순 비교였다면': 대조 집단·계절 보정·검증 없이 정책 대상만 시행 전후로 비교한 값.

판정관이 왜 다른 결론을 냈는지 보여 주기 위한 대비용 숫자다. 판정에는 쓰지 않는다.
- 정책 대상 계열의 시행 후 N기간 평균을 시행 직전 N기간 평균과 비교한다(N = 설계의 사후 기간, 자료 끝에서 자름).
- 값이 0인 기간(결측·집계 누락)은 뺀다. 흔한 엑셀식 비교와 같은 방식이다.
- 잠긴 설계(design.json)와 정책 카드만으로 결정적으로 계산하므로, 예전 기록(Claude 포함)에도 같은 값이 나온다.
"""
from __future__ import annotations


import pandas as pd

POST_DEFAULT = {"D": 182, "MS": 6}
THRESH = 0.05          # 단순 비교에서 '변화가 있다'고 읽힐 크기(±5%)


def _target_spec(design: dict) -> tuple[str, dict] | None:
    if design.get("method") == "did_panel":
        p = design["panel"]
        return p["treated"], {"groups": [p["treated"]], "units": None}
    for s in design.get("series", []):
        if s["name"] == design.get("target"):
            return s["name"], {"groups": s["groups"], "units": s.get("units_tier") or (s.get("units_list") or None)}
    return None


def _reading(pct: float, expected: str | None) -> tuple[str, str]:
    """(단순 비교의 결론, 등급 이름)."""
    if abs(pct) < THRESH:
        return "변화 없음 → '효과 없음'으로 보였을 것", "효과 없음"
    up = pct > 0
    if expected == "decrease":
        return ("목표와 반대로 늘어남 → '정책 실패'로 보였을 것", "역효과") if up else \
            ("줄어듦 → '효과 있음'으로 보였을 것", "효과 있음")
    if expected == "increase":
        return ("늘어남 → '효과 있음'으로 보였을 것", "효과 있음") if up else \
            ("줄어듦 → '정책 실패(역효과)'로 보였을 것", "역효과")
    return (f"{'늘어남' if up else '줄어듦'} → '효과 있음({'증가' if up else '감소'})'으로 보였을 것", "효과 있음")


def naive_compare(card: dict, design: dict, seed: int = 20261104) -> dict | None:
    from jverdict.agent import catalog
    spec = _target_spec(design)
    if spec is None or design.get("dataset") not in catalog.DATASETS:
        return None
    name, sp = spec
    freq = catalog.DATASETS[design["dataset"]].freq
    _, wide = catalog.series_frame(design["dataset"], {name: sp}, seed)
    y = wide[name].astype(float)
    t0 = pd.Timestamp(card["start_date"])
    n = int(design.get("post") or POST_DEFAULT[freq])
    post = y[y.index >= t0].iloc[:n]
    pre = y[y.index < t0].iloc[-len(post):] if len(post) else y.iloc[:0]
    post, pre = post[post > 0], pre[pre > 0]
    if len(post) == 0 or len(pre) == 0:
        return None
    pct = float(post.mean() / pre.mean() - 1)
    fmt = (lambda t: t.strftime("%Y-%m-%d")) if freq == "D" else (lambda t: t.strftime("%Y-%m"))
    unit = "일" if freq == "D" else "개월"
    reading, level = _reading(pct, card.get("expected"))
    return {"method": "단순 전후 비교(대조·계절 보정·검증 없음)", "target": name, "unit": unit,
            "pre": [fmt(pre.index[0]), fmt(pre.index[-1])], "post": [fmt(post.index[0]), fmt(post.index[-1])],
            "n_pre": int(len(pre)), "n_post": int(len(post)), "pct": round(pct, 6),
            "reading": reading, "naive_level": level,
            "sentence": f"'{name}'의 시행 후 {len(post)}{unit} 평균이 직전 {len(pre)}{unit}보다 {pct * 100:+.1f}%: {reading}"}


def contrast(naive: dict | None, final_level: str) -> str | None:
    """단순 비교와 판정관의 결론이 어떻게 다른지 한 문장."""
    if not naive:
        return None
    if naive["naive_level"] == final_level or (naive["naive_level"] == "효과 있음" and final_level == "효과 있음"):
        return "판정관도 같은 방향의 결론을 냈지만, 대조 집단·위약검정·사전 등록으로 근거를 확인했습니다."
    return f"판정관은 같은 숫자를 그대로 믿지 않고 '{final_level}'로 판정했습니다. 이유는 아래 근거와 검증 결과에 있습니다."


def _safe(card, design, seed):
    try:
        return naive_compare(card, design, seed)
    except Exception:  # noqa: BLE001  보기 전용 숫자: 실패해도 판정서는 그대로
        return None


