"""정책 카드 직접 만들기: 등록 사례가 아닌 정책을 API 키 없이(템플릿 모드) 판정하기 위한 카드.

사용자는 데이터셋, 시행일, 정책 대상(집단·업종), 대조 후보(선택)만 고른다. 나머지는 Agent가 한다.
- Data Scout: 고른 계열로 측정 단절·커버리지를 점검한다.
- Design Agent: 대조 후보를 비워 두면 정책 전 유사도로 대조를 고른다(효과 가림). 그 뒤 설계를 잠근다.
- 이후 추정·반론·판정서는 등록 사례와 같다.
"""
from __future__ import annotations

import pandas as pd

from . import catalog

AUTO_MAX = 6          # 대조 후보를 자동으로 만들 때 최대 개수(규모 순)


def options(dataset_id: str) -> dict:
    """화면에 보여 줄 선택지: 집단(규모 순), 업종(카드 데이터), 기간과 시행일로 고를 수 있는 범위."""
    ds = catalog.DATASETS[dataset_id]
    prof = catalog.profile(dataset_id)
    lo, hi = pd.Timestamp(prof["period"][0]), pd.Timestamp(prof["period"][1])
    if ds.freq == "MS":
        first, last = lo + pd.DateOffset(months=15), hi - pd.DateOffset(months=2)
    else:
        first, last = lo + pd.Timedelta(days=400), hi - pd.Timedelta(days=30)
    units: list[str] = []
    if dataset_id == "card_sector":
        units = sorted(catalog.load(dataset_id)["unit"].unique().tolist())
    return {"dataset": dataset_id, "title": ds.title, "freq": ds.freq, "group_label": ds.group_label,
            "unit_label": ds.unit_label, "value_label": ds.value_label,
            "groups": [g["group"] for g in prof["groups"]], "units": units,
            "period": (str(lo.date()), str(hi.date())), "start_range": (str(first.date()), str(last.date()))}


def _name(groups: list[str], units: list[str] | None) -> str:
    base = "+".join(groups)
    if units:
        base += f"·{units[0]}" + (f" 외 {len(units) - 1}" if len(units) > 1 else "")
    return base


def build_card(dataset_id: str, policy: str, start_date: str, target_groups: list[str],
               target_units: list[str] | None = None, references: list[dict] | None = None,
               expected: str | None = "increase", reference_effect: float = 0.10) -> tuple[dict | None, list[str]]:
    """(정책 카드, 오류 목록). references: [{"groups": [...], "units": [...]}]. 비우면 Agent가 고른다."""
    errs: list[str] = []
    if dataset_id not in catalog.DATASETS:
        return None, [f"없는 데이터셋: {dataset_id}"]
    opt = options(dataset_id)
    policy = (policy or "").strip()
    if not policy:
        errs.append("정책 이름을 적어 주세요")
    if not target_groups:
        errs.append(f"정책 대상 {opt['group_label']}을(를) 하나 이상 골라 주세요")
    bad = [g for g in target_groups if g not in opt["groups"]]
    if bad:
        errs.append(f"데이터에 없는 {opt['group_label']}: {bad}")
    target_units = [u for u in (target_units or []) if u]
    if target_units and not opt["units"]:
        errs.append(f"이 데이터는 {opt['unit_label']}을(를) 고를 수 없습니다")
    try:
        t0 = pd.Timestamp(start_date)
    except Exception:  # noqa: BLE001
        return None, errs + ["시행일 형식이 맞지 않습니다(YYYY-MM-DD)"]
    assumptions = []
    if opt["freq"] == "MS" and t0.day != 1:
        assumptions.append(f"월 단위 자료라 시행일 {t0.date()}을 그 달 1일로 둠(시행한 달은 사후에 넣음, 절반만 처치된 달일 수 있음)")
        t0 = t0.replace(day=1)
    a, b = pd.Timestamp(opt["start_range"][0]), pd.Timestamp(opt["start_range"][1])
    if not a <= t0 <= b:
        errs.append(f"시행일은 {a.date()} ~ {b.date()} 사이여야 합니다(정책 전 비교 기간과 정책 후 자료가 필요)")

    tgt = {"name": _name(target_groups, target_units), "groups": list(target_groups), "units": target_units or None}
    auto = not references
    if auto:
        refs = [{"groups": [g], "units": target_units or None} for g in opt["groups"] if g not in target_groups][:AUTO_MAX]
        if dataset_id == "synthetic_bus":
            refs = refs[:4]
    else:
        refs = [{"groups": list(r.get("groups") or []), "units": [u for u in (r.get("units") or []) if u] or None}
                for r in references]
    out_refs, seen = [], {tgt["name"]}
    for r in refs:
        if not r["groups"]:
            errs.append("대조 후보마다 집단을 하나 이상 골라 주세요")
            continue
        if set(r["groups"]) == set(target_groups) and (r["units"] or None) == (target_units or None):
            errs.append("대조 후보가 정책 대상과 같습니다")
            continue
        nm = _name(r["groups"], r["units"])
        if nm in seen:
            continue
        seen.add(nm)
        out_refs.append({"name": nm, **r})
    if not out_refs:
        errs.append("대조 후보가 없습니다")
    if errs:
        return None, errs

    verb = {"increase": "늘었는가", "decrease": "줄었는가"}.get(expected, "달라졌는가")
    what = f"{tgt['name']}의 {opt['value_label'].split('(')[0]}"
    card = {"policy": policy, "question": f"{policy}({t0.date()})로 {what}이(가) {verb}?",
            "start_date": str(t0.date()), "end_date": None, "treated": tgt["name"],
            "outcome": f"{opt['title']}: {opt['value_label']}", "expected": expected, "region": "제주도",
            "source": "사용자 입력(정책 카드 직접 작성)", "reference_effect": float(reference_effect),
            "needed_hint": [], "prereg": None, "library_case": None,
            "assumptions": assumptions + ["정책 시행일과 대상은 사용자가 입력함(확인 필요)",
                                          "대조 집단은 Agent가 정책 전 유사도로 고름" if auto else "대조 집단은 사용자가 정함"],
            "manual": {"dataset": dataset_id, "target": tgt, "references": out_refs, "auto": auto}}
    return card, []
