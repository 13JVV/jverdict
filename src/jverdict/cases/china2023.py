"""사례 D: 중국 단체관광 재개(2023-08-10) → 제주 내 중국인 카드 소비.

사전 등록 문서(docs/prereg/case_china_group_tour_2023.md)를 그대로 구현한다.
등록 문서의 해시가 lock 파일과 다르면 실행을 멈춘다(결과를 본 뒤 설계를 고치는 것을 막는다).

실행: python -m jverdict.cases.china2023
결과: outputs/cases/china2023/
"""
from __future__ import annotations

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from jverdict.config import OUTPUT_DIR, RAW_DIR, ROOT

SRC = RAW_DIR / "업종별성별카드이용정보.csv"
PREREG = ROOT / "docs" / "prereg" / "case_china_group_tour_2023.md"
LOCK = ROOT / "docs" / "prereg" / "case_china_group_tour_2023.lock.json"
OUT = OUTPUT_DIR / "cases" / "china2023"

TREAT = "중국"
CONTROLS = ["일본", "동남아", "기타외국"]
PRE = pd.date_range("2023-03-01", "2023-07-01", freq="MS")          # 원 등록
PRE_AMEND1 = pd.DatetimeIndex(["2023-03-01", "2023-04-01", "2023-07-01"])  # 수정 1: 외국인 2023-05·06 결측
POST = pd.date_range("2023-08-01", "2024-01-01", freq="MS")
REF = pd.Timestamp("2023-07-01")
DUTY_FREE = "면세점"
MDE_LIMIT = 0.20


def check_lock() -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    digest = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if digest != lock["sha256"]:
        raise SystemExit("사전 등록 문서가 잠금 이후 바뀌었습니다. 변경 기록을 확인하세요.")
    return lock


def load() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        """
        SELECT strptime(ym || '01', '%Y%m%d')::DATE AS month, replace(biz, '?', '') AS biz,
               ut AS grp, sum(amt)::DOUBLE AS amount
        FROM read_csv(?, header=false, skip=1, delim=',',
             columns={'ym':'VARCHAR','biz':'VARCHAR','ut':'VARCHAR','sex':'VARCHAR',
                      'users':'BIGINT','cnt':'BIGINT','amt':'BIGINT'})
        WHERE ut IN ('중국','일본','동남아','기타외국')
        GROUP BY ALL
        """,
        [str(SRC)],
    ).df()
    df["month"] = pd.to_datetime(df["month"])
    return df


def build_panel(df: pd.DataFrame, pre: pd.DatetimeIndex = PRE) -> pd.DataFrame:
    base = df[df.month.dt.year == 2019].assign(cal=lambda d: d.month.dt.month)[["grp", "biz", "cal", "amount"]]
    base = base.rename(columns={"amount": "base"})
    win = df[df.month.isin(pre.union(POST))].assign(cal=lambda d: d.month.dt.month)
    p = win.merge(base, on=["grp", "biz", "cal"], how="inner")
    p = p[(p.amount > 0) & (p.base > 0)].copy()
    p["y"] = np.log(p.amount / p.base)
    n_months = len(pre) + len(POST)
    full = p.groupby(["grp", "biz"]).month.nunique().loc[lambda s: s == n_months].index
    p = p.set_index(["grp", "biz"]).loc[full].reset_index()
    p["unit"] = p.grp + "|" + p.biz
    p["sm"] = p.biz + "|" + p.month.dt.strftime("%Y-%m")
    p["post"] = p.month.isin(POST).astype(int)
    annual = df[df.month.dt.year == 2019].groupby(["grp", "biz"]).amount.sum().rename("w2019")
    return p.merge(annual, on=["grp", "biz"], how="left")


def did(p: pd.DataFrame, treat: str, weights: bool = False) -> dict:
    d = p.assign(tp=lambda x: ((x.grp == treat) & (x.post == 1)).astype(int))
    model = smf.wls if weights else smf.ols
    kw = {"weights": d.w2019} if weights else {}
    fit = model("y ~ tp + C(unit) + C(sm)", data=d, **kw).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.unit)[0]})
    b, se = float(fit.params["tp"]), float(fit.bse["tp"])
    lo, hi = (float(v) for v in fit.conf_int().loc["tp"])
    return {"beta": b, "se": se, "ci": [lo, hi], "pct": float(np.expm1(b) * 100),
            "ci_pct": [float(np.expm1(lo) * 100), float(np.expm1(hi) * 100)],
            "n_obs": int(fit.nobs), "n_units": int(d.unit.nunique()),
            "n_treat_units": int(d.loc[d.grp == treat, "unit"].nunique())}


def event_study(p: pd.DataFrame, pre: pd.DatetimeIndex = PRE) -> tuple[pd.DataFrame, float]:
    d = p.copy()
    months = [m for m in pre.union(POST) if m != REF]
    names = []
    for m in months:
        n = f"e_{m:%Y_%m}"
        d[n] = ((d.grp == TREAT) & (d.month == m)).astype(int)
        names.append(n)
    fit = smf.ols("y ~ " + " + ".join(names) + " + C(unit) + C(sm)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.unit)[0]})
    ci = fit.conf_int()
    es = pd.DataFrame({"month": [f"{m:%Y-%m}" for m in months],
                       "coef": [fit.params[n] for n in names], "se": [fit.bse[n] for n in names],
                       "ci_lo": [ci.loc[n, 0] for n in names], "ci_hi": [ci.loc[n, 1] for n in names]})
    leads = [f"e_{m:%Y_%m}" for m in pre if m != REF]
    wald = fit.wald_test(", ".join(f"{n} = 0" for n in leads), scalar=True)
    return es, float(wald.pvalue)


def two_by_two_pandas(p: pd.DataFrame) -> float:
    u = p.groupby(["grp", "unit", "post"]).y.mean().unstack("post")
    change = (u[1] - u[0]).reset_index()
    return float(change.loc[change.grp == TREAT, 0].mean() - change.loc[change.grp != TREAT, 0].mean())


def two_by_two_sql(units: list[str]) -> float:
    """pandas 경로와 독립: 원본 CSV에서 SQL만으로 같은 2×2 값을 계산한다."""
    con = duckdb.connect()
    q = """
    WITH raw AS (
      SELECT strptime(ym||'01','%Y%m%d')::DATE AS month, replace(biz,'?','') AS biz, ut AS grp, sum(amt)::DOUBLE AS amt
      FROM read_csv(?, header=false, skip=1, delim=',',
           columns={'ym':'VARCHAR','biz':'VARCHAR','ut':'VARCHAR','sex':'VARCHAR','users':'BIGINT','cnt':'BIGINT','amt':'BIGINT'})
      WHERE ut IN ('중국','일본','동남아','기타외국') GROUP BY ALL
    ), cur AS (
      SELECT grp, biz, month, amt, month(month) AS cal,
             CASE WHEN month >= DATE '2023-08-01' THEN 1 ELSE 0 END AS post
      FROM raw WHERE month BETWEEN DATE '2023-03-01' AND DATE '2024-01-01'
    ), b AS (SELECT grp, biz, month(month) AS cal, amt AS base FROM raw WHERE year(month) = 2019),
    y AS (
      SELECT c.grp, c.grp || '|' || c.biz AS unit, c.post, ln(c.amt / b.base) AS y
      FROM cur c JOIN b USING (grp, biz, cal)
      WHERE c.amt > 0 AND b.base > 0 AND list_contains(?, c.grp || '|' || c.biz)
    ), u AS (
      SELECT grp, unit, avg(y) FILTER (WHERE post=1) - avg(y) FILTER (WHERE post=0) AS ch FROM y GROUP BY 1, 2
    )
    SELECT avg(ch) FILTER (WHERE grp='중국') - avg(ch) FILTER (WHERE grp<>'중국') FROM u
    """
    return float(con.execute(q, [str(SRC), units]).fetchone()[0])


def decide(main: dict, c1: bool, c2: bool, c3_pass: int) -> tuple[str, list[str]]:
    lo, hi = main["ci"]
    mde = float(np.expm1(2.8 * main["se"]))
    reasons = []
    sig = lo > 0 or hi < 0
    if not sig:
        if mde <= MDE_LIMIT:
            return "효과 없음", [f"신뢰구간이 0을 포함하고 MDE {mde:.1%} ≤ 20%"]
        return "판정 보류", [f"신뢰구간이 0을 포함하고 MDE {mde:.1%} > 20%"]
    if not c1 and not c2:
        return "판정 보류", ["사전추세(C1)와 공간 위약검정(C2)이 모두 실패"]
    fails = [n for n, ok in (("C1 사전추세", c1), ("C2 공간 위약검정", c2), ("C3 민감도", c3_pass >= 2)) if not ok]
    direction = "(증가)" if lo > 0 else "(감소)"
    if fails:
        return f"조건부 {direction}", [f"실패: {', '.join(fails)}"]
    return f"효과 있음 {direction}", ["C1·C2·C3 모두 통과"]


def run() -> dict:
    lock = check_lock()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    original = analyze(df, PRE, OUT / "original", lock, label="원 등록")
    amended = analyze(df, PRE_AMEND1, OUT / "amend1", lock, label="등록 후 수정 1(결과 보기 전, 외국인 2023-05·06 결측)")
    summary = {"original": original, "amend1": amended}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def analyze(df: pd.DataFrame, pre: pd.DatetimeIndex, out, lock: dict, label: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    p = build_panel(df, pre)

    g1 = {"china_units": int(p.loc[p.grp == TREAT, "unit"].nunique()),
          "control_units": int(p.loc[p.grp != TREAT, "unit"].nunique())}
    g1["pass"] = g1["china_units"] >= 10 and g1["control_units"] >= 10
    if not g1["pass"]:
        return {"label": label, "verdict": "판정 보류", "reasons": ["G1 진입 조건 실패(분석 기간의 모든 달이 있는 단위 부족)"], "G1": g1}

    main = did(p, TREAT)
    es, lead_p = event_study(p, pre)
    es.to_csv(out / "event_study.csv", index=False, encoding="utf-8-sig")
    c1 = lead_p >= 0.10

    placebo = {}
    for g in CONTROLS:
        placebo[g] = did(p[p.grp != TREAT], g)
    c2 = all(main["beta"] > v["beta"] for v in placebo.values())

    sens = {
        "S1 사후 2023-09부터(8월 제외)": did(p[p.month != pd.Timestamp("2023-08-01")], TREAT),
        "S2 2019 연간 금액 가중": did(p, TREAT, weights=True),
        "S3 면세점 제외": did(p[p.biz != DUTY_FREE], TREAT),
    }
    c3_pass = sum(1 for v in sens.values() if v["ci"][0] > 0)
    pd.DataFrame([{"spec": k, **{kk: vv for kk, vv in v.items() if kk not in ("ci", "ci_pct")},
                   "ci_lo": v["ci"][0], "ci_hi": v["ci"][1]} for k, v in sens.items()]
                 ).to_csv(out / "sensitivity.csv", index=False, encoding="utf-8-sig")

    units = sorted(p.unit.unique())
    c4_pd, c4_sql = two_by_two_pandas(p), two_by_two_sql(units)
    c4 = abs(c4_pd - c4_sql) < 1e-9
    verdict, reasons = decide(main, c1, c2, c3_pass) if c4 else ("판정 없음", ["C4 이중 계산 불일치"])

    return {
        "label": label, "pre_months": [f"{m:%Y-%m}" for m in pre],
        "prereg": {"file": lock["file"], "sha256": lock["sha256"], "locked_at": lock["locked_at"], "history": lock.get("history", [])},
        "verdict": verdict, "reasons": reasons, "G1": g1, "main": main,
        "mde_pct": float(np.expm1(2.8 * main["se"]) * 100),
        "C1_pretrend": {"lead_joint_p": lead_p, "pass": c1},
        "C2_placebo_space": {"placebo": {k: {"beta": v["beta"], "pct": v["pct"], "ci": v["ci"]} for k, v in placebo.items()},
                             "pass": c2},
        "C3_sensitivity": {"specs": {k: {"beta": v["beta"], "pct": v["pct"], "ci": v["ci"]} for k, v in sens.items()},
                           "n_pass": c3_pass, "pass": c3_pass >= 2},
        "C4_double_compute": {"pandas": c4_pd, "sql": c4_sql, "pass": c4},
        "china_units": sorted(p.loc[p.grp == TREAT, "biz"].unique().tolist()),
    }


def report(s: dict) -> None:
    print(f"[{s['label']}] 판정: {s['verdict']} — {'; '.join(s['reasons'])}")
    if "main" not in s:
        print(f"   G1: {s['G1']}")
        return
    m = s["main"]
    print(f"   주분석 β={m['beta']:+.3f} ({m['pct']:+.1f}%), 95% CI [{m['ci_pct'][0]:+.1f}%, {m['ci_pct'][1]:+.1f}%], "
          f"MDE {s['mde_pct']:.1f}%, 단위 {m['n_units']}개(중국 {m['n_treat_units']}), 관측 {m['n_obs']}")
    print(f"   C1 사전추세 결합 p={s['C1_pretrend']['lead_joint_p']:.3f} → {'통과' if s['C1_pretrend']['pass'] else '실패'}")
    print("   C2 공간 위약: " + ", ".join(f"{k} {v['pct']:+.1f}%" for k, v in s['C2_placebo_space']['placebo'].items())
          + f" → {'통과' if s['C2_placebo_space']['pass'] else '실패'}")
    print("   C3 민감도: " + "; ".join(f"{k} {v['pct']:+.1f}% CI[{np.expm1(v['ci'][0])*100:+.1f}, {np.expm1(v['ci'][1])*100:+.1f}]"
                                        for k, v in s['C3_sensitivity']['specs'].items())
          + f" → {s['C3_sensitivity']['n_pass']}/3")
    print(f"   C4 이중 계산: pandas {s['C4_double_compute']['pandas']:.6f} / SQL {s['C4_double_compute']['sql']:.6f} → "
          f"{'일치' if s['C4_double_compute']['pass'] else '불일치'}")


if __name__ == "__main__":
    res = run()
    report(res["original"])
    report(res["amend1"])
