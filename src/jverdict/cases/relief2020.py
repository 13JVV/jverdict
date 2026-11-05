"""사례 B: 1차 긴급재난지원금(2020-05)이 제주도민의 카드 소비를 늘렸는가.

설계: 삼중차분(전년동월비 로그 증가율 기준)
  - 처치 집단: 제주도민(지원금 수령, 주소지 광역시도 안에서만 사용 가능)
  - 비교 집단: 내국인 관광객(제주에서 쓸 수 있는 지원금이 없음)
  - 처치 업종: 지원금 사용 가능 업종 / 대조 업종: 사용 불가 업종
  TD_t = (도민 가능 − 도민 불가) − (관광객 가능 − 관광객 불가)
  ※ 각 항은 log(금액_t / 금액_{t-12})
  관광객 차분은 업종별 코로나 충격(관광 급감)을, 업종 차분은 집단별 충격을 걷어낸다.

효과 통계량: 점프 J_t = TD_t − mean(TD_{t-3}, TD_{t-2}, TD_{t-1})
  코로나가 2020-02부터 기준선을 옮겼기 때문에 직전 3개월을 기준으로 쓴다.
검증:
  - 시점 위약검정: 정책과 무관한 달들의 J 분포 안에서 2020-05의 순위 p값
  - 민감도: 대조 업종 정의 3종, 기준 기간 2·3개월, 관광객 차분 제외
  - 이중 계산: 핵심 수치를 DuckDB(SQL)와 pandas로 각각 계산해 대조
  - 업종별 이질성: 필수재·소매 vs 음식·대면서비스 (KDI 결과와 비교)

실행: python -m jverdict.cases.relief2020
결과: outputs/cases/relief2020/
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from jverdict.config import OUTPUT_DIR, RAW_DIR

SRC = RAW_DIR / "업종별성별카드이용정보.csv"
SECTORS = Path(__file__).with_name("relief2020_sectors.csv")
OUT = OUTPUT_DIR / "cases" / "relief2020"

RES, TOUR = "제주도민", "내국인"
POLICY_MONTH = pd.Timestamp("2020-05-01")
# 위약검정에서 뺄 달: 정책 기간(지급~사용기한 8/31 + 여파 1개월),
# 그리고 다른 대규모 지원금 시기(2021-09~12 상생 국민지원금)
EXCLUDE_FROM_PLACEBO = (
    list(pd.date_range("2020-05-01", "2020-09-01", freq="MS"))
    + list(pd.date_range("2021-09-01", "2021-12-01", freq="MS"))
)
PLACEBO_RANGE = ("2019-04-01", "2024-12-01")

ESSENTIAL_RETAIL = [
    "체인화 편의점", "과실 및 채소 소매업", "수산물 소매업", "육류 소매업",
    "기타음식료품위주종합소매업", "그외 기타 종합 소매업", "건강보조식품 소매업",
    "차량용 주유소 운영업", "차량용 가스 충전업", "빵 및 과자류 소매업",
]
FOOD_FACE = [
    "한식 음식점업", "서양식 음식점업", "일식 음식점업", "중식 음식점업",
    "기타 외국식 음식점업", "피자 햄버거 샌드위치 및 유사 음식점업",
    "비알콜 음료점업", "기타 주점업", "욕탕업", "마사지업",
]


# ---------------------------------------------------------------- 데이터 ----
def load_card() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        """
        SELECT strptime(ym || '01', '%Y%m%d')::DATE AS month,
               replace(biz, '?', '') AS biz,      -- '기타음?식료품…' 인코딩 오류 병합
               ut AS user_type,
               sum(amt) AS amount
        FROM read_csv(?, header=false, skip=1, delim=',',
             columns={'ym':'VARCHAR','biz':'VARCHAR','ut':'VARCHAR','sex':'VARCHAR',
                      'users':'BIGINT','cnt':'BIGINT','amt':'BIGINT'})
        WHERE ut IN (?, ?)
        GROUP BY ALL
        """,
        [str(SRC), RES, TOUR],
    ).df()
    df["month"] = pd.to_datetime(df["month"])
    return df


def load_tiers() -> pd.DataFrame:
    return pd.read_csv(SECTORS)


def control_sets(tiers: pd.DataFrame) -> dict[str, list[str]]:
    sure_no = tiers.loc[tiers.tier == "불가", "biz"].tolist()
    unsure = tiers.loc[tiers.tier == "불확실", "biz"].tolist()
    return {
        "주분석: 확실한 불가 업종": sure_no,
        "민감도: 불가 + 면세점·사행업": sure_no + ["면세점", "기타 갬블링 및 베팅업"],
        "민감도: 불가 + 불확실 전체": sure_no + unsure,
    }


# ---------------------------------------------------------------- 계산 ------
def yoy_log(series: pd.Series) -> pd.Series:
    s = series.sort_index().asfreq("MS")
    return np.log(s / s.shift(12))


def triple_diff(df: pd.DataFrame, treat: list[str], control: list[str],
                use_tourist: bool = True) -> pd.Series:
    def agg(group: str, sectors: list[str]) -> pd.Series:
        m = df[(df.user_type == group) & (df.biz.isin(sectors))]
        return yoy_log(m.groupby("month")["amount"].sum())

    gap_res = agg(RES, treat) - agg(RES, control)
    if not use_tourist:
        return gap_res.dropna()
    gap_tour = agg(TOUR, treat) - agg(TOUR, control)
    return (gap_res - gap_tour).dropna()


def jump(td: pd.Series, window: int = 3) -> pd.Series:
    return td - td.shift(1).rolling(window).mean()


def placebo_rank(j: pd.Series, month: pd.Timestamp) -> dict:
    pool = j.loc[PLACEBO_RANGE[0]:PLACEBO_RANGE[1]].drop(index=EXCLUDE_FROM_PLACEBO, errors="ignore").dropna()
    actual = float(j.loc[month])
    # 단측(증가 방향) 순위 p값: 실제 값 이상인 위약 수 + 1 / (위약 수 + 1)
    p = (int((pool >= actual).sum()) + 1) / (len(pool) + 1)
    return {"actual": actual, "n_placebo": int(len(pool)), "p_rank": p,
            "placebo_p95": float(pool.quantile(0.95)), "placebo_max": float(pool.max())}


def sql_double_check(treat: list[str], control: list[str]) -> float:
    """pandas 경로와 독립적으로 2020-05 TD를 SQL로만 계산한다."""
    con = duckdb.connect()
    q = """
    WITH base AS (
      SELECT ym, ut, replace(biz,'?','') AS biz, sum(amt)::DOUBLE AS amt
      FROM read_csv(?, header=false, skip=1, delim=',',
           columns={'ym':'VARCHAR','biz':'VARCHAR','ut':'VARCHAR','sex':'VARCHAR',
                    'users':'BIGINT','cnt':'BIGINT','amt':'BIGINT'})
      WHERE ym IN ('201905','202005') AND ut IN (?, ?)
      GROUP BY ALL
    ), cell AS (
      SELECT ut,
             CASE WHEN list_contains(?, biz) THEN 'T' WHEN list_contains(?, biz) THEN 'C' END AS arm,
             sum(amt) FILTER (WHERE ym='202005') / sum(amt) FILTER (WHERE ym='201905') AS ratio
      FROM base GROUP BY 1, 2
    )
    SELECT  (ln(max(ratio) FILTER (WHERE ut=? AND arm='T')) - ln(max(ratio) FILTER (WHERE ut=? AND arm='C')))
          - (ln(max(ratio) FILTER (WHERE ut=? AND arm='T')) - ln(max(ratio) FILTER (WHERE ut=? AND arm='C')))
    FROM cell
    """
    return float(con.execute(q, [str(SRC), RES, TOUR, treat, control, RES, RES, TOUR, TOUR]).fetchone()[0])


# ---------------------------------------------------------------- 실행 ------
def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_card()
    tiers = load_tiers()
    unknown = sorted(set(df.biz) - set(tiers.biz))
    treat = tiers.loc[tiers.tier == "가능", "biz"].tolist()
    controls = control_sets(tiers)
    main_ctrl = controls["주분석: 확실한 불가 업종"]

    # 주분석
    td = triple_diff(df, treat, main_ctrl)
    j3 = jump(td, 3)
    main = placebo_rank(j3, POLICY_MONTH)
    june = float(j3.loc["2020-06-01"])
    series = pd.DataFrame({"TD": td, "J3": j3}).loc["2019-01-01":"2021-12-01"]
    series.to_csv(OUT / "main_series.csv", encoding="utf-8-sig")

    # 민감도
    sens = []
    for name, ctrl in controls.items():
        for w in (2, 3):
            jj = jump(triple_diff(df, treat, ctrl), w)
            r = placebo_rank(jj, POLICY_MONTH)
            sens.append({"spec": name, "baseline_months": w, **r})
    jj = jump(triple_diff(df, treat, main_ctrl, use_tourist=False), 3)
    sens.append({"spec": "민감도: 관광객 차분 제외(이중차분)", "baseline_months": 3,
                 **placebo_rank(jj, POLICY_MONTH)})
    pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False, encoding="utf-8-sig")

    # 이중 계산 (2020-05 TD: pandas vs SQL)
    pandas_td = float(td.loc[POLICY_MONTH])
    sql_td = sql_double_check(treat, main_ctrl)
    double_ok = abs(pandas_td - sql_td) < 1e-9

    # 업종별 이질성: 각 가능 업종 단독을 처치로
    het = []
    for b in treat:
        s = jump(triple_diff(df, [b], main_ctrl), 3)
        if POLICY_MONTH in s.index and not np.isnan(s.loc[POLICY_MONTH]):
            base = df[(df.user_type == RES) & (df.biz == b) & (df.month.dt.year == 2019)].amount.sum()
            grp = "필수재·소매" if b in ESSENTIAL_RETAIL else ("음식·대면서비스" if b in FOOD_FACE else "기타")
            het.append({"biz": b, "group": grp, "J3_may": float(s.loc[POLICY_MONTH]),
                        "resident_2019_amount": int(base)})
    het = pd.DataFrame(het).sort_values("J3_may", ascending=False)
    het.to_csv(OUT / "heterogeneity.csv", index=False, encoding="utf-8-sig")
    grp_mean = (het[het.group != "기타"]
                .assign(w=lambda d: d.resident_2019_amount)
                .groupby("group").apply(lambda g: np.average(g.J3_may, weights=g.w), include_groups=False))

    # 도민 내부 이중차분 (관광객 비교군은 2020-05 황금연휴 관광 반등 때문에 부적절 — findings F5)
    resident_specs = []
    resident_ctrls = {
        "대형종합+유흥": controls["주분석: 확실한 불가 업종"],
        "대형종합만": ["기타 대형 종합 소매업"],
        "대형종합+유흥+면세": controls["주분석: 확실한 불가 업종"] + ["면세점"],
    }
    for cn, c in resident_ctrls.items():
        d = triple_diff(df, treat, c, use_tourist=False)
        for w in (1, 2, 3):
            resident_specs.append({"control": cn, "baseline_months": w, **placebo_rank(jump(d, w), POLICY_MONTH)})
    pd.DataFrame(resident_specs).to_csv(OUT / "resident_did_specs.csv", index=False, encoding="utf-8-sig")

    # 업종 단위 부호 검정: 가능 업종 중 5월 점프가 양수인 비율을 위약 달들과 비교
    jz = pd.DataFrame({b: jump(triple_diff(df, [b], main_ctrl, use_tourist=False), 1) for b in treat})
    jz = jz.loc[:, jz.notna().sum() > 60]
    share = (jz > 0).mean(axis=1)
    pool = share.loc[PLACEBO_RANGE[0]:PLACEBO_RANGE[1]].drop(index=EXCLUDE_FROM_PLACEBO, errors="ignore").dropna()
    sign_test = {"n_sectors": int(jz.shape[1]), "share_positive_may": float(share.loc[POLICY_MONTH]),
                 "placebo_mean": float(pool.mean()), "placebo_p95": float(pool.quantile(0.95)),
                 "p_rank": (int((pool >= share.loc[POLICY_MONTH]).sum()) + 1) / (len(pool) + 1)}

    summary = {
        "resident_did": resident_specs,
        "sector_sign_test": sign_test,
        "policy": "1차 긴급재난지원금 (2020-05 지급, 사용기한 2020-08-31)",
        "data": "제주데이터허브 '업종별 성별 카드 이용 정보' (비씨카드 실적, 2018-01~2025-06)",
        "design": "삼중차분(도민·관광객 × 사용가능·불가 업종), 전년동월비 로그 증가율, 직전 3개월 기준 점프",
        "n_treat_sectors": len(treat),
        "control_sectors": main_ctrl,
        "unclassified_sectors": unknown,
        "main": {**main, "effect_pct": (np.exp(main["actual"]) - 1) * 100,
                 "june_J3": june, "june_effect_pct": (np.exp(june) - 1) * 100},
        "double_compute": {"pandas": pandas_td, "sql": sql_td, "match": double_ok},
        "heterogeneity_weighted_mean_J3": {k: float(v) for k, v in grp_mean.items()},
        "sensitivity_all_positive": all(s["actual"] > 0 for s in sens),
        "sensitivity_p_below_0.05": sum(s["p_rank"] < 0.05 for s in sens),
        "n_sensitivity": len(sens),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = run()
    m = s["main"]
    print(f"[주분석] 2020-05 점프 {m['actual']:+.3f} (약 {m['effect_pct']:+.1f}%), "
          f"위약 {m['n_placebo']}개월 중 순위 p={m['p_rank']:.3f} (위약 95% 분위 {m['placebo_p95']:+.3f}, 최대 {m['placebo_max']:+.3f})")
    print(f"[6월] 점프 {m['june_J3']:+.3f} (약 {m['june_effect_pct']:+.1f}%)")
    print(f"[이중 계산] pandas {s['double_compute']['pandas']:.6f} / SQL {s['double_compute']['sql']:.6f} → 일치 {s['double_compute']['match']}")
    print(f"[민감도] {s['n_sensitivity']}개 중 효과 양수 전부={s['sensitivity_all_positive']}, p<0.05 {s['sensitivity_p_below_0.05']}개")
    print(f"[이질성] 가중 평균 J3: {s['heterogeneity_weighted_mean_J3']}")
    print(f"[미분류 업종] {s['unclassified_sectors']}")
