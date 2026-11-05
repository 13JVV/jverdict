"""외부 통계 대조: 데이터 집계값이 공식 발표 수치와 같은 규모인지 확인한다.

기간이 서로 다르면(예: 공식 발표는 7개월 누적, 데이터는 3개월) 기간당 평균으로 맞춰 비교하고,
그 사실을 결과에 남긴다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ._panel import fmt, freq_params


@dataclass
class ExternalStat:
    label: str                 # 예: "청소년·어린이 무료 이용 누적 건수(도 발표)"
    official_total: float      # 공식 발표 누적값
    start: str                 # 공식 집계 기간 시작
    end: str                   # 공식 집계 기간 끝
    groups: list[str]          # 데이터에서 대응하는 집단
    source: str = ""           # 출처
    same_definition: bool = False  # 데이터와 집계 대상·기준이 같은 통계인가
    tolerance: float | None = None  # 비율이 [1/tol, tol] 밖이면 불일치. 없으면 같은 정의 1.25, 아니면 2.0

    @property
    def tol(self) -> float:
        if self.tolerance is not None:
            return self.tolerance
        return 1.25 if self.same_definition else 2.0


@dataclass
class ExternalCheck:
    label: str
    source: str
    official_per_period: float
    data_per_period: float
    ratio: float               # 공식 ÷ 데이터
    contradiction: bool
    data_window: tuple[str, str]
    official_window: tuple[str, str]
    period_mismatch: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def external_check(wide: pd.DataFrame, stat: ExternalStat, freq: str) -> ExternalCheck:
    fp = freq_params(freq)
    s, e = pd.Timestamp(stat.start), pd.Timestamp(stat.end)
    official_periods = len(pd.date_range(s, e, freq=fp.code))
    official_mean = stat.official_total / max(official_periods, 1)

    cols = [g for g in stat.groups if g in wide.columns]
    if not cols:
        raise ValueError(f"데이터에 없는 집단: {stat.groups}")
    ds, de = max(s, wide.index[0]), min(e, wide.index[-1])
    seg = wide.loc[ds:de, cols].sum(axis=1, min_count=1).dropna()
    data_mean = float(seg.mean()) if len(seg) else float("nan")
    ratio = official_mean / data_mean if data_mean and np.isfinite(data_mean) else float("inf")
    contra = not (1 / stat.tol <= ratio <= stat.tol)
    mismatch = (ds != s) or (de != e)
    unit = fp.label
    msg = (f"{stat.label}: 공식 {unit}평균 {official_mean:,.0f} vs 데이터 {unit}평균 {data_mean:,.0f} "
           f"(공식이 데이터의 {ratio:.2f}배, 허용 {1 / stat.tol:.2f}~{stat.tol:.2f}배)")
    if mismatch:
        msg += f". 기간이 달라 {unit}평균으로 비교(데이터 {fmt(ds, freq)}~{fmt(de, freq)})"
    return ExternalCheck(stat.label, stat.source, round(official_mean, 2), round(data_mean, 2),
                         round(ratio, 3), contra, (fmt(ds, freq), fmt(de, freq)),
                         (fmt(s, freq), fmt(e, freq)), mismatch, msg)
