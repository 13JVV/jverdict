"""품질 점검 툴이 공통으로 쓰는 패널 도우미.

입력은 항상 '긴 형식' 표 하나다: 시점(time) × 집단(group) [× 단위(unit)] × 값(value).
여기서 빈틈 없는 달력 위의 넓은 형식(시점 × 집단)으로 바꾼다. 행이 없는 칸은 NaN이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FreqParams:
    """시간 단위별 기본 창 크기. 모든 값은 '기간 수'(일 또는 월)다."""

    code: str          # pandas 빈도 코드
    label: str         # 사람이 읽는 단위
    window: int        # 전후 비교 창
    settle: int        # 정책 시점 뒤 정착 기간(점진 이탈을 기다림)
    season: int        # 계절 주기(요일을 맞추려고 일 단위는 364)
    years_back: int    # 계절 보정에 쓸 과거 주기 수
    near: int          # '같은 시점'으로 보는 허용 오차
    onset_horizon: int  # 전환 모양을 볼 기간
    persist: tuple[int, int]  # 지속성 확인 창(시점 뒤 a~b 기간)
    partial_ref: int   # 부분 집계 판단용 이동 중앙값 창


DAILY = FreqParams("D", "일", window=28, settle=14, season=364, years_back=3, near=7,
                   onset_horizon=56, persist=(90, 180), partial_ref=29)
MONTHLY = FreqParams("MS", "개월", window=3, settle=0, season=12, years_back=3, near=1,
                     onset_horizon=6, persist=(6, 12), partial_ref=7)


def freq_params(freq: str) -> FreqParams:
    if freq in ("D", "day", "daily"):
        return DAILY
    if freq in ("MS", "M", "month", "monthly"):
        return MONTHLY
    raise ValueError(f"지원하지 않는 시간 단위: {freq}")


def infer_freq(times: pd.Series) -> str:
    t = pd.to_datetime(pd.Series(times).dropna().unique())
    if len(t) < 3:
        return "D"
    t = np.sort(t)
    gap = pd.Series(np.diff(t)).dt.days.median()
    if gap >= 27 and pd.DatetimeIndex(t).day.max() == 1:
        return "MS"
    return "D"


def to_wide(df: pd.DataFrame, time: str, group: str, value: str, freq: str) -> pd.DataFrame:
    """긴 형식 → 빈틈 없는 달력 × 집단 합계. 행이 없으면 NaN(0과 구별)."""
    d = df[[time, group, value]].copy()
    d[time] = pd.to_datetime(d[time])
    w = d.groupby([time, group])[value].sum(min_count=1).unstack(group)
    full = pd.date_range(w.index.min(), w.index.max(), freq=freq_params(freq).code)
    w = w.reindex(full)
    w.index.name = "time"
    w.columns.name = None
    return w.astype(float)


def ranges(index: pd.DatetimeIndex, mask: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """참(True)인 연속 구간을 (시작, 끝, 길이)로 돌려준다."""
    out = []
    mask = np.asarray(mask, dtype=bool)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((index[i], index[j], j - i + 1))
            i = j + 1
        else:
            i += 1
    return out


def fmt(t: pd.Timestamp, freq: str) -> str:
    return t.strftime("%Y-%m") if freq_params(freq).code == "MS" else t.strftime("%Y-%m-%d")


def rolling_median_back(x: np.ndarray, w: int, minp: int) -> np.ndarray:
    """t 직전 w개(t 미포함)의 중앙값."""
    s = pd.Series(x).rolling(w, min_periods=minp).median().shift(1)
    return s.to_numpy()


def rolling_median_fwd(x: np.ndarray, w: int, minp: int, lag: int = 0) -> np.ndarray:
    """t+lag부터 w개(t+lag 포함)의 중앙값."""
    r = pd.Series(x[::-1]).rolling(w, min_periods=minp).median().to_numpy()[::-1]
    out = np.full_like(r, np.nan)
    if lag == 0:
        return r
    out[:-lag] = r[lag:]
    return out
